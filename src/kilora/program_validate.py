# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .errors import KiLoRAError, ModFormatError


class ProgramError(ModFormatError):
    pass


@dataclass(frozen=True)
class ValueType:
    axes: tuple[str, ...]
    dtype: str = "float32"
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class Signature:
    inputs: Mapping[str, ValueType]
    outputs: Mapping[str, ValueType]
    layer_change: bool = False


@dataclass(frozen=True)
class ValidatedProgram:
    instructions: tuple[dict[str, Any], ...]
    value_types: Mapping[str, ValueType]
    identity: str
    outputs: tuple[str, ...] = ()
    version: int = 1
    op_set: str = "kilora.ops"
    upgraded_identity: str = ""


INSTRUCTION_LIMIT = 96


def layer_change_signature(hook_sizes: tuple[int, int]) -> Signature:
    in_size, out_size = hook_sizes
    return Signature(
        {
            "x": ValueType(("batch", "tokens", "in"), shape=(1, 4096, in_size)),
            "y": ValueType(("batch", "tokens", "out"), shape=(1, 4096, out_size)),
        },
        {"out": ValueType(("batch", "tokens", "out"), shape=(1, 4096, out_size))},
        True,
    )


def check_program_count(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 16:
        raise ProgramError("program <mod>, instruction -, op -: program count must be at most 16")


def program_identity(program: Mapping[str, Any]) -> str:
    fields = ("op_set", "op_set_version", "hooks", "inputs", "outputs", "bindings", "instructions")
    data = {key: copy.deepcopy(program[key]) for key in fields if key in program}
    for binding in data.get("bindings", {}).values():
        if isinstance(binding, dict):
            binding.pop("tensor", None)
            binding.pop("tensors", None)
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _validate(program: Mapping[str, Any], signature: Signature) -> ValidatedProgram:
    from .program import AXES, op_table

    name = program.get("name", "<unnamed>") if isinstance(program, Mapping) else "<unnamed>"
    part = program.get("part", "-") if isinstance(program, Mapping) else "-"

    def fail(rule: str, ins: Mapping[str, Any] | None = None) -> None:
        ins = ins or {}
        raise ProgramError(
            f"program {name}, part {part}, instruction {ins.get('id', '-')}, "
            f"op {ins.get('op', '-')}: {rule}"
        )

    if not isinstance(program, Mapping):
        fail("program must be a record")
    if not isinstance(program.get("op_set"), str) or type(program.get("op_set_version")) is not int:
        fail("op set name and version have invalid types")
    try:
        table = op_table(program.get("op_set"), program.get("op_set_version"))
    except (KiLoRAError, TypeError) as exc:
        fail(f"op set: {exc}")
    instructions = program.get("instructions")
    if not isinstance(instructions, list):
        fail("instructions must be a list")
    if len(instructions) > INSTRUCTION_LIMIT:
        fail(f"instruction limit {INSTRUCTION_LIMIT} exceeded")
    for i, ins in enumerate(instructions):
        if not isinstance(ins, Mapping):
            fail(f"instruction {i} must be a record")
        if not all(k in ins for k in ("id", "op", "args", "result", "attributes")):
            fail("missing instruction field", ins)
        if type(ins["id"]) is not int or ins["id"] != i:
            fail("instruction ids must be in order", ins)
        op = ins["op"]
        if not isinstance(op, str) or op not in table:
            fail("unknown op", ins)
        schema = table[op]
        if not isinstance(ins["args"], list) or (
            len(ins["args"]) != schema["arity"] if schema["arity"] != -1 else len(ins["args"]) < 1
        ):
            fail("wrong arity", ins)
        attrs = ins["attributes"]
        if not isinstance(attrs, Mapping) or set(attrs) != set(schema["attributes"]):
            fail("attributes missing or unknown", ins)
        for key, rule in schema["attributes"].items():
            value = attrs[key]
            typ = rule["type"]
            valid = (
                (typ == "bool" and type(value) is bool)
                or (typ == "int" and type(value) is int)
                or (typ == "float" and type(value) in (int, float) and math.isfinite(value))
                or (typ in ("string", "input_name") and isinstance(value, str))
            )
            if (
                not valid
                or ("values" in rule and value not in rule["values"])
                or ("min" in rule and value < rule["min"])
                or ("max" in rule and value > rule["max"])
                or (
                    "greater_than_attribute" in rule
                    and (
                        type(attrs.get(rule["greater_than_attribute"])) not in (int, float)
                        or value <= attrs[rule["greater_than_attribute"]]
                    )
                )
            ):
                fail(f"attribute {key} violates schema", ins)
    bindings = program.get("bindings", {})
    if not isinstance(bindings, Mapping):
        fail("bindings must be a record")
    declared_inputs = program.get("inputs", {})
    if not isinstance(declared_inputs, Mapping) or set(declared_inputs) != set(signature.inputs):
        fail("inputs do not match signature")
    declared_outputs = program.get("outputs", {})
    if not signature.layer_change and (
        not isinstance(declared_outputs, Mapping) or set(declared_outputs) != set(signature.outputs)
    ):
        fail("declared outputs do not match signature")
    names = set(signature.inputs)
    if names.intersection(bindings):
        fail("SSA value written twice")
    names.update(bindings)
    for ins in instructions:
        if not isinstance(ins["result"], str) or not ins["result"] or ins["result"] in names:
            fail("SSA value written twice or invalid result", ins)
        if any(not isinstance(arg, str) or arg not in names for arg in ins["args"]):
            fail("SSA read before definition", ins)
        names.add(ins["result"])

    dtypes = {"float64", "float32", "float16", "bfloat16", "bool", "int64"}
    sizes: dict[str, int] = {"batch": 1, "tokens": 4096}

    def check_type(value: ValueType, label: str) -> ValueType:
        if (
            not isinstance(value, ValueType)
            or any(a not in AXES for a in value.axes)
            or len(set(value.axes)) != len(value.axes)
        ):
            fail(f"{label}: unknown or repeated axis")
        if value.dtype not in dtypes:
            fail(f"{label}: invalid dtype")
        shape = value.shape or tuple(sizes.get(axis, 1) for axis in value.axes)
        if len(shape) != len(value.axes) or any(type(n) is not int or n < 1 for n in shape):
            fail(f"{label}: invalid shape")
        for axis, size in zip(value.axes, shape, strict=True):
            if axis not in ("batch", "tokens"):
                if axis in sizes and sizes[axis] != size:
                    fail(f"{label}: axis {axis} shape mismatch")
                sizes[axis] = size
        return ValueType(value.axes, value.dtype, shape)

    types: dict[str, ValueType] = {}
    for key, value in signature.inputs.items():
        types[key] = check_type(value, f"input {key}")
        decl = declared_inputs[key]
        if (
            not isinstance(decl, Mapping)
            or tuple(decl.get("axes", ())) != value.axes
            or decl.get("dtype", value.dtype) != value.dtype
        ):
            fail(f"input {key} axes or dtype does not match signature")
    expected = {key: check_type(value, f"output {key}") for key, value in signature.outputs.items()}
    for key, binding in bindings.items():
        if not isinstance(key, str) or not isinstance(binding, Mapping):
            fail("invalid binding")
        shape = binding.get("shape")
        dtype = binding.get("dtype")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 1 for n in shape)
            or dtype not in dtypes
        ):
            fail(f"binding {key}: invalid shape or dtype")
        axes = binding.get("axes")
        if axes is not None:
            types[key] = check_type(ValueType(tuple(axes), dtype, tuple(shape)), f"binding {key}")
        else:
            types[key] = ValueType((), dtype, tuple(shape))

    def broadcast(a: ValueType, b: ValueType, ins: Mapping[str, Any]) -> ValueType:
        if a.dtype != b.dtype or not a.axes and a.shape or not b.axes and b.shape:
            fail("type mismatch in broadcast", ins)
        axes = list(a.axes)
        for axis in b.axes:
            if axis not in axes:
                axes.append(axis)
        shape = []
        for axis in axes:
            x = a.shape[a.axes.index(axis)] if axis in a.axes else 1
            y = b.shape[b.axes.index(axis)] if axis in b.axes else 1
            if x != y and x != 1 and y != 1:
                fail("shape mismatch in broadcast", ins)
            shape.append(max(x, y))
        return ValueType(tuple(axes), a.dtype, tuple(shape))

    for ins in instructions:
        op, attrs = ins["op"], ins["attributes"]
        args = [types[a] for a in ins["args"]]
        rule = table[op]["type_rule"]
        result = args[0] if args else None
        if rule == "matmul":
            a, b = args
            if not a.axes or len(b.shape) != 2 or a.dtype != b.dtype:
                fail("matmul type or binding shape mismatch", ins)
            k, n = (b.shape[1], b.shape[0]) if attrs["transpose_b"] else b.shape
            if a.shape[-1] != k:
                fail("matmul binding shape mismatch", ins)
            descendants = {ins["result"]}
            feeds_matmul = False
            for later in instructions[ins["id"] + 1 :]:
                if any(arg in descendants for arg in later["args"]):
                    if later["op"] == "matmul" and later["args"][0] in descendants:
                        feeds_matmul = True
                    descendants.add(later["result"])
            axis = (
                b.axes[0 if attrs["transpose_b"] else 1]
                if b.axes
                else "out"
                if sizes.get("out") == n and not feeds_matmul
                else "rank"
            )
            if b.axes and b.axes[1 if attrs["transpose_b"] else 0] != a.axes[-1]:
                fail("matmul contracted axis mismatch", ins)
            if axis in a.axes[:-1] or (axis in sizes and sizes[axis] != n):
                fail("matmul output axis shape mismatch", ins)
            sizes[axis] = n
            result = ValueType((*a.axes[:-1], axis), a.dtype, (*a.shape[:-1], n))
        elif rule == "narrow":
            axes, shape = list(result.axes), list(result.shape)
            source = types.get(attrs["size_of"])
            if (
                attrs["axis"] not in axes
                or attrs["new_axis"] in axes
                or source is None
                or attrs["size_of"] not in signature.inputs
                or attrs["new_axis"] not in source.axes
            ):
                fail("invalid narrow axes or size source", ins)
            i = axes.index(attrs["axis"])
            extent = source.shape[source.axes.index(attrs["new_axis"])]
            if extent > shape[i]:
                fail("narrow exceeds axis size", ins)
            axes[i], shape[i] = attrs["new_axis"], extent
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule in ("svd_values", "svd_vh"):
            if len(result.axes) != 2 or result.dtype != "float32" or attrs["axis"] in result.axes:
                fail("SVD requires a float32 matrix and a new axis", ins)
            extent = min(result.shape)
            if attrs["axis"] in sizes and sizes[attrs["axis"]] != extent:
                fail("SVD axis shape mismatch", ins)
            sizes[attrs["axis"]] = extent
            if rule == "svd_values":
                result = ValueType((attrs["axis"],), result.dtype, (extent,))
            else:
                result = ValueType(
                    (attrs["axis"], result.axes[-1]), result.dtype, (extent, result.shape[-1])
                )
        elif rule == "unflatten":
            axes, shape = list(result.axes), list(result.shape)
            source = types.get(attrs["size_of"])
            new_axes = (attrs["axis1"], attrs["axis2"])
            if (
                attrs["axis"] not in axes
                or source is None
                or attrs["size_of"] not in signature.inputs
                or new_axes[0] == new_axes[1]
                or any(axis in axes or axis not in source.axes for axis in new_axes)
            ):
                fail("invalid unflatten axes or size source", ins)
            i = axes.index(attrs["axis"])
            extents = [source.shape[source.axes.index(axis)] for axis in new_axes]
            if extents[0] * extents[1] != shape[i]:
                fail("unflatten axis shape mismatch", ins)
            axes[i : i + 1] = new_axes
            shape[i : i + 1] = extents
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "flatten":
            axes, shape = list(result.axes), list(result.shape)
            if (
                attrs["axis1"] not in axes
                or attrs["axis2"] not in axes
                or axes.index(attrs["axis2"]) != axes.index(attrs["axis1"]) + 1
                or attrs["new_axis"] in axes
            ):
                fail("flatten requires adjacent axes and a new axis", ins)
            i = axes.index(attrs["axis1"])
            extent = shape[i] * shape[i + 1]
            if attrs["new_axis"] in sizes and sizes[attrs["new_axis"]] != extent:
                fail("flatten axis shape mismatch", ins)
            sizes[attrs["new_axis"]] = extent
            axes[i : i + 2] = [attrs["new_axis"]]
            shape[i : i + 2] = [extent]
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "reduce_pair":
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis1"] == attrs["axis2"] or any(
                axis not in axes for axis in (attrs["axis1"], attrs["axis2"])
            ):
                fail("sum_pair requires two distinct input axes", ins)
            for index in sorted(
                (axes.index(attrs["axis1"]), axes.index(attrs["axis2"])), reverse=True
            ):
                axes.pop(index)
                shape.pop(index)
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "transpose":
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis1"] not in axes or attrs["axis2"] not in axes:
                fail("transpose axis absent", ins)
            i, j = axes.index(attrs["axis1"]), axes.index(attrs["axis2"])
            axes[i], axes[j] = axes[j], axes[i]
            shape[i], shape[j] = shape[j], shape[i]
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "rename_axis":
            axes = list(result.axes)
            if (
                attrs["axis"] not in axes
                or attrs["new_axis"] in axes
                or attrs["axis"] in ("batch", "tokens")
                or attrs["new_axis"] in ("batch", "tokens")
            ):
                fail("invalid axis rename", ins)
            index = axes.index(attrs["axis"])
            extent = result.shape[index]
            if attrs["new_axis"] in sizes and sizes[attrs["new_axis"]] != extent and extent != 1:
                fail("renamed axis shape mismatch", ins)
            if extent != 1 or attrs["new_axis"] not in sizes:
                sizes[attrs["new_axis"]] = extent
            axes[index] = attrs["new_axis"]
            result = ValueType(tuple(axes), result.dtype, result.shape)
        elif rule == "unsqueeze":
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis"] in axes or attrs["before"] not in axes:
                fail("invalid singleton axis", ins)
            i = axes.index(attrs["before"]) + int(attrs["after"])
            axes.insert(i, attrs["axis"])
            shape.insert(i, 1)
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "stack":
            if any(arg != args[0] for arg in args[1:]):
                fail("stack input types or shapes differ", ins)
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis"] in axes or attrs["before"] not in axes:
                fail("invalid stack axis", ins)
            i = axes.index(attrs["before"])
            axes.insert(i, attrs["axis"])
            shape.insert(i, len(args))
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "arange_axis":
            if attrs["axis"] not in result.axes:
                fail("arange axis absent", ins)
            index = result.axes.index(attrs["axis"])
            result = ValueType((attrs["axis"],), "int64", (result.shape[index],))
        elif rule == "select":
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis"] not in axes:
                fail("select axis absent", ins)
            i = axes.index(attrs["axis"])
            if attrs["index"] >= shape[i]:
                fail("select index out of bounds", ins)
            axes.pop(i)
            shape.pop(i)
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "key_projection":
            query, keys = args
            if (
                query.axes != ("batch", "tokens", "key")
                or keys.axes != ("experts", "per", "key")
                or query.dtype != keys.dtype
                or query.shape[-1] != keys.shape[-1]
            ):
                fail("key projection axes, shape or dtype mismatch", ins)
            result = ValueType(
                ("batch", "tokens", "experts", "per"),
                query.dtype,
                (*query.shape[:-1], *keys.shape[:-1]),
            )
        elif rule == "repeat_interleave":
            axes, shape = list(result.axes), list(result.shape)
            if attrs["axis"] not in axes or (
                attrs["new_axis"] in axes and attrs["new_axis"] != attrs["axis"]
            ):
                fail("invalid repeat axis", ins)
            i = axes.index(attrs["axis"])
            axes[i] = attrs["new_axis"]
            shape[i] *= attrs["repeats"]
            if axes[i] in sizes and sizes[axes[i]] != shape[i]:
                fail("repeat axis shape mismatch", ins)
            sizes[axes[i]] = shape[i]
            result = ValueType(tuple(axes), result.dtype, tuple(shape))
        elif rule == "batch_matmul":
            a, b = args
            if (
                len(a.axes) < 2
                or len(b.axes) < 2
                or a.dtype != b.dtype
                or a.axes[-1] != b.axes[-2]
                or a.shape[-1] != b.shape[-2]
            ):
                fail("batch matmul contracted axis or dtype mismatch", ins)
            ab, bb = a.axes[:-2], b.axes[:-2]
            if ab and bb and ab != bb:
                fail("batch matmul batch axes mismatch", ins)
            batch = ab or bb
            shape = a.shape[:-2] if ab else b.shape[:-2]
            if ab and bb and a.shape[:-2] != b.shape[:-2]:
                fail("batch matmul batch shape mismatch", ins)
            axes = (*batch, a.axes[-2], b.axes[-1])
            if len(set(axes)) != len(axes):
                fail("batch matmul repeated output axis", ins)
            result = ValueType(axes, a.dtype, (*shape, a.shape[-2], b.shape[-1]))
        elif rule in ("broadcast", "compare"):
            result = broadcast(args[0], args[1], ins)
            if rule == "compare":
                result = ValueType(result.axes, "bool", result.shape)
        elif rule == "where":
            if args[0].dtype != "bool":
                fail("where condition must be bool", ins)
            result = broadcast(args[1], args[2], ins)
            cond = ValueType(args[0].axes, result.dtype, args[0].shape)
            result = broadcast(result, cond, ins)
        elif rule in ("reduce", "argmax", "same") and "axis" in attrs:
            if op == "vector_norm" and result.dtype != "float32":
                fail("vector_norm requires float32 input", ins)
            axis = attrs["axis"]
            if axis not in result.axes:
                fail("axis is absent from input", ins)
            if rule != "same":
                ix = result.axes.index(axis)
                axes, shape = list(result.axes), list(result.shape)
                if rule == "reduce" and attrs["keep"]:
                    shape[ix] = 1
                else:
                    axes.pop(ix)
                    shape.pop(ix)
                result = ValueType(
                    tuple(axes), "int64" if rule == "argmax" else result.dtype, tuple(shape)
                )
        elif rule == "cast":
            result = ValueType(result.axes, attrs["dtype"], result.shape)
        elif rule == "one_hot":
            axis, source = attrs["axis"], attrs["size_of"]
            if (
                result.dtype != "int64"
                or axis in result.axes
                or source not in signature.inputs
                or axis not in types[source].axes
            ):
                fail("one_hot axis, indices or size_of mismatch", ins)
            size = types[source].shape[types[source].axes.index(axis)]
            result = ValueType((*result.axes, axis), "int64", (*result.shape, size))
        elif rule == "gather":
            if args[1].dtype != "int64" or attrs["axis"] not in args[0].axes:
                fail("gather axis or indices mismatch", ins)
            result = ValueType(args[1].axes, args[0].dtype, args[1].shape)
        elif rule not in ("same", "output_add"):
            fail(f"unknown type rule {rule}", ins)
        types[ins["result"]] = result

    output_ops = [ins for ins in instructions if ins["op"] == "output_add"]
    if signature.layer_change:
        if len(output_ops) != 1 or instructions[-1] is not output_ops[0]:
            fail("layer change must end with exactly one output_add")
        ins = output_ops[0]
        if types[ins["result"]] != next(iter(expected.values())):
            fail("output_add type does not match signature", ins)
    else:
        if output_ops:
            fail("output_add is only valid for layer changes", output_ops[0])
        for key, value in expected.items():
            decl = declared_outputs[key]
            if (
                not isinstance(decl, Mapping)
                or tuple(decl.get("axes", ())) != value.axes
                or decl.get("dtype", value.dtype) != value.dtype
                or types.get(key) != value
                or key not in {i["result"] for i in instructions}
            ):
                fail(f"output {key} missing or type does not match signature")
    for ins in instructions:
        value = types[ins["result"]]
        shape = [
            1 if a == "batch" else 4096 if a == "tokens" else n
            for a, n in zip(value.axes, value.shape, strict=True)
        ]
        if (
            math.prod(shape) * (8 if value.dtype in ("float64", "int64") else 4)
            >= 256 * 1024 * 1024
        ):
            fail("working-bytes preflight exceeds 256 MiB", ins)
    for key, binding in bindings.items():
        if binding.get("kind") == "derived":
            fail(f"binding {key}: derived bindings are not supported in 1.0")
        if binding.get("kind") not in ("embedded", "ref"):
            fail(f"binding {key}: unknown kind")
    try:
        identity = program_identity(program)
    except (TypeError, ValueError, AttributeError) as exc:
        fail(f"identity canonical JSON: {exc}")
    if program.get("identity") != identity:
        fail("identity mismatch")
    return ValidatedProgram(
        tuple(copy.deepcopy(instructions)),
        types,
        identity,
        (instructions[-1]["result"],) if signature.layer_change else tuple(signature.outputs),
        program["op_set_version"],
        program["op_set"],
        identity,
    )


def validate(program: Mapping[str, Any], signature: Signature) -> ValidatedProgram:
    try:
        from .program import upgrade_program

        if not isinstance(program, Mapping):
            return _validate(program, signature)
        instructions = program.get("instructions")
        if isinstance(instructions, list) and len(instructions) > 64:
            return _validate(program, signature)
        try:
            upgraded = upgrade_program(program)
        except KiLoRAError as exc:
            name = program.get("name", "<unnamed>")
            part = program.get("part", "-")
            raise ProgramError(
                f"program {name}, part {part}, instruction -, op -: op set: {exc}"
            ) from exc
        validated = _validate(upgraded, signature)
        return replace(validated, identity=program["identity"])
    except ProgramError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError, IndexError) as exc:
        name = program.get("name", "<unnamed>") if isinstance(program, Mapping) else "<unnamed>"
        part = program.get("part", "-") if isinstance(program, Mapping) else "-"
        raise ProgramError(
            f"program {name}, part {part}, instruction -, op -: invalid program data ({exc})"
        ) from None
