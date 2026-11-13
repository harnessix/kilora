# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from .errors import KiLoRAError

AXES = [
    "batch",
    "tokens",
    "in",
    "out",
    "rank",
    "experts",
    "per",
    "cols",
    "key",
    "axes",
    "r1",
    "r2",
    "r3",
]


def _attr(type_name: str, **constraints: Any) -> dict[str, Any]:
    return {"type": type_name, "required": True, **constraints}


def _op(name: str, arity: int, rule: str, **attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "arity": arity,
        "attributes": attributes,
        "type_rule": rule,
        "since_version": 1,
    }


def _core_ops() -> dict[str, dict[str, Any]]:
    ops: dict[str, dict[str, Any]] = {}

    def add(name: str, arity: int, rule: str, **attrs: dict[str, Any]) -> None:
        ops[name] = _op(name, arity, rule, **attrs)

    def axis() -> dict[str, Any]:
        return _attr("string", values=AXES.copy())

    add("matmul", 2, "matmul", transpose_b=_attr("bool"))
    add("transpose", 1, "transpose", axis1=axis(), axis2=axis())
    add("rename_axis", 1, "rename_axis", axis=axis(), new_axis=axis())
    add("unsqueeze", 1, "unsqueeze", axis=axis(), before=axis(), after=_attr("bool"))
    add("batch_matmul", 2, "batch_matmul")
    add("key_projection", 2, "key_projection")
    add("stack", 4, "stack", axis=axis(), before=axis())
    add("stack_rows", -1, "stack", axis=axis(), before=axis())
    add("select", 1, "select", axis=axis(), index=_attr("int", min=0))
    add("arange_axis", 1, "arange_axis", axis=axis())
    add(
        "repeat_interleave",
        1,
        "repeat_interleave",
        axis=axis(),
        new_axis=axis(),
        repeats=_attr("int", min=1),
    )
    for name in ("add", "mul", "sub", "div", "pow"):
        add(name, 2, "broadcast")
    add("scale", 1, "same", value=_attr("float", min=-64, max=64), by_strength=_attr("bool"))
    for name in ("silu", "tanh", "sigmoid", "relu", "abs", "exp", "log", "sqrt", "neg", "detach"):
        add(name, 1, "same")
    add("gelu", 1, "same", approximate=_attr("string", values=["none", "tanh"]))
    add("clamp", 1, "same", low=_attr("float"), high=_attr("float", greater_than_attribute="low"))
    add("clamp_min", 1, "same", low=_attr("float"))
    add("vector_norm", 1, "reduce", axis=axis(), keep=_attr("bool"))
    add("pow_int", 1, "same", exponent=_attr("int"))
    add("mask", 1, "same", source=_attr("string", values=["region", "roles"]))
    add("output_add", 1, "output_add")
    for name in ("sum", "mean", "max", "min"):
        add(name, 1, "reduce", axis=axis(), keep=_attr("bool"))
    add("narrow", 1, "narrow", axis=axis(), new_axis=axis(), size_of=_attr("input_name"))
    add("svd_vh", 1, "svd_vh", axis=axis())
    add("svd_values", 1, "svd_values", axis=axis())
    add(
        "unflatten",
        1,
        "unflatten",
        axis=axis(),
        axis1=axis(),
        axis2=axis(),
        size_of=_attr("input_name"),
    )
    add("flatten", 1, "flatten", axis1=axis(), axis2=axis(), new_axis=axis())
    add("sum_pair", 1, "reduce_pair", axis1=axis(), axis2=axis())
    add("softmax", 1, "same", axis=axis())
    add("sort", 1, "same", axis=axis(), descending=_attr("bool"))
    add("cumsum", 1, "same", axis=axis())
    for name in ("lt", "le", "gt", "ge", "eq"):
        add(name, 2, "compare")
    add("where", 3, "where")
    add("argmax", 1, "argmax", axis=axis())
    add("one_hot", 1, "one_hot", axis=axis(), size_of=_attr("input_name"))
    add("gather", 2, "gather", axis=axis())
    add(
        "cast",
        1,
        "cast",
        dtype=_attr(
            "string", values=["float64", "float32", "float16", "bfloat16", "bool", "int64"]
        ),
    )
    return ops


_TABLES: dict[str, dict[int, dict[str, Any]]] = {"kilora.ops": {1: _core_ops()}}
_UPGRADES: dict[tuple[str, int], Callable[[dict[str, Any]], None]] = {}


def op_set_digest(table: Mapping[str, Any]) -> str:
    data = json.dumps(
        table, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def register_ops(op_set: str, version: int, ops: Mapping[str, Any]) -> None:
    if op_set == "kilora.ops":
        raise KiLoRAError("Plugins cannot replace the kilora.ops op set.")
    if not isinstance(op_set, str) or not op_set.endswith(".ops") or len(op_set) <= 4:
        raise KiLoRAError("A plugin op set must have a name of the form <plugin>.ops.")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise KiLoRAError("An op set version must be a positive integer.")
    if not isinstance(ops, Mapping):
        raise KiLoRAError("An op table must map names to operation schemas.")
    try:
        op_set_digest(ops)
    except (TypeError, ValueError) as exc:
        raise KiLoRAError("An op table must contain only JSON data.") from exc
    versions = _TABLES.setdefault(op_set, {})
    if version in versions:
        raise KiLoRAError(f"Op set {op_set!r} version {version} is already registered.")
    versions[version] = copy.deepcopy(dict(ops))


def op_table(op_set: str, version: int) -> dict[str, Any]:
    versions = _TABLES.get(op_set)
    if versions is None:
        raise KiLoRAError(f"Unknown op set {op_set!r}; install the plugin that provides it.")
    if version not in versions:
        if isinstance(version, int) and version > max(versions):
            raise KiLoRAError(
                f"Op set {op_set!r} version {version} is newer; update KiLoRA or its plugin."
            )
        raise KiLoRAError(f"Op set {op_set!r} version {version} is not available.")
    return copy.deepcopy(versions[version])


def register_upgrade(
    op_set: str, from_version: int, adapter: Callable[[dict[str, Any]], None]
) -> None:
    if type(from_version) is not int or from_version < 1 or not callable(adapter):
        raise KiLoRAError("An upgrade requires a positive version and a callable adapter.")
    op_table(op_set, from_version)
    op_table(op_set, from_version + 1)
    key = (op_set, from_version)
    if key in _UPGRADES:
        raise KiLoRAError(f"Upgrade for {op_set!r} version {from_version} is already registered.")
    _UPGRADES[key] = adapter


def upgrade_program(
    program: Mapping[str, Any], target_version: int | None = None
) -> dict[str, Any]:
    name, version = program.get("op_set"), program.get("op_set_version")
    if not isinstance(name, str) or type(version) is not int or version < 1:
        raise KiLoRAError("An op set name and positive integer version are required.")
    op_table(name, version)
    target = max(_TABLES[name]) if target_version is None else target_version
    if type(target) is not int or target < version:
        raise KiLoRAError("A program cannot be downgraded.")
    op_table(name, target)
    result = copy.deepcopy(dict(program))
    if target == version:
        return result
    if result.get("identity") != program_identity(result):
        raise KiLoRAError("Program identity mismatch before upgrade.")
    while version < target:
        op_table(name, version + 1)
        adapter = _UPGRADES.get((name, version))
        if adapter is None:
            raise KiLoRAError(f"No upgrade for op set {name!r} version {version}.")
        adapter(result)
        if result.get("op_set") != name or result.get("op_set_version") != version:
            raise KiLoRAError("An upgrade adapter cannot change the op set or version.")
        version += 1
        result["op_set_version"] = version
        result["identity"] = program_identity(result)
    return result


from .program_validate import (
    ProgramError,
    Signature,
    ValidatedProgram,
    ValueType,
    check_program_count,
    layer_change_signature,
    program_identity,
    validate,
)

__all__ = [
    "ProgramError",
    "Signature",
    "ValidatedProgram",
    "ValueType",
    "check_program_count",
    "layer_change_signature",
    "program_identity",
    "validate",
    "op_table",
    "op_set_digest",
    "register_ops",
    "register_upgrade",
    "upgrade_program",
    "AXES",
]

from .program_eager import LOWERINGS, interpret

__all__ += ["LOWERINGS", "interpret"]
