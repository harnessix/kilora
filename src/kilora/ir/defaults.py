# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

import torch

from ..program import ProgramError, Signature, ValueType, interpret, validate
from .parts import PARTS

_SIGNATURES = {
    ("score", "projection"): Signature(
        {
            "query": ValueType(("batch", "tokens", "key")),
            "keys": ValueType(("batch", "key", "cols")),
        },
        {"projection": ValueType(("batch", "tokens", "cols"))},
    ),
    ("score", "weighted"): Signature(
        {
            "projection": ValueType(("batch", "tokens", "experts", "per")),
            "weights": ValueType(("batch", "tokens", "experts")),
        },
        {"weighted": ValueType(("batch", "tokens", "experts", "per"))},
    ),
    ("score", "router"): Signature(
        {
            "query": ValueType(("batch", "tokens", "key")),
            "projection": ValueType(("batch", "tokens", "experts", "per")),
            "rows": ValueType(("experts",)),
            "key_width": ValueType(()),
            "temperature": ValueType(()),
        },
        {"logits": ValueType(("batch", "tokens", "experts"))},
    ),
    ("score", "keys"): Signature(
        {
            "query": ValueType(("batch", "tokens", "key")),
            "keys": ValueType(("experts", "per", "key")),
            "rows": ValueType(("experts",)),
            "key_width": ValueType(()),
            "temperature": ValueType(()),
        },
        {"logits": ValueType(("batch", "tokens", "experts"))},
    ),
    ("score", "read_share"): Signature(
        {"reads": ValueType(("batch", "tokens", "experts", "per")), "temperature": ValueType(())},
        {"logits": ValueType(("batch", "tokens", "experts"))},
    ),
}


_BTE = ("batch", "tokens", "experts")
_SIGNATURES[("gate", "active_count")] = Signature(
    {"active": ValueType(("batch", "tokens", "r1"))},
    {"token_count": ValueType(("batch",))},
)
_SIGNATURES.update(
    {
        ("select", f"{scope}_{selection}"): Signature(
            {
                "logits": ValueType(_BTE),
                "noise": ValueType(_BTE),
                **(
                    {"region": ValueType(("batch", "tokens")), "has_region": ValueType((), "bool")}
                    if scope == "tensor"
                    else {}
                ),
            },
            {"weights": ValueType(_BTE)},
        )
        for scope in ("token", "tensor")
        for selection in ("sparsemax", "top1")
    }
)


_SIGNATURES[("select", "region_mean")] = Signature(
    {
        "logits": ValueType(_BTE),
        "region": ValueType(("batch", "tokens")),
        "has_region": ValueType((), "bool"),
        "prior": ValueType(()),
    },
    {"mean_logits": ValueType(_BTE, "float32", (1, 1, 1))},
)


_PATH_COLS = ValueType(("batch", "tokens", "cols"))
_PATH_OUT = ValueType(("batch", "tokens", "out"))
_SIGNATURES.update(
    {
        ("iterate", "path_step"): Signature(
            {"read": _PATH_COLS, "weight": _PATH_COLS},
            {"state": _PATH_COLS},
        ),
        ("iterate", "path_stop"): Signature(
            {
                "stopped": ValueType(("batch", "tokens", "cols"), "bool"),
                "previous": _PATH_COLS,
                "state": _PATH_COLS,
            },
            {"result": _PATH_COLS},
        ),
        ("iterate", "path_delta"): Signature(
            {"coefficients": _PATH_COLS, "up": ValueType(("cols", "out"))},
            {"delta": _PATH_OUT},
        ),
        ("iterate", "path_add"): Signature(
            {"base": _PATH_OUT, "delta": _PATH_OUT},
            {"output": _PATH_OUT},
        ),
    }
)


_GROUPED = ValueType(("batch", "tokens", "experts", "per"))
_SIGNATURES.update(
    {
        ("keep", "default"): PARTS["keep"].signature,
        ("carry", "drop"): PARTS["carry"].signature,
        ("carry", "dropped"): PARTS["carry"].signature,
        ("carry", "mask_sample"): Signature(
            {"change": _GROUPED, "base_read": _GROUPED, "uniforms": _GROUPED}, {"mask": _GROUPED}
        ),
        ("carry", "mask_expect"): Signature(
            {"change": _GROUPED, "base_read": _GROUPED, "uniforms": _GROUPED}, {"mask": _GROUPED}
        ),
        ("carry", "transfer"): Signature(
            {"change": _GROUPED, "base_read": _GROUPED}, {"state": _GROUPED}
        ),
    }
)


_SIGNATURES.update(
    {
        ("iterate", "accumulate"): PARTS["iterate"].signature,
        ("iterate", "gated_read"): Signature(
            {
                "token": ValueType(("batch", "tokens", "r1")),
                "change": ValueType(("batch", "tokens", "cols")),
                "up": ValueType(("batch", "cols", "out")),
                "channel": ValueType(("batch", "out")),
                "target": ValueType(("out", "key")),
                "base": ValueType(("batch", "tokens", "key")),
            },
            {"result": ValueType(("batch", "tokens", "key"))},
        ),
        ("iterate", "gated_read_shared"): Signature(
            {
                "token": ValueType(("batch", "tokens", "r1")),
                "change": ValueType(("batch", "tokens", "cols")),
                "up": ValueType(("cols", "out")),
                "channel": ValueType(("out",)),
                "target": ValueType(("out", "key")),
                "base": ValueType(("batch", "tokens", "key")),
            },
            {"result": ValueType(("batch", "tokens", "key"))},
        ),
        ("iterate", "output_split"): Signature(
            {
                "base": ValueType(("batch", "tokens", "out")),
                "coefficients": ValueType(("batch", "tokens", "cols")),
                "up": ValueType(("cols", "out")),
                "token": ValueType(("batch", "tokens", "r1")),
                "channel": ValueType(("batch", "out")),
            },
            {"output": ValueType(("batch", "tokens", "out"))},
        ),
        ("iterate", "output_split_tokens"): Signature(
            {
                "base": ValueType(("batch", "tokens", "out")),
                "coefficients": ValueType(("batch", "tokens", "cols")),
                "up": ValueType(("cols", "out")),
                "token": ValueType(("batch", "tokens", "r1")),
                "channel": ValueType(("batch", "tokens", "out")),
            },
            {"output": ValueType(("batch", "tokens", "out"))},
        ),
        ("iterate", "output"): Signature(
            {
                "base": ValueType(("batch", "tokens", "out")),
                "coefficients": ValueType(("batch", "tokens", "cols")),
                "up": ValueType(("cols", "out")),
            },
            {"output": ValueType(("batch", "tokens", "out"))},
        ),
    }
)


for _mode in ("sample", "expect"):
    _mask_type = ValueType(("batch", "tokens", "cols"))
    _SIGNATURES[("gate", f"mask_{_mode}")] = Signature(
        {"probability": _mask_type, "uniforms": _mask_type}, {"mask": _mask_type}
    )


_LIVE_PARTS = ("guard", "live_write")


def _record(part, variant):
    if not isinstance(part, str) or not isinstance(variant, str):
        raise ProgramError(f"unknown default program {part!r}, {variant!r}")
    if not part.isidentifier() or not variant.isidentifier():
        raise ProgramError(f"unknown default program {part!r}, {variant!r}")
    if (
        part == "budget"
        and variant
        in {
            "scalar",
            "scalar_sum",
            "auto",
            "auto_sum",
            "auto_scalar",
            "auto_add_scalar",
            "auto_seed_scalar",
        }
    ) or (part == "freeze" and variant == "step"):
        name = f"{part}_{variant}"
    else:
        name = (
            part
            if variant == "default" and part in {"budget", "freeze", "transition", "entry"}
            else f"{part}.{variant}"
        )
    path = files("kilora.ir").joinpath("default", f"{name}.json")
    if not path.is_file():
        raise ProgramError(f"unknown default program {part!r}, {variant!r}")
    return json.loads(path.read_text())


def default_program(part, variant="default", sizes=None):
    if part in _LIVE_PARTS:
        return _live_program(part, variant, sizes=sizes)[0]
    if part in {"budget", "freeze", "transition", "entry"}:
        from .default import default_program as part_program

        return part_program(part, variant, sizes=sizes)
    signature = _SIGNATURES.get((part, variant))
    if (part, variant) in {("gate", "adaptive"), ("gate", "adaptive_active")}:
        signature = PARTS[part].sized_signature(sizes)
        if variant == "adaptive_active":
            signature = Signature(
                {**signature.inputs, "token_count": ValueType(("batch",))},
                signature.outputs,
            )
    record = _record(part, variant)
    return validate(record, signature)


def default_bindings(part, variant="default", *, tokens=None, kappa=1.0, coherence=True):
    if part in _LIVE_PARTS:
        return _live_program(part, variant)[1]
    record = _record(part, variant)
    bindings = record["bindings"]
    values = {name: item["value"] for name, item in bindings.items()}
    if (part, variant) in {("gate", "adaptive"), ("gate", "adaptive_active")}:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        if not float(kappa) > 0:
            raise ValueError("kappa must be positive")
        if type(coherence) is not bool:
            raise ValueError("coherence must be bool")
        values.update(kappa=float(kappa), coherence=coherence)
        if variant == "adaptive":
            values["token_count"] = float(tokens)
    return {
        name: torch.tensor(values[name], dtype=getattr(torch, item["dtype"]))
        for name, item in bindings.items()
    }


def _live_program(part: str, variant: str = "default", *, sizes=None):
    if (part, variant) not in {
        ("guard", "default"),
        ("live_write", "default"),
        ("live_write", "split"),
        ("live_write", "norm"),
        ("live_write", "root"),
        ("live_write", "scale"),
    }:
        raise ProgramError(f"unknown default program {part!r}, {variant!r}")
    record = json.loads(
        files("kilora.ir").joinpath("default", f"{part}.{variant}.json").read_text()
    )
    if variant in {"default", "split"}:
        signature = PARTS[part].sized_signature(sizes or {})
        if variant == "split":
            signature = Signature(
                {name: signature.inputs[name] for name in ("slopes", "shares", "multiplier")},
                {"raw": signature.outputs["update"]},
            )
    elif variant == "norm":
        signature = Signature(
            {
                "raw": ValueType(("cols",), shape=((sizes or {}).get("cols", 1),)),
                "total": ValueType((), "float64"),
            },
            {"next_total": ValueType((), "float64")},
        )
    elif variant == "root":
        signature = Signature(
            {"total": ValueType((), "float64")}, {"norm": ValueType((), "float64")}
        )
    else:
        signature = Signature(
            {"norm": ValueType((), "float64"), "write_norm": ValueType((), "float64")},
            {"write_scale": ValueType(())},
        )
    program = validate(record, signature)
    bindings = {
        name: torch.tensor(binding["value"], dtype=getattr(torch, binding["dtype"]))
        for name, binding in record["bindings"].items()
    }
    return program, bindings


def interpret_live_write(
    histories, coefficients, shares, write_norm, decay, multipliers=None, norm=None
):

    if multipliers is None:
        multipliers = {name: torch.ones_like(value[0]) for name, value in histories.items()}
    total = torch.tensor(0.0, dtype=torch.float64)
    raw_chunks = {}
    for name, history in histories.items():
        sizes = {"r1": history.shape[0], "cols": history.shape[1], "experts": shares.numel()}
        program, bindings = _live_program("live_write", "split", sizes=sizes)
        inputs = {"slopes": history, "shares": shares, "multiplier": multipliers[name]}
        raw = interpret(program, inputs, bindings)["raw"]
        raw_chunks[name] = raw
        if norm is None:
            program, bindings = _live_program("live_write", "norm", sizes={"cols": raw.numel()})
            total = interpret(program, {"raw": raw.reshape(-1), "total": total}, bindings)[
                "next_total"
            ]
    if norm is None:
        program, bindings = _live_program("live_write", "root")
        magnitude = interpret(program, {"total": total}, bindings)["norm"]
    else:
        magnitude = torch.tensor(float(norm(raw_chunks)), dtype=torch.float64)
    program, bindings = _live_program("live_write", "scale")
    scale = interpret(
        program,
        {"norm": magnitude, "write_norm": torch.tensor(write_norm, dtype=torch.float64)},
        bindings,
    )["write_scale"]
    result = {}
    for name, history in histories.items():
        sizes = {"r1": history.shape[0], "cols": history.shape[1], "experts": shares.numel()}
        from .driver import _run

        inputs = {
            "slopes": history,
            "shares": shares,
            "coefficients": coefficients[name],
            "multiplier": multipliers[name],
            "write_scale": scale,
            "retention": torch.tensor(1 - decay),
        }
        result[name] = _run("live_write", "default", inputs, sizes)["update"]
    return result


def key_preparation_program(experts, per, width):

    return _key_preparation_program(experts, per, width)


@lru_cache(maxsize=128)
def _key_preparation_program(experts, per, width):
    from ..program import program_identity

    if per < 1 or per > 3:
        raise ProgramError("key preparation exceeds the instruction limit")
    axes = {"keys": ["experts", "per", "key"], "rows": ["experts"]}
    record = {
        "name": "prepare.keys",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {
            name: {"axes": a, "dtype": "int64" if name == "rows" else "float32"}
            for name, a in axes.items()
        },
        "outputs": {"prepared": {"axes": axes["keys"], "dtype": "float32"}},
        "bindings": {},
        "instructions": [],
    }

    def add(op, args, result, **attributes):
        record["instructions"].append(
            {
                "id": len(record["instructions"]),
                "op": op,
                "args": args,
                "result": result,
                "attributes": attributes,
            }
        )
        return result

    normalized = []
    for j in range(per):
        v = add("select", ["keys"], f"row_{j}", axis="per", index=j)
        for repeat in range(2):
            for i, q in enumerate(normalized):
                name = f"{j}_{repeat}_{i}"
                product = add("mul", [v, q], f"product_{name}")
                dot = add("sum", [product], f"dot_{name}", axis="key", keep=True)
                component = add("mul", [dot, q], f"component_{name}")
                v = add("sub", [v, component], f"residual_{name}")
        norm = add("vector_norm", [v], f"norm_{j}", axis="key", keep=True)
        floor = add("clamp_min", [norm], f"floor_{j}", low=1e-12)
        normalized.append(add("div", [v, floor], f"unit_{j}"))
    add("stack_rows", normalized, "stacked", axis="per", before="key")
    add("arange_axis", ["keys"], "positions", axis="per")
    add("gt", ["rows", "positions"], "used")
    add("cast", ["used"], "used_float", dtype="float32")
    add("mul", ["stacked", "used_float"], "prepared")
    record["identity"] = program_identity(record)
    signature = Signature(
        {
            "keys": ValueType(("experts", "per", "key"), shape=(experts, per, width)),
            "rows": ValueType(("experts",), "int64", (experts,)),
        },
        {"prepared": ValueType(("experts", "per", "key"), shape=(experts, per, width))},
    )
    return validate(record, signature)


@lru_cache(maxsize=128)
def key_preparation_plan(experts, per, width, floating=False):
    from ..program import program_identity

    if min(experts, per, width) < 1:
        raise ProgramError("key dimensions must be positive")
    vector = ValueType(("experts", "key"), shape=(experts, width))
    keys_type = ValueType(("experts", "per", "key"), shape=(experts, per, width))
    rows_type = ValueType(("experts",), "float64" if floating else "int64", (experts,))

    def compile_part(name, inputs, outputs, instructions):
        record = {
            "name": name,
            "op_set": "kilora.ops",
            "op_set_version": 1,
            "inputs": {k: {"axes": list(v.axes), "dtype": v.dtype} for k, v in inputs.items()},
            "outputs": {k: {"axes": list(v.axes), "dtype": v.dtype} for k, v in outputs.items()},
            "bindings": {},
            "instructions": instructions,
        }
        record["identity"] = program_identity(record)
        return validate(record, Signature(inputs, outputs))

    steps = []
    for row in range(per):
        projections = list(range(row)) * 2
        chunks = [projections[i : i + 12] for i in range(0, len(projections), 12)] or [[]]
        for chunk_index, chunk in enumerate(chunks):
            first, last = chunk_index == 0, chunk_index == len(chunks) - 1
            inputs = {"keys": keys_type} if first else {"value": vector}
            inputs.update({f"q{i}": vector for i in sorted(set(chunk))})
            instructions = []

            def add(op, args, result, *, _instructions=instructions, **attributes):
                _instructions.append(
                    {
                        "id": len(_instructions),
                        "op": op,
                        "args": args,
                        "result": result,
                        "attributes": attributes,
                    }
                )
                return result

            v = add("select", ["keys"], "selected", axis="per", index=row) if first else "value"
            for i, q in enumerate(chunk):
                product = add("mul", [v, f"q{q}"], f"product{i}")
                dot = add("sum", [product], f"dot{i}", axis="key", keep=True)
                component = add("mul", [dot, f"q{q}"], f"component{i}")
                v = add("sub", [v, component], f"residual{i}")
            if last:
                norm = add("vector_norm", [v], "norm", axis="key", keep=True)
                floor = add("clamp_min", [norm], "floor", low=1e-12)
                v = add("div", [v, floor], "unit")
            steps.append(
                (
                    row,
                    first,
                    last,
                    compile_part(
                        f"prepare.keys.{row}.{chunk_index}", inputs, {v: vector}, instructions
                    ),
                    v,
                    tuple(inputs),
                )
            )

    inputs = {"keys": keys_type, "rows": rows_type, **{f"q{i}": vector for i in range(per)}}
    instructions = [
        {
            "id": 0,
            "op": "stack_rows",
            "args": [f"q{i}" for i in range(per)],
            "result": "stacked",
            "attributes": {"axis": "per", "before": "key"},
        },
        {
            "id": 1,
            "op": "arange_axis",
            "args": ["keys"],
            "result": "positions",
            "attributes": {"axis": "per"},
        },
        {"id": 2, "op": "gt", "args": ["rows", "positions"], "result": "used", "attributes": {}},
        {
            "id": 3,
            "op": "cast",
            "args": ["used"],
            "result": "used_float",
            "attributes": {"dtype": "float32"},
        },
        {
            "id": 4,
            "op": "mul",
            "args": ["stacked", "used_float"],
            "result": "prepared",
            "attributes": {},
        },
    ]
    if floating:
        instructions.insert(
            2,
            {
                "id": 2,
                "op": "cast",
                "args": ["positions"],
                "result": "float_positions",
                "attributes": {"dtype": "float64"},
            },
        )
        instructions[3]["args"][1] = "float_positions"
        for index, instruction in enumerate(instructions):
            instruction["id"] = index
    final = compile_part("prepare.keys.stack", inputs, {"prepared": keys_type}, instructions)
    return steps, final


def interpret_key_preparation(keys, rows=None):
    experts, per, width = keys.shape
    keys = keys.float()
    floating = rows is not None and rows.is_floating_point()
    if rows is None:
        rows = torch.full((experts,), per, dtype=torch.int64, device=keys.device)
    else:
        rows = rows.to(device=keys.device, dtype=torch.float64 if floating else torch.int64)
    if per <= 3 and not floating:
        return interpret(
            key_preparation_program(experts, per, width), {"keys": keys, "rows": rows}, {}
        )["prepared"]
    steps, final = key_preparation_plan(experts, per, width, floating)
    normalized = {}
    value = None
    for row, _first, last, program, output, names in steps:
        inputs = {
            name: keys if name == "keys" else value if name == "value" else normalized[name]
            for name in names
        }
        value = interpret(program, inputs, {})[output]
        if last:
            normalized[f"q{row}"] = value
    return interpret(final, {"keys": keys, "rows": rows, **normalized}, {})["prepared"]
