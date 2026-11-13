# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import torch

from ..program import ProgramError, ValidatedProgram, program_identity
from .defaults import _record
from .identities import NATIVE_IDENTITIES


@dataclass(frozen=True)
class PartOverride:
    variant: str
    program: Any = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    bindings: Mapping[str, torch.Tensor] = field(default_factory=dict)


_OVERRIDES: ContextVar[list[tuple[int | None, Mapping[str, PartOverride]]] | None] = ContextVar(
    "kilora_ir_overrides", default=None
)
_USED: ContextVar[set[tuple[int | None, str]] | None] = ContextVar("kilora_ir_used", default=None)
_SHARED_SELECT: ContextVar[bool] = ContextVar("kilora_ir_shared_select", default=False)
_PARENT: ContextVar[tuple | None] = ContextVar("kilora_ir_override_parent", default=None)


@contextlib.contextmanager
def override_scope():
    from ..compute import _IR_ACTIVE_OVERRIDES

    token = _IR_ACTIVE_OVERRIDES.set(_IR_ACTIVE_OVERRIDES.get() + 1)
    try:
        yield
    finally:
        _IR_ACTIVE_OVERRIDES.reset(token)


@contextlib.contextmanager
def interpret_all():
    from ..compute import _IR_INTERPRET_ALL

    token = _IR_INTERPRET_ALL.set(True)
    try:
        yield
    finally:
        _IR_INTERPRET_ALL.reset(token)


def interpreting_all():
    from ..compute import _IR_INTERPRET_ALL

    return _IR_INTERPRET_ALL.get()


@cache
def default_identity(part, variant):
    return NATIVE_IDENTITIES.get((part, variant))


def lowering(part, variant, program=None, parameters=None):

    record = _record(part, variant)
    if program_identity(record) != record["identity"]:
        raise ProgramError(f"built-in program identity mismatch for {part}.{variant}")
    identity = record["identity"] if program is None else program.identity
    if interpreting_all() or identity is None or identity != default_identity(part, variant):
        return None
    for name, value in (parameters or {}).items():
        binding = record["bindings"].get(name)
        if binding is None or value != binding.get("value"):
            return None
    return _PART_LOWERINGS.get((part, variant), "native")


def _projection(inputs):
    return {"projection": inputs["query"] @ inputs["keys"]}


def _weighted(inputs):
    return {"weighted": inputs["projection"] * inputs["weights"].unsqueeze(-1)}


def _carry_drop(inputs):
    return {"state": inputs["mask"] * inputs["change"]}


def _carry_dropped(inputs):
    return {
        "state": inputs["carried"] + (1 - inputs["mask"]) * (inputs["change"] - inputs["carried"])
    }


def _path_stop(inputs):
    return {"result": torch.where(inputs["stopped"], inputs["previous"], inputs["state"])}


def _path_delta(inputs):
    return {"delta": inputs["coefficients"] @ inputs["up"]}


def _path_add(inputs):
    return {"output": inputs["base"] + inputs["delta"]}


def _accumulate(inputs):
    return {
        "state": inputs["previous_state"] + inputs["change"],
        "output": inputs["previous_state"] + inputs["change"],
    }


def _output(inputs):
    return {"output": inputs["base"] + inputs["coefficients"] @ inputs["up"]}


_PART_LOWERINGS = {
    ("score", "projection"): _projection,
    ("score", "weighted"): _weighted,
    ("carry", "drop"): _carry_drop,
    ("carry", "dropped"): _carry_dropped,
    ("iterate", "path_stop"): _path_stop,
    ("iterate", "path_delta"): _path_delta,
    ("iterate", "path_add"): _path_add,
    ("iterate", "accumulate"): _accumulate,
    ("iterate", "output"): _output,
}


def interpreted_part(sets, *, exclude=()):
    first = None
    for setting in sets:
        for part, override in getattr(setting, "ir_parts", {}).items():
            if part in exclude:
                continue
            if not isinstance(override, PartOverride):
                raise TypeError(f"{part} override must be a PartOverride")
            record = _record(part, override.variant)
            for name in override.parameters:
                if name not in record["bindings"] and not (
                    (part == "keep" and name == "DROPOUT_DELTA")
                    or (
                        part in {"guard", "live_write", "budget", "freeze"}
                        and name in record["inputs"]
                        and not record["inputs"][name]["axes"]
                    )
                ):
                    raise ValueError(f"unknown parameter {name!r} for {part}.{override.variant}")
            if override.program is not None:
                if not isinstance(override.program, ValidatedProgram):
                    raise TypeError(f"{part} replacement must be a validated program")
                from .parts import PARTS

                if part in {"budget", "freeze", "guard", "live_write"}:
                    from .defaults import default_program

                    default = default_program(part, override.variant)
                    expected_types = {
                        name: default.value_types[name]
                        for name in (*record["inputs"], *record["outputs"])
                    }
                    outputs = set(default.outputs)
                else:
                    signature = PARTS[part].signature
                    expected_types = {**signature.inputs, **signature.outputs}
                    outputs = set(signature.outputs)
                for name, expected in expected_types.items():
                    actual = override.program.value_types.get(name)
                    if (
                        actual is None
                        or expected is None
                        or (actual.axes, actual.dtype) != (expected.axes, expected.dtype)
                    ):
                        raise ValueError(f"{part} replacement has incompatible value {name!r}")
                if set(override.program.outputs) != outputs:
                    raise ValueError(f"{part} replacement has incompatible outputs")
            if first is None and (
                override.program is not None
                or bool(override.parameters)
                or lowering(part, override.variant, override.program, override.parameters) is None
            ):
                first = part
    if first is not None:
        return first
    if interpreting_all() and sets:
        return "score"
    return None


@contextlib.contextmanager
def active_overrides(overrides, sets=None, *, shared=False, exclude=()):
    groups = (
        [(id(s), {k: v for k, v in s.ir_parts.items() if k not in exclude}) for s in sets]
        if sets is not None
        else [(None, overrides)]
    )
    from ..compute import _IR_ACTIVE_OVERRIDES

    active_token = _IR_ACTIVE_OVERRIDES.set(_IR_ACTIVE_OVERRIDES.get() + 1)
    parent_token = _PARENT.set((_OVERRIDES.get(), _USED.get(), _PARENT.get()))
    token = _OVERRIDES.set(groups)
    used_token = _USED.set(set())
    shared_token = _SHARED_SELECT.set(shared)
    try:
        yield
        used = _USED.get()
        for key, mapping in groups:
            for part, override in mapping.items():
                if (key, part) not in used:
                    raise NotImplementedError(
                        f"{part}.{override.variant} was not invoked by the interpreted driver"
                    )
    finally:
        _SHARED_SELECT.reset(shared_token)
        _USED.reset(used_token)
        _OVERRIDES.reset(token)
        _PARENT.reset(parent_token)
        _IR_ACTIVE_OVERRIDES.reset(active_token)


def dispatch_active():
    return _USED.get() is not None


def peek_override(part):
    groups, parent = _OVERRIDES.get(), _PARENT.get()
    while True:
        if groups and len(groups) == 1 and part in groups[0][1]:
            return groups[0][1][part]
        if parent is None:
            return None
        groups, _, parent = parent


def current_override(part, variant, setting=None):
    groups, used, parent = _OVERRIDES.get(), _USED.get(), _PARENT.get()
    while True:
        groups = groups or []
        if (
            setting is None
            and groups
            and (len(groups) == 1 or (part == "select" and _SHARED_SELECT.get()))
        ):
            key, mapping = groups[0]
        else:
            key = id(setting)
            mapping = next((m for k, m in groups if k == key), {})
        override = mapping.get(part)
        matches_select = (
            part == "select"
            and override is not None
            and override.program is not None
            and variant in {"token_sparsemax", "token_top1", "tensor_sparsemax", "tensor_top1"}
        )
        matches_carry = (
            part == "carry"
            and override is not None
            and override.program is not None
            and variant in {"drop", "dropped"}
        )
        matches_gate = (
            part == "gate"
            and override is not None
            and override.variant == "adaptive"
            and variant == "adaptive_active"
        )
        if override is not None and (
            override.variant == variant or matches_select or matches_carry or matches_gate
        ):
            if used is not None:
                used.add((key, part))
                if part == "select" and _SHARED_SELECT.get():
                    for other_key, other_mapping in groups:
                        if same_override(override, other_mapping.get(part)):
                            used.add((other_key, part))
            return override
        if parent is None:
            return None
        groups, used, parent = parent


def same_override(left, right):
    if left is None or right is None:
        return left is right
    return (
        left.variant == right.variant
        and getattr(left.program, "identity", None) == getattr(right.program, "identity", None)
        and left.parameters == right.parameters
        and left.bindings.keys() == right.bindings.keys()
        and all(torch.equal(left.bindings[k], right.bindings[k]) for k in left.bindings)
    )
