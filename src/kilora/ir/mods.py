# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import replace

import torch

from ..program import ProgramError, interpret, validate
from .dispatch import PartOverride
from .parts import PARTS, check_invariants, probe_inputs

_PROBE_SIZES = {"batch": 2, "tokens": 3, "experts": 4, "per": 2, "cols": 3, "key": 3, "out": 3}
_UNAVAILABLE = {"budget", "freeze", "guard", "live_write"}


def apply_mod_parts(sets, sections):
    result = [replace(s, ir_parts=dict(s.ir_parts)) for s in sets]
    for section in sections:
        meta = section.meta
        if section.kind not in {"program", "params"}:
            continue
        if section.kind == "program" and "part" not in meta:
            continue
        part = meta.get("part")
        if not isinstance(part, str) or part not in PARTS:
            raise ProgramError(f"unknown part {part!r}")
        if part in _UNAVAILABLE:
            raise ProgramError(f"part {part}: overrides are not yet supported")
        scope = meta.get("scope")
        if not isinstance(scope, str) or not scope:
            raise ProgramError(f"part {part}: scope must be a set name or '*'")
        matched = [s for s in result if scope == "*" or s.name == scope]
        if not matched:
            raise ProgramError(f"part {part}: scope {scope!r} matches no set")
        if section.kind == "params":
            values = meta.get("params")
            allowed = {
                "gate": {"kappa", "coherence"},
                "keep": {"decay_base", "DROPOUT_DELTA"},
            }.get(part, set())
            if not isinstance(values, dict) or not values:
                raise ProgramError(f"part {part}: params must be a nonempty record")
            for name, value in values.items():
                if name not in allowed:
                    raise ProgramError(f"part {part}: unknown parameter {name!r}")
                if name == "coherence":
                    if type(value) is not bool:
                        raise ProgramError(f"part {part}: parameter {name} must be boolean")
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    or (name == "decay_base" and value > 1)
                ):
                    raise ProgramError(f"part {part}: parameter {name} is out of range")
            override = PartOverride(
                "adaptive" if part == "gate" else "default", parameters=dict(values)
            )
        else:
            variants = {
                "select": "token_sparsemax",
                "score": "keys",
                "carry": "drop",
                "gate": "adaptive",
                "keep": "default",
                "iterate": "accumulate",
            }
            if part not in variants:
                raise ProgramError(f"part {part}: replacement is not yet supported by the driver")
            variant = meta.get("variant", variants[part])
            if variant != variants[part]:
                raise ProgramError(f"part {part}: unsupported replacement variant {variant!r}")
            program = validate(meta, PARTS[part].sized_signature(_PROBE_SIZES))
            bindings = {}
            for name, spec in meta.get("bindings", {}).items():
                if spec.get("kind") != "embedded":
                    raise ProgramError(f"part {part}: binding {name} must be embedded")
                value = section.data.get(name)
                if (
                    not isinstance(value, torch.Tensor)
                    or list(value.shape) != spec.get("shape")
                    or str(value.dtype) != "torch." + str(spec.get("dtype"))
                ):
                    raise ProgramError(f"part {part}: binding {name} has wrong shape or dtype")
                bindings[name] = value
            outputs = interpret(program, probe_inputs(part, _PROBE_SIZES, 119), bindings)
            check_invariants(part, outputs, sizes=_PROBE_SIZES)
            override = PartOverride(variant, program=program, bindings=bindings)
        for setting in matched:
            if part in setting.ir_parts:
                raise ProgramError(f"part {part}: duplicate override for set {setting.name!r}")
            setting.ir_parts[part] = override
    return result
