# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..program import AXES, ProgramError, Signature, ValueType


@dataclass(frozen=True)
class Part:
    name: str
    signature: Signature
    invariants: tuple[str, ...]
    singleton: tuple[tuple[str, str], ...] = ()

    def sized_signature(self, sizes: Mapping[str, int]) -> Signature:
        sizes = dict(sizes)
        if self.name == "transition":
            sizes["r1"] = sizes.get("experts", 1)
        elif self.name in ("keep", "gate", "freeze"):
            sizes["r1"] = 1

        def bind(name: str, value: ValueType, output: bool) -> ValueType:
            shape = tuple(
                1 if output and (name, axis) in self.singleton else sizes.get(axis, 1)
                for axis in value.axes
            )
            return ValueType(value.axes, value.dtype, shape)

        return Signature(
            {n: bind(n, t, False) for n, t in self.signature.inputs.items()},
            {n: bind(n, t, True) for n, t in self.signature.outputs.items()},
        )


def _v(*axes: str, dtype: str = "float32") -> ValueType:
    return ValueType(axes, dtype)


def _part(name, inputs, outputs, invariants=(), singleton=()):
    return Part(name, Signature(inputs, outputs), tuple(invariants), tuple(singleton))


_BTE = ("batch", "tokens", "experts")
_BTC = ("batch", "tokens", "cols")
PARTS = {
    "score": _part(
        "score",
        {
            "query": _v("batch", "tokens", "key"),
            "keys": _v("experts", "per", "key"),
            "temperature": _v(),
        },
        {"logits": _v(*_BTE)},
        ("finite logits",),
    ),
    "select": _part(
        "select",
        {"logits": _v(*_BTE), "noise": _v(*_BTE)},
        {"weights": _v(*_BTE)},
        ("simplex weights",),
    ),
    "keep": _part(
        "keep",
        {
            "carried": _v("batch", "tokens", "experts", "per"),
            "base_scale": _v("batch", "tokens", "experts", "r1"),
        },
        {"probability": _v("batch", "tokens", "experts", "per")},
        ("keep probability",),
    ),
    "carry": _part(
        "carry",
        {
            "change": _v(*_BTC),
            "carried": _v(*_BTC),
            "mask": _v(*_BTC),
            "base_read": _v(*_BTC),
            "uniforms": _v(*_BTC),
        },
        {"state": _v(*_BTC)},
        ("shape and dtype",),
    ),
    "gate": _part(
        "gate",
        {
            "coefficients": _v(*_BTC),
            "before": _v(*_BTC),
            "prior_history": _v(*_BTC),
            "has_history": _v(dtype="bool"),
            "up": _v("cols", "out"),
            "base_norm": _v("batch", "tokens"),
            "base_rms": _v("batch", "out"),
            "first_norm": _v("batch", "tokens"),
            "first_rms": _v("batch", "out"),
            "first_step": _v(dtype="bool"),
        },
        {
            "token": _v("batch", "tokens", "r1"),
            "channel": _v("batch", "r1", "out"),
            "history": _v(*_BTC),
        },
        ("gate parts",),
        (("token", "r1"), ("channel", "r1")),
    ),
    "transition": _part(
        "transition",
        {"last_weights": _v("batch", "tokens", "r1"), "matrix": _v("r1", "experts")},
        {"weights": _v(*_BTE)},
        ("simplex weights",),
    ),
    "entry": _part(
        "entry",
        {
            "reads": _v("batch", "tokens", "experts", "per"),
            "energy": _v("batch", "tokens"),
            "rows": _v("experts", dtype="int64"),
            "key_width": _v(),
            "tau": _v(),
            "epsilon": _v(),
            "temperature_in": _v(),
        },
        {"logits": _v(*_BTE), "temperature": _v()},
        ("finite logits", "temperature"),
    ),
    "budget": _part(
        "budget",
        {"weights": _v("batch", "tokens", "axes")},
        {"scales": _v("batch", "tokens", "axes")},
        ("axis weights",),
    ),
    "freeze": _part(
        "freeze",
        {
            "progress": _v(dtype="float64"),
            "threshold": _v(dtype="float64"),
            "enabled": _v(dtype="bool"),
            "history": _v(*_BTE),
            "last_weights": _v(*_BTE),
            "strength": _v("batch", "cols"),
            "gate_token": _v("batch", "tokens", "r1"),
            "gate_channel": _v("batch", "r1", "out"),
            "up": _v("cols", "out"),
        },
        {
            "frozen": _v(dtype="bool"),
            "weights": _v("batch", "experts"),
            "scales": _v("batch", "cols"),
            "up_gated": _v("batch", "cols", "out"),
        },
        ("simplex weights",),
    ),
    "iterate": _part(
        "iterate",
        {"previous_state": _v(*_BTC), "change": _v(*_BTC)},
        {"state": _v(*_BTC), "output": _v(*_BTC)},
        ("shape and dtype",),
    ),
    "live_write": _part(
        "live_write",
        {
            "slopes": _v("r1", "cols"),
            "shares": _v("experts"),
            "coefficients": _v("experts", "cols"),
            "multiplier": _v("cols"),
            "write_scale": _v(),
            "retention": _v(),
        },
        {"update": _v("experts", "cols")},
        ("finite update",),
    ),
    "guard": _part(
        "guard",
        {
            name: _v(dtype="float64")
            for name in (
                "gap_before",
                "gap_after",
                "spread_before",
                "spread_after",
                "noise",
                "gap_drop",
                "gap_noise_k",
                "gap_floor",
                "spread_low",
                "spread_high",
            )
        },
        {"refused": _v(dtype="bool")},
        ("boolean",),
    ),
}


def _part_value(part: str | Part) -> Part:
    if isinstance(part, Part):
        return part
    try:
        return PARTS[part]
    except (KeyError, TypeError) as exc:
        raise ProgramError(f"unknown part {part!r}") from exc


def check_invariants(
    part: str | Part, outputs: Mapping[str, torch.Tensor], *, sizes: Mapping[str, int] | None = None
) -> None:

    part = _part_value(part)

    def fail(invariant: str) -> None:
        raise ProgramError(f"part {part.name}: {invariant}")

    if not isinstance(outputs, Mapping) or set(outputs) != set(part.signature.outputs):
        fail("output shapes: missing or unknown output")
    axes = dict(sizes or {})
    for name, spec in part.signature.outputs.items():
        value = outputs[name]
        if not isinstance(value, torch.Tensor) or value.ndim != len(spec.axes):
            fail(f"output shapes: {name}")
        if value.dtype != getattr(torch, spec.dtype):
            fail(f"output dtype: {name}")
        for axis, extent in zip(spec.axes, value.shape, strict=True):
            if (name, axis) in part.singleton:
                if extent != 1:
                    fail(f"output shapes: {name} {axis}")
            elif extent < 1 or (axis in axes and axes[axis] != extent):
                fail(f"output shapes: {name} {axis}")
            else:
                axes[axis] = extent
    for invariant in part.invariants:
        if invariant == "simplex weights":
            w = outputs["weights"]
            if not bool(
                torch.isfinite(w).all() and (w >= 0).all() and ((w.sum(-1) - 1).abs() <= 1e-5).all()
            ):
                fail(invariant)
        elif invariant == "keep probability":
            q = outputs["probability"]
            if not bool(torch.isfinite(q).all() and (q > 0).all() and (q <= 1).all()):
                fail(invariant)
        elif (
            invariant == "finite logits" and not bool(torch.isfinite(outputs["logits"]).all())
        ) or (invariant == "finite update" and not bool(torch.isfinite(outputs["update"]).all())):
            fail(invariant)
        elif invariant == "temperature":
            t = outputs["temperature"]
            if not bool(torch.isfinite(t).all() and (t > 0).all()):
                fail(invariant)
        elif invariant == "gate parts":
            for name in ("token", "channel"):
                v = outputs[name]
                if not bool(torch.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
                    fail(invariant)
        elif invariant == "axis weights":
            v = outputs["scales"]
            if not bool(torch.isfinite(v).all() and (v >= 0).all()):
                fail(invariant)


def probe_inputs(part: str | Part, sizes: Mapping[str, int], seed: int) -> dict[str, torch.Tensor]:
    part = _part_value(part)
    if not isinstance(sizes, Mapping) or any(
        k not in AXES or type(v) is not int or v < 1 for k, v in sizes.items()
    ):
        raise ProgramError(f"part {part.name}: invalid probe sizes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for name, spec in part.sized_signature(sizes).inputs.items():
        shape = spec.shape
        if spec.dtype == "bool":
            value = torch.zeros(shape, dtype=torch.bool)
        elif spec.dtype == "int64":
            value = torch.ones(shape, dtype=torch.int64)
        else:
            value = torch.randn(shape, generator=generator, dtype=getattr(torch, spec.dtype)) * 0.1
        if name in {
            "temperature",
            "temperature_in",
            "energy",
            "key_width",
            "tau",
            "epsilon",
            "base_scale",
            "base_norm",
            "base_rms",
            "spread_before",
            "spread_after",
        }:
            value = value.abs() + 1
        if name in {"weights", "last_weights", "shares"} or (
            part.name == "freeze" and name in {"history", "last_weights"}
        ):
            value = torch.softmax(value, dim=-1)
        if name == "matrix":
            value = torch.softmax(value, dim=-1)
        if name == "mask":
            value = torch.ones_like(value)
        if name == "uniforms":
            value = torch.rand(shape, generator=generator)
        result[name] = value
    return result
