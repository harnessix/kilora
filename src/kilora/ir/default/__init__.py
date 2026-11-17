# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files

import torch

from ...program import ProgramError, Signature, ValueType, interpret, program_identity, validate
from ..parts import PARTS

_PARTS = ("transition", "entry", "budget", "freeze")
_VARIANTS = {
    ("entry", "chance"): "entry.chance",
    ("freeze", "step"): "freeze_step",
    ("budget", "auto_sum"): "budget_auto_sum",
    ("budget", "auto"): "budget_auto",
    ("budget", "scalar_sum"): "budget_scalar_sum",
    ("budget", "scalar"): "budget_scalar",
    ("budget", "auto_add_scalar"): "budget_auto_add_scalar",
    ("budget", "auto_seed_scalar"): "budget_auto_seed_scalar",
    ("budget", "auto_scalar"): "budget_auto_scalar",
}


def default_program(
    part: str, variant: str | None = None, *, sizes: Mapping[str, int] | None = None
):
    if (part, variant) == ("entry", "derived_basis"):
        size = sizes or {}
        per, key = size.get("per", 1), size.get("key", 1)
        rank = min(per, key)
        width = min(size.get("width", 8), rank)
        signature = Signature(
            {
                "block": ValueType(("per", "key"), shape=(per, key)),
                "indices": ValueType(("r1",), "int64", (rank,)),
                "width_indices": ValueType(("r2",), "int64", (width,)),
                "max_dim": ValueType(()),
                "eps": ValueType(()),
                "limit": ValueType((), "int64"),
                "zero": ValueType(()),
            },
            {
                "basis": ValueType(("r2", "key"), shape=(width, key)),
                "count": ValueType((), "int64"),
            },
        )
        data = json.loads(files(__package__).joinpath("entry.derived_basis.json").read_text())
        return validate(data, signature)
    if part == "entry" and variant in ("from_input", "chance_from_input"):
        size = sizes or {}
        experts, per, key = (size.get(axis, 1) for axis in ("experts", "per", "key"))
        batch, tokens = (size.get(axis, 1) for axis in ("batch", "tokens"))
        original = PARTS["entry"].sized_signature(size)
        inputs = {k: v for k, v in original.inputs.items() if k not in ("reads", "energy")}
        inputs.update(
            x=ValueType(("batch", "tokens", "key"), shape=(batch, tokens, key)),
            keys=ValueType(("experts", "per", "key"), shape=(experts, per, key)),
        )
        data = json.loads(files(__package__).joinpath(f"entry.{variant}.json").read_text())
        return validate(data, Signature(inputs, original.outputs))
    if (part, variant) == ("transition", "cosine_row"):
        size = sizes or {}
        experts, per, key = (size.get(axis, 1) for axis in ("experts", "per", "key"))
        signature = Signature(
            {
                "down": ValueType(("experts", "per", "key"), shape=(experts, per, key)),
                "up": ValueType(("r1", "key"), shape=(per, key)),
            },
            {"row": ValueType(("experts",), shape=(experts,))},
        )
        data = json.loads(files(__package__).joinpath("transition.cosine_row.json").read_text())
        return validate(data, signature)
    if part not in _PARTS or (
        variant not in (None, "default") and (part, variant) not in _VARIANTS
    ):
        raise ProgramError(f"unknown default program {part!r}, variant {variant!r}")
    name = _VARIANTS.get((part, variant), part)
    data = json.loads(files(__package__).joinpath(f"{name}.json").read_text(encoding="utf-8"))
    sizes = dict(sizes or {})
    if name == "freeze":
        if data.get("identity") != program_identity(data):
            raise ProgramError("built-in freeze program identity mismatch")
        per = sizes.get("per", 1)
        sizes.setdefault("cols", sizes.get("experts", 1) * per)
        data["instructions"][4]["attributes"]["repeats"] = per
        data["identity"] = program_identity(data)
    if name in _VARIANTS.values():

        def values(declarations):
            return {
                n: ValueType(
                    tuple(d["axes"]), d["dtype"], tuple(sizes.get(a, 1) for a in d["axes"])
                )
                for n, d in declarations.items()
            }

        signature = Signature(values(data["inputs"]), values(data["outputs"]))
    else:
        signature = PARTS[part].sized_signature(sizes)
    return validate(data, signature)


def prepare_cosine_transition(down, up, per_expert):
    if down.ndim != 2 or up.shape != down.shape or per_expert < 1 or down.shape[0] % per_expert:
        raise ValueError("expected equal square-input weight layouts and a valid per_expert")
    down = down.detach().float().reshape(-1, per_expert, down.shape[-1])
    up = up.detach().float().reshape(-1, per_expert, up.shape[-1])
    program = default_program(
        "transition",
        "cosine_row",
        sizes={
            "experts": down.shape[0],
            "per": per_expert,
            "key": down.shape[-1],
        },
    )
    return torch.stack([interpret(program, {"down": down, "up": row}, {})["row"] for row in up])


def prepare_derived_bases(weight, per_expert, width=8):
    if weight.ndim != 2 or per_expert < 1 or weight.shape[0] % per_expert or width < 1:
        raise ValueError("expected a matrix, positive width and valid per_expert")
    blocks = weight.detach().float().reshape(-1, per_expert, weight.shape[-1])
    key = blocks.shape[-1]
    rank = min(per_expert, key)
    width = min(width, rank)
    program = default_program(
        "entry", "derived_basis", sizes={"per": per_expert, "key": key, "width": width}
    )
    device = blocks.device
    fixed = {
        "indices": torch.arange(rank, device=device),
        "width_indices": torch.arange(width, device=device),
        "max_dim": torch.tensor(float(max(per_expert, key)), device=device),
        "eps": torch.tensor(torch.finfo(torch.float32).eps, device=device),
        "limit": torch.tensor(width, dtype=torch.int64, device=device),
        "zero": torch.tensor(0.0, device=device),
    }
    results = [interpret(program, {**fixed, "block": block}, {}) for block in blocks]
    return torch.stack([result["basis"] for result in results]), torch.stack(
        [result["count"] for result in results]
    )
