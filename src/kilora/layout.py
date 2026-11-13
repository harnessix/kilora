# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Layout:
    id: str
    targets: tuple[str, ...]
    widths: tuple[int, ...]
    in_widths: tuple[int, ...]
    experts: int
    per_expert: int
    key_dim: int
    write_scale: float = 0.01

    def __post_init__(self) -> None:
        if len(self.widths) != len(self.targets) or len(self.in_widths) != len(self.targets):
            raise ValueError("a layout needs one input and one output width per target")

    def width_of(self, target: str) -> int:
        return self.widths[self.targets.index(target)]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "targets": list(self.targets),
            "widths": list(self.widths),
            "in_widths": list(self.in_widths),
            "experts": self.experts,
            "per_expert": self.per_expert,
            "key_dim": self.key_dim,
            "write_scale": self.write_scale,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Layout:
        return cls(
            id=str(d["id"]),
            targets=tuple(d["targets"]),
            widths=tuple(int(w) for w in d["widths"]),
            in_widths=tuple(int(w) for w in d["in_widths"]),
            experts=int(d["experts"]),
            per_expert=int(d["per_expert"]),
            key_dim=int(d["key_dim"]),
            write_scale=float(d.get("write_scale", 0.01)),
        )

    def fits(self, other: Layout) -> bool:
        mine, theirs = self.to_dict(), other.to_dict()
        mine.pop("write_scale"), theirs.pop("write_scale")
        return mine == theirs


State = dict[str, dict[str, torch.Tensor]]


def flat(state: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for parts in state.values() for v in parts.values()])


def relative_change(new: State, old: State, layout: Layout) -> float:
    a, b = flat(new), flat(old)
    floor = layout.write_scale * math.sqrt(b.numel())
    return float((a - b).norm() / max(float(b.norm()), floor))
