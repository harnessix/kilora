# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

SOFTMAX_SCOPES = ("shared", "per_set")
ROUTING_SCOPES = ("token", "tensor")
ROUTING_SELECTS = ("sparsemax", "top1")
CARRIES = ("drop", "dropped")


@dataclass
class KiLoRAConfig:
    target_modules: Sequence[str] = field(default_factory=lambda: ["*"])
    num_experts: int = 16
    routes: int = 4
    rank: int = 1
    router_dim: int = 32
    router_seed: int = 0
    depth: int = 1
    train_depth: tuple[int, int] | None = (1, 4)
    last_step_training: bool = False
    routing_noise: float = 0.0
    softmax_scope: str = "shared"
    routing_scope: str = "token"
    routing_select: str = "sparsemax"
    carry: str = "dropped"
    m_in: int = 4
    m_out: int = 4
    iteration: str = "accumulate"
    train_iteration: str = "accumulate"
    fixed_point_stop: bool = False
    output_gate: str = "separable"
    carry_across_steps: bool = False
    seed: int = 0
    aggregation: str = "accumulate"

    def __post_init__(self) -> None:
        self.target_modules = list(self.target_modules)
        if not self.target_modules:
            raise ValueError("target_modules must list at least one pattern")
        for name in ("num_experts", "routes", "rank", "router_dim", "m_in", "m_out"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if max(self.m_in, self.m_out) > self.router_dim:
            raise ValueError("m_in and m_out must be at most router_dim")
        if self.output_gate not in ("none", "separable", "neuron", "lowrank"):
            raise ValueError(f"unknown output_gate {self.output_gate!r}")
        if self.iteration not in ("accumulate", "path"):
            raise ValueError(f"unknown iteration {self.iteration!r}")
        if self.train_iteration not in ("accumulate", "path", "mix"):
            raise ValueError(f"unknown train_iteration {self.train_iteration!r}")
        check_depth(self.depth)
        if self.train_depth is not None:
            self.train_depth = check_depth_range(self.train_depth)
        if self.softmax_scope not in SOFTMAX_SCOPES:
            raise ValueError(
                f"softmax_scope must be one of {SOFTMAX_SCOPES}, got {self.softmax_scope!r}"
            )
        if self.carry not in CARRIES:
            raise ValueError(f"carry must be one of {CARRIES}, got {self.carry!r}")
        if self.routing_scope not in ROUTING_SCOPES:
            raise ValueError(
                f"routing_scope must be one of {ROUTING_SCOPES}, got {self.routing_scope!r}"
            )
        if self.routing_select not in ROUTING_SELECTS:
            raise ValueError(
                f"routing_select must be one of {ROUTING_SELECTS}, got {self.routing_select!r}"
            )
        if self.aggregation not in ("accumulate", "replace"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if not math.isfinite(self.routing_noise) or self.routing_noise < 0:
            raise ValueError(f"routing_noise must be finite and >= 0, got {self.routing_noise}")


def check_depth(depth: int) -> int:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError(f"depth must be an integer >= 0, got {depth!r}")
    return depth


def check_depth_range(value: Sequence[int]) -> tuple[int, int]:
    low, high = (int(v) for v in value)
    check_depth(low)
    check_depth(high)
    if low > high:
        raise ValueError(f"train_depth must be (low, high) with low <= high, got {value!r}")
    return (low, high)
