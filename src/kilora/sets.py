# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .compute import DEFAULT_TAU, new_gate, orthonormal_keys
from .errors import IncompatibleAdapterSetError

PART_TYPES = ("lora",)
AGGREGATIONS = ("accumulate", "replace")
FORMS = ("state", "input")


@dataclass
class KeyedExperts:
    keys_in: torch.Tensor
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    keys_out: torch.Tensor | None = None
    rows_in: torch.Tensor | None = None
    rows_out: torch.Tensor | None = None
    gate: dict[str, torch.Tensor] | None = None
    exact: dict[int, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    ranks: tuple[int, ...] | None = None
    routing_down: torch.Tensor | None = None
    routing_up: torch.Tensor | None = None
    routing_source: tuple | None = None
    exact_unavailable: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.keys_out is None:
            self.keys_out = self.keys_in.clone()
        e = self.keys_in.shape[0]
        if self.rows_in is None:
            self.rows_in = torch.full((e,), self.keys_in.shape[1], dtype=torch.long)
        if self.rows_out is None:
            self.rows_out = torch.full((e,), self.keys_out.shape[1], dtype=torch.long)
        self.rows_in, self.rows_out = self.rows_in.long(), self.rows_out.long()

    @property
    def has_lora(self) -> bool:
        return self.down is not None

    @property
    def in_features(self) -> int | None:
        return self.down.shape[-1] if self.down is not None else None

    @property
    def out_features(self) -> int:
        return self.up.shape[-1]


@dataclass
class AdapterSet:
    name: str
    part_type: tuple[str, ...]
    experts: dict[str, Any]
    routers: dict[int, torch.Tensor] = field(default_factory=dict)
    aggregation: str = "accumulate"
    strength: float = 1.0
    prior: float = 1.0
    imported_from: dict[str, Any] | None = None
    frozen: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    form: str = "state"
    log_tau: float = math.log(DEFAULT_TAU)
    key_source: str = "random"
    learned_cross: dict[str, torch.Tensor] | None = None
    learned_cross_zero: bool = False
    ir_parts: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        check_set_name(self.name)
        self.part_type = tuple(self.part_type)
        if not self.part_type or any(p not in PART_TYPES for p in self.part_type):
            raise ValueError(f"unknown part types {self.part_type!r}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.form not in FORMS:
            raise ValueError(f"unknown form {self.form!r}")
        if not self.experts:
            raise ValueError(f"adapter set {self.name!r} has no target layers")
        check_strength(self.strength)
        check_prior(self.prior)
        for target, ex in self.experts.items():
            if not isinstance(ex, KeyedExperts):
                raise TypeError(f"{target}: expected KeyedExperts")
            if not ex.has_lora:
                raise ValueError(f"{target}: the experts have no LoRA parts")
            if ex.has_lora and (
                ex.down.dim() != 4 or ex.up is None or ex.down.shape[:3] != ex.up.shape[:3]
            ):
                raise ValueError(
                    f"{target}: down must be [E, routes, rank, in], up [E, routes, rank, out]"
                )
            router = self.routers.get(ex.out_features)
            if router is None:
                raise ValueError(f"{target}: no router for output width {ex.out_features}")
            for keys, rows in ((ex.keys_in, ex.rows_in), (ex.keys_out, ex.rows_out)):
                if keys.dim() != 3 or (self.form != "state" and keys.shape[-1] != router.shape[0]):
                    raise ValueError(f"{target}: keys must be [experts, M, router_dim]")
                if keys.shape[0] != ex.down.shape[0] or rows.shape != (keys.shape[0],):
                    raise ValueError(f"{target}: one key and one row count per expert")
                if int(rows.min()) < (0 if self.form == "state" else 1) or int(rows.max()) > min(
                    keys.shape[1], keys.shape[2]
                ):
                    raise ValueError(f"{target}: row counts must be 1 to min(M, router_dim)")

    @property
    def target_modules(self) -> list[str]:
        return list(self.experts)

    def _first(self) -> KeyedExperts:
        return next(iter(self.experts.values()))

    @property
    def has_lora(self) -> bool:
        return "lora" in self.part_type

    @property
    def num_experts(self) -> int:
        return self._first().keys_in.shape[0]

    @property
    def routes(self) -> int:
        return self._first().down.shape[1] if self.has_lora else 1

    @property
    def rank(self) -> int:
        return self._first().down.shape[2] if self.has_lora else 0

    @property
    def router_dim(self) -> int:
        return next(iter(self.routers.values())).shape[0]

    def router_for(self, target: str) -> torch.Tensor:
        return self.routers[self.experts[target].out_features]


def routes_alone(extra: dict[str, Any]) -> bool:
    return bool(extra.get("live") or extra.get("learned"))


def check_set_name(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ValueError(f"set name must be a non-empty string of at most 64 characters: {name!r}")
    if not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError(f"set name may use letters, digits, '-', '_' and '.' only: {name!r}")
    return name


def check_strength(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"strength must be a finite number, got {value}")
    return value


def check_prior(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"prior must be a finite number >= 0, got {value}")
    return value


def make_router(router_dim: int, width: int, seed: int) -> torch.Tensor:

    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(width))
    return torch.randn(router_dim, width, generator=gen, dtype=torch.float32) / math.sqrt(width)


def new_keyed_set(
    name: str,
    shapes: dict[str, tuple[int, int]],
    *,
    num_experts: int,
    routes: int,
    rank: int,
    router_dim: int,
    router_seed: int,
    seed: int,
    routers: dict[int, torch.Tensor] | None = None,
    aggregation: str = "accumulate",
    form: str = "state",
    m_in: int = 4,
    m_out: int = 4,
    keys: dict[str, dict[str, torch.Tensor]] | None = None,
    key_source: str = "random",
    output_gate: str = "none",
) -> AdapterSet:
    routers = dict(routers or {})
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    experts = {}
    for target, (n_in, n_out) in shapes.items():
        if form == "state":
            n_in = n_out
        if n_out not in routers:
            routers[n_out] = make_router(router_dim, n_out, router_seed)
        if max(m_in, m_out) > router_dim:
            raise ValueError(f"key rows ({m_in}, {m_out}) exceed router_dim {router_dim}")
        parts = {
            "keys_in": orthonormal_keys(torch.randn(num_experts, m_in, router_dim, generator=gen)),
            "keys_out": orthonormal_keys(
                torch.randn(num_experts, m_out, router_dim, generator=gen)
            ),
        }
        if form != "state" and keys is not None and target in keys:
            given = keys[target]
            parts = {
                "keys_in": orthonormal_keys(given["keys_in"].float(), given.get("rows_in")),
                "keys_out": orthonormal_keys(given["keys_out"].float(), given.get("rows_out")),
                "rows_in": given.get("rows_in"),
                "rows_out": given.get("rows_out"),
            }
            if parts["keys_in"].shape[0] != num_experts:
                raise ValueError(f"{target}: the given keys have another number of experts")
        down = torch.randn(num_experts, routes, rank, n_in, generator=gen) / math.sqrt(n_in)
        experts[target] = KeyedExperts(
            down=down,
            up=torch.zeros(num_experts, routes, rank, n_out),
            gate=new_gate(output_gate, num_experts, n_out, gen, num_experts * routes * rank),
            **parts,
        )
    return AdapterSet(
        name=name,
        part_type=("lora",),
        experts=experts,
        routers=routers,
        aggregation=aggregation,
        form=form,
        key_source=key_source if keys is not None else "random",
        log_tau=math.log(8.0 * m_in / router_dim),
    )


def needs_state_keys(s: AdapterSet) -> bool:
    if s.key_source != "random" or s.extra.get("trained_with_kilora") is not False:
        return False
    if s.num_experts < 2:
        return False
    for ex in s.experts.values():
        for t in (ex.down, ex.up):
            if t is not None and not all(torch.equal(t[0], t[e]) for e in range(1, t.shape[0])):
                return True
    return False


def set_identity(s: AdapterSet) -> str:
    import hashlib
    import json

    record = s.imported_from or {}
    sources = record.get("sources")
    if sources:
        text = json.dumps([[x.get("sha256"), x.get("weight")] for x in sources])
    elif record.get("digest"):
        text = str(record["digest"])
    else:
        h = hashlib.sha256()
        for target in sorted(s.experts):
            ex = s.experts[target]
            for t in (ex.down, ex.up):
                if t is not None:
                    h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
        text = h.hexdigest()
    return hashlib.sha256(f"{s.name}|{text}".encode()).hexdigest()


def check_compatible(first: AdapterSet, other: AdapterSet) -> None:
    if first.form != "state" and other.form != "state" and first.router_dim != other.router_dim:
        raise IncompatibleAdapterSetError(
            f"adapter sets {first.name!r} and {other.name!r} differ in router_dim: "
            f"{first.router_dim} != {other.router_dim}"
        )
