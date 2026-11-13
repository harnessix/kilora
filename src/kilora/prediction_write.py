# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

LEARN_MODES = ("measure", "prediction")
DEFAULT_LEARN = "measure"
SOURCES = ("depth", "cross", "hier", "clamp", "images")
DEFAULT_WEIGHTS = {"depth": 0.25, "cross": 0.25, "hier": 0.25, "clamp": 1.0, "images": 1.0}
DEFAULT_PRIOR_SCALE = 1.0
NO_TARGET = "no target: self-consistency terms only"


def check_learn(value: Any) -> str:
    if value not in LEARN_MODES:
        raise ValueError(f"learn must be one of {', '.join(LEARN_MODES)}")
    return str(value)


def check_weights(value: Mapping[str, Any]) -> dict[str, float]:
    out = dict(DEFAULT_WEIGHTS)
    for name, weight in dict(value).items():
        if name not in SOURCES:
            raise ValueError(f"unknown prediction source {name!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("prediction weights must be numbers")
        weight = float(weight)
        if not weight >= 0 or weight == float("inf"):
            raise ValueError("prediction weights must be finite and nonnegative")
        out[name] = weight
    return out


def check_prior_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("the prior scale must be a positive number")
    value = float(value)
    if not value > 0 or value == float("inf"):
        raise ValueError("the prior scale must be a positive number")
    return value


def session_learn(session: Any) -> str:
    return getattr(session, "learn", DEFAULT_LEARN) or DEFAULT_LEARN


def session_weights(session: Any) -> dict[str, float]:
    return check_weights(getattr(session, "prediction_weights", None) or {})


def session_prior_scale(session: Any) -> float:
    return float(getattr(session, "prediction_prior", DEFAULT_PRIOR_SCALE) or DEFAULT_PRIOR_SCALE)


@dataclass
class Term:
    source: str
    node: Hashable
    error: torch.Tensor
    response: torch.Tensor
    precision: torch.Tensor | float
    step: Hashable = None


@dataclass
class NodePrior:
    c: torch.Tensor
    mu: torch.Tensor
    inv_cov: torch.Tensor


@dataclass
class SolveResult:
    delta: dict[Hashable, torch.Tensor]
    counts: dict[str, int] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)
    nodes: dict[str, int] = field(default_factory=dict)
    energy: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def has_target(self) -> bool:
        return self.counts.get("clamp", 0) > 0

    def note(self, norm: float | None = None) -> str:
        parts = [" source=prediction"]
        for name in SOURCES:
            parts.append(
                f"eps_{name}_steps={self.steps.get(name, 0)}"
                f" eps_{name}_nodes={self.nodes.get(name, 0)}"
                f" pi_eps2_{name}={self.energy.get(name, 0.0):.4g}"
            )
        parts.append(f"solve_s={self.seconds:.4g}")
        if norm is not None:
            parts.append(f"write_norm={float(norm):.4g}")
        line = " ".join(parts) + "."
        if not self.has_target:
            line += f" {NO_TARGET}."
        return line


def _groups(term: Term):
    error = torch.as_tensor(term.error).double()
    response = torch.as_tensor(term.response).double()
    if error.dim() == 1:
        error, response = error.unsqueeze(0), response.unsqueeze(0)
    pi = torch.as_tensor(term.precision, dtype=torch.float64).reshape(-1)
    if pi.numel() == 1 and error.shape[0] > 1:
        pi = pi.expand(error.shape[0])
    return error, response, pi


def solve_map(
    terms: Iterable[Term],
    priors: Mapping[Hashable, NodePrior],
    weights: Mapping[str, float] | None = None,
) -> SolveResult:
    started = time.perf_counter()
    weights = check_weights(weights or {})
    h: dict[Hashable, torch.Tensor] = {}
    g: dict[Hashable, torch.Tensor] = {}
    result = SolveResult({})
    steps: dict[str, set] = {name: set() for name in SOURCES}
    nodes: dict[str, set] = {name: set() for name in SOURCES}
    sums: dict[str, float] = {name: 0.0 for name in SOURCES}
    for term in terms:
        prior = priors.get(term.node)
        if prior is None or term.source not in SOURCES:
            continue
        error, response, pi = _groups(term)
        r = prior.c.numel()
        if response.shape[-1] != r or response.shape[:2] != error.shape:
            continue
        name = term.source
        energy = (pi * error.square().sum(-1)).sum()
        result.counts[name] = result.counts.get(name, 0) + error.shape[0]
        sums[name] += float(energy)
        steps[name].add(term.step)
        nodes[name].add(term.node)
        lam = float(weights.get(name, 0.0))
        if lam == 0.0:
            continue
        w = (lam * pi).reshape(-1, 1, 1)
        hh = (response.transpose(-1, -2) @ (w * response)).sum(0)
        gg = (response.transpose(-1, -2) @ (w * error.unsqueeze(-1))).sum(0).squeeze(-1)
        h[term.node] = h.get(term.node, 0) + hh
        g[term.node] = g.get(term.node, 0) + gg
    by_size: dict[int, list[Hashable]] = {}
    for node, prior in priors.items():
        by_size.setdefault(prior.c.numel(), []).append(node)
    for r, keys in by_size.items():
        if r == 0:
            continue
        mats, rhs = [], []
        for node in keys:
            prior = priors[node]
            inv = prior.inv_cov.double().reshape(r, r)
            pull = inv @ (prior.c.double().reshape(r) - prior.mu.double().reshape(r))
            mats.append(h.get(node, torch.zeros(r, r, dtype=torch.float64)) + inv)
            rhs.append(g.get(node, torch.zeros(r, dtype=torch.float64)) + pull)
        solved = torch.linalg.solve(torch.stack(mats), -torch.stack(rhs).unsqueeze(-1))
        for node, value in zip(keys, solved.squeeze(-1), strict=True):
            result.delta[node] = value
    for name in SOURCES:
        count = result.counts.get(name, 0)
        result.energy[name] = sums[name] / count if count else 0.0
        result.steps[name] = len(steps[name])
        result.nodes[name] = len(nodes[name])
    result.seconds = time.perf_counter() - started
    return result


Response = Callable[[torch.Tensor], torch.Tensor]


def _sigma_key(sigma: float) -> float:
    return round(float(sigma), 9)


def build_terms(
    cross: Mapping[str, list],
    responses: Mapping[tuple[str, int], tuple[Hashable, Response]],
    p_out: Mapping[tuple[str, int], torch.Tensor],
    depth: Mapping[str, list] | None = None,
    depth_proj: Mapping[str, torch.Tensor] | None = None,
    hier: Mapping[str, list] | None = None,
    clamp: Mapping[str, list] | None = None,
    block_frame: Callable[[str, int], torch.Tensor | None] | None = None,
    block_of: Callable[[str], int | None] | None = None,
) -> list[Term]:

    terms: list[Term] = []
    reads: dict[tuple[str, int], dict[float, torch.Tensor]] = {}
    for target, items in (cross or {}).items():
        for item in items:
            key = (target, int(item.node))
            if key not in responses or key not in p_out:
                continue
            reads.setdefault(key, {})[_sigma_key(item.sigma)] = item.z
            z_prev = getattr(item, "z_prev", None)
            if z_prev is None or z_prev.shape != item.z.shape:
                continue
            node, response = responses[key]
            span = float(item.sigma_prev) - float(item.sigma)
            if span == 0:
                continue
            frame = p_out[key].double()
            dz = (item.z.double() - z_prev.double()) / span
            terms.append(
                Term("cross", node, item.error, frame @ response(dz), item.precision, item.step)
            )
    for target, items in (depth or {}).items():
        key = (target, 0)
        proj = (depth_proj or {}).get(target)
        if key not in responses or proj is None:
            continue
        node, response = responses[key]
        for item in items:
            if item.z is None:
                continue
            a = response(item.z.double())
            if a.shape[-2] != proj.shape[-1]:
                continue
            error = item.error.double().mean(0) if item.error.dim() > 1 else item.error
            terms.append(
                Term("depth", node, error, -(proj.double() @ a), item.precision, item.progress)
            )
    if block_frame is None or block_of is None:
        return terms
    by_block: dict[int, list[tuple[str, int]]] = {}
    for key in responses:
        index = block_of(key[0])
        if index is not None:
            by_block.setdefault(index, []).append(key)
    for name, errors, sign in (("hier", hier or {}, 1.0), ("clamp", clamp or {}, -1.0)):
        for items in errors.values():
            for item in items:
                for key in by_block.get(int(item.level), []):
                    z = reads.get(key, {}).get(_sigma_key(item.sigma))
                    if z is None:
                        continue
                    node, response = responses[key]
                    a = response(z.double())
                    frame = block_frame(key[0], a.shape[-2])
                    if frame is None or frame.shape[-1] != a.shape[-2]:
                        continue
                    error = item.error.double()
                    if error.shape[-1] != frame.shape[0] or error.dim() != z.dim():
                        continue
                    terms.append(
                        Term(
                            name,
                            node,
                            error,
                            sign * (frame.double() @ a),
                            item.precision,
                            item.step,
                        )
                    )
    return terms


@dataclass
class DeferredGuard:
    before: dict[str, dict[str, torch.Tensor]]
    commits: int
    revision: int = 0
