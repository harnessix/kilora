# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
import statistics

import torch

from .path_walks import transition_prior


def enabled() -> bool:
    return os.environ.get("KILORA_KEY_GRAPH") == "1"


def build(s, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    prior = transition_prior(s).detach().float().cpu()
    sums = prior.sum(-1, keepdim=True)
    identity = torch.eye(prior.shape[0], dtype=prior.dtype)
    matrix = torch.where(sums > 0, prior / sums.clamp_min(1e-30), identity)
    powers = [identity]
    for _ in range(max(1, depth)):
        powers.append(powers[-1] @ matrix)
    return matrix, torch.stack(powers)


def distances(weights: list[torch.Tensor], powers: torch.Tensor) -> torch.Tensor:
    def normalize(w):
        w = w.detach().float().cpu()
        total = w.sum(-1, keepdim=True)
        return torch.where(total > 0, w / total.clamp_min(1e-30), torch.nan)

    first = normalize(weights[0])
    return torch.stack(
        [
            (normalize(w) - first @ powers[i]).abs().sum(-1).mean() * 0.5
            for i, w in enumerate(weights[: len(powers)])
        ]
    )


def graph_report(plan, threshold: float = 0.1) -> dict:
    plans = plan if isinstance(plan, (list, tuple)) else [plan]
    result = {}
    for module in plans:
        targets = [part[2] for part in module.parts] if hasattr(module, "parts") else [module]
        for target in targets:
            for index, s in enumerate(target.sets):
                d = s.graph_distances
                if s.graph_transition is not None:
                    result[(target.name, s.key, index)] = {
                        "distances": [] if d is None else d.tolist(),
                        **_diagnostics(s),
                        "qualifies": bool(
                            d is not None
                            and d.numel()
                            and s.graph_powers is not None
                            and len(d) == len(s.graph_powers) - 1
                            and torch.all(d < threshold)
                        ),
                    }
    return result


def loop_gain(s, keep: torch.Tensor | None) -> tuple[float | None, float | None]:
    if s.form != "state" or s.up is None or s.down is None:
        return None, None
    g = s.up.detach().float().cpu() @ s.down.detach().float().cpu().t()
    rho = float(torch.linalg.eigvals(g).abs().max())
    damped = None if keep is None else float(torch.linalg.eigvals(keep[:, None] * g).abs().max())
    return rho, damped


def sequence_metrics(
    weights: list[torch.Tensor],
) -> tuple[float | None, float | None, torch.Tensor]:

    e = weights[0].shape[-1]
    seq = torch.stack([w.detach().cpu().reshape(-1, e).argmax(-1) for w in weights], 1)
    if seq.shape[1] < 2 or not seq.shape[0]:
        return None, None, torch.empty((0, 4), dtype=torch.long)
    counts = torch.bincount((seq[:, :-1] * e + seq[:, 1:]).reshape(-1), minlength=e * e)
    pairs = counts.reshape(e, e).float()
    totals = pairs.sum(-1, keepdim=True)
    probs = pairs / totals.clamp_min(1)
    entropy = float(-(pairs / pairs.sum() * probs.clamp_min(1e-30).log2()).sum())
    loop = float((seq == seq[:, :1]).all(1).float().mean())
    singles = {}
    for row in seq.tolist():
        for i in range(1, len(row) - 1):
            singles.setdefault((row[i - 1], row[i + 1]), set()).add(row[i])
    contexts = {}
    for row in seq.tolist():
        for i in range(1, len(row) - 2):
            contexts.setdefault((row[i], row[i + 1]), set()).add((row[i - 1], row[i + 2]))
    ranked = sorted(range(e * e), key=lambda i: (-int(counts[i]), i))
    merged = []
    for idx in ranked[:3]:
        count = int(counts[idx])
        if not count:
            break
        pair = (idx // e, idx % e)
        alternatives = set().union(*(singles.get(c, set()) for c in contexts.get(pair, set())))
        merged.append([*pair, count, min(alternatives) if alternatives else -1])
    return entropy, loop, torch.tensor(merged, dtype=torch.long).reshape(-1, 4)


def hop_distances(weights: list[torch.Tensor], powers: torch.Tensor) -> torch.Tensor:
    normalized = []
    for w in weights:
        w = w.detach().float().cpu()
        total = w.sum(-1, keepdim=True)
        normalized.append(torch.where(total > 0, w / total.clamp_min(1e-30), torch.nan))
    rows = []
    for i in range(1, len(normalized)):
        actual = normalized[i]
        one = (actual - normalized[i - 1] @ powers[1]).abs().sum(-1).mean() * 0.5
        two = (
            (actual - normalized[i - 2] @ powers[2]).abs().sum(-1).mean() * 0.5
            if i >= 2
            else torch.tensor(float("nan"))
        )
        rows.append(torch.stack((one, two)))
    return torch.stack(rows) if rows else torch.empty((0, 2))


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _diagnostics(s):
    stats = s.graph_stats
    values = [None] * 4 if stats is None else [_number(v) for v in stats[:4]]
    hops = (
        []
        if s.graph_hops is None
        else [{"one_hop": _number(a), "two_hop": _number(b)} for a, b in s.graph_hops]
    )
    merges = (
        []
        if s.graph_merges is None
        else [
            {
                "experts": [int(a), int(b)],
                "count": int(n),
                "frequency": int(n) / int(stats[4]) if stats is not None and stats[4] else 0.0,
                "substitute": None if int(sub) < 0 else int(sub),
            }
            for a, b, n, sub in s.graph_merges
        ]
    )
    return {
        "rho": values[0],
        "damped_rho": values[1],
        "entropy_rate": values[2],
        "loop_share": values[3],
        "hops": hops,
        "merge_candidates": merges,
    }


def summary(report: dict) -> str:
    layers = {}
    for (name, _, _), row in report.items():
        layers.setdefault(name, []).append(row)
    gains, entropies, hops = [], [], []
    for rows in layers.values():
        gain = [r["damped_rho"] for r in rows if r["damped_rho"] is not None]
        entropy = [r["entropy_rate"] for r in rows if r["entropy_rate"] is not None]
        distances = [h["two_hop"] for r in rows for h in r["hops"] if h["two_hop"] is not None]
        if gain:
            gains.append(max(gain))
        if entropy:
            entropies.append(statistics.mean(entropy))
        if distances:
            hops.append(statistics.mean(distances))

    def fraction(values, threshold):
        return f"{sum(v < threshold for v in values)}/{len(values)}" if values else "n/a"

    median = f"{statistics.median(entropies):.3f} bits" if entropies else "n/a"
    return (
        f"Key graph diagnostics: damped gain < 0.9: {fraction(gains, 0.9)} layers; "
        f"median entropy rate: {median}; two-hop TV < 0.1: {fraction(hops, 0.1)} layers."
    )
