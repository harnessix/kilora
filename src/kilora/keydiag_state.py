# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from . import compute


def _basis(keys: torch.Tensor, count: int) -> torch.Tensor:
    rows = keys[:count].detach().float()
    if not rows.numel():
        return rows[:0]
    _, values, vh = torch.linalg.svd(rows, full_matrices=False)
    if not values.numel() or values[0] == 0:
        return vh[:0]
    tol = max(rows.shape) * torch.finfo(values.dtype).eps * values[0]
    return vh[values > tol]


def state_keys(s) -> dict:

    if s.form != "state":
        return {"skipped": "not state"}
    kin, ri, kout, ro, _ = compute.derived_keys(s)
    ins = [_basis(k, int(ri[i]) if ri is not None else len(k)) for i, k in enumerate(kin)]
    outs = [_basis(k, int(ro[i]) if ro is not None else len(k)) for i, k in enumerate(kout)]
    matrix = []
    for out in outs:
        row = []
        for key in ins:
            value = (
                torch.linalg.svdvals(out @ key.T).clamp(0, 1).mean()
                if len(out) and len(key)
                else 0.0
            )
            row.append(float(value))
        matrix.append(row)
    own = [row[i] for i, row in enumerate(matrix)]
    other = [
        max((v for j, v in enumerate(row) if j != i), default=None) for i, row in enumerate(matrix)
    ]
    return {"own": own, "best_other": other, "cosines": matrix}


def query_motion(s, x, y0, depth) -> dict:

    if depth < 1:
        return {"skipped": "depth below 1"}
    with torch.no_grad(), compute.eager(), compute.key_probes() as records:
        compute.run_layer(x, y0, [s], depth, target="keydiag_state")
    steps = [entry for entry in records.get("keydiag_state", []) if isinstance(entry["k"], int)]
    if not steps or any(not entry.get("gates") or entry["gates"][0] is None for entry in steps):
        return {"skipped": "loop does not expose queries and gates"}
    motion, changes, entropy = [], [], []
    prev_q = prev_choice = None
    for entry in steps:
        q = entry["queries"][0].detach().float()
        gates = entry["gates"][0].detach().float()
        choice = gates.argmax(-1)
        if prev_q is None:
            motion.append(0.0)
            changes.append(0.0)
        else:
            diff = torch.linalg.vector_norm(q - prev_q, dim=-1)
            norm = torch.linalg.vector_norm(q, dim=-1)
            motion.append(float((diff / norm.clamp_min(1e-30)).mean()))
            changes.append(float((choice != prev_choice).float().mean()))
        entropy.append(float(-(gates * gates.clamp_min(1e-30).log()).sum(-1).mean()))
        prev_q, prev_choice = q, choice
    return {"query_motion": motion, "argmax_changes": changes, "routing_entropy": entropy}
