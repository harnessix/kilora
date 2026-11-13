# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import torch

from . import compute


def _report(energy: torch.Tensor) -> dict:
    rows = energy.sum(dim=1, keepdim=True)
    shares = torch.where(rows > 0, energy / rows.clamp_min(torch.finfo(energy.dtype).tiny), 0)
    total = energy.sum()
    diagonal = energy.diagonal().sum()
    return {
        "matrix": shares.cpu().tolist(),
        "diagonal_share": (diagonal / total).item() if total > 0 else 0.0,
        "self_share": shares.diagonal().cpu().tolist(),
    }


def depth_transition(s: compute.LayerSet, x: torch.Tensor) -> dict:

    if s.form == "input" and s.down.shape[1] != s.up.shape[1]:
        return {"skipped": "not square"}

    with torch.no_grad():
        device = x.device
        down = s.down.detach().to(device=device, dtype=torch.float32)
        up = s.up.detach().to(device=device, dtype=torch.float32)
        x = x.detach().to(dtype=torch.float32).reshape(-1, down.shape[1])
        n = down.shape[0] // s.per_expert
        a = down.reshape(n, s.per_expert, -1)
        b = up.reshape(n, s.per_expert, -1)
        coefficients = x @ down.T
        tensors = {}
        for name in ("router", "keys_in", "keys_out", "rows_in", "rows_out", "tau"):
            value = getattr(s, name)
            if isinstance(value, torch.Tensor):
                tensors[name] = value.to(device)
        routed_set = replace(s, **tensors)
        if s.form == "input" and compute.import_score(s) == "read_share":
            gate = compute._read_share_gates([routed_set], [coefficients], [None])[0]
        else:
            gate = compute._gates(x, [routed_set], False, 0.0)[0]
        scale = torch.as_tensor(s.scale, dtype=torch.float32, device=device)
        c = coefficients.reshape(-1, n, s.per_expert) * gate.unsqueeze(-1) * scale
        writes = torch.einsum("ner,erd->ned", c, b)
        reads = torch.einsum("frd,ned->nfer", a, writes)
        energy = reads.square().sum(dim=-1).mean(dim=0)
        weight_energy = torch.einsum("frd,esd->fers", a, b).square().sum(dim=(-1, -2))
        result = _report(energy)
        result["weight_only"] = _report(weight_energy)
        return result
