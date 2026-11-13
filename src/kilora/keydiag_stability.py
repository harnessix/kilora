# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import torch

from . import compute


def spectral_radius(matrix: torch.Tensor) -> float:
    if not matrix.numel():
        return 0.0
    return float(torch.linalg.eigvals(matrix.float()).abs().max())


def graph_stability(transition: torch.Tensor) -> dict:
    t = transition.detach().float()
    off = t.clone()
    off.diagonal().zero_()
    two = t @ t
    return {
        "off_diagonal_radius": spectral_radius(off),
        "return_mass_2": float(two.diagonal().mean()) if len(t) else 0.0,
        "return_mass_3": float((two @ t).diagonal().mean()) if len(t) else 0.0,
    }


def stability(s: compute.LayerSet, x: torch.Tensor, y0: torch.Tensor, depth: float) -> dict:

    if s.down is None or s.up is None:
        return {"skipped": "no expert weights"}
    result = {}
    if s.down.shape[1] == s.up.shape[1]:
        graph_set = replace(s, form="input")
        graph = compute._share_transition_base(graph_set, states=x if s.form == "input" else y0)
        if graph is not None:
            result.update(graph_stability(graph))
    else:
        result["graph_skipped"] = "not square"
    if s.form != "state":
        return result
    keep = []
    with (
        torch.no_grad(),
        compute.eager(),
        compute.derived_call(),
        compute.key_probes() as records,
        compute.recording_read_keep(lambda _target, rates: keep.append(rates)),
    ):
        compute.run_layer(x, y0, [s], depth, target="stability")
    steps = [entry for entry in records.get("stability", []) if isinstance(entry["k"], int)]
    gates = [e["gates"][0] for e in steps if e.get("gates") and e["gates"][0] is not None]
    if not keep or not gates:
        result["depth_skipped"] = "routing or read keep rates unavailable"
        return result
    w = torch.stack([g.float().reshape(-1, g.shape[-1]).mean(0) for g in gates]).mean(0)
    d = torch.stack(keep).mean(0)
    g = s.up.detach().float() @ s.down.detach().float().T
    scale = torch.as_tensor(s.scale, device=g.device, dtype=g.dtype)
    step = d[:, None] * g * (w.repeat_interleave(s.per_expert) * scale)[None, :]
    result["depth_step_radius"] = spectral_radius(step)
    return result
