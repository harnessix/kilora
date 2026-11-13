# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import statistics

import torch

from . import compute, key_graph
from .keydiag_state import _basis


def _normalize(matrix):
    total = matrix.sum(-1, keepdim=True)
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return torch.where(total > 0, matrix / total.clamp_min(1e-30), identity)


def _ratio(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), 0)


def _equivalence(matrix):
    n = len(matrix)
    differences = [
        abs(float(matrix[a, b] * matrix[b, c] - matrix[a, c]))
        for a, b, c in itertools.permutations(range(n), 3)
    ]
    return {
        "median": statistics.median(differences) if differences else None,
        "share_below_0_1": sum(v < 0.1 for v in differences) / len(differences)
        if differences
        else None,
        "triples": len(differences),
    }


def _embeddings(matrix):
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    result = {}
    for rank in (4, 8, 16):
        if rank >= len(matrix):
            result[f"embed_r{rank}"] = matrix
        else:
            root = singular[:rank].sqrt()
            result[f"embed_r{rank}"] = (
                (u[:, :rank] * root) @ (root[:, None] * vh[:rank])
            ).clamp_min(0)
    return result


def _trace(record, index):
    if not key_graph.enabled():
        return None
    sets = record["sets"]
    device = sets[index].down.device
    weights = []

    def observe(name, gates):
        if name == record["target"] and index < len(gates):
            weights.append(gates[index].detach().float())

    devices = [device.index] if device.type == "cuda" else []
    with (
        torch.random.fork_rng(devices=devices),
        torch.no_grad(),
        compute.eager(),
        compute.recording_shares(observe),
    ):
        compute.run_layer(
            record["x"].to(device).unsqueeze(0),
            record["y0"].to(device).unsqueeze(0),
            sets,
            record["depth"],
            target=record["target"],
        )
    return weights


def compare_paths(record, index):

    s = record["sets"][index]
    if s.down is None or s.up is None:
        return {"skipped": "no read or write matrix"}
    if s.down.shape[1] != s.up.shape[1]:
        return {"skipped": "not square"}
    with torch.no_grad():
        a = s.down.detach().float().reshape(-1, s.per_expert, s.down.shape[1])
        b = s.up.detach().to(device=a.device, dtype=torch.float32).reshape_as(a)
        n = len(a)
        product = torch.einsum("erd,fsd->efrs", b, a)
        energy = product.square().sum((-1, -2))
        anorm = a.square().sum((-1, -2))
        bnorm = b.square().sum((-1, -2))
        cosine = _ratio(energy.sqrt(), (bnorm[:, None] * anorm[None, :]).sqrt())
        ins = [_basis(rows, len(rows)) for rows in a]
        outs = [_basis(rows, len(rows)) for rows in b]
        angles = torch.zeros_like(energy)
        for e, out in enumerate(outs):
            for f, inp in enumerate(ins):
                if len(out) and len(inp):
                    angles[e, f] = torch.linalg.svdvals(out @ inp.T).clamp(0, 1).square().mean()
        x = record["x"] if s.form == "input" else record["y0"]
        x = x.detach().to(device=a.device, dtype=torch.float32).reshape(-1, a.shape[-1])
        x = x - x.mean(0, keepdim=True)
        ar = torch.einsum("frd,nd->frn", a, x)
        br = torch.einsum("erd,nd->ern", b, x)
        cross = torch.einsum("ern,fsn->efrs", br, ar)
        denom = (br.square().sum((-1, -2))[:, None] * ar.square().sum((-1, -2))[None, :]).sqrt()
        weighted = _ratio(cross.square().sum((-1, -2)).sqrt(), denom)
        aa = torch.einsum("erd,esd->ers", a, a)
        bb = torch.einsum("erd,esd->ers", b, b)
        weighted_aa = torch.einsum("ern,esn->ers", ar, ar) / max(len(x), 1)
        weighted_cross = cross / max(len(x), 1)
        norms = torch.einsum("ers,esr->e", aa, bb).clamp_min(0).sqrt()
        weighted_norms = torch.einsum("ers,esr->e", weighted_aa, bb).clamp_min(0).sqrt()
        trace = torch.einsum("efrs,fesr->ef", product, product)
        trace = _ratio(trace, norms[:, None] * norms[None, :])
        weighted_trace = torch.einsum("efrs,fesr->ef", product, weighted_cross)
        weighted_trace = _ratio(weighted_trace, weighted_norms[:, None] * weighted_norms[None, :])
        transmission_sq = torch.einsum("efij,fjk,eflk,eli->ef", product, bb, product, aa)
        weighted_sq = torch.einsum("efij,fjk,eflk,eli->ef", product, bb, product, weighted_aa)
        transmission = _ratio(transmission_sq.clamp_min(0).sqrt(), norms[:, None] * norms[None, :])
        weighted_transmission = _ratio(
            weighted_sq.clamp_min(0).sqrt(), weighted_norms[:, None] * norms[None, :]
        )
        raw = {
            "energy": energy,
            "cosine": cosine,
            "angles": angles,
            "weighted": weighted,
            "trace": trace.clamp_min(0),
            "transmission": transmission,
            "weighted_trace": weighted_trace.clamp_min(0),
            "weighted_transmission": weighted_transmission,
            **_embeddings(transmission),
            **{f"weighted_{k}": v for k, v in _embeddings(weighted_transmission).items()},
        }
        matrices = {name: _normalize(value) for name, value in raw.items()}
        weights = _trace(record, index)
        result = {}
        for name, matrix in matrices.items():
            hops = None
            if weights is not None:
                hops = []
                for before, after in zip(weights, weights[1:], strict=False):
                    before = before.to(matrix.device)
                    after = after.to(matrix.device)
                    bs, ass = before.sum(-1, keepdim=True), after.sum(-1, keepdim=True)
                    valid = (bs[..., 0] > 0) & (ass[..., 0] > 0)
                    if valid.any():
                        p = before / bs.clamp_min(1e-30)
                        q = after / ass.clamp_min(1e-30)
                        hops.append(float(((p @ matrix - q).abs().sum(-1) * 0.5)[valid].mean()))
                    else:
                        hops.append(None)
            result[name] = {
                "matrix": matrix.cpu().tolist(),
                "diagonal_share": float(matrix.diagonal().sum() / n) if n else 0.0,
                "one_hop_tv": hops,
            }
        result["trace"]["negative_share"] = float((trace < 0).float().mean()) if n else 0.0
        result["weighted_trace"]["negative_share"] = (
            float((weighted_trace < 0).float().mean()) if n else 0.0
        )
        result["path_equivalence"] = _equivalence(cosine)
        result["transmission_path_equivalence"] = _equivalence(transmission)
        return result
