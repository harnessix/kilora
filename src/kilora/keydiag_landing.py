# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re

import torch


def landing(write_set, read_set, x_write) -> dict:

    a_write, b_write, a_read = write_set.down, write_set.up, read_set.down
    if a_write is None or b_write is None or a_read is None:
        raise ValueError("landing requires write down/up and read down matrices")
    rw, rr = write_set.per_expert, read_set.per_expert
    if rw <= 0 or rr <= 0 or a_write.shape[0] % rw or a_read.shape[0] % rr:
        raise ValueError("invalid expert row counts")
    nw, nr = a_write.shape[0] // rw, a_read.shape[0] // rr
    if not nw or nw != nr or b_write.shape[0] != a_write.shape[0]:
        raise ValueError("landing requires matching, nonempty expert sets")
    if x_write.ndim != 2 or not x_write.shape[0] or x_write.shape[1] != a_write.shape[1]:
        raise ValueError("x_write must be a nonempty [tokens, input width] matrix")
    if b_write.shape[1] != a_read.shape[1]:
        raise ValueError("write output and read input widths must match")

    with torch.no_grad():
        device = x_write.device
        x = x_write.detach().to(dtype=torch.float32)
        aw = a_write.detach().to(device=device, dtype=torch.float32).reshape(nw, rw, -1)
        bw = b_write.detach().to(device=device, dtype=torch.float32).reshape(nw, rw, -1)
        ar = a_read.detach().to(device=device, dtype=torch.float32).reshape(nr, rr, -1)
        read_norm = ar.square().sum(dim=(1, 2))
        scores = torch.zeros((nr, nw), device=device)
        overlap = torch.zeros_like(scores)
        for e in range(nw):
            change = (x @ aw[e].T) @ bw[e]
            change_norm = change.square().sum(dim=1)
            energy = torch.einsum("frd,nd->fnr", ar, change).square().sum(dim=2)
            denom = read_norm[:, None] * change_norm[None, :]
            scores[:, e] = torch.where(
                denom > 0, energy / denom.clamp_min(torch.finfo(x.dtype).tiny), 0
            ).mean(dim=1)

            _, singular, vh = torch.linalg.svd(bw[e], full_matrices=False)
            if singular.numel():
                tol = max(bw[e].shape) * torch.finfo(singular.dtype).eps * singular[0]
                basis = vh[singular > tol]
                energy = torch.einsum("frd,kd->frk", ar, basis).square().sum(dim=(1, 2))
                overlap[:, e] = torch.where(
                    read_norm > 0, energy / read_norm.clamp_min(torch.finfo(x.dtype).tiny), 0
                )

        def comparison(matrix):
            own = [float(matrix[e, e]) for e in range(nw)]
            other = [
                max((float(matrix[f, e]) for f in range(nr) if f != e), default=None)
                for e in range(nw)
            ]
            return {"matrix": matrix.cpu().tolist(), "own": own, "best_other": other}

        return {
            "activation": comparison(scores),
            "weight_only": comparison(overlap),
            "limits": ["ignores norm scale", "ignores residual add"],
        }


def pairs(names) -> list[tuple[str, str]]:

    available = set(names)
    result = set()
    pattern = re.compile(r"^(.*?)(?<![A-Za-z_])(transformer_blocks|blocks)\.(\d+)\.(.+)$")
    for name in available:
        match = pattern.fullmatch(name)
        if not match:
            continue
        prefix, family, number, suffix = match.groups()
        number = int(number)
        if family == "blocks":
            successors = {
                "self_attn.output_proj": (number, ("cross_attn.q_proj",)),
                "cross_attn.output_proj": (number, ("mlp.layer1",)),
                "mlp.layer2": (number + 1, tuple(f"self_attn.{p}_proj" for p in "qkv")),
            }
        else:
            successors = {
                "attn1.to_out.0": (number, ("attn2.to_q",)),
                "attn2.to_out.0": (number, ("ff.net.0.proj",)),
                "ff.net.2": (number + 1, tuple(f"attn1.to_{p}" for p in "qkv")),
            }
        if suffix in successors:
            block, reads = successors[suffix]
            for read in reads:
                target = f"{prefix}{family}.{block}.{read}"
                if target in available:
                    result.add((name, target))
    return sorted(result)
