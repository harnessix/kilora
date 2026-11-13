# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


def make_exact(tensors, kind, weight, budget):
    if kind == "lokr":
        from .multi_import import lokr_parts

        pair = lokr_parts(tensors)
        if pair is None or any(t.ndim != 2 for t in pair):
            return None
        parts = (pair[0] * weight, pair[1])
    elif kind == "loha":
        names = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
        parts = tuple(tensors[n].float().contiguous() for n in names)
        if any(t.ndim != 2 for t in parts):
            return None
        alpha = tensors.get("alpha")
        scale = weight * (
            float(alpha.reshape(-1)[0]) / parts[1].shape[0] if alpha is not None else 1
        )
        parts = (parts[0] * scale, *parts[1:])
    else:
        return None
    if sum(t.numel() * t.element_size() for t in parts) > budget:
        return None
    return tuple(t.contiguous() for t in parts)


def apply_exact(parts, x, block=128):
    parts = tuple(t.to(device=x.device, dtype=x.dtype) for t in parts)
    if len(parts) == 2:
        w1, w2 = parts
        shaped = x.reshape(*x.shape[:-1], w1.shape[1], w2.shape[1])
        return (w1 @ (shaped @ w2.t())).reshape(*x.shape[:-1], -1)
    a, b, c, d = parts
    outputs = []
    for start in range(0, a.shape[0], block):
        delta = (a[start : start + block] @ b) * (c[start : start + block] @ d)
        outputs.append(F.linear(x, delta))
    return torch.cat(outputs, dim=-1)
