# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import torch


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + end - 1) / 2
        start = end
    return ranks


def _pearson(a, b):
    if len(a) < 2:
        return None
    am, bm = sum(a) / len(a), sum(b) / len(b)
    da, db = [v - am for v in a], [v - bm for v in b]
    denom = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    if denom == 0:
        return None
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(da, db, strict=True)) / denom))


def _correlations(a, b):
    return {"pearson": _pearson(a, b), "spearman": _pearson(_ranks(a), _ranks(b))}


def _shares(energy):
    return energy / (energy.sum(dim=-1, keepdim=True) + 1e-12)


def share_bias(s, x, *, neutral_seed=0):

    if s.down is None or x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("share_bias requires a read matrix and nonempty 2-D activations")
    if s.per_expert <= 0 or s.down.shape[0] % s.per_expert or s.down.shape[0] == 0:
        raise ValueError("read matrix rows must form nonempty equal-sized expert blocks")
    if x.shape[1] != s.down.shape[1]:
        raise ValueError("activation width must match the read matrix")

    a = s.down.detach().to(device="cpu", dtype=torch.float64)
    x = x.detach().to(device="cpu", dtype=torch.float64)
    a = a.reshape(-1, s.per_expert, a.shape[-1])
    energy = torch.einsum("ni,eri->ner", x, a).square().sum(dim=-1)
    norm_sq = a.square().sum(dim=(1, 2))
    effective_rank = (a != 0).any(dim=-1).sum(dim=-1)
    singular_values = torch.linalg.svdvals(a)
    spectral_sq = singular_values[:, 0].square()
    stable_rank = torch.where(spectral_sq > 0, norm_sq / spectral_sq.clamp_min(1e-300), 0)
    generator = torch.Generator(device="cpu").manual_seed(neutral_seed)
    neutral = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    neutral = neutral * x.std(dim=0, unbiased=False) + x.mean(dim=0)
    neutral_energy = torch.einsum("ni,eri->ner", neutral, a).square().sum(dim=-1)
    normalized_energy = energy / torch.where(norm_sq > 0, norm_sq, 1)

    means = _shares(energy).mean(dim=0).tolist()
    ranks = effective_rank.tolist()
    norms = norm_sq.sqrt().tolist()
    neutral_means = _shares(neutral_energy).mean(dim=0).tolist()
    normalized_means = _shares(normalized_energy).mean(dim=0).tolist()
    stable = stable_rank.tolist()
    experts = [
        {
            "mean_share": means[i],
            "effective_rank": ranks[i],
            "stable_rank": stable[i],
            "frobenius_norm": norms[i],
            "neutral_mean_share": neutral_means[i],
            "normalized_mean_share": normalized_means[i],
        }
        for i in range(len(means))
    ]
    return {
        "experts": experts,
        "share_vs_effective_rank": _correlations(means, ranks),
        "share_vs_frobenius_norm": _correlations(means, norms),
        "largest_share_is_largest_norm": means.index(max(means)) == norms.index(max(norms)),
    }
