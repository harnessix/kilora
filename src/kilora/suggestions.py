# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def suggest(current: Sequence[str], installed: Mapping[str, str], singles, pairs):

    selected = set(current)
    scores = []
    for name, digest in installed.items():
        if digest in selected or singles.get(digest, 0) <= 0:
            continue
        score = sum(
            pairs.get(tuple(sorted((digest, other))), 0)
            / math.sqrt(singles[digest] * singles[other])
            for other in selected
            if singles.get(other, 0) > 0
        )
        if score > 0:
            scores.append((name, score))
    return sorted(scores, key=lambda item: (-item[1], item[0].casefold(), item[0]))[:5]


CO_USAGE_WEIGHT = 0.25


def embedding_suggest(current, installed, singles, pairs, vectors, query, mode):
    import torch

    selected = set(current)
    rows = [vectors[h].float().flatten() for h in query if h in vectors]
    if len(rows) != len(query) or not rows or len({v.numel() for v in rows}) != 1:
        raise ValueError("missing query embedding")
    matrix = torch.stack(rows)
    mean = matrix.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean)
    if not torch.isfinite(matrix).all() or mean_norm <= 1e-12:
        raise ValueError("invalid query embedding")
    basis = None
    if mode == "complementary":
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = int(
            (singular > singular.max() * max(matrix.shape) * torch.finfo(matrix.dtype).eps).sum()
        )
        basis = vh[:rank]
    scores, missing = [], False
    for name, digest in installed.items():
        if digest in selected:
            continue
        co = sum(
            pairs.get(tuple(sorted((digest, other))), 0)
            / math.sqrt(singles[digest] * singles[other])
            for other in selected
            if singles.get(digest, 0) > 0 and singles.get(other, 0) > 0
        )
        vector = vectors.get(digest)
        if vector is None or vector.numel() != mean.numel():
            missing = True
            if co > 0:
                scores.append((name, co))
            continue
        vector = vector.float().flatten()
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector).all() or norm <= 1e-12:
            missing = True
            if co > 0:
                scores.append((name, co))
            continue
        cosine = float(torch.dot(vector, mean) / (norm * mean_norm))
        score = cosine
        if basis is not None:
            score *= float(torch.linalg.vector_norm(vector - (basis @ vector) @ basis))
        scores.append((name, score + CO_USAGE_WEIGHT * co))
    return sorted(scores, key=lambda item: (-item[1], item[0].casefold(), item[0]))[:5], missing
