# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .codebook import StreamingCodebook


@dataclass(frozen=True)
class RetrievalResult:
    digest: str
    score: float


def _cosine(a: Tensor | None, b: Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    a = torch.as_tensor(a, dtype=torch.float32, device="cpu").flatten()
    b = torch.as_tensor(b, dtype=torch.float32, device="cpu").flatten()
    if a.shape != b.shape:
        raise ValueError("feature widths must match")
    denom = a.norm() * b.norm()
    return float(torch.dot(a, b) / denom) if denom > 0 else 0.0


def _probe_for(group: str, probe: Mapping[str, Tensor]) -> Tensor | None:
    if group in probe:
        return probe[group]
    band, _, rest = group.partition("-")
    if not band.isdigit() or not rest:
        return None
    same = [g for g in probe if g.partition("-")[2] == rest and g.partition("-")[0].isdigit()]
    return probe[max(same, key=lambda g: int(g.partition("-")[0]))] if same else None


def _content_share(
    summary: Tensor | None,
    probe: Mapping[str, Tensor],
    groups: Mapping[str, int] | None = None,
    directions: Mapping[str, Tensor] | None = None,
) -> float:

    if (summary is None and not directions) or not probe:
        return 0.0
    if groups is None:
        groups = {g: int(torch.as_tensor(t).shape[-1]) for g, t in probe.items()}
    flat = (
        None
        if summary is None
        else torch.as_tensor(summary, dtype=torch.float32, device="cpu").flatten()
    )
    if flat is not None and flat.numel() and flat.numel() != sum(groups.values()):
        raise ValueError("summary width does not match the family groups")
    offset = 0
    read_energy = total_energy = 0.0
    for group, width in groups.items():
        segment = None if flat is None or not flat.numel() else flat[offset : offset + width]
        offset += width
        tokens = _probe_for(group, probe)
        if tokens is None:
            continue
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device="cpu")
        if tokens.shape[-1] != width:
            raise ValueError(f"probe width of group {group} does not match")
        tokens = tokens.reshape(-1, width)
        if not tokens.shape[0]:
            continue
        total_energy += float(tokens.square().sum(dim=-1).mean())
        rows = None if directions is None else directions.get(group)
        if rows is not None and torch.as_tensor(rows).numel():
            basis = torch.as_tensor(rows, dtype=torch.float32, device="cpu").reshape(-1, width)
            basis = torch.linalg.qr(basis.t()).Q.t()
            read_energy += float((tokens @ basis.t()).square().sum(dim=-1).mean())
        elif segment is not None:
            norm = segment.norm()
            if norm > 0:
                read_energy += float((tokens @ (segment / norm)).square().mean())
    return min(1.0, max(0.0, read_energy / total_energy)) if total_energy else 0.0


def _tags(values: Iterable[str] | str) -> set[str]:
    if isinstance(values, str):
        values = values.split(",")
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def retrieve(
    records: Mapping[str, Mapping],
    summaries: Mapping[str, Tensor],
    text_features: Mapping[str, Tensor],
    codebook: StreamingCodebook,
    *,
    text_query: Tensor,
    probe: Mapping[str, Tensor],
    family: str,
    k: int,
    seed: int = 0,
    min_relevance: float = 0.2,
    include_tags: Iterable[str] | str = (),
    exclude_tags: Iterable[str] | str = (),
    w_text: float = 0.5,
    w_content: float = 0.5,
    groups: Mapping[str, int] | None = None,
    directions: Mapping[str, Mapping[str, Tensor]] | None = None,
    cluster_query: Tensor | None = None,
    fallback_query: Tensor | None = None,
    fallback_digests: Iterable[str] = (),
) -> list[RetrievalResult]:

    del seed
    if k < 1:
        raise ValueError("k must be positive")
    if not all(math.isfinite(x) for x in (min_relevance, w_text, w_content)):
        raise ValueError("threshold and weights must be finite")
    includes, excludes = _tags(include_tags), _tags(exclude_tags)
    eligible = []
    for digest, record in records.items():
        if family == "unknown" or record.get("family") != family:
            continue
        tags = _tags(record.get("tags", ()))
        if tags & excludes or (record.get("axis") == "functional" and not tags & includes):
            continue
        eligible.append(digest)
    eligible.sort()
    if not eligible:
        return []

    fallback_digests = set(fallback_digests)
    assignments = codebook.assignments
    centroids = codebook.centroids
    candidates = {digest for digest in eligible if digest not in assignments}
    clustered = {assignments[digest] for digest in eligible if digest in assignments}
    aligned = None
    if codebook.alignment is not None:
        aligned = torch.as_tensor(
            text_query if cluster_query is None else cluster_query,
            dtype=torch.float32,
            device="cpu",
        ).flatten()
        aligned = aligned @ codebook.alignment.to(dtype=torch.float32, device="cpu")
    ranked_clusters = sorted(
        clustered,
        key=lambda cluster: (
            -(
                w_text * _cosine(aligned, centroids[cluster])
                + w_content * _content_share(centroids[cluster], probe, groups)
            ),
            cluster,
        ),
    )
    for cluster in ranked_clusters:
        if len(candidates) >= 4 * k:
            break
        candidates.update(d for d in eligible if assignments.get(d) == cluster)

    results = []
    for digest in sorted(candidates):
        score = w_text * _cosine(
            fallback_query if digest in fallback_digests else text_query, text_features.get(digest)
        ) + w_content * (
            _content_share(
                summaries.get(digest),
                probe,
                groups,
                None if directions is None else directions.get(digest),
            )
        )
        if score >= min_relevance or (min_relevance == 0 and k >= len(eligible)):
            results.append(RetrievalResult(digest, score))
    results.sort(key=lambda result: (-result.score, result.digest))
    return results[:k]
