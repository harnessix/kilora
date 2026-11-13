# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path

import torch

from ..regions import token_weights
from . import text
from .codebook import FamilyCodebooks, StreamingCodebook
from .index import LibraryIndex, SyncReport
from .loading import ByteLRU, LibraryLoader, LoadReport, SelectedFile
from .retrieve import RetrievalResult, retrieve

__all__ = [
    "ByteLRU",
    "FamilyCodebooks",
    "LibraryIndex",
    "LibraryLoader",
    "LoadReport",
    "RetrievalResult",
    "SelectionResult",
    "SelectedFile",
    "StreamingCodebook",
    "SyncReport",
    "retrieve",
    "select",
]


@dataclass(frozen=True)
class SelectionResult:
    path: str
    sha256: str
    score: float
    text_fallback: bool = False
    regions: tuple[str, ...] = ()

    @property
    def digest(self):
        return self.sha256


def select(
    index,
    prompt,
    probe,
    family,
    count,
    *,
    clip=None,
    seed=0,
    min_relevance=0.2,
    include_tags=(),
    exclude_tags=(),
    regions=None,
):

    inputs = dict(index.retrieval_inputs(family, clip))
    fallback = inputs.pop("fallback")
    query = text.embed_text(prompt, clip)
    trigram_query = query if clip is None else text.embed_text(prompt)
    records = inputs["records"]

    def run(tokens):
        return retrieve(
            **inputs,
            text_query=query,
            cluster_query=trigram_query,
            fallback_query=trigram_query,
            fallback_digests=fallback,
            probe=tokens,
            family=family,
            k=count,
            seed=seed,
            min_relevance=min_relevance,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

    if regions is None:
        results = [(item, ()) for item in run(probe)]
    else:
        chosen = {}
        for label, mask in regions.items():
            tokens = _region_probe(probe, mask)
            if not tokens:
                continue
            for item in run(tokens):
                previous = chosen.get(item.digest)
                if previous is None:
                    chosen[item.digest] = (item, (label,))
                else:
                    best, labels = previous
                    best = item if item.score > best.score else best
                    chosen[item.digest] = (best, (*labels, label))
        results = sorted(chosen.values(), key=lambda entry: (-entry[0].score, entry[0].digest))
    return [
        SelectionResult(
            str(Path(records[item.digest]["folder"]) / records[item.digest]["filename"]),
            item.digest,
            item.score,
            item.digest in fallback,
            labels,
        )
        for item, labels in results
    ]


def _region_probe(probe, mask):
    mask = torch.as_tensor(mask).detach().float().cpu()
    if mask.ndim != 2 or not bool(torch.isfinite(mask).all()):
        raise ValueError("region masks must be finite [H, W] tensors")
    if mask.numel() == 0 or bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("region masks must be nonempty and in [0, 1]")
    mask = mask[None, None]
    result = {}
    for group, value in probe.items():
        tokens = torch.as_tensor(value).detach().float().cpu()
        if tokens.ndim < 2:
            raise ValueError("probe tokens must have a token and feature dimension")
        if tokens.ndim > 2:
            tokens = tokens[:1]
        tokens = tokens.reshape(1, -1, tokens.shape[-1])
        weights = token_weights(mask, tokens.shape[1])
        if weights is None or not bool(weights.sum() > 0):
            continue
        result[group] = tokens * (weights / weights.mean()).sqrt().unsqueeze(-1)
    return result
