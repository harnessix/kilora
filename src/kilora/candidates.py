# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def expert_summaries(keys: torch.Tensor) -> torch.Tensor:

    if keys.ndim != 3 or keys.shape[1] != 1:
        raise ValueError("content keys must have shape [experts, 1, width]")
    return keys[:, 0, :]


def top_candidates(
    summary: torch.Tensor, query: torch.Tensor, count: int, rows: torch.Tensor | None = None
) -> torch.Tensor:
    if summary.ndim not in (2, 3) or query.ndim < 1 or query.shape[-1] != summary.shape[-1]:
        raise ValueError("summary and query must have shapes [E, d] or [E, rows, d] and [..., d]")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if summary.ndim == 2:
        summary = summary.unsqueeze(1)
    scores = (query @ summary.flatten(0, 1).transpose(0, 1)).abs()
    scores = scores.unflatten(-1, summary.shape[:2])
    if rows is not None:
        valid = (
            torch.arange(summary.shape[1], device=summary.device) < rows.to(summary.device)[:, None]
        )
        scores = scores.masked_fill(~valid, -torch.inf)
    scores = scores.amax(-1)
    return scores.topk(min(count, summary.shape[0]), dim=-1).indices
