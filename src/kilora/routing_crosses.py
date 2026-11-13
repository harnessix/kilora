# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .routing_features import RoutingFeatures

CROSSES = {
    "token_region": "region",
    "token_prompt": "prompt",
    "token_timestep": "timestep_embedding",
}


def cross_scores(
    logits: torch.Tensor,
    keys: torch.Tensor,
    features: RoutingFeatures,
    weights: Mapping[str, float],
    rows: torch.Tensor | None = None,
) -> torch.Tensor:

    unknown = set(weights) - CROSSES.keys()
    if unknown:
        raise ValueError(f"unknown routing crosses: {', '.join(sorted(unknown))}")
    active = []
    for name, weight in weights.items():
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError(f"routing cross {name} weight must be finite")
        if weight:
            active.append((name, weight))
    if not active:
        return logits
    if keys.ndim == 2:
        keys = keys.unsqueeze(1)
    if keys.ndim != 3 or keys.shape[0] != logits.shape[-1]:
        raise ValueError("routing cross keys must have rows for each expert")

    def project(value, name):
        if value is None or value.shape[-1] != keys.shape[-1]:
            raise ValueError(f"routing cross {name} needs a feature in the content-key space")
        return torch.einsum(
            "...d,emd->...em", value.float(), keys.to(device=value.device, dtype=torch.float32)
        )

    token = project(features.token, "token")
    result = logits
    for name, weight in active:
        terms = token * project(getattr(features, CROSSES[name]), name)
        if rows is not None:
            valid = (
                torch.arange(keys.shape[1], device=terms.device) < rows.to(terms.device)[:, None]
            )
            terms = terms.masked_fill(~valid, float("-inf"))
        result = result + weight * terms.max(dim=-1).values
    return result
