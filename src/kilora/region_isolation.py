# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionIsolation:
    strength: float = 1.0
    until: float = 0.6
    feather: int = 32
    timeline: tuple = ()

    def active(self, progress: float) -> bool:
        return self.strength > 0 and progress < self.until

    def at(self, clock) -> RegionIsolation:
        if not self.timeline:
            return self
        from dataclasses import replace

        from .timeline import window_value

        v = window_value(self.timeline, "region_isolation", clock)
        if v is None:
            return replace(self, timeline=())
        return replace(self, strength=max(v, 0.0), until=2.0, timeline=())


def outside_weight(
    masks: Sequence[torch.Tensor], shape: Sequence[int], feather: int, strength: float
) -> torch.Tensor:
    h, w = int(shape[-2]), int(shape[-1])
    union = None
    for mask in masks:
        m = mask.float()
        m = m[None] if m.dim() == 2 else m[:1]
        cells = F.adaptive_avg_pool2d(m[:, None], (h, w))[:, 0]
        union = cells if union is None else torch.maximum(union, cells)
    if union is None:
        union = torch.zeros(1, h, w)
    k = max(int(round(feather / 8)), 0)
    if k:
        union = F.avg_pool2d(union[:, None], 2 * k + 1, 1, k, count_include_pad=False)[:, 0]
    weight = float(strength) * (1 - union.clamp(0, 1))
    return weight.view(1, 1, *([1] * (len(shape) - 4)), h, w)


@dataclass
class BaseBranch:
    latent: torch.Tensor | None = None
    sigma: float | None = None
    outputs: list | None = None
    scale: float = 1.0
    calls: int = field(default=0)

    def latent_at(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.latent is None or self.sigma is None or sigma >= self.sigma:
            self.latent, self.outputs, self.calls = x.detach().clone(), None, 0
        elif self.outputs is not None:
            d = self.guided()
            self.latent = d + (sigma / self.sigma) * (self.latent - d)
        self.sigma = sigma
        return self.latent

    def guided(self) -> torch.Tensor:
        cond = self.outputs[0]
        uncond = self.outputs[1] if len(self.outputs) > 1 else None
        if uncond is None:
            return cond
        return uncond + self.scale * (cond - uncond)


def isolate(
    calc: Callable[[torch.Tensor, bool], list],
    x: torch.Tensor,
    sigma: float,
    weight: torch.Tensor,
    branch: BaseBranch,
) -> list:
    base_latent = branch.latent_at(x, sigma)
    w = weight.to(x.device, x.dtype)
    base = calc(base_latent, True)
    branch.outputs = list(base)
    branch.calls += 1
    modded = calc(x + w * (base_latent - x), False)
    return [
        m if b is None or m is None else m + w * (b - m) for m, b in zip(modded, base, strict=True)
    ]
