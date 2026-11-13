# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(eq=False)
class DepthMap:
    values: torch.Tensor
    min_depth: float
    max_depth: float
    start: float
    base: float

    def depths(self) -> torch.Tensor:
        return self.min_depth + self.values * (self.max_depth - self.min_depth)

    def applies(self, progress: float) -> tuple[bool, float]:
        from .call_values import current_token_depth

        on, base = current_token_depth()
        if on is None:
            on = not (self.start > 0 and progress < self.start)
        return on, self.base if base is None else base

    def on_tokens(self, region, before_start: bool, base: float | None = None):
        base = self.base if base is None else float(base)
        mapped = region(self.values)
        if mapped is None:
            return None
        if before_start:
            return torch.full_like(mapped, base)
        inside = region(self._inside())
        depth = self.min_depth + mapped * (self.max_depth - self.min_depth)
        return torch.where(inside > 0.5, depth, torch.full_like(depth, base))

    def _inside(self) -> torch.Tensor:
        ones = self.__dict__.get("_ones")
        if ones is None:
            ones = self.__dict__["_ones"] = torch.ones_like(self.values)
        return ones


def luminance(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image[None]
    if image.shape[-1] == 1:
        return image[..., 0].float()
    rgb = image[..., :3].float()
    return rgb @ torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype)


def feather(values: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return values
    sigma = radius / 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=values.device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).to(values.dtype)
    m = values[:, None]
    m = F.pad(m, (radius, radius, radius, radius), mode="replicate")
    m = F.conv2d(m, kernel.view(1, 1, 1, -1))
    m = F.conv2d(m, kernel.view(1, 1, -1, 1))
    return m[:, 0]


def make_depth_map(
    values: torch.Tensor,
    min_depth: float = 1.0,
    max_depth: float = 3.0,
    invert: bool = False,
    gamma: float = 1.0,
    feather_px: int = 0,
    start: float = 0.3,
    base: float | None = None,
) -> DepthMap:

    if min(min_depth, max_depth) < 0:
        raise ValueError(f"depths must be at least 0, got {min_depth} and {max_depth}")
    if gamma <= 0:
        raise ValueError(f"gamma must be above 0, got {gamma}")
    if not 0.0 <= float(start) <= 1.0:
        raise ValueError(f"start must be a sampling progress in [0, 1], got {start}")
    v = values.detach().float().cpu()
    if v.dim() == 2:
        v = v[None]
    v = v.clamp(0, 1)
    if invert:
        v = 1 - v
    if gamma != 1:
        v = v**gamma
    v = feather(v, feather_px).clamp(0, 1)
    out = DepthMap(v, float(min_depth), float(max_depth), float(start), 0.0)
    if base is None or base < 0:
        base = math.floor(float(out.depths().mean()) + 0.5)
    out.base = float(base)
    return out


__all__ = ["DepthMap", "feather", "luminance", "make_depth_map"]
