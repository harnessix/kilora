# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F

_LATENT: list[tuple[int, ...]] = []


def tokens_from_image(
    mask: torch.Tensor,
    grid: tuple[int, int],
    before: int = 0,
    after: int = 0,
    other: float = 0.0,
) -> torch.Tensor:
    m = mask.float()
    if m.dim() == 2:
        m = m[None]
    h, w = grid
    cells = m if tuple(m.shape[-2:]) == (h, w) else F.adaptive_avg_pool2d(m[:, None], (h, w))[:, 0]
    tokens = cells.reshape(m.shape[0], h * w).clamp(0, 1)
    if before or after:
        pad_b = tokens.new_full((tokens.shape[0], before), other)
        pad_a = tokens.new_full((tokens.shape[0], after), other)
        tokens = torch.cat([pad_b, tokens, pad_a], -1)
    return tokens


def grid_for(tokens: int, latent: tuple[int, int]) -> tuple[int, int] | None:
    hl, wl = latent
    for f in (1, 2, 4, 8, 16, 32, 64):
        for rnd in (math.ceil, math.floor):
            h, w = rnd(hl / f), rnd(wl / f)
            if h > 0 and w > 0 and h * w == tokens:
                return h, w
    return None


@contextlib.contextmanager
def latent_size(size: tuple[int, ...]) -> Iterator[None]:
    _LATENT.append(tuple(int(s) for s in size))
    try:
        yield
    finally:
        _LATENT.pop()


def current_latent() -> tuple[int, int] | None:
    if not _LATENT or len(_LATENT[-1]) < 2:
        return None
    h, w = _LATENT[-1][-2:]
    return h, w


def current_latent_shape() -> tuple[int, ...] | None:
    return _LATENT[-1] if _LATENT else None


_LAYOUTS: list = []


@contextlib.contextmanager
def token_layouts(resolve, text_tokens: int | None = None) -> Iterator[None]:
    _LAYOUTS.append((resolve, text_tokens))
    try:
        yield
    finally:
        _LAYOUTS.pop()


def current_text_tokens() -> int | None:
    return _LAYOUTS[-1][1] if _LAYOUTS else None


def current_layout(target: str):
    if not _LAYOUTS or _LAYOUTS[-1][0] is None:
        return None
    return _LAYOUTS[-1][0](target)


def layer_region(
    mask: torch.Tensor,
    tokens: int,
    latent: tuple[int, ...] | None = None,
    target: str | None = None,
):

    if target is not None and _LAYOUTS:
        resolve, text_tokens = _LAYOUTS[-1]
        layout = resolve(target) if resolve is not None else None
        if layout is not None and (layout.patch is not None or layout.kind in ("text1d", "set")):
            from .family_spec import map_region

            shape = latent or current_latent_shape()
            if shape is None and layout.kind not in ("text1d", "set"):
                raise ValueError("a region needs the latent size (latent_size)")
            return map_region(mask, tokens, layout, shape or (), text_tokens=text_tokens)
    latent = latent[-2:] if latent is not None and len(latent) > 2 else latent
    latent = latent or current_latent()
    if latent is None:
        raise ValueError("a region needs the latent size (latent_size)")
    grid = grid_for(tokens, latent)
    if grid is None:
        raise ValueError(f"no token grid of a {latent[0]}x{latent[1]} latent has {tokens} tokens")
    return tokens_from_image(mask, grid)
