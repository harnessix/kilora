# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

PIXELS_PER_LATENT = 8
LATENT_MULTIPLE = 2
SIZE_MULTIPLE = PIXELS_PER_LATENT * LATENT_MULTIPLE
DEFAULT_MEASURE_SIZE = 512
MIN_MEASURE_SIZE = 256
MAX_MEASURE_SIZE = 4096
DEFAULT_REDUCER = "latent"

Reducer = Callable[[torch.Tensor, int, Any], torch.Tensor]


def check_measure_size(size: int) -> int:
    size = int(size)
    if not MIN_MEASURE_SIZE <= size <= MAX_MEASURE_SIZE:
        raise ValueError(
            f"the measurement size must be {MIN_MEASURE_SIZE} to {MAX_MEASURE_SIZE} pixels"
        )
    return max(MIN_MEASURE_SIZE, int(round(size / SIZE_MULTIPLE)) * SIZE_MULTIPLE)


def _target_cells(h: int, w: int, size: int) -> tuple[int, int]:
    scale = size / (max(h, w) * PIXELS_PER_LATENT)

    def cells(n):
        return max(LATENT_MULTIPLE, int(round(n * scale / LATENT_MULTIPLE)) * LATENT_MULTIPLE)

    return cells(h), cells(w)


def latent_area(latent: torch.Tensor, size: int, vae: Any = None) -> torch.Tensor:
    h, w = _target_cells(*latent.shape[-2:], size)
    if latent.dim() == 4:
        return F.interpolate(latent.float(), size=(h, w), mode="area").to(latent.dtype)
    b, c, *frames = latent.shape[:-2]
    flat = latent.reshape(b, c, -1, *latent.shape[-2:]).movedim(2, 1)
    flat = flat.reshape(-1, c, *latent.shape[-2:]).float()
    out = F.interpolate(flat, size=(h, w), mode="area")
    out = out.reshape(b, -1, c, h, w).movedim(1, 2).reshape(b, c, *frames, h, w)
    return out.to(latent.dtype)


def decode_resize_encode(latent: torch.Tensor, size: int, vae: Any) -> torch.Tensor:
    if vae is None:
        raise ValueError("decode-resize-encode needs a VAE")
    images = vae.decode(latent)
    if images.dim() == 5:
        images = images.reshape(-1, *images.shape[-3:])
    h, w = _target_cells(*latent.shape[-2:], size)
    small = F.interpolate(
        images.movedim(-1, 1).float(),
        size=(h * PIXELS_PER_LATENT, w * PIXELS_PER_LATENT),
        mode="area",
    )
    return vae.encode(small.movedim(1, -1)[..., :3]).to(latent.device, latent.dtype)


REDUCERS: dict[str, Reducer] = {
    "latent": latent_area,
    "decode-resize-encode": decode_resize_encode,
}


def register_reducer(name: str, fn: Reducer) -> None:
    REDUCERS[name] = fn


def measurement_latent(
    latent: torch.Tensor, *, size: int, reducer: str, vae: Any = None
) -> tuple[torch.Tensor, str]:
    if size <= 0 or latent.dim() < 4 or max(latent.shape[-2:]) * PIXELS_PER_LATENT <= size:
        return latent, ""
    name = reducer if reducer in REDUCERS else DEFAULT_REDUCER
    if name == "decode-resize-encode" and vae is None:
        return latent_area(latent, size), "No VAE is connected, so the latent was reduced."
    return REDUCERS[name](latent, size, vae), ""


def resolution_clause(measure: Any) -> str:
    size = getattr(measure, "resolution", None)
    tokens = getattr(measure, "tokens", None)
    if not size or not tokens:
        return ""
    h, w = size
    return f", samples at {w}x{h} px, {tokens} tokens per row"
