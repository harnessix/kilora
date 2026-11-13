# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch

__all__ = [
    "SIZE_NORMS",
    "S_REF",
    "applied_scale",
    "base_identity",
    "base_key",
    "base_norms",
    "dora_delta",
    "layer_delta_sq",
    "low_rank_sq",
    "size_note",
    "size_scale",
    "weight_sq",
]

SIZE_NORMS = ("off", "weight")

S_REF = 0.01785


def low_rank_sq(down: torch.Tensor, up: torch.Tensor, scale: float = 1.0) -> float:
    d = down.detach().reshape(down.shape[0], -1).to(torch.float64)
    u = up.detach().reshape(up.shape[0], -1).to(torch.float64)
    gram = (u.t() @ u) * (d @ d.t())
    return float(scale) ** 2 * float(gram.sum())


def dora_delta(w0: torch.Tensor, delta: torch.Tensor, dora_scale: torch.Tensor) -> torch.Tensor:
    w0 = w0.detach().to(torch.float64)
    calc = w0 + delta.detach().to(torch.float64)
    m = dora_scale.detach().to(torch.float64)
    if m.dim() == 2 and m.shape[0] == calc.shape[0] and m.shape[1] == 1 and calc.shape[0] != 1:
        norm = calc.norm(dim=1, keepdim=True)
    else:
        m = m.reshape(1, -1)
        norm = calc.norm(dim=0, keepdim=True)
    return m * calc / norm - w0


def layer_delta_sq(
    tensors: Mapping[str, torch.Tensor],
    scale: float = 1.0,
    w0: torch.Tensor | None = None,
    rows: tuple[int, int] | None = None,
) -> float | None:

    if "down" in tensors and "up" in tensors:
        down, up = tensors["down"], tensors["up"]
        if rows is not None:
            up = up.chunk(rows[1], dim=0)[rows[0]]
        if "dora_scale" in tensors:
            if w0 is None:
                return None
            delta = (up.to(torch.float64) @ down.to(torch.float64)) * float(scale)
            dora = tensors["dora_scale"]
            if rows is not None and dora.shape[0] == tensors["up"].shape[0]:
                dora = dora.chunk(rows[1], dim=0)[rows[0]]
            return float(dora_delta(w0, delta, dora).pow(2).sum())
        return low_rank_sq(down, up, scale)
    from .multi_import import lokr_parts, lycoris_delta

    if not any(k.startswith("lokr_t") for k in tensors):
        pair = lokr_parts(dict(tensors))
        if pair is not None:
            a, b = (p.detach().to(torch.float64) for p in pair)
            return float(a.pow(2).sum()) * float(b.pow(2).sum())
    found = lycoris_delta(dict(tensors))
    if found is None:
        return None
    return float(found[0].to(torch.float64).pow(2).sum())


def size_scale(size: float | None) -> float:
    if size is None or not math.isfinite(size) or size <= 0.0:
        return 1.0
    return S_REF / size


def applied_scale(scale: float, radial_budget: float = 0.0) -> tuple[float, str]:

    del radial_budget
    return min(float(scale), 1.0), "shrink"


def size_note(
    name: str,
    size: float | None,
    scale: float,
    applied: float | None = None,
    rule: str | None = None,
) -> str:
    if size is None:
        text = (
            f"{name}: size_norm=weight, size S_f unknown (no float base weights), "
            f"scale c_f {scale:.4g}"
        )
    else:
        text = f"{name}: size_norm=weight, size S_f {size:.4g}, scale c_f {scale:.4g}"
    if applied is not None:
        text += f", applied scale s {applied:.4g} ({rule or 'shrink'})"
    return text + "."


def weight_sq(w: torch.Tensor) -> float:
    w = w.detach()
    if not w.dtype.is_floating_point:
        w = w.float()
    norm = torch.linalg.vector_norm(w.reshape(-1), dtype=torch.float32)
    return float(norm) ** 2


_BASE_NORMS: dict[tuple, dict[str, float]] = {}


def weight_identity(w: torch.Tensor) -> tuple:

    version = None if w.is_inference() else w._version
    return (w.data_ptr(), version, tuple(w.shape), str(w.dtype), str(w.device))


def base_norms(weights: Mapping[str, torch.Tensor]) -> dict[str, float]:
    identity = getattr(weights, "weight_identity", None)
    key = (
        tuple(sorted((t, *identity(t)) for t in weights))
        if callable(identity)
        else tuple(sorted((t, *weight_identity(w)) for t, w in weights.items()))
    )
    found = _BASE_NORMS.get(key)
    if found is None:
        found = {t: weight_sq(w) for t, w in weights.items()}
        if len(_BASE_NORMS) > 8:
            _BASE_NORMS.clear()
        _BASE_NORMS[key] = found
    return found


def base_identity(
    norms: Mapping[str, float], shapes: Mapping[str, tuple[int, ...]] | None = None
) -> str | None:

    if not norms:
        return None
    values = [float(v) for v in norms.values()]
    total = math.fsum(values)
    second = math.fsum(v * v for v in values)
    shape = sorted([int(n) for n in (shapes or {}).get(t, ())] for t in norms)
    text = json.dumps([shape, f"{total:.3e}", f"{second:.3e}"])
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def base_key(weights: Mapping[str, torch.Tensor]) -> tuple[dict[str, float], str | None]:
    norms = base_norms(weights)
    shapes = getattr(weights, "shapes", None)
    if shapes is None:
        shapes = {t: tuple(w.shape) for t, w in weights.items()}
    return norms, base_identity(norms, shapes)
