# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import weakref
from collections.abc import Iterator, Mapping, Sequence

import torch

from .recording import ContextStack

_DEPTH = ContextStack("kilora_call_depth")
_EXPERTS = ContextStack("kilora_call_experts")
_ALPHAS = ContextStack("kilora_call_alphas")
_AXES = ContextStack("kilora_call_axes")
_ROW_AXES = ContextStack("kilora_call_row_axes")
_ALPHA_CACHES = ContextStack("kilora_call_alpha_caches")
_IP_EARLY = ContextStack("kilora_call_ip_early")


def ip_early_threshold() -> float:
    try:
        value = float(os.environ.get("KILORA_AXIS_IP_EARLY", "") or 0)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) and 0 < value <= 1 else 0.0


@contextlib.contextmanager
def ip_early(active: bool, mounted: bool | None = None) -> Iterator[None]:
    with _IP_EARLY.pushed((active, mounted)):
        yield


def current_ip_early() -> bool:
    return _IP_EARLY[-1][0] if _IP_EARLY else False


def current_ip_mounted() -> bool | None:
    return _IP_EARLY[-1][1] if _IP_EARLY else None


@contextlib.contextmanager
def token_depth_values(on: bool | None = None, base: float | None = None) -> Iterator[None]:
    with _DEPTH.pushed((on, None if base is None else float(base))):
        yield


def current_token_depth() -> tuple[bool | None, float | None]:
    return _DEPTH[-1] if _DEPTH else (None, None)


@contextlib.contextmanager
def expert_scales(scales: Mapping[str, float]) -> Iterator[None]:
    with _EXPERTS.pushed({str(k): float(v) for k, v in scales.items()}):
        yield


def expert_scale_vector(names: Sequence[str]) -> list[float] | None:
    if not _EXPERTS or not names:
        return None
    given = _EXPERTS[-1]
    if not any(n in given for n in names):
        return None
    return [given.get(n, 1.0) for n in names]


@contextlib.contextmanager
def axis_alphas(alphas: Mapping[str, float | torch.Tensor]) -> Iterator[None]:
    with _ALPHAS.pushed(dict(alphas)), _ALPHA_CACHES.pushed({}):
        yield


def current_alpha_cache():
    return _ALPHA_CACHES[-1] if _ALPHA_CACHES else None


def current_axis_alphas() -> Mapping[str, float | torch.Tensor]:
    return _ALPHAS[-1] if _ALPHAS else {}


@contextlib.contextmanager
def active_axes(axes: Sequence[str]) -> Iterator[None]:
    with _AXES.pushed(frozenset(axes)):
        yield


def current_active_axes() -> frozenset[str] | None:
    return _AXES[-1] if _AXES else None


@contextlib.contextmanager
def row_axes(ids: torch.Tensor | None) -> Iterator[None]:
    with _ROW_AXES.pushed(None if ids is None else tuple(ids.detach().reshape(-1).tolist())):
        yield


def current_row_axes() -> tuple[int, ...] | None:
    return _ROW_AXES[-1] if _ROW_AXES else None


def contiguous_row_slices(values: tuple[int, ...]) -> dict[int, slice] | None:
    runs = {}
    for i, value in enumerate(values):
        if value not in runs:
            runs[value] = slice(i, i + 1)
        elif runs[value].stop == i:
            runs[value] = slice(runs[value].start, i + 1)
        else:
            return None
    return runs


__all__ = [
    "active_axes",
    "current_active_axes",
    "current_token_depth",
    "expert_scale_vector",
    "expert_scales",
    "token_depth_values",
    "axis_alphas",
    "current_axis_alphas",
]


_PASSES = ContextStack("kilora_call_passes")
_UP_DELTAS = ContextStack("kilora_call_up_deltas")


@contextlib.contextmanager
def depth_pass(depth: int, grad_step: int | None = None) -> Iterator[None]:
    if grad_step is not None and not 1 <= grad_step <= depth:
        raise ValueError(f"grad_step {grad_step} is not within 1..{depth}")
    with _PASSES.pushed((int(depth), None if grad_step is None else int(grad_step))):
        yield


def current_depth_pass() -> tuple[int, int | None] | None:
    return _PASSES[-1] if _PASSES else None


@contextlib.contextmanager
def up_deltas(deltas: Mapping[str, Mapping[str, torch.Tensor]]) -> Iterator[None]:
    with _UP_DELTAS.pushed({str(k): dict(v) for k, v in deltas.items()}):
        yield


def current_up_deltas() -> Mapping[str, Mapping[str, torch.Tensor]]:
    return _UP_DELTAS[-1] if _UP_DELTAS else {}


_CROPS = ContextStack("kilora_call_crops")


@contextlib.contextmanager
def crop_window(boxes):
    value = None if boxes is None else tuple(tuple(map(int, row)) for row in boxes)
    with _CROPS.pushed(value):
        yield


def current_crop_window():
    return _CROPS[-1] if _CROPS else None


def crop_map(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    boxes = current_crop_window()
    if boxes is None or values.ndim != 3:
        return values
    h, w = values.shape[-2:]
    rows = []
    for i, (top, left, full_h, full_w) in enumerate(boxes):
        y0, x0 = round(top * h / full_h), round(left * w / full_w)
        y1 = round((top + size[0]) * h / full_h)
        x1 = round((left + size[1]) * w / full_w)
        source = values[i % values.shape[0] : i % values.shape[0] + 1]
        if y1 <= y0 or x1 <= x0:
            source = F.interpolate(source[:, None].float(), size=(full_h, full_w), mode="nearest")[
                :, 0
            ]
            y0, x0, y1, x1 = top, left, top + size[0], left + size[1]
        row = source[:, y0:y1, x0:x1]
        target = (max(1, round(size[0] * h / full_h)), max(1, round(size[1] * w / full_w)))
        if row.shape[-2:] != target:
            row = F.interpolate(row[:, None].float(), size=target, mode="area")[:, 0]
        rows.append(row)
    return torch.cat(rows)


_MOUNT_SHARES = ContextStack("kilora_call_mount_shares")


@contextlib.contextmanager
def mount_shares(shares: Mapping[str, torch.Tensor]) -> Iterator[None]:
    with _MOUNT_SHARES.pushed(shares):
        yield


def current_mount_share(tag: str, expert: int, count: int):
    if count == 1:
        return None
    shares = _MOUNT_SHARES[-1].get(tag) if _MOUNT_SHARES else None
    return 1.0 / count if shares is None else shares[..., expert]


def selected_mount_shares(indices: torch.Tensor) -> Mapping[str, torch.Tensor]:
    if not _MOUNT_SHARES:
        return {}
    return {
        key: value.index_select(0, indices.to(value.device))
        for key, value in _MOUNT_SHARES[-1].items()
    }


_ROUTE_ROWS = ContextStack("kilora_call_route_rows")


@contextlib.contextmanager
def route_rows(indices: torch.Tensor) -> Iterator[None]:
    with _ROUTE_ROWS.pushed(indices):
        yield


def current_route_rows() -> torch.Tensor | None:
    return _ROUTE_ROWS[-1] if _ROUTE_ROWS else None


_COMPOSE = ContextStack("kilora_call_compose")


@contextlib.contextmanager
def compose_layout(layouts, rows: torch.Tensor | None) -> Iterator[None]:
    with _COMPOSE.pushed(None if layouts is None else (tuple(layouts), rows)):
        yield


def current_compose_layout() -> tuple[tuple, torch.Tensor] | None:
    return _COMPOSE[-1] if _COMPOSE else None


@dataclasses.dataclass(eq=False)
class ComposedAlpha:
    image: torch.Tensor
    text: torch.Tensor | None = None


_HOST_SCALARS = ContextStack("kilora_call_host_scalars")
_HOST_LISTS: dict = {}
_HOST_LIST_LIMIT = 4


def _host_key(t: torch.Tensor) -> tuple:
    try:
        version = t._version
    except RuntimeError:
        version = None
    return (version, t.data_ptr(), tuple(t.shape), t.dtype, str(t.device))


@contextlib.contextmanager
def host_values() -> Iterator[None]:
    with _HOST_SCALARS.pushed({}):
        yield


def host_scalar(t) -> float:
    if not isinstance(t, torch.Tensor):
        return float(t)
    if not _HOST_SCALARS:
        return float(t.detach().flatten()[0])
    cache = _HOST_SCALARS[-1]
    key = _host_key(t)
    entry = cache.get(id(t))
    if entry is not None and entry[0] is t and entry[1] == key:
        return entry[2]
    value = float(t.detach().flatten()[0])
    cache[id(t)] = (t, key, value)
    return value


def host_list(t: torch.Tensor) -> tuple:
    key = _host_key(t)
    entry = _HOST_LISTS.get(id(t))
    if entry is not None and entry[0]() is t and entry[1] == key:
        return entry[2]
    values = tuple(t.detach().flatten().tolist())
    _HOST_LISTS.pop(id(t), None)
    while len(_HOST_LISTS) >= _HOST_LIST_LIMIT:
        _HOST_LISTS.pop(next(iter(_HOST_LISTS)))
    _HOST_LISTS[id(t)] = (weakref.ref(t), key, values)
    return values
