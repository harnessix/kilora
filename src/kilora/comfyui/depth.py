# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch

from ..depth_schedule import KINDS, MAX_DEPTH, PRESETS, DepthSchedule, compile_schedule

DEPTH = "KILORA_DEPTH"
MANIFEST_KEY = "depth_schedule"
PRESET_NAMES = (*PRESETS, "auto")
PREVIEW_BLOCKS = 28
PREVIEW_STEPS = 20
CELL = 8


def compile_depth(spec: str | Sequence[float] | None) -> DepthSchedule | None:

    return compile_schedule(spec)


def from_manifest(value: Any) -> DepthSchedule | None:
    if isinstance(value, dict):
        return DepthSchedule.from_dict(value)
    return compile_depth(value)


def node_spec(text: str, preset: str) -> Any:
    if text.strip():
        return compile_depth(text)
    return None if preset == "none" else compile_depth(preset)


def effective(mods: Sequence[Any], requested: Any) -> tuple[Any, str | None]:
    if requested is not None:
        return requested, "workflow"
    for m in mods:
        value = (getattr(m, "manifest", None) or {}).get(MANIFEST_KEY)
        if value not in (None, "", []):
            return from_manifest(value), m.mod_id
    return None, None


def is_auto(schedule: Any) -> bool:
    return isinstance(schedule, DepthSchedule) and schedule.is_auto


def grid(
    schedule: Any,
    blocks: int = PREVIEW_BLOCKS,
    steps: int = PREVIEW_STEPS,
    kind: str = "self_attn",
) -> torch.Tensor:
    out = torch.zeros(steps, blocks)
    if schedule is None:
        return out
    for s in range(steps):
        t = s / max(steps - 1, 1)
        for i in range(blocks):
            out[s, i] = float(schedule.depth(i, blocks, kind, t, ""))
    return out


def max_depth(schedule: Any, blocks: int = PREVIEW_BLOCKS) -> int:
    if type(schedule) is DepthSchedule:
        return _cached_max_depth(schedule, blocks, PREVIEW_STEPS, torch.get_default_dtype())
    return _max_depth(schedule, blocks)


@lru_cache(maxsize=128)
def _cached_max_depth(
    schedule: DepthSchedule, blocks: int, preview_steps: int, dtype: torch.dtype
) -> int:
    if schedule.mode != "expr" or blocks <= 0 or preview_steps <= 0:
        return _max_depth(schedule, blocks)
    names = {node.id for node in ast.walk(schedule._tree) if isinstance(node, ast.Name)}
    indices = range(blocks) if "i" in names else (0,)
    steps = range(preview_steps) if "t" in names else (0,)
    kinds = KINDS if names.intersection(("self", "cross", "mlp")) else (KINDS[0],)
    values = [
        schedule.depth(i, blocks, kind, step / max(preview_steps - 1, 1), "")
        for kind in kinds
        for step in steps
        for i in indices
    ]
    top = float(torch.tensor(values, dtype=dtype).max())
    return int(math.ceil(top - 1e-9))


def _max_depth(schedule: Any, blocks: int) -> int:
    if is_auto(schedule):
        return int(MAX_DEPTH)
    top = max(float(grid(schedule, blocks, PREVIEW_STEPS, k).max()) for k in KINDS)
    return int(math.ceil(top - 1e-9))


_STOPS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.23, 0.11, 0.45],
        [0.13, 0.45, 0.56],
        [0.21, 0.72, 0.47],
        [0.99, 0.91, 0.14],
    ]
)


def heatmap(values: torch.Tensor, cell: int = CELL) -> torch.Tensor:
    top = max(float(values.max()), 1.0)
    x = (values / top).clamp(0, 1) * (len(_STOPS) - 1)
    low = x.floor().long().clamp(max=len(_STOPS) - 2)
    f = (x - low).unsqueeze(-1)
    rgb = _STOPS[low] * (1 - f) + _STOPS[low + 1] * f
    rgb = rgb.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
    return rgb.unsqueeze(0).contiguous()


def describe(schedule: Any, source: str | None = None) -> str:
    if schedule is None:
        return "Depth schedule: none (the depth input applies)."
    where = f" (from {source})" if source and source != "workflow" else ""
    return f"Depth schedule: {schedule!r}{where}."
