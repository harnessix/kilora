# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

OFF = "Trigger attention maps are off: "
_log = logging.getLogger(__name__)
_quantile_ignored_logged = False


def trigger_positions(family: str, conditioning: Mapping, triggers: Mapping[str, Sequence[int]]):

    if family.lower() != "anima":
        return {}, f"{OFF}trigger token positions are known on Anima only, not {family}."
    if not triggers:
        return {}, f"{OFF}no IP LoRA has trigger token ids."
    ids = conditioning.get("t5xxl_ids")
    if ids is None:
        return {}, f"{OFF}the positive conditioning has no t5xxl_ids."
    ids = torch.as_tensor(ids)
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1:
        return {}, f"{OFF}t5xxl_ids has shape {tuple(ids.shape)}, not one row."
    haystack = ids.tolist()
    positions = {}
    for name, tokens in triggers.items():
        tokens = list(tokens)
        found = set()
        if tokens:
            for start in range(len(haystack) - len(tokens) + 1):
                if haystack[start : start + len(tokens)] == tokens:
                    found.update(range(start, start + len(tokens)))
        if not found:
            return {}, f"{OFF}the trigger of {os.path.basename(str(name))} is not in the prompt."
        positions[name] = tuple(sorted(found))
    return positions, None


@dataclass
class AttentionMaps:
    positions: Mapping[str, Sequence[int]]
    blocks: int
    kind: str | None = None
    block: int | None = None
    _cond_or_uncond: Sequence[int] | None = field(default=None, init=False)
    _sums: dict[str, torch.Tensor] = field(default_factory=dict, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)

    def kind_patches(self) -> dict[str, Callable]:
        def mark(kind):
            def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
                self.kind = kind
                options = extra_options or {}
                self.block = options.get("block_index", options.get("transformer_index"))
                self._cond_or_uncond = options.get("cond_or_uncond")
                return {}

            return patch

        return {"attn1_patch": mark("self"), "attn2_patch": mark("cross")}

    def patch_options(self, transformer_options: Mapping) -> dict:
        options = dict(transformer_options)
        patches = {name: list(items) for name, items in (options.get("patches") or {}).items()}
        for name, patch in self.kind_patches().items():
            patches.setdefault(name, []).append(patch)
        options["patches"] = patches
        options["optimized_attention_override"] = self.comfy_override(
            options.get("optimized_attention_override")
        )
        return options

    def comfy_override(self, previous: Callable | None = None) -> Callable:
        def override(func, *args, **kwargs):
            kind, self.kind = self.kind, None
            if kind == "cross" and self.block is not None:
                self.collect(
                    args[0],
                    args[1],
                    args[3],
                    kwargs.get("skip_reshape", False),
                    kwargs.get("mask"),
                    self._cond_or_uncond,
                )
            if previous is not None:
                return previous(func, *args, **kwargs)
            return func(*args, **kwargs)

        return override

    def collect(self, q, k, heads: int, skip_reshape: bool = False, mask=None, cond_or_uncond=None):
        if not self.positions or self.blocks <= 0 or self.block is None:
            return
        if not (self.blocks // 3 <= self.block < max(self.blocks // 3 + 1, 2 * self.blocks // 3)):
            return
        if not skip_reshape:
            q = q.reshape(q.shape[0], q.shape[1], heads, -1).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], heads, -1).transpose(1, 2)
        if any(not p or min(p) < 0 or max(p) >= k.shape[-2] for p in self.positions.values()):
            return
        with torch.no_grad():
            batch = q.shape[0]
            if cond_or_uncond is not None:
                groups = list(cond_or_uncond)
                if not groups or batch % len(groups):
                    return
                rows_per_group = batch // len(groups)
                rows = [
                    i
                    for group, kind in enumerate(groups)
                    if kind == 0
                    for i in range(group * rows_per_group, (group + 1) * rows_per_group)
                ]
                if not rows:
                    return
            else:
                rows = list(range(batch))
            if mask is not None:
                mask = torch.broadcast_to(mask, (batch, heads, q.shape[-2], k.shape[-2]))
            q, k = q[rows], k[rows].float()
            values = {name: [] for name in self.positions}
            for start in range(0, q.shape[-2], 1024):
                scores = torch.matmul(q[..., start : start + 1024, :].float(), k.transpose(-1, -2))
                scores = scores / math.sqrt(q.shape[-1])
                if mask is not None:
                    part = mask[..., start : start + 1024, :][rows]
                    scores = (
                        scores.masked_fill(~part, float("-inf"))
                        if part.dtype == torch.bool
                        else scores + part
                    )
                normalizer = torch.logsumexp(scores, dim=-1, keepdim=True)
                for name, positions in self.positions.items():
                    value = (scores[..., list(positions)] - normalizer).exp().sum(-1).mean(1)
                    values[name].append(value)
            for name, chunks in values.items():
                value = torch.cat(chunks, dim=-1).detach()
                if name in self._sums and self._sums[name].shape != value.shape:
                    raise ValueError("attention map shapes differ within a step")
                self._sums[name] = self._sums.get(name, 0) + value
                self._counts[name] = self._counts.get(name, 0) + 1

    def finish_step(self) -> dict[str, torch.Tensor]:
        result = {name: value / self._counts[name] for name, value in self._sums.items()}
        self._sums.clear()
        self._counts.clear()
        return result


class _Alphas(dict):
    def __init__(self, values, history):
        super().__init__(values)
        self.history = history


FULL_CONTRAST = 20.0
LEVEL_FLOOR = 0.01
LEVEL_FULL = 0.05


def map_confidence(value: torch.Tensor) -> torch.Tensor:
    floor, full = LEVEL_FLOOR, LEVEL_FULL
    override = os.environ.get("KILORA_ALPHA_LEVEL")
    if override is not None:
        try:
            floor, full = (float(part) for part in override.split(","))
            if not (math.isfinite(floor) and math.isfinite(full) and 0 <= floor < full):
                raise ValueError
        except ValueError as exc:
            raise ValueError("KILORA_ALPHA_LEVEL must be two floats: 0 <= floor < full") from exc
    value = value.float()
    p99 = torch.quantile(value, 0.99, dim=-1)
    p50 = torch.quantile(value, 0.5, dim=-1)
    ratio = torch.where(p99 > 0, p99 / p50.clamp_min(p99 * 1e-6 + 1e-30), torch.ones_like(p99))
    contrast = (ratio.clamp_min(1.0).log() / math.log(FULL_CONTRAST)).clamp(0, 1)
    level = ((p99 - floor) / (full - floor)).clamp(0, 1)
    return contrast * level


def alpha_from_maps(maps: Mapping[str, torch.Tensor], previous=None, *, grid=None, share=1.0):

    global _quantile_ignored_logged
    hard_mode = os.environ.get("KILORA_ALPHA_HARD") == "1"
    quantile_setting = os.environ.get("KILORA_ALPHA_QUANTILE")
    quantile = None
    if quantile_setting is not None:
        if hard_mode:
            try:
                quantile = float(quantile_setting)
                if not 0 < quantile < 1:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("KILORA_ALPHA_QUANTILE must be a float in (0, 1)") from exc
        elif not _quantile_ignored_logged:
            _log.warning("KILORA_ALPHA_QUANTILE has no effect unless KILORA_ALPHA_HARD=1")
            _quantile_ignored_logged = True
    if not maps:
        return {}
    history = getattr(previous, "history", {})
    current = {}
    placed = []
    for name, value in maps.items():
        if value.ndim != 2:
            raise ValueError("attention maps must have shape [batch, image tokens]")
        size = grid
        if size is None:
            side = math.isqrt(value.shape[-1])
            size = (side, side)
        if size[0] * size[1] != value.shape[-1]:
            raise ValueError("a matching image token grid is required")
        value = value.float()
        confidence = map_confidence(value)[:, None]
        scale = torch.quantile(value, 0.99, dim=-1, keepdim=True)
        value = torch.where(scale > 0, value / scale.clamp_min(1e-12), 0).clamp(0, 1)
        image = value.reshape(value.shape[0], 1, *size)
        image = F.avg_pool2d(F.pad(image, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        value = image.flatten(1)
        if name in history:
            value = 0.5 * (value + history[name].to(value))
        current[name] = value
        if hard_mode:
            if quantile is None:
                owner = value >= 0.5
            else:
                threshold = torch.quantile(value, 1 - quantile, dim=-1, keepdim=True)
                owner = (value >= threshold) & (
                    value.amax(-1, keepdim=True) > value.amin(-1, keepdim=True)
                )
            mask = owner.to(value.dtype).reshape(value.shape[0], 1, *size)
            mask = F.avg_pool2d(F.pad(mask, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            placed.append((share + confidence * (mask.flatten(1) - share)).clamp(0, 1))
        else:
            centered = value - value.mean(-1, keepdim=True)
            placed.append((share + confidence * centered).clamp(0, 1))
    ip = torch.stack(placed).amax(0)
    return _Alphas({"ip": ip, "creative": torch.ones_like(ip)}, current)
