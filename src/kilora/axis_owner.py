# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import os

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
_current = contextvars.ContextVar("kilora_axis_owner", default=None)


def mode():
    return os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower()


def nonnegative_setting(name, default):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


@contextlib.contextmanager
def recording(owner, kinds=()):
    token = _current.set((owner, tuple(kinds)))
    try:
        yield
    finally:
        _current.reset(token)


def record(target, y0, sets):
    current = _current.get()
    if current is not None:
        owner, kinds = current
        if kinds:
            if y0.shape[0] % len(kinds):
                return
            chunks = y0.chunk(len(kinds), dim=0)
            positive = [chunk for chunk, kind in zip(chunks, kinds, strict=True) if kind == 0]
            if not positive:
                return
            y0 = torch.cat(positive, dim=0)
        owner.record(target, y0, sets)


class AxisOwner:
    def __init__(self, shape, batch=1, device=None, axes=None):
        self.shape = tuple(shape)
        self.batch = batch
        self.device = device
        self.axes = None if axes is None else frozenset(axes)
        self.scores = {}
        self.ema = {}
        self.masks = None
        self.bias_sum = None
        self.bias_count = 0
        self._warned_frames = False

    def record(self, target, y0, sets):
        from .compute import _set_keys, norm_energy
        from .region import current_latent_shape, current_layout, current_text_tokens, grid_for

        if y0.ndim < 3 or len(self.shape) != 2:
            return
        latent = current_latent_shape() or self.shape
        layout = current_layout(target)
        tokens = math.prod(y0.shape[1:-1])
        start = 0
        if layout is not None:
            if layout.kind not in ("image2d", "video3d") or layout.role == "reference":
                return
            if layout.kind == "video3d" and any(size != 1 for size in latent[:-2]):
                if not self._warned_frames:
                    log.info("axis owner: disabled for multiple frames")
                    self._warned_frames = True
                return
            if layout.patch is not None:
                full_grid = layout.grid(latent)
                n = math.prod(full_grid)
                grid = full_grid[-2:]
                if n > tokens:
                    return
                if layout.text == "before":
                    padded = math.ceil(n / layout.pad) * layout.pad
                    start = tokens - padded
                    text = current_text_tokens()
                    if layout.pad == 1 and text is not None and 0 < text <= tokens - n:
                        start = text
            else:
                grid = grid_for(tokens, tuple(latent[-2:]))
        else:
            grid = grid_for(tokens, tuple(latent[-2:]))
        if grid is None:
            return
        n = math.prod(grid)
        if start < 0 or start + n > tokens:
            return
        with torch.no_grad():
            y = y0.detach().reshape(y0.shape[0], tokens, y0.shape[-1])[:, start : start + n]
            values, identities = [], []
            for set_index, s in enumerate(sets):
                if self.axes is not None and s.axis not in self.axes:
                    continue
                if s.keys_in is None or s.keys_in.shape[0] == 0:
                    continue
                q = F.linear(y.float(), s.router.float())
                kin, _ = _set_keys(s)
                values.append(norm_energy(q, kin, s.rows_in))
                name = s.key or f"#{set_index}"
                identities.extend((s.axis, name, i) for i in range(kin.shape[0]))
            if not values:
                return
            self.record_scores(torch.cat(values, -1), identities, grid)

    def record_scores(self, scores, identities, grid):
        with torch.no_grad():
            scores = torch.nan_to_num(scores.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            has_scores = (scores > 0).any(-1, keepdim=True)
            ranks = (scores.unsqueeze(-1) > scores.unsqueeze(-2)).float().sum(-1) + 1
            ranks = ranks / ranks.sum(-1, keepdim=True) * has_scores
            if ranks.shape[0] != self.batch:
                if ranks.shape[0] % self.batch:
                    return
                ranks = ranks.reshape(-1, self.batch, *ranks.shape[1:]).mean(0)
            for i, identity in enumerate(identities):
                value = ranks[..., i].reshape(ranks.shape[0], 1, *grid)
                image = F.interpolate(value, size=self.shape, mode="bilinear", align_corners=False)[
                    :, 0
                ]
                old = self.scores.get(identity)
                if old is None or old.shape == image.shape:
                    self.scores[identity] = image if old is None else old + image

    def add_bias(self, bias):
        with torch.no_grad():
            total = bias.detach().float().sum()
            self.bias_sum = total if self.bias_sum is None else self.bias_sum + total
            self.bias_count += bias.numel()

    @torch.no_grad()
    def finish_step(self):
        if not self.scores:
            self.ema = {}
            self.masks = owner_masks({}, shape=self.shape, batch=self.batch, device=self.device)
            if mode() == "soft":
                mean_bias = (self.bias_sum / self.bias_count).item() if self.bias_count else 0.0
                log.info(
                    "axis owner: shares=%s overlap=0.000 mean margin=0.000 mean bias=%.6f",
                    {"creative": 1.0},
                    mean_bias,
                )
            self.bias_sum, self.bias_count = None, 0
            return self.masks
        axes = {}
        for (axis, _name, _index), score in self.scores.items():
            axes[axis] = score if axis not in axes else torch.maximum(axes[axis], score)
        for axis in self.ema.keys() - axes.keys():
            axes[axis] = torch.zeros_like(self.ema[axis])
        for axis, score in axes.items():
            previous = self.ema.get(axis)
            self.ema[axis] = (
                score
                if previous is None or previous.shape != score.shape
                else (previous + score) * 0.5
            )
        self.scores = {}
        scores = self.ema
        if mode() == "soft" and self.masks:
            beta = nonnegative_setting("KILORA_AXIS_INHIBIT", 0.5)
            if beta:
                blurred = {
                    name: F.avg_pool2d(
                        F.pad(mask[:, None], (1, 1, 1, 1), mode="replicate"), 3, stride=1
                    )[:, 0]
                    for name, mask in self.masks.items()
                }
                scores = {
                    name: (
                        score
                        - beta
                        * sum(
                            (other for key, other in blurred.items() if key != name),
                            torch.zeros_like(score),
                        )
                    ).clamp_min(0)
                    for name, score in self.ema.items()
                }
        self.masks = owner_masks(scores)
        if mode() == "soft":
            weights = torch.stack(list(self.masks.values()), dim=-1)
            top = weights.topk(min(2, weights.shape[-1]), dim=-1).values
            overlap = (top[..., 1] > 0.25).float().mean().item() if top.shape[-1] > 1 else 0.0
            sample = next(iter(scores.values()))
            values = torch.stack(list(scores.values()), dim=-1).clamp_min(0)
            if "creative" not in scores:
                values = torch.cat((values, torch.zeros_like(sample)[..., None]), dim=-1)
            norm = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
            ranked = norm.topk(min(2, norm.shape[-1]), dim=-1).values
            margin = ranked[..., 0] - ranked[..., 1] if ranked.shape[-1] > 1 else ranked[..., 0]
            mean_bias = (self.bias_sum / self.bias_count).item() if self.bias_count else 0.0
            log.info(
                "axis owner: shares=%s overlap=%.3f mean margin=%.3f mean bias=%.6f",
                {name: mask.mean().item() for name, mask in self.masks.items()},
                overlap,
                margin.mean().item(),
                mean_bias,
            )
        self.bias_sum, self.bias_count = None, 0
        return self.masks


def owner_masks(scores, margin=None, far="creative", *, shape=None, batch=1, device=None):

    if not scores:
        if shape is None:
            return {}
        log.info("axis owner: shares=%s mean margin=0.000 blended=0.000", {far: 1.0})
        return {far: torch.ones((batch, *shape), device=device)}
    names = list(scores)
    if far not in names:
        names.append(far)
    sample = next(iter(scores.values()))
    values = torch.stack([scores.get(name, torch.zeros_like(sample)) for name in names], -1)
    values = values.clamp_min(0)
    total = values.sum(-1, keepdim=True)
    norm = values / total.clamp_min(1e-12)
    top, indices = norm.topk(min(2, len(names)), dim=-1)
    if margin is None:
        try:
            margin = float(os.environ.get("KILORA_AXIS_ZBUFFER_MARGIN", "0.1"))
        except ValueError:
            margin = 0.1
    difference = top[..., 0] - top[..., 1] if len(names) > 1 else top[..., 0]
    blended = (difference < margin) & (total[..., 0] > 0)
    weights = torch.zeros_like(norm)
    first = torch.ones_like(top[..., 0])
    if len(names) > 1:
        pair = top[..., :2] / top[..., :2].sum(-1, keepdim=True).clamp_min(1e-12)
        first = torch.where(blended, pair[..., 0], first)
        weights.scatter_(-1, indices[..., 1:2], torch.where(blended, pair[..., 1], 0)[..., None])
    weights.scatter_(-1, indices[..., :1], first[..., None])
    creative_index = names.index(far)
    creative_close = (norm[..., creative_index] > 0) & (
        top[..., 0] - norm[..., creative_index] < margin
    )
    far_in_pair = (indices == creative_index).any(-1)
    missing = (total[..., 0] <= 0) | (creative_close & ~far_in_pair)
    fallback = F.one_hot(torch.full_like(indices[..., 0], names.index(far)), len(names)).to(
        weights.dtype
    )
    weights = torch.where(missing[..., None], fallback, weights)
    masks = F.avg_pool2d(
        F.pad(weights.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate"), 3, stride=1
    )
    shares = masks.mean((0, 2, 3)).tolist()
    log.info(
        "axis owner: shares=%s mean margin=%.3f blended=%.3f",
        dict(zip(names, shares, strict=True)),
        difference.mean().item(),
        blended.float().mean().item(),
    )
    return {name: masks[:, i] for i, name in enumerate(names)}
