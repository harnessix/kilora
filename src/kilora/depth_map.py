# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .recording import ContextStack


@dataclass(eq=False)
class DepthMap:
    values: torch.Tensor
    min_depth: float
    max_depth: float
    start: float
    base: float
    auto: AutoDepth | None = None
    on_measured: Callable[[torch.Tensor], None] | None = None

    def depths(self) -> torch.Tensor:
        return self.min_depth + self.values * (self.max_depth - self.min_depth)

    @property
    def ready(self) -> bool:
        return self.auto is None or self.__dict__.get("_measured_at") is not None

    def mean_target(self) -> float:
        a = self.auto
        if a is not None and a.mean_depth >= 0:
            return float(a.mean_depth)
        depth = self.__dict__.get("_apply_depth")
        if depth is not None:
            return float(depth)
        return (self.min_depth + self.max_depth) / 2

    def use_apply_depth(self, depth: float | None) -> None:
        if self.auto is None or depth is None:
            return
        self.__dict__["_apply_depth"] = float(depth)
        if not self.auto.fixed_base:
            self.base = float(math.floor(self.mean_target() + 0.5))

    def reset(self) -> None:
        if self.auto is None:
            return
        self.__dict__["_measured_at"] = None
        self._set_values(torch.full((1, 1, 1), 0.5))

    def _set_values(self, values: torch.Tensor) -> None:
        self.values = values
        self.__dict__.pop("_ones", None)
        self.__dict__.pop("_token_depth_cache", None)

    def measured(self) -> torch.Tensor | None:
        if self.auto is None or not self.ready:
            return None
        return self.values

    def begin_call(self, progress: float) -> Measurement | None:
        a = self.auto
        if a is None:
            return None
        at = self.__dict__.get("_measured_at")
        if at is not None and progress < at:
            self.reset()
            at = None
        if at is not None or progress < self.start:
            return None
        return Measurement(want_change=a.change_weight != 0, want_routes=a.disagreement_weight != 0)

    def finish(
        self,
        measurement: Measurement,
        x: torch.Tensor,
        x0: torch.Tensor | None,
        progress: float,
        chunks: int = 1,
    ) -> None:
        a = self.auto
        if x0 is None or tuple(x0.shape) != tuple(x.shape):
            _log_once(
                "auto-x0",
                "KiLoRA Depth Map auto: no clean-image estimate at the measured step, "
                "the detail is measured on the model input",
            )
            x0 = x
        if x.dim() < 4:
            _log_once("auto-1d", "KiLoRA Depth Map auto: the latent has no image grid")
            self._set_values(torch.full((1, 1, 1), 0.5, device=x.device))
            self.__dict__["_measured_at"] = float(progress)
            return
        batch = int(x.shape[0])
        grids = [g for g, _ in measurement.change] + [g for g, _ in measurement.routes]
        if grids:
            grid = max(grids, key=lambda g: g[0] * g[1])
        else:
            grid = (math.ceil(x.shape[-2] / 2), math.ceil(x.shape[-1] / 2))
        score = None
        signals = [
            (a.detail_weight, lambda: _per_image(detail_energy(x0, grid), batch, chunks)),
            (a.change_weight, lambda: _mean_on(measurement.change, grid, batch, chunks)),
            (a.disagreement_weight, lambda: _mean_on(measurement.routes, grid, batch, chunks)),
        ]
        for weight, read in signals:
            if weight == 0:
                continue
            signal = read()
            if signal is None:
                continue
            part = weight * normalized(signal)
            score = part if score is None else score + part
        if score is None:
            score = torch.zeros(1, *grid, device=x.device)
        score = smooth_grid(score, a.smooth)
        pixels = 8 * x.shape[-2] / grid[0]
        score = feather(score, round(a.feather_px / pixels))
        depths = budget_depths(score, self.min_depth, self.max_depth, self.mean_target())
        span = self.max_depth - self.min_depth
        values = (depths - self.min_depth) / span if abs(span) > 1e-12 else depths * 0
        self._set_values(values.clamp(0, 1))
        self.__dict__["_measured_at"] = float(progress)
        if self.on_measured is not None:
            try:
                self.on_measured(self.values)
            except Exception as exc:
                _log_once("auto-preview", f"KiLoRA Depth Map auto: no preview on the node ({exc})")

    def call(self, invoke, args: dict, progress: float, chunks: int = 1):
        measurement = self.begin_call(progress)
        if measurement is None:
            return invoke(args)
        from .compute import recording_shares

        routes = (
            recording_shares(measurement.note_routes)
            if measurement.want_routes
            else contextlib.nullcontext()
        )
        with measuring(measurement), routes:
            out = invoke(args)
        x = args["input"]
        x0 = out if isinstance(out, torch.Tensor) else None
        self.finish(measurement, x, x0, progress, chunks)
        return out

    def applies(self, progress: float) -> tuple[bool, float]:
        from .call_values import current_token_depth

        on, base = current_token_depth()
        if on is None:
            on = not (self.start > 0 and progress < self.start)
        if not self.ready:
            on = False
        return on, self.base if base is None else base

    def on_tokens(self, region, before_start: bool, base: float | None = None):
        base = self.base if base is None else float(base)
        mapped = region(self.values)
        if mapped is None:
            return None
        if before_start or not self.ready:
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


def smooth_grid(values: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return values
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=values.device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).to(values.dtype)
    m = F.pad(values[:, None], (radius, radius, 0, 0), mode="replicate")
    m = F.conv2d(m, kernel.view(1, 1, 1, -1))
    m = F.pad(m, (0, 0, radius, radius), mode="replicate")
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


def make_auto_depth_map(
    min_depth: float = 1.0,
    max_depth: float = 3.0,
    start: float = 0.3,
    base: float | None = None,
    mean_depth: float = -1.0,
    detail_weight: float = 1.0,
    change_weight: float = 1.0,
    disagreement_weight: float = 0.0,
    feather_px: int = 0,
    smooth: float = 2.0,
) -> DepthMap:

    if min(min_depth, max_depth) < 0:
        raise ValueError(f"depths must be at least 0, got {min_depth} and {max_depth}")
    if not 0.0 <= float(start) <= 1.0:
        raise ValueError(f"start must be a sampling progress in [0, 1], got {start}")
    weights = (detail_weight, change_weight, disagreement_weight)
    if min(weights) < 0:
        raise ValueError(f"the auto weights must be at least 0, got {weights}")
    if smooth < 0:
        raise ValueError(f"smooth must be at least 0, got {smooth}")
    fixed = base is not None and base >= 0
    auto = AutoDepth(float(mean_depth), *map(float, weights), int(feather_px), fixed, float(smooth))
    out = DepthMap(
        torch.full((1, 1, 1), 0.5), float(min_depth), float(max_depth), float(start), 0.0, auto
    )
    out.base = float(base) if fixed else float(math.floor(out.mean_target() + 0.5))
    out.reset()
    return out


_MEASURING = ContextStack("kilora_depth_measurements")
_LOGGED: set = set()


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    import logging

    logging.getLogger("kilora").info(message)


@dataclass(eq=False)
class AutoDepth:
    mean_depth: float = -1.0
    detail_weight: float = 1.0
    change_weight: float = 1.0
    disagreement_weight: float = 0.0
    feather_px: int = 0
    fixed_base: bool = False
    smooth: float = 2.0


class Measurement:
    def __init__(self, want_change: bool = True, want_routes: bool = False) -> None:
        self.want_change = want_change
        self.want_routes = want_routes
        self.change: list = []
        self.routes: list = []

    def note_change(self, h: torch.Tensor, y: torch.Tensor) -> None:
        if not self.want_change or h.dim() < 3 or y.shape != h.shape:
            return
        grid = _image_grid(h)
        if grid is None:
            return
        norm = (y.detach().float() - h.detach().float()).norm(dim=-1)
        self.change.append((grid, norm.reshape(norm.shape[0], *grid)))

    def note_routes(self, target: str, gates: list) -> None:
        if not self.want_routes:
            return
        from .region import current_latent, grid_for

        latent = current_latent()
        for g in gates:
            if g.dim() < 2 or g.shape[-1] < 2 or latent is None:
                continue
            grid = grid_for(int(g.shape[-2]), latent)
            if grid is None:
                continue
            p = g.detach().float().clamp_min(0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
            self.routes.append((grid, entropy.reshape(-1, *grid)))


def _image_grid(h: torch.Tensor) -> tuple[int, int] | None:
    from .region import current_latent, grid_for

    dims = tuple(h.shape[1:-1])
    if len(dims) == 2:
        return dims
    latent = current_latent()
    if len(dims) != 1 or latent is None:
        return None
    return grid_for(dims[0], latent)


@contextlib.contextmanager
def measuring(measurement: Measurement) -> Iterator[Measurement]:
    with _MEASURING.pushed(measurement):
        yield measurement


def current_measurement() -> Measurement | None:
    return _MEASURING[-1] if _MEASURING else None


def note_change(h: torch.Tensor, y: torch.Tensor) -> None:
    if _MEASURING:
        _MEASURING[-1].note_change(h, y)


def detail_energy(x0: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    x = x0.detach().float()
    x = x.reshape(x.shape[0], -1, *x.shape[-2:])
    channels = x.shape[1]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device)
    kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=channels)
    energy = lap.abs().sum(1, keepdim=True)
    return F.adaptive_avg_pool2d(energy, grid)[:, 0]


def normalized(signal: torch.Tensor) -> torch.Tensor:
    flat = signal.float().reshape(signal.shape[0], -1)
    q = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device), dim=1)
    lo, hi = q[0][:, None], q[1][:, None]
    span = hi - lo
    out = torch.where(span > 1e-12, (flat - lo) / span.clamp_min(1e-12), torch.zeros_like(flat))
    return out.clamp(0, 1).reshape(signal.shape)


def budget_depths(
    score: torch.Tensor, min_depth: float, max_depth: float, mean_depth: float
) -> torch.Tensor:
    lo_d, hi_d = float(min(min_depth, max_depth)), float(max(min_depth, max_depth))
    if hi_d - lo_d < 1e-12:
        return torch.full_like(score.float(), lo_d)
    p = (min(max(float(mean_depth), lo_d), hi_d) - lo_d) / (hi_d - lo_d)
    flat = score.detach().double().reshape(score.shape[0], -1)
    n = flat.shape[1]
    ordered = flat.sort(-1).values.contiguous()
    left = torch.searchsorted(ordered, flat.contiguous(), right=False)
    right = torch.searchsorted(ordered, flat.contiguous(), right=True)
    q = (left + right - 1).double() / 2 / max(n - 1, 1)
    if n == 1:
        q = torch.full_like(q, p)
    lo = torch.full((flat.shape[0], 1), -1.0, dtype=torch.float64, device=flat.device)
    hi = torch.full_like(lo, 1.0)
    for _ in range(60):
        c = (lo + hi) / 2
        below = (q + c).clamp(0, 1).mean(-1, keepdim=True) < p
        lo, hi = torch.where(below, c, lo), torch.where(below, hi, c)
    f = (q + (lo + hi) / 2).clamp(0, 1)
    return (lo_d + f * (hi_d - lo_d)).float().reshape(score.shape)


def _per_image(t: torch.Tensor, batch: int, chunks: int) -> torch.Tensor | None:
    if t.shape[0] == batch:
        if chunks > 1 and batch % chunks == 0:
            return t.reshape(chunks, batch // chunks, *t.shape[1:]).mean(0)
        return t
    if t.shape[0] % batch == 0:
        return _per_image(t.reshape(batch, -1, *t.shape[1:]).mean(1), batch, chunks)
    return None


def _mean_on(items: list, grid, batch: int, chunks: int) -> torch.Tensor | None:
    picked = []
    for g, t in items:
        v = _per_image(t, batch, chunks)
        if v is None:
            continue
        if tuple(g) != tuple(grid):
            v = F.adaptive_avg_pool2d(v[:, None], grid)[:, 0]
        picked.append(v)
    if not picked:
        return None
    return torch.stack(picked).mean(0)


__all__ = [
    "AutoDepth",
    "DepthMap",
    "Measurement",
    "budget_depths",
    "current_measurement",
    "detail_energy",
    "feather",
    "luminance",
    "make_auto_depth_map",
    "make_depth_map",
    "measuring",
    "normalized",
    "note_change",
    "smooth_grid",
]
