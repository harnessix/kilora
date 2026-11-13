# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..compute import recording_directions, recording_shares
from ..i18n import text
from .patching import KiLoRAError

ROUTES = "KILORA_ROUTES"
LEGEND = 24
OPACITY = 0.85
BACKDROP = 0.4
COLORS = (
    (0.90, 0.16, 0.16),
    (0.12, 0.47, 0.95),
    (0.15, 0.75, 0.30),
    (0.98, 0.72, 0.05),
    (0.64, 0.24, 0.88),
    (0.05, 0.80, 0.82),
    (0.97, 0.45, 0.70),
    (0.55, 0.40, 0.20),
)
COLOR_NAMES = ("red", "blue", "green", "yellow", "purple", "cyan", "pink", "brown")


class RouteShares:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self._step: float | None = None
        self.directions: dict[tuple, tuple[torch.Tensor, int, list, int]] = {}

    def clear(self) -> None:
        self.sums.clear()
        self.counts.clear()
        self.directions.clear()

    def new_call(self, timestep: Any) -> None:
        if isinstance(timestep, torch.Tensor):
            step = float(timestep.detach().flatten()[0])
        else:
            step = float(timestep)
        if self._step is not None and step > self._step:
            self.clear()
        self._step = step

    def observe(self, target: str, gates: Sequence[torch.Tensor]) -> None:
        parts = []
        for g in gates:
            g = g.float()
            if g.dim() == 1:
                g = g.reshape(1, 1, -1)
            parts.append(g.reshape(-1, g.shape[-2], g.shape[-1]))
        tokens = max(p.shape[1] for p in parts)
        rows = max(p.shape[0] for p in parts)
        parts = [p.expand(rows, tokens, p.shape[-1]) for p in parts]
        w = torch.cat(parts, -1).mean(0) / len(parts)
        s = self.sums.get(target)
        if s is None or s.shape[0] != w.shape[0]:
            self.sums[target] = w.clone()
            self.counts[target] = 1
            return
        if s.shape[1] < w.shape[1]:
            s = F.pad(s, (0, w.shape[1] - s.shape[1]))
        elif w.shape[1] < s.shape[1]:
            w = F.pad(w, (0, s.shape[1] - w.shape[1]))
        self.sums[target] = s + w.to(s.device)
        self.counts[target] += 1

    def observe_directions(self, target, step, index, layer, gate) -> None:
        weights = gate.float().reshape(-1, gate.shape[-2], gate.shape[-1]).mean(0).cpu()
        key = (target, index, step)
        previous = self.directions.get(key)
        if previous is not None and previous[0].shape == weights.shape:
            weights = weights + previous[0]
            count = previous[1] + 1
        else:
            count = 1
        self.directions[key] = (weights, count, layer.direction_sources, layer.per_expert)

    def shares(self) -> dict[str, torch.Tensor]:
        return {t: (s / self.counts[t]).cpu() for t, s in self.sums.items()}

    @property
    def loras(self) -> int:
        return max((s.shape[1] for s in self.sums.values()), default=0)

    @contextlib.contextmanager
    def recording(self):
        with recording_shares(self.observe), recording_directions(self.observe_directions):
            yield self


def record(patcher: Any) -> tuple[Any, RouteShares]:
    handle = RouteShares()
    out = patcher.clone()
    options = getattr(out, "model_options", None)
    if options is None:
        raise KiLoRAError(text("err.no_routes"))
    previous = options.get("model_function_wrapper")

    def wrapper(apply_model, args):
        handle.new_call(args["timestep"])
        with handle.recording():
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper
    return out, handle


def token_grid(tokens: int, height: int, width: int) -> tuple[int, int]:
    want = math.log(height / width)
    best = (1, tokens)
    for rows in range(1, tokens + 1):
        if tokens % rows:
            continue
        cols = tokens // rows
        if abs(math.log(rows / cols) - want) < abs(math.log(best[0] / best[1]) - want):
            best = (rows, cols)
    return best


def share_maps(shares: dict[str, torch.Tensor], height: int, width: int) -> torch.Tensor:
    loras = max(s.shape[1] for s in shares.values())
    total = torch.zeros(loras, height, width)
    for s in shares.values():
        s = F.pad(s, (0, loras - s.shape[1]))
        rows, cols = token_grid(s.shape[0], height, width)
        grid = s.t().reshape(1, loras, rows, cols)
        total += F.interpolate(grid, size=(height, width), mode="nearest")[0]
    return total / len(shares)


def labels_for(given: str, loras: int) -> list[str]:
    names = [n.strip() for n in (given or "").split(",") if n.strip()]
    return [names[i] if i < len(names) else f"LoRA {i + 1}" for i in range(loras)]


def _legend(width: int, color: torch.Tensor, label: str) -> torch.Tensor:
    strip = torch.full((LEGEND, width, 3), 0.08)
    pad = 4
    swatch = min(LEGEND - 2 * pad, max(width - 2 * pad, 0))
    strip[pad : pad + swatch, pad : pad + swatch] = color
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return strip
    img = Image.fromarray((strip.numpy() * 255).round().astype("uint8"))
    ImageDraw.Draw(img).text((2 * pad + swatch, pad), label, fill=(235, 235, 235))
    return torch.from_numpy(np.asarray(img).copy()).float() / 255


def render(handle: RouteShares, image: torch.Tensor, labels: str = "") -> tuple[torch.Tensor, str]:
    shares = handle.shares()
    if not shares:
        raise KiLoRAError(text("err.no_routes"))
    batch, height, width, _ = image.shape
    maps = share_maps(shares, height, width)
    names = labels_for(labels, maps.shape[0])
    base = image[..., :3].float().cpu().mean(-1, keepdim=True) * BACKDROP
    panels = []
    lines = [f"Routing map: {len(names)} LoRAs, {len(shares)} layers."]
    for e, name in enumerate(names):
        color = torch.tensor(COLORS[e % len(COLORS)])
        a = (maps[e] * OPACITY).clamp(0, 1)[None, :, :, None]
        panel = base * (1 - a) + color * a
        legend = _legend(width, color, name).expand(batch, LEGEND, width, 3)
        panels.append(torch.cat([panel, legend], 1))
        lines.append(
            f"{e + 1}. {name} ({COLOR_NAMES[e % len(COLOR_NAMES)]}): "
            f"mean share {float(maps[e].mean()):.2f}"
        )
    if handle.directions:
        lines.append("Direction routes (token grid regions):")
        for (target, index, step), (total, count, sources, chunk) in sorted(
            handle.directions.items()
        ):
            weights = total / count
            rows, cols = token_grid(weights.shape[0], height, width)
            for token, row in enumerate(weights):
                top = torch.argsort(row, descending=True, stable=True)[:3]
                entries = []
                for node in top.tolist():
                    if row[node] <= 0:
                        continue
                    first = node * chunk
                    last = min(first + chunk, len(sources))
                    for direction in range(first, last):
                        files = (
                            ", ".join(
                                os.path.basename(str(source.get("file", "")))
                                for source in sources[direction][:3]
                            )
                            or "unknown source"
                        )
                        entries.append(f"direction {direction + 1} ({files}; {row[node]:.3f})")
                lines.append(
                    f"{target}, set {index + 1}, region {token // cols + 1},"
                    f"{token % cols + 1}/{rows}x{cols}, step {step}: " + "; ".join(entries)
                )
    return torch.cat(panels, 2).clamp(0, 1).contiguous(), "\n".join(lines)


class KiLoRARouteRecord:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", ROUTES)
    RETURN_NAMES = ("model", "routes")
    FUNCTION = "record"

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def record(self, model):
        return record(model)


class KiLoRARouteMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"routes": (ROUTES,), "image": ("IMAGE",)},
            "optional": {"labels": ("STRING", {"default": "", "multiline": False})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("map", "report")
    FUNCTION = "draw"

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def draw(self, routes, image, labels=""):
        return render(routes, image, labels)
