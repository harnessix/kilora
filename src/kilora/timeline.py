# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import contextlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .compute import AXES
from .depth_schedule import _BINOPS, _CMPOPS, FUNCTIONS, DepthScheduleError, _evaluate
from .errors import KiLoRAError

CLOCKS = ("progress", "step", "sigma")
INTERPOLATIONS = ("linear", "step")
VARIABLES = ("t", "step", "steps", "sigma")

QUANTITIES: dict[str, tuple[float, float]] = {
    "strength": (-4.0, 4.0),
    "active": (0.0, 1.0),
    "route_freeze": (0.0, 1.0),
    "gate_kappa": (0.0, 16.0),
    "region_isolation": (0.0, 1.0),
    "region_attention": (0.0, 20.0),
    "caption_window": (0.0, 1.0),
    "depth_map": (0.0, 1.0),
    "depth_base": (0.0, 16.0),
}
GLOBAL = ("depth_map", "depth_base")
SCALARS = ("strength", "active")
POINTS = ("route_freeze", "gate_kappa")


class TimelineError(KiLoRAError, ValueError):
    pass


@dataclass(frozen=True)
class Clock:
    t: float = 0.0
    step: int = 0
    steps: int = 1
    sigma: float = 1.0

    def env(self) -> dict[str, float]:
        return {
            "t": self.t,
            "step": float(self.step),
            "steps": float(self.steps),
            "sigma": self.sigma,
        }


@dataclass(frozen=True)
class Row:
    target: str
    quantity: str
    curve: str
    clock: str = "progress"
    interpolation: str = "linear"

    def value(self, clock: Clock) -> float:
        return _compiled(self)(clock)


class Timeline(tuple):
    def add(self, *rows: Row) -> Timeline:
        return Timeline((*self, *rows))

    def rows_for(self, quantity: str) -> list[Row]:
        return [r for r in self if r.quantity == quantity]


def _check_expression(tree: ast.AST, source: str) -> None:
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Call,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.And,
        ast.Or,
    )
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.operator, ast.cmpop)):
            if type(node) not in _BINOPS and type(node) not in _CMPOPS:
                raise TimelineError(f"{type(node).__name__} is not allowed in {source!r}")
            continue
        if not isinstance(node, allowed):
            raise TimelineError(f"{type(node).__name__} is not allowed in {source!r}")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise TimelineError(f"only numbers are allowed as constants in {source!r}")
        if isinstance(node, ast.Name) and id(node) not in called and node.id not in VARIABLES:
            raise TimelineError(
                f"unknown name {node.id!r} in {source!r}; variables are {', '.join(VARIABLES)}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise TimelineError(f"unknown function in {source!r}")
            if node.keywords or any(isinstance(x, ast.Starred) for x in node.args):
                raise TimelineError(f"function calls take plain arguments in {source!r}")


def parse_keyframes(text: str) -> list[tuple[float, float]] | None:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts or not all(p.count(":") == 1 for p in parts):
        return None
    try:
        keys = [(float(a), float(b)) for a, b in (p.split(":") for p in parts)]
    except ValueError:
        return None
    return sorted(keys)


def _interpolate(keys: list[tuple[float, float]], x: float, kind: str) -> float:
    if x <= keys[0][0]:
        return keys[0][1]
    for (x0, y0), (x1, y1) in zip(keys, keys[1:], strict=False):
        if x < x1:
            if kind == "step" or x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return keys[-1][1]


_CACHE: dict[Row, object] = {}


def _compiled(row: Row):
    if row not in _CACHE:
        keys = parse_keyframes(row.curve)
        if keys is not None:
            position = {"progress": "t", "step": "step", "sigma": "sigma"}[row.clock]
            kind = row.interpolation

            def fn(clock: Clock, keys=keys, position=position, kind=kind) -> float:
                return _interpolate(keys, clock.env()[position], kind)
        else:
            try:
                tree = ast.parse(row.curve.strip(), mode="eval")
            except SyntaxError as exc:
                raise TimelineError(f"cannot parse curve {row.curve!r}: {exc.msg}") from None
            _check_expression(tree, row.curve)

            def fn(clock: Clock, tree=tree) -> float:
                try:
                    return float(_evaluate(tree.body, clock.env()))
                except (DepthScheduleError, ZeroDivisionError, OverflowError, ValueError) as e:
                    raise TimelineError(f"curve {row.curve!r}: {e}") from None

        _CACHE[row] = fn
    return _CACHE[row]


def check_row(row: Row, targets: Sequence[str] = ()) -> None:
    if row.quantity not in QUANTITIES:
        raise TimelineError(
            f"unknown quantity {row.quantity!r}; use one of {', '.join(QUANTITIES)}"
        )
    if row.clock not in CLOCKS:
        raise TimelineError(f"unknown clock {row.clock!r}; use one of {', '.join(CLOCKS)}")
    if row.interpolation not in INTERPOLATIONS:
        raise TimelineError(f"unknown interpolation {row.interpolation!r}")
    if row.quantity in GLOBAL and row.target != "all":
        raise TimelineError(f"{row.quantity} acts on the whole model call; use the target all")
    known = {"all", *AXES, *targets}
    if targets and row.target not in known:
        raise TimelineError(
            f"unknown target {row.target!r}; use all, an axis ({', '.join(AXES)}) or one of "
            + ", ".join(sorted(targets))
        )
    low, high = QUANTITIES[row.quantity]
    for clock in sample_clocks(30):
        v = row.value(clock)
        if not math.isfinite(v) or not low <= v <= high:
            raise TimelineError(
                f"{row.quantity} of {row.target!r} is {v:g} at t={clock.t:g}; "
                f"the range is {low:g} to {high:g}"
            )


def sample_clocks(steps: int, sigma_max: float = 1.0) -> list[Clock]:
    n = max(int(steps), 2)
    return [Clock(t=i / (n - 1), step=i, steps=n, sigma=sigma_max * (1 - i / n)) for i in range(n)]


def clock_of(sigma: float, sigmas: Sequence[float] | None) -> Clock:
    if sigmas is None or len(sigmas) < 2:
        return Clock(sigma=float(sigma))
    values = [float(s) for s in sigmas][:-1]
    index = min(range(len(values)), key=lambda i: abs(values[i] - float(sigma)))
    steps = len(values)
    return Clock(t=index / max(steps - 1, 1), step=index, steps=steps, sigma=float(sigma))


def parse_rows(text: str) -> list[Row]:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if not 3 <= len(parts) <= 5:
            raise TimelineError(
                f"row {n}: write target | quantity | curve [| clock [| interpolation]]"
            )
        rows.append(Row(*parts))
    return rows


_SCALES: list = []


def scale_for(rows: Sequence[Row], clock: Clock, entry: str, axis: str) -> float:
    value = 1.0
    for r in rows:
        if r.quantity in SCALARS and r.target in ("all", axis, entry):
            v = r.value(clock)
            value *= (1.0 if v > 0.5 else 0.0) if r.quantity == "active" else v
    return value


@contextlib.contextmanager
def time_scales(rows: Sequence[Row], clock: Clock) -> Iterator[None]:
    _SCALES.append((tuple(r for r in rows if r.quantity in SCALARS), clock))
    try:
        yield
    finally:
        _SCALES.pop()


def current_scale(entry: str, axis: str) -> float | None:
    if not _SCALES or not _SCALES[-1][0]:
        return None
    rows, clock = _SCALES[-1]
    return scale_for(rows, clock, entry, axis)


def window_value(rows: Sequence[Row], quantity: str, clock: Clock) -> float | None:
    found = [r for r in rows if r.quantity == quantity]
    return found[-1].value(clock) if found else None


def point_value(rows: Sequence[Row], quantity: str) -> float | None:
    found = [r for r in rows if r.quantity == quantity]
    return found[-1].value(Clock()) if found else None


def preview(timeline: Sequence[Row], width: int = 640, height: int = 360, steps: int = 60):
    import torch
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, right, top, bottom = 40, width - 10, 10, height - 24
    colors = [
        (230, 120, 80),
        (90, 170, 240),
        (130, 210, 120),
        (220, 200, 90),
        (190, 120, 220),
        (240, 240, 240),
        (90, 220, 210),
        (240, 110, 170),
    ]
    clocks = sample_clocks(steps)
    series = [[r.value(c) for c in clocks] for r in timeline]
    flat = [v for s in series for v in s] or [0.0, 1.0]
    low, high = min(0.0, *flat), max(1.0, *flat)
    span = (high - low) or 1.0
    draw.rectangle([left, top, right, bottom], outline=(90, 90, 100))
    for frac in (0.0, 0.5, 1.0):
        y = bottom - (bottom - top) * frac
        draw.text((2, y - 6), f"{low + span * frac:.2g}", fill=(160, 160, 170), font=font)
    for frac in (0.0, 0.5, 1.0):
        x = left + (right - left) * frac
        draw.text((x - 8, bottom + 6), f"{frac:g}", fill=(160, 160, 170), font=font)
    for k, (row, values) in enumerate(zip(timeline, series, strict=True)):
        color = colors[k % len(colors)]
        pts = [
            (left + (right - left) * c.t, bottom - (bottom - top) * (v - low) / span)
            for c, v in zip(clocks, values, strict=True)
        ]
        draw.line(pts, fill=color, width=2)
        draw.text(
            (left + 6, top + 4 + 12 * k), f"{row.target} {row.quantity}", fill=color, font=font
        )
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return (data.view(height, width, 3).float() / 255)[None]
