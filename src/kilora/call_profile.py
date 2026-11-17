# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import threading
import time

import torch

_LOCAL = threading.local()


def _stack() -> list[CallProfile]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = _LOCAL.stack = []
    return stack


def active() -> bool:
    return bool(getattr(_LOCAL, "stack", None))


STEP_PARTS = ("route", "update", "skips", "chain", "stats", "gate", "blend")
PARTS = ("base", "layer", "prep", *STEP_PARTS, "output")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CallProfile:
    def __init__(self) -> None:
        self.seconds: dict[tuple[str, int], float] = {}
        self.walks: dict[int, int] = {}
        self.calls = 0
        self.total = 0.0
        self._last: float | None = None
        self._layers = 0
        self.in_chain = 0

    def begin(self) -> None:
        _sync()
        self._last = time.perf_counter()

    def tick(self, part: str, step: int = 0) -> None:
        if self._last is None:
            return
        _sync()
        now = time.perf_counter()
        key = (part, int(step))
        self.seconds[key] = self.seconds.get(key, 0.0) + now - self._last
        self._last = now

    def end(self) -> None:
        self.tick("base")
        self._last = None
        self.calls += 1

    def part(self, name: str) -> float:
        return sum(v for (p, _k), v in self.seconds.items() if p == name)

    def steps(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for (p, k), v in self.seconds.items():
            if k and p in STEP_PARTS:
                out[k] = out.get(k, 0.0) + v
        return dict(sorted(out.items()))

    def sum(self) -> float:
        return sum(self.seconds.values())

    def line(self) -> str:
        if not self.calls:
            return ""
        n = self.calls
        routed = self.sum() - self.part("base")
        parts = ", ".join(
            f"{name} {self.part(name) / n:.3f}" for name in PARTS[1:] if self.part(name)
        )
        steps = ", ".join(f"{k}: {v / n:.3f}" for k, v in self.steps().items())
        text = (
            f"call split (mean of {n} calls, s): total {self.total / n:.3f},"
            f" base {self.part('base') / n:.3f}, routed {routed / n:.3f}"
        )
        if parts:
            text += f" ({parts})"
        if steps:
            text += f", depth steps {steps}"
        if self.walks:
            text += ", fused chains " + ", ".join(
                f"{count} x single walk"
                if not width
                else f"{count} x {width} stacked walk{'s' if width > 1 else ''}"
                for width, count in sorted(self.walks.items())
            )
        return text


def tick(part: str, step: int = 0) -> None:
    stack = _stack()
    if stack:
        stack[-1].tick(part, step)


def outside_chain() -> bool:
    stack = _stack()
    return bool(stack) and not stack[-1].in_chain


def note_walks(width: int) -> None:
    stack = _stack()
    if stack:
        walks = stack[-1].walks
        key = max(int(width), 0)
        walks[key] = walks.get(key, 0) + 1


@contextlib.contextmanager
def profiling(profile: CallProfile):
    stack = _stack()
    stack.append(profile)
    profile.begin()
    try:
        yield profile
    finally:
        profile.end()
        stack.remove(profile)


def run_layer(body, *args, **kwargs):
    profile = _stack()[-1]
    if profile._layers:
        return body(*args, **kwargs)
    profile.tick("base")
    profile._layers += 1
    try:
        return body(*args, **kwargs)
    finally:
        profile._layers -= 1
        profile.tick("layer")


def run_chain(body, *args, **kwargs):
    profile = _stack()[-1]
    profile.tick("layer")
    profile.in_chain += 1
    try:
        return body(*args, **kwargs)
    finally:
        profile.in_chain -= 1
        profile.tick("output")
