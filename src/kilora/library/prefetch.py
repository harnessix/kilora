# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class BlockPrefetch:
    def __init__(self, count: int, window: int, source: Callable, backend: Any, *, enabled=True):
        if count < 1 or window not in (1, 2):
            raise ValueError("block count must be positive and window must be 1 or 2")
        self.count, self.window, self.source, self.backend = count, window, source, backend
        self.enabled = enabled
        self.pending: dict[int, tuple[Any, Any]] = {}
        self.active: dict[int, Any] = {}

    def _copy(self, block: int) -> None:
        if block not in self.pending and block not in self.active:
            self.pending[block] = self.backend.copy(block, self.source(block))

    def begin(self, block: int) -> Any:
        self._copy(block)
        value, event = self.pending.pop(block)
        self.backend.wait(event, value)
        self.active[block] = value
        for offset in range(1, min(self.window, self.count - 1) + 1 if self.enabled else 1):
            self._copy((block + offset) % self.count)
        return value

    def end(self, block: int) -> None:
        self.active.pop(block, None)
        if self.enabled and block == self.count - 1:
            self._copy(0)

    def reset(self) -> None:
        self.pending.clear()
        self.active.clear()


class CudaCopies:
    def __init__(self, device: torch.device, report: Any):
        self.device, self.report = device, report
        self.stream = torch.cuda.Stream(device=device)
        self._timings: list[tuple[Any, Any]] = []
        report._prefetch_collect = self.collect

    def copy(self, _block: int, source: dict) -> tuple[dict, Any]:
        with torch.cuda.stream(self.stream):
            value = {
                key: tensor.to(self.device, non_blocking=True) for key, tensor in source.items()
            }
            event = torch.cuda.Event()
            event.record(self.stream)
        return value, event

    def wait(self, event: Any, value: dict) -> None:
        stream = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        stream.wait_event(event)
        end.record(stream)
        self._timings.append((start, end))
        for tensor in value.values():
            tensor.record_stream(stream)
        self.collect()

    def collect(self) -> None:
        remaining = []
        for start, end in self._timings:
            if end.query():
                self.report.prefetch_wait_seconds += start.elapsed_time(end) / 1000
            else:
                remaining.append((start, end))
        self._timings = remaining


class SynchronousCopies:
    def __init__(self, device: torch.device):
        self.device = device

    def copy(self, _block: int, source: dict) -> tuple[dict, None]:
        return {key: tensor.to(self.device) for key, tensor in source.items()}, None

    def wait(self, _event: Any, _value: dict) -> None:
        pass


class PlanCopies:
    def __init__(self, device, dtype, report, asynchronous):
        self.device, self.dtype, self.report = device, dtype, report
        self.asynchronous = asynchronous
        self.stream = torch.cuda.Stream(device=device) if asynchronous else None
        self._timings = []
        if asynchronous:
            report._prefetch_collect = self.collect

    def copy(self, _block, source):
        from ..bake import ModulePlan

        def move():
            with torch.inference_mode(False):
                return {
                    id(plan): ModulePlan(
                        parts=[
                            (
                                offset,
                                rows,
                                target.to(self.device, self.dtype, non_blocking=self.asynchronous),
                            )
                            for offset, rows, target in plan.parts
                        ]
                    )
                    for plan in source
                }

        if not self.asynchronous:
            return move(), None
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            result = move()
            event = torch.cuda.Event()
            event.record(self.stream)
        return result, event

    def wait(self, event, plans):
        if not self.asynchronous:
            return
        stream = torch.cuda.current_stream(self.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        stream.wait_event(event)
        end.record(stream)
        self._timings.append((start, end))
        for plan in plans.values():
            for _, _, target in plan.parts:
                for layer in target.sets:
                    for value in vars(layer).values():
                        values = value.values() if isinstance(value, dict) else (value,)
                        for tensor in values:
                            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                                tensor.record_stream(stream)
        self.collect()

    def collect(self):
        remaining = []
        for start, end in self._timings:
            if end.query():
                self.report.prefetch_wait_seconds += start.elapsed_time(end) / 1000
            else:
                remaining.append((start, end))
        self._timings = remaining


class PlanPrefetch:
    def __init__(self, plans, config):
        groups = {}
        self.locations = {}
        for plan in plans:
            block = plan.parts[0][2].place[0]
            groups.setdefault(block, []).append(plan)
        self.groups = list(groups.values())
        for block, group in enumerate(self.groups):
            for plan in group:
                self.locations[id(plan)] = block
        self.config = config
        self.scheduler = None
        self.seen = set()
        self.current = None
        self.values = None
        for group in self.groups:
            for plan in group:
                plan._prefetch = self
        self.pin()

    def pin(self):
        if not torch.cuda.is_available():
            return
        for group in self.groups:
            for plan in group:
                for _, _, target in plan.parts:
                    for layer in target.sets:
                        for key, value in vars(layer).items():
                            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                                if not value.is_pinned():
                                    setattr(layer, key, value.pin_memory())
                            elif key == "gate" and value is not None:
                                layer.gate = {
                                    k: v.pin_memory()
                                    if v.device.type == "cpu" and not v.is_pinned()
                                    else v
                                    for k, v in value.items()
                                }

    def reset(self):
        if self.scheduler is not None:
            self.scheduler.reset()
        self.current, self.values = None, None
        self.seen.clear()
        self.pin()

    def get(self, plan, device, dtype):
        block = self.locations[id(plan)]
        if self.scheduler is None or (
            self.scheduler.backend.device,
            self.scheduler.backend.dtype,
        ) != (device, dtype):
            backend = PlanCopies(
                device, dtype, self.config["report"], self.config["mode"] == "auto"
            )
            self.scheduler = BlockPrefetch(
                len(self.groups),
                self.config["window"],
                lambda i: self.groups[i],
                backend,
                enabled=self.config["mode"] == "auto",
            )
            self.current, self.values = None, None
            self.seen.clear()
        if self.current != block:
            if self.current is not None:
                self.scheduler.end(self.current)
            self.values = self.scheduler.begin(block)
            self.current = block
            self.seen.clear()
        return self.values[id(plan)]

    def finished(self, plan):
        block = self.locations[id(plan)]
        if self.current != block:
            return
        self.seen.add(id(plan))
        if len(self.seen) == len(self.groups[block]):
            self.scheduler.end(block)
            self.current, self.values = None, None
            self.seen.clear()
