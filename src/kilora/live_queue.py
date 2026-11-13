# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import collections
import contextlib
import contextvars
import copy
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("kilora.live")

DEFAULT_MAX_PENDING = 2

STARVE_PROMPTS = 3

_YIELD: contextvars.ContextVar[Callable[[], None] | None] = contextvars.ContextVar(
    "kilora_yield", default=None
)


_Hook = Callable[[], None]
_PAUSE: contextvars.ContextVar[Callable[[_Hook, _Hook], None] | None] = contextvars.ContextVar(
    "kilora_pause", default=None
)


def yield_point() -> float:
    fn = _YIELD.get()
    if fn is not None:
        return fn() or 0.0
    return 0.0


class Preempted(BaseException):
    def __init__(self, seconds: float) -> None:
        super().__init__(seconds)
        self.seconds = seconds


_PREEMPT: contextvars.ContextVar[Callable[[], None] | None] = contextvars.ContextVar(
    "kilora_preempt", default=None
)
_QUEUE: contextvars.ContextVar[WriteQueue | None] = contextvars.ContextVar(
    "kilora_queue", default=None
)


def preempt_check() -> None:
    fn = _PREEMPT.get()
    if fn is not None:
        fn()


def recorders_active() -> bool:
    from . import compute, recording

    return bool(
        compute._PROBES
        or compute._SHARES
        or compute._DIRECTION_SHARES
        or compute._SET_SHARES
        or compute._LAYER_TIMERS
        or recording._BLOCK_OBSERVERS
    )


@contextlib.contextmanager
def preemptible(enabled: bool = True) -> Iterator[None]:

    queue = _QUEUE.get()
    if queue is None or not enabled or queue._busy() or recorders_active():
        yield
        return

    def check() -> None:
        if queue._busy() and not queue._stop:
            began = queue._generation_began if queue._generating else queue._prompt_began
            raise Preempted(max(time.perf_counter() - began, 0.0))

    token = _PREEMPT.set(check)
    try:
        yield
    finally:
        _PREEMPT.reset(token)


def on_pause(release: Callable[[], None], restore: Callable[[], None]) -> None:
    fn = _PAUSE.get()
    if fn is not None:
        fn(release, restore)


def empty_device_cache() -> None:
    import importlib

    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        importlib.import_module("comfy.model_management").soft_empty_cache()


def async_enabled() -> bool:
    return os.environ.get("KILORA_LIVE_ASYNC", "1").strip().lower() not in ("0", "false", "off")


@dataclass
class Event:
    number: int
    run: Callable[[], str]
    branch: str | None = None
    action: str | None = None
    revision: tuple | None = None
    snapshot: Any = None
    context: contextvars.Context | None = None


_CALL_MODULES = (
    "kilora.call_values",
    "kilora.compute",
    "kilora.linear_jump",
    "kilora.recording",
    "kilora.route_freeze",
)


def _clear_call_values() -> None:
    import sys

    recording = sys.modules.get("kilora.recording")
    if recording is not None:
        for name in _CALL_MODULES:
            module = sys.modules.get(name)
            for value in vars(module).values() if module is not None else ():
                if isinstance(value, recording.ContextStack) and value._var.get() is not None:
                    value._var.set(None)
        for var in (recording._ACTIVE, recording._PERTURB):
            if var.get() is not None:
                var.set(None)
    ablation = sys.modules.get("kilora.ablation_settings")
    if ablation is not None:
        if ablation._ACTIVE.get() is not None:
            ablation._ACTIVE.set(None)
        if ablation._TIMESTEP.get() != 0:
            ablation._TIMESTEP.set(0)
    for var in (_YIELD, _PAUSE, _PREEMPT, _QUEUE):
        if var.get() is not None:
            var.set(None)


def write_context() -> contextvars.Context:
    context = contextvars.copy_context()
    context.run(_clear_call_values)
    return context


_ALL_QUEUES: weakref.WeakSet[WriteQueue] = weakref.WeakSet()
_ALL_LOCK = threading.Lock()
_PROMPTS_RUNNING = 0


def begin_prompts() -> None:
    global _PROMPTS_RUNNING
    with _ALL_LOCK:
        _PROMPTS_RUNNING += 1
        queues = list(_ALL_QUEUES)
    for q in queues:
        q.begin_prompt()


def end_prompts() -> None:
    global _PROMPTS_RUNNING
    with _ALL_LOCK:
        _PROMPTS_RUNNING = max(0, _PROMPTS_RUNNING - 1)
        queues = list(_ALL_QUEUES)
    for q in queues:
        q.end_prompt()


def note_timing(timing: dict[str, Any] | None) -> bool:
    queue = _QUEUE.get()
    if queue is None:
        return False
    if timing:
        totals = queue._write_timing
        for key in ("preempts", "preempt_wait_s", "preempt_dropped_s"):
            totals[key] = totals.get(key, 0) + timing.get(key, 0)
    return True


class WriteQueue:
    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self.max_pending = max_pending
        self._pending: collections.deque[Event] = collections.deque()
        self._lock = threading.RLock()
        self._batch = threading.RLock()
        self._pause_hooks: list[tuple[Callable[[], None], Callable[[], None]]] = []
        self._cond = threading.Condition()
        self._generating = 0
        self._generation_began = 0.0
        self._prompt_began = time.perf_counter()
        self._starved = 0
        self._write_timing: dict[str, float] = {}
        self._count = 0
        self._running: Event | None = None
        self._stop = False
        self._side_branches: dict[tuple, str] = {}
        self._keeps: dict[str, list[tuple[Event, bool]]] = {}
        self.reports: list[tuple[int, str]] = []
        self.dropped: list[int] = []
        with _ALL_LOCK:
            self._prompts = _PROMPTS_RUNNING
            _ALL_QUEUES.add(self)
        self._worker = threading.Thread(target=self._loop, name="kilora-live-writer", daemon=True)
        self._worker.start()

    def begin_generation(self) -> None:
        with self._cond:
            if not self._generating:
                self._generation_began = time.perf_counter()
            self._generating += 1

    def end_generation(self) -> None:
        with self._cond:
            self._generating = max(0, self._generating - 1)
            self._cond.notify_all()

    @contextlib.contextmanager
    def generation(self) -> Iterator[None]:
        self.begin_generation()
        try:
            yield
        finally:
            self.end_generation()

    def begin_prompt(self) -> None:
        with self._cond:
            if not self._prompts:
                self._prompt_began = time.perf_counter()
            self._prompts += 1

    def end_prompt(self) -> None:
        with self._cond:
            self._prompts = max(0, self._prompts - 1)
            if self._pending or self._running is not None:
                self._starved += 1
            self._cond.notify_all()

    def _busy(self) -> bool:
        if self._generating:
            return True
        return bool(self._prompts) and self._starved < STARVE_PROMPTS

    def _wait_for_gap(self) -> float:
        start, paused = time.perf_counter(), False
        try:
            while True:
                with self._cond:
                    if not self._busy() or self._stop:
                        break
                if not paused:
                    paused = True
                    for release, _ in self._pause_hooks:
                        release()
                    empty_device_cache()
                self._batch.release()
                try:
                    with self._cond:
                        while self._busy() and not self._stop:
                            self._cond.wait(0.05)
                finally:
                    self._batch.acquire()
        finally:
            if paused:
                for _, restore in reversed(self._pause_hooks):
                    restore()
        return time.perf_counter() - start if paused else 0.0

    def _add_pause_hook(self, release: Callable[[], None], restore: Callable[[], None]) -> None:
        self._pause_hooks.append((release, restore))

    def submit(self, run: Callable[[], str]) -> int:
        with self._cond:
            self._count += 1
            event = Event(self._count, run, context=write_context())
            session = getattr(run, "session", None)
            state = getattr(session, "editor_state", None)
            forest = getattr(state, "forest", None)
            if forest is not None and hasattr(run, "run_many"):
                event.revision = (forest, forest.current, session.revision)
                event.snapshot = (copy.deepcopy(forest.state), copy.deepcopy(state.regions))
            self._pending.append(event)
            self._cond.notify_all()
            return event.number

    def pending(self) -> int:
        with self._cond:
            return len(self._pending) + (self._running is not None)

    def drop_pending(self, why: str) -> int:
        with self._cond:
            numbers = [e.number for e in self._pending]
            self._pending.clear()
        for n in numbers:
            self.dropped.append(n)
            log.info("live write %d dropped: %s", n, why)
        return len(numbers)

    def keep_pending(self, editor_state: Any, action: str) -> list[tuple[str, list[int]]] | None:
        with self._cond:
            forest = getattr(editor_state, "forest", None)
            groups: dict[tuple, list[Event]] = {}
            for event in self._pending:
                if event.branch is None and event.revision and event.revision[0] is forest:
                    groups.setdefault(event.revision, []).append(event)
            kept = []
            for revision, events in groups.items():
                branch = self._side_branches.get(revision)
                created = branch not in forest.branches
                if created:
                    branch = editor_state.fork()
                    forest.branches[branch].parent = revision[1]
                    forest.branches[branch].state = copy.deepcopy(events[0].snapshot[0])
                    editor_state._branch_regions[branch] = copy.deepcopy(events[0].snapshot[1])
                    self._side_branches[revision] = branch
                numbers = [e.number for e in events]
                item = forest.branches[branch]
                item.action = action
                item.writes.extend(numbers)
                for event in events:
                    event.branch, event.action = branch, action
                self._keeps[branch] = [(event, created) for event in events]
                kept.append((branch, numbers))
            return kept or None

    def cancel_keep(self, editor_state: Any, branch: str) -> None:
        with self._cond:
            entries = self._keeps.pop(branch, [])
            item = editor_state.forest.branches[branch]
            numbers = {event.number for event, _ in entries}
            item.writes = [n for n in item.writes if n not in numbers]
            for event, _ in entries:
                event.branch = event.action = None
            if entries and entries[0][1]:
                self._side_branches.pop(entries[0][0].revision, None)
                editor_state.delete(branch)

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._lock, self._batch:
            yield

    @contextlib.contextmanager
    def copy_guard(self) -> Iterator[None]:
        with self._batch:
            yield

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._pending and self._running is None, timeout=timeout
            )

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._worker.join(timeout=5)

    def _loop(self) -> None:
        token = _YIELD.set(self._wait_for_gap)
        pause_token = _PAUSE.set(self._add_pause_hook)
        queue_token = _QUEUE.set(self)
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(
                        lambda: (self._pending and not self._busy()) or self._stop,
                        timeout=None,
                    )
                    if self._stop:
                        return
                taken = self._take_and_run()
                if taken is None:
                    continue
                with self._cond:
                    waited, self._starved = self._starved, 0
                    totals, self._write_timing = self._write_timing, {}
                    self.reports.append(taken)
                    self._running = None
                    self._cond.notify_all()
                log.info(
                    "live write %d: %d stops for a render, %.2f s of overlap, "
                    "%.2f s dropped, %d prompts waited",
                    taken[0],
                    int(totals.get("preempts", 0)),
                    totals.get("preempt_wait_s", 0.0),
                    totals.get("preempt_dropped_s", 0.0),
                    waited,
                )
        finally:
            _QUEUE.reset(queue_token)
            _PAUSE.reset(pause_token)
            _YIELD.reset(token)

    def _run_events(self, events: list[Event]) -> str:
        token = _YIELD.set(self._wait_for_gap)
        pause_token = _PAUSE.set(self._add_pause_hook)
        queue_token = _QUEUE.set(self)
        try:
            with self._batch:
                event = events[0]
                if hasattr(event.run, "run_many"):
                    return event.run.run_many(events)
                return event.run()
        finally:
            _QUEUE.reset(queue_token)
            _PAUSE.reset(pause_token)
            _YIELD.reset(token)

    def _take_and_run(self) -> tuple[int, str] | None:
        events: list[Event] = []
        try:
            with self._lock:
                with self._cond:
                    if not self._pending or self._busy():
                        return None
                    event = self._pending.popleft()
                    events.append(event)
                    if len(self._pending) + 1 > self.max_pending and hasattr(event.run, "run_many"):
                        while (
                            self._pending
                            and self._pending[0].branch == event.branch
                            and hasattr(self._pending[0].run, "run_many")
                        ):
                            events.append(self._pending.popleft())
                    self._running = event
                number = event.number
                try:
                    if event.context is None:
                        report = self._run_events(events)
                    else:
                        context, event.context = event.context, None
                        report = context.run(self._run_events, events)
                except Exception as exc:
                    report = f"The live write failed: {exc}"
                    log.exception("live write %d failed", number)
            return number, report
        finally:
            self._pause_hooks = []
            ran = bool(events)
            for item in events:
                release = getattr(item.run, "release", None)
                if release is not None:
                    release()
                item.snapshot = None
            events.clear()
            event = None
            if ran:
                empty_device_cache()


_QUEUES_LOCK = threading.Lock()


def queue_for(session: Any) -> WriteQueue:
    with _QUEUES_LOCK:
        q = getattr(session, "write_queue", None)
        if q is None:
            q = WriteQueue()
            session.write_queue = q
        return q
