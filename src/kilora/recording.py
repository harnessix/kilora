# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

_ACTIVE: contextvars.ContextVar[PathRecorder | None] = contextvars.ContextVar(
    "kilora_path_recorder", default=None
)


def active_recorder() -> PathRecorder | None:
    return _ACTIVE.get()


class ContextStack:
    def __init__(self, name: str) -> None:
        self._var: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(name, default=None)

    def get(self) -> tuple:
        return self._var.get() or ()

    def __bool__(self) -> bool:
        return self._var.get() is not None

    def __len__(self) -> int:
        return len(self._var.get() or ())

    def __iter__(self) -> Iterator:
        return iter(self._var.get() or ())

    def __getitem__(self, index):
        return (self._var.get() or ())[index]

    def __contains__(self, entry) -> bool:
        return entry in (self._var.get() or ())

    def holds(self, entry) -> bool:
        current = self._var.get()
        return current is not None and any(e is entry for e in current)

    @contextlib.contextmanager
    def pushed(self, entry) -> Iterator[Any]:
        value = (*(self._var.get() or ()), entry)
        token = self._var.set(value)
        try:
            yield entry
        finally:
            self._pop(token, value, entry)

    @contextlib.contextmanager
    def hidden(self) -> Iterator[tuple]:
        entries = self.get()
        token = self._var.set(None)
        try:
            yield entries
        finally:
            self._var.reset(token)

    def _pop(self, token, value: tuple, entry) -> None:
        current = self._var.get()
        if current is value:
            try:
                self._var.reset(token)
                return
            except ValueError:
                pass
        if current is None:
            return
        for index in range(len(current) - 1, -1, -1):
            if current[index] is entry:
                rest = current[:index] + current[index + 1 :]
                self._var.set(rest or None)
                return


@dataclass
class Perturbation:
    proj: dict[str, torch.Tensor]
    du: dict[str, torch.Tensor]

    def delta(self, target: str, h: torch.Tensor) -> torch.Tensor | None:
        if target not in self.du:
            return None
        proj = self.proj[target].to(h.device, h.dtype)
        return (h @ proj.t()) @ self.du[target].to(h.device, h.dtype).t()


_PERTURB: contextvars.ContextVar[Perturbation | None] = contextvars.ContextVar(
    "kilora_perturbation", default=None
)


def active_perturbation() -> Perturbation | None:
    return _PERTURB.get()


@contextlib.contextmanager
def perturbing(p: Perturbation) -> Iterator[Perturbation]:
    token = _PERTURB.set(p)
    try:
        yield p
    finally:
        _PERTURB.reset(token)


@dataclass
class _Sums:
    vk: torch.Tensor
    cc: torch.Tensor
    c2: torch.Tensor
    n: int = 0


@dataclass
class PredictionError:
    progress: float
    read: torch.Tensor
    short: torch.Tensor
    deep: torch.Tensor
    error: torch.Tensor
    precision: torch.Tensor
    node: int
    z: torch.Tensor | None = None


@dataclass
class PathStatistics:
    c_kk: dict[str, torch.Tensor]
    c_vk: dict[str, torch.Tensor]
    tokens: dict[str, int]
    pairs: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    pair_inputs: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    prediction_errors: dict[str, list[PredictionError]] = field(default_factory=dict)
    cross_errors: dict[str, list[CrossStepError]] = field(default_factory=dict)
    block_reads: list[BlockRead] = field(default_factory=list)
    hier_errors: dict[str, list[LevelError]] = field(default_factory=dict)
    clamp_errors: dict[str, list[LevelError]] = field(default_factory=dict)


@dataclass
class PathRecorder:
    a: dict[str, torch.Tensor]
    v0: dict[str, torch.Tensor]
    rows: Sequence[int] | None = None
    enabled: Callable[[], bool] | None = None
    pair_targets: Sequence[str] = ()
    sums: dict[str, _Sums] = field(default_factory=dict)
    pairs: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    capture_pair_inputs: bool | set[tuple[str, str]] = False
    prediction_set_key: str | None = None
    pair_inputs: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    prediction_errors: dict[str, list[PredictionError]] = field(default_factory=dict)
    _depth_steps: dict[str, dict[float, bool]] = field(default_factory=dict, repr=False)
    _raw_call: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _call: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _proj: dict[tuple[str, Any], torch.Tensor] = field(default_factory=dict, repr=False)
    cross: CrossStepRecorder | None = None
    blocks: BlockStreamRecorder | None = None

    def wants(self, target: str) -> bool:
        return target in self.a

    def wants_output(self, target: str) -> bool:
        return self.cross is not None and self.cross.wants_output(target)

    def observe_output(self, target: str, x: torch.Tensor, h: torch.Tensor, o: torch.Tensor):
        if self.cross is not None and (self.enabled is None or self.enabled()):
            self.cross.rows = self.rows
            self.cross.observe_output(target, x, h, o)

    def set_sigma(self, sigma: float | None) -> None:
        if self.cross is not None:
            self.cross.set_sigma(sigma)
        if self.blocks is not None:
            self.blocks.rows = self.rows
            self.blocks.enabled = self.enabled
            self.blocks.set_sigma(sigma)

    def observe(self, target: str, y: torch.Tensor) -> None:
        if target not in self.a or (self.enabled is not None and not self.enabled()):
            return
        with torch.no_grad():
            if self.rows is not None and y.dim() > 1:
                idx = torch.as_tensor(list(self.rows), device=y.device, dtype=torch.long)
                y = y.index_select(0, idx)
            raw = y.reshape(-1, y.shape[-1]).detach()
            y = raw.double()
            key = (target, y.device)
            if key not in self._proj:
                self._proj[key] = (self.v0[target].double() @ self.a[target].double()).to(y.device)
            c = y @ self._proj[key].t()
            if self.pair_targets and target in self.pair_targets:
                index = self.pair_targets.index(target)
                previous = self._call.pop(self.pair_targets[index - 1], None) if index else None
                prev_raw = self._raw_call.pop(self.pair_targets[index - 1], None) if index else None
                if previous is not None and previous.shape[0] == c.shape[0] and c.shape[0] > 0:
                    count = min(256, c.shape[0])
                    indices = torch.linspace(0, c.shape[0] - 1, count, device=c.device).long()
                    pair = (self.pair_targets[index - 1], target)
                    previous = previous.to(c.device, non_blocking=True)
                    self.pairs.setdefault(pair, []).append((previous[indices], c[indices]))
                    if prev_raw is not None:
                        self.pair_inputs.setdefault(pair, []).append(
                            (prev_raw[indices.to(prev_raw.device)].clone(), raw[indices].clone())
                        )
                self._call[target] = c.detach()
                next_pair = (
                    (target, self.pair_targets[index + 1])
                    if index + 1 < len(self.pair_targets)
                    else None
                )
                if next_pair is not None and (
                    self.capture_pair_inputs is True
                    or (
                        isinstance(self.capture_pair_inputs, set)
                        and next_pair in self.capture_pair_inputs
                    )
                ):
                    self._raw_call[target] = raw
            s = self.sums.get(target)
            if s is None:
                rank, kw = c.shape[1], {"dtype": torch.float64, "device": y.device}
                s = _Sums(
                    torch.zeros(y.shape[1], rank, **kw),
                    torch.zeros(rank, rank, **kw),
                    torch.zeros((), **kw),
                )
                self.sums[target] = s
            elif s.vk.device != y.device:
                s.vk, s.cc, s.c2 = s.vk.to(y.device), s.cc.to(y.device), s.c2.to(y.device)
            s.vk += y.t() @ c
            s.cc += c.t() @ c
            s.c2 += (c * c).sum()
            s.n += c.shape[0]

    def observe_depth(self, target, progress, deep):
        steps = self._depth_steps.setdefault(target, {})
        steps[float(progress)] = steps.get(float(progress), False) or deep

    def observe_prediction(self, target, base, short, deep, keep, progress, node):
        if not self.wants(target) or (self.enabled is not None and not self.enabled()):
            return
        with torch.no_grad():
            if self.rows is not None and base.dim() > 1:
                idx = torch.as_tensor(list(self.rows), device=base.device, dtype=torch.long)
                base, short, deep, keep = (
                    value.index_select(0, idx) for value in (base, short, deep, keep)
                )
            proj = (self.v0[target].double() @ self.a[target].double()).to(base.device)
            read, first, last = (
                (value.detach().double().reshape(-1, value.shape[-1]) @ proj.t()).cpu()
                for value in (base, short, deep)
            )
            precision = keep.detach().float().mean().cpu()
            self.prediction_errors.setdefault(target, []).append(
                PredictionError(
                    float(progress),
                    read,
                    first,
                    last,
                    last - first,
                    precision,
                    node,
                    read.mean(0) if read.shape[0] else None,
                )
            )

    def discard(self) -> None:
        if self.cross is not None:
            self.cross.discard()
        if self.blocks is not None:
            self.blocks.discard()
        self.sums.clear()
        self.prediction_errors.clear()
        self._depth_steps.clear()
        self.pairs.clear()
        self._call.clear()
        self._raw_call.clear()
        self.pair_inputs.clear()

    def finish(self) -> PathStatistics:
        c_kk, c_vk, tokens = {}, {}, {}
        for t, s in self.sums.items():
            if s.n == 0:
                continue
            m = s.c2.cpu() / s.n
            if not bool(m > 0):
                continue
            c_kk[t] = (s.cc.cpu() / s.n / m).float()
            c_vk[t] = (s.vk.cpu() / s.n / torch.sqrt(m)).float()
            tokens[t] = s.n
        self.sums.clear()
        pairs, self.pairs = self.pairs, {}
        self._call.clear()
        self._raw_call.clear()
        pair_inputs, self.pair_inputs = self.pair_inputs, {}
        errors, self.prediction_errors = self.prediction_errors, {}
        for target, items in errors.items():
            steps = self._depth_steps.get(target, {})
            weight = sum(steps.values()) / len(steps) if steps else 0.0
            for item in items:
                item.precision = item.precision * weight
        self._depth_steps.clear()
        cross = self.cross.finish_errors() if self.cross is not None else {}
        reads = self.blocks.finish_reads() if self.blocks is not None else []
        return PathStatistics(
            c_kk, c_vk, tokens, pairs, pair_inputs, errors, cross, block_reads=reads
        )

    @contextlib.contextmanager
    def recording(self) -> Iterator[PathRecorder]:
        self._call.clear()
        self._raw_call.clear()
        token = _ACTIVE.set(self)
        try:
            yield self
        except BaseException:
            self.discard()
            raise
        finally:
            self._call.clear()
            self._raw_call.clear()
            _ACTIVE.reset(token)


@dataclass
class CrossFrame:
    p_in: torch.Tensor
    p_out: torch.Tensor
    reads: str = "input"
    weight: torch.Tensor | None = None
    up: torch.Tensor | None = None
    down: torch.Tensor | None = None
    base: torch.Tensor | None = None
    edit: tuple = ()


@dataclass
class CrossStepError:
    step: int
    sigma: float
    progress: float
    z: torch.Tensor
    y: torch.Tensor
    predicted: torch.Tensor
    error: torch.Tensor
    precision: torch.Tensor
    node: int
    z_prev: torch.Tensor | None = None
    sigma_prev: float | None = None


def noise_weight(sigma: float) -> float:
    return 1.0 / (1.0 + float(sigma) ** 2)


def cross_step_error(z, y, z_prev, y_prev, jhat, sigma_prev, sigma):
    predicted = y_prev + (z - z_prev) @ jhat.t()
    error = (y - predicted) / (float(sigma_prev) - float(sigma))
    cos = torch.nn.functional.cosine_similarity(z, z_prev, dim=-1, eps=1e-12)
    return predicted, error, 1.0 - cos.abs()


@dataclass
class _StepSums:
    sigma: float
    progress: float
    x: torch.Tensor
    o: torch.Tensor
    n: torch.Tensor


@dataclass
class CrossStepRecorder:
    frames: dict[str, list[CrossFrame]]
    rows: Sequence[int] | None = None
    enabled: Callable[[], bool] | None = None
    region_weights: Callable[[str, int], torch.Tensor | None] | None = None
    sigma: float | None = None
    errors: dict[str, list[CrossStepError]] = field(default_factory=dict)
    _pending: dict[tuple[str, int], _StepSums] = field(default_factory=dict, repr=False)
    _last: dict[tuple[str, int], tuple[float, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, repr=False
    )
    _steps: dict[tuple[str, int], int] = field(default_factory=dict, repr=False)
    _jhat: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict, repr=False)
    blocks: BlockStreamRecorder | None = None

    def wants(self, target: str) -> bool:
        return False

    def observe(self, target: str, y: torch.Tensor) -> None:
        return None

    def wants_output(self, target: str) -> bool:
        return target in self.frames and (self.enabled is None or self.enabled())

    def set_sigma(self, sigma: float | None) -> None:
        self.sigma = None if sigma is None else float(sigma)
        if self.blocks is not None:
            self.blocks.rows = self.rows
            self.blocks.enabled = self.enabled
            self.blocks.set_sigma(sigma)

    def _rows(self, value: torch.Tensor) -> torch.Tensor:
        if self.rows is not None and value.dim() > 1:
            idx = torch.as_tensor(list(self.rows), device=value.device, dtype=torch.long)
            value = value.index_select(0, idx)
        return value.detach().reshape(-1, value.shape[-1]).double()

    def _pool(self, target, value):
        ones = torch.ones(value.shape[0], 1, dtype=value.dtype, device=value.device)
        weights = ones
        extra = self.region_weights(target, value.shape[0]) if self.region_weights else None
        if extra is not None:
            extra = extra.to(value.device, value.dtype).reshape(value.shape[0], -1)
            weights = torch.cat([ones, extra], 1)
        return weights.t() @ value, weights.sum(0)

    def observe_output(self, target: str, x, h, o) -> None:
        if self.sigma is None or not self.wants_output(target):
            return
        from .compute import current_progress

        with torch.no_grad():
            out = self._rows(o)
            if out.shape[0] == 0:
                return
            pooled_out, n = self._pool(target, out)
            inputs = {}
            for node, frame in enumerate(self.frames[target]):
                if frame.reads not in inputs:
                    source = self._rows(x if frame.reads == "input" else h)
                    if source.shape[0] != out.shape[0]:
                        continue
                    inputs[frame.reads] = self._pool(target, source)[0]
                if frame.reads not in inputs:
                    continue
                key = (target, node)
                pending = self._pending.get(key)
                if pending is not None and pending.sigma != self.sigma:
                    self._close(key)
                    pending = None
                if pending is None:
                    self._pending[key] = _StepSums(
                        self.sigma, current_progress(), inputs[frame.reads], pooled_out, n
                    )
                else:
                    pending.x = pending.x + inputs[frame.reads]
                    pending.o = pending.o + pooled_out
                    pending.n = pending.n + n

    def _j(self, key, device):
        jhat = self._jhat.get(key)
        if jhat is None or jhat.device != device:
            frame = self.frames[key[0]][key[1]]
            p_in = frame.p_in.to(device, torch.float64)
            p_out = frame.p_out.to(device, torch.float64)
            if frame.base is not None:
                jhat = frame.base.to(device, torch.float64)
            elif frame.weight is None:
                jhat = p_out @ p_in.t()
            else:
                w = frame.weight.detach()
                jhat = p_out @ (w @ frame.p_in.to(w.device, w.dtype).t()).to(device).double()
            extra = self._edit_term(frame, p_in, p_out, device)
            if extra is not None:
                jhat = jhat + extra
            self._jhat[key] = jhat
        return jhat

    def _edit_term(self, frame, p_in, p_out, device):

        if frame.up is not None and frame.down is not None:
            left = p_out @ frame.up.detach().to(device, torch.float64)
            return left @ (frame.down.detach().to(device, torch.float64) @ p_in.t())
        if not frame.edit:
            return None
        ups, downs = [], []
        with torch.no_grad():
            for up_e, down_e, rank, strength, core in frame.edit:
                b = up_e.detach().double().T
                mix = strength.detach().double().reshape(()) * (
                    core.detach().double()
                    if core is not None
                    else torch.eye(int(rank), dtype=torch.float64)
                )
                ups.append(b @ mix.to(b.device))
                downs.append(down_e.detach().double())
            up = torch.cat(ups, 1)
            down = torch.cat(downs, 0)
            left = p_out @ up.detach().to(device, torch.float64)
            return left @ (down.detach().to(device, torch.float64) @ p_in.t())

    def _close(self, key) -> None:
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        frame = self.frames[key[0]][key[1]]
        device = pending.x.device
        weight = pending.n.clamp_min(1e-12).unsqueeze(-1)
        z = (pending.x / weight) @ frame.p_in.to(device, torch.float64).t()
        y = (pending.o / weight) @ frame.p_out.to(device, torch.float64).t()
        if pending.n.shape[0] == 1:
            z, y = z[0], y[0]
        step = self._steps.get(key, 0) + 1
        self._steps[key] = step
        last = self._last.get(key)
        self._last[key] = (pending.sigma, z, y)
        if last is None or last[0] == pending.sigma or last[1].shape != z.shape:
            return
        predicted, error, change = cross_step_error(
            z, y, last[1], last[2], self._j(key, device), last[0], pending.sigma
        )
        precision = noise_weight(pending.sigma) * change
        self.errors.setdefault(key[0], []).append(
            CrossStepError(
                step,
                pending.sigma,
                pending.progress,
                z.cpu(),
                y.cpu(),
                predicted.cpu(),
                error.cpu(),
                precision.float().cpu(),
                key[1],
                last[1].cpu(),
                float(last[0]),
            )
        )

    def _clear(self) -> None:
        self._pending.clear()
        self._last.clear()
        self._steps.clear()
        self._jhat.clear()
        self.sigma = None

    def discard(self) -> None:
        self.errors = {}
        self._clear()
        if self.blocks is not None:
            self.blocks.discard()

    def finish_errors(self) -> dict[str, list[CrossStepError]]:
        for key in list(self._pending):
            self._close(key)
        errors, self.errors = self.errors, {}
        self._clear()
        return errors

    def finish(self) -> PathStatistics:
        reads = self.blocks.finish_reads() if self.blocks is not None else []
        return PathStatistics({}, {}, {}, cross_errors=self.finish_errors(), block_reads=reads)

    @contextlib.contextmanager
    def recording(self) -> Iterator[CrossStepRecorder]:
        token = _ACTIVE.set(self)
        try:
            yield self
        except BaseException:
            self.discard()
            raise
        finally:
            _ACTIVE.reset(token)


def orthonormal_rows(columns: torch.Tensor, tol: float = 1e-9) -> torch.Tensor | None:
    value = columns.detach().double()
    if value.numel() == 0:
        return None
    q, r = torch.linalg.qr(value)
    diag = r.diagonal().abs()
    if not bool((diag > tol * max(float(diag.max()), 1.0)).all()):
        return None
    return q.t().contiguous()


BLOCK_RANK = 32
TOP_BLOCKS = 4
CLAMP_DECAY = 0.7
MAP_DECAY = 0.9
MAP_RIDGE = 1e-3


def block_frame(rows: Sequence[torch.Tensor], k: int = BLOCK_RANK) -> torch.Tensor | None:
    if not rows:
        return None
    value = torch.cat([row.detach().to("cpu", torch.float64) for row in rows], 0)
    if value.numel() == 0:
        return None
    _, sv, vh = torch.linalg.svd(value, full_matrices=False)
    keep = min(int(k), int((sv > 1e-9 * max(float(sv[0]), 1.0)).sum()))
    return vh[:keep].contiguous() if keep > 0 else None


def block_index(target: str, attr: str) -> int | None:
    import re

    match = re.search(rf"(?:^|\.){re.escape(attr)}\.(\d+)\.", target)
    return int(match[1]) if match else None


def _output_rows(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(value)


@dataclass
class BlockFrames:
    outputs: dict
    k: int = BLOCK_RANK
    _cache: dict[tuple[str, int, int], torch.Tensor | None] = field(
        default_factory=dict, repr=False
    )

    def frame(self, attr: str, index: int, width: int) -> torch.Tensor | None:
        key = (attr, int(index), int(width))
        if key not in self._cache:
            rows = []
            for target, value in sorted(self.outputs.items()):
                if block_index(target, attr) != index:
                    continue
                rows.extend(row for row in _output_rows(value) if row.shape[-1] == width)
            self._cache[key] = block_frame(rows, self.k)
        return self._cache[key]


@dataclass
class BlockRead:
    step: int
    sigma: float
    progress: float
    reads: dict[int, torch.Tensor]


@dataclass
class BlockStreamRecorder:
    frames: BlockFrames
    attr: str = "blocks"
    rows: Sequence[int] | None = None
    enabled: Callable[[], bool] | None = None
    region_weights: Callable[[int, int], torch.Tensor | None] | None = None
    sigma: float | None = None
    levels: set[int] | None = None
    reads: list[BlockRead] = field(default_factory=list)
    _pending: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)
    _pending_sigma: float | None = field(default=None, repr=False)
    _pending_progress: float = field(default=0.0, repr=False)
    _step: int = field(default=0, repr=False)

    def set_sigma(self, sigma: float | None) -> None:
        self.sigma = None if sigma is None else float(sigma)

    def observe_block(self, index: int, h: torch.Tensor) -> None:
        if self.sigma is None or (self.enabled is not None and not self.enabled()):
            return
        if self.levels is not None and index not in self.levels:
            return
        if not isinstance(h, torch.Tensor) or h.dim() < 2:
            return
        from .compute import current_progress

        with torch.no_grad():
            frame = self.frames.frame(self.attr, index, h.shape[-1])
            if frame is None:
                return
            if self.rows is not None:
                idx = torch.as_tensor(list(self.rows), device=h.device, dtype=torch.long)
                h = h.index_select(0, idx)
            value = h.detach().reshape(-1, h.shape[-1]).double()
            if value.shape[0] == 0:
                return
            if self._pending_sigma is not None and self._pending_sigma != self.sigma:
                self._close()
            if self._pending_sigma is None:
                self._pending_sigma, self._pending_progress = self.sigma, current_progress()
            ones = torch.ones(value.shape[0], 1, dtype=value.dtype, device=value.device)
            weights = ones
            extra = self.region_weights(index, value.shape[0]) if self.region_weights else None
            if extra is not None:
                extra = extra.to(value.device, value.dtype).reshape(value.shape[0], -1)
                weights = torch.cat([ones, extra], 1)
            total, n = weights.t() @ value, weights.sum(0)
            pending = self._pending.get(index)
            if pending is not None and pending[0].shape == total.shape:
                total, n = pending[0] + total, pending[1] + n
            self._pending[index] = (total, n)

    def _close(self) -> None:
        if self._pending_sigma is None:
            return
        reads = {}
        for index, (total, n) in sorted(self._pending.items()):
            frame = self.frames.frame(self.attr, index, total.shape[-1])
            mean = total / n.clamp_min(1e-12).unsqueeze(-1)
            s = (mean @ frame.to(mean.device, torch.float64).t()).cpu()
            reads[index] = s[0] if s.shape[0] == 1 else s
        self._step += 1
        if reads:
            sigma, progress = self._pending_sigma, self._pending_progress
            self.reads.append(BlockRead(self._step, sigma, progress, reads))
        self._pending.clear()
        self._pending_sigma = None

    def discard(self) -> None:
        self.reads = []
        self._pending.clear()
        self._pending_sigma = None
        self._step = 0
        self.sigma = None

    def finish_reads(self) -> list[BlockRead]:
        self._close()
        reads, self.reads = self.reads, []
        self.discard()
        return reads


_BLOCK_OBSERVERS = ContextStack("kilora_block_observers")


@contextlib.contextmanager
def observing_blocks(modules, recorder: BlockStreamRecorder | None) -> Iterator[None]:

    if recorder is None:
        yield
        return
    marker = object()
    handles = []
    for index, module in modules:

        def hook(_module, _args, output, index=index):
            if not _BLOCK_OBSERVERS.holds(marker):
                return
            value = output[0] if isinstance(output, (tuple, list)) and output else output
            if isinstance(value, torch.Tensor):
                recorder.observe_block(index, value)

        handles.append(module.register_forward_hook(hook))
    try:
        with _BLOCK_OBSERVERS.pushed(marker):
            yield
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class LevelError:
    step: int
    sigma: float
    progress: float
    level: int
    s: torch.Tensor
    predicted: torch.Tensor
    error: torch.Tensor
    precision: torch.Tensor
    node: int = 0


def _first(read: torch.Tensor) -> torch.Tensor:
    return read[0] if read.dim() > 1 else read


@dataclass
class BlockMaps:
    decay: float = MAP_DECAY
    sums: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)

    PREFIX = "prior/hier/"

    def map(self, level: int) -> torch.Tensor | None:
        item = self.sums.get(level)
        if item is None:
            return None
        auto = item["auto"]
        ridge = max(MAP_RIDGE * float(auto.diagonal().mean()), 1e-12)
        eye = torch.eye(auto.shape[0], dtype=auto.dtype)
        return torch.linalg.solve(auto + ridge * eye, item["cross"].t()).t()

    def residual(self, level: int, g: torch.Tensor | None = None) -> float | None:
        item = self.sums.get(level)
        if item is None:
            return None
        g = self.map(level) if g is None else g
        value = item["self"].trace() - 2 * (g * item["cross"]).sum() + (g @ item["auto"] * g).sum()
        return max(float(value), 1e-12)

    def update(self, reads: Sequence[BlockRead]) -> None:
        pairs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for read in reads:
            for level, s in read.reads.items():
                up = read.reads.get(level + 1)
                if up is not None:
                    pairs.setdefault(level, []).append((_first(s).double(), _first(up).double()))
        for level, items in pairs.items():
            low = torch.stack([a for a, _ in items], 1)
            high = torch.stack([b for _, b in items], 1)
            n = low.shape[1]
            new = {
                "cross": low @ high.t() / n,
                "auto": high @ high.t() / n,
                "self": low @ low.t() / n,
            }
            old = self.sums.get(level)
            if old is not None and all(old[k].shape == v.shape for k, v in new.items()):
                new = {k: self.decay * old[k] + (1 - self.decay) * v for k, v in new.items()}
            self.sums[level] = new

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            f"{self.PREFIX}{level}/{name}": value.clone()
            for level, item in self.sums.items()
            for name, value in item.items()
        }

    @classmethod
    def from_tensors(cls, tensors, decay: float = MAP_DECAY) -> BlockMaps:
        maps = cls(decay)
        for key, value in tensors.items():
            if not key.startswith(cls.PREFIX):
                continue
            level, _, name = key[len(cls.PREFIX) :].partition("/")
            if level.isdigit() and name in ("cross", "auto", "self"):
                maps.sums.setdefault(int(level), {})[name] = value.detach().cpu().double().clone()
        maps.sums = {k: v for k, v in maps.sums.items() if len(v) == 3}
        return maps


def level_name(level: int, attr: str = "blocks") -> str:
    return f"{attr}.{level}"


def _fitted(maps: BlockMaps):
    out = {}
    for level in maps.sums:
        g = maps.map(level)
        out[level] = (g, maps.residual(level, g))
    return out


def _like(value: float, s: torch.Tensor) -> torch.Tensor:
    shape = (s.shape[0],) if s.dim() > 1 else ()
    return torch.full(shape, float(value), dtype=torch.float32)


def hierarchy_errors(maps: BlockMaps, reads: Sequence[BlockRead], attr: str = "blocks"):
    fitted = _fitted(maps)
    out: dict[str, list[LevelError]] = {}
    for read in reads:
        w = noise_weight(read.sigma)
        for level, s in read.reads.items():
            up = read.reads.get(level + 1)
            g, residual = fitted.get(level, (None, None))
            if up is None or g is None or g.shape != (s.shape[-1], up.shape[-1]):
                continue
            if s.dim() != up.dim():
                continue
            predicted = up.double() @ g.t()
            error = s.double() - predicted
            out.setdefault(level_name(level, attr), []).append(
                LevelError(
                    read.step,
                    read.sigma,
                    read.progress,
                    level,
                    s,
                    predicted,
                    error,
                    _like(w / residual, s),
                )
            )
    return out


def clamp_errors(
    maps: BlockMaps,
    reads: Sequence[BlockRead],
    targets: dict[float, dict[int, torch.Tensor]],
    rho: float = CLAMP_DECAY,
    top: int = TOP_BLOCKS,
    attr: str = "blocks",
):
    fitted = _fitted(maps)
    out: dict[str, list[LevelError]] = {}

    def add(read, level, predicted, top_level):
        s = read.reads[level].double()
        pi = noise_weight(read.sigma) * rho ** (top_level - level)
        out.setdefault(level_name(level, attr), []).append(
            LevelError(
                read.step,
                read.sigma,
                read.progress,
                level,
                read.reads[level],
                predicted,
                predicted - s,
                _like(pi, s),
            )
        )

    for read in reads:
        star = targets.get(float(read.sigma))
        if not star or not read.reads:
            continue
        levels = sorted(read.reads)
        top_level = levels[-1]
        clamped = [
            level
            for level in levels[-top:]
            if level in star and star[level].shape == read.reads[level].shape
        ]
        if not clamped:
            continue
        for level in clamped:
            add(read, level, star[level].double(), top_level)
        level = min(clamped)
        carried = star[level].double()
        while level - 1 in read.reads:
            g = fitted.get(level - 1, (None, None))[0]
            below = read.reads[level - 1]
            if g is None or g.shape != (below.shape[-1], carried.shape[-1]):
                break
            if below.dim() != carried.dim():
                break
            carried = carried @ g.t()
            add(read, level - 1, carried, top_level)
            level -= 1
    return out


def record_hierarchy(maps: BlockMaps | None, stats, extra=None, attr: str = "blocks") -> None:
    if stats is None or maps is None or getattr(stats, "_hierarchy_done", False):
        return
    reads = getattr(stats, "block_reads", None) or []
    targets = (extra or {}).get("target_reads") or {}
    stats.hier_errors = hierarchy_errors(maps, reads, attr)
    stats.clamp_errors = clamp_errors(maps, reads, targets, attr=attr) if targets else {}
    maps.update(reads)
    stats._hierarchy_done = True


def prediction_debug() -> bool:
    return os.environ.get("KILORA_PREDICTION_DEBUG", "").strip() == "1"


def prediction_counts(depth, cross, hier=None, clamp=None) -> dict[str, int]:
    depth, cross, hier, clamp = depth or {}, cross or {}, hier or {}, clamp or {}
    out = {
        "eps_depth_steps": len({item.progress for items in depth.values() for item in items}),
        "eps_depth_nodes": len({(t, item.node) for t, items in depth.items() for item in items}),
    }
    for name, errors in (("cross", cross), ("hier", hier), ("clamp", clamp)):
        out[f"eps_{name}_steps"] = len({item.step for items in errors.values() for item in items})
        out[f"eps_{name}_nodes"] = len(
            {(t, item.node) for t, items in errors.items() for item in items}
        )
    names = ("depth", "cross", "hier", "clamp")
    out["eps_steps"] = sum(out[f"eps_{name}_steps"] for name in names)
    out["eps_nodes"] = sum(out[f"eps_{name}_nodes"] for name in names)
    return out


def prediction_note(depth, cross, hier=None, clamp=None, target_seconds=None) -> str:
    counts = prediction_counts(depth, cross, hier, clamp)
    line = ""
    if prediction_debug() and (counts["eps_steps"] or counts["eps_nodes"]):
        order = ("eps_steps", "eps_nodes", "eps_depth_steps", "eps_depth_nodes")
        order += ("eps_cross_steps", "eps_cross_nodes")
        if hier or clamp:
            order += ("eps_hier_steps", "eps_hier_nodes", "eps_clamp_steps", "eps_clamp_nodes")
        line = " " + " ".join(f"{k}={counts[k]}" for k in order)
    if target_seconds is not None:
        line += f" target_forward_s={float(target_seconds):.2f}"
    return line


def prediction_report(depth, cross, hier=None, clamp=None) -> str:
    parts = []
    sources = (("depth", depth or {}), ("cross", cross or {}))
    sources += (("hier", hier or {}), ("clamp", clamp or {}))
    for name, errors in sources:
        items = [(t, e) for t, entries in errors.items() for e in entries]
        if not items:
            parts.append(f"{name}: count=0")
            continue
        abs_error = sum(float(e.error.abs().mean()) for _, e in items) / len(items)
        precision = sum(float(torch.as_tensor(e.precision).mean()) for _, e in items) / len(items)
        score: dict[str, list[float]] = {}
        for t, e in items:
            pi = torch.as_tensor(e.precision, dtype=torch.float64)
            err = e.error.double()
            sq = err.pow(2).sum(-1) if err.dim() else err.pow(2)
            if sq.dim() and pi.dim() == 0:
                sq = sq.mean()
            value = (pi * sq).mean() if pi.dim() else pi * sq.mean()
            score.setdefault(t, []).append(float(value))
        ranked = sorted(score, key=lambda t: -sum(score[t]) / len(score[t]))[:3]
        top = ",".join(f"{t}:{sum(score[t]) / len(score[t]):.4g}" for t in ranked)
        parts.append(
            f"{name}: count={len(items)} mean_abs_error={abs_error:.4g}"
            f" mean_precision={precision:.4g} top=[{top}]"
        )
    return " Prediction debug: " + "; ".join(parts) + "."


@dataclass
class GenerationRecord:
    stats: PathStatistics | None = None
    latent: torch.Tensor | None = None
    conditioning: dict[str, Any] | None = None
    timesteps: list[float] = field(default_factory=list)
    images: int = 1
    measure: Callable[[Any], float] | None = None
    guard_score: Callable[[Any], dict[str, float]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
