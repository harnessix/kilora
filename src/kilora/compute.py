# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import collections
import contextlib
import contextvars
import copy
import dataclasses
import hashlib
import logging
import math
import os
import struct
import weakref
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from . import ablation_settings as _ablation
from . import gate_stats as _gate_stats
from . import route_freeze as _route_freeze
from .adaptive_gate import (
    DEFAULT_KAPPA,
    HISTORY_RATE,
    adaptive_keep,
    base_reference,
    change_stats,
    resolve_mode,
)
from .live_queue import preempt_check as _preempt_check
from .recording import ContextStack
from .routing_features import enabled as routing_features_enabled
from .routing_features import layer_features

_missing_feature_input_noted = False
_IR_ACTIVE_OVERRIDES: ContextVar[int] = ContextVar("kilora_ir_active_overrides", default=0)
_IR_INTERPRET_ALL: ContextVar[bool] = ContextVar("kilora_ir_interpret_all", default=False)
_IR_FALLBACK: ContextVar[bool] = ContextVar("kilora_ir_fallback", default=False)

_HARD_GATES = ContextStack("kilora_hard_gates")

MASK_MODES = ("sample", "expect")
_MASKS = ["sample"]


def set_masks(mode: str) -> None:
    if mode not in MASK_MODES:
        raise ValueError(f"masks must be one of {MASK_MODES}, got {mode!r}")
    _MASKS[0] = mode


def mask_mode() -> str:
    env = os.environ.get("KILORA_MASKS", "").strip()
    return env if env in MASK_MODES else _MASKS[0]


def expect_masks() -> bool:
    return mask_mode() == "expect"


@contextlib.contextmanager
def hard_gates() -> Iterator[None]:
    with _HARD_GATES.pushed(True):
        yield


def top1_gate(logits: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:

    if _HARD_GATES:
        choice = (logits if noise is None else logits + noise).argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter(-1, choice, 1.0).detach()
    probs = torch.softmax(logits, dim=-1)
    choice = (logits if noise is None else logits + noise).argmax(dim=-1, keepdim=True)
    picked = probs.gather(-1, choice)
    weight = 1.0 + picked - picked.detach()
    return torch.zeros_like(probs).scatter(-1, choice, weight)


ROUTING_SCOPES = ("token", "tensor")
ROUTING_SELECTS = ("sparsemax", "top1")


def check_routing(scope: str, select: str) -> None:
    if scope not in ROUTING_SCOPES:
        raise ValueError(f"routing_scope must be one of {ROUTING_SCOPES}, got {scope!r}")
    if select not in ROUTING_SELECTS:
        raise ValueError(f"routing_select must be one of {ROUTING_SELECTS}, got {select!r}")


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    zs = torch.sort(z, dim=-1, descending=True).values
    r = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    k = (1 + r * zs > cs).sum(-1, keepdim=True).clamp(min=1)
    tau = (cs.gather(-1, k - 1) - 1) / k.to(z.dtype)
    return (z - tau).clamp(min=0)


def region_mean(logits: torch.Tensor, region: torch.Tensor | None) -> torch.Tensor:
    if logits.dim() < 2:
        return logits
    if region is None:
        return logits.mean(-2, keepdim=True)
    w = region.to(logits.dtype)
    total = w.sum(-2, keepdim=True)
    return (logits * w).sum(-2, keepdim=True) / total.clamp(min=torch.finfo(w.dtype).tiny)


def tensor_gate(
    logits: torch.Tensor,
    noise_scale: float = 0.0,
    region: torch.Tensor | None = None,
    select: str = "sparsemax",
) -> torch.Tensor:
    mean = region_mean(logits, region)
    if noise_scale > 0:
        mean = mean + gumbel(mean, noise_scale)
    return _select(mean, select).expand_as(logits)


def _select(logits: torch.Tensor, select: str) -> torch.Tensor:
    return top1_gate(logits) if select == "top1" else sparsemax(logits)


def token_gate(
    logits: torch.Tensor, noise_scale: float = 0.0, select: str = "sparsemax"
) -> torch.Tensor:
    if noise_scale > 0:
        noise = gumbel(logits, noise_scale)
        if select == "top1":
            return top1_gate(logits, noise)
        logits = logits + noise
    return _select(logits, select)


def route_shares(
    logits: torch.Tensor, scope: str = "token", select: str = "sparsemax"
) -> torch.Tensor:
    lg = logits.detach().float()
    with hard_gates():
        if scope == "tensor":
            p = tensor_gate(lg, select=select)
            return p[..., :1, :].reshape(-1, lg.shape[-1]).sum(0) if lg.dim() > 1 else p
        return token_gate(lg, select=select).reshape(-1, lg.shape[-1]).sum(0)


CARRIES = ("drop", "dropped")


SCORE_EPS = 1e-6
DEFAULT_TAU = 1.0


def orthonormal_keys(keys: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:

    k = keys.float()
    out = []
    for j in range(k.shape[1]):
        v = k[:, j]
        for _ in range(2):
            for q in out:
                v = v - (v * q).sum(-1, keepdim=True) * q
        out.append(v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12))
    q = torch.stack(out, 1)
    if rows is None:
        return q
    used = torch.arange(keys.shape[1], device=keys.device)[None, :] < rows.to(keys.device)[:, None]
    return q * used[..., None]


def key_rows(keys: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    if rows is None:
        return torch.full((keys.shape[0],), float(keys.shape[1]), device=keys.device)
    return rows.to(keys.device).float()


def norm_energy(v: torch.Tensor, keys: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    proj = torch.einsum("...r,emr->...em", v, keys)
    share = proj.pow(2).sum(-1) / (v.pow(2).sum(-1, keepdim=True) + SCORE_EPS)
    return share / (key_rows(keys, rows) / keys.shape[-1])


def cross_energy(
    delta: torch.Tensor, token: torch.Tensor, keys: torch.Tensor, rows: torch.Tensor | None
) -> torch.Tensor:
    proj = torch.einsum("...r,emr->...em", delta, keys)
    share = proj.pow(2).sum(-1) / (token.pow(2).sum(-1, keepdim=True) + SCORE_EPS)
    return share / (key_rows(keys, rows) / keys.shape[-1])


def out_project(q: torch.Tensor, keys_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    proj = torch.einsum("...r,emr->...em", q, keys_out)
    return torch.einsum("...e,...em,emr->...r", gate.to(q.dtype), proj, keys_out)


def router_logits(
    q: torch.Tensor,
    keys: torch.Tensor,
    tau: torch.Tensor | float = DEFAULT_TAU,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    return tau * norm_energy(q.float(), orthonormal_keys(keys, rows), rows)


DROPOUT_DELTA = 1e-6
_M32 = 0xFFFFFFFF


class _BaseStack(ContextStack):
    def __init__(self, name: str, base) -> None:
        super().__init__(name)
        self._base = (base,)

    def get(self) -> tuple:
        return self._var.get() or self._base

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.get())

    def __iter__(self) -> Iterator:
        return iter(self.get())

    def __getitem__(self, index):
        return self.get()[index]

    def __contains__(self, entry) -> bool:
        return entry in self.get()

    @contextlib.contextmanager
    def pushed(self, entry) -> Iterator[Any]:
        value = (*self.get(), entry)
        token = self._var.set(value)
        try:
            yield entry
        finally:
            self._pop(token, value, entry)


_DROPOUT_KEY = _BaseStack("kilora_dropout_key", (0, 0))


def _mix32(x):
    if isinstance(x, torch.Tensor):
        x = x & _M32
        x ^= x >> 16
        x *= 0x7FEB352D
        x &= _M32
        x ^= x >> 15
        x *= 0x846CA68B
        x &= _M32
        x ^= x >> 16
        return x
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    return x ^ (x >> 16)


def step_key(step: float | int | torch.Tensor) -> int:
    if isinstance(step, torch.Tensor):
        step = float(step.detach().reshape(-1)[0]) if step.numel() else 0.0
    if isinstance(step, int):
        return step & _M32
    return struct.unpack("<I", struct.pack("<f", float(step)))[0]


@contextlib.contextmanager
def dropout_key(seed: int = 0, step: float | int | torch.Tensor = 0) -> Iterator[None]:
    with _DROPOUT_KEY.pushed((int(seed) & _M32, step_key(step))), _ablation.timestep(step):
        yield


def dropout_uniforms(
    shape: tuple[int, ...], target: str, k: int, device=None, columns: torch.Tensor | None = None
) -> torch.Tensor:
    seed, step = _DROPOUT_KEY[-1]
    base = _mix32(seed ^ _mix32(step ^ _mix32(zlib.crc32(target.encode()) ^ _mix32(k))))
    if columns is None:
        h = _mix32(_index_hash(math.prod(shape), device) ^ base)
    else:
        c = columns.numel()
        rows = _mix32(torch.arange(math.prod(shape) // c, device=device, dtype=torch.int64))
        idx = rows[:, None] ^ columns.to(device=device, dtype=torch.int64)[None, :]
        h = _mix32(_mix32(idx) ^ base)
    return ((h >> 8).to(torch.float32) / float(1 << 24)).reshape(shape)


def column_keys(s: LayerSet) -> torch.Tensor | None:
    if s.expert_ids is None:
        return None
    ids = _mix32(s.expert_ids.to(torch.int64).clone())
    j = torch.arange(s.per_expert, dtype=torch.int64, device=ids.device)
    return _mix32(ids[:, None] ^ j[None, :]).reshape(-1)


def expert_ids_of(adapter_set) -> torch.Tensor | None:
    if getattr(adapter_set, "form", None) != "input":
        return None
    ids = expert_ids_from((adapter_set.imported_from or {}).get("sources"))
    return ids if ids is not None and ids.numel() == adapter_set.num_experts else None


def expert_ids_from(sources) -> torch.Tensor | None:
    hashes = [x.get("sha256") for x in sources or []]
    if not hashes or any(not h for h in hashes):
        return None
    return torch.tensor([int(h[:8], 16) for h in hashes], dtype=torch.int64)


_INDEX_HASH: dict[tuple[int, str], torch.Tensor] = {}


def _index_hash(n: int, device) -> torch.Tensor:
    key = (n, str(device if device is not None else "cpu"))
    out = _INDEX_HASH.get(key)
    if out is None:
        if len(_INDEX_HASH) >= 8:
            _INDEX_HASH.pop(next(iter(_INDEX_HASH)))
        out = _mix32(torch.arange(n, device=device, dtype=torch.int64))
        _INDEX_HASH[key] = out
    return out


def keep_probability(c: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
    if _ablation.read_dropout_off():
        return torch.ones_like(c)
    return torch.pow(0.5, c.abs() / sigma0)


_READ_KEEP_OBSERVERS = ContextStack("kilora_read_keep_observers")
_READ_KEEP_DETAIL_OBSERVERS = ContextStack("kilora_read_keep_detail_observers")
_READ_KEEP_RAW_OBSERVERS = ContextStack("kilora_read_keep_raw_observers")


@contextlib.contextmanager
def recording_read_keep(fn):
    with _READ_KEEP_OBSERVERS.pushed(fn):
        yield


@contextlib.contextmanager
def recording_read_keep_detail(fn):
    with _READ_KEEP_DETAIL_OBSERVERS.pushed(fn):
        yield


@contextlib.contextmanager
def recording_read_keep_raw(fn):
    with _READ_KEEP_RAW_OBSERVERS.pushed(fn):
        yield


def read_dropout(
    z: torch.Tensor,
    base: torch.Tensor,
    per_expert: int,
    target: str,
    k: int,
    columns: torch.Tensor | None = None,
    set_key: str = "",
) -> torch.Tensor:
    experts = z.shape[-1] // per_expert
    shape = (*z.shape[:-1], experts, per_expert)
    with torch.no_grad():
        c0 = base.detach().float().reshape(shape)
        sigma0 = c0.pow(2).mean(-1, keepdim=True).sqrt() + DROPOUT_DELTA
        u = dropout_uniforms(shape, target, k, z.device, columns)
        q = keep_probability(z.detach().float().reshape(shape), sigma0)
        for observer in _READ_KEEP_RAW_OBSERVERS:
            observer(target, k, set_key, q.detach())
        if _READ_KEEP_OBSERVERS or _READ_KEEP_DETAIL_OBSERVERS:
            rates = q.reshape(-1, experts * per_expert).mean(0).detach()
            for observer in _READ_KEEP_OBSERVERS:
                observer(target, rates)
            for observer in _READ_KEEP_DETAIL_OBSERVERS:
                observer(target, k, set_key, rates)
        m = (q if expect_masks() else (u < q)).to(z.dtype).reshape(z.shape)
    if _HARD_GATES or not (torch.is_grad_enabled() and z.requires_grad):
        return m
    q = keep_probability(z.float().reshape(shape), sigma0).to(z.dtype).reshape(z.shape)
    return m + (q - q.detach())


def _read_dropout_for_set(z, base, per_expert, target, k, columns, set_key):
    if _READ_KEEP_DETAIL_OBSERVERS or _READ_KEEP_RAW_OBSERVERS:
        return read_dropout(z, base, per_expert, target, k, columns, set_key)
    return read_dropout(z, base, per_expert, target, k, columns)


def carry_transfer(z: torch.Tensor, base: torch.Tensor, per_expert: int) -> torch.Tensor:
    experts = z.shape[-1] // per_expert
    shape = (*z.shape[:-1], experts, per_expert)
    with torch.no_grad():
        c0 = base.detach().float().reshape(shape)
        sigma0 = c0.pow(2).mean(-1, keepdim=True).sqrt() + DROPOUT_DELTA
        zf = z.detach().float().reshape(shape)
        transferred = (
            z.detach()
            if _ablation.read_dropout_off()
            else (keep_probability(zf, sigma0) * zf).reshape(z.shape).to(z.dtype)
        )
        _ablation.note_transfer(transferred, z, base, per_expert)
        return transferred


def gumbel(shape_like: torch.Tensor, scale: float) -> torch.Tensor:
    u = torch.rand_like(shape_like).clamp(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u)) * scale


@dataclass
class LayerSet:
    router: torch.Tensor
    keys_in: torch.Tensor
    per_expert: int = 1
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    exact: dict[int, tuple[torch.Tensor, ...]] = dataclasses.field(default_factory=dict)
    scale: torch.Tensor | float = 1.0
    log_prior: torch.Tensor | float = 0.0
    tau: torch.Tensor | float = DEFAULT_TAU
    aggregation: str = "accumulate"
    key: str = ""
    expert_mask: torch.Tensor | None = None
    expert_inputs: tuple[torch.Tensor | None, ...] | None = None
    expert_offsets: tuple[torch.Tensor | None, ...] | None = None
    expert_ids: torch.Tensor | None = None
    import_score: str | None = None
    direction_core: torch.Tensor | None = None
    derived_entry_temp: float | None = None
    routing_down: torch.Tensor | None = None
    routing_up: torch.Tensor | None = None
    routing_source: tuple | None = None
    derived_cache_source: tuple | None = dataclasses.field(default=None, repr=False, compare=False)
    host_weight_norm: float | None = None
    fine: bool = False
    direction_candidates: int = 16
    form: str = "state"
    apply_once: bool = False
    routing_scope: str = "token"
    routing_select: str = "sparsemax"
    carry: str = "dropped"
    keys_out: torch.Tensor | None = None
    rows_in: torch.Tensor | None = None
    rows_out: torch.Tensor | None = None
    iteration: str = "accumulate"
    fixed_point_stop: bool = False
    path_width: int = 1
    path_select: str = "converged"
    route_freeze: float | None = None
    radial_step: float = 1.0
    orthogonal_step: float = 1.0
    radial_budget: float = 0.0
    step_metric: torch.Tensor | None = None
    share_transition_weighted: torch.Tensor | None = None
    share_transition_base: torch.Tensor | None = None
    graph_transition: torch.Tensor | None = None
    graph_powers: torch.Tensor | None = None
    graph_distances: torch.Tensor | None = None
    graph_stats: torch.Tensor | None = None
    graph_hops: torch.Tensor | None = None
    graph_merges: torch.Tensor | None = None
    jump_fits: dict | None = None
    jump_threshold: float = 0.05
    axis: str = "creative"
    axis_joint: bool = False
    fixed_depth: float | None = None
    path_starts: str = "top"
    path_prior_min: float = 0.0
    path_merge: bool = False
    path_beam: bool = False
    output_gate: str = "none"
    gate: dict | None = None
    gate_mode: str = "learned"
    gate_kappa: float = DEFAULT_KAPPA
    gate_head_limit: int | None = None
    gate_coherence: bool = True
    region: torch.Tensor | None = None
    direction_sources: list[list[dict]] | None = None
    candidates: int = 0
    ir_parts: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    candidate_summary: str = "keys"
    cross_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    learned_cross: dict[str, torch.Tensor] | None = None
    learned_cross_zero: bool = False
    entry_windows: tuple[tuple[float, float], ...] = ()
    window_indices: tuple[int, ...] = ()
    window_count: int = 0


DEPTH_STEP_DEFAULTS = (1.0, 1.0, 0.0)
_DEPTH_STEP_RANGES = {"radial_step": 2.0, "orthogonal_step": 2.0, "radial_budget": 4.0}


def check_depth_step(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= _DEPTH_STEP_RANGES[name]:
        raise ValueError(f"{name} must be in [0, {_DEPTH_STEP_RANGES[name]:g}], got {value}")
    return value


def depth_split_on(s: LayerSet) -> bool:
    return (
        (s.radial_step, s.orthogonal_step, s.radial_budget) != DEPTH_STEP_DEFAULTS
        and s.aggregation != "replace"
        and s.iteration == "accumulate"
        and s.up is not None
        and not s.apply_once
    )


def depth_step_metric(up: torch.Tensor) -> torch.Tensor:
    u = up.detach().float()
    return u @ u.t()


def _step_metric(s: LayerSet, device) -> torch.Tensor:
    if s.step_metric is not None:
        return s.step_metric.to(device, torch.float32)
    key = _tensor_key(f"depth-metric-{device}", [s.up])
    hit = _cache_get(key)
    if hit is not None:
        return hit
    g = depth_step_metric(s.up).to(device)
    _cache_put(key, g)
    return g


def metric_inner(a: torch.Tensor, b: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return ((a.float() @ metric) * b.float()).sum(-1)


def radial_budget_scale(c, zz, oo, radial_step, orthogonal_step, limit):
    b = radial_step * c
    rest = limit * limit - orthogonal_step * orthogonal_step * oo
    neg = b < 0
    back = -1 / torch.where(neg, b, -torch.ones_like(b))
    smallest = torch.where(neg, back.clamp(max=1), torch.zeros_like(b))
    full = (1 + b) ** 2 * zz <= rest
    fits = full | ((1 + smallest * b) ** 2 * zz <= rest)
    r = torch.sqrt((rest / zz.clamp_min(1e-30)).clamp_min(0))
    nonzero = torch.where(b == 0, torch.ones_like(b), b)
    root = torch.where(neg, -1 - r, r - 1) / nonzero
    t = torch.maximum(root.clamp(max=1), smallest)
    t = torch.where(full, torch.ones_like(t), torch.where(fits, t, smallest))
    return t, ~fits


_BUDGET_RECORDS = ContextStack("kilora_budget_records")


@contextlib.contextmanager
def recording_radial_budget():
    record: dict = {}
    with _BUDGET_RECORDS.pushed(record):
        yield record


def depth_split_step(z, part, s: LayerSet, metric, ref=None, weight=None, target=""):
    zf, pf = z.float(), part.float()
    zw, pw = (zf, pf) if weight is None else (zf * weight, pf * weight)
    zz = metric_inner(zw, zw, metric)
    has = zz > 0
    pz = metric_inner(pw, zw, metric)
    c = torch.where(has, pz / zz.clamp_min(1e-30), torch.zeros_like(zz))
    o = pf - c[..., None] * zf
    radial = s.radial_step * c
    if s.radial_budget > 0 and ref is not None:
        with torch.no_grad():
            ow = o if weight is None else o * weight
            oo = metric_inner(ow, ow, metric).clamp_min(0)
            t, over = radial_budget_scale(
                c.detach(), zz, oo, s.radial_step, s.orthogonal_step, s.radial_budget * ref
            )
            over = over & has
        radial = radial * t
        for record in _BUDGET_RECORDS:
            record[target] = record.get(target, 0) + int(over.sum())
    new = zf + radial[..., None] * zf + s.orthogonal_step * o
    new = torch.where(has[..., None], new, zf + pf)
    return new.to(z.dtype)


def _metric_size(z, metric, weight=None) -> torch.Tensor:
    with torch.no_grad():
        zw = z.float() if weight is None else z.float() * weight
        return metric_inner(zw, zw, metric).clamp_min(0).sqrt()


ITERATIONS = ("accumulate", "path")
PATH_SELECTS = ("converged", "score", "mix")


def _set_keys(s: LayerSet) -> tuple[torch.Tensor, torch.Tensor]:
    if s.form == "state" and s.down is not None:
        kin, _, kout, _, _ = derived_keys(s)
        return kin, kout
    tensors = [s.keys_in, s.rows_in, s.keys_out, s.rows_out]
    key = None
    if not (torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)):
        key = _tensor_key("keys", tensors)
        hit = _cache_get(key)
        if hit is not None:
            return hit
    kin = orthonormal_keys(s.keys_in, s.rows_in)
    value = (kin, kin) if s.keys_out is None else (kin, orthonormal_keys(s.keys_out, s.rows_out))
    _cache_put(key, value)
    return value


_DERIVED: dict = {}
_DERIVED_MAX = 4096


def _tensor_key(kind: str, tensors) -> tuple | None:
    try:
        key = (kind,) + tuple(
            None
            if t is None
            else (
                t.data_ptr(),
                t._version,
                tuple(t.shape),
                tuple(t.stride()),
                t.dtype,
                str(t.device),
            )
            for t in tensors
        )
        refs = tuple(None if t is None else weakref.ref(_owner(t)) for t in tensors)
    except (RuntimeError, TypeError):
        return None
    return key, refs


def _owner(t: torch.Tensor) -> torch.Tensor:
    base = t._base
    return t if base is None else base


@dataclass(frozen=True)
class _Source:
    index: int
    view: tuple | None = None


def _strip(value, owners: list):
    if isinstance(value, torch.Tensor):
        for i, owner in enumerate(owners):
            if owner is None:
                continue
            if value is owner:
                return _Source(i), True
            try:
                same = (
                    value.device == owner.device
                    and value.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
                )
            except (RuntimeError, NotImplementedError):
                same = False
            if same:
                if value.dtype != owner.dtype:
                    return value.clone(), True
                view = (tuple(value.shape), tuple(value.stride()), value.storage_offset())
                return _Source(i, view), True
        return value, False
    if isinstance(value, (tuple, list)):
        items = [_strip(v, owners) for v in value]
        if not any(found for _, found in items):
            return value, False
        return type(value)(v for v, _ in items), True
    if isinstance(value, dict):
        items = {k: _strip(v, owners) for k, v in value.items()}
        if not any(found for _, found in items.values()):
            return value, False
        return {k: v for k, (v, _) in items.items()}, True
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and not isinstance(value, type):
        items = {k: _strip(v, owners) for k, v in attrs.items()}
        if not any(found for _, found in items.values()):
            return value, False
        out = copy.copy(value)
        for k, (v, _) in items.items():
            object.__setattr__(out, k, v)
        return out, True
    return value, False


def _fill(value, owners: list):
    if isinstance(value, _Source):
        owner = owners[value.index]
        return owner if value.view is None else owner.as_strided(*value.view)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_fill(v, owners) for v in value)
    if isinstance(value, dict):
        return {k: _fill(v, owners) for k, v in value.items()}
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and not isinstance(value, type):
        out = copy.copy(value)
        for k, v in attrs.items():
            object.__setattr__(out, k, _fill(v, owners))
        return out
    return value


def _cache_get(key):
    if key is None:
        return None
    hit = _DERIVED.get(key[0])
    if hit is None:
        return None
    refs, value, stripped = hit
    owners = [None if r is None else r() for r in refs]
    if all(
        (a is None and b is None) or (a is not None and b is not None and o is b())
        for a, o, b in zip(refs, owners, key[1], strict=True)
    ):
        return _fill(value, owners) if stripped else value
    return None


def _drop_derived(cache_key, refs) -> None:
    hit = _DERIVED.get(cache_key)
    if hit is not None and hit[0] is refs:
        _DERIVED.pop(cache_key, None)


def _cache_put(key, value) -> None:
    if key is None:
        return
    if len(_DERIVED) >= _DERIVED_MAX:
        _DERIVED.pop(next(iter(_DERIVED)))
    owners = [None if r is None else r() for r in key[1]]
    stored, stripped = _strip(value, owners)
    _DERIVED[key[0]] = (key[1], stored, stripped)
    for owner in owners:
        if owner is not None:
            weakref.finalize(owner, _drop_derived, key[0], key[1])


_cross_missing_noted = set()


def _cross_logits(logits, s, keys):
    if not any(s.cross_weights):
        return logits
    from .routing_crosses import CROSSES, cross_scores
    from .routing_features import RoutingFeatures, features_for_set

    features = features_for_set(s)
    state = s.form == "state"
    width = keys.shape[-1] if state else s.router.shape[-1]
    weights = dict(zip(CROSSES, s.cross_weights, strict=True))
    projected = {}
    for name, attr in CROSSES.items():
        if not weights[name]:
            continue
        value = getattr(features, attr) if features is not None else None
        token = features.token if features is not None else None
        if value is None or token is None or value.shape[-1] != width or token.shape[-1] != width:
            missing_projection = (
                name == "token_prompt"
                and features is not None
                and features.prompt_projection_missing
            )
            note_key = (name, "projection") if missing_projection else name
            if note_key not in _cross_missing_noted:
                reason = (
                    "prompt projection is missing (needed for Anima); "
                    "feature missing or width mismatch"
                    if missing_projection
                    else "feature missing or width mismatch"
                )
                logging.getLogger("kilora").warning(
                    "Routing cross %s is unavailable: %s.", name, reason
                )
                _cross_missing_noted.add(note_key)
            weights[name] = 0.0
            continue
        projected[attr] = value.float() if state else F.linear(value.float(), s.router.float())
    if not any(weights.values()):
        return logits
    token = features.token.float() if state else F.linear(features.token.float(), s.router.float())
    transformed = RoutingFeatures(
        token, projected.get("region"), projected.get("prompt"), projected.get("timestep_embedding")
    )
    rows = derived_keys(s)[1] if state else s.rows_in
    return cross_scores(logits, keys, transformed, weights, rows)


def _learned_cross_logits(logits, s, q, keys, scale=None):
    if s.learned_cross is None:
        return logits
    from .learned_cross import cross
    from .routing_features import features_for_set

    features = features_for_set(s)
    state = s.form == "state"
    width = keys.shape[-1] if state else s.router.shape[-1]
    if features is None or features.token.shape[-1] != width:
        if "learned" not in _cross_missing_noted:
            logging.getLogger("kilora").warning(
                "Learned cross is unavailable: token feature missing or width mismatch."
            )
            _cross_missing_noted.add("learned")
        return logits
    token = features.token.reshape(*q.shape[:-1], features.token.shape[-1])
    token = token.float() if state else F.linear(token.float(), s.router.float())
    if state and token.shape[-1] != s.learned_cross["bias"].numel():
        width = s.learned_cross["bias"].numel()
        small = F.pad(token, (0, max(0, width - token.shape[-1])))[..., :width]
        delta = cross(small, small, s.learned_cross) - small
        delta = F.pad(delta, (0, max(0, token.shape[-1] - width)))[..., : token.shape[-1]]
    else:
        delta = cross(token, token, s.learned_cross) - token
    rows = derived_keys(s)[1] if state else s.rows_in
    term = cross_energy(delta, token, keys, rows)
    return logits + (s.tau if scale is None else scale) * term


def _route(qs, sets, keys, prev, shared, noise_scale, regions=None):
    if (
        sets
        and (not shared or len(sets) == 1)
        and noise_scale == 0
        and all(q.ndim >= 2 for q in qs)
        and all(
            s.routing_scope == "tensor"
            and 0 < s.candidates < kin.shape[0]
            and not any(s.cross_weights)
            and s.learned_cross is None
            for s, (kin, _) in zip(sets, keys, strict=True)
        )
    ):
        result = []
        for i, (s, q, (kin, kout)) in enumerate(zip(sets, qs, keys, strict=True)):
            check_routing(s.routing_scope, s.routing_select)
            v = q if prev is None else out_project(q, kout, prev[i])
            region = None if regions is None else regions[i]
            indices = _candidate_indices(s, v, kin, region).sort(-1).values
            selected = kin[indices.squeeze(-2)]
            proj = torch.einsum("...tr,...kmr->...tkm", v, selected)
            rows = key_rows(kin, s.rows_in)[indices]
            scores = proj.pow(2).sum(-1) / (v.pow(2).sum(-1, keepdim=True) + SCORE_EPS)
            scores = s.tau * (scores / (rows / kin.shape[-1]))
            mean = region_mean(scores, region)
            gate = token_gate(mean, 0.0, s.routing_select)
            full = torch.zeros(*gate.shape[:-1], kin.shape[0], dtype=gate.dtype, device=gate.device)
            result.append(full.scatter(-1, indices, gate).expand(*v.shape[:-1], kin.shape[0]))
        return result
    logits = []
    candidates = []
    for i, (s, q, (kin, kout)) in enumerate(zip(sets, qs, keys, strict=True)):
        v = q if prev is None else out_project(q, kout, prev[i])
        if (
            s.direction_core is not None
            and s.fine
            and s.routing_scope == "token"
            and not shared
            and s.direction_candidates < kin.shape[0]
            and s.candidates <= 0
        ):
            from .library.nodes import node_candidates

            indices = node_candidates(v, kin[:, 0, :], s.direction_candidates)
            selected = kin[indices]
            proj = torch.einsum("...r,...kmr->...km", v.float(), selected)
            rows = key_rows(kin, s.rows_in)[indices]
            scores = proj.square().sum(-1) / (v.float().square().sum(-1, keepdim=True) + SCORE_EPS)
            scores = s.tau * scores / (rows / kin.shape[-1])
            score = torch.full(
                (*v.shape[:-1], kin.shape[0]), -1e9, dtype=scores.dtype, device=v.device
            ).scatter(-1, indices, scores)
        else:
            score = s.tau * norm_energy(v, kin, s.rows_in)
        score = _cross_logits(score, s, kin)
        logits.append(_learned_cross_logits(score, s, q, kin))
        region = None if regions is None else regions[i]
        candidates.append(_candidate_mask(s, v, kin, region))
    if all(mask is None for mask in candidates):
        return _gates_from_logits(logits, sets, shared, noise_scale, regions)
    return _gates_from_logits(logits, sets, shared, noise_scale, regions, candidates)


def _gates(
    q: torch.Tensor,
    sets: Sequence[LayerSet],
    shared: bool,
    noise_scale: float,
    prev: list[torch.Tensor] | None = None,
    keys: list | None = None,
    regions: list | None = None,
) -> list[torch.Tensor]:
    if _derived_state_sets(sets):
        regions = regions if regions is not None else [None] * len(sets)
        return _mixed_route(q, sets, None, prev, shared, noise_scale, regions, True, x=q)
    keys = keys if keys is not None else [_set_keys(s) for s in sets]
    qs = [F.linear(q.float(), s.router.float()) for s in sets]
    return _route(qs, sets, keys, prev, shared, noise_scale, regions)


def _gates_from_logits(
    logits: list[torch.Tensor],
    sets: Sequence[LayerSet],
    shared: bool,
    noise_scale: float,
    regions: list | None = None,
    candidates: list | None = None,
) -> list[torch.Tensor]:
    regions = regions if regions is not None else [None] * len(sets)
    candidates = candidates if candidates is not None else [None] * len(sets)
    has_region = any(r is not None for r in regions)
    if shared and len(sets) > 1 and has_region and sets[0].routing_scope == "tensor":
        means = []
        for lg, r, s in zip(logits, regions, sets, strict=True):
            mean = region_mean(lg, r) + s.log_prior
            if r is not None:
                empty = r.sum(-2, keepdim=True) <= 0
                mean = mean.masked_fill(empty, -1e9)
            means.append(mean)
        means = [
            m if c is None else m.masked_fill(~c, -torch.inf)
            for m, c in zip(means, candidates, strict=True)
        ]
        joint = torch.cat(means, -1)
        if noise_scale > 0:
            joint = joint + gumbel(joint, noise_scale)
        gate = _select(joint, sets[0].routing_select)
        sizes = [lg.shape[-1] for lg in logits]
        return [
            g.expand_as(lg) for g, lg in zip(torch.split(gate, sizes, dim=-1), logits, strict=True)
        ]
    if shared and len(sets) > 1:
        parts = [lg + s.log_prior for lg, s in zip(logits, sets, strict=True)]
        parts = [
            p if c is None else p.masked_fill(~c, -torch.inf)
            for p, c in zip(parts, candidates, strict=True)
        ]
        joint = torch.cat(parts, -1)
        gate = _gate(joint, sets[0], noise_scale)
        sizes = [lg.shape[-1] for lg in logits]
        return list(torch.split(gate, sizes, dim=-1))
    return [
        _gate(lg, s, noise_scale, r, c)
        for lg, s, r, c in zip(logits, sets, regions, candidates, strict=True)
    ]


def _read_map_summary(s: LayerSet) -> torch.Tensor:
    from .library.content import _layer_rows, _svd_rows

    if s.down is None or s.up is None:
        raise ValueError("read_map candidate summary requires expert read and write maps")
    key = _tensor_key(
        f"candidate-read-{s.form}-{s.per_expert}",
        [s.down, s.up, s.router if s.form != "state" else None],
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        downs = s.down.detach().float().reshape(-1, s.per_expert, s.down.shape[-1])
        ups = s.up.detach().float().reshape(-1, s.per_expert, s.up.shape[-1])
        result = []
        for down, up in zip(downs, ups, strict=True):
            if s.form not in ("input", "state"):
                down = down @ s.router.float().T
            weighted = _layer_rows({"down": down, "up": up.T}, 4)
            directions, values = _svd_rows(weighted, 4)
            directions = directions * (values > 0)[:, None]
            result.append(directions)
        summary = torch.stack(result)
    _cache_put(key, summary)
    return summary


def _candidate_indices(s: LayerSet, query: torch.Tensor, summary: torch.Tensor, region=None):
    if s.candidates <= 0 or s.candidates >= summary.shape[0]:
        return None
    from .candidates import top_candidates

    if s.candidate_summary == "read_map":
        summary, rows = _read_map_summary(s), None
    elif s.candidate_summary == "keys":
        rows = (
            derived_keys(s)[1]
            if s.form == "state" and s.down is not None
            else s.rows_in
            if s.form != "input"
            else None
        )
    else:
        raise ValueError(f"unknown candidate_summary {s.candidate_summary!r}")
    query = query.float()
    if s.routing_scope == "tensor":
        query = region_mean(query, region)
    return top_candidates(summary.float(), query, s.candidates, rows)


def _candidate_mask(s: LayerSet, query: torch.Tensor, summary: torch.Tensor, region=None):
    indices = _candidate_indices(s, query, summary, region)
    if indices is None:
        return None
    mask = torch.zeros(*indices.shape[:-1], summary.shape[0], dtype=torch.bool, device=query.device)
    return mask.scatter(-1, indices, True)


def _gate(
    logits: torch.Tensor,
    s: LayerSet,
    noise_scale: float,
    region: torch.Tensor | None = None,
    candidates: torch.Tensor | None = None,
) -> torch.Tensor:
    check_routing(s.routing_scope, s.routing_select)
    if candidates is None:
        if s.routing_scope == "tensor":
            return tensor_gate(logits, noise_scale, region, s.routing_select)
        return token_gate(logits, noise_scale, s.routing_select)
    if s.routing_scope == "tensor":
        mean = region_mean(logits, region)
        return token_gate(
            mean.masked_fill(~candidates, -torch.inf), noise_scale, s.routing_select
        ).expand_as(logits)
    return token_gate(logits.masked_fill(~candidates, -torch.inf), noise_scale, s.routing_select)


_IR_LAYER_PARTS: ContextVar[bool] = ContextVar("kilora_ir_layer_parts", default=False)


_LAYER_TIMERS = ContextStack("kilora_layer_timers")
_LAYER_DEPTH: ContextVar[int] = ContextVar("kilora_layer_depth", default=0)


@contextlib.contextmanager
def timing_layers() -> Iterator[dict[str, float]]:

    timer = {"layers_s": 0.0, "hooks_s": 0.0, "layer_calls": 0}
    with _LAYER_TIMERS.pushed(timer):
        yield timer


def _sync(t: Any) -> None:
    if isinstance(t, torch.Tensor) and t.is_cuda:
        torch.cuda.synchronize(t.device)


def run_layer(*args, **kwargs):
    _preempt_check()
    _ablation.read_dropout_off()
    _ablation.carry_steps(False)
    coherence = _ablation.gate_coherence(True)
    if not coherence:
        sets = args[2] if len(args) > 2 else kwargs["sets"]
        sets = [dataclasses.replace(s, gate_coherence=False) for s in sets]
        if len(args) > 2:
            args = (*args[:2], sets, *args[3:])
        else:
            kwargs = {**kwargs, "sets": sets}
    base = args[1] if len(args) > 1 else kwargs["y0"]
    with _ablation.recording(kwargs.get("target", ""), base):
        return _timed_run_layer(*args, **kwargs)


def _timed_run_layer(*args, **kwargs):
    timers = _LAYER_TIMERS.get()
    if not timers or _LAYER_DEPTH.get():
        return _run_layer_overrides(*args, **kwargs)
    import time

    y0 = args[1] if len(args) > 1 else kwargs.get("y0")
    _sync(y0)
    start = time.perf_counter()
    depth = _LAYER_DEPTH.set(1)
    try:
        out = _run_layer_overrides(*args, **kwargs)
        _sync(out if isinstance(out, torch.Tensor) else y0)
    finally:
        _LAYER_DEPTH.reset(depth)
        spent = time.perf_counter() - start
        for timer in _LAYER_TIMERS:
            timer["layers_s"] += spent
            timer["layer_calls"] += 1
    return out


def _run_layer_overrides(*args, **kwargs):
    if not _IR_ACTIVE_OVERRIDES.get():
        sets = args[2] if len(args) > 2 else kwargs["sets"]
        if any(s.ir_parts for s in sets):
            from .ir.dispatch import override_scope

            with override_scope():
                return _run_layer_overrides(*args, **kwargs)
    if _IR_ACTIVE_OVERRIDES.get() and not _IR_LAYER_PARTS.get():
        from .ir.dispatch import active_overrides, interpreted_part, same_override

        sets = args[2] if len(args) > 2 else kwargs["sets"]
        chosen = {}
        for part in ("budget", "freeze"):
            values = [setting.ir_parts.get(part) for setting in sets]
            if values and any(not same_override(values[0], value) for value in values[1:]):
                raise NotImplementedError(f"layer cannot use different {part} overrides")
            if values and values[0] is not None:
                chosen[part] = values[0]
        if chosen:
            from types import SimpleNamespace

            interpreted_part([SimpleNamespace(ir_parts=chosen)])
            token = _IR_LAYER_PARTS.set(True)
            try:
                with active_overrides(chosen):
                    return _run_layer_dispatch(*args, **kwargs)
            finally:
                _IR_LAYER_PARTS.reset(token)
    return _run_layer_dispatch(*args, **kwargs)


def _run_layer_dispatch(*args, **kwargs):
    sets = args[2] if len(args) > 2 else kwargs["sets"]
    if any(s.exact for s in sets) and (_IR_ACTIVE_OVERRIDES.get() or _IR_INTERPRET_ALL.get()):
        if _IR_ACTIVE_OVERRIDES.get():
            raise NotImplementedError("exact routed changes do not support IR overrides")
        token = _IR_INTERPRET_ALL.set(False)
        try:
            return _run_layer_dispatch(*args, **kwargs)
        finally:
            _IR_INTERPRET_ALL.reset(token)
    if _IR_ACTIVE_OVERRIDES.get() or _IR_INTERPRET_ALL.get():
        from .ir.dispatch import active_overrides, interpreted_part, interpreting_all, same_override

        sets = args[2] if len(args) > 2 else kwargs["sets"]
        joint_select = kwargs.get("shared", False) or (
            _derived_state_sets(sets) and any(s.form == "input" for s in sets)
        )
        if joint_select and len(sets) > 1:
            selections = [s.ir_parts.get("select") for s in sets]
            if any(not same_override(selection, selections[0]) for selection in selections[1:]):
                raise NotImplementedError("shared routing cannot use different select overrides")
        external = ("budget", "freeze", "guard", "live_write")
        part = interpreted_part(sets, exclude=external)
        if part is None:
            part = next(
                (name for setting in sets for name in setting.ir_parts if name not in external),
                None,
            )
        if part is not None:
            from .fused import _note
            from .ir.driver import run_layer_ir

            overrides = sets[0].ir_parts if sets else {}
            try:
                with active_overrides(
                    overrides,
                    sets,
                    shared=joint_select,
                    exclude=("budget", "freeze", "guard", "live_write"),
                ):
                    result = run_layer_ir(*args, **kwargs)
            except NotImplementedError as exc:
                if not interpreting_all() or any(s.ir_parts for s in sets):
                    raise
                token = _IR_INTERPRET_ALL.set(False)
                fallback = _IR_FALLBACK.set(True)
                try:
                    result = run_layer(*args, **kwargs)
                finally:
                    _IR_FALLBACK.reset(fallback)
                    _IR_INTERPRET_ALL.reset(token)
                _note(f"IR driver refused layer: {exc}")
                return result
            else:
                native = []
                if any(
                    s.iteration == "path" and learned_kind(s) not in ("none", "separable")
                    for s in sets
                ):
                    native.append("path learned output gate")
                tensors = [args[1] if len(args) > 1 else kwargs.get("y0")]
                tensors.extend(
                    tensor
                    for setting in sets
                    for tensor in (
                        setting.down,
                        setting.up,
                        setting.router,
                        setting.keys_in,
                        setting.keys_out,
                        setting.scale,
                        setting.tau,
                        *(setting.gate or {}).values(),
                    )
                )
                if torch.is_grad_enabled() and any(
                    isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors
                ):
                    native.append("gradient score and keep_probability")
                if native:
                    _note("interpreted layer retains native " + ", ".join(native))
                else:
                    _note(f"interpreted part {part}")
                return result
    global _missing_feature_input_noted
    sets = args[2] if len(args) > 2 else kwargs["sets"]
    learned = any(s.learned_cross is not None for s in sets)
    if not _SET_SHARE_SCOPES and not routing_features_enabled() and not learned:
        return _run_layer(*args, **kwargs)
    target = kwargs.get("target", "")
    x = args[0] if args else kwargs["x"]
    y0 = args[1] if len(args) > 1 else kwargs["y0"]
    with contextlib.ExitStack() as stack:
        if _SET_SHARE_SCOPES:
            stack.enter_context(layer_set_share_observation(target, sets))
        if routing_features_enabled() or learned:
            if x is None and not _missing_feature_input_noted:
                logging.getLogger("kilora").warning(
                    "Routing features are unavailable when the layer input is missing."
                )
                _missing_feature_input_noted = True
            elif x is not None and x.ndim <= 3:
                stack.enter_context(layer_features(x, y0, sets, kwargs.get("regions"), target))
        return _run_layer(*args, **kwargs)


def _run_layer(
    x: torch.Tensor,
    y0: torch.Tensor,
    sets: Sequence[LayerSet],
    depth: int,
    *,
    shared: bool = False,
    noise_scale: float = 0.0,
    query_policy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    grad_steps: Callable[[int], bool] | None = None,
    target: str = "",
    regions: Sequence[torch.Tensor | None] | None = None,
    schedule: Any = None,
    expert_regions: Sequence[Any] | None = None,
    index: int = 0,
    count: int = 1,
    kind: str = "other",
    t: float | None = None,
    token_depth: torch.Tensor | None = None,
    expert_scales: Sequence[Any] | None = None,
    _apply_alphas: bool = True,
) -> torch.Tensor:

    if not sets:
        return y0
    from .call_values import (
        ComposedAlpha,
        active_axes,
        axis_alphas,
        contiguous_row_slices,
        current_active_axes,
        current_axis_alphas,
        current_ip_early,
        current_ip_mounted,
        current_row_axes,
        route_rows,
        row_axes,
    )

    rows = current_row_axes()
    if rows is not None:
        if len(rows) != y0.shape[0]:
            raise ValueError("layer row count does not match model batch")
        values = rows
        if any(v < -1 or v >= len(AXES) for v in values):
            raise ValueError("invalid layer row selection")
        batch = y0.shape[0]
        slices = contiguous_row_slices(values)

        def subset(value, indices, row_slice):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 1
                and value.shape[0] > 1
                and batch % value.shape[0] == 0
            ):
                if value.shape[0] != batch:
                    value = value.repeat((batch // value.shape[0],) + (1,) * (value.ndim - 1))
                if row_slice is not None:
                    return value[row_slice]
                return value.index_select(0, indices.to(value.device))
            if isinstance(value, (list, tuple)):
                return type(value)(subset(v, indices, row_slice) for v in value)
            return value

        result = y0
        parts = []
        for value in dict.fromkeys(values):
            indices = torch.tensor(
                [i for i, v in enumerate(values) if v == value], device=y0.device
            )
            row_slice = None if slices is None else slices[value]

            def select(tensor, row_slice=row_slice, indices=indices):
                return (
                    tensor[row_slice] if row_slice is not None else tensor.index_select(0, indices)
                )

            names = (
                None
                if value == -1
                else (("creative", "functional") if AXES[value] == "creative" else (AXES[value],))
            )
            selected = active_axes(names) if names is not None else contextlib.nullcontext()
            sub_sets = [
                dataclasses.replace(
                    s,
                    region=subset(s.region, indices, row_slice),
                    expert_mask=subset(s.expert_mask, indices, row_slice),
                    expert_inputs=subset(s.expert_inputs, indices, row_slice),
                    expert_offsets=subset(s.expert_offsets, indices, row_slice),
                )
                for s in sets
            ]
            sub_alphas = {
                k: subset(v, indices, row_slice) for k, v in current_axis_alphas().items()
            }
            freeze_cache = (
                _route_freeze.freezing(
                    _route_freeze._CACHES[-1].setdefault(("kilora_rows", tuple(values), value), {})
                )
                if _route_freeze._CACHES
                else contextlib.nullcontext()
            )
            carry_cache = (
                step_carry(_STEP_CARRY[-1].setdefault(("kilora_rows", tuple(values), value), {}))
                if _STEP_CARRY
                else contextlib.nullcontext()
            )
            from .call_values import crop_window, current_crop_window

            boxes = current_crop_window()
            selected_boxes = None if boxes is None else [boxes[int(i)] for i in indices]
            with (
                row_axes(None),
                route_rows(indices),
                selected,
                axis_alphas(sub_alphas),
                freeze_cache,
                carry_cache,
                crop_window(selected_boxes),
            ):
                out = run_layer(
                    None if x is None else select(x),
                    select(y0),
                    sub_sets,
                    depth,
                    shared=shared,
                    noise_scale=noise_scale,
                    query_policy=query_policy,
                    grad_steps=grad_steps,
                    target=target,
                    regions=subset(regions, indices, row_slice),
                    schedule=schedule,
                    expert_regions=subset(expert_regions, indices, row_slice),
                    index=index,
                    count=count,
                    kind=kind,
                    t=t,
                    token_depth=subset(token_depth, indices, row_slice),
                    expert_scales=expert_scales,
                    _apply_alphas=_apply_alphas,
                )
            if slices is None:
                result = result.index_copy(0, indices, out)
            else:
                parts.append(out)
        return result if slices is None or not parts else torch.cat(parts, dim=0)

    active = current_active_axes()
    mounted = current_ip_mounted()
    early = (
        _apply_alphas
        and current_ip_early()
        and (active is None or "ip" in active)
        and (mounted if mounted is not None else any(s.axis == "ip" for s in sets))
    )
    if any(s.axis == "shared" for s in sets):
        alphas = current_axis_alphas() if _apply_alphas else {}
        axes = list(dict.fromkeys(s.axis for s in sets if s.axis != "shared"))
        if not axes:
            axes = list(alphas) if alphas else ["creative"]
        if active is not None:
            axes = [a for a in axes if a in active] or sorted(active)
        weights = {a: _axis_alpha(alphas.get(a, 1.0), y0, target) for a in axes}
        if early:
            weights = {
                a: (w if a in UNBUDGETED_AXES else float(a == "ip")) for a, w in weights.items()
            }
        y = y0
        for axis, weight in axis_budget(weights).items():
            if early and axis != "ip" and axis not in UNBUDGETED_AXES:
                continue
            indices = [i for i, s in enumerate(sets) if s.axis in (axis, "shared")]
            sub = [dataclasses.replace(sets[i], axis=axis, axis_joint=True) for i in indices]
            own = next((s.fixed_depth for s in sets if s.axis == axis), None)
            sub_regions = None if regions is None else [regions[i] for i in indices]
            if isinstance(alphas.get(axis), ComposedAlpha):
                sub_regions = _composed_regions(sub, sub_regions, weight, y0)
            out = run_layer(
                x,
                y0,
                sub,
                depth if own is None else own,
                shared=True,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=axis_target(target, axis),
                regions=sub_regions,
                expert_regions=None
                if expert_regions is None
                else [expert_regions[i] for i in indices],
                expert_scales=None
                if expert_scales is None
                else [expert_scales[i] for i in indices],
                schedule=schedule if own is None else None,
                index=index,
                count=count,
                kind=kind,
                t=t,
                token_depth=token_depth if own is None else None,
                _apply_alphas=False,
            )
            y = y + (out - y0) * weight
        return y
    if active is not None and not any(s.axis in active for s in sets):
        return y0
    alphas = current_axis_alphas() if _apply_alphas else {}
    if early or (alphas and any(s.axis in alphas for s in sets)):
        outs, weights = {}, {}
        for axis in dict.fromkeys(s.axis for s in sets):
            if active is not None and axis not in active:
                continue
            if early and axis != "ip" and axis not in UNBUDGETED_AXES:
                continue
            indices = [i for i, s in enumerate(sets) if s.axis == axis]
            sub = [sets[i] for i in indices]
            own = sub[0].fixed_depth if len({s.axis for s in sets}) > 1 else None
            weights[axis] = (
                1.0 if early and axis == "ip" else _axis_alpha(alphas.get(axis, 1.0), y0, target)
            )
            sub_regions = None if regions is None else [regions[i] for i in indices]
            if isinstance(alphas.get(axis), ComposedAlpha):
                sub_regions = _composed_regions(sub, sub_regions, weights[axis], y0)
            out = run_layer(
                x,
                y0,
                sub,
                depth if own is None else own,
                shared=shared,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=axis_target(target, axis) if len({s.axis for s in sets}) > 1 else target,
                regions=sub_regions,
                schedule=schedule if own is None else None,
                expert_regions=None
                if expert_regions is None
                else [expert_regions[i] for i in indices],
                expert_scales=None
                if expert_scales is None
                else [expert_scales[i] for i in indices],
                index=index,
                count=count,
                kind=kind,
                t=t,
                token_depth=token_depth if own is None else None,
                _apply_alphas=False,
            )
            outs[axis] = out
        y = y0
        for axis, w in axis_budget(weights).items():
            y = y + (outs[axis] - y0) * w
        return y
    if y0.dim() > 3:
        n = y0.dim()
        y = run_layer(
            None if x is None else x.flatten(1, -2),
            y0.flatten(1, -2),
            [
                s
                if (s.region is None and s.expert_inputs is None and s.expert_offsets is None)
                else dataclasses.replace(
                    s,
                    region=_flat_tokens(s.region, n),
                    expert_inputs=None
                    if s.expert_inputs is None
                    else tuple(None if v is None else v.flatten(1, -2) for v in s.expert_inputs),
                    expert_offsets=None
                    if s.expert_offsets is None
                    else tuple(None if v is None else v.flatten(1, -2) for v in s.expert_offsets),
                )
                for s in sets
            ],
            depth,
            shared=shared,
            noise_scale=noise_scale,
            query_policy=query_policy,
            grad_steps=grad_steps,
            target=target,
            regions=None if regions is None else [_flat_tokens(r, n) for r in regions],
            schedule=schedule,
            expert_regions=None
            if expert_regions is None
            else [None if m is None else [_flat_tokens(r, n) for r in m] for m in expert_regions],
            index=index,
            count=count,
            kind=kind,
            t=t,
            token_depth=_flat_tokens(token_depth, n),
            _apply_alphas=_apply_alphas,
        )
        return y.reshape(y0.shape)
    if expert_regions is not None and any(expert_regions):
        sets = _expert_masks(sets, expert_regions, y0)
    if expert_scales is not None and any(v is not None for v in expert_scales):
        sets = _expert_scaled(sets, expert_scales, y0)
    if regions is not None and len(regions) != len(sets):
        raise ValueError(f"{len(regions)} regions for {len(sets)} sets")
    if any(s.entry_windows for s in sets):
        from .entry_window import active_layer_sets

        sets, positions = active_layer_sets(sets, current_progress() if t is None else float(t))
        if not sets:
            return y0
        if regions is not None:
            regions = [regions[i] for i in positions]
    axes = list(dict.fromkeys(s.axis for s in sets))
    if len(axes) > 1:
        y = y0
        shares = axis_budget({a: 1.0 for a in axes if active is None or a in active})
        for axis in axes:
            if active is not None and axis not in active:
                continue
            idx = [i for i, s in enumerate(sets) if s.axis == axis]
            sub = [sets[i] for i in idx]
            own = sub[0].fixed_depth
            out = run_layer(
                x,
                y0,
                sub,
                depth if own is None else own,
                shared=shared,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=axis_target(target, axis),
                regions=None if regions is None else [regions[i] for i in idx],
                schedule=schedule if own is None else None,
                index=index,
                count=count,
                kind=kind,
                t=t,
                token_depth=token_depth if own is None else None,
            )
            y = y + (out - y0) * shares[axis]
        return y
    plain = [i for i, s in enumerate(sets) if s.apply_once and s.down is not None]
    if plain:
        regs = [
            _token_region(regions[i] if regions is not None else s.region, y0)
            for i, s in enumerate(sets)
        ]
        out = y0
        for i in plain:
            once = _run_input(x, y0, sets, [i], 1, regs, target)
            out = out + (once - y0)
        rest = [i for i in range(len(sets)) if i not in plain]
        if rest:
            probe_starts = [(record, len(record.get(target, []))) for record in _PROBES]
            other = run_layer(
                x,
                y0,
                [sets[i] for i in rest],
                depth,
                shared=shared,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=target,
                regions=None if regions is None else [regions[i] for i in rest],
                schedule=schedule,
                index=index,
                count=count,
                kind=kind,
                t=t,
                token_depth=token_depth,
                _apply_alphas=False,
            )
            mapped = set()
            for record, start in probe_starts:
                for entry in record.get(target, [])[start:]:
                    if id(entry) in mapped:
                        continue
                    mapped.add(id(entry))
                    for key in ("queries", "gates", "prev", "masks", "reads"):
                        if entry.get(key) is not None:
                            entry[key] = _at_places(entry[key], (rest, len(sets)))
                    if entry["k"] == "output":
                        entry["_outer_places"] = (rest, len(sets))
            out = out + (other - y0)
        return out
    if (
        sets
        and sets[0].fixed_depth is not None
        and all(s.fixed_depth == sets[0].fixed_depth for s in sets)
    ):
        depth = sets[0].fixed_depth
        schedule = None
        token_depth = None
    regs = [
        _token_region(regions[i] if regions is not None else s.region, y0)
        for i, s in enumerate(sets)
    ]
    auto = False
    if schedule is not None:
        key = next((s.key for s in sets if s.down is not None and s.form == "state"), sets[0].key)
        progress = current_progress() if t is None else float(t)
        depth = float(schedule.depth(index, count, kind, progress, key))
        if depth <= 0:
            return y0
        if getattr(schedule, "is_auto", False):
            depth = min(depth, AUTO_DEPTH_CAP)
            auto = True
    depth = _whole(depth)
    td = None
    if token_depth is not None:
        values = _token_values(token_depth, y0)
        low, high = _tensor_metadata(
            token_depth,
            "depth_bounds",
            lambda: torch.stack((values.min(), values.max())).tolist(),
        )
        if low == high:
            depth = _whole(low)
        else:
            td = _TokenDepth(values, high)
            depth = td.steps
        if depth <= 0:
            return y0
    if any(s.direction_core is not None for s in sets):
        limit = 64 if all(s.direction_core is not None and s.fine for s in sets) else 16
        if depth > limit:
            raise ValueError(f"depth exceeds the limit of {limit} for these sets")
    keyed_pos = [i for i, s in enumerate(sets) if s.down is not None]
    keyed = [sets[i] for i in keyed_pos]
    mixed_derived = _derived_state_sets(keyed)
    freeze = (
        depth > 0
        and not any(
            s.axis_joint
            or any(s.cross_weights)
            or (s.learned_cross is not None and not s.learned_cross_zero)
            for s in sets
        )
        and not any(s.expert_inputs is not None or s.expert_offsets is not None for s in sets)
        and td is None
        and not any(0 < s.candidates < s.keys_in.shape[0] for s in keyed)
        and not (
            derived_mode() == "full"
            and any(import_score(s) == "derived" for s in keyed if s.form == "input")
        )
        and _route_freeze.active(keyed, current_progress() if t is None else float(t))
    )
    if freeze and not torch.is_grad_enabled() and query_policy is None and noise_scale == 0:

        def call():
            return run_layer(
                x,
                y0,
                sets,
                depth,
                shared=shared,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=target,
                regions=regions,
                index=index,
                count=count,
                kind=kind,
                t=t,
                _apply_alphas=_apply_alphas,
            )

        kregs = [regs[i] for i in keyed_pos]
        if not _ablation.force_loop() and not (_STEP_CARRY and _ablation.carry_steps(False)):
            return _route_freeze.run(call, x, y0, keyed, keyed_pos, target, kregs, depth)
    once = [i for i, s in enumerate(sets) if s.form == "input" and s.down is not None]
    state = [i for i, s in enumerate(sets) if s.down is not None and s.form == "state"]
    if any(sets[i].direction_core is not None for i in once) and state:
        raise ValueError("direction sets cannot yet run with state-form sets at the same mount")
    if any(sets[i].direction_core is not None and sets[i].path_width > 1 for i in once):
        raise ValueError("direction sets do not support multiple path walks")
    joint = bool(once and state) and depth > 0
    if mixed_derived and depth > 0:
        if any(
            s.path_width > 1 and s.gate_head_limit is not None and uses_adaptive(s)
            for s in keyed
            if s.iteration == "path"
        ):
            raise ValueError("gate_history head:k is not supported with multiple path walks")
        if td is not None and any(s.iteration == "path" for s in keyed):
            raise ValueError("a per-token depth needs accumulate iteration (path mode: later)")
        if keyed[0].iteration == "path" and keyed[0].path_width > 1:
            from .path_walks import run_wide

            def walk(n):
                return run_wide(
                    y0,
                    keyed,
                    n,
                    shared,
                    noise_scale,
                    grad_steps,
                    target,
                    [regs[i] for i in keyed_pos],
                    None,
                    x,
                )

            low = math.floor(depth)
            high = walk(math.ceil(depth))
            if low == depth:
                return high
            before = y0 if low == 0 else walk(low)
            return before + (depth - low) * (high - before)
        fused = _fused_module()
        if (
            fused is not None
            and not _EAGER
            and not auto
            and _fast_path_ok(
                y0,
                keyed,
                noise_scale,
                query_policy,
                grad_steps,
                allow_depth_split=len(keyed) > 1,
            )
            and fused.fused_ok(y0, keyed, depth, td=td)
        ):
            return fused.run_fused(
                y0,
                keyed,
                depth,
                shared,
                target,
                [regs[i] for i in keyed_pos],
                x=x,
                td=td,
                positions=keyed_pos,
            )
        if keyed[0].iteration == "path":
            return _run_input(
                x,
                y0,
                sets,
                keyed_pos,
                math.ceil(depth) if auto else depth,
                regs,
                target,
                td,
                grad_steps=grad_steps,
                noise_scale=noise_scale,
                query_policy=query_policy,
                shared=shared,
                stop=True if auto else None,
            )
        if _PROBES or len(keyed) != len(sets):
            return _run_input(
                x,
                y0,
                sets,
                keyed_pos,
                depth,
                regs,
                target,
                td,
                grad_steps=grad_steps,
                noise_scale=noise_scale,
                query_policy=query_policy,
                shared=shared,
            )
        if td is None and _fast_path_ok(y0, keyed, noise_scale, query_policy, grad_steps):
            return _run_lowrank(y0, keyed, depth, shared, target, [regs[i] for i in keyed_pos], x)
        return _run_loop(
            y0,
            keyed,
            depth,
            shared,
            noise_scale,
            query_policy,
            grad_steps,
            target,
            [regs[i] for i in keyed_pos],
            x=x,
            td=td,
        )
    if td is not None and any(sets[i].iteration == "path" for i in once + state):
        raise ValueError("a per-token depth needs accumulate iteration (path mode: later)")
    if once and not joint and depth > 0:
        ins = [sets[i] for i in once]
        if any(s.axis_joint for s in ins) or (
            (depth > 1 or td is not None) and any(depth_split_on(s) for s in ins)
        ):
            y0 = _run_input(x, y0, sets, once, depth, regs, target, td)
        elif any(import_score(s) == "derived" for s in ins) and derived_mode() == "full":
            y0 = _run_derived_walks(
                x,
                y0,
                sets,
                once,
                depth,
                regs,
                target,
                td,
                index,
                query_policy is None and noise_scale == 0,
                current_progress() if t is None else float(t),
            )
        elif any(import_score(s) == "derived" for s in ins) or td is not None:
            y0 = _run_input_fused(x, y0, sets, once, depth, regs, target, td)
        elif depth >= 1 and ins[0].iteration == "path" and ins[0].path_width > 1:
            y0 = _run_input_walks(x, y0, ins, depth, [regs[i] for i in once], target)
        else:
            y0 = _run_input_fused(x, y0, sets, once, depth, regs, target)
    if depth <= 0:
        return y0
    keep = sorted(state + once) if joint else state
    inputs = x if joint else None
    if not keep:
        return y0
    lora = [sets[i] for i in keep]
    lregs = [regs[i] for i in keep]
    if sets[state[0]].iteration == "path":
        if auto:
            depth = math.ceil(depth)
            return _run_path(
                y0, lora, depth, shared, noise_scale, grad_steps, target, lregs, True, inputs
            )
        if isinstance(depth, int):
            return _run_path(
                y0, lora, depth, shared, noise_scale, grad_steps, target, lregs, x=inputs
            )
        low = math.floor(depth)
        hi = _run_path(y0, lora, low + 1, shared, noise_scale, grad_steps, target, lregs, x=inputs)
        lo = (
            y0
            if low == 0
            else _run_path(y0, lora, low, shared, noise_scale, grad_steps, target, lregs, x=inputs)
        )
        return lo + (depth - low) * (hi - lo)
    if any(s.learned_cross is not None and not s.learned_cross_zero for s in sets):
        fused = _fused_module()
        if fused is not None and not _EAGER:
            fused.fused_ok(y0, sets, td=td)
    if _fast_path_ok(y0, sets, noise_scale, query_policy, grad_steps):
        fused = _fused_module()
        if (
            fused is not None
            and not _READ_KEEP_DETAIL_OBSERVERS
            and fused.fused_ok(y0, sets, td=td)
        ):
            try:
                return fused.run_fused(y0, sets, depth, shared, target, lregs, td=td)
            except Exception as err:
                if not fused.resource_error(err):
                    raise
                fused.mark_too_big(y0, sets)
        if td is None:
            return _run_lowrank(y0, sets, depth, shared, target, lregs, inputs)
    places = (keep, len(sets))
    return _run_loop(
        y0,
        sets,
        depth,
        shared,
        noise_scale,
        query_policy,
        grad_steps,
        target,
        lregs,
        places,
        inputs,
        td,
    )


def _add_expert_offsets(y, sets, gates):
    for s, gate in zip(sets, gates, strict=True):
        if s.expert_offsets is None:
            continue
        if len(s.expert_offsets) != gate.shape[-1]:
            raise ValueError("expert offset count does not match routing weights")
        for i, offset in enumerate(s.expert_offsets):
            if offset is not None:
                y = y + offset.to(y.dtype) * gate[..., i : i + 1].to(y.dtype)
    return y


run_layer.__wrapped__ = _run_layer
run_layer.__doc__ = _run_layer.__doc__
run_layer.__annotations__ = _run_layer.__annotations__


def _input_read(s: LayerSet, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if s.expert_inputs is None:
        a = F.linear(x.to(dtype), _as(s.down, dtype))
    else:
        inputs = s.expert_inputs
        if len(inputs) * s.per_expert != s.down.shape[0]:
            raise ValueError("expert input count does not match the input-form set")
        weights = _as(s.down, dtype).split(s.per_expert)
        a = torch.cat(
            [
                F.linear((x if value is None else value).to(dtype), weight)
                for value, weight in zip(inputs, weights, strict=True)
            ],
            dim=-1,
        )
    return a if s.expert_mask is None else a * s.expert_mask.to(a.dtype)


def _state_read(s, value, dtype):
    read = F.linear(value.to(dtype), _as(s.down, dtype))
    return read if s.expert_mask is None else read * s.expert_mask.to(read.dtype)


def _expert_masks(sets, expert_regions, y0):
    out = list(sets)
    for i, (s, masks) in enumerate(zip(sets, expert_regions, strict=True)):
        if not masks or s.form != "input" or s.down is None or all(m is None for m in masks):
            continue
        ref = next(m for m in masks if m is not None)
        ones = torch.ones_like(ref, dtype=torch.float32)
        cols = [
            (ones if m is None else m.float()).expand(*ref.shape[:-1], s.per_expert) for m in masks
        ]
        mask = torch.cat(cols, -1)
        if mask.shape[-1] != s.down.shape[0]:
            raise ValueError(f"{len(masks)} expert regions for {s.down.shape[0]} columns")
        out[i] = dataclasses.replace(s, expert_mask=mask.to(y0.device))
    return out


def _expert_scaled(sets, expert_scales, y0):

    if len(expert_scales) != len(sets):
        raise ValueError(f"{len(expert_scales)} expert scale lists for {len(sets)} sets")
    out = list(sets)
    for i, (s, v) in enumerate(zip(sets, expert_scales, strict=True)):
        if v is None:
            continue
        if s.form != "input" or s.down is None:
            raise ValueError("expert scales need an imported (input-form) set")
        v = torch.as_tensor(v, dtype=torch.float32).flatten()
        experts = s.down.shape[0] // s.per_expert
        if v.numel() != experts:
            raise ValueError(f"{v.numel()} expert scales for {experts} experts")
        cols = v.repeat_interleave(s.per_expert).to(y0.device)
        mask = cols if s.expert_mask is None else s.expert_mask * cols
        out[i] = dataclasses.replace(s, expert_mask=mask)
    return out


def _looped(sets, x) -> list:
    forms = ("state", "input") if x is not None else ("state",)
    return [s for s in sets if s.down is not None and s.form in forms]


def _reads(lora, x, dtype) -> list:
    return [_input_read(s, x, dtype) if s.form == "input" else None for s in lora]


def _derived_state_sets(sets):
    keyed = [s for s in sets if s.down is not None]
    return any(s.form == "state" for s in keyed)


def _mixed_route(
    qs,
    lora,
    keys,
    prev,
    shared,
    noise_scale,
    regions,
    by_state=False,
    x=None,
    interpret=False,
):

    if _derived_state_sets(lora):
        if not by_state:
            raise ValueError("derived state routing needs the state the experts read")
        joint = shared or any(s.form == "input" for s in lora)
        logits, candidates = [], []
        for s, region in zip(lora, regions, strict=True):
            value = x if s.form == "input" else qs
            mode = import_score(s)
            if mode == "derived":
                basis, rows, *_ = derived_keys(s)
                score = derived_logits(s, value)
                query = value
            elif mode == "read_share":
                read = _input_read(s, x, x.dtype)
                energy = read_energy(s, read)
                score = s.tau * energy / energy.sum(-1, keepdim=True).clamp_min(1e-12)
                basis, _ = _set_keys(s)
                rows = key_rows(basis, s.rows_in)
                query = F.linear(qs.float(), s.router.float())
            else:
                basis, output = _set_keys(s)
                query = F.linear(qs.float(), s.router.float())
                if prev is not None:
                    query = out_project(query, output, prev[len(logits)])
                rows = key_rows(basis, s.rows_in)
                score = s.tau * norm_energy(query, basis, s.rows_in)
                score = _cross_logits(score, s, basis)
                score = _learned_cross_logits(score, s, query, basis)
            if s.form == "state":
                basis = derived_keys(s)[0]
                score = _cross_logits(score, s, basis)
                score = _learned_cross_logits(score, s, value, basis)
            if joint and len(lora) > 1 and os.environ.get("KILORA_FILE_BALANCE", "1") != "0":
                score = score + 1.0 / (len(lora) * score.shape[-1])
            score = score.masked_fill(rows == 0, -1e9)
            logits.append(score)
            candidates.append(_candidate_mask(s, query, basis, region))
        if interpret:
            from .ir.driver import _joint_derived_select

            if joint:
                return _joint_derived_select(logits, lora, regions, candidates, noise_scale)
            return [
                _joint_derived_select([lg], [s], [r], [mask], noise_scale)[0]
                for lg, s, r, mask in zip(logits, lora, regions, candidates, strict=True)
            ]
        if all(mask is None for mask in candidates):
            return _gates_from_logits(logits, lora, joint, noise_scale, regions)
        return _gates_from_logits(logits, lora, joint, noise_scale, regions, candidates)
    if any(s.axis_joint for s in lora):
        if by_state:
            return _gates(qs, lora, True, noise_scale, prev, keys, regions)
        return _route(qs, lora, keys, prev, True, noise_scale, regions)
    inp = [s.form == "input" for s in lora]
    if not any(inp):
        if by_state:
            return _gates(qs, lora, shared, noise_scale, prev, keys, regions)
        return _route(qs, lora, keys, prev, shared, noise_scale, regions)
    out: list = [None] * len(lora)
    for flag, sh, ns in ((False, shared, noise_scale), (True, False, 0.0)):
        idx = [i for i, f in enumerate(inp) if f == flag]
        if not idx:
            continue
        part = [lora[i] for i in idx]
        pv = None if prev is None else [prev[i] for i in idx]
        ks = [keys[i] for i in idx]
        rg = [regions[i] for i in idx]
        if by_state:
            gates = _gates(qs, part, sh, ns, pv, ks, rg)
        else:
            gates = _route([qs[i] for i in idx], part, ks, pv, sh, ns, rg)
        for i, g in zip(idx, gates, strict=True):
            out[i] = g
    return out


AXES = ("creative", "functional", "ip")
RECORD_AXES = (*AXES, "shared")


def axis_target(target: str, axis: str) -> str:
    return target if axis == "creative" else f"{target}@{axis}"


UNBUDGETED_AXES = frozenset({"functional"})


def budget_share(axes) -> float:
    return 1.0


def axis_budget(weights: Mapping[str, Any]) -> dict[str, Any]:

    budgeted = [a for a in weights if a not in UNBUDGETED_AXES]
    from .ir.dispatch import peek_override

    override = peek_override("budget") if _IR_ACTIVE_OVERRIDES.get() else None
    if _IR_INTERPRET_ALL.get() or override is not None:
        scalar = all(
            type(weights[a]) in (int, float) and math.isfinite(weights[a]) for a in budgeted
        )
        if override is not None and not scalar:
            from .ir.part_driver import budget_weights_ir

            return budget_weights_ir(weights)
        if override is not None and override.variant not in {"scalar", "scalar_sum"}:
            raise NotImplementedError(
                f"budget override variant {override.variant} needs mapped alphas"
            )
        if scalar:
            from .ir.driver import _run

            total = torch.tensor(0.0, dtype=torch.float64)
            for axis in budgeted:
                alpha = torch.tensor(weights[axis], dtype=torch.float64)
                total = _run("budget", "scalar_sum", {"previous": total, "alpha": alpha})["total"]
            result = dict(weights)
            for axis in budgeted:
                alpha = torch.tensor(weights[axis], dtype=torch.float64)
                result[axis] = _run("budget", "scalar", {"total": total, "alpha": alpha})[
                    "scale"
                ].item()
            return result
        from .fused import _note

        if override is not None:
            raise NotImplementedError("budget override requires finite scalar alphas")
        _note("IR driver refused budget: tensor or nonfinite alpha")
    return dict(weights)


def _axis_alpha(alpha, y0: torch.Tensor, target: str):
    from .call_values import ComposedAlpha, current_alpha_cache
    from .region import current_latent_shape, current_layout, current_text_tokens

    if isinstance(alpha, ComposedAlpha):
        text = _composed_text_alpha(alpha, y0, target)
        if text is not None:
            return text
        alpha = alpha.image
    cache = current_alpha_cache()
    if cache is None or not isinstance(alpha, torch.Tensor):
        return _map_axis_alpha(alpha, y0, target)
    key = (
        id(alpha),
        tuple(y0.shape[:-1]),
        y0.device,
        y0.dtype,
        current_layout(target),
        current_latent_shape(),
        current_text_tokens(),
    )
    if key not in cache:
        cache[key] = (alpha, _map_axis_alpha(alpha, y0, target))
    return cache[key][1]


def _composed_text_alpha(alpha, y0: torch.Tensor, target: str):
    from .region import current_layout, current_text_tokens

    layout = current_layout(target)
    tokens = math.prod(y0.shape[1:-1])
    if layout is not None:
        if layout.kind != "text1d":
            return None
    elif y0.dim() != 3 or current_text_tokens() != tokens:
        return None
    text = alpha.text
    if text is None or text.shape[-1] != tokens:
        return 1.0
    batch = y0.shape[0]
    if text.shape[0] != batch:
        if batch % text.shape[0]:
            raise ValueError(f"a text weight batch of {text.shape[0]} for a batch of {batch}")
        text = text.repeat(batch // text.shape[0], 1)
    return text.reshape(batch, *y0.shape[1:-1], 1).to(y0.device, y0.dtype)


def _composed_regions(sub, regions, weight, y0: torch.Tensor):
    if not isinstance(weight, torch.Tensor) or not any(s.routing_scope == "tensor" for s in sub):
        return regions
    flat = y0.flatten(1, -2) if y0.dim() > 3 else y0
    gate = (weight > 0).to(torch.float32)
    gate = _flat_tokens(gate, y0.dim()) if y0.dim() > 3 else gate
    gate = torch.broadcast_to(gate, (*flat.shape[:-1], 1))
    out = []
    for i, s in enumerate(sub):
        own = regions[i] if regions is not None else s.region
        if s.routing_scope != "tensor":
            out.append(own)
            continue
        if y0.dim() > 3:
            own = _flat_tokens(own, y0.dim())
        mine = _token_region(own, flat)
        out.append(gate if mine is None else mine * gate)
    return out


def _map_axis_alpha(alpha, y0: torch.Tensor, target: str):
    from .call_values import crop_map, current_crop_window
    from .region import current_latent_shape, current_layout, current_text_tokens

    layout = current_layout(target)
    latent = current_latent_shape()
    dims = tuple(y0.shape[1:-1])
    if isinstance(alpha, torch.Tensor) and alpha.ndim == 3 and latent and current_crop_window():
        import torch.nn.functional as F

        alpha = crop_map(alpha, latent[-2:])
        if tuple(alpha.shape[-2:]) != tuple(latent[-2:]):
            alpha = F.adaptive_avg_pool2d(alpha[:, None], latent[-2:])[:, 0]
    tokens = math.prod(dims)
    if layout is not None and layout.kind in ("text1d", "set"):
        return 1.0
    if layout is not None and layout.role == "reference":
        return 1.0
    if layout is not None and layout.patch is not None and latent is not None:
        from .family_spec import map_region

        grid = layout.grid(latent)
        if not isinstance(alpha, torch.Tensor):
            mask = torch.full((1, *grid), 1.0 - float(alpha))
        else:
            mask = 1 - alpha.detach().float()
            if math.prod(mask.shape[1:]) == math.prod(latent):
                mask = mask.reshape(-1, *latent)
            elif math.prod(mask.shape[1:]) == math.prod(grid):
                mask = mask.reshape(-1, *grid)
            else:
                marker = map_region(
                    torch.ones((1, *grid)),
                    tokens,
                    layout,
                    latent,
                    text_tokens=current_text_tokens(),
                )
                flat = mask.reshape(mask.shape[0], -1)
                if flat.shape[-1] != tokens:
                    raise ValueError("axis alpha does not fit the image or layer tokens")
                mask = (flat * marker.to(flat.device)).reshape(mask.shape[0], *dims)
                return 1 - mask.to(y0.device, y0.dtype)[..., None]
        mapped = map_region(mask, tokens, layout, latent, text_tokens=current_text_tokens())
        if mapped is None:
            return 1.0
        alpha = (1 - mapped).reshape(mapped.shape[0], *dims)
    elif isinstance(alpha, torch.Tensor) and latent is not None:
        from .region import layer_region

        if math.prod(alpha.shape[1:]) == math.prod(latent):
            shape = latent[-2:] if len(latent) > 2 and all(n == 1 for n in latent[:-2]) else latent
            mask = (1 - alpha.detach().float()).reshape(-1, *shape)
            try:
                mapped = layer_region(mask, tokens, latent, target=target)
            except ValueError:
                return 1.0
            if mapped is None:
                return 1.0
            alpha = (1 - mapped).reshape(mapped.shape[0], *dims)
    if not isinstance(alpha, torch.Tensor):
        return float(alpha)
    flat = _flat_tokens(alpha, y0.dim()) if y0.dim() > 3 else alpha
    gate = _token_region(flat, y0.flatten(1, -2) if y0.dim() > 3 else y0)
    return 1.0 if gate is None else gate.reshape(*y0.shape[:-1], 1).to(y0.dtype)


def _flat_tokens(mask: torch.Tensor | None, dims: int) -> torch.Tensor | None:
    if mask is None or mask.dim() < dims - 1:
        return mask
    if mask.dim() == dims:
        return mask.reshape(mask.shape[0], -1, 1)
    return mask.reshape(mask.shape[0], -1)


_TENSOR_METADATA: dict[tuple[int, str], tuple[Any, int | None, Any]] = {}


def _tensor_metadata(tensor: torch.Tensor, name: str, calculate):
    base = tensor._base
    if (
        base is not None
        and tensor.is_contiguous()
        and base.is_contiguous()
        and tensor.numel() == base.numel()
        and tensor.storage_offset() == base.storage_offset()
    ):
        tensor = base
    try:
        version = tensor._version
    except RuntimeError:
        return calculate()
    key = (id(tensor), name)
    entry = _TENSOR_METADATA.get(key)
    if entry is not None and entry[0]() is tensor and entry[1] == version:
        return entry[2]
    value = calculate()

    def discard(ref):
        current = _TENSOR_METADATA.get(key)
        if current is not None and current[0] is ref:
            _TENSOR_METADATA.pop(key, None)

    _TENSOR_METADATA[key] = (weakref.ref(tensor, discard), version, value)
    return value


def _token_region(region: torch.Tensor | None, y0: torch.Tensor) -> torch.Tensor | None:
    if region is None:
        return None
    r = region.detach().to(y0.device, torch.float32)
    if r.dim() == y0.dim():
        r = r[..., 0]
    lead = tuple(y0.shape[:-1])
    if r.shape[-1] != lead[-1]:
        raise ValueError(f"region has {r.shape[-1]} tokens, the layer {lead[-1]}")
    if r.dim() == len(lead) and len(lead) > 1 and r.shape[0] not in (1, lead[0]):
        if lead[0] % r.shape[0]:
            raise ValueError(f"region batch {r.shape[0]} does not divide {lead[0]}")
        r = r.repeat(lead[0] // r.shape[0], *([1] * (r.dim() - 1)))
    r = r.clamp(0, 1)
    if _tensor_metadata(region, "region_all_ones", lambda: bool((r == 1).all())):
        return None
    return torch.broadcast_to(r, lead)[..., None]


def _run_loop(
    y0,
    sets,
    depth,
    shared,
    noise_scale,
    query_policy,
    grad_steps,
    target,
    regions=None,
    places=None,
    x=None,
    td=None,
):
    lora = _looped(sets, x)
    if not lora:
        return y0
    regions = regions if regions is not None else [None] * len(lora)
    if _derived_state_sets(lora):
        return _run_input(
            x,
            y0,
            lora,
            list(range(len(lora))),
            depth,
            regions,
            target,
            td,
            noise_scale=noise_scale,
            query_policy=query_policy,
            grad_steps=grad_steps,
            shared=shared,
        )
    dtype = y0.dtype
    reads = _reads(lora, x, dtype)
    with torch.no_grad():
        base = [
            F.linear(y0.float(), s.down.float()) if a is None else a.float()
            for s, a in zip(lora, reads, strict=True)
        ]
    zs: list = [None] * len(lora)
    hold = None
    if lora[0].carry == "dropped":
        hold = [
            (F.linear(y0.to(dtype), s.down.to(dtype)) if a is None else a) * _scale(s, dtype)
            for s, a in zip(lora, reads, strict=True)
        ]
    keys = [_set_keys(s) for s in lora]
    kinds = [learned_kind(s) for s in lora]
    gated = [kind not in ("none", "separable") for kind in kinds]
    adaptive = [uses_adaptive(s) for s in lora]
    split = [kind == "separable" or a for kind, a in zip(kinds, adaptive, strict=True)]
    blend_coefficients = (
        td is not None
        and not any(split)
        and not any(gated)
        and all(s.expert_offsets is None for s in lora)
        and not any(s.exact for s in lora)
    )
    blended_zs = [None] * len(lora)
    if any(split):
        y0_norm, y0_rms = _y0_gate_stats(y0)
    if any(gated):
        with torch.no_grad():
            rms = y0.float().pow(2).mean(-2, keepdim=True).sqrt() + DROPOUT_DELTA
        prev_delta: list = [torch.zeros_like(y0) for _ in lora]
    steps, low, frac = _depth_parts(depth)
    splits = [depth_split_on(s) for s in lora]
    metrics = [
        _step_metric(s, y0.device) if on else None for s, on in zip(lora, splits, strict=True)
    ]
    refs: list = [None] * len(lora)
    y, carried, frozen, prev = y0, y0, False, None
    start = _carry_start(target, tuple(y0.shape[:-1]), len(lora))
    if start is not None and any(s.exact for s in lora):
        raise NotImplementedError("exact routed changes do not support carry across sampler steps")
    exact_weights = [None] * len(lora)
    if start is not None:
        if hold is not None:
            hold = [t.to(dtype) for t in start["carried"]]
        zs = [
            t.to(dtype) if s.aggregation != "replace" else None
            for s, t in zip(lora, start["change"], strict=True)
        ]
    hists = [gate_history(start, i, zs[i], regions[i]) for i in range(len(lora))]
    if start is not None:
        for i, (s, t) in enumerate(zip(lora, start["carried"], strict=True)):
            q = hists[i].get("q") if adaptive[i] else None
            if q is None:
                carried = carried + t.to(dtype) @ s.up.to(dtype)
            else:
                upq = s.up.to(dtype) * q[1].to(dtype)
                carried = carried + (q[0].to(dtype) * t.to(dtype)) @ upq
    snap = None
    if frac and low == 0:
        snap = (y0, _loop_initial(lora, start, hold, y0, dtype), list(zs))
    out = y0
    for k in range(1, steps + 1):
        old_zs, old_carried = list(zs), carried
        old_exact_weights = list(exact_weights)
        old_hold = list(hold) if hold is not None else None
        last = k == steps if td is None else td.last(k)
        act = None if td is None else td.active(k)
        keep = grad_steps is None or grad_steps(k)
        with contextlib.nullcontext() if keep else torch.no_grad():
            if _PROBES:
                _probe(target, k, carried, prev, lora, places, regions, reads)
            state = carried.detach() if frozen else carried
            q = state if query_policy is None else query_policy(state)
            gates = _mixed_route(q, lora, keys, prev, shared, noise_scale, regions, True)
            _note_routes(target, gates, lora)
            _note_directions(target, k, lora, gates)
            if _PROBES:
                _probe_gates(target, k, gates, places)
            prev = gates
            for i, (s, gate) in enumerate(zip(lora, gates, strict=True)):
                coef = (
                    F.linear(carried.to(dtype), s.down.to(dtype)) if reads[i] is None else reads[i]
                )
                weight = gate.to(dtype).repeat_interleave(s.per_expert, dim=-1)
                scale = s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale
                part = coef * weight * scale
                if not keep:
                    part = part.detach()
                if s.aggregation == "replace" or zs[i] is None:
                    zs[i] = part
                elif splits[i]:
                    zs[i] = depth_split_step(
                        zs[i], part, s, metrics[i], refs[i], target=_set_target(target, s, i)
                    )
                else:
                    zs[i] = zs[i] + part
                if splits[i] and k == 1 and s.radial_budget > 0:
                    refs[i] = _metric_size(zs[i], metrics[i])
                if s.exact:
                    weighted_gate = gate * scale
                    exact_weights[i] = (
                        weighted_gate
                        if s.aggregation == "replace" or exact_weights[i] is None
                        else exact_weights[i] + weighted_gate
                    )
            y = y0
            masks: list = [None] * len(lora)
            for i, (s, z) in enumerate(zip(lora, zs, strict=True)):
                r = regions[i]
                if blend_coefficients:
                    if r is not None:
                        masks[i] = (r.to(dtype), None)
                    continue
                if split[i] and not gated[i]:
                    args = (y0_norm, y0_rms, gates[i], target, i, k, r, last)
                    a, b = split_gate(s, z, hists[i], *args, active=act)
                    masks[i] = (a.to(dtype), b.to(dtype))
                    y = y + ((masks[i][0] * z) @ s.up.to(dtype)) * masks[i][1]
                    continue
                if r is not None and not gated[i]:
                    masks[i] = (r.to(dtype), None)
                    y = y + (masks[i][0] * z) @ s.up.to(dtype)
                    continue
                delta = z @ s.up.to(dtype)
                if gated[i]:
                    masks[i] = output_gate(s, delta, y0, prev_delta[i], gates[i], rms, target, i, k)
                    if adaptive[i]:
                        args = (y0_norm, y0_rms, gates[i], target, i, k, r, last)
                        a, b = split_gate(s, z, hists[i], *args, active=act)
                        masks[i] = masks[i] * (a * b).to(masks[i].dtype)
                    elif r is not None:
                        masks[i] = masks[i] * r.to(masks[i].dtype)
                    prev_delta[i] = delta
                    delta = masks[i] * delta
                y = y + delta
            for i, s in enumerate(lora):
                if not s.exact:
                    continue
                extra = _exact_correction(s, x, reads[i], exact_weights[i], None, dtype)
                mask = masks[i]
                if isinstance(mask, tuple):
                    extra = extra * mask[0]
                    if mask[1] is not None:
                        extra = extra * mask[1]
                elif mask is not None:
                    extra = extra * mask
                y = y + extra
            y = _add_expert_offsets(y, lora, gates)
            if td is not None:
                if blend_coefficients:
                    for i, z in enumerate(zs):
                        value = z if regions[i] is None else z * regions[i].to(dtype)
                        old = blended_zs[i]
                        blended_zs[i] = td.blend(
                            torch.zeros_like(value) if old is None else old, value, k
                        )
                else:
                    out = td.blend(out, y, k)
                zs = [td.keep(z, o, k) for z, o in zip(zs, old_zs, strict=True)]
                exact_weights = [
                    td.keep(z, o, k) if z is not None else None
                    for z, o in zip(exact_weights, old_exact_weights, strict=True)
                ]
            _ablation.note_output(target, k, y, y0)
            if k < steps:
                carried = y0
                czs = []
                for i, (s, z) in enumerate(zip(lora, zs, strict=True)):
                    m = _read_dropout_for_set(
                        z,
                        base[i],
                        s.per_expert,
                        _set_target(target, s, i),
                        k,
                        column_keys(s),
                        s.key,
                    )
                    if hold is None:
                        cz = m * z
                    else:
                        hold[i] = (
                            z
                            if _ablation.read_dropout_off()
                            else (hold[i] + (1 - m) * (z - hold[i]))
                        )
                        cz = hold[i]
                    czs.append(cz)
                    if isinstance(masks[i], tuple):
                        part = (masks[i][0] * cz) @ s.up.to(dtype)
                        if masks[i][1] is not None:
                            part = part * masks[i][1]
                    elif masks[i] is not None:
                        part = masks[i] * (cz @ s.up.to(dtype))
                    else:
                        part = cz @ s.up.to(dtype)
                    carried = carried + part
                if td is not None:
                    carried = td.keep(carried, old_carried, k)
                    if hold is not None:
                        hold = [td.keep(h, o, k) for h, o in zip(hold, old_hold, strict=True)]
                if frac and k == low:
                    snap = (y, czs, list(zs))
        frozen = not keep
    if frac:
        y = snap[0] + frac * (y - snap[0])
    if td is not None:
        if blend_coefficients:
            y = y0
            for z, s in zip(blended_zs, lora, strict=True):
                y = y + z.to(dtype) @ s.up.to(dtype)
        else:
            y = out
    if _STEP_CARRY:
        changes = _blend(snap[2], zs, frac) if frac else zs
        ends = [
            carry_transfer(z, base[i], s.per_expert)
            for i, (s, z) in enumerate(zip(lora, changes, strict=True))
        ]
        _carry_end(target, ends, changes, start, depth, _gate_ends(lora, hists))
    if _PROBES:
        y = _probe_output(target, y, lora, places)
    return y.detach() if frozen else y


def _run_path(
    y0, lora, depth, shared, noise_scale, grad_steps, target, regions=None, stop=None, x=None
):
    if _derived_state_sets(lora):
        if depth <= 0:
            return y0
        paths = [dataclasses.replace(s, iteration="path") for s in lora]
        regs = regions if regions is not None else [None] * len(paths)
        if paths[0].path_width > 1:
            from .path_walks import run_wide

            return run_wide(
                y0, paths, depth, shared, noise_scale, grad_steps, target, regs, stop, x
            )
        return _run_input(
            x,
            y0,
            paths,
            list(range(len(paths))),
            depth,
            regs,
            target,
            noise_scale=noise_scale,
            grad_steps=grad_steps,
            shared=shared,
            stop=stop,
        )
    if lora[0].path_width > 1 and any(
        s.gate_head_limit is not None and uses_adaptive(s) for s in lora
    ):
        raise ValueError("gate_history head:k is not supported with multiple path walks")
    if lora[0].path_width > 1:
        from .path_walks import run_wide

        result = run_wide(
            y0, lora, depth, shared, noise_scale, grad_steps, target, regions, stop, x
        )
        if _ablation.recording_outputs():
            for k in range(1, depth):
                with torch.no_grad(), _ablation.suspend_recording():
                    observed = run_wide(
                        y0.detach(),
                        lora,
                        k,
                        shared,
                        noise_scale,
                        None,
                        target,
                        regions,
                        stop,
                        x,
                    )
                _ablation.note_output(target, k, observed, y0)
            _ablation.note_output(target, depth, result, y0)
        return result
    dtype = y0.dtype
    regions = regions if regions is not None else [None] * len(lora)
    keys = [_set_keys(s) for s in lora]
    c0 = [
        F.linear(y0.to(dtype), s.down.to(dtype)) if s.form == "state" else _input_read(s, x, dtype)
        for s in lora
    ]
    qs = [F.linear(y0.float(), s.router.float()) for s in lora]
    zs: list = []
    prev_zs: list = []
    prev_node = done = last_gates = exact_gates = None
    last_k = 0
    frozen = False
    fixed = lora[0].fixed_point_stop if stop is None else stop
    head_hists = [{} for _ in lora]
    if any(s.gate_head_limit is not None and uses_adaptive(s) for s in lora):
        y0_norm, y0_rms = _y0_gate_stats(y0)
    for k in range(1, depth + 1):
        keep = grad_steps is None or grad_steps(k)
        with contextlib.nullcontext() if keep else torch.no_grad():
            if frozen:
                qs = [q.detach() for q in qs]
            gates = _mixed_route(qs, lora, keys, None, shared, noise_scale, regions)
            _note_routes(target, gates, lora)
            _note_directions(target, k, lora, gates)
            new = []
            for s, gate, c in zip(lora, gates, c0, strict=True):
                weight = gate.to(dtype).repeat_interleave(s.per_expert, dim=-1)
                scale = s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale
                z = c * weight * scale
                new.append(z if keep else z.detach())
            node = torch.cat(gates, -1).detach().argmax(-1)
            if fixed and prev_node is not None:
                same = (node == prev_node).all(-1)
                done = same if done is None else done | same
                stop = done[..., None, None]
                new = [torch.where(stop, old, z) for old, z in zip(zs, new, strict=True)]
                exact_gates = [
                    torch.where(stop, old, gate)
                    for old, gate in zip(exact_gates, gates, strict=True)
                ]
            else:
                exact_gates = gates
            for i, (s, z) in enumerate(zip(lora, new, strict=True)):
                if k == 1 and uses_adaptive(s) and not base_reference():
                    up = s.up.detach().float()
                    head_hists[i]["first_norm"], head_hists[i]["first_rms"] = change_stats(
                        z, up, up @ up.t()
                    )
                if s.gate_head_limit is not None and uses_adaptive(s):
                    split_gate(
                        s, z, head_hists[i], y0_norm, y0_rms, gates[i], target, i, k, None, True
                    )
            prev_zs, zs, prev_node, last_gates, last_k = zs, new, node, gates, k
            if _ablation.recording_outputs():
                with torch.no_grad():
                    observed = _path_output(
                        y0.detach(),
                        lora,
                        [z.detach() for z in zs],
                        [z.detach() for z in prev_zs],
                        last_gates,
                        last_k,
                        regions,
                        target,
                        [dict(hist) for hist in head_hists],
                        x=x,
                        exact_gates=exact_gates,
                    )
                    _ablation.note_output(target, k, observed, y0)
            qs = [out_project(q, kout, g) for q, (_, kout), g in zip(qs, keys, gates, strict=True)]
        frozen = not keep
        if done is not None and bool(done.all()):
            break
    y = _path_output(
        y0,
        lora,
        zs,
        prev_zs,
        last_gates,
        last_k,
        regions,
        target,
        head_hists,
        x=x,
        exact_gates=exact_gates,
    )
    return y.detach() if frozen else y


def _path_output(
    y0,
    lora,
    zs,
    prev_zs,
    last_gates,
    last_k,
    regions,
    target,
    head_hists=None,
    x=None,
    exact_gates=None,
):
    dtype = y0.dtype
    y = y0
    rms = None
    for i, (s, z) in enumerate(zip(lora, zs, strict=True)):
        delta = z @ s.up.to(dtype)
        extra = None
        if s.exact:
            gate = last_gates[i] if exact_gates is None else exact_gates[i]
            coef = _input_read(s, x, dtype)
            extra = _exact_correction(s, x, coef, gate, None, dtype) * _scale(s, dtype)
        kind = learned_kind(s)
        if kind in ("none", "separable") and (kind == "separable" or uses_adaptive(s)):
            y0_norm, y0_rms = _y0_gate_stats(y0)
            before = prev_zs[i] if prev_zs else None
            hist = {"prev": before, "before": before}
            if head_hists:
                hist.update(
                    {
                        key: head_hists[i][key]
                        for key in ("first_norm", "first_rms")
                        if key in head_hists[i]
                    }
                )
            if head_hists and s.gate_head_limit is not None and uses_adaptive(s):
                hist.update(head_hists[i])
                hist["prev"] = before
                hist["head_count"] = s.gate_head_limit
            gw = last_gates[i]
            a, b = split_gate(s, z, hist, y0_norm, y0_rms, gw, target, i, last_k, None, True)
            delta = a.to(dtype) * delta * b.to(dtype)
            if extra is not None:
                extra = a.to(dtype) * extra * b.to(dtype)
        elif kind != "none":
            if rms is None:
                with torch.no_grad():
                    rms = y0.float().pow(2).mean(-2, keepdim=True).sqrt() + DROPOUT_DELTA
            before = prev_zs[i] @ s.up.to(dtype) if prev_zs else torch.zeros_like(delta)
            mask = output_gate(s, delta, y0, before, last_gates[i], rms, target, i, last_k)
            if uses_adaptive(s):
                y0_norm, y0_rms = _y0_gate_stats(y0)
                hist = {"before": prev_zs[i] if prev_zs else None}
                if head_hists:
                    hist.update(
                        {
                            key: head_hists[i][key]
                            for key in ("first_norm", "first_rms")
                            if key in head_hists[i]
                        }
                    )
                if head_hists and s.gate_head_limit is not None and uses_adaptive(s):
                    hist.update(head_hists[i])
                    hist["head_count"] = s.gate_head_limit
                gw = last_gates[i]
                a, b = split_gate(s, z, hist, y0_norm, y0_rms, gw, target, i, last_k, None, True)
                mask = mask * (a * b).to(mask.dtype)
            delta = mask * delta
            if extra is not None:
                extra = mask * extra
        if extra is not None:
            delta = delta + extra
        if regions[i] is not None:
            delta = regions[i].to(dtype) * delta
        y = y + delta
    return _add_expert_offsets(y, lora, last_gates)


OUTPUT_GATES = ("none", "separable", "neuron", "lowrank")
GATE_START = 0.95
GATE_HIDDEN = 4
GATE_CHUNK = 256
_GATE_STATS = ContextStack("kilora_gate_stats")


def _gate_mode(s: LayerSet) -> str:
    return resolve_mode(s.gate_mode, s.gate is not None and s.output_gate != "none")


def learned_kind(s: LayerSet) -> str:
    if s.gate is None or _gate_mode(s) not in ("learned", "both"):
        return "none"
    return s.output_gate


def uses_adaptive(s: LayerSet) -> bool:
    return _gate_mode(s) in ("adaptive", "both")


def gate_fixed_after(layer_set: LayerSet) -> int | None:
    return layer_set.gate_head_limit if _gate_mode(layer_set) == "adaptive" else None


def split_gate(
    s: LayerSet,
    z,
    hist: dict,
    y0_norm,
    y0_rms,
    gate_w,
    target,
    i,
    k,
    region,
    last=False,
    active=None,
    uniforms=None,
):
    a = b = None
    if learned_kind(s) == "separable":
        a, b = separable_gate(s, z, hist.get("prev"), y0_norm, y0_rms, gate_w, target, i, k)
    hist["prev"] = z
    if uses_adaptive(s):
        limit = s.gate_head_limit
        if limit is not None and hist.get("head_count", 0) >= limit:
            qa, qb = hist["head_q"]
        else:
            zr = z if region is None else z * region.to(z.dtype)
            if active is not None:
                zr = zr * active.to(zr.dtype)
            if "adaptive_gram" not in hist:
                up = s.up.detach().float()
                hist["adaptive_gram"] = up @ up.t()
            reference = {}
            if not base_reference():
                first = "first_norm" not in hist
                if first:
                    before = hist.get("before")
                    dz = zr.detach().float()
                    if before is not None:
                        dz = dz - before.detach().float()
                    hist["first_norm"], hist["first_rms"] = change_stats(
                        dz, s.up, hist["adaptive_gram"], active
                    )
                reference = dict(
                    first_norm=hist["first_norm"], first_rms=hist["first_rms"], first_step=k == 1
                )
            qa, qb, hist["m"] = adaptive_keep(
                zr,
                hist.get("before"),
                hist.get("m"),
                s.up,
                y0_norm,
                y0_rms,
                s.gate_kappa,
                s.gate_coherence,
                active=active,
                gram=hist["adaptive_gram"],
                **reference,
            )
            hist["before"] = zr
            hist["q"] = (qa if region is None else qa * region, qb)
            if limit is not None:
                hist["head_count"] = hist.get("head_count", 0) + 1
                hist["head_q"] = (qa, qb)
        if _route_freeze.RECORDS:
            _route_freeze.note_gate(target, i, qa, qb)
        for stats in _GATE_STATS:
            stats.append(qa.flatten())
            stats.append(qb.flatten())
        if last is True or expect_masks():
            ha, hb = qa, qb
        else:
            n = qa.numel()
            name = f"{target}#{i}#adaptive"
            u = (
                dropout_uniforms((n + qb.numel(),), name, k, qa.device)
                if uniforms is None
                else uniforms(n + qb.numel(), name, k, qa.device)
            )
            masks = (u < torch.cat((qa.reshape(-1), qb.reshape(-1)))).float()
            ha = masks[:n].reshape(qa.shape)
            hb = masks[n:].reshape(qb.shape)
            if isinstance(last, torch.Tensor):
                ha = torch.where(last, qa, ha)
                hb = torch.where(last, qb, hb)
        a = ha if a is None else a * ha
        b = hb if b is None else b * hb
    if a is None:
        return None
    return (a if region is None else a * region), b


def gate_history(start: dict | None, i: int, z, region) -> dict:
    hist: dict = {}
    gates = (start or {}).get("gates")
    if gates is not None and gates[i] is not None:
        hist["m"], hist["q"] = gates[i]["m"], gates[i]["q"]
        if "head_count" in gates[i]:
            hist["head_count"] = gates[i]["head_count"]
            hist["head_q"] = gates[i]["head_q"]
    if z is not None:
        hist["before"] = z if region is None else z * region.to(z.dtype)
    return hist


def _gate_ends(lora, hists) -> list | None:

    if not any(uses_adaptive(s) for s in lora):
        return None
    return [
        {key: h[key] for key in ("m", "q", "head_count", "head_q") if key in h}
        if "m" in h
        else None
        for h in hists
    ]


def new_gate(
    kind: str, experts: int, width: int, gen: torch.Generator, cols: int = 0
) -> dict | None:
    if kind == "none":
        return None
    bias = math.log(GATE_START / (1 - GATE_START))
    h = GATE_HIDDEN
    if kind == "separable":
        return {
            "cw1": torch.randn(experts, width, 3, h, generator=gen) * 0.5,
            "cb1": torch.zeros(experts, width, h),
            "cw2": torch.randn(experts, width, h, generator=gen) * 0.01,
            "cb2": torch.full((experts, width), bias),
            "tp": torch.randn(experts, cols, 2, generator=gen) / math.sqrt(max(cols, 1)),
            "tw1": torch.randn(experts, 4, 8, generator=gen) * 0.5,
            "tb1": torch.zeros(experts, 8),
            "tw2": torch.randn(experts, 8, generator=gen) * 0.01,
            "tb2": torch.full((experts,), bias),
        }
    if kind == "neuron":
        return {
            "w1": torch.randn(experts, width, 3, h, generator=gen) * 0.5,
            "b1": torch.zeros(experts, width, h),
            "w2": torch.randn(experts, width, h, generator=gen) * 0.01,
            "b2": torch.full((experts, width), bias),
        }
    if kind == "lowrank":
        return {
            "v": torch.randn(experts, 3, width, h, generator=gen) / math.sqrt(3 * width),
            "u": torch.randn(experts, h, width, generator=gen) * 0.01,
            "b": torch.full((experts, width), bias),
        }
    raise ValueError(f"unknown output gate {kind!r}")


def gate_score(kind: str, gate: dict, feats: torch.Tensor, e: int) -> torch.Tensor:
    if kind == "neuron":
        w1, b1 = gate["w1"][e].float(), gate["b1"][e].float()
        w2, b2 = gate["w2"][e].float(), gate["b2"][e].float()
        parts = []
        for c in range(0, feats.shape[-2], GATE_CHUNK):
            sl = slice(c, c + GATE_CHUNK)
            hidden = torch.relu(
                torch.einsum("...cf,cfh->...ch", feats[..., sl, :], w1[sl]) + b1[sl]
            )
            parts.append(torch.sigmoid((hidden * w2[sl]).sum(-1) + b2[sl]))
        return torch.cat(parts, -1)
    v, u, b = gate["v"][e].float(), gate["u"][e].float(), gate["b"][e].float()
    low = torch.einsum("...cf,fch->...h", feats, v)
    return torch.sigmoid(low @ u + b)


def output_gate(
    s: LayerSet, delta, y0, prev_delta, gate_w, rms, target: str, i: int, k: int
) -> torch.Tensor:
    feats = torch.stack([delta, y0, prev_delta], -1).float() / rms[..., None]
    w = gate_w.float()
    used = (
        (w.detach() > 0).flatten(0, -2).any(0).nonzero().flatten().tolist()
        if torch.is_grad_enabled()
        else range(w.shape[-1])
    )
    q = None
    for e in used:
        part = w[..., e : e + 1] * gate_score(s.output_gate, s.gate, feats, e)
        q = part if q is None else q + part
    for stats in _GATE_STATS:
        stats.append(q.detach())
    u = dropout_uniforms(tuple(q.shape), f"{target}#{i}#gate", k, q.device)
    with torch.no_grad():
        g = (q if expect_masks() else (u < q)).to(delta.dtype)
    if _HARD_GATES or not (torch.is_grad_enabled() and q.requires_grad):
        return g
    q = q.to(delta.dtype)
    return g + (q - q.detach())


def separable_gate(s: LayerSet, z, z_prev, y0_norm, y0_rms, gate_w, target: str, i: int, k: int):

    g = s.gate
    zp = None if z_prev is None else z_prev.float()
    qa, qb = _gate_scores()(
        z.float(),
        zp,
        s.up.float(),
        y0_norm,
        y0_rms,
        gate_w.float(),
        *(g[n].float() for n in _SEPARABLE_PARTS),
    )
    for stats in _GATE_STATS:
        stats.append(qa.detach().flatten())
        stats.append(qb.detach().flatten())
    n = qa.numel()
    u = dropout_uniforms((n + qb.numel(),), f"{target}#{i}#gate", k, qa.device)
    out = []
    for q, uq in ((qa, u[:n].reshape(qa.shape)), (qb, u[n:].reshape(qb.shape))):
        with torch.no_grad():
            hard = q.detach().float() if expect_masks() else (uq < q).float()
        if _HARD_GATES or not (torch.is_grad_enabled() and q.requires_grad):
            out.append(hard)
        else:
            out.append(hard + (q - q.detach()))
    return out[0], out[1]


_SEPARABLE_PARTS = ("tp", "tw1", "tb1", "tw2", "tb2", "cw1", "cb1", "cw2", "cb2")


def _separable_scores(zf, zp, up, y0_norm, y0_rms, w, tp, tw1, tb1, tw2, tb2, cw1, cb1, cw2, cb2):
    gram = up @ up.t()
    tokens = zf.shape[-2]
    v = zf if zp is None else torch.stack([zf, zp])
    norms = ((v @ gram) * v).sum(-1).clamp(min=0).sqrt()
    chans = (((v.transpose(-1, -2) @ v / tokens) @ up) * up).sum(-2).clamp(min=0).sqrt()
    if zp is None:
        norm, norm_p = norms, torch.zeros_like(norms)
        chan_z, chan_p = chans, torch.zeros_like(chans)
    else:
        (norm, norm_p), (chan_z, chan_p) = norms.unbind(0), chans.unbind(0)
    scale = y0_rms.mean(-1, keepdim=True)
    ch = torch.stack([chan_z, y0_rms, chan_p], -1) / scale[..., None]
    wbar = w.mean(-2)
    ratios = torch.stack([norm / y0_norm, norm_p / y0_norm], -1)
    proj = torch.einsum("...ti,eij->...tej", zf, tp)
    x = torch.cat([ratios.unsqueeze(-2).expand(*proj.shape[:-1], 2), proj], -1)
    h = torch.relu(torch.einsum("...tef,efh->...teh", x, tw1) + tb1)
    qa_e = torch.sigmoid(torch.einsum("...teh,eh->...te", h, tw2) + tb2)
    qa = (w * qa_e).sum(-1)
    hc = torch.relu(torch.einsum("...cf,ecfh->...ech", ch, cw1) + cb1)
    qb_e = torch.sigmoid((hc * cw2).sum(-1) + cb2)
    qb = torch.einsum("...e,...ec->...c", wbar, qb_e)
    return qa[..., None], qb[..., None, :]


_COMPILED: dict = {}


def _gate_scores():
    if os.environ.get("KILORA_COMPILE", "0") != "1":
        return _separable_scores
    mode = os.environ.get("KILORA_COMPILE_MODE", "max-autotune-no-cudagraphs")
    if mode not in _COMPILED:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _COMPILED[mode] = torch.compile(_separable_scores, dynamic=False, mode=mode)
    return _COMPILED[mode]


_GATE_STATS_MODE: ContextVar[str | None] = ContextVar("kilora_gate_stats_mode", default=None)


def resolve_gate_stats(value: str) -> str:
    value = os.environ.get("KILORA_GATE_STATS", value)
    if value not in ("fused", "torch"):
        raise ValueError("gate_stats must be fused or torch")
    return value


@contextlib.contextmanager
def gate_stats_mode(value: str):
    token = _GATE_STATS_MODE.set(_GATE_STATS_MODE.get() or resolve_gate_stats(value))
    try:
        yield
    finally:
        _GATE_STATS_MODE.reset(token)


def _y0_gate_stats(y0: torch.Tensor):
    mode = resolve_gate_stats(_GATE_STATS_MODE.get() or "fused")
    with torch.no_grad():
        if (
            y0.is_cuda
            and os.environ.get("TRITON_INTERPRET") != "1"
            and mode == "fused"
            and _gate_stats.supported(y0)
        ):
            return _gate_stats.fused_stats(y0, DROPOUT_DELTA)
        sq = y0.float().square()
        return sq.sum(-1).sqrt() + DROPOUT_DELTA, sq.mean(-2).sqrt() + DROPOUT_DELTA


@contextlib.contextmanager
def recording_gates() -> Iterator[list]:
    stats: list = []
    with _GATE_STATS.pushed(stats):
        yield stats


def gate_summary(qs: list) -> dict:
    if not qs:
        return {}
    flat = torch.cat([q.float().flatten().cpu() for q in qs])
    return {
        "mean": float(flat.mean()),
        "low": float((flat < 0.1).float().mean()),
        "mid": float(((flat >= 0.1) & (flat <= 0.9)).float().mean()),
        "high": float((flat > 0.9).float().mean()),
    }


_STEP_CARRY = ContextStack("kilora_step_carry")
_DERIVED_CARRY = ContextStack("kilora_derived_carry")
_DERIVED_FIRST_BLOCK = ContextStack("kilora_derived_first_block")


@contextlib.contextmanager
def step_carry(cache: dict) -> Iterator[dict]:
    with _STEP_CARRY.pushed(cache):
        yield cache


_SPECULATE = ContextStack("kilora_speculate")


def _speculate_prepare(cache):
    groups = {}
    for key, rec in list(cache.items()):
        if rec.get("chain"):
            _, _, _, prefix = _derived_chain_paths(
                rec["set"],
                None,
                None,
                None,
                rec["target"],
                list(rec["entry"]),
                rec["depth"],
                prediction=rec,
                stop_before_last=True,
            )
            cache[key] = {"signature": rec["signature"], "chain_prefix": prefix}
            continue
        if "c" not in rec:
            del cache[key]
            continue
        signature = (
            rec["c"].shape,
            rec["c"].device,
            rec["logits"].shape,
            rec["entry"].shape,
            rec["depth"],
            rec["per_expert"],
            rec["select"],
            rec["scope"],
            rec["iteration"],
            rec["gamma"],
            rec["adaptive"],
            rec["scale"],
        )
        groups.setdefault(signature, []).append((key, rec))
    for group in groups.values():
        records = [rec for _, rec in group]
        first = records[0]
        c = torch.stack([rec["c"] for rec in records])
        logits = torch.stack([rec["logits"] for rec in records])
        prev = torch.stack([rec["entry"].expand(-1, *rec["logits"].shape) for rec in records])
        logits = logits.unsqueeze(1).expand_as(prev)
        c = c.unsqueeze(1)
        z = c * prev.repeat_interleave(first["per_expert"], -1)
        before = z * first["scale"]
        first_changes = before
        history = (1 - HISTORY_RATE) * before if first["adaptive"] else None
        state = {}
        _derived_path_note(state, prev[..., :1, :] if first["scope"] == "tensor" else prev, logits)
        transitions = None
        if first["gamma"]:
            transitions = torch.stack([rec["transition"] for rec in records])
        for _ in range(2, first["depth"]):
            route = logits * derived_lift_after()
            if transitions is not None:
                matrix = transitions.reshape(
                    transitions.shape[0], *([1] * (prev.ndim - 2)), *transitions.shape[1:]
                )
                route = (
                    route
                    + first["gamma"] * ((prev.unsqueeze(-2) @ matrix).squeeze(-2) + 1e-6).log()
                )
            gates = (
                tensor_gate(route, select=first["select"])
                if first["scope"] == "tensor"
                else token_gate(route, select=first["select"])
            )
            part = c * gates.repeat_interleave(first["per_expert"], -1)
            z = part if first["iteration"] == "path" else z + part
            if history is not None:
                current = z * first["scale"]
                history = HISTORY_RATE * history + (1 - HISTORY_RATE) * (current - before)
                before = current
            prev = gates
            _derived_path_note(
                state, gates[..., :1, :] if first["scope"] == "tensor" else gates, logits
            )
        for n, (key, rec) in enumerate(group):
            reference = {}
            if rec["adaptive"] and not base_reference():
                up = rec["set"].up.float()
                reference["first_norm"], reference["first_rms"] = change_stats(
                    first_changes[n], up, up @ up.t()
                )
            cache[key] = {
                **reference,
                "signature": rec["signature"],
                "z": z[n],
                "prev": prev[n],
                "state": {name: value[n] for name, value in state.items()},
                "history": None if history is None else history[n],
            }


def speculative_cache_bytes(cache: dict) -> int:
    seen = set()

    def size(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(item) for item in value)
        return 0

    return size(cache)


@contextlib.contextmanager
def speculate(cache):
    _speculate_prepare(cache)
    with _SPECULATE.pushed(cache):
        yield cache


@contextlib.contextmanager
def derived_call():
    with _DERIVED_CARRY.pushed({}), _DERIVED_FIRST_BLOCK.pushed({}):
        yield


@contextlib.contextmanager
def generation(
    module: torch.nn.Module, carry: bool = True, seed: int = 0, steps: int | None = None
) -> Iterator[dict]:
    caches: dict = {}
    frozen: dict = {}
    speculative: dict = {}
    shape: list = [None]
    position = {"t": None, "n": 0, "i": -1}
    stack: list = []

    def before(mod, args, kwargs):
        t = kwargs.get("timestep", kwargs.get("t"))
        if t is None:
            t = args[1] if len(args) > 1 else 0
        x = args[0] if args else next(iter(kwargs.values()))
        if isinstance(x, (list, tuple)):
            x = x[0]
        if shape[0] != tuple(x.shape):
            caches.clear()
            frozen.clear()
            speculative.clear()
            shape[0] = tuple(x.shape)
        step = float(t.detach().flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        position["n"] = position["n"] + 1 if position["t"] == step else 0
        if position["n"] == 0:
            position["i"] += 1
        position["t"] = step
        ctx = contextlib.ExitStack()
        ctx.enter_context(dropout_key(seed, t))
        if steps is not None:
            ctx.enter_context(sampling_progress(position["i"] / max(steps - 1, 1)))
        ctx.enter_context(derived_call())
        if os.environ.get("KILORA_SPECULATE") == "1":
            ctx.enter_context(speculate(speculative.setdefault(position["n"], {})))
        if _ablation.carry_steps(carry):
            ctx.enter_context(step_carry(caches.setdefault(position["n"], {})))
        ctx.enter_context(_route_freeze.freezing(frozen.setdefault(position["n"], {})))
        stack.append(ctx)

    def after(mod, args, kwargs, out):
        stack.pop().close()

    handles = [
        module.register_forward_pre_hook(before, with_kwargs=True),
        module.register_forward_hook(after, with_kwargs=True),
    ]
    try:
        yield caches
    finally:
        for h in handles:
            h.remove()
        while stack:
            stack.pop().close()


def _carry_start(target: str, shape: tuple, n_sets: int) -> dict | None:
    if not _STEP_CARRY:
        return None
    entry = _STEP_CARRY[-1].get(target)
    if entry is None or len(entry["carried"]) != n_sets:
        return None
    if any(tuple(t.shape[:-1]) != shape for t in entry["carried"]):
        return None
    return entry


def _carry_end(
    target: str, carried: list, change: list, start: dict | None, depth: float, gates=None
):
    if _STEP_CARRY:
        _STEP_CARRY[-1][target] = {
            "carried": [t.detach() for t in carried],
            "change": [t.detach() for t in change],
            "depth": (start["depth"] if start is not None else 0) + depth,
            "gates": gates,
        }


_PROBES = ContextStack("kilora_key_probes")
_EAGER = ContextStack("kilora_eager")


@contextlib.contextmanager
def eager() -> Iterator[None]:
    with _EAGER.pushed(True):
        yield


@contextlib.contextmanager
def key_probes() -> Iterator[dict]:
    record: dict = {}
    with _PROBES.pushed(record):
        yield record


def _at_places(values, places):
    if values is None or places is None:
        return values
    keep, n = places
    out: list = [None] * n
    for pos, v in zip(keep, values, strict=True):
        out[pos] = v
    return out


def _probe_mask(s: LayerSet, region: torch.Tensor | None) -> torch.Tensor | None:
    m = None if region is None else region.detach().float()
    if s.expert_mask is not None:
        em = s.expert_mask[..., :: s.per_expert].detach().float()
        m = em if m is None else m * em
    return m


def _probe(target, k, state, prev, lora, places=None, regions=None, reads=None):
    regions = regions if regions is not None else [None] * len(lora)
    with torch.no_grad():
        queries = _at_places(
            [
                state.detach().float()
                if s.form == "state"
                else F.linear(state.detach().float(), s.router.float())
                for s in lora
            ],
            places,
        )
        masks = _at_places([_probe_mask(s, r) for s, r in zip(lora, regions, strict=True)], places)
        energy = None
        if reads is not None:
            pairs = zip(lora, reads, strict=True)
            each = [None if a is None else read_energy(s, a) for s, a in pairs]
            energy = _at_places(each, places)
    prev = None if prev is None else _at_places([g.detach() for g in prev], places)
    entry = {
        "k": k,
        "queries": queries,
        "prev": prev,
        "gates": None,
        "masks": masks,
        "reads": energy,
    }
    for record in _PROBES:
        record.setdefault(target, []).append(entry)


class _GradTap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, callback):
        ctx.callback = callback
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad):
        ctx.callback(grad)
        return grad, None


def read_energy(s: LayerSet, a: torch.Tensor) -> torch.Tensor:
    e = a.detach().float().pow(2)
    return e.reshape(*e.shape[:-1], -1, s.per_expert).sum(-1)


def _probe_input(target, k, state, sets, once, gates, prev, regions=None, reads=None):
    n = len(sets)
    with torch.no_grad():
        queries: list = [None] * n
        kept: list = [None] * n
        masks: list = [None] * n
        energy: list = [None] * n
        before: list | None = None if prev is None else [None] * n
        for j, (pos, gate) in enumerate(zip(once, gates, strict=True)):
            queries[pos] = F.linear(state.detach().float(), sets[pos].router.float())
            kept[pos] = gate.detach()
            masks[pos] = _probe_mask(sets[pos], None if regions is None else regions[j])
            if reads is not None:
                energy[pos] = read_energy(sets[pos], reads[j])
            if before is not None:
                before[pos] = prev[j].detach()
    sample = (
        state.detach().float().cpu()
        if k == 0 and share_transition_gamma() and share_transition_kind() == "weighted"
        else None
    )
    entry = {
        "k": k,
        "queries": queries,
        "prev": before,
        "gates": kept,
        "form": "input",
        "masks": masks,
        "reads": energy,
        "state": sample,
    }
    for record in _PROBES:
        record.setdefault(target, []).append(entry)


IMPORT_SCORES = ("keys", "read_share", "derived")


def import_score(s: LayerSet | None = None, *, form: str | None = None) -> str:
    if form == "state" or (s is not None and getattr(s, "form", None) == "state"):
        return "derived"
    if s is not None and getattr(s, "direction_core", None) is not None:
        return "keys"
    value = os.environ.get("KILORA_IMPORT_SCORE") or (
        (s.import_score or ("derived" if s.form == "state" else "keys"))
        if s is not None
        else "keys"
    )
    if value not in IMPORT_SCORES:
        raise ValueError(f"KILORA_IMPORT_SCORE must be one of {IMPORT_SCORES}, got {value!r}")
    return value


DEFAULT_DERIVED_MODE = "keys"


def derived_mode():
    mode = os.environ.get("KILORA_DERIVED_MODE", DEFAULT_DERIVED_MODE)
    if mode not in ("full", "keys"):
        raise ValueError("KILORA_DERIVED_MODE must be full or keys")
    return mode


def derived_diffusion():
    value = os.environ.get("KILORA_DERIVED_DIFFUSE", "0").lower()
    if value in ("on", "true", "1"):
        return 0.5
    if value in ("off", "false", ""):
        return 0.0
    strength = float(value)
    if not math.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("KILORA_DERIVED_DIFFUSE must be in [0, 1)")
    return strength


def derived_default_depth():
    return 2 if derived_diffusion() else 4


def share_transition_gamma(s=None):
    derived = import_score(s) == "derived"
    if derived and derived_mode() == "keys":
        return 0.0
    value = float(
        os.environ.get(
            "KILORA_SHARE_TRANSITION", ("0.5" if derived_diffusion() else "1.0") if derived else "0"
        )
    )
    if not math.isfinite(value) or value < 0:
        raise ValueError("KILORA_SHARE_TRANSITION must be a finite non-negative number")
    return value


TRANSITION_KINDS = (
    "cosine",
    "angles",
    "energy",
    "weighted",
    "trace",
    "transmission",
    "embed",
    "base",
)


def share_transition_kind() -> str:
    kind = os.environ.get("KILORA_SHARE_TRANSITION_KIND", "cosine")
    if kind not in TRANSITION_KINDS:
        raise ValueError(
            f"KILORA_SHARE_TRANSITION_KIND must be one of {TRANSITION_KINDS}, got {kind!r}"
        )
    return kind


def _share_transition_base(
    s: LayerSet,
    *,
    states: torch.Tensor | None = None,
    _kind: str | None = None,
    _raw: bool = False,
) -> torch.Tensor | None:

    if s.form != "input" or s.down is None or s.up is None or s.down.shape[1] != s.up.shape[1]:
        return None
    kind = _kind or share_transition_kind()
    if kind == "weighted" and states is None:
        return s.share_transition_weighted
    if kind == "base":
        return s.share_transition_base
    rank = os.environ.get("KILORA_SHARE_TRANSITION_RANK", "8")
    key = (
        None
        if states is not None
        else _tensor_key(
            f"share-transition-{kind}-{s.per_expert}-{rank}-{_raw}-"
            f"{derived_diffusion() if not _raw and import_score(s) == 'derived' else 0}",
            [s.down, s.up],
        )
    )
    hit = _cache_get(key)
    if hit is not None:
        return hit
    from . import derived_cache

    source = derived_cache.layer_source(s) if states is None else None
    settings = {
        "kind": kind,
        "rank": rank,
        "raw": _raw,
        "per_expert": s.per_expert,
        "diffusion": derived_diffusion() if import_score(s) == "derived" else 0,
        "backend": derived_cache.backend(s.down.device),
        "down_dtype": str(s.down.dtype),
        "up_dtype": str(s.up.dtype),
        "down_shape": tuple(s.down.shape),
        "up_shape": tuple(s.up.shape),
    }
    stored = derived_cache.load(source, "transition", settings, ("transition",), s.down.device)
    if stored is not None:
        _cache_put(key, stored[0])
        return stored[0]
    with torch.no_grad():
        down = s.down.detach().float().reshape(-1, s.per_expert, s.down.shape[1])
        up = s.up.detach().float().reshape(-1, s.per_expert, s.up.shape[1])
        if kind == "weighted":
            x = (
                states.detach()
                .to(device=down.device, dtype=torch.float32)
                .reshape(-1, down.shape[-1])
            )
            x = x - x.mean(0, keepdim=True)
            down = down @ x.T
            up = up @ x.T
        if kind in ("trace", "transmission", "embed"):
            aa = down @ down.transpose(-1, -2)
            bb = up @ up.transpose(-1, -2)
            norms = (aa * bb).sum((-1, -2)).clamp_min(0).sqrt()
            graph = []
            for e in range(len(down)):
                row = []
                for f in range(len(down)):
                    m = down[f] @ up[e].T
                    if kind == "trace":
                        value = (m * (down[e] @ up[f].T).T).sum()
                    else:
                        value = torch.trace(m.T @ bb[f] @ m @ aa[e]).clamp_min(0).sqrt()
                    row.append(value / (norms[f] * norms[e]).clamp_min(1e-30))
                graph.append(torch.stack(row))
            match = torch.stack(graph)
            if kind == "embed":
                rank = int(os.environ.get("KILORA_SHARE_TRANSITION_RANK", "8"))
                if rank < 1:
                    raise ValueError("KILORA_SHARE_TRANSITION_RANK must be positive")
                u, singular, vh = torch.linalg.svd(match, full_matrices=False)
                rank = min(rank, len(singular))
                match = (u[:, :rank] * singular[:rank]) @ vh[:rank]
            match = match.clamp_min(0)
        elif kind == "angles":

            def basis(block):
                _, values, vh = torch.linalg.svd(block, full_matrices=False)
                tol = values.max() * max(block.shape) * torch.finfo(values.dtype).eps
                return vh[values > tol]

            reads = [basis(a) for a in down]
            writes = [basis(b) for b in up]
            match = torch.stack(
                [
                    torch.stack(
                        [
                            (a @ b.T).square().sum() / min(a.shape[0], b.shape[0])
                            if a.shape[0] and b.shape[0]
                            else down.new_zeros(())
                            for a in reads
                        ]
                    )
                    for b in writes
                ]
            )
        else:
            match = torch.stack([(down @ b.T).square().sum((-2, -1)) for b in up])
            if kind != "energy":
                norms = (
                    up.square().sum((-2, -1)).sqrt()[:, None]
                    * down.square().sum((-2, -1)).sqrt()[None, :]
                )
                match = match.sqrt() / norms.clamp_min(1e-30)
        result = match if _raw else match / match.sum(-1, keepdim=True).clamp_min(1e-30)
        if not _raw and import_score(s) == "derived" and derived_diffusion():
            matrices = derived_diffusion_matrices(s)
            if matrices is not None:
                rout, rin = matrices
                result = (rout @ match @ rin.T).clamp_min(0)
                result = result / result.sum(-1, keepdim=True).clamp_min(1e-30)
    derived_cache.store(source, "transition", settings, ("transition",), (result,))
    _cache_put(key, result)
    return result


def base_weight_transition(s: LayerSet, weight: torch.Tensor) -> torch.Tensor | None:
    if (
        s.form != "input"
        or s.down is None
        or s.up is None
        or weight.ndim != 2
        or s.down.shape[1] != s.up.shape[1]
        or weight.shape[1] != s.down.shape[1]
        or not weight.dtype.is_floating_point
        or weight.device.type == "meta"
    ):
        return None
    with torch.no_grad():
        w = weight.detach().float()
        down = (
            s.down.detach()
            .to(device=w.device, dtype=torch.float32)
            .reshape(-1, s.per_expert, s.down.shape[1])
        )
        up = (
            s.up.detach()
            .to(device=w.device, dtype=torch.float32)
            .reshape(-1, s.per_expert, s.up.shape[1])
        )
        reads = down @ w.T
        writes = up @ w.T
        pair = torch.stack([(reads @ b.T).square().sum((-2, -1)) for b in writes])
        norm = (
            writes.square().sum((-2, -1)).sqrt()[:, None]
            * reads.square().sum((-2, -1)).sqrt()[None, :]
        )
        match = pair.sqrt() / norm.clamp_min(1e-30)
        return match / match.sum(-1, keepdim=True).clamp_min(1e-30)


def transition_resolvent(transition: torch.Tensor, gamma: float) -> torch.Tensor:
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("KILORA_SHARE_TRANSITION_RESOLVENT must be in (0, 1)")
    matrix = transition.float()
    radius = torch.linalg.eigvals(matrix).abs().max().item() if matrix.numel() else 0.0
    if gamma * radius >= 1:
        raise ValueError("transition resolvent does not converge: gamma * spectral radius >= 1")
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    result = (1 - gamma) * torch.linalg.solve(eye - gamma * matrix, eye)
    return result / result.sum(-1, keepdim=True).clamp_min(1e-30)


def share_transition(
    s: LayerSet,
    *,
    states: torch.Tensor | None = None,
    _kind: str | None = None,
    _raw: bool = False,
) -> torch.Tensor | None:
    if s.routing_down is not None and s.form == "input" and import_score(s) == "derived":
        s = dataclasses.replace(
            s, down=s.routing_down, up=s.routing_up, routing_down=None, routing_up=None
        )
    force = _IR_INTERPRET_ALL.get()
    override = s.ir_parts.get("transition") if _IR_ACTIVE_OVERRIDES.get() else None
    interpreted = False
    if force or override is not None:
        from .ir.dispatch import active_overrides, interpreted_part

        needs_interpreter = force or interpreted_part([s]) is not None
        supported = (
            s.form == "input"
            and s.down is not None
            and s.up is not None
            and s.down.shape == s.up.shape
            and (_kind or share_transition_kind()) == "cosine"
            and not _raw
            and not (import_score(s) == "derived" and derived_diffusion())
        )
        if needs_interpreter and supported:
            from .ir.driver import _run

            down = s.down.detach().float().reshape(-1, s.per_expert, s.down.shape[-1])
            up = s.up.detach().float().reshape(-1, s.per_expert, s.up.shape[-1])
            sizes = {"experts": down.shape[0], "per": s.per_expert, "key": down.shape[-1]}
            chosen = {} if override is None else {"transition": override}
            with active_overrides(chosen):
                transition = torch.stack(
                    [
                        _run("transition", "cosine_row", {"down": down, "up": row}, sizes)["row"]
                        for row in up
                    ]
                )
            interpreted = True
        elif needs_interpreter and override is not None:
            raise NotImplementedError("transition override requires a normalized cosine transition")
        else:
            transition = _share_transition_base(s, states=states, _kind=_kind, _raw=_raw)
            if needs_interpreter:
                from .fused import _note

                _note("IR driver refused transition: only normalized cosine is supported")
    else:
        transition = _share_transition_base(s, states=states, _kind=_kind, _raw=_raw)
    if _raw:
        return transition
    value = os.environ.get("KILORA_SHARE_TRANSITION_RESOLVENT", "")
    if not value or float(value) == 0:
        return transition
    gamma = float(value)
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("KILORA_SHARE_TRANSITION_RESOLVENT must be in (0, 1)")
    if transition is None:
        return None
    kind = _kind or share_transition_kind()
    rank = os.environ.get("KILORA_SHARE_TRANSITION_RANK", "8")
    diffusion = derived_diffusion() if import_score(s) == "derived" else 0
    sources = [s.down, s.up]
    if kind == "weighted":
        sources.append(s.share_transition_weighted)
    key = (
        _tensor_key(
            f"transition-resolvent-{gamma}-{kind}-{s.per_expert}-{rank}-{diffusion}", sources
        )
        if states is None
        else None
    )
    hit = None if interpreted else _cache_get(key)
    if hit is not None:
        return hit
    result = transition_resolvent(transition, gamma)
    if not interpreted:
        _cache_put(key, result)
    return result


_FROZEN_DERIVED_KEYS = contextvars.ContextVar("kilora_frozen_derived_keys", default=None)


class DerivedKeyVersions:
    def __init__(self):
        self._version = object()
        self._keys = {}

    def advance(self, version):
        if version != self._version:
            self._version, self._keys = version, {}

    @contextlib.contextmanager
    def write_window(self, version, layers):

        self.advance(version)
        for layer in layers:
            key = id(layer)
            if key not in self._keys or self._keys[key][0] is not layer:
                self._keys[key] = (layer, _derived_keys_current(layer, fresh=True))
        token = _FROZEN_DERIVED_KEYS.set(dict(self._keys))
        try:
            yield
        finally:
            _FROZEN_DERIVED_KEYS.reset(token)


_DERIVED_KEYS_BY_DIGEST: dict = {}
_DERIVED_KEYS_BUDGET = 256 * 2**20


def _derived_keys_bytes() -> int:
    return sum(t.numel() * t.element_size() for v in _DERIVED_KEYS_BY_DIGEST.values() for t in v)


def derived_keys(s: LayerSet):
    frozen = _FROZEN_DERIVED_KEYS.get()
    if frozen is not None:
        hit = frozen.get(id(s))
        if hit is None or hit[0] is not s:
            raise ValueError("layer was not registered before the write window")
        return hit[1]
    return _derived_keys_current(s)


DERIVED_STATS: collections.Counter = collections.Counter()


def _zero_expert(s: LayerSet, n: int, i: int) -> bool:
    down = s.down.reshape(n, s.per_expert, -1)[i]
    up = s.up.reshape(n, s.per_expert, -1)[i]
    return not bool(down.count_nonzero()) and not bool(up.count_nonzero())


def _zero_expert_bases(s: LayerSet, m: int):

    DERIVED_STATS["zero_experts"] += 1
    device = s.down.device
    result = []
    for weight in (s.down, s.up):
        width = min(m, s.per_expert, weight.shape[-1])
        result.append(torch.zeros(width, weight.shape[-1], dtype=torch.float32, device=device))
        result.append(torch.zeros((), dtype=torch.int64, device=device))
    return tuple(result)


EXPERT_KEY_NAMES = ("kin", "rin", "kout", "rout")


def _block_bases(weight: torch.Tensor, per_expert: int, m: int):
    blocks = weight.detach().float().reshape(-1, per_expert, weight.shape[-1])
    width = min(m, per_expert, weight.shape[-1])
    _, values, vh = torch.linalg.svd(blocks, full_matrices=False)
    tol = values.amax(dim=-1, keepdim=True) * max(blocks.shape[-2:]) * torch.finfo(values.dtype).eps
    counts = (values > tol).sum(-1).clamp(max=width)
    mask = torch.arange(width, device=blocks.device)[None, :] < counts[:, None]
    return torch.where(mask[:, :, None], vh[:, :width], 0.0), counts, values


def _expert_key_settings(down: torch.Tensor, up: torch.Tensor, per_expert: int, m: int):
    from . import derived_cache

    return {
        "identity": "strength-free-2",
        "rows": m,
        "backend": derived_cache.backend(down.device),
        "per_expert": per_expert,
        "down_dtype": str(down.dtype),
        "up_dtype": str(up.dtype),
        "down_width": down.shape[-1],
        "up_width": up.shape[-1],
    }


def _expert_bases(down, up, n: int, per_expert: int, batch: list[int], m: int):
    kin, rin, _ = _block_bases(down.reshape(n, per_expert, -1)[batch], per_expert, m)
    kout, rout, _ = _block_bases(up.reshape(n, per_expert, -1)[batch], per_expert, m)
    return [(kin[j], rin[j], kout[j], rout[j]) for j in range(len(batch))]


def derived_rows() -> int:
    m = int(os.environ.get("KILORA_DERIVED_ROWS", "8"))
    if m < 1:
        raise ValueError("KILORA_DERIVED_ROWS must be positive")
    return m


def derived_weights(s: LayerSet):
    from . import derived_cache

    if s.routing_down is not None and s.form == "input":
        s = dataclasses.replace(
            s, down=s.routing_down, up=s.routing_up, routing_down=None, routing_up=None
        )
        if s.routing_source:
            derived_cache.bind(s, s.routing_source)
    return s, s.down, _unscaled_up(s, derived_cache.unit_up(derived_cache.layer_source(s)))


def padded_expert_bases(s: LayerSet, padding: int, m: int):

    s, down, up = derived_weights(s)
    if down.shape[0] != s.per_expert or padding < s.per_expert:
        raise ValueError("padding needs a single expert and a larger rank")
    padded = []
    for weight in (down, up):
        block = torch.zeros(padding, weight.shape[-1], dtype=weight.dtype, device=weight.device)
        block[: weight.shape[0]] = weight.detach()
        padded.append(block)
    entry = _expert_bases(padded[0], padded[1], 1, padding, [0], m)[0]
    return entry, _expert_key_settings(padded[0], padded[1], padding, m)


def _derived_keys_current(s: LayerSet, *, fresh: bool = False, prepare_experts: bool = False):
    if s.routing_down is not None and s.form == "input":
        s = dataclasses.replace(
            s, down=s.routing_down, up=s.routing_up, routing_down=None, routing_up=None
        )
        if s.routing_source:
            from . import derived_cache

            derived_cache.bind(s, s.routing_source)
    m = int(os.environ.get("KILORA_DERIVED_ROWS", "8"))
    if m < 1:
        raise ValueError("KILORA_DERIVED_ROWS must be positive")
    individual_cuda = s.down.device.type == "cuda" and (
        prepare_experts or os.environ.get("KILORA_CONTENT_KEYS", "per_expert") == "per_expert"
    )
    key = _tensor_key(f"derived-keys-{s.per_expert}-{m}-{individual_cuda}", [s.down, s.up])
    hit = None if fresh or prepare_experts else _cache_get(key)
    if hit is not None:
        return hit
    with torch.inference_mode(False), torch.no_grad():
        from . import derived_cache

        source = None if fresh else derived_cache.layer_source(s)
        up_keys = _unscaled_up(s, derived_cache.unit_up(derived_cache.layer_source(s)))
        settings = {
            "keys_from": "unscaled-1",
            "rows": m,
            "per_expert": s.per_expert,
            "backend": derived_cache.backend(s.down.device),
            "down_dtype": str(s.down.dtype),
            "up_dtype": str(s.up.dtype),
            "down_shape": tuple(s.down.shape),
            "up_shape": tuple(s.up.shape),
        }
        if individual_cuda:
            settings["algorithm"] = "individual-cuda"
        if source:
            from . import key_cache

            content_key = (
                source,
                key_cache.cache_dir(),
                s.per_expert,
                m,
                tuple(sorted((k, str(v)) for k, v in settings["backend"].items())),
                str(s.down.dtype),
                str(s.up.dtype),
                tuple(s.down.shape),
                tuple(s.up.shape),
                individual_cuda,
            )
        else:
            digest = hashlib.sha256()
            for weight in (s.down, up_keys):
                cpu = weight.detach().contiguous().cpu()
                digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
                digest.update(cpu.view(torch.uint8).numpy().tobytes())
            content_key = (
                digest.hexdigest(),
                s.per_expert,
                m,
                tuple(sorted((k, str(v)) for k, v in settings["backend"].items())),
                individual_cuda,
            )
        stored = None if prepare_experts else _DERIVED_KEYS_BY_DIGEST.get(content_key)
        hit = None
        if stored is not None:
            hit = tuple(t.to(s.down.device, copy=True) for t in stored)
        names = ("kin", "rin", "kout", "rout", "similarity")
        if hit is None and source and not prepare_experts:
            hit = derived_cache.load(source, "keys", settings, names, s.down.device)
        if hit is None:

            def bases(weight):
                return _block_bases(weight, s.per_expert, m)

            ids = source[1] if isinstance(source, tuple) and len(source) == 2 else None
            n = s.down.shape[0] // s.per_expert
            if (
                ids is not None
                and len(ids) == n
                and (s.down.device.type == "cpu" or individual_cuda)
            ):
                expert_settings = _expert_key_settings(s.down, s.up, s.per_expert, m)
                expert_names = EXPERT_KEY_NAMES
                entries = [
                    derived_cache.load(
                        identity, "expert-keys", expert_settings, expert_names, s.down.device
                    )
                    for identity in ids
                ]
                missing = [i for i, entry in enumerate(entries) if entry is None]
                zero = [i for i in missing if _zero_expert(s, n, i)]
                if zero and (individual_cuda or len(zero) == len(missing)):
                    for i in zero:
                        entries[i] = _zero_expert_bases(s, m)
                        derived_cache.store(
                            ids[i], "expert-keys", expert_settings, expert_names, entries[i]
                        )
                    missing = [i for i in missing if entries[i] is None]
                if missing:
                    batches = [[i] for i in missing] if individual_cuda else [missing]
                    for batch in batches:
                        found = _expert_bases(s.down, up_keys, n, s.per_expert, batch, m)
                        for i, entry in zip(batch, found, strict=True):
                            entries[i] = entry
                            derived_cache.store(
                                ids[i], "expert-keys", expert_settings, expert_names, entries[i]
                            )
                kin, rin, kout, rout = (
                    torch.stack(values) for values in zip(*entries, strict=True)
                )
            else:
                kin, rin, _ = bases(s.down)
                kout, rout, _ = bases(up_keys)

            def similarity(bases, counts):
                overlap = torch.einsum("erd,fsd->efrs", bases, bases).square().sum((-1, -2))
                return overlap / torch.minimum(counts[:, None], counts[None, :]).clamp_min(1)

            similarity_matrix = (similarity(kin, rin) + similarity(kout, rout)) * 0.5
            similarity_matrix.fill_diagonal_(0)
            hit = (kin, rin, kout, rout, similarity_matrix)
            if not prepare_experts:
                derived_cache.store(source, "keys", settings, names, hit)
        if not prepare_experts and content_key not in _DERIVED_KEYS_BY_DIGEST:
            copies = tuple(t.detach().to("cpu", copy=True) for t in hit)
            _DERIVED_KEYS_BY_DIGEST[content_key] = copies
            while _DERIVED_KEYS_BY_DIGEST and _derived_keys_bytes() > _DERIVED_KEYS_BUDGET:
                _DERIVED_KEYS_BY_DIGEST.pop(next(iter(_DERIVED_KEYS_BY_DIGEST)))
    if not prepare_experts:
        _cache_put(key, hit)
    return hit


def _unscaled_up(s: LayerSet, unit: torch.Tensor | None) -> torch.Tensor:
    if unit is None or unit.numel() != s.up.numel() or unit.shape[-1] != s.up.shape[-1]:
        return s.up
    return unit.reshape(s.up.shape).to(device=s.up.device, dtype=s.up.dtype)


def derived_diffusion_matrices(s: LayerSet):
    if s.routing_down is not None and s.form == "input":
        s = dataclasses.replace(
            s, down=s.routing_down, up=s.routing_up, routing_down=None, routing_up=None
        )
    strength = derived_diffusion()
    if not strength or s.down.shape[1] != s.up.shape[1]:
        return None
    key = _tensor_key(f"derived-diffusion-{s.per_expert}-{strength}", [s.down, s.up])
    hit = _cache_get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        coupling = share_transition(s, _kind="cosine", _raw=True).clamp_min(0).clone()
        coupling.fill_diagonal_(0)
        pout = coupling / coupling.sum(-1, keepdim=True).clamp_min(1e-30)
        pin_t = (coupling / coupling.sum(0, keepdim=True).clamp_min(1e-30)).T
        eye = torch.eye(len(coupling), device=coupling.device, dtype=coupling.dtype)
        hit = tuple(
            (1 - strength) * torch.linalg.solve(eye - strength * p, eye) for p in (pout, pin_t)
        )
    _cache_put(key, hit)
    return hit


def derived_exclusion():
    value = float(os.environ.get("KILORA_DERIVED_EXCLUSION", "0"))
    if not math.isfinite(value) or value < 0:
        raise ValueError("KILORA_DERIVED_EXCLUSION must be finite and non-negative")
    return value


def derived_conflict(s):
    if s.routing_down is not None and s.form == "input":
        s = dataclasses.replace(
            s, down=s.routing_down, up=s.routing_up, routing_down=None, routing_up=None
        )
    key = _tensor_key(f"derived-conflict-{s.per_expert}", [s.down, s.up])
    hit = _cache_get(key)
    if hit is None:
        from .keydiag_conflict import overlaps

        with torch.no_grad():
            hit = (-overlaps(s)[0]).clamp_min(0)
            hit.fill_diagonal_(0)
        _cache_put(key, hit)
    return hit


def _mask_derived_logits(s: LayerSet | None, logits: torch.Tensor) -> torch.Tensor:
    if s is None or s.expert_mask is None:
        return logits
    mask = s.expert_mask.to(logits.device).unflatten(-1, (-1, s.per_expert))
    active = mask.ne(0).any(-1)
    excluded = ~active & active.any(-1, keepdim=True)
    return logits.masked_fill(excluded, -1e9)


def derived_logits(s: LayerSet, x: torch.Tensor) -> torch.Tensor:
    return _mask_derived_logits(s, _derived_logits(s, x))


def _derived_logits(s: LayerSet, x: torch.Tensor) -> torch.Tensor:
    kin, rows, _, _, _ = derived_keys(s)
    force = _IR_INTERPRET_ALL.get()
    override = s.ir_parts.get("entry") if _IR_ACTIVE_OVERRIDES.get() else None
    if force or override is not None:
        from .ir.dispatch import active_overrides, interpreted_part

        needs_interpreter = force or interpreted_part([s]) is not None
        supported = x.ndim == 3 and x.dtype == torch.float32 and not derived_diffusion()
        if needs_interpreter and supported:
            from .ir.driver import _run

            sizes = {
                "batch": x.shape[0],
                "tokens": x.shape[1],
                "experts": kin.shape[0],
                "per": kin.shape[1],
                "key": kin.shape[2],
            }

            def scalar(value):
                return torch.as_tensor(value, dtype=torch.float32, device=x.device)

            inputs = {
                "x": x,
                "keys": kin.to(x.device),
                "rows": rows.to(x.device),
                "key_width": scalar(kin.shape[-1]),
                "tau": scalar(s.tau),
                "epsilon": scalar(SCORE_EPS),
                "temperature_in": scalar(1.0),
            }
            chosen = {} if override is None else {"entry": override}
            with active_overrides(chosen):
                return _run("entry", "from_input", inputs, sizes)["logits"]
        if needs_interpreter and override is not None:
            raise NotImplementedError("entry override requires float32 input without diffusion")
        if needs_interpreter:
            from .fused import _note

            _note("IR driver refused entry: unsupported dtype, layout, or diffusion")
    keys = _as(kin, x.dtype).flatten(0, 1)
    proj = (x @ keys.T).float().unflatten(-1, kin.shape[:2])
    energy = torch.linalg.vector_norm(
        x.float() if s.form == "state" else x,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    ).square()
    logits = s.tau * proj.square().sum(-1) / (energy + SCORE_EPS)
    logits = logits / (rows.clamp_min(1).float() / kin.shape[-1])
    matrices = derived_diffusion_matrices(s) if derived_diffusion() else None
    return logits @ matrices[1].T if matrices is not None else logits


def pooled_derived_logits(s, x, target):
    return _mask_derived_logits(s, _pooled_derived_logits(s, x, target))


def _pooled_derived_logits(s, x, target):
    logits = derived_logits(s, x)
    raw = os.environ.get("KILORA_DERIVED_POOL", "1,0,0")
    try:
        weights = tuple(float(v.strip()) for v in raw.split(","))
    except ValueError as exc:
        raise ValueError("KILORA_DERIVED_POOL must contain three finite weights") from exc
    if len(weights) != 3 or not all(math.isfinite(v) for v in weights):
        raise ValueError("KILORA_DERIVED_POOL must contain three finite weights")
    if weights == (1.0, 0.0, 0.0) or logits.ndim < 2:
        return logits
    if s is not None and s.expert_mask is not None:
        logits = _derived_logits(s, x)
    from .region import current_latent_shape, current_layout, current_text_tokens, grid_for

    latent = current_latent_shape()
    layout = current_layout(target)
    if latent is None or len(latent) != 2:
        return logits
    tokens = logits.shape[-2]
    start = 0
    if layout is not None:
        if layout.kind != "image2d" or layout.role == "reference":
            return logits
        if layout.patch is None:
            if layout.text:
                return logits
            grid = grid_for(tokens, latent)
        else:
            grid = layout.grid(latent)
            count = math.prod(grid)
            if count > tokens:
                return logits
            if layout.text == "before":
                padded = math.ceil(count / layout.pad) * layout.pad
                text = tokens - padded
                known = current_text_tokens()
                if layout.pad == 1 and known is not None and 0 < known <= tokens - count:
                    text = known
                start = text
    else:
        grid = grid_for(tokens, latent)
    if grid is None or start + math.prod(grid) > tokens:
        return logits
    count = math.prod(grid)
    image = logits[..., start : start + count, :]
    shape = image.shape
    cells = image.reshape(-1, *grid, shape[-1]).permute(0, 3, 1, 2)
    local = F.avg_pool2d(F.pad(cells, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    mean = cells.mean(dim=(-2, -1), keepdim=True)
    pooled = weights[0] * cells + weights[1] * local + weights[2] * mean
    pooled = pooled.permute(0, 2, 3, 1).reshape(shape)
    if start == 0 and count == tokens:
        return pooled
    result = logits.clone()
    result[..., start : start + count, :] = pooled
    return result


def derived_lift_after():
    value = float(os.environ.get("KILORA_DERIVED_LIFT_AFTER", "0.25"))
    if not math.isfinite(value) or value < 0:
        raise ValueError("KILORA_DERIVED_LIFT_AFTER must be finite and non-negative")
    return value


def _import_gates(ins, coefs, regions, x, prev=None, transitions=None, gamma=0.0, target=""):
    out = []
    for i, (s, c, region) in enumerate(zip(ins, coefs, regions, strict=True)):
        if import_score(s) == "derived":
            logits = pooled_derived_logits(s, x, target)
        else:
            energy = read_energy(s, c)
            logits = s.tau * energy / (energy.sum(-1, keepdim=True) + 1e-12)
        if prev is not None and import_score(s) == "derived" and derived_mode() == "full":
            logits = logits * derived_lift_after()
        if prev is not None and transitions is not None and transitions[i] is not None:
            logits = logits + gamma * (prev[i].float() @ transitions[i] + 1e-6).log()
        if import_score(s) == "derived":
            logits = _mask_derived_logits(s, logits)
        candidate_mask = (
            _candidate_mask(s, x, derived_keys(s)[0], region)
            if 0 < s.candidates < logits.shape[-1]
            else None
        )
        mu = derived_exclusion() if import_score(s) == "derived" else 0.0
        if mu:
            seed = prev[i] if prev is not None else _gate(logits, s, 0.0, region, candidate_mask)
            logits = logits - mu * (seed.float() @ derived_conflict(s))
        if (
            import_score(s) == "derived"
            and derived_mode() == "keys"
            and s.routing_select == "top1"
            and not torch.is_grad_enabled()
        ):
            check_routing(s.routing_scope, s.routing_select)
            choice_logits = region_mean(logits, region) if s.routing_scope == "tensor" else logits
            if candidate_mask is not None:
                choice_logits = choice_logits.masked_fill(~candidate_mask, -torch.inf)
            choice = choice_logits.argmax(-1, keepdim=True)
            gate = torch.zeros_like(choice_logits).scatter(-1, choice, 1.0).expand_as(logits)
        else:
            gate = _gate(logits, s, 0.0, region, candidate_mask)
        out.append(gate)
    return out


def _axis_import_gates(ins, coefs, regions, x, query, prev=None, transitions=None, gamma=0.0):
    logits = []
    candidates = []
    for i, (s, c, region) in enumerate(zip(ins, coefs, regions, strict=True)):
        mode = import_score(s)
        if mode == "derived":
            score = derived_logits(s, x)
            if prev is not None and derived_mode() == "full":
                score = score * derived_lift_after()
        elif mode == "read_share":
            energy = read_energy(s, c)
            score = s.tau * energy / (energy.sum(-1, keepdim=True) + 1e-12)
        else:
            kin, kout = _set_keys(s)
            q = F.linear(query.float(), s.router.float())
            if prev is not None:
                q = out_project(q, kout, prev[i])
            score = _cross_logits(s.tau * norm_energy(q, kin, s.rows_in), s, kin)
            score = _learned_cross_logits(score, s, q, kin)
        if prev is not None and transitions is not None and transitions[i] is not None:
            score = score + gamma * (prev[i].float() @ transitions[i] + 1e-6).log()
        mask = (
            _candidate_mask(s, x, derived_keys(s)[0], region)
            if 0 < s.candidates < score.shape[-1]
            else None
        )
        mu = derived_exclusion() if mode == "derived" else 0.0
        if mu:
            seed = prev[i] if prev is not None else _gate(score, s, 0.0, region, mask)
            score = score - mu * (seed.float() @ derived_conflict(s))
        if mode == "derived":
            score = _mask_derived_logits(s, score)
        logits.append(score)
        candidates.append(mask)
    return _gates_from_logits(logits, ins, True, 0.0, regions, candidates)


def _read_share_gates(
    ins, coefs, regions, noise_scale=0.0, prev=None, transitions=None, gamma=0.0, x=None
):
    out = []
    for i, (st, c, r) in enumerate(zip(ins, coefs, regions, strict=True)):
        energy = read_energy(st, c)
        share = energy / (energy.sum(-1, keepdim=True) + 1e-12)
        logits = st.tau * share
        if prev is not None and transitions is not None and transitions[i] is not None:
            logits = logits + gamma * (prev[i].float() @ transitions[i] + 1e-6).log()
        mask = (
            _candidate_mask(st, x, derived_keys(st)[0], r)
            if x is not None and 0 < st.candidates < logits.shape[-1]
            else None
        )
        out.append(_gate(logits, st, noise_scale, r, mask))
    return out


def _with_read_share(
    gates, by_read, ins, coefs, regions, prev=None, transitions=None, gamma=0.0, x=None
):
    if not any(by_read):
        return gates
    shared = _read_share_gates(
        ins, coefs, regions, prev=prev, transitions=transitions, gamma=gamma, x=x
    )
    return [r if b else g for g, r, b in zip(gates, shared, by_read, strict=True)]


def _import_fold_enabled():
    return os.environ.get("KILORA_IMPORT_FOLD", "1").strip() != "0"


def _import_fold_history(part, scale, steps, iteration, region, hist, up=None):
    count = steps - 1
    change = part if iteration == "path" else part * count
    before = change * scale
    unit = part * scale
    if region is not None:
        before = before * region.to(before.dtype)
        unit = unit * region.to(unit.dtype)
    weight = HISTORY_RATE**count if iteration == "path" else 1 - HISTORY_RATE**count
    hist["before"], hist["m"] = before, unit * weight
    if up is not None and not base_reference():
        up = up.detach().float()
        hist["first_norm"], hist["first_rms"] = change_stats(unit, up, up @ up.t())
    return change


def _run_input_fused(x, y0, sets, once, depth, regs, target, td=None, entry=None, stats=None):
    if _STEP_CARRY and _ablation.carry_steps(False):
        return _run_input(x, y0, sets, once, depth, regs, target, td, entry, stats)
    skip_active = False
    if len(once) == 1 and depth >= 3:
        s = sets[once[0]]
        if import_score(s) == "derived" and derived_mode() == "full":
            from .segments import has_skips, skip_decisions

            skip_active = has_skips(skip_decisions(s))
    if (
        not skip_active
        and len(once) == 1
        and sets[once[0]].direction_core is None
        and depth >= 1
        and not (import_score(sets[once[0]]) == "derived" and derived_exclusion())
        and _fast_path_ok(y0, [sets[once[0]]], 0.0, None, None)
    ):
        fused = _fused_module()
        s = sets[once[0]]
        if (
            fused is not None
            and not _READ_KEEP_DETAIL_OBSERVERS
            and fused.fused_ok(y0, [s], depth, td=td)
        ):
            name = f"{target}#input{once[0]}"
            reg = [regs[once[0]]]
            try:
                return fused.run_fused(
                    y0,
                    [s],
                    depth,
                    False,
                    target,
                    reg,
                    x=x,
                    name=name,
                    td=td,
                    entry=entry,
                    path_stats=stats,
                )
            except Exception as err:
                if not fused.resource_error(err):
                    raise
                fused.mark_too_big(y0, [s])
    return _run_input(x, y0, sets, once, depth, regs, target, td, entry, stats)


def _replay_import_chain(x, y0, sets, once, depth, regs, target, stats, entry=None):
    if len(once) == 1:
        st = sets[once[0]]
        fused = _fused_module()
        if (
            fused is not None
            and not _READ_KEEP_DETAIL_OBSERVERS
            and _fast_path_ok(y0, [st], 0.0, None, None)
            and fused.fused_ok(y0, [st], depth)
        ):
            try:
                return fused.run_fused(
                    y0,
                    [st],
                    depth,
                    False,
                    target,
                    [regs[once[0]]],
                    x=x,
                    name=f"{target}#input{once[0]}",
                    entry=entry,
                    path_stats=stats,
                )
            except Exception as err:
                if not fused.resource_error(err):
                    raise
                fused.mark_too_big(y0, [st])
    return _run_input(x, y0, sets, once, depth, regs, target, entry=entry, stats=stats)


def derived_entries(s, x, target, width, region=None, prior=None, omega=0.0, pairs=()):
    temp = (
        float(os.environ.get("KILORA_DERIVED_ENTRY_TEMP", "1.0"))
        if s.derived_entry_temp is None
        else float(s.derived_entry_temp)
    )
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("KILORA_DERIVED_ENTRY_TEMP must be finite and nonnegative")
    if pairs:
        from .segments import entry_logits

        logits = entry_logits(s, x, pairs)
    else:
        logits = pooled_derived_logits(s, x, target)
    if prior is not None and omega:
        if pairs:
            extra = torch.stack([(prior[..., a] + prior[..., b]) * 0.5 for a, b in pairs], dim=-1)
            prior = torch.cat((prior, extra), dim=-1)
        logits = (1 - omega) * logits + omega * (prior + 1e-6).log()
    if s.routing_scope == "tensor":
        if region is None:
            logits = logits.mean(-2, keepdim=True)
        else:
            weight = region.float()
            logits = (logits * weight).sum(-2, keepdim=True) / weight.sum(
                -2, keepdim=True
            ).clamp_min(1e-30)
    ids = s.expert_ids
    if ids is None:
        ids = torch.arange(s.down.shape[0] // s.per_expert, device=logits.device, dtype=torch.int64)
    if pairs:
        ids = torch.cat(
            (ids.to(logits.device), torch.arange(len(pairs), device=logits.device) + 0x40000000)
        )
    columns = _mix32(ids.to(device=logits.device, dtype=torch.int64).clone())
    if temp == 0:
        scores = logits
    else:
        u = dropout_uniforms(logits.shape, target + "#entry", 1, logits.device, columns)
        noise = -(-u.clamp(1e-7, 1 - 1e-7).log()).log()
        scores = logits / temp + noise
    count = s.down.shape[0] // s.per_expert
    mapping = torch.eye(count, device=logits.device, dtype=logits.dtype)
    if pairs:
        extra = mapping.new_zeros((len(pairs), count))
        for n, (a, b) in enumerate(pairs):
            extra[n, a] += 1
            extra[n, b] += 1
        mapping = torch.cat((mapping, extra))
    candidate_mask = (
        _candidate_mask(s, x, derived_keys(s)[0], region) if 0 < s.candidates < count else None
    )
    if candidate_mask is not None:
        if pairs:
            members = candidate_mask.to(mapping.dtype) @ mapping.T
            candidate_mask = members >= mapping.sum(-1)
        scores = scores.masked_fill(~candidate_mask, -torch.inf)
        width = min(width, s.candidates)
    mu = derived_exclusion()
    if mu:
        penalty = derived_conflict(s)
        if pairs:
            norm = mapping / mapping.sum(-1, keepdim=True)
            penalty = norm @ penalty @ norm.T
        chosen = torch.zeros_like(scores)
        picks = []
        for _ in range(min(width, logits.shape[-1])):
            adjusted = (scores - mu * (chosen @ penalty)).masked_fill(chosen.bool(), -torch.inf)
            pick = adjusted.argmax(-1)
            picks.append(pick)
            chosen = chosen + F.one_hot(pick, logits.shape[-1]).to(chosen.dtype)
        order = torch.stack(picks, dim=-1)
    else:
        order = scores.topk(min(width, logits.shape[-1]), dim=-1).indices
    if s.expert_mask is not None and s.routing_scope == "token":
        active = s.expert_mask.to(scores.device).unflatten(-1, (count, s.per_expert))
        active = active.ne(0).any(-1)
        valid = (active.to(mapping.dtype) @ mapping.T) == mapping.sum(-1)
        valid = valid | ~active.any(-1, keepdim=True)
        best = scores.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
        order = torch.where(valid.gather(-1, order), order, best)
    return [mapping[order[..., i]] for i in range(order.shape[-1])]


def _derived_path_note(stats, gates, logits, active=None):
    if stats is None:
        return
    if stats.get("record_macro"):
        stats.setdefault("routes", []).append(gates.detach().clone())
    if gates.shape[-2] == 1 and logits.shape[-2] != 1:
        logits = logits.mean(-2, keepdim=True)
    if active is not None and "gates" in stats:
        gates_final = torch.where(active.bool(), gates, stats["gates"])
    else:
        gates_final = gates
    stats["gates"] = gates_final / gates_final.sum(-1, keepdim=True).clamp_min(1e-30)
    node = gates.argmax(-1)
    gained = logits.gather(-1, node[..., None])[..., 0] * (gates.amax(-1) > 0)
    stats["score"] = gained if "score" not in stats else stats["score"] + gained
    if "node" in stats:
        same = node == stats["node"]
        stats["converged"] = same if "converged" not in stats else stats["converged"] | same
    stats["node"] = node


def _derived_chain_paths(
    s,
    x,
    y0,
    region,
    target,
    entries,
    depth,
    *,
    prediction=None,
    prefix=None,
    stop_before_last=False,
):
    from .path_walks import select_weights

    dtype = y0.dtype if y0 is not None else prediction["dtype"]
    c0 = _input_read(s, x, dtype).float() if prediction is None else prediction["c"]
    if region is not None:
        c0 = c0 * region.float()
    logits = pooled_derived_logits(s, x, target) if prediction is None else prediction["logits"]
    scale = s.scale.float() if isinstance(s.scale, torch.Tensor) else s.scale
    up = _as(s.up, dtype)
    cross = _import_read_cross(s, [up])
    bounded = _import_depth_settings(s)
    metric = _step_metric(s, c0.device)
    steps, low, frac = _depth_parts(depth)
    width = len(entries)
    states = [{} for _ in entries]
    histories = [{} for _ in entries]
    norm_rms = None
    if uses_adaptive(s):
        norm_rms = _y0_gate_stats(y0) if prediction is None else prediction["gate_stats"]
    gamma = share_transition_gamma(s)
    transition = share_transition(s) if gamma else None
    z = prev = None
    held = c0 * scale if s.carry == "dropped" else None
    carried = [None] * width
    survival = [None] * width
    references = [None] * width
    split_references = [None] * width
    extras = [None] * width
    ends = []
    fused = _fused_module()
    use_fused = y0 is not None and fused is not None and fused.fused_ok(y0, [s], depth)
    begin = 1
    if prefix is not None:
        z, prev = prefix["z"], prefix["prev"]
        carried, survival = list(prefix["carried"]), list(prefix["survival"])
        references = prefix["references"]
        split_references = prefix["split_references"]
        histories = [dict(h) for h in prefix["histories"]]
        states = [dict(st) for st in prefix["states"]]
        begin = steps
    for k in range(begin, steps if stop_before_last else steps + 1):
        reads = [
            c0 if k == 1 else _import_depth_coefs(s, c0, [carried[n]], [up], region, cross)
            for n in range(width)
        ]
        if k == 1:
            route_logits = torch.stack([entry.expand_as(logits) * 20 for entry in entries])
        else:
            route_logits = (logits * derived_lift_after()).expand(width, *logits.shape)
            if transition is not None:
                route_logits = route_logits + gamma * (prev.float() @ transition + 1e-6).log()
        route_logits = _mask_derived_logits(s, route_logits)
        additions = []
        for n, read in enumerate(reads):
            alive = survival[n]
            if alive is not None and s.exact:
                alive = alive.reshape(*alive.shape[:-1], -1, s.per_expert)
                alive = alive.mean(-1).repeat_interleave(s.per_expert, -1)
            additions.append(read if alive is None else read * alive)
        if use_fused:
            try:
                gates, candidate = fused.derived_batched_step(
                    s,
                    torch.stack(additions),
                    route_logits,
                    region,
                    z,
                    entry=torch.stack(entries) if k == 1 else None,
                )
            except Exception as err:
                if not fused.resource_error(err):
                    raise
                fused.mark_too_big(y0, [s])
                use_fused = False
        if not use_fused:
            gates = (
                torch.stack([entry.expand_as(logits) for entry in entries])
                if k == 1
                else _gate(route_logits, s, 0.0, region)
            )
            part = torch.stack(additions) * gates.repeat_interleave(s.per_expert, -1)
            candidate = part if z is None else z + part
        current, output_gates = [], []
        for n, state in enumerate(states):
            note = (
                entries[n]
                if k == 1
                else gates[n][..., :1, :]
                if use_fused and s.routing_scope == "tensor"
                else gates[n]
            )
            _derived_path_note(state, note, logits)
            value = candidate[n]
            if z is not None:
                addition = value - z[n] if use_fused else part[n]
                value = depth_split_step(
                    z[n],
                    addition,
                    bounded,
                    metric,
                    split_references[n],
                    _weight_of(scale),
                    target,
                )
            if s.exact:
                source = x
                if carried[n] is not None and up.shape[-1] == x.shape[-1]:
                    source = source + carried[n].to(y0.dtype) @ up
                layer = s
                if s.expert_inputs is not None:
                    change = source - x
                    layer = dataclasses.replace(
                        s,
                        expert_inputs=[None if v is None else v + change for v in s.expert_inputs],
                    )
                weights = gates[n]
                if survival[n] is not None:
                    alive = survival[n].reshape(*survival[n].shape[:-1], -1, s.per_expert).mean(-1)
                    weights = weights * alive
                extra = (
                    _exact_correction(layer, source, reads[n], weights, region, y0.dtype) * scale
                )
                extras[n] = extra if extras[n] is None else extras[n] + extra
                size = ((value * scale).to(y0.dtype) @ up + extras[n]).float().norm(dim=-1)
            else:
                size = _metric_size(value, metric, _weight_of(scale))
            if k == 1:
                references[n] = size
                split_references[n] = _metric_size(value, metric, _weight_of(scale))
            else:
                ratio = (bounded.radial_budget * references[n] / size.clamp_min(1e-30)).clamp(max=1)
                value = value * ratio[..., None]
                if extras[n] is not None:
                    extras[n] = extras[n] * ratio[..., None]
            current.append(value)
            output_gates.append(
                split_gate(
                    s,
                    value * scale,
                    histories[n],
                    *norm_rms,
                    gates[n],
                    target,
                    0,
                    k,
                    region,
                    k == steps,
                )
                if norm_rms is not None
                else None
            )
            if k < steps:
                change = value * scale
                keep = _read_dropout_for_set(
                    change, c0, s.per_expert, target, k, column_keys(s), s.key
                )
                if s.carry == "dropped":
                    old = held if k == 1 else carried[n]
                    carried[n] = (
                        change
                        if _ablation.read_dropout_off()
                        else old + (1 - keep) * (change - old)
                    )
                else:
                    carried[n] = keep * change
                survival[n] = keep if survival[n] is None else survival[n] * keep
        z = torch.stack(current)
        prev = gates
        if k == steps or (frac and k == low):
            ends.append((z.clone(), output_gates, [e for e in extras]))
    if stop_before_last:
        return (
            None,
            None,
            None,
            {
                "z": z,
                "prev": prev,
                "carried": carried,
                "survival": survival,
                "references": references,
                "split_references": split_references,
                "histories": histories,
                "states": states,
            },
        )
    for state in states:
        state.setdefault("converged", torch.zeros_like(state["score"], dtype=torch.bool))
    weights = select_weights(states, s.path_select)
    if use_fused:
        prepared = fused._prepared(s, y0.dtype)
        paths = []
        for n in range(width):
            outputs = []
            for zs, gates, exact in ends:
                output = fused._input_output(
                    y0, zs[n] * scale, gates[n], prepared, y0.shape[:-1], y0.dtype
                )
                if exact[n] is not None:
                    extra = exact[n]
                    if gates[n] is not None:
                        extra = extra * gates[n][0].to(y0.dtype) * gates[n][1].to(y0.dtype)
                    output = output + extra
                outputs.append(output)
            paths.append(
                outputs[0] if len(outputs) == 1 else outputs[0] + frac * (outputs[1] - outputs[0])
            )
        change = sum((p - y0) * weights[..., n : n + 1].to(p.dtype) for n, p in enumerate(paths))
        return change, states, weights
    groups = []
    exact_delta = 0
    for end, (zs, gates, exact) in enumerate(ends):
        blend = (1 - frac if end == 0 else frac) if frac else 1.0
        for n in range(width):
            a, b = gates[n] if gates[n] is not None else (None, None)
            coef = zs[n] * scale * weights[..., n : n + 1] * blend
            if a is not None:
                coef = coef * a.float()
            for group in groups:
                if (b is None and group[1] is None) or (
                    b is not None
                    and group[1] is not None
                    and torch.allclose(b.float(), group[1].float(), rtol=1e-3, atol=0)
                ):
                    group[0] = group[0] + coef
                    break
            else:
                groups.append([coef, b])
            if exact[n] is not None:
                extra = exact[n]
                if a is not None:
                    extra = extra * a * b
                exact_delta = exact_delta + extra * weights[..., n : n + 1] * blend
    write_up = _as(s.up, torch.float32)
    delta = sum(
        (coef @ write_up) if b is None else (coef @ write_up) * b.float() for coef, b in groups
    )
    return (delta + exact_delta).to(y0.dtype), states, weights


def _derived_batched_paths(s, x, y0, region, target, entries, depth):
    from .path_walks import select_weights

    if not target.rsplit("#input", 1)[-1].isdigit() or "#input" not in target:
        target = f"{target}#input0"
    if depth > 1 and _import_depth_chain(s):
        return _derived_chain_paths(s, x, y0, region, target, entries, depth)
    width = len(entries)
    c0 = _input_read(s, x, y0.dtype).float()
    if region is not None:
        c0 = c0 * region.float()
    logits = pooled_derived_logits(s, x, target)
    transition_gamma = share_transition_gamma(s)
    transition = share_transition(s) if transition_gamma else None
    scale = s.scale.float() if isinstance(s.scale, torch.Tensor) else s.scale
    steps, low, frac = _depth_parts(depth) if depth > 1 else (1, 1, 0.0)
    states = [{} for _ in entries]
    up = _as(s.up, torch.float32)
    gate_stats = _y0_gate_stats(y0) if uses_adaptive(s) else None
    gate_set = dataclasses.replace(s, up=up) if gate_stats is not None else s
    gram = up @ up.t() if gate_stats is not None else None
    histories = [{"adaptive_gram": gram} for _ in entries]
    prev = z = None
    exact_weights = None
    ends = []
    fused = _fused_module()
    use_fused = fused is not None and fused.fused_ok(y0, [s], depth)
    fused_c0 = c0.expand(width, *c0.shape).contiguous() if use_fused else None
    precise = use_fused and os.environ.get("KILORA_DERIVED_BATCHED", "1") != "fast"
    for k in range(1, steps + 1):
        if k == 1:
            route_logits = torch.stack([entry.expand_as(logits) * 20 for entry in entries])
        else:
            route_logits = (logits * derived_lift_after()).expand(width, *logits.shape)
            if transition is not None:
                route_logits = (
                    route_logits + transition_gamma * (prev.float() @ transition + 1e-6).log()
                )
        route_logits = _mask_derived_logits(s, route_logits)
        if use_fused:
            try:
                gates, z = fused.derived_batched_step(
                    s,
                    fused_c0,
                    route_logits,
                    region,
                    z,
                    scale if precise else 1.0,
                    entry=torch.stack(entries) if k == 1 else None,
                )
            except Exception as err:
                if not fused.resource_error(err):
                    raise
                fused.mark_too_big(y0, [s])
                use_fused = False
                if precise and z is not None:
                    return _derived_batched_paths(s, x, y0, region, target, entries, depth)
                precise = False
        if not use_fused:
            gates = (
                torch.stack([entry.expand_as(logits) for entry in entries])
                if k == 1
                else _gate(route_logits, s, 0.0, region)
            )
            part = c0 * gates.float().repeat_interleave(s.per_expert, dim=-1)
            z = part if z is None or s.iteration == "path" else z + part
        prev = gates
        if s.exact:
            exact_weights = (
                gates.clone()
                if exact_weights is None or s.iteration == "path"
                else exact_weights + gates
            )
        output_gates = []
        for n, st in enumerate(states):
            if k == 1:
                note_gate = entries[n]
            elif use_fused and s.routing_scope == "tensor":
                note_gate = gates[n][..., :1, :]
            else:
                note_gate = gates[n]
            _derived_path_note(st, note_gate, logits)
            if gate_stats is not None:
                output_gates.append(
                    split_gate(
                        gate_set,
                        z[n] if precise else z[n] * scale,
                        histories[n],
                        *gate_stats,
                        gates[n],
                        target,
                        0,
                        k,
                        region,
                        k == steps,
                    )
                )
        if k == steps or (frac and k == low):
            ends.append(
                (
                    z.clone(),
                    output_gates,
                    1 - frac if k == low and frac else frac if frac else 1.0,
                    None if exact_weights is None else exact_weights.clone(),
                )
            )
    for state in states:
        state.setdefault("converged", torch.zeros_like(state["score"], dtype=torch.bool))
    weights = select_weights(states, s.path_select)
    if precise and use_fused:
        prepared = fused._prepared(s, y0.dtype)
        paths = []
        for n in range(width):
            outputs = [
                fused._input_output(
                    y0, zs[n], gates[n] if gates else None, prepared, y0.shape[:-1], y0.dtype
                )
                + (
                    _derived_exact_extra(
                        s, x, c0, ew[n], region, scale, gates[n] if gates else None, y0.dtype
                    )
                    if ew is not None
                    else 0
                )
                for zs, gates, _, ew in ends
            ]
            path = (
                outputs[0] if len(outputs) == 1 else outputs[0] + frac * (outputs[1] - outputs[0])
            )
            paths.append(path)
        change = sum(
            (path - y0) * weights[..., n : n + 1].to(path.dtype) for n, path in enumerate(paths)
        )
        return change, states, weights
    groups = []
    for zs, gates, blend, _ in ends:
        for n in range(width):
            a, b = gates[n] if gates else (None, None)
            coef = zs[n] * scale * weights[..., n : n + 1] * blend
            if a is not None:
                coef = coef * a.float()
            for group in groups:
                if (b is None and group[1] is None) or (
                    b is not None
                    and group[1] is not None
                    and torch.allclose(b.float(), group[1].float(), rtol=1e-3, atol=0)
                ):
                    group[0] = group[0] + coef
                    break
            else:
                groups.append([coef, b])
    delta = sum((coef @ up) if b is None else (coef @ up) * b.float() for coef, b in groups)
    if s.exact:
        for _, gates, blend, ew in ends:
            for n in range(width):
                extra = _derived_exact_extra(
                    s, x, c0, ew[n], region, scale, gates[n] if gates else None, y0.dtype
                )
                delta = delta + extra * weights[..., n : n + 1] * blend
    return delta.to(y0.dtype), states, weights


def _derived_exact_extra(s, x, coefs, weights, region, scale, gate, dtype):
    extra = _exact_correction(s, x, coefs, weights, region, dtype) * scale
    if gate is not None:
        extra = extra * gate[0] * gate[1]
    return extra


def _run_derived_walks(x, y0, sets, once, depth, regs, target, td, index, macro_allowed, progress):
    from .path_walks import select_weights

    out = y0
    for i in once:
        s = sets[i]
        if import_score(s) != "derived":
            out = _run_input_fused(x, out, sets, [i], depth, regs, target, td)
            continue
        omega = float(os.environ.get("KILORA_DERIVED_CARRY", "0.5"))
        if not math.isfinite(omega) or not 0 <= omega <= 1:
            raise ValueError("KILORA_DERIVED_CARRY must be between 0 and 1")
        anchor = int(os.environ.get("KILORA_DERIVED_ANCHOR", "0"))
        if anchor < 0:
            raise ValueError("KILORA_DERIVED_ANCHOR must be nonnegative")
        cache = _DERIVED_CARRY[-1] if _DERIVED_CARRY and omega > 0 else None
        ids = (
            None
            if cache is None or s.expert_ids is None
            else tuple(s.expert_ids.detach().cpu().tolist())
        )
        key = (s.key, s.axis, ids)
        record = cache.get(key) if cache is not None else None
        first_block = (
            _DERIVED_FIRST_BLOCK[-1].setdefault(key, index) == index
            if _DERIVED_FIRST_BLOCK
            else record is None or record.get("first_block", index) == index
        )
        shape = x.shape[:-1]
        compatible = record is None or (
            record["shape"] == shape and record["experts"] == s.down.shape[0] // s.per_expert
        )
        first_in_block = record is not None and record["block"] != index
        anchored = anchor > 0 and (index + 1) % anchor == 0 and first_in_block
        prior = record["gates"] if record is not None and compatible else None
        if prior is not None:
            spread = float(os.environ.get("KILORA_DERIVED_CARRY_SPREAD", "0.25"))
            if not math.isfinite(spread) or spread < 0:
                raise ValueError("KILORA_DERIVED_CARRY_SPREAD must be finite and non-negative")
            if spread:
                similarity = derived_keys(s)[4]
                cross = (
                    torch.eye(similarity.shape[0], device=similarity.device) + spread * similarity
                )
                prior = prior @ (cross / cross.sum(-1, keepdim=True))
        from .segments import has_skips, skip_decisions

        skip_active = depth >= 3 and has_skips(skip_decisions(s))
        nofold = (
            os.environ.get("KILORA_DERIVED_FREEZE_NOFOLD") == "1"
            and os.environ.get("KILORA_DERIVED_MACRO", "1") == "0"
        )
        macro = (
            macro_allowed
            and not _ablation.force_loop()
            and not (_STEP_CARRY and _ablation.carry_steps(False))
            and not (0 < s.candidates < s.keys_in.shape[0])
            and not skip_active
            and derived_mode() == "full"
            and (nofold or os.environ.get("KILORA_DERIVED_MACRO", "1") != "0")
            and _route_freeze.active([s], progress)
            and not torch.is_grad_enabled()
            and td is None
            and len(once) == 1
            and s.aggregation == "accumulate"
            and learned_kind(s) == "none"
            and not (
                _ROUTES
                or _SHARES
                or _DIRECTION_SHARES
                or _GATE_STATS
                or _PROBES
                or _route_freeze.RECORDS
            )
        )
        macro_key = ("derived_nofold" if nofold else "derived_macro", target, i)
        tensor_key = _tensor_key("macro", [s.down, s.up]) if macro else None
        macro = macro and tensor_key is not None
        if macro:
            cache = None
            prior = None
        signature = (
            tuple(x.shape),
            x.device,
            y0.device,
            y0.dtype,
            tuple(y0.shape),
            s.gate_mode,
            s.gate_kappa,
            s.gate_head_limit,
            s.gate_coherence,
            depth,
            s.iteration,
            s.path_width,
            s.path_select,
            s.candidates,
            s.per_expert,
            import_fixed_read(),
            tensor_key,
        )
        saved = _route_freeze._CACHES[-1].get(macro_key) if macro else None
        folded = saved is not None and saved["signature"] == signature
        if macro:
            _route_freeze.note_fold(folded, nofold)
        if folded and (nofold or saved.get("chain")):
            paths = []
            for state in saved["paths"]:
                stats = {
                    "forced_routes": state["routes"],
                    "forced_gates": state.get("output_gates"),
                    "forced_read_keeps": state.get("read_keeps"),
                }
                paths.append(
                    _replay_import_chain(
                        x, y0, sets, [i], depth, regs, target, stats, state["routes"][0]
                    )
                    if saved.get("chain")
                    else _run_input_fused(
                        x, y0, sets, [i], depth, regs, target, td, state["routes"][0], stats
                    )
                )
            out = out + sum(
                (path - y0) * saved["path_weights"][..., n : n + 1].to(path.dtype)
                for n, path in enumerate(paths)
            )
            continue
        if folded:
            coef = _input_read(s, x, y0.dtype).float()
            if regs[i] is not None:
                coef = coef * regs[i].float()
            scale = s.scale.float() if isinstance(s.scale, torch.Tensor) else s.scale
            gates = saved.get("output_gates")
            if gates is None:
                weighted = coef * saved["weight"].repeat_interleave(s.per_expert, dim=-1) * scale
                delta = weighted @ _as(s.up, torch.float32)
                if s.exact:
                    delta = (
                        delta
                        + _exact_correction(s, x, coef, saved["weight"], regs[i], torch.float32)
                        * scale
                    )
            else:
                up = _as(s.up, torch.float32)
                delta = sum(
                    ((coef * w * scale) @ up) * b.float()
                    for w, b in zip(saved["weight"], gates, strict=True)
                )
                if s.exact:
                    delta = delta + sum(
                        _exact_correction(
                            s, x, coef, w[..., :: s.per_expert], regs[i], torch.float32
                        )
                        * scale
                        * b.float()
                        for w, b in zip(saved["weight"], gates, strict=True)
                    )
            out = out + delta.to(y0.dtype)
            if cache is not None and compatible:
                cache[key] = {
                    "shape": shape,
                    "experts": saved["gates"].shape[-1],
                    "block": index,
                    "first_block": index if record is None else record.get("first_block", index),
                    "gates": saved["gates"],
                }
            continue
        pairs = ()
        if first_block and derived_mode() == "full":
            from . import segments

            pairs = segments.prepare(s)
        if prior is not None and not anchored and not pairs:
            scores = prior
            candidate_mask = (
                _candidate_mask(s, x, derived_keys(s)[0], regs[i])
                if 0 < s.candidates < prior.shape[-1]
                else None
            )
            if candidate_mask is not None:
                scores = scores.masked_fill(~candidate_mask, -torch.inf)
            entry_width = (
                min(s.path_width, s.candidates) if candidate_mask is not None else s.path_width
            )
            mu = derived_exclusion()
            if mu:
                chosen = torch.zeros_like(scores)
                picks = []
                for _ in range(min(entry_width, scores.shape[-1])):
                    adjusted = (scores - mu * (chosen @ derived_conflict(s))).masked_fill(
                        chosen.bool(), -torch.inf
                    )
                    pick = adjusted.argmax(-1)
                    picks.append(pick)
                    chosen = chosen + F.one_hot(pick, scores.shape[-1]).to(scores.dtype)
                order = torch.stack(picks, dim=-1)
            else:
                order = scores.topk(min(entry_width, prior.shape[-1]), dim=-1).indices
            entries = [
                F.one_hot(order[..., j], prior.shape[-1]).to(prior.dtype)
                for j in range(order.shape[-1])
            ]
        else:
            entries = derived_entries(
                s, x, f"{target}#input{i}", s.path_width, regs[i], prior, omega, pairs
            )
        walk_depth = depth
        if prior is not None:
            later_depth = int(
                os.environ.get("KILORA_DERIVED_LATER_DEPTH", "1" if derived_diffusion() else "2")
            )
            if later_depth < 1:
                raise ValueError("KILORA_DERIVED_LATER_DEPTH must be positive")
            walk_depth = min(depth, later_depth)
        from . import linear_jump

        fused = _fused_module()
        batched = (
            os.environ.get("KILORA_DERIVED_BATCHED", "1") != "0"
            and not _ablation.force_loop()
            and not (_STEP_CARRY and _ablation.carry_steps(False))
            and not macro
            and not (0 < s.candidates < s.keys_in.shape[0])
            and not pairs
            and not skip_active
            and not linear_jump._RECORDS
            and not (fused is not None and fused._TRACE)
            and not s.jump_fits
            and not (y0.is_cuda and torch.cuda.is_current_stream_capturing())
            and derived_mode() == "full"
            and (
                not y0.is_cuda
                or os.environ.get("KILORA_DERIVED_BATCHED", "1") == "fast"
                or (
                    fused is not None
                    and fused.fused_ok(y0, [s], walk_depth)
                    and not (
                        uses_adaptive(s)
                        and fused.adaptive_kernel()
                        and s.up.shape[0] <= fused._COLUMN_BLOCK
                    )
                )
            )
            and td is None
            and not torch.is_grad_enabled()
            and not derived_exclusion()
            and s.aggregation == "accumulate"
            and learned_kind(s) == "none"
            and not (
                _ROUTES
                or _SHARES
                or _DIRECTION_SHARES
                or _GATE_STATS
                or _PROBES
                or _route_freeze.RECORDS
            )
        )
        speculate_ok = (
            batched
            and not s.exact
            and _SPECULATE
            and isinstance(walk_depth, int)
            and walk_depth >= 2
            and regs[i] is None
            and s.expert_mask is None
            and s.expert_inputs is None
            and s.gate_head_limit is None
            and isinstance(s.scale, (int, float))
            and not s.fixed_point_stop
        )
        if speculate_ok:
            speculative = _SPECULATE[-1]
            skey = (target, i)
            signature = (
                tuple(x.shape),
                x.device,
                y0.dtype,
                walk_depth,
                len(entries),
                s.per_expert,
                s.routing_select,
                s.routing_scope,
                s.iteration,
                uses_adaptive(s),
                s.scale,
                _tensor_key("speculate", [s.down, s.up]),
                import_fixed_read(),
            )
            previous = speculative.get(skey)
            c = _input_read(s, x, y0.dtype).float()
            logits = derived_logits(s, x)
            gamma = share_transition_gamma(s)
            transition = share_transition(s) if gamma else None
            if transition is None:
                gamma = 0.0
            if depth > 1 and _import_depth_chain(s):
                prefix = (
                    previous.get("chain_prefix")
                    if previous is not None and previous.get("signature") == signature
                    else None
                )
                if prefix is None:
                    change, states, weights = _derived_batched_paths(
                        s, x, y0, regs[i], f"{target}#input{i}", entries, walk_depth
                    )
                else:
                    change, states, weights = _derived_chain_paths(
                        s, x, y0, regs[i], f"{target}#input{i}", entries, walk_depth, prefix=prefix
                    )
            elif (
                previous is not None
                and "prev" in previous
                and previous.get("signature") == signature
            ):
                route = logits * derived_lift_after()
                if transition is not None:
                    route = route + gamma * (previous["prev"] @ transition + 1e-6).log()
                gates = _gate(route, s, 0.0)
                part = c.unsqueeze(0) * gates.repeat_interleave(s.per_expert, -1)
                z = part if s.iteration == "path" else previous["z"] + part
                states = []
                for n in range(len(entries)):
                    state = {name: value[n] for name, value in previous["state"].items()}
                    note = gates[n][..., :1, :] if s.routing_scope == "tensor" else gates[n]
                    _derived_path_note(state, note, logits)
                    state.setdefault(
                        "converged", torch.zeros_like(state["score"], dtype=torch.bool)
                    )
                    states.append(state)
                weights = select_weights(states, s.path_select)
                scale = s.scale
                up = _as(s.up, torch.float32)
                if uses_adaptive(s):
                    norm, rms = _y0_gate_stats(y0)
                    gate_set = dataclasses.replace(s, up=up)
                    gram = up @ up.t()
                    change = 0
                    for n in range(len(entries)):
                        hist = {
                            "m": previous["history"][n],
                            "before": previous["z"][n] * scale,
                            "adaptive_gram": gram,
                            **{
                                key: previous[key][n]
                                for key in ("first_norm", "first_rms")
                                if key in previous
                            },
                        }
                        a, b = split_gate(
                            gate_set,
                            z[n] * scale,
                            hist,
                            norm,
                            rms,
                            gates[n],
                            target,
                            0,
                            walk_depth,
                            None,
                            True,
                        )
                        coef = z[n] * scale * weights[..., n : n + 1] * a.float()
                        change = change + (coef @ up) * b.float()
                    change = change.to(y0.dtype)
                else:
                    coef = sum(z[n] * weights[..., n : n + 1] for n in range(len(entries)))
                    change = ((coef * scale) @ up).to(y0.dtype)
            else:
                change, states, weights = _derived_batched_paths(
                    s, x, y0, regs[i], f"{target}#input{i}", entries, walk_depth
                )
            speculative[skey] = {
                "signature": signature,
                "chain": depth > 1 and _import_depth_chain(s),
                "set": s,
                "target": f"{target}#input{i}",
                "dtype": y0.dtype,
                "gate_stats": _y0_gate_stats(y0) if uses_adaptive(s) else None,
                "c": c.detach(),
                "logits": logits.detach(),
                "entry": torch.stack([entry.detach() for entry in entries]),
                "depth": walk_depth,
                "per_expert": s.per_expert,
                "select": s.routing_select,
                "scope": s.routing_scope,
                "iteration": s.iteration,
                "gamma": gamma,
                "transition": transition,
                "adaptive": uses_adaptive(s),
                "scale": s.scale,
            }
        elif batched:
            change, states, weights = _derived_batched_paths(
                s, x, y0, regs[i], f"{target}#input{i}", entries, walk_depth
            )
        else:
            paths, states = [], []
            for walk_index, entry in enumerate(entries):
                stats = {"ablation_walk": walk_index}
                if macro:
                    stats["record_macro"] = True
                paths.append(
                    (_run_input if pairs else _run_input_fused)(
                        x, y0, sets, [i], walk_depth, regs, target, td, entry, stats
                    )
                )
                stats.setdefault("converged", torch.zeros_like(stats["score"], dtype=torch.bool))
                states.append(stats)
            weights = select_weights(states, s.path_select)
            change = sum(
                (path - y0) * weights[..., n : n + 1].to(path.dtype) for n, path in enumerate(paths)
            )
        out = out + change
        if macro and all(
            st.get("routes") and (not uses_adaptive(s) or st.get("output_gates")) for st in states
        ):
            steps, low, frac = _depth_parts(walk_depth) if walk_depth > 1 else (1, 1, 0.0)
            path_weights = []
            output_gates = []
            for n, st in enumerate(states):
                routes = st["routes"]

                def route_at(end, routes=routes, iteration=s.iteration):
                    return routes[end - 1] if iteration == "path" else sum(routes[:end])

                if uses_adaptive(s):
                    ends = [(low, 1 - frac), (steps, frac)] if frac else [(steps, 1.0)]
                    for end, blend in ends:
                        a, b = st["output_gates"][end - 1]
                        w = route_at(end).repeat_interleave(s.per_expert, dim=-1)
                        path_weights.append(w * a.float() * weights[..., n : n + 1] * blend)
                        output_gates.append(b.detach())
                else:
                    route = (
                        route_at(steps)
                        if not frac
                        else (route_at(low) * (1 - frac) + route_at(steps) * frac)
                    )
                    path_weights.append(route * weights[..., n : n + 1])
            if output_gates and all(
                torch.allclose(output_gates[0].float(), b.float(), rtol=1e-3, atol=0)
                for b in output_gates[1:]
            ):
                path_weights = [sum(path_weights)]
                output_gates = output_gates[:1]
            folded = path_weights if output_gates else sum(path_weights)
            final_gates = sum(
                st["gates"].expand(*shape, -1) * weights[..., n : n + 1]
                for n, st in enumerate(states)
            )
            _route_freeze._CACHES[-1][macro_key] = {
                "signature": signature,
                "chain": depth > 1 and _import_depth_chain(s),
                "paths": states if nofold or (depth > 1 and _import_depth_chain(s)) else None,
                "path_weights": (
                    weights.detach() if nofold or (depth > 1 and _import_depth_chain(s)) else None
                ),
                "weight": [w.detach() for w in folded] if output_gates else folded.detach(),
                "output_gates": output_gates or None,
                "gates": final_gates.detach(),
            }
        if cache is not None and compatible:
            gates = sum(
                st["gates"].expand(*shape, -1) * weights[..., n : n + 1]
                for n, st in enumerate(states)
            )
            cache[key] = {
                "shape": shape,
                "experts": gates.shape[-1],
                "block": index,
                "first_block": index if record is None else record.get("first_block", index),
                "gates": gates.detach(),
            }
    return out


def _run_input_walks(x, y0, ins, depth, regs, target):
    if isinstance(depth, int):
        return _run_path(y0, ins, depth, False, 0.0, None, target, regs, x=x)
    low = math.floor(depth)
    hi = _run_path(y0, ins, low + 1, False, 0.0, None, target, regs, x=x)
    lo = _run_path(y0, ins, low, False, 0.0, None, target, regs, x=x) if low else y0
    return lo + (depth - low) * (hi - lo)


def _log_direction_keep(st, z, base, name):
    if st.direction_core is None or not st.fine:
        return
    with torch.no_grad():
        z = z.detach().float()
        shape = (*z.shape[:-1], -1, st.per_expert)
        base = base.detach().float().reshape(shape)
        sigma = base.square().mean(-1, keepdim=True).sqrt() + DROPOUT_DELTA
        mean = keep_probability(z.reshape(shape), sigma).mean().item()
    logging.getLogger(__name__).info(
        "direction fine mode %s: last-step mean keep probability %.6f", name, mean
    )


def import_fixed_read() -> bool:
    return os.environ.get("KILORA_IMPORT_FIXED_READ", "").strip() == "1"


def _import_depth_chain(st):
    return (
        (st.form == "input" or import_score(st) == "derived")
        and st.direction_core is None
        and st.iteration == "accumulate"
        and not import_fixed_read()
    )


IMPORT_DEPTH_BUDGET = 2.0


def _import_depth_settings(st):
    return (
        st if st.radial_budget > 0 else dataclasses.replace(st, radial_budget=IMPORT_DEPTH_BUDGET)
    )


def _import_read_cross(st, ups):
    key = None if torch.is_grad_enabled() else _tensor_key("import-read-cross", [st.down, *ups])
    hit = _cache_get(key)
    if hit is not None:
        return hit
    result = [
        up.float() @ st.down.float().t() if up.shape[-1] == st.down.shape[-1] else None
        for up in ups
    ]
    _cache_put(key, result)
    return result


def _import_depth_coefs(st, base, carried, ups, region, cross=None):

    added = None
    cross = _import_read_cross(st, ups) if cross is None else cross
    for value, mapping in zip(carried, cross, strict=True):
        if value is not None and mapping is not None:
            change = value.float() @ mapping
            added = change if added is None else added + change
    if added is None:
        return base
    if st.expert_mask is not None:
        added = added * st.expert_mask.to(added.dtype)
    if region is not None:
        added = added * region.float()
    return base + added


def _exact_correction(st, x, coefs, weights, region, dtype):
    from .exact_lycoris import apply_exact

    correction = None
    for expert, components in st.exact.items():
        start = expert * st.per_expert
        end = start + st.per_expert
        value = (
            x
            if st.expert_inputs is None or st.expert_inputs[expert] is None
            else st.expert_inputs[expert]
        )
        exact = apply_exact(components, value.to(dtype))
        if st.expert_mask is not None:
            exact = exact * st.expert_mask[..., start : start + 1].to(dtype)
        if region is not None:
            exact = exact * region.to(dtype)
        approx = coefs[..., start:end].to(dtype) @ st.up[start:end].to(dtype)
        part = (exact - approx) * weights[..., expert : expert + 1].to(dtype)
        correction = part if correction is None else correction + part
    return correction


def _run_input(
    x,
    y0,
    sets,
    once,
    depth,
    regs,
    target,
    td=None,
    entry=None,
    stats=None,
    *,
    interpret=False,
    grad_steps=None,
    noise_scale=0.0,
    query_policy=None,
    shared=False,
    stop=None,
):
    if interpret:
        from .ir.driver import _loop_output, _read_mask, _run, _split_gate_grad

    dtype = y0.dtype
    ins = [sets[i] for i in once]
    rg = [regs[i] for i in once]
    joint_derived = _derived_state_sets(ins)
    names = [
        f"{target}#input{i}" if st.form == "input" else _set_target(target, st, i)
        for i, st in zip(once, ins, strict=True)
    ]
    gate_names = [
        target if st.form == "state" else name for st, name in zip(ins, names, strict=True)
    ]
    gate_indices = [i if st.form == "state" else 0 for i, st in zip(once, ins, strict=True)]
    coefs = []
    direction_queries = []
    for st, r in zip(ins, rg, strict=True):
        if st.direction_core is not None:
            q = F.linear(x.to(dtype), _as(st.router, dtype)).float()
            c = F.linear(q, st.direction_core.float())
            if st.expert_mask is not None:
                c = c * st.expert_mask.to(c.dtype)
            direction_queries.append(q)
        else:
            c = (
                _state_read(st, y0, dtype) if st.form == "state" else _input_read(st, x, dtype)
            ).float()
            direction_queries.append(None)
        coefs.append(c * r.float() if r is not None else c)
    scales = [st.scale.float() if isinstance(st.scale, torch.Tensor) else st.scale for st in ins]
    ups = [_as(st.up, dtype) for st in ins]
    steps, low, frac = _depth_parts(depth) if depth > 1 else (1, 1, 0.0)
    if td is not None:
        steps, low, frac = td.steps, td.steps, 0.0
    split = [learned_kind(st) == "separable" or uses_adaptive(st) for st in ins]
    blend_coefficients = (
        td is not None
        and not any(split)
        and not any(joint_derived and learned_kind(st) in ("neuron", "lowrank") for st in ins)
        and all(st.expert_offsets is None for st in ins)
        and not any(st.exact for st in ins)
    )
    if any(split):
        y0_norm, y0_rms = _y0_gate_stats(y0)
    hists: list = [{} for _ in ins]
    elementwise = [joint_derived and learned_kind(st) in ("neuron", "lowrank") for st in ins]
    previous_delta = [torch.zeros_like(y0) if on else None for on in elementwise]
    if any(elementwise):
        with torch.no_grad():
            gate_rms = y0.float().square().mean(-2, keepdim=True).sqrt() + DROPOUT_DELTA

    chain_extras: list = [None] * len(ins)

    def correction(i, st, weights):
        if not st.exact:
            return None
        if chain_extras[i] is not None:
            return chain_extras[i]
        return _exact_correction(st, x, coefs[i], weights, rg[i], dtype) * scales[i]

    def gated_change(i, st, gate, z, k, weights, output):
        last = k == steps if td is None else td.last(k)
        act = None if td is None else td.active(k)
        args = (y0_norm, y0_rms, gate, gate_names[i], gate_indices[i], k, rg[i], last)
        forced = stats.get("forced_gates") if stats is not None else None
        if interpret:
            a, b = _split_gate_grad(st, z, hists[i], *args, active=act)
        else:
            a, b = forced[k - 1] if forced else split_gate(st, z, hists[i], *args, active=act)
        if stats is not None and stats.get("record_macro"):
            stats.setdefault("output_gates", []).append((a.detach(), b.detach()))
        from . import fused

        if interpret:
            return _loop_output(output, z, ups[i], (a, b), dtype)
        delta = fused.token_gate(z, a, dtype) @ ups[i]
        extra = correction(i, st, weights)
        if extra is not None:
            delta = delta + extra * a.to(dtype)
        return fused.channel_change(delta, b, output)

    by_read = [import_score(st) in ("read_share", "derived") for st in ins]
    has_derived = any(import_score(st) == "derived" for st in ins)
    gamma = share_transition_gamma(ins[0]) if any(by_read) and not joint_derived else 0.0
    transitions = [
        share_transition(st) if gamma and b else None for st, b in zip(ins, by_read, strict=True)
    ]
    chains = [steps > 1 and _import_depth_chain(st) for st in ins]
    read_cross = [
        _import_read_cross(st, ups) if on else None for st, on in zip(ins, chains, strict=True)
    ]
    fold = (
        not joint_derived
        and not any(chains)
        and entry is None
        and not _ablation.force_loop()
        and not (_STEP_CARRY and _ablation.carry_steps(False))
        and not any(depth_split_on(st) for st in ins)
        and not any(st.axis_joint for st in ins)
        and not (has_derived and derived_exclusion())
        and not any(t is not None for t in transitions)
        and _import_fold_enabled()
        and steps > 1
        and (not frac or (not any(split) and dtype == torch.float32))
        and td is None
        and all(st.gate_head_limit is None or not uses_adaptive(st) for st in ins)
        and all(by_read)
        and all(learned_kind(st) == "none" for st in ins)
        and _fast_path_ok(y0, ins, 0.0, None, None)
        and not (
            _ROUTES
            or _SHARES
            or _DIRECTION_SHARES
            or _SET_SHARES
            or _GATE_STATS
            or _route_freeze.RECORDS
        )
    )
    if fold:
        gates = (
            _import_gates(ins, coefs, rg, x, target=target)
            if has_derived
            else _read_share_gates(ins, coefs, rg, x=x)
        )
        y = y0
        for i, (st, gate) in enumerate(zip(ins, gates, strict=True)):
            weight = gate.float().repeat_interleave(st.per_expert, dim=-1)
            part = coefs[i] * weight
            if split[i]:
                _import_fold_history(part, scales[i], steps, st.iteration, rg[i], hists[i], st.up)
            count = low + frac if frac else steps
            change = part if st.iteration == "path" else part * count
            z = change * scales[i]
            weights = gate if st.iteration == "path" else gate * count
            if split[i]:
                y = gated_change(i, st, gate, z, steps, weights, y)
            else:
                delta = z.to(dtype) @ ups[i]
                if st.exact:
                    delta = delta + correction(i, st, weights)
                y = y + delta
        return y

    def input_gates(state, prev=None):
        if joint_derived:
            return _mixed_route(
                state if query_policy is None else query_policy(state),
                ins,
                None,
                prev,
                shared,
                noise_scale,
                rg,
                True,
                x=x,
                interpret=interpret,
            )
        keys = [_set_keys(st) for st in ins]
        qs = [
            torch.zeros((*state.shape[:-1], st.router.shape[0]), device=state.device)
            if by_read[i]
            else q
            if q is not None
            else F.linear(state.float(), st.router.float())
            for i, (st, q) in enumerate(zip(ins, direction_queries, strict=True))
        ]
        return _route(qs, ins, keys, prev, False, 0.0, rg)

    from .recording import active_recorder

    recorder = active_recorder()
    prediction = (
        recorder is not None
        and recorder.wants(target)
        and len(ins) == 1
        and ins[0].form == "state"
        and ins[0].down.shape[0] == ins[0].per_expert
        and (recorder.prediction_set_key is None or ins[0].key == recorder.prediction_set_key)
        and not frac
        and not any(split)
        and not any(elementwise)
        and rg[0] is None
        and not ins[0].exact
        and not _STEP_CARRY
    )
    if prediction:
        recorder.observe_depth(target, current_progress(), steps > 1)
    short = keep_rate = None
    fixed = ins[0].fixed_point_stop if stop is None else stop
    fixed = fixed and ins[0].iteration == "path"
    previous_node = done = None
    carry_input = bool(_STEP_CARRY) and _ablation.carry_steps(joint_derived)
    walk_index = (stats or {}).get("ablation_walk", 0)
    carry_target = (
        target
        if all(st.form == "state" for st in ins)
        else f"{target}#input{tuple(once)}#walk{walk_index}"
    )
    start = _carry_start(carry_target, tuple(y0.shape[:-1]), len(ins))
    if not carry_input:
        start = None
    if (
        steps == 1
        and not blend_coefficients
        and not carry_input
        and (grad_steps is None or grad_steps(1))
    ):
        if all(st.apply_once for st in ins):
            gates = [
                c.new_ones((*c.shape[:-1], c.shape[-1] // st.per_expert))
                for st, c in zip(ins, coefs, strict=True)
            ]
        else:
            gates = input_gates(y0)
            if has_derived and not joint_derived:
                selected = _import_gates(ins, coefs, rg, x, target=target)
                gates = [r if b else g for g, r, b in zip(gates, selected, by_read, strict=True)]
            elif not joint_derived:
                gates = _with_read_share(gates, by_read, ins, coefs, rg, x=x)
            if not joint_derived and any(st.axis_joint for st in ins):
                gates = _axis_import_gates(ins, coefs, rg, x, y0)
            if entry is not None:
                gates[0] = (
                    stats["forced_routes"][0] if stats and "forced_routes" in stats else entry
                )
                _derived_path_note(stats, gates[0], derived_logits(ins[0], x))
        if stats is not None and stats.get("filter_routes"):
            gates = stats["filter_routes"](y0, gates, 1)
        _note_routes(target, gates, ins)
        _note_directions(target, 1, ins, gates)
        if _PROBES:
            if joint_derived:
                _probe(target, 1, y0, None, ins, (once, len(sets)), rg, coefs)
                _probe_gates(target, 1, gates, (once, len(sets)))
            else:
                _probe_input(target, 0, y0, sets, once, gates, None, rg, coefs)
        y = y0
        for i, gate in enumerate(gates):
            weight = gate.float().repeat_interleave(ins[i].per_expert, dim=-1)
            _log_direction_keep(ins[i], coefs[i] * weight * scales[i], coefs[i], names[i])
            if learned_kind(ins[i]) in ("neuron", "lowrank") and (
                ins[i].apply_once or joint_derived
            ):
                z = coefs[i] * weight * scales[i]
                delta = z.to(dtype) @ ups[i]
                with torch.no_grad():
                    rms = y0.float().pow(2).mean(-2, keepdim=True).sqrt() + DROPOUT_DELTA
                gate_weights = (
                    gate / gate.sum(-1, keepdim=True).clamp_min(1e-12)
                    if ins[i].apply_once
                    else gate
                )
                mask = output_gate(
                    ins[i], delta, y0, torch.zeros_like(y0), gate_weights, rms, target, i, 1
                )
                if ins[i].exact:
                    delta = delta + correction(i, ins[i], gate)
                if split[i]:
                    gate_fn = _split_gate_grad if interpret else split_gate
                    a, b = gate_fn(
                        ins[i],
                        z,
                        hists[i],
                        y0_norm,
                        y0_rms,
                        gate_weights,
                        gate_names[i],
                        gate_indices[i],
                        1,
                        None,
                        True,
                    )
                    mask = mask * (a * b).to(mask.dtype)
                y = y + delta * mask
            elif split[i]:
                y = gated_change(i, ins[i], gate, coefs[i] * weight * scales[i], 1, gate, y)
            else:
                sc = ins[i].scale
                sc = sc.to(dtype) if isinstance(sc, torch.Tensor) else sc
                delta = (
                    (coefs[i] * weight * scales[i]).to(dtype) @ ups[i]
                    if joint_derived
                    else ((coefs[i] * weight).to(dtype) @ ups[i]) * sc
                )
                if ins[i].exact:
                    delta = delta + correction(i, ins[i], gate)
                if interpret:
                    if joint_derived:
                        y = _loop_output(
                            y, (coefs[i] * weight * scales[i]).to(dtype), ups[i], None, dtype
                        )
                    else:
                        product = _loop_output(
                            torch.zeros_like(y), (coefs[i] * weight).to(dtype), ups[i], None, dtype
                        )
                        y = y + product * sc
                else:
                    y = y + delta
        y = _add_expert_offsets(y, ins, gates)
        if stats is not None and stats.get("capture_step"):
            stats["capture_step"](
                y0,
                gates,
                [
                    c * g.repeat_interleave(st.per_expert, -1) * scale
                    for c, g, st, scale in zip(coefs, gates, ins, scales, strict=True)
                ],
                y,
                1,
            )
        if td is not None:
            y = td.blend(y0, y, 1)
        if joint_derived and depth < 1:
            y = y0 + depth * (y - y0)
        _ablation.note_output(target, 1, y, y0)
        if _PROBES:
            y = _probe_output(target, y, ins, (once, len(sets)))
        return y
    keys = None if joint_derived else [_set_keys(st) for st in ins]
    fast = LOWRANK_ENABLED and not _PROBES and not _ablation.force_loop()
    if fast:
        q0 = [
            torch.zeros((*y0.shape[:-1], st.router.shape[0]), device=y0.device)
            if by_read[i]
            else q
            if q is not None
            else F.linear(y0.float(), st.router.float())
            for i, (st, q) in enumerate(zip(ins, direction_queries, strict=True))
        ]
        cross = [
            [
                None
                if st.direction_core is not None or by_read[i]
                else u.float() @ st.router.float().t()
                for i, st in enumerate(ins)
            ]
            for u in ups
        ]
    holds = [
        c * sc if st.carry == "dropped" else None
        for c, sc, st in zip(coefs, scales, ins, strict=True)
    ]
    carried: list = [None] * len(ins)
    zs: list = [None] * len(ins)
    exact_weights: list = [None] * len(ins)
    blended_zs: list = [None] * len(ins)
    step_sets = [
        _import_depth_settings(st) if on and (not joint_derived or st.form == "input") else st
        for st, on in zip(ins, chains, strict=True)
    ]
    splits = [depth_split_on(st) for st in step_sets]
    metrics = [
        _step_metric(st, y0.device) if on else None for st, on in zip(ins, splits, strict=True)
    ]
    refs: list = [None] * len(ins)
    alive: list = [None] * len(ins)
    chain_refs: list = [None] * len(ins)
    chain_metrics = [
        _step_metric(st, y0.device) if on else None for st, on in zip(ins, chains, strict=True)
    ]
    if carry_input and any(st.exact for st in ins):
        raise NotImplementedError("exact routed changes do not support carry across sampler steps")
    if start is not None:
        carried = [c.to(coefs[i]) for i, c in enumerate(start["carried"])]
        holds = [c if st.carry == "dropped" else None for st, c in zip(ins, carried, strict=True)]
        zs = []
        for i, (st, change) in enumerate(zip(ins, start["change"], strict=True)):
            scale = torch.as_tensor(scales[i], device=change.device, dtype=change.dtype)
            value = torch.where(scale != 0, change / scale, torch.zeros_like(change))
            zs.append(value.to(coefs[i]) if st.iteration != "path" else None)
        hists = [
            gate_history(start, i, zs[i] * scales[i] if zs[i] is not None else None, rg[i])
            for i in range(len(ins))
        ]
    prev = None
    snap = None
    carry_snap = None
    y = out = y0
    skip_tables = [None] * len(ins)
    if steps >= 3:
        from .segments import skip_decisions

        skip_tables = [
            skip_decisions(st)
            if not joint_derived and import_score(st) == "derived" and derived_mode() == "full"
            else None
            for st in ins
        ]
    frozen = False
    for k in range(1, steps + 1):
        keep = grad_steps is None or grad_steps(k)
        with contextlib.nullcontext() if keep else torch.no_grad():
            old_zs, old_carried, old_holds = list(zs), list(carried), list(holds)
            old_exact_weights = list(exact_weights)
            old_alive = list(alive)
            old_extras = list(chain_extras)
            step_coefs = [
                _import_depth_coefs(st, coefs[i], carried, ups, rg[i], read_cross[i])
                if (chains[i] or (joint_derived and st.form == "state")) and k > 1
                else coefs[i]
                for i, st in enumerate(ins)
            ]
            if joint_derived:
                state = y0
                for value, up in zip(carried, ups, strict=True):
                    if value is not None:
                        state = state + value.to(dtype) @ up
                if frozen:
                    state = state.detach()
                for i, st in enumerate(ins):
                    if st.form == "state":
                        value = _state_read(st, state, dtype).float()
                        step_coefs[i] = value if rg[i] is None else value * rg[i].float()
            forced_routes = stats.get("forced_routes_all") if stats is not None else None
            forced_single = stats.get("forced_routes") if stats is not None else None
            forced = forced_routes is not None or forced_single is not None
            if forced_routes is not None:
                gates = list(forced_routes[k - 1])
            elif forced_single is not None:
                gates = [forced_single[k - 1]]
            elif joint_derived:
                gates = input_gates(state, prev)
            elif k == 1 and start is None:
                state = y0
                gates = input_gates(y0)
            elif fast:
                qs = [
                    q0[i]
                    if direction_queries[i] is not None or by_read[i]
                    else q0[i] + sum(c.float() @ cross[j][i] for j, c in enumerate(carried))
                    for i in range(len(ins))
                ]
                gates = _route(qs, ins, keys, prev, False, 0.0, rg)
            else:
                state = y0 + sum(c.to(dtype) @ u for c, u in zip(carried, ups, strict=True))
                gates = input_gates(state, prev)
            if has_derived and not forced and not joint_derived:
                selected = _import_gates(ins, step_coefs, rg, x, prev, transitions, gamma, target)
                gates = [r if b else g for g, r, b in zip(gates, selected, by_read, strict=True)]
            elif not forced and not joint_derived:
                gates = _with_read_share(
                    gates, by_read, ins, step_coefs, rg, prev, transitions, gamma, x=x
                )
            if not forced and not joint_derived and any(st.axis_joint for st in ins):
                query = (
                    y0
                    if k == 1
                    else y0 + sum(c.to(dtype) @ u for c, u in zip(carried, ups, strict=True))
                )
                gates = _axis_import_gates(ins, step_coefs, rg, x, query, prev, transitions, gamma)
            if entry is not None:
                if stats is not None and "forced_routes" in stats:
                    gates[0] = stats["forced_routes"][k - 1]
                elif k == 1:
                    gates[0] = entry
                _derived_path_note(
                    stats, gates[0], derived_logits(ins[0], x), None if td is None else td.active(k)
                )
            if stats is not None and stats.get("filter_routes"):
                gates = stats["filter_routes"](state, gates, k)
            if fixed:
                node = torch.cat(gates, -1).detach().argmax(-1)
                if previous_node is not None:
                    same = node == previous_node
                    done = same if done is None else done | same
                previous_node = node
            _note_routes(target, gates, ins)
            _note_directions(target, k, ins, gates)
            if _PROBES:
                if joint_derived:
                    _probe(target, k, state, prev, ins, (once, len(sets)), rg, step_coefs)
                    _probe_gates(target, k, gates, (once, len(sets)))
                else:
                    _probe_input(target, k - 1, state, sets, once, gates, prev, rg, coefs)
            previous = prev
            prev = (
                [g / g.sum(-1, keepdim=True).clamp_min(1e-30) for g in gates]
                if entry is not None and k == 1
                else gates
            )
            y = y0
            for i, (st, gate) in enumerate(zip(ins, gates, strict=True)):
                effective = gate
                if previous is not None and k < steps and skip_tables[i] is not None:
                    from .segments import skip_mask

                    following = _import_gates(
                        [st], [step_coefs[i]], [rg[i]], x, [prev[i]], [transitions[i]], gamma
                    )[0]
                    mask = skip_mask(skip_tables[i], previous[i], prev[i], following)
                    effective = torch.where(mask[..., None], torch.zeros_like(gate), gate)
                weight = effective.float().repeat_interleave(st.per_expert, dim=-1)
                part = (
                    _run(
                        "iterate",
                        "path_step",
                        {"read": step_coefs[i], "weight": weight},
                        _setting=st,
                    )["state"]
                    if interpret and st.iteration == "path"
                    else step_coefs[i] * weight
                )
                survival = alive[i]
                if survival is not None:
                    if st.exact:
                        survival = survival.reshape(*survival.shape[:-1], -1, st.per_expert)
                        survival = survival.mean(-1).repeat_interleave(st.per_expert, dim=-1)
                    part = part * survival
                if chains[i] and st.exact:
                    value = x
                    for c, up in zip(carried, ups, strict=True):
                        if c is not None and up.shape[-1] == x.shape[-1]:
                            value = value + c.to(dtype) @ up
                    layer = st
                    if st.expert_inputs is not None:
                        change = value - x
                        layer = dataclasses.replace(
                            st,
                            expert_inputs=[
                                None if v is None else v + change for v in st.expert_inputs
                            ],
                        )
                    weights = effective
                    if survival is not None:
                        weights = weights * survival[..., :: st.per_expert]
                    extra = _exact_correction(layer, value, step_coefs[i], weights, rg[i], dtype)
                    extra = extra * scales[i]
                    chain_extras[i] = extra if chain_extras[i] is None else chain_extras[i] + extra
                if fixed and done is not None and zs[i] is not None:
                    part = torch.where(done[..., None], zs[i], part)
                if (
                    splits[i]
                    and zs[i] is not None
                    and not (joint_derived and st.aggregation == "replace")
                ):
                    zs[i] = depth_split_step(
                        zs[i],
                        part,
                        step_sets[i],
                        metrics[i],
                        refs[i],
                        _weight_of(scales[i]),
                        names[i],
                    )
                else:
                    if (
                        st.iteration == "path"
                        or zs[i] is None
                        or (joint_derived and st.aggregation == "replace")
                    ):
                        zs[i] = part
                    elif interpret:
                        zs[i] = _run(
                            "iterate",
                            "accumulate",
                            {"previous_state": zs[i], "change": part},
                            _setting=st,
                        )["state"]
                    else:
                        zs[i] = zs[i] + part
                if chains[i]:
                    metric = chain_metrics[i]
                    if st.exact:
                        change = (zs[i] * scales[i]).to(dtype) @ ups[i] + chain_extras[i]
                        size = change.detach().float().norm(dim=-1)
                    else:
                        size = _metric_size(zs[i], metric, _weight_of(scales[i]))
                    if k == 1:
                        chain_refs[i] = size
                    else:
                        limit = (step_sets[i].radial_budget or IMPORT_DEPTH_BUDGET) * chain_refs[i]
                        ratio = (limit / size.clamp_min(1e-30)).clamp(max=1)
                        if joint_derived:
                            ratio = torch.where(size == 0, torch.ones_like(ratio), ratio)
                        zs[i] = zs[i] * ratio[..., None]
                        if st.exact:
                            chain_extras[i] = chain_extras[i] * ratio[..., None]
                if splits[i] and k == 1 and step_sets[i].radial_budget > 0:
                    refs[i] = _metric_size(zs[i], metrics[i], _weight_of(scales[i]))
                if st.exact:
                    exact_weights[i] = (
                        effective
                        if st.iteration == "path" or exact_weights[i] is None
                        else exact_weights[i] + effective
                    )
                if blend_coefficients:
                    continue
                if elementwise[i]:
                    z = zs[i] * scales[i]
                    delta = z.to(dtype) @ ups[i]
                    mask = output_gate(
                        st, delta, y0, previous_delta[i], gate, gate_rms, target, once[i], k
                    )
                    previous_delta[i] = delta
                    if split[i]:
                        last = k == steps if td is None else td.last(k)
                        gate_fn = _split_gate_grad if interpret else split_gate
                        a, b = gate_fn(
                            st,
                            z,
                            hists[i],
                            y0_norm,
                            y0_rms,
                            gate,
                            gate_names[i],
                            gate_indices[i],
                            k,
                            rg[i],
                            last,
                            active=None if td is None else td.active(k),
                        )
                        mask = mask * (a * b).to(mask.dtype)
                    y = y + delta * mask
                elif split[i]:
                    y = gated_change(i, st, gate, zs[i] * scales[i], k, exact_weights[i], y)
                else:
                    delta = (zs[i] * scales[i]).to(dtype) @ ups[i]
                    if st.exact:
                        delta = delta + correction(i, st, exact_weights[i])
                    y = (
                        _loop_output(y, (zs[i] * scales[i]).to(dtype), ups[i], None, dtype)
                        if interpret
                        else y + delta
                    )
            y = _add_expert_offsets(y, ins, gates)
            if stats is not None and stats.get("capture_step"):
                stats["capture_step"](
                    state, gates, [z * sc for z, sc in zip(zs, scales, strict=True)], y, k
                )
            if td is not None:
                if blend_coefficients:
                    for i, z in enumerate(zs):
                        value = z * scales[i]
                        old = blended_zs[i]
                        blended_zs[i] = td.blend(
                            torch.zeros_like(value) if old is None else old, value, k
                        )
                else:
                    out = td.blend(out, y, k)
                zs = [td.keep(z, o, k) for z, o in zip(zs, old_zs, strict=True)]
                exact_weights = [
                    td.keep(z, o, k) if z is not None else None
                    for z, o in zip(exact_weights, old_exact_weights, strict=True)
                ]
                chain_extras = [
                    td.keep(z, o, k) if z is not None else None
                    for z, o in zip(chain_extras, old_extras, strict=True)
                ]
            if prediction and k > 1:
                recorder.observe_prediction(
                    target, y0, short, y, keep_rate, current_progress(), k - 2
                )
            _ablation.note_output(target, k, y, y0)
            if frac and k <= low:
                snap = y
                carry_snap = list(zs)
            if fixed and done is not None and bool(done.all()):
                break
            if k < steps:
                for i, st in enumerate(ins):
                    z = zs[i] * scales[i]
                    if stats is not None and stats.get("forced_read_keeps"):
                        m = stats["forced_read_keeps"][k - 1]
                    elif stats is not None and stats.get("forced_read_keeps_all"):
                        m = stats["forced_read_keeps_all"][k - 1][i]
                    elif interpret:
                        shape = (*z.shape[:-1], z.shape[-1] // st.per_expert, st.per_expert)
                        m = _read_mask(
                            "mask_expect" if expect_masks() else "mask_sample",
                            {
                                "change": z.detach().float().reshape(shape),
                                "base_read": coefs[i].detach().reshape(shape),
                                "uniforms": dropout_uniforms(
                                    shape, names[i], k, z.device, column_keys(st)
                                ),
                            },
                            st,
                        ).reshape_as(z)
                        if torch.is_grad_enabled() and z.requires_grad:
                            surrogate = _read_dropout_for_set(
                                z, coefs[i], st.per_expert, names[i], k, column_keys(st), st.key
                            )
                            m = m.detach() + (surrogate - surrogate.detach())
                    else:
                        m = _read_dropout_for_set(
                            z, coefs[i], st.per_expert, names[i], k, column_keys(st), st.key
                        )
                    if stats is not None and stats.get("record_macro"):
                        stats.setdefault("read_keeps", []).append(m.detach())
                    for rec in _route_freeze.RECORDS:
                        keeps = rec.setdefault("input_keeps", {}).setdefault((target, k), {})
                        keeps[i] = m.detach()
                    if holds[i] is None:
                        carried[i] = m * z
                    else:
                        holds[i] = (
                            z
                            if _ablation.read_dropout_off()
                            else holds[i] + (1 - m) * (z - holds[i])
                        )
                        carried[i] = holds[i]
                    if interpret:
                        carried[i] = _run(
                            "carry",
                            st.carry,
                            {
                                "change": z,
                                "carried": (
                                    old_holds[i]
                                    if old_holds[i] is not None
                                    else torch.zeros_like(z)
                                ),
                                "mask": m,
                                "base_read": coefs[i],
                                "uniforms": torch.zeros_like(z),
                            },
                            _setting=st,
                        )["state"]
                        if holds[i] is not None:
                            holds[i] = carried[i]
                    if chains[i]:
                        alive[i] = m if alive[i] is None else alive[i] * m
                if prediction:
                    short = y0 + carried[0].to(dtype) @ ups[0]
                    z = zs[0] * scales[0]
                    shape = (*z.shape[:-1], 1, ins[0].per_expert)
                    sigma = coefs[0].reshape(shape).pow(2).mean(-1, keepdim=True).sqrt()
                    keep_rate = keep_probability(z.reshape(shape), sigma + DROPOUT_DELTA)
                if td is not None:
                    carried = [td.keep(c, o, k) for c, o in zip(carried, old_carried, strict=True)]
                    holds = [td.keep(h, o, k) for h, o in zip(holds, old_holds, strict=True)]
                    alive = [
                        a if a is None else td.keep(a, torch.ones_like(a) if o is None else o, k)
                        for a, o in zip(alive, old_alive, strict=True)
                    ]
        frozen = not keep
    if carry_input:
        changes = [z * sc for z, sc in zip(zs, scales, strict=True)]
        if frac:
            changes = _blend(
                [z * sc for z, sc in zip(carry_snap, scales, strict=True)], changes, frac
            )
        ends = [
            carry_transfer(z, c, st.per_expert)
            for st, z, c in zip(ins, changes, coefs, strict=True)
        ]
        _carry_end(carry_target, ends, changes, start, depth, _gate_ends(ins, hists))
    for i, st in enumerate(ins):
        _log_direction_keep(st, zs[i] * scales[i], coefs[i], names[i])
    if frac:
        y = snap + frac * (y - snap)
    if td is not None:
        if blend_coefficients:
            y = y0
            for z, up in zip(blended_zs, ups, strict=True):
                y = y + z.to(dtype) @ up
        else:
            y = out
    if joint_derived and depth < 1 and td is None:
        y = y0 + depth * (y - y0)
    if _PROBES:
        y = _probe_output(target, y, ins, (once, len(sets)))
    return y.detach() if frozen else y


def _probe_output(target, y, lora, places=None):
    entry = {"k": "output", "grads": None}
    for record in _PROBES:
        record.setdefault(target, []).append(entry)
    routers = [None if s.form == "state" else s.router.detach().float() for s in lora]

    def keep(grad):
        with torch.no_grad():
            g = grad.detach().float()
            grads = _at_places(
                [g if r is None else F.linear(g, r.to(g.device)) for r in routers], places
            )
            entry["grads"] = _at_places(grads, entry.get("_outer_places"))

    if not torch.is_grad_enabled():
        return y
    if not y.requires_grad:
        y = y + torch.zeros((), dtype=y.dtype, device=y.device, requires_grad=True)
    return _GradTap.apply(y, keep)


def _probe_gates(target, k, gates, places=None):
    for record in _PROBES:
        record[target][-1]["gates"] = _at_places([g.detach() for g in gates], places)


LOWRANK_MODES = ("accumulate", "replace")
LOWRANK_ENABLED = os.environ.get("KILORA_DEPTH_PATH", "fast").strip().lower() != "loop"


def use_depth_loop(loop: bool = True) -> None:
    global LOWRANK_ENABLED
    LOWRANK_ENABLED = not loop


_PROGRAM_LAYER = ContextVar("kilora_program_layer", default=False)


def _fast_path_ok(
    y0, sets, noise_scale, query_policy, grad_steps, *, allow_depth_split=False
) -> bool:
    if (
        _PROGRAM_LAYER.get()
        or _ablation.force_loop()
        or not LOWRANK_ENABLED
        or any(
            any(s.cross_weights) or (s.learned_cross is not None and not s.learned_cross_zero)
            for s in sets
        )
        or noise_scale > 0
        or query_policy is not None
        or _PROBES
        or any(s.expert_inputs is not None or s.expert_offsets is not None for s in sets)
    ):
        return False
    if any(learned_kind(s) in ("neuron", "lowrank") for s in sets):
        return False
    if not allow_depth_split and any(depth_split_on(s) for s in sets if s.down is not None):
        return False
    lora = [s for s in sets if s.down is not None and s.form in ("state", "input")]
    if not lora or any(s.aggregation not in LOWRANK_MODES for s in lora):
        return False
    if torch.is_grad_enabled():
        tensors = [y0] + [t for s in lora for t in (s.down, s.up, s.router, s.keys_in)]
        tensors += [s.keys_out for s in lora if s.keys_out is not None]
        tensors += [t for s in lora for t in (s.gate or {}).values()]
        tensors += [s.tau for s in lora if isinstance(s.tau, torch.Tensor)]
        if any(t.requires_grad for t in tensors) or grad_steps is not None:
            return False
    return True


_FUSED: list = []


def _fused_module():
    if os.environ.get("KILORA_FUSED", "1").strip() == "0":
        return None
    if not _FUSED:
        from . import fused

        _FUSED.append(fused)
    return _FUSED[0]


def _lowrank_mats(lora: list[LayerSet]) -> tuple[torch.Tensor, list, list]:
    forms = "".join(s.form[0] for s in lora)
    key = _tensor_key(f"lowrank-{forms}", [t for s in lora for t in (s.up, s.down, s.router)])
    up = torch.cat([s.up.float() for s in lora], 0)
    hit = _cache_get(key)
    if hit is not None:
        gs, hs = hit
        return up, gs, hs
    gs = [up @ s.down.float().t() if s.form == "state" else None for s in lora]
    hs = [up @ s.router.float().t() for s in lora]
    _cache_put(key, (gs, hs))
    return up, gs, hs


def _run_lowrank(
    y0: torch.Tensor,
    sets: Sequence[LayerSet],
    depth: float,
    shared: bool,
    target: str = "",
    regions: list | None = None,
    x: torch.Tensor | None = None,
):
    lora = _looped(sets, x)
    regions = regions if regions is not None else [None] * len(lora)
    if _derived_state_sets(lora):
        return _run_input(
            x, y0, lora, list(range(len(lora))), depth, regions, target, shared=shared
        )
    with torch.no_grad():
        up, gs, hs = _lowrank_mats(lora)
        dt = y0.dtype
        c0 = [
            (F.linear(y0, _as(s.down, dt)) if s.form == "state" else _input_read(s, x, dt)).float()
            for s in lora
        ]
        q0 = [F.linear(y0, _as(s.router, dt)).float() for s in lora]
        keys = [_set_keys(s) for s in lora]
        prev = None
        sizes = [s.up.shape[0] for s in lora]
        z = torch.zeros((*y0.shape[:-1], sum(sizes)), device=y0.device)
        offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        zm = z
        hold = None
        if lora[0].carry == "dropped":
            hold = torch.cat([c * _scale(s, c.dtype) for c, s in zip(c0, lora, strict=True)], -1)
        start = _carry_start(target, tuple(y0.shape[:-1]), len(lora))
        if start is not None and any(s.exact for s in lora):
            raise NotImplementedError(
                "exact routed changes do not support carry across sampler steps"
            )
        exact_weights = [None] * len(lora)
        if start is not None:
            zm = torch.cat([t.float() for t in start["carried"]], -1)
            if hold is not None:
                hold = zm.clone()
            z = torch.cat(
                [
                    t.float() if s.aggregation != "replace" else torch.zeros_like(t.float())
                    for s, t in zip(lora, start["change"], strict=True)
                ],
                -1,
            )
        split = [learned_kind(s) == "separable" or uses_adaptive(s) for s in lora]
        masks: list = [None] * len(lora)
        if any(split):
            y0_norm, y0_rms = _y0_gate_stats(y0)
        hists = []
        for i, (s, off, n) in enumerate(zip(lora, offsets, sizes, strict=True)):
            zi = z[..., off : off + n] if start is not None and s.aggregation != "replace" else None
            hists.append(gate_history(start, i, zi, regions[i]))
            if uses_adaptive(s) and "q" in hists[i]:
                masks[i] = hists[i]["q"]
        steps, low, frac = _depth_parts(depth)

        def output(z, masks):
            if not any(m is not None for m in masks):
                y = y0 + z.to(dt) @ _as(up, dt)
            else:
                y = y0
                for s, off, n, mask in zip(lora, offsets, sizes, masks, strict=True):
                    zi = z[..., off : off + n]
                    if mask is None:
                        y = y + zi.to(dt) @ _as(s.up, dt)
                    else:
                        y = y + ((mask[0] * zi).to(dt) @ _as(s.up, dt)) * mask[1].to(dt)
            for i, s in enumerate(lora):
                if s.exact:
                    extra = _exact_correction(s, x, c0[i], exact_weights[i], None, dt)
                    if masks[i] is not None:
                        extra = extra * masks[i][0].to(dt) * masks[i][1].to(dt)
                    y = y + extra
            return y

        from .recording import active_recorder

        recorder = active_recorder()
        supported = (
            recorder is not None
            and recorder.wants(target)
            and len(lora) == 1
            and (recorder.prediction_set_key is None or lora[0].key == recorder.prediction_set_key)
            and lora[0].form == "state"
            and lora[0].keys_in.shape[0] == 1
            and not frac
            and start is None
            and not any(split)
            and regions[0] is None
            and not lora[0].exact
        )
        if supported:
            recorder.observe_depth(target, current_progress(), steps > 1)
        capture = supported and steps > 1
        short = keep_rate = None
        snap = None
        if frac and low == 0:
            snap = (y0, (zm if hold is None else hold).clone(), z.clone())
        for k in range(1, steps + 1):
            if any(m is not None for m in masks):
                qs, coefs = _gated_reads(zm, masks, lora, offsets, sizes, q0, c0, hs, gs)
            else:
                qs = [q + zm @ h for q, h in zip(q0, hs, strict=True)]
                coefs = [c if g is None else c + zm @ g for c, g in zip(c0, gs, strict=True)]
            gates = _mixed_route(qs, lora, keys, prev, shared, 0.0, regions)
            prev = gates
            _note_routes(target, gates, lora)
            _note_directions(target, k, lora, gates)
            parts = []
            for i, (s, gate, coef, off, n) in enumerate(
                zip(lora, gates, coefs, offsets, sizes, strict=True)
            ):
                weight = gate.repeat_interleave(s.per_expert, dim=-1)
                scale = float(s.scale) if not isinstance(s.scale, torch.Tensor) else s.scale.float()
                part = coef * weight * scale
                parts.append(part if s.aggregation == "replace" else z[..., off : off + n] + part)
                if s.exact:
                    weighted_gate = gate * scale
                    exact_weights[i] = (
                        weighted_gate
                        if s.aggregation == "replace" or exact_weights[i] is None
                        else exact_weights[i] + weighted_gate
                    )
            z = parts[0] if len(parts) == 1 else torch.cat(parts, -1)
            for i, (s, off, n) in enumerate(zip(lora, offsets, sizes, strict=True)):
                r = regions[i]
                if split[i]:
                    zi = z[..., off : off + n]
                    args = (y0_norm, y0_rms, gates[i], target, i, k, r, k == steps)
                    g = split_gate(s, zi, hists[i], *args)
                    masks[i] = g
                elif r is not None:
                    masks[i] = (r, torch.ones((1, s.up.shape[1]), device=y0.device))
            if capture and k > 1:
                recorder.observe_prediction(
                    target, y0, short, output(z, masks), keep_rate, current_progress(), k - 2
                )
            if k < steps:
                zm = z.clone() if hold is None else hold.clone()
                for i, (s, off, n) in enumerate(zip(lora, offsets, sizes, strict=True)):
                    name = _set_target(target, s, i)
                    sl = z[..., off : off + n]
                    m = _read_dropout_for_set(
                        sl, c0[i], s.per_expert, name, k, column_keys(s), s.key
                    )
                    if hold is None:
                        zm[..., off : off + n] *= m
                    else:
                        old = hold[..., off : off + n]
                        zm[..., off : off + n] = (
                            sl if _ablation.read_dropout_off() else old + (1 - m) * (sl - old)
                        )
                if hold is not None:
                    hold = zm
                if capture:
                    short = output(zm, masks)
                    shape = (*z.shape[:-1], 1, lora[0].per_expert)
                    sigma = (
                        c0[0].reshape(shape).pow(2).mean(-1, keepdim=True).sqrt() + DROPOUT_DELTA
                    )
                    keep_rate = keep_probability(z.reshape(shape), sigma)
                if frac and k == low:
                    snap = (output(z, masks), zm.clone(), z.clone())
        if _STEP_CARRY:
            changes = [z[..., off : off + n] for off, n in zip(offsets, sizes, strict=True)]
            if frac:
                parts = list(zip(offsets, sizes, strict=True))
                changes = _blend([snap[2][..., o : o + n] for o, n in parts], changes, frac)
            ends = [
                carry_transfer(c, c0[i], s.per_expert)
                for i, (s, c) in enumerate(zip(lora, changes, strict=True))
            ]
            _carry_end(target, ends, changes, start, depth, _gate_ends(lora, hists))
        if not any(m is not None for m in masks) and not frac:
            return output(z, masks).to(y0.dtype)
        y = output(z, masks)
        if frac:
            y = snap[0] + frac * (y - snap[0])
        return y.to(y0.dtype)


def _gated_reads(zm, masks, lora, offsets, sizes, q0, c0, hs, gs):
    qs = [q.clone() for q in q0]
    coefs = [c.clone() for c in c0]
    for sj, off, n, mask in zip(lora, offsets, sizes, masks, strict=True):
        zj = zm[..., off : off + n]
        if mask is None:
            for i in range(len(lora)):
                qs[i] = qs[i] + zj @ hs[i][off : off + n]
                if gs[i] is not None:
                    coefs[i] = coefs[i] + zj @ gs[i][off : off + n]
            continue
        a, b = mask
        za = a * zj
        upb = sj.up.float() * b
        for i, si in enumerate(lora):
            qs[i] = qs[i] + za @ (upb @ si.router.float().t())
            if gs[i] is not None:
                coefs[i] = coefs[i] + za @ (upb @ si.down.float().t())
    return qs, coefs


def _as(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype == dtype:
        return t
    if torch.is_grad_enabled() and t.requires_grad:
        return t.to(dtype)
    key = _tensor_key(f"as-{dtype}", [t])
    hit = _cache_get(key)
    if hit is None:
        hit = t.to(dtype)
        _cache_put(key, hit)
    return hit


def _weight_of(scale) -> torch.Tensor | float | None:
    if isinstance(scale, torch.Tensor):
        return scale.float()
    return None if scale == 1 else float(scale)


def _scale(s: LayerSet, dtype) -> torch.Tensor | float:
    return s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale


def _set_target(target: str, s: LayerSet, i: int) -> str:
    return f"{target}#{i}"


def _whole(depth: float) -> int | float:
    if isinstance(depth, bool):
        raise ValueError(f"depth must be a number >= 0, got {depth!r}")
    if isinstance(depth, int):
        return depth
    depth = float(depth)
    if not math.isfinite(depth):
        raise ValueError(f"depth must be a finite number >= 0, got {depth!r}")
    return int(depth) if depth.is_integer() else depth


def _token_values(d: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    lead = tuple(y0.shape[:-1])
    d = d.detach().to(y0.device, torch.float32)
    if tuple(d.shape) == lead + (1,):
        return d
    if tuple(d.shape) == lead:
        return d[..., None]
    if d.dim() == 2 and d.shape[1] == math.prod(lead[1:]):
        if d.shape[0] != lead[0]:
            if lead[0] % d.shape[0]:
                raise ValueError(f"token depth batch {d.shape[0]} does not divide {lead[0]}")
            d = d.repeat(lead[0] // d.shape[0], 1)
        return d.reshape(lead + (1,))
    raise ValueError(f"token depth {tuple(d.shape)} does not fit the layer {lead}")


class _TokenDepth:
    def __init__(self, d: torch.Tensor, maximum: float):
        self.d = d.clamp(min=0)
        self.steps = max(0, math.ceil(maximum))

    def full(self, k: int) -> torch.Tensor:
        return self.d >= k

    def active(self, k: int) -> torch.Tensor:
        return (self.d > k - 1).float()

    def last(self, k: int) -> torch.Tensor:
        return torch.ceil(self.d) == k

    def blend(self, out: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
        w = (self.d - (k - 1)).clamp(0, 1).to(y.dtype)
        mid = out + w * (y - out)
        return torch.where(w >= 1, y, torch.where(w <= 0, out, mid))

    def keep(self, new, old, k: int):
        if new is None:
            return None
        return torch.where(self.full(k), new, torch.zeros_like(new) if old is None else old)


def _depth_parts(depth: int | float) -> tuple[int, int, float]:
    low = math.floor(depth)
    frac = float(depth - low)
    return (low + 1 if frac else low), low, frac


def _blend(lo: list, hi: list, frac: float) -> list:
    out = []
    for a, b in zip(lo, hi, strict=True):
        a = torch.zeros_like(b) if a is None else a.to(b.dtype)
        out.append(a + frac * (b - a))
    return out


def _loop_initial(lora, start, hold, y0, dtype) -> list:
    if start is not None:
        return [t.to(dtype) for t in start["carried"]]
    if hold is not None:
        return list(hold)
    return [y0.new_zeros((*y0.shape[:-1], s.up.shape[0])) for s in lora]


AUTO_DEPTH_CAP = 16

LAYER_KINDS = ("self_attn", "cross_attn", "mlp", "other")
_KIND_PARTS = {
    "self_attn": {"attn1", "self_attn", "self_attention", "attn", "attention", "sa"},
    "cross_attn": {"attn2", "cross_attn", "cross_attention", "ca"},
    "mlp": {"ff", "mlp", "ffn", "feed_forward", "ff_context", "mlp_fc"},
}


def _kind_at(parts: list[str]) -> tuple[str, int]:
    for j, part in enumerate(parts):
        for kind in ("cross_attn", "self_attn", "mlp"):
            if part.lower() in _KIND_PARTS[kind]:
                return kind, j
    return "other", len(parts)


def layer_kind(target: str) -> str:
    return _kind_at(target.split("."))[0]


def layer_places(targets: Sequence[str]) -> dict[str, tuple[int, int, str]]:
    blocks: dict[str, int] = {}
    found = {}
    for target in targets:
        parts = target.split(".")
        kind, j = _kind_at(parts)
        head = parts[:j]
        nums = [n for n, p in enumerate(head) if p.isdigit()]
        block = ".".join(head[: nums[-1] + 1]) if nums else ".".join(head) or target
        found[target] = (blocks.setdefault(block, len(blocks)), kind)
    return {t: (i, len(blocks), kind) for t, (i, kind) in found.items()}


_PROGRESS = ContextStack("kilora_progress")


@contextlib.contextmanager
def sampling_progress(t: float) -> Iterator[None]:
    with _PROGRESS.pushed(min(max(float(t), 0.0), 1.0)):
        yield


def current_progress() -> float:
    return _PROGRESS[-1] if _PROGRESS else 0.0


_ROUTES = ContextStack("kilora_route_records")


@contextlib.contextmanager
def recording_routes() -> Iterator[dict[str, list[torch.Tensor]]]:
    routes: dict[str, list[torch.Tensor]] = {}
    with _ROUTES.pushed(routes):
        yield routes


_DIRECTION_SHARES = ContextStack("kilora_direction_shares")


@contextlib.contextmanager
def recording_directions(fn: Callable) -> Iterator[None]:
    with _DIRECTION_SHARES.pushed(fn):
        yield


def _note_directions(target, step, sets, gates):
    if not _DIRECTION_SHARES:
        return
    for index, (layer, gate) in enumerate(zip(sets, gates, strict=True)):
        if layer.direction_sources is not None:
            for fn in _DIRECTION_SHARES:
                fn(target, step, index, layer, gate.detach())


_SHARES = ContextStack("kilora_shares")
_SET_SHARES = ContextStack("kilora_set_shares")
_SET_SHARE_SCOPES = ContextStack("kilora_set_share_scopes")


@contextlib.contextmanager
def observing_set_share_layers(wants: Callable, observe: Callable) -> Iterator[None]:
    entry = (wants, observe)
    with _SET_SHARE_SCOPES.pushed(entry):
        yield


@contextlib.contextmanager
def layer_set_share_observation(target: str, sets) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for wants, observe in tuple(_SET_SHARE_SCOPES):
            if observe not in _SET_SHARES and wants(target, sets):
                stack.enter_context(recording_set_shares(observe))
        yield


@contextlib.contextmanager
def recording_shares(fn: Callable[[str, list[torch.Tensor]], None]) -> Iterator[None]:
    with _SHARES.pushed(fn):
        yield


@contextlib.contextmanager
def recording_set_shares(fn: Callable) -> Iterator[None]:
    with _SET_SHARES.pushed(fn):
        yield


def _note_routes(target: str, gates: list[torch.Tensor], sets=None) -> None:
    if not _LAYER_TIMERS:
        return _note_routes_now(target, gates, sets)
    import time

    _sync(gates[0] if gates else None)
    start = time.perf_counter()
    try:
        _note_routes_now(target, gates, sets)
        _sync(gates[0] if gates else None)
    finally:
        spent = time.perf_counter() - start
        for timer in _LAYER_TIMERS:
            timer["hooks_s"] += spent


def _note_routes_now(target: str, gates: list[torch.Tensor], sets=None) -> None:
    if sets is not None and any(getattr(s, "window_indices", None) for s in sets):
        from .entry_window import full_window_gates

        gates = full_window_gates(gates, sets)
    if _route_freeze.RECORDS and gates:
        _route_freeze.note_routes(target, gates)
    if not gates or not (_ROUTES or _SHARES or _SET_SHARES):
        return
    for fn in _SHARES:
        fn(target, [g.detach() for g in gates])
    for fn in tuple(_SET_SHARES):
        fn(target, [g.detach() for g in gates], sets)
    if not _ROUTES:
        return
    choice = torch.stack([g.detach().argmax(-1) for g in gates], -1).cpu()
    for routes in _ROUTES:
        routes.setdefault(target, []).append(choice)


def routing_flips(
    base: Mapping[str, list[torch.Tensor]], other: Mapping[str, list[torch.Tensor]]
) -> int:
    flips = 0
    for target in set(base) | set(other):
        a, b = base.get(target, []), other.get(target, [])
        for x, y in zip(a, b, strict=False):
            flips += int((x != y).sum()) if x.shape == y.shape else int(max(x.numel(), y.numel()))
        for extra in a[len(b) :] if len(a) > len(b) else b[len(a) :]:
            flips += int(extra.numel())
    return flips
