# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import logging
import math
import os
import threading
import time
import zlib

import torch
import torch.nn.functional as F

from . import compute, linear_jump

log = logging.getLogger("kilora")

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:
    triton = None
    tl = None
    HAVE_TRITON = False

_M32 = 0xFFFFFFFF
MAX_EXPERTS = 16
_NOTED: set = set()
_TOO_BIG: set = set()
_TOKEN_BLOCK = 32
_COLUMN_BLOCK = 64
_TRACE: list = []
_NOTE_LOCK = threading.Lock()


COUNTS: dict[str, int] = {"fused": 0, "eager": 0}
LAST_REASON: list[str] = [""]
REASONS: dict[str, int] = {}
_CALL_REASON: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kilora_fused_call_reason", default=""
)


def _note(reason: str) -> None:
    LAST_REASON[0] = reason
    _CALL_REASON.set(reason)
    with _NOTE_LOCK:
        if reason in _NOTED:
            return
        _NOTED.add(reason)
    log.info("KiLoRA fused kernels not used (%s); running the eager fast path", reason)


def _shared_limit(t: torch.Tensor) -> int | None:
    if not t.is_cuda:
        return None
    props = torch.cuda.get_device_properties(t.device)
    limit = getattr(props, "shared_memory_per_block_optin", None)
    return int(limit) if limit else None


def fits(t: torch.Tensor, cols: int, router_dim: int | None = None, *, tiled: bool = False) -> bool:
    cp = _p2(cols)
    key = (str(t.device), cp, router_dim is not None)
    if (key + ("tiled",) if tiled else key) in _TOO_BIG:
        return False
    limit = _shared_limit(t)
    return limit is None or _shared_need(cols, router_dim, tiled) <= limit


def _shared_need(cols: int, router_dim: int | None = None, tiled: bool = False) -> int:
    cp = _p2(cols)
    if tiled:
        return 4 * _TOKEN_BLOCK * _COLUMN_BLOCK
    if router_dim is None:
        return 4 * (cp * cp + _TOKEN_BLOCK * cp)
    return 4 * (cp * _p2(router_dim) + _TOKEN_BLOCK * cp)


def columns_for_limit(limit: int, router_dim: int) -> int:
    if _shared_need(16, router_dim) > limit:
        return 0
    cols = 16
    while _shared_need(cols * 2, router_dim) <= limit:
        cols *= 2
    return cols


def max_columns(
    router_dim: int, device: torch.device | str | None = None, *, tiled: bool = False
) -> int | None:
    if tiled:
        return None
    try:
        if device is None:
            if not torch.cuda.is_available():
                return None
            device = torch.device("cuda", torch.cuda.current_device())
        device = torch.device(device)
        if device.type != "cuda":
            return None
        probe = torch.empty(0, device=device)
    except (RuntimeError, AssertionError):
        return None
    limit = _shared_limit(probe)
    if limit is None:
        return None
    cols = columns_for_limit(limit, router_dim)
    while cols > 16 and not fits(probe, cols, router_dim):
        cols //= 2
    return cols if cols and fits(probe, cols, router_dim) else 0


def resource_error(err: BaseException) -> bool:
    return type(err).__name__ in ("OutOfResources", "OutOfResourcesError")


def mark_too_big(t: torch.Tensor, sets) -> None:
    for s in sets:
        if s.down is None:
            continue
        key = (str(t.device), _p2(s.down.shape[0]), s.form == "input")
        if _tiled_import(s):
            key += ("tiled",)
        if key not in _TOO_BIG:
            _TOO_BIG.add(key)
            log.warning(
                "KiLoRA fused kernels for %d low-rank columns do not fit the shared memory "
                "of %s; those layers run on the eager fast path",
                s.down.shape[0],
                t.device,
            )


def adaptive_kernel() -> bool:
    return os.environ.get("KILORA_ADAPTIVE_KERNEL", "0").strip() == "1"


def enabled() -> bool:
    return os.environ.get("KILORA_FUSED", "1").strip() != "0"


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def device_ok(t: torch.Tensor) -> bool:
    if t.is_cuda:
        return True
    return t.device.type == "cpu" and _interpret() and os.environ.get("KILORA_FUSED_CPU") == "1"


def _tiled_import(s) -> bool:
    return (
        s.form == "input"
        and compute.import_score(s) in ("read_share", "derived")
        and compute.learned_kind(s) == "none"
    )


def fused_ok(y0: torch.Tensor, sets, depth=None, td=None) -> bool:
    _CALL_REASON.set("IR fallback" if compute._IR_FALLBACK.get() else "")
    ok = not compute._IR_FALLBACK.get() and _fused_ok(y0, sets, depth, td)
    COUNTS["fused" if ok else "eager"] += 1
    if not ok:
        reason = _CALL_REASON.get() or "off"
        REASONS[reason] = REASONS.get(reason, 0) + 1
    return ok


def _fused_ok(y0: torch.Tensor, sets, depth=None, td=None) -> bool:
    from .ir.dispatch import interpreted_part

    if compute._ablation.force_loop():
        _note("ablation comparison step loop")
        return False
    part = interpreted_part(sets)
    if part is not None:
        _note(f"interpreted part {part}")
        return False
    if compute._PROGRAM_LAYER.get():
        _note("a layer-change program")
        return False
    if any(
        s.form == "state" and (any(s.cross_weights) or compute.derived_diffusion()) for s in sets
    ):
        _note("routing crosses or derived diffusion")
        return False
    if any(s.learned_cross is not None and not s.learned_cross_zero for s in sets):
        _note("learned cross")
        return False
    if any(0 < s.candidates < s.keys_in.shape[0] for s in sets):
        _note("candidate selection")
        return False
    if not enabled() or compute._EAGER:
        return False
    if not HAVE_TRITON:
        _note("Triton is not installed")
        return False
    if not device_ok(y0):
        _note(f"device {y0.device.type} is not CUDA or ROCm")
        return False
    if y0.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        _note(f"dtype {y0.dtype}")
        return False
    if (
        compute._ROUTES
        or compute._SHARES
        or compute._SET_SHARES
        or compute._GATE_STATS
        or compute._PROBES
        or any(not r["fused"] for r in compute._route_freeze.RECORDS)
    ):
        _note("routes or gates are being recorded")
        return False
    if any(s.expert_mask is not None and s.form not in ("input", "state") for s in sets):
        _note("per-LoRA regions on a trained set")
        return False
    lora = [s for s in sets if s.down is not None and s.form in ("state", "input")]
    if len(lora) > 1 and compute._derived_state_sets(lora):
        reason = _derived_state_reason(y0, lora, td)
        if reason is not None:
            _note(reason)
            return False
        return True
    if len(lora) != 1:
        _note("more than one keyed set at a layer")
        return False
    s = lora[0]
    if s.gate_head_limit is not None and compute.uses_adaptive(s):
        _note("gate head history")
        return False
    if not fits(
        y0,
        s.down.shape[0],
        s.router.shape[0] if s.form == "input" else None,
        tiled=_tiled_import(s),
    ):
        mark_too_big(y0, [s])
        return False
    if td is not None:
        if s.iteration != "accumulate" or s.aggregation != "accumulate":
            _note("a per-token depth with path iteration or replace")
            return False
        if s.form == "state" and (compute.uses_adaptive(s) or compute.learned_kind(s) != "none"):
            _note("a per-token depth with a gate on a trained set")
            return False
    if s.form == "input":
        if s.aggregation != "accumulate" or s.iteration not in ("accumulate", "path"):
            _note(f"an imported set with iteration {s.iteration}")
            return False
        if s.iteration == "path" and s.path_width > 1 and compute.import_score(s) != "derived":
            _note("an imported set with path width above 1")
            return False
        if compute.learned_kind(s) != "none":
            _note("an imported set with a learned gate")
            return False
    experts = s.keys_in.shape[0]
    if experts > MAX_EXPERTS or routing_mode(s) is None:
        _note(f"{experts} experts, routing {s.routing_scope} {s.routing_select}")
        return False
    if compute.learned_kind(s) not in ("none", "separable"):
        _note(f"output gate {s.output_gate}")
        return False
    for v in (s.tau, s.scale):
        if isinstance(v, torch.Tensor) and v.numel() != 1:
            _note("a per-element temperature or strength")
            return False
    if y0.dim() < 2:
        _note("a layer output without a token dimension")
        return False
    return True


def routing_mode(s: compute.LayerSet) -> tuple[bool, bool] | None:
    scope, select = s.routing_scope, s.routing_select
    if scope not in ("token", "tensor") or select not in ("sparsemax", "top1"):
        return None
    return scope == "token", select == "top1"


def _p2(n: int) -> int:
    return max(16, 1 << max(0, (int(n) - 1).bit_length()))


def _i32(x: int) -> int:
    x &= _M32
    return x - (1 << 32) if x >= (1 << 31) else x


def hash_base(target: str, k: int) -> int:
    seed, step = compute._DROPOUT_KEY[-1]
    mix = compute._mix32
    return mix(seed ^ mix(step ^ mix(zlib.crc32(target.encode()) ^ mix(k))))


if HAVE_TRITON:
    try:
        from triton.language.extra import libdevice as _libdevice

        _HAVE_LIBDEVICE = True
    except Exception:
        _libdevice = None
        _HAVE_LIBDEVICE = False

    @triton.jit
    def _mix(x):
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        x = x * 0x846CA68B
        x = x ^ (x >> 16)
        return x

    @triton.jit
    def _uniform(idx, base):
        h = _mix(_mix(idx) ^ base)
        return (h >> 8).to(tl.float32) / 16777216.0

    @triton.jit
    def _keep_rate(x, USE_LIBDEVICE: tl.constexpr):
        if USE_LIBDEVICE:
            return _libdevice.pow(tl.full(x.shape, 0.5, tl.float32), x)
        else:
            return tl.exp2(-x)

    @triton.jit(do_not_specialize=["n", "start", "base"])
    def _uniform_kernel(out_ptr, n, start, base, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        idx = (i + start).to(tl.uint32)
        tl.store(out_ptr + i, _uniform(idx, base.to(tl.uint32)), mask=m)

    @triton.jit
    def _cast(x, dtype: tl.constexpr):
        if dtype == tl.float32:
            return x
        else:
            return x.to(dtype, fp_downcast_rounding="rtne")

    @triton.jit
    def _sparsemax_rows(z, ok):
        ge = z[:, None, :] >= z[:, :, None]
        kk = tl.sum(ge.to(tl.float32), axis=2)
        ss = tl.sum(tl.where(ge, z[:, None, :], 0.0), axis=2)
        sup = ((1.0 + kk * z) > ss) & ok[None, :]
        th = tl.max(tl.where(sup, tl.div_rn(ss - 1.0, tl.maximum(kk, 1.0)), float("-inf")), 1)
        p = tl.maximum(z - th[:, None], 0.0)
        return tl.where(ok[None, :], p, 0.0)

    _ROUTE_DYNAMIC = ["T", "C", "RD", "MO", "q0_stride", "c0_stride", "h_bstride", "h_rstride"]
    _ROUTE_DYNAMIC += ["p_bstride", "p_tstride"]

    @triton.jit(do_not_specialize=_ROUTE_DYNAMIC)
    def _route_kernel(
        q0_ptr,
        q0_stride,
        zm_ptr,
        c0_ptr,
        c0_stride,
        a_ptr,
        h_ptr,
        h_bstride,
        h_rstride,
        p_ptr,
        p_bstride,
        p_tstride,
        kin_ptr,
        kout_ptr,
        seg_ptr,
        den_ptr,
        tau_ptr,
        tau_val,
        scale_ptr,
        scale_val,
        out_ptr,
        T,
        C,
        RD,
        MO,
        HAS_Z: tl.constexpr,
        HAS_A: tl.constexpr,
        HAS_PREV: tl.constexpr,
        TAU_PTR: tl.constexpr,
        SCALE_PTR: tl.constexpr,
        BT: tl.constexpr,
        CP: tl.constexpr,
        RDP: tl.constexpr,
        EMP: tl.constexpr,
        EOP: tl.constexpr,
        EP: tl.constexpr,
        DERIVED: tl.constexpr = False,
        INTERPRET: tl.constexpr = False,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        tm = t < T
        row = (b * T + t).to(tl.int64)
        em = tl.arange(0, EMP)
        if DERIVED:
            rb = tl.arange(0, 128)
            pin = tl.zeros([BT, EMP], tl.float32)
            vv = tl.zeros([BT], tl.float32)
            for r0 in range(0, RD, 128):
                rr = r0 + rb
                q = tl.load(
                    q0_ptr + row[:, None] * q0_stride + rr[None, :],
                    mask=tm[:, None] & (rr[None, :] < RD),
                    other=0.0,
                ).to(tl.float32)
                kin = tl.load(
                    kin_ptr + em[:, None] * RDP + rr[None, :],
                    mask=rr[None, :] < RD,
                    other=0.0,
                )
                pin += tl.dot(q, tl.trans(kin), input_precision="ieee")
                vv += tl.sum(q * q, 1)
            pin = _output_cast(pin, q0_ptr.dtype.element_ty, INTERPRET).to(tl.float32)
            vv += 1e-6
        else:
            rd = tl.arange(0, RDP)
            q = tl.load(
                q0_ptr + row[:, None] * q0_stride + rd[None, :],
                mask=tm[:, None] & (rd[None, :] < RD),
                other=0.0,
            ).to(tl.float32)
            if HAS_Z:
                c = tl.arange(0, CP)
                cm = tm[:, None] & (c[None, :] < C)
                z = tl.load(zm_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
                if HAS_A:
                    z = z * tl.load(a_ptr + row, mask=tm, other=0.0)[:, None]
                hm = tl.load(h_ptr + b * h_bstride + c[:, None] * h_rstride + rd[None, :])
                q = q + tl.dot(z, hm, input_precision="ieee")
            if HAS_PREV:
                eo_r = tl.arange(0, EOP)
                kout = tl.load(kout_ptr + eo_r[:, None] * RDP + rd[None, :])
                pr = tl.dot(q, tl.trans(kout), input_precision="ieee")
                eo = eo_r // MO
                pw = tl.load(
                    p_ptr + b * p_bstride + t[:, None] * p_tstride + eo[None, :],
                    mask=tm[:, None] & (eo[None, :] < EP),
                    other=0.0,
                )
                v = tl.dot(pr * pw, kout, input_precision="ieee")
            else:
                v = q
            kin = tl.load(kin_ptr + em[:, None] * RDP + rd[None, :])
            pin = tl.dot(v, tl.trans(kin), input_precision="ieee")
            vv = tl.sum(v * v, axis=1) + 1e-6
        e = tl.arange(0, EP)
        seg = tl.load(seg_ptr + em[:, None] * EP + e[None, :])
        energy = tl.dot(pin * pin, seg, input_precision="ieee")
        share = tl.div_rn(energy, vv[:, None])
        den = tl.load(den_ptr + e)[None, :]
        if DERIVED:
            den = tl.maximum(den, 1e-30)
        share = tl.div_rn(share, den)
        if TAU_PTR:
            lg = tl.load(tau_ptr) * share
        else:
            lg = tau_val * share
        if DERIVED:
            lg = tl.where(tl.load(den_ptr + e)[None, :] > 0, lg, -1e9)
            lg = lg - tl.max(lg, 1)[:, None]
        tl.store(out_ptr + row[:, None] * EP + e[None, :], lg, mask=tm[:, None])

    @triton.jit
    def _state_score_read_kernel(
        base_ptr,
        y_ptr,
        energy_ptr,
        z_ptr,
        key_ptr,
        gram_ptr,
        read_map_ptr,
        c0_ptr,
        mask_ptr,
        region_ptr,
        den_ptr,
        tau_ptr,
        seg_ptr,
        logits_ptr,
        read_ptr,
        N,
        C0_STRIDE,
        WIDTH: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        EP: tl.constexpr,
        CP: tl.constexpr,
        KP: tl.constexpr,
        BT: tl.constexpr,
        FIRST: tl.constexpr,
        HAS_Z: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_REGION: tl.constexpr,
        INTERPRET: tl.constexpr,
    ):
        row = tl.program_id(0) * BT + tl.arange(0, BT)
        ok = row < N
        c = tl.arange(0, CP)
        m = tl.arange(0, KP)
        e = tl.arange(0, EP)
        proj = tl.load(
            base_ptr + row[:, None] * (K + 2 * C) + m[None, :],
            mask=ok[:, None] & (m[None, :] < K),
            other=0,
        ).to(tl.float32)
        if FIRST:
            energy = tl.zeros([BT], tl.float32)
            for start in range(0, WIDTH, 128):
                w = start + tl.arange(0, 128)
                y = tl.load(
                    y_ptr + row[:, None] * WIDTH + w[None, :],
                    mask=ok[:, None] & (w[None, :] < WIDTH),
                    other=0,
                ).to(tl.float32)
                energy += tl.sum(y * y, 1)
            tl.store(energy_ptr + row, energy, mask=ok)
        else:
            energy = tl.load(energy_ptr + row, mask=ok, other=0)
        if HAS_Z:
            z = tl.load(
                z_ptr + row[:, None] * C + c[None, :], mask=ok[:, None] & (c[None, :] < C), other=0
            )
            km = tl.load(key_ptr + c[:, None] * KP + m[None, :])
            proj += tl.dot(z, km, input_precision="ieee")
            yu = tl.load(
                base_ptr + row[:, None] * (K + 2 * C) + K + c[None, :],
                mask=ok[:, None] & (c[None, :] < C),
                other=0,
            ).to(tl.float32)
            gram = tl.load(gram_ptr + c[:, None] * CP + c[None, :])
            energy += 2 * tl.sum(z * yu, 1) + tl.sum(tl.dot(z, gram, input_precision="ieee") * z, 1)
            rm = tl.load(read_map_ptr + c[:, None] * CP + c[None, :])
            zr = _output_cast(z, y_ptr.dtype.element_ty, INTERPRET).to(tl.float32)
            read = tl.load(
                c0_ptr + row[:, None] * C0_STRIDE + c[None, :],
                mask=ok[:, None] & (c[None, :] < C),
                other=0,
            ).to(tl.float32)
            read += tl.dot(zr, rm, input_precision="ieee")
            if HAS_MASK:
                read *= tl.load(
                    mask_ptr + row[:, None] * C + c[None, :],
                    mask=ok[:, None] & (c[None, :] < C),
                    other=0,
                )
            if HAS_REGION:
                read *= tl.load(region_ptr + row, mask=ok, other=0)[:, None]
            tl.store(
                read_ptr + row[:, None] * C + c[None, :], read, mask=ok[:, None] & (c[None, :] < C)
            )
        proj = _output_cast(proj, y_ptr.dtype.element_ty, INTERPRET).to(tl.float32)
        seg = tl.load(seg_ptr + m[:, None] * EP + e[None, :])
        sums = tl.dot(proj * proj, seg, input_precision="ieee")
        den = tl.load(den_ptr + e)
        score = tl.div_rn(tl.load(tau_ptr) * sums, energy[:, None] + 1e-6)
        score = tl.div_rn(score, tl.maximum(den[None, :], 1e-30))
        score = tl.where((e[None, :] < E) & (den[None, :] > 0), score, -1e9)
        score -= tl.max(score, 1)[:, None]
        tl.store(logits_ptr + row[:, None] * EP + e[None, :], score, mask=ok[:, None])

    @triton.jit
    def _derived_score_kernel(
        proj_ptr,
        energy_ptr,
        den_ptr,
        tau_ptr,
        out_ptr,
        N,
        E: tl.constexpr,
        M: tl.constexpr,
        MP: tl.constexpr,
        BT: tl.constexpr,
    ):
        row = tl.program_id(0) * BT + tl.arange(0, BT)
        e = tl.program_id(1)
        m = tl.arange(0, MP)
        proj = tl.load(
            proj_ptr + row[:, None] * (E * M) + e * M + m[None, :],
            mask=(row[:, None] < N) & (m[None, :] < M),
            other=0.0,
        ).to(tl.float32)
        energy = tl.load(energy_ptr + row, mask=row < N, other=0.0)
        den = tl.load(den_ptr + e)
        tau = tl.load(tau_ptr + e)
        score = tl.div_rn(tau * tl.sum(proj * proj, 1), energy + 1e-6)
        score = tl.div_rn(score, tl.maximum(den, 1e-30))
        score = tl.where(den > 0, score, -1e9)
        tl.store(out_ptr + row * E + e, score, mask=row < N)

    @triton.jit
    def _derived_column_gate_kernel(
        value_ptr, gate_ptr, out_ptr, N, BLOCK: tl.constexpr, INTERPRET: tl.constexpr
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(value_ptr + i, mask=i < N, other=0)
        gate = tl.load(gate_ptr + i, mask=i < N, other=1)
        tl.store(
            out_ptr + i, _output_cast(value * gate, out_ptr.dtype.element_ty, INTERPRET), mask=i < N
        )

    @triton.jit
    def _derived_update_kernel(
        read_ptr,
        gates_ptr,
        expert_ptr,
        alive_ptr,
        old_ptr,
        part_ptr,
        out_ptr,
        N,
        C,
        E: tl.constexpr,
        HAS_OLD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = i // C, i % C
        ok = row < N
        expert = tl.load(expert_ptr + col)
        gate = tl.load(gates_ptr + row * E + expert, mask=ok, other=0.0)
        read = tl.load(read_ptr + i, mask=ok, other=0.0)
        alive = tl.load(alive_ptr + i, mask=ok, other=0.0)
        z = (read * gate) * alive
        tl.store(part_ptr + i, z, mask=ok)
        if HAS_OLD:
            z += tl.load(old_ptr + i, mask=ok, other=0.0)
        tl.store(out_ptr + i, z, mask=ok)

    @triton.jit
    def _read_share_kernel(
        c0_ptr,
        stride,
        tau_ptr,
        tau_val,
        out_ptr,
        T,
        C,
        R: tl.constexpr,
        E: tl.constexpr,
        EP: tl.constexpr,
        RB: tl.constexpr,
        TAU_PTR: tl.constexpr,
        BT: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        row = b * T + t
        e = tl.arange(0, EP)
        r = tl.arange(0, RB)
        energy = tl.zeros([BT, EP], tl.float32)
        for r0 in range(0, R, RB):
            rr = r0 + r
            col = e[:, None] * R + rr[None, :]
            a = tl.load(
                c0_ptr + row[:, None, None] * stride + col[None, :, :],
                mask=(t[:, None, None] < T) & (col[None, :, :] < C) & (rr[None, None, :] < R),
                other=0.0,
            ).to(tl.float32)
            energy += tl.sum(a * a, axis=2)
        total = tl.sum(energy, axis=1) + 1e-12
        tau = tl.load(tau_ptr) if TAU_PTR else tau_val
        logits = tau * tl.div_rn(energy, total[:, None])
        tl.store(out_ptr + row[:, None] * EP + e[None, :], logits, mask=t[:, None] < T)

    @triton.jit(do_not_specialize=["stride", "T", "C", "R", "E"])
    def _read_share_wide_kernel(
        c0_ptr,
        stride,
        tau_ptr,
        tau_val,
        out_ptr,
        T,
        C,
        R,
        E,
        EP: tl.constexpr,
        RB: tl.constexpr,
        TAU_PTR: tl.constexpr,
        BT: tl.constexpr,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        row = b * T + t
        e = tl.arange(0, EP)
        r = tl.arange(0, RB)
        energy = tl.zeros([BT, EP], tl.float32)
        for r0 in range(0, R, RB):
            rr = r0 + r
            col = e[:, None] * R + rr[None, :]
            a = tl.load(
                c0_ptr + row[:, None, None] * stride + col[None, :, :],
                mask=(t[:, None, None] < T) & (col[None, :, :] < C) & (rr[None, None, :] < R),
                other=0.0,
            ).to(tl.float32)
            energy += tl.sum(a * a, axis=2)
        total = tl.sum(energy, axis=1) + 1e-12
        tau = tl.load(tau_ptr) if TAU_PTR else tau_val
        logits = tau * tl.div_rn(energy, total[:, None])
        tl.store(out_ptr + row[:, None] * EP + e[None, :], logits, mask=t[:, None] < T)

    @triton.jit(do_not_specialize=["T", "OUT", "N"])
    def _channel_output_kernel(y_ptr, delta_ptr, gate_ptr, out_ptr, T, OUT, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        gate = tl.load(gate_ptr + (i // (T * OUT)) * OUT + i % OUT, mask=mask, other=0.0)
        delta = tl.load(delta_ptr + i, mask=mask, other=0.0)
        y = tl.load(y_ptr + i, mask=mask, other=0.0)
        dtype: tl.constexpr = y_ptr.dtype.element_ty
        gate = _cast(gate, dtype).to(tl.float32)
        product = _cast(delta.to(tl.float32) * gate, dtype).to(tl.float32)
        tl.store(out_ptr + i, _cast(y.to(tl.float32) + product, dtype), mask=mask)

    @triton.jit(do_not_specialize=["N"])
    def _add_output_kernel(y_ptr, delta_ptr, out_ptr, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        y = tl.load(y_ptr + i, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + i, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + i, _cast(y + delta, y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _output_cast(x, dtype: tl.constexpr, INTERPRET: tl.constexpr):
        if INTERPRET and dtype == tl.bfloat16:
            bits = x.to(tl.uint32, bitcast=True)
            magnitude = bits & 0x7FFFFFFF
            rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            bits = tl.where(magnitude > 0x7F800000, bits | 0x00400000, rounded)
            x = (bits & 0xFFFF0000).to(tl.float32, bitcast=True)
        return _cast(x, dtype)

    @triton.jit(do_not_specialize=["N", "WIDTH", "GATE_STRIDE"])
    def _elementwise_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        out_ptr,
        N,
        WIDTH,
        GATE_STRIDE,
        MODE: tl.constexpr,
        SCALAR: tl.constexpr,
        VALUE: tl.constexpr,
        INTERPRET: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        x = tl.load(x_ptr + i, mask=mask, other=0).to(tl.float32)
        dtype: tl.constexpr = out_ptr.dtype.element_ty
        if MODE == 0:
            g = tl.load(gate_ptr + i // WIDTH, mask=mask, other=0).to(tl.float32)
            result = _output_cast(x * g, dtype, INTERPRET)
        elif MODE == 1:
            y = tl.load(y_ptr + i, mask=mask, other=0).to(tl.float32)
            result = _output_cast(
                y + _output_cast(x, dtype, INTERPRET).to(tl.float32), dtype, INTERPRET
            )
        elif MODE == 2:
            y = tl.load(y_ptr + i, mask=mask, other=0).to(tl.float32)
            g = tl.load(gate_ptr + (i // GATE_STRIDE) * WIDTH + i % WIDTH, mask=mask, other=0)
            g = _output_cast(g.to(tl.float32), dtype, INTERPRET).to(tl.float32)
            product = _output_cast(x * g, dtype, INTERPRET).to(tl.float32)
            result = _output_cast(y + product, dtype, INTERPRET)
        elif MODE == 3:
            g = tl.load(gate_ptr + (i // GATE_STRIDE) * WIDTH + i % WIDTH, mask=mask, other=0)
            g = _output_cast(g.to(tl.float32), dtype, INTERPRET).to(tl.float32)
            result = _output_cast(x * g, dtype, INTERPRET)
        else:
            y = tl.load(y_ptr + i, mask=mask, other=0).to(tl.float32)
            if SCALAR:
                g = VALUE
            else:
                g = tl.load(gate_ptr + i // GATE_STRIDE, mask=mask, other=0).to(tl.float32)
            if not SCALAR:
                g = _output_cast(g, dtype, INTERPRET).to(tl.float32)
            diff = _output_cast(x - y, dtype, INTERPRET).to(tl.float32)
            product = _output_cast(diff * g, dtype, INTERPRET).to(tl.float32)
            result = _output_cast(y + product, dtype, INTERPRET)
        tl.store(out_ptr + i, result, mask=mask)

    @triton.jit(do_not_specialize=["C", "OUT", "N"])
    def _gated_up_kernel(up_ptr, gate_ptr, out_ptr, C, OUT, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        col = i % OUT
        sample = i // (C * OUT)
        up = tl.load(up_ptr + i % (C * OUT), mask=mask, other=0.0)
        gate = tl.load(gate_ptr + sample * OUT + col, mask=mask, other=0.0)
        tl.store(out_ptr + i, up * gate, mask=mask)

    _UPDATE_DYNAMIC = ["T", "C", "R", "E", "gate_base", "drop_base", "frac", "c0_stride"]
    _UPDATE_DYNAMIC += ["g_bstride", "g_rstride", "p_bstride", "p_tstride", "SP", "step"]

    @triton.jit(do_not_specialize=_UPDATE_DYNAMIC)
    def _update_kernel(
        lg_ptr,
        entry_ptr,
        reg_ptr,
        c0_ptr,
        c0_stride,
        zm_ptr,
        a_ptr,
        g_ptr,
        g_bstride,
        g_rstride,
        zin_ptr,
        zout_ptr,
        p_ptr,
        p_bstride,
        p_tstride,
        scale_ptr,
        scale_val,
        segc_ptr,
        segt_ptr,
        zp_ptr,
        gram_ptr,
        tp0_ptr,
        tp1_ptr,
        tw1_ptr,
        tb1_ptr,
        tw2_ptr,
        tb2_ptr,
        y0n_ptr,
        gate_base,
        aout_ptr,
        drop_base,
        colkey_ptr,
        sig_ptr,
        sig_delta,
        zl_ptr,
        frac,
        out_ptr,
        depth_ptr,
        step,
        kept_ptr,
        blend_ptr,
        T,
        C,
        R,
        E,
        TOKEN: tl.constexpr,
        TOP1: tl.constexpr,
        HAS_ENTRY: tl.constexpr,
        HAS_REGION: tl.constexpr,
        HAS_Z: tl.constexpr,
        READ_G: tl.constexpr,
        HOLD0: tl.constexpr,
        HAS_A: tl.constexpr,
        ACCUM: tl.constexpr,
        SCALE_PTR: tl.constexpr,
        SEPARABLE: tl.constexpr,
        HAS_ZP: tl.constexpr,
        DO_DROP: tl.constexpr,
        CARRY_END: tl.constexpr,
        DROPPED: tl.constexpr,
        WRITE_OUT: tl.constexpr,
        HAS_ZL: tl.constexpr,
        WRITE_A: tl.constexpr,
        USE_LIBDEVICE: tl.constexpr,
        EXPECT: tl.constexpr,
        HAS_COLKEY: tl.constexpr,
        HAS_DEPTH: tl.constexpr,
        HAS_OLD: tl.constexpr,
        BLEND_DEPTH: tl.constexpr,
        BLEND_REGION: tl.constexpr,
        BT: tl.constexpr,
        CP: tl.constexpr,
        SP,
        EP: tl.constexpr,
    ):
        b = tl.program_id(0)
        cb = tl.program_id(2)
        tb = tl.program_id(1)
        t = tb * BT + tl.arange(0, BT)
        tm = t < T
        row = (b * T + t).to(tl.int64)
        e = tl.arange(0, EP)
        em_ok = e < E
        c = cb * CP + tl.arange(0, CP)
        cm = tm[:, None] & (c[None, :] < C)
        if SCALE_PTR:
            scale = tl.load(scale_ptr)
        else:
            scale = scale_val
        if TOKEN:
            lg = tl.load(lg_ptr + row[:, None] * EP + e[None, :], mask=tm[:, None], other=0.0)
            zt = tl.where(em_ok[None, :], lg, float("-inf"))
            if TOP1:
                pick = tl.argmax(zt, axis=1)
                w = (e[None, :] == pick[:, None]).to(tl.float32)
            else:
                w = _sparsemax_rows(zt, em_ok)
            if cb == 0:
                tl.store(
                    p_ptr + b * p_bstride + t[:, None] * p_tstride + e[None, :], w, mask=tm[:, None]
                )
        else:
            acc = tl.zeros([EP], tl.float32)
            wsum = 0.0
            for t0 in range(0, T, BT):
                tt = t0 + tl.arange(0, BT)
                mm = tt < T
                rr = (b * T + tt).to(tl.int64)
                lg = tl.load(lg_ptr + rr[:, None] * EP + e[None, :], mask=mm[:, None], other=0.0)
                if HAS_REGION:
                    rw = tl.load(reg_ptr + rr, mask=mm, other=0.0)
                    acc += tl.sum(lg * rw[:, None], axis=0)
                    wsum += tl.sum(rw, axis=0)
                else:
                    acc += tl.sum(lg, axis=0)
            if HAS_REGION:
                mean = tl.div_rn(acc, tl.maximum(wsum, 1.1754943508222875e-38))
            else:
                mean = tl.div_rn(acc, T.to(tl.float32))
            zv = tl.where(em_ok, mean, float("-inf"))
            if TOP1:
                p = (e == tl.argmax(zv, axis=0)).to(tl.float32)
            else:
                p = tl.sum(_sparsemax_rows(zv[None, :], em_ok), axis=0)
            if tb == 0 and cb == 0:
                tl.store(p_ptr + b * p_bstride + e, p)
            w = tl.zeros([BT, EP], tl.float32) + p[None, :]
        if HAS_ENTRY:
            w = tl.load(
                entry_ptr + row[:, None] * E + e[None, :],
                mask=tm[:, None] & em_ok[None, :],
                other=0.0,
            )
            denom = tl.maximum(tl.sum(w, axis=1), 1e-30)
            normalized = w / denom[:, None]
            if cb == 0:
                if TOKEN:
                    tl.store(
                        p_ptr + b * p_bstride + t[:, None] * p_tstride + e[None, :],
                        normalized,
                        mask=tm[:, None],
                    )
                elif tb == 0:
                    tl.store(
                        p_ptr + b * p_bstride + e,
                        tl.sum(tl.where((t == 0)[:, None], normalized, 0.0), axis=0),
                    )
        segt = tl.load(segt_ptr + e[:, None] * SP + c[None, :], mask=c[None, :] < SP, other=0.0)
        wcol = tl.dot(w, segt, input_precision="ieee")
        c0 = tl.load(c0_ptr + row[:, None] * c0_stride + c[None, :], mask=cm, other=0.0)
        c0 = c0.to(tl.float32)
        coef = c0
        hold = tl.zeros([BT, CP], tl.float32)
        if HOLD0:
            hold = c0 * scale
        if HAS_Z:
            hold = tl.load(zm_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
            za = hold
            if HAS_A:
                za = hold * tl.load(a_ptr + row, mask=tm, other=0.0)[:, None]
            if READ_G:
                gm = tl.load(g_ptr + b * g_bstride + c[:, None] * g_rstride + c[None, :])
                coef = c0 + tl.dot(za, gm, input_precision="ieee")
        z = coef * wcol * scale
        if ACCUM:
            z = tl.load(zin_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0) + z
        tl.store(zout_ptr + row[:, None] * C + c[None, :], z, mask=cm)
        if HAS_DEPTH:
            d = tl.load(depth_ptr + row, mask=tm, other=0.0)
            old = tl.zeros([BT, CP], tl.float32)
            if HAS_OLD:
                old = tl.load(zin_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
            kept = tl.where(d[:, None] >= step, z, old)
            tl.store(kept_ptr + row[:, None] * C + c[None, :], kept, mask=cm)
            if BLEND_DEPTH:
                weight = tl.minimum(tl.maximum(d - (step - 1), 0.0), 1.0)
                prior = tl.zeros([BT, CP], tl.float32)
                if step > 1:
                    prior = tl.load(blend_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
                current = z
                if BLEND_REGION:
                    current = current * tl.load(reg_ptr + row, mask=tm, other=0.0)[:, None]
                blended = prior + weight[:, None] * (current - prior)
                blended = tl.where(
                    weight[:, None] >= 1.0,
                    current,
                    tl.where(weight[:, None] <= 0.0, prior, blended),
                )
                tl.store(blend_ptr + row[:, None] * C + c[None, :], blended, mask=cm)
        a_new = tl.full([BT], 1.0, tl.float32)
        if SEPARABLE:
            gram = tl.load(gram_ptr + c[:, None] * CP + c[None, :])
            nz = tl.sqrt_rn(tl.maximum(tl.sum(tl.dot(z, gram, input_precision="ieee") * z, 1), 0.0))
            y0n = tl.load(y0n_ptr + row, mask=tm, other=1.0)
            r0 = tl.div_rn(nz, y0n)
            r1 = tl.zeros([BT], tl.float32)
            if HAS_ZP:
                zp = tl.load(zp_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
                npv = tl.sum(tl.dot(zp, gram, input_precision="ieee") * zp, 1)
                r1 = tl.div_rn(tl.sqrt_rn(tl.maximum(npv, 0.0)), y0n)
            tp0 = tl.load(tp0_ptr + c[:, None] * EP + e[None, :])
            tp1 = tl.load(tp1_ptr + c[:, None] * EP + e[None, :])
            pj0 = tl.dot(z, tp0, input_precision="ieee")
            pj1 = tl.dot(z, tp1, input_precision="ieee")
            sacc = tl.zeros([BT, EP], tl.float32)
            for h in tl.static_range(8):
                hid = (
                    r0[:, None] * tl.load(tw1_ptr + (0 * 8 + h) * EP + e)[None, :]
                    + r1[:, None] * tl.load(tw1_ptr + (1 * 8 + h) * EP + e)[None, :]
                    + pj0 * tl.load(tw1_ptr + (2 * 8 + h) * EP + e)[None, :]
                    + pj1 * tl.load(tw1_ptr + (3 * 8 + h) * EP + e)[None, :]
                    + tl.load(tb1_ptr + h * EP + e)[None, :]
                )
                sacc += tl.maximum(hid, 0.0) * tl.load(tw2_ptr + h * EP + e)[None, :]
            qa_e = tl.sigmoid(sacc + tl.load(tb2_ptr + e)[None, :])
            qa = tl.sum(w * qa_e, axis=1)
            u = _uniform(row.to(tl.uint32), gate_base.to(tl.uint32))
            if EXPECT:
                a_new = qa
            else:
                a_new = (u < qa).to(tl.float32)
        if HAS_REGION:
            a_new = a_new * tl.load(reg_ptr + row, mask=tm, other=0.0)
        if WRITE_A:
            tl.store(aout_ptr + row, a_new, mask=tm)
        if DO_DROP or CARRY_END:
            sig = tl.load(
                sig_ptr + row[:, None] * E + e[None, :],
                mask=tm[:, None] & em_ok[None, :],
                other=1.0,
            )
            sig = tl.where(em_ok[None, :], sig + sig_delta, 1.0)
            sigc = tl.dot(sig, segt, input_precision="ieee")
            keep = _keep_rate(tl.div_rn(tl.abs(z), tl.maximum(sigc, 1e-30)), USE_LIBDEVICE)
            if CARRY_END:
                zn = keep * z
            else:
                if HAS_COLKEY:
                    ck = tl.load(colkey_ptr + c, mask=c < C, other=0).to(tl.uint32)
                    idx = _mix(row.to(tl.uint32))[:, None] ^ ck[None, :]
                else:
                    idx = (row[:, None] * C + c[None, :]).to(tl.uint32)
                u = _uniform(idx, drop_base.to(tl.uint32))
                if EXPECT:
                    m = keep
                else:
                    m = (u < keep).to(tl.float32)
                if DROPPED:
                    zn = hold + (1.0 - m) * (z - hold)
                else:
                    zn = m * z
            if HAS_DEPTH:
                prior_hold = tl.zeros([BT, CP], tl.float32)
                if HAS_Z:
                    prior_hold = hold
                zn = tl.where(d[:, None] >= step, zn, prior_hold)
            tl.store(zm_ptr + row[:, None] * C + c[None, :], zn, mask=cm)
        if WRITE_OUT:
            zz = z
            if HAS_ZL:
                zl = tl.load(zl_ptr + row[:, None] * C + c[None, :], mask=cm, other=0.0)
                zz = zl + frac * (z - zl)
            if HAS_REGION or SEPARABLE:
                zz = zz * a_new[:, None]
            tl.store(
                out_ptr + row[:, None] * C + c[None, :],
                _cast(zz, out_ptr.dtype.element_ty),
                mask=cm,
            )

    @triton.jit(do_not_specialize=["N", "C", "step"])
    def _import_bound_kernel(
        z_ptr,
        old_ptr,
        projected_ptr,
        old_projected_ptr,
        next_projected_ptr,
        ref_ptr,
        depth_ptr,
        N,
        C,
        step,
        HAS_OLD: tl.constexpr,
        HAS_DEPTH: tl.constexpr,
        BUDGET: tl.constexpr,
        RADIAL: tl.constexpr,
        ORTHOGONAL: tl.constexpr,
        BT: tl.constexpr,
        CP: tl.constexpr,
        SPLIT: tl.constexpr = True,
        KEEP_ZERO: tl.constexpr = False,
    ):
        row = tl.program_id(0) * BT + tl.arange(0, BT)
        col = tl.arange(0, CP)
        ok = (row[:, None] < N) & (col[None, :] < C)
        offsets = row[:, None] * C + col[None, :]
        z = tl.load(z_ptr + offsets, mask=ok, other=0.0)
        projected = tl.load(projected_ptr + offsets, mask=ok, other=0.0)
        if HAS_OLD:
            limit = BUDGET * tl.load(ref_ptr + row, mask=row < N, other=0.0)
        if HAS_OLD and SPLIT:
            old = tl.load(old_ptr + offsets, mask=ok, other=0.0)
            old_projected = tl.load(old_projected_ptr + offsets, mask=ok, other=0.0)
            part, part_projected = z - old, projected - old_projected
            zz = tl.sum(old * old_projected, 1)
            has = zz > 0
            safe_zz = tl.maximum(zz, 1e-30)
            coefficient = tl.where(has, tl.sum(part_projected * old, 1) / safe_zz, 0.0)
            orthogonal = part - coefficient[:, None] * old
            orthogonal_projected = part_projected - coefficient[:, None] * old_projected
            oo = tl.maximum(tl.sum(orthogonal * orthogonal_projected, 1), 0.0)
            limit = BUDGET * tl.load(ref_ptr + row, mask=row < N, other=0.0)
            radial = RADIAL * coefficient
            negative = radial < 0
            back = -1.0 / tl.where(negative, radial, -1.0)
            smallest = tl.where(negative, tl.minimum(back, 1.0), 0.0)
            rest = limit * limit - ORTHOGONAL * ORTHOGONAL * oo
            full = (1 + radial) * (1 + radial) * zz <= rest
            minimal = (1 + smallest * radial) * (1 + smallest * radial) * zz <= rest
            radius = tl.sqrt_rn(tl.maximum(rest / safe_zz, 0.0))
            root = tl.where(negative, -1 - radius, radius - 1) / tl.where(radial == 0, 1.0, radial)
            t = tl.maximum(tl.minimum(root, 1.0), smallest)
            t = tl.where(full, 1.0, tl.where(full | minimal, t, smallest))
            radial *= t
            new = old + radial[:, None] * old + ORTHOGONAL * orthogonal
            new_projected = (
                old_projected + radial[:, None] * old_projected + ORTHOGONAL * orthogonal_projected
            )
            z = tl.where(has[:, None], new, z)
            projected = tl.where(has[:, None], new_projected, projected)
        size = tl.sqrt_rn(tl.maximum(tl.sum(z * projected, 1), 0.0))
        if HAS_OLD:
            ratio = tl.minimum(limit / tl.maximum(size, 1e-30), 1.0)
            if KEEP_ZERO:
                ratio = tl.where(size == 0, 1.0, ratio)
            z *= ratio[:, None]
            projected *= ratio[:, None]
        else:
            tl.store(ref_ptr + row, size, mask=row < N)
        tl.store(z_ptr + offsets, z, mask=ok)
        if HAS_DEPTH:
            d = tl.load(depth_ptr + row, mask=row < N, other=0.0)
            if HAS_OLD:
                old_projected = tl.load(old_projected_ptr + offsets, mask=ok, other=0.0)
                projected = tl.where(d[:, None] >= step, projected, old_projected)
            else:
                projected = tl.where(d[:, None] >= step, projected, 0.0)
        tl.store(next_projected_ptr + offsets, projected, mask=ok)

    @triton.jit(do_not_specialize=["N", "C", "R", "E", "step", "drop_base"])
    def _import_carry_kernel(
        z_ptr,
        old_ptr,
        hold_ptr,
        alive_ptr,
        sig_ptr,
        colkey_ptr,
        ratio_ptr,
        kept_ptr,
        blend_ptr,
        depth_ptr,
        keep_ptr,
        forced_ptr,
        N,
        C,
        R,
        E,
        step,
        drop_base,
        HAS_OLD: tl.constexpr,
        HAS_DEPTH: tl.constexpr,
        BLEND: tl.constexpr,
        DO_DROP: tl.constexpr,
        DROPPED: tl.constexpr,
        HAS_RATIO: tl.constexpr,
        HAS_COLKEY: tl.constexpr,
        EXPECT: tl.constexpr,
        RECORD_KEEP: tl.constexpr,
        FORCED_KEEP: tl.constexpr,
        USE_LIBDEVICE: tl.constexpr,
        DELTA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < N * C
        row, col = i // C, i % C
        z = tl.load(z_ptr + i, mask=ok, other=0.0)
        if HAS_RATIO:
            z *= tl.load(ratio_ptr + row, mask=ok, other=1.0)
        tl.store(z_ptr + i, z, mask=ok)
        active = tl.full((BLOCK,), True, tl.int1)
        if HAS_DEPTH:
            d = tl.load(depth_ptr + row, mask=ok, other=0.0)
            active = d >= step
            old = tl.full((BLOCK,), 0.0, tl.float32)
            if HAS_OLD:
                old = tl.load(old_ptr + i, mask=ok, other=0.0)
            tl.store(kept_ptr + i, tl.where(active, z, old), mask=ok)
            if BLEND:
                w = tl.minimum(tl.maximum(d - (step - 1), 0.0), 1.0)
                prior = tl.full((BLOCK,), 0.0, tl.float32)
                if step > 1:
                    prior = tl.load(blend_ptr + i, mask=ok, other=0.0)
                value = prior + w * (z - prior)
                value = tl.where(w >= 1, z, tl.where(w <= 0, prior, value))
                tl.store(blend_ptr + i, value, mask=ok)
        if DO_DROP:
            sigma = tl.load(sig_ptr + row * E + col // R, mask=ok, other=1.0) + DELTA
            keep = _keep_rate(tl.div_rn(tl.abs(z), tl.maximum(sigma, 1e-30)), USE_LIBDEVICE)
            if EXPECT:
                m = keep
            else:
                if HAS_COLKEY:
                    ck = tl.load(colkey_ptr + col, mask=ok, other=0).to(tl.uint32)
                    idx = _mix(row.to(tl.uint32)) ^ ck
                else:
                    idx = i.to(tl.uint32)
                m = (_uniform(idx, drop_base.to(tl.uint32)) < keep).to(tl.float32)
            if FORCED_KEEP:
                m = tl.load(forced_ptr + i, mask=ok, other=1.0)
            if RECORD_KEEP:
                tl.store(keep_ptr + i, m, mask=ok)
            alive = tl.load(alive_ptr + i, mask=ok, other=1.0)
            tl.store(alive_ptr + i, tl.where(active, alive * m, alive), mask=ok)
            hold = tl.load(hold_ptr + i, mask=ok, other=0.0)
            if DROPPED:
                carried = hold + (1.0 - m) * (z - hold)
            else:
                carried = m * z
            prior = hold if HAS_OLD else tl.full((BLOCK,), 0.0, tl.float32)
            tl.store(hold_ptr + i, tl.where(active, carried, prior), mask=ok)

    @triton.jit(do_not_specialize=["T", "C", "OUT", "y_stride", "base"])
    def _adaptive_token_kernel(
        z_ptr,
        before_ptr,
        hist_ptr,
        up_ptr,
        y_ptr,
        reg_ptr,
        qa_ptr,
        raw_qa_ptr,
        first_norm_ptr,
        a_ptr,
        before_out,
        hist_out,
        za_ptr,
        T,
        C,
        OUT,
        y_stride,
        kappa,
        base,
        HAS_BEFORE: tl.constexpr,
        HAS_HIST: tl.constexpr,
        HAS_REGION: tl.constexpr,
        RECORD_QA: tl.constexpr,
        EXPECT: tl.constexpr,
        BASE_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        WRITE_ZA: tl.constexpr,
        USE_LIBDEVICE: tl.constexpr,
        BT: tl.constexpr,
        CP: tl.constexpr,
        BO: tl.constexpr,
        FOLDED: tl.constexpr = False,
        FOLD_COUNT: tl.constexpr = 1,
        PATH_FOLD: tl.constexpr = False,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        row = b * T + t
        c = tl.arange(0, CP)
        mask = (t[:, None] < T) & (c[None, :] < C)
        z = tl.load(z_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
        reg = (
            tl.load(reg_ptr + row, mask=t < T, other=0.0)
            if HAS_REGION
            else tl.full([BT], 1.0, tl.float32)
        )
        zr = z * reg[:, None]
        before = (
            tl.load(before_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
            if HAS_BEFORE
            else tl.zeros([BT, CP], tl.float32)
        )
        hist = (
            tl.load(hist_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
            if HAS_HIST
            else tl.zeros([BT, CP], tl.float32)
        )
        dz = zr - before
        if FOLDED:
            unit = zr if PATH_FOLD else zr / FOLD_COUNT
            dz = tl.zeros([BT, CP], tl.float32) if PATH_FOLD else unit
            history_weight = 0.5 ** (FOLD_COUNT - 1) if PATH_FOLD else 1.0 - 0.5 ** (FOLD_COUNT - 1)
            hist = unit * history_weight
        tl.store(before_out + row[:, None] * C + c[None, :], zr, mask=mask)
        tl.store(hist_out + row[:, None] * C + c[None, :], 0.5 * (hist + dz), mask=mask)
        ds = tl.zeros([BT], tl.float32)
        ys = tl.zeros([BT], tl.float32)
        for o0 in range(0, OUT, BO):
            o = o0 + tl.arange(0, BO)
            up = tl.load(
                up_ptr + c[:, None] * OUT + o[None, :],
                mask=(c[:, None] < C) & (o[None, :] < OUT),
                other=0.0,
            )
            d = tl.dot(dz, up, input_precision="ieee")
            y = tl.load(
                y_ptr + row[:, None] * y_stride + o[None, :],
                mask=(t[:, None] < T) & (o[None, :] < OUT),
                other=0.0,
            ).to(tl.float32)
            ds += tl.sum(d * d, 1)
            ys += tl.sum(y * y, 1)
        step = tl.sqrt_rn(ds)
        reference = tl.sqrt_rn(ys)
        if not BASE_REFERENCE:
            if FOLDED:
                reference = tl.maximum(reference, step)
            elif FIRST_STEP:
                tl.store(first_norm_ptr + row, step, mask=t < T)
            else:
                first = tl.load(first_norm_ptr + row, mask=t < T, other=0.0)
                reference = tl.maximum(reference, first)
        q = _keep_rate(tl.div_rn(step, kappa * reference + 1e-6), USE_LIBDEVICE)
        if FIRST_STEP and not BASE_REFERENCE:
            q = tl.full([BT], 1.0, tl.float32)
        a = q if EXPECT else (_uniform(row.to(tl.uint32), base.to(tl.uint32)) < q).to(tl.float32)
        tl.store(qa_ptr + row, q * reg, mask=t < T)
        if RECORD_QA:
            tl.store(raw_qa_ptr + row, q, mask=t < T)
        tl.store(a_ptr + row, a * reg, mask=t < T)
        if WRITE_ZA:
            tl.store(
                za_ptr + row[:, None] * C + c[None, :],
                _cast(z * (a * reg)[:, None], za_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit(do_not_specialize=["T", "C", "OUT", "y_stride", "base"])
    def _adaptive_channel_kernel(
        z_ptr,
        before_ptr,
        hist_ptr,
        up_ptr,
        y_ptr,
        reg_ptr,
        qb_ptr,
        first_rms_ptr,
        b_ptr,
        delta_ptr,
        out_ptr,
        T,
        C,
        OUT,
        y_stride,
        kappa,
        base,
        HAS_BEFORE: tl.constexpr,
        HAS_HIST: tl.constexpr,
        HAS_REGION: tl.constexpr,
        COHERENCE: tl.constexpr,
        EXPECT: tl.constexpr,
        BASE_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        WRITE_OUT: tl.constexpr,
        USE_LIBDEVICE: tl.constexpr,
        BT: tl.constexpr,
        CP: tl.constexpr,
        BO: tl.constexpr,
        FOLDED: tl.constexpr = False,
        FOLD_COUNT: tl.constexpr = 1,
        PATH_FOLD: tl.constexpr = False,
    ):
        b = tl.program_id(0)
        o = tl.program_id(1) * BO + tl.arange(0, BO)
        c = tl.arange(0, CP)
        up = tl.load(
            up_ptr + c[:, None] * OUT + o[None, :],
            mask=(c[:, None] < C) & (o[None, :] < OUT),
            other=0.0,
        )
        first_sq = tl.zeros([BO], tl.float32)
        acc = tl.zeros([BO], tl.float32)
        yy = tl.zeros([BO], tl.float32)
        dot = tl.zeros([BO], tl.float32)
        nd = tl.zeros([BO], tl.float32)
        nm = tl.zeros([BO], tl.float32)
        for t0 in range(0, T, BT):
            t = t0 + tl.arange(0, BT)
            row = b * T + t
            mask = (t[:, None] < T) & (c[None, :] < C)
            z = tl.load(z_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
            if HAS_REGION:
                z *= tl.load(reg_ptr + row, mask=t < T, other=0.0)[:, None]
            v = tl.dot(z, up, input_precision="ieee")
            y = tl.load(
                y_ptr + row[:, None] * y_stride + o[None, :],
                mask=(t[:, None] < T) & (o[None, :] < OUT),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(v * v, 0)
            yy += tl.sum(y * y, 0)
            if (COHERENCE and HAS_HIST) or (FIRST_STEP and not BASE_REFERENCE):
                before = (
                    tl.load(before_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
                    if HAS_BEFORE
                    else tl.zeros([BT, CP], tl.float32)
                )
                d = tl.dot(z - before, up, input_precision="ieee")
                if FIRST_STEP and not BASE_REFERENCE:
                    first_sq += tl.sum(d * d, 0)
                if COHERENCE and HAS_HIST:
                    hist = tl.load(hist_ptr + row[:, None] * C + c[None, :], mask=mask, other=0.0)
                    m = tl.dot(hist, up, input_precision="ieee")
                    dot += tl.sum(d * m, 0)
                    nd += tl.sum(d * d, 0)
                    nm += tl.sum(m * m, 0)
        rms = tl.sqrt_rn(tl.div_rn(yy, T.to(tl.float32)))
        size = tl.sqrt_rn(tl.div_rn(acc, T.to(tl.float32)))
        if not BASE_REFERENCE:
            if FOLDED:
                first = size if PATH_FOLD else size / FOLD_COUNT
                rms = tl.maximum(rms, first)
                size = tl.maximum(size - first, 0.0)
            elif FIRST_STEP:
                first = tl.sqrt_rn(tl.div_rn(first_sq, T.to(tl.float32)))
                tl.store(first_rms_ptr + b * OUT + o, first, mask=o < OUT)
            else:
                first = tl.load(first_rms_ptr + b * OUT + o, mask=o < OUT, other=0.0)
                rms = tl.maximum(rms, first)
                size = tl.maximum(size - first, 0.0)
        q = _keep_rate(tl.div_rn(size, kappa * rms + 1e-6), USE_LIBDEVICE)
        if COHERENCE and HAS_HIST:
            norms = tl.sqrt_rn(nd) * tl.sqrt_rn(nm)
            cos = tl.where(norms > 1e-6, tl.div_rn(dot, tl.maximum(norms, 1e-6)), 1.0)
            q *= tl.minimum(tl.maximum(0.5 + 0.5 * cos, 0.0), 1.0)
        if FIRST_STEP and not BASE_REFERENCE:
            q = tl.full([BO], 1.0, tl.float32)
        u = _uniform((T * tl.num_programs(0) + b * OUT + o).to(tl.uint32), base.to(tl.uint32))
        bg = q if EXPECT else (u < q).to(tl.float32)
        tl.store(qb_ptr + b * OUT + o, q, mask=o < OUT)
        tl.store(b_ptr + b * OUT + o, bg, mask=o < OUT)
        if WRITE_OUT:
            for t0 in range(0, T, BT):
                t = t0 + tl.arange(0, BT)
                row = b * T + t
                mask = (t[:, None] < T) & (o[None, :] < OUT)
                y = tl.load(y_ptr + row[:, None] * y_stride + o[None, :], mask=mask, other=0.0)
                d = tl.load(delta_ptr + row[:, None] * OUT + o[None, :], mask=mask, other=0.0)
                bv = _cast(bg, y_ptr.dtype.element_ty).to(tl.float32)
                prod = _cast(d.to(tl.float32) * bv[None, :], y_ptr.dtype.element_ty).to(tl.float32)
                tl.store(
                    out_ptr + row[:, None] * OUT + o[None, :], y.to(tl.float32) + prod, mask=mask
                )

    @triton.jit(do_not_specialize=["OUT", "TOTAL", "STEP"])
    def _depth_blend_kernel(
        old_ptr,
        new_ptr,
        depth_ptr,
        out_ptr,
        OUT,
        TOTAL,
        STEP,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < TOTAL
        old = tl.load(old_ptr + i, mask=mask, other=0).to(tl.float32)
        new = tl.load(new_ptr + i, mask=mask, other=0).to(tl.float32)
        depth = tl.load(depth_ptr + i // OUT, mask=mask, other=0)
        weight = tl.minimum(tl.maximum(depth - (STEP - 1), 0.0), 1.0)
        value = old + weight * (new - old)
        value = tl.where(weight >= 1.0, new, tl.where(weight <= 0.0, old, value))
        tl.store(out_ptr + i, value, mask=mask)

    @triton.jit(do_not_specialize=["T", "OUT", "y_stride"])
    def _stats_kernel(
        y_ptr,
        y_stride,
        norm_ptr,
        part_ptr,
        T,
        OUT,
        BT: tl.constexpr,
        BO: tl.constexpr,
    ):
        b = tl.program_id(0)
        tb = tl.program_id(1)
        t = tb * BT + tl.arange(0, BT)
        tm = t < T
        row = (b * T + t).to(tl.int64)
        nt = tl.num_programs(1)
        acc = tl.zeros([BT], tl.float32)
        for o0 in range(0, OUT, BO):
            o = o0 + tl.arange(0, BO)
            om = o < OUT
            y = tl.load(
                y_ptr + row[:, None] * y_stride + o[None, :],
                mask=tm[:, None] & om[None, :],
                other=0.0,
            ).to(tl.float32)
            sq = y * y
            acc += tl.sum(sq, axis=1)
            tl.store(part_ptr + (b * nt + tb).to(tl.int64) * OUT + o, tl.sum(sq, axis=0), mask=om)
        tl.store(norm_ptr + row, tl.sqrt_rn(acc) + 1e-6, mask=tm)

    _CHANNEL_DYNAMIC = ["OUT", "C", "E", "NA", "gate_base", "wb_bstride", "NT", "T"]

    @triton.jit(do_not_specialize=_CHANNEL_DYNAMIC)
    def _channel_kernel(
        s_ptr,
        sp_ptr,
        up_ptr,
        part_ptr,
        wbar_ptr,
        wb_bstride,
        cw1_ptr,
        cb1_ptr,
        cw2_ptr,
        cb2_ptr,
        out_ptr,
        upb_ptr,
        OUT,
        C,
        E,
        NA,
        gate_base,
        NT,
        T,
        HAS_ZP: tl.constexpr,
        WRITE_UPB: tl.constexpr,
        EXPECT: tl.constexpr,
        CP: tl.constexpr,
        BO: tl.constexpr,
        BS: tl.constexpr,
    ):
        b = tl.program_id(0)
        o = tl.program_id(1) * BO + tl.arange(0, BO)
        om = o < OUT
        c = tl.arange(0, CP)
        tf = T.to(tl.float32)
        ssum = tl.zeros([BO], tl.float32)
        for k in range(NT):
            ssum += tl.load(part_ptr + (b * NT + k).to(tl.int64) * OUT + o, mask=om, other=0.0)
        rms = tl.sqrt_rn(tl.div_rn(ssum, tf)) + 1e-6
        tot = tl.zeros([BS], tl.float32)
        for o0 in range(0, OUT, BS):
            oo = o0 + tl.arange(0, BS)
            mo = oo < OUT
            acc = tl.zeros([BS], tl.float32)
            for k in range(NT):
                acc += tl.load(part_ptr + (b * NT + k).to(tl.int64) * OUT + oo, mask=mo, other=0.0)
            tot += tl.where(mo, tl.sqrt_rn(tl.div_rn(acc, tf)) + 1e-6, 0.0)
        scl = tl.div_rn(tl.sum(tot, axis=0), OUT.to(tl.float32))
        upt = tl.load(up_ptr + c[:, None] * OUT + o[None, :], mask=om[None, :], other=0.0)
        smask = (c[:, None] < C) & (c[None, :] < C)
        sm = tl.load(s_ptr + b * C * C + c[:, None] * C + c[None, :], mask=smask, other=0.0)
        sm = tl.div_rn(sm, tf)
        chz = tl.sqrt_rn(tl.maximum(tl.sum(tl.dot(sm, upt, input_precision="ieee") * upt, 0), 0.0))
        chp = tl.zeros([BO], tl.float32)
        if HAS_ZP:
            spm = tl.load(sp_ptr + b * C * C + c[:, None] * C + c[None, :], mask=smask, other=0.0)
            spm = tl.div_rn(spm, tf)
            chp = tl.sum(tl.dot(spm, upt, input_precision="ieee") * upt, 0)
            chp = tl.sqrt_rn(tl.maximum(chp, 0.0))
        f0 = tl.div_rn(chz, scl)
        f1 = tl.div_rn(rms, scl)
        f2 = tl.div_rn(chp, scl)
        qb = tl.zeros([BO], tl.float32)
        o64 = o.to(tl.int64)
        for ex in range(E):
            base = ex * OUT + o64
            s = tl.load(cb2_ptr + base, mask=om, other=0.0)
            for h in tl.static_range(4):
                hid = (
                    f0 * tl.load(cw1_ptr + (base * 3 + 0) * 4 + h, mask=om, other=0.0)
                    + f1 * tl.load(cw1_ptr + (base * 3 + 1) * 4 + h, mask=om, other=0.0)
                    + f2 * tl.load(cw1_ptr + (base * 3 + 2) * 4 + h, mask=om, other=0.0)
                    + tl.load(cb1_ptr + base * 4 + h, mask=om, other=0.0)
                )
                s += tl.maximum(hid, 0.0) * tl.load(cw2_ptr + base * 4 + h, mask=om, other=0.0)
            qb += tl.load(wbar_ptr + b * wb_bstride + ex) * tl.sigmoid(s)
        idx = (NA + b * OUT + o).to(tl.uint32)
        u = _uniform(idx, gate_base.to(tl.uint32))
        if EXPECT:
            bg = qb
        else:
            bg = (u < qb).to(tl.float32)
        tl.store(out_ptr + b * OUT + o, bg, mask=om)
        if WRITE_UPB:
            upb = _cast(upt * bg[None, :], upb_ptr.dtype.element_ty)
            dst = upb_ptr + (b * C + c[:, None]).to(tl.int64) * OUT + o[None, :]
            tl.store(dst, upb, mask=(c[:, None] < C) & om[None, :])


def uniforms(n: int, target: str, k: int, device, start: int = 0) -> torch.Tensor:
    out = torch.empty(n - start, dtype=torch.float32, device=device)
    block = 256
    _uniform_kernel[(triton.cdiv(n - start, block),)](
        out, n - start, start, _i32(hash_base(target, k)), BLOCK=block
    )
    return out


def _pad(t: torch.Tensor, shape) -> torch.Tensor:
    out = torch.zeros(shape, dtype=torch.float32, device=t.device)
    out[tuple(slice(0, n) for n in t.shape)] = t.float()
    return out


class _Prepared:
    pass


def _prepared(s: compute.LayerSet, dt: torch.dtype) -> _Prepared:
    gate = s.gate if compute.learned_kind(s) == "separable" else {}
    tensors = [s.up, s.down, s.router, s.keys_in, s.keys_out, s.rows_in, s.rows_out]
    tensors += [gate[n] for n in compute._SEPARABLE_PARTS if n in gate]
    tensors += [s.expert_ids]
    if s.form == "state":
        tensors += [v for v in (s.tau, s.scale) if isinstance(v, torch.Tensor)]
    kind = f"fused-{dt}-{s.per_expert}-{compute.learned_kind(s)}-{s.form}"
    if s.form == "state":
        values = tuple(None if isinstance(v, torch.Tensor) else v for v in (s.tau, s.scale))
        kind += f"-{values}"
    key = compute._tensor_key(kind, tensors)
    hit = compute._cache_get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        p = _Prepared()
        dev = s.up.device
        cols, width = s.up.shape
        kin, kout = compute._set_keys(s)
        rd = kin.shape[-1]
        experts, m = kin.shape[0], kin.shape[1]
        p.C, p.RD, p.E, p.M, p.R, p.OUT = cols, rd, experts, m, s.per_expert, width
        p.CP, p.RDP, p.EP, p.EMP = _p2(cols), _p2(rd), _p2(experts), _p2(experts * m)
        p.down_dt = compute._as(s.down, dt)
        p.router_dt = compute._as(s.router, dt)
        p.router_f32 = compute._as(s.router, torch.float32)
        p.up_dt = compute._as(s.up, dt).contiguous()
        up = s.up.float() if s.form == "state" or dt == torch.float32 else p.up_dt.float()
        if s.form == "state":
            p.G = _pad(up @ s.down.float().t(), (p.CP, p.CP))
        else:
            p.G = torch.empty(0, device=dev)
        p.H = (
            torch.empty((p.CP, 1), device=dev)
            if s.form == "state"
            else _pad(up @ s.router.float().t(), (p.CP, p.RDP))
        )
        if s.form == "state":
            kin = compute._as(kin, dt)
        p.kin = _pad(kin.reshape(experts * m, rd), (p.EMP, p.RDP))
        mo = kout.shape[1]
        p.MO, p.EOP = mo, _p2(experts * mo)
        p.kout = _pad(kout.reshape(experts * mo, rd), (p.EOP, p.RDP))
        den = torch.zeros(p.EP, device=dev) if s.form == "state" else torch.ones(p.EP, device=dev)
        rows = compute.derived_keys(s)[1] if s.form == "state" else s.rows_in
        den[:experts] = compute.key_rows(kin, rows) / rd
        p.den = den
        j = torch.arange(p.EMP, device=dev)
        eidx = torch.arange(p.EP, device=dev)
        p.seg = ((j[:, None] // m == eidx[None, :]) & (j[:, None] < experts * m)).float()
        cidx = torch.arange(p.CP, device=dev)
        segc = (cidx[:, None] // s.per_expert == torch.arange(p.EP, device=dev)[None, :]) & (
            cidx[:, None] < cols
        )
        p.segc = segc.float().contiguous()
        p.segt = p.segc.t().contiguous()
        ck = compute.column_keys(s)
        p.has_colkey = ck is not None
        colkey = torch.zeros(p.CP, dtype=torch.int64, device=dev)
        if ck is not None:
            colkey[:cols] = ck.to(dev)
        p.colkey = torch.where(colkey >= 2**31, colkey - 2**32, colkey).to(torch.int32)
        p.separable = bool(gate)
        if gate:
            g = {n: gate[n].float().to(dev) for n in compute._SEPARABLE_PARTS}
            p.gram = _pad(up @ up.t(), (p.CP, p.CP))
            p.tp0 = _pad(g["tp"][:, :, 0].t(), (p.CP, p.EP))
            p.tp1 = _pad(g["tp"][:, :, 1].t(), (p.CP, p.EP))
            p.tw1 = _pad(g["tw1"].permute(1, 2, 0), (4, 8, p.EP)).contiguous()
            p.tb1 = _pad(g["tb1"].t(), (8, p.EP))
            p.tw2 = _pad(g["tw2"].t(), (8, p.EP))
            p.tb2 = _pad(g["tb2"], (p.EP,))
            p.cw1 = g["cw1"].contiguous()
            p.cb1 = g["cb1"].contiguous()
            p.cw2 = g["cw2"].contiguous()
            p.cb2 = g["cb2"].contiguous()
        p.up_pad = _pad(up, (p.CP, width))
        if s.form == "state":
            u = p.up_dt.float()
            same_up = dt == torch.float32 or dt == s.up.dtype
            same_down = dt == torch.float32 or dt == s.down.dtype
            p.state_read = p.G[:cols, :cols] if same_up and same_down else u @ p.down_dt.float().t()
            p.state_keys = u @ kin.flatten(0, 1).float().t()
            p.state_gram = p.gram[:cols, :cols] if gate and same_up else u @ u.t()
            p.state_joint = torch.cat((kin.flatten(0, 1), p.up_dt, p.down_dt), 0).contiguous()
            p.state_cp = max(16, p.CP)
            p.state_kp = max(16, p.EMP)
            p.state_key_pad = _pad(p.state_keys, (p.state_cp, p.state_kp))
            p.state_gram_pad = _pad(p.state_gram, (p.state_cp, p.state_cp))
            p.state_read_pad = _pad(p.state_read, (p.state_cp, p.state_cp))
            p.state_seg = _pad(p.seg, (p.state_kp, p.EP))
            p.state_tau = torch.as_tensor(s.tau, dtype=torch.float32, device=dev).reshape(())
            p.state_scale = torch.as_tensor(s.scale, dtype=torch.float32, device=dev).reshape(())
        if s.form == "state":
            p.dr_t = _pad(s.down.float().t(), (width, p.CP))
        else:
            down_t = torch.zeros(width, cols, device=dev)
            p.dr_t = torch.cat(
                [
                    _pad(down_t, (width, p.CP)),
                    _pad(s.router.float().t(), (width, p.RDP)),
                ],
                1,
            ).contiguous()
    compute._cache_put(key, p)
    return p


def _arg(v):
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            raise ValueError("a temperature or strength must contain one value")
        return compute._as(v, torch.float32), 0.0, True
    return None, float(v), False


def _gated_read(p, gate):
    gate = gate.contiguous()
    batch, width = gate.shape
    up = torch.empty((batch, p.CP, width), dtype=gate.dtype, device=gate.device)
    size = batch * p.CP * width
    _gated_up_kernel[(triton.cdiv(size, 256),)](p.up_pad, gate, up, p.CP, width, size, BLOCK=256)
    return torch.matmul(up, p.dr_t)


def _state_base(y, up, keys):
    return (
        (y @ keys.t()).float(),
        torch.linalg.vector_norm(y.float(), dim=-1, dtype=torch.float32).square(),
        y.float() @ up.t(),
    )


def _state_logits(base, z, key_map, gram, den, tau, dt, experts, rows):
    proj, energy, y_up = base
    if z is not None:
        proj = proj + z @ key_map
        energy = energy + 2 * (z * y_up).sum(-1) + ((z @ gram) * z).sum(-1)
    proj = proj.to(dt).contiguous()
    energy = energy.contiguous()
    n = energy.numel()
    out = torch.empty((n, experts), dtype=torch.float32, device=proj.device)
    _derived_score_kernel[(triton.cdiv(n, 32), experts)](
        proj,
        energy,
        den,
        tau,
        out,
        n,
        E=experts,
        M=rows,
        MP=_p2(rows),
        BT=32,
        enable_fp_fusion=False,
    )
    return out


def _derived_state_reason(y0, lora, td):
    if td is not None:
        return "a per-token depth with joint derived state routing"
    modes = {(compute.learned_kind(s), s.gate_mode) for s in lora}
    mixed_separable = all(
        compute.learned_kind(s) == "none" for s in lora if s.form == "input"
    ) and all(compute.learned_kind(s) == "separable" for s in lora if s.form == "state")
    if len(modes) > 1 and not mixed_separable:
        return "derived state routing with different output gate modes"
    if any(compute.learned_kind(s) not in ("none", "separable") for s in lora):
        return "joint derived state routing with an element output gate"
    if any(s.iteration != "accumulate" or s.aggregation != "accumulate" for s in lora):
        return "joint derived state routing without accumulate iteration"
    if any(s.exact or s.apply_once or s.axis_joint for s in lora):
        return "joint derived state routing with exact, single or axis changes"
    if any(s.direction_core is not None or s.expert_offsets is not None for s in lora):
        return "joint derived state routing with direction reads or expert offsets"
    if not mixed_separable and any(
        s.gate_head_limit is not None and compute.uses_adaptive(s) for s in lora
    ):
        return "joint derived state routing with gate head history"
    if any(s.form == "input" and compute.import_score(s) != "derived" for s in lora):
        return "joint derived state routing with a non-derived import score"
    imports = [s for s in lora if s.form == "input"]
    if not imports:
        return "multiple state sets without imports"
    settings = {
        (s.radial_step, s.orthogonal_step, s.radial_budget, compute._import_depth_chain(s))
        for s in imports
    }
    if len(settings) > 1:
        return "joint derived state routing with different import depth settings"
    if compute.import_fixed_read():
        return "joint derived state routing with fixed import reads"
    if any(any(s.cross_weights) for s in lora) or compute.derived_diffusion():
        return "routing crosses or derived diffusion"
    if any(s.learned_cross is not None and not s.learned_cross_zero for s in lora):
        return "joint derived state routing with learned crosses"
    if compute._IR_INTERPRET_ALL.get() or any(s.ir_parts for s in lora):
        return "joint derived state routing with IR parts"
    if compute._STEP_CARRY and compute._ablation.carry_steps(True):
        return "joint derived state routing with carry across sampler steps"
    if compute._BUDGET_RECORDS:
        return "joint derived state routing with radial budget recording"
    if os.environ.get("KILORA_DERIVED_POOL", "1,0,0").replace(" ", "") != "1,0,0":
        return "joint derived state routing with import pooling"
    experts = sum(compute.derived_keys(s)[0].shape[0] for s in lora)
    if experts > MAX_EXPERTS:
        return f"joint derived state routing with {experts} experts (limit {MAX_EXPERTS})"
    if any(routing_mode(s) is None for s in lora):
        return "joint derived state routing with unsupported selection or scope"
    if any(0 < s.candidates < s.keys_in.shape[0] for s in lora):
        return "joint derived state routing with candidate selection"
    if any(s.up.shape[-1] != y0.shape[-1] for s in lora):
        return "joint derived state routing with different output widths"
    if any(isinstance(v, torch.Tensor) and v.numel() != 1 for s in lora for v in (s.tau, s.scale)):
        return "joint derived state routing with per-element temperature or strength"
    if y0.dim() < 2:
        return "a layer output without a token dimension"
    if not fits(y0, sum(s.down.shape[0] for s in lora), tiled=True):
        return "joint derived state update tiles exceed shared memory"
    return None


def _kernel_adaptive_import(s) -> bool:
    return (
        s.form == "input"
        and compute.uses_adaptive(s)
        and compute.learned_kind(s) == "none"
        and s.gate_head_limit is None
        and not compute._GATE_STATS
        and not compute._route_freeze.RECORDS
        and os.environ.get("KILORA_FUSED_GATE", "kernel").strip() != "torch"
        and adaptive_kernel()
        and _p2(s.up.shape[0]) <= _COLUMN_BLOCK
    )


def _adaptive_import_gate(s, z, hist, y0, region, name, k, last):

    dev = y0.device
    lead, out = tuple(y0.shape[:-1]), y0.shape[-1]
    tokens = lead[-1]
    n = math.prod(lead)
    batch = n // tokens
    cols = z.shape[-1]
    f32 = torch.float32
    if "up_f32" not in hist:
        hist["up_f32"] = compute._as(s.up, f32).detach().contiguous()
        hist["first_norm"] = torch.empty(n, dtype=f32, device=dev)
        hist["first_rms"] = torch.empty((batch, out), dtype=f32, device=dev)
    zf = z.detach().reshape(n, cols).to(f32).contiguous()
    y2 = y0.reshape(n, out)
    reg = None if region is None else region.expand(*lead, 1).reshape(n).float().contiguous()
    before, history = hist.get("before"), hist.get("m")
    qa = torch.empty(n, dtype=f32, device=dev)
    a = torch.empty_like(qa)
    qb = torch.empty((batch, out), dtype=f32, device=dev)
    b = torch.empty_like(qb)
    next_before = torch.empty_like(zf)
    next_history = torch.empty_like(zf)
    expect = bool(last) or compute.expect_masks()
    reference = compute.base_reference()
    base = _i32(hash_base(f"{name}#0#adaptive", k))
    cp = _p2(cols)
    _adaptive_token_kernel[(batch, triton.cdiv(tokens, 32))](
        zf,
        before if before is not None else zf,
        history if history is not None else zf,
        hist["up_f32"],
        y2,
        reg if reg is not None else qa,
        qa,
        qa,
        hist["first_norm"],
        a,
        next_before,
        next_history,
        zf,
        tokens,
        cols,
        out,
        y2.stride(0),
        s.gate_kappa,
        base,
        HAS_BEFORE=before is not None,
        HAS_HIST=history is not None,
        HAS_REGION=reg is not None,
        RECORD_QA=False,
        EXPECT=expect,
        BASE_REFERENCE=reference,
        FIRST_STEP=k == 1,
        WRITE_ZA=False,
        USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
        BT=32,
        CP=cp,
        BO=32,
    )
    _adaptive_channel_kernel[(batch, triton.cdiv(out, 32))](
        zf,
        before if before is not None else zf,
        history if history is not None else zf,
        hist["up_f32"],
        y2,
        reg if reg is not None else qb,
        qb,
        hist["first_rms"],
        b,
        qb,
        qb,
        tokens,
        cols,
        out,
        y2.stride(0),
        s.gate_kappa,
        base,
        HAS_BEFORE=before is not None,
        HAS_HIST=history is not None,
        HAS_REGION=reg is not None,
        COHERENCE=compute._ablation.gate_coherence(s.gate_coherence),
        EXPECT=expect,
        BASE_REFERENCE=reference,
        FIRST_STEP=k == 1,
        WRITE_OUT=False,
        USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
        BT=32,
        CP=cp,
        BO=32,
    )
    hist["before"], hist["m"] = next_before, next_history
    hist["q"] = (qa.view(*lead, 1), qb.view(*lead[:-1], 1, out))
    return a.view(*lead, 1), b.view(*lead[:-1], 1, out)


def _derived_state_tables(lora, dt):
    keys = [compute.derived_keys(s) for s in lora]
    tensors = [v for s, k in zip(lora, keys, strict=True) for v in (s.up, s.down, k[0], k[1])]
    layout = tuple((s.form, s.per_expert) for s in lora)
    key = compute._tensor_key(f"fused-joint-derived-{dt}-{layout}", tensors)
    hit = compute._cache_get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        p = _Prepared()
        p.cols = [s.up.shape[0] for s in lora]
        p.experts = [k[0].shape[0] for k in keys]
        p.up = torch.cat([compute._as(s.up, dt).float() for s in lora], 0)
        p.gram = p.up @ p.up.t()
        p.keys, p.key_maps, p.read_maps, p.den = [], [], [], []
        indices = []
        offset = 0
        for s, k, e in zip(lora, keys, p.experts, strict=True):
            kin = compute._as(k[0], dt).flatten(0, 1)
            p.keys.append(kin)
            p.den.append((k[1].float() / k[0].shape[-1]).contiguous())
            p.key_maps.append(p.up @ kin.float().t() if s.form == "state" else None)
            down = compute._as(s.down, dt).float() if s.form == "state" else s.down.float()
            p.read_maps.append(p.up @ down.t() if down.shape[-1] == p.up.shape[-1] else None)
            indices.append(
                torch.arange(e, device=p.up.device).repeat_interleave(s.per_expert) + offset
            )
            offset += e
        p.column_expert = torch.cat(indices).contiguous()
    compute._cache_put(key, p)
    return p


def _run_derived_state(y0, lora, depth, target, regions, x, positions, shared=False):

    dt, dev = y0.dtype, y0.device
    lead, width = tuple(y0.shape[:-1]), y0.shape[-1]
    n = math.prod(lead)
    p = _derived_state_tables(lora, dt)
    regs = list(regions) if regions is not None else [None] * len(lora)
    pos = list(positions) if positions is not None else list(range(len(lora)))
    names = [
        f"{target}#input{i}" if s.form == "input" else compute._set_target(target, s, i)
        for i, s in zip(pos, lora, strict=True)
    ]
    ups = [compute._as(s.up, dt) for s in lora]
    scales = [s.scale.float() if isinstance(s.scale, torch.Tensor) else s.scale for s in lora]
    coefs, score_bases, fixed_logits, taus = [], [], [], []
    for i, (s, r) in enumerate(zip(lora, regs, strict=True)):
        base = (
            compute._state_read(s, y0, dt).float()
            if s.form == "state"
            else compute._input_read(s, x, dt).float()
        )
        coefs.append(base if r is None else base * r.float())
        score_bases.append(
            _state_base(y0.reshape(n, width), p.up, p.keys[i]) if s.form == "state" else None
        )
        fixed_logits.append(
            compute.pooled_derived_logits(s, x, target) if s.form == "input" else None
        )
        tau = torch.as_tensor(s.tau, device=dev, dtype=torch.float32).reshape(())
        taus.append(tau.expand(p.experts[i]).contiguous())
    joint = shared or any(s.form == "input" for s in lora)
    balance = joint and len(lora) > 1 and os.environ.get("KILORA_FILE_BALANCE", "1") != "0"
    keys = [compute.derived_keys(s) for s in lora]
    steps, low, frac = compute._depth_parts(depth) if depth > 1 else (1, 1, 0.0)
    chains = [steps > 1 and compute._import_depth_chain(s) for s in lora]
    settings = [
        compute._import_depth_settings(s) if chain and s.form == "input" else s
        for s, chain in zip(lora, chains, strict=True)
    ]
    splits = [compute.depth_split_on(s) for s in settings]
    metrics = [compute._step_metric(s, dev) for s in lora]
    split = [compute.learned_kind(s) == "separable" or compute.uses_adaptive(s) for s in lora]
    column_gated = any(split) and not all(split)
    kernel_gates = [_kernel_adaptive_import(s) for s in lora]
    if any(split):
        y0_norm, y0_rms = compute._y0_gate_stats(y0)
    hists = [{} for _ in lora]
    columns = [compute.column_keys(s) for s in lora]
    holds = [
        c * sc if s.carry == "dropped" else None
        for c, sc, s in zip(coefs, scales, lora, strict=True)
    ]
    zs, carried, alive = [None] * len(lora), [None] * len(lora), [None] * len(lora)
    refs, chain_refs = [None] * len(lora), [None] * len(lora)
    snap = None
    for k in range(1, steps + 1):
        z = None if k == 1 else torch.cat(carried, -1).reshape(n, -1)
        reads, logits = [], []
        for i, (s, r) in enumerate(zip(lora, regs, strict=True)):
            read = coefs[i]
            if z is not None and (s.form == "state" or chains[i]) and p.read_maps[i] is not None:
                added = (z.to(dt).float() if s.form == "state" else z) @ p.read_maps[i]
                added = added.reshape(*lead, p.cols[i])
                if s.expert_mask is not None:
                    added = added * s.expert_mask.to(added.dtype)
                if r is not None:
                    added = added * r.float()
                read = read + added
            reads.append(read)
            score = fixed_logits[i]
            if s.form == "state":
                score = _state_logits(
                    score_bases[i],
                    z,
                    p.key_maps[i],
                    p.gram,
                    p.den[i],
                    taus[i],
                    dt,
                    p.experts[i],
                    keys[i][0].shape[1],
                ).reshape(*lead, p.experts[i])
                score = compute._mask_derived_logits(s, score)
            if balance:
                score = score + 1.0 / (len(lora) * p.experts[i])
            logits.append(score.masked_fill(keys[i][1] == 0, -1e9))
        gates = compute._gates_from_logits(logits, lora, joint, 0.0, regs)
        reads_all = torch.cat(reads, -1).reshape(n, -1).contiguous()
        gates_all = torch.cat(gates, -1).reshape(n, -1).contiguous()
        survival = (
            torch.cat(
                [torch.ones_like(c) if a is None else a for c, a in zip(coefs, alive, strict=True)],
                -1,
            )
            .reshape(n, -1)
            .contiguous()
        )
        old = reads_all if k == 1 else torch.cat(zs, -1).reshape(n, -1).contiguous()
        updated = torch.empty_like(reads_all)
        additions = torch.empty_like(reads_all)
        _derived_update_kernel[(triton.cdiv(updated.numel(), 256),)](
            reads_all,
            gates_all,
            p.column_expert,
            survival,
            old,
            additions,
            updated,
            n,
            updated.shape[-1],
            E=sum(p.experts),
            HAS_OLD=k > 1,
            BLOCK=256,
            enable_fp_fusion=False,
        )
        parts = updated.reshape(*lead, -1).split(p.cols, -1)
        increments = additions.reshape(*lead, -1).split(p.cols, -1)
        y = y0
        last = k == steps
        write_output = last or (frac and k == low)
        output_gates = []
        output_values = []
        for i, (s, gate) in enumerate(zip(lora, gates, strict=True)):
            value = parts[i]
            if k > 1 and splits[i]:
                value = compute.depth_split_step(
                    zs[i],
                    increments[i],
                    settings[i],
                    metrics[i],
                    refs[i],
                    compute._weight_of(scales[i]),
                    names[i],
                )
            if chains[i]:
                size = compute._metric_size(value, metrics[i], compute._weight_of(scales[i]))
                if k == 1:
                    chain_refs[i] = size
                else:
                    budget = settings[i].radial_budget or compute.IMPORT_DEPTH_BUDGET
                    ratio = (budget * chain_refs[i] / size.clamp_min(1e-30)).clamp(max=1)
                    value = value * torch.where(size == 0, torch.ones_like(ratio), ratio)[..., None]
            zs[i] = value
            if splits[i] and k == 1 and settings[i].radial_budget > 0:
                refs[i] = compute._metric_size(value, metrics[i], compute._weight_of(scales[i]))
            if split[i]:
                if kernel_gates[i]:
                    a, b = _adaptive_import_gate(
                        s, value * scales[i], hists[i], y0, regs[i], names[i], k, last
                    )
                else:
                    a, b = compute.split_gate(
                        s,
                        value * scales[i],
                        hists[i],
                        y0_norm,
                        y0_rms,
                        gate,
                        target if s.form == "state" else names[i],
                        pos[i] if s.form == "state" else 0,
                        k,
                        regs[i],
                        last,
                    )
                if write_output:
                    if column_gated:
                        output_values.append(value * scales[i])
                        output_gates.append((a, b))
                    else:
                        y = channel_change(token_gate(value * scales[i], a, dt) @ ups[i], b, y)
            elif write_output:
                if column_gated:
                    output_values.append(value * scales[i])
                    output_gates.append((None, None))
                else:
                    y = y + (value * scales[i]).to(dt) @ ups[i]
        if column_gated and write_output:
            values = torch.cat(output_values, -1).contiguous()
            token_gates = torch.cat(
                [
                    torch.ones_like(v) if a is None else a.expand_as(v)
                    for v, (a, _) in zip(output_values, output_gates, strict=True)
                ],
                -1,
            ).contiguous()
            gated = torch.empty(values.shape, dtype=dt, device=dev)
            _derived_column_gate_kernel[(triton.cdiv(values.numel(), 256),)](
                values,
                token_gates,
                gated,
                values.numel(),
                BLOCK=256,
                INTERPRET=_interpret(),
                enable_fp_fusion=False,
            )
            for v, up, (_, b) in zip(gated.split(p.cols, -1), ups, output_gates, strict=True):
                delta = v @ up
                y = y + delta if b is None else channel_change(delta, b, y)
        if frac and k == low:
            snap = y
        if not last:
            for i, s in enumerate(lora):
                value = zs[i] * scales[i]
                m = compute._read_dropout_for_set(
                    value, coefs[i], s.per_expert, names[i], k, columns[i], s.key
                )
                if holds[i] is None:
                    carried[i] = m * value
                else:
                    holds[i] = (
                        value
                        if compute._ablation.read_dropout_off()
                        else holds[i] + (1 - m) * (value - holds[i])
                    )
                    carried[i] = holds[i]
                if chains[i]:
                    alive[i] = m if alive[i] is None else alive[i] * m
    if frac:
        y = snap + frac * (y - snap)
    if depth < 1:
        y = y0 + depth * (y - y0)
    return y


def derived_batched_step(s, c0, logits, region, previous, scale=1.0, *, entry=None):
    p = _prepared(s, c0.dtype)
    lead = tuple(logits.shape[:-1])
    tokens = lead[-1]
    batch = math.prod(lead[:-1])
    rows = batch * tokens
    c = p.C
    experts = p.E
    reads = c0.expand(*lead, c).reshape(rows, c).contiguous()
    lg = torch.zeros((rows, p.EP), dtype=torch.float32, device=c0.device)
    lg[:, :experts] = logits.reshape(rows, experts)
    reg = None if region is None else region.expand(*lead, 1).reshape(rows).contiguous().float()
    weights = torch.empty(
        (batch, tokens if routing_mode(s)[0] else 1, p.EP), dtype=torch.float32, device=c0.device
    )
    token, top1 = routing_mode(s)
    result = torch.empty_like(reads)
    old = reads if previous is None else previous.reshape(rows, c).contiguous()
    dummy = reads
    entry_values = (
        dummy if entry is None else entry.expand(*lead, experts).reshape(rows, experts).contiguous()
    )
    scale_ptr, scale_value, scale_is_tensor = _arg(scale)
    bt = 32
    cp = min(p.CP, _COLUMN_BLOCK)
    _update_kernel[(batch, triton.cdiv(tokens, bt), triton.cdiv(c, cp))](
        lg,
        entry_values,
        reg if reg is not None else dummy,
        reads,
        reads.stride(0),
        dummy,
        dummy,
        p.G,
        0,
        p.CP,
        old,
        result,
        weights,
        weights.stride(0),
        p.EP if token else 0,
        scale_ptr if scale_is_tensor else dummy,
        scale_value,
        p.segc,
        p.segt,
        dummy,
        *(dummy,) * 8,
        0,
        dummy,
        0,
        p.colkey,
        dummy,
        compute.DROPOUT_DELTA,
        dummy,
        0.0,
        dummy,
        dummy,
        1,
        dummy,
        dummy,
        tokens,
        c,
        p.R,
        experts,
        HAS_DEPTH=False,
        HAS_OLD=False,
        BLEND_DEPTH=False,
        BLEND_REGION=False,
        TOKEN=token,
        TOP1=top1,
        HAS_ENTRY=entry is not None,
        HAS_REGION=reg is not None,
        HAS_Z=False,
        READ_G=False,
        HOLD0=False,
        HAS_A=False,
        ACCUM=previous is not None and s.iteration != "path",
        SCALE_PTR=scale_is_tensor,
        SEPARABLE=False,
        HAS_ZP=False,
        DO_DROP=False,
        CARRY_END=False,
        DROPPED=False,
        WRITE_OUT=False,
        HAS_ZL=False,
        WRITE_A=False,
        USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
        EXPECT=False,
        HAS_COLKEY=False,
        BT=bt,
        CP=cp,
        SP=p.CP,
        EP=p.EP,
    )
    gates = weights[..., :experts].expand(batch, tokens, experts).reshape(*lead, experts)
    return gates, result.reshape(*lead, c)


def run_fused(
    y0,
    sets,
    depth,
    shared=False,
    target="",
    regions=None,
    carry=None,
    _drop_last=False,
    x=None,
    name=None,
    td=None,
    entry=None,
    path_stats=None,
    positions=None,
):
    if depth <= 0 and compute._derived_state_sets(sets):
        return y0
    if compute._derived_state_sets(sets) and len([s for s in sets if s.down is not None]) > 1:
        keep = [i for i, s in enumerate(sets) if s.down is not None]
        lora = [sets[i] for i in keep]
        regs = list(regions) if regions is not None else [None] * len(sets)
        reason = _derived_state_reason(y0, lora, td)
        if not HAVE_TRITON or not device_ok(y0):
            reason = "joint derived state routing needs Triton on CUDA, ROCm or the CPU interpreter"
        if carry is not None or _drop_last or entry is not None or path_stats is not None:
            reason = "joint derived state routing with explicit carry or path diagnostics"
        if reason is not None:
            _note(reason)
            return compute._run_input(x, y0, sets, keep, depth, regs, target, td, shared=shared)
        with torch.no_grad():
            return _run_derived_state(
                y0, lora, depth, target, [regs[i] for i in keep], x, positions, shared
            )
    if not _CACHE_LOGGED:
        _configure_cache()
    lora = [s for s in sets if s.down is not None and s.form in ("state", "input")]
    s = lora[0]
    region = (regions or [None])[0]
    state = s.form == "state"
    inp = True
    position = positions[0] if state and positions is not None else 0
    drop_name = compute._set_target(target, s, position) if state else name
    gate_target = target if state else name
    dt = y0.dtype
    dev = y0.device
    lead = tuple(y0.shape[:-1])
    T = lead[-1]
    B = math.prod(lead[:-1]) if len(lead) > 1 else 1
    N = B * T
    p = _prepared(s, dt)
    C, E, OUT = p.C, p.E, p.OUT
    f32 = torch.float32
    with torch.no_grad():
        y2 = y0.reshape(N, OUT)
        by_read = compute.import_score(s) in ("read_share", "derived")
        by_derived = compute.import_score(s) == "derived"
        gamma = compute.share_transition_gamma(s) if by_read and not state else 0.0
        transition = compute.share_transition(s) if gamma else None
        if state or by_read:
            q0 = None
        elif inp:
            q0 = F.linear(y2.float(), p.router_f32)
        else:
            q0 = F.linear(y2, p.router_dt)
        reg = region.reshape(N).contiguous().float() if region is not None else None
        if inp:
            value = y0 if state else x
            if state:
                state_base = F.linear(y2, p.state_joint)
                c0 = F.linear(y2, p.down_dt)
            else:
                c0 = F.linear(value.to(dt).reshape(N, value.shape[-1]), p.down_dt)
            state_c0 = c0 if state else None
            if s.expert_mask is not None:
                em = s.expert_mask.to(dev, dt).expand(*lead, C).reshape(N, C)
                c0 = c0 * em
            if reg is not None:
                c0 = c0.float() * reg[:, None]
        else:
            c0 = F.linear(y2, p.down_dt)
        tau_p, tau_v, tau_is = _arg(s.tau)
        sc_p, sc_v, sc_is = _arg(s.scale)
        if state:
            state_energy = torch.empty(N, dtype=f32, device=dev)
            tau_p, tau_v, tau_is = p.state_tau, 0.0, True
            sc_p, sc_v, sc_is = p.state_scale, 0.0, True
        dummy = c0
        steps, low, frac = compute._depth_parts(depth)
        if state and depth < 1:
            steps, low, frac = 1, 1, 0.0
        original_steps = steps
        chain = inp and steps > 1 and compute._import_depth_chain(s)
        band = (
            "after"
            if s.route_freeze is not None and compute.current_progress() >= s.route_freeze
            else "before"
        )
        fitted = None if s.jump_fits is None else s.jump_fits.get((band, depth))
        jump = (
            fitted is not None
            and not state
            and transition is None
            and band == "before"
            and fitted[1] <= s.jump_threshold
            and steps > 1
            and not frac
            and td is None
            and not compute._STEP_CARRY
            and carry is None
            and not linear_jump._RECORDS
            and not compute._route_freeze.RECORDS
            and compute.learned_kind(s) != "separable"
            and (compute.base_reference() or not compute.uses_adaptive(s))
        )
        if jump:
            steps, low = 1, 1
        first_z = None
        if td is not None:
            steps, low, frac = td.steps, td.steps, 0.0
            depth_rows = td.d.contiguous()
            out = y0
        fold = (
            not state
            and by_read
            and not chain
            and entry is None
            and transition is None
            and compute._import_fold_enabled()
            and steps > 1
            and (not frac or (not compute.uses_adaptive(s) and dt == torch.float32))
            and td is None
            and not jump
            and not _drop_last
            and not compute._route_freeze.RECORDS
            and not linear_jump._RECORDS
            and not _TRACE
            and compute.learned_kind(s) == "none"
            and (s.aggregation != "replace" or s.iteration == "path")
        )
        fold_count = low + frac if frac else steps
        if fold:
            frac = 0.0
        if not state:
            start, carry_out = None, False
        elif carry is None:
            start = compute._carry_start(target, lead, 1)
            carry_out = bool(compute._STEP_CARRY)
        else:
            start = carry.get(target)
            if start is not None and tuple(start["carried"][0].shape[:-1]) != lead:
                start = None
            carry_out = True
        sig0 = c0
        if steps > 1 or _drop_last or carry_out:
            sig0 = c0.float().reshape(N, p.E, p.R).pow(2).mean(-1).sqrt()
        dropped = s.carry == "dropped"
        accum_set = s.aggregation != "replace" and not (inp and s.iteration == "path")
        zm = None
        hold0 = False
        z = None
        if start is not None:
            zm = start["carried"][0].reshape(N, C).to(f32).contiguous()
            if accum_set:
                z = start["change"][0].reshape(N, C).float()
        elif dropped:
            hold0 = True
        zm_valid = zm is not None
        if zm is None and (steps > 1 or carry_out or _drop_last):
            zm = torch.empty((N, C), dtype=f32, device=dev)
        if chain:
            chain_set = compute._import_depth_settings(s)
            chain_metric = compute._step_metric(s, dev)
            chain_ref = split_ref = None
            bound_ref = torch.empty(N, dtype=f32, device=dev)
            projected_prev = None
            survival = torch.ones((N, C), dtype=f32, device=dev)
            if start is None:
                zm = c0.float() * s.scale if dropped else torch.zeros((N, C), dtype=f32, device=dev)
            read_cross = compute._import_read_cross(s, [p.up_dt])[0]
            read_mask = None
            if s.expert_mask is not None:
                read_mask = s.expert_mask.to(dev, f32).expand(*lead, C).reshape(N, C)
        token, top1 = routing_mode(s)
        expect = compute.expect_masks()
        lg = torch.empty((N, p.EP), dtype=f32, device=dev)
        gates_w = torch.empty((B, T if token else 1, p.EP), dtype=f32, device=dev)
        pb, pt = gates_w.stride(0), (p.EP if token else 0)
        prev = None
        a_prev = None
        g_mat, g_bs, g_rs = p.G, 0, p.CP
        h_mat, h_bs, h_rs = p.H, 0, p.RDP
        region_t = region.reshape(*lead, 1) if region is not None else None
        z_start = z.view(*lead, C) if z is not None else None
        adaptive = (
            compute.uses_adaptive(s)
            and compute.learned_kind(s) == "none"
            and td is None
            and not compute._GATE_STATS
            and os.environ.get("KILORA_FUSED_GATE", "kernel").strip() != "torch"
            and adaptive_kernel()
            and not (by_read and p.CP > _COLUMN_BLOCK)
            and (not fold or not compute.base_reference())
            and not (path_stats and path_stats.get("forced_gates"))
        )
        tgate = None if adaptive else _torch_gate(s, p, start, z_start, region_t, position)
        hist = compute.gate_history(start, 0, z_start, region_t) if adaptive else None
        gate_base_reference = compute.base_reference()
        first_norm = torch.empty(N, dtype=f32, device=dev) if adaptive else None
        first_rms = torch.empty((B, OUT), dtype=f32, device=dev) if adaptive else None
        before_gate = hist.get("before") if adaptive else None
        history_gate = hist.get("m") if adaptive else None
        if before_gate is not None:
            before_gate = before_gate.reshape(N, C).contiguous()
        if history_gate is not None:
            history_gate = history_gate.reshape(N, C).contiguous()
        sep = p.separable and tgate is None
        if tgate is not None:
            y0_stats = compute._y0_gate_stats(y0)
        if sep:
            y0_norm = torch.empty(N, dtype=f32, device=dev)
            ST = 256
            n_st = triton.cdiv(T, ST)
            parts = torch.empty((B, n_st, OUT), dtype=f32, device=dev)
            _stats_kernel[(B, n_st)](y2, y2.stride(0), y0_norm, parts, T, OUT, BT=ST, BO=32)
        z_prev = None
        sm_prev = None
        b_gate = None
        gate_start = (
            None
            if state
            else hist.get("q")
            if adaptive
            else tgate.start
            if tgate is not None
            else None
        )
        if gate_start is not None:
            a_prev = gate_start[0].reshape(N).float().contiguous()
            b_gate = gate_start[1].reshape(B, OUT).float()
            gh = _gated_read(p, b_gate)
            g_mat, g_bs, g_rs = gh, gh.stride(0), gh.stride(1)
            h_mat, h_bs, h_rs = gh[:, :, p.CP :], gh.stride(0), gh.stride(1)
        if sep:
            gate_parts = (p.gram, p.tp0, p.tp1, p.tw1, p.tb1, p.tw2, p.tb2, y0_norm)
        snap = None
        if frac and low == 0:
            before = _hold_now(zm, hold0, c0, s) if (zm_valid or hold0) else None
            snap = (None, before, None if z is None else z.clone(), None, None)
        BT = 32
        odt = torch.float32 if _interpret() else dt
        grid = (B, triton.cdiv(T, BT))
        za_out = None
        inp_gate = None
        y_low = None
        if by_derived and not state:
            lg[:, :E] = compute.pooled_derived_logits(
                s, x.reshape(*lead, x.shape[-1]), target
            ).reshape(N, E)
        elif by_read and not state:
            (_read_share_kernel if p.R <= _NARROW_RANK else _read_share_wide_kernel)[grid](
                c0,
                c0.stride(0),
                tau_p if tau_is else dummy,
                tau_v,
                lg,
                T,
                C,
                R=p.R,
                E=E,
                EP=p.EP,
                RB=min(64, _p2(p.R)),
                TAU_PTR=tau_is,
                BT=BT,
            )
        base_logits = (
            lg.clone()
            if transition is not None or entry is not None or (by_derived and not state)
            else None
        )
        entry_values = (
            entry.expand(*lead, E).reshape(N, E).float().contiguous()
            if entry is not None
            else dummy
        )
        coefficient_blend = td is not None and tgate is None and not s.exact
        if coefficient_blend:
            z_blend = torch.empty((N, C), dtype=f32, device=dev)
        exact_weights = None
        exact_extra = None
        exact_low = None
        for k in [steps] if fold else range(1, steps + 1):
            step_read = c0
            if state:
                state_z = zm if k > 1 or start is not None else None
                if state_z is not None:
                    step_read = torch.empty((N, C), dtype=f32, device=dev)
                _state_score_read_kernel[(triton.cdiv(N, 16),)](
                    state_base,
                    y2,
                    state_energy,
                    state_z if state_z is not None else c0,
                    p.state_key_pad,
                    p.state_gram_pad,
                    p.state_read_pad,
                    state_c0,
                    em if s.expert_mask is not None else c0,
                    reg if reg is not None else c0,
                    p.den,
                    p.state_tau,
                    p.state_seg,
                    lg,
                    step_read,
                    N,
                    state_c0.stride(0),
                    OUT,
                    C,
                    E * p.M,
                    E=E,
                    EP=p.EP,
                    CP=p.state_cp,
                    KP=p.state_kp,
                    BT=16,
                    FIRST=k == 1,
                    HAS_Z=state_z is not None,
                    HAS_MASK=s.expert_mask is not None,
                    HAS_REGION=reg is not None,
                    INTERPRET=_interpret(),
                    enable_fp_fusion=False,
                )
            if chain and k > 1 and not state:
                if read_cross is not None:
                    added = zm @ read_cross
                    if read_mask is not None:
                        added = added * read_mask
                    if reg is not None:
                        added = added * reg[:, None]
                    step_read = c0.float() + added
                if by_read and not by_derived:
                    (_read_share_kernel if p.R <= _NARROW_RANK else _read_share_wide_kernel)[grid](
                        step_read,
                        step_read.stride(0),
                        tau_p if tau_is else dummy,
                        tau_v,
                        lg,
                        T,
                        C,
                        R=p.R,
                        E=E,
                        EP=p.EP,
                        RB=min(64, _p2(p.R)),
                        TAU_PTR=tau_is,
                        BT=BT,
                    )
                    if transition is not None:
                        base_logits = lg.clone()
            step_survival = survival if chain else None
            if chain and s.exact:
                step_survival = survival.reshape(N, E, p.R).mean(-1).repeat_interleave(p.R, -1)
            update_read = step_read * step_survival if chain else step_read
            use_hold0 = hold0 and k == 1
            has_z = zm_valid
            if entry is not None and k == 1:
                lg[:, :E] = entry.expand(*lead, E).reshape(N, E) * 20.0
            elif entry is not None and k == 2:
                lg[:, :E] = base_logits[:, :E]
            if by_derived and not state and k > 1 and compute.derived_mode() == "full":
                lg[:, :E] = base_logits[:, :E] * compute.derived_lift_after()
            if transition is not None and prev is not None:
                prior = gates_w[..., :E].expand(B, T, E).reshape(N, E).float() @ transition
                lg[:, :E] = (
                    base_logits[:, :E]
                    * (
                        compute.derived_lift_after()
                        if by_derived and compute.derived_mode() == "full"
                        else 1.0
                    )
                    + gamma * (prior + 1e-6).log()
                )
            if path_stats is not None and "forced_routes" in path_stats:
                routes = path_stats["forced_routes"][k - 1]
                lg[:, :E] = routes.expand(*lead, E).reshape(N, E)
            if not state and not by_read and not (path_stats and "forced_routes" in path_stats):
                _route_kernel[grid](
                    q0,
                    q0.stride(0),
                    zm if zm is not None else dummy,
                    c0,
                    c0.stride(0),
                    a_prev if a_prev is not None else dummy,
                    h_mat,
                    h_bs,
                    h_rs,
                    gates_w if prev is not None else dummy,
                    pb,
                    pt,
                    p.kin,
                    p.kout,
                    p.seg,
                    p.den,
                    tau_p if tau_is else dummy,
                    tau_v,
                    sc_p if sc_is else dummy,
                    sc_v,
                    lg,
                    T,
                    C,
                    p.RD,
                    p.MO,
                    HAS_Z=has_z and not state,
                    HAS_A=a_prev is not None and not state,
                    HAS_PREV=prev is not None and not state,
                    TAU_PTR=tau_is,
                    SCALE_PTR=sc_is,
                    DERIVED=state,
                    INTERPRET=_interpret(),
                    BT=BT,
                    CP=p.CP,
                    RDP=p.RDP,
                    EMP=p.EMP,
                    EOP=p.EOP,
                    EP=p.EP,
                )
            if by_derived and s.expert_mask is not None:
                lg[:, :E] = compute._mask_derived_logits(s, lg[:, :E].reshape(*lead, E)).reshape(
                    N, E
                )
            last = k == steps
            do_drop = not last or _drop_last
            carry_end = last and carry_out and not frac and not _drop_last and td is None
            write_out = (
                last
                and not (frac and sep)
                and tgate is None
                and not adaptive
                and not inp
                and td is None
                and not jump
            )
            zl = None
            if write_out and frac:
                zl = snap[2] if low > 0 else None
                if low == 0:
                    zl = torch.zeros((N, C), dtype=f32, device=dev)
            z_new = torch.empty((N, C), dtype=f32, device=dev)
            z_kept = torch.empty_like(z_new) if td is not None else None
            a_new = (
                torch.empty(N, dtype=f32, device=dev)
                if (sep or reg is not None) and tgate is None and not inp
                else None
            )
            if write_out:
                za_out = torch.empty((N, C), dtype=odt, device=dev)
            update_cp = min(p.CP, _COLUMN_BLOCK) if by_read else p.CP
            update_grid = (B, triton.cdiv(T, BT), triton.cdiv(C, update_cp))
            _update_kernel[update_grid](
                lg,
                entry_values,
                reg if reg is not None else dummy,
                update_read,
                update_read.stride(0),
                zm if zm is not None else dummy,
                a_prev if a_prev is not None else dummy,
                g_mat,
                g_bs,
                g_rs,
                z if z is not None else dummy,
                z_new,
                gates_w,
                pb,
                pt,
                sc_p if sc_is else dummy,
                sc_v,
                p.segc,
                p.segt,
                z_prev if (sep and z_prev is not None) else dummy,
                *(gate_parts if sep else (dummy,) * 8),
                _i32(hash_base(f"{target}#0#gate", k)) if sep else 0,
                a_new if a_new is not None else dummy,
                _i32(hash_base(drop_name, k)) if do_drop else 0,
                p.colkey,
                sig0,
                compute.DROPOUT_DELTA,
                zl if zl is not None else dummy,
                float(frac),
                za_out if write_out else dummy,
                depth_rows if td is not None else dummy,
                k,
                z_kept if z_kept is not None else dummy,
                z_blend if coefficient_blend else dummy,
                T,
                C,
                p.R,
                E,
                HAS_DEPTH=td is not None and not chain,
                HAS_OLD=z is not None,
                BLEND_DEPTH=coefficient_blend,
                BLEND_REGION=reg is not None and not inp,
                TOKEN=token,
                TOP1=top1,
                HAS_ENTRY=entry is not None and k == 1,
                HAS_REGION=reg is not None,
                HAS_Z=has_z,
                READ_G=not inp,
                HOLD0=use_hold0,
                HAS_A=a_prev is not None,
                ACCUM=accum_set and z is not None,
                SCALE_PTR=sc_is,
                SEPARABLE=sep,
                HAS_ZP=sep and z_prev is not None,
                DO_DROP=do_drop and not chain,
                CARRY_END=carry_end,
                DROPPED=dropped,
                WRITE_OUT=write_out,
                HAS_ZL=zl is not None,
                WRITE_A=a_new is not None,
                USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
                EXPECT=expect,
                HAS_COLKEY=p.has_colkey and do_drop,
                BT=BT,
                CP=update_cp,
                SP=p.CP,
                EP=p.EP,
            )
            old_exact_extra = exact_extra
            if chain:
                ratio = None
                if not s.exact:
                    projected = z_new @ chain_metric
                    next_projected = torch.empty_like(z_new)
                    _import_bound_kernel[(triton.cdiv(N, 4),)](
                        z_new,
                        z if z is not None else dummy,
                        projected,
                        projected_prev if projected_prev is not None else dummy,
                        next_projected,
                        bound_ref,
                        depth_rows if td is not None else dummy,
                        N,
                        C,
                        k,
                        HAS_OLD=z is not None and k > 1,
                        HAS_DEPTH=td is not None,
                        BUDGET=chain_set.radial_budget,
                        RADIAL=chain_set.radial_step,
                        ORTHOGONAL=chain_set.orthogonal_step,
                        SPLIT=not state or compute.depth_split_on(s),
                        KEEP_ZERO=state,
                        BT=4,
                        CP=p.CP,
                        enable_fp_fusion=False,
                    )
                    projected_prev = next_projected
                else:
                    if z is not None:
                        z_new = compute.depth_split_step(
                            z, z_new - z, chain_set, chain_metric, split_ref
                        )
                    if s.exact:
                        weights = gates_w[..., :E].expand(B, T, E).reshape(*lead, E).clone()
                        alive = survival.view(*lead, E, p.R).mean(-1)
                        weights = weights * alive
                        value = x
                        if k > 1 and x.shape[-1] == OUT:
                            value = value + (zm.to(dt) @ p.up_dt).view(*lead, OUT)
                        layer = s
                        if s.expert_inputs is not None:
                            change = value - x
                            layer = dataclasses.replace(
                                s,
                                expert_inputs=[
                                    None if v is None else v + change for v in s.expert_inputs
                                ],
                            )
                        extra = (
                            compute._exact_correction(
                                layer, value, step_read.view(*lead, C), weights, region_t, dt
                            )
                            * s.scale
                        )
                        exact_extra = extra if exact_extra is None else exact_extra + extra
                        exact_change = (z_new.to(dt) @ p.up_dt).view(*lead, OUT) + exact_extra
                        size = exact_change.float().norm(dim=-1).reshape(N)
                    else:
                        size = compute._metric_size(z_new, chain_metric)
                    ratio = None
                    if chain_ref is None:
                        chain_ref = size
                        split_ref = compute._metric_size(z_new, chain_metric)
                    else:
                        ratio = (chain_set.radial_budget * chain_ref / size.clamp_min(1e-30)).clamp(
                            max=1
                        )
                        if s.exact:
                            exact_extra = exact_extra * ratio.reshape(*lead, 1)
                record_keep = do_drop and (
                    bool(compute._route_freeze.RECORDS)
                    or bool(path_stats and path_stats.get("record_macro"))
                )
                keeps = torch.empty_like(z_new) if record_keep else dummy
                forced_keep = path_stats.get("forced_read_keeps") if path_stats else None
                _import_carry_kernel[(triton.cdiv(N * C, 256),)](
                    z_new,
                    z if z is not None else dummy,
                    zm,
                    survival,
                    sig0,
                    p.colkey,
                    ratio.reshape(N) if ratio is not None else dummy,
                    z_kept if z_kept is not None else dummy,
                    z_blend if coefficient_blend else dummy,
                    depth_rows if td is not None else dummy,
                    keeps,
                    forced_keep[k - 1].reshape(N, C) if forced_keep and do_drop else dummy,
                    N,
                    C,
                    p.R,
                    E,
                    k,
                    _i32(hash_base(drop_name, k)),
                    HAS_OLD=z is not None,
                    HAS_DEPTH=td is not None,
                    BLEND=coefficient_blend,
                    DO_DROP=do_drop,
                    DROPPED=dropped,
                    HAS_RATIO=ratio is not None,
                    HAS_COLKEY=p.has_colkey,
                    EXPECT=expect,
                    RECORD_KEEP=record_keep,
                    FORCED_KEEP=bool(forced_keep) and do_drop,
                    USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
                    DELTA=compute.DROPOUT_DELTA,
                    BLOCK=256,
                )
                if record_keep:
                    if path_stats is not None and path_stats.get("record_macro"):
                        path_stats.setdefault("read_keeps", []).append(keeps.view(*lead, C).clone())
                    for rec in compute._route_freeze.RECORDS:
                        rec.setdefault("input_keeps", {})[(target, k)] = {
                            0: keeps.view(*lead, C).clone()
                        }
            if entry is not None:
                path_gates = gates_w[..., :E].reshape(*lead[:-1], T if token else 1, E)
                compute._derived_path_note(
                    path_stats,
                    entry if k == 1 else path_gates.clone(),
                    compute.derived_logits(s, x),
                    None if td is None else td.active(k),
                )
            if fold:
                part = z_new.view(*lead, C)
                if tgate is not None:
                    compute._import_fold_history(
                        part, 1.0, steps, s.iteration, region_t, tgate.hist, s.up
                    )
                z_new = (part if s.iteration == "path" else part * fold_count).reshape(N, C)
            if compute._route_freeze.RECORDS:
                weights = gates_w[..., :E].reshape(*lead[:-1], T if token else 1, E)
                compute._route_freeze.note_routes(target, [weights.clone()])
            if linear_jump._RECORDS and k == 1:
                first_z = z_new.clone()
            if (
                linear_jump._RECORDS
                and k == original_steps
                and first_z is not None
                and not frac
                and td is None
            ):
                linear_jump.note(target, depth, first_z, z_new)
            if jump:
                z_new = z_new + z_new @ fitted[0].to(device=dev, dtype=z_new.dtype)
            if sep:
                zz = z_new.reshape(B, T, C)
                sm = torch.matmul(zz.transpose(-1, -2), zz)
                spm = sm
                if z_prev is not None:
                    if sm_prev is not None:
                        spm = sm_prev
                    else:
                        zpp = z_prev.reshape(B, T, C)
                        spm = torch.matmul(zpp.transpose(-1, -2), zpp)
                sm_prev = sm
                wbar = gates_w.mean(1) if token else gates_w[:, 0]
                b_gate = torch.empty((B, OUT), dtype=f32, device=dev)
                upb = None
                BO = 64
                _channel_kernel[(B, triton.cdiv(OUT, BO))](
                    sm,
                    spm,
                    p.up_pad,
                    parts,
                    wbar.contiguous(),
                    p.EP,
                    p.cw1,
                    p.cb1,
                    p.cw2,
                    p.cb2,
                    b_gate,
                    upb if upb is not None else b_gate,
                    OUT,
                    C,
                    E,
                    N,
                    _i32(hash_base(f"{target}#0#gate", k)),
                    n_st,
                    T,
                    HAS_ZP=z_prev is not None,
                    WRITE_UPB=upb is not None,
                    EXPECT=expect,
                    CP=p.CP,
                    BO=BO,
                    BS=256,
                )
                if not last:
                    gh = _gated_read(p, b_gate)
                    g_mat, g_bs, g_rs = gh, gh.stride(0), gh.stride(1)
                    h_mat, h_bs, h_rs = gh[:, :, p.CP :], gh.stride(0), gh.stride(1)
            if adaptive:
                qa = torch.empty(N, dtype=f32, device=dev)
                raw_qa = torch.empty_like(qa) if compute._route_freeze.RECORDS else None
                a_new = torch.empty_like(qa)
                qb = torch.empty((B, OUT), dtype=f32, device=dev)
                b_gate = torch.empty_like(qb)
                next_before = torch.empty_like(z_new)
                next_history = torch.empty_like(z_new)
                za = torch.empty((N, C), dtype=odt, device=dev)
                base = _i32(hash_base(f"{gate_target}#0#adaptive", k))
                _adaptive_token_kernel[grid](
                    z_new,
                    before_gate if before_gate is not None else dummy,
                    history_gate if history_gate is not None else dummy,
                    p.up_pad,
                    y2,
                    reg if reg is not None else dummy,
                    qa,
                    raw_qa if raw_qa is not None else dummy,
                    first_norm,
                    a_new,
                    next_before,
                    next_history,
                    za,
                    T,
                    C,
                    OUT,
                    y2.stride(0),
                    s.gate_kappa,
                    base,
                    HAS_BEFORE=before_gate is not None,
                    HAS_HIST=history_gate is not None,
                    HAS_REGION=reg is not None,
                    RECORD_QA=raw_qa is not None,
                    EXPECT=last or expect,
                    BASE_REFERENCE=gate_base_reference,
                    FIRST_STEP=k == 1,
                    FOLDED=fold,
                    FOLD_COUNT=steps,
                    PATH_FOLD=s.iteration == "path",
                    WRITE_ZA=True,
                    USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
                    BT=32,
                    CP=p.CP,
                    BO=32,
                )
                fast_output = last and not frac and (not _interpret() or dt == torch.float32)
                delta = torch.matmul(za.to(dt).view(*lead, C), p.up_dt) if fast_output else None
                result = torch.empty_like(y0) if fast_output else None
                _adaptive_channel_kernel[(B, triton.cdiv(OUT, 32))](
                    z_new,
                    before_gate if before_gate is not None else dummy,
                    history_gate if history_gate is not None else dummy,
                    p.up_pad,
                    y2,
                    reg if reg is not None else dummy,
                    qb,
                    first_rms,
                    b_gate,
                    delta if delta is not None else dummy,
                    result if result is not None else dummy,
                    T,
                    C,
                    OUT,
                    y2.stride(0),
                    s.gate_kappa,
                    base,
                    HAS_BEFORE=before_gate is not None,
                    HAS_HIST=history_gate is not None,
                    HAS_REGION=reg is not None,
                    COHERENCE=compute._ablation.gate_coherence(s.gate_coherence),
                    EXPECT=last or expect,
                    BASE_REFERENCE=gate_base_reference,
                    FIRST_STEP=k == 1,
                    FOLDED=fold,
                    FOLD_COUNT=steps,
                    PATH_FOLD=s.iteration == "path",
                    WRITE_OUT=fast_output,
                    USE_LIBDEVICE=_HAVE_LIBDEVICE and not _interpret(),
                    BT=32,
                    CP=p.CP,
                    BO=32,
                )
                if compute._route_freeze.RECORDS:
                    compute._route_freeze.note_gate(
                        gate_target, 0, raw_qa.view(*lead, 1), qb.view(*lead[:-1], 1, OUT)
                    )
                hist["m"] = next_history.view(*lead, C)
                hist["q"] = (qa.view(*lead, 1), qb.view(*lead[:-1], 1, OUT))
                before_gate, history_gate = next_before, next_history
                if inp:
                    inp_gate = (a_new.view(*lead, 1), b_gate.view(*lead[:-1], 1, OUT))
                elif not last:
                    gh = _gated_read(p, b_gate)
                    g_mat, g_bs, g_rs = gh, gh.stride(0), gh.stride(1)
                    h_mat, h_bs, h_rs = gh[:, :, p.CP :], gh.stride(0), gh.stride(1)
            if tgate is not None:
                zz = z_new.view(*lead, C)
                gw = gates_w[..., :E].expand(B, T, E).reshape(*lead, E)
                forced = path_stats.get("forced_gates") if path_stats is not None else None
                if forced:
                    ab = forced[k - 1]
                elif td is None:
                    ab = tgate(zz, y0_stats, gw, gate_target, steps if fold else k, last)
                else:
                    ab = tgate(zz, y0_stats, gw, gate_target, k, td.last(k), td.active(k))
                if inp:
                    inp_gate = ab
                elif ab is None:
                    a_new, b_gate = None, None
                    a_prev = None
                    g_mat, g_bs, g_rs = p.G, 0, p.CP
                    h_mat, h_bs, h_rs = p.H, 0, p.RDP
                else:
                    a_t, b_t = ab
                    a_new = a_t.reshape(N).float().contiguous()
                    b_gate = b_t.reshape(B, OUT).float()
                    if not last:
                        gh = _gated_read(p, b_gate)
                        g_mat, g_bs, g_rs = gh, gh.stride(0), gh.stride(1)
                        h_mat, h_bs, h_rs = gh[:, :, p.CP :], gh.stride(0), gh.stride(1)
            if (
                entry is not None
                and path_stats is not None
                and path_stats.get("record_macro")
                and inp_gate is not None
            ):
                path_stats.setdefault("output_gates", []).append(
                    tuple(v.detach().clone() for v in inp_gate)
                )
            if inp and s.exact and not chain:
                weights = gates_w[..., :E].expand(B, T, E).reshape(*lead, E).clone()
                if fold and s.iteration != "path":
                    weights = weights * fold_count
                old_exact_weights = exact_weights
                exact_weights = (
                    weights
                    if exact_weights is None or s.iteration == "path"
                    else exact_weights + weights
                )
                exact_extra = (
                    compute._exact_correction(
                        s, x, c0.reshape(*lead, C), exact_weights, region_t, dt
                    )
                    * s.scale
                )
                if inp_gate is not None:
                    a, b = inp_gate
                    exact_extra = exact_extra * a.to(dt) * b.to(dt)
                if td is not None:
                    exact_weights = td.keep(exact_weights, old_exact_weights, k)
            if chain and exact_extra is not None:
                exact_output = exact_extra
                if inp_gate is not None:
                    exact_output = exact_output * inp_gate[0].to(dt) * inp_gate[1].to(dt)
            else:
                exact_output = exact_extra
            if td is not None and not coefficient_blend:
                if inp:
                    y_k = _input_output(y0, z_new, inp_gate, p, lead, dt)
                    if exact_output is not None:
                        y_k = y_k + exact_output
                else:
                    za = z_new if reg is None else z_new * reg[:, None]
                    y_k = y0 + torch.matmul(za.view(*lead, C).to(dt), p.up_dt)
                if dt == torch.float32:
                    blended = torch.empty_like(y_k)
                    _depth_blend_kernel[(triton.cdiv(N * OUT, 256),)](
                        out, y_k, depth_rows, blended, OUT, N * OUT, k, BLOCK=256
                    )
                    out = blended
                else:
                    out = td.blend(out, y_k, k)
            if td is not None:
                z_new = z_kept
                if chain and exact_extra is not None:
                    exact_extra = td.keep(exact_extra, old_exact_extra, k)
            if do_drop:
                zm_valid = True
            if _TRACE:
                _TRACE[-1].append(
                    {
                        "k": k,
                        "p": gates_w.clone(),
                        "z": z_new.clone(),
                        "a": None if a_new is None else a_new.clone(),
                        "b": None if b_gate is None else b_gate.clone(),
                        "carried": zm.clone() if (do_drop and zm is not None) else None,
                    }
                )
            if a_new is not None and not (adaptive and inp):
                a_prev = a_new
            z_prev = z_new
            z = z_new
            prev = True
            if inp and frac and k == low:
                y_low = _input_output(y0, z, inp_gate, p, lead, dt)
                if exact_output is not None:
                    exact_low = exact_output
            if frac and k == low:
                snap = (None, zm.clone() if zm is not None else None, z.clone(), a_prev, b_gate)
        if td is not None:
            if carry_out:
                change = z.reshape(*lead, C)
                ends = compute.carry_transfer(change, c0.reshape(*lead, C), s.per_expert)
                if carry is None:
                    compute._carry_end(target, [ends], [change], start, depth, None)
                else:
                    carry[target] = {"carried": [ends], "change": [change], "depth": depth}
            if coefficient_blend:
                return y0 + torch.matmul(z_blend.view(*lead, C).to(dt), p.up_dt)
            return out
        if carry_out:
            change = z.reshape(*lead, C)
            if frac:
                lo_z = snap[2].reshape(*lead, C) if snap[2] is not None else None
                change = compute._blend([lo_z], [change], frac)[0]
                ends = compute.carry_transfer(change, c0.reshape(*lead, C), s.per_expert)
            else:
                ends = (
                    compute.carry_transfer(change, c0.reshape(*lead, C), s.per_expert)
                    if state
                    else zm.reshape(*lead, C)
                )
            if carry is None:
                gates = (
                    compute._gate_ends([s], [hist if adaptive else tgate.hist])
                    if adaptive or tgate is not None
                    else None
                )
                compute._carry_end(target, [ends], [change], start, depth, gates)
            else:
                carry[target] = {"carried": [ends], "change": [change], "depth": depth}
                if _drop_last:
                    carry[target]["dropped"] = zm.reshape(*lead, C)
        if jump and not inp and not adaptive and tgate is None and not sep:
            za = z if reg is None else z * reg[:, None]
            return y0 + torch.matmul(za.view(*lead, C).to(dt), p.up_dt)
        if inp:
            if adaptive and not frac and result is not None:
                return result if exact_output is None else result + exact_output
            y = _input_output(y0, z, inp_gate, p, lead, dt)
            if exact_output is not None:
                y = y + exact_output
                if exact_low is not None:
                    y_low = y_low + exact_low
            if state and depth < 1:
                return y0 + depth * (y - y0)
            return y if not frac else y_low + frac * (y - y_low)
        if adaptive and not frac and result is not None:
            return result
        if (adaptive or tgate is not None) and not frac:
            if a_prev is None:
                return y0 + torch.matmul(z.view(*lead, C).to(dt), p.up_dt)
            za = (a_prev.view(*lead, 1) * z.view(*lead, C)).to(dt)
            return y0 + _channel(torch.matmul(za, p.up_dt), b_gate, lead, OUT, dt)
        if za_out is not None:
            za = za_out.to(dt)
            delta = torch.matmul(za.view(*lead, C), p.up_dt)
            if sep:
                if (
                    y0.is_contiguous()
                    and delta.is_contiguous()
                    and not (_interpret() and dt != torch.float32)
                ):
                    result = torch.empty_like(y0)
                    _channel_output_kernel[(triton.cdiv(N * OUT, 256),)](
                        y0,
                        delta,
                        b_gate,
                        result,
                        T,
                        OUT,
                        N * OUT,
                        BLOCK=256,
                        enable_fp_fusion=False,
                    )
                    return result
                delta = _channel(delta, b_gate, lead, OUT, dt)
            elif (
                y0.is_contiguous()
                and delta.is_contiguous()
                and not (_interpret() and dt != torch.float32)
            ):
                result = torch.empty_like(y0)
                _add_output_kernel[(triton.cdiv(N * OUT, 256),)](
                    y0, delta, result, N * OUT, BLOCK=256
                )
                return result
            return y0 + delta
        y_hi = _gated_output(y0, z, a_prev, b_gate, p, lead, dt)
        y_lo = y0 if low == 0 else _gated_output(y0, snap[2], snap[3], snap[4], p, lead, dt)
        return (y_lo + frac * (y_hi - y_lo)).to(dt)


def _torch_gate(s: compute.LayerSet, p, start, z_start, region, index=0):
    adaptive = compute.uses_adaptive(s)
    learned = compute.learned_kind(s) == "separable"
    forced = s.form == "state" or os.environ.get("KILORA_FUSED_GATE", "kernel").strip() == "torch"
    if not (adaptive or (learned and forced)):
        return None
    hist = compute.gate_history(start, 0, z_start, region)

    def gate(z, y0_stats, gate_w, target, k, last, active=None):
        return compute.split_gate(
            s,
            z,
            hist,
            *y0_stats,
            gate_w,
            target,
            index,
            k,
            region,
            last,
            active=active,
            uniforms=uniforms,
        )

    gate.hist = hist
    gate.start = hist.get("q") if adaptive else None
    return gate


def _elementwise_ok(*tensors):
    return (
        HAVE_TRITON
        and enabled()
        and not compute._EAGER
        and not torch.is_grad_enabled()
        and all(
            device_ok(t)
            and t.is_contiguous()
            and t.numel() > 0
            and t.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and t.device == tensors[0].device
            for t in tensors
        )
    )


def token_gate(z, gate, dtype):
    if (
        z.ndim > 0
        and z.dtype == torch.float32
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and gate.shape == (*z.shape[:-1], 1)
        and _elementwise_ok(z, gate)
    ):
        out = torch.empty(z.shape, dtype=dtype, device=z.device)
        _elementwise_kernel[(triton.cdiv(z.numel(), 256),)](
            z,
            z,
            gate,
            out,
            z.numel(),
            z.shape[-1],
            1,
            MODE=0,
            SCALAR=False,
            VALUE=0.0,
            BLOCK=256,
            INTERPRET=_interpret(),
            enable_fp_fusion=False,
        )
        return out
    return (gate.float() * z).to(dtype)


def add_cast(y, delta):
    if y.shape == delta.shape and _elementwise_ok(y, delta):
        out = torch.empty_like(y)
        _elementwise_kernel[(triton.cdiv(y.numel(), 256),)](
            delta,
            y,
            y,
            out,
            y.numel(),
            1,
            1,
            MODE=1,
            SCALAR=False,
            VALUE=0.0,
            BLOCK=256,
            INTERPRET=_interpret(),
            enable_fp_fusion=False,
        )
        return out
    return y + delta.to(y.dtype)


def channel_change(delta, gate, y=None):
    if delta.ndim == 0 or delta.shape[-1] == 0:
        change = delta * gate.to(delta.dtype)
        return change if y is None else y + change
    width = delta.shape[-1]
    try:
        shape = torch.broadcast_shapes(delta.shape, gate.shape)
    except RuntimeError:
        shape = None
    rows = delta.numel() // width
    grows = gate.numel() // width if gate.numel() % width == 0 else 0
    padded = (1,) * (delta.ndim - gate.ndim) + tuple(gate.shape)
    per_token = padded[:-1] == tuple(delta.shape[:-1])
    per_sample = padded[:-1] == (delta.shape[0],) + (1,) * (delta.ndim - 2)
    global_gate = all(size == 1 for size in padded[:-1])
    valid = (
        shape == delta.shape
        and padded[-1:] == (width,)
        and (per_token or per_sample or global_gate)
    )
    stride = width if grows == rows else delta.numel() // max(grows, 1)
    tensors = (delta, gate) if y is None else (delta, gate, y)
    if (
        valid
        and (y is None or y.shape == delta.shape and y.dtype == delta.dtype)
        and _elementwise_ok(*tensors)
    ):
        out = torch.empty_like(delta)
        _elementwise_kernel[(triton.cdiv(delta.numel(), 256),)](
            delta,
            delta if y is None else y,
            gate,
            out,
            delta.numel(),
            width,
            stride,
            MODE=3 if y is None else 2,
            SCALAR=False,
            VALUE=0.0,
            BLOCK=256,
            INTERPRET=_interpret(),
            enable_fp_fusion=False,
        )
        return out
    change = delta * gate.to(delta.dtype)
    return change if y is None else y + change


def scaled_difference(y, changed, mult, *, cast_mult=False):
    scalar = not isinstance(mult, torch.Tensor)
    valid = (
        scalar
        and isinstance(mult, float)
        and math.isfinite(mult)
        and abs(mult) <= 3.4028234663852886e38
    )
    stride = 1
    if not scalar:
        prefix = tuple(mult.shape)
        while prefix and prefix[-1] == 1:
            prefix = prefix[:-1]
        valid = (
            (cast_mult or mult.dtype == y.dtype)
            and mult.numel() > 0
            and prefix == tuple(y.shape[: len(prefix)])
            and y.numel() % mult.numel() == 0
        )
        stride = y.numel() // mult.numel() if valid else 1
    tensors = (y, changed) if scalar else (y, changed, mult)
    if (
        valid
        and y.shape == changed.shape
        and y.dtype == changed.dtype
        and _elementwise_ok(*tensors)
    ):
        out = torch.empty_like(y)
        _elementwise_kernel[(triton.cdiv(y.numel(), 256),)](
            changed,
            y,
            y if scalar else mult,
            out,
            y.numel(),
            1,
            stride,
            MODE=4,
            SCALAR=scalar,
            VALUE=float(mult) if scalar else 0.0,
            BLOCK=256,
            INTERPRET=_interpret(),
            enable_fp_fusion=False,
        )
        return out
    if cast_mult:
        mult = mult.to(device=y.device, dtype=y.dtype)
    return y + mult * (changed - y)


def _input_output(y0, z, gate, p, lead, dt):
    zz = z.view(*lead, p.C)
    if gate is None:
        delta = torch.matmul(zz.to(dt), p.up_dt)
        if (
            y0.is_contiguous()
            and delta.is_contiguous()
            and not (_interpret() and dt != torch.float32)
        ):
            result = torch.empty_like(y0)
            _add_output_kernel[(triton.cdiv(y0.numel(), 256),)](
                y0, delta, result, y0.numel(), BLOCK=256
            )
            return result
        return y0 + delta
    a_t, b_t = gate
    za = token_gate(zz, a_t.reshape(*lead, 1), dt)
    y = torch.matmul(za, p.up_dt)
    if b_t.numel() == math.prod(lead) * p.OUT:
        return channel_change(y, b_t.reshape(*lead, p.OUT), y0)
    return channel_change(y, b_t.reshape(*lead[:-1], 1, p.OUT), y0)


def _hold_now(zm, hold0, c0, s):
    if hold0:
        return c0.float() * compute._scale(s, torch.float32)
    return zm.clone()


def _channel(y, b, lead, out, dt):
    return y * b.reshape(*lead[:-1], 1, out).to(dt)


def _gated_output(y0, z, a, b, p, lead, dt):
    za = (z * a[:, None]).to(dt).reshape(*lead, p.C)
    y = torch.matmul(za, p.up_dt)
    return y0 + _channel(y, b, lead, p.OUT, dt)


_NARROW_RANK = 64


def _int_key(value: int) -> tuple[bool, bool]:
    return value == 1, value % 16 == 0


def shape_key(s: compute.LayerSet, depth: float = 1, tokens: int = 4096) -> tuple:
    kout = s.keys_out if s.keys_out is not None else s.keys_in
    cols, width = s.up.shape
    experts, rows = s.keys_in.shape[:2]
    rd = s.router.shape[0]
    output_rows = kout.shape[1]
    if s.form == "state":
        rd = width
        rows = output_rows = min(compute.derived_rows(), s.per_expert, width)
    tiled = _tiled_import(s)
    return (
        min(_p2(cols), _COLUMN_BLOCK) if tiled and not adaptive_kernel() else _p2(cols),
        _p2(rd),
        _p2(experts),
        _p2(experts * rows),
        _p2(experts * output_rows),
        _int_key(cols),
        _int_key(s.per_expert),
        s.per_expert if s.per_expert <= _NARROW_RANK else "wide",
        _int_key(experts),
        _int_key(rd),
        _int_key(rows),
        _int_key(output_rows),
        _int_key(width),
        _int_key(tokens),
        routing_mode(s),
        s.carry,
        s.aggregation,
        compute.learned_kind(s),
        compute.uses_adaptive(s),
        isinstance(s.tau, torch.Tensor),
        isinstance(s.scale, torch.Tensor),
        compute._depth_parts(depth),
        s.form,
        s.iteration,
        tiled,
        s.expert_ids is not None,
        s.expert_mask is not None,
        s.region is not None,
        s.gate_coherence,
    )


def _matching_size(n: int) -> int:
    if n == 1:
        return 1
    return 64 if n % 16 == 0 else 65


def _synthetic(t: compute.LayerSet, dev, gen, width: int | None = None) -> compute.LayerSet:
    cols, rd = t.up.shape[0], t.router.shape[0]
    if width is None:
        width = t.up.shape[1] if t.form == "state" else _matching_size(t.up.shape[1])
    n_in = width if t.form == "state" else 2 * width
    experts, rows = t.keys_in.shape[0], t.keys_in.shape[1]
    kout = t.keys_out if t.keys_out is not None else t.keys_in
    sep = t.output_gate == "separable" and t.gate is not None
    gate = compute.new_gate("separable", experts, width, gen, cols) if sep else None
    router = torch.randn(rd, width, generator=gen).to(dev)
    keys_in = torch.randn(experts, rows, rd, generator=gen).to(dev)
    keys_out = torch.randn(experts, kout.shape[1], rd, generator=gen).to(dev)
    down = torch.randn(cols, n_in, generator=gen).to(dev) / 8
    up = torch.randn(cols, width, generator=gen).to(dev) / 8
    return compute.LayerSet(
        router=router,
        keys_in=keys_in,
        keys_out=keys_out,
        per_expert=t.per_expert,
        down=down,
        up=up,
        routing_down=None if t.routing_down is None else down.clone(),
        routing_up=None if t.routing_up is None else up.clone(),
        form=t.form,
        import_score=t.import_score,
        expert_ids=None if t.expert_ids is None else t.expert_ids.to(dev),
        expert_mask=None if t.expert_mask is None else torch.ones(1, 1, cols, device=dev),
        iteration=t.iteration if t.form == "input" else "accumulate",
        gate_mode=t.gate_mode,
        gate_kappa=t.gate_kappa,
        gate_coherence=t.gate_coherence,
        region=None if t.region is None else torch.ones(2, 32, device=dev),
        tau=torch.ones((), device=dev) if isinstance(t.tau, torch.Tensor) else 1.0,
        scale=torch.ones((), device=dev) if isinstance(t.scale, torch.Tensor) else 1.0,
        aggregation=t.aggregation,
        routing_scope=t.routing_scope,
        routing_select=t.routing_select,
        carry=t.carry,
        output_gate="separable" if sep else "none",
        gate=None if gate is None else {k: v.to(dev) for k, v in gate.items()},
    )


def _default_templates() -> list:
    out = []
    for per in (32, 16, 64):
        for scope in ("token", "tensor"):
            t = dataclasses.replace(
                _default_template(per=per),
                form="input",
                output_gate="none",
                gate=None,
                gate_mode="adaptive",
                routing_scope=scope,
            )
            out += [(t, 1), (t, 3)]
    out.append((dataclasses.replace(out[0][0], iteration="path"), 3))
    out.append((_default_template(), 1))
    return out


def _default_template(experts=4, per=16, rd=32, rows=4) -> compute.LayerSet:
    gen = torch.Generator().manual_seed(0)
    cols, width = experts * per, 8
    return compute.LayerSet(
        router=torch.zeros(rd, width),
        keys_in=torch.zeros(experts, rows, rd),
        keys_out=torch.zeros(experts, rows, rd),
        per_expert=per,
        down=torch.zeros(cols, width),
        up=torch.zeros(cols, width),
        output_gate="separable",
        gate=compute.new_gate("separable", experts, width, gen, cols),
    )


_WARMED: set = set()
_WARMED_LOCK = threading.Lock()
_WARM_STATS = [0, 0.0]
_CACHE_LOCK = threading.Lock()
_CACHE_LOGGED = False
_WARM_THREADS: list[threading.Thread] = []
_WARM_THREADS_LOCK = threading.Lock()
_WARM_PENDING: set = set()
_WARM_PENDING_LOCK = threading.Lock()
_COMPILE_WORKERS = max(1, int(os.environ.get("KILORA_COMPILE_WORKERS", "1")))
_COMPILE_LIMIT = threading.BoundedSemaphore(_COMPILE_WORKERS)


def warm_up(device=None, templates=None, dtypes=None) -> bool:
    try:
        if not (enabled() and HAVE_TRITON):
            return False
        if device is None:
            if not torch.cuda.is_available():
                return False
            device = "cuda"
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return False
        if dev.type == "cpu" and not _interpret():
            return False
        if templates is None:
            templates = _default_templates()
        if dtypes is None:
            dtypes = (torch.float32,) if dev.type == "cpu" else (torch.bfloat16, torch.float16)
        _configure_cache()
        started = time.perf_counter()
        jobs = []
        for dt in dtypes:
            for template, depth in templates:
                key = (str(dev), dt, shape_key(template, depth))
                with _WARMED_LOCK:
                    if key in _WARMED:
                        continue
                    _WARMED.add(key)
                jobs.append((dt, template, depth, key))

        def compile_shape(job):
            dt, template, depth, key = job
            gen = torch.Generator().manual_seed(0)
            try:
                s = _synthetic(template, dev, gen)
                width = s.up.shape[1]
                y0 = torch.randn(2, 32, width, generator=gen).to(dev, dt)
                rd = s.router.shape[0] if s.form == "input" else None
                tiled = _tiled_import(s)
                if not fits(y0, s.down.shape[0], rd, tiled=tiled):
                    mark_too_big(y0, [s])
                    return 0
                cache: dict = {}
                with torch.no_grad():
                    if s.form == "input":
                        x = torch.randn(2, 32, s.down.shape[1], generator=gen).to(dev, dt)
                        run_fused(
                            y0,
                            [s],
                            depth,
                            target="warm-up",
                            x=x,
                            name="warm-up#input0",
                            regions=[s.region],
                        )
                    else:
                        for _ in range(2):
                            run_fused(
                                y0, [s], depth, target="warm-up", carry=cache, regions=[s.region]
                            )
                return 1
            except Exception as err:
                with _WARMED_LOCK:
                    _WARMED.discard(key)
                if not resource_error(err):
                    raise
                mark_too_big(y0, [s])
                return 0

        workers = 1 if dev.type == "cpu" else min(len(jobs) or 1, _COMPILE_WORKERS)

        def bounded_compile(job):
            with _COMPILE_LIMIT:
                return compile_shape(job)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            compiled = sum(pool.map(bounded_compile, jobs))

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        if compiled:
            with _WARMED_LOCK:
                _WARM_STATS[0] += compiled
                _WARM_STATS[1] += time.perf_counter() - started
            log.info(
                "KiLoRA fused kernels ready: %d shapes in %.1f s",
                compiled,
                time.perf_counter() - started,
            )
        return True
    except Exception as exc:
        log.info("KiLoRA fused kernel warm-up skipped: %s", exc)
        return False


def _configure_cache() -> str:
    global _CACHE_LOGGED
    with _CACHE_LOCK:
        path = os.environ.get("TRITON_CACHE_DIR")
        if not path:
            try:
                import folder_paths

                path = os.path.join(folder_paths.get_user_directory(), "kilora", "triton")
                os.environ["TRITON_CACHE_DIR"] = path
            except (ImportError, AttributeError):
                path = os.path.join(os.path.expanduser("~"), ".triton", "cache")
        if not _CACHE_LOGGED:
            log.info("KiLoRA Triton cache: %s", path)
            _CACHE_LOGGED = True
        return path


def cache_dir() -> str:
    return _configure_cache()


@contextlib.contextmanager
def sample_compile_report():
    try:
        from triton import knobs

        compilation = knobs.compilation
        previous = compilation.listener
    except (ImportError, AttributeError):
        compilation = None
        previous = None
    count = [0]

    def listener(*args, **kwargs):
        if not kwargs.get("cache_hit", False):
            count[0] += 1
        if previous is not None:
            previous(*args, **kwargs)

    if compilation is not None:
        compilation.listener = listener
    try:
        yield
    finally:
        if compilation is not None and compilation.listener is listener:
            compilation.listener = previous
        with _WARMED_LOCK:
            shapes, seconds = _WARM_STATS
            _WARM_STATS[:] = [0, 0.0]
        value = str(count[0]) if compilation is not None else "unavailable"
        log.info(
            "KiLoRA first model call: %s kernels compiled; %d shapes warmed in %.1f s",
            value,
            shapes,
            seconds,
        )


def warm_up_sets(sets_by_depth, device=None, dtypes=None) -> threading.Thread | None:
    try:
        if not enabled() or not HAVE_TRITON:
            return None
        if device is None and not torch.cuda.is_available():
            return None
        dev = torch.device(device or _comfy_device() or "cuda")
        if dtypes is None:
            dtypes = (torch.float32,) if dev.type == "cpu" else (torch.bfloat16, torch.float16)
        else:
            dtypes = tuple(dtypes)
        seen = {}
        for sets, depth in sets_by_depth:
            lora = [s for s in sets if s.down is not None and s.form in ("state", "input")]
            if len(lora) != 1 or lora[0].iteration != "accumulate":
                continue
            seen.setdefault(shape_key(lora[0], depth), (lora[0], depth))
        if not seen:
            return None
        dev = dev or _comfy_device() or torch.device("cuda")
        dev = torch.device(dev)
        types = dtypes
        if types is None:
            types = (torch.float32,) if dev.type == "cpu" else (torch.bfloat16, torch.float16)
        with _WARM_PENDING_LOCK, _WARMED_LOCK:
            pending = {
                (str(dev), dt, shape)
                for shape in seen
                for dt in types
                if (str(dev), dt, shape) not in _WARMED
                and (str(dev), dt, shape) not in _WARM_PENDING
            }
            if not pending:
                return None
            _WARM_PENDING.update(pending)
        templates = [
            value
            for shape, value in seen.items()
            if any((str(dev), dt, shape) in pending for dt in types)
        ]

        def work():
            try:
                warm_up(dev, templates, types)
            finally:
                with _WARM_PENDING_LOCK:
                    _WARM_PENDING.difference_update(pending)

        try:
            return _spawn(work)
        except Exception:
            with _WARM_PENDING_LOCK:
                _WARM_PENDING.difference_update(pending)
            raise
    except Exception as exc:
        log.info("KiLoRA fused kernel warm-up not started: %s", exc)
        return None


def _comfy_device():
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return None


def _spawn(work) -> threading.Thread:
    thread = threading.Thread(target=work, name="kilora-fused-warm-up", daemon=True)
    with _WARM_THREADS_LOCK:
        _WARM_THREADS[:] = [item for item in _WARM_THREADS if item.is_alive()]
        _WARM_THREADS.append(thread)
        thread.start()
    atexit.register(thread.join, 120)
    return thread


def wait_for_warm_up() -> None:
    with _WARM_THREADS_LOCK:
        threads = tuple(_WARM_THREADS)
    for thread in threads:
        if thread.is_alive():
            thread.join()


_WARM: list = []


def start_warm_up() -> threading.Thread | None:
    try:
        if _WARM or not enabled() or not HAVE_TRITON or not torch.cuda.is_available():
            return None

        thread = _spawn(lambda: warm_up(_comfy_device()))
        _WARM.append(thread)
        return thread
    except Exception as exc:
        log.info("KiLoRA fused kernel warm-up not started: %s", exc)
        return None
