# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch

from .ablation_settings import gate_coherence


def parse_gate_history(value: str) -> int | None:
    if value == "full":
        return None
    if isinstance(value, str) and value.startswith("head:"):
        number = value[5:]
        if number.isascii() and number.isdecimal() and int(number) >= 1:
            return int(number)
    raise ValueError("gate_history must be full or head:k with an integer k >= 1")


try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def gate_mix(x):
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        x = x * 0x846CA68B
        x = x ^ (x >> 16)
        return x

    @triton.jit
    def gate_draw(idx, base, q):
        h = gate_mix(gate_mix(idx) ^ base)
        u = (h >> 8).to(tl.float32) / 16777216.0
        return (u < q).to(tl.float32)

    @triton.jit
    def gate_halving(ratio, INTERPRET: tl.constexpr):
        if INTERPRET:
            return tl.exp2(-ratio)
        else:
            return libdevice.pow(tl.full(ratio.shape, 0.5, tl.float32), ratio)

    @triton.jit
    def gate_change(z, z_before, offs, mask, HAS_BEFORE: tl.constexpr):
        d = tl.load(z + offs, mask, 0).to(tl.float32)
        if HAS_BEFORE:
            d -= tl.load(z_before + offs, mask, 0).to(tl.float32)
        return d

    @triton.jit
    def gate_token_tile(
        z,
        z_before,
        m,
        new,
        gram,
        rows,
        rmask,
        C,
        history_rate: tl.constexpr,
        HAS_BEFORE: tl.constexpr,
        HAS_HISTORY: tl.constexpr,
        BT: tl.constexpr,
        BJ: tl.constexpr,
    ):
        sq = tl.zeros([BT], tl.float32)
        for j0 in range(0, C, BJ):
            j = j0 + tl.arange(0, BJ)
            jm = rmask[:, None] & (j[None, :] < C)
            prod = tl.zeros([BT, BJ], tl.float32)
            for i0 in range(0, C, BJ):
                i = i0 + tl.arange(0, BJ)
                im = rmask[:, None] & (i[None, :] < C)
                di = gate_change(z, z_before, rows[:, None] * C + i[None, :], im, HAS_BEFORE)
                gm = (i[:, None] < C) & (j[None, :] < C)
                g = tl.load(gram + i[:, None] * C + j[None, :], gm, 0).to(tl.float32)
                prod += tl.dot(di, g, input_precision="ieee")
            offs = rows[:, None] * C + j[None, :]
            dj = gate_change(z, z_before, offs, jm, HAS_BEFORE)
            sq += tl.sum(prod * dj, 1)
            h = tl.load(m + offs, jm, 0) if HAS_HISTORY else 0.0
            tl.store(new + offs, history_rate * h + (1 - history_rate) * dj, jm)
        return sq

    @triton.jit
    def gate_token_keep(
        sq,
        y0_norm,
        first_norm,
        kappa,
        HAS_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        INTERPRET: tl.constexpr,
    ):
        step = tl.sqrt_rn(tl.maximum(sq, 0.0))
        n = y0_norm
        if HAS_REFERENCE:
            n = tl.maximum(n, first_norm)
        q = gate_halving(tl.div_rn(step, kappa * n + 1e-6), INTERPRET)
        if FIRST_STEP:
            q = tl.full(q.shape, 1.0, tl.float32)
        return q

    @triton.jit
    def gate_moment(
        a,
        a_before,
        b,
        b_before,
        rows,
        rmask,
        i,
        j,
        C,
        A_BEFORE: tl.constexpr,
        B_BEFORE: tl.constexpr,
    ):
        am = rmask[:, None] & (i[None, :] < C)
        bm = rmask[:, None] & (j[None, :] < C)
        left = gate_change(a, a_before, rows[:, None] * C + i[None, :], am, A_BEFORE)
        right = gate_change(b, b_before, rows[:, None] * C + j[None, :], bm, B_BEFORE)
        return tl.dot(tl.trans(left), right, input_precision="ieee")

    @triton.jit
    def gate_channel_keep(
        acc,
        dot,
        dd,
        mm,
        y0_rms,
        first_rms,
        kappa,
        COHERENCE: tl.constexpr,
        HAS_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        INTERPRET: tl.constexpr,
    ):
        size = tl.sqrt_rn(tl.maximum(acc, 0.0))
        y = y0_rms
        if HAS_REFERENCE:
            y = tl.maximum(y, first_rms)
            size = tl.maximum(size - first_rms, 0.0)
        q = gate_halving(tl.div_rn(size, kappa * y + 1e-6), INTERPRET)
        if COHERENCE:
            norms = tl.sqrt_rn(tl.maximum(dd, 0.0)) * tl.sqrt_rn(tl.maximum(mm, 0.0))
            cos = tl.where(norms > 1e-6, tl.div_rn(dot, tl.maximum(norms, 1e-6)), 1.0)
            q *= tl.minimum(tl.maximum(0.5 + 0.5 * cos, 0.0), 1.0)
        if FIRST_STEP:
            q = tl.full(q.shape, 1.0, tl.float32)
        return q

    @triton.jit(do_not_specialize=["T", "C", "R", "NI", "S", "TPS", "kappa", "base"])
    def _gate_token_kernel(
        z,
        z_before,
        m,
        gram,
        y0_norm,
        first_norm,
        qa,
        new,
        moments,
        mask_a,
        T,
        C,
        R,
        NI,
        S,
        TPS,
        kappa,
        base,
        history_rate: tl.constexpr,
        HAS_BEFORE: tl.constexpr,
        HAS_HISTORY: tl.constexpr,
        COHERENCE: tl.constexpr,
        HAS_REFERENCE: tl.constexpr,
        FIRST_CHANGE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        MASKS: tl.constexpr,
        INTERPRET: tl.constexpr,
        BT: tl.constexpr,
        BJ: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = tl.program_id(1)
        if pid < R:
            t = pid * BT + tl.arange(0, BT)
            tm = t < T
            rows = b.to(tl.int64) * T + t
            sq = gate_token_tile(
                z,
                z_before,
                m,
                new,
                gram,
                rows,
                tm,
                C,
                history_rate,
                HAS_BEFORE,
                HAS_HISTORY,
                BT,
                BJ,
            )
            n = tl.load(y0_norm + rows, tm, 0).to(tl.float32)
            if FIRST_CHANGE:
                first = tl.sqrt_rn(tl.maximum(sq, 0.0))
                tl.store(first_norm + rows, first, tm)
            elif HAS_REFERENCE:
                first = tl.load(first_norm + rows, tm, 0).to(tl.float32)
            else:
                first = n
            q = gate_token_keep(
                sq, n, first, kappa, HAS_REFERENCE or FIRST_CHANGE, FIRST_STEP, INTERPRET
            )
            tl.store(qa + rows, q, tm)
            if MASKS:
                tl.store(mask_a + rows, gate_draw(rows.to(tl.uint32), base.to(tl.uint32), q), tm)
        else:
            w = pid - R
            split = w % S
            block = w // S
            i = (block // NI) * BJ + tl.arange(0, BJ)
            j = (block % NI) * BJ + tl.arange(0, BJ)
            t_start = split * TPS
            zz = tl.zeros([BJ, BJ], tl.float32)
            dm = tl.zeros([BJ, BJ], tl.float32)
            dd = tl.zeros([BJ, BJ], tl.float32)
            mm = tl.zeros([BJ, BJ], tl.float32)
            for t0 in range(0, TPS, BT):
                t = t_start + t0 + tl.arange(0, BT)
                tm = t < T
                rows = b.to(tl.int64) * T + t
                zz += gate_moment(z, z, z, z, rows, tm, i, j, C, False, False)
                if COHERENCE or FIRST_CHANGE:
                    dd += gate_moment(
                        z, z_before, z, z_before, rows, tm, i, j, C, HAS_BEFORE, HAS_BEFORE
                    )
                if COHERENCE:
                    dm += gate_moment(z, z_before, m, m, rows, tm, i, j, C, HAS_BEFORE, False)
                    mm += gate_moment(m, m, m, m, rows, tm, i, j, C, False, False)
            dst = moments + ((b.to(tl.int64) * S + split) * 4) * C * C
            offs = i[:, None] * C + j[None, :]
            om = (i[:, None] < C) & (j[None, :] < C)
            tl.store(dst + offs, zz, om)
            if COHERENCE:
                tl.store(dst + C * C + offs, dm, om)
                tl.store(dst + 3 * C * C + offs, mm, om)
            if COHERENCE or FIRST_CHANGE:
                tl.store(dst + 2 * C * C + offs, dd, om)

    @triton.jit
    def gate_moment_sum(moments, b, P, s, i, j, C, mask, BJ: tl.constexpr):
        acc = tl.zeros([BJ, BJ], tl.float32)
        for p in range(P):
            src = moments + (((b.to(tl.int64) * P + p) * 4 + s) * C) * C
            acc += tl.load(src + i[:, None] * C + j[None, :], mask, 0)
        return acc

    @triton.jit(do_not_specialize=["T", "C", "OUT", "P", "NA", "TOKENS", "kappa", "base"])
    def _gate_channel_kernel(
        moments,
        quads,
        up,
        active,
        y0_rms,
        first_rms,
        qb,
        mask_b,
        T,
        C,
        OUT,
        P,
        NA,
        TOKENS,
        kappa,
        base,
        PRE_QUADS: tl.constexpr,
        NQ: tl.constexpr,
        DD_INDEX: tl.constexpr,
        COHERENCE: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        HAS_REFERENCE: tl.constexpr,
        FIRST_CHANGE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        MASKS: tl.constexpr,
        INTERPRET: tl.constexpr,
        BT: tl.constexpr,
        BJ: tl.constexpr,
        BO: tl.constexpr,
    ):
        ob = tl.program_id(0)
        b = tl.program_id(1)
        o = ob * BO + tl.arange(0, BO)
        om = o < OUT
        if HAS_ACTIVE:
            count = tl.zeros([BT], tl.float32)
            for t0 in range(0, T, BT):
                t = t0 + tl.arange(0, BT)
                count += tl.load(active + b.to(tl.int64) * T + t, t < T, 0).to(tl.float32)
            tokens = tl.maximum(tl.sum(count, 0), 1.0)
        else:
            tokens = TOKENS
        if PRE_QUADS:
            src = quads + b.to(tl.int64) * NQ * OUT + o
            acc = tl.load(src, om, 0)
            dot = tl.zeros([BO], tl.float32)
            dd = tl.zeros([BO], tl.float32)
            mm = tl.zeros([BO], tl.float32)
            if COHERENCE:
                dot = tl.load(src + OUT, om, 0)
                mm = tl.load(src + 3 * OUT, om, 0)
            if COHERENCE or FIRST_CHANGE:
                dd = tl.load(src + DD_INDEX * OUT, om, 0)
        else:
            acc = tl.zeros([BO], tl.float32)
            dot = tl.zeros([BO], tl.float32)
            dd = tl.zeros([BO], tl.float32)
            mm = tl.zeros([BO], tl.float32)
            for i0 in range(0, C, BJ):
                i = i0 + tl.arange(0, BJ)
                im = i < C
                pz = tl.zeros([BJ, BO], tl.float32)
                pd = tl.zeros([BJ, BO], tl.float32)
                pm = tl.zeros([BJ, BO], tl.float32)
                pq = tl.zeros([BJ, BO], tl.float32)
                for j0 in range(0, C, BJ):
                    j = j0 + tl.arange(0, BJ)
                    jm = j < C
                    sm = im[:, None] & jm[None, :]
                    uj = tl.load(up + j[:, None] * OUT + o[None, :], jm[:, None] & om[None, :], 0)
                    uj = uj.to(tl.float32)
                    part = gate_moment_sum(moments, b, P, 0, i, j, C, sm, BJ)
                    pz += tl.dot(part, uj, input_precision="ieee")
                    if COHERENCE or FIRST_CHANGE:
                        part = gate_moment_sum(moments, b, P, 2, i, j, C, sm, BJ)
                        pd += tl.dot(part, uj, input_precision="ieee")
                    if COHERENCE:
                        part = gate_moment_sum(moments, b, P, 1, i, j, C, sm, BJ)
                        pm += tl.dot(part, uj, input_precision="ieee")
                        part = gate_moment_sum(moments, b, P, 3, i, j, C, sm, BJ)
                        pq += tl.dot(part, uj, input_precision="ieee")
                ui = tl.load(up + i[:, None] * OUT + o[None, :], im[:, None] & om[None, :], 0)
                ui = ui.to(tl.float32)
                acc += tl.sum(pz * ui, 0)
                if COHERENCE or FIRST_CHANGE:
                    dd += tl.sum(pd * ui, 0)
                if COHERENCE:
                    dot += tl.sum(pm * ui, 0)
                    mm += tl.sum(pq * ui, 0)
            acc = tl.div_rn(acc, tokens)
        cols = b.to(tl.int64) * OUT + o
        y = tl.load(y0_rms + cols, om, 0).to(tl.float32)
        if FIRST_CHANGE:
            first = tl.sqrt_rn(tl.maximum(tl.div_rn(dd, tokens), 0.0))
            tl.store(first_rms + cols, first, om)
        elif HAS_REFERENCE:
            first = tl.load(first_rms + cols, om, 0).to(tl.float32)
        else:
            first = y
        q = gate_channel_keep(
            acc,
            dot,
            dd,
            mm,
            y,
            first,
            kappa,
            COHERENCE,
            HAS_REFERENCE or FIRST_CHANGE,
            FIRST_STEP,
            INTERPRET,
        )
        tl.store(qb + cols, q, om)
        if MASKS:
            tl.store(mask_b + cols, gate_draw((NA + cols).to(tl.uint32), base.to(tl.uint32), q), om)


GATE_MODES = ("auto", "adaptive", "learned", "both", "none")
DEFAULT_KAPPA = 1.0
HISTORY_RATE = 0.5
_EPS = 1e-6
_MOMENT_PROGRAMS = 64
KERNEL_MOMENT_COLS = 64


def check_gate_mode(mode: str) -> str:
    if mode not in GATE_MODES:
        raise ValueError(f"unknown output gate mode {mode!r}, expected one of {GATE_MODES}")
    return mode


def check_kappa(kappa: float) -> float:
    kappa = float(kappa)
    if not kappa > 0:
        raise ValueError(f"the adaptive gate scale kappa must be > 0, got {kappa!r}")
    return kappa


def resolve_mode(mode: str, has_weights: bool) -> str:
    check_gate_mode(mode)
    if mode == "auto":
        return "both" if has_weights else "adaptive"
    return mode


def _diag_quad(s: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return ((s @ up) * up).sum(-2)


def token_cross(a, b):
    tokens = a.shape[-2]
    for chunks in (64, 32, 16, 8, 4, 2):
        if tokens % chunks == 0:
            left = a.reshape(*a.shape[:-2], chunks, tokens // chunks, a.shape[-1])
            right = b.reshape(*b.shape[:-2], chunks, tokens // chunks, b.shape[-1])
            return (left.transpose(-1, -2) @ right).sum(-3)
    return a.transpose(-1, -2) @ b


def _token_gram(v):
    return token_cross(v, v)


def base_reference() -> bool:
    return os.environ.get("KILORA_GATE_BASE_REFERENCE") == "1"


def change_stats(dz, up, gram, active=None):
    with torch.no_grad():
        dz, up = dz.detach().float(), up.detach().float()
        norm = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
        tokens = (
            dz.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
        )
        rms = _diag_quad(_token_gram(dz) / tokens, up).clamp(min=0).sqrt()
        return norm, rms


def fused_gate_enabled(z: torch.Tensor) -> bool:
    if triton is None or z.numel() == 0:
        return False
    if z.is_cuda:
        return True
    return os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("KILORA_FUSED_CPU") == "1"


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def gate_call(
    z: torch.Tensor,
    z_before: torch.Tensor | None,
    history: torch.Tensor | None,
    up: torch.Tensor,
    gram: torch.Tensor,
    y0_norm: torch.Tensor,
    y0_rms: torch.Tensor,
    kappa: float = DEFAULT_KAPPA,
    coherence: bool = True,
    active: torch.Tensor | None = None,
    first_norm: torch.Tensor | None = None,
    first_rms: torch.Tensor | None = None,
    first_step: bool = False,
    first_change: bool = False,
    base: int | None = None,
):

    coherence = gate_coherence(coherence)
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    tokens, cols = z.shape[-2], z.shape[-1]
    out = up.shape[-1]
    z = z.contiguous()
    batch = z.numel() // max(tokens * cols, 1)
    z_before = z_before.contiguous() if z_before is not None else None
    m = history.contiguous() if history is not None else None
    has_coherence = coherence and m is not None
    gram = gram.contiguous()
    up = up.contiguous()
    norm = y0_norm.contiguous()
    rms = y0_rms.contiguous()
    has_reference = first_norm is not None and not first_change
    if first_change:
        first_norm = torch.empty(z.shape[:-1], dtype=torch.float32, device=z.device)
        first_rms = torch.empty(rms.shape, dtype=torch.float32, device=z.device)
    ref_norm = first_norm.contiguous() if first_norm is not None else norm
    ref_rms = first_rms.contiguous() if first_rms is not None else rms
    if active is not None:
        if active.numel() != batch * tokens:
            active = active.expand(*z.shape[:-1], 1)
        active = active.contiguous()
    qa = torch.empty((*z.shape[:-1], 1), dtype=torch.float32, device=z.device)
    qb = torch.empty((*rms.shape[:-1], 1, out), dtype=torch.float32, device=z.device)
    new = torch.empty(z.shape, dtype=torch.float32, device=z.device)
    masks = base is not None
    mask_a = torch.empty_like(qa) if masks else qa
    mask_b = torch.empty_like(qb) if masks else qb
    bj = 16 if cols <= 16 else 32
    bt = 32
    rate_programs = triton.cdiv(tokens, bt)
    blocks = triton.cdiv(cols, bj)
    pre = cols > KERNEL_MOMENT_COLS
    nq, dd_index = 1, 1
    if pre:
        zf = z.float()
        dz = zf if z_before is None else zf - z_before.float()
        count = tokens if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
        parts = [_token_gram(zf) / count]
        if has_coherence:
            mf = m.float()
            parts += [token_cross(dz, mf), _token_gram(dz), _token_gram(mf)]
            dd_index = 2
        elif first_change:
            parts.append(_token_gram(dz))
        nq = len(parts)
        quads = _diag_quad(torch.stack(parts, dim=-3), up.float()).contiguous()
        moments, splits, per, moment_programs = quads, 1, bt, 0
    else:
        splits = max(1, min(rate_programs, triton.cdiv(_MOMENT_PROGRAMS, blocks * blocks)))
        per = triton.cdiv(rate_programs, splits) * bt
        splits = triton.cdiv(tokens, per)
        moments = torch.empty((batch, splits, 4, cols, cols), dtype=torch.float32, device=z.device)
        quads, moment_programs = moments, blocks * blocks * splits
    key = _i32(base) if masks else 0
    _gate_token_kernel[(rate_programs + moment_programs, batch)](
        z,
        z_before if z_before is not None else z,
        m if m is not None else z,
        gram,
        norm,
        ref_norm,
        qa,
        new,
        moments,
        mask_a,
        tokens,
        cols,
        rate_programs,
        blocks,
        splits,
        per,
        kappa,
        key,
        HISTORY_RATE,
        z_before is not None,
        m is not None,
        has_coherence,
        has_reference,
        first_change,
        first_step,
        masks,
        interpret,
        bt,
        bj,
        num_stages=1,
        enable_fp_fusion=False,
    )
    bo = 64
    _gate_channel_kernel[(triton.cdiv(out, bo), batch)](
        moments,
        quads,
        up,
        active if active is not None else z,
        rms,
        ref_rms,
        qb,
        mask_b,
        tokens,
        cols,
        out,
        splits,
        batch * tokens,
        float(tokens),
        kappa,
        key,
        pre,
        nq,
        dd_index,
        has_coherence,
        active is not None,
        has_reference,
        first_change,
        first_step,
        masks,
        interpret,
        bt,
        bj,
        bo,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if not first_change:
        first_norm = first_rms = None
    if not masks:
        mask_a = mask_b = None
    return qa, qb, new, first_norm, first_rms, mask_a, mask_b


def adaptive_keep(
    z: torch.Tensor,
    z_before: torch.Tensor | None,
    history: torch.Tensor | None,
    up: torch.Tensor,
    y0_norm: torch.Tensor,
    y0_rms: torch.Tensor,
    kappa: float = DEFAULT_KAPPA,
    coherence: bool = True,
    active: torch.Tensor | None = None,
    gram: torch.Tensor | None = None,
    first_norm: torch.Tensor | None = None,
    first_rms: torch.Tensor | None = None,
    first_step: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coherence = gate_coherence(coherence)
    if base_reference():
        first_norm = first_rms = None
        first_step = False
    elif first_step is None:
        first_step = z_before is None
    with torch.no_grad():
        if fused_gate_enabled(z):
            if gram is None:
                u = up.detach().float()
                gram = u @ u.t()
            qa, qb, new = gate_call(
                z,
                z_before,
                history,
                up,
                gram,
                y0_norm,
                y0_rms,
                kappa,
                coherence,
                active,
                first_norm,
                first_rms,
                first_step,
            )[:3]
            return qa, qb, new
        z = z.detach().float()
        up = up.detach().float()
        dz = z if z_before is None else z - z_before.detach().float()
        if gram is None:
            gram = up @ up.t()
        step = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
        token_ref = y0_norm if first_norm is None else torch.maximum(y0_norm, first_norm)
        channel_ref = y0_rms if first_rms is None else torch.maximum(y0_rms, first_rms)
        qa = torch.pow(0.5, step / (kappa * token_ref + _EPS))
        tokens = (
            z.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
        )
        if coherence and history is not None:
            m = history.float()
            moments = torch.stack(
                (_token_gram(z) / tokens, token_cross(dz, m), _token_gram(dz), _token_gram(m)),
                dim=-3,
            )
            acc, dot, nd, nm = _diag_quad(moments, up).unbind(-2)
            acc = acc.clamp(min=0).sqrt()
            size = acc if first_rms is None else (acc - first_rms).clamp(min=0)
            qb = torch.pow(0.5, size / (kappa * channel_ref + _EPS))
            nd = nd.clamp(min=0).sqrt()
            nm = nm.clamp(min=0).sqrt()
            norms = nd * nm
            cos = torch.where(norms > _EPS, dot / norms.clamp(min=_EPS), torch.ones_like(dot))
            qb = qb * (0.5 + 0.5 * cos).clamp(0, 1)
        else:
            acc = _diag_quad(_token_gram(z) / tokens, up).clamp(min=0).sqrt()
            size = acc if first_rms is None else (acc - first_rms).clamp(min=0)
            qb = torch.pow(0.5, size / (kappa * channel_ref + _EPS))
        if first_step:
            qa, qb = torch.ones_like(qa), torch.ones_like(qb)
        old = 0.0 if history is None else HISTORY_RATE * history.float()
        new = old + (1 - HISTORY_RATE) * dz
        return qa[..., None], qb[..., None, :], new
