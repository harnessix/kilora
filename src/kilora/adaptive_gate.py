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

    @triton.jit(do_not_specialize=["N", "C", "kappa"])
    def _token_rates_kernel(
        dz,
        product,
        norm,
        history,
        first_norm,
        qa,
        new,
        N,
        C,
        kappa,
        history_rate: tl.constexpr,
        HAS_HISTORY: tl.constexpr,
        HAS_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        INTERPRET: tl.constexpr,
        BC: tl.constexpr,
    ):
        row = tl.program_id(0)
        c = tl.arange(0, BC)
        d = tl.load(dz + row * C + c, c < C, 0)
        p = tl.load(product + row * C + c, c < C, 0)
        n = tl.load(norm + row)
        if HAS_REFERENCE:
            n = tl.maximum(n, tl.load(first_norm + row))
        step = tl.sqrt_rn(tl.maximum(tl.sum(p * d, 0), 0.0))
        ratio = tl.div_rn(step, kappa * n + 1e-6)
        q = (
            tl.exp2(-ratio)
            if INTERPRET
            else libdevice.pow(tl.full(ratio.shape, 0.5, tl.float32), ratio)
        )
        if FIRST_STEP:
            q = tl.full(q.shape, 1.0, tl.float32)
        tl.store(qa + row, q)
        h = tl.load(history + row * C + c, c < C, 0) if HAS_HISTORY else 0.0
        tl.store(new + row * C + c, history_rate * h + (1 - history_rate) * d, c < C)

    @triton.jit(do_not_specialize=["OUT", "N", "kappa"])
    def _channel_rates_kernel(
        quads,
        rms,
        first_rms,
        qb,
        OUT,
        N,
        kappa,
        COHERENCE: tl.constexpr,
        HAS_REFERENCE: tl.constexpr,
        FIRST_STEP: tl.constexpr,
        INTERPRET: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch = i // OUT
        col = i % OUT
        nquad = 4 if COHERENCE else 1
        acc = tl.load(quads + batch * nquad * OUT + col, i < N, 0)
        y = tl.load(rms + i, i < N, 0)
        size = tl.sqrt_rn(tl.maximum(acc, 0.0))
        if HAS_REFERENCE:
            first = tl.load(first_rms + i, i < N, 0)
            y = tl.maximum(y, first)
            size = tl.maximum(size - first, 0.0)
        ratio = tl.div_rn(size, kappa * y + 1e-6)
        q = (
            tl.exp2(-ratio)
            if INTERPRET
            else libdevice.pow(tl.full(ratio.shape, 0.5, tl.float32), ratio)
        )
        if COHERENCE:
            dot = tl.load(quads + batch * 4 * OUT + OUT + col, i < N, 0)
            nd = tl.load(quads + batch * 4 * OUT + 2 * OUT + col, i < N, 0)
            nm = tl.load(quads + batch * 4 * OUT + 3 * OUT + col, i < N, 0)
            norms = tl.sqrt_rn(tl.maximum(nd, 0.0)) * tl.sqrt_rn(tl.maximum(nm, 0.0))
            cos = tl.where(norms > 1e-6, tl.div_rn(dot, tl.maximum(norms, 1e-6)), 1.0)
            q *= tl.minimum(tl.maximum(0.5 + 0.5 * cos, 0.0), 1.0)
        if FIRST_STEP:
            q = tl.full(q.shape, 1.0, tl.float32)
        tl.store(qb + i, q, i < N)


GATE_MODES = ("auto", "adaptive", "learned", "both", "none")
DEFAULT_KAPPA = 1.0
HISTORY_RATE = 0.5
_EPS = 1e-6


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


def base_reference() -> bool:
    return os.environ.get("KILORA_GATE_BASE_REFERENCE") == "1"


def change_stats(dz, up, gram, active=None):
    with torch.no_grad():
        dz, up = dz.detach().float(), up.detach().float()
        norm = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
        tokens = (
            dz.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
        )
        rms = _diag_quad(dz.transpose(-1, -2) @ dz / tokens, up).clamp(min=0).sqrt()
        return norm, rms


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
        z = z.detach().float()
        up = up.detach().float()
        dz = z if z_before is None else z - z_before.detach().float()
        if gram is None:
            gram = up @ up.t()
        interpret = os.environ.get("TRITON_INTERPRET") == "1"
        if triton is not None and (
            z.is_cuda or (interpret and os.environ.get("KILORA_FUSED_CPU") == "1")
        ):
            tokens = (
                z.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
            )
            moment = z.transpose(-1, -2) @ z / tokens
            has_coherence = coherence and history is not None
            if has_coherence:
                m = history.float()
                dzt, mt = dz.transpose(-1, -2), m.transpose(-1, -2)
                moments = torch.stack((moment, dzt @ m, dzt @ dz, mt @ m), dim=-3)
            else:
                moments = moment.unsqueeze(-3)
            quads = _diag_quad(moments, up).contiguous()
            product = (dz @ gram).contiguous()
            dz = dz.contiguous()
            norm = y0_norm.contiguous()
            rms = y0_rms.contiguous()
            ref_norm = first_norm.contiguous() if first_norm is not None else norm
            ref_rms = first_rms.contiguous() if first_rms is not None else rms
            h = history.float().contiguous() if history is not None else dz
            qa = torch.empty_like(norm)
            qb = torch.empty_like(rms)
            new = torch.empty_like(dz)
            n = norm.numel()
            cols = dz.shape[-1]
            _token_rates_kernel[(n,)](
                dz,
                product,
                norm,
                h,
                ref_norm,
                qa,
                new,
                n,
                cols,
                kappa,
                HISTORY_RATE,
                history is not None,
                first_norm is not None,
                first_step,
                interpret,
                triton.next_power_of_2(cols),
                enable_fp_fusion=False,
            )
            size = rms.numel()
            _channel_rates_kernel[(triton.cdiv(size, 128),)](
                quads,
                rms,
                ref_rms,
                qb,
                up.shape[-1],
                size,
                kappa,
                has_coherence,
                first_rms is not None,
                first_step,
                interpret,
                128,
                enable_fp_fusion=False,
            )
            return qa[..., None], qb[..., None, :], new
        step = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
        token_ref = y0_norm if first_norm is None else torch.maximum(y0_norm, first_norm)
        channel_ref = y0_rms if first_rms is None else torch.maximum(y0_rms, first_rms)
        qa = torch.pow(0.5, step / (kappa * token_ref + _EPS))
        tokens = (
            z.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)
        )
        if coherence and history is not None:
            m = history.float()
            dzt, mt = dz.transpose(-1, -2), m.transpose(-1, -2)
            moments = torch.stack(
                (z.transpose(-1, -2) @ z / tokens, dzt @ m, dzt @ dz, mt @ m), dim=-3
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
            acc = _diag_quad(z.transpose(-1, -2) @ z / tokens, up).clamp(min=0).sqrt()
            size = acc if first_rms is None else (acc - first_rms).clamp(min=0)
            qb = torch.pow(0.5, size / (kappa * channel_ref + _EPS))
        if first_step:
            qa, qb = torch.ones_like(qa), torch.ones_like(qb)
        old = 0.0 if history is None else HISTORY_RATE * history.float()
        new = old + (1 - HISTORY_RATE) * dz
        return qa[..., None], qb[..., None, :], new
