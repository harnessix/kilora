# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


TOKEN_BLOCK = 64


if triton is not None:

    @triton.jit
    def _stats_kernel(
        Y,
        NORM,
        PARTIAL,
        T: tl.constexpr,
        W: tl.constexpr,
        BLOCKS: tl.constexpr,
        TOTAL: tl.constexpr,
        DELTA: tl.constexpr,
        WIDTH: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        while block < TOTAL:
            sample = block // BLOCKS
            token_block = block % BLOCKS
            rows = tl.arange(0, 4)
            cols = tl.arange(0, WIDTH)
            channel_sum = tl.full((WIDTH,), 0, tl.float32)
            for group in range(TOKEN_BLOCK // 4):
                t = token_block * TOKEN_BLOCK + group * 4 + rows
                values = tl.load(
                    Y + sample * T * W + t[:, None] * W + cols[None, :],
                    (t[:, None] < T) & (cols[None, :] < W),
                    0,
                ).to(tl.float32)
                square = values * values
                norm = tl.sqrt_rn(tl.sum(square, axis=1)) + DELTA
                tl.store(NORM + sample * T + t, norm, t < T)
                channel_sum += tl.sum(square, axis=0)
            tl.store(PARTIAL + block * W + cols, channel_sum, cols < W)
            block += tl.num_programs(0)

    @triton.jit
    def _finish_kernel(
        PARTIAL,
        RMS,
        T: tl.constexpr,
        W: tl.constexpr,
        BLOCKS: tl.constexpr,
        DELTA: tl.constexpr,
        COL_BLOCK: tl.constexpr,
    ):
        sample = tl.program_id(0)
        cols = tl.program_id(1) * COL_BLOCK + tl.arange(0, COL_BLOCK)
        channel_sum = tl.full((COL_BLOCK,), 0, tl.float32)
        for block in range(BLOCKS):
            channel_sum += tl.load(PARTIAL + (sample * BLOCKS + block) * W + cols, cols < W, 0)
        rms = tl.sqrt_rn(channel_sum / T) + DELTA
        tl.store(RMS + sample * W + cols, rms, cols < W)


def supported(y0: torch.Tensor) -> bool:
    return (
        triton is not None
        and y0.ndim >= 2
        and y0.is_contiguous()
        and y0.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and y0.shape[-2] > 0
        and 0 < y0.shape[-1] <= 8192
        and y0.numel() > 0
    )


def fused_stats(y0: torch.Tensor, delta: float, *, grid_size: int | None = None):
    if not supported(y0):
        raise ValueError("unsupported gate statistics input")
    tokens, width = y0.shape[-2:]
    batch = math.prod(y0.shape[:-2])
    norm = torch.empty(y0.shape[:-1], dtype=torch.float32, device=y0.device)
    rms = torch.empty((*y0.shape[:-2], width), dtype=torch.float32, device=y0.device)
    blocks = triton.cdiv(tokens, TOKEN_BLOCK)
    total = batch * blocks
    partial = torch.empty((total, width), dtype=torch.float32, device=y0.device)
    grid = total if grid_size is None else min(total, grid_size)
    if grid < 1:
        raise ValueError("grid_size must be positive")
    _stats_kernel[(grid,)](
        y0,
        norm,
        partial,
        tokens,
        width,
        blocks,
        total,
        delta,
        triton.next_power_of_2(width),
        TOKEN_BLOCK,
        enable_fp_fusion=False,
    )
    _finish_kernel[(batch, triton.cdiv(width, 256))](
        partial, rms, tokens, width, blocks, delta, 256, enable_fp_fusion=False
    )
    return norm, rms
