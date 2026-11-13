# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch

V0_LAYOUT = "nested-v1"
NESTED_MAX_RANK = 64
BASE_RANK = 4


def _base_v0(seed: int, index: int, columns: int) -> torch.Tensor:
    h = hashlib.sha256(f"{int(seed)}:{int(index)}".encode()).hexdigest()
    gen = torch.Generator().manual_seed(int(h[:15], 16))
    q, _ = torch.linalg.qr(torch.randn(columns, BASE_RANK, generator=gen, dtype=torch.float64))
    return q.t().to(torch.float32).contiguous()


def make_v0(seed: int, index: int, rank: int, columns: int) -> torch.Tensor:
    if not 1 <= rank <= NESTED_MAX_RANK:
        raise ValueError(f"live rank {rank} is outside 1 to {NESTED_MAX_RANK}")
    v0_4 = _base_v0(seed, index, columns)
    if rank <= BASE_RANK:
        return v0_4[:rank].clone()
    h = hashlib.sha256(f"nested:{int(seed)}:{int(index)}".encode()).hexdigest()
    gen = torch.Generator().manual_seed(int(h[:15], 16))
    extra = torch.randn(columns, NESTED_MAX_RANK - BASE_RANK, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.cat([v0_4.t().double(), extra], 1))
    out = q.t()[:rank].to(torch.float32).contiguous()
    out[:BASE_RANK] = v0_4
    return out


def default_v0(targets: Sequence[str], columns: Mapping[str, int], seed: int, rank: int = 4):
    return {t: make_v0(seed, i, rank, columns[t]) for i, t in enumerate(targets)}


def columns_of(down: torch.Tensor) -> int:
    return int(down.numel() // down.shape[-1])
