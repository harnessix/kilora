# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

_STEP_STRIDE = 1_000_003


def draw_depth(seed: int, step: int, low: int, high: int) -> int:

    if low == high:
        return low
    gen = torch.Generator(device="cpu").manual_seed((int(seed) * _STEP_STRIDE + int(step)) % 2**63)
    return int(torch.randint(low, high + 1, (), generator=gen))
