# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.recording import PathRecorder


def _held(rec):
    s = rec.sums["l"]
    out = sum(t.numel() * t.element_size() for t in (s.vk, s.cc, s.c2))
    return out + sum(t.numel() * t.element_size() for t in rec._proj.values())


def test_held_memory_matches_the_equation_and_stays_fixed():
    width, rank = 48, 6
    gen = torch.Generator().manual_seed(0)
    rec = PathRecorder(a={"l": torch.randn(rank, width, generator=gen)}, v0={"l": torch.eye(rank)})
    rec.observe("l", torch.randn(1, 7, width, generator=gen))
    first = _held(rec)
    assert first == 8 * (width * rank + rank * rank + 1) + 8 * rank * width
    for tokens in (3, 50, 400):
        rec.observe("l", torch.randn(2, tokens, width, generator=gen))
    assert _held(rec) == first
    assert rec.sums["l"].n == 7 + 2 * (3 + 50 + 400)
