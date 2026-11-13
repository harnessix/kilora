# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenMap:
    count: int = 0
    offset: int = 0
    from_end: bool = False
    end_pad: int = 0

    def observed_slice(self, tokens: int) -> slice:
        if self.count <= 0:
            return slice(0, tokens)
        start = tokens - self.end_pad - self.count if self.from_end else self.offset
        if start < 0 or start + self.count > tokens:
            raise ValueError(f"{self.count} observed tokens do not fit {tokens} tokens")
        return slice(start, start + self.count)


class Backend:
    family: str = ""

    def forward(
        self, items: Sequence[Any], events: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def token_map(self, items: Sequence[Any]) -> TokenMap:
        return TokenMap()

    def sample_events(self, count: int, generator: torch.Generator) -> torch.Tensor:
        return torch.rand(count, generator=generator) * 0.9 + 0.05

    def observe_events(self, count: int, rng: random.Random) -> list[float]:
        width = 0.9 / count
        levels = [0.05 + width * (i + rng.random()) for i in range(count)]
        rng.shuffle(levels)
        return levels


class FlowMatchingBackend(Backend):
    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction.float() - target.float()).pow(2).mean()
