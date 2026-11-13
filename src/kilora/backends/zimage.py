# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from . import FlowMatchingBackend, TokenMap

PATCH = 2


def patchify(x: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // patch) * (w // patch), patch * patch * c)


class ZImageBackend(FlowMatchingBackend):
    family = "z-image"

    def __init__(self, denoiser: torch.nn.Module, dtype: torch.dtype, channels: int = 16):
        self.denoiser = denoiser
        self.dtype = dtype

    def forward(
        self, items: Sequence[Any], events: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.denoiser.parameters()).device
        x0 = torch.stack([ex.latent for ex in items]).to(device)
        noise = torch.randn(x0.shape, generator=generator).to(device)
        s = events.to(device).float().view(-1, 1, 1, 1)
        noisy = (1 - s) * x0 + s * noise
        inputs = [x.unsqueeze(1).to(self.dtype) for x in noisy.unbind(0)]
        captions = [ex.caption.to(device, self.dtype) for ex in items]
        out = self.denoiser(inputs, (1 - s.view(-1)).to(self.dtype), captions, return_dict=False)[0]
        pred = torch.stack([o.float() for o in out]).squeeze(2)
        return patchify(pred), patchify(x0 - noise)

    def token_map(self, items: Sequence[Any]) -> TokenMap:
        h, w = items[0].latent.shape[-2:]
        return TokenMap(count=(h // PATCH) * (w // PATCH))
