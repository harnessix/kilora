# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from .network import KiLoRAMusubiNetwork, create_network, create_network_from_weights

FAMILY = "z-image"


def create_arch_network(
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    vae: Any,
    text_encoders: Any,
    unet: Any,
    neuron_dropout: float | None = None,
    **kwargs: Any,
) -> KiLoRAMusubiNetwork:
    return create_network(FAMILY, multiplier, network_dim, network_alpha, unet, **kwargs)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Any,
    text_encoders: Any = None,
    unet: Any = None,
    for_inference: bool = False,
    **kwargs: Any,
) -> KiLoRAMusubiNetwork:
    return create_network_from_weights(FAMILY, multiplier, weights_sd, unet, **kwargs)
