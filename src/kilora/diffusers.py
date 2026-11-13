# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from torch import nn

from .adapter import KiLoRA, attach, compose
from .config import KiLoRAConfig
from .sets import AdapterSet


def denoiser_of(pipe: Any) -> nn.Module:
    for attr in ("transformer", "unet"):
        module = getattr(pipe, attr, None)
        if isinstance(module, nn.Module):
            return module
    raise AttributeError("the pipeline has no 'transformer' or 'unet' module")


def attach_to_pipeline(pipe: Any, config: KiLoRAConfig, *, name: str = "main") -> KiLoRA:
    return attach(denoiser_of(pipe), config, name=name)


def compose_into_pipeline(
    pipe: Any,
    sources: Sequence[str | os.PathLike[str] | AdapterSet],
    **kwargs: Any,
) -> KiLoRA:
    return compose(denoiser_of(pipe), sources, **kwargs)
