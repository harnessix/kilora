# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

HOST_API = (1,)
EXPORTS_API = 1


@dataclass(frozen=True)
class HostServices:
    api: int = 1
    host: str = "python"
    model_dirs: Callable[[str], list[str]] | None = None
    comfy: Any = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("kilora"))
    settings: Mapping[str, Any] = field(default_factory=dict)
    cache: str | None = None


@dataclass(frozen=True)
class ModelDescription:
    family: str
    target_modules: tuple[str, ...]
    full_target_modules: tuple[str, ...] = ()


@dataclass
class PackageExports:
    name: str
    version: str
    api: int = EXPORTS_API
    nodes: dict[str, type] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)
    web_directory: str | None = None
    editors: list[Any] = field(default_factory=list)
    trainers: list[Any] = field(default_factory=list)
    models: list[ModelDescription] = field(default_factory=list)


SERVICES = HostServices()


def services() -> HostServices:
    return SERVICES


def model_descriptions() -> list[ModelDescription]:
    from .presets import PRESETS

    return [
        ModelDescription(p.family, tuple(p.target_modules), tuple(p.full_target_modules))
        for p in PRESETS.values()
    ]


def _web_directory() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "comfyui", "web"), os.path.join(here, "..", "..", "web")):
        if os.path.isdir(path):
            return os.path.normpath(path)
    return None


def register(host_services: HostServices) -> PackageExports:
    global SERVICES
    if host_services.api not in HOST_API:
        raise RuntimeError(
            f"This KiLoRA runtime supports host contract {HOST_API}, "
            f"not {host_services.api}. Update the KiLoRA plugin."
        )
    SERVICES = host_services
    from ._version import __version__
    from .comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .editors import available_editors
    from .trainers import available_trainers

    return PackageExports(
        name="kilora",
        version=__version__,
        nodes=dict(NODE_CLASS_MAPPINGS),
        display_names=dict(NODE_DISPLAY_NAME_MAPPINGS),
        web_directory=_web_directory(),
        editors=list(available_editors().values()),
        trainers=list(available_trainers().values()),
        models=model_descriptions(),
    )


def selftest() -> None:
    from .adapter import attach
    from .config import KiLoRAConfig
    from .editors import available_editors
    from .testing import TinyDenoiser
    from .trainers import available_trainers

    model = TinyDenoiser(channels=4, width=16, num_layers=1, seed=0).eval()
    adapter = attach(
        model, KiLoRAConfig(target_modules=["layers.*.attention.to_out.0"], num_experts=2)
    )
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 4))
    adapter.remove()
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("the self-test output is not finite")
    from .comfyui import NODE_CLASS_MAPPINGS

    if not NODE_CLASS_MAPPINGS:
        raise RuntimeError("no ComfyUI nodes")
    available_editors()
    if "static" not in available_trainers("mod"):
        raise RuntimeError("the static trainer is missing")
