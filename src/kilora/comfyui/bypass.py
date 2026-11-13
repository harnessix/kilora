# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..family_spec import linear_shape

__all__ = ["INJECTION_KEY", "ForwardHook", "injection", "is_quantized", "linear_shape"]

INJECTION_KEY = "kilora"

_reloading = ContextVar("kilora_reloading", default=None)


@contextmanager
def same_patcher_reload(patcher: Any):
    token = _reloading.set(patcher)
    try:
        yield
    finally:
        _reloading.reset(token)


def retaining(patcher: Any) -> bool:
    model = getattr(patcher, "model", None)
    injections = getattr(patcher, "injections", {})
    return (
        _reloading.get() is patcher
        and all(key == INJECTION_KEY or not value for key, value in injections.items())
        and getattr(model, "model_loaded_weight_memory", 0) > 0
        and getattr(model, "device", None) == getattr(patcher, "load_device", None)
        and getattr(patcher, "load_device", None) is not None
    )


_LOW_BIT = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        torch.int8,
        torch.uint8,
    )
    if d is not None
)


def is_quantized(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None:
        return not isinstance(module, nn.Linear)
    data = weight.data if isinstance(weight, nn.Parameter) else weight
    if type(data) is not torch.Tensor:
        return True
    return data.dtype in _LOW_BIT


class ForwardHook:
    def __init__(
        self,
        path: str,
        after: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        full_after: Callable | None = None,
    ):
        self.path = path
        self.after = after
        self.full_after = full_after
        self.module: nn.Module | None = None
        self.original: Any = None
        self.had_forward = False
        self.force: Any = None

    def _forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        from ..bake import base_only_active

        if base_only_active():
            return self.original(x, *args, **kwargs)
        if self.full_after is not None:
            return self.full_after(x, self.original, *args, **kwargs)
        return self.after(x, self.original(x, *args, **kwargs))

    def inject(self, module: nn.Module) -> None:
        if self.module is not None:
            return
        self.module = module
        self.had_forward = "forward" in module.__dict__
        self.original = module.forward
        self.force = getattr(module, "comfy_force_forward", None)
        module.comfy_force_forward = True
        module.forward = self._forward

    def eject(self) -> None:
        module = self.module
        if module is None:
            return
        if self.had_forward:
            module.forward = self.original
        else:
            del module.forward
        if self.force is None:
            module.__dict__.pop("comfy_force_forward", None)
        else:
            module.comfy_force_forward = self.force
        self.module = self.original = None

    @property
    def injected(self) -> bool:
        return self.module is not None


@dataclass
class _Injection:
    inject: Callable[[Any], None]
    eject: Callable[[Any], None]


def injection(hooks: list[ForwardHook], on_unloaded: Callable[[], None] | None = None) -> Any:
    def inject_all(patcher: Any) -> None:
        model = getattr(patcher, "model", None)
        if on_unloaded is not None and getattr(model, "model_loaded_weight_memory", None) == 0:
            on_unloaded()
        for hook in hooks:
            module = _module(patcher, hook.path)
            if hook.injected and hook.module is not module:
                hook.eject()
            hook.inject(module)

    def eject_all(patcher: Any) -> None:
        if retaining(patcher):
            return
        for hook in hooks:
            hook.eject()

    try:
        from comfy.patcher_extension import PatcherInjection

        return PatcherInjection(inject=inject_all, eject=eject_all)
    except Exception:
        return _Injection(inject=inject_all, eject=eject_all)


def _module(patcher: Any, path: str) -> nn.Module:
    obj = patcher.model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj
