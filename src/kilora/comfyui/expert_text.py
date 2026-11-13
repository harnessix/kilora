# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
from collections import OrderedDict

import torch

log = logging.getLogger(__name__)
_SELECTION = contextvars.ContextVar("kilora_text_selection", default=None)
_PREFIX = "kilora_expert_text_"


@contextlib.contextmanager
def select_text(expert):
    token = _SELECTION.set((expert,))
    try:
        yield
    finally:
        _SELECTION.reset(token)


def selected_text():
    return _SELECTION.get()


def _identity(value):
    if not isinstance(value, torch.Tensor):
        return ("value", value)
    try:
        version = value._version
    except RuntimeError:
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return (
            "content",
            hashlib.sha256(raw).digest(),
            value.device,
            value.dtype,
            tuple(value.shape),
        )
    return (id(value), version, value.device, value.dtype, tuple(value.shape))


class ExpertTextCache:
    def __init__(self, experts, limit=16):
        self.experts = tuple(experts)
        self.limit = limit
        self.cache = OrderedDict()
        log.debug("Expert text cache enabled for %d experts, capacity %d", len(experts), limit)

    def prepare(self, forward, inputs, version):
        identities = tuple(_identity(value) for value in inputs)
        cacheable = all(value is not None for value in identities)
        key = (identities, version)
        if cacheable and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][1]
        with select_text(""):
            base = forward()
        values = {"base": base}
        for name in self.experts:
            with select_text(name):
                values["expert_" + name] = forward()
        if cacheable:
            self.cache[key] = (inputs, values)
            while len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        return values


_VALUES = contextvars.ContextVar("kilora_expert_conditionings", default=None)


@contextlib.contextmanager
def conditioning_values(values):
    token = _VALUES.set(values)
    try:
        yield
    finally:
        _VALUES.reset(token)


def current_conditionings():
    return _VALUES.get() or {}


def install(patcher, fixed, enabled, always):
    groups = {}
    for path, changes in fixed.items():
        if ".llm_adapter." not in path:
            continue
        for change in changes:
            name = getattr(change, "text_expert", None)
            if name:
                groups.setdefault(name, set()).add(path)
    if not groups:
        return {}
    if not enabled or not hasattr(patcher, "add_object_patch"):
        return groups
    model = getattr(patcher, "model", None)
    try:
        original = (
            patcher.get_model_object("extra_conds")
            if hasattr(patcher, "get_model_object")
            else getattr(model, "extra_conds", None)
        )
    except AttributeError:
        original = None
    if original is None:
        return groups
    original = getattr(original, "_kilora_original_extra_conds", original)
    cache = ExpertTextCache(groups)

    def extra_conds(**kwargs):
        shared = original(**kwargs)
        if not (always or kwargs.get("kilora_layer") is not None):
            return shared
        if kwargs.get("cross_attn") is None or kwargs.get("t5xxl_ids") is None:
            return shared
        inputs = tuple(kwargs.get(key) for key in ("cross_attn", "t5xxl_ids", "t5xxl_weights"))
        values = cache.prepare(
            lambda: original(**kwargs).get("c_crossattn"), inputs, str(kwargs.get("device"))
        )
        if any(value is None for value in values.values()):
            return shared
        return {
            **shared,
            "c_crossattn": values["base"],
            **{_PREFIX + name: value for name, value in values.items()},
        }

    extra_conds._kilora_original_extra_conds = original
    extra_conds._kilora_previous_extra_patch = getattr(patcher, "object_patches", {}).get(
        "extra_conds"
    )
    patcher.add_object_patch("extra_conds", extra_conds)
    return groups
