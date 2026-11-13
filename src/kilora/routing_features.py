# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingFeatures:
    token: torch.Tensor
    region: torch.Tensor
    prompt: torch.Tensor | None
    timestep_embedding: torch.Tensor | None
    prompt_projection_missing: bool = False


_enabled = contextvars.ContextVar("kilora_routing_features_enabled", default=False)
_values = contextvars.ContextVar("kilora_routing_feature_values", default=(None, None, None))
_feature_sets = contextvars.ContextVar("kilora_score_feature_sets", default=())
_features = contextvars.ContextVar("kilora_score_features", default=None)


def timestep_features(timestep: torch.Tensor, width: int = 256) -> torch.Tensor:
    t = torch.as_tensor(timestep).reshape(-1).float()
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(width // 2, device=t.device) / (width // 2)
    )
    phase = t[:, None] * frequencies[None, :]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


@contextlib.contextmanager
def routing_features(enabled: bool = True):
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def enabled() -> bool:
    return _enabled.get()


def prompt_projection(model, family):

    if family != "anima":
        return None

    def project(target, prompt, y):
        from .presets import anima_name

        name = target.split("@", 1)[0]
        mapped = anima_name(name)
        if mapped is not None:
            index = mapped.split(".")[1]
        elif name.startswith("blocks.") and len(name.split(".")) > 2:
            index = name.split(".")[1]
            if not index.isdigit():
                return None
        else:
            return None
        if model is None or not isinstance(prompt, torch.Tensor):
            return None
        module = None
        for path in (f"blocks.{index}.cross_attn.k_proj", f"transformer_blocks.{index}.attn2.to_k"):
            try:
                module = model.get_submodule(path)
                break
            except (AttributeError, KeyError):
                continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return None
        if weight.shape != (y.shape[-1], prompt.shape[-1]):
            return None
        pooled = prompt.mean(dim=1) if prompt.ndim == 3 else prompt
        if pooled.ndim != 2:
            return None
        bias = getattr(module, "bias", None)
        return torch.nn.functional.linear(
            pooled.to(device=y.device, dtype=y.dtype),
            weight.to(device=y.device, dtype=y.dtype),
            None if bias is None else bias.to(device=y.device, dtype=y.dtype),
        )

    return project


@contextlib.contextmanager
def call_features(
    prompt: torch.Tensor | None, timestep_embedding: torch.Tensor | None, prompt_projection=None
):

    token = _values.set((prompt, timestep_embedding, prompt_projection))
    try:
        yield
    finally:
        _values.reset(token)


def current_score_features() -> tuple[RoutingFeatures, ...] | None:
    return _features.get()


def features_for_set(layer_set):
    features = _features.get()
    if features is not None:
        for original, feature in zip(_feature_sets.get(), features, strict=True):
            if original is layer_set or original.keys_in is layer_set.keys_in:
                return feature
    return None


def _expand(value: torch.Tensor | None, x: torch.Tensor, *, pool: bool):
    if value is None:
        return None
    if pool and value.ndim == 3:
        value = value.mean(dim=1)
    if value.ndim != 2:
        raise ValueError("routing features must have a batch and feature dimension")
    if value.shape[0] not in (1, x.shape[0]):
        from .call_values import current_route_rows

        rows = current_route_rows()
        if rows is None or rows.numel() != x.shape[0]:
            raise ValueError("routing feature batch does not match the layer input")
        value = value.index_select(0, rows.to(value.device))
    value = value.to(device=x.device)
    return value.reshape(value.shape[0], *([1] * (x.ndim - 2)), value.shape[-1]).expand(
        *x.shape[:-1], value.shape[-1]
    )


@contextlib.contextmanager
def layer_features(x, y0, sets, regions, target=""):

    from .compute import _token_region, region_mean

    if y0.ndim == 2:
        y0 = y0.unsqueeze(1)
    prompt, timestep, projection = _values.get()
    from .region import current_layout, current_text_tokens

    layout, count = current_layout(target), current_text_tokens()
    from .presets import anima_name

    missing_projection = projection is None and (
        anima_name(target.split("@", 1)[0]) is not None or target.startswith("blocks.")
    )
    if layout is not None and layout.kind == "text1d":
        missing_projection = False
        prompt = y0.mean(dim=1)
    elif layout is not None and count and 0 < count <= y0.shape[-2]:
        if layout.text == "before":
            missing_projection = False
            prompt = y0[:, :count].mean(dim=1)
        elif layout.text == "end":
            missing_projection = False
            prompt = y0[:, -count:].mean(dim=1)
        elif projection is not None:
            prompt = projection(target, prompt, y0)
    elif projection is not None:
        prompt = projection(target, prompt, y0)
    prompt = _expand(prompt, y0, pool=True)
    timestep = _expand(timestep, y0, pool=False)
    if regions is not None and len(regions) != len(sets):
        raise ValueError("routing feature regions do not match the sets")
    features = []
    for i, layer_set in enumerate(sets):
        mask = _token_region(regions[i] if regions is not None else layer_set.region, y0)
        pooled = region_mean(y0, mask).expand_as(y0)
        features.append(RoutingFeatures(y0, pooled, prompt, timestep, missing_projection))
    token = _features.set(tuple(features))
    set_token = _feature_sets.set(tuple(sets))
    try:
        yield
    finally:
        _feature_sets.reset(set_token)
        _features.reset(token)
