# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from inspect import signature


def _options(args, kwargs):
    def find(value):
        if isinstance(value, Mapping):
            if "kilora_layer" in value:
                return value
            for key in ("transformer_options", "extra_options", "c", "model_options"):
                if key in value:
                    result = find(value[key])
                    if result is not None:
                        return result
        return None

    for value in (*args, kwargs):
        result = find(value)
        if result is not None:
            return result
    return {}


def _gate(fn, layer, fallback):
    @wraps(fn)
    def gated(*args, **kwargs):
        if _options(args, kwargs).get("kilora_layer") == layer:
            return fn(*args, **kwargs)
        return fallback(*args, **kwargs)

    if hasattr(fn, "to"):

        def move(device):
            nonlocal fn
            fn = fn.to(device)
            return gated

        gated.to = move
    if hasattr(fn, "models"):
        gated.models = lambda: fn.models()
    return gated


def _pass_patch(*args, **kwargs):
    if len(args) >= 4 and isinstance(args[3], Mapping):
        return args[:3]
    return args[0]


def _patch_fallback(patch):
    try:
        params = signature(patch).parameters
    except (ValueError, TypeError):
        params = {}
    if "pe" in params and "attn_mask" in params:

        def anima_attention(*args, **kwargs):
            bound = signature(patch).bind(*args, **kwargs)
            bound.apply_defaults()
            return {key: bound.arguments[key] for key in ("q", "k", "v", "pe", "attn_mask")}

        return anima_attention
    return _pass_patch


def _attention(q, k, v, options, **kwargs):
    from comfy.ldm.modules.attention import optimized_attention

    return optimized_attention(
        q, k, v, options["n_heads"], attn_precision=options.get("attn_precision")
    )


def _pass_wrapper(executor, *args, **kwargs):
    return executor(*args, **kwargs)


def _copy_tree(value):
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


def _added(item, old):
    return not any(item is previous for previous in old)


def filter_model(model, before, layer):
    out = model.clone()
    options = _copy_tree(getattr(model, "model_options", {}) or {})
    prior = getattr(before, "model_options", {}) or {}
    trans = options.get("transformer_options", {})
    old_trans = prior.get("transformer_options", {})
    count = 0
    for name, patches in trans.get("patches", {}).items():
        old = old_trans.get("patches", {}).get(name, [])
        for i, patch in enumerate(patches):
            if callable(patch) and _added(patch, old):
                patches[i] = _gate(patch, layer, _patch_fallback(patch))
                count += 1
    for name, entries in trans.get("patches_replace", {}).items():
        old = old_trans.get("patches_replace", {}).get(name, {})
        for key, patch in entries.items():
            previous = old.get(key)
            if callable(patch) and patch is not previous:
                entries[key] = _gate(patch, layer, previous or _attention)
                count += 1
    for current, previous in ((options, prior), (trans, old_trans)):
        for kind, groups in current.get("wrappers", {}).items():
            old_groups = previous.get("wrappers", {}).get(kind, {})
            for key, wrappers in groups.items():
                old = old_groups.get(key, [])
                for i, wrapper in enumerate(wrappers):
                    if callable(wrapper) and _added(wrapper, old):
                        wrappers[i] = _gate(wrapper, layer, _pass_wrapper)
                        count += 1
    fn = options.get("model_function_wrapper")
    old_fn = prior.get("model_function_wrapper")
    if callable(fn) and fn is not old_fn:

        def baseline(apply_model, args):
            if old_fn is not None:
                return old_fn(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        options["model_function_wrapper"] = _gate(fn, layer, baseline)
        count += 1
    if count:
        options["kilora_layer_filter"] = True
    out.model_options = options
    if count:
        from .layer_batch import install_layer_batch

        install_layer_batch(out)
    global_patches = []
    for attr, label in (("patches", "weight"), ("object_patches", "object")):
        current = getattr(model, attr, {}) or {}
        previous = getattr(before, attr, {}) or {}
        for key, value in current.items():
            old = previous.get(key)
            same = value is old
            if isinstance(value, list) and isinstance(old, list):
                same = len(value) == len(old) and all(
                    a is b for a, b in zip(value, old, strict=True)
                )
            if key not in previous or not same:
                global_patches.append(f"{label}: {key}")
    report = f"Filtered {count} function patches for layer {layer}."
    if global_patches:
        report += " Global patches: " + ", ".join(global_patches)
    return out, report


class KiLoRALayerFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "before": ("MODEL",),
                "layer": (["ip", "creative", "functional"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "filter"
    CATEGORY = "KiLoRA"

    def filter(self, model, before, layer):
        out, report = filter_model(model, before, layer)
        return {"ui": {"text": [report]}, "result": (out, report)}
