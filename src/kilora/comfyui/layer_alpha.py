# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F

from ..attention_maps import AttentionMaps, alpha_from_maps, trigger_positions

log = logging.getLogger(__name__)


def remove_layer_alpha(options):
    for key in ("model_function_wrapper", "sampler_calc_cond_batch_function"):
        current = options.get(key)
        while getattr(current, "_kilora_layer_alpha", False):
            current = current._kilora_previous
        if current is None:
            options.pop(key, None)
        else:
            options[key] = current


def install_layer_alpha(patcher, family, triggers, blocks, calc=None, *, ip_mounted=None):
    options = patcher.model_options
    remove_layer_alpha(options)
    previous_calc = options.get("sampler_calc_cond_batch_function")
    previous_model = options.get("model_function_wrapper")
    state = {"step": None, "shape": None, "collector": None, "alphas": {}, "active": False}
    warned = False
    from ..axis_owner import mode

    apply_owner = mode() == "layer"
    state["owner"] = None

    def model_wrapper(apply_model, args):
        c = args.get("c") or {}
        topts = c.get("transformer_options") or {}
        collector = state["collector"]
        if state["active"] and collector is not None and topts.get("kilora_layer") == "ip":
            c = {**c, "transformer_options": collector.patch_options(topts)}
            args = {**args, "c": c}
        from ..axis_owner import recording

        kinds = topts.get("cond_or_uncond") or ()
        owner = state["owner"] if state["active"] and apply_owner and mode() == "layer" else None
        scope = recording(owner, kinds) if owner is not None else contextlib.nullcontext()
        with scope:
            if previous_model is not None:
                return previous_model(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

    def sampler(args):
        nonlocal warned
        conds = args["conds"]
        if apply_owner and mode() == "layer":
            return owner_sampler(args)
        state["owner"] = None
        selected = [
            cond
            for group in conds
            if group is not None
            for cond in group
            if cond.get("kilora_alpha_source") == "attention" and cond.get("kilora_layer") == "ip"
        ]
        if not selected:
            return evaluate(args)
        sigma = args["sigma"]
        step = float(sigma.flatten()[0]) if isinstance(sigma, torch.Tensor) else float(sigma)
        shape = tuple(args["input"].shape[-2:])
        if state["step"] is None or step > state["step"] or shape != state["shape"]:
            state.update(collector=None, alphas={})
        elif step != state["step"] and state["collector"] is not None:
            maps = state["collector"].finish_step()
            if maps:
                count = next(iter(maps.values())).shape[-1]
                grid = next(
                    (
                        (shape[0] // n, shape[1] // n)
                        for n in range(1, min(shape) + 1)
                        if shape[0] % n == 0
                        and shape[1] % n == 0
                        and (shape[0] // n) * (shape[1] // n) == count
                    ),
                    None,
                )
                if grid is not None:
                    from ..compute import budget_share

                    layers = {
                        c.get("kilora_layer")
                        for group in conds
                        if group is not None
                        for c in group
                        if c.get("kilora_layer") is not None
                    }
                    state["alphas"] = alpha_from_maps(
                        maps, state["alphas"], grid=grid, share=budget_share(layers)
                    )
        state["step"], state["shape"] = step, shape
        if state["collector"] is None:
            positions, message = trigger_positions(family or "", selected[0], triggers)
            if positions and blocks:
                state["collector"] = AttentionMaps(positions, blocks)
            elif not warned:
                log.warning(message or "Trigger attention maps are off: block count is unknown.")
                warned = True
        alphas = state["alphas"]
        if alphas:

            def replace(cond):
                axis = cond.get("kilora_layer")
                if cond.get("kilora_alpha_source") != "attention" or axis not in alphas:
                    return cond
                alpha = alphas[axis]
                h, w = shape
                count = alpha.shape[-1]
                grid = next(
                    (
                        (h // n, w // n)
                        for n in range(1, min(h, w) + 1)
                        if h % n == 0 and w % n == 0 and (h // n) * (w // n) == count
                    ),
                    None,
                )
                if grid is None:
                    return cond
                image = alpha.reshape(alpha.shape[0], 1, *grid)
                image = F.interpolate(image, size=shape, mode="bilinear", align_corners=False)
                mask = image[:, 0]
                if args["input"].ndim == 5:
                    mask = mask.unsqueeze(1).expand(-1, args["input"].shape[2], -1, -1)
                return {**cond, "mask": mask, "mask_strength": 1.0}

            args = {
                **args,
                "conds": [
                    None if group is None else [replace(c) for c in group] for group in conds
                ],
            }
        state["active"] = True
        try:
            return evaluate(args)
        finally:
            state["active"] = False

    def owner_sampler(args):
        from ..axis_owner import AxisOwner

        dimensions = args["input"].shape[2:]
        if len(dimensions) > 2 and any(n != 1 for n in dimensions[:-2]):
            if not state.get("warned_owner_frames"):
                log.info("axis owner: disabled for multiple frames")
                state["warned_owner_frames"] = True
            state.update(owner=None, step=None, shape=None)
            return evaluate(args)

        conds = args["conds"]
        layers = [
            c
            for group in conds
            if group is not None
            for c in group
            if c.get("kilora_layer") in ("ip", "creative", "functional")
        ]
        if not layers:
            return evaluate(args)
        sigma = args["sigma"]
        step = float(sigma.flatten()[0]) if isinstance(sigma, torch.Tensor) else float(sigma)
        shape = tuple(args["input"].shape[-2:])
        if (
            state["step"] is None
            or step > state["step"]
            or shape != state["shape"]
            or state["owner"] is None
            or state["owner"].batch != args["input"].shape[0]
        ):
            state["owner"] = AxisOwner(shape, args["input"].shape[0], args["input"].device)
        elif step != state["step"]:
            state["owner"].finish_step()
        state["step"], state["shape"] = step, shape
        masks = state["owner"].masks
        from ..call_values import ip_early_threshold

        threshold = ip_early_threshold()
        early = bool(
            threshold
            and step > threshold
            and ip_mounted is not False
            and any(c.get("kilora_layer") == "ip" for c in layers)
        )
        if early:
            sample = args["input"].new_ones((args["input"].shape[0], *shape))
            masks = {"ip": sample, "creative": torch.zeros_like(sample)}
        if masks:

            def replace(cond):
                axis = cond.get("kilora_layer")
                if axis not in ("ip", "creative", "functional"):
                    return cond
                if early and axis == "functional":
                    return cond
                sample = next(iter(masks.values()))
                mask = (
                    torch.ones_like(sample)
                    if axis == "functional"
                    else masks.get(axis, torch.zeros_like(sample))
                )
                if args["input"].ndim == 5:
                    mask = mask.unsqueeze(1).expand(-1, args["input"].shape[2], -1, -1)
                return {**cond, "mask": mask, "mask_strength": 1.0}

            args = {
                **args,
                "conds": [
                    None if group is None else [replace(c) for c in group] for group in conds
                ],
            }
        state["active"] = True
        try:
            return evaluate(args)
        finally:
            state["active"] = False

    def evaluate(args):
        if previous_calc is not None:
            return previous_calc(args)
        fn = calc
        if fn is None:
            from comfy.samplers import calc_cond_batch as fn
        return fn(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])

    model_wrapper._kilora_layer_alpha = True
    model_wrapper._kilora_previous = previous_model
    sampler._kilora_layer_alpha = True
    sampler._kilora_previous = previous_calc
    options["model_function_wrapper"] = model_wrapper
    options["sampler_calc_cond_batch_function"] = sampler
