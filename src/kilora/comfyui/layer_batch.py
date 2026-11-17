# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os

import torch

from ..ablation_settings import carry_steps
from ..compute import AXES

log = logging.getLogger(__name__)

SPATIAL = (
    "area",
    "strength",
    "mask",
    "mask_strength",
    "set_area_to_bounds",
    "timestep_start",
    "timestep_end",
)


def _follow(negative, layer):

    out = {k: v for k, v in negative.items() if not k.startswith("kilora_") and k not in SPATIAL}
    out.update({k: v for k, v in layer.items() if k.startswith("kilora_") or k in SPATIAL})
    own = layer.get("model_conds") or {}
    if "model_conds" in negative or own:
        text = negative.get("model_conds") or {}
        model_conds = {k: v for k, v in text.items() if not k.startswith("kilora_")}
        model_conds.update({k: v for k, v in own.items() if k.startswith("kilora_")})
        out["model_conds"] = model_conds
    if "area" in layer and "control" in negative and "control" in layer:
        out["control"] = layer["control"]
    return out


def pair_negatives(conds):

    if len(conds) < 2 or conds[0] is None or conds[1] is None:
        return conds, 0
    positive, negative = conds[0], conds[1]
    covered = {c.get("kilora_layer") for c in negative if c.get("kilora_layer") in AXES}
    plain = [c for c in negative if c.get("kilora_layer") not in AXES]
    layered = [
        c for c in positive if c.get("kilora_layer") in AXES and c["kilora_layer"] not in covered
    ]
    if not layered or not plain:
        return conds, 0
    paired = [c for c in negative if c.get("kilora_layer") in AXES]
    for cond in layered:
        paired.extend(_follow(n, cond) for n in plain)
    whole = 0.0
    for cond in positive:
        if cond.get("kilora_layer") in AXES:
            continue
        if any(k in cond for k in ("area", "mask", "timestep_start", "timestep_end")):
            paired.extend(_follow(n, cond) for n in plain)
        else:
            whole += float(cond.get("strength", 1.0))
    if whole == 1.0:
        paired.extend(plain)
    elif whole > 0:
        paired.extend({**n, "strength": float(n.get("strength", 1.0)) * whole} for n in plain)
    return [positive, paired, *conds[2:]], len(layered) * len(plain)


def _separate_layers(entries, options, separate):
    return (
        not all(
            c.get("kilora_layer_batch", False) for c in entries if c.get("kilora_layer") in AXES
        )
        or os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower() == "layer"
        or options.get("kilora_layer_separate", separate)
        or options.get("kilora_layer_filter")
        or any(c.get("kilora_alpha_source") == "attention" for c in entries)
    )


def install_layer_batch(patcher, *, separate=False, calc=None, family=None):
    options = patcher.model_options
    options["kilora_layer_separate"] = bool(separate)
    if family is not None:
        options["kilora_layer_family"] = family
    if options.get("kilora_layer_batch_installed"):
        return
    options["kilora_layer_batch_installed"] = True
    previous = options.get("sampler_calc_cond_batch_function")
    groups = {}
    warned = False
    run = {"seen": False, "step": None}
    told = set()

    def tell(message):
        if message not in told:
            told.add(message)
            log.info("KiLoRA: %s", message)

    def sampler(args):
        nonlocal warned
        conds = args["conds"]
        entries = [c for group in conds if group is not None for c in group]
        layered = any(c.get("kilora_layer") in AXES for c in entries)
        report = None
        composed = None
        if layered:
            step = _step(args.get("sigma"))
            later = step is not None and run["step"] is not None and step > run["step"]
            if not run["seen"] or later:
                report = [0, 0]
                told.clear()
            run["seen"], run["step"] = True, step
            composed = _composed(conds, args["model_options"], separate, tell)
        if layered and composed is not None:
            conds, added = composed, 0
            args = {**args, "conds": conds}
        elif layered:
            x = args.get("input")
            if isinstance(x, torch.Tensor) and x.ndim == 5:

                def add_frames(c):
                    mask = c.get("mask")
                    if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                        return {**c, "mask": mask.unsqueeze(1).expand(-1, x.shape[2], -1, -1)}
                    return c

                conds = [
                    None if group is None else [add_frames(c) for c in group] for group in conds
                ]
            conds, added = pair_negatives(conds)
            fallback = _separate_layers(entries, args["model_options"], separate)
            if fallback:
                if not warned:
                    log.info(
                        "Layers run as separate model calls (batch off, a layer filter, "
                        "an attention alpha source, or carry across steps)."
                    )
                    warned = True
                import comfy.hooks as hooks

                def convert(c):
                    layer = c.get("kilora_layer")
                    if layer not in AXES:
                        return c
                    old = c.get("hooks")
                    key = (layer, old)
                    if key not in groups:
                        group = hooks.HookGroup()
                        group.add(
                            hooks.TransformerOptionsHook(
                                {"kilora_layer": layer}, hook_scope=hooks.EnumHookScope.HookedOnly
                            )
                        )
                        groups[key] = hooks.HookGroup.combine_all_hooks([old, group])
                    return {**c, "hooks": groups[key]}
            else:
                from comfy.conds import CONDRegular

                def convert(c):
                    layer = c.get("kilora_layer")
                    value = AXES.index(layer) if layer in AXES else -1
                    model_conds = dict(c["model_conds"])
                    model_conds["kilora_layer_ids"] = CONDRegular(torch.tensor([value]))
                    return {**c, "model_conds": model_conds}

            args = {
                **args,
                "conds": [
                    None if group is None else [convert(c) for c in group] for group in conds
                ],
            }
        if report is not None:
            args = {**args, "model_options": _counting(args["model_options"], report)}
        result = _evaluate(args, previous, calc)
        if report is not None:
            count = sum(len(group) for group in args["conds"] if group is not None)
            log.info(
                "KiLoRA: layered sample, %d model calls and %d batch rows per step for "
                "%d conds (%d negatives paired with a layer, compose %s).",
                report[0],
                report[1],
                count,
                added,
                "single" if composed is not None else "separate",
            )
        return result

    options["sampler_calc_cond_batch_function"] = sampler


def _composed(conds, options, separate, tell):
    from . import layer_compose

    positive = conds[0] if conds else None
    layers = [c for c in positive or () if c.get("kilora_layer") in AXES]
    if not any(c.get("kilora_compose") == "single" for c in layers):
        return None
    if not layer_compose.SINGLE_READY:
        return None
    groups, info = layer_compose.compose(
        conds,
        options=options,
        family=options.get("kilora_layer_family"),
        carry=carry_steps(bool(options.get("kilora_layer_separate", separate))),
    )
    if groups is None:
        tell(f"compose single falls back to separate: {info}.")
        return None
    if any(c.get("kilora_crop") == "auto" for c in layers):
        tell("compose single does not use the crop input of the layers.")
    tell(layer_compose.settings_note(info[0]))
    from comfy.conds import CONDRegular

    key = layer_compose.register(info)

    def tagged(c):
        model_conds = dict(c["model_conds"])
        model_conds["kilora_compose_id"] = CONDRegular(torch.tensor([key]))
        return {**c, "model_conds": model_conds}

    return [None if group is None else [tagged(c) for c in group] for group in groups]


def _evaluate(args, previous, calc):
    if previous is not None:
        return previous(args)
    fn = calc
    if fn is None:
        from comfy.samplers import calc_cond_batch as fn
    return fn(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])


def _step(sigma):
    if isinstance(sigma, torch.Tensor):
        return float(sigma.flatten()[0]) if sigma.numel() else None
    try:
        return float(sigma)
    except (TypeError, ValueError):
        return None


def _counting(options, report):
    inner = options.get("model_function_wrapper")

    def counting(apply_model, args):
        report[0] += 1
        x = args.get("input")
        if len(report) > 1 and isinstance(x, torch.Tensor) and x.ndim:
            report[1] += int(x.shape[0])
        if inner is not None:
            return inner(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    return {**options, "model_function_wrapper": counting}
