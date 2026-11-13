# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import math
import random
from collections import deque

log = logging.getLogger("kilora")

RING_MAX = 0.5
RING_MIN = 0.0
RING_POWER = 1.0


def ring_fraction(progress, high=RING_MAX, low=RING_MIN, power=RING_POWER):
    progress = min(1.0, max(0.0, float(progress)))
    return low + (high - low) * ((1 + math.cos(math.pi * progress)) / 2) ** power


def ring_width(bounds, progress, **schedule):
    top, left, bottom, right = bounds
    side = max(bottom - top, right - left)
    return math.ceil(side * ring_fraction(progress, **schedule))


def base_extent(bounds, margin=0.125):
    top, left, bottom, right = bounds
    h, w = bottom - top, right - left
    return h + 2 * max(8, math.ceil(h * margin)), w + 2 * max(8, math.ceil(w * margin))


def ring_cost(h, w, m):
    return (h + 2 * m) * (w + 2 * m) / (h * w)


def ring_shift(m, seed, step, key):

    if m <= 0:
        return 0, 0
    rng = random.Random(f"{seed}:{step}:{key}")
    return rng.randint(-m, m), rng.randint(-m, m)


def blend_weight(mask, area, strength=1.0):

    h, w, top, left = area
    return mask[..., top : top + h, left : left + w] * strength


def crop_box(bounds, size, patch=2, margin=0.125, ring=0, shift=(0, 0)):

    height, width = size
    top, left, bottom, right = bounds
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("Mask bounds must lie inside the latent")
    if patch < 1 or margin < 0 or ring < 0 or any(abs(d) > ring for d in shift):
        raise ValueError("Invalid crop grid, margin, ring or shift")
    grid = math.lcm(16, patch)

    def axis(start, end, extent, offset):
        extra = max(8, math.ceil((end - start) * margin)) + ring
        start, end = max(0, start - extra + offset), min(extent, end + extra + offset)
        length = math.ceil((end - start) / grid) * grid
        if length >= extent or extent % patch:
            return 0, extent
        start = (start // patch) * patch
        if start + length < end:
            start = math.ceil((end - length) / patch) * patch
        start = min(start, ((extent - length) // patch) * patch)
        return start, length

    y, h = axis(top, bottom, height, shift[0])
    x, w = axis(left, right, width, shift[1])
    return h, w, y, x


class CropHistory:
    def __init__(self, steps=4):
        if steps < 4:
            raise ValueError("Crop history must cover at least four steps")
        self.boxes = deque(maxlen=steps)

    def update(self, bounds):
        self.boxes.append(bounds)
        return (
            min(box[0] for box in self.boxes),
            min(box[1] for box in self.boxes),
            max(box[2] for box in self.boxes),
            max(box[3] for box in self.boxes),
        )


def sampling_progress(sigma, sample_sigmas=None, first=None):

    values = []
    if sample_sigmas is not None:
        import torch

        from ..call_values import host_list

        try:
            source = (
                host_list(sample_sigmas)
                if isinstance(sample_sigmas, torch.Tensor)
                else sample_sigmas.flatten()
            )
            values = [float(v) for v in source if float(v) > 0]
        except (AttributeError, TypeError, ValueError):
            values = []
    if values:
        high, low = max(values), min(values)
        if high <= low:
            return 0.0
        return min(1.0, max(0.0, (high - sigma) / (high - low)))
    if first is None or first <= 0:
        return 0.0
    return min(1.0, max(0.0, 1 - sigma / first))


def _shift_seed():
    import os

    value = os.environ.get("KILORA_CROP_RING_SHIFT", "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        log.warning("KiLoRA: KILORA_CROP_RING_SHIFT must be an integer seed. The shift is off.")
        return None


def install_layer_crop(patcher, patch=2, *, safe=True, calc=None, shift_seed=None):

    import torch
    import torch.nn.functional as F

    from ..call_values import host_scalar

    options = patcher.model_options
    previous = options.get("sampler_calc_cond_batch_function")
    histories = {}
    controls = {}
    state = {"step": None, "shape": None, "areas": {}, "costs": {}, "first": None, "index": -1}
    stats = {"costs": [], "step_costs": {}, "mean": None, "runs": []}
    if shift_seed is None:
        shift_seed = _shift_seed()

    def finish_run():
        if stats["costs"] and stats["mean"] is None:
            stats["mean"] = sum(stats["costs"]) / len(stats["costs"])
            stats["runs"].append(stats["mean"])
            log.info(
                "KiLoRA: layer crop ring, mean token cost %.3g over %d crop steps.",
                stats["mean"],
                len(stats["costs"]),
            )

    def start_run():
        finish_run()
        stats["costs"], stats["step_costs"], stats["mean"] = [], {}, None

    def sampler(args):
        x = args["input"]
        sigma = args["sigma"]
        step = host_scalar(sigma)
        shape = tuple(x.shape[-2:])
        if state["step"] is None or step > state["step"] or shape != state["shape"]:
            histories.clear()
            state["areas"], state["costs"] = {}, {}
            state["first"], state["index"] = step, -1
            start_run()
        new_step = step != state["step"] or shape != state["shape"]
        if new_step:
            state["index"] += 1
        state["step"], state["shape"] = step, shape
        transformer = args.get("model_options", {}).get("transformer_options", {})
        sample_sigmas = transformer.get("sample_sigmas")
        progress = sampling_progress(step, sample_sigmas, state["first"])
        patched = bool(transformer.get("patches") or transformer.get("patches_replace"))
        unsupported_area = any(
            c.get("area") is not None and len(c["area"]) != 4
            for group in args["conds"]
            if group is not None
            for c in group
        )

        def crop(cond, key):
            if (
                not safe
                or patched
                or unsupported_area
                or x.ndim not in (4, 5)
                or cond.get("kilora_crop") != "auto"
                or cond.get("kilora_alpha_source") != "mask"
                or cond.get("area") is not None
                or not isinstance(cond.get("mask"), torch.Tensor)
            ):
                return cond
            control = cond.get("control")
            chain = []
            while control is not None:
                if (
                    not isinstance(getattr(control, "cond_hint_original", None), torch.Tensor)
                    or not callable(getattr(control, "copy", None))
                    or any(control is c for c in chain)
                ):
                    return cond
                chain.append(control)
                control = getattr(control, "previous_controlnet", None)
            mask = cond["mask"].detach().float()
            if mask.ndim < 2:
                return cond
            mask = mask.reshape(-1, 1, *mask.shape[-2:]).amax(dim=0, keepdim=True)
            if mask.shape[-2:] != shape:
                if mask.shape[-2] >= shape[0] and mask.shape[-1] >= shape[1]:
                    mask = F.adaptive_max_pool2d(mask, shape)
                else:
                    mask = F.interpolate(mask, size=shape, mode="nearest")
            if new_step or key not in state["areas"]:
                points = torch.nonzero(mask[0, 0] > 0, as_tuple=False)
                if not points.numel():
                    state["areas"].pop(key, None)
                    state["costs"].pop(key, None)
                    return cond
                lo, hi = points.amin(0), points.amax(0) + 1
                bounds = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
                bounds = histories.setdefault(key, CropHistory()).update(bounds)
                ring = ring_width(bounds, progress)
                shift = (
                    ring_shift(ring, shift_seed, state["index"], key)
                    if shift_seed is not None
                    else (0, 0)
                )
                area = crop_box(bounds, shape, patch=patch or 2, ring=ring, shift=shift)
                state["areas"][key] = area
                state["costs"][key] = (ring, ring_cost(*base_extent(bounds), ring))
            area = state["areas"][key]
            if area[0] * area[1] >= 0.7 * shape[0] * shape[1]:
                return cond
            if any(
                round((area[2] + area[0]) * c.cond_hint_original.shape[-2] / shape[0])
                <= round(area[2] * c.cond_hint_original.shape[-2] / shape[0])
                or round((area[3] + area[1]) * c.cond_hint_original.shape[-1] / shape[1])
                <= round(area[3] * c.cond_hint_original.shape[-1] / shape[1])
                for c in chain
            ):
                return cond
            result = {**cond, "area": area}
            step_key = (state["index"], key)
            if key in state["costs"] and step_key not in stats["step_costs"]:
                ring, cost = state["costs"][key]
                stats["costs"].append(cost)
                stats["step_costs"][step_key] = cost
                log.debug(
                    "KiLoRA: layer crop ring %d at progress %.3f, token cost %.3g.",
                    ring,
                    progress,
                    cost,
                )
            if chain:
                h, w, top, left = area
                previous_control = None
                for original in reversed(chain):
                    cache_key = (id(original), area, shape)
                    cached = controls.get(cache_key)
                    if cached is None or cached[0] is not original:
                        hint = original.cond_hint_original
                        hh, hw = hint.shape[-2:]
                        y0, x0 = round(top * hh / shape[0]), round(left * hw / shape[1])
                        y1, x1 = round((top + h) * hh / shape[0]), round((left + w) * hw / shape[1])
                        copied = original.copy()
                        copied.cond_hint_original = hint[..., y0:y1, x0:x1]
                        copied.cond_hint = None
                        copied.previous_controlnet = previous_control
                        controls[cache_key] = (original, copied)
                    previous_control = controls[cache_key][1]
                result["control"] = previous_control
            return result

        groups = args["conds"]
        new_groups = [
            None if group is None else [crop(cond, (i, j)) for j, cond in enumerate(group)]
            for i, group in enumerate(groups)
        ]
        if any(
            c is not old and "area" in c
            for group, old_group in zip(new_groups, groups, strict=True)
            if group is not None
            for c, old in zip(group, old_group, strict=True)
        ):
            from comfy.conds import CONDRegular

            def add_window(cond):
                h, w, top, left = cond.get("area", (*shape, 0, 0))
                model_conds = dict(cond.get("model_conds", {}))
                model_conds["kilora_crop_window"] = CONDRegular(torch.tensor([[top, left, *shape]]))
                return {**cond, "model_conds": model_conds}

            new_groups = [
                None if group is None else [add_window(c) for c in group] for group in new_groups
            ]
        args = {**args, "conds": new_groups}
        try:
            if previous is not None:
                return previous(args)
            fn = calc
            if fn is None:
                from comfy.samplers import calc_cond_batch as fn
            return fn(args["model"], args["conds"], x, sigma, args["model_options"])
        finally:
            if sample_sigmas is not None and progress >= 1.0:
                finish_run()

    sampler._kilora_layer_alpha = True
    sampler._kilora_previous = previous
    sampler._kilora_ring_stats = stats
    options["sampler_calc_cond_batch_function"] = sampler
