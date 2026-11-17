# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import re
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import families, recording
from ..library import contrast
from ..presets import pack_license
from ..regions import token_weights
from . import image_learn, patching
from . import prompt_solve as P


@dataclass(frozen=True)
class _Image:
    latent: torch.Tensor
    caption: str
    weights: torch.Tensor | None = None


def _prompt(caption, cue):
    return f"{caption.strip().rstrip('.')}. {cue.strip().rstrip('.')}." if caption.strip() else cue


def _crop_spatial(channels_first, size):
    height, width = channels_first.shape[-2:]
    if not height or not width:
        raise ValueError("Images must have a nonzero width and height")
    scale = size / min(height, width)
    resized = F.interpolate(
        channels_first,
        size=(max(size, round(height * scale)), max(size, round(width * scale))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top, left = (resized.shape[-2] - size) // 2, (resized.shape[-1] - size) // 2
    return resized[..., top : top + size, left : left + size]


def _encode_image(patcher, vae, image, size):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError("Each image must be an H x W x RGB tensor")
    height, width = image.shape[1:3]
    if not height or not width:
        raise ValueError("Images must have a nonzero width and height")
    pixels = _crop_spatial(image[..., :3].movedim(-1, 1), size).movedim(1, -1)
    latent = vae.encode(pixels)
    return patcher.model.process_latent_in(latent).detach().cpu()


def _mask_plane(mask, height, width):
    if not isinstance(mask, torch.Tensor):
        raise ValueError("each mask must be a height by width tensor")
    plane = mask.detach().float()
    if plane.ndim == 3 and plane.shape[0] == 1:
        plane = plane[0]
    if plane.ndim != 2:
        raise ValueError("each mask must be a height by width tensor")
    if not bool(torch.isfinite(plane).all()):
        raise ValueError("masks must be finite")
    if plane.shape != (height, width):
        plane = F.interpolate(
            plane[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0]
    return plane


def _mask_weights(image, mask, latent, size):
    if image.ndim == 3:
        height, width = int(image.shape[0]), int(image.shape[1])
    elif image.ndim == 4:
        height, width = int(image.shape[1]), int(image.shape[2])
    else:
        raise ValueError("Each image must be an H x W x RGB tensor")
    cropped = _crop_spatial(_mask_plane(mask, height, width)[None, None], size)
    tokens = int(latent.shape[-2] * latent.shape[-1])
    weights = token_weights(cropped, tokens)
    if weights is None:
        raise ValueError("mask does not reduce to the latent token grid")
    if float(weights.sum()) <= 0:
        raise ValueError("mask covers no latent tokens")
    return weights.reshape(1, 1, *latent.shape[-2:]).detach().cpu()


def _mask_schedule(steps, mask_start=0.4, mask_samples=""):
    try:
        fraction = float(mask_start)
    except (TypeError, ValueError) as exc:
        raise ValueError("mask_start must be a number from 0 to 0.95") from exc
    if not math.isfinite(fraction) or not 0 <= fraction <= 0.95:
        raise ValueError("mask_start must be a number from 0 to 0.95")
    last = steps - 1
    start = round(fraction * last)
    if mask_samples != "":
        if not isinstance(mask_samples, str):
            raise ValueError("mask_samples must list numeric schedule positions")
        try:
            positions = [
                float(value) for value in re.split(r"[\s,]+", mask_samples.strip()) if value
            ]
        except ValueError as exc:
            raise ValueError("mask_samples must list numeric schedule positions") from exc
        if not positions:
            raise ValueError("mask_samples must contain at least one position")
        if any(not math.isfinite(value) or value < fraction or value > 1 for value in positions):
            raise ValueError(
                "mask_samples positions must be at or after mask_start and at most 1.0"
            )
        return start, tuple(sorted({round(value * last) for value in positions}))
    span = last - start
    chosen = []
    for index in range(1, 4):
        step = start + index * span // 3
        if step <= start or step in chosen:
            continue
        chosen.append(step)
    if last > start and last not in chosen:
        chosen.append(last)
    if not chosen:
        chosen.append(last)
    return start, tuple(chosen)


def _prepare(patcher, vae, items, size):
    prepared = []
    for item in items:
        if isinstance(item, _Image):
            prepared.append(item)
            continue
        image, caption = item[0], item[1]
        mask = item[2] if len(item) > 2 else None
        latent = _encode_image(patcher, vae, image, size)
        weights = None if mask is None else _mask_weights(image, mask, latent, size)
        prepared.append(_Image(latent, caption, weights))
    return prepared


def _style_items(items):
    result = []
    for item in items:
        weights = None if item.weights is None else 1 - item.weights
        if weights is not None and float(weights.sum()) <= 0:
            raise ValueError("two_stage needs nonempty background coverage for every image")
        result.append(_Image(item.latent, item.caption, weights))
    return result


def _saved_mounts(path):
    from safetensors.torch import load_file

    metadata = _record(path)
    weights = load_file(str(path))
    mounts = {}
    for name, fit in metadata["fit"].items():
        original = metadata.get("export_names", {}).get(name, contrast.anima_name(name))
        stem = (
            original
            if metadata.get("canonical_export")
            else "lora_unet_" + original.replace(".", "_")
        )
        mounts[name] = contrast.MountSolution(
            weights[stem + ".lora_down.weight"],
            weights[stem + ".lora_up.weight"].T * 0.7,
            fit["ridge"],
            fit["fit_share"],
        )
    return mounts


def _holdout_indices(count):
    if count < 3:
        raise ValueError("Supply at least 3 images so at least 2 remain after holding out images")
    wanted = max(1, round(count / 5))
    indices = list(range(4, count, 5))[:wanted]
    if len(indices) < wanted:
        indices.append(count - 1)
    if count - len(indices) < 2:
        raise ValueError("At least 2 images must remain after holding out images")
    return indices


def _pair_masks(items, masks):
    if masks is None:
        return items
    if not isinstance(masks, torch.Tensor) or masks.ndim not in (2, 3):
        raise ValueError("masks must be a MASK batch with one mask per image")
    batch = masks.unsqueeze(0) if masks.ndim == 2 else masks
    if len(batch) != len(items):
        raise ValueError(f"Mask count {len(batch)} does not match image count {len(items)}")
    return [(image, caption, mask) for (image, caption), mask in zip(items, batch, strict=True)]


def _load_items(images, captions, folder, masks=None):
    has_images = images is not None
    has_folder = bool(folder and folder.strip())
    if has_images == has_folder:
        raise ValueError("Supply either images or folder, with exactly one of them set")
    if has_images:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be an IMAGE batch")
        lines = captions.splitlines() if captions else [""] * len(images)
        if len(lines) != len(images):
            raise ValueError(f"Caption count {len(lines)} does not match image count {len(images)}")
        return _pair_masks(list(zip(images, lines, strict=True)), masks)
    if captions.strip():
        raise ValueError("Use .txt caption files with folder; captions is for the images batch")
    import numpy as np
    from PIL import Image, ImageOps

    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Image folder does not exist: {root}")
    items = []
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        with Image.open(path) as image:
            pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255
        caption_path = path.with_suffix(".txt")
        caption = (
            caption_path.read_text(encoding="utf-8-sig").strip() if caption_path.exists() else ""
        )
        items.append((torch.from_numpy(pixels), caption))
    return _pair_masks(items, masks)


@torch.inference_mode()
def image_samples(
    patcher,
    vae,
    items,
    cue,
    settings,
    progress,
    *,
    clip=None,
    mask_start=0.4,
    mask_samples="",
    student_change=None,
):

    if clip is None:
        raise ValueError("Image solve requires the connected CLIP encoder")
    if student_change is not None:
        adapter = P.SolveAdapter(patcher)
        changes = adapter.changes()
        try:
            adapter.install()
            for name, solution in student_change.mounts.items():
                changes.add(name, solution)
            return _image_samples(
                patcher,
                vae,
                items,
                cue,
                settings,
                progress,
                clip=clip,
                mask_start=mask_start,
                mask_samples=mask_samples,
                changes=changes,
            )
        finally:
            changes.close()
            adapter.close()
    return _image_samples(
        patcher,
        vae,
        items,
        cue,
        settings,
        progress,
        clip=clip,
        mask_start=mask_start,
        mask_samples=mask_samples,
    )


def _image_samples(
    patcher,
    vae,
    items,
    cue,
    settings,
    progress,
    *,
    clip,
    mask_start=0.4,
    mask_samples="",
    changes=None,
):
    import comfy.sample
    import comfy.samplers

    spec = families.get_family(patching.detect_family(patcher).family)
    sampling, _ = P._solve_sampling(spec)
    items = _prepare(patcher, vae, items, settings.size)
    masked = any(item.weights is not None for item in items)
    if masked and any(item.weights is None for item in items):
        raise ValueError("every image needs a mask when masks are connected")
    t_start = _mask_schedule(settings.steps, mask_start, mask_samples)[0] if masked else 0
    single = P.empty_latent(patcher, settings.size)
    count = len(settings.seeds)
    latent = torch.cat([single] * count).to(patcher.load_device)
    samples = []
    negative = P._encode(clip, settings.negative_prompt)
    for image_index, item in enumerate(items):
        conditioning = P._encode(clip, _prompt(item.caption, cue))
        noise = torch.cat([comfy.sample.prepare_noise(single, seed) for seed in settings.seeds]).to(
            patcher.load_device
        )
        z = item.latent.to(patcher.load_device).expand_as(noise)
        original = patcher.model_options.get("model_function_wrapper")
        options = dict(patcher.model_options)
        options.pop("model_function_wrapper", None)
        seen = []
        per_seed = [[] for _ in settings.seeds]

        row_weights = None if item.weights is None else item.weights.detach().cpu()

        def wrapped(
            function,
            arguments,
            conditioning=conditioning,
            noise=noise,
            z=z,
            original=original,
            options=options,
            seen=seen,
            per_seed=per_seed,
            image_index=image_index,
            row_weights=row_weights,
            t_start=t_start,
        ):
            chunks, timestep = arguments["cond_or_uncond"], arguments["timestep"]
            sigma = float(timestep.flatten()[0])
            if 0 in chunks and (not seen or sigma != seen[-1]):
                seen.append(sigma)
                if t_start + len(seen) - 1 in settings.sample_steps:
                    start = chunks.index(0) * count
                    x = arguments["input"][start : start + count]
                    times = (
                        timestep.expand(count)
                        if timestep.numel() == 1
                        else timestep[start : start + count]
                    )
                    noised = patcher.model.model_sampling.noise_scaling(
                        times.reshape(-1, *([1] * (noise.ndim - 1))), noise, z
                    )
                    target = P._capture_call(patcher, conditioning, noised, times, options)
                    student = P._capture_call(patcher, conditioning, x, times, options)
                    targets, students = (
                        P._split_sample_call(target, count),
                        P._split_sample_call(student, count),
                    )
                    if targets is None or students is None:
                        targets, students = [], []
                        for row in range(count):
                            row_time = times[row : row + 1]
                            targets.append(
                                P._capture_call(
                                    patcher, conditioning, noised[row : row + 1], row_time, options
                                )
                            )
                            students.append(
                                P._capture_call(
                                    patcher, conditioning, x[row : row + 1], row_time, options
                                )
                            )
                    for row, (target, student) in enumerate(zip(targets, students, strict=True)):
                        per_seed[row].append(
                            contrast.Sample(
                                target, student, image_index=image_index, weights=row_weights
                            )
                        )
            with changes.active(True) if changes is not None else nullcontext():
                if original is not None:
                    return original(function, arguments)
                return function(arguments["input"], timestep, **arguments["c"])

        guider = comfy.samplers.CFGGuider(patcher)
        guider.model_options = {**patcher.model_options, "model_function_wrapper": wrapped}
        guider.set_conds(conditioning, negative)
        guider.set_cfg(settings.cfg)
        guider.inner_model = patcher.model
        guider.conds = copy.deepcopy(guider.original_conds)
        sigmas = comfy.samplers.calculate_sigmas(
            patcher.model.model_sampling, sampling["scheduler"], settings.steps
        ).to(patcher.load_device)
        patcher.pre_run()
        start_latent = z if item.weights is not None else latent
        run_sigmas = sigmas[t_start:] if item.weights is not None else sigmas
        try:
            guider.inner_sample(
                noise,
                start_latent,
                patcher.load_device,
                comfy.samplers.sampler_object(sampling["sampler"]),
                run_sigmas,
                None,
                None,
                True,
                settings.seeds[0],
                **(
                    {"latent_shapes": [tuple(start_latent.shape)]}
                    if "latent_shapes" in inspect.signature(guider.inner_sample).parameters
                    else {}
                ),
            )
        finally:
            patcher.cleanup()
        if len(seen) < settings.steps - t_start:
            raise ValueError("Sampler returned fewer trajectory steps than requested")
        for seed_samples in per_seed:
            samples.extend(seed_samples)
            progress.update(1)
    return samples


_GUARD_SCALES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


class _WhileTrial:
    def __init__(self, trial):
        self.trial = trial

    def __bool__(self):
        return recording.active_perturbation() is self.trial


def _held_out_loss(denoised, target, weights):
    if weights is None:
        return float(F.mse_loss(denoised.float(), target.float()))
    err = (denoised.float() - target.float()).square()
    rows = contrast._token_row_weights(weights, target.movedim(1, -1), require_image=True)
    scale = rows.float().reshape(target.shape[0], 1, target.shape[-2], target.shape[-1])
    covered = scale.sum()
    if float(covered) <= 0:
        raise ValueError("mask covers no latent tokens")
    return float((err * scale).sum() / (covered * err.shape[1]))


def _trajectory_sample(patcher, clip, settings, sampling, prompt, interrupt):
    import comfy.sample
    import comfy.samplers

    latent = P.empty_latent(patcher, settings.size).to(patcher.load_device)
    seed = settings.seeds[0]
    noise = comfy.sample.prepare_noise(latent, seed).to(patcher.load_device)
    guider = comfy.samplers.CFGGuider(patcher)
    guider.set_conds(P._encode(clip, prompt), P._encode(clip, settings.negative_prompt))
    guider.set_cfg(settings.cfg)
    guider.inner_model = patcher.model
    guider.conds = copy.deepcopy(guider.original_conds)
    sigmas = comfy.samplers.calculate_sigmas(
        patcher.model.model_sampling, sampling["scheduler"], 8
    ).to(patcher.load_device)
    patcher.pre_run()
    try:
        with contrast._without_malloc_graph():
            return guider.inner_sample(
                noise,
                latent,
                patcher.load_device,
                comfy.samplers.sampler_object(sampling["sampler"]),
                sigmas,
                None,
                lambda *args, **kwargs: interrupt(),
                True,
                seed,
                **(
                    {"latent_shapes": [tuple(latent.shape)]}
                    if "latent_shapes" in inspect.signature(guider.inner_sample).parameters
                    else {}
                ),
            )
    finally:
        patcher.cleanup()


def _trajectory_guard(patcher, clip, settings, sampling, changes, guard, interrupt):
    checks = guard["trajectory"] = []
    candidates = sorted(
        (
            entry
            for entry in guard["losses"]
            if entry["allowed"] and entry["scale"] <= guard["scale"]
        ),
        key=lambda entry: entry["scale"],
        reverse=True,
    )
    guard["scale"] = guard["relative_gain"] = 0.0
    if not candidates:
        return guard
    started = time.perf_counter()
    bases = {}
    for prompt in image_learn.UNRELATED_PROMPTS[:2]:
        interrupt()
        with changes.active(False):
            bases[prompt] = _trajectory_sample(
                patcher, clip, settings, sampling, prompt, interrupt
            ).float()
    for entry in candidates:
        changes.scale = entry["scale"]
        prompts = []
        for prompt, base in bases.items():
            interrupt()
            with changes.active(True):
                changed = _trajectory_sample(
                    patcher, clip, settings, sampling, prompt, interrupt
                ).float()
            base_std = float(base.std(dim=(-2, -1), correction=0).mean())
            changed_std = float(changed.std(dim=(-2, -1), correction=0).mean())
            ratio = 1.0 if base_std == changed_std == 0.0 else changed_std / max(base_std, 1e-30)
            distance = float(torch.linalg.vector_norm(changed - base)) / max(
                float(torch.linalg.vector_norm(base)), 1e-30
            )
            collapse = (
                not math.isfinite(ratio)
                or not math.isfinite(distance)
                or ratio < 0.5
                or distance > 1.0
            )
            prompts.append(
                {"prompt": prompt, "std_ratio": ratio, "distance": distance, "collapse": collapse}
            )
        collapse = any(item["collapse"] for item in prompts)
        checks.append(
            {
                "scale": entry["scale"],
                "prompts": prompts,
                "std_ratio": min(item["std_ratio"] for item in prompts),
                "distance": max(item["distance"] for item in prompts),
                "collapse": collapse,
                "seconds": time.perf_counter() - started,
            }
        )
        if collapse:
            entry["allowed"] = False
            started = time.perf_counter()
        else:
            guard["scale"] = entry["scale"]
            base_loss = guard["losses"][0]["loss"]
            guard["relative_gain"] = (base_loss - entry["loss"]) / max(base_loss, 1e-30)
            break
    return guard


@torch.inference_mode()
def _guard(
    patcher,
    clip,
    items,
    cue,
    settings,
    adapter,
    result,
    interrupt,
    *,
    guard_rule="loss",
    trajectory_check=False,
):
    import comfy.sample
    import comfy.samplers
    from comfy.sampler_helpers import convert_cond

    sampling, _ = P._solve_sampling(adapter.spec)
    sigmas = comfy.samplers.calculate_sigmas(
        patcher.model.model_sampling, sampling["scheduler"], settings.steps
    ).to(patcher.load_device)
    options = dict(patcher.model_options)
    options.pop("model_function_wrapper", None)
    changes = adapter.changes()
    model = patcher.model.diffusion_model
    training = {module: module.training for module in model.modules()}
    grouped = {scale: [] for scale in _GUARD_SCALES}
    drifts = {}
    started = False
    try:
        model.eval()
        adapter.install()
        for name, solution in result.mounts.items():
            changes.add(name, solution)
        patcher.pre_run()
        started = True
        for item in items:
            conditioning = P._encode(clip, _prompt(item.caption, cue))
            z = item.latent.to(patcher.load_device)
            for seed in settings.seeds:
                noise = comfy.sample.prepare_noise(torch.zeros_like(item.latent), seed).to(z.device)
                for step in settings.sample_steps:
                    interrupt()
                    sigma = sigmas[step : step + 1]
                    x = patcher.model.model_sampling.noise_scaling(sigma, noise, z)
                    cond = comfy.samplers.encode_model_conds(
                        patcher.model.extra_conds,
                        convert_cond(conditioning),
                        x,
                        x.device,
                        "positive",
                    )
                    for scale in _GUARD_SCALES:
                        changes.scale = scale
                        with changes.active(True), contrast._without_malloc_graph():
                            denoised = comfy.samplers.calc_cond_batch(
                                patcher.model, [cond], x, sigma, options
                            )[0]
                        grouped[scale].append(_held_out_loss(denoised, z, item.weights))
        started = False
        patcher.cleanup()
        for scale in _GUARD_SCALES:
            if scale == 0.0:
                continue
            interrupt()
            trial = recording.Perturbation(proj={}, du={})
            changes.scale = scale
            with changes.active(_WhileTrial(trial)):
                drifts[scale] = float(image_learn.unrelated_drift(patcher, clip, settings, trial))
        means = {scale: sum(values) / len(values) for scale, values in grouped.items()}
        guard = _guard_result(means, drifts, rule=guard_rule)
        if trajectory_check:
            _trajectory_guard(patcher, clip, settings, sampling, changes, guard, interrupt)
    finally:
        changes.close()
        adapter.close()
        for module, was_training in training.items():
            module.training = was_training
        if started:
            patcher.cleanup()
    return guard


def _guard_result(means, drifts, bound=None, rule="loss"):

    from ..live import DEFAULT_DRIFT_THRESHOLD

    if rule not in ("loss", "drift"):
        raise ValueError("guard_rule must be loss or drift")
    bound = DEFAULT_DRIFT_THRESHOLD if bound is None else float(bound)
    base = means[0.0]
    entries = []
    for scale in _GUARD_SCALES:
        loss = means[scale]
        drift = 0.0 if scale == 0.0 else drifts[scale]
        allowed = (
            scale > 0
            and math.isfinite(loss)
            and math.isfinite(drift)
            and math.isfinite(base)
            and drift <= bound
            and (loss < base if rule == "loss" else loss <= 1.25 * base)
        )
        entries.append({"scale": scale, "loss": loss, "drift": drift, "allowed": allowed})
    allowed = [entry for entry in entries if entry["allowed"]]
    if rule == "drift":
        best = max(allowed, key=lambda entry: entry["scale"]) if allowed else None
    else:
        best = min(allowed, key=lambda entry: (entry["loss"], entry["scale"])) if allowed else None
    scale = best["scale"] if best else 0.0
    gain = (base - best["loss"]) / max(base, 1e-30) if best else 0.0
    result = {"losses": entries, "drift_bound": bound, "scale": scale, "relative_gain": gain}
    if rule != "loss":
        result["rule"] = rule
    return result


def _scale_label(scale):
    return f"{float(scale):g}"


def _guard_report(guard):
    lines = [f"Guard rule: {guard.get('rule', 'loss')}"]
    for item in guard["losses"]:
        line = f"Scale {_scale_label(item['scale'])}: held-out loss {item['loss']:.3g}"
        if item["scale"] > 0:
            line += f", drift {item['drift']:.3g}, "
            line += "allowed" if item["allowed"] else "not allowed"
        lines.append(line)
    for check in guard.get("trajectory", []):
        lines.append(
            f"Trajectory check at scale {_scale_label(check['scale'])}: "
            f"std ratio {check['std_ratio']:.3g}, distance {check['distance']:.3g}, "
            f"{'collapse' if check['collapse'] else 'passed'}, {check['seconds']:.3f} s"
        )
    if _guard_passed(guard):
        lines.append(f"Chose scale {_scale_label(guard['scale'])}, saved for strength 0.7")
    else:
        lines.append("No scale is within the drift bound, not saved")
    return "\n".join(lines)


def _guard_passed(guard):
    entries = guard["losses"]
    numbers = [item["loss"] for item in entries]
    numbers.extend(item.get("drift", math.nan) for item in entries)
    numbers.extend((guard["scale"], guard["relative_gain"]))
    chosen = [item for item in entries if item["scale"] == guard["scale"]]
    return (
        all(math.isfinite(number) for number in numbers)
        and guard["scale"] > 0
        and bool(chosen)
        and all(item.get("allowed") is True for item in chosen)
    )


def _record_save_scale(result, guard):

    if _guard_passed(guard):
        multiplier = guard["scale"] / 0.7
        for solution in result.mounts.values():
            solution.up = solution.up * multiplier
    else:
        multiplier = 1.0
    guard["multiplier"] = multiplier


def _record(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        return json.loads((file.metadata() or {}).get("kilora_directions", "{}"))


class KiLoRASolveFromImages:
    @classmethod
    def INPUT_TYPES(cls):
        defaults = P.KiLoRASolveFromPrompts.INPUT_TYPES()["required"]
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "name": ("STRING", {"default": "image-solve"}),
                "cue": ("STRING", {"default": "style"}),
                **{key: defaults[key] for key in ("rank", "steps", "cfg", "size", "seeds")},
                "ignore_guard": ("BOOLEAN", {"default": False}),
                "overwrite": defaults["overwrite"],
            },
            "optional": {
                "images": ("IMAGE",),
                "captions": ("STRING", {"default": "", "multiline": True}),
                "folder": ("STRING", {"default": ""}),
                "contrast_components": (
                    "INT",
                    {"default": 4, "min": 0, "max": 4, "advanced": True},
                ),
                "masks": ("MASK",),
                "mask_contrast": (["tokens", "mean"], {"default": "tokens"}),
                "two_stage": ("BOOLEAN", {"default": False}),
                "mask_start": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 0.95, "advanced": True},
                ),
                "mask_samples": ("STRING", {"default": "", "advanced": True}),
                "refit_rounds": ("INT", {"default": 0, "min": 0, "max": 8, "advanced": True}),
                "guard_rule": (["loss", "drift"], {"default": "loss", "advanced": True}),
                "trajectory_check": ("BOOLEAN", {"default": False, "advanced": True}),
            },
        }

    RETURN_TYPES = ("STRING", "KILORA_SETS", "STRING")
    RETURN_NAMES = ("paths", "sets", "report")
    FUNCTION = "solve"
    CATEGORY = "KiLoRA"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @torch.inference_mode()
    def solve(
        self,
        model,
        clip,
        vae,
        name,
        cue,
        rank=16,
        steps=0,
        cfg=-1.0,
        size=0,
        seeds="811 1811",
        ignore_guard=False,
        overwrite=False,
        images=None,
        captions="",
        folder="",
        contrast_components=4,
        masks=None,
        mask_contrast="tokens",
        two_stage=False,
        mask_start=0.4,
        mask_samples="",
        refit_rounds=0,
        guard_rule="loss",
        trajectory_check=False,
        _stage=None,
        _earlier=None,
    ):
        if two_stage:
            if not isinstance(name, str) or not re.fullmatch(r"[\w-]+", name):
                raise ValueError("name must contain only letters, numbers, underscores or dashes")
            common = dict(
                model=model,
                clip=clip,
                vae=vae,
                cue=cue,
                rank=rank,
                steps=steps,
                cfg=cfg,
                size=size,
                seeds=seeds,
                ignore_guard=ignore_guard,
                overwrite=overwrite,
                images=images,
                captions=captions,
                folder=folder,
                contrast_components=contrast_components,
                masks=masks,
                mask_contrast=mask_contrast,
                mask_start=mask_start,
                mask_samples=mask_samples,
                refit_rounds=refit_rounds,
                guard_rule=guard_rule,
                trajectory_check=trajectory_check,
            )
            style = self.solve(name=name + "-style", _stage="creative", **common)["result"]
            if not style[0]:
                report = style[2] + "\nStyle guard failed; character solve skipped."
                return {"ui": {"text": [report]}, "result": ("", style[1], report)}
            earlier = _saved_mounts(style[0])
            character = self.solve(
                name=name + "-character", _stage="ip", _earlier=earlier, **common
            )["result"]
            paths = "\n".join(path for path in (style[0], character[0]) if path)
            from .nodes import FOLDERS, KiLoRAImportLoRA

            chain = style[1]
            if character[0]:
                relative = Path(character[0]).relative_to(Path(FOLDERS.first_dir("loras")))
                (chain,) = KiLoRAImportLoRA().add(relative.as_posix(), 0.7, sets=chain)
            report = "Style stage\n" + style[2] + "\nCharacter stage\n" + character[2]
            return {"ui": {"text": [report]}, "result": (paths, chain, report)}
        if guard_rule not in ("loss", "drift"):
            raise ValueError("guard_rule must be loss or drift")
        if type(refit_rounds) is not int or not 0 <= refit_rounds <= 8:
            raise ValueError("refit_rounds must be an integer from 0 to 8")
        if mask_contrast not in ("tokens", "mean"):
            raise ValueError("mask_contrast must be tokens or mean")
        items = _load_items(images, captions, folder, masks)
        held = _holdout_indices(len(items))
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]+", name):
            raise ValueError("name must contain only letters, numbers, underscores or dashes")
        if not cue.strip():
            raise ValueError("cue must be nonempty")
        try:
            seed_values = tuple(int(s) for s in re.split(r"[\s,]+", seeds.strip()) if s)
        except ValueError as exc:
            raise ValueError("seeds must be a string of integers") from exc
        if any(seed < 0 or seed >= 2**64 for seed in seed_values):
            raise ValueError("seeds must be integers from 0 to 18446744073709551615")
        adapter = P.SolveAdapter(model)
        sampling, sampling_report = P._solve_sampling(adapter.spec, steps=steps, cfg=cfg, size=size)
        steps, cfg, size = sampling["steps"], sampling["cfg"], sampling["size"]
        if type(contrast_components) is not int or not 0 <= contrast_components <= 4:
            raise ValueError("contrast_components must be an integer from 0 to 4")
        mask_schedule = _mask_schedule(steps, mask_start, mask_samples)
        if masks is None:
            sample_steps = tuple(sorted({0, (steps - 1) // 2, steps - 1}))
            contrast_mode = "mean"
        else:
            sample_steps = mask_schedule[1]
            contrast_mode = mask_contrast
        settings = contrast.Settings(
            seeds=seed_values,
            size=size,
            steps=steps,
            cfg=cfg,
            rank=rank,
            contrast_components=min(contrast_components, len(items) - len(held) - 1),
            sample_steps=sample_steps,
            contrast_mode=contrast_mode,
            refit_rounds=refit_rounds,
        )
        settings.validate()
        sha = P._model_sha(model)
        items = _prepare(model, vae, items, size)
        if _stage == "creative":
            items = _style_items(items)
        adapter.latent_hw = tuple(items[0].latent.shape[-2:])
        record = json.loads(
            json.dumps(
                {
                    "image_hashes": [
                        hashlib.sha256(
                            item.latent.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                        ).hexdigest()
                        for item in items
                    ],
                    "captions": [item.caption for item in items],
                    "cue": cue,
                    "settings": asdict(settings),
                    "model_sha": sha,
                    "family": adapter.family,
                    "sampler": sampling["sampler"],
                    "scheduler": sampling["scheduler"],
                    "holdout_indices": held,
                    "version": 2,
                }
            )
        )
        for key, value, default in (
            ("mask_start", mask_start, 0.4),
            ("mask_samples", mask_samples, ""),
            ("guard_rule", guard_rule, "loss"),
            ("trajectory_check", trajectory_check, False),
        ):
            if value != default:
                record[key] = value
        if masks is not None:
            record["mask_hashes"] = [
                hashlib.sha256(
                    item.weights.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for item in items
            ]
            record["mask_contrast"] = mask_contrast
            record["mask_layout"] = "latent_plane"
        if _stage is not None:
            record["stage"] = _stage
        if _earlier is not None:
            record["style_hash"] = hashlib.sha256(
                b"".join(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    for solution in _earlier.values()
                    for tensor in (solution.down, solution.up)
                )
            ).hexdigest()
        from .nodes import FOLDERS, KiLoRAImportLoRA

        root = Path(FOLDERS.first_dir("loras")) / "kilora-solved"
        path = root / f"{name}.safetensors"
        metadata = None
        kept = None
        if not overwrite and path.exists():
            metadata = _record(path)
            if metadata.get("image_solve") != record:
                kept = path
                digest = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:8]
                path = path.with_stem(f"{path.stem}-{digest}")
                metadata = _record(path) if path.exists() else None
                if metadata is not None and metadata.get("image_solve") != record:
                    raise ValueError(
                        f"Output exists with different solve inputs: {path}; enable overwrite"
                    )
        lines = [f"Images: {len(items)}", f"Hold-out images: {len(held)}", sampling_report]
        if metadata is None:
            import comfy.model_management as mm
            from comfy.utils import ProgressBar

            fitted = [item for index, item in enumerate(items) if index not in held]
            pairs = [
                contrast.PromptPair(
                    _prompt(item.caption, cue), _prompt(item.caption, cue), name, cue
                )
                for item in fitted
            ]
            estimate = contrast.memory_estimate(
                model.model.diffusion_model, settings, len(fitted), shapes=adapter.shapes
            )
            message = (
                f"Estimated GPU memory: {estimate / 1024**3:.2f} GiB (weights and solve workspace)"
            )
            print(message)
            lines.append(message)
            try:
                load_options = inspect.signature(mm.load_models_gpu).parameters
                if "force_full_load" not in load_options:
                    mm.load_models_gpu([model], memory_required=estimate)
                    lines.append("Model load mode unchanged (full-load option unavailable)")
                elif hasattr(mm, "get_free_memory") and estimate <= mm.get_free_memory(
                    adapter.device
                ):
                    mm.load_models_gpu([model], memory_required=estimate, force_full_load=True)
                    lines.append("Model loaded fully for the solve")
                else:
                    mm.load_models_gpu([model], memory_required=estimate)
                    lines.append("Model loaded in parts (not enough free GPU memory)")
                mm.throw_exception_if_processing_interrupted()
                progress = ProgressBar(len(fitted) * len(settings.seeds))
                started = time.perf_counter()
                sample_options = {"clip": clip}
                if mask_start != 0.4 or mask_samples != "":
                    sample_options.update(mask_start=mask_start, mask_samples=mask_samples)
                samples = image_samples(
                    model, vae, fitted, cue, settings, progress, **sample_options
                )
                trajectory_seconds = time.perf_counter() - started
                blocks = ProgressBar(len(adapter.blocks))
                started = time.perf_counter()

                def solve_round(samples, round_index=0):
                    return contrast.solve(
                        model.model.diffusion_model,
                        samples,
                        pairs,
                        settings=settings,
                        model_sha=sha,
                        license=pack_license(adapter.family)[0],
                        adapter=adapter,
                        progress=lambda message: blocks.update(1),
                        interrupt=mm.throw_exception_if_processing_interrupted,
                        **({"refit_round": round_index} if settings.refit_rounds else {}),
                        **(
                            {
                                "coupling": contrast.ConceptCoupling(
                                    deflate=True, earlier=[_earlier], order=("style", "character")
                                )
                            }
                            if _earlier is not None
                            else {}
                        ),
                    )

                result = solve_round(samples)
                replay_seconds = time.perf_counter() - started - result.fit_seconds
                lines.append(
                    f"Times: trajectories {trajectory_seconds:.3f} s, "
                    f"block replays {max(0.0, replay_seconds):.3f} s, "
                    f"fits {result.fit_seconds:.3f} s "
                    f"({result.metadata['exact_fits']} of {len(result.mounts)} "
                    "mounts on the exact path)"
                )

                def resample(*, student_change):
                    mm.throw_exception_if_processing_interrupted()
                    return image_samples(
                        model,
                        vae,
                        fitted,
                        cue,
                        settings,
                        ProgressBar(len(fitted) * len(settings.seeds)),
                        student_change=student_change,
                        **sample_options,
                    )

                for round_index, round_result, seconds in contrast.refit_results(
                    result, settings, resample, solve_round
                ):
                    result = round_result
                    lines.append(f"Refit round {round_index}: {seconds:.3f} s")
                background = adapter.changes()
                try:
                    for mount, solution in (_earlier or {}).items():
                        background.add(mount, solution)
                    with background.active(True):
                        guard = _guard(
                            model,
                            clip,
                            [items[index] for index in held],
                            cue,
                            settings,
                            adapter,
                            result,
                            mm.throw_exception_if_processing_interrupted,
                            **({"guard_rule": guard_rule} if guard_rule != "loss" else {}),
                            **({"trajectory_check": True} if trajectory_check else {}),
                        )
                finally:
                    background.close()
                _record_save_scale(result, guard)
                metadata = result.metadata
                metadata.update({"concept": name, "image_solve": record, "image_guard": guard})
                if _stage is not None:
                    metadata["axis"] = _stage
                lines.append(_guard_report(guard))
                if _guard_passed(guard) or ignore_guard:
                    if not _guard_passed(guard):
                        lines.append("Warning: guard failed; ignore_guard saves anyway.")
                    import folder_paths

                    temp_root = folder_paths.get_temp_directory()
                    Path(temp_root).mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
                        temporary = Path(tmp) / path.name
                        contrast.write_lora(temporary, result)
                        mm.throw_exception_if_processing_interrupted()
                        path.parent.mkdir(parents=True, exist_ok=True)
                        P._publish(temporary, path, overwrite=overwrite)
                    if kept is not None:
                        lines.append(f"Kept {kept} (other solve inputs); wrote {path}")
                    lines.append(f"Wrote {path}")
            finally:
                mm.soft_empty_cache()
        else:
            guard = metadata["image_guard"]
            lines.append(_guard_report(guard))
            if _guard_passed(guard) or ignore_guard:
                if not _guard_passed(guard):
                    lines.append("Warning: guard failed; ignore_guard saves anyway.")
                lines.append(f"Reused {path}")
        lines.append(f"Contrast components: {settings.contrast_components}")
        ranks = Counter(
            fit["measured_rank"]
            for fit in metadata.get("fit", {}).values()
            if "measured_rank" in fit
        )
        if ranks:
            summary = ", ".join(f"{rank} x{count}" for rank, count in sorted(ranks.items()))
            lines.append(f"Fitted rank per mount: {summary}")
        chain = patching.ModChain()
        if _guard_passed(guard) or ignore_guard:
            (chain,) = KiLoRAImportLoRA().add(
                path.relative_to(root.parent).as_posix(), 0.7, sets=chain
            )
            paths = str(path)
        else:
            paths = ""
            lines.append("ignore_guard saves anyway.")
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (paths, chain, report)}
