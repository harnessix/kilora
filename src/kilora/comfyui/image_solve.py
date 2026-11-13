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
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import families, recording
from ..library import contrast
from ..presets import pack_license
from . import image_learn, patching
from . import prompt_solve as P


@dataclass(frozen=True)
class _Image:
    latent: torch.Tensor
    caption: str


def _prompt(caption, cue):
    return f"{caption.strip().rstrip('.')}. {cue.strip().rstrip('.')}." if caption.strip() else cue


def _encode_image(patcher, vae, image, size):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError("Each image must be an H x W x RGB tensor")
    height, width = image.shape[1:3]
    if not height or not width:
        raise ValueError("Images must have a nonzero width and height")
    scale = size / min(height, width)
    resized = F.interpolate(
        image[..., :3].movedim(-1, 1),
        size=(max(size, round(height * scale)), max(size, round(width * scale))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top, left = (resized.shape[-2] - size) // 2, (resized.shape[-1] - size) // 2
    pixels = resized[..., top : top + size, left : left + size].movedim(1, -1)
    latent = vae.encode(pixels)
    return patcher.model.process_latent_in(latent).detach().cpu()


def _prepare(patcher, vae, items, size):
    return [
        item
        if isinstance(item, _Image)
        else _Image(_encode_image(patcher, vae, item[0], size), item[1])
        for item in items
    ]


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


def _load_items(images, captions, folder):
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
        return list(zip(images, lines, strict=True))
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
    return items


@torch.inference_mode()
def image_samples(patcher, vae, items, cue, settings, progress, *, clip=None):

    import comfy.sample
    import comfy.samplers

    if clip is None:
        raise ValueError("Image solve requires the connected CLIP encoder")
    spec = families.get_family(patching.detect_family(patcher).family)
    sampling, _ = P._solve_sampling(spec)
    items = _prepare(patcher, vae, items, settings.size)
    single = P.empty_latent(patcher, settings.size)
    count = len(settings.seeds)
    latent = torch.cat([single] * count).to(patcher.load_device)
    samples = []
    negative = P._encode(clip, settings.negative_prompt)
    for item in items:
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
        ):
            chunks, timestep = arguments["cond_or_uncond"], arguments["timestep"]
            sigma = float(timestep.flatten()[0])
            if 0 in chunks and (not seen or sigma != seen[-1]):
                seen.append(sigma)
                if len(seen) - 1 in settings.sample_steps:
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
                        per_seed[row].append(contrast.Sample(target, student))
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
        try:
            guider.inner_sample(
                noise,
                latent,
                patcher.load_device,
                comfy.samplers.sampler_object(sampling["sampler"]),
                sigmas,
                None,
                None,
                True,
                settings.seeds[0],
                **(
                    {"latent_shapes": [tuple(latent.shape)]}
                    if "latent_shapes" in inspect.signature(guider.inner_sample).parameters
                    else {}
                ),
            )
        finally:
            patcher.cleanup()
        if len(seen) < settings.steps:
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


@torch.inference_mode()
def _guard(patcher, clip, items, cue, settings, adapter, result, interrupt):
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
                        grouped[scale].append(float(F.mse_loss(denoised.float(), z.float())))
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
    finally:
        changes.close()
        adapter.close()
        for module, was_training in training.items():
            module.training = was_training
        if started:
            patcher.cleanup()
    means = {scale: sum(values) / len(values) for scale, values in grouped.items()}
    return _guard_result(means, drifts)


def _guard_result(means, drifts, bound=None):

    from ..live import DEFAULT_DRIFT_THRESHOLD

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
            and loss < base
        )
        entries.append({"scale": scale, "loss": loss, "drift": drift, "allowed": allowed})
    allowed = [entry for entry in entries if entry["allowed"]]
    best = min(allowed, key=lambda entry: (entry["loss"], entry["scale"])) if allowed else None
    scale = best["scale"] if best else 0.0
    gain = (base - best["loss"]) / max(base, 1e-30) if best else 0.0
    return {"losses": entries, "drift_bound": bound, "scale": scale, "relative_gain": gain}


def _scale_label(scale):
    return f"{float(scale):g}"


def _guard_report(guard):
    lines = []
    for item in guard["losses"]:
        line = f"Scale {_scale_label(item['scale'])}: held-out loss {item['loss']:.3g}"
        if item["scale"] > 0:
            line += f", drift {item['drift']:.3g}, "
            line += "allowed" if item["allowed"] else "not allowed"
        lines.append(line)
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
    ):
        items = _load_items(images, captions, folder)
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
        settings = contrast.Settings(
            seeds=seed_values,
            size=size,
            steps=steps,
            cfg=cfg,
            rank=rank,
            sample_steps=tuple(sorted({0, (steps - 1) // 2, steps - 1})),
            contrast_mode="mean",
        )
        settings.validate()
        sha = P._model_sha(model)
        items = _prepare(model, vae, items, size)
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
                samples = image_samples(model, vae, fitted, cue, settings, progress, clip=clip)
                trajectory_seconds = time.perf_counter() - started
                blocks = ProgressBar(len(adapter.blocks))
                started = time.perf_counter()
                result = contrast.solve(
                    model.model.diffusion_model,
                    samples,
                    pairs,
                    settings=settings,
                    model_sha=sha,
                    license=pack_license(adapter.family)[0],
                    adapter=adapter,
                    progress=lambda message: blocks.update(1),
                    interrupt=mm.throw_exception_if_processing_interrupted,
                )
                replay_seconds = time.perf_counter() - started - result.fit_seconds
                lines.append(
                    f"Times: trajectories {trajectory_seconds:.3f} s, "
                    f"block replays {max(0.0, replay_seconds):.3f} s, "
                    f"fits {result.fit_seconds:.3f} s "
                    f"({result.metadata['exact_fits']} of {len(result.mounts)} "
                    "mounts on the exact path)"
                )
                guard = _guard(
                    model,
                    clip,
                    [items[index] for index in held],
                    cue,
                    settings,
                    adapter,
                    result,
                    mm.throw_exception_if_processing_interrupted,
                )
                _record_save_scale(result, guard)
                metadata = result.metadata
                metadata.update({"concept": name, "image_solve": record, "image_guard": guard})
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
