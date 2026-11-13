# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import errno
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import time
import weakref
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from .. import families
from ..errors import KiLoRAError
from ..library import contrast
from ..presets import anima_name, pack_license
from . import patching

PROMPT_SET = "KILORA_PROMPT_SET"
_CAPTION_MOUNT = re.compile(r"(?:cross_attn|attn2)\.(?:k_proj|v_proj|to_k|to_v)$")


def _solve_families() -> list[str]:
    return sorted(
        name
        for name in families.BLOCK_LISTS
        if (spec := families.FAMILIES.get(name)) is not None
        and not any(layout.text for _, layout in spec.layouts)
        and any(
            layout.kind == "text1d" and _CAPTION_MOUNT.search(pattern)
            for pattern, layout in spec.layouts
        )
        and all(
            layout.kind != "text1d" or _CAPTION_MOUNT.search(pattern)
            for pattern, layout in spec.layouts
        )
    )


@dataclass(frozen=True)
class PromptSet:
    groups: dict[str, list[contrast.PromptPair]]

    def validated(self):
        return contrast.prompt_set_from_data(
            {"pairs": [asdict(pair) for pairs in self.groups.values() for pair in pairs]}
        )


class KiLoRAPromptSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subjects": ("STRING", {"default": "a tree\na house", "multiline": True}),
                "style": ("STRING", {"default": "watercolor", "multiline": True}),
                "cue": ("STRING", {"default": "watercolor"}),
                "label": ("STRING", {"default": "watercolor"}),
            },
            "optional": {
                "json": ("STRING", {"default": "", "multiline": True}),
                "prompt_set": (PROMPT_SET,),
            },
        }

    RETURN_TYPES = (PROMPT_SET, "STRING")
    RETURN_NAMES = ("prompt_set", "report")
    FUNCTION = "build"
    CATEGORY = "KiLoRA"

    def build(self, subjects, style, cue, label, json="", prompt_set=None):
        import json as json_module

        if json.strip():
            try:
                data = json_module.loads(json)
            except json_module.JSONDecodeError as exc:
                raise ValueError(f"Invalid prompt set JSON: {exc}") from exc
        else:
            data = {
                "concepts": [
                    {
                        "label": label,
                        "style": style,
                        "cue": cue,
                        "subjects": [
                            line.strip() for line in subjects.splitlines() if line.strip()
                        ],
                    }
                ]
            }
        added = contrast.prompt_set_from_data(data)
        groups = prompt_set.validated() if prompt_set is not None else {}
        for name, pairs in added.items():
            if name in groups:
                raise ValueError(f"duplicate concept label: {name}")
            groups[name] = pairs
        report = "\n".join(f"{name}: {len(pairs)} pairs" for name, pairs in groups.items())
        return PromptSet(groups), report


def _rows(output, part):
    if part is None:
        return output
    index, count = part
    width = output.shape[-1] // count
    return output[..., index * width : (index + 1) * width]


class _MountView:
    def __init__(self, module, part):
        self.module, self.part = module, part

    def register_forward_pre_hook(self, callback):
        return self.module.register_forward_pre_hook(callback)

    def register_forward_hook(self, callback):
        return self.module.register_forward_hook(
            lambda module, args, output: callback(module, args, _rows(output, self.part))
        )


class _AdapterChanges(contrast._Changes):
    def __init__(self, adapter):
        self.adapter = adapter
        self.enabled = False
        self.scale = 1.0
        self.handles = []

    def add(self, name, solution):
        module, part = self.adapter.locations[name]
        down = solution.down.to(self.adapter.device)
        up = solution.up.T.to(self.adapter.device)

        def apply(module, args, output):
            if not self.enabled or self.scale == 0.0:
                return output
            x = args[0].float()
            change = F.linear(F.linear(x, down), up)
            if part is not None:
                index, count = part
                width = change.shape[-1]
                change = F.pad(change, (index * width, (count - index - 1) * width))
            return output + (self.scale * change).to(output.dtype)

        self.handles.append(module.register_forward_hook(apply))


class SolveAdapter:
    def __init__(self, patcher, latent_hw=None):
        supported = _solve_families()
        try:
            self.binding = patching.detect_family(patcher)
        except KiLoRAError as exc:
            declared = getattr(patcher.model.diffusion_model, "kilora_family", "unknown")
            raise ValueError(
                f"Prompt solve has no adapter for family {declared!r}. "
                f"Supported families: {', '.join(supported)}."
            ) from exc
        self.family = self.binding.family
        self.spec = families.FAMILIES.get(self.family)
        self.latent_hw = latent_hw
        self.select_image_rows = (
            self.family not in supported
            and self.spec is not None
            and bool(self.spec.image_token_target)
            and callable(getattr(self.spec, "image_token_rows", None))
        )
        if self.family not in supported and not self.select_image_rows:
            raise ValueError(
                f"Prompt solve does not support single-stream family {self.family} yet: "
                "its text and image tokens share one sequence. "
                f"Supported families: {', '.join(supported)}."
            )
        model = patcher.model.diffusion_model
        try:
            attribute, blocks = families.block_list(self.family, model)
        except (ValueError, KiLoRAError) as exc:
            raise ValueError(
                f"Prompt solve has no block adapter for family {self.family!r}. "
                f"Supported families: {', '.join(supported)}."
            ) from exc
        self.device = getattr(patcher, "load_device", None)
        if self.device is None:
            import comfy.model_management

            self.device = comfy.model_management.get_torch_device()
        self.blocks = dict(blocks)
        self.shapes = patching.diffusers_shapes(patcher, self.binding)
        tree = dict(model.named_modules())
        self.locations, self.owners = {}, {}
        for name in self.shapes:
            place = self.binding.to_comfy(name)
            path = place.path.removeprefix("diffusion_model.")
            for index, _ in blocks:
                if path.startswith(f"{attribute}.{index}."):
                    self.locations[name] = (tree[path], place.part)
                    self.owners[name] = index
                    break
        self.mounts = list(self.locations)
        self.forced = []
        self.modules = {n: _MountView(*where) for n, where in self.locations.items()}
        if not self.mounts:
            raise ValueError(f"Family {self.family!r} has no supported mounts in its block adapter")

    def install(self):
        for module, _ in self.locations.values():
            if any(previous is module for previous, _, _ in self.forced):
                continue
            present = "comfy_force_forward" in module.__dict__
            self.forced.append((module, present, getattr(module, "comfy_force_forward", None)))
            module.comfy_force_forward = True

    def close(self):
        for module, present, value in self.forced:
            if present:
                module.comfy_force_forward = value
            else:
                module.__dict__.pop("comfy_force_forward", None)
        self.forced.clear()

    def owner(self, name):
        return self.owners[name]

    def rows(self, tokens):
        if not self.select_image_rows:
            return None
        if self.latent_hw is None:
            raise ValueError(f"Family {self.family!r} requires latent_hw for image rows")
        rows = self.spec.image_token_rows(tokens, self.latent_hw)
        if rows is None:
            raise ValueError(f"Family {self.family!r} has no target-image rows for {tokens} tokens")
        return rows

    def is_caption(self, name):
        if self.select_image_rows:
            return False
        path = self.binding.to_comfy(name).path
        return bool(_CAPTION_MOUNT.search(path))

    def changes(self):
        return _AdapterChanges(self)

    def metadata(self):
        license, notice = pack_license(self.family)
        canonical = self.family != "anima"
        return {
            "family": self.family,
            "license": license,
            "license_notice": notice,
            "canonical_export": canonical,
            "export_names": {
                name: "diffusion_model." + name if canonical else anima_name(name)
                for name in self.mounts
            },
        }

    def groups(self, index, inputs):
        ordered, handles = [], []
        previous_input = None
        previous_version = None
        physical = {}
        for name, (module, _) in self.locations.items():
            if self.owner(name) == index:
                physical.setdefault(module, []).append(name)
        try:
            for module, names in physical.items():

                def record(module, args, output, names=names):
                    nonlocal previous_input, previous_version
                    x = args[0]
                    version = None if torch.is_inference(x) else x._version
                    if x is previous_input and version == previous_version:
                        ordered[-1].extend(names)
                    else:
                        ordered.append(list(names))
                    previous_input, previous_version = x, version

                handles.append(module.register_forward_hook(record))
            contrast._run_block(self.blocks[index], inputs)
        finally:
            for handle in handles:
                handle.remove()
        called = [name for group in ordered for name in group]
        expected = [name for names in physical.values() for name in names]
        if len(called) != len(set(called)) or set(called) != set(expected):
            raise ValueError(f"Family {self.family!r}: block mounts must be called exactly once")
        return ordered


_MODEL_HASHES = weakref.WeakKeyDictionary()


def _model_sha(patcher):
    if getattr(patcher, "patches", {}) or patching.applied(patcher):
        raise ValueError("Prompt solve requires a base MODEL without applied LoRAs or KiLoRA mods")
    return state_sha(patcher)


def state_sha(patcher):
    model = patcher.model
    state = (weakref.ref(model.diffusion_model), getattr(patcher, "patches_uuid", None))
    cached = _MODEL_HASHES.get(model)
    if cached is not None and cached[0] == state:
        return cached[1]
    digest = hashlib.sha256()
    for name, value in sorted(patcher.model.diffusion_model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        if not isinstance(value, torch.Tensor):
            digest.update(repr(value).encode())
            continue
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    result = digest.hexdigest()
    _MODEL_HASHES[model] = (state, result)
    return result


def _encode(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _capture_call(patcher, conditioning, x, timestep, options):
    import comfy.samplers

    captured = None
    model = patcher.model.diffusion_model

    def stop(module, args, kwargs):
        nonlocal captured
        call = inspect.signature(module.forward).bind(*args, **kwargs)
        arguments = dict(call.arguments)
        for name, parameter in inspect.signature(module.forward).parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                arguments.update(arguments.pop(name, {}))
        captured = contrast._map_tensors(arguments, lambda t: t.detach().cpu().clone())
        raise contrast._BlockReached

    handle = model.register_forward_pre_hook(stop, with_kwargs=True)
    try:
        from comfy.sampler_helpers import convert_cond

        cond = comfy.samplers.encode_model_conds(
            patcher.model.extra_conds, convert_cond(conditioning), x, x.device, "positive"
        )
        with suppress(contrast._BlockReached), contrast._without_malloc_graph():
            comfy.samplers.calc_cond_batch(patcher.model, [cond], x, timestep, options)
    finally:
        handle.remove()
    if captured is None:
        raise ValueError("ComfyUI did not call the diffusion model while preparing a solve sample")
    return captured


def empty_latent(patcher, size):
    latent_format = patcher.model.latent_format
    channels = latent_format.latent_channels
    compression = getattr(latent_format, "spacial_downscale_ratio", 8)
    shape = (1, channels, size // compression, size // compression)
    if getattr(latent_format, "latent_dimensions", 2) == 3:
        shape = (shape[0], shape[1], 1, *shape[2:])
    return torch.zeros(shape, device="cpu")


def _solve_sampling(spec, *, steps=0, cfg=-1.0, size=0):
    defaults = contrast.Settings()
    family = spec.solve_sampling or {}
    values, sources = {}, {}
    for key, value, auto in (("steps", steps, 0), ("cfg", cfg, -1), ("size", size, 0)):
        if value != auto:
            values[key], sources[key] = value, "explicit"
        elif key in family:
            values[key], sources[key] = family[key], f"family {spec.family}"
        else:
            values[key], sources[key] = getattr(defaults, key), "default"
    for key, fallback in (("sampler", "er_sde"), ("scheduler", "simple")):
        values[key] = family.get(key, fallback)
        sources[key] = f"family {spec.family}" if key in family else "default"
    report = ", ".join(f"{key} {value} ({sources[key]})" for key, value in values.items())
    return values, report


def _split_sample_call(call, count):
    if count == 1:
        return [call]
    hidden = next((value for value in call.values() if isinstance(value, torch.Tensor)), None)
    if hidden is None or hidden.ndim < 3 or hidden.shape[0] != count:
        return None
    return [
        contrast._map_tensors(
            call,
            lambda tensor, row=row: (
                tensor[row : row + 1] if tensor.ndim and tensor.shape[0] == count else tensor
            ),
        )
        for row in range(count)
    ]


def trajectory_samples(patcher, captions, pairs, settings, progress):
    import comfy.sample
    import comfy.samplers

    spec = families.get_family(patching.detect_family(patcher).family)
    sampling, _ = _solve_sampling(spec)
    single_latent = empty_latent(patcher, settings.size)
    count = len(settings.seeds)
    latent = torch.cat([single_latent] * count)
    samples = []
    for pair in pairs:
        original = patcher.model_options.get("model_function_wrapper")
        options = dict(patcher.model_options)
        options.pop("model_function_wrapper", None)
        seen = []
        per_seed = [[] for _ in settings.seeds]

        def wrapped(
            function,
            arguments,
            pair=pair,
            seen=seen,
            options=options,
            original=original,
            per_seed=per_seed,
        ):
            chunks = arguments["cond_or_uncond"]
            timestep = arguments["timestep"]
            sigma = float(timestep.flatten()[0])
            if 0 in chunks and (not seen or sigma != seen[-1]):
                seen.append(sigma)
                step = len(seen) - 1
                if step in settings.sample_steps:
                    start = chunks.index(0) * count
                    x = arguments["input"][start : start + count]
                    times = (
                        timestep.expand(count)
                        if timestep.numel() == 1
                        else timestep[start : start + count]
                    )
                    target = _capture_call(patcher, captions[pair.target], x, times, options)
                    student = _capture_call(patcher, captions[pair.student], x, times, options)
                    targets = _split_sample_call(target, count)
                    students = _split_sample_call(student, count)
                    if targets is None or students is None:
                        targets, students = [], []
                        for row in range(count):
                            row_x, row_time = x[row : row + 1], times[row : row + 1]
                            targets.append(
                                _capture_call(
                                    patcher, captions[pair.target], row_x, row_time, options
                                )
                            )
                            students.append(
                                _capture_call(
                                    patcher, captions[pair.student], row_x, row_time, options
                                )
                            )
                    for row, (target, student) in enumerate(zip(targets, students, strict=True)):
                        per_seed[row].append(contrast.Sample(target, student))
            if original is not None:
                return original(function, arguments)
            return function(arguments["input"], arguments["timestep"], **arguments["c"])

        guider = comfy.samplers.CFGGuider(patcher)
        guider.model_options = {**patcher.model_options, "model_function_wrapper": wrapped}
        guider.set_conds(captions[pair.target], captions[settings.negative_prompt])
        guider.set_cfg(settings.cfg)
        guider.inner_model = patcher.model
        guider.conds = copy.deepcopy(guider.original_conds)
        sigmas = comfy.samplers.calculate_sigmas(
            patcher.model.model_sampling, sampling["scheduler"], settings.steps
        ).to(patcher.load_device)
        noise = torch.cat(
            [comfy.sample.prepare_noise(single_latent, seed) for seed in settings.seeds]
        ).to(patcher.load_device)
        patcher.pre_run()
        try:
            guider.inner_sample(
                noise,
                latent.to(patcher.load_device),
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


class KiLoRASolveFromPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        defaults = contrast.Settings()
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt_set": (PROMPT_SET,),
                "name": ("STRING", {"default": "prompt-solve"}),
                "mode": (["single", "joint"], {"default": "single"}),
                "rank": ("INT", {"default": defaults.rank, "min": 1, "max": 4096}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "size": ("INT", {"default": 0, "min": 0, "step": 16}),
                "cfg": ("FLOAT", {"default": -1.0, "min": -1.0}),
                "seeds": ("STRING", {"default": "811 1811"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "output": (["lora", "kmod", "both"], {"default": "lora"}),
            },
            "optional": {
                "sets": ("KILORA_SETS",),
                "preview": ("IMAGE",),
                "overlap_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "deflate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "KILORA_SETS", "STRING")
    RETURN_NAMES = ("paths", "sets", "report")
    FUNCTION = "solve"
    CATEGORY = "KiLoRA"
    INPUT_IS_LIST = True
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def solve(
        self,
        model,
        clip,
        prompt_set,
        name,
        mode="single",
        rank=16,
        steps=0,
        size=0,
        cfg=-1.0,
        seeds="811 1811",
        negative_prompt="",
        overwrite=False,
        sets=None,
        output="lora",
        preview=None,
        overlap_penalty=0.0,
        deflate=False,
    ):
        def one(value):
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError("Only prompt_set and name may vary within a solve batch")
                return value[0]
            return value

        model, clip = one(model), one(clip)
        mode, rank, steps, size = one(mode), one(rank), one(steps), one(size)
        cfg, seeds = one(cfg), one(seeds)
        negative_prompt, overwrite = one(negative_prompt), one(overwrite)
        sets = one(sets)
        output, preview = one(output), one(preview)
        overlap_penalty, deflate = float(one(overlap_penalty)), bool(one(deflate))
        contrast.check_coupling(overlap_penalty, deflate)
        if output not in ("lora", "kmod", "both"):
            raise ValueError("output must be lora, kmod or both")
        batches = prompt_set if isinstance(prompt_set, list) else [prompt_set]
        names = name if isinstance(name, list) else [name]
        if not batches or len(names) not in (1, len(batches)):
            raise ValueError("Supply one name, or one name per prompt set")
        if mode not in ("single", "joint"):
            raise ValueError("mode must be single or joint")
        try:
            seed_values = tuple(int(s) for s in re.split(r"[\s,]+", seeds.strip()) if s)
        except ValueError as exc:
            raise ValueError("seeds must be a string of integers") from exc
        if any(seed < 0 or seed >= 2**64 for seed in seed_values):
            raise ValueError("seeds must be integers from 0 to 18446744073709551615")
        adapter = SolveAdapter(model)
        sampling, sampling_report = _solve_sampling(adapter.spec, steps=steps, cfg=cfg, size=size)
        steps, cfg, size = sampling["steps"], sampling["cfg"], sampling["size"]
        settings = contrast.Settings(
            seeds=seed_values,
            size=size,
            steps=steps,
            cfg=cfg,
            rank=rank,
            negative_prompt=negative_prompt,
            sample_steps=tuple(sorted({0, (steps - 1) // 2, steps - 1})),
        )
        settings.validate()
        compression = getattr(
            getattr(model.model, "latent_format", None), "spacial_downscale_ratio", 8
        )
        latent_hw = (settings.size // compression, settings.size // compression)
        adapter.latent_hw = latent_hw
        jobs, paths, lines = [], [], [sampling_report]
        from .nodes import FOLDERS, KiLoRAImportLoRA

        root = Path(FOLDERS.first_dir("loras")) / "kilora-solved"
        sha = _model_sha(model)
        records, conflicts = {}, set()
        for i, batch in enumerate(batches):
            stem = names[i] if len(names) > 1 else names[0]
            if not isinstance(stem, str) or not re.fullmatch(r"[\w-]+", stem):
                raise ValueError("name must contain only letters, numbers, underscores or dashes")
            if len(batches) > 1 and len(names) == 1:
                stem += f"-{i + 1}"
            try:
                groups = batch.validated()
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"Prompt set {i + 1} ({stem}): {exc}") from exc
            selected = (
                groups.items()
                if mode == "single"
                else [("joint", [pair for pairs in groups.values() for pair in pairs])]
            )
            coupled = mode == "single" and len(groups) > 1 and (overlap_penalty or deflate)
            for label, pairs in selected:
                filename = (
                    f"{stem}-{label}.safetensors" if mode == "single" else f"{stem}.safetensors"
                )
                path = root / filename
                record = json.loads(
                    json.dumps(
                        {
                            "pairs": [asdict(p) for p in pairs],
                            "settings": asdict(settings),
                            "model_sha": sha,
                            "family": adapter.family,
                            "mode": mode,
                            "sampler": sampling["sampler"],
                            "scheduler": sampling["scheduler"],
                            "version": 1,
                            **(
                                {
                                    "overlap_penalty": overlap_penalty,
                                    "deflate": deflate,
                                    "concept_order": list(groups),
                                    "set_pairs": [
                                        asdict(p) for group in groups.values() for p in group
                                    ],
                                }
                                if coupled
                                else {}
                            ),
                        }
                    )
                )
                if path in records and records[path] != record:
                    conflicts.add(path)
                records.setdefault(path, record)
                jobs.append((i, label, pairs, path, record))
        resolved, used = [], set()
        for i, label, pairs, path, record in jobs:
            reuse = False
            kept = None
            if not overwrite:
                existing_record = record
                if path.exists():
                    with safe_open(str(path), framework="pt", device="cpu") as file:
                        metadata = json.loads(
                            (file.metadata() or {}).get("kilora_directions", "{}")
                        )
                    existing_record = metadata.get("prompt_solve")
                    reuse = existing_record == record
                if existing_record != record or (not reuse and path in conflicts):
                    if existing_record != record:
                        kept = path
                    digest = hashlib.sha256(
                        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()[:8]
                    path = path.with_stem(f"{path.stem}-{digest}")
                    reuse = False
                    if path.exists():
                        with safe_open(str(path), framework="pt", device="cpu") as file:
                            metadata = json.loads(
                                (file.metadata() or {}).get("kilora_directions", "{}")
                            )
                        if metadata.get("prompt_solve") != record:
                            raise ValueError(
                                f"Prompt set {i + 1}, concept {label!r}: "
                                "Output exists with different solve inputs: "
                                f"{path}; enable overwrite"
                            )
                        reuse = True
            if path in used:
                raise ValueError(f"Batch output names collide: {path.name}")
            used.add(path)
            if kept is not None and not reuse:
                lines.append(f"Kept {kept} (other solve inputs); wrote {path}")
            resolved.append((i, label, pairs, path, record, reuse))
        jobs = resolved
        coupled_sets = {
            i for i, _, _, _, record, reuse in jobs if "concept_order" in record and not reuse
        }
        replaced = {path for i, _, _, path, _, reuse in jobs if reuse and i in coupled_sets}
        jobs = [
            (i, label, pairs, path, record, reuse and i not in coupled_sets)
            for i, label, pairs, path, record, reuse in jobs
        ]
        pending = [job for job in jobs if not job[-1]]
        captions = {}
        if pending:
            for _, _, pairs, _, _, _ in pending:
                for text in [
                    settings.negative_prompt,
                    *[p.target for p in pairs],
                    *[p.student for p in pairs],
                ]:
                    if text not in captions:
                        captions[text] = _encode(clip, text)
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        chain = patching.ModChain() if sets is None else sets
        loaded = False
        try:
            if pending:
                estimate = contrast.memory_estimate(
                    model.model.diffusion_model,
                    settings,
                    max(len(job[2]) for job in pending),
                    shapes=adapter.shapes,
                )
                message = (
                    f"Estimated GPU memory: {estimate / 1024**3:.2f} GiB "
                    "(weights and solve workspace)"
                )
                print(message)
                lines.append(message)
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
                loaded = True

            def run(pairs, coupling=None):
                mm.throw_exception_if_processing_interrupted()
                progress = ProgressBar(len(pairs) * len(settings.seeds))
                started = time.perf_counter()
                samples = trajectory_samples(model, captions, pairs, settings, progress)
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
                    progress=lambda message, bar=blocks: bar.update(1),
                    interrupt=mm.throw_exception_if_processing_interrupted,
                    **({} if coupling is None else {"coupling": coupling}),
                )
                replay_seconds = time.perf_counter() - started - result.fit_seconds
                lines.append(
                    f"Times: trajectories {trajectory_seconds:.3f} s, "
                    f"block replays {max(0.0, replay_seconds):.3f} s, "
                    f"fits {result.fit_seconds:.3f} s "
                    f"({result.metadata['exact_fits']} of {len(result.mounts)} "
                    "mounts on the exact path)"
                )
                return result

            coupled_results = {}
            for set_index in sorted(coupled_sets):
                members = {label: pairs for i, label, pairs, *_ in jobs if i == set_index}
                try:
                    solved = contrast.solve_concepts(
                        list(members),
                        lambda label, coupling, members=members: run(members[label], coupling),
                        overlap_penalty=overlap_penalty,
                        deflate=deflate,
                    )
                except Exception as exc:
                    message = f"Prompt set {set_index + 1}: {exc}"
                    interruption = getattr(mm, "InterruptProcessingException", InterruptedError)
                    if isinstance(exc, (InterruptedError, interruption)):
                        exc.args = (message,)
                        raise
                    raise ValueError(message) from exc
                coupled_results.update({(set_index, k): v for k, v in solved.items()})
            for i, label, pairs, path, record, reuse in jobs:
                try:
                    if reuse:
                        lines.append(f"Reused {path}")
                    else:
                        result = coupled_results.get((i, label))
                        if result is None:
                            result = run(pairs)
                        result.metadata.update({"concept": label, "prompt_solve": record})
                        import folder_paths

                        temp_root = folder_paths.get_temp_directory()
                        Path(temp_root).mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
                            temporary = Path(tmp) / path.name
                            contrast.write_lora(temporary, result)
                            mm.throw_exception_if_processing_interrupted()
                            path.parent.mkdir(parents=True, exist_ok=True)
                            _publish(temporary, path, overwrite=overwrite or path in replaced)
                        lines.append(f"Wrote {path}")
                    relative = path.relative_to(root.parent).as_posix()
                    if output == "lora":
                        (chain,) = KiLoRAImportLoRA().add(relative, 0.7, sets=chain)
                    if output in ("kmod", "both"):
                        from .pack_nodes import solved_mod

                        mod_path = solved_mod(
                            path,
                            adapter,
                            record,
                            label,
                            pairs,
                            preview,
                            overwrite=overwrite,
                        )
                        chain = chain.add(patching.ModEntry("mod", str(mod_path), 0.7))
                        paths.append(str(mod_path))
                        lines.append(f"Wrote {mod_path}")
                    if output in ("lora", "both"):
                        paths.append(str(path))
                    elif not reuse:
                        path.unlink()
                except Exception as exc:
                    message = f"Prompt set {i + 1}, concept {label!r}: {exc}"
                    interruption = getattr(mm, "InterruptProcessingException", InterruptedError)
                    if isinstance(exc, (InterruptedError, interruption)):
                        exc.args = (message,)
                        raise
                    raise ValueError(message) from exc
        finally:
            if loaded:
                mm.soft_empty_cache()
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": ("\n".join(paths), chain, report)}


def _publish(source, destination, *, overwrite=False):
    try:
        if overwrite:
            os.replace(source, destination)
        else:
            os.link(source, destination)
            source.unlink()
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
        staged = Path(name)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            if overwrite:
                os.replace(staged, destination)
            else:
                os.link(staged, destination)
            source.unlink()
        finally:
            staged.unlink(missing_ok=True)
