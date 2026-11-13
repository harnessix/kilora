# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os

import torch

from .. import library
from ..i18n import text
from ..library.probe import StepZeroProbe, install
from ..presets import PRESETS
from ..recipes import save_recipe
from ..regions import latent_mask
from . import patching
from .bypass import INJECTION_KEY


class _Choice:
    def __init__(self, patcher, finish=None):
        self.patcher = patcher
        self.finish = finish
        self.injections = list(patcher.injections.get(INJECTION_KEY, ()))
        self.wrapper = patcher.model_options.get("model_function_wrapper")
        self.active = []

    def __call__(self, model, args):
        if not self.active:
            try:
                for item in self.injections:
                    self.active.append(item)
                    item.inject(self.patcher)
            except Exception:
                self.close()
                raise
        if self.wrapper is not None:
            return self.wrapper(model, args)
        return model(args["input"], args["timestep"], **(args.get("c") or {}))

    def close(self):
        try:
            if self.active and self.finish is not None:
                self.finish()
        finally:
            for item in reversed(self.active):
                item.eject(self.patcher)
            self.active.clear()


def _probe_layout(model, binding):
    blocks = []
    for name in ("double_blocks", "single_blocks", "transformer_blocks", "blocks"):
        value = getattr(model, name, None)
        if value is not None:
            blocks.extend(value)
    if not blocks:
        raise ValueError(text("library.no_blocks"))
    groups = {}
    preset = PRESETS[binding.family]
    for path, module in model.named_modules():
        shape = patching.linear_shape(module)
        if shape is None:
            continue
        for target, _ in binding.from_comfy(path):
            group = preset.content_group(target, shape[0])
            if group:
                groups.setdefault(group, []).append(path)
    return blocks, groups


class KiLoRALoRALibrary:
    def __init__(self):
        self.loader = None
        self.folders = None
        self.device = None
        self.last_report = ""

    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import FOLDERS

        try:
            folder = FOLDERS.first_dir("loras")
        except (ImportError, AttributeError, IndexError):
            folder = ""
        return {
            "required": {
                "model": ("MODEL",),
                "folders": ("STRING", {"default": folder, "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "count": ("INT", {"default": 5, "min": 1, "max": 16}),
                "strength": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "min_relevance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "include_tags": ("STRING", {"default": ""}),
                "exclude_tags": ("STRING", {"default": ""}),
            },
            "optional": {"clip": ("CLIP",), "mask_a": ("MASK",), "mask_b": ("MASK",)},
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "report", "recipe")
    FUNCTION = "apply"
    CATEGORY = "KiLoRA"

    def apply(
        self,
        model,
        folders,
        prompt,
        count=5,
        strength=0.8,
        seed=0,
        min_relevance=0.2,
        include_tags="",
        exclude_tags="",
        clip=None,
        mask_a=None,
        mask_b=None,
    ):
        from .nodes import COMFY_DEPTH_STEPS, FOLDERS

        paths = tuple(
            os.path.abspath(os.path.expanduser(p.strip()))
            for p in folders.splitlines()
            if p.strip()
        )
        device = getattr(model, "load_device", None)
        if device is None:
            try:
                import comfy.model_management as model_management

                device = model_management.get_torch_device()
            except ImportError:
                device = torch.device("cpu")
        if self.loader is None or paths != self.folders or device != self.device:
            self.loader = library.LibraryLoader(paths, device=device)
            self.folders = paths
            self.device = device
        loader = self.loader
        index = library.LibraryIndex.open()
        sync = index.sync(paths, clip)
        sync_text = text(
            "library.sync",
            added=sync.added,
            removed=sync.removed,
            skipped=sync.skipped,
            seconds=sync.seconds,
        )
        binding = patching.detect_family(model)
        denoiser = model.model.diffusion_model
        try:
            blocks, groups = _probe_layout(denoiser, binding)
        except ValueError as exc:
            return model, sync_text + "\n" + str(exc), ""
        shapes = patching.shape_model(patching.diffusers_shapes(model, binding))
        identity = [
            paths,
            prompt,
            count,
            strength,
            seed,
            min_relevance,
            include_tags,
            exclude_tags,
            type(model.model.diffusion_model).__qualname__,
        ]
        masks = {label: mask for label, mask in (("a", mask_a), ("b", mask_b)) if mask is not None}
        for label, mask in masks.items():
            data = mask.detach().float().cpu().contiguous()
            mask_hash = hashlib.sha256(data.numpy().tobytes()).hexdigest()
            identity.append((label, list(data.shape), mask_hash))
        if clip is not None:
            identity.append(library.text.encoder_identity(clip))
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        recipe = os.path.join(FOLDERS.first_dir("kilora"), f"library-{digest}.kmod")

        def tags(value):
            return [tag.strip() for tag in value.split(",") if tag.strip()]

        def select(tokens, args):
            scaled = {
                label: latent_mask(mask[:1] if mask.ndim >= 3 else mask, args["input"][:1])[0, 0]
                for label, mask in masks.items()
            }
            chosen = library.select(
                index,
                prompt,
                tokens,
                binding.family,
                count,
                clip=clip,
                seed=seed,
                min_relevance=min_relevance,
                include_tags=tags(include_tags),
                exclude_tags=tags(exclude_tags),
                **({"regions": scaled} if masks else {}),
            )
            if not chosen:
                self.last_report = sync_text + "\n" + text("library.none")
                save_recipe(
                    [],
                    None,
                    recipe,
                    "Library selection",
                    route_freeze=0.4,
                    axis_alpha="auto",
                    library=True,
                )
                return None
            selected = []
            skipped = []
            for item in chosen:
                entry = library.SelectedFile(item.path, item.sha256, strength)
                try:
                    path = loader._path(entry)
                    digest = index.hash_index.hash(path)
                    if digest != item.sha256:
                        raise ValueError(text("library.changed"))
                    selected.append(entry)
                except (OSError, ValueError) as exc:
                    skipped.append(
                        text("library.skipped", filename=os.path.basename(item.path), error=exc)
                    )
            chosen = [item for item in chosen if any(s.sha256 == item.sha256 for s in selected)]
            if not selected:
                self.last_report = "\n".join([sync_text, *skipped, text("library.failed")])
                save_recipe(
                    [],
                    None,
                    recipe,
                    "Library selection",
                    route_freeze=0.4,
                    axis_alpha="auto",
                    library=True,
                )
                return None
            choice, timing = loader.load(selected, model, binding, shapes)
            selected.sort(key=lambda c: (c.sha256, float(c.strength)))
            by_digest = {item.sha256: item for item in chosen}
            chain = []
            for c in selected:
                labels = by_digest[c.sha256].regions
                region = None
                if labels:
                    region = torch.stack([scaled[label] for label in labels]).amax(dim=0)[None]
                chain.append(
                    patching.ModEntry(
                        "lora", str(c.path), strength=c.strength, axis="creative", region=region
                    )
                )
            scratch = model.clone()
            scratch.model_options.pop("model_function_wrapper", None)
            if not callable(getattr(scratch, "set_injections", None)):
                raise TypeError(text("library.injections"))
            share_samples = {}

            def observe_share(shares, args):
                if shares.shape[-1] == len(selected):
                    mean = shares.detach().float().reshape(-1, len(selected)).mean(0)
                    timestep = args["timestep"]
                    step = (
                        float(timestep.detach().flatten()[0])
                        if isinstance(timestep, torch.Tensor)
                        else float(timestep)
                    )
                    total, count = share_samples.get(step, (0, 0))
                    share_samples[step] = total + mean, count + 1

            scratch, _ = patching.apply(
                scratch,
                chain,
                None,
                joint=True,
                bypass="on",
                freeze=0.4,
                axis_alpha="auto",
                alpha_clip=clip,
                joint_choice_override=choice,
                share_observer=observe_share,
                depth_steps=COMFY_DEPTH_STEPS,
            )
            recipe_notes = save_recipe(
                chain,
                None,
                recipe,
                "Library selection",
                route_freeze=0.4,
                axis_alpha="auto",
                library=True,
                hash_index=index.hash_index,
            )
            lines = [
                text(
                    "library.choice",
                    filename=os.path.basename(c.path),
                    score=c.score,
                    strength=strength,
                )
                for c in chosen
            ]
            self.last_report = "\n".join(
                [
                    sync_text,
                    *lines,
                    *skipped,
                    *[line for line in recipe_notes if "region was not saved" in line],
                    str(timing),
                    text("library.batch"),
                ]
            )

            def finish_report():
                if not share_samples:
                    return
                weights = torch.stack([total / count for total, count in share_samples.values()])
                weights = weights.mean(0).cpu().tolist()
                by_digest = {
                    entry.sha256: weight for entry, weight in zip(selected, weights, strict=True)
                }
                weighted_lines = [
                    line + text("library.mean_weight", weight=by_digest[item.sha256])
                    for line, item in zip(lines, chosen, strict=True)
                ]
                self.last_report = "\n".join(
                    [
                        sync_text,
                        *weighted_lines,
                        *skipped,
                        *[line for line in recipe_notes if "region was not saved" in line],
                        str(timing),
                        text("library.batch"),
                    ]
                )

            return _Choice(scratch, finish_report)

        base_function = getattr(model.model, "apply_model", denoiser)
        probe = StepZeroProbe(denoiser, base_function, blocks, groups, select, pass_args=True)
        installed = install(model, probe)
        report = sync_text + "\n" + text("library.pending")
        if self.last_report:
            report += "\n" + self.last_report
        return installed, report, recipe
