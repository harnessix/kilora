# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

from ..prepare import prepare_file, prepare_lokr, prepare_mix


def apply_device(model=None):
    device = getattr(model, "load_device", None) if model is not None else None
    if device is not None:
        return device
    try:
        from comfy.model_management import get_torch_device

        return get_torch_device()
    except (ImportError, AttributeError):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selected_files(roots, selection):
    roots = [os.path.abspath(root) for root in roots]
    names = [name.strip() for name in selection.splitlines() if name.strip()] or ["."]
    files = {}
    missing = []
    for name in names:
        found = False
        unsupported = False
        for root in roots:
            candidate = os.path.abspath(os.path.join(root, name))
            try:
                inside = os.path.commonpath((root, candidate)) == root
            except ValueError:
                inside = False
            if not inside or not os.path.exists(candidate):
                continue
            found = True
            if os.path.isfile(candidate) and candidate.lower().endswith(".safetensors"):
                files.setdefault(os.path.realpath(candidate), candidate)
            elif os.path.isfile(candidate):
                unsupported = True
            elif os.path.isdir(candidate):
                visited = set()
                for folder, dirs, filenames in os.walk(candidate, followlinks=True):
                    resolved = os.path.realpath(folder)
                    if resolved in visited:
                        dirs[:] = []
                        continue
                    visited.add(resolved)
                    dirs.sort()
                    for filename in sorted(filenames):
                        path = os.path.join(folder, filename)
                        if path.lower().endswith(".safetensors") and os.path.isfile(path):
                            files.setdefault(os.path.realpath(path), path)
        if not found:
            missing.append(name)
        elif unsupported:
            missing.append(f"{name} (unsupported file type)")
    return list(files.values()), missing


def display_path(path, roots):
    for root in roots:
        root = os.path.abspath(root)
        try:
            if os.path.commonpath((root, path)) == root:
                return os.path.relpath(path, root)
        except ValueError:
            continue
    return path


class KiLoRAPrepareLoRAs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"paths": ("STRING", {"default": "", "multiline": True})},
            "optional": {"model": ("MODEL",), "family": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "prepare"
    CATEGORY = "KiLoRA"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def prepare(self, paths, model=None, family=""):
        from . import patching
        from .nodes import FOLDERS

        started = time.monotonic()
        roots = FOLDERS.dirs("loras")
        files, missing = selected_files(roots, paths)
        lines = [f"Failed: {name}" for name in missing]
        device = apply_device(model)
        valid = []
        for selected in files:
            path = os.path.realpath(selected)
            label = display_path(selected, roots)
            try:
                new, current, skipped = prepare_lokr(path, detailed=True, device=device)
                valid.append((selected, label))
                if skipped:
                    lines.append(f"Skipped (current): {label} ({current} LoKr parts)")
                elif new:
                    lines.append(f"Prepared LoKr parts: {label} ({new})")
                elif current:
                    lines.append(f"LoKr parts current: {label} ({current})")
                else:
                    lines.append(f"Read file: {label} (no LoKr parts)")
            except Exception as exc:
                lines.append(f"Failed: {label}: {exc}")
        if model is not None and valid:
            from ..multi_import import LoRASource
            from ..presets import PRESETS

            try:
                binding = patching.detect_family(model)
                shape_model = patching.shape_model(patching.diffusers_shapes(model, binding))
                family = binding.family if binding.family in PRESETS else None
                sources = [
                    LoRASource(path=path, name=f"{i}-{os.path.splitext(os.path.basename(path))[0]}")
                    for i, (path, _) in enumerate(valid)
                ]
                prepare_mix(
                    sources,
                    shape_model,
                    family=family,
                    full_model=model.model.diffusion_model,
                    device=device,
                )
                lines.append(f"Prepared the selected ordered mix for this model on {device}.")
            except Exception as exc:
                lines.append(f"Mix preparation failed: {exc}")
        elif valid:
            for path, label in valid:
                try:
                    prepare_file(path, family=family or None, device=device)
                    lines.append(f"Prepared derived keys: {label}")
                except Exception as exc:
                    lines.append(f"Connect a model to prepare derived keys for {label}: {exc}")
        lines.append(f"Time: {time.monotonic() - started:.2f} s")
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report,)}
