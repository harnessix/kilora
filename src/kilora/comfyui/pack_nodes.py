# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path

from ..adapter import imported_schedule_fields, imported_settings
from ..lora_import import import_lora
from ..modfile import load_mod, read_manifest, save_mod
from ..modlist import LoadedMod
from ..packs import pack_from_mods, save_pack
from ..presets import pack_license
from ..sets import check_strength
from . import patching


def solved_mod(source, adapter, record, label, pairs, preview, *, overwrite):
    from .nodes import FOLDERS

    mod_id = f"{source.stem}-{label}" if record["mode"] == "joint" else source.stem
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", mod_id):
        raise ValueError("Mod ID must use 1 to 64 lowercase letters, digits, '.', '_' or '-'")
    destination = Path(FOLDERS.first_dir("kilora")) / "kilora-solved" / f"{mod_id}.kmod.safetensors"
    previews = []
    if preview is not None:
        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(
            np.clip(preview[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        ).save(buffer, format="PNG")
        previews.append(buffer.getvalue())
    if destination.exists() and not overwrite:
        existing = load_mod(destination)
        if existing.manifest.get("prompt_solve") != record or existing.previews != previews:
            digest = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]
            mod_id = f"{mod_id}-{digest}"
            if len(mod_id) > 64:
                raise ValueError(
                    "Mod ID must use 1 to 64 lowercase letters, digits, '.', '_' or '-'"
                )
            destination = destination.with_name(f"{mod_id}.kmod.safetensors")
            if destination.exists():
                existing = load_mod(destination)
                if existing.manifest.get("prompt_solve") != record or existing.previews != previews:
                    raise ValueError(
                        f"Output exists with different solve inputs or preview: {destination}"
                    )
                return destination
        else:
            return destination
    result = import_lora(
        source, patching.shape_model(adapter.shapes), family=adapter.family, experts=1
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        path = Path(temporary) / destination.name
        save_mod(
            path,
            [result.adapter_set],
            imported_settings(),
            sections=result.sections,
            previews=previews,
            id=mod_id,
            version="1.0.0",
            name=label,
            license=pack_license(adapter.family)[0],
            content_source="original",
            solve_source="prompt-set",
            base_model={"family": adapter.family},
            prompt_solve=record,
            trigger_words=list(dict.fromkeys(p.cue for p in pairs if p.cue)),
            **imported_schedule_fields(),
        )
        from .prompt_solve import _publish

        _publish(path, destination, overwrite=overwrite)
    return destination


class KiLoRAPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mods": ("STRING", {"default": "", "multiline": True}),
                "name": ("STRING", {"default": "my-pack"}),
                "strength": ("FLOAT", {"default": 0.7, "min": -4.0, "max": 4.0}),
                "depth": ("INT", {"default": -1, "min": -1}),
                "copy_mods": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "report")
    FUNCTION = "save"
    CATEGORY = "KiLoRA"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, mods, name, strength=0.7, depth=-1, copy_mods=False):
        from .nodes import FOLDERS

        check_strength(strength)
        if not re.fullmatch(r"[\w-]+", name):
            raise ValueError("name must contain only letters, numbers, underscores or dashes")
        if depth < -1:
            raise ValueError("depth must be -1 or greater")
        loaded, sources, lines = [], [], []
        for line in mods.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith(".safetensors") and not line.endswith(".kmod.safetensors"):
                continue
            source = Path(line)
            if not source.is_file():
                try:
                    source = Path(FOLDERS.path("kilora", line))
                except (OSError, KeyError) as exc:
                    raise FileNotFoundError(f"Mod file not found: {line}") from exc
            if not source.is_file():
                raise FileNotFoundError(f"Mod file not found: {line}")
            manifest = read_manifest(source)
            loaded.append(
                LoadedMod(
                    manifest["id"],
                    manifest["version"],
                    manifest["name"],
                    manifest,
                    [],
                    imported_settings(),
                    strength,
                    len(loaded),
                    manifest["name"],
                    path=str(source),
                )
            )
            sources.append(source)
            mod = load_mod(source)
            cue = ", ".join(manifest.get("trigger_words", []))
            lines.append(
                f"{manifest['id']} {manifest['version']}: cue={cue}; "
                f"preview={'yes' if mod.previews else 'no'}"
            )
        if not loaded:
            raise ValueError("Supply at least one mod file")
        if len({p.name for p in sources}) != len(sources):
            raise ValueError("Mod file names must be unique")
        root = Path(FOLDERS.first_dir("kilora"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.kpack.json"
        pack = pack_from_mods(name, loaded, None if depth == -1 else depth)
        save_pack(path, pack)
        if copy_mods:
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            for source in [*sources, path]:
                target = folder / source.name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (str(path), report)}
