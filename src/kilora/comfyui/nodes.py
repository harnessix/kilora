# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from ..adaptive_gate import GATE_MODES
from ..compute import AXES, PATH_SELECTS, RECORD_AXES, ROUTING_SCOPES, ROUTING_SELECTS
from ..i18n import text
from ..naming import (
    MOD_SUFFIX,
    MOD_SUFFIXES,
    PACK_SUFFIX,
    PACK_SUFFIXES,
    UPDATER_SUFFIX,
    UPDATER_SUFFIXES,
    mod_filename,
)
from ..packs import scan_library
from ..size_norm import SIZE_NORMS
from . import depth as depth_schedules
from . import dropin, live_write, patching, takeover
from . import help as help_texts
from . import settings as kilora_settings
from .image_solve import KiLoRASolveFromImages
from .layer_filter import KiLoRALayerFilter
from .library_node import KiLoRALoRALibrary
from .pack_nodes import KiLoRAPack
from .patching import ModChain, ModEntry
from .prepare_node import KiLoRAPrepareLoRAs
from .prompt_solve import KiLoRAPromptSet, KiLoRASolveFromPrompts
from .route_map import KiLoRARouteMap, KiLoRARouteRecord
from .trigger_words import KiLoRATriggerWords

SETS = "KILORA_SETS"
TIMELINE = "TIMELINE"
CATEGORY = "KiLoRA"
COMFY_RADIAL_STEP = 0.5
COMFY_DEPTH_STEPS = (COMFY_RADIAL_STEP, 1.0, 0.0)
MOD_EXTENSION = MOD_SUFFIX
PACK_EXTENSION = PACK_SUFFIX
UPDATER_EXTENSION = UPDATER_SUFFIX


_FILE_KINDS = {
    "mod": ("kind.mod", "ComfyUI/models/kilora"),
    "pack": ("kind.pack", "ComfyUI/models/kilora"),
    "lora": ("kind.lora", "ComfyUI/models/loras"),
}


def _file(folder: str, name: str, kind: str) -> str:
    kind_key, where = _FILE_KINDS[kind]
    label = text(kind_key)
    if not name:
        raise patching.KiLoRAError(text("err.no_file", kind=label, where=where))
    path = FOLDERS.path(folder, name)
    if not path or not os.path.exists(path):
        raise patching.KiLoRAError(text("err.not_found", kind=label, name=name, where=where))
    return path


def _strength(value: float, low: float = -2.0, high: float = 2.0) -> float:
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise patching.KiLoRAError(text("err.strength", low=f"{low:g}", high=f"{high:g}"))
    return v


def _lora_stack_items(lora_stack) -> list[tuple[str, float]]:

    if not lora_stack:
        return []
    rows = lora_stack
    if isinstance(rows, (tuple, list)) and rows and isinstance(rows[0], str):
        rows = [rows]
    items = []
    for row in rows:
        if not row:
            continue
        name = row[0]
        if not name or name == "None":
            continue
        model_strength = 1.0 if len(row) < 2 or row[1] is None else float(row[1])
        items.append((str(name), model_strength))
    return items


class Folders:
    def __init__(self) -> None:
        self._registered = False

    def _fp(self):
        from ..host import services

        comfy = services().comfy
        if comfy is not None and getattr(comfy, "folder_paths", None) is not None:
            folder_paths = comfy.folder_paths
        else:
            import folder_paths

        if not self._registered:
            if "kilora" not in folder_paths.folder_names_and_paths:
                folder_paths.folder_names_and_paths["kilora"] = (
                    [os.path.join(folder_paths.models_dir, "kilora")],
                    {".safetensors", ".json", *MOD_SUFFIXES, *PACK_SUFFIXES, *UPDATER_SUFFIXES},
                )
            self._registered = True
        return folder_paths

    def list(self, folder: str, extension: str | tuple[str, ...] | None = None) -> list[str]:
        names = self._fp().get_filename_list(folder)
        return [n for n in names if extension is None or n.lower().endswith(extension)]

    def path(self, folder: str, name: str) -> str:
        path = self._fp().get_full_path(folder, name)
        if path is None:
            raise FileNotFoundError(f"{name} is not in the {folder} model folder")
        return path

    def first_dir(self, folder: str) -> str:
        return self._fp().get_folder_paths(folder)[0]

    def dirs(self, folder: str) -> list[str]:
        return list(self._fp().get_folder_paths(folder))


FOLDERS = Folders()


def live_env() -> patching.LiveEnv:
    try:
        saves = os.path.join(FOLDERS.first_dir("kilora"), "saves")
        editors = tuple(FOLDERS.path("kilora", n) for n in FOLDERS.list("kilora", UPDATER_SUFFIXES))
    except Exception:
        return patching.LiveEnv()
    try:
        fp = FOLDERS._fp()
        dirs = tuple(fp.get_folder_paths("loras") + fp.get_folder_paths("kilora"))
    except (ImportError, AttributeError):
        try:
            dirs = (FOLDERS.first_dir("loras"), FOLDERS.first_dir("kilora"))
        except Exception:
            dirs = ()
    return patching.LiveEnv(saves, editors, dirs)


def _live_revisions(sets: Any = None) -> str:

    from ..live import SESSIONS
    from ..modfile import read_manifest

    if sets is None:
        return ";".join(sorted(f"{s.mod_id}:{id(s)}:{s.revision}" for s in list(SESSIONS.values())))
    parts = []
    for entry in _chain(sets):
        if entry.kind != "mod":
            continue
        try:
            session = SESSIONS.get(read_manifest(entry.path).get("digest", ""))
        except Exception:
            continue
        if session is not None:
            parts.append(f"{session.mod_id}:{session.revision}")
    return ";".join(parts)


def _chain(sets: Any) -> ModChain:
    return sets if isinstance(sets, ModChain) else ModChain()


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip(".-")[:64]


class KiLoRASelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (FOLDERS.list("kilora", MOD_SUFFIXES),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Signed strength. Negative values push away from the mod.",
                    },
                ),
                "prior": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "How much the router prefers this mod's experts.",
                    },
                ),
            },
            "optional": {
                "sets": (SETS,),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, file, strength, prior, sets=None, enabled=True):
        entry = ModEntry("mod", _file("kilora", file, "mod"), _strength(strength), prior, enabled)
        return (_chain(sets).add(entry),)


def _segs_mask(segs):

    import torch

    if not isinstance(segs, (tuple, list)) or len(segs) < 2:
        raise patching.KiLoRAError(text("err.segs"))
    shape, segments = segs[0], segs[1]
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) < 2
        or not isinstance(segments, (tuple, list))
    ):
        raise patching.KiLoRAError(text("err.segs"))
    height, width = int(shape[0]), int(shape[1])
    if height < 0 or width < 0:
        raise patching.KiLoRAError(text("err.segs"))
    canvas = torch.zeros((height, width), dtype=torch.uint8)
    for seg in segments:
        cropped = getattr(seg, "cropped_mask", None)
        region = getattr(seg, "crop_region", None)
        if cropped is None or region is None:
            continue
        if not isinstance(cropped, torch.Tensor):
            cropped = torch.as_tensor(cropped)
        cropped = cropped.detach().cpu().float()
        while cropped.dim() > 2 and cropped.shape[0] == 1:
            cropped = cropped[0]
        if cropped.dim() != 2:
            raise patching.KiLoRAError(text("err.segs"))
        x1, y1, x2, y2 = (int(value) for value in region)
        if y2 < y1 or x2 < x1 or y1 < 0 or x1 < 0 or y2 > height or x2 > width:
            raise patching.KiLoRAError(text("err.segs"))
        if tuple(cropped.shape) != (y2 - y1, x2 - x1):
            raise patching.KiLoRAError(text("err.segs"))
        canvas[y1:y2, x1:x2] |= (cropped * 255).to(torch.uint8)
    return canvas.float() / 255.0


class KiLoRALayerConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",), "layer": (list(AXES),)},
            "optional": {
                "mask": ("MASK",),
                "alpha_source": (["mask", "attention"], {"default": "mask"}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 1024}),
                "crop": (["off", "auto"], {"default": "off"}),
                "batch": ("BOOLEAN", {"default": False}),
                "compose": (["separate", "single"], {"default": "separate"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "layer"
    CATEGORY = CATEGORY

    def layer(
        self,
        conditioning,
        layer,
        mask=None,
        alpha_source="mask",
        feather=16,
        crop="off",
        batch=False,
        compose="separate",
    ):
        import comfy.hooks as hooks

        if mask is not None and feather:
            import torch.nn.functional as F

            mask = mask if mask.ndim >= 3 else mask.unsqueeze(0)
            radius = int(feather)
            mask = F.avg_pool2d(
                F.pad(mask.unsqueeze(1), (radius,) * 4, mode="replicate"),
                2 * radius + 1,
                stride=1,
            ).squeeze(1)
        _warn_mention_layer(conditioning, layer)
        result = hooks.set_mask_for_conditioning(conditioning, mask, "default", 1.0)
        result = [
            [
                value,
                {
                    **meta,
                    "kilora_layer": layer,
                    "kilora_alpha_source": alpha_source,
                    "kilora_crop": crop,
                    "kilora_layer_batch": bool(batch),
                    "kilora_compose": compose,
                },
            ]
            for value, meta in result
        ]
        return (result,)


def _warn_mention_layer(conditioning, layer) -> None:
    for _, meta in conditioning or ():
        if not isinstance(meta, dict) or not meta.get("kilora_mentions"):
            continue
        bound = meta.get("kilora_mention_layer")
        if bound is not None and bound != layer:
            logging.getLogger("kilora").warning(
                text("mention.layer_differs", mention=bound, layer=layer)
            )
            return


MENTION_STRENGTH = 0.7
_MENTION = re.compile(
    r"(?<![\w@])@(?P<name>[\w\-]+(?:\.[\w\-]+)*)"
    r"(?::(?P<strength>[+-]?(?:\d+(?:\.\d*)?|\.\d+)))?(?![\w@])"
)


@dataclass(frozen=True)
class Mention:
    name: str
    strength: float
    span: tuple[int, int]


def parse_mentions(value: str) -> list[Mention]:
    out = []
    for match in _MENTION.finditer(value):
        raw = match.group("strength")
        strength = MENTION_STRENGTH if raw is None else _strength(float(raw), -4.0, 4.0)
        out.append(Mention(match.group("name"), strength, match.span()))
    return out


def strip_mentions(value: str, mentions: list[Mention]) -> str:
    if not mentions:
        return value
    out = ""
    last = 0
    for mention in mentions:
        start, end = mention.span
        out += value[last:start]
        last = end
        out = out.rstrip(" \t")
        rest = value[last:]
        stripped = rest.lstrip(" \t")
        if stripped[:1] in (",", ";") and (not out or out[-1] in ",;(\n"):
            stripped = stripped[1:].lstrip(" \t")
        last = len(value) - len(stripped)
        if not stripped.strip():
            out = out.rstrip(",; \t")
        elif out and stripped and out[-1] != "\n" and stripped[0] not in ",.;:!?)\n":
            out += " "
    return out + value[last:]


def _mention_files(names: list[str]) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for name in names:
        stem = os.path.splitext(os.path.basename(name.replace("\\", "/")))[0]
        files.setdefault(stem.lower(), []).append(name)
    return files


_SOLVED_MENTIONS: dict[tuple[str, int, int], str | None] = {}


def _solved_mention(path: str) -> str | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _SOLVED_MENTIONS:
        name = None
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as file:
                record = json.loads((file.metadata() or {}).get("kilora_directions", "{}"))
            solve = record.get("prompt_solve")
            concept = record.get("concept")
            if isinstance(solve, dict) and solve.get("mode") == "single" and concept:
                name = str(concept)
        except Exception:
            name = None
        _SOLVED_MENTIONS[key] = name
    return _SOLVED_MENTIONS[key]


def _solved_mention_files(names: list[str]) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for name in names:
        if not name.replace("\\", "/").startswith("kilora-solved/"):
            continue
        try:
            path = FOLDERS.path("loras", name)
        except FileNotFoundError:
            continue
        label = _solved_mention(path)
        if label:
            files.setdefault(label.lower(), []).append(name)
    return files


def _mention_file(
    name: str, files: dict[str, list[str]], solved: dict[str, list[str]] | None = None
) -> str:
    import difflib

    found = files.get(name.lower())
    if found is None and solved:
        found = solved.get(name.lower())
    if found is None:
        stems = {key: os.path.splitext(os.path.basename(f[0]))[0] for key, f in files.items()}
        close = difflib.get_close_matches(name.lower(), list(stems), n=3, cutoff=0.4)
        closest = ", ".join(stems[c] for c in close) if close else text("mention.none_close")
        raise patching.KiLoRAError(text("err.mention_unknown", name=name, closest=closest))
    if len(found) > 1:
        raise patching.KiLoRAError(text("err.mention_ambiguous", name=name, files=", ".join(found)))
    return found[0]


class KiLoRAMentionPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "sets": (SETS,),
                "layer": (["global", *AXES], {"default": "global"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", SETS, "STRING")
    RETURN_NAMES = ("conditioning", "sets", "report")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, text, sets=None, layer="global"):
        from ..i18n import text as _text
        from ..library.bake import find_baked
        from .trigger_words import trigger_words_of

        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint "
                "loader node your checkpoint does not contain a valid clip or text "
                "encoder model."
            )
        if layer not in ("global", *AXES):
            raise patching.KiLoRAError(f"unknown layer {layer!r}; use global or one of {AXES}")
        user_text = text
        mentions = parse_mentions(user_text)
        chain = _chain(sets)
        if not mentions:
            tokens = clip.tokenize(user_text)
            return (clip.encode_from_tokens_scheduled(tokens), chain, "")
        listed = FOLDERS.list("loras", ".safetensors")
        files = _mention_files(listed)
        solved = None
        axis = "auto" if layer == "global" else layer
        records, lines, added = [], [], []
        for mention in mentions:
            if mention.name.lower() not in files and solved is None:
                solved = _solved_mention_files(listed)
            source = _file("loras", _mention_file(mention.name, files, solved), "lora")
            triggers = trigger_words_of(source)
            baked, count = find_baked(source)
            name = os.path.splitext(os.path.basename(source))[0]
            if baked is not None:
                lines.append(_text("mention.baked", name=name, file=str(baked)))
            else:
                added.extend(w for w in triggers if w not in added)
                key = "mention.several" if count > 1 else "mention.fallback"
                lines.append(_text(key, name=name, words=", ".join(triggers) or "-"))
            path = str(baked) if baked is not None else source
            chain = chain.add(ModEntry("lora", path, mention.strength, 1.0, True, axis=axis))
            records.append(
                {
                    "name": name,
                    "path": source,
                    "strength": mention.strength,
                    "trigger_words": list(triggers),
                    "span": list(mention.span),
                    "baked": baked is not None,
                    "baked_path": None if baked is None else str(baked),
                }
            )
        clean = strip_mentions(user_text, mentions)
        encoder_text = ", ".join(p for p in (", ".join(added), clean) if p.strip())
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(encoder_text))
        conditioning = [
            [value, {**meta, "kilora_mentions": records, "kilora_mention_layer": layer}]
            for value, meta in conditioning
        ]
        return (conditioning, chain, "\n".join(lines))


class KiLoRARegion:
    APPLY_TO = ("whole list", "last mod")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sets": (SETS,),
                "mask": ("MASK",),
            },
            "optional": {
                "apply_to": (list(cls.APPLY_TO), {"default": "whole list"}),
                "caption": ("CONDITIONING",),
                "segs": ("SEGS",),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "region"
    CATEGORY = CATEGORY

    def region(self, sets, mask, apply_to="whole list", caption=None, segs=None):
        import dataclasses

        chain = _chain(sets)
        if not chain:
            raise patching.KiLoRAError(text("err.region_empty"))
        if segs is not None:
            mask = _segs_mask(segs)
        m = mask if mask.dim() == 3 else mask.reshape(-1, *mask.shape[-2:])
        start = 0 if apply_to == "whole list" else len(chain) - 1
        entries = [
            dataclasses.replace(e, region=m, region_caption=caption) if i >= start else e
            for i, e in enumerate(chain)
        ]
        return (ModChain(entries),)


class KiLoRATokenRoles:
    APPLY_TO = ("whole list", "last mod")

    @classmethod
    def INPUT_TYPES(cls):
        from ..roles import POLICIES

        return {
            "required": {
                "sets": (SETS,),
                "roles": (list(POLICIES), {"default": "auto"}),
            },
            "optional": {
                "apply_to": (list(cls.APPLY_TO), {"default": "whole list"}),
                "custom": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "set_roles"
    CATEGORY = CATEGORY

    def set_roles(self, sets, roles="auto", apply_to="whole list", custom=""):
        import dataclasses

        from ..roles import parse_policy

        chain = _chain(sets)
        if not chain:
            raise patching.KiLoRAError(text("err.roles_empty"))
        try:
            policy = parse_policy(custom if custom.strip() else roles)
        except ValueError as err:
            raise patching.KiLoRAError(str(err)) from None
        start = 0 if apply_to == "whole list" else len(chain) - 1
        entries = [
            dataclasses.replace(e, roles=policy) if i >= start else e for i, e in enumerate(chain)
        ]
        return (ModChain(entries),)


class KiLoRATwoPanels:
    @classmethod
    def INPUT_TYPES(cls):
        from ..roles import PANEL_SIDES

        return {
            "required": {
                "sets": (SETS,),
                "reference_side": (list(PANEL_SIDES), {"default": "left"}),
            },
            "optional": {
                "split": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "panels"
    CATEGORY = CATEGORY

    def panels(self, sets, reference_side="left", split=0.5):
        import dataclasses

        from ..roles import check_panels

        chain = _chain(sets)
        if not chain:
            raise patching.KiLoRAError(text("err.panels_empty"))
        try:
            panels = check_panels(reference_side, split)
        except ValueError as err:
            raise patching.KiLoRAError(str(err)) from None
        return (ModChain(dataclasses.replace(e, panels=panels) for e in chain),)


class KiLoRARegionAttention:
    @classmethod
    def INPUT_TYPES(cls):
        from ..region_attention import MODES

        return {
            "required": {
                "model": ("MODEL",),
                "mode": (list(MODES), {"default": "off"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "until": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "outside_region": ("BOOLEAN", {"default": False}),
                "start": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "region_captions": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "attend"
    CATEGORY = CATEGORY

    def attend(
        self,
        model,
        mode,
        strength=0.5,
        until=0.6,
        outside_region=False,
        start=0.3,
        region_captions=False,
    ):
        from ..region_attention import RegionAttention

        config = RegionAttention(
            mode,
            float(strength),
            float(until),
            bool(outside_region),
            float(start),
            bool(region_captions),
        )
        out, n = patching.apply_region_attention(model, config)
        if mode == "off" and not region_captions:
            return out, text("msg.region_attention_off")
        if n == 0:
            return out, text("msg.region_attention_none")
        report = text(
            "msg.region_attention_on", mode=mode, regions=n, start=f"{start:g}", until=f"{until:g}"
        )
        if region_captions:
            captions = len(patching.region_captions(model))
            report += " " + text("msg.region_attention_captions", captions=captions)
        return out, report


class KiLoRARegionIsolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "until": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "sets": (SETS,),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "isolate"
    CATEGORY = CATEGORY

    def isolate(self, model, strength, until=0.6, feather=32, sets=None):
        from ..region_isolation import RegionIsolation

        config = RegionIsolation(float(strength), float(until), int(feather))
        chain = _chain(sets) if sets is not None else None
        out, n = patching.apply_region_isolation(model, config, chain=chain)
        if strength <= 0:
            return out, text("msg.region_isolation_off")
        if n == 0:
            return out, text("msg.region_attention_none")
        return out, text(
            "msg.region_isolation_on", regions=n, strength=f"{strength:g}", until=f"{until:g}"
        )


class KiLoRATimeline:
    @classmethod
    def INPUT_TYPES(cls):
        from ..timeline import CLOCKS, INTERPOLATIONS, QUANTITIES

        return {
            "required": {
                "target": ("STRING", {"default": "all"}),
                "quantity": (list(QUANTITIES), {"default": "strength"}),
                "curve": ("STRING", {"default": "0:1, 1:1"}),
            },
            "optional": {
                "clock": (list(CLOCKS), {"default": "progress"}),
                "interpolation": (list(INTERPOLATIONS), {"default": "linear"}),
                "timeline": (TIMELINE,),
                "more_rows": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (TIMELINE, "IMAGE")
    RETURN_NAMES = ("timeline", "preview")
    FUNCTION = "row"
    CATEGORY = CATEGORY

    def row(
        self,
        target,
        quantity,
        curve,
        clock="progress",
        interpolation="linear",
        timeline=None,
        more_rows="",
    ):
        from ..timeline import Row, Timeline, TimelineError, check_row, parse_rows, preview

        rows = [Row(target.strip() or "all", quantity, curve, clock, interpolation)]
        try:
            rows += parse_rows(more_rows)
            for r in rows:
                check_row(r)
        except TimelineError as err:
            raise patching.KiLoRAError(str(err)) from None
        out = Timeline(timeline or ()).add(*rows)
        return out, preview(out)


class KiLoRALiveExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mod": ("STRING", {"default": ""}),
                "include_learned_state": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include what the mod learned from the images you picked",
                    },
                ),
                "new_id": ("STRING", {"default": ""}),
                "format": (["kmod", "lora"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def export(self, model, mod, include_learned_state, new_id, format="kmod"):
        from ..live import LEARNED_STATE_WARNING
        from ..live_imports import ImportedLiveSession

        sessions = [
            s for s in patching.sessions_of(model) if not mod.strip() or s.mod_id == mod.strip()
        ]
        if len(sessions) > 1:
            raise patching.KiLoRAError(text("err.choose_live_session"))
        if not sessions:
            raise patching.KiLoRAError(f"no live mod {mod!r} is loaded")
        session = sessions[0]
        family = (session.mod.manifest.get("base_model") or {}).get("family")
        if isinstance(session, ImportedLiveSession):
            name = f"{session.mod_id}-share." + ("kmod" if format == "kmod" else "safetensors")
            path = os.path.join(FOLDERS.first_dir("kilora"), "exports", name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            note = session.export(path, format=format, include_sources=include_learned_state)
            report = f"Saved exports/{name}."
            if format == "lora":
                report += f" {note['note']} Relative SVD residuals: {note['relative_svd_residual']}"
            return {"ui": {"text": [report]}, "result": (report,)}
        if format != "kmod":
            raise patching.KiLoRAError("plain LoRA export requires an imported live session")
        if include_learned_state:
            target_id = _safe_id(new_id)
            if not target_id:
                raise patching.KiLoRAError("set new_id for a mod with a learned state")
        else:
            target_id = f"{session.mod_id}-share"
        name = mod_filename(target_id, family)
        path = os.path.join(FOLDERS.first_dir("kilora"), "exports", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        session.export(
            path,
            include_learned_state=include_learned_state,
            new_id=target_id if include_learned_state else None,
        )
        report = f"Saved exports/{name}."
        if include_learned_state:
            report += f" Warning: {LEARNED_STATE_WARNING}"
        else:
            report += " It has the state that shipped with the mod, not what it learned here."
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRAModInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file": (FOLDERS.list("kilora", MOD_SUFFIXES),)}}

    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("strength", "trigger_words", "example_prompt", "depth", "info")
    FUNCTION = "info"
    CATEGORY = CATEGORY

    def info(self, file):
        from ..card import read_card

        card = read_card(_file("kilora", file, "mod"))
        strength = float(card["strength"]["recommended"])
        words = ", ".join(card["trigger"])
        prompts = card["example_prompts"]
        depth = int(card["depth"] if card["depth"] is not None else -1)
        report = card_text(card)
        result = (strength, words, prompts[0] if prompts else "", depth, report)
        return {"ui": {"text": [report]}, "result": result}


def card_text(card: dict) -> str:
    s = card["strength"]
    lines = [f"{card['name']} ({card['id']} {card.get('version') or ''})".rstrip()]
    if card.get("description"):
        lines.append(str(card["description"]))
    lines.append(text("card.kind_live" if card.get("kind") == "live" else "card.kind_static"))
    words = ", ".join(card["trigger"])
    lines.append(text("card.trigger", value=words) if words else text("card.no_trigger"))
    lines.append(
        text(
            "card.strength",
            value=f"{s['recommended']:g}",
            low=f"{s['min']:g}",
            high=f"{s['max']:g}",
        )
    )
    lines.append(text("card.depth", value=card["depth"]))
    lines.append(text("card.hooks", count=card["hooks"]["count"]))
    lines.append(text("card.base", value=card["base_model"]["family"]))
    lines.append(text("card.license", value=card["license"]))
    source = card.get("data_source")
    if source:
        lines.append(
            text("card.data", images=source.get("images", "?"), name=source.get("name", ""))
        )
        if source.get("generators"):
            lines.append(text("card.data_generated", value=", ".join(source["generators"])))
        if source.get("license"):
            lines.append(text("card.data_license", value=source["license"]))
    else:
        lines.append(text("card.data_missing"))
    return "\n".join(lines)


@dataclass(frozen=True)
class SizedLoRAEntry(ModEntry):
    size_norm: str = "weight"


def size_normed_chain(
    model, chain, every_lora: bool = False, radial_budget: float = 0.0
) -> tuple[ModChain, list[str]]:

    from dataclasses import fields

    from ..multi_import import lora_file_size
    from ..presets import PRESETS
    from ..size_norm import applied_scale, base_key, size_note, size_scale

    picked = [
        e
        for e in chain
        if e.kind == "lora"
        and e.enabled
        and (every_lora or getattr(e, "size_norm", "off") == "weight")
    ]
    if not picked:
        return ModChain(chain), []
    binding = patching.detect_family(model)
    shapes = patching.diffusers_shapes(model, binding)
    tree = patching.shape_model(shapes)
    full = model.model.diffusion_model
    family = binding.family if binding.family in PRESETS else None
    weights = patching.host_weights(model, shapes)
    norms, base_id = base_key(weights)
    names = [f.name for f in fields(ModEntry)]
    scales: dict[str, float] = {}
    lines = []
    out = []
    for e in chain:
        if not any(e is p for p in picked):
            out.append(e)
            continue
        if e.path not in scales:
            size = lora_file_size(
                e.path,
                tree,
                family=family,
                full_model=full,
                base_weight=weights.get,
                base_norms=norms,
                base_id=base_id,
            )
            c = size_scale(size)
            s, rule = applied_scale(c)
            scales[e.path] = s
            lines.append(size_note(os.path.basename(e.path), size, c, s, rule))
        values = {n: getattr(e, n) for n in names}
        values["strength"] = float(e.strength) * scales[e.path]
        out.append(ModEntry(**values))
    return ModChain(out), lines


class KiLoRAImportLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (FOLDERS.list("loras", ".safetensors"),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "The LoRA weight you would use in a LoRA loader.",
                    },
                ),
            },
            "optional": {
                "sets": (SETS,),
                "enabled": ("BOOLEAN", {"default": True}),
                "lora_stack": ("LORA_STACK",),
                "axis": (["auto", *RECORD_AXES], {"default": "auto"}),
                "size_norm": (list(SIZE_NORMS), {"default": "off"}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(
        self,
        lora_name,
        strength,
        sets=None,
        enabled=True,
        lora_stack=None,
        axis="auto",
        size_norm="off",
        start=0.0,
        end=1.0,
    ):

        chain = _chain(sets)
        items = _lora_stack_items(lora_stack)
        if not items:
            items = [(lora_name, strength)]
        for name, model_strength in items:
            if axis not in (*RECORD_AXES, "auto"):
                raise patching.KiLoRAError(
                    f"unknown axis {axis!r}; use auto or one of {RECORD_AXES}"
                )
            if size_norm not in SIZE_NORMS:
                raise patching.KiLoRAError(f"unknown size_norm {size_norm!r}; use off or weight")
            entry_type = SizedLoRAEntry if size_norm == "weight" else ModEntry
            extra = {"size_norm": size_norm} if size_norm == "weight" else {}
            entry = entry_type(
                "lora",
                _file("loras", name, "lora"),
                _strength(model_strength, -4.0, 4.0),
                1.0,
                enabled,
                axis=axis,
                start=float(start),
                end=float(end),
                **extra,
            )
            chain = chain.add(entry)
        return (chain,)


class KiLoRADirectionNodes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "chunk": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "sets": (SETS,),
                "library_path": ("STRING", {"default": ""}),
                "fine": ("BOOLEAN", {"default": False}),
                "candidates": ("INT", {"default": 16, "min": 1, "max": 4096}),
                "direction_file": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (SETS,)
    RETURN_NAMES = ("sets",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(
        self,
        strength,
        chunk,
        sets=None,
        library_path="",
        fine=False,
        candidates=16,
        direction_file="",
    ):
        from ..library.index import LibraryIndex

        if direction_file.strip():
            path = os.path.abspath(os.path.expanduser(direction_file.strip()))
        else:
            path = str(LibraryIndex.open(library_path or None).path)
        return (
            _chain(sets).add(
                ModEntry(
                    "direction_file" if direction_file.strip() else "direction",
                    path,
                    _strength(strength),
                    float(chunk),
                    fine=bool(fine),
                    candidates=int(candidates),
                )
            ),
        )


class KiLoRAPalette:
    @classmethod
    def INPUT_TYPES(cls):
        from ..palette import list_palettes

        packs = [n for n in FOLDERS.list("kilora", PACK_SUFFIXES)]
        return {
            "required": {
                "palette": (list_palettes() + packs,),
                "shuffle": ("BOOLEAN", {"default": False}),
                "pick_min": ("INT", {"default": 2, "min": 1, "max": 64}),
                "pick_max": ("INT", {"default": 3, "min": 1, "max": 64}),
                "strength_min": ("FLOAT", {"default": 0.6, "min": -4.0, "max": 4.0, "step": 0.05}),
                "strength_max": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True},
                ),
            },
            "optional": {
                "sets": (SETS,),
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "mode": (["random", "learn", "learn_embedding"],),
                "reset_state": ("BOOLEAN", {"default": False}),
                "reward": (["none", "keep", "discard"],),
                "embedding_model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (SETS, "STRING", "KILORA_SETTINGS", "STRING")
    RETURN_NAMES = ("sets", "trigger_words", "settings", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        palette,
        shuffle=False,
        pick_min=2,
        pick_max=3,
        strength_min=0.6,
        strength_max=1.0,
        seed=0,
        sets=None,
        mask_a=None,
        mask_b=None,
        mode="random",
        reset_state=False,
        reward="none",
        embedding_model_path="",
    ):
        from ..key_cache import file_sha256
        from ..palette import (
            load_palette,
            load_shuffle_state,
            palette_dir,
            reset_shuffle_state,
            reward_shuffle,
            save_shuffle_state,
        )
        from ..palette import resolve as resolve_palette
        from ..palette import shuffle as shuffle_entries

        if palette.endswith(PACK_SUFFIXES):
            path = _file("kilora", palette, "pack")
        else:
            path = os.path.join(palette_dir(), palette)
            if not os.path.isfile(path):
                raise patching.KiLoRAError(f"palette {palette} not found in {palette_dir()}")
        data = load_palette(path)

        def find(kind, name):
            try:
                return FOLDERS.path("loras" if kind == "lora" else "kilora", name)
            except FileNotFoundError:
                return None

        def library():
            files = [FOLDERS.path("kilora", n) for n in FOLDERS.list("kilora", MOD_SUFFIXES)]
            return scan_library(files)

        found, problems = resolve_palette(data, find, file_sha256, library)
        lines = [f"Palette {data.name}: {len(found)} of {len(data.entries)} entries."]
        lines += problems
        if reset_state:
            reset_shuffle_state(path)
        if shuffle:
            state = load_shuffle_state(path) if mode in ("learn", "learn_embedding") else None
            embeddings = {}
            if mode == "learn_embedding" and embedding_model_path.strip():
                embeddings, notices = _palette_embeddings(found, embedding_model_path)
                lines.extend(notices)
            elif mode == "learn_embedding":
                lines.append(text("palette.embedding_empty"))
            found = shuffle_entries(
                found,
                seed,
                (pick_min, pick_max),
                (strength_min, strength_max),
                mode=mode,
                state=state,
                embeddings=embeddings,
            )
            if mode in ("learn", "learn_embedding") and reward != "none":
                reward_shuffle(
                    state,
                    found,
                    (strength_min, strength_max),
                    reward,
                    embeddings=embeddings if mode == "learn_embedding" else None,
                )
                save_shuffle_state(path, state)
                lines.append(f"Shuffle feedback recorded: {reward}.")
            picks = ", ".join(f"{r.entry.label} {r.entry.strength:g}" for r in found)
            lines.append(f"Shuffle (seed {seed}) picked {len(found)}: {picks}.")
            lines.append("Connect sets to KiLoRA Save Palette to keep this pick.")
        masks = {"a": mask_a, "b": mask_b}
        entries, words = [], []
        for r in found:
            e = r.entry
            region = masks.get(e.region) if e.region else None
            if e.region and region is None:
                lines.append(
                    f"{e.label}: its region {e.region} needs mask_{e.region}; whole image."
                )
            entries.append(
                ModEntry(
                    e.kind,
                    r.path,
                    e.strength,
                    1.0,
                    e.enabled,
                    region=region,
                    roles=e.roles,
                    axis=e.axis,
                )
            )
            if e.enabled:
                words += [w for w in e.triggers if w not in words]
        settings = _palette_settings(data)
        return (_chain(sets).add(*entries), ", ".join(words), settings, "\n".join(lines))


def _palette_embeddings(entries, model_path):
    import hashlib
    from pathlib import Path

    from safetensors import SafetensorError

    from ..library import LibraryIndex

    result, notices = {}, []
    try:
        index = LibraryIndex.open()
        index.sync(FOLDERS.dirs("loras"), reuse_text=True)
        model = Path(model_path).expanduser()
        if model.is_file():
            model = model.parent
        digest = hashlib.sha256()
        for artifact in (
            index.path / "basis.json",
            index.path / "basis.safetensors",
            model / "config.json",
            model / "model.safetensors",
        ):
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        model_id = digest.hexdigest()
        families = {}
        for item in entries:
            if item.entry.kind != "lora":
                continue
            file_id = index.paths.get(os.path.realpath(os.path.expanduser(item.path)))
            record = index.records.get(file_id, {})
            family = record.get("family")
            if not family:
                continue
            if family not in families:
                try:
                    families[family] = index.embedding_inputs(family, model_path)
                except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
                    families[family] = {}
                    notices.append(
                        text("palette.embedding_family_unavailable", family=family, reason=str(exc))
                    )
            vector = families[family].get(file_id)
            if (
                vector is not None
                and vector.ndim == 1
                and vector.numel()
                and vector.isfinite().all()
            ):
                key = hashlib.sha256(f"{model_id}/{family}".encode()).hexdigest()
                result[item.path] = (key, vector.tolist())
    except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
        notices.append(text("palette.embedding_unavailable", reason=str(exc)))
    return result, notices


def _palette_settings(data) -> dict:
    from .settings import DEFAULTS, resolve_settings

    given = {k: v for k, v in data.settings.items() if k in DEFAULTS}
    if data.depth is not None and "depth" not in given:
        given["depth"] = data.depth
    return resolve_settings(given)


class KiLoRASuggestions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"sets": (SETS,)},
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "suggest_by": (["co_usage", "similar", "complementary", "near_liked"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("suggestions",)
    FUNCTION = "show"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def show(self, sets, model_path="", suggest_by="co_usage"):
        from .. import usage_history
        from ..errors import KiLoRAError
        from ..key_cache import file_sha256
        from ..palette import list_palettes, palette_dir
        from ..suggestions import suggest

        index = None
        index_error = None
        if suggest_by != "co_usage" and model_path.strip():
            try:
                from ..library import LibraryIndex

                index = LibraryIndex.open()
                index.sync(FOLDERS.dirs("loras"), reuse_text=True)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                index_error = exc

        def digest(path):
            indexed = index.paths.get(os.path.realpath(os.path.expanduser(path))) if index else None
            return indexed if indexed is not None else file_sha256(path)

        current = {digest(e.path) for e in _chain(sets) if e.enabled and os.path.isfile(e.path)}
        installed = {}
        for name in FOLDERS.list("loras", ".safetensors"):
            try:
                path = FOLDERS.path("loras", name)
                if os.path.isfile(path):
                    installed[name] = digest(path)
            except (OSError, FileNotFoundError):
                continue
        singles, pairs = usage_history.counts()
        sources = [("palette", os.path.join(palette_dir(), name)) for name in list_palettes()]
        for name in FOLDERS.list("kilora", MOD_SUFFIXES):
            try:
                sources.append(("recipe", FOLDERS.path("kilora", name)))
            except (OSError, FileNotFoundError):
                continue
        for kind, path in sources:
            try:
                one_singles, one_pairs = usage_history.counts(
                    palettes=[path] if kind == "palette" else (),
                    recipes=[path] if kind == "recipe" else (),
                    history=False,
                )
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, KiLoRAError):
                continue
            singles.update(one_singles)
            pairs.update(one_pairs)
        results = suggest(current, installed, singles, pairs)
        notices = []
        if suggest_by != "co_usage":
            from safetensors import SafetensorError

            from ..palette import load_palette, load_shuffle_state, shuffle_entry_key
            from ..suggestions import embedding_suggest

            try:
                if not model_path.strip():
                    raise ValueError("empty model path")
                if index_error is not None:
                    raise ValueError(str(index_error))
                if index is None:
                    raise ValueError("library index unavailable")
                families = {index.records[h]["family"] for h in current if h in index.records}
                if len(families) != 1 or any(h not in index.records for h in current):
                    raise ValueError("current entries have no common indexed family")
                vectors = index.embedding_inputs(next(iter(families)), model_path)
                query = sorted(current)
                mode = suggest_by
                if mode == "near_liked":
                    liked = []
                    for kind, path in sources:
                        if kind != "palette":
                            continue
                        try:
                            palette = load_palette(path)
                            state = load_shuffle_state(path)
                            for entry in palette.entries:
                                count = state.counts.get(shuffle_entry_key(entry), [[0, 0]])[0][0]
                                if entry.kind == "lora" and entry.sha256 and count:
                                    liked.extend([entry.sha256] * count)
                        except (OSError, ValueError, KeyError, TypeError, KiLoRAError):
                            continue
                    if liked:
                        query = liked
                    else:
                        notices.append(text("suggestions.no_likes"))
                        mode = "similar"
                results, missing = embedding_suggest(
                    current, installed, singles, pairs, vectors, query, mode
                )
                if missing:
                    notices.append(text("suggestions.missing"))
            except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
                notices.append(text("suggestions.fallback", reason=str(exc)))
        report = "\n".join(notices + [f"{name}: {score:.3f}" for name, score in results])
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRASavePalette:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sets": (SETS,),
                "name": ("STRING", {"default": "my palette"}),
                "filename": ("STRING", {"default": "my-palette"}),
            },
            "optional": {"settings": ("KILORA_SETTINGS",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, sets, name, filename, settings=None):
        from ..key_cache import file_sha256
        from ..palette import (
            PALETTE_SUFFIX,
            REGION_LABELS,
            Palette,
            PaletteEntry,
            palette_dir,
            save_palette,
        )
        from .settings import resolve_settings
        from .trigger_words import trigger_words_of

        labels: dict[int, str] = {}
        entries, notes = [], []
        for e in _chain(sets):
            label = None
            if e.region is not None:
                if id(e.region) not in labels and len(labels) < len(REGION_LABELS):
                    labels[id(e.region)] = REGION_LABELS[len(labels)]
                label = labels.get(id(e.region))
                if label is None:
                    notes.append(f"{os.path.basename(e.path)}: only two regions are saved.")
            entries.append(
                PaletteEntry(
                    kind=e.kind,
                    name=os.path.basename(e.path),
                    sha256=file_sha256(e.path),
                    strength=float(e.strength),
                    enabled=bool(e.enabled),
                    roles=e.roles,
                    region=label,
                    axis=patching.resolve_axis(e)[0],
                    triggers=trigger_words_of(e.path) if e.kind == "lora" else [],
                )
            )
        safe = "".join(c for c in filename if c.isalnum() or c in "-_.").strip(".") or "palette"
        path = os.path.join(palette_dir(), safe + PALETTE_SUFFIX)
        given = resolve_settings(settings) if settings is not None else {}
        save_palette(path, Palette(name, entries, given))
        report = " ".join([f"Saved {safe}{PALETTE_SUFFIX} with {len(entries)} entries.", *notes])
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRALoadOrder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "sets": (SETS,)},
            "optional": {"depth": ("INT", {"default": -1, "min": -1, "max": 16})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "show"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def show(self, model, sets, depth=-1):
        plan, _ = patching.plan_chain(model, _chain(sets), live_env())
        report = plan.report(plan.depth(depth))
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRAApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sets": (SETS,),
                "depth": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 64,
                        "tooltip": "Iteration passes. -1 uses the depth that the mods recommend.",
                    },
                ),
            },
            "optional": {
                "iteration": (
                    ["mod", "accumulate", "path"],
                    {
                        "default": "mod",
                        "tooltip": (
                            "How depth passes work. mod: the mode the mod chose. accumulate: "
                            "each pass adds its change. path: each pass picks the next "
                            "expert, and the result is that expert's change."
                        ),
                    },
                ),
                "depth_schedule": (depth_schedules.DEPTH, {}),
                "routing_scope": (
                    ["mod", *ROUTING_SCOPES],
                    {
                        "default": "mod",
                        "tooltip": (
                            "Which logits choose the experts. mod: the scope stored in each "
                            "mod. token: each token on its "
                            "own. tensor: the mean over all tokens, one choice for the "
                            "whole tensor."
                        ),
                    },
                ),
                "routing_select": (
                    ["mod", *ROUTING_SELECTS],
                    {
                        "default": "mod",
                        "tooltip": (
                            "How the logits become expert weights. mod: the selection "
                            "stored in each mod. sparsemax: weights "
                            "that sum to 1, weak experts at 0. top1: only the best expert."
                        ),
                    },
                ),
                "output_gate": (["mod", *GATE_MODES], {"default": "mod"}),
                "gate_kappa": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.05},
                ),
                "path_width": ("INT", {"default": 0, "min": 0, "max": 8}),
                "path_select": (["mod", *PATH_SELECTS], {"default": "mod"}),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional. The text encoder layers of the mods and LoRA files "
                            "apply to this CLIP, as ComfyUI's LoRA loader does. Use the "
                            "clip output for the prompt."
                        )
                    },
                ),
                "route_freeze": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "timeline": (TIMELINE,),
                "settings": (kilora_settings.SETTINGS,),
                "depth_map": (DEPTH_MAP,),
                "linear_jump": ("BOOLEAN", {"default": False}),
                "jump_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "axis_alpha": (["off", "auto"], {"default": "off"}),
                "alpha_mask": ("MASK",),
                "gate_history": ("STRING", {"default": "full"}),
                "expert_text": (["auto", "off"], {"default": "auto"}),
                "variation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": (
                            "Entry sampling temperature; acts only with derived routing. "
                            "Zero selects deterministically."
                        ),
                    },
                ),
                "record_history": ("BOOLEAN", {"default": True}),
                "candidates": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "attention_route": (["multiply", "add", "plain"], {"default": "multiply"}),
                "attention_route_cap": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 100.0, "step": 0.1},
                ),
                "size_norm": (list(SIZE_NORMS), {"default": "off"}),
                "radial_step": (
                    "FLOAT",
                    {"default": COMFY_RADIAL_STEP, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "orthogonal_step": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "radial_budget": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "CLIP")
    RETURN_NAMES = ("model", "report", "clip")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, model=None, sets=None, **_):
        return _live_revisions(sets)

    def apply(
        self,
        model,
        sets,
        depth,
        iteration="mod",
        depth_schedule=None,
        routing_scope="mod",
        routing_select="mod",
        output_gate="mod",
        gate_kappa=0.0,
        path_width=0,
        path_select="mod",
        clip=None,
        route_freeze=0.0,
        settings=None,
        timeline=None,
        depth_map=None,
        linear_jump=False,
        jump_threshold=0.05,
        axis_alpha="off",
        alpha_mask=None,
        gate_history="full",
        expert_text="auto",
        variation=1.0,
        record_history=True,
        candidates=0,
        attention_route="multiply",
        attention_route_cap=2.0,
        size_norm="off",
        radial_step=COMFY_RADIAL_STEP,
        orthogonal_step=1.0,
        radial_budget=0.0,
    ):
        if size_norm not in SIZE_NORMS:
            raise patching.KiLoRAError(f"unknown size_norm {size_norm!r}; use off or weight")
        given = _chain(sets)
        size_lines: list[str] = []
        if size_norm == "weight" or any(isinstance(e, SizedLoRAEntry) for e in given):
            given, size_lines = size_normed_chain(model, given, size_norm == "weight")
        chain, recipe_settings = patching.expand_chain(given, live_env())
        if settings is None and recipe_settings is not None:
            settings = {k: v for k, v in recipe_settings.items() if k in kilora_settings.DEFAULTS}
            if isinstance(settings.get("depth_schedule"), str):
                settings["depth_schedule"] = depth_schedules.compile_depth(
                    settings["depth_schedule"]
                )
            if output_gate == "mod":
                output_gate = recipe_settings["output_gate"]
            if gate_kappa == 0:
                gate_kappa = recipe_settings["gate_kappa"]
            if path_width == 0:
                path_width = recipe_settings["path_width"]
            if path_select == "mod":
                path_select = recipe_settings["path_select"]
            if route_freeze == 0:
                route_freeze = recipe_settings["route_freeze"]
            if axis_alpha == "off":
                axis_alpha = recipe_settings.get("axis_alpha", "off")
        s = kilora_settings.apply_inputs(
            settings,
            depth,
            _iteration(iteration),
            depth_schedule,
            _from_mod(routing_scope),
            _from_mod(routing_select),
        )
        kilora_settings.resolve_settings(s)
        stack = s["mix"] == kilora_settings.STACK
        patched, report = patching.apply(
            model,
            chain,
            s["depth"],
            live_env(),
            s["iteration"],
            s["depth_schedule"],
            s["routing_scope"],
            s["routing_select"],
            gate=("none", None) if stack else (output_gate, gate_kappa),
            gate_history=gate_history,
            walks=(path_width, path_select),
            freeze=False if stack else (route_freeze or None),
            joint=not stack,
            timeline=timeline,
            token_depth=depth_map,
            axis_alpha=axis_alpha,
            alpha_clip=clip,
            alpha_mask=alpha_mask,
            expert_text=expert_text,
            import_score=s.get("import_score"),
            content_keys=s["content_keys"],
            gate_stats=s["gate_stats"],
            lycoris_apply=s["lycoris_apply"],
            variation=variation,
            linear_jump=linear_jump,
            jump_threshold=jump_threshold,
            candidates=candidates or s["candidates"],
            candidate_summary=s["candidate_summary"],
            routing_features_enabled=s["routing_features"],
            cross_weights=tuple(
                s.get(k, 0.0)
                for k in ("cross_token_region", "cross_token_prompt", "cross_token_timestep")
            ),
            depth_steps=(float(radial_step), float(orthogonal_step), float(radial_budget)),
        )
        if size_lines:
            report += "\n" + "\n".join(size_lines)
        if record_history:
            from ..key_cache import file_sha256
            from ..usage_history import record

            try:
                record([file_sha256(e.path) for e in chain if e.enabled and os.path.isfile(e.path)])
            except OSError as exc:
                report += f"\nCould not record local history: {exc}"
        if clip is None:
            return patched, report, None
        clip_out, clip_report = patching.apply_clip(model, clip, chain, live_env(), timeline)
        return patched, report + ("\n" + clip_report if clip_report else ""), clip_out


class KiLoRAClearHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "clear"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def clear(self):
        from ..usage_history import clear

        clear()
        return ("Local history cleared.",)


class KiLoRADepthSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("STRING", {"default": "", "multiline": True}),
                "preset": (["none", *depth_schedules.PRESET_NAMES], {"default": "none"}),
            },
            "optional": {
                "blocks": (
                    "INT",
                    {"default": depth_schedules.PREVIEW_BLOCKS, "min": 1, "max": 256},
                ),
                "preview_kind": (list(depth_schedules.KINDS), {"default": "self_attn"}),
            },
        }

    RETURN_TYPES = (depth_schedules.DEPTH, "IMAGE")
    RETURN_NAMES = ("depth_schedule", "preview")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, schedule, preset, blocks=depth_schedules.PREVIEW_BLOCKS, preview_kind=None):
        sched = depth_schedules.node_spec(schedule, preset)
        kind = preview_kind or "self_attn"
        values = depth_schedules.grid(sched, int(blocks), depth_schedules.PREVIEW_STEPS, kind)
        image = depth_schedules.heatmap(values)
        report = depth_schedules.describe(sched)
        if sched is not None:
            report += f" Depth {float(values.min()):g} to {float(values.max()):g}."
        return {"ui": {"text": [report]}, "result": (sched, image)}


DEPTH_MAP = dropin.DEPTH_MAP


class KiLoRADepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.5}),
                "max_depth": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 16.0, "step": 0.5}),
                "invert": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 512}),
                "start": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 16.0, "step": 0.5}),
            },
            "optional": {"mask": ("MASK",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = (DEPTH_MAP, "IMAGE")
    RETURN_NAMES = ("depth_map", "preview")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self, min_depth, max_depth, invert, gamma, feather, start, base, mask=None, image=None
    ):
        from ..depth_map import luminance, make_depth_map

        if mask is not None:
            values = mask if mask.dim() == 3 else mask.reshape(-1, *mask.shape[-2:])
            source = "mask"
        elif image is not None:
            values, source = luminance(image), "image"
        else:
            raise patching.KiLoRAError("connect a mask or an image to KiLoRA Depth Map")
        try:
            m = make_depth_map(
                values,
                min_depth,
                max_depth,
                invert,
                gamma,
                feather,
                start,
                None if base < 0 else base,
            )
        except ValueError as exc:
            raise patching.KiLoRAError(str(exc)) from None
        depths = m.depths()
        report = (
            f"Depth map from the {source}: depth {float(depths.min()):g} to "
            f"{float(depths.max()):g}, mean {float(depths.mean()):.2f}; depth {m.base:g} "
            f"for every token until progress {m.start:g}."
        )
        if mask is not None and image is not None:
            report += " The image is not used: the mask comes first."
        preview = m.values[..., None].expand(*m.values.shape, 3).contiguous()
        return {"ui": {"text": [report]}, "result": (m, preview)}


class KiLoRALoadMod:
    @classmethod
    def IS_CHANGED(cls, **_):
        return _live_revisions()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "file": (FOLDERS.list("kilora", MOD_SUFFIXES),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Signed strength. 0 turns the mod off. "
                            "Negative values push away from the mod."
                        ),
                    },
                ),
            },
            "optional": {
                "depth": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 16,
                        "tooltip": "Iteration passes. -1 uses the depth that the mod recommends.",
                    },
                ),
                "iteration": (
                    ["mod", "accumulate", "path"],
                    {
                        "default": "mod",
                        "tooltip": (
                            "How depth passes work. mod: the mode the mod chose. accumulate: "
                            "each pass adds its change. path: each pass picks the next "
                            "expert, and the result is that expert's change."
                        ),
                    },
                ),
                "routing_scope": (
                    ["mod", *ROUTING_SCOPES],
                    {
                        "default": "mod",
                        "tooltip": (
                            "Which logits choose the experts. mod: the scope stored in each "
                            "mod. token: each token on its "
                            "own. tensor: the mean over all tokens, one choice for the "
                            "whole tensor."
                        ),
                    },
                ),
                "routing_select": (
                    ["mod", *ROUTING_SELECTS],
                    {
                        "default": "mod",
                        "tooltip": (
                            "How the logits become expert weights. mod: the selection "
                            "stored in each mod. sparsemax: weights "
                            "that sum to 1, weak experts at 0. top1: only the best expert."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model,
        file,
        strength,
        depth=-1,
        iteration="mod",
        routing_scope="mod",
        routing_select="mod",
    ):
        entry = ModEntry("mod", _file("kilora", file, "mod"), _strength(strength), 1.0, True)
        chain = ModChain().add(entry)
        from ..modfile import read_manifest

        kinds = {section.get("kind") for section in read_manifest(entry.path).get("sections", [])}
        if "weight-ref" in kinds or "logic" in kinds:
            patched, report, _ = KiLoRAApply().apply(
                model,
                chain,
                depth,
                iteration=iteration,
                routing_scope=routing_scope,
                routing_select=routing_select,
            )
            return patched, report
        return patching.apply(
            model,
            chain,
            depth,
            live_env(),
            _iteration(iteration),
            None,
            _from_mod(routing_scope),
            _from_mod(routing_select),
            depth_steps=COMFY_DEPTH_STEPS,
        )


def _from_mod(value: str) -> str | None:
    return None if value == "mod" else value


def _iteration(value: str) -> str | None:
    if value not in ("mod", "accumulate", "path"):
        raise ValueError(f"unknown iteration {value!r}")
    return None if value == "mod" else value


class KiLoRAClear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "clear"
    CATEGORY = CATEGORY

    def clear(self, model):
        return (patching.clear(model),)


class KiLoRALiveSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "action": (list(patching.LIVE_ACTIONS),),
                "name": (
                    "STRING",
                    {"default": "", "tooltip": "Save point name, or the save to load"},
                ),
            },
            "optional": {"mod": ("STRING", {"default": "all"})},
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "act"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def act(self, model, action, name, mod="all"):
        out, report = patching.live_action(model, action, name, mod)
        return {"ui": {"text": [report]}, "result": (out, report)}


def _editor_options(measurement: str, time_bins: str, path_merge: str = "unchanged") -> dict:
    out: dict = {}
    if measurement != "unchanged":
        out["diff"] = measurement
    if time_bins != "unchanged":
        out["time_bins"] = int(time_bins)
    if path_merge != "unchanged":
        out["path_merge"] = path_merge == "on"
    return out


class KiLoRALiveOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mod": ("STRING", {"default": ""}),
                "learning": ("BOOLEAN", {"default": True}),
                "speed": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "keep_previous": (["unchanged", "on", "off"], {"default": "unchanged"}),
                "measure_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "reduce_with": (
                    ["unchanged", "latent", "decode-resize-encode"],
                    {"default": "unchanged"},
                ),
                "pin": (["unchanged", "on", "off"], {"default": "unchanged"}),
                "controller": (["unchanged", "r1", "rprop"], {"default": "unchanged"}),
                "measure": (["unchanged", "compressed", "full"], {"default": "unchanged"}),
                "measure_combos": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "edits": (["unchanged", "on", "off"], {"default": "unchanged"}),
                "measure_samples": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "learn": (["unchanged", "measure", "prediction"], {"default": "unchanged"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "set"
    CATEGORY = CATEGORY

    def set(
        self,
        model,
        mod,
        learning,
        speed,
        keep_previous="unchanged",
        measure_size=0,
        reduce_with="unchanged",
        pin="unchanged",
        controller="unchanged",
        measure="unchanged",
        measure_combos=0,
        edits="unchanged",
        measure_samples=0,
        learn="unchanged",
    ):
        from ..live_imports import ImportedLiveSession

        sessions = [
            s for s in patching.sessions_of(model) if not mod.strip() or s.mod_id == mod.strip()
        ]
        if len(sessions) > 1:
            raise patching.KiLoRAError(text("err.choose_live_session"))
        if not sessions:
            raise patching.KiLoRAError(f"no live mod {mod!r} is loaded")
        imported_options = {}
        if edits != "unchanged" or controller != "unchanged":
            if not isinstance(sessions[0], ImportedLiveSession):
                raise patching.KiLoRAError(text("err.import_options"))
            imported_options = {
                "edits": None if edits == "unchanged" else edits,
                "controller": None if controller == "unchanged" else controller,
            }
        keep = {"on": True, "off": False}.get(keep_previous)
        message = sessions[0].set_options(
            learning,
            None if speed < 0 else speed,
            keep_previous=keep,
            measure_size=measure_size or None,
            measure_reducer=None if reduce_with == "unchanged" else reduce_with,
            learn=None if learn in ("unchanged", 0, "0", None) else learn,
            **imported_options,
        )
        if controller != "unchanged":
            from ..live_imports import ImportedLiveSession

            if not isinstance(sessions[0], ImportedLiveSession):
                raise patching.KiLoRAError("controller is only available for imported sessions")
            sessions[0].set_options(controller=controller)
        if measure != "unchanged" or measure_combos or measure_samples:
            from ..live_imports import ImportedLiveSession

            if not isinstance(sessions[0], ImportedLiveSession):
                raise patching.KiLoRAError("measure is only available for imported sessions")
            sessions[0].set_options(
                measure=None if measure == "unchanged" else measure,
                measure_combos=measure_combos or None,
                measure_samples=measure_samples or None,
            )
        if pin != "unchanged":
            from kilora.live_queue import queue_for

            with queue_for(sessions[0]).exclusive():
                message += " " + (sessions[0].pin() if pin == "on" else sessions[0].unpin())
        out, report = patching.reapply(model)
        return (out, f"{mod}: {message}\n{report}")


class KiLoRALivePacks:
    @classmethod
    def INPUT_TYPES(cls):
        from ..editor_packs import PACK_MODES

        return {
            "required": {
                "model": ("MODEL",),
                "mod": ("STRING", {"default": ""}),
                "packs": (list(PACK_MODES), {"default": "auto"}),
                "turned_off": ("STRING", {"default": ""}),
            },
            "optional": {
                "editor_kind": ("STRING", {"default": ""}),
                "measurement": (["unchanged", "central", "one-sided"], {"default": "unchanged"}),
                "time_bins": (["unchanged", "1", "2"], {"default": "unchanged"}),
                "families": ("STRING", {"default": ""}),
                "path_merge": (["unchanged", "off", "on"], {"default": "unchanged"}),
                "path_angle": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 3.14159, "step": 0.01},
                ),
                "cross_layer": (["unchanged", "off", "rls", "jvp"], {"default": "unchanged"}),
                "keep_merge_pairs": (["unchanged", "off", "on"], {"default": "unchanged"}),
                "merge_residual": (["unchanged", "keep", "drop"], {"default": "unchanged"}),
                "term_budget": ("INT", {"default": 64, "min": 1, "max": 4096}),
                "explore": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "choose"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def choose(
        self,
        model,
        mod,
        packs,
        turned_off,
        editor_kind="",
        measurement="unchanged",
        time_bins="unchanged",
        families="",
        path_merge="unchanged",
        path_angle=None,
        cross_layer="unchanged",
        keep_merge_pairs="unchanged",
        merge_residual="unchanged",
        term_budget=None,
        explore=None,
    ):
        from ..editor_packs import select_packs

        sessions = [
            s for s in patching.sessions_of(model) if not mod.strip() or s.mod_id == mod.strip()
        ]
        if not sessions:
            raise patching.KiLoRAError(f"no live mod {mod!r} is loaded")
        if len(sessions) > 1:
            raise patching.KiLoRAError("several live mods are loaded: set mod to one mod ID")
        session = sessions[0]
        installed, problems = _installed_packs()
        manifest = session.mod.manifest
        pins = {p["id"]: p["version"] for p in session.loaded_packs if "id" in p}
        off = [x.strip() for x in turned_off.split(",") if x.strip()]
        chosen, notes = select_packs(
            installed,
            mode=packs,
            disabled=off,
            base_family=(manifest.get("base_model") or {}).get("family", ""),
            mod_family=manifest.get("mod_family"),
            editor_kind=editor_kind.strip() or None,
            pins=pins,
            families=[f for f in families.split(",") if f.strip()],
        )
        session.editor_packs = chosen
        options = _editor_options(measurement, time_bins, path_merge)
        if path_angle is not None:
            options["path_angle"] = path_angle
        if cross_layer != "unchanged":
            options["cross_layer"] = cross_layer
        if keep_merge_pairs != "unchanged":
            options["keep_merge_pairs"] = keep_merge_pairs == "on"
        if merge_residual != "unchanged":
            options["merge_residual"] = merge_residual
        if term_budget is not None:
            options["term_budget"] = term_budget
        if explore is not None:
            options["explore"] = explore
        if options:
            session.set_options(editor_options=options)
        session.revision += 1
        return (model, _packs_report(session, installed, chosen, off, notes + problems))


def _installed_packs() -> tuple[list, list[str]]:
    from ..editor_packs import PACK_SUFFIX, scan_packs

    try:
        names = FOLDERS.list("kilora", PACK_SUFFIX)
        return scan_packs(FOLDERS.path("kilora", n) for n in names)
    except Exception as exc:
        return [], [str(exc)]


def _packs_report(session, installed, chosen, off, notes) -> str:
    from ..editor_packs import check_save_record

    used = {p.id for p in chosen}
    lines = [f"{session.mod_id}: editor packs"]
    for p in sorted(installed, key=lambda p: (p.role != "common", p.id, p.version)):
        state = "on" if p.id in used and p in chosen else ("off" if p.id in off else "not used")
        fams = f" ({', '.join(p.families)})" if p.families else ""
        lines.append(f"  [{state}] {p.id} {p.version}, {p.role}{fams}")
    if not installed:
        lines.append("  none installed: put .kedit-pack.safetensors files in models/kilora")
    if session.loaded_packs:
        saved = ", ".join(f"{r.get('id')} {r.get('version')}" for r in session.loaded_packs)
        lines.append(f"The loaded save used: {saved}.")
        warnings = check_save_record(session.loaded_packs, chosen)
        lines += [f"Warning: {w}" for w in warnings]
    lines += notes
    return "\n".join(lines)


class KiLoRAFreeze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mod": ("STRING", {"default": ""}),
                "new_id": ("STRING", {"default": "my-frozen-mod"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "freeze"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def freeze(self, model, mod, new_id):
        sessions = [s for s in patching.sessions_of(model) if s.mod_id == mod.strip()]
        if not sessions:
            raise patching.KiLoRAError(f"no live mod {mod!r} is loaded")
        safe = _safe_id(new_id) or "frozen"
        family = (sessions[0].mod.manifest.get("base_model") or {}).get("family")
        name = mod_filename(safe, family)
        path = os.path.join(FOLDERS.first_dir("kilora"), name)
        if os.path.exists(path):
            raise patching.KiLoRAError(f"{name} exists; pick another ID")
        sessions[0].freeze(path, safe)
        from ..live import LEARNED_STATE_WARNING

        report = f"Saved {name}, a static copy of {mod}. Note: {LEARNED_STATE_WARNING}"
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRALiveRecord:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "editor": (
                    FOLDERS.list("kilora", UPDATER_SUFFIXES) + [text("choice.import_editor")],
                ),
            },
            "optional": {"clip": ("CLIP",)},
        }

    RETURN_TYPES = ("MODEL", live_write.PATH_STATE)
    RETURN_NAMES = ("model", "path")
    FUNCTION = "record"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def record(self, model, editor, clip=None):
        from ..editors import editor_for

        try:
            path = FOLDERS.path("kilora", editor) if editor else ""
        except FileNotFoundError:
            path = ""
        rule = editor_for(path) if path and os.path.exists(path) else None
        out, handle = live_write.record(model, rule)
        if clip is not None:
            handle.encode_empty = live_write.comfy_empty_caption(clip, model.model)
            handle.encode_caption = live_write.comfy_caption_encoder(clip, model.model)
        return out, handle


class KiLoRALiveWrite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"path": (live_write.PATH_STATE,), "samples": ("LATENT",)},
            "optional": {
                "vae": ("VAE",),
                "mask": ("MASK",),
                "region_caption": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "report")
    FUNCTION = "write"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def write(self, path, samples, vae=None, mask=None, region_caption=""):
        report = live_write.write(
            path, samples, vae=vae, mask=mask, region_caption=region_caption or ""
        )
        return {"ui": {"text": [report]}, "result": (samples, report)}


class KiLoRALiveLearnFromImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (live_write.PATH_STATE,),
                "vae": ("VAE",),
                "cue": ("STRING", {"default": "style"}),
                "ignore_guard": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "captions": ("STRING", {"default": "", "multiline": True}),
                "folder": ("STRING", {"default": ""}),
                "clip": ("CLIP",),
                "mounts": ("STRING", {"default": "", "multiline": True}),
                "t_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask": ("MASK",),
                "axis": (["all", *RECORD_AXES], {"default": "all"}),
                "single_image": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (live_write.PATH_STATE, "STRING")
    RETURN_NAMES = ("path", "report")
    FUNCTION = "learn"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def learn(
        self,
        path,
        vae,
        cue,
        ignore_guard=False,
        images=None,
        captions="",
        folder="",
        clip=None,
        mounts="",
        t_min=0.0,
        t_max=1.0,
        mask=None,
        axis="all",
        single_image=False,
    ):
        from ..image_terms import holdout_indices, learn_from_images
        from . import image_learn
        from .image_solve import _load_items, _prepare, image_samples

        items = _load_items(images, captions, folder)
        if single_image:
            if len(items) != 1:
                raise ValueError("Single-image mode needs exactly 1 image")
        else:
            holdout_indices(len(items))
        if not 0.0 <= float(t_min) <= float(t_max) <= 1.0:
            raise ValueError("t_min and t_max must be numbers from 0 to 1, with t_min <= t_max")
        if not cue.strip():
            raise ValueError("cue must be nonempty")
        if clip is None:
            raise ValueError("Learn From Images needs the CLIP input for the captions")
        patcher = path.patcher
        if patcher is None:
            raise ValueError("Run the sampler with KiLoRA Live: 1. Record once first")
        session = path.session
        if not session.learning:
            report = text("msg.learning_off")
            return {"ui": {"text": [report]}, "result": (path, report)}
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        settings = image_learn.settings_for(patcher, float(t_min), float(t_max))
        mm.load_models_gpu([patcher])
        prepared = _prepare(patcher, vae, items, settings.size)
        token_mask = None
        if mask is not None:
            from ..regions import is_full, latent_mask

            token_mask = latent_mask(mask, prepared[0].latent)
            if is_full(token_mask):
                token_mask = None
        bar = ProgressBar(max(1, len(items) * len(settings.seeds)))

        def samples(chosen, cue_, settings_):
            mm.throw_exception_if_processing_interrupted()
            return image_samples(patcher, vae, chosen, cue_, settings_, bar, clip=clip)

        def losses(chosen, cue_, settings_, trial):
            mm.throw_exception_if_processing_interrupted()
            return image_learn.held_out_losses(patcher, clip, chosen, cue_, settings_, trial)

        def drift(trial):
            return image_learn.unrelated_drift(patcher, clip, settings, trial)

        queue = getattr(path, "queue", None)
        lock = queue.exclusive() if queue is not None else contextlib.nullcontext()
        kept = (path.queue, path.generating, session.learning)
        path.queue, session.learning = None, False
        try:
            with lock:
                result = learn_from_images(
                    session,
                    prepared,
                    cue,
                    settings,
                    ignore_guard=bool(ignore_guard),
                    progress=print,
                    samples=samples,
                    losses=losses,
                    drift=drift,
                    editor=path.rule,
                    model=patcher.model.diffusion_model,
                    mounts=mounts or None,
                    axis=None if axis == "all" else axis,
                    mask=token_mask,
                    single_image=bool(single_image),
                )
        finally:
            path.queue, path.generating, session.learning = kept
            mm.soft_empty_cache()
        report = result.text()
        return {"ui": {"text": [report]}, "result": (path, report)}


class KiLoRASimilarLoRAs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (FOLDERS.list("loras", ".safetensors"),),
                "count": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
            "optional": {
                "similar_space": (["keys", "embedding"],),
                "model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "find"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def find(self, lora, count, similar_space="keys", model_path=""):
        from safetensors import SafetensorError

        from ..library import LibraryIndex

        path = _file("loras", lora, "lora")
        folders = FOLDERS.dirs("loras")
        index = LibraryIndex.open()
        index.sync(folders, reuse_text=similar_space == "embedding")
        fallback = ""
        vectors = None
        if similar_space == "embedding":
            try:
                if not model_path.strip():
                    raise ValueError("empty model path")
                digest = index.paths[os.path.realpath(os.path.expanduser(path))]
                family = index.records[digest]["family"]
                vectors = index.embedding_inputs(family, model_path)
                if digest not in vectors:
                    raise ValueError("no matching coefficients for query")
            except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
                fallback = text("similar.fallback", reason=str(exc))
                vectors = None
        results = index.similar(path, count, vectors=vectors)
        report = "\n".join(f"{item.score:.4f}  {item.path}" for item in results)
        if not report:
            report = text("similar.empty")
        if fallback:
            report = fallback + "\n" + report
        return {"ui": {"text": [report]}, "result": (report,)}


class KiLoRASaveRecipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sets": (SETS,),
                "filename": ("STRING", {"default": "recipe"}),
                "name": ("STRING", {"default": "My recipe"}),
                "license": ("STRING", {"default": "NOASSERTION"}),
            },
            "optional": {"settings": (kilora_settings.SETTINGS,)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, sets, filename, name, license="NOASSERTION", settings=None):
        from ..recipes import save_recipe

        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename).strip(".-") or "recipe"
        if not safe.lower().endswith(".kmod"):
            safe += ".kmod"
        path = os.path.join(FOLDERS.first_dir("kilora"), safe)
        lines = save_recipe(_chain(sets), settings, path, name, license)
        return ("\n".join([f"Saved {safe}.", *lines]),)


PACK_PLUGIN = "kilora-editor-chunk"
PACK_FAMILIES = ("auto", "z-image", "anima", "sd15", "sdxl")


def _pack_builder():
    import importlib

    from ..editors import plugin_for_kind

    plugin = plugin_for_kind("chunk")
    package = getattr(getattr(plugin, "load", None), "__module__", "") or ""
    if plugin is None or not package:
        raise patching.KiLoRAError(text("err.pack_plugin_missing", plugin=PACK_PLUGIN))
    try:
        return importlib.import_module(package.split(".")[0] + ".comfy")
    except ImportError:
        raise patching.KiLoRAError(text("err.pack_plugin_missing", plugin=PACK_PLUGIN)) from None


def _pack_prompts(value: str) -> dict[str, list[str]]:
    import json

    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise patching.KiLoRAError(text("err.pack_prompts", reason=str(exc))) from None
    if not isinstance(data, dict) or not data:
        raise patching.KiLoRAError(text("err.pack_prompts", reason="not a nonempty object"))
    for family, items in data.items():
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}", str(family)):
            raise patching.KiLoRAError(text("err.pack_prompts", reason=f"family {family!r}"))
        if (
            not isinstance(items, list)
            or not items
            or not all(isinstance(p, str) and p.strip() for p in items)
        ):
            raise patching.KiLoRAError(
                text("err.pack_prompts", reason=f"{family}: a nonempty list of prompts")
            )
    return data


def _with_loras(model, loras: str) -> tuple[Any, list[dict]]:
    import hashlib

    lines = [line.strip() for line in loras.splitlines() if line.strip()]
    if not lines:
        return model, []
    import comfy.sd
    import comfy.utils

    record = []
    for line in lines:
        name, strength = line, 1.0
        head, sep, tail = line.rpartition(":")
        if sep and re.fullmatch(r"-?\d+(\.\d+)?", tail.strip()):
            name, strength = head.strip(), _strength(float(tail), -4.0, 4.0)
        path = _file("loras", name, "lora")
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda f=f: f.read(1 << 20), b""):
                digest.update(block)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0.0)
        record.append({"name": name, "strength": strength, "sha256": digest.hexdigest()})
    return model, record


class KiLoRABuildEditorPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "name": ("STRING", {"default": "my-packs"}),
                "prompts": (
                    "STRING",
                    {
                        "default": '{"_common": ["a photo of a cat"], '
                        '"watercolor": ["a watercolor painting of a cat"]}',
                        "multiline": True,
                    },
                ),
                "base_family": (list(PACK_FAMILIES), {"default": "auto"}),
                "draws": ("INT", {"default": 2, "min": 1, "max": 64}),
                "common": ("INT", {"default": 48, "min": 1, "max": 4096}),
                "family": ("INT", {"default": 16, "min": 1, "max": 4096}),
            },
            "optional": {
                "loras": ("STRING", {"default": "", "multiline": True}),
                "orthogonalize": ("BOOLEAN", {"default": False}),
                "size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "report")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def build(
        self,
        model,
        clip,
        name,
        prompts,
        base_family="auto",
        draws=2,
        common=48,
        family=16,
        loras="",
        orthogonalize=False,
        size=0,
        steps=0,
        seed=0,
    ):
        builder = _pack_builder()
        safe = _safe_id(name)
        if not safe:
            raise patching.KiLoRAError(text("err.pack_name"))
        data = _pack_prompts(prompts)
        if base_family not in PACK_FAMILIES:
            raise patching.KiLoRAError(text("err.pack_prompts", reason=str(base_family)))
        model, lora_record = _with_loras(model, loras or "")
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        root = FOLDERS.first_dir("kilora")
        os.makedirs(root, exist_ok=True)
        bar = ProgressBar(sum(len(items) for items in data.values()))
        try:
            result = builder.build_pack(
                model,
                clip,
                data,
                os.path.join(root, safe),
                cache=os.path.join(root, ".cache", "editor-events"),
                base_family=None if base_family == "auto" else base_family,
                draws=draws,
                common=common,
                family=family,
                orthogonalize=bool(orthogonalize),
                size=size or None,
                steps=steps or None,
                seed=seed,
                extra={"loras": lora_record},
                interrupt=mm.throw_exception_if_processing_interrupted,
                progress=bar.update,
                log=lambda line: logging.getLogger("kilora").info("%s", line),
            )
        except (ValueError, KeyError) as exc:
            raise patching.KiLoRAError(str(exc)) from exc
        installed, _ = _installed_packs()
        folder = os.path.abspath(result.folder)
        found = sum(
            1 for p in installed if os.path.dirname(os.path.abspath(p.path or "")) == folder
        )
        directions = ", ".join(f"{fam} {count}" for fam, count in result.directions.items())
        report = "\n".join(
            [
                text(
                    "pack_build.report",
                    packs=len(result.packs),
                    folder=f"models/kilora/{safe}",
                    directions=directions,
                ),
                text(
                    "pack_build.events",
                    events=result.events,
                    reused=result.reused,
                    seconds=f"{result.seconds:.0f}",
                ),
                result.estimate or text("pack_build.no_estimate"),
                text("pack_build.found", found=found),
            ]
        )
        return {"ui": {"text": [report]}, "result": (str(result.folder), report)}


NODE_CLASS_MAPPINGS = {
    "KiLoRASelect": KiLoRASelect,
    "KiLoRAImportLoRA": KiLoRAImportLoRA,
    "KiLoRATriggerWords": KiLoRATriggerWords,
    "KiLoRAPalette": KiLoRAPalette,
    "KiLoRASavePalette": KiLoRASavePalette,
    "KiLoRASuggestions": KiLoRASuggestions,
    "KiLoRALoadOrder": KiLoRALoadOrder,
    "KiLoRAApply": KiLoRAApply,
    "KiLoRAClearHistory": KiLoRAClearHistory,
    "KiLoRAClear": KiLoRAClear,
    "KiLoRALiveSave": KiLoRALiveSave,
    "KiLoRALiveOptions": KiLoRALiveOptions,
    "KiLoRALivePacks": KiLoRALivePacks,
    "KiLoRAFreeze": KiLoRAFreeze,
    "KiLoRAModInfo": KiLoRAModInfo,
    "KiLoRALiveExport": KiLoRALiveExport,
    "KiLoRALoadMod": KiLoRALoadMod,
    "KiLoRALiveRecord": KiLoRALiveRecord,
    "KiLoRALiveWrite": KiLoRALiveWrite,
    "KiLoRALiveLearnFromImages": KiLoRALiveLearnFromImages,
    "KiLoRARegion": KiLoRARegion,
    "KiLoRADepthSchedule": KiLoRADepthSchedule,
    "KiLoRARegionAttention": KiLoRARegionAttention,
    "KiLoRASimple": kilora_settings.KiLoRASimple,
    "KiLoRASettings": kilora_settings.KiLoRASettings,
    "KiLoRALoRALoader": dropin.KiLoRALoRALoader,
    "KiLoRARouteLoaded": takeover.KiLoRARouteLoaded,
    "KiLoRARouteRecord": KiLoRARouteRecord,
    "KiLoRARouteMap": KiLoRARouteMap,
    "KiLoRATokenRoles": KiLoRATokenRoles,
    "KiLoRATwoPanels": KiLoRATwoPanels,
    "KiLoRARegionIsolation": KiLoRARegionIsolation,
    "KiLoRATimeline": KiLoRATimeline,
    "KiLoRADepthMap": KiLoRADepthMap,
    "KiLoRALayerConditioning": KiLoRALayerConditioning,
    "KiLoRALayerFilter": KiLoRALayerFilter,
    "KiLoRASaveRecipe": KiLoRASaveRecipe,
    "KiLoRADirectionNodes": KiLoRADirectionNodes,
    "KiLoRALoRALibrary": KiLoRALoRALibrary,
    "KiLoRASimilarLoRAs": KiLoRASimilarLoRAs,
    "KiLoRAPrepareLoRAs": KiLoRAPrepareLoRAs,
    "KiLoRAMentionPrompt": KiLoRAMentionPrompt,
    "KiLoRAPromptSet": KiLoRAPromptSet,
    "KiLoRASolveFromPrompts": KiLoRASolveFromPrompts,
    "KiLoRASolveFromImages": KiLoRASolveFromImages,
    "KiLoRAPack": KiLoRAPack,
    "KiLoRABuildEditorPack": KiLoRABuildEditorPack,
}

NODE_DISPLAY_NAME_MAPPINGS = help_texts.decorate(NODE_CLASS_MAPPINGS)
