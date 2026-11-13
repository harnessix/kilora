# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .sections import Section, SectionKind, register_section

__all__ = ["KIND", "clip_lora_state", "is_text_encoder_key", "text_encoder_sections"]

KIND = "text-encoder"

TEXT_ENCODER_PREFIXES = (
    "lora_te_",
    "lora_te1_",
    "lora_te2_",
    "lora_te3_",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoders.",
    "te.",
    "te1.",
    "te2.",
    "cond_stage_model.",
    "conditioner.",
)


def is_text_encoder_key(key: str) -> bool:
    return key.startswith(TEXT_ENCODER_PREFIXES)


def _encode(data: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {}
    for stem, parts in data.items():
        if "/" in stem or not stem:
            raise ValueError(f"a text-encoder key stem cannot be empty or contain '/': {stem!r}")
        for part, value in parts.items():
            out[f"{stem}/{part}"] = value.float()
    return out


def _decode(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in tensors.items():
        stem, part = name.rsplit("/", 1)
        out.setdefault(stem, {})[part] = value.float()
    return out


def _scale(data, w: float):
    return {
        stem: {p: (v * w if p in ("up", "delta") else v) for p, v in parts.items()}
        for stem, parts in data.items()
    }


register_section(
    SectionKind(
        kind=KIND,
        since="0.1.1",
        encode=_encode,
        decode=_decode,
        changes=lambda data, s: {},
        scale=_scale,
    )
)


def text_encoder_sections(
    name: str, state: Mapping[str, torch.Tensor], weight: float = 1.0
) -> tuple[list[Section], list[str]]:
    from .multi_import import _split_any, lycoris_delta

    groups: dict[str, dict[str, torch.Tensor]] = {}
    keys: dict[str, list[str]] = {}
    skipped = []
    for key, tensor in state.items():
        if not is_text_encoder_key(key):
            continue
        split = _split_any(key)
        if split is None:
            skipped.append(key)
            continue
        stem, part = split
        groups.setdefault(stem, {})[part] = tensor
        keys.setdefault(stem, []).append(key)
    data: dict[str, dict[str, torch.Tensor]] = {}
    for stem, parts in groups.items():
        if "down" in parts and "up" in parts and parts["down"].dim() == 2:
            down, up = parts["down"].float(), parts["up"].float()
            rank = down.shape[0]
            alpha = parts.get("alpha")
            scale = float(alpha.float().reshape(-1)[0]) / rank if alpha is not None else 1.0
            data[stem] = {"down": down, "up": up * (scale * float(weight))}
            continue
        found = lycoris_delta(parts) if "down" not in parts else None
        if found is not None:
            data[stem] = {"delta": found[0] * float(weight)}
        else:
            skipped.extend(keys[stem])
    if not data:
        return [], skipped
    return [Section(KIND, name, data, {"layers": len(data)})], skipped


def clip_lora_state(sections: list[Section]) -> dict[str, torch.Tensor]:
    by_stem: dict[str, list[dict[str, torch.Tensor]]] = {}
    for s in sections:
        if s.kind == KIND:
            for stem, parts in s.data.items():
                by_stem.setdefault(stem, []).append(parts)
    out: dict[str, torch.Tensor] = {}
    for stem, items in by_stem.items():
        low = [p for p in items if "down" in p]
        dense = [p["delta"] for p in items if "delta" in p]
        if low and not dense:
            down = torch.cat([p["down"] for p in low], 0)
            out[f"{stem}.lora_down.weight"] = down
            out[f"{stem}.lora_up.weight"] = torch.cat([p["up"] for p in low], 1)
            out[f"{stem}.alpha"] = torch.tensor(float(down.shape[0]))
        else:
            total = sum(dense) + sum(p["up"] @ p["down"] for p in low)
            out[f"{stem}.diff"] = total
    return out


def apply_to_clip(clip: Any, mods: list, key_map: Any = None, load_lora: Any = None):
    work = [(m, [s for s in m.sections if s.kind == KIND]) for m in mods]
    work = [(m, s) for m, s in work if s]
    if clip is None or not work:
        return clip, []
    if key_map is None or load_lora is None:
        import comfy.lora

        key_map = key_map or comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        load_lora = load_lora or comfy.lora.load_lora
    out = clip.clone()
    lines = []
    for m, sections in work:
        state = clip_lora_state(sections)
        loaded = load_lora(state, key_map)
        out.add_patches(loaded, float(m.strength))
        stems = {k.rsplit(".", 2)[0] if ".lora_" in k else k.rsplit(".", 1)[0] for k in state}
        lines.append(
            f"{m.label}: {len(loaded)} text encoder layers patched ({len(stems)} in the mod)."
        )
    return out, lines
