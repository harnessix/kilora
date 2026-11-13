# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os


def _split_words(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _metadata(path: str) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _load_json(value: str):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return None
    return parsed


def _frequency_tags(raw: str) -> list[str]:
    parsed = _load_json(raw)
    if not isinstance(parsed, dict):
        return []
    words: list[str] = []
    for folder in parsed.values():
        if not isinstance(folder, dict):
            continue
        ranked: list[tuple[float, str]] = []
        for tag, count in folder.items():
            try:
                score = float(count)
            except (TypeError, ValueError):
                continue
            ranked.append((score, str(tag)))
        ranked.sort(key=lambda item: -item[0])
        words.extend(tag for _, tag in ranked[:3])
    return words


def _trained_words(payload) -> list[str] | None:
    if not isinstance(payload, dict) or "trainedWords" not in payload:
        return None
    value = payload["trainedWords"]
    if isinstance(value, str):
        return _split_words(value)
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _sidecar_words(path: str) -> list[str]:
    folder = os.path.dirname(path)
    name = os.path.basename(path)
    stem, _ext = os.path.splitext(name)
    candidates = [
        os.path.join(folder, name + ".civitai.info"),
        os.path.join(folder, name + ".json"),
        os.path.join(folder, stem + ".civitai.info"),
        os.path.join(folder, stem + ".json"),
    ]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        words = _trained_words(payload)
        if words is not None:
            return words
    return []


def _dedupe(words: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        word = word.strip()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def trigger_words_of(path: str) -> list[str]:
    meta = _metadata(path)
    words: list[str] = []
    phrase = meta.get("modelspec.trigger_phrase") or ""
    if phrase:
        words.extend(_split_words(phrase))
    frequency = meta.get("ss_tag_frequency") or ""
    if frequency:
        words.extend(_frequency_tags(frequency))
    words.extend(_sidecar_words(path))
    return _dedupe(words)


class KiLoRATriggerWords:
    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import FOLDERS

        return {
            "required": {
                "lora_name": (FOLDERS.list("loras", ".safetensors"),),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trigger_words",)
    FUNCTION = "trigger_words"
    CATEGORY = "KiLoRA"

    def trigger_words(self, lora_name, lora_stack=None):
        from .nodes import _file, _lora_stack_items

        names = [lora_name]
        names.extend(name for name, _model_strength in _lora_stack_items(lora_stack))
        words: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            words.extend(trigger_words_of(_file("loras", name, "lora")))
        return (", ".join(_dedupe(words)),)
