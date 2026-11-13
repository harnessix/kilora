# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re

MOD_SUFFIX = ".kmod.safetensors"
MOD_SUFFIXES = (MOD_SUFFIX, ".kmod")
UPDATER_SUFFIX = ".kedit.safetensors"
UPDATER_SUFFIXES = (UPDATER_SUFFIX, ".kedit")
PACK_SUFFIX = ".kpack.json"
PACK_SUFFIXES = (PACK_SUFFIX, ".kpack")

FAMILY_SHORT = {
    "z-image": "zimage",
    "sd15": "sd15",
    "sdxl": "sdxl",
    "anima": "anima",
    "kilora-tiny-1": "tiny",
}


def has_suffix(name: str | os.PathLike[str], suffixes: tuple[str, ...]) -> bool:
    return os.fspath(name).lower().endswith(suffixes)


def strip_suffix(name: str | os.PathLike[str], suffixes: tuple[str, ...]) -> str:
    text = os.fspath(name)
    for suffix in suffixes:
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return os.path.splitext(text)[0]


def mod_filename(mod_id: str, family: str | None = None) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", mod_id.lower()).strip(".-") or "mod"
    short = FAMILY_SHORT.get(family or "", re.sub(r"[^a-z0-9]+", "", (family or "").lower()))
    if short and short != "unknown" and not base.endswith("." + short):
        base = f"{base}.{short}"
    return base + MOD_SUFFIX
