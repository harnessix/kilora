# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

LOCALES = (
    "en",
    "ar",
    "de",
    "es",
    "fa",
    "fr",
    "he",
    "it",
    "ja",
    "ko",
    "pt-BR",
    "ru",
    "tr",
    "zh",
    "zh-TW",
)
FRONTEND_VERSION = "1.58.5"
MACHINE_MADE = tuple(lang for lang in LOCALES if lang != "en")

_DIR = Path(__file__).parent


@functools.cache
def catalog(lang: str) -> dict[str, str]:
    with open(_DIR / "en.json", encoding="utf-8") as f:
        texts = json.load(f)
    if lang != "en":
        path = _DIR / f"{lang}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                texts.update(json.load(f))
    return texts


def language() -> str:
    lang = os.environ.get("KILORA_LANG", "en")
    return lang if lang in LOCALES else "en"


def text(key: str, lang: str | None = None, **values: Any) -> str:
    return catalog(lang or language())[key].format(**values)
