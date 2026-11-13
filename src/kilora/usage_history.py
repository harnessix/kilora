# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import date
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from .palette import palette_dir

_LOCK = threading.RLock()
MAX_HISTORY_CHAINS = 10_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def history_path() -> Path:
    return Path(palette_dir()) / "usage-history.json"


def _hashes(values):
    return sorted({v.lower() for v in values if isinstance(v, str) and _HASH.fullmatch(v.lower())})


def read_history():
    try:
        data = json.loads(history_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def record(hashes, enabled=True):
    if not enabled:
        return
    hashes = _hashes(hashes)
    if not hashes:
        return
    with _LOCK:
        path = history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = read_history()
        rows.append({"date": date.today().isoformat(), "sha256": hashes})
        rows = rows[-MAX_HISTORY_CHAINS:]
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".usage-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def clear():
    with _LOCK:
        history_path().unlink(missing_ok=True)


@lru_cache(maxsize=2048)
def _recipe_entries(path, size, mtime_ns):
    from .modfile import load_mod

    mod = load_mod(path)
    return tuple(
        s.meta.get("sha256")
        for s in mod.sections
        if s.kind == "weight-ref" and s.meta.get("transforms", {}).get("enabled", True)
    )


def counts(palettes=(), recipes=(), history=True):

    from .palette import load_palette

    chains = (
        [row.get("sha256", []) for row in read_history() if isinstance(row, dict)]
        if history
        else []
    )
    for path in palettes:
        palette = load_palette(path)
        chains.append([e.sha256 for e in palette.entries if e.enabled])
    for path in recipes:
        path = Path(path).resolve()
        stat = path.stat()
        chains.append(list(_recipe_entries(path, stat.st_size, stat.st_mtime_ns)))
    singles, pairs = Counter(), Counter()
    for chain in chains:
        hashes = _hashes(chain if isinstance(chain, list) else [])
        singles.update(hashes)
        pairs.update(combinations(hashes, 2))
    return singles, pairs
