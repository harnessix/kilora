# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fast_keys import Keys

log = logging.getLogger("kilora.keys")

PARTS = ("keys_in", "keys_out", "rows_in", "rows_out")
FORMAT = 4
GRAPH_PARTS = (
    "graph_transition",
    "graph_powers",
    "graph_distances",
    "graph_stats",
    "graph_hops",
    "graph_merges",
)


def _format() -> int:
    from .compute import share_transition_gamma, share_transition_kind
    from .key_graph import enabled

    weighted = bool(share_transition_gamma()) and share_transition_kind() == "weighted"
    return (6 if enabled() else 5) if weighted else (FORMAT if enabled() else 2)


_META = "kilora_key_cache"

CACHE_DIR: str | None = None

_FILE_HASHES: dict[tuple, str] = {}


def cache_dir() -> str:
    if CACHE_DIR is not None:
        return CACHE_DIR
    user = None
    try:
        from .host import services

        folder_paths = getattr(services().comfy, "folder_paths", None)
        if folder_paths is None:
            import folder_paths
        user = folder_paths.get_user_directory()
    except Exception:
        user = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(user, "kilora", "key-cache")


def file_sha256(path: str | os.PathLike[str]) -> str:
    path = os.path.abspath(os.fspath(path))
    for _ in range(3):
        stat = os.stat(path)
        memo = (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if memo in _FILE_HASHES and time.time_ns() - stat.st_ctime_ns > 2_000_000_000:
            return _FILE_HASHES[memo]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            opened = os.fstat(f.fileno())
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
            finished = os.fstat(f.fileno())
        after = os.stat(path)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if not signature(stat) == signature(opened) == signature(finished) == signature(after):
            continue
        if len(_FILE_HASHES) > 256:
            _FILE_HASHES.clear()
        _FILE_HASHES[memo] = h.hexdigest()
        return _FILE_HASHES[memo]
    raise OSError(f"file changed while hashing: {path}")


def cache_key(lora_sha256: str, family: str, settings: Mapping[str, Any]) -> str:
    text = json.dumps(
        {"format": _format(), "lora": lora_sha256, "family": family, "settings": dict(settings)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _path(key: str) -> str:
    return os.path.join(cache_dir(), key + ".safetensors")


def load(key: str) -> Keys | None:
    path = _path(key)
    if not os.path.isfile(path):
        return None
    try:
        with safe_open(path, "pt") as f:
            meta = json.loads((f.metadata() or {}).get(_META, "{}"))
            if meta.get("format") != _format() or meta.get("key") != key:
                return None
            out = Keys(
                {
                    t: {
                        p: f.get_tensor(f"{t}/{p}")
                        for p in PARTS
                        + (
                            tuple(p for p in GRAPH_PARTS if f"{t}/{p}" in f.keys())
                            if _format() in (FORMAT, 6)
                            else ()
                        )
                        + (
                            ("share_transition_weighted",)
                            if f"{t}/share_transition_weighted" in f.keys()
                            else ()
                        )
                    }
                    for t in meta["targets"]
                }
            )
        out.log_tau = float(meta["log_tau"])
        return out
    except Exception as e:
        log.warning("key cache %s cannot be read (%s); computing the keys again", path, e)
        return None


def store(key: str, keys: Mapping[str, Mapping[str, torch.Tensor]], log_tau: float) -> str:
    folder = cache_dir()
    os.makedirs(folder, exist_ok=True)
    tensors = {
        f"{t}/{p}": v[p].detach().cpu().contiguous()
        for t, v in keys.items()
        for p in PARTS
        + (tuple(p for p in GRAPH_PARTS if p in v) if _format() in (FORMAT, 6) else ())
        + (
            ("share_transition_weighted",)
            if _format() in (5, 6) and "share_transition_weighted" in v
            else ()
        )
    }
    meta = {"format": _format(), "key": key, "targets": sorted(keys), "log_tau": float(log_tau)}
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        save_file(tensors, tmp, metadata={_META: json.dumps(meta)})
        os.replace(tmp, _path(key))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return _path(key)


def state_keys(
    lora_path: str | os.PathLike[str],
    family: str,
    settings: Mapping[str, Any],
    compute: Callable[[], Keys | dict],
) -> Keys:
    key = cache_key(file_sha256(lora_path), family, settings)
    cached = load(key)
    if cached is not None:
        log.info("state keys of %s from the cache", os.path.basename(os.fspath(lora_path)))
        return cached
    keys = compute()
    log_tau = float(getattr(keys, "log_tau", 0.0))
    try:
        store(key, keys, log_tau)
    except (OSError, KeyError) as e:
        log.warning("cannot write the key cache (%s)", e)
    result = Keys(
        {
            t: {p: v[p] for p in PARTS + GRAPH_PARTS + ("share_transition_weighted",) if p in v}
            if _format() in (FORMAT, 6)
            else {p: v[p] for p in PARTS + ("share_transition_weighted",) if p in v}
            for t, v in keys.items()
        }
    )
    result.log_tau = log_tau
    return result


__all__ = ["cache_dir", "cache_key", "file_sha256", "load", "state_keys", "store"]
