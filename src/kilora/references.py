# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import KiLoRAError

logger = logging.getLogger(__name__)


class MissingReferenceError(KiLoRAError, FileNotFoundError):
    pass


def _index_path() -> Path:
    override = os.environ.get("KILORA_HASH_INDEX")
    if override:
        return Path(override).expanduser()
    try:
        from .host import services

        folder_paths = getattr(services().comfy, "folder_paths", None)
        if folder_paths is None:
            import folder_paths
        user = Path(folder_paths.get_user_directory())
    except (ImportError, AttributeError):
        user = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return user / "kilora" / "hash_index.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HashIndex:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else _index_path()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            self.entries = {}

    def hash(self, path: str | os.PathLike[str], *, force: bool = False, write: bool = True) -> str:
        file = Path(path).resolve()
        stat = file.stat()
        entry = self.entries.get(str(file))
        if (
            not force
            and isinstance(entry, dict)
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in entry["sha256"])
        ):
            return entry["sha256"]
        digest = _sha256(file)
        after = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError(f"File changed while hashing: {file}")
        self.entries[str(file)] = {
            "size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": digest,
        }
        if write:
            self._write()
        return digest

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                name = stream.name
                json.dump(self.entries, stream, sort_keys=True)
            os.replace(name, self.path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)


_VERIFIED: dict[str, tuple] = {}


def _stamp(file: Path) -> tuple | None:
    try:
        st = file.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _verified(file: Path, wanted: str) -> bool:
    stamp = _stamp(file)
    return stamp is not None and _VERIFIED.get(str(file)) == (*stamp, wanted)


def resolve(meta: Mapping[str, Any], search_dirs: Iterable[str | os.PathLike[str]]) -> Path:
    wanted = meta["sha256"].lower()
    index = HashIndex()
    candidates = []
    for folder in search_dirs:
        root = Path(folder)
        if root.is_dir():
            candidates.extend(
                file
                for file in sorted(root.rglob("*"))
                if file.suffix.lower() in {".safetensors", ".kmod"} and file.is_file()
            )

    def priority(file: Path) -> int:
        if meta.get("filename") and file.name == meta["filename"]:
            return 0
        try:
            if meta.get("size") is not None and file.stat().st_size == meta["size"]:
                return 1
        except OSError:
            pass
        return 2

    try:
        for file in sorted(candidates, key=priority):
            try:
                if _verified(file.resolve(), wanted):
                    return file
                before = dict(index.entries.get(str(file.resolve())) or {})
                digest = index.hash(file, write=False)
                if digest == wanted:
                    fresh = index.entries.get(str(file.resolve())) != before or not before
                    actual = digest if fresh else index.hash(file, force=True, write=False)
                    if actual != wanted:
                        raise KiLoRAError(
                            f"Hash mismatch for {file}: expected {wanted}, got {actual}"
                        )
                    stamp = _stamp(file.resolve())
                    if stamp is not None:
                        _VERIFIED[str(file.resolve())] = (*stamp, actual)
                    return file
            except OSError as exc:
                logger.warning("Cannot read reference file %s: %s", file, exc)
    finally:
        index._write()
    filename = meta.get("filename") or "unknown file"
    size = meta.get("size")
    size_hint = f"{size / (1024 * 1024):.0f} MB, " if isinstance(size, int) else ""
    source = f" Source: {meta['source_url']}" if meta.get("source_url") else ""
    raise MissingReferenceError(
        f'The mod needs "{filename}" ({size_hint}sha256 {wanted[:12]}...). '
        f"Put it in ComfyUI's loras folder.{source}"
    )
