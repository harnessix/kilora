# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import ModFormatError, UnsupportedFeatureError
from .modfile import read_manifest
from .modlist import LoadedMod, ModChoice

PACK_FORMAT = "kilora-pack"
PACK_VERSION = "1.0"
PACK_FEATURES: dict[str, str] = {}
_MAX_PACK_BYTES = 256 * 1024
_MAX_MODS = 512


@dataclass
class PackEntry:
    mod_id: str
    version: str | None
    digest: str | None
    strength: float = 1.0
    prior: float = 1.0
    enabled: bool = True
    file_sha256: str | None = None
    source: str | None = None


@dataclass
class Pack:
    name: str
    entries: list[PackEntry]
    depth: int | None = None


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack_from_mods(
    name: str,
    mods: Sequence[LoadedMod],
    depth: int | None,
    sources: dict[str, str] | None = None,
) -> Pack:
    entries = [
        PackEntry(
            m.mod_id,
            m.version,
            m.digest,
            strength=m.strength,
            file_sha256=file_sha256(m.path) if m.path else None,
            source=(sources or {}).get(m.mod_id),
        )
        for m in mods
        if m.digest is not None
    ]
    return Pack(name=name, entries=entries, depth=depth)


def save_pack(path: str | os.PathLike[str], pack: Pack) -> dict[str, Any]:
    data = {
        "format": PACK_FORMAT,
        "format_version": PACK_VERSION,
        "requires_features": [],
        "name": pack.name,
        "created_with": f"kilora {__version__}",
        "depth": pack.depth,
        "mods": [
            {
                "id": e.mod_id,
                "version": e.version,
                "digest": e.digest,
                "strength": e.strength,
                "prior": e.prior,
                "enabled": e.enabled,
                "file_sha256": e.file_sha256,
                "source": e.source,
            }
            for e in pack.entries
        ],
    }
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return data


def load_pack(path: str | os.PathLike[str]) -> Pack:
    label = os.path.basename(os.fspath(path))
    if os.path.getsize(path) > _MAX_PACK_BYTES:
        raise ModFormatError(f"{label} is larger than {_MAX_PACK_BYTES} bytes")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModFormatError(f"{label} is not a JSON mod pack ({exc})") from None
    if not isinstance(data, dict) or data.get("format") != PACK_FORMAT:
        raise ModFormatError(f"{label}: 'format' must be {PACK_FORMAT!r}")
    version = str(data.get("format_version", ""))
    major = version.split(".")[0]
    if not major.isdigit():
        raise ModFormatError(f"{label}: 'format_version' must be a 'major.minor' string")
    if int(major) > int(PACK_VERSION.split(".")[0]):
        raise UnsupportedFeatureError(
            f"{label} uses pack format {version}. KiLoRA {__version__} reads pack format "
            f"{PACK_VERSION.split('.')[0]}.x. Update KiLoRA to load it."
        )
    for feature in data.get("requires_features", []):
        if feature not in PACK_FEATURES:
            raise UnsupportedFeatureError(
                f"{label} needs the feature {feature}, which KiLoRA {__version__} does not "
                "support. Update KiLoRA to load it."
            )
    mods = data.get("mods")
    if not isinstance(mods, list) or len(mods) > _MAX_MODS:
        raise ModFormatError(f"{label}: 'mods' must be a list of at most {_MAX_MODS} entries")
    entries = []
    for item in mods:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ModFormatError(f"{label}: each mod entry needs an 'id'")
        entries.append(
            PackEntry(
                mod_id=item["id"],
                version=item.get("version"),
                digest=item.get("digest"),
                strength=float(item.get("strength", 1.0)),
                prior=float(item.get("prior", 1.0)),
                enabled=bool(item.get("enabled", True)),
                file_sha256=item.get("file_sha256"),
                source=item.get("source"),
            )
        )
    depth = data.get("depth")
    if depth is not None and (not isinstance(depth, int) or depth < 0):
        raise ModFormatError(f"{label}: 'depth' must be an integer >= 0 or null")
    return Pack(name=str(data.get("name", label)), entries=entries, depth=depth)


@dataclass
class LibraryItem:
    path: Path
    mod_id: str
    version: str
    digest: str


def scan_library(paths: Iterable[str | os.PathLike[str]]) -> list[LibraryItem]:
    items = []
    for path in paths:
        try:
            m = read_manifest(path)
        except Exception:
            continue
        items.append(LibraryItem(Path(path), m["id"], str(m["version"]), m.get("digest", "")))
    return items


def pack_choices(pack: Pack, library: Sequence[LibraryItem]) -> list[ModChoice]:
    choices = []
    for e in pack.entries:
        found = None
        for item in library:
            if item.mod_id != e.mod_id:
                continue
            if e.digest:
                if item.digest == e.digest:
                    found = item
                    break
                continue
            if e.file_sha256:
                if file_sha256(item.path) == e.file_sha256:
                    found = item
                    break
                continue
            if e.version is None or item.version == e.version:
                found = item
                break
        if found is None:
            want = e.digest or e.version or "any version"
            raise ModFormatError(f"pack {pack.name!r} needs {e.mod_id} ({want}), not found")
        choices.append(
            ModChoice(
                path=found.path,
                strength=e.strength,
                prior=e.prior,
                enabled=e.enabled,
                label=found.path.name,
            )
        )
    return choices
