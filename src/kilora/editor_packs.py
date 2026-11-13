# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ModFormatError
from .modlist import parse_version, satisfies
from .quant import FORMATS as QUANT_FORMATS
from .quant import dequantize, quantize

PACK_FORMAT = "kilora-editor-pack"
PACK_FORMAT_VERSION = "1.0"
PACK_SUFFIX = ".kedit-pack.safetensors"
MANIFEST_KEY = "kilora.manifest"
ROLES = ("common", "family")
STORAGES = ("float32", *QUANT_FORMATS)
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_DIGEST = re.compile(r"^sha256-[0-9a-f]{64}$")
MAX_PACK_BYTES = 512 * 2**20


@dataclass(frozen=True)
class PackRef:
    id: str
    version: str
    digest: str

    def to_manifest(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version, "digest": self.digest}


@dataclass
class PackManifest:
    id: str
    version: str
    role: str
    editor_kind: str
    editor_version: str
    base_families: list[str]
    families: list[str]
    requires: dict[str, str]
    storage: str
    license: str
    digest: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    path: str | None = field(default=None, repr=False, compare=False)

    @property
    def ref(self) -> PackRef:
        return PackRef(self.id, self.version, self.digest)


def _need(m: Mapping[str, Any], key: str, kind: type, label: str):
    if key not in m:
        raise ModFormatError(f"{label}: the pack manifest has no {key!r}")
    value = m[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ModFormatError(f"{label}: {key!r} must be a {kind.__name__}")
    return value


def parse_manifest(m: Mapping[str, Any], label: str = "pack") -> PackManifest:
    if not isinstance(m, Mapping):
        raise ModFormatError(f"{label}: the pack manifest is not a table")
    if m.get("format") != PACK_FORMAT:
        raise ModFormatError(f"{label}: not an editor pack (format {m.get('format')!r})")
    if m.get("format_version") != PACK_FORMAT_VERSION:
        raise ModFormatError(
            f"{label}: pack format version {m.get('format_version')!r} is not supported. "
            f"Update KiLoRA."
        )
    pack_id = _need(m, "id", str, label)
    if not _ID.match(pack_id):
        raise ModFormatError(f"{label}: the pack ID {pack_id!r} is not valid")
    version = _need(m, "version", str, label)
    if not _VERSION.match(version):
        raise ModFormatError(f"{label}: the version must be X.Y.Z, not {version!r}")
    role = _need(m, "role", str, label)
    if role not in ROLES:
        raise ModFormatError(f"{label}: the role must be one of {', '.join(ROLES)}")
    editor = _need(m, "editor", dict, label)
    kind = editor.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ModFormatError(f"{label}: 'editor.kind' names the editor that reads the pack")
    editor_version = editor.get("version", "")
    if not isinstance(editor_version, str) or not editor_version:
        raise ModFormatError(f"{label}: 'editor.version' is a version range of the editor")
    _range(editor_version, label, "editor.version")
    bases = _need(m, "base_models", list, label)
    base_families = [b.get("family") for b in bases if isinstance(b, dict)]
    if not base_families or not all(isinstance(f, str) and f for f in base_families):
        raise ModFormatError(f"{label}: 'base_models' needs at least one family")
    families = m.get("mod_families", [])
    if not isinstance(families, list) or not all(isinstance(f, str) for f in families):
        raise ModFormatError(f"{label}: 'mod_families' must be a list of family IDs")
    if role == "family" and not families:
        raise ModFormatError(f"{label}: a family pack names its mod families")
    if role == "common" and families:
        raise ModFormatError(f"{label}: a common pack works for every family, so it names none")
    requires = m.get("requires", {})
    if not isinstance(requires, dict):
        raise ModFormatError(f"{label}: 'requires' must be a table of pack ID to version range")
    for rid, spec in requires.items():
        if not isinstance(rid, str) or not _ID.match(rid) or not isinstance(spec, str):
            raise ModFormatError(f"{label}: 'requires' has a bad entry {rid!r}")
        _range(spec, label, f"requires.{rid}")
    if role == "family" and not requires:
        raise ModFormatError(f"{label}: a family pack requires its common pack")
    storage = _need(m, "storage", str, label)
    if storage not in STORAGES:
        raise ModFormatError(f"{label}: the storage must be one of {', '.join(STORAGES)}")
    license_ = _need(m, "license", str, label)
    if (m.get("training") or {}).get("state") != "trained":
        raise ModFormatError(f"{label}: the pack is not trained")
    digest = _need(m, "digest", str, label)
    if not _DIGEST.match(digest):
        raise ModFormatError(f"{label}: the digest must be sha256-<64 hex digits>")
    return PackManifest(
        pack_id,
        version,
        role,
        kind,
        editor_version,
        base_families,
        list(families),
        dict(requires),
        storage,
        license_,
        digest,
        dict(m),
    )


def _range(spec: str, label: str, key: str) -> None:
    try:
        satisfies("0.0.0", spec)
    except ValueError:
        raise ModFormatError(f"{label}: {key!r} is not a version range: {spec!r}") from None


def write_pack(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    exact: Iterable[str] = (),
) -> PackManifest:
    import json

    from .modfile import compute_digest, write_safetensors

    m = {"format": PACK_FORMAT, "format_version": PACK_FORMAT_VERSION, **manifest}
    storage = m.get("storage")
    stored: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    for name, t in tensors.items():
        if "/" in name:
            raise ModFormatError(f"tensor name {name!r}: a pack tensor name has no '/'")
        shapes[name] = list(t.shape)
        keep = any(name == e or (e.endswith("*") and name.startswith(e[:-1])) for e in exact)
        if keep and t.is_floating_point():
            stored[name] = t.float().contiguous()
        elif storage in QUANT_FORMATS and t.is_floating_point():
            for part, value in quantize(t, storage).items():
                stored[f"{name}/{part}"] = value.contiguous()
        else:
            stored[name] = t.contiguous()
    m["tensors"] = shapes
    m["digest"] = compute_digest({k: v for k, v in m.items() if k != "digest"}, stored)
    checked = parse_manifest(m, os.path.basename(os.fspath(path)))
    write_safetensors(path, stored, {MANIFEST_KEY: json.dumps(m, sort_keys=True)})
    return checked


def read_pack(
    path: str | os.PathLike[str], load_tensors: bool = True
) -> tuple[PackManifest, dict[str, torch.Tensor]]:
    import json

    from safetensors import safe_open

    from .modfile import _check_safetensors, _check_sizes, compute_digest

    label = os.path.basename(os.fspath(path))
    if os.path.getsize(path) > MAX_PACK_BYTES:
        raise ModFormatError(f"{label}: larger than {MAX_PACK_BYTES // 2**20} MiB")
    _check_safetensors(path)
    with safe_open(os.fspath(path), framework="pt") as f:
        meta = (f.metadata() or {}).get(MANIFEST_KEY)
        if meta is None:
            raise ModFormatError(f"{label}: no KiLoRA manifest")
        raw = json.loads(meta)
        _check_sizes(raw, 0)
        manifest = parse_manifest(raw, label)
        manifest.path = os.fspath(path)
        stored = {k: f.get_tensor(k) for k in f.keys()}
    if compute_digest(raw, stored) != manifest.digest:
        raise ModFormatError(f"{label}: the digest does not match. The file is damaged.")
    if not load_tensors:
        return manifest, {}
    shapes = raw.get("tensors")
    if not isinstance(shapes, dict):
        raise ModFormatError(f"{label}: the manifest has no 'tensors' table")
    out = {}
    for name, shape in shapes.items():
        if name in stored:
            out[name] = stored[name].float()
            continue
        parts = {k.split("/", 1)[1]: v for k, v in stored.items() if k.split("/", 1)[0] == name}
        if not parts:
            raise ModFormatError(f"{label}: the tensor {name!r} is missing")
        out[name] = dequantize(parts, manifest.storage, tuple(shape))
    return manifest, out


def choose_packs(
    installed: Iterable[PackManifest],
    editor_kind: str,
    editor_version: str | None,
    base_family: str,
    mod_family: str | None,
    pins: Mapping[str, str] | None = None,
) -> list[PackManifest]:
    pins = dict(pins or {})
    fit = [
        p
        for p in installed
        if p.editor_kind == editor_kind
        and (editor_version is None or satisfies(editor_version, p.editor_version))
        and base_family in p.base_families
        and (p.id not in pins or p.version == pins[p.id])
    ]

    def newest(items: Sequence[PackManifest]) -> PackManifest | None:
        return max(items, key=lambda p: parse_version(p.version), default=None)

    common = newest([p for p in fit if p.role == "common"])
    if common is None:
        return []
    chosen = [common]
    if mod_family:
        family = newest(
            [
                p
                for p in fit
                if p.role == "family"
                and mod_family in p.families
                and all(
                    rid == common.id and satisfies(common.version, spec)
                    for rid, spec in p.requires.items()
                )
            ]
        )
        if family is not None:
            chosen.append(family)
    return chosen


PACK_MODES = ("auto", "common only", "pinned")


def _family_packs(
    installed: Iterable[PackManifest],
    common: PackManifest,
    editor_kind: str,
    editor_version: str | None,
    base_family: str,
) -> dict[str, PackManifest]:
    out: dict[str, PackManifest] = {}
    for p in installed:
        if (
            p.role != "family"
            or p.editor_kind != editor_kind
            or (editor_version is not None and not satisfies(editor_version, p.editor_version))
            or base_family not in p.base_families
            or not all(
                rid == common.id and satisfies(common.version, spec)
                for rid, spec in p.requires.items()
            )
        ):
            continue
        for fam in p.families:
            if fam not in out or parse_version(p.version) > parse_version(out[fam].version):
                out[fam] = p
    return out


def select_packs(
    installed: Sequence[PackManifest],
    *,
    mode: str,
    disabled: Iterable[str],
    base_family: str,
    mod_family: str | None,
    editor_kind: str | None = None,
    editor_version: str | None = None,
    pins: Mapping[str, str] | None = None,
    families: Iterable[str] = (),
) -> tuple[list[PackManifest], list[str]]:

    if mode not in PACK_MODES:
        raise ValueError(f"mode must be one of {', '.join(PACK_MODES)}")
    notes = []
    if editor_kind is None:
        kinds = sorted({p.editor_kind for p in installed})
        if len(kinds) > 1:
            return [], [
                f"Packs of several editors are installed ({', '.join(kinds)}). Set editor_kind."
            ]
        editor_kind = kinds[0] if kinds else None
    if editor_kind is None:
        return [], ["No editor pack is installed."]
    use_pins = dict(pins or {}) if mode == "pinned" else None
    wanted = [f.strip() for f in families if f and f.strip()]
    chosen = choose_packs(
        installed,
        editor_kind,
        editor_version,
        base_family,
        None if mode == "common only" or wanted else mod_family,
        pins=use_pins,
    )
    if mode == "auto" and chosen and (wanted or len(chosen) == 1):
        fits = _family_packs(installed, chosen[0], editor_kind, editor_version, base_family)
        if wanted:
            missing = [f for f in wanted if f not in fits]
            if missing:
                notes.append(f"No installed family pack for: {', '.join(missing)}.")
            chosen = [chosen[0], *(fits[f] for f in wanted if f in fits)]
        else:
            chosen = [chosen[0], *(fits[f] for f in sorted(fits))]
    if mode == "pinned" and pins:
        missing = [f"{i} {v}" for i, v in pins.items() if i not in {p.id for p in chosen}]
        if missing:
            notes.append(f"Pinned packs that are not installed: {', '.join(missing)}.")
    off = set(disabled)
    out = [p for p in chosen if p.id not in off]
    if not any(p.role == "common" for p in out):
        notes.append("No common pack is in use, so this editor cannot learn here.")
    return out, notes


def scan_packs(paths: Iterable[str | os.PathLike[str]]) -> tuple[list[PackManifest], list[str]]:
    packs, problems = [], []
    for path in paths:
        try:
            packs.append(read_pack(path, load_tensors=False)[0])
        except (ModFormatError, OSError, ValueError) as exc:
            problems.append(str(exc))
    return packs, problems


def save_record(packs: Iterable[PackManifest]) -> list[dict[str, str]]:
    return [p.ref.to_manifest() for p in packs]


def check_save_record(
    recorded: Sequence[Mapping[str, str]], in_use: Iterable[PackManifest]
) -> list[str]:
    warnings = []
    have = {p.id: p for p in in_use}
    for ref in recorded:
        p = have.get(ref.get("id", ""))
        if p is None:
            warnings.append(
                f"The save was made with the pack {ref.get('id')}, which is not in use."
            )
        elif p.version != ref.get("version") or p.digest != ref.get("digest"):
            warnings.append(
                f"The save was made with {ref.get('id')} {ref.get('version')}. "
                f"Now {p.version} is in use."
            )
    known = {r.get("id") for r in recorded}
    for pid in have:
        if pid not in known:
            warnings.append(f"The pack {pid} is in use, but the save was made without it.")
    return warnings


INDEX_KIND = "editor-pack"


def check_index_entry(name: str, entry: Mapping[str, Any]) -> list[str]:
    problems = []
    if entry.get("kind") != INDEX_KIND:
        problems.append(f"{name}: kind must be {INDEX_KIND!r}")
    if not _VERSION.match(str(entry.get("version", ""))):
        problems.append(f"{name}: version must be X.Y.Z")
    if entry.get("role") not in ROLES:
        problems.append(f"{name}: role must be one of {', '.join(ROLES)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
        problems.append(f"{name}: sha256 must be 64 hex digits")
    if not _DIGEST.match(str(entry.get("digest", ""))):
        problems.append(f"{name}: digest must be the pack's manifest digest")
    size = entry.get("size")
    if not isinstance(size, int) or not 0 < size <= MAX_PACK_BYTES:
        problems.append(f"{name}: size must be the file size in bytes, at most 512 MiB")
    url = str(entry.get("url", ""))
    if not url.endswith(PACK_SUFFIX):
        problems.append(f"{name}: the url must name a {PACK_SUFFIX} file")
    return problems
