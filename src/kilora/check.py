# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from safetensors import safe_open

from .errors import KiLoRAError, ModFormatError
from .modfile import MANIFEST_KEY, load_mod
from .presets import PRESETS

LARGE_FILE = 4 * 1024**3
_HASH = re.compile(r"^(sha256[-:])?[0-9a-f]{64}$")
_NO_LICENSE = ("", "NOASSERTION", None)


@dataclass
class Report:
    path: str
    kind: str = "unknown"
    facts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def lines(self) -> list[str]:
        name = os.path.basename(self.path)
        head = f"{name}: {self.kind}, " + ("OK" if self.ok else "NOT USABLE")
        out = [head, *(f"  {f}" for f in self.facts)]
        out += [f"  warning: {w}" for w in self.warnings]
        out += [f"  problem: {p}" for p in self.problems]
        return out


def _size_text(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _manifest(path: str) -> dict[str, Any] | None:
    with safe_open(path, framework="pt") as f:
        raw = (f.metadata() or {}).get(MANIFEST_KEY)
    return None if raw is None else json.loads(raw)


def _base_model(r: Report, families: list[str], hashes: list[Any]) -> None:
    known = [f for f in families if f in PRESETS]
    if not families or families == ["unknown"]:
        r.warnings.append("The file does not say which base model it is for.")
    elif not known:
        r.warnings.append(
            f"This KiLoRA release has no preset for the base model {', '.join(families)}."
        )
    r.facts.append(f"base model: {', '.join(families) or 'not given'}")
    for h in hashes:
        if not isinstance(h, str) or not _HASH.match(h):
            r.problems.append(f"The base model hash {h!r} is not a SHA-256 hash.")


def _license(r: Report, value: Any) -> None:
    if value in _NO_LICENSE:
        r.warnings.append("The license field is empty. Others cannot know if they can use it.")
    else:
        r.facts.append(f"license: {value}")


def _hooks(r: Report, targets: list[str]) -> None:
    if not targets:
        r.problems.append("The file names no hook layers.")
        return
    bad = [t for t in targets if not isinstance(t, str) or "/" in t or not t]
    if bad:
        r.problems.append(f"The hook layer name {bad[0]!r} is not valid.")
    r.facts.append(f"hook layers: {len(targets)} (first: {targets[0]})")


def _check_mod(r: Report, path: str) -> dict[str, Any]:
    mod = load_mod(path)
    m = mod.manifest
    live = m.get("live") if isinstance(m.get("live"), dict) else None
    r.kind = "live mod" if live else "static mod"
    r.facts.append(f"{m.get('name')} ({m.get('id')} {m.get('version')})")
    r.facts.append(f"format {m['format_version']}, digest matches")
    base = m.get("base_model") or {}
    hashes = [base[k] for k in ("sha256", "hash") if k in base]
    _base_model(r, [base.get("family", "unknown")], hashes)
    _hooks(r, list(m["adapter"]["target_modules"]))
    _license(r, m.get("license"))
    _keyed_sets(r, mod)
    if live:
        spec = live.get("layout_spec") or {}
        missing = [t for t in spec.get("targets", []) if t not in m["adapter"]["target_modules"]]
        if missing:
            r.problems.append(f"The live layer {missing[0]} is not one of the mod's hook layers.")
        r.facts.append(f"live layers: {len(spec.get('targets', []))}")
    return m


def _keyed_sets(r: Report, mod: Any) -> None:
    for s in mod.sets:
        if "lora" not in s.part_type or s.form != "state" or s.imported_from:
            continue
        if s.num_experts < 2:
            r.warnings.append(
                f"The set {s.name} has {s.num_experts} expert per layer. A static KiLoRA mod "
                "needs 2 or more, chosen by their read subspaces. Train it again with more experts."
            )


def _check_editor(r: Report, path: str, manifest: dict[str, Any]) -> dict[str, Any]:
    kind = manifest.get("kind")
    from .editors import editor_for, plugin_for_kind

    plugin = plugin_for_kind(kind)
    if plugin is None:
        raise ModFormatError(
            f"{os.path.basename(path)}: no installed editor plugin reads the kind '{kind}'"
        )
    info = plugin.inspect(path) if plugin.inspect is not None else None
    if info is None:
        editor_for(path)
    r.kind = info.label if info else f"editor ({kind})"
    targets = list(info.targets) if info else list(manifest.get("targets") or [])
    if info:
        r.facts.extend(info.facts)
    if not targets:
        with safe_open(path, framework="pt") as f:
            names = list(f.keys())
        targets = sorted(k.split("/")[-2] for k in names if k.endswith("/v0"))
    if "license" in manifest:
        _license(r, manifest["license"])
    r.facts.append(
        f"{manifest.get('id')} {manifest.get('version')}, format {manifest.get('format_version')}"
    )
    families = [b.get("family", "unknown") for b in manifest.get("base_models") or []]
    _base_model(
        r, families, [b["sha256"] for b in manifest.get("base_models") or [] if "sha256" in b]
    )
    _hooks(r, targets)
    support = editor_support(manifest, targets)
    r.facts.append(
        f"supports live rank {support.rank_min} to {support.rank_max}, "
        f"A columns {support.columns_min} to {support.columns_max}, "
        f"{len(support.hooks)} hook points"
        + ("" if "supports" in manifest else " (" + NO_SUPPORTS + ")")
    )
    manifest = dict(manifest)
    manifest["_targets"] = targets
    manifest["_support"] = support
    manifest["_v0_ranks"] = _v0_ranks(path)
    return manifest


DEFAULT_EDITOR_RANK = 4
NO_SUPPORTS = "without supports, the editor is treated as live rank 4"


def editor_support(manifest: dict[str, Any], targets: list[str]) -> Any:
    from .editors import EditorSupport

    if "supports" in manifest:
        return EditorSupport.from_manifest(manifest["supports"])
    rank = DEFAULT_EDITOR_RANK
    return EditorSupport(rank_min=rank, rank_max=rank, hooks=list(targets))


def _v0_ranks(path: str) -> dict[str, int]:
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.endswith("/v0"):
                out[k.split("/")[-2]] = int(f.get_slice(k).get_shape()[0])
    return out


def mod_live_shape(m: dict[str, Any], mod_path: str) -> tuple[dict[str, int], dict[str, int]]:
    live = m.get("live") or {}
    mod = load_mod(mod_path)
    sets = [s for s in mod.sets if s.name == live.get("live_set")]
    columns = {}
    if sets:
        for t, ex in sets[0].experts.items():
            columns[t] = int(ex.down.numel() // ex.down.shape[-1])
    rank = live.get("rank")
    ranks = {t: int(rank) for t in columns} if rank is not None else {}
    return ranks, columns


def _check_save(r: Report, path: str) -> dict[str, Any]:
    from .live import read_save

    save = read_save(path)
    m = save.manifest
    r.kind = "live state (save)"
    mod = m.get("mod") or {}
    r.facts.append(f"for mod {mod.get('id')} {mod.get('version')}, layout {m.get('layout')}")
    r.facts.append(f"{len(m.get('commits', []))} commits, format {m.get('format_version')}")
    _hooks(r, sorted(save.state))
    return m


def check_file(
    path: str | os.PathLike[str], *, mod: str | os.PathLike[str] | None = None
) -> Report:
    path = os.fspath(path)
    r = Report(path)
    if not os.path.isfile(path):
        r.problems.append("The file does not exist.")
        return r
    size = os.path.getsize(path)
    r.facts.append(f"size: {_size_text(size)}")
    if size == 0:
        r.problems.append("The file is empty.")
        return r
    if size > LARGE_FILE:
        r.warnings.append("The file is larger than 4 GB.")
    try:
        manifest = _manifest(path)
    except Exception:
        r.problems.append("This is not a safetensors file. KiLoRA files are safetensors files.")
        return r
    if manifest is None:
        r.problems.append("The file has no KiLoRA manifest. It is not a KiLoRA file.")
        return r
    fmt = manifest.get("format")
    try:
        if fmt == "kilora-mod":
            info = _check_mod(r, path)
        elif fmt == "kilora-editor":
            info = _check_editor(r, path, manifest)
        elif fmt == "kilora-save":
            info = _check_save(r, path)
        else:
            r.problems.append(f"Unknown file format {fmt!r}.")
            return r
        if mod is not None and fmt in ("kilora-editor", "kilora-save"):
            _check_fit(r, fmt, info, os.fspath(mod))
    except KiLoRAError as exc:
        r.problems.append(_sentence(str(exc)))
    except (KeyError, TypeError, ValueError) as exc:
        r.problems.append(f"The manifest is not valid ({exc}).")
    return r


def _check_fit(r: Report, fmt: str, info: dict[str, Any], mod_path: str) -> None:
    m = load_mod(mod_path).manifest
    live = m.get("live") if isinstance(m.get("live"), dict) else None
    name = os.path.basename(mod_path)
    if live is None:
        r.problems.append(f"{name} is a static mod. Editors and saves are for live mods.")
        return
    spec = live.get("layout_spec") or {}
    if fmt == "kilora-save":
        if info.get("layout") != spec.get("id"):
            r.problems.append(
                f"The save uses layout {info.get('layout')}, but {name} uses {spec.get('id')}."
            )
        elif (info.get("mod") or {}).get("digest") != m.get("digest"):
            r.warnings.append(f"The save was made with another version of {name}.")
        else:
            r.facts.append(f"fits {name}")
        return
    family = (m.get("base_model") or {}).get("family")
    families = [b.get("family") for b in info.get("base_models") or []]
    if families and family not in families:
        r.problems.append(f"The editor is for {', '.join(families)}, but {name} is for {family}.")
    want = set(spec.get("targets", []))
    have = set(info.get("_targets", []))
    if have and want and have != want:
        first = sorted(want ^ have)[0]
        r.problems.append(f"The editor and {name} differ at the hook layer {first}.")
    from .editors import check_mod

    ranks, columns = mod_live_shape(m, mod_path)
    if not ranks:
        v0 = info.get("_v0_ranks") or {}
        ranks = {t: v0[t] for t in columns if t in v0}
    for message in check_mod(info["_support"], ranks, columns):
        r.problems.append(message)
    if not r.problems:
        r.facts.append(f"fits {name}")


def _sentence(text: str) -> str:
    text = text.strip()
    text = text[0].upper() + text[1:] if text else text
    return text if text.endswith(".") else text + "."
