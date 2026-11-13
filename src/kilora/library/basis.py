# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file

from ..families import _ANIMA_OTHER, anima_never_target_comfy
from ..lora_import import _resolver, _scale
from ..multi_import import _group, lycoris_delta
from ..presets import _ANIMA_LAYERS, anima_name, get_preset
from .direction_files import direction_metadata, direction_source
from .index import LibraryIndex, _atomic

log = logging.getLogger(__name__)


@dataclass
class MountBasis:
    u: torch.Tensor
    v: torch.Tensor
    output_values: torch.Tensor
    input_values: torch.Tensor
    output_sources: list[list[dict]]
    input_sources: list[list[dict]]


@dataclass
class LibraryBasis:
    mounts: dict[str, MountBasis]
    files: dict[str, dict]
    skipped: list[dict]
    flagged: list[dict]
    rank: int
    top: int
    device: str = "cpu"
    _parts: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict, repr=False
    )


def _license_status(metadata):
    fields = {k: v for k, v in metadata.items() if "license" in k.lower() or "usage" in k.lower()}
    for key, value in fields.items():
        if re.search(r"no[\s_-]*derivative|no[\s_-]*merge|(?<![a-z])nd(?![a-z])", value, re.I):
            return f"{key}: {value}", False
    return None, not any("license" in k.lower() and v.strip() for k, v in fields.items())


def _resolve_groups(groups, layout):
    names = [f"transformer_blocks.{i}.{diff}" for i in range(28) for diff, _ in _ANIMA_LAYERS]
    names += [
        f"transformer_blocks.{i}.norm{norm}.linear_{j}"
        for i in range(28)
        for norm in (1, 2, 3)
        for j in (1, 2)
    ]
    names += list(_ANIMA_OTHER) + list(_ANIMA_OTHER.values())
    names += [anima_never_target_comfy(n) for n in names if anima_never_target_comfy(n)]
    for stem in groups:
        if not stem.startswith("lora_unet_"):
            body = stem.removeprefix("diffusion_model.")
            for prefix in ("transformer.", "unet.", "base_model.model."):
                body = body.removeprefix(prefix)
            if body not in names:
                names.append(body)
    resolve = _resolver(layout, names, get_preset("anima").aliases(names), strict=False)
    result = {}
    for stem, (_, keys) in groups.items():
        matches = resolve(stem)
        if len(matches) > 1:
            raise ValueError(f"{stem}: mount maps to multiple targets")
        name = matches[0] if matches else stem.removeprefix("lora_unet_")
        name = anima_name(name) or anima_never_target_comfy(name) or name
        if name in result:
            raise ValueError(f"multiple groups resolve to {name}")
        result[name] = keys
    return result


def _delta(file, keys, device):
    parts = {
        part: file.get_tensor(key).to(device=device, dtype=torch.float32)
        for part, key in keys.items()
    }
    if "down" in parts and "up" in parts:
        down, up = parts["down"], parts["up"]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError("only linear LoRA weights are supported")
        return (up @ down) * _scale(str(keys), parts.get("alpha"), down.shape[0], None)
    result = lycoris_delta(parts)
    if result is None or result[0].ndim != 2:
        raise ValueError(f"unsupported or incomplete mount: {list(keys.values())}")
    return result[0]


def _directions(file, keys, top, device, paths, mount):
    if "down" in keys and "up" in keys:
        paths[mount] = "low_rank_qr"
        down = file.get_tensor(keys["down"]).to(device=device, dtype=torch.float32)
        up = file.get_tensor(keys["up"]).to(device=device, dtype=torch.float32)
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"invalid linear LoRA weights: {list(keys.values())[:5]}")
        alpha = file.get_tensor(keys["alpha"]) if "alpha" in keys else None
        scale = _scale(str(keys), alpha, down.shape[0], None)
        left, left_r = torch.linalg.qr(up, mode="reduced")
        right, right_r = torch.linalg.qr(down.T, mode="reduced")
        u, s, vh = torch.linalg.svd((left_r @ right_r.T) * scale, full_matrices=False)
        u, v = left @ u, right @ vh.T
        n = min(top, up.shape[0], down.shape[1])
        if s.numel() < n:
            missing = n - s.numel()
            u = torch.nn.functional.pad(u, (0, missing))
            v = torch.nn.functional.pad(v, (0, missing))
            s = torch.nn.functional.pad(s, (0, missing))
    else:
        w = _delta(file, keys, device)
        n = min(top, *w.shape)
        if min(w.shape) > 4 * top:
            paths[mount] = "svd_lowrank"
            devices = [w.device] if w.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                u, s, v = torch.svd_lowrank(w, q=min(top + 8, *w.shape), niter=2)
        else:
            paths[mount] = "full_svd"
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            v = vh.T
    return (
        (u[:, :n] * s[:n]).cpu().contiguous(),
        (v[:, :n] * s[:n]).cpu().contiguous(),
    )


def _read(path, top, device, paths=None, source_info=None, *, allow_restricted=False):
    if paths is None:
        paths = {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        record = direction_metadata(metadata)
        if record is not None and source_info is not None:
            source_info.update(direction_source(record, path))
        keys = list(file.keys())
        text_count = sum(k.startswith(("lora_te", "text_encoder")) for k in keys)
        reason, flagged = _license_status(metadata)
        if reason and not allow_restricted:
            return {}, text_count, reason, flagged
        image_keys = [k for k in keys if not k.startswith(("lora_te", "text_encoder"))]
        groups, skipped, layout = _group(dict.fromkeys(image_keys), str(path))
        if skipped:
            raise ValueError(f"unsupported keys: {skipped[:5]}")
        result = {}
        for mount, names in _resolve_groups(groups, layout).items():
            result[mount] = _directions(file, names, top, device, paths, mount)
        return result, text_count, reason, flagged


def _cut(parts, rank, device):
    stack = torch.cat([p.to(device) for _, p in parts], dim=1)
    u, s, _ = torch.linalg.svd(stack, full_matrices=False)
    count = (
        int((s > s[0] * max(stack.shape) * torch.finfo(s.dtype).eps).sum())
        if s.numel() and s[0] > 0
        else 0
    )
    n = min(rank, count)
    u, s = u[:, :n], s[:n]
    sources = []
    for direction in u.T:
        weights = [(name, float((direction @ p.to(device)).square().sum())) for name, p in parts]
        total = sum(weight for _, weight in weights)
        sources.append(
            [
                (name, weight / total)
                for name, weight in sorted(weights, key=lambda x: (-x[1], x[0]))[:3]
            ]
            if total
            else []
        )
    return u.cpu().contiguous(), s.cpu().contiguous(), sources


def _assemble(files, parts, rank, top, device):
    mounts = {}
    for mount in sorted({m for p in parts.values() for m in p}):
        entries = [(name, parts[name][mount]) for name in sorted(parts) if mount in parts[name]]
        shapes = {(p[0].shape[0], p[1].shape[0]) for _, p in entries}
        if len(shapes) != 1:
            raise ValueError(f"{mount}: incompatible library shapes {sorted(shapes)}")
        u, us, u_sources = _cut([(n, p[0]) for n, p in entries], rank, device)
        v, vs, v_sources = _cut([(n, p[1]) for n, p in entries], rank, device)

        def annotate(sources):
            return [
                [
                    {
                        "file": n,
                        "cluster": files[n].get("cluster"),
                        "weight": weight,
                        **files[n].get("direction_source", {}),
                    }
                    for n, weight in row
                ]
                for row in sources
            ]

        mounts[mount] = MountBasis(u, v, us, vs, annotate(u_sources), annotate(v_sources))
    skipped = [{"file": n, "reason": f["reason"]} for n, f in files.items() if f.get("reason")]
    flagged = [
        {"file": n, "reason": "no license metadata"} for n, f in files.items() if f.get("flagged")
    ]
    return LibraryBasis(mounts, files, skipped, flagged, rank, top, str(device), parts)


def _save(basis, path):
    tensors, mapping, part_mapping = {}, {}, {}
    for i, (mount, b) in enumerate(sorted(basis.mounts.items())):
        key = f"mount/{i}"
        mapping[mount] = {
            "key": key,
            "output_sources": b.output_sources,
            "input_sources": b.input_sources,
        }
        for name in ("u", "v", "output_values", "input_values"):
            tensors[f"{key}/{name}"] = getattr(b, name)
    for i, (file, mounts) in enumerate(sorted(basis._parts.items())):
        part_mapping[file] = {}
        for j, (mount, (left, right)) in enumerate(sorted(mounts.items())):
            key = f"part/{i}/{j}"
            part_mapping[file][mount] = key
            tensors[key + "/left"], tensors[key + "/right"] = left, right
    manifest = {
        "format": "kilora-library-basis",
        "version": 1,
        "rank": basis.rank,
        "top": basis.top,
        "files": basis.files,
        "skipped": basis.skipped,
        "flagged": basis.flagged,
        "mounts": mapping,
        "parts": part_mapping,
    }
    path.mkdir(parents=True, exist_ok=True)
    _atomic(path / "basis.safetensors", lambda name: save_file(tensors, name))
    _atomic(
        path / "basis.json",
        lambda name: Path(name).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8"),
    )


def load_basis(index: LibraryIndex, *, device="cpu") -> LibraryBasis:
    data = json.loads((index.path / "basis.json").read_text(encoding="utf-8"))
    if data.get("format") != "kilora-library-basis" or data.get("version") != 1:
        raise ValueError("unsupported library basis format")
    tensors = load_file(str(index.path / "basis.safetensors"))
    parts = {
        f: {m: (tensors[k + "/left"], tensors[k + "/right"]) for m, k in mounts.items()}
        for f, mounts in data["parts"].items()
    }
    mounts = {}
    for name, entry in data["mounts"].items():
        key = entry["key"]
        mounts[name] = MountBasis(
            *(tensors[f"{key}/{n}"] for n in ("u", "v", "output_values", "input_values")),
            entry["output_sources"],
            entry["input_sources"],
        )
    return LibraryBasis(
        mounts,
        data["files"],
        data["skipped"],
        data["flagged"],
        data["rank"],
        data["top"],
        str(device),
        parts,
    )


def build_basis(index: LibraryIndex, *, rank=32, top=16, device="cpu", store=True) -> LibraryBasis:
    if rank < 1 or top < 1:
        raise ValueError("rank and top must be positive")
    return _update(None, index, rank, top, device, store=store)


def update_basis(basis: LibraryBasis, index: LibraryIndex) -> LibraryBasis:
    return _update(basis, index, basis.rank, basis.top, basis.device)


def _update(previous, index, rank, top, device, *, store=True):
    files, parts = {}, {}
    if store and any(
        index.records[d].get("license_flag") == "noderiv" for d in index.paths.values()
    ):
        raise ValueError("cannot store a basis built from noderiv files")
    if (
        store
        and previous is not None
        and any(item.get("license_flag") == "noderiv" for item in previous.files.values())
    ):
        raise ValueError("cannot store a basis built from noderiv files")
    for name, digest in sorted(index.paths.items()):
        started = time.perf_counter()
        record = index.records[digest]
        cluster = record.get("cluster_id")
        restricted = False
        if previous is not None and previous.files.get(name, {}).get("hash") == digest:
            files[name] = dict(previous.files[name], cluster=cluster)
            if name in previous._parts:
                parts[name] = previous._parts[name]
        else:
            paths, source_info = {}, {}
            try:
                if store:
                    value, text_count, reason, flagged = _read(
                        Path(name), top, device, paths, source_info
                    )
                else:
                    value, text_count, reason, flagged = _read(
                        Path(name), top, device, paths, source_info, allow_restricted=True
                    )
                    restricted = bool(reason)
                    if restricted:
                        reason = None
            except (OSError, SafetensorError, ValueError, KeyError) as exc:
                value, text_count, flagged = {}, 0, False
                paths = {}
                reason = f"{type(exc).__name__}: {exc}"
            files[name] = {
                "hash": digest,
                "cluster": cluster,
                "text_encoder_keys": text_count,
                "reason": reason,
                "flagged": flagged,
                "svd_paths": paths,
            }
            if source_info:
                files[name]["direction_source"] = source_info
            if not reason:
                parts[name] = value
        files[name]["license_flag"] = (
            "noderiv" if restricted else record.get("license_flag", files[name].get("license_flag"))
        )
        paths = files[name].get("svd_paths", {})
        log.info(
            "%s: %d mounts, %.3f seconds (low_rank_qr=%d, svd_lowrank=%d, full_svd=%d)",
            Path(name).name,
            len(parts.get(name, {})),
            time.perf_counter() - started,
            sum(path == "low_rank_qr" for path in paths.values()),
            sum(path == "svd_lowrank" for path in paths.values()),
            sum(path == "full_svd" for path in paths.values()),
        )
    result = _assemble(files, parts, rank, top, device)
    if store:
        _save(result, index.path)
    return result
