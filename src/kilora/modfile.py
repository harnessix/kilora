# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import safe_open

from ._version import __version__
from .adapter import Settings
from .adaptive_gate import DEFAULT_KAPPA, check_gate_mode, check_kappa
from .compute import check_routing
from .config import check_depth, check_depth_range
from .errors import ModFormatError, UnsupportedFeatureError
from .host_copy import host_copies, host_copy_map
from .path_walks import check_path_select, check_path_starts, check_path_width
from .roles import parse_policy
from .route_freeze import check_route_freeze
from .sets import PART_TYPES, AdapterSet, KeyedExperts, check_prior, check_strength

MANIFEST_KEY = "kilora.manifest"
NAMESPACE = "kilora/"
ROUTER = NAMESPACE + "router"
PREVIEW_PREFIX = NAMESPACE + "preview/"


def tensor_name(target: str, part: str) -> str:
    if "/" in target:
        raise ValueError(f"target module names cannot contain '/': {target!r}")
    return f"{NAMESPACE}{target}/{part}"


MOD_FORMAT = "kilora-mod"
FORMAT_VERSION = "1.0"
BEHAVIOR_VERSION = 1

FEATURES: dict[str, str] = {
    "experts.lora": "0.1.0",
    "live.state": "0.1.0",
    "storage.int8-block": "0.1.0",
}

_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_STRING = 4096
_MAX_ITEMS = 65536
_MAX_NESTING = 16
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CONTENT_SOURCES = ("original", "licensed", "public-domain", "unknown")


@dataclass
class Mod:
    manifest: dict[str, Any]
    sets: list[AdapterSet]
    settings: Settings
    previews: list[bytes] = field(default_factory=list)
    sections: list = field(default_factory=list)

    @property
    def digest(self) -> str:
        return self.manifest["digest"]


def _check_learned_cross_scope(sets, sections, manifest):
    crosses = [s for s in sections if s.kind == "learned-cross"]
    if crosses and (
        not sets
        or manifest.get("live")
        or any(s.imported_from is not None or s.extra.get("live") for s in sets)
    ):
        raise ValueError("learned cross requires static, non-imported sets")
    if len(crosses) > 1:
        raise ValueError("at most one learned cross section is allowed")
    if crosses:
        from .learned_cross import validate

        data = validate(crosses[0].data)
        if any(
            router.shape[0] != data["bias"].numel()
            for s in sets
            if s.form != "state"
            for router in s.routers.values()
        ):
            raise ValueError("learned cross width must match the set key width")


def save_mod(
    path: str | os.PathLike[str],
    sets: list[AdapterSet],
    settings: Settings,
    *,
    dtype: torch.dtype = torch.float32,
    previews: list[bytes] | None = None,
    extra_tensors: dict[str, torch.Tensor] | None = None,
    storage: str | None = None,
    sections: list | None = None,
    **fields: Any,
) -> dict[str, Any]:

    if not sets and not sections and "edits" not in fields:
        raise ValueError("a mod needs adapter sets or sections")
    for setting in sets:
        if setting.ir_parts:
            part = next(iter(setting.ir_parts))
            raise NotImplementedError(f"cannot save {part} override in a mod yet")
    tensors, adapter = _pack_sets(sets, settings, dtype)
    if storage is not None:
        tensors = _store(tensors, storage)
    _store_state_keys(tensors, sets)
    extra: dict[str, torch.Tensor] = {}
    for name, value in (extra_tensors or {}).items():
        if not name.startswith(NAMESPACE) or name in tensors or name in extra:
            raise ValueError(f"extra tensor name {name!r} is not allowed")
        extra[name] = value
    tensors.update(host_copy_map(extra, lambda v: v.cpu().contiguous()))
    for i, data in enumerate(previews or []):
        tensors[f"{PREVIEW_PREFIX}{i}"] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    manifest = _base_manifest(sets, settings, adapter, fields)
    if storage is not None:
        manifest["requires_features"].append(f"storage.{storage}")
    if sections:
        _check_learned_cross_scope(sets, sections, manifest)
        from .sections import SECTION_KINDS, encode_sections

        section_tensors, records = encode_sections(list(sections))
        tensors.update(section_tensors)
        manifest["sections"] = records
        for kind in sorted({r["kind"] for r in records}):
            manifest["requires_features"].append(SECTION_KINDS[kind].feature)
    _check_manifest(manifest)
    manifest["digest"] = compute_digest(manifest, tensors)
    metadata = {MANIFEST_KEY: json.dumps(manifest, sort_keys=True, separators=(",", ":"))}
    metadata.update(modelspec_fields(manifest))
    write_safetensors(path, tensors, metadata)
    return manifest


STORAGE_FORMATS = ("int8-block",)
INT8_BLOCK = 256


def _blocks_of(width: int) -> int:
    return INT8_BLOCK if width % INT8_BLOCK == 0 else width


def _store(tensors: dict[str, torch.Tensor], storage: str) -> dict[str, torch.Tensor]:
    if storage not in STORAGE_FORMATS:
        raise ValueError(f"unknown storage {storage!r}; known: {', '.join(STORAGE_FORMATS)}")
    out = {}
    for name, t in tensors.items():
        part = name.rsplit("/", 1)[-1]
        if part in ("down", "up") and name.startswith(NAMESPACE):
            x = t.float()
            b = _blocks_of(x.shape[-1])
            blocks = x.reshape(*x.shape[:-1], x.shape[-1] // b, b)
            scale = blocks.abs().amax(-1).clamp(min=1e-30).float() / 127.0
            q = torch.round(blocks / scale[..., None]).clamp(-127, 127)
            out[name] = q.reshape(x.shape).to(torch.int8).contiguous()
            out[f"{name}.scale"] = scale.contiguous()
        elif part in ("keys_in", "keys_out") or name.startswith(ROUTER):
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t
    return out


def _unstore(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        scale = tensors.get(f"{name}.scale")
        if scale is not None:
            b = _blocks_of(t.shape[-1])
            blocks = t.float().reshape(*t.shape[:-1], t.shape[-1] // b, b)
            out[name] = (blocks * scale.float()[..., None]).reshape(t.shape)
        else:
            out[name] = t
    return out


def _store_state_keys(tensors: dict[str, torch.Tensor], sets: list[AdapterSet]) -> None:
    from .compute import _block_bases, derived_rows

    if not any(s.form == "state" for s in sets):
        return
    restored = _unstore(tensors)
    for i, setting in enumerate(sets):
        if setting.form != "state":
            continue
        for target in setting.experts:
            for weight_part, key_part, row_part in (
                ("down", "keys_in", "rows_in"),
                ("up", "keys_out", "rows_out"),
            ):
                weight = restored[tensor_name(target, f"s{i}/{weight_part}")].to(
                    "cpu", torch.float32
                )
                n, routes, rank, width = weight.shape
                keys, rows, _ = _block_bases(
                    weight.reshape(n * routes * rank, width), routes * rank, derived_rows()
                )
                tensors[tensor_name(target, f"s{i}/{key_part}")] = keys.contiguous()
                tensors[tensor_name(target, f"s{i}/{row_part}")] = rows.to(torch.int32)


def _pack_sets(
    sets: list[AdapterSet], settings: Settings, dtype: torch.dtype
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    targets: list[str] = []
    for s in sets:
        targets.extend(t for t in s.target_modules if t not in targets)
    set_records: list[dict[str, Any]] = []
    adapter: dict[str, Any] = {
        "target_modules": targets,
        "softmax_scope": settings.softmax_scope,
        "routing_scope": settings.routing_scope,
        "routing_select": settings.routing_select,
        "carry": settings.carry,
        "iteration": settings.iteration,
        "trained_iterations": _trained_iterations(settings.train_iteration),
        "fixed_point_stop": settings.fixed_point_stop,
        "path_width": settings.path_width,
        "path_select": settings.path_select,
        "route_freeze": settings.route_freeze,
        "path_starts": settings.path_starts,
        "path_prior_min": settings.path_prior_min,
        "path_merge": settings.path_merge,
        "path_beam": settings.path_beam,
        "output_gate": settings.output_gate,
        "gate_mode": settings.gate_mode,
        "gate_kappa": settings.gate_kappa,
        "carry_across_steps_experimental": settings.carry_across_steps,
        "depth": settings.depth,
        "train_depth": list(settings.train_depth) if settings.train_depth else None,
        "last_step_training": settings.last_step_training,
        "routing_noise": settings.routing_noise,
        "seed": settings.seed,
        "sets": set_records,
    }
    if settings.derived_entry_temp is not None:
        adapter["derived_entry_temp"] = settings.derived_entry_temp
    adapter["import_score"] = settings.import_score
    if settings.candidate_summary != "keys":
        adapter["candidate_summary"] = settings.candidate_summary
    if settings.candidates:
        adapter["candidates"] = settings.candidates
    if settings.token_roles != "auto":
        adapter["token_roles"] = settings.token_roles
    for i, s in enumerate(sets):
        for width, router in s.routers.items():
            tensors[f"{ROUTER}/{i}/{width}"] = router.to(dtype)
        for target, ex in s.experts.items():
            for part in ("down", "up", "keys_in", "keys_out"):
                value = getattr(ex, part)
                if value is not None:
                    tensors[tensor_name(target, f"s{i}/{part}")] = value.to(dtype)
            for part in ("routing_down", "routing_up"):
                value = getattr(ex, part)
                if value is not None:
                    tensors[tensor_name(target, f"s{i}/{part}")] = value.to(dtype)
            for name, value in (ex.gate or {}).items():
                tensors[tensor_name(target, f"s{i}/gate_{name}")] = value.float()
            for expert, components in ex.exact.items():
                for part, value in enumerate(components):
                    tensors[tensor_name(target, f"s{i}/exact_{expert}_{part}")] = value.float()
            for part in ("rows_in", "rows_out"):
                tensors[tensor_name(target, f"s{i}/{part}")] = getattr(ex, part).to(torch.int32)
        record: dict[str, Any] = {
            "name": s.name,
            "part_type": list(s.part_type),
            "aggregation": s.aggregation,
            "form": s.form,
            "strength": s.strength,
            "prior": s.prior,
            "log_tau": float(s.log_tau),
            "key_source": s.key_source,
            "experts": s.num_experts,
            "routes": s.routes,
            "rank": s.rank,
            "router_dim": s.router_dim,
            "targets": list(s.target_modules),
        }
        if s.imported_from is not None:
            record["imported_from"] = dict(s.imported_from)
            record["trained_with_kilora"] = bool(s.extra.get("trained_with_kilora", False))
        if "axis" in s.extra:
            from .compute import RECORD_AXES

            if s.extra["axis"] not in RECORD_AXES:
                raise ValueError(f"unknown set axis {s.extra['axis']!r}")
            record["axis"] = s.extra["axis"]
        if s.extra.get("live"):
            record["live"] = True
        if s.extra.get("learned"):
            record["learned"] = True
        if s.frozen:
            record["frozen"] = True
        set_records.append(record)
    return tensors, adapter


def _base_manifest(
    sets: list[AdapterSet], settings: Settings, adapter: dict[str, Any], fields: dict[str, Any]
) -> dict[str, Any]:
    features = ["experts.lora"]
    trained = list(settings.train_depth) if settings.train_depth else [settings.depth] * 2
    default_name = sets[0].name if sets else "mod"
    default_id = re.sub(r"[^a-z0-9._-]", "-", default_name.lower())[:64].strip("-") or "mod"
    manifest: dict[str, Any] = {
        "format": MOD_FORMAT,
        "format_version": FORMAT_VERSION,
        "requires_features": features,
        "extensions": {},
        "behavior_version": BEHAVIOR_VERSION,
        "id": default_id,
        "name": default_name,
        "version": "0.1.0",
        "author": "",
        "license": "NOASSERTION",
        "content_source": "unknown",
        "base_model": {"family": "unknown"},
        "engine": ">=0.1",
        "strength": {"recommended": 1.0, "min": -2.0, "max": 2.0},
        "depth": {"recommended": settings.depth, "trained": trained},
        "previews": [],
        "created_with": f"kilora {__version__}",
        "adapter": adapter,
    }
    for key, value in fields.items():
        if key in ("format", "format_version", "digest", "adapter", "requires_features"):
            raise ValueError(f"manifest field {key!r} is set by KiLoRA")
        manifest[key] = value
    live = manifest.get("live")
    if live:
        features.append("live.state")
    return manifest


NOTICE = (
    "KiLoRA mod. Load it with a KiLoRA node (ComfyUI) or the kilora Python package. "
    "Plain LoRA loaders cannot use this file."
)
IMPLEMENTATION = "https://github.com/harnessix/kilora"


def modelspec_fields(manifest: dict[str, Any]) -> dict[str, str]:
    family = (manifest.get("base_model") or {}).get("family", "unknown")
    strength = (manifest.get("strength") or {}).get("recommended")
    description = str(manifest.get("description", "")).strip()
    hint = "Needs a KiLoRA node (ComfyUI) or the kilora Python package."
    if strength is not None:
        hint += f" Recommended strength: {strength:g}."
    fields = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": f"{family}/kilora",
        "modelspec.implementation": IMPLEMENTATION,
        "modelspec.title": str(manifest.get("name", manifest.get("id", ""))),
        "modelspec.description": (description + "\n\n" if description else "") + NOTICE,
        "modelspec.usage_hint": hint,
        "kilora.notice": NOTICE,
    }
    if manifest.get("author"):
        fields["modelspec.author"] = str(manifest["author"])
    if manifest.get("license"):
        fields["modelspec.license"] = str(manifest["license"])
    words = manifest.get("trigger_words") or []
    if words:
        fields["modelspec.trigger_phrase"] = ", ".join(words)
    tags = ["kilora"] + ([manifest["mod_family"]] if manifest.get("mod_family") else [])
    fields["modelspec.tags"] = ",".join(tags)
    return fields


_DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}


def _layout(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, Any], list[bytes]]:
    entries: dict[str, Any] = {}
    chunks = []
    offset = 0
    names = sorted(tensors)
    host = host_copies((tensors[n] for n in names), lambda v: v.cpu().contiguous())
    for name, t in zip(names, host, strict=True):
        if t.dtype not in _DTYPES:
            raise ValueError(f"tensor {name}: dtype {t.dtype} cannot be written")
        data = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        entries[name] = {
            "dtype": _DTYPES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    return entries, chunks


def _header(entries: dict[str, Any], metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items())), **entries}
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def _serialize(tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> bytes:
    entries, chunks = _layout(tensors)
    return _header(entries, metadata) + b"".join(chunks)


def write_safetensors(
    path: str | os.PathLike[str], tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:

    entries, chunks = _layout(tensors)
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    meta = {**metadata, "modelspec.hash_sha256": "0x" + h.hexdigest()}
    with open(path, "wb") as f:
        f.write(_header(entries, meta))
        for chunk in chunks:
            f.write(chunk)


def compute_digest(manifest: dict[str, Any], tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    body = {k: v for k, v in manifest.items() if k != "digest"}
    h.update(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    names = sorted(tensors)
    host = host_copies((tensors[n] for n in names), lambda v: v.contiguous().cpu())
    for name, t in zip(names, host, strict=True):
        h.update(f"\n{name}|{str(t.dtype).removeprefix('torch.')}|{list(t.shape)}\n".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return "sha256-" + h.hexdigest()


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    _check_safetensors(path)
    with safe_open(os.fspath(path), framework="pt") as f:
        metadata = f.metadata() or {}
    raw = metadata.get(MANIFEST_KEY)
    if raw is None:
        raise ModFormatError(f"{_label(path)} has no KiLoRA manifest; it is not a mod file")
    if len(raw.encode()) > _MAX_MANIFEST_BYTES:
        raise ModFormatError(f"{_label(path)}: manifest is larger than {_MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModFormatError(f"{_label(path)}: manifest is not valid JSON ({exc})") from None
    check_support(manifest, _label(path))
    _check_manifest(manifest)
    return manifest


def load_mod(path: str | os.PathLike[str], *, verify: bool = True) -> Mod:

    manifest = read_manifest(path)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(os.fspath(path), framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    if verify and compute_digest(manifest, tensors) != manifest.get("digest"):
        raise ModFormatError(f"{_label(path)}: digest does not match; the file is damaged")
    if "storage.int8-block" in manifest.get("requires_features", []):
        tensors = _unstore(tensors)
    try:
        sets, settings = _unpack_sets(manifest["adapter"], tensors)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModFormatError(f"{_label(path)}: adapter data is invalid ({exc})") from None
    if not sets and not manifest.get("sections") and "edits" not in manifest:
        raise ModFormatError(f"{_label(path)}: the mod has no adapter sets or sections")
    previews = []
    i = 0
    while f"{PREVIEW_PREFIX}{i}" in tensors:
        previews.append(tensors[f"{PREVIEW_PREFIX}{i}"].numpy().tobytes())
        i += 1
    sections = []
    if manifest.get("sections"):
        from .sections import decode_sections

        try:
            sections = decode_sections(manifest["sections"], tensors)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModFormatError(f"{_label(path)}: section data is invalid ({exc})") from None
    try:
        _check_learned_cross_scope(sets, sections, manifest)
    except ValueError as exc:
        raise ModFormatError(f"{_label(path)}: {exc}") from None
    from .ir.mods import apply_mod_parts

    sets = apply_mod_parts(sets, sections)
    cross = next((section.data for section in sections if section.kind == "learned-cross"), None)
    for adapter_set in sets:
        adapter_set.learned_cross = cross
        adapter_set.learned_cross_zero = cross is not None and not (
            torch.count_nonzero(cross["up"]).item() or torch.count_nonzero(cross["bias"]).item()
        )
    return Mod(
        manifest=manifest, sets=sets, settings=settings, previews=previews, sections=sections
    )


def check_support(manifest: dict[str, Any], label: str = "this mod") -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != MOD_FORMAT:
        raise ModFormatError(f"{label}: 'format' must be {MOD_FORMAT!r}")
    version = manifest.get("format_version")
    match = re.fullmatch(r"(\d+)\.(\d+)", str(version))
    if not isinstance(version, str) or match is None:
        raise ModFormatError(f"{label}: 'format_version' must be a 'major.minor' string")
    if int(match.group(1)) > int(FORMAT_VERSION.split(".")[0]):
        raise UnsupportedFeatureError(
            f"{label} uses mod format {version}. KiLoRA {__version__} reads mod format "
            f"{FORMAT_VERSION.split('.')[0]}.x. Update KiLoRA to load it."
        )
    required = manifest.get("requires_features", [])
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
        raise ModFormatError(f"{label}: 'requires_features' must be a list of strings")
    for feature in required:
        if feature not in FEATURES:
            raise UnsupportedFeatureError(
                f"{label} needs the feature {feature}, which KiLoRA {__version__} does not "
                "support. Update KiLoRA to load it."
            )
    behavior = manifest.get("behavior_version", 1)
    if not isinstance(behavior, int) or behavior < 1:
        raise ModFormatError(f"{label}: 'behavior_version' must be an integer >= 1")
    if behavior > BEHAVIOR_VERSION:
        raise UnsupportedFeatureError(
            f"{label} uses behavior version {behavior}. KiLoRA {__version__} supports "
            f"behavior versions up to {BEHAVIOR_VERSION}. Update KiLoRA to load it."
        )


def _unpack_sets(
    adapter: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> tuple[list[AdapterSet], Settings]:
    train_depth = adapter.get("train_depth")
    settings = Settings(
        import_score=adapter.get(
            "import_score",
            "derived"
            if any(r.get("form", "state") == "state" for r in adapter["sets"])
            else "keys",
        ),
        derived_entry_temp=adapter.get("derived_entry_temp"),
        depth=check_depth(int(adapter["depth"])),
        train_depth=check_depth_range(train_depth) if train_depth else None,
        last_step_training=bool(adapter.get("last_step_training", False)),
        routing_noise=float(adapter.get("routing_noise", 0.0)),
        softmax_scope=adapter.get("softmax_scope", "shared"),
        routing_scope=adapter["routing_scope"],
        routing_select=adapter["routing_select"],
        candidates=adapter.get("candidates", 0),
        candidate_summary=adapter.get("candidate_summary", "keys"),
        carry=adapter["carry"],
        iteration=adapter["iteration"],
        train_iteration=_train_iteration(adapter["trained_iterations"]),
        fixed_point_stop=bool(adapter["fixed_point_stop"]),
        path_width=check_path_width(adapter.get("path_width", 1)),
        path_select=check_path_select(adapter.get("path_select", "converged")),
        route_freeze=check_route_freeze(adapter.get("route_freeze")),
        path_starts=check_path_starts(adapter.get("path_starts", "top")),
        path_prior_min=float(adapter.get("path_prior_min", 0.0)),
        path_merge=bool(adapter.get("path_merge", False)),
        path_beam=bool(adapter.get("path_beam", False)),
        output_gate=adapter["output_gate"],
        gate_mode=check_gate_mode(adapter.get("gate_mode", "learned")),
        gate_kappa=check_kappa(adapter.get("gate_kappa", DEFAULT_KAPPA)),
        carry_across_steps=bool(adapter.get("carry_across_steps_experimental", False)),
        seed=int(adapter.get("seed", 0)),
        token_roles=parse_policy(adapter.get("token_roles", "auto")),
    )
    if settings.candidate_summary not in ("keys", "read_map"):
        raise ValueError("candidate_summary must be keys or read_map")
    if (
        not isinstance(settings.candidates, int)
        or isinstance(settings.candidates, bool)
        or settings.candidates < 0
    ):
        raise ValueError("candidates must be a nonnegative integer")
    if settings.softmax_scope not in ("shared", "per_set"):
        raise ValueError(f"unknown softmax_scope {settings.softmax_scope!r}")
    check_routing(settings.routing_scope, settings.routing_select)
    if settings.iteration not in ("accumulate", "path"):
        raise ValueError(f"unknown iteration {settings.iteration!r}")
    if settings.carry not in ("drop", "dropped"):
        raise ValueError(f"unknown carry {settings.carry!r}")
    sets = []
    for i, record in enumerate(adapter["sets"]):
        part_type = tuple(record["part_type"])
        routers = {}
        prefix = f"{ROUTER}/{i}/"
        for name, value in tensors.items():
            if name.startswith(prefix):
                routers[int(name[len(prefix) :])] = value.float()
        if any(p not in PART_TYPES for p in part_type):
            raise ValueError(f"set {record['name']!r} has part types {list(part_type)}")
        experts: dict[str, Any] = {}
        needed = ["keys_in", "keys_out", "rows_in", "rows_out", "down", "up"]
        for target in record["targets"]:
            parts = {}
            for part in needed:
                name = tensor_name(target, f"s{i}/{part}")
                if name not in tensors:
                    raise ValueError(f"set {record['name']!r} is missing the tensor {name!r}")
                value = tensors[name]
                parts[part] = value.long() if part.startswith("rows") else value.float().clone()
            for part in ("routing_down", "routing_up"):
                name = tensor_name(target, f"s{i}/{part}")
                if name in tensors:
                    parts[part] = tensors[name].float().clone()
            prefix = tensor_name(target, f"s{i}/gate_")
            gate = {
                n[len(prefix) :]: v.float().clone()
                for n, v in tensors.items()
                if n.startswith(prefix)
            }
            exact_prefix = tensor_name(target, f"s{i}/exact_")
            exact_parts: dict[int, dict[int, torch.Tensor]] = {}
            for name, value in tensors.items():
                if name.startswith(exact_prefix):
                    expert, part = (int(v) for v in name[len(exact_prefix) :].split("_"))
                    exact_parts.setdefault(expert, {})[part] = value.float().clone()
            exact = {}
            for expert, components in exact_parts.items():
                if (
                    expert < 0
                    or expert >= parts["down"].shape[0]
                    or set(components) not in ({0, 1}, {0, 1, 2, 3})
                ):
                    raise ValueError(
                        f"set {record['name']!r}: invalid exact components for {target}"
                    )
                exact[expert] = tuple(components[j] for j in range(len(components)))
            experts[target] = KeyedExperts(gate=gate or None, exact=exact, **parts)
        extra: dict[str, Any] = {k: True for k in ("live", "learned") if record.get(k)}
        if "axis" in record:
            from .compute import RECORD_AXES

            if record["axis"] not in RECORD_AXES:
                raise ValueError(f"unknown set axis {record['axis']!r}")
            extra["axis"] = record["axis"]
        if "trained_with_kilora" in record:
            extra["trained_with_kilora"] = bool(record["trained_with_kilora"])
        sets.append(
            AdapterSet(
                name=record["name"],
                part_type=part_type,
                experts=experts,
                routers=routers,
                aggregation=record.get("aggregation", "accumulate"),
                form=record.get("form", "state"),
                strength=check_strength(record.get("strength", 1.0)),
                prior=check_prior(record.get("prior", 1.0)),
                imported_from=record.get("imported_from"),
                log_tau=float(record["log_tau"]),
                key_source=record.get("key_source", "random"),
                frozen=bool(record.get("frozen", False)),
                extra=extra,
            )
        )
    return sets, settings


def _check_safetensors(path: str | os.PathLike[str]) -> None:
    label = _label(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(9)
    if len(head) < 9:
        raise ModFormatError(f"{label} is too small to be a safetensors file")
    (length,) = struct.unpack("<Q", head[:8])
    if length > _MAX_HEADER_BYTES or 8 + length > size or head[8:9] != b"{":
        raise ModFormatError(
            f"{label} is not a safetensors file. KiLoRA only loads safetensors files, "
            "because other formats such as pickle can run code when they load."
        )


def _check_manifest(manifest: dict[str, Any]) -> None:
    _check_sizes(manifest, 0)
    for key in ("id", "name", "version", "license", "base_model", "adapter"):
        if key not in manifest:
            raise ModFormatError(f"manifest field {key!r} is missing")
    if not isinstance(manifest["id"], str) or not _ID_PATTERN.match(manifest["id"]):
        raise ModFormatError(
            "manifest 'id' must be 1 to 64 lowercase letters, digits, '.', '_' or '-'"
        )
    for key in ("name", "version", "license"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ModFormatError(f"manifest field {key!r} must be a non-empty string")
    source = manifest.get("content_source", "unknown")
    if source not in _CONTENT_SOURCES:
        raise ModFormatError(f"manifest 'content_source' must be one of {_CONTENT_SOURCES}")
    base = manifest["base_model"]
    if not isinstance(base, dict) or not isinstance(base.get("family"), str):
        raise ModFormatError("manifest 'base_model' must be a table with a 'family' string")
    if not isinstance(manifest.get("extensions", {}), dict):
        raise ModFormatError("manifest 'extensions' must be a table")
    adapter = manifest["adapter"]
    if not isinstance(adapter, dict) or not isinstance(adapter.get("sets"), list):
        raise ModFormatError("manifest 'adapter' must be a table with a 'sets' list")
    for key in ("trigger_words", "example_prompts"):
        value = manifest.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ModFormatError(f"manifest {key!r} must be a list of strings")
    if not isinstance(manifest.get("description", ""), str):
        raise ModFormatError("manifest 'description' must be a string")
    strength = manifest.get("strength", {})
    if not isinstance(strength, dict) or not all(
        isinstance(strength.get(k, 0.0), (int, float)) for k in ("recommended", "min", "max")
    ):
        raise ModFormatError("manifest 'strength' must hold numbers")


def _check_sizes(value: Any, depth: int) -> None:
    if depth > _MAX_NESTING:
        raise ModFormatError(f"manifest is nested deeper than {_MAX_NESTING} levels")
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ModFormatError(f"a manifest string is longer than {_MAX_STRING} characters")
    elif isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            raise ModFormatError(f"a manifest table has more than {_MAX_ITEMS} entries")
        for k, v in value.items():
            _check_sizes(k, depth + 1)
            _check_sizes(v, depth + 1)
    elif isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise ModFormatError(f"a manifest list has more than {_MAX_ITEMS} entries")
        for v in value:
            _check_sizes(v, depth + 1)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ModFormatError(f"manifest values must be JSON data, got {type(value).__name__}")


def _label(path: str | os.PathLike[str]) -> str:
    return os.path.basename(os.fspath(path))


def _trained_iterations(train_iteration: str) -> list[str]:
    return ["accumulate", "path"] if train_iteration == "mix" else [train_iteration]


def _train_iteration(modes: list[str]) -> str:
    if sorted(modes) == ["accumulate", "path"]:
        return "mix"
    if modes in (["accumulate"], ["path"]):
        return modes[0]
    raise ValueError(f"unknown trained_iterations {modes!r}")


from . import sections as _sections
