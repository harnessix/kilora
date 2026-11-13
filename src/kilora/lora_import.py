# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from .base_weights import read_base_weight
from .compute import orthonormal_keys
from .errors import LoRAImportError, ModFormatError
from .presets import Alias, get_preset
from .sets import AdapterSet, KeyedExperts, make_router

_TEXT_ENCODER_PREFIXES = (
    "lora_te_",
    "lora_te1_",
    "lora_te2_",
    "lora_te3_",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoders.",
    "te.",
    "te1.",
    "te2.",
    "cond_stage_model.",
    "conditioner.",
)
_MODEL_PREFIXES = ("transformer.", "unet.", "base_model.model.")

_WEIGHT_SUFFIXES = {
    ".lora_A.weight": "down",
    ".lora_B.weight": "up",
    ".lora_A.default.weight": "down",
    ".lora_B.default.weight": "up",
    ".lora.down.weight": "down",
    ".lora.up.weight": "up",
    ".lora_down.weight": "down",
    ".lora_up.weight": "up",
    ".alpha": "alpha",
}


@dataclass
class LoRAImport:
    adapter_set: AdapterSet
    layout: str
    scale: float | None
    skipped_keys: list[str] = field(default_factory=list)
    excluded_keys: list[str] = field(default_factory=list)
    sections: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"Imported {len(self.adapter_set.experts)} target modules "
            f"as adapter set {self.adapter_set.name!r} ({self.layout} layout).",
            f"Strength: {self.adapter_set.strength:g}.",
        ]
        source = self.adapter_set.imported_from or {}
        if "size_scale" in source:
            from .size_norm import size_note

            lines.append(
                size_note(
                    self.adapter_set.name,
                    source["size"],
                    source["size_scale"],
                    source.get("applied_scale"),
                    source.get("size_rule"),
                )
            )
        if self.scale is None:
            lines.append("Scales differ between modules; each module scale is part of its weights.")
        lines.append(
            "This set has not been trained with KiLoRA. Train it before sharing it as a KiLoRA mod."
        )
        te = sum(int(x.meta.get("layers", 0)) for x in self.sections if x.kind == "text-encoder")
        if te:
            lines.append(
                f"Kept {te} text encoder layers. Connect the CLIP to KiLoRA Apply to apply "
                "them, as a plain LoRA loader does."
            )
        if self.skipped_keys:
            lines.append(
                f"Skipped {len(self.skipped_keys)} text encoder keys that are not LoRA "
                "weights, so results differ from the original LoRA where it also changes "
                "the text encoder."
            )
        layers = sum(
            int(x.meta.get("layers", 0)) for x in self.sections if x.kind != "text-encoder"
        )
        if layers:
            lines.append(
                f"{layers} layers that are not targets (modulation, embedders, head) "
                "get the LoRA as a fixed plain change, without routing."
            )
        if self.excluded_keys:
            lines.append(
                f"Skipped {len(self.excluded_keys)} keys on layers that are not targets of "
                "this family (modulation, embedders, output head)."
            )
        return "\n".join(lines)


def import_lora(
    source: str | os.PathLike[str] | Mapping[str, torch.Tensor],
    model: nn.Module,
    *,
    name: str | None = None,
    weight: float = 1.0,
    family: str | None = None,
    adapter_config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    experts: int = 4,
    router_dim: int = 32,
    seed: int = 0,
    full_model: nn.Module | None = None,
    size_norm: str = "off",
    base_weight: Any = None,
) -> LoRAImport:

    if size_norm not in ("off", "weight"):
        raise ValueError("size_norm must be off or weight")
    size_record: dict[str, Any] = {}
    if size_norm == "weight":
        from .multi_import import lora_file_size
        from .size_norm import applied_scale, size_scale

        config_in = adapter_config
        if config_in is None and not isinstance(source, Mapping):
            candidate = os.path.join(os.path.dirname(os.fspath(source)), "adapter_config.json")
            config_in = candidate if os.path.isfile(candidate) else None
        size = lora_file_size(
            dict(source) if isinstance(source, Mapping) else source,
            model,
            family=family,
            full_model=full_model,
            base_weight=base_weight,
            config=_read_config(config_in),
        )
        c_f = size_scale(size)
        applied, rule = applied_scale(c_f)
        size_record = {
            "size": size,
            "size_scale": c_f,
            "applied_scale": applied,
            "size_rule": rule,
        }
        weight = float(weight) * applied
    record: dict[str, Any] = {}
    if isinstance(source, Mapping):
        state = dict(source)
        metadata: dict[str, str] = {}
        default_name = "imported"
    else:
        state, metadata, record = _read_file(source)
        default_name = re.sub(r"[^A-Za-z0-9._-]", "-", os.path.splitext(record["original_name"])[0])
        if adapter_config is None:
            candidate = os.path.join(os.path.dirname(os.fspath(source)), "adapter_config.json")
            if os.path.isfile(candidate):
                adapter_config = candidate
    original_keys = set(state)
    config = _read_config(adapter_config)

    from .families import normalize_for
    from .sections import fixed_sections, mount_kind
    from .text_encoder_lora import is_text_encoder_key, text_encoder_sections

    te_sections, te_skipped = [], []
    clip_keys = (
        {k for k in state if is_text_encoder_key(k)} if not _targets_text_encoder(family) else set()
    )
    if not _targets_text_encoder(family):
        te_sections, te_skipped = text_encoder_sections(name or default_name, state, weight)
        state = {k: v for k, v in state.items() if not is_text_encoder_key(k)}
    norm = normalize_for(family, state, model, full_model)
    state, excluded = norm.state, norm.skipped
    sections = [*fixed_sections(name or default_name, norm.fixed, norm.fixed_dense, weight)]
    sections += te_sections
    if sections:
        logging.getLogger("kilora.import").info(
            "%s: %d keys on %d layers that are not targets (modulation, embedders, head) "
            "apply as a fixed plain change",
            name or default_name,
            norm.fixed_keys,
            len(norm.fixed) + len(norm.fixed_dense),
        )
    groups, skipped, layout = _group_keys(state)
    skipped = [*te_skipped, *skipped]
    if not groups:
        raise LoRAImportError("the file has no LoRA weights for the diffusion model")

    modules = {n: m for n, m in model.named_modules() if n}
    linear_names = [n for n, m in modules.items() if isinstance(m, nn.Linear)]
    aliases: dict[str, Alias] = get_preset(family).aliases(linear_names) if family else {}
    resolve = _resolver(layout, list(modules), aliases)

    routed_stems: set[str] = set()
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    scales: dict[str, float] = {}
    never: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for stem, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            raise LoRAImportError(f"{stem}: needs both a down (A) and an up (B) weight")
        down, up = parts["down"].float(), parts["up"].float()
        if down.dim() != 2 or up.dim() != 2:
            raise LoRAImportError(
                f"{stem}: only LoRA on linear layers is supported, found weights with "
                f"{down.dim()} and {up.dim()} dimensions"
            )
        rank = down.shape[0]
        if up.shape[1] != rank:
            raise LoRAImportError(f"{stem}: down has rank {rank}, up has rank {up.shape[1]}")
        scale = _scale(stem, parts.get("alpha"), rank, config)
        targets = resolve(stem)
        if not targets:
            never[stem] = (down, up * scale)
            continue
        routed_stems.add(stem)
        chunks = up.chunk(len(targets), dim=0) if len(targets) > 1 else (up,)
        if len(targets) > 1 and up.shape[0] % len(targets):
            raise LoRAImportError(
                f"{stem}: fused output of {up.shape[0]} rows does not split evenly"
            )
        for target, up_part in zip(targets, chunks, strict=True):
            module = modules[target]
            if not isinstance(module, nn.Linear):
                raise LoRAImportError(f"{stem}: {target} is not a linear layer")
            if down.shape[1] != module.in_features or up_part.shape[0] != module.out_features:
                raise LoRAImportError(
                    f"{stem}: weight shapes do not fit {target} "
                    f"(in {down.shape[1]} vs {module.in_features}, "
                    f"out {up_part.shape[0]} vs {module.out_features})"
                )
            if target in weights:
                raise LoRAImportError(f"two LoRA entries change the same module {target}")
            weights[target] = (down.clone(), up_part.clone())
            scales[target] = scale

    lost = []
    if never:
        tree = full_model if full_model is not None else model
        fixed, _, lost = _never_target_changes(never, tree, family)
        sections = [*sections, *fixed_sections(name or default_name, fixed, {}, weight)]
        log = logging.getLogger("kilora.import")
        if fixed:
            log.info(
                "%s: %d layers that are never targets (modulation, embedders, head) apply "
                "as a fixed plain change",
                name or default_name,
                len(fixed),
            )
        if lost:
            log.info(
                "%s: %d keys match no module of the model and are skipped: %s",
                name or default_name,
                len(lost),
                ", ".join(lost[:4]),
            )
    if not weights:
        raise LoRAImportError("the file has no LoRA weights for the target layers")
    distinct = set(scales.values())
    if len(distinct) == 1:
        scale: float | None = distinct.pop()
        strength = weight * scale
    else:
        scale = None
        strength = weight
    router_seed = get_preset(family).router_seed if family else 0
    gen = torch.Generator().manual_seed(int(seed))
    keyed: dict[str, KeyedExperts] = {}
    routers: dict[int, torch.Tensor] = {}
    for target, (down, up) in weights.items():
        if scale is None:
            up = up * scales[target]
        rank, n_out = down.shape[0], up.shape[0]
        routers.setdefault(n_out, make_router(router_dim, n_out, router_seed))
        keyed[target] = KeyedExperts(
            down=down.reshape(1, 1, rank, -1).repeat(experts, 1, 1, 1),
            up=up.t().reshape(1, 1, rank, n_out).repeat(experts, 1, 1, 1),
            keys_in=orthonormal_keys(torch.randn(experts, 1, router_dim, generator=gen)),
        )

    if record:
        record["format"] = layout
        record["original_license"] = _license(metadata)
        record["scale"] = scale if scale is not None else "per-module"
    if size_record:
        record = {**(record or {"original_name": "state dict"}), **size_record}
    imported = AdapterSet(
        name=name or default_name,
        part_type=("lora",),
        experts=keyed,
        routers=routers,
        aggregation="accumulate",
        form="input",
        strength=strength,
        imported_from=record or {"original_name": "state dict"},
        extra={"trained_with_kilora": False},
    )

    def originals(keys):
        return {original for key in keys for original in norm.origins.get(key, [key])}

    routed_keys = originals(
        k for k in state if (split := _split_key(k)) and split[0] in routed_stems
    )
    global_paths = dict(norm.fixed_original_paths)
    global_keys = set(norm.fixed_originals)
    for key in state:
        split = _split_key(key)
        if split and split[0] in never and split[0] not in lost:
            for original in norm.origins.get(key, [key]):
                global_keys.add(original)
                global_paths[original] = split[0]
    global_keys.update(clip_keys - set(te_skipped))
    routed_keys &= original_keys
    global_keys &= original_keys
    skipped_originals = original_keys - routed_keys - global_keys
    missing = originals(k for k in state if (split := _split_key(k)) and split[0] in lost)
    coverage = {
        "name": name or default_name,
        "coverage_counts": {"routed": len(routed_keys), "skipped": len(skipped_originals)},
        "global_mount_counts": {
            kind: sum(mount_kind(global_paths.get(key, key)) == kind for key in global_keys)
            for kind in ("adaln", "text-adapter", "embedder", "final", "other")
        },
        "coverage": {"skipped": sorted(skipped_originals)},
        "skipped_reasons": {
            key: "no matching model layer or shape"
            if key in missing
            else "unsupported or excluded weight"
            for key in skipped_originals
        },
    }
    return LoRAImport(
        adapter_set=imported,
        layout=layout,
        scale=scale,
        skipped_keys=skipped,
        excluded_keys=excluded,
        sections=sections,
        coverage=coverage,
    )


def _targets_text_encoder(family: str | None) -> bool:
    from .families import FAMILIES

    spec = FAMILIES.get(family or "")
    return spec is not None and spec.root == "text_encoder"


def _read_file(path: str | os.PathLike[str], *, checkpoint=None) -> tuple[dict, dict, dict]:
    from .modfile import _check_safetensors

    try:
        _check_safetensors(path)
    except ModFormatError as exc:
        raise LoRAImportError(str(exc)) from None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    state = {}
    with safe_open(os.fspath(path), framework="pt") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            if checkpoint is not None:
                checkpoint()
            state[key] = f.get_tensor(key)
    record = {
        "digest": "sha256-" + digest.hexdigest(),
        "original_name": os.path.basename(os.fspath(path)),
    }
    return state, metadata, record


def _read_config(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        with open(config, encoding="utf-8") as f:
            config = json.load(f)
    config = dict(config)
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise LoRAImportError(
            "adapter_config.json uses rank_pattern or alpha_pattern, which this import "
            "does not support yet"
        )
    if config.get("use_dora"):
        raise LoRAImportError("DoRA adapters are not plain LoRA and cannot be imported")
    return config


def _group_keys(state: Mapping[str, torch.Tensor]) -> tuple[dict, list[str], str]:
    groups: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    layouts: set[str] = set()
    for key, tensor in state.items():
        if key.startswith(_TEXT_ENCODER_PREFIXES):
            skipped.append(key)
            continue
        split = _split_key(key)
        if split is None:
            raise LoRAImportError(
                f"unknown key {key!r}: this is not a plain LoRA weight. LyCORIS, DoRA and "
                "other formats cannot be imported."
            )
        stem, part = split
        if stem.startswith("lora_unet_"):
            layouts.add("kohya")
        elif stem.startswith("diffusion_model."):
            layouts.add("comfyui")
        else:
            layouts.add("peft")
        groups.setdefault(stem, {})[part] = tensor
    if len(layouts) > 1:
        raise LoRAImportError(f"the file mixes key layouts {sorted(layouts)}")
    return groups, skipped, layouts.pop() if layouts else "peft"


def _split_key(key: str) -> tuple[str, str] | None:
    for suffix, part in _WEIGHT_SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


def _never_target_changes(
    never: Mapping, tree: nn.Module, family: str | None = None
) -> tuple[dict, dict, list[str]]:
    from .families import FAMILIES
    from .family_spec import linear_shape

    spec = FAMILIES.get(family or "")
    to_comfy = spec.comfy_path if spec is not None else None
    paths: dict[str, str] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for n, m in tree.named_modules():
        shape = linear_shape(m) if n else None
        if shape is None:
            continue
        name = (to_comfy(n) if to_comfy is not None else None) or n
        shapes[name] = shape
        for alias in {n, name}:
            paths.setdefault(alias.replace(".", "_"), name)
            paths.setdefault(alias, name)
    low, dense, lost = {}, {}, []
    for stem, value in never.items():
        body = stem.removeprefix("lora_unet_").removeprefix("diffusion_model.")
        for prefix in _MODEL_PREFIXES:
            body = body.removeprefix(prefix)
        path = paths.get(body)
        shape = shapes.get(path) if path is not None else None
        if isinstance(value, tuple):
            want = (value[0].shape[1], value[1].shape[0])
        else:
            want = (value.shape[1], value.shape[0])
        if shape is None or shape != want:
            lost.append(stem)
        elif isinstance(value, tuple):
            low[path] = value
        else:
            dense[path] = value
    return low, dense, lost


def _never_target(body: str) -> bool:
    from .family_spec import GENERIC_EXCLUDE

    dotted = body.replace("_", ".")
    return any(
        fnmatch.fnmatchcase(body, p) or fnmatch.fnmatchcase(dotted, p) for p in GENERIC_EXCLUDE
    )


def _resolver(
    layout: str, module_names: list[str], aliases: dict[str, Alias], *, strict: bool = True
):
    known = {n: (n,) for n in module_names}
    known.update({a: alias.targets for a, alias in aliases.items()})
    underscored = {}
    for n, targets in known.items():
        underscored.setdefault(n.replace(".", "_"), set()).add(targets)

    def resolve(stem: str) -> tuple[str, ...]:
        if layout == "kohya":
            body = stem.removeprefix("lora_unet_")
            options = underscored.get(body, set())
            if len(options) > 1:
                raise LoRAImportError(f"{stem}: the module name is ambiguous")
            if options:
                return next(iter(options))
        else:
            body = stem.removeprefix("diffusion_model.")
            for prefix in _MODEL_PREFIXES:
                body = body.removeprefix(prefix)
            if body in known:
                return known[body]
        if not strict or _never_target(body):
            return ()
        raise LoRAImportError(f"{stem}: no module with this name in the model")

    return resolve


def _scale(stem: str, alpha: torch.Tensor | None, rank: int, config: dict | None) -> float:
    if alpha is not None:
        value = float(alpha.float().reshape(-1)[0])
    elif config is not None and "lora_alpha" in config:
        value = float(config["lora_alpha"])
        rank = int(config.get("r", rank))
    else:
        return 1.0
    scale = value / rank
    if not math.isfinite(scale):
        raise LoRAImportError(f"{stem}: alpha gives a scale that is not finite")
    return scale


def _license(metadata: Mapping[str, str]) -> str:
    for key in ("modelspec.license", "license"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return "unknown"


def to_state_form(
    adapter_set: AdapterSet, model: nn.Module, *, ridge: float = 1e-4
) -> tuple[AdapterSet, dict[str, float]]:

    if adapter_set.form != "input":
        raise ValueError(f"set {adapter_set.name!r} is already in the {adapter_set.form} form")
    modules = dict(model.named_modules())
    experts, lost = {}, {}
    for target, ex in adapter_set.experts.items():
        w = read_base_weight(modules[target], target).float()
        gram = w @ w.t()
        lam = ridge * float(gram.diagonal().mean())
        solve = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], device=w.device), w)
        a = ex.down.float().to(w.device)
        a_state = a @ solve.t()
        back = a_state @ w
        lost[target] = round(float((a - back).norm() / a.norm().clamp(min=1e-12)), 6)
        experts[target] = dataclasses.replace(ex, down=a_state.to(ex.down.dtype).cpu())
    extra = {**adapter_set.extra, "trained_with_kilora": False}
    return (
        dataclasses.replace(adapter_set, experts=experts, form="state", extra=extra),
        lost,
    )


def import_to_mod(
    source: str | os.PathLike[str],
    model: nn.Module,
    output: str | os.PathLike[str],
    *,
    family: str | None = None,
    weight: float = 1.0,
    experts: int = 4,
    state_form: bool = False,
    mod_id: str | None = None,
    license: str | None = None,
    ridge: float = 1e-4,
) -> tuple[dict[str, Any], str]:

    from .adapter import imported_schedule_fields, imported_settings
    from .modfile import save_mod

    result = import_lora(source, model, weight=weight, family=family, experts=experts)
    adapter_set, lines = result.adapter_set, [result.report()]
    if state_form:
        adapter_set, lost = to_state_form(adapter_set, model, ridge=ridge)
        worst = max(lost, key=lost.get)
        lines.append(
            f"Converted for training (A' = A W+). The largest lost part is "
            f"{lost[worst]:.1%} at {worst}. Depth above 1 changes the result until you train."
        )
    else:
        lines.append("Form: input. The LoRA runs once, before the loop, at every depth.")
    original = (adapter_set.imported_from or {}).get("original_license")
    name = adapter_set.name
    manifest = save_mod(
        output,
        [adapter_set],
        imported_settings(),
        id=mod_id or re.sub(r"[^a-z0-9._-]", "-", name.lower())[:64].strip("-.") or "lora",
        name=name,
        license=license or (original if original not in (None, "unknown") else "NOASSERTION"),
        base_model={"family": family or "unknown"},
        sections=result.sections,
        **({} if state_form else imported_schedule_fields()),
    )
    lines.append(f"Wrote {os.fspath(output)}.")
    return manifest, "\n".join(lines)
