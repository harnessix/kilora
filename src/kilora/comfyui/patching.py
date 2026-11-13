# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import math
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..adaptive_gate import check_gate_mode, check_kappa, parse_gate_history
from ..bake import baked_copy, is_baked, module_plans, plan_targets
from ..base_weights import BaseWeights, is_quantized_weight, read_base_weight
from ..bindings import BINDINGS, COMFY_PREFIX, Binding
from ..compute import DEPTH_STEP_DEFAULTS, check_depth_step, check_routing
from ..errors import KiLoRAError
from ..families import detect_binding, layout_resolver
from ..fixed_change import fixed_copy, is_fixed
from ..i18n import text
from ..lora_import import import_lora
from ..modlist import LoadPlan, ModChoice, require_ok, resolve
from ..path_walks import check_path_select, check_path_width
from ..presets import PRESETS
from ..route_freeze import check_route_freeze
from ..routing_features import call_features, prompt_projection, routing_features, timestep_features
from ..routing_features import enabled as routing_features_enabled
from . import depth as depth_schedules
from .bypass import INJECTION_KEY, ForwardHook, injection, is_quantized, linear_shape

log = logging.getLogger("kilora")

_MEMORY_KEY = "kilora_working_memory"
_MEMORY_CLASSES: dict[type, type] = {}


def _prepare_sampling_memory(executor, patcher, *args, **kwargs):
    model = patcher.model
    original = model.memory_required
    had_local = "memory_required" in model.__dict__
    local = model.__dict__.get("memory_required")
    extra = patcher.model_options.get(_MEMORY_KEY, 0)
    previous = patcher.__dict__.get("_kilora_estimating", False)
    patcher.__dict__["_kilora_estimating"] = True
    model.memory_required = lambda *a, **kw: original(*a, **kw) + extra
    try:
        return executor(patcher, *args, **kwargs)
    finally:
        if had_local:
            model.memory_required = local
        else:
            del model.memory_required
        if previous:
            patcher.__dict__["_kilora_estimating"] = previous
        else:
            patcher.__dict__.pop("_kilora_estimating", None)


def _sampling_wrappers(options: dict) -> dict:
    transformer = dict(options.get("transformer_options", {}))
    options["transformer_options"] = transformer
    wrappers = dict(transformer.get("wrappers", {}))
    transformer["wrappers"] = wrappers
    sampling = dict(wrappers.get("prepare_sampling", {}))
    wrappers["prepare_sampling"] = sampling
    return sampling


def _working_memory(patcher: Any, plans: dict, changes: dict) -> None:
    model = patcher.model
    dtype = getattr(model, "manual_cast_dtype", None)
    if dtype is None and hasattr(model, "get_dtype"):
        dtype = model.get_dtype()
    if not isinstance(dtype, torch.dtype):
        dtype = next(model.parameters()).dtype
    size = torch.empty((), dtype=dtype).element_size()
    count = 0
    for plan in plans.values():
        for _, _, target in plan.parts:
            for layer in target.sets:
                for name in (
                    "router",
                    "keys_in",
                    "keys_out",
                    "rows_in",
                    "rows_out",
                    "down",
                    "up",
                    "share_transition_weighted",
                ):
                    value = getattr(layer, name, None)
                    if isinstance(value, torch.Tensor):
                        count += value.numel()
                if layer.gate:
                    count += sum(t.numel() for t in layer.gate.values())
    for items in changes.values():
        for item in items:
            count += sum(v.numel() for v in vars(item).values() if isinstance(v, torch.Tensor))
    patcher.model_options[_MEMORY_KEY] = count * size
    _sampling_wrappers(patcher.model_options)[_MEMORY_KEY] = [_prepare_sampling_memory]
    if not hasattr(patcher, "memory_required"):
        return
    base = type(patcher)
    if getattr(base, "_kilora_memory_class", False):
        return
    if base not in _MEMORY_CLASSES:

        def memory_required(self, *args, **kwargs):
            extra = (
                0
                if self.__dict__.get("_kilora_estimating")
                else self.model_options.get(_MEMORY_KEY, 0)
            )
            return base.memory_required(self, *args, **kwargs) + extra

        _MEMORY_CLASSES[base] = type(
            f"KiLoRAMemory{base.__name__}",
            (base,),
            {"memory_required": memory_required, "_kilora_memory_class": True},
        )
    patcher.__class__ = _MEMORY_CLASSES[base]


@dataclass(frozen=True)
class LiveEnv:
    saves_dir: str | None = None
    editor_paths: tuple[str, ...] = ()
    search_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModEntry:
    kind: str
    path: str
    strength: float = 1.0
    prior: float = 1.0
    enabled: bool = True
    region: Any = field(default=None, compare=False)
    region_caption: Any = field(default=None, compare=False)
    roles: str = "auto"
    panels: Any = field(default=None, compare=False)
    axis: str = "auto"
    timeline: tuple = ()
    fine: bool = False
    candidates: int = 16
    start: float = 0.0
    end: float = 1.0

    def __post_init__(self):
        from ..entry_window import check_window

        check_window(self.start, self.end)


FUNCTIONAL_DEFAULTS = {"gate_mode": "none", "fixed_depth": 1, "route_freeze": None}
IP_DEFAULTS = {"route_freeze": None}


class ModChain(tuple):
    def add(self, *entries: ModEntry) -> ModChain:
        return ModChain((*self, *entries))


def detect_family(patcher: Any) -> Binding:
    denoiser = patcher.model.diffusion_model
    declared = getattr(denoiser, "kilora_family", None)
    if declared in BINDINGS:
        return BINDINGS[declared]
    names = [n for n, _ in denoiser.named_modules()]
    for binding in BINDINGS.values():
        if binding.matches_comfy(names):
            return binding
    found = detect_binding(denoiser)
    if found is not None:
        return found
    raise KiLoRAError(
        "KiLoRA does not support this base model yet. Supported families: "
        + ", ".join(b for b in BINDINGS if not b.startswith("kilora-"))
    )


def shape_model(shapes: dict[str, tuple[int, int]]) -> nn.Module:
    root = nn.Module()
    for name, (fan_in, fan_out) in shapes.items():
        parent = root
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(parts[-1], nn.Linear(fan_in, fan_out, device="meta"))
    return root


def diffusers_shapes(patcher: Any, binding: Binding) -> dict[str, tuple[int, int]]:
    shapes = {}
    for name, module in patcher.model.diffusion_model.named_modules():
        shape = linear_shape(module)
        if shape is None:
            continue
        for target, part in binding.from_comfy(name):
            rows = shape[1] // part[1] if part else shape[1]
            shapes[target] = (shape[0], rows)
    return shapes


def host_weight_norms(patcher: Any, targets: Any) -> dict[str, float]:
    binding = detect_family(patcher)
    want = set(targets)
    out: dict[str, float] = {}
    for name, module in patcher.model.diffusion_model.named_modules():
        if linear_shape(module) is None:
            continue
        hits = [(t, p) for t, p in binding.from_comfy(name) if t in want]
        w = read_base_weight(module, name) if hits else None
        if not hits or not isinstance(w, torch.Tensor) or w.dim() != 2 or w.device.type == "meta":
            continue
        if not w.dtype.is_floating_point:
            continue
        wf = w.detach().float()
        for target, part in hits:
            if part:
                index, count = part
                rows = wf.shape[0] // count
                out[target] = float(wf[index * rows : (index + 1) * rows].norm())
            else:
                out[target] = float(wf.norm())
    return out


def host_weights(patcher: Any, targets: Any) -> dict[str, torch.Tensor] | BaseWeights:
    binding = detect_family(patcher)
    want = set(targets)
    entries = {}
    for name, module in patcher.model.diffusion_model.named_modules():
        if linear_shape(module) is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or weight.device.type == "meta":
            continue
        if not is_quantized_weight(weight) and not weight.dtype.is_floating_point:
            continue
        for target, part in binding.from_comfy(name):
            if target in want:
                entries[target] = (module, part)
    if any(is_quantized_weight(module.weight) for module, _ in entries.values()):
        return BaseWeights(entries)
    out = {}
    for target, (module, part) in entries.items():
        wf = module.weight.detach()
        if part:
            index, count = part
            rows = wf.shape[0] // count
            wf = wf[index * rows : (index + 1) * rows]
        out[target] = wf
    return out


_LORA_CACHE: dict[tuple, Any] = {}
_ROUTED_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_ROUTED_CACHE_LOCK = threading.Lock()


def _import_result(
    path: str,
    binding: Binding,
    shapes_model,
    full_model: nn.Module | None,
    lycoris_apply: str = "truncated",
):
    from .key_cache import file_sha256

    model = shapes_model if isinstance(shapes_model, nn.Module) else shapes_model()
    shapes = tuple((n, linear_shape(m)) for n, m in model.named_modules() if linear_shape(m))
    full_shapes = (
        tuple((n, linear_shape(m)) for n, m in full_model.named_modules() if linear_shape(m))
        if full_model is not None
        else ()
    )
    config = os.path.join(os.path.dirname(path), "adapter_config.json")
    config_hash = file_sha256(config) if os.path.isfile(config) else None
    key = (
        file_sha256(path),
        os.path.basename(path),
        config_hash,
        binding.family,
        shapes,
        full_shapes,
        os.environ.get("KILORA_LOKR_RANKS"),
        os.environ.get("KILORA_LOKR_BUDGET_MIB"),
        os.environ.get("KILORA_IMPORT_SCORE"),
        os.environ.get("KILORA_LYCORIS_APPLY"),
        lycoris_apply,
    )
    if key not in _LORA_CACHE:
        if len(_LORA_CACHE) > 32:
            _LORA_CACHE.clear()
        family = binding.family if binding.family in PRESETS else None
        if _is_lycoris(path):
            from ..multi_import import import_loras

            _LORA_CACHE[key] = import_loras(
                [path], model, family=family, full_model=full_model, lycoris_mode="exact"
            )
        else:
            _LORA_CACHE[key] = import_lora(
                path, model, family=family, full_model=full_model, experts=1
            )
    return _LORA_CACHE[key]


_LYCORIS_PARTS = (".lokr_w1", ".lokr_w2", ".hada_w1", ".hada_w2")


def _is_lycoris(path: str) -> bool:
    from safetensors import safe_open

    if not os.path.isfile(path):
        return False
    with safe_open(path, "pt") as f:
        names = list(f.keys())
    return any(part in key for key in names for part in _LYCORIS_PARTS)


def _imported(path: str, binding: Binding, shapes_model: nn.Module, full_model=None):
    return _import_result(path, binding, shapes_model, full_model).adapter_set


def _fixed_changes(plan: LoadPlan, chain: Sequence[ModEntry] = ()) -> dict[str, list]:
    from ..sections import layer_changes

    out: dict[str, list] = {}
    for m in plan.mods:
        e = _entry_of(m, chain)
        joint = m.label is not None and m.label.startswith("routed LoRAs")
        axis = (
            "ip"
            if joint and m.label.startswith("routed LoRAs (ip)")
            else e.axis
            if e is not None and not joint
            else "creative"
        )
        tag = JOINT_SETS[axis] if joint else (entry_name(e) if e is not None else (m.label or ""))
        for section in m.sections:
            for path, changes in layer_changes([section], m.strength).items():
                for index, c in enumerate(changes):
                    c.tag, c.axis = tag, axis
                    if (
                        m.session is not None
                        and hasattr(m.session, "_section_expert")
                        and section.kind in ("fixed-lokr", "fixed-delta")
                        and section.meta.get("lycoris")
                    ):
                        expert = m.session._section_expert(section)
                        if (path, expert) in m.session._section_pairs:
                            clean_section = next(
                                (
                                    item
                                    for item in m.session._sections
                                    if item.name == section.name and item.kind == section.kind
                                ),
                                None,
                            )
                            if clean_section is not None:
                                clean_changes = layer_changes([clean_section], m.strength).get(
                                    path, []
                                )
                                if index < len(clean_changes):
                                    c._kilora_import_clean = clean_changes[index]
                                    c._kilora_import_key = (m.session.mod_id, path, expert)
                    if path.startswith("llm_adapter."):
                        name = section.name
                        if joint and name.split("-", 1)[0].isdigit():
                            name = name.split("-", 1)[1]
                        c.text_expert = name if joint else tag
                out.setdefault(COMFY_PREFIX + path, []).extend(changes)
    return out


def live_session(entry: ModEntry, env: LiveEnv | None):
    from ..live import session_for
    from ..modfile import read_manifest

    if entry.kind != "mod" or not read_manifest(entry.path).get("live"):
        return None
    env = env or LiveEnv()
    return session_for(entry.path, saves_dir=env.saves_dir)


JOINT_SET = "routed-loras"
IP_JOINT_SET = "routed-ip"
JOINT_SETS = {"creative": JOINT_SET, "ip": IP_JOINT_SET}


def joint_choice(
    patcher: Any,
    entries: Sequence[ModEntry],
    binding: Binding,
    shapes_model,
    *,
    read_file=None,
    key_time=None,
    content_keys="per_expert",
    lycoris_apply="truncated",
    open_session=False,
    saves_dir=None,
) -> tuple[ModChoice, list[tuple[str, float]]]:
    from ..multi_import import LoRASource, import_loras

    files = [(e.path, float(e.strength)) for e in entries]
    names = [os.path.splitext(os.path.basename(p))[0] for p, _ in files]
    sources = [
        LoRASource(path=p, weight=w, name=f"{i}-{n}")
        for i, ((p, w), n) in enumerate(zip(files, names, strict=True))
    ]
    family = binding.family if binding.family in PRESETS else None
    from .key_cache import file_sha256

    full_model = patcher.model.diffusion_model
    from ..derived_cache import backend
    from .prepare_node import apply_device

    device = apply_device(patcher)
    identities = (
        tuple((file_sha256(p), w, n) for (p, w), n in zip(files, names, strict=True))
        if read_file is None
        else ()
    )
    shapes = tuple((n, linear_shape(m)) for n, m in shapes_model.named_modules() if linear_shape(m))
    full_shapes = tuple(
        (n, linear_shape(m)) for n, m in full_model.named_modules() if linear_shape(m)
    )
    settings = tuple(
        os.environ.get(k)
        for k in (
            "KILORA_LOKR_RANKS",
            "KILORA_LOKR_BUDGET_MIB",
            "KILORA_IMPORT_SCORE",
            "KILORA_LYCORIS_APPLY",
            "KILORA_CONTENT_KEYS",
        )
    )
    key = (
        tuple(sorted((h, w) for h, w, _ in identities)),
        identities,
        family,
        settings,
        content_keys,
        lycoris_apply,
        shapes,
        full_shapes,
        repr(backend(device)),
    )
    if read_file is not None:
        result = import_loras(
            sources,
            shapes_model,
            family=family,
            full_model=full_model,
            read_file=read_file,
            key_time=key_time,
            device=device,
            content_keys=content_keys,
            lycoris_apply=lycoris_apply,
        )
    else:
        with _ROUTED_CACHE_LOCK:
            if key not in _ROUTED_CACHE:
                _ROUTED_CACHE[key] = import_loras(
                    sources,
                    shapes_model,
                    family=family,
                    full_model=full_model,
                    key_time=key_time,
                    device=device,
                    content_keys=content_keys,
                    lycoris_apply=lycoris_apply,
                )
                while len(_ROUTED_CACHE) > 2:
                    _ROUTED_CACHE.popitem(last=False)
            _ROUTED_CACHE.move_to_end(key)
            original = _ROUTED_CACHE[key]
            with torch.inference_mode(False):
                result = copy.deepcopy(original)
            if original.adapter_set is not None:
                from .. import derived_cache

                for target, expert in original.adapter_set.experts.items():
                    source_id = derived_cache.source(expert)
                    if source_id:
                        derived_cache.stamp(result.adapter_set.experts[target], source_id)
    axis = entries[0].axis
    routed = dataclasses.replace(result.adapter_set, name=JOINT_SETS[axis])
    from ..live_imports import imported_session_for

    source_info = (
        [{"path": p, "sha256": file_sha256(p), "strength": w, "axis": axis} for p, w in files]
        if read_file is None
        else []
    )
    session = (
        imported_session_for(
            routed,
            source_info,
            binding.family,
            shapes,
            saves_dir,
            sections=result.sections,
            create=open_session,
        )
        if source_info
        else None
    )
    choice = ModChoice(
        adapter_sets=[routed],
        session=session,
        sections=list(result.sections),
        entry_windows=tuple((e.start, e.end) for e in entries)
        if any((e.start, e.end) != (0.0, 1.0) for e in entries)
        else (),
        strength=1.0,
        label=("routed LoRAs: " if axis == "creative" else "routed LoRAs (ip): ")
        + ", ".join(names),
        mod_id="lora-routed" if axis == "creative" else "lora-routed-ip",
        notes=(
            ([result.content_key_summary()] if result.content_key_counts is not None else [])
            + result.cut_notes()
            + _joint_coverage_notes(result.sources)
        ),
    )
    return choice, files


def _joint_coverage_notes(sources: list[dict], *, single: bool = False) -> list[str]:
    notes = []
    for src in sources:
        kinds = src["global_mount_counts"]
        counts = ", ".join(f"{kind} {count}" for kind, count in kinds.items())
        notes.append(
            f"{src['name']}: {src['coverage_counts']['routed']} routed / "
            f"scaled-global ({counts}) / {src['coverage_counts']['skipped']} skipped; "
            + (
                "single."
                if single
                else f"previous-step mean share, 1/{len(sources)} before the first step."
            )
        )
        skipped = src["coverage"]["skipped"]
        notes.extend(
            f"{src['name']}: {key}: skipped ({src['skipped_reasons'][key]})" for key in skipped[:20]
        )
        if len(skipped) > 20:
            notes.append(f"{src['name']}: and {len(skipped) - 20} more skipped keys")
    return notes


def _single_coverage_notes(result) -> list[str]:
    if not hasattr(result, "sources"):
        return _joint_coverage_notes([result.coverage], single=True)
    sources = copy.deepcopy(result.sources)
    for src in sources:
        kept = [
            key
            for key in src["coverage"]["skipped"]
            if src["skipped_reasons"][key] == "text encoder kept separately"
        ]
        src["coverage"]["skipped"] = [key for key in src["coverage"]["skipped"] if key not in kept]
        src["coverage_counts"]["skipped"] -= len(kept)
        src["global_mount_counts"]["other"] += len(kept)
    return _joint_coverage_notes(sources, single=True)


def _joint_lora(e: ModEntry, axis: str = "creative") -> bool:
    return e.kind == "lora" and e.enabled and e.axis == axis


def _routed_groups(chain: Sequence[ModEntry], joint: bool) -> dict[str, list[ModEntry]]:
    if not joint:
        return {}
    groups = {axis: [e for e in chain if _joint_lora(e, axis)] for axis in JOINT_SETS}
    return {
        axis: entries
        for axis, entries in groups.items()
        if len(entries) > 1
        or (axis == "creative" and len(entries) == 1 and not _is_lycoris(entries[0].path))
    }


def choices_for(
    patcher: Any,
    chain: Sequence[ModEntry],
    binding: Binding,
    env: LiveEnv | None = None,
    joint: bool = False,
    content_keys: str = "per_expert",
    lycoris_apply: str = "truncated",
    open_imports: bool = False,
) -> list[ModChoice]:
    choices = []
    shapes_model = None
    groups = _routed_groups(chain, joint)
    joined = set()
    for entry in chain:
        if entry.axis in groups and _joint_lora(entry, entry.axis):
            if entry.axis in joined:
                continue
            if shapes_model is None:
                shapes_model = shape_model(diffusers_shapes(patcher, binding))
            choice, _ = joint_choice(
                patcher,
                groups[entry.axis],
                binding,
                shapes_model,
                content_keys=content_keys,
                lycoris_apply=lycoris_apply,
                open_session=open_imports,
                saves_dir=(env or LiveEnv()).saves_dir,
            )
            choices.append(choice)
            joined.add(entry.axis)
            continue
        if entry.kind == "mod":
            choices.append(
                ModChoice(
                    path=entry.path,
                    strength=entry.strength,
                    prior=entry.prior,
                    enabled=entry.enabled,
                    label=os.path.basename(entry.path),
                    session=live_session(entry, env) if entry.enabled else None,
                )
            )
        elif entry.kind in ("direction", "direction_file"):
            from ..library.basis import load_basis
            from ..library.index import LibraryIndex
            from ..library.nodes import direction_adapters, direction_file_adapters

            shapes = diffusers_shapes(patcher, binding)
            if entry.kind == "direction_file":
                adapters = direction_file_adapters(entry.path, shapes)
                source_description = (
                    f"the direction file has no mounts matching this model; file: {entry.path}; "
                )
            else:
                basis = load_basis(LibraryIndex.open(entry.path or None))
                source_description = (
                    "the stored basis has no mounts matching this model; "
                    f"basis mounts: {list(basis.mounts)[:3]}; "
                )
                adapters = direction_adapters(
                    basis,
                    shapes,
                    chunk=int(entry.prior),
                    fine=entry.fine,
                    candidates=entry.candidates,
                )
            if not adapters:
                raise KiLoRAError(
                    source_description
                    + f"model targets: {list(shapes)[:3]}"
                    + ("\n" + "\n".join(adapters.notes) if adapters.notes else "")
                )
            choices.append(
                ModChoice(
                    adapter_sets=adapters,
                    strength=entry.strength,
                    enabled=entry.enabled,
                    label=os.path.basename(entry.path),
                    mod_id="direction-nodes",
                    notes=adapters.notes,
                )
            )
        elif entry.kind == "lora":
            if shapes_model is None:
                shapes_model = shape_model(diffusers_shapes(patcher, binding))
            result = _import_result(
                entry.path, binding, shapes_model, patcher.model.diffusion_model, lycoris_apply
            )
            label = os.path.basename(entry.path)
            choices.append(
                ModChoice(
                    adapter_sets=[result.adapter_set] if result.adapter_set is not None else [],
                    sections=list(result.sections),
                    strength=entry.strength,
                    enabled=entry.enabled,
                    label=label,
                    mod_id="lora-" + "".join(c if c.isalnum() else "-" for c in label.lower()),
                    notes=_single_coverage_notes(result),
                    entry_windows=((entry.start, entry.end),)
                    if (entry.start, entry.end) != (0.0, 1.0)
                    else (),
                )
            )
        else:
            raise KiLoRAError(f"unknown chain entry kind {entry.kind!r}")
    return choices


def expand_chain(chain: Sequence[ModEntry], env: LiveEnv | None = None):
    from ..modlist import load_choice
    from ..timeline import Row

    search_dirs = (env or LiveEnv()).search_dirs
    out = []
    logic = None

    def add(entry):
        nonlocal logic
        if entry.kind != "mod" or not entry.enabled:
            out.append(entry)
            return
        loaded = load_choice(ModChoice(path=entry.path), 0, search_dirs)
        if loaded.logic is not None and logic is None:
            logic = loaded.logic
        if not loaded.references:
            if (
                loaded.logic is None and not any(s.kind == "weight-ref" for s in loaded.sections)
            ) or any(s.kind == "program" for s in loaded.sections):
                out.append(entry)
            return
        for ref in loaded.references:
            t = ref.transforms
            add(
                ModEntry(
                    ref.kind,
                    ref.path,
                    entry.strength * t["strength"],
                    entry.prior,
                    t["enabled"],
                    roles=t["roles"],
                    axis=t["axis"],
                    timeline=entry.timeline + tuple(Row(**row) for row in t["timeline"]),
                    start=max(entry.start, t.get("start", 0.0)),
                    end=min(entry.end, t.get("end", 1.0)),
                )
            )

    for item in chain:
        add(item)
    return ModChain(out), logic


def plan_chain(
    patcher: Any,
    chain: Sequence[ModEntry],
    env: LiveEnv | None = None,
    joint: bool = False,
    joint_choice_override: ModChoice | None = None,
    content_keys: str = "per_expert",
    lycoris_apply: str = "truncated",
    open_imports: bool = False,
) -> tuple[LoadPlan, Binding]:
    chain, _ = expand_chain(chain, env)
    chain = ModChain(dataclasses.replace(e, axis=resolve_axis(e)[0]) for e in chain)
    binding = detect_family(patcher)
    plan = resolve(
        (
            [joint_choice_override]
            if joint_choice_override is not None
            else choices_for(
                patcher,
                chain,
                binding,
                env,
                joint,
                content_keys=content_keys,
                lycoris_apply=lycoris_apply,
                open_imports=open_imports,
            )
        ),
        family=binding.family,
        search_dirs=(env or LiveEnv()).search_dirs,
    )
    return plan, binding


APPLIED_KEY = "kilora"
BYPASS_MODES = ("auto", "on", "off")


def applied(patcher: Any) -> dict[str, Any] | None:
    options = getattr(patcher, "model_options", None)
    if isinstance(options, dict) and APPLIED_KEY in options:
        return options[APPLIED_KEY]
    return None


def _remember(
    patcher: Any,
    chain: Sequence[ModEntry],
    depth: int | None,
    env: Any,
    iteration=None,
    schedule: Any = None,
    used_schedule: Any = None,
    routing: tuple | None = None,
) -> None:
    options = getattr(patcher, "model_options", None)
    if isinstance(options, dict):
        options[APPLIED_KEY] = {
            "chain": ModChain(chain),
            "depth": depth,
            "env": env,
            "iteration": iteration,
            "routing": routing,
            "schedule": schedule,
            "depth_schedule": used_schedule,
        }


def record_composition(plan: LoadPlan, depth: int) -> None:
    for m in plan.mods:
        if m.session is None:
            continue
        m.session.composition = [
            {"id": o.mod_id, "digest": o.digest, "strength": float(o.strength), "depth": depth}
            for o in plan.mods
            if o is not m
        ]


def live_sessions(plan: LoadPlan) -> list[Any]:
    return [m.session for m in plan.mods if m.session is not None]


def _working_copy_events():
    try:
        from comfy.patcher_extension import CallbacksMP

        return CallbacksMP.ON_DETACH, CallbacksMP.ON_CLEANUP
    except ImportError:
        return "on_detach_after", "on_cleanup"


def clear(patcher: Any) -> Any:
    out = patcher.clone()
    text_patch = out.object_patches.get("extra_conds")
    previous_text_patch = getattr(text_patch, "_kilora_previous_extra_patch", None)
    if hasattr(out, "remove_callbacks_with_key"):
        for event in _working_copy_events():
            out.remove_callbacks_with_key(event, "kilora")
    for name in [
        k
        for k, v in out.object_patches.items()
        if is_baked(v)
        or is_fixed(v)
        or getattr(v, "_kilora_learn", False) is True
        or hasattr(v, "_kilora_original_extra_conds")
    ]:
        del out.object_patches[name]
    if previous_text_patch is not None:
        out.object_patches["extra_conds"] = previous_text_patch
    options = getattr(out, "model_options", None)
    if isinstance(options, dict):
        from .layer_alpha import remove_layer_alpha

        remove_layer_alpha(options)
        options.pop(APPLIED_KEY, None)
        options.pop(_MEMORY_KEY, None)
        _sampling_wrappers(options).pop(_MEMORY_KEY, None)
    injections = getattr(out, "injections", None)
    if isinstance(injections, dict):
        injections.pop(INJECTION_KEY, None)
    return out


def _require_applied(patcher: Any) -> dict[str, Any]:
    info = applied(patcher)
    if info is None:
        raise KiLoRAError("connect the model output of KiLoRA Apply")
    return info


def sessions_of(patcher: Any) -> list[Any]:
    info = _require_applied(patcher)
    plan, _ = plan_chain(
        patcher,
        info["chain"],
        info["env"],
        bool(info.get("joint")),
        lycoris_apply=info.get("lycoris_apply", "truncated"),
    )
    return live_sessions(plan)


def reapply(patcher: Any) -> tuple[Any, str]:
    info = _require_applied(patcher)
    return apply(
        patcher,
        info["chain"],
        info["depth"],
        info["env"],
        info.get("iteration"),
        info.get("schedule"),
        *(info.get("routing") or (None, None)),
        gate=info.get("gate"),
        gate_history=info.get("gate_history", "full"),
        walks=info.get("walks"),
        freeze=info.get("freeze"),
        timeline=info.get("timeline"),
        token_depth=info.get("token_depth"),
        alpha_mask=info.get("alpha_mask"),
        axis_alpha=info.get("axis_alpha", "off"),
        alpha_clip=info.get("alpha_clip"),
        expert_text=info.get("expert_text", "auto"),
        joint=bool(info.get("joint", False)),
        import_score=info.get("import_score"),
        lycoris_apply=info.get("lycoris_apply", "truncated"),
        gate_stats=info.get("gate_stats", "fused"),
        variation=info.get("variation"),
        candidates=info.get("candidates", 0),
        candidate_summary=info.get("candidate_summary", "keys"),
        routing_features_enabled=info.get("routing_features", False),
        cross_weights=info.get("cross_weights", (0.0, 0.0, 0.0)),
        linear_jump=info.get("linear_jump", False),
        jump_threshold=info.get("jump_threshold", 0.05),
        bypass=info.get("bypass"),
        depth_steps=info.get("depth_steps"),
    )


LIVE_ACTIONS = ("commit", "discard", "undo", "reset", "save point", "load")


def live_action(patcher: Any, action: str, name: str, mod: str):
    sessions = sessions_of(patcher)
    if mod.strip() and mod.strip().lower() != "all":
        sessions = [s for s in sessions if s.mod_id == mod.strip()]
    if not sessions:
        raise KiLoRAError("no matching live mod is loaded")
    lines = []
    for s in sessions:
        queue = getattr(s, "write_queue", None)
        guard = queue.exclusive() if queue is not None else contextlib.nullcontext()
        with guard:
            kept = None
            if queue is not None and action in ("undo", "reset", "load"):
                kept = queue.keep_pending(s.editor_state, action)
            try:
                result = _live_action_one(s, action, name)
            except Exception:
                for branch, _ in kept or []:
                    if branch in s.editor_state.forest.branches:
                        queue.cancel_keep(s.editor_state, branch)
                raise
            for branch, numbers in kept or []:
                lines.append(f"{s.mod_id}: kept on side branch {branch} after {action}")
                log.info("live writes %s kept on side branch %s after %s", numbers, branch, action)
            lines.append(f"{s.mod_id}: {result}")
    out, report = reapply(patcher)
    return out, "\n".join(lines) + "\n" + report


def _live_action_one(s: Any, action: str, name: str) -> str:
    if action == "commit":
        return "nothing pending" if s.pending is None else s.commit()
    if action == "discard":
        s.discard()
        return "Discarded the pending update."
    if action == "undo":
        return s.undo()
    if action == "reset":
        return s.reset()
    if action == "save point":
        return f"Saved {s.save_point(name or 'save')}."
    if action == "load":
        return s.load(name)
    raise KiLoRAError(f"unknown action {action!r}")


def resolve_axis(entry: ModEntry) -> tuple[str, str]:
    if entry.axis != "auto":
        return entry.axis, "node selection"
    if entry.kind != "lora":
        return "creative", "no clear axis signal"
    from ..roles import suggest_axis

    return suggest_axis(entry.path)


def apply(
    patcher: Any,
    chain: Sequence[ModEntry],
    depth: int | None,
    env: LiveEnv | None = None,
    iteration: str | None = None,
    schedule: Any = None,
    routing_scope: str | None = None,
    routing_select: str | None = None,
    bypass: str | None = None,
    gate: tuple | None = None,
    walks: tuple | None = None,
    joint: bool = False,
    freeze: float | bool | None = None,
    timeline: Any = None,
    token_depth: Any = None,
    import_score: str | None = None,
    lycoris_apply: str = "truncated",
    variation: float | None = None,
    linear_jump: bool = False,
    jump_threshold: float = 0.05,
    axis_alpha: str = "off",
    alpha_clip: Any = None,
    alpha_mask: Any = None,
    gate_history: str = "full",
    expert_text: str = "auto",
    joint_choice_override: ModChoice | None = None,
    candidates: int = 0,
    candidate_summary: str = "keys",
    routing_features_enabled: bool = False,
    cross_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    share_observer: Any = None,
    content_keys: str = "per_expert",
    gate_stats: str = "fused",
    depth_steps: tuple[float, float, float] | None = None,
) -> tuple[Any, str]:

    from .settings import resolve_settings

    resolve_settings(
        {
            "lycoris_apply": lycoris_apply,
            "gate_stats": gate_stats,
        }
    )
    from ..compute import resolve_gate_stats

    resolve_gate_stats(gate_stats)
    gate_head_limit = parse_gate_history(gate_history)
    if expert_text not in ("auto", "off"):
        raise ValueError("expert_text must be auto or off")
    chain, _ = expand_chain(chain, env)
    timeline = tuple(timeline or ()) + tuple(row for e in chain for row in e.timeline)
    axis_report = [
        f"{os.path.basename(e.path)}: axis {axis} ({reason})"
        for e in chain
        if e.kind == "lora" and e.enabled
        for axis, reason in [resolve_axis(e)]
    ]
    axis_report += [
        text("report.entry_window", name=os.path.basename(e.path), start=e.start, end=e.end)
        for e in chain
        if e.kind == "lora" and e.enabled and (e.start, e.end) != (0.0, 1.0)
    ]
    chain = ModChain(dataclasses.replace(e, axis=resolve_axis(e)[0]) for e in chain)
    overrides = _timeline_overrides(timeline, chain, joint)
    if "route_freeze" in overrides:
        freeze = overrides["route_freeze"]
    if "gate_kappa" in overrides:
        gate = ((gate or (None, None))[0], overrides["gate_kappa"])
    plan, binding = plan_chain(
        patcher,
        chain,
        env,
        joint,
        joint_choice_override,
        content_keys=content_keys,
        lycoris_apply=lycoris_apply,
    )
    mods = plan.mods
    if schedule is None and depth is not None and depth >= 0:
        mods = [m for m in mods if not (m.manifest or {}).get("schedule_yields_to_depth")]
    used_schedule, source = depth_schedules.effective(mods, schedule)
    used = plan.depth(depth) if used_schedule is None else depth_schedules.max_depth(used_schedule)
    require_ok(plan, used)
    scoped = plan.scoped_sets()
    if used > 16 and any(
        item.adapter_set.key_source != "directions" or not item.adapter_set.extra.get("fine")
        for item in scoped
    ):
        raise ValueError("depth exceeds the limit of 16 for these sets")
    if used > 64:
        raise ValueError("depth exceeds the limit of 64 for direction sets")
    if gate_head_limit is not None:
        mode, _ = gate or (None, None)
        width, _ = walks or (None, None)
        if width:
            width = check_path_width(width)
        if mode not in (None, "mod"):
            mode = check_gate_mode(mode)
        if any(
            (width or s.path_width) > 1
            and (mode if mode not in (None, "mod") else s.gate_mode) in ("adaptive", "both", "auto")
            for s in plan.scoped_sets()
        ):
            raise KiLoRAError(text("err.gate_history_width"))
    record_composition(plan, used)
    out = clear(patcher)
    routing = (routing_scope, routing_select)
    _remember(out, chain, depth, env, iteration, schedule, used_schedule, routing)
    if isinstance(getattr(out, "model_options", None), dict):
        out.model_options[APPLIED_KEY]["effective_depth"] = used
        out.model_options[APPLIED_KEY]["gate"] = gate
        out.model_options[APPLIED_KEY]["gate_history"] = gate_history
        out.model_options[APPLIED_KEY]["walks"] = walks
        out.model_options[APPLIED_KEY]["freeze"] = freeze
        out.model_options[APPLIED_KEY]["timeline"] = timeline
        out.model_options[APPLIED_KEY]["token_depth"] = token_depth
        out.model_options[APPLIED_KEY]["alpha_mask"] = alpha_mask
        out.model_options[APPLIED_KEY]["axis_alpha"] = axis_alpha
        out.model_options[APPLIED_KEY]["expert_text"] = expert_text
        out.model_options[APPLIED_KEY]["alpha_clip"] = alpha_clip
        out.model_options[APPLIED_KEY]["joint"] = joint
        out.model_options[APPLIED_KEY]["import_score"] = import_score
        out.model_options[APPLIED_KEY]["lycoris_apply"] = lycoris_apply
        out.model_options[APPLIED_KEY]["gate_stats"] = gate_stats
        out.model_options[APPLIED_KEY]["variation"] = variation
        out.model_options[APPLIED_KEY]["candidates"] = candidates
        out.model_options[APPLIED_KEY]["candidate_summary"] = candidate_summary
        out.model_options[APPLIED_KEY]["routing_features"] = routing_features_enabled
        out.model_options[APPLIED_KEY]["cross_weights"] = cross_weights
        out.model_options[APPLIED_KEY]["linear_jump"] = linear_jump
        out.model_options[APPLIED_KEY]["jump_threshold"] = jump_threshold
        out.model_options[APPLIED_KEY]["bypass"] = bypass
        out.model_options[APPLIED_KEY]["depth_steps"] = depth_steps

    def report() -> str:
        lines = plan.report(used)
        from ..key_graph import enabled as graph_enabled

        if graph_enabled() and plan.mods and used:
            lines += "\nKey graph: qualification will be reported after the first sample."
        if axis_report:
            lines += "\n" + "\n".join(axis_report)
        if timeline:
            lines += "\n" + _timeline_report(timeline)
        hints = [n for m in plan.mods for n in m.manifest.get("notes", [])]
        hints += _role_hints(plan, chain) + _axis_hints(chain)
        if hints:
            lines += "\n" + "\n".join(hints)
        if used_schedule is not None:
            lines += "\n" + depth_schedules.describe(used_schedule, source)
        return lines

    if not plan.mods or used == 0:
        return out, report()
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 0:
        raise ValueError("candidates must be a nonnegative integer")
    if candidate_summary not in ("keys", "read_map"):
        raise ValueError("candidate_summary must be keys or read_map")
    scoped = [
        dataclasses.replace(
            s,
            candidates=candidates,
            candidate_summary=candidate_summary,
            cross_weights=cross_weights,
        )
        for s in _with_regions(plan, chain)
    ]
    if gate_head_limit is not None:
        scoped = [dataclasses.replace(s, gate_head_limit=gate_head_limit) for s in scoped]
    if iteration in ("accumulate", "path"):
        scoped = [dataclasses.replace(s, iteration=iteration) for s in scoped]
    if routing_scope is not None:
        scoped = [dataclasses.replace(s, routing_scope=routing_scope) for s in scoped]
    if import_score is not None:
        scoped = [dataclasses.replace(s, import_score=import_score) for s in scoped]
    if variation is not None:
        scoped = [dataclasses.replace(s, derived_entry_temp=variation) for s in scoped]
    if routing_select is not None:
        scoped = [dataclasses.replace(s, routing_select=routing_select) for s in scoped]
    mode, kappa = gate or (None, None)
    if mode not in (None, "mod"):
        scoped = [dataclasses.replace(s, gate_mode=check_gate_mode(mode)) for s in scoped]
    if kappa:
        scoped = [dataclasses.replace(s, gate_kappa=check_kappa(kappa)) for s in scoped]
    width, select = walks or (None, None)
    if width:
        scoped = [dataclasses.replace(s, path_width=check_path_width(width)) for s in scoped]
    if select not in (None, "mod"):
        scoped = [dataclasses.replace(s, path_select=check_path_select(select)) for s in scoped]
    if freeze is False:
        scoped = [dataclasses.replace(s, route_freeze=None) for s in scoped]
    elif freeze is not None:
        value = check_route_freeze(freeze)
        scoped = [dataclasses.replace(s, route_freeze=value) for s in scoped]
    if depth_steps is not None and tuple(depth_steps) != DEPTH_STEP_DEFAULTS:
        names = ("radial_step", "orthogonal_step", "radial_budget")
        values = {n: check_depth_step(n, v) for n, v in zip(names, depth_steps, strict=True)}
        scoped = [dataclasses.replace(s, **values) for s in scoped]
    for s in scoped:
        check_routing(s.routing_scope, s.routing_select)
    plans = plan_targets(
        scoped,
        used,
        used_schedule,
        merge_plain=not timeline
        and os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower() != "layer",
    )
    from ..compute import derived_mode
    from ..compute import import_score as effective_import_score

    needs_host = {
        target
        for target, plan in plans.items()
        if any(s.form == "input" and effective_import_score(s) == "derived" for s in plan.sets)
    }
    if needs_host and derived_mode() != "full":
        needs_host.clear()
    host_norms = host_weight_norms(out, needs_host) if needs_host else {}
    for target in needs_host:
        for layer_set in plans[target].sets:
            layer_set.host_weight_norm = host_norms.get(target)
    if token_depth is not None:
        for target_plan in plans.values():
            target_plan.token_depth = token_depth
    resolve = layout_resolver(binding.family, _patch_size(out))
    if resolve is not None:
        for target, target_plan in plans.items():
            target_plan.layout = resolve(target)
    _warm_up_shapes(out, plans)

    def place(target: str):
        try:
            p = binding.to_comfy(target)
        except KeyError:
            raise KiLoRAError(
                text("err.mod_not_fit", layer=target, family=binding.family)
            ) from None
        return p.path, p.part

    def out_features(path: str) -> int:
        return linear_shape(_base_module(out, path))[1]

    mode = bypass if bypass is not None else os.environ.get("KILORA_BYPASS", "auto")
    if mode not in BYPASS_MODES:
        raise KiLoRAError(f"bypass must be one of {BYPASS_MODES}, not {mode!r}")
    hooks = []
    can_inject = callable(getattr(out, "set_injections", None))
    fallback_logged = False

    def place_change(path: str, base: nn.Module, after, patch: nn.Module, full_after=None) -> None:
        nonlocal fallback_logged
        hook = mode != "off" or is_quantized(base)
        if hook and can_inject:
            hooks.append(ForwardHook(path, after, full_after))
        else:
            if hook and not fallback_logged:
                log.warning("KiLoRA: patcher has no set_injections; using layer copies")
                fallback_logged = True
            out.add_object_patch(path, patch)

    linear_jump = linear_jump or os.environ.get("KILORA_LINEAR_JUMP", "0").strip() == "1"
    if linear_jump:
        for target_plan in plans.values():
            for layer_set in target_plan.sets:
                layer_set.jump_fits = {}
                layer_set.jump_threshold = jump_threshold
    placed = module_plans(plans, place, out_features)
    transfer = next(
        (
            item.adapter_set.extra.get("_library_transfer")
            for item in scoped
            if item.adapter_set.extra.get("_library_transfer")
        ),
        None,
    )
    if transfer is not None:
        from ..library.prefetch import PlanPrefetch

        PlanPrefetch(list(placed.values()), transfer)
    fixed_by_path = _fixed_changes(plan, chain)
    from .expert_text import install as install_expert_text

    text_groups = install_expert_text(
        out,
        fixed_by_path,
        expert_text == "auto" and binding.family == "anima",
        axis_alpha == "auto" or alpha_mask is not None,
    )
    if expert_text == "auto" and binding.family == "anima" and text_groups:
        denoiser = getattr(getattr(out, "model", None), "diffusion_model", None)
        targets = set()
        if denoiser is not None:
            for path, _ in denoiser.named_modules():
                for target, _ in binding.from_comfy(path):
                    if target.endswith((".attn2.to_k", ".attn2.to_v")):
                        targets.add(target)
        for name, mounts in text_groups.items():
            reached = 0
            for target in targets:
                target_plan = plans.get(target)
                if target_plan is None:
                    continue
                for i, layer_set in enumerate(target_plan.sets):
                    names = (
                        target_plan.expert_names[i] if i < len(target_plan.expert_names) else None
                    )
                    if not names:
                        tag = target_plan.tags[i] if i < len(target_plan.tags) else ""
                        names = [tag] if tag else []
                    if layer_set.form == "input" and name in names:
                        reached += 1
                        break
            axis_report.append(
                f"Expert {name}: {len(mounts)} text mounts; text change reaches "
                f"{reached} of {len(targets)} cross-attention projections."
            )
    _working_memory(out, placed, fixed_by_path)
    clear_working_copies = None
    if hasattr(out, "add_callback_with_key"):
        plan_refs = [weakref.ref(item) for item in placed.values()]
        change_refs = [weakref.ref(change) for group in fixed_by_path.values() for change in group]

        def clear_working_copies(*args, **kwargs):
            for ref in plan_refs:
                module_plan = ref()
                if module_plan is not None:
                    module_plan._cache.clear()
                    if module_plan._prefetch is not None:
                        module_plan._prefetch.reset()
            for ref in change_refs:
                change = ref()
                cache = getattr(change, "_cast", None)
                if cache is not None:
                    cache.clear()

        def clear_on_detach(patcher, *args, **kwargs):
            from .bypass import retaining

            if not retaining(patcher):
                for hook in hooks:
                    hook.eject()
                clear_working_copies()

        def clear_on_cleanup(patcher, *args, **kwargs):
            eject_model = getattr(patcher, "eject_model", None)
            if callable(eject_model):
                eject_model()
            for hook in hooks:
                hook.eject()
            clear_working_copies()

        detach, cleanup = _working_copy_events()
        out.add_callback_with_key(detach, "kilora", clear_on_detach)
        out.add_callback_with_key(cleanup, "kilora", clear_on_cleanup)
    share_tags = {
        c.tag
        for changes in fixed_by_path.values()
        for c in changes
        if getattr(c, "expert_count", 1) > 1 and c.tag in JOINT_SETS.values()
    }
    for path, module_plan in placed.items():
        from ..bake import run_plan_forward

        base = _base_module(out, path)
        if os.environ.get("KILORA_SHARE_TRANSITION_KIND") == "base":
            from ..compute import base_weight_transition

            weight = read_base_weight(base, path)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                for offset, rows, target_plan in module_plan.parts:
                    part = weight if rows == -1 else weight[offset : offset + rows]
                    for layer_set in target_plan.sets:
                        layer_set.share_transition_base = base_weight_transition(layer_set, part)
        live = out.get_model_object(path)
        live_layer = live if is_baked(live) or is_fixed(live) else None
        patch, after = baked_copy(base, module_plan, source=live_layer), module_plan.run
        changes = fixed_by_path.pop(path, None)
        if changes:
            patch = fixed_copy(patch, changes)
            from ..sections import ProgramChange

            has_program = any(isinstance(change, ProgramChange) for change in changes)
            if has_program:
                from .. import compute, fused

                def program_run(x, y, run=module_plan.run):
                    fused._note("a layer-change program")
                    token = compute._PROGRAM_LAYER.set(True)
                    try:
                        return run(x, y)
                    finally:
                        compute._PROGRAM_LAYER.reset(token)

                after = _both(program_run, patch._kilora_change)
            else:
                after = _both(module_plan.run, patch._kilora_change)
        else:
            has_program = False

        def full_after(
            x,
            original,
            *args,
            _plan=module_plan,
            _fixed=patch._kilora_change if changes else None,
            _program=has_program,
            **kwargs,
        ):
            if _program:
                from .. import compute, fused

                fused._note("a layer-change program")
                token = compute._PROGRAM_LAYER.set(True)
                try:
                    result = run_plan_forward(_plan, original, x, *args, **kwargs)
                finally:
                    compute._PROGRAM_LAYER.reset(token)
            else:
                result = run_plan_forward(_plan, original, x, *args, **kwargs)
            return _fixed(x, result) if _fixed is not None else result

        place_change(path, base, after, patch, full_after)
    for path, changes in fixed_by_path.items():
        base = _base_module(out, path)
        live = out.get_model_object(path)
        live_layer = live if is_baked(live) or is_fixed(live) else None
        fixed = fixed_copy(base, changes, source=live_layer)
        place_change(path, base, fixed._kilora_change, fixed)
    if hooks:
        out.set_injections(INJECTION_KEY, [injection(hooks, clear_working_copies)])
    _key_pass(out, placed, chain, plan, binding, routing_scope, joint, linear_jump)
    carry = any(m.settings.carry_across_steps for m in plan.mods)
    auto_alpha = None
    if axis_alpha == "auto" and alpha_mask is None:
        from .auto_alpha import AutoAlpha, install_sampler_hook, trigger_ids

        missing: list[str] = []
        triggers = trigger_ids(chain, alpha_clip, missing)
        denoiser = getattr(getattr(out, "model", None), "diffusion_model", None)
        blocks = getattr(denoiser, "blocks", ())
        from ..compute import budget_share

        share = budget_share({e.axis for e in chain if e.enabled})
        auto_alpha = AutoAlpha(
            binding.family,
            triggers,
            len(blocks),
            _patch_size(out),
            share,
            chain=chain,
            clip=alpha_clip,
            denoiser=denoiser,
            reports=axis_report,
        )
        install_sampler_hook(out.model_options)
        if binding.family != "anima":
            missing.insert(
                0, f"trigger token positions are known on Anima only, not {binding.family}"
            )
        if not blocks:
            missing.append("the model has no transformer block list")
        if missing:
            axis_report.append("Trigger attention maps are off: " + "; ".join(missing) + ".")
        if os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower() in ("single", "soft"):
            axis_report.append(
                "Axis owner masks override trigger attention alpha after the first step."
            )
    _key_mask_by_step(
        out,
        carry,
        binding.family,
        _patch_size(out),
        timeline,
        joined_names(chain, joint),
        auto_alpha,
        plans=plans,
        mount_share_tags=share_tags | ({JOINT_SET} if share_observer is not None else set()),
        share_observer=share_observer,
        collect_routing_features=routing_features_enabled or any(cross_weights),
        graph_mutable=bool(
            transfer
            or timeline
            or auto_alpha
            or alpha_mask is not None
            or token_depth is not None
            or live_sessions(plan)
        )
        or any(
            e.region is not None or e.panels is not None or (e.start, e.end) != (0.0, 1.0)
            for e in chain
        ),
    )
    if axis_alpha == "auto" and alpha_mask is not None:
        _mask_alpha_wrapper(out, alpha_mask)
    _gate_stats_wrapper(out, gate_stats)
    from .auto_alpha import trigger_ids
    from .layer_alpha import install_layer_alpha

    denoiser = getattr(getattr(out, "model", None), "diffusion_model", None)
    blocks = getattr(denoiser, "blocks", ())
    count = len(blocks) if hasattr(blocks, "__len__") else 0
    install_layer_alpha(
        out,
        binding.family,
        trigger_ids(chain, alpha_clip),
        count,
        ip_mounted=any(
            getattr(s, "axis", None) == "ip" for plan in plans.values() for s in plan.sets
        ),
    )
    from .layer_crop import install_layer_crop

    install_layer_crop(
        out,
        _patch_size(out),
        safe=(
            not carry
            and auto_alpha is None
            and os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower() != "layer"
        ),
    )
    return out, report()


def _gate_stats_wrapper(out: Any, mode: str) -> None:
    from ..compute import gate_stats_mode

    options = out.model_options
    previous = options.get("model_function_wrapper")

    def wrapper(apply_model, args):
        with gate_stats_mode(mode):
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper


def _both(first, second):
    return lambda x, y: second(x, first(x, y))


def _mask_alpha_wrapper(out: Any, mask: torch.Tensor, feather_px: int = 16) -> None:
    from ..call_values import axis_alphas
    from ..depth_map import feather

    values = mask.detach().float().clamp(0, 1)
    if values.dim() == 2:
        values = values[None]
    if values.dim() != 3:
        raise ValueError("alpha_mask must have shape [batch, height, width]")
    values = feather(values, feather_px)
    options = out.model_options
    options["kilora_alpha_mask"] = True
    previous = options.get("model_function_wrapper")

    def wrapper(apply_model, args):
        import torch.nn.functional as F

        shape = args["input"].shape[2:]
        if len(shape) != 2:
            raise ValueError("alpha_mask requires a two-dimensional latent")
        boxes = (args.get("c") or {}).get("kilora_crop_window")
        image = values if boxes is not None else F.adaptive_avg_pool2d(values[:, None], shape)[:, 0]
        with axis_alphas({"ip": image, "creative": 1 - image}):
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper


def _key_pass(
    out: Any,
    placed: dict,
    chain: Sequence[ModEntry],
    plan: Any,
    binding: Binding,
    scope: str | None,
    joint: bool,
    linear_jump: bool = False,
) -> None:
    from ..sets import needs_state_keys, set_identity
    from .state_keys import KeyPass, joint_sha256

    options = getattr(out, "model_options", None)
    if options is None:
        return
    jobs = []
    for axis, name in JOINT_SETS.items():
        loras = [e for e in chain if _joint_lora(e, axis)]
        if joint and len(loras) > 1:
            jobs.append((name, joint_sha256([(e.path, float(e.strength)) for e in loras])))
    for m in plan.mods:
        for s in m.sets:
            if s.name not in JOINT_SETS.values() and needs_state_keys(s):
                jobs.append((s.extra.get("live_id", s.name), set_identity(s)))
    from ..compute import import_score as effective_import_score
    from ..compute import share_transition_gamma
    from ..key_graph import enabled as graph_enabled

    required = set()
    for module_plan in placed.values():
        for _, _, target in module_plan.parts:
            for layer_set in target.sets:
                if (
                    layer_set.form != "input"
                    or effective_import_score(layer_set) != "derived"
                    or (
                        share_transition_gamma(layer_set)
                        and os.environ.get("KILORA_SHARE_TRANSITION_KIND") == "weighted"
                    )
                ):
                    required.add(layer_set.key)
    jobs = [(name, identity) for name, identity in jobs if name in required]
    if not jobs and not linear_jump and not graph_enabled():
        return
    key_pass = KeyPass(
        list(placed.values()), jobs, binding.family, scope or "token", linear_jump, model=out.model
    )
    options.setdefault(APPLIED_KEY, {})["key_pass"] = key_pass
    previous = options.get("model_function_wrapper")
    wrapper = key_pass.wrap(previous)
    wrapper._kilora_graph_safe = previous is None
    wrapper._kilora_key_pass = key_pass
    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper


def _warm_up_shapes(patcher: Any, plans: dict) -> None:
    try:
        from .. import fused

        dtypes = None
        model = getattr(patcher, "model", None)
        dtype = getattr(model, "manual_cast_dtype", None)
        if dtype is None and hasattr(model, "get_dtype"):
            dtype = model.get_dtype()
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtypes = (dtype,)
        fused.warm_up_sets([(p.sets, p.depth) for p in plans.values()], dtypes=dtypes)
    except Exception:
        pass


def apply_clip(
    patcher: Any,
    clip: Any,
    chain: Sequence[ModEntry],
    env: LiveEnv | None = None,
    timeline: Any = None,
) -> tuple[Any, str]:
    from ..text_encoder_lora import apply_to_clip

    if clip is None:
        return None, ""
    chain, _ = expand_chain(chain, env)
    timeline = tuple(timeline or ()) + tuple(row for e in chain for row in e.timeline)
    chain = ModChain(dataclasses.replace(e, axis=resolve_axis(e)[0]) for e in chain)
    plan, _ = plan_chain(patcher, chain, env)
    mods = plan.mods
    if timeline:
        from ..timeline import Clock, scale_for

        mods = []
        for m in plan.mods:
            e = _entry_of(m, chain)
            name = entry_name(e) if e is not None else (m.label or "")
            mult = scale_for(timeline, Clock(), name, e.axis if e is not None else "creative")
            mods.append(m if mult == 1.0 else dataclasses.replace(m, strength=m.strength * mult))
    out, lines = apply_to_clip(clip, mods)
    return out, "\n".join(lines)


def _entry_of(m: Any, chain: Sequence[ModEntry]) -> ModEntry | None:
    for e in reversed(chain):
        if m.path is not None and os.fspath(e.path) == m.path:
            return e
        if m.path is None and os.path.basename(e.path) == m.label:
            return e
    return None


def _is_joint(s: Any) -> bool:
    return getattr(s.adapter_set, "name", None) in JOINT_SETS.values()


def entry_name(e: ModEntry) -> str:
    name = os.path.basename(os.fspath(e.path))
    for ext in (".safetensors", ".kmod"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def _with_regions(plan: Any, chain: Sequence[ModEntry]) -> list:
    from ..roles import file_policy, first_policy

    by_path = {os.fspath(e.path): e.region for e in chain if e.region is not None}
    by_label = {os.path.basename(e.path): e.region for e in chain if e.region is not None}
    scoped = plan.scoped_sets()
    regions, roles, axes, tags = [], [], [], []
    for m in plan.mods:
        e = _entry_of(m, chain)
        axes.extend([e.axis if e is not None else "creative"] * len(m.sets))
        tags.extend([entry_name(e) if e is not None else (m.label or "")] * len(m.sets))
        lora_file = e.path if e is not None and e.kind == "lora" else None
        policy = first_policy(
            e.roles if e is not None else None, m.settings.token_roles, file_policy(lora_file)
        )
        r = by_path.get(m.path) if m.path is not None else by_label.get(m.label)
        regions.extend([r] * len(m.sets))
        roles.extend([policy] * len(m.sets))
    groups = {axis: [e for e in chain if _joint_lora(e, axis)] for axis in JOINT_SETS}
    panels = next((e.panels for e in reversed(chain) if e.panels is not None), None)
    out = []
    for s, r, p, a, tag in zip(scoped, regions, roles, axes, tags, strict=True):
        joint_axis = next(
            (
                axis
                for axis, name in JOINT_SETS.items()
                if getattr(s.adapter_set, "name", None) == name
            ),
            None,
        )
        loras = groups.get(joint_axis, [])
        if joint_axis is not None:
            a = joint_axis
        s = dataclasses.replace(
            s,
            region=r,
            roles="all" if joint_axis else p,
            panels=panels,
            tag=JOINT_SETS[joint_axis] if joint_axis else tag,
            expert_regions=(
                [e.region for e in loras] if any(e.region is not None for e in loras) else None
            )
            if joint_axis
            else None,
            expert_roles=[first_policy(e.roles, file_policy(e.path)) for e in loras]
            if joint_axis
            else None,
            expert_names=[entry_name(e) for e in loras] if joint_axis else None,
        )
        if a != "creative":
            defaults = {"functional": FUNCTIONAL_DEFAULTS, "ip": IP_DEFAULTS}.get(a, {})
            if a == "ip":
                mode = os.environ.get("KILORA_IP_DEPTH", "1")
                if mode not in ("chain", "1"):
                    raise ValueError("KILORA_IP_DEPTH must be chain or 1")
                defaults = dict(defaults, fixed_depth=1 if mode == "1" else None)
            s = dataclasses.replace(s, axis=a, **defaults)
        out.append(s)
    return out


def _axis_hints(chain: Sequence[ModEntry]) -> list[str]:
    from ..roles import looks_functional

    return [
        text("msg.axis_hint", name=os.path.basename(e.path))
        for e in chain
        if e.kind == "lora" and e.enabled and e.axis == "creative" and looks_functional(e.path)
    ]


def _role_hints(plan: Any, chain: Sequence[ModEntry]) -> list[str]:
    from ..roles import file_policy, first_policy, looks_like_edit

    lines = []
    for e in chain:
        if e.kind != "lora" or not e.enabled or not looks_like_edit(e.path):
            continue
        if first_policy(e.roles, file_policy(e.path)) == "auto":
            lines.append(text("msg.roles_edit_hint", name=os.path.basename(e.path)))
    return lines


class StepCarry:
    def __init__(self) -> None:
        self.caches: dict = {}

    def cache_for(self, args: dict) -> dict:
        kind = tuple(
            (args.get("c") or {}).get("transformer_options", {}).get("cond_or_uncond") or ()
        )
        from ..call_values import host_list, host_scalar

        x, t = args["input"], args["timestep"]
        step = host_scalar(t)
        shape = tuple(x.shape)
        entry = self.caches.get(kind)
        sigmas = (args.get("c") or {}).get("transformer_options", {}).get("sample_sigmas")
        first = (
            sigmas is not None
            and len(sigmas)
            and (host_list(sigmas)[0] if isinstance(sigmas, torch.Tensor) else float(sigmas[0]))
            == step
        )
        if entry is None or first or step >= entry["step"] or entry["shape"] != shape:
            entry = {"step": step, "shape": shape, "state": {}}
            self.caches[kind] = entry
        entry["step"] = step
        return entry["state"]


def _patch_size(patcher: Any) -> int | None:
    denoiser = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    patch = getattr(denoiser, "patch_size", None)
    return patch if isinstance(patch, int) and not isinstance(patch, bool) else None


def _key_mask_by_step(
    out: Any,
    carry: bool = False,
    family: str | None = None,
    patch: int | None = None,
    timeline: Any = None,
    joined: frozenset = frozenset(),
    auto_alpha: Any = None,
    plans: dict | None = None,
    graph_mutable: bool = False,
    mount_share_tags: frozenset[str] | set[str] = frozenset(),
    collect_routing_features: bool = False,
    share_observer: Any = None,
) -> None:
    from ..call_values import (
        active_axes,
        crop_window,
        host_scalar,
        host_values,
        ip_early,
        ip_early_threshold,
        mount_shares,
        row_axes,
    )
    from ..compute import (
        derived_call,
        dropout_key,
        observing_set_share_layers,
        sampling_progress,
        speculate,
        speculative_cache_bytes,
        step_carry,
    )
    from ..region import latent_size, token_layouts
    from ..roles import reference_shapes, shapes_of
    from ..route_freeze import fold_counts, freezing

    options = getattr(out, "model_options", None)
    if options is None:
        return
    previous = options.get("model_function_wrapper")
    from .. import ablation_settings

    carrier = StepCarry()
    try:
        patcher_ref = weakref.ref(out)
    except TypeError:

        def patcher_ref():
            return None

    freezer = StepCarry()
    share_cache = StepCarry()
    speculative_cache = StepCarry()
    resolve = layout_resolver(family, patch)
    from .cuda_graphs import Graphs
    from .cuda_graphs import requested as graphs_requested
    from .cuda_graphs import safe as graph_safe

    graphs = Graphs()
    from .layer_batch import install_layer_batch

    install_layer_batch(out, separate=carry, family=family)
    targets = {
        name: next((s.route_freeze for s in p.sets if s.down is not None), None)
        for name, p in (plans or {}).items()
        if any(s.down is not None for s in p.sets)
    }
    share_keys = {
        s.key
        for p in (plans or {}).values()
        for s in p.sets
        if s.key in mount_share_tags and s.keys_in.shape[0] > 1 and s.down is not None
    }
    observe_shares = bool(share_keys)
    graph_mutable = (
        any(getattr(s, "entry_windows", ()) for p in (plans or {}).values() for s in p.sets)
        or graph_mutable
        or observe_shares
        or os.environ.get("KILORA_SPECULATE") == "1"
        or (previous is not None and not getattr(previous, "_kilora_graph_safe", False))
    )

    speculation_reported = False
    from .crop_positions import diffusion_model, global_positions, mixed_note

    mixed_windows = mixed_note()
    owner_state = {"owner": None, "step": None, "shape": None, "warned_frames": False}
    early_log = {"step": None, "active": None}
    compose_log = {"other_maps": False}
    ip_mounted = (
        None
        if plans is None
        else any(getattr(s, "axis", None) == "ip" for plan in plans.values() for s in plan.sets)
    )

    def model_call(apply_model, args):
        nonlocal speculation_reported
        c = dict(args.get("c") or {})
        ids = c.pop("kilora_layer_ids", None)
        compose_ids = c.pop("kilora_compose_id", None)
        boxes = c.pop("kilora_crop_window", None)
        from .expert_text import _PREFIX, conditioning_values

        texts = {k[len(_PREFIX) :]: c.pop(k) for k in tuple(c) if k.startswith(_PREFIX)}
        args = {**args, "c": c}
        call_carry = ablation_settings.carry_steps(carry)
        state = carrier.cache_for(args) if call_carry else None
        if not call_carry:
            carrier.caches.clear()
        carried = step_carry(state) if state is not None else contextlib.nullcontext()
        spec_cache = (
            speculative_cache.cache_for(args) if os.environ.get("KILORA_SPECULATE") == "1" else None
        )
        speculation = speculate(spec_cache) if spec_cache is not None else contextlib.nullcontext()
        cache = freezer.cache_for(args)
        shares = share_cache.cache_for(args)
        collected: dict = {}
        sample_targets: dict = {}
        sample_rows: dict = {}
        sample_attempts = 0

        def wants(target, sets):
            nonlocal sample_attempts
            if sample_attempts >= 4 or share_keys <= collected.keys():
                return False
            from ..region import current_layout

            layout = current_layout(target.split("@", 1)[0])
            eligible = not (
                layout is not None and (layout.kind == "text1d" or layout.role == "reference")
            ) and any(
                s.key in share_keys and s.key not in collected and s.down is not None for s in sets
            )
            if eligible:
                sample_attempts += 1
            return eligible

        def observe(target, gates, sets):
            if sets is None:
                return
            from ..call_values import current_route_rows
            from ..region import current_latent_shape, current_layout, layer_region

            target = target.split("@", 1)[0]
            layout = current_layout(target)
            if layout is not None and (layout.kind == "text1d" or layout.role == "reference"):
                return
            for gate, layer_set in zip(gates, sets, strict=True):
                if (
                    layer_set.key not in share_keys
                    or (layer_set.key in sample_targets and sample_targets[layer_set.key] != target)
                    or gate.shape[-1] < 2
                ):
                    continue
                g = gate.float().reshape(-1, gate.shape[-2], gate.shape[-1])
                mask = None
                latent = current_latent_shape()
                if layout is not None and latent is not None and g.shape[1] > 1:
                    try:
                        mask = layer_region(
                            g.new_ones((1, *latent[-2:])), g.shape[1], target=target
                        )
                    except ValueError:
                        continue
                if mask is None:
                    tokens = 1
                    if g.shape[1] == 1 and layout is not None and latent is not None:
                        with contextlib.suppress(ValueError):
                            tokens = math.prod(layout.grid(latent))
                    mask = g.new_full((1, g.shape[1]), tokens)
                mask = mask.to(g.device, g.dtype)
                total = (g * mask[..., None]).sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).expand(g.shape[0], 1)
                rows = current_route_rows()
                row_key = None if rows is None else tuple(rows.tolist())
                seen = sample_rows.setdefault(layer_set.key, set())
                if row_key in seen:
                    continue
                seen.add(row_key)
                sample_targets[layer_set.key] = target
                if rows is not None:
                    rows = rows.to(g.device)
                    batch = args["input"].shape[0]
                    total = total.new_zeros((batch, total.shape[-1])).index_add(0, rows, total)
                    count = count.new_zeros((batch, 1)).index_add(0, rows, count)
                if layer_set.key in collected:
                    old, n = collected[layer_set.key]
                    total, count = old + total, n + count
                collected[layer_set.key] = (total, count)

        kind = tuple(
            (args.get("c") or {}).get("transformer_options", {}).get("cond_or_uncond") or ()
        )
        from ..axis_owner import AxisOwner
        from ..axis_owner import mode as owner_mode
        from ..axis_owner import recording as record_owner

        owner = None
        dimensions = args["input"].shape[2:]
        owner_enabled = owner_mode() in ("single", "soft")
        if compose_ids is not None and (owner_enabled or auto_alpha is not None):
            if not compose_log["other_maps"]:
                log.info(
                    "KiLoRA: compose single uses the layer axis maps; the axis owner and "
                    "the trigger attention maps are not used."
                )
                compose_log["other_maps"] = True
            owner_enabled = False
        threshold = ip_early_threshold()
        step = None
        if threshold or owner_enabled:
            step = host_scalar(args["timestep"])
        early = bool(threshold and step > threshold)
        if threshold and (early_log["step"], early_log["active"]) != (step, early):
            log.info("axis ip early: sigma=%s active=%s", step, early)
            early_log.update(step=step, active=early)
        if owner_enabled and len(dimensions) >= 2 and all(n == 1 for n in dimensions[:-2]):
            spatial = tuple(args["input"].shape[-2:])
            batch = args["input"].shape[0]
            if kind and batch % len(kind) == 0:
                batch //= len(kind)
            if (
                owner_state["owner"] is None
                or owner_state["step"] is None
                or step > owner_state["step"]
                or spatial != owner_state["shape"]
                or batch != owner_state["owner"].batch
            ):
                owner_state["owner"] = AxisOwner(
                    spatial, batch, args["input"].device, axes=("ip", "creative")
                )
            elif step != owner_state["step"]:
                owner_state["owner"].finish_step()
            owner_state["step"], owner_state["shape"] = step, spatial
            owner = owner_state["owner"]
            if owner.masks:
                from ..region_attention import (
                    CallState,
                    RegionAttention,
                    comfy_override,
                    kind_patches,
                )

                c = dict(args.get("c") or {})
                topts = dict(c.get("transformer_options") or {})
                state = CallState(
                    latent=spatial,
                    owners=owner.masks,
                    cond_rows=_cond_rows(topts, args["input"].shape[0]),
                    owner_mode=owner_mode(),
                    owner_tracker=owner,
                )
                patches = {k: list(v) for k, v in (topts.get("patches") or {}).items()}
                for name, fn in kind_patches(state).items():
                    patches[name] = [*patches.get(name, []), fn]
                topts["patches"] = patches
                topts["optimized_attention_override"] = comfy_override(
                    (), RegionAttention(), state, topts.get("optimized_attention_override")
                )
                c["transformer_options"] = topts
                args = {**args, "c": c}
        else:
            if (
                owner_enabled
                and len(dimensions) > 2
                and any(n != 1 for n in dimensions[:-2])
                and not owner_state["warned_frames"]
            ):
                log.info("axis owner: disabled for multiple frames")
                owner_state["warned_frames"] = True
            owner_state.update(owner=None, step=None, shape=None)
        graphs.generation(kind, cache)
        frozen = freezing(cache)
        size = tuple(args["input"].shape[2:])
        progress_value = _progress(args)
        progress = sampling_progress(progress_value)
        layouts = (
            token_layouts(resolve, _text_tokens(args)) if resolve else contextlib.nullcontext()
        )
        refs = reference_shapes(shapes_of(args.get("c") or {}))
        timed = _time_scales(timeline, args, joined)
        layer = ((args.get("c") or {}).get("transformer_options") or {}).get("kilora_layer")
        axes = ("creative", "functional") if layer == "creative" else (layer,)
        selected = active_axes(axes) if layer is not None else contextlib.nullcontext()
        collect = collect_routing_features or routing_features_enabled()
        prompt = c.get("c_crossattn") if family == "anima" else c.get("y")
        if prompt is None:
            prompt = c.get("c_crossattn")
        patcher = patcher_ref()
        model = getattr(getattr(patcher, "model", None), "diffusion_model", None)
        projection = prompt_projection(model, family)
        feature_values = (
            call_features(prompt, timestep_features(args["timestep"]), projection)
            if collect
            else contextlib.nullcontext()
        )
        scopes = contextlib.ExitStack()
        scopes.enter_context(selected)
        scopes.enter_context(ip_early(early, ip_mounted))
        composed_alphas = False
        if compose_ids is not None:
            scopes.enter_context(_compose_scope(compose_ids, args))
            axis_scope = _compose_axis_scope(args, progress_value)
            if axis_scope is not None:
                scopes.enter_context(axis_scope)
                composed_alphas = True
        with (
            record_owner(owner, kind) if owner is not None else contextlib.nullcontext(),
            scopes,
            routing_features(collect),
            feature_values,
            conditioning_values(texts),
            row_axes(ids),
            crop_window(boxes),
            timed,
            dropout_key(0, _shared_step(args["timestep"])),
            carried,
            speculation,
            derived_call(),
            frozen,
            latent_size(size),
            progress,
            layouts,
            refs,
            mount_shares(shares),
            observing_set_share_layers(wants, observe)
            if observe_shares
            else contextlib.nullcontext(),
        ):
            if spec_cache and not speculation_reported:
                log.info(
                    "KiLoRA: prediction cache holds %d bytes", speculative_cache_bytes(spec_cache)
                )
                speculation_reported = True

            def invoke(call_args):
                from ..call_values import axis_alphas

                masks = owner.masks if owner is not None and not composed_alphas else None
                if masks:
                    sample = next(iter(masks.values()))
                    alphas = {
                        "ip": masks.get("ip", torch.zeros_like(sample)),
                        "creative": masks.get("creative", torch.zeros_like(sample)),
                        "functional": 1.0,
                    }
                    scope = axis_alphas(alphas)
                else:
                    scope = contextlib.nullcontext()
                positions = (
                    global_positions(
                        diffusion_model(patcher_ref()),
                        boxes,
                        call_args["input"].shape[0],
                        mixed_windows,
                    )
                    if boxes is not None
                    else contextlib.nullcontext()
                )
                with scope, positions:
                    if previous is not None:
                        return previous(apply_model, call_args)
                    return apply_model(call_args["input"], call_args["timestep"], **call_args["c"])

            if auto_alpha is not None and compose_ids is None:
                result = auto_alpha.call(args, invoke)
                for key, (total, count) in collected.items():
                    shares[key] = torch.where(
                        count > 0,
                        total / count.clamp_min(1),
                        shares.get(key, torch.full_like(total, 1.0 / total.shape[-1])),
                    )
                if share_observer is not None and JOINT_SET in shares:
                    share_observer(shares[JOINT_SET], args)
                return result
            key_pass = getattr(previous, "_kilora_key_pass", None)
            if (
                graphs_requested()
                and not ablation_settings.force_loop()
                and os.environ.get("KILORA_CARRY_STEPS") is None
                and os.environ.get("KILORA_GATE_COHERENCE") is None
                and not threshold
                and os.environ.get("KILORA_AXIS_ZBUFFER", "").strip().lower()
                not in ("layer", "single", "soft")
                and not (
                    os.environ.get("KILORA_IMPORT_SCORE") == "derived"
                    and float(os.environ.get("KILORA_DERIVED_CARRY", "0.5")) != 0
                )
                and ids is None
                and boxes is None
                and compose_ids is None
                and (key_pass is None or key_pass.done)
                and patcher_ref() is not None
                and graph_safe(
                    patcher_ref(),
                    args,
                    cache,
                    targets,
                    carry,
                    progress_value,
                    mutable=graph_mutable,
                )
            ):
                result = graphs.call(args, cache, invoke)
            else:
                result = invoke(args)
            for key, (total, count) in collected.items():
                shares[key] = torch.where(
                    count > 0,
                    total / count.clamp_min(1),
                    shares.get(key, torch.full_like(total, 1.0 / total.shape[-1])),
                )
            if share_observer is not None and JOINT_SET in shares:
                share_observer(shares[JOINT_SET], args)
            return result

    first_sample = True
    fold_report = {"folded": 0, "replayed": 0, "recording": 0}
    reported = False

    def run_model_wrapper(apply_model, args):
        nonlocal first_sample, reported
        progress_value = _progress(args)
        if reported and progress_value == 0:
            fold_report.update(folded=0, replayed=0, recording=0)
            reported = False
        with fold_counts(fold_report):
            if first_sample:
                first_sample = False
                from .. import fused

                fused.wait_for_warm_up()
                with fused.sample_compile_report():
                    result = model_call(apply_model, args)
            else:
                result = model_call(apply_model, args)
        if progress_value >= 1 and not reported:
            if any(fold_report.values()):
                log.info(
                    "KiLoRA derived fold after freeze: %d folded, %d replayed unfolded, "
                    "%d recording calls",
                    fold_report["folded"],
                    fold_report["replayed"],
                    fold_report["recording"],
                )
            reported = True
        return result

    def wrapper(apply_model, args):
        from .startup_prepare import model_call

        with model_call(), host_values():
            return run_model_wrapper(apply_model, args)

    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper


def _text_tokens(args: dict) -> int | None:
    context = (args.get("c") or {}).get("c_crossattn")
    return (
        int(context.shape[1]) if isinstance(context, torch.Tensor) and context.dim() == 3 else None
    )


def _progress(args: dict) -> float:
    from ..call_values import host_list, host_scalar

    sigmas = (args.get("c") or {}).get("transformer_options", {}).get("sample_sigmas")
    t = args.get("timestep")
    if sigmas is None or t is None or len(sigmas) < 2:
        return 0.0
    sigma = host_scalar(t)
    if not isinstance(sigmas, torch.Tensor):
        values = torch.as_tensor(sigmas, dtype=torch.float64).flatten()[:-1]
        index = int(torch.argmin((values - sigma).abs()))
        return index / max(len(values) - 1, 1)
    distances = [abs(float(v) - sigma) for v in host_list(sigmas)[:-1]]
    index = next((i for i, d in enumerate(distances) if d != d), None)
    if index is None:
        index = min(range(len(distances)), key=distances.__getitem__)
    return index / max(len(distances) - 1, 1)


def _shared_step(t: Any) -> Any:
    from ..call_values import host_scalar

    if isinstance(t, torch.Tensor) and t.numel():
        return host_scalar(t)
    return t


def _base_module(patcher: Any, path: str) -> nn.Module:
    try:
        module = patcher.get_model_object(path)
    except AttributeError:
        raise KiLoRAError(f"the model has no module {path[len(COMFY_PREFIX) :]}") from None
    if getattr(module, "_kilora_learn", False) is True:
        module = module.inner
    if is_fixed(module):
        module = module.__dict__["_kilora_base"]
    if is_baked(module):
        module = module.__dict__["_kilora_base"]
    return module


def region_masks(patcher: Any, chain: Sequence[ModEntry] | None = None) -> list:
    if chain is not None:
        return [m for m, _ in chain_regions(chain, all_regions=True)]
    return [m for m, _ in region_captions(patcher, all_regions=True)]


def region_captions(patcher: Any, all_regions: bool = False) -> list:
    info = applied(patcher)
    return chain_regions((info or {}).get("chain", ()), all_regions)


def chain_regions(chain: Sequence[ModEntry], all_regions: bool = False) -> list:
    out: list = []
    for e in chain:
        if not e.enabled or e.region is None:
            continue
        for i, (m, cap) in enumerate(out):
            if m is e.region:
                if cap is None and e.region_caption is not None:
                    out[i] = (m, e.region_caption)
                break
        else:
            out.append((e.region, e.region_caption))
    return out if all_regions else [(m, c) for m, c in out if c is not None]


def encode_region_caption(model: Any, caption: Any, device, dtype) -> torch.Tensor:
    tensor, extra = caption[0][0], caption[0][1] if len(caption[0]) > 1 else {}
    tensor = tensor.to(device, dtype)
    denoiser = getattr(model, "diffusion_model", None)
    ids = (extra or {}).get("t5xxl_ids")
    if ids is None or not hasattr(denoiser, "preprocess_text_embeds"):
        return tensor
    weights = extra.get("t5xxl_weights")
    if weights is not None:
        weights = weights.unsqueeze(0).unsqueeze(-1).to(device, dtype)
    with torch.no_grad():
        return denoiser.preprocess_text_embeds(
            tensor, ids.unsqueeze(0).to(device), t5xxl_weights=weights
        )


def _compose_scope(compose_ids: Any, args: dict):
    from ..call_values import compose_layout
    from .layer_compose import lookup

    values = torch.as_tensor(compose_ids).reshape(-1)
    layouts = lookup(int(values[0])) if values.numel() else None
    if layouts is not None:
        _install_compose_attention(args)
    batch = args["input"].shape[0]
    topts = (args.get("c") or {}).get("transformer_options") or {}
    kinds = topts.get("cond_or_uncond")
    if kinds and batch % len(kinds) == 0:
        per = batch // len(kinds)
        rows = torch.tensor([int(k) for k in kinds for _ in range(per)], dtype=torch.long)
    else:
        rows = torch.zeros(batch, dtype=torch.long)
    return compose_layout(layouts, rows)


def _install_compose_attention(args: dict) -> None:
    from ..region_attention import CallState, RegionAttention, comfy_override, kind_patches

    c = args["c"]
    topts = dict(c.get("transformer_options") or {})
    context = c.get("c_crossattn")
    real = None
    if isinstance(context, torch.Tensor) and context.ndim == 3:
        real = context.detach().abs().amax(-1) > 0
    state = CallState(
        progress=_call_clock(args).t,
        latent=tuple(args["input"].shape[-2:]),
        compose=True,
        real=real,
    )
    patches = {k: list(v) for k, v in (topts.get("patches") or {}).items()}
    for name, fn in kind_patches(state).items():
        patches[name] = [*patches.get(name, []), fn]
    topts["patches"] = patches
    topts["optimized_attention_override"] = comfy_override(
        (), RegionAttention(), state, topts.get("optimized_attention_override")
    )
    c["transformer_options"] = topts


def _compose_axis_scope(args: dict, progress: float):
    from ..call_values import axis_alphas, current_compose_layout
    from .layer_compose import axis_maps

    current = current_compose_layout()
    if current is None:
        return None
    layouts, rows = current
    if not layouts or layouts[0] is None:
        return None
    shape = tuple(args["input"].shape[2:])
    if len(shape) < 2 or any(n != 1 for n in shape[:-2]):
        return None
    try:
        maps = axis_maps(layouts, rows, shape[-2:], float(progress or 0.0))
    except ValueError as exc:
        log.info("KiLoRA: compose single has no axis maps for this call: %s", exc)
        return None
    device = args["input"].device
    for alpha in maps.values():
        alpha.image = alpha.image.to(device)
        if alpha.text is not None:
            alpha.text = alpha.text.to(device)
    return axis_alphas(maps)


def _cond_rows(topts: dict, batch: int) -> torch.Tensor:
    kinds = topts.get("cond_or_uncond")
    if not kinds or batch % len(kinds):
        return torch.ones(batch, dtype=torch.bool)
    per = batch // len(kinds)
    return torch.tensor([k == 0 for k in kinds for _ in range(per)], dtype=torch.bool)


def apply_region_attention(patcher: Any, config: Any) -> tuple[Any, int]:
    from ..region_attention import CallState, CaptionLayout, comfy_override, kind_patches

    config = _with_timeline(config, patcher, ("region_attention", "caption_window"))
    out = patcher.clone()
    masks = region_masks(patcher)
    captions = region_captions(patcher) if config.captions else []
    options = getattr(out, "model_options", None)
    acts = config.mode != "off" or bool(captions)
    if not acts or not masks or options is None:
        return out, len(masks) if acts else 0
    previous = options.get("model_function_wrapper")
    encoded: dict = {}

    def contexts(context: torch.Tensor) -> list:
        key = (context.device, context.dtype)
        if key not in encoded:
            model = getattr(out, "model", None)
            encoded[key] = [
                encode_region_caption(model, cap, context.device, context.dtype)
                for _, cap in captions
            ]
        return encoded[key]

    def wrapper(apply_model, args):
        c = dict(args["c"])
        topts = dict(c.get("transformer_options") or {})
        progress = _progress(args)
        cfg = config.at(_call_clock(args))
        state = CallState(progress=progress, latent=tuple(args["input"].shape[-2:]))
        context = c.get("c_crossattn")
        if captions and context is not None and cfg.captions_active(progress):
            extra = contexts(context)
            batch = context.shape[0]
            c["c_crossattn"] = torch.cat([context, *(e.expand(batch, -1, -1) for e in extra)], 1)
            state.captions = CaptionLayout(
                prompt=context.shape[1],
                lengths=tuple(e.shape[1] for e in extra),
                masks=tuple(m for m, _ in captions),
                cond_rows=_cond_rows(topts, batch),
            )
        patches = {k: list(v) for k, v in (topts.get("patches") or {}).items()}
        for name, patch in kind_patches(state).items():
            patches[name] = [*patches.get(name, []), patch]
        topts["patches"] = patches
        topts["optimized_attention_override"] = comfy_override(
            masks, cfg, state, topts.get("optimized_attention_override")
        )
        c["transformer_options"] = topts
        args = {**args, "c": c}
        if previous is not None:
            return previous(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper
    return out, len(masks)


def apply_region_isolation(
    patcher: Any, config: Any, calc: Any = None, chain: Sequence[ModEntry] | None = None
) -> tuple[Any, int]:
    from ..bake import base_only
    from ..region_isolation import BaseBranch, isolate, outside_weight

    config = _with_timeline(config, patcher, ("region_isolation",))
    out = patcher.clone()
    masks = region_masks(patcher, chain)
    options = getattr(out, "model_options", None)
    acts = config.strength > 0 or bool(config.timeline)
    if not acts or not masks or options is None:
        return out, len(masks) if acts else 0
    previous = options.get("sampler_calc_cond_batch_function")
    branch = BaseBranch()
    weights: dict = {}

    def evaluate(args: dict, x: torch.Tensor) -> list:
        a = {**args, "input": x}
        if previous is not None:
            return previous(a)
        fn = calc
        if fn is None:
            from comfy.samplers import calc_cond_batch as fn
        return fn(a["model"], a["conds"], x, a["sigma"], a["model_options"])

    def hook(args: dict) -> list:
        x, sigma = args["input"], args["sigma"]
        topts = (args.get("model_options") or {}).get("transformer_options", {})
        call = {"c": {"transformer_options": topts}, "timestep": sigma}
        progress = _progress(call)
        cfg = config.at(_call_clock(call))
        if not cfg.active(progress):
            return evaluate(args, x)
        key = (tuple(x.shape), x.device, cfg.strength)
        if key not in weights:
            weights[key] = outside_weight(masks, x.shape, cfg.feather, cfg.strength)

        def run(latent: torch.Tensor, base: bool) -> list:
            if base:
                with base_only():
                    return evaluate(args, latent)
            return evaluate(args, latent)

        s = float(sigma.detach().flatten()[0]) if isinstance(sigma, torch.Tensor) else float(sigma)
        return isolate(run, x, s, weights[key], branch)

    def scale(args: dict) -> torch.Tensor:
        branch.scale = float(args.get("cond_scale", 1.0))
        return args["denoised"]

    options["sampler_calc_cond_batch_function"] = hook
    if hasattr(out, "set_model_sampler_post_cfg_function"):
        out.set_model_sampler_post_cfg_function(scale)
    else:
        options["sampler_post_cfg_function"] = [
            *options.get("sampler_post_cfg_function", []),
            scale,
        ]
    return out, len(masks)


def _call_clock(args: dict):
    from ..call_values import host_list, host_scalar
    from ..timeline import clock_of

    t = args.get("timestep")
    sigma = host_scalar(t) if isinstance(t, torch.Tensor) else float(t or 0.0)
    sigmas = (args.get("c") or {}).get("transformer_options", {}).get("sample_sigmas")
    if sigmas is None:
        values = None
    elif isinstance(sigmas, torch.Tensor) and sigmas.dim() == 1:
        values = [float(x) for x in host_list(sigmas)]
    else:
        values = [float(x) for x in sigmas]
    return clock_of(sigma, values)


def _time_scales(timeline: Any, args: dict, joined: frozenset = frozenset()):
    from ..call_values import expert_scales, token_depth_values
    from ..timeline import SCALARS, time_scales, window_value

    if not timeline:
        return contextlib.nullcontext()
    clock = _call_clock(args)
    stack = contextlib.ExitStack()
    if any(r.quantity in SCALARS and r.target not in joined for r in timeline):
        stack.enter_context(time_scales(timeline, clock))
    scales = {}
    for r in timeline:
        if r.quantity in SCALARS and r.target in joined:
            v = r.value(clock)
            v = (1.0 if v > 0.5 else 0.0) if r.quantity == "active" else v
            scales[r.target] = scales.get(r.target, 1.0) * v
    if scales:
        stack.enter_context(expert_scales(scales))
    on = window_value(timeline, "depth_map", clock)
    base = window_value(timeline, "depth_base", clock)
    if on is not None or base is not None:
        stack.enter_context(token_depth_values(None if on is None else on > 0, base))
    return stack


def _timeline_overrides(timeline: Any, chain: Sequence[ModEntry], joint: bool) -> dict:
    from ..timeline import POINTS, check_row, point_value

    if not timeline:
        return {}
    names = [entry_name(e) for e in chain if e.enabled]
    for row in timeline:
        check_row(row, names)
    return {q: v for q in POINTS if (v := point_value(timeline, q)) is not None}


def joined_names(chain: Sequence[ModEntry], joint: bool) -> frozenset:
    return frozenset(
        entry_name(e) for entries in _routed_groups(chain, joint).values() for e in entries
    )


def _timeline_report(timeline: Any) -> str:
    lines = ["Timeline:"]
    for r in timeline:
        lines.append(f"  {r.target} {r.quantity} = {r.curve} ({r.clock}, {r.interpolation})")
    return "\n".join(lines)


def _with_timeline(config: Any, patcher: Any, quantities: tuple) -> Any:
    info = applied(patcher) or {}
    rows = tuple(r for r in (info.get("timeline") or ()) if r.quantity in quantities)
    return dataclasses.replace(config, timeline=rows) if rows else config
