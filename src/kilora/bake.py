# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .adaptive_gate import DEFAULT_KAPPA
from .base_weights import read_base_weight
from .compute import (
    LayerSet,
    depth_split_on,
    depth_step_metric,
    expert_ids_of,
    layer_places,
    run_layer,
)
from .recording import active_perturbation as _active_perturbation
from .recording import active_recorder as _active_recorder
from .sets import AdapterSet, routes_alone


@dataclass
class TargetPlan:
    sets: list[LayerSet] = field(default_factory=list)
    depth: int = 1
    shared: bool = False
    name: str = ""
    regions: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    panels: list = field(default_factory=list)
    layout: Any = None
    expert_regions: list = field(default_factory=list)
    expert_roles: list = field(default_factory=list)
    schedule: Any = None
    place: tuple[int, int, str] = (0, 1, "other")
    tags: list = field(default_factory=list)
    token_depth: Any = None
    expert_names: list = field(default_factory=list)

    def _sets(self) -> list:
        from .timeline import current_scale

        out = None
        for i, s in enumerate(self.sets):
            tag = self.tags[i] if i < len(self.tags) else ""
            mult = current_scale(tag, s.axis)
            if mult is None:
                return self.sets
            if mult != 1.0:
                out = out if out is not None else list(self.sets)
                out[i] = dataclasses.replace(s, scale=s.scale * mult)
        return self.sets if out is None else out

    def run(
        self, x: torch.Tensor, h: torch.Tensor, conditionings=None, offsets=None, trial=None
    ) -> torch.Tensor:
        regions = self._token_regions(h) if any(r is not None for r in self.regions) else None
        regions = self._with_roles(h, regions)
        index, count, kind = self.place
        experts = None
        if any(self.expert_regions) or any(self.expert_roles):
            experts = self._expert_masks(h)
        sets = self._sets()
        from .axis_owner import record as record_owner

        record_owner(self.name, h, sets)
        if trial is not None:
            sets = trial(self.name, sets)
        if conditionings:
            updated = []
            for i, s in enumerate(sets):
                names = self.expert_names[i] if i < len(self.expert_names) else None
                if not names:
                    tag = self.tags[i] if i < len(self.tags) else ""
                    names = [tag] if tag else []
                if s.form == "input" and names:
                    inputs = tuple(conditionings.get("expert_" + name) for name in names)
                    if any(value is not None for value in inputs):
                        s = dataclasses.replace(
                            s,
                            expert_inputs=inputs,
                            expert_offsets=tuple(offsets.get("expert_" + name) for name in names),
                        )
                updated.append(s)
            sets = updated
        if any(s.ir_parts for s in sets):
            from .ir import override_scope

            scope = override_scope()
        else:
            scope = contextlib.nullcontext()
        with scope:
            y = run_layer(
                x,
                h,
                sets,
                self.depth,
                shared=self.shared,
                target=self.name,
                regions=regions,
                schedule=self.schedule,
                expert_regions=experts,
                index=index,
                count=count,
                kind=kind,
                token_depth=self._token_depth(h),
                expert_scales=self._expert_scales(),
            )
        return y

    def _expert_scales(self) -> list | None:
        if not any(self.expert_names):
            return None
        from .call_values import expert_scale_vector

        out = [expert_scale_vector(n) if n else None for n in self.expert_names]
        out = [None if v is not None and all(x == 1.0 for x in v) else v for v in out]
        return out if any(v is not None for v in out) else None

    def _token_depth(self, h: torch.Tensor):
        m = self.token_depth
        if m is None or h.dim() < 3:
            return None
        from .call_values import current_crop_window
        from .compute import current_progress

        def region(t):
            try:
                return self._token_region(t, h)
            except ValueError:
                return None

        on, base = m.applies(current_progress())
        from .region import current_latent_shape, current_layout, current_text_tokens

        try:
            version = m.values._version
        except RuntimeError:
            version = None
        key = (
            m.values,
            current_crop_window(),
            version,
            tuple(h.shape[1:-1]),
            h.device,
            current_latent_shape(),
            current_text_tokens(),
            current_layout(self.name),
            on,
            base,
            m.min_depth,
            m.max_depth,
        )
        cache = m.__dict__.setdefault("_token_depth_cache", {}) if version is not None else {}
        if key not in cache:
            if len(cache) >= 16:
                cache.clear()
            value = m.on_tokens(region, not on, base)
            if value is not None and value.is_inference():
                with torch.inference_mode(False):
                    value = value.clone()
            cache[key] = value
        return cache[key]

    def _token_regions(self, h: torch.Tensor) -> list:
        if h.dim() < 3:
            raise ValueError(f"{self.name}: a region needs token outputs, got {tuple(h.shape)}")
        return [self._token_region(r, h) for r in self.regions]

    def _token_region(self, r: Any, h: torch.Tensor):
        from .call_values import crop_map, current_crop_window
        from .region import current_latent_shape, current_layout, current_text_tokens, layer_region

        if r is None:
            return None
        dims = tuple(h.shape[1:-1])
        tokens, latent = math.prod(dims), current_latent_shape()
        try:
            version = r._version
        except RuntimeError:
            version = None
        key = (
            r,
            version,
            dims,
            latent,
            current_text_tokens(),
            current_layout(self.name),
            h.device,
            current_crop_window(),
        )
        cache = self.__dict__.setdefault("_region_cache", {}) if version is not None else {}
        if key not in cache:
            source = (
                crop_map(r if r.ndim == 3 else r[None], latent[-2:])
                if current_crop_window() and latent and r.ndim in (2, 3)
                else r
            )
            found = layer_region(source, tokens, latent, target=self.name)
            if found is not None and len(dims) > 1:
                found = found.reshape(found.shape[0], *dims)
            found = None if found is None else found.to(h.device)
            if found is not None and found.is_inference():
                with torch.inference_mode(False):
                    found = found.clone()
            cache[key] = found
        return cache[key]

    def _expert_masks(self, h: torch.Tensor) -> list:
        from .compute import _token_region as as_gate

        n = len(self.sets)
        regions = self.expert_regions or [None] * n
        roles = self.expert_roles or [None] * n
        panels = self.panels or [None] * n
        out = []
        for regs, pols, panel in zip(regions, roles, panels, strict=True):
            count = len(regs) if regs is not None else len(pols) if pols is not None else 0
            if not count:
                out.append(None)
                continue
            masks = []
            for j in range(count):
                r = as_gate(self._token_region(regs[j], h), h) if regs is not None else None
                m = self._role_mask(pols[j], panel, h) if pols is not None else None
                m = as_gate(m, h) if m is not None else None
                masks.append(r if m is None else m if r is None else r * m)
            out.append(masks if any(m is not None for m in masks) else None)
        return out

    def _role_mask(self, policy: str, panel: Any, h: torch.Tensor):
        from .call_values import current_crop_window
        from .region import current_latent_shape, current_layout, current_text_tokens
        from .roles import current_references, needs_mask, role_mask

        if policy == "all" or h.dim() < 3:
            return None
        refs = current_references()
        if (
            not refs
            and panel is None
            and getattr(self.layout, "role", "") != "reference"
            and not needs_mask(policy, False)
        ):
            return None
        layout = current_layout(self.name) or self.layout
        dims = tuple(h.shape[1:-1])
        tokens, latent = math.prod(dims), current_latent_shape() or ()
        text_tokens = current_text_tokens()
        cache = self.__dict__.setdefault("_role_cache", {})
        key = (policy, panel, dims, latent, refs, text_tokens, id(layout), current_crop_window())
        if key not in cache:
            found = role_mask(policy, layout, tokens, latent, refs, text_tokens, self.name, panel)
            if found is not None and len(dims) > 1:
                found = found.reshape(found.shape[0], *dims)
            found = None if found is None else found.to(h.device)
            if found is not None and found.is_inference():
                with torch.inference_mode(False):
                    found = found.clone()
            cache[key] = found
        return cache[key]

    def _with_roles(self, h: torch.Tensor, regions: list | None) -> list | None:
        n = len(self.sets)
        policies = self.roles or ["auto"] * n
        panels = self.panels or [None] * n
        if all(p == "all" for p in policies):
            return regions
        out = list(regions) if regions is not None else [None] * n
        for i, (policy, panel) in enumerate(zip(policies, panels, strict=True)):
            mask = self._role_mask(policy, panel, h)
            if mask is not None:
                out[i] = mask if out[i] is None else out[i] * mask
        return out if any(r is not None for r in out) else None

    def to(
        self, device: torch.device, dtype: torch.dtype, *, non_blocking: bool = False
    ) -> TargetPlan:
        def move(t, kind):
            if t is None:
                return None
            value = t.to(device, kind, non_blocking=non_blocking)
            return value.clone() if value.is_inference() else value

        moved = []
        from . import derived_cache

        for s in self.sets:
            moved.append(
                LayerSet(
                    router=s.router.to(device, torch.float32, non_blocking=non_blocking),
                    keys_in=s.keys_in.to(device, torch.float32, non_blocking=non_blocking),
                    keys_out=None
                    if s.keys_out is None
                    else s.keys_out.to(device, torch.float32, non_blocking=non_blocking),
                    rows_in=None
                    if s.rows_in is None
                    else s.rows_in.to(device, non_blocking=non_blocking),
                    rows_out=None
                    if s.rows_out is None
                    else s.rows_out.to(device, non_blocking=non_blocking),
                    per_expert=s.per_expert,
                    expert_ids=s.expert_ids,
                    entry_windows=s.entry_windows,
                    down=move(s.down, dtype),
                    up=move(s.up, dtype),
                    routing_down=move(s.routing_down, dtype),
                    routing_up=move(s.routing_up, dtype),
                    routing_source=s.routing_source,
                    exact={
                        i: tuple(move(part, dtype) for part in parts)
                        for i, parts in s.exact.items()
                    },
                    scale=s.scale,
                    log_prior=s.log_prior,
                    tau=s.tau,
                    aggregation=s.aggregation,
                    form=s.form,
                    apply_once=s.apply_once,
                    key=s.key,
                    routing_scope=s.routing_scope,
                    routing_select=s.routing_select,
                    candidates=s.candidates,
                    candidate_summary=s.candidate_summary,
                    cross_weights=s.cross_weights,
                    ir_parts=s.ir_parts,
                    learned_cross=None
                    if s.learned_cross is None
                    else {k: move(v, torch.float32) for k, v in s.learned_cross.items()},
                    learned_cross_zero=s.learned_cross_zero,
                    carry=s.carry,
                    iteration=s.iteration,
                    fixed_point_stop=s.fixed_point_stop,
                    path_width=s.path_width,
                    path_select=s.path_select,
                    route_freeze=s.route_freeze,
                    radial_step=s.radial_step,
                    orthogonal_step=s.orthogonal_step,
                    radial_budget=s.radial_budget,
                    step_metric=move(s.step_metric, torch.float32),
                    jump_fits=s.jump_fits,
                    jump_threshold=s.jump_threshold,
                    import_score=s.import_score,
                    derived_entry_temp=s.derived_entry_temp,
                    host_weight_norm=s.host_weight_norm,
                    share_transition_weighted=move(s.share_transition_weighted, torch.float32),
                    direction_core=move(s.direction_core, torch.float32),
                    direction_sources=s.direction_sources,
                    share_transition_base=move(s.share_transition_base, torch.float32),
                    fine=s.fine,
                    direction_candidates=s.direction_candidates,
                    axis=s.axis,
                    fixed_depth=s.fixed_depth,
                    path_starts=s.path_starts,
                    path_prior_min=s.path_prior_min,
                    path_merge=s.path_merge,
                    path_beam=s.path_beam,
                    output_gate=s.output_gate,
                    gate_mode=s.gate_mode,
                    gate_kappa=s.gate_kappa,
                    gate_head_limit=s.gate_head_limit,
                    gate_coherence=s.gate_coherence,
                    gate=None
                    if s.gate is None
                    else {
                        k: v.to(device, torch.float32, non_blocking=non_blocking)
                        for k, v in s.gate.items()
                    },
                )
            )
            derived_cache.bind(moved[-1], derived_cache.layer_source(s))
        return TargetPlan(
            sets=moved,
            depth=self.depth,
            shared=self.shared,
            name=self.name,
            regions=list(self.regions),
            roles=list(self.roles),
            panels=list(self.panels),
            layout=self.layout,
            expert_regions=list(self.expert_regions),
            expert_roles=list(self.expert_roles),
            schedule=self.schedule,
            place=self.place,
            tags=list(self.tags),
            token_depth=self.token_depth,
            expert_names=list(self.expert_names),
        )


def _observe_outputs(recorder, plan, x, h, out) -> None:
    wants = getattr(recorder, "wants_output", None)
    if wants is None:
        return
    for offset, rows, target in plan.parts:
        if wants(target.name):
            if rows == -1:
                recorder.observe_output(target.name, x, h, out)
            else:
                part = slice(offset, offset + rows)
                recorder.observe_output(target.name, x, h[..., part], out[..., part])


@dataclass
class ModulePlan:
    parts: list[tuple[int, int, TargetPlan]]
    _cache: dict[tuple[Any, Any], ModulePlan] = field(default_factory=dict, repr=False)
    _prefetch: Any = field(default=None, repr=False, compare=False)

    def run(
        self, x: torch.Tensor, h: torch.Tensor, conditionings=None, offsets=None
    ) -> torch.Tensor:
        if x.ndim == 2 and h.ndim == 2:
            return self.run(x.unsqueeze(0), h.unsqueeze(0), conditionings, offsets).squeeze(0)
        plan = self._on(h.device, h.dtype)
        try:
            return self._run_on(plan, x, h, conditionings, offsets)
        finally:
            if self._prefetch is not None and h.device.type == "cuda":
                self._prefetch.finished(self)

    def _run_on(self, plan, x, h, conditionings=None, offsets=None):
        recorder = _active_recorder()
        if recorder is not None:
            for offset, rows, target in plan.parts:
                if recorder.wants(target.name):
                    recorder.observe(
                        target.name, h if rows == -1 else h[..., offset : offset + rows]
                    )
        perturb = _active_perturbation()

        def apply(target, part, part_offsets):
            trial = getattr(perturb, "attention_sets", None)
            can_replay = (
                trial is not None
                and getattr(perturb, "set_key", None) is not None
                and any(
                    s.key == perturb.set_key
                    and (s.form == "state" or getattr(perturb, "replay_all", False))
                    for s in target.sets
                )
                and target.name in perturb.du
            )
            d = perturb.delta(target.name, part) if perturb is not None else None
            read = getattr(perturb, "input_delta", None)
            e = read(target.name, x) if read is not None else None
            if can_replay:
                out = target.run(x, part, conditionings, part_offsets, trial=trial)
            else:
                out = target.run(x, part, conditionings, part_offsets)
                out = out if d is None else out + d
            return out if e is None else out + e.to(out)

        if len(plan.parts) == 1 and plan.parts[0][1] == -1:
            out = apply(plan.parts[0][2], h, offsets)
            _observe_outputs(recorder, plan, x, h, out)
            return out
        out = h.clone()
        for offset, rows, target in plan.parts:
            part = h[..., offset : offset + rows]
            part_offsets = (
                None
                if offsets is None
                else {k: v[..., offset : offset + rows] for k, v in offsets.items()}
            )
            out[..., offset : offset + rows] = apply(target, part, part_offsets)
        _observe_outputs(recorder, plan, x, h, out)
        return out

    def _on(self, device: torch.device, dtype: torch.dtype) -> ModulePlan:
        if self._prefetch is not None and device.type == "cuda":
            return self._prefetch.get(self, device, dtype)
        key = (device, dtype)
        if key not in self._cache:
            self._cache.clear()
            with torch.inference_mode(False):
                parts = [(o, r, t.to(device, dtype)) for o, r, t in self.parts]
            self._cache[key] = ModulePlan(parts=parts)
        return self._cache[key]


@dataclass
class ScopedSet:
    adapter_set: AdapterSet
    scope: str = "shared"
    routing_scope: str = "token"
    routing_select: str = "sparsemax"
    carry: str = "dropped"
    iteration: str = "accumulate"
    fixed_point_stop: bool = False
    output_gate: str = "separable"
    gate_mode: str = "learned"
    gate_kappa: float = DEFAULT_KAPPA
    gate_head_limit: int | None = None
    path_width: int = 1
    path_select: str = "converged"
    route_freeze: float | None = None
    radial_step: float = 1.0
    orthogonal_step: float = 1.0
    radial_budget: float = 0.0
    axis: str = "creative"
    fixed_depth: float | None = None
    path_starts: str = "top"
    path_prior_min: float = 0.0
    path_merge: bool = False
    path_beam: bool = False
    region: Any = None
    roles: str = "auto"
    import_score: str = "derived"
    derived_entry_temp: float | None = None
    panels: Any = None
    expert_regions: Any = None
    expert_roles: Any = None
    expert_names: Any = None
    tag: str = ""
    candidates: int = 0
    candidate_summary: str = "keys"
    cross_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)


def plan_targets(
    sets: Sequence[ScopedSet], depth: int, schedule: Any = None, merge_plain: bool = False
) -> dict[str, TargetPlan]:
    targets: list[str] = []
    for item in sets:
        targets.extend(t for t in item.adapter_set.target_modules if t not in targets)
    places = layer_places(targets)
    plans: dict[str, TargetPlan] = {}
    for target in targets:
        here = [item for item in sets if target in item.adapter_set.experts]
        dims = {item.adapter_set.router_dim for item in here}
        lora_sets = [item for item in here if item.adapter_set.has_lora]
        shared = all(item.scope == "shared" for item in lora_sets) and (
            len(dims) == 1 or any(item.adapter_set.form == "state" for item in lora_sets)
        )
        if shared and sum(i.adapter_set.prior for i in here) <= 0:
            raise ValueError(f"{target}: at least one adapter set needs a prior above 0")
        plan = TargetPlan(
            depth=depth,
            shared=shared and len(lora_sets) > 1,
            name=target,
            schedule=schedule,
            place=places[target],
        )
        from . import derived_cache

        for item in here:
            s = item.adapter_set
            ex = s.experts[target]
            once = bool(s.extra.get("apply_once", False))
            down = up = None
            per = 1
            if ex.has_lora:
                experts, routes, rank, n_in = ex.down.shape
                per = routes * rank
                down = ex.down.float().reshape(experts * per, n_in)
                up = ex.up.float().reshape(experts * per, ex.out_features)
            scale = s.strength
            plan.sets.append(
                LayerSet(
                    expert_ids=expert_ids_of(s),
                    entry_windows=s.extra.get("entry_windows", ()),
                    down=down,
                    up=up,
                    routing_down=(
                        None
                        if ex.routing_down is None
                        else ex.routing_down.float().reshape_as(down)
                    ),
                    routing_up=(
                        None if ex.routing_up is None else ex.routing_up.float().reshape_as(up)
                    ),
                    routing_source=derived_cache.routing_source(ex),
                    exact={
                        i: tuple(part.float() for part in parts) for i, parts in ex.exact.items()
                    },
                    router=s.router_for(target).float(),
                    keys_in=ex.keys_in.float(),
                    keys_out=ex.keys_out.float(),
                    rows_in=ex.rows_in,
                    rows_out=ex.rows_out,
                    per_expert=per,
                    scale=scale,
                    log_prior=math.log(s.prior) if s.prior > 0 else -math.inf,
                    tau=math.exp(s.log_tau),
                    aggregation=s.aggregation,
                    form=s.form,
                    apply_once=once,
                    key=s.extra.get("live_id", s.name),
                    routing_scope=item.routing_scope,
                    routing_select=item.routing_select,
                    candidates=item.candidates,
                    candidate_summary=item.candidate_summary,
                    cross_weights=item.cross_weights,
                    ir_parts=s.ir_parts,
                    learned_cross=s.learned_cross,
                    learned_cross_zero=s.learned_cross_zero,
                    carry=item.carry,
                    iteration=item.iteration,
                    fixed_point_stop=item.fixed_point_stop,
                    path_width=item.path_width,
                    path_select=item.path_select,
                    route_freeze=item.route_freeze,
                    radial_step=item.radial_step,
                    orthogonal_step=item.orthogonal_step,
                    radial_budget=item.radial_budget,
                    axis=item.axis,
                    fixed_depth=item.fixed_depth,
                    path_starts=item.path_starts,
                    path_prior_min=item.path_prior_min,
                    path_merge=item.path_merge,
                    path_beam=item.path_beam,
                    output_gate=item.output_gate,
                    gate_mode="none" if once else item.gate_mode,
                    gate_kappa=item.gate_kappa,
                    gate_head_limit=item.gate_head_limit,
                    import_score=item.import_score,
                    direction_core=s.extra.get("direction_core"),
                    direction_sources=s.extra.get("direction_sources"),
                    derived_entry_temp=item.derived_entry_temp,
                    fine=s.extra.get("fine", False),
                    direction_candidates=s.extra.get("candidates", 16),
                    gate=None if ex.gate is None else {k: v.float() for k, v in ex.gate.items()},
                )
            )
            if depth_split_on(plan.sets[-1]):
                plan.sets[-1].step_metric = depth_step_metric(up)
            derived_cache.bind(plan.sets[-1], derived_cache.source(ex))
            plan.tags.append(item.tag)
            plan.regions.append(item.region)
            plan.roles.append(item.roles)
            plan.panels.append(item.panels)
            plan.expert_regions.append(item.expert_regions)
            plan.expert_roles.append(item.expert_roles)
            plan.expert_names.append(item.expert_names)
        if merge_plain:
            _merge_plain_sets(plan)
        plans[target] = plan
    return plans


def _plain(plan: TargetPlan, i: int) -> bool:
    s = plan.sets[i]
    once = s.fixed_depth == 1 or (s.fixed_depth is None and plan.depth == 1)
    return (
        not s.apply_once
        and not s.entry_windows
        and not s.exact
        and s.direction_core is None
        and not s.ir_parts
        and s.down is not None
        and s.form == "input"
        and s.gate_mode == "none"
        and s.aggregation == "accumulate"
        and s.iteration == "accumulate"
        and int(s.keys_in.shape[0]) == 1
        and isinstance(s.scale, (int, float))
        and once
        and plan.regions[i] is None
        and plan.roles[i] == "auto"
        and plan.panels[i] is None
        and plan.expert_regions[i] is None
        and plan.expert_roles[i] is None
    )


def _merge_plain_sets(plan: TargetPlan) -> None:
    groups: dict[str, list[int]] = {}
    for i in range(len(plan.sets)):
        if _plain(plan, i):
            groups.setdefault(plan.sets[i].axis, []).append(i)
    merged: dict[int, LayerSet] = {}
    drop: set[int] = set()
    for idx in groups.values():
        if len(idx) < 2:
            continue
        first = plan.sets[idx[0]]
        down = torch.cat([plan.sets[i].down for i in idx])
        up = torch.cat([plan.sets[i].up * float(plan.sets[i].scale) for i in idx])
        merged[idx[0]] = dataclasses.replace(
            first, down=down, up=up, per_expert=int(down.shape[0]), scale=1.0
        )
        drop.update(idx[1:])
    if not merged:
        return
    keep = [i for i in range(len(plan.sets)) if i not in drop]
    plan.sets = [merged.get(i, plan.sets[i]) for i in keep]
    for name in (
        "tags",
        "regions",
        "roles",
        "panels",
        "expert_regions",
        "expert_roles",
        "expert_names",
    ):
        values = getattr(plan, name)
        setattr(plan, name, [values[i] for i in keep])
    plan.shared = plan.shared and sum(s.down is not None for s in plan.sets) > 1


_BAKED_CLASSES: dict[type, type] = {}


_BASE_ONLY: list[bool] = []


@contextlib.contextmanager
def base_only() -> Iterator[None]:
    _BASE_ONLY.append(True)
    try:
        yield
    finally:
        _BASE_ONLY.pop()


def base_only_active() -> bool:
    return bool(_BASE_ONLY)


def run_plan_forward(plan, original, x, *args, **kwargs):
    if base_only_active():
        return original(x, *args, **kwargs)
    from .comfyui.expert_text import current_conditionings
    from .region import current_layout

    parts = getattr(plan, "parts", None)
    is_text = bool(parts) and all(
        getattr(current_layout(target.name) or target.layout, "kind", None) == "text1d"
        for _, _, target in parts
    )
    values = current_conditionings() if is_text else {}
    if values and values.get("base") is not None:
        x = values["base"]
        h = original(x, *args, **kwargs)
        offsets = {
            key: original(value, *args, **kwargs) - h
            for key, value in values.items()
            if key.startswith("expert_")
        }
        return plan.run(x, h, values, offsets)
    return plan.run(x, original(x, *args, **kwargs))


def _separate_shared_parameters(module: nn.Module) -> None:
    base = module.__dict__["_kilora_base"]
    for name, parameter in module._parameters.items():
        if parameter is not None and base._parameters.get(name) is parameter:
            module._parameters[name] = copy.copy(parameter)


def _ensure_input_device(module: nn.Module, x: torch.Tensor) -> None:
    if module.weight.device != x.device and not getattr(module, "comfy_cast_weights", False):
        module.to(x.device)


def _copy_layer_state(module: nn.Module, source: nn.Module | None = None) -> dict:
    state = {}
    for key, value in module.__dict__.items():
        if key in {"comfy_patched_weights", "prev_comfy_cast_weights", "comfy_force_cast_weights"}:
            continue
        if key in {"weight_function", "bias_function"}:
            state[key] = []
        elif isinstance(value, (dict, set, list)):
            state[key] = type(value)(value)
        else:
            state[key] = value
    if hasattr(module, "prev_comfy_cast_weights"):
        state["comfy_cast_weights"] = module.prev_comfy_cast_weights
    for key in ("weight_function", "bias_function"):
        if hasattr(module, key):
            state[key] = []
    if source is not None and source is not module:
        state["_parameters"] = {
            name: copy.copy(value) if value is not None else None
            for name, value in source._parameters.items()
        }
        state["_buffers"] = source._buffers.copy()
    return state


def _baked_class(base: type) -> type:
    if base not in _BAKED_CLASSES:

        def forward(self, x, *args, **kwargs):
            _ensure_input_device(self, x)
            return run_plan_forward(
                self._kilora_plan,
                lambda value, *a, **kw: base.forward(self, value, *a, **kw),
                x,
                *args,
                **kwargs,
            )

        def _apply(self, fn, recurse=True):
            _separate_shared_parameters(self)
            result = base._apply(self, fn, recurse=recurse)
            if self.weight.device.type != "cuda":
                self._kilora_plan._cache.clear()
                if self._kilora_plan._prefetch is not None:
                    self._kilora_plan._prefetch.reset()
            return result

        _BAKED_CLASSES[base] = type(
            f"Baked{base.__name__}", (base,), {"forward": forward, "_apply": _apply}
        )
    return _BAKED_CLASSES[base]


def baked_plans(root: nn.Module) -> list[ModulePlan]:
    return [m.__dict__["_kilora_plan"] for m in root.modules() if is_baked(m)]


def is_baked(module: Any) -> bool:
    return isinstance(module, nn.Module) and "_kilora_plan" in module.__dict__


def baked_copy(
    module: nn.Module, plan: ModulePlan, *, source: nn.Module | None = None
) -> nn.Module:

    if is_baked(module):
        if source is None:
            source = module
        module = module.__dict__["_kilora_base"]
    cls = _baked_class(type(module))
    new = cls.__new__(cls)
    new.__dict__.update(_copy_layer_state(module, source))
    new.__dict__["_kilora_plan"] = plan
    new.__dict__["_kilora_base"] = module
    new.__dict__["comfy_force_forward"] = True
    return new


def module_plans(
    plans: dict[str, TargetPlan],
    place: Callable[[str], tuple[str, tuple[int, int] | None]],
    out_features: Callable[[str], int],
) -> dict[str, ModulePlan]:

    grouped: dict[str, list[tuple[int, int, TargetPlan]]] = {}
    for target, plan in plans.items():
        path, part = place(target)
        if part is None:
            grouped.setdefault(path, []).append((0, -1, plan))
        else:
            index, count = part
            rows = out_features(path) // count
            grouped.setdefault(path, []).append((index * rows, rows, plan))
    out = {}
    for path, parts in grouped.items():
        if any(r == -1 for _, r, _ in parts) and len(parts) > 1:
            raise ValueError(f"{path}: a whole-module plan cannot share the module")
        out[path] = ModulePlan(parts=sorted(parts, key=lambda p: p[0]))
    return out


class BakedModel:
    def __init__(self, model: nn.Module, plans: dict[str, ModulePlan]) -> None:
        self.model = model
        self._originals: dict[str, nn.Module] = {}
        for path, plan in plans.items():
            parent_path, _, name = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            original = getattr(parent, name)
            self._originals[path] = original
            setattr(parent, name, baked_copy(original, plan))

    @property
    def paths(self) -> list[str]:
        return list(self._originals)

    def remove(self) -> None:
        for path, original in self._originals.items():
            parent_path, _, name = path.rpartition(".")
            parent = self.model.get_submodule(parent_path) if parent_path else self.model
            setattr(parent, name, original)
        self._originals = {}


def bake(model: nn.Module, sets: Sequence[ScopedSet | AdapterSet], depth: int) -> BakedModel:

    plain = [s for s in sets if not isinstance(s, ScopedSet)]
    independent = any(routes_alone(s.extra) or s.imported_from is not None for s in plain)
    scope = "per_set" if independent and len(plain) > 1 else "shared"
    scoped = [s if isinstance(s, ScopedSet) else ScopedSet(s, scope) for s in sets]
    modules = dict(model.named_modules())
    for item in scoped:
        for target in item.adapter_set.target_modules:
            if target not in modules:
                raise KeyError(f"target module {target!r} does not exist in this model")
    plans = plan_targets(scoped, depth)
    from .compute import derived_mode, import_score

    for target, plan in plans.items():
        if not any(s.form == "input" and import_score(s) == "derived" for s in plan.sets):
            continue
        if derived_mode() != "full":
            continue
        weight = read_base_weight(modules[target], target)
        if (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and weight.dtype.is_floating_point
            and weight.device.type != "meta"
        ):
            norm = float(weight.detach().float().norm())
            for layer_set in plan.sets:
                layer_set.host_weight_norm = norm
    import os

    if os.environ.get("KILORA_SHARE_TRANSITION_KIND") == "base":
        from .compute import base_weight_transition

        for target, plan in plans.items():
            weight = read_base_weight(modules[target], target)
            if isinstance(weight, torch.Tensor):
                for layer_set in plan.sets:
                    layer_set.share_transition_base = base_weight_transition(layer_set, weight)
    grouped = module_plans(
        plans,
        place=lambda t: (t, None),
        out_features=lambda p: modules[p].out_features,
    )
    return BakedModel(model, grouped)
