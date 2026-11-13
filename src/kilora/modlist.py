# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import heapq
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .adapter import Settings, imported_schedule_fields, imported_settings
from .bake import ScopedSet
from .bindings import BINDINGS
from .compute import derived_default_depth, derived_mode
from .errors import KiLoRAError, ModFormatError
from .modfile import load_mod
from .roles import parse_policy
from .sets import AdapterSet, check_prior, check_strength

HOOK_KINDS = ("observe", "transform", "wrap", "inject", "route")


class LoadOrderError(KiLoRAError):
    pass


def parse_version(text: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+].*)?\s*", str(text))
    if m is None:
        raise ValueError(f"not a version: {text!r}")
    return tuple(int(g) if g else 0 for g in m.groups())


_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def satisfies(version: str, spec: str) -> bool:
    have = parse_version(version)
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        m = re.fullmatch(r"(>=|<=|==|!=|>|<)?\s*(\S+)", clause)
        if m is None:
            raise ValueError(f"not a version range: {spec!r}")
        op = m.group(1) or "=="
        if not _OPS[op](have, parse_version(m.group(2))):
            return False
    return True


def parse_requirement(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        raise ModFormatError(f"a mod reference must be a string, got {text!r}")
    m = re.fullmatch(r"\s*([a-z0-9][a-z0-9._-]*)\s*(.*?)\s*", text)
    if m is None:
        raise ModFormatError(f"not a mod reference: {text!r}")
    spec = m.group(2)
    if spec:
        satisfies("0.0.0", spec)
    return m.group(1), spec


@dataclass
class ModChoice:
    path: str | os.PathLike[str] | None = None
    adapter_sets: list[AdapterSet] | None = None
    strength: float = 1.0
    prior: float = 1.0
    enabled: bool = True
    label: str | None = None
    mod_id: str | None = None
    session: Any = None
    sections: list | None = None
    notes: list[str] | None = None

    entry_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        check_strength(self.strength)
        check_prior(self.prior)
        if (self.path is None) == (self.adapter_sets is None):
            raise ValueError("a mod choice needs either a path or adapter sets")


@dataclass
class LoadedMod:
    mod_id: str
    version: str
    name: str
    manifest: dict[str, Any]
    sets: list[AdapterSet]
    settings: Settings
    strength: float
    position: int
    label: str
    session: Any = None
    path: str | None = None
    sections: list = field(default_factory=list)
    references: list = field(default_factory=list)
    logic: dict | None = None

    @property
    def digest(self) -> str | None:
        return self.manifest.get("digest")

    @property
    def family(self) -> str:
        base = self.manifest.get("base_model") or {}
        return base.get("family", "unknown")

    @property
    def trained_depth(self) -> tuple[int, int] | None:
        trained = (self.manifest.get("depth") or {}).get("trained")
        return (int(trained[0]), int(trained[1])) if trained else None

    @property
    def recommended_depth(self) -> int:
        return int((self.manifest.get("depth") or {}).get("recommended", self.settings.depth))

    def refs(self, key: str) -> list[tuple[str, str]]:
        values = self.manifest.get(key, [])
        if not isinstance(values, list):
            raise ModFormatError(f"{self.mod_id}: manifest {key!r} must be a list")
        return [parse_requirement(v) for v in values]

    def ids(self, key: str) -> list[str]:
        values = self.manifest.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ModFormatError(f"{self.mod_id}: manifest {key!r} must be a list of mod IDs")
        return values

    @property
    def priority(self) -> int:
        value = self.manifest.get("priority", 0)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ModFormatError(f"{self.mod_id}: manifest 'priority' must be an integer")
        return value


_CACHE: dict[tuple[str, int, int, int, str], Any] = {}


def _cache_key(path: str) -> tuple[str, int, int, int, str] | None:

    import hashlib

    try:
        with open(path, "rb") as f:
            stat = os.fstat(f.fileno())
            prefix = f.read(8)
            if len(prefix) < 8:
                return None
            length = int.from_bytes(prefix, "little")
            if length > stat.st_size - 8:
                return None
            header = f.read(length)
    except OSError:
        return None
    if len(header) != length:
        return None
    digest = hashlib.sha256(prefix + header).hexdigest()
    return (os.path.abspath(path), stat.st_size, stat.st_mtime_ns, stat.st_ino, digest)


def _cached_mod(path: str | os.PathLike[str]):
    path = os.fspath(path)
    key = _cache_key(path)
    if key is None:
        return load_mod(path)
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = load_mod(path)
    return _CACHE[key]


def _reference_paths(path, search_dirs, chain=()):
    from pathlib import Path

    from .references import resolve as resolve_reference

    current = Path(path).resolve()
    names = (*chain, current)
    if current in chain:
        raise ModFormatError("Reference cycle: " + " -> ".join(map(str, names)))
    if len(names) > 5:
        raise ModFormatError("Reference depth exceeds 4: " + " -> ".join(map(str, names)))
    for section in _cached_mod(current).sections:
        if section.kind != "weight-ref" or not section.meta["transforms"]["enabled"]:
            continue
        target = resolve_reference(section.meta, search_dirs)
        if section.meta["form"] == "mod":
            yield from _reference_paths(target, search_dirs, names)
        else:
            yield target


@dataclass(frozen=True)
class ReferenceEntry:
    kind: str
    path: str
    transforms: dict


def load_choice(choice: ModChoice, position: int, search_dirs=()) -> LoadedMod:
    references = []
    logic = None
    if choice.path is not None:
        from .references import resolve as resolve_reference

        mod = _cached_mod(choice.path)
        manifest = mod.manifest
        sets = choice.session.sets() if choice.session is not None else mod.sets
        settings = replace(mod.settings)
        if (
            manifest.get("schedule_yields_to_depth")
            and (os.environ.get("KILORA_IMPORT_SCORE") or settings.import_score) == "derived"
            and derived_mode() == "full"
        ):
            manifest = {**manifest, "depth": {"recommended": derived_default_depth()}}
            manifest.pop("depth_schedule", None)
            settings.depth = derived_default_depth()
            if settings.path_width == 1:
                settings.path_width = 3
        label = choice.label or os.path.basename(os.fspath(choice.path))
        sections = list(mod.sections)
        if any(
            section.kind == "logic" and section.meta["settings"].get("library")
            for section in sections
        ):
            from .library import LibraryIndex

            records = LibraryIndex.open().records.values()
            search_dirs = (*search_dirs, *(r["folder"] for r in records))
        if any(section.kind == "weight-ref" for section in sections):
            list(_reference_paths(choice.path, search_dirs))
            for section in sections:
                if section.kind == "weight-ref" and section.meta["transforms"]["enabled"]:
                    transforms = dict(section.meta["transforms"])
                    transforms["strength"] *= choice.strength
                    references.append(
                        ReferenceEntry(
                            section.meta["form"],
                            str(resolve_reference(section.meta, search_dirs)),
                            transforms,
                        )
                    )
        from .sections import resolve_program_bindings

        sections = resolve_program_bindings(sections, search_dirs)
        for section in sections:
            if section.kind == "logic":
                logic = dict(section.meta["settings"])
                updates = {
                    k: logic[k]
                    for k in ("iteration", "routing_scope", "routing_select")
                    if logic[k] is not None
                }
                if logic["depth"] >= 0:
                    updates["depth"] = logic["depth"]
                settings = replace(settings, **updates)
    else:
        sections = (
            choice.session.sections()
            if choice.session is not None and hasattr(choice.session, "sections")
            else list(getattr(choice, "sections", None) or [])
        )
        sets = choice.session.sets() if choice.session is not None else choice.adapter_sets or []
        name = choice.label or sets[0].name
        mod_id = choice.mod_id or re.sub(r"[^a-z0-9._-]", "-", name.lower()).strip("-") or "mod"
        manifest = {
            "id": mod_id,
            "name": name,
            "version": "0.0.0",
            "base_model": {"family": "unknown"},
            "depth": {"recommended": imported_settings().depth},
            **imported_schedule_fields(),
        }
        if choice.notes:
            manifest["notes"] = list(choice.notes)
        settings = imported_settings()
        label = name
    if choice.entry_windows:
        from .entry_window import windowed_sections, windowed_sets

        sets = windowed_sets(sets, choice.entry_windows)
        sections = windowed_sections(sections, choice.entry_windows)
    scaled = [
        replace(s, strength=s.strength * choice.strength, prior=s.prior * choice.prior)
        for s in sets
    ]
    return LoadedMod(
        mod_id=choice.mod_id or manifest["id"],
        version=str(manifest.get("version", "0.0.0")),
        name=str(manifest.get("name", manifest["id"])),
        manifest=manifest,
        sets=scaled,
        settings=settings,
        strength=choice.strength,
        position=position,
        label=label,
        session=choice.session,
        path=os.fspath(choice.path) if choice.path is not None else None,
        sections=sections,
        references=references,
        logic=logic,
    )


@dataclass
class Problem:
    level: str
    message: str


@dataclass
class HookUse:
    mod_id: str
    kind: str
    exclusive: bool
    declared: bool


@dataclass
class LoadPlan:
    mods: list[LoadedMod]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    hooks: dict[str, list[HookUse]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(p.level == "error" for p in self.problems)

    def scoped_sets(self) -> list[ScopedSet]:
        return [
            ScopedSet(
                s,
                m.settings.softmax_scope,
                m.settings.routing_scope,
                m.settings.routing_select,
                m.settings.carry,
                m.settings.iteration,
                m.settings.fixed_point_stop,
                m.settings.output_gate,
                m.settings.gate_mode,
                m.settings.gate_kappa,
                path_width=m.settings.path_width,
                path_select=m.settings.path_select,
                route_freeze=m.settings.route_freeze,
                path_starts=m.settings.path_starts,
                path_prior_min=m.settings.path_prior_min,
                path_merge=m.settings.path_merge,
                path_beam=m.settings.path_beam,
                roles=parse_policy(m.settings.token_roles),
                import_score=m.settings.import_score,
                derived_entry_temp=m.settings.derived_entry_temp,
                candidates=m.settings.candidates,
                candidate_summary=m.settings.candidate_summary,
            )
            for m in self.mods
            for s in m.sets
        ]

    def depth(self, requested: int | None) -> int:
        if requested is not None and requested >= 0:
            return requested
        if not self.mods:
            return 0
        return max(m.recommended_depth for m in self.mods)

    def report(self, depth: int | None = None) -> str:
        lines = ["KiLoRA load order:"]
        if not self.mods:
            lines.append("  (no mods)")
        for i, m in enumerate(self.mods, 1):
            sets = ", ".join(_set_strength(s) for s in m.sets)
            experts = sum(s.num_experts for s in m.sets)
            lines.append(f"  {i}. {m.name} ({m.mod_id} {m.version}): strength {m.strength:+g}")
            lines.append(f"     sets: {sets}; {experts} experts per target")
            for reason in self.reasons.get(m.mod_id, []):
                lines.append(f"     {reason}")
            words = m.manifest.get("trigger_words") or []
            if words:
                lines.append(f"     trigger words: {', '.join(words)}")
        for label in self.disabled:
            lines.append(f"  off: {label}")
        for m in self.mods:
            if m.session is None:
                continue
            s = m.session
            state = "on" if s.learning else "off"
            note = f" ({s.pairing.message})" if s.pairing.message else ""
            pending = ", pending update" if s.pending is not None else ""
            lines.append(
                f"  live {m.mod_id}: learning {state}{note}, speed {s.speed:g}, "
                f"{len(s.commits)} commits{pending}"
            )
        if depth is not None:
            lines.append(f"Depth: {depth}")
            for m in self.mods:
                trained = m.trained_depth
                if trained and depth > trained[1]:
                    lines.append(
                        f"  {m.mod_id}: depth above trained range ({trained[0]} to {trained[1]})"
                    )
        if self.problems:
            lines.append("Problems:")
            for p in self.problems:
                lines.append(f"  {p.level}: {p.message}")
        return "\n".join(lines)


def resolve(
    choices: Sequence[ModChoice],
    *,
    family: str | None = None,
    search_dirs=(),
) -> LoadPlan:
    loaded: list[LoadedMod] = []
    disabled: list[str] = []
    for position, choice in enumerate(choices):
        if not choice.enabled:
            disabled.append(choice.label or os.fspath(choice.path or "mod"))
            continue
        loaded.append(load_choice(choice, position, search_dirs))
    plan = LoadPlan(mods=[], disabled=disabled)
    err = plan.problems.append
    for m in loaded:
        if m.references and not m.sets:
            err(f"{m.label}: a recipe; load it with KiLoRA Apply or KiLoRA Load Mod")

    by_id: dict[str, LoadedMod] = {}
    for m in loaded:
        if m.mod_id in by_id:
            err(Problem("error", f"the mod {m.mod_id} is in the list twice"))
            continue
        by_id[m.mod_id] = m
    mods = list(by_id.values())

    if family is not None:
        for m in mods:
            if m.family not in ("unknown", family):
                err(Problem("error", f"{m.mod_id} is made for {m.family}, the model is {family}"))

    edges: dict[str, set[str]] = {m.mod_id: set() for m in mods}
    reasons: dict[str, list[str]] = {m.mod_id: [] for m in mods}

    def edge(before: str, after: str, why: str) -> None:
        if after not in edges[before]:
            edges[before].add(after)
            reasons[after].append(why)

    try:
        for m in mods:
            for other, spec in m.refs("requires"):
                dep = by_id.get(other)
                if dep is None:
                    err(Problem("error", f"{m.mod_id} requires {other} {spec}".rstrip()))
                elif spec and not satisfies(dep.version, spec):
                    err(
                        Problem("error", f"{m.mod_id} requires {other} {spec}, found {dep.version}")
                    )
                else:
                    edge(other, m.mod_id, f"after {other}: requires")
            for other, spec in m.refs("optional"):
                dep = by_id.get(other)
                if dep is None:
                    continue
                if spec and not satisfies(dep.version, spec):
                    err(
                        Problem(
                            "warning", f"{m.mod_id} works with {other} {spec}, found {dep.version}"
                        )
                    )
                edge(other, m.mod_id, f"after {other}: optional")
            for other, spec in m.refs("conflicts"):
                dep = by_id.get(other)
                if dep is not None and (not spec or satisfies(dep.version, spec)):
                    err(Problem("error", f"{m.mod_id} conflicts with {other} {dep.version}"))
            for other in m.ids("load_after"):
                if other in by_id:
                    edge(other, m.mod_id, f"after {other}: load_after")
            for other in m.ids("load_before"):
                if other in by_id:
                    edge(m.mod_id, other, f"after {m.mod_id}: its load_before")
                    reasons[m.mod_id].append(f"before {other}: load_before")
            for pin in m.manifest.get("pins", []):
                if not isinstance(pin, dict) or "id" not in pin or "digest" not in pin:
                    raise ModFormatError(f"{m.mod_id}: each pin needs 'id' and 'digest'")
                dep = by_id.get(pin["id"])
                if dep is None:
                    err(Problem("warning", f"{m.mod_id} pins {pin['id']}, which is not loaded"))
                elif dep.digest != pin["digest"]:
                    err(
                        Problem(
                            "error",
                            f"{m.mod_id} pins an exact file of {pin['id']}; "
                            f"the loaded {pin['id']} {dep.version} is a different file",
                        )
                    )
    except (ModFormatError, ValueError) as exc:
        err(Problem("error", str(exc)))
        plan.mods = mods
        return plan

    plan.mods = _order(mods, edges, plan)
    plan.reasons = {k: v for k, v in reasons.items() if v}
    plan.hooks = _hooks(plan.mods, family)
    _check_live(plan, family)
    for point, uses in plan.hooks.items():
        transforms = [u for u in uses if u.kind == "transform"]
        exclusive = [u for u in transforms if u.exclusive]
        if exclusive and len(transforms) > 1:
            names = ", ".join(sorted({u.mod_id for u in transforms}))
            err(Problem("error", f"exclusive transform conflict at {point}: {names}"))
        wraps = [u for u in uses if u.kind == "wrap" and u.exclusive]
        if len(wraps) > 1:
            names = ", ".join(sorted({u.mod_id for u in wraps}))
            err(Problem("error", f"exclusive wrap conflict at {point}: {names}"))
    return plan


def _check_live(plan: LoadPlan, family: str | None) -> None:
    binding = BINDINGS.get(family or "")
    learners: dict[str, list[tuple[str, bool]]] = {}
    for m in plan.mods:
        live = m.manifest.get("live")
        if not live:
            continue
        exclusive = bool(live.get("exclusive", False))
        for target in live.get("layout_spec", {}).get("targets", []):
            point = binding.hook_point(target) if binding else f"base::{target}"
            learners.setdefault(point, []).append((m.mod_id, exclusive))
    shared_points: dict[tuple[str, ...], list[str]] = {}
    for point, mods in learners.items():
        if len(mods) < 2:
            continue
        names = tuple(sorted(n for n, _ in mods))
        if any(ex for _, ex in mods):
            plan.problems.append(
                Problem("error", f"live learning conflict at {point}: {', '.join(names)}")
            )
        else:
            shared_points.setdefault(names, []).append(point)
    for names, points in shared_points.items():
        plan.problems.append(
            Problem(
                "warning",
                f"{' and '.join(names)} both learn at {len(points)} hook points "
                f"(for example {points[0]}); each keeps its own live state",
            )
        )


def _order(mods: list[LoadedMod], edges: dict[str, set[str]], plan: LoadPlan) -> list[LoadedMod]:
    by_id = {m.mod_id: m for m in mods}
    indegree = {m.mod_id: 0 for m in mods}
    for afters in edges.values():
        for a in afters:
            indegree[a] += 1
    ready = [(by_id[i].priority, by_id[i].position, i) for i, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, _, current = heapq.heappop(ready)
        order.append(by_id[current])
        for a in sorted(edges[current]):
            indegree[a] -= 1
            if indegree[a] == 0:
                heapq.heappush(ready, (by_id[a].priority, by_id[a].position, a))
    if len(order) != len(mods):
        stuck = sorted(i for i, d in indegree.items() if d > 0)
        plan.problems.append(
            Problem("error", f"load order has a cycle between: {', '.join(stuck)}")
        )
        order += [by_id[i] for i in stuck]
    return order


def _hooks(mods: list[LoadedMod], family: str | None) -> dict[str, list[HookUse]]:
    binding = BINDINGS.get(family or "")
    table: dict[str, list[HookUse]] = {}
    for m in mods:
        declared = m.manifest.get("hooks", [])
        if not isinstance(declared, list):
            raise ModFormatError(f"{m.mod_id}: manifest 'hooks' must be a list")
        points = set()
        for hook in declared:
            if not isinstance(hook, dict) or hook.get("kind") not in HOOK_KINDS:
                raise ModFormatError(f"{m.mod_id}: each hook needs a kind from {HOOK_KINDS}")
            point = str(hook.get("point", ""))
            points.add(point)
            table.setdefault(point, []).append(
                HookUse(m.mod_id, hook["kind"], bool(hook.get("exclusive", False)), True)
            )
        for s in m.sets:
            for target in s.target_modules:
                point = binding.hook_point(target) if binding else f"base::{target}"
                if point in points:
                    continue
                points.add(point)
                table.setdefault(point, []).append(HookUse(m.mod_id, "transform", False, False))
    return table


def require_ok(plan: LoadPlan, depth: int | None = None) -> None:
    if not plan.ok:
        raise LoadOrderError(plan.report(depth))


def load_mods(
    model: Any,
    choices: Sequence[ModChoice],
    *,
    depth: int | None = None,
    family: str | None = None,
):

    from .bake import bake

    plan = resolve(choices, family=family)
    used = plan.depth(depth)
    require_ok(plan, used)
    return bake(model, plan.scoped_sets(), used), plan, used


def _set_strength(s: Any) -> str:
    scale = (getattr(s, "imported_from", None) or {}).get("scale")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale != 1:
        return f"{s.name} {s.strength:+g} (file alpha/rank {scale:g})"
    return f"{s.name} {s.strength:+g}"
