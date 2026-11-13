# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import math
import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch

from .. import key_graph
from ..compute import (
    _share_transition_base,
    key_probes,
    orthonormal_keys,
    share_transition_gamma,
    share_transition_kind,
)
from ..fast_keys import Keys, calibrate_log_tau, key_settings
from ..subspace_keys import keys_from_records
from . import key_cache

log = logging.getLogger("kilora.keys")

_MEMORY_KEYS = weakref.WeakKeyDictionary()
_MEMORY_LOCK = threading.Lock()


def _memory_key(model, key, args):
    if model is None:
        return None
    try:
        tensors = (*model.parameters(), *model.buffers())
        versions = tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in tensors)
    except (AttributeError, RuntimeError):
        return None
    return (
        key_cache.cache_dir(),
        key,
        versions,
        tuple(args["input"].shape),
        str(args["input"].dtype),
        str(args["input"].device),
        tuple(args["timestep"].detach().cpu().flatten().tolist()),
    )


def _plan_ref(plan):
    try:
        return weakref.ref(plan)
    except TypeError:
        return lambda: plan


FRACTIONS = (1.0, 0.66, 0.33)
DEPTH = 2
M_IN = 4
M_OUT = 4


def joint_sha256(files: Sequence[tuple[str, float]]) -> str:
    text = "|".join(f"{key_cache.file_sha256(path)}:{float(w)!r}" for path, w in files)
    return hashlib.sha256(text.encode()).hexdigest()


def _cond_sha256(args: dict) -> str:
    c = (args.get("c") or {}).get("c_crossattn")
    if not isinstance(c, torch.Tensor):
        return "none"
    data = c.detach().float().cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def _probe_rows(args: dict):

    from ..call_values import current_row_axes

    x = args["input"]
    batch = x.shape[0]
    c = args.get("c") or {}
    options = c.get("transformer_options") or {}
    kinds = options.get("cond_or_uncond")
    ids = current_row_axes()
    if ids is None or not kinds or batch % len(kinds) or len(ids) != batch:
        return args, None
    width = batch // len(kinds)
    row_kinds = [kind for kind in kinds for _ in range(width)]
    positive = [i for i, kind in enumerate(row_kinds) if kind == 0]
    if not positive:
        return args, None
    first_id = ids[positive[0]]
    rows = [i for i in positive if ids[i] == first_id]
    if len(rows) == batch:
        return args, None
    index = torch.tensor(rows, device=x.device)

    def select(value):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
            return value.index_select(0, index.to(value.device))
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(select(item) for item in value)
        if isinstance(value, list):
            return [select(item) for item in value]
        return value

    selected = select(args)
    selected["c"]["transformer_options"]["cond_or_uncond"] = [0] * sum(
        row_kinds[i] == 0 and ids[i] == first_id for i in range(0, batch, width)
    )
    return selected, first_id


class KeyPass:
    def __init__(
        self,
        module_plans: Sequence[Any],
        jobs: Sequence[tuple[str, str]],
        family: str,
        scope: str = "token",
        linear_jump: bool = False,
        model: Any = None,
    ):
        self._module_plan_refs = [_plan_ref(plan) for plan in module_plans]
        self.jobs = list(jobs)
        self.family = family
        self.scope = scope
        self.linear_jump = linear_jump
        self._model_ref = weakref.ref(model) if model is not None else lambda: None
        self.done = False
        self.report = ""

    @property
    def module_plans(self):
        return [plan for ref in self._module_plan_refs if (plan := ref()) is not None]

    def _targets(self, set_key: str) -> list[tuple[Any, int]]:
        out = []
        for plan in self.module_plans:
            for _, _, target in plan.parts:
                for pos, s in enumerate(target.sets):
                    if s.key == set_key:
                        out.append((target, pos))
        return out

    def _clear(self) -> None:
        for plan in self.module_plans:
            plan._cache.clear()
            prefetch = getattr(plan, "_prefetch", None)
            if prefetch is not None:
                prefetch.reset()

    def wrap(self, previous):
        def wrapper(apply_model, args):
            if not self.done:
                from ..call_values import active_axes, row_axes
                from ..compute import AXES

                probe_args, layer_id = _probe_rows(args)
                selected = contextlib.nullcontext()
                if layer_id is not None and 0 <= layer_id < len(AXES):
                    layer = AXES[layer_id]
                    axes = ("creative", "functional") if layer == "creative" else (layer,)
                    selected = active_axes(axes)
                if probe_args is args:
                    self.run(apply_model, args)
                else:
                    with selected, row_axes(None):
                        self.run(apply_model, probe_args)
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        return wrapper

    def run(self, apply_model, args: dict) -> None:
        self.done = True
        settings = {
            "cond": _cond_sha256(args),
            "fractions": FRACTIONS,
            "depth": DEPTH,
            "m_in": M_IN,
            "m_out": M_OUT,
            "scope": self.scope,
            **key_settings(),
        }
        graph_on = key_graph.enabled()
        if graph_on:
            settings["graph_depths"] = sorted(
                (t.name, s.key, self._graph_depth(t, s))
                for plan in self.module_plans
                for _, _, t in plan.parts
                for s in t.sets
                if not s.apply_once
            )
        found: dict[str, tuple[Keys, str]] = {}
        missing = []
        for set_key, identity in self.jobs:
            if not self._targets(set_key):
                continue
            key = key_cache.cache_key(identity, self.family, settings)
            model = self._model_ref()
            memory_key = _memory_key(model, key, args)
            cached = None
            if memory_key is not None and not graph_on:
                with _MEMORY_LOCK:
                    memory = _MEMORY_KEYS.get(model, {})
                    cached = copy.deepcopy(memory.get(memory_key))
            if cached is None:
                cached = key_cache.load(key)
            if cached is not None:
                found[set_key] = (cached, "the cache")
            else:
                missing.append((set_key, key))
        if missing:
            computed = self._compute(apply_model, args, [k for k, _ in missing])
            for set_key, key in missing:
                keys = computed.get(set_key)
                if keys is None:
                    continue
                found[set_key] = (keys, "a key pass")
                if not graph_on:
                    memory_key = _memory_key(model, key, args)
                    if memory_key is not None:
                        with _MEMORY_LOCK:
                            memory = _MEMORY_KEYS.setdefault(model, OrderedDict())
                            memory[memory_key] = copy.deepcopy(keys)
                            while len(memory) > 8:
                                memory.popitem(last=False)
                    try:
                        key_cache.store(key, keys, keys.log_tau)
                    except OSError as e:
                        log.warning("cannot write the key cache (%s)", e)
        lines = []
        for set_key, (keys, source) in found.items():
            self._write(keys, self._targets(set_key))
            lines.append(
                f"State keys of {set_key} from {source} "
                f"(routing temperature {math.exp(keys.log_tau):.3g})."
            )
        if graph_on:
            if any(
                not s.apply_once and s.graph_distances is None
                for plan in self.module_plans
                for _, _, t in plan.parts
                for s in t.sets
            ):
                x = args["input"]
                generator = torch.Generator().manual_seed(0)
                noise = torch.randn(x.shape, generator=generator).to(x.device, x.dtype)
                self._measure_graph(apply_model, {**args, "input": noise})
            for set_key, key in missing:
                if set_key not in found:
                    continue
                keys = found[set_key][0]
                for target, pos in self._targets(set_key):
                    if target.name in keys:
                        layer_set = target.sets[pos]
                        if layer_set.apply_once:
                            continue
                        for part in key_cache.GRAPH_PARTS:
                            keys[target.name][part] = getattr(layer_set, part)
                try:
                    key_cache.store(key, keys, keys.log_tau)
                except (OSError, AttributeError, KeyError, TypeError) as e:
                    log.warning("cannot write the graph cache (%s)", e)
            report = key_graph.graph_report(self.module_plans)
            if report:
                layers = {}
                for (name, _, _), row in report.items():
                    layers[name] = layers.get(name, True) and row["qualifies"]
                count = sum(layers.values())
                lines.append(f"Key graph: {count}/{len(layers)} layers qualify (threshold 0.1).")
                lines.append(key_graph.summary(report))
        if self.linear_jump:
            self._fit_jump(apply_model, args)
        self.report = " ".join(lines)
        if lines:
            log.info(self.report)

    @staticmethod
    def _graph_depth(target, layer_set) -> int:
        from ..depth_schedule import MAX_DEPTH

        depth = layer_set.fixed_depth if layer_set.fixed_depth is not None else target.depth
        if target.schedule is not None and layer_set.fixed_depth is None:
            depth = MAX_DEPTH
        return max(1, math.ceil(depth))

    def _measure_graph(self, apply_model, args: dict) -> None:
        from ..compute import recording_read_keep, recording_set_shares

        targets = [t for plan in self.module_plans for _, _, t in plan.parts]
        saved = [
            (
                t,
                t.depth,
                t.schedule,
                t.token_depth,
                [(s, s.route_freeze, s.fixed_depth, s.fixed_point_stop) for s in t.sets],
            )
            for t in targets
        ]
        records = {}
        keeps = {}

        def observe_keep(name, rates):
            keeps.setdefault(name, []).append(rates.cpu())

        def observe(name, gates, sets):
            for gate, layer_set in zip(gates, sets, strict=True):
                if not layer_set.apply_once:
                    records.setdefault((name, layer_set.key), []).append(gate.cpu())

        for t in targets:
            iterated = [s for s in t.sets if not s.apply_once]
            depths = [self._graph_depth(t, s) for s in iterated]
            for s, depth in zip(iterated, depths, strict=True):
                s.graph_transition, s.graph_powers = key_graph.build(s, depth)
                s.graph_distances = None
                s.route_freeze = None
                s.fixed_point_stop = False
                if s.fixed_depth is not None:
                    s.fixed_depth = depth
            t.depth = max(depths, default=1)
            t.schedule = None
            t.token_depth = None
        self._clear()
        devices = sorted({args["input"].device.index} if args["input"].is_cuda else set())
        try:
            with (
                torch.random.fork_rng(devices=devices),
                torch.no_grad(),
                recording_set_shares(observe),
                recording_read_keep(observe_keep),
            ):
                apply_model(args["input"], args["timestep"], **args["c"])
            for t in targets:
                for pos, s in enumerate(t.sets):
                    if s.apply_once:
                        continue
                    name, local = _axis_place(t, pos)
                    weights = records.get((name, s.key), [])
                    if weights:
                        weights = weights[: len(s.graph_powers) - 1]
                        s.graph_distances = key_graph.distances(weights, s.graph_powers)
                        axis = s.axis
                        local = sum(
                            1
                            for prior in t.sets[:pos]
                            if not prior.apply_once and prior.axis == axis
                        )
                        rates = keeps.get(f"{name}#{local}", [])
                        keep = torch.stack(rates).mean(0) if rates else None
                        rho, damped = key_graph.loop_gain(s, keep)
                        entropy, loop, merges = key_graph.sequence_metrics(weights)
                        s.graph_stats = torch.tensor(
                            [float("nan") if v is None else v for v in (rho, damped, entropy, loop)]
                            + [weights[0].numel() // weights[0].shape[-1] * (len(weights) - 1)]
                        )
                        s.graph_hops = key_graph.hop_distances(weights, s.graph_powers)
                        s.graph_merges = merges
        finally:
            for t, depth, schedule, token_depth, sets in saved:
                t.depth, t.schedule, t.token_depth = depth, schedule, token_depth
                for s, freeze, fixed, stop in sets:
                    s.route_freeze, s.fixed_depth, s.fixed_point_stop = freeze, fixed, stop
            self._clear()

    def _fit_jump(self, apply_model, args: dict) -> None:
        from .. import linear_jump
        from ..compute import sampling_progress, step_carry

        targets = [t for plan in self.module_plans for _, _, t in plan.parts]
        saved = [(t, t.schedule, [s.route_freeze for s in t.sets]) for t in targets]
        for t, _, _ in saved:
            t.schedule = None
            for s in t.sets:
                s.route_freeze = None
        self._clear()
        try:
            for band, progress, fraction in (("before", 0.2, 1.0), ("after", 0.8, 0.33)):
                with (
                    torch.no_grad(),
                    sampling_progress(progress),
                    step_carry({"__linear_jump_probe__": None}),
                    linear_jump.recording(band) as rec,
                ):
                    apply_model(args["input"], args["timestep"] * fraction, **args["c"])
                for t in targets:
                    fit = rec["layers"].get((t.name, t.depth))
                    if fit is not None and len(t.sets) == 1 and t.sets[0].jump_fits is not None:
                        t.sets[0].jump_fits[(band, t.depth)] = fit
        finally:
            for t, schedule, freezes in saved:
                t.schedule = schedule
                for s, freeze in zip(t.sets, freezes, strict=True):
                    s.route_freeze = freeze
            self._clear()

    def _compute(self, apply_model, args: dict, set_keys: list[str]) -> dict[str, Keys]:
        targets = {k: self._targets(k) for k in set_keys}
        every = [tp for tps in targets.values() for tp in tps]
        depths = {id(t): t.depth for t, _ in every}
        frozen = {id(ls): ls.route_freeze for t, _ in every for ls in t.sets}
        for t, _ in every:
            t.depth = DEPTH
            for ls in t.sets:
                ls.route_freeze = None
        self._clear()
        x, t0 = args["input"], args["timestep"]
        gen = torch.Generator().manual_seed(0)
        try:
            with torch.no_grad(), key_probes() as everything:
                for frac in FRACTIONS:
                    noise = torch.randn(x.shape, generator=gen).to(x.device, x.dtype)
                    apply_model(noise, t0 * frac, **args["c"])
        finally:
            for t, _ in every:
                t.depth = depths[id(t)]
                for ls in t.sets:
                    ls.route_freeze = frozen[id(ls)]
            self._clear()
        out = {}
        for set_key, tps in targets.items():
            seen = {t.name: _axis_place(t, pos) for t, pos in tps}
            positions = {name: pos for name, (_, pos) in seen.items()}
            record = {name: everything.get(at, []) for name, (at, _) in seen.items()}
            current = {
                t.name: (
                    t.sets[pos].keys_in,
                    t.sets[pos].keys_out,
                    _rows(t.sets[pos].rows_in, t.sets[pos].keys_in),
                    _rows(t.sets[pos].rows_out, _out_keys(t.sets[pos])),
                )
                for t, pos in tps
            }
            found = keys_from_records(record, positions, current, M_IN, M_OUT)
            queries = {}
            for name, pos in positions.items():
                first = next(
                    (
                        e
                        for e in record.get(name, [])
                        if e.get("k") in (0, 1)
                        and e.get("queries") is not None
                        and e["queries"][pos] is not None
                    ),
                    None,
                )
                if first is not None:
                    queries[name] = first["queries"][pos].float().cpu()
            log_tau = calibrate_log_tau(
                queries,
                {n: (found[n]["keys_in"], found[n]["rows_in"]) for n in queries if n in found},
                scope=self.scope,
            )
            if share_transition_gamma() and share_transition_kind() == "weighted":
                for t, pos in tps:
                    samples = [
                        e["state"].reshape(-1, e["state"].shape[-1])
                        for e in record.get(t.name, [])
                        if e.get("state") is not None
                    ]
                    if samples and t.name in found:
                        transition = _share_transition_base(t.sets[pos], states=torch.cat(samples))
                        if transition is not None:
                            found[t.name]["share_transition_weighted"] = transition.cpu()
            keys = Keys(found)
            keys.log_tau = float(log_tau)
            out[set_key] = keys
        return out

    def _write(self, keys: Keys, targets: list) -> None:
        tau = math.exp(keys.log_tau)
        for t, pos in targets:
            k = keys.get(t.name)
            if k is None:
                continue
            s = t.sets[pos]
            device = s.keys_in.device
            s.keys_in = orthonormal_keys(k["keys_in"].float(), k["rows_in"]).to(device)
            s.keys_out = orthonormal_keys(k["keys_out"].float(), k["rows_out"]).to(device)
            s.rows_in = k["rows_in"].to(device)
            s.rows_out = k["rows_out"].to(device)
            s.share_transition_weighted = (
                k["share_transition_weighted"].to(device)
                if "share_transition_weighted" in k
                else None
            )
            s.tau = tau
            if key_graph.enabled() and "graph_transition" in k:
                for part in key_cache.GRAPH_PARTS:
                    setattr(s, part, k[part])
        self._clear()


def _axis_place(t: Any, pos: int) -> tuple[str, int]:
    from ..compute import axis_target

    axis = t.sets[pos].axis
    if len({s.axis for s in t.sets}) < 2:
        return t.name, pos
    local = sum(1 for s in t.sets[:pos] if s.axis == axis)
    return axis_target(t.name, axis), local


def _out_keys(s: Any) -> torch.Tensor:
    return s.keys_in if s.keys_out is None else s.keys_out


def _rows(rows: torch.Tensor | None, keys: torch.Tensor) -> torch.Tensor:
    if rows is not None:
        return rows
    return torch.full((keys.shape[0],), keys.shape[1], dtype=torch.long)


__all__ = ["KeyPass", "joint_sha256"]
