# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import hashlib
import logging
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .adaptive_gate import DEFAULT_KAPPA, check_gate_mode, check_kappa
from .base_weights import read_base_weight
from .compute import (
    CARRIES,
    LayerSet,
    check_routing,
    derived_default_depth,
    derived_mode,
    expert_ids_of,
    import_score,
    layer_places,
    orthonormal_keys,
    run_layer,
    share_transition,
    share_transition_gamma,
)
from .config import KiLoRAConfig, check_depth, check_depth_range
from .errors import KiLoRAError
from .path_walks import check_path_select, check_path_starts, check_path_width
from .route_freeze import check_route_freeze
from .schedule import draw_depth
from .sets import (
    AdapterSet,
    KeyedExperts,
    check_compatible,
    check_prior,
    check_strength,
    needs_state_keys,
    new_keyed_set,
    routes_alone,
    set_identity,
)

log = logging.getLogger("kilora.keys")
AUTO_KEYS = os.environ.get("KILORA_AUTO_KEYS", "1") != "0"


def _layer_token_depth(depth_map: torch.Tensor | None, h: torch.Tensor):
    import math

    import torch.nn.functional as F

    if depth_map is None:
        return None
    dims = tuple(h.shape[1:-1])
    if depth_map.dim() == 2 and depth_map.shape[1] == math.prod(dims):
        return depth_map
    if tuple(depth_map.shape[1:]) == dims:
        return depth_map
    if depth_map.dim() != 3:
        return None
    if len(dims) >= 2:
        gh, gw = dims[-2], dims[-1]
    else:
        side = math.isqrt(dims[0])
        if side * side != dims[0]:
            return None
        gh = gw = side
    m = F.interpolate(depth_map[:, None].to(h.device), size=(gh, gw), mode="area")[:, 0]
    if len(dims) > 2:
        m = m.reshape(m.shape[0], *([1] * (len(dims) - 2)), gh, gw).expand(m.shape[0], *dims)
    return m.reshape(m.shape[0], -1)


def _call_hash(args: tuple, kwargs: dict) -> str:
    words = ("encoder", "context", "text", "prompt", "cond")
    picked = [
        v
        for k, v in sorted(kwargs.items())
        if isinstance(v, torch.Tensor) and any(w in k for w in words)
    ]
    if not picked:
        picked = [a for a in args[1:] if isinstance(a, torch.Tensor)]
    h = hashlib.sha256()
    for t in picked:
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class Settings:
    depth: int = 1
    train_depth: tuple[int, int] | None = None
    last_step_training: bool = False
    routing_noise: float = 0.0
    softmax_scope: str = "shared"
    routing_scope: str = "token"
    routing_select: str = "sparsemax"
    carry: str = "dropped"
    iteration: str = "accumulate"
    train_iteration: str = "accumulate"
    fixed_point_stop: bool = False
    path_width: int = 1
    path_select: str = "converged"
    route_freeze: float | None = None
    path_starts: str = "top"
    path_prior_min: float = 0.0
    path_merge: bool = False
    path_beam: bool = False
    output_gate: str = "separable"
    gate_mode: str = "learned"
    gate_kappa: float = DEFAULT_KAPPA
    carry_across_steps: bool = False
    seed: int = 0
    token_roles: str = "auto"
    import_score: str = "derived"
    derived_entry_temp: float | None = None
    candidates: int = 0
    candidate_summary: str = "keys"

    @classmethod
    def from_config(cls, config: KiLoRAConfig) -> Settings:
        return cls(
            depth=config.depth,
            train_depth=config.train_depth,
            last_step_training=config.last_step_training,
            routing_noise=config.routing_noise,
            softmax_scope=config.softmax_scope,
            routing_scope=config.routing_scope,
            routing_select=config.routing_select,
            carry=config.carry,
            iteration=config.iteration,
            train_iteration=config.train_iteration,
            fixed_point_stop=config.fixed_point_stop,
            output_gate=config.output_gate,
            carry_across_steps=config.carry_across_steps,
            seed=config.seed,
        )


IMPORTED_DEPTH = 3
IMPORTED_ROUTE_FREEZE = 0.4
IMPORTED_DEPTH_SCHEDULE = "1 + 2 * (t >= 0.3)"
IMPORTED_SCORE = "derived"


def imported_settings(**overrides) -> Settings:
    values = dict(
        depth=IMPORTED_DEPTH,
        softmax_scope="per_set",
        gate_mode="auto",
        route_freeze=IMPORTED_ROUTE_FREEZE,
        import_score=IMPORTED_SCORE,
    )
    if (
        os.environ.get("KILORA_IMPORT_SCORE") or IMPORTED_SCORE
    ) == "derived" and derived_mode() == "full":
        values.update(depth=derived_default_depth(), path_width=3)
    values.update(overrides)
    return Settings(**values)


def imported_schedule_fields() -> dict:
    if (
        os.environ.get("KILORA_IMPORT_SCORE") or IMPORTED_SCORE
    ) == "derived" and derived_mode() == "full":
        return {}
    return {"depth_schedule": IMPORTED_DEPTH_SCHEDULE, "schedule_yields_to_depth": True}


def imported_schedule():
    from .depth_schedule import compile_schedule

    if (
        os.environ.get("KILORA_IMPORT_SCORE") or IMPORTED_SCORE
    ) == "derived" and derived_mode() == "full":
        return None
    return compile_schedule(IMPORTED_DEPTH_SCHEDULE)


def _axis_fields(axis: str) -> dict:
    if axis == "creative":
        return {}
    if axis in ("functional", "ip"):
        return {"axis": axis, "gate_mode": "none", "fixed_depth": 1, "route_freeze": None}
    return {"axis": axis}


def find_target_modules(model: nn.Module, patterns: Sequence[str]) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            names.append(name)
    if not names:
        raise KiLoRAError(f"no nn.Linear module matches target_modules {list(patterns)}")
    return names


class _Site(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensors = nn.ParameterDict()
        self.keyed: list[int] = []
        self.rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.derived_sources: dict[int, tuple] = {}
        self.routing_sources: dict[int, tuple] = {}
        self.exact: dict[int, dict[int, int]] = {}

    def add(self, key: str, value: torch.Tensor, device: torch.device, frozen: bool) -> None:
        with torch.inference_mode(False):
            param = nn.Parameter(value.detach().to(device=device, dtype=torch.float32).clone())
            param.requires_grad_(not frozen)
        self.tensors[key] = param


class KiLoRA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        sets: Sequence[AdapterSet],
        settings: Settings,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not sets:
            raise KiLoRAError("at least one adapter set is needed")
        names = [s.name for s in sets]
        if len(set(names)) != len(names):
            twice = sorted({n for n in names if names.count(n) > 1})
            raise KiLoRAError(
                f"adapter set names must be unique, got {names} ({', '.join(twice)} more "
                "than once). Give each set its own name: names=[...] in compose, or "
                "--names when you run kilora import."
            )
        if settings.softmax_scope not in ("shared", "per_set"):
            raise ValueError(f"unknown softmax_scope {settings.softmax_scope!r}")
        if settings.candidate_summary not in ("keys", "read_map"):
            raise ValueError("candidate_summary must be keys or read_map")
        if (
            not isinstance(settings.candidates, int)
            or isinstance(settings.candidates, bool)
            or settings.candidates < 0
        ):
            raise ValueError("candidates must be a nonnegative integer")
        check_routing(settings.routing_scope, settings.routing_select)
        check_gate_mode(settings.gate_mode)
        settings.path_width = check_path_width(settings.path_width)
        check_path_select(settings.path_select)
        settings.route_freeze = check_route_freeze(settings.route_freeze)
        check_path_starts(settings.path_starts)
        settings.gate_kappa = check_kappa(settings.gate_kappa)
        if settings.carry not in CARRIES:
            raise ValueError(f"unknown carry {settings.carry!r}")
        if settings.train_depth is not None:
            settings.train_depth = check_depth_range(settings.train_depth)
        if settings.softmax_scope == "shared":
            for other in sets[1:]:
                check_compatible(sets[0], other)
        self.settings = settings
        transfers = [s.extra.get("_library_transfer") for s in sets]
        self._library_transfers = {i: config for i, config in enumerate(transfers) if config}
        self._block_prefetch = None
        self._prefetched = {}
        self.metadata = dict(metadata or {})
        self._set_info = [
            {
                "name": s.name,
                "part_type": s.part_type,
                "aggregation": s.aggregation,
                "form": s.form,
                "imported_from": s.imported_from,
                "extra": dict(s.extra),
                "alone": routes_alone(s.extra),
                "key_source": s.key_source,
                "ir_parts": dict(s.ir_parts),
                "learned_cross": s.learned_cross,
                "learned_cross_zero": s.learned_cross_zero,
                "identity": (
                    set_identity(s)
                    if needs_state_keys(s)
                    and (
                        s.form != "input"
                        or import_score(settings, form=s.form) != "derived"
                        or (
                            share_transition_gamma(settings)
                            and os.environ.get("KILORA_SHARE_TRANSITION_KIND") == "weighted"
                        )
                    )
                    else None
                ),
                "expert_ids": expert_ids_of(s),
            }
            for s in sets
        ]
        self.log_taus = nn.ParameterDict(
            {
                f"s{i}": nn.Parameter(torch.tensor(float(s.log_tau)), requires_grad=not s.frozen)
                for i, s in enumerate(sets)
            }
        )
        self._keyed_index = list(range(len(sets)))
        self.register_buffer(
            "strengths", torch.tensor([check_strength(s.strength) for s in sets]), persistent=False
        )
        self.register_buffer(
            "priors", torch.tensor([check_prior(s.prior) for s in sets]), persistent=False
        )
        self._check_priors()
        self.routers = nn.ParameterDict(
            {
                f"s{i}_{w}": nn.Parameter(r.detach().float().clone(), requires_grad=False)
                for i, s in enumerate(sets)
                for w, r in s.routers.items()
            }
        )

        targets: list[str] = []
        for s in sets:
            targets.extend(t for t in s.target_modules if t not in targets)
        self.targets = targets
        self.sites = nn.ModuleList()
        self._handles = []
        modules = dict(model.named_modules())
        self._out: list[int] = []
        for target in targets:
            module = modules.get(target)
            if module is None:
                raise KiLoRAError(f"target module {target!r} does not exist in this model")
            device = _module_device(module)
            site = _Site()
            out_width = None
            for i, s in enumerate(sets):
                ex = s.experts.get(target)
                if ex is None:
                    continue
                _check_shape(target, module, ex, s.form)
                for part in ("down", "up", "keys_in", "keys_out"):
                    value = getattr(ex, part)
                    if value is not None:
                        storage_device = (
                            torch.device("cpu") if i in self._library_transfers else device
                        )
                        site.add(
                            f"s{i}_{part}",
                            value,
                            storage_device,
                            s.frozen or (s.form == "state" and part.startswith("keys_")),
                        )
                        if i in self._library_transfers:
                            site.tensors[f"s{i}_{part}"].data = site.tensors[
                                f"s{i}_{part}"
                            ].data.pin_memory()
                if ex.routing_down is not None:
                    if ex.routing_source:
                        site.routing_sources[i] = ex.routing_source
                    site.add(f"s{i}_routing_down", ex.routing_down, device, True)
                    site.add(f"s{i}_routing_up", ex.routing_up, device, True)
                site.exact[i] = {expert: len(components) for expert, components in ex.exact.items()}
                for expert, components in ex.exact.items():
                    for part, value in enumerate(components):
                        storage_device = (
                            torch.device("cpu") if i in self._library_transfers else device
                        )
                        key = f"s{i}_exact_{expert}_{part}"
                        site.add(key, value, storage_device, True)
                        if i in self._library_transfers:
                            site.tensors[key].data = site.tensors[key].data.pin_memory()
                from . import derived_cache

                source_id = derived_cache.source(ex)
                if source_id and ex.down is not None:
                    a, b = site.tensors[f"s{i}_down"], site.tensors[f"s{i}_up"]
                    site.derived_sources[i] = (source_id, a, a._version, b, b._version)
                if (
                    s.form == "input"
                    and ex.down is not None
                    and device.type != "meta"
                    and (
                        import_score(settings, form=s.form) != "derived"
                        and share_transition_gamma(settings)
                        and os.environ.get("KILORA_SHARE_TRANSITION_KIND", "cosine") != "weighted"
                    )
                ):
                    down = site.tensors[f"s{i}_down"]
                    up = site.tensors[f"s{i}_up"]
                    per_expert = down.shape[1] * down.shape[2]
                    prepared_set = LayerSet(
                        router=site.tensors[f"s{i}_keys_in"],
                        keys_in=site.tensors[f"s{i}_keys_in"],
                        per_expert=per_expert,
                        down=down.reshape(-1, down.shape[-1]),
                        up=up.reshape(-1, up.shape[-1]),
                        form="input",
                        import_score=settings.import_score,
                    )
                    if share_transition_gamma(settings):
                        share_transition(prepared_set)
                if i in self._library_transfers:
                    site.rows[i] = (ex.rows_in.cpu().pin_memory(), ex.rows_out.cpu().pin_memory())
                else:
                    site.rows[i] = (ex.rows_in.to(device), ex.rows_out.to(device))
                for name, value in (ex.gate or {}).items():
                    storage_device = torch.device("cpu") if i in self._library_transfers else device
                    site.add(f"s{i}_gate_{name}", value, storage_device, s.frozen)
                    if i in self._library_transfers:
                        site.tensors[f"s{i}_gate_{name}"].data = site.tensors[
                            f"s{i}_gate_{name}"
                        ].data.pin_memory()
                site.keyed.append(i)
                out_width = ex.out_features
            self.sites.append(site)
            self._out.append(out_width)
        device = _module_device(modules[targets[0]])
        self.strengths = self.strengths.to(device)
        self.priors = self.priors.to(device)
        self.routers.to(device)
        self.log_taus.to(device)

        self._depth = check_depth(settings.depth)
        self.depth_schedule: Any = None
        self.token_depth: torch.Tensor | None = None
        self.token_depth_start = 0.0
        self.token_depth_base = 0.0
        self._places = layer_places(targets)
        self._step_depth: int | None = None
        self._step_index = 0
        self.eval_mode = False
        for index, target in enumerate(targets):
            handle = modules[target].register_forward_hook(_Hook(self, index))
            self._handles.append(handle)
        self._keys_todo = [i for i, info in enumerate(self._set_info) if info["identity"]]
        self._keys_busy = False
        if self._keys_todo:
            self._handles.append(model.register_forward_pre_hook(self._auto_keys, with_kwargs=True))
        if self._library_transfers:
            self._install_library_prefetch(modules)

    def _install_library_prefetch(self, modules):
        from .library.prefetch import BlockPrefetch, CudaCopies, SynchronousCopies

        grouped = {}
        for index, target in enumerate(self.targets):
            block = self._places[target][0]
            grouped.setdefault(block, []).append((index, target))
        config = next(iter(self._library_transfers.values()))
        groups = list(grouped.values())

        def source(block):
            values = {}
            for index, _target in groups[block]:
                site = self.sites[index]
                for name, tensor in site.tensors.items():
                    if any(name.startswith(f"s{i}_") for i in self._library_transfers):
                        values[(index, name)] = tensor
                for i, rows in site.rows.items():
                    if i in self._library_transfers:
                        values[(index, i, 0)], values[(index, i, 1)] = rows
            return values

        def before(block, module, _args):
            device = _module_device(module)
            if device.type != "cuda":
                if self._block_prefetch is not None:
                    self._block_prefetch.reset()
                self._prefetched = {}
                return
            if self._block_prefetch is None or self._block_prefetch.backend.device != device:
                backend = (
                    CudaCopies(device, config["report"])
                    if config["mode"] == "auto"
                    else SynchronousCopies(device)
                )
                self._block_prefetch = BlockPrefetch(
                    len(groups), config["window"], source, backend, enabled=config["mode"] == "auto"
                )
            self._prefetched = self._block_prefetch.begin(block)

        def after(block, module, _args, _output):
            if self._block_prefetch is not None and _module_device(module).type == "cuda":
                self._block_prefetch.end(block)
            self._prefetched = {}

        for block, group in enumerate(groups):
            paths = [name.split(".") for _, name in group]
            common = []
            for parts in zip(*paths, strict=False):
                if len(set(parts)) != 1:
                    break
                common.append(parts[0])
            path = ".".join(common)
            module = modules[path]
            self._handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, b=block: before(b, module, args)
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    lambda module, args, output, b=block: after(b, module, args, output),
                    always_call=True,
                )
            )

    def _auto_keys(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if self._keys_busy or not self._keys_todo:
            return None
        todo, self._keys_todo = self._keys_todo, []
        names = ", ".join(self._set_info[i]["name"] for i in todo)
        if not AUTO_KEYS:
            log.warning("routing of %s runs on random keys (KILORA_AUTO_KEYS=0)", names)
            return None
        self._keys_busy = True
        try:
            with torch.no_grad():
                for i in todo:
                    self._state_keys_for(i, lambda: module(*args, **kwargs), args, kwargs)
        except Exception as e:
            log.warning("state keys for %s failed (%s); routing runs on random keys", names, e)
        finally:
            self._keys_busy = False
        return None

    def _state_keys_for(self, i: int, call, args: tuple, kwargs: dict) -> None:
        from . import key_cache
        from .fast_keys import key_settings, state_keys_from_call

        settings = {
            "cond": _call_hash(args, kwargs),
            "m_in": 4,
            "m_out": 4,
            "depth": 2,
            "scope": self.settings.routing_scope,
            **key_settings(),
        }
        family = str(self.metadata.get("family") or self.metadata.get("base_model") or "")
        key = key_cache.cache_key(self._set_info[i]["identity"], family, settings)
        keys = key_cache.load(key)
        source = "the cache"
        if keys is None:
            keys = state_keys_from_call(self, call, i)
            source = "a key pass"
            try:
                key_cache.store(key, keys, keys.log_tau)
            except OSError as e:
                log.warning("cannot write the key cache (%s)", e)
        self.write_state_keys(i, keys, keys.log_tau)
        log.info("state keys of %s from %s", self._set_info[i]["name"], source)

    def write_state_keys(self, i: int, keys: Mapping[str, Mapping], log_tau: float) -> None:
        if self._block_prefetch is not None:
            self._block_prefetch.reset()
        for target, k in keys.items():
            if target not in self.targets:
                continue
            site = self.sites[self.targets.index(target)]
            if i not in site.keyed:
                continue
            old = site.tensors[f"s{i}_keys_in"]
            frozen, device = not old.requires_grad, old.device
            site.add(f"s{i}_keys_in", k["keys_in"], device, frozen)
            site.add(f"s{i}_keys_out", k["keys_out"], device, frozen)
            site.rows[i] = (k["rows_in"].to(device), k["rows_out"].to(device))
            if i in self._library_transfers:
                for part in ("keys_in", "keys_out"):
                    tensor = site.tensors[f"s{i}_{part}"]
                    tensor.data = tensor.data.pin_memory()
                site.rows[i] = tuple(row.pin_memory() for row in site.rows[i])
            if "share_transition_weighted" in k:
                site.add(
                    f"s{i}_share_transition_weighted", k["share_transition_weighted"], device, True
                )
                if i in self._library_transfers:
                    tensor = site.tensors[f"s{i}_share_transition_weighted"]
                    tensor.data = tensor.data.pin_memory()
        with torch.no_grad():
            self.log_taus[f"s{i}"].fill_(float(log_tau))
        self._set_info[i]["key_source"] = "states"
        self._set_info[i]["identity"] = None

    def set_token_depth(
        self, depth_map: torch.Tensor | None, start: float = 0.0, base: float | None = None
    ) -> None:

        if not 0.0 <= float(start) <= 1.0:
            raise ValueError(f"token depth start must be a sampling progress in [0, 1]: {start}")
        self.token_depth = None if depth_map is None else depth_map.detach().float()
        self.token_depth_start = float(start)
        if base is None and self.token_depth is not None:
            base = math.floor(float(self.token_depth.mean()) + 0.5)
        self.token_depth_base = float(base or 0.0)

    def _token_depth_at(self, h: torch.Tensor) -> torch.Tensor | None:
        if self.token_depth is None:
            return None
        from .call_values import crop_map, current_crop_window, current_token_depth
        from .compute import current_progress
        from .region import current_latent

        on, base = current_token_depth()
        if on is None:
            on = not (self.token_depth_start > 0 and current_progress() < self.token_depth_start)
        try:
            version = self.token_depth._version
        except RuntimeError:
            version = None
        key = (
            self.token_depth,
            current_crop_window(),
            version,
            tuple(h.shape[1:-1]),
            h.device,
            on,
            base,
            self.token_depth_base,
        )
        cache = self.__dict__.setdefault("_token_depth_cache", {}) if version is not None else {}
        if key not in cache:
            if len(cache) >= 16:
                cache.clear()
            latent = current_latent()
            source = crop_map(self.token_depth, latent) if latent else self.token_depth
            mapped = _layer_token_depth(source, h)
            if mapped is not None and not on:
                mapped = torch.full_like(mapped, self.token_depth_base if base is None else base)
            if mapped is not None and mapped.is_inference():
                with torch.inference_mode(False):
                    mapped = mapped.clone()
            cache[key] = mapped
        return cache[key]

    @property
    def set_names(self) -> list[str]:
        return [info["name"] for info in self._set_info]

    @property
    def current_depth(self) -> int:
        return self._depth

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def set_depth(self, depth: int) -> None:
        self._depth = check_depth(depth)

    def set_depth_schedule(self, schedule: Any) -> None:
        self.depth_schedule = schedule

    @contextlib.contextmanager
    def depth(self, depth: int) -> Iterator[None]:
        old = self._depth
        self.set_depth(depth)
        try:
            yield
        finally:
            self._depth = old

    def set_strengths(self, strengths: Mapping[str, float] | Sequence[float]) -> None:
        self._write_controls(self.strengths, strengths, check_strength)

    def set_priors(self, priors: Mapping[str, float] | Sequence[float]) -> None:
        old = self.priors.clone()
        self._write_controls(self.priors, priors, check_prior)
        try:
            self._check_priors()
        except ValueError:
            self.priors.copy_(old)
            raise

    def bind_set(self, name: str, parts: Mapping[str, Mapping[str, nn.Parameter]]) -> None:
        index = self._set_index(name)
        for target, site in zip(self.targets, self.sites, strict=True):
            for part, param in parts.get(target, {}).items():
                key = f"s{index}_{part}"
                if key not in site.tensors:
                    raise KeyError(f"{target}: set {name!r} has no part {part!r}")
                site.tensors[key] = param

    def freeze_set(self, name: str, frozen: bool = True) -> None:
        index = self._set_index(name)
        for site in self.sites:
            for key, param in site.tensors.items():
                if key.startswith(f"s{index}_"):
                    inactive = self._set_info[index]["form"] == "state" and "_keys_" in key
                    param.requires_grad_(not frozen and not inactive)

    def start_step(self, step: int) -> int:

        self._step_index = int(step)
        train_depth = self.settings.train_depth
        if train_depth is None:
            self._step_depth = self._depth
        else:
            self._step_depth = draw_depth(self.settings.seed, step, *train_depth)
        return self._step_depth

    def wrap_loss(self, loss: torch.Tensor) -> torch.Tensor:

        if self._step_depth == 0:
            zero = sum(p.sum() for p in self.trainable_parameters())
            if isinstance(zero, torch.Tensor):
                loss = loss + zero.to(loss.dtype) * 0.0
        return loss

    def add_sections(self, model: nn.Module, sections: Sequence[Any], family: str | None = None):
        from .families import FAMILIES
        from .sections import Section, layer_changes

        family = family or (self.metadata.get("base_model") or {}).get("family", "")
        spec = FAMILIES.get(family)
        to_diffusers = spec.diffusers_path if spec is not None else None
        from .bindings import BINDINGS

        binding = BINDINGS.get(family)
        modules = dict(model.named_modules())
        missing = []
        for item in sections:
            section, strength = (item, 1.0) if isinstance(item, Section) else item
            for path, changes in layer_changes([section], float(strength)).items():
                name = path if path in modules else None
                if name is None and to_diffusers is not None:
                    name = to_diffusers(path)
                if name not in modules and binding is not None:
                    found = binding.from_comfy(path)
                    if len(found) == 1 and found[0][1] is None:
                        name = found[0][0]
                if name not in modules:
                    missing.append(path)
                    continue
                module = modules[name]
                for change in changes:
                    validate_for = getattr(change, "validate_for", None)
                    if (
                        validate_for is not None
                        and hasattr(module, "in_features")
                        and hasattr(module, "out_features")
                    ):
                        validate_for(module.in_features, module.out_features)
                self._handles.append(module.register_forward_hook(_SectionHook(changes)))
        return missing

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def to_sets(self) -> list[AdapterSet]:
        out = []
        for i, info in enumerate(self._set_info):
            experts: dict[str, Any] = {}
            frozen = True
            for target, site in zip(self.targets, self.sites, strict=True):
                if i in site.keyed:
                    t = site.tensors
                    parts = {
                        part: _export(t[f"s{i}_{part}"]) if f"s{i}_{part}" in t else None
                        for part in ("down", "up")
                    }
                    rows_in, rows_out = (r.cpu() for r in site.rows[i])
                    prefix = f"s{i}_gate_"
                    gate = {
                        k[len(prefix) :]: _export(v) for k, v in t.items() if k.startswith(prefix)
                    }
                    experts[target] = KeyedExperts(
                        keys_in=orthonormal_keys(_export(t[f"s{i}_keys_in"]), rows_in),
                        keys_out=orthonormal_keys(_export(t[f"s{i}_keys_out"]), rows_out),
                        rows_in=rows_in.clone(),
                        rows_out=rows_out.clone(),
                        gate=gate or None,
                        exact={
                            expert: tuple(
                                _export(t[f"s{i}_exact_{expert}_{part}"]) for part in range(count)
                            )
                            for expert, count in site.exact.get(i, {}).items()
                        },
                        **parts,
                    )
                    frozen = frozen and not any(
                        t[f"s{i}_{part}"].requires_grad
                        for part in ("down", "up")
                        if f"s{i}_{part}" in t
                    )
            if info["form"] == "state":
                from .compute import _block_bases, derived_rows

                for ex in experts.values():
                    if ex.down is None:
                        continue
                    n, routes, rank, width = ex.down.shape
                    ex.keys_in, ex.rows_in, _ = _block_bases(
                        ex.down.reshape(n * routes * rank, width), routes * rank, derived_rows()
                    )
                    ex.keys_out, ex.rows_out, _ = _block_bases(
                        ex.up.reshape(n * routes * rank, ex.up.shape[-1]),
                        routes * rank,
                        derived_rows(),
                    )
            routers = {
                int(k.split("_")[1]): (
                    torch.eye(int(k.split("_")[1])) if info["form"] == "state" else _export(v)
                )
                for k, v in self.routers.items()
                if k.startswith(f"s{i}_")
            }
            out.append(
                AdapterSet(
                    name=info["name"],
                    part_type=info["part_type"],
                    experts=experts,
                    routers=routers,
                    aggregation=info["aggregation"],
                    form=info["form"],
                    strength=float(self.strengths[i]),
                    prior=float(self.priors[i]),
                    imported_from=info["imported_from"],
                    frozen=frozen,
                    extra=dict(info["extra"]),
                    ir_parts=dict(info["ir_parts"]),
                    log_tau=float(self.log_taus[f"s{i}"].detach()),
                    key_source=info["key_source"],
                    learned_cross=info["learned_cross"],
                    learned_cross_zero=info["learned_cross_zero"],
                )
            )
        return out

    def save(self, path: str | os.PathLike[str], **manifest: Any) -> None:

        from .modfile import save_mod

        fields = dict(self.metadata)
        fields.update(manifest)
        save_mod(path, self.to_sets(), self.settings, **fields)

    def _active_depth(self, training: bool) -> int:
        if training and self._step_depth is not None:
            return self._step_depth
        return self._depth

    def _layer_sets(self, index: int, site: _Site, device: torch.device) -> list[LayerSet]:
        grad_enabled = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
            t = {
                name: self._prefetched.get((index, name), value).to(device)
                for name, value in site.tensors.items()
            }
        strengths = self.strengths.to(device, copy=torch.is_grad_enabled())
        priors = self.priors.to(device)
        per_set = self.settings.softmax_scope == "per_set"
        out = []
        for i in site.keyed:
            info = self._set_info[i]
            down, up = t.get(f"s{i}_down"), t.get(f"s{i}_up")
            per_expert, flat_down, flat_up = 1, None, None
            if down is not None:
                experts, routes, rank, n_in = down.shape
                per_expert = routes * rank
                flat_down = down.reshape(experts * per_expert, n_in)
                flat_up = up.reshape(experts * per_expert, up.shape[-1])
            out.append(
                LayerSet(
                    router=self.routers[f"s{i}_{self._out[index]}"].to(device),
                    share_transition_weighted=t.get(f"s{i}_share_transition_weighted"),
                    keys_in=t[f"s{i}_keys_in"],
                    keys_out=t[f"s{i}_keys_out"],
                    rows_in=self._prefetched.get((index, i, 0), site.rows[i][0]).to(device),
                    rows_out=self._prefetched.get((index, i, 1), site.rows[i][1]).to(device),
                    iteration=self.settings.iteration,
                    fixed_point_stop=self.settings.fixed_point_stop,
                    path_width=self.settings.path_width,
                    path_select=self.settings.path_select,
                    route_freeze=self.settings.route_freeze,
                    path_starts=self.settings.path_starts,
                    path_prior_min=self.settings.path_prior_min,
                    path_merge=self.settings.path_merge,
                    path_beam=self.settings.path_beam,
                    output_gate=self.settings.output_gate,
                    gate_mode=self.settings.gate_mode,
                    gate_kappa=self.settings.gate_kappa,
                    gate={
                        k[len(f"s{i}_gate_") :]: v
                        for k, v in t.items()
                        if k.startswith(f"s{i}_gate_")
                    }
                    or None,
                    per_expert=per_expert,
                    exact={
                        expert: tuple(t[f"s{i}_exact_{expert}_{part}"] for part in range(count))
                        for expert, count in site.exact.get(i, {}).items()
                    },
                    down=flat_down,
                    up=flat_up,
                    routing_source=site.routing_sources.get(i),
                    routing_down=(
                        t[f"s{i}_routing_down"].reshape_as(flat_down)
                        if f"s{i}_routing_down" in t
                        else None
                    ),
                    routing_up=(
                        t[f"s{i}_routing_up"].reshape_as(flat_up)
                        if f"s{i}_routing_up" in t
                        else None
                    ),
                    scale=strengths[i],
                    log_prior=torch.log(
                        priors[i]
                        if any(self._set_info[j]["form"] == "state" for j in site.keyed)
                        else priors[i].clamp(min=1e-30)
                    )
                    if len(site.keyed) > 1
                    and (
                        not per_set
                        or {self._set_info[j]["form"] for j in site.keyed} == {"state", "input"}
                    )
                    else 0.0,
                    tau=_exp(self.log_taus[f"s{i}"].to(device)),
                    aggregation=info["aggregation"],
                    form=info["form"],
                    key=info["extra"].get("live_id", info["name"]),
                    ir_parts=info["ir_parts"],
                    routing_scope=self.settings.routing_scope,
                    routing_select=self.settings.routing_select,
                    candidates=self.settings.candidates,
                    candidate_summary=self.settings.candidate_summary,
                    carry=self.settings.carry,
                    expert_ids=info["expert_ids"],
                    entry_windows=info["extra"].get("entry_windows", ()),
                    import_score=self.settings.import_score,
                    derived_entry_temp=self.settings.derived_entry_temp,
                    learned_cross=info["learned_cross"],
                    learned_cross_zero=info["learned_cross_zero"],
                )
            )
        from . import derived_cache

        for layer, i in zip(out, site.keyed, strict=True):
            source = site.derived_sources.get(i)
            if (
                source
                and source[1] is site.tensors[f"s{i}_down"]
                and source[3] is site.tensors[f"s{i}_up"]
                and source[1]._version == source[2]
                and source[3]._version == source[4]
            ):
                derived_cache.bind(layer, source[0])
        for n, i in enumerate(site.keyed):
            fields = _axis_fields(self._set_info[i]["extra"].get("axis", "creative"))
            if fields:
                out[n] = dataclasses.replace(out[n], **fields)
        return out

    def _run(self, index: int, module: nn.Module, x: torch.Tensor, h: torch.Tensor):
        if x.ndim == 2 and h.ndim == 2:
            return self._run(index, module, x.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        out = self._run_layer(index, module, x, h)
        _observe_output(self.targets[index], x, h, out)
        return out

    def _run_layer(self, index: int, module: nn.Module, x: torch.Tensor, h: torch.Tensor):
        training = module.training and not self.eval_mode
        depth = self._active_depth(training)
        site = self.sites[index]
        schedule = None if training else self.depth_schedule
        if (depth == 0 and schedule is None) or not site.keyed:
            return h
        place, count, kind = self._places[self.targets[index]]
        settings = self.settings
        sets = self._layer_sets(index, site, h.device)
        from .compute import _cache_get, _cache_put, _tensor_key, base_weight_transition
        from .compute import import_score as effective_import_score

        needs_norm = (
            any(s.form == "input" and effective_import_score(s) == "derived" for s in sets)
            and derived_mode() == "full"
        )
        needs_base = os.environ.get("KILORA_SHARE_TRANSITION_KIND") == "base"
        source_weight = getattr(module, "weight", None)
        if (
            (needs_norm or needs_base)
            and isinstance(source_weight, torch.Tensor)
            and source_weight.ndim == 2
            and source_weight.device.type != "meta"
        ):
            weight = None
            key = _tensor_key(f"adapter-host-norm-{h.device}", [source_weight])
            norm = _cache_get(key)
            if norm is None:
                weight = read_base_weight(module, self.targets[index], device=h.device)
                norm = float(weight.float().norm())
                _cache_put(key, norm)
            prepared = []
            for s in sets:
                transition = None
                if needs_base and s.down is not None and s.up is not None:
                    key = _tensor_key(
                        f"adapter-base-transition-{s.per_expert}-{h.device}",
                        [source_weight, s.down, s.up],
                    )
                    transition = _cache_get(key)
                    if transition is None:
                        if weight is None:
                            weight = read_base_weight(module, self.targets[index], device=h.device)
                        transition = base_weight_transition(s, weight)
                        if transition is not None:
                            _cache_put(key, transition)
                prepared.append(
                    dataclasses.replace(
                        s,
                        host_weight_norm=norm if needs_norm else None,
                        share_transition_base=transition,
                    )
                )
            sets = prepared
        query_policy = None
        grad_steps = None
        if training and settings.last_step_training:

            def grad_steps(k: int, _d=depth) -> bool:
                return k == _d

        from .call_values import current_depth_pass, current_up_deltas

        fixed = current_depth_pass()
        if fixed is not None:
            depth, step = fixed if fixed[1] is None else (fixed[1], fixed[1])
            schedule = None
            grad_steps = None if step is None else (lambda k, _s=step: k == _s)
        deltas = current_up_deltas()
        if deltas:
            target = self.targets[index]
            sets = [
                dataclasses.replace(s, up=s.up + deltas[s.key][target].to(s.up))
                if s.up is not None and target in deltas.get(s.key, {})
                else s
                for s in sets
            ]

        def run(mode):
            if any(s.ir_parts for s in sets):
                from .ir import override_scope

                scope = override_scope()
            else:
                scope = contextlib.nullcontext()
            with scope:
                return run_layer(
                    x,
                    h,
                    [dataclasses.replace(s, iteration=mode) for s in sets],
                    depth,
                    shared=settings.softmax_scope == "shared",
                    noise_scale=settings.routing_noise if training else 0.0,
                    query_policy=query_policy,
                    grad_steps=grad_steps,
                    target=self.targets[index],
                    schedule=schedule,
                    index=place,
                    count=count,
                    kind=kind,
                    token_depth=None if fixed is not None else self._token_depth_at(h),
                )

        mode = settings.train_iteration if training else settings.iteration
        if mode != "mix":
            return run(mode)
        gen = torch.Generator().manual_seed(settings.seed * 1_000_003 + self._step_index)
        path = torch.rand(h.shape[0], generator=gen) < 0.5
        path = path.to(h.device).reshape(-1, *([1] * (h.dim() - 1)))
        return torch.where(path, run("path"), run("accumulate"))

    def _write_controls(self, buf: torch.Tensor, values: Any, check: Any) -> None:
        if isinstance(values, Mapping):
            for name, value in values.items():
                buf[self._set_index(name)] = check(value)
        else:
            values = list(values)
            if len(values) != len(self._set_info):
                raise ValueError(f"expected {len(self._set_info)} values, got {len(values)}")
            buf.copy_(torch.tensor([check(v) for v in values], dtype=buf.dtype))

    def _check_priors(self) -> None:
        if float(self.priors.sum()) <= 0:
            raise ValueError("at least one adapter set needs a prior above 0")

    def _set_index(self, name: str) -> int:
        for i, info in enumerate(self._set_info):
            if info["name"] == name:
                return i
        raise KeyError(f"no adapter set named {name!r}; sets are {self.set_names}")

    def extra_repr(self) -> str:
        return (
            f"sets={self.set_names}, targets={len(self.targets)}, depth={self.current_depth}, "
            f"softmax_scope={self.settings.softmax_scope!r}, "
            f"routing={self.settings.routing_scope}/{self.settings.routing_select}"
        )


def _observe_output(target: str, x: torch.Tensor, h: torch.Tensor, out: torch.Tensor) -> None:
    from .recording import active_recorder

    recorder = active_recorder()
    wants = getattr(recorder, "wants_output", None)
    if wants is not None and wants(target):
        recorder.observe_output(target, x, h, out)


class _Hook:
    def __init__(self, adapter: KiLoRA, index: int) -> None:
        self.adapter = adapter
        self.index = index

    def __call__(self, module: nn.Module, args: tuple, output: Any) -> Any:
        if not isinstance(output, torch.Tensor):
            raise KiLoRAError(
                f"target module {self.adapter.targets[self.index]!r} must return a tensor"
            )
        site = self.adapter.sites[self.index]
        first = next(iter(site.tensors.values()), None)
        if (
            first is not None
            and first.device != output.device
            and not self.adapter._library_transfers
        ):
            site.to(output.device)
        return self.adapter._run(self.index, module, args[0], output)


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    return torch.device("cpu")


def _check_shape(target: str, module: nn.Module, ex: KeyedExperts, form: str = "state") -> None:
    n_in = getattr(module, "in_features", None)
    n_out = getattr(module, "out_features", None)
    if form == "state":
        n_in = n_out
    if (n_in is not None and ex.in_features is not None and n_in != ex.in_features) or (
        n_out is not None and n_out != ex.out_features
    ):
        raise KiLoRAError(
            f"{target}: experts have in={ex.in_features}, out={ex.out_features}; "
            f"the layer has in={n_in}, out={n_out}"
        )


def _export(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32).clone()


def attach(
    model: nn.Module,
    config: KiLoRAConfig,
    *,
    name: str = "main",
    keys: dict[str, dict[str, torch.Tensor]] | None = None,
    key_source: str = "random",
) -> KiLoRA:

    targets = find_target_modules(model, config.target_modules)
    modules = dict(model.named_modules())
    shapes = {t: (modules[t].in_features, modules[t].out_features) for t in targets}
    new = new_keyed_set(
        name,
        shapes,
        num_experts=config.num_experts,
        routes=config.routes,
        rank=config.rank,
        router_dim=config.router_dim,
        router_seed=config.router_seed,
        seed=config.seed,
        m_in=config.m_in,
        m_out=config.m_out,
        output_gate=config.output_gate,
        keys=keys,
        key_source=key_source,
        aggregation=config.aggregation,
    )
    return KiLoRA(model, [new], Settings.from_config(config))


class _SectionHook:
    def __init__(self, changes) -> None:
        self.changes = list(changes)

    def __call__(self, module, args, out):
        x = args[0]
        for change in self.changes:
            out = change.add(x, out)
        return out


def compose(
    model: nn.Module,
    sources: Sequence[str | os.PathLike[str] | AdapterSet],
    *,
    sections: Sequence[Any] | None = None,
    family: str | None = None,
    section_model: nn.Module | None = None,
    names: Sequence[str] | None = None,
    strengths: Sequence[float] | None = None,
    priors: Sequence[float] | None = None,
    depth: int | None = None,
    softmax_scope: str | None = None,
    routing_scope: str | None = None,
    routing_select: str | None = None,
    token_depth: torch.Tensor | None = None,
    token_depth_start: float = 0.0,
    token_depth_base: float | None = None,
    mix: str | None = None,
) -> KiLoRA:

    from .modfile import load_mod

    sets: list[AdapterSet] = []
    settings: Settings | None = None
    metadata: dict[str, Any] = {}
    mod_sections: list[tuple[Any, int]] = []
    yielding_schedule = None
    for source in sources:
        if isinstance(source, AdapterSet):
            sets.append(dataclasses.replace(source))
            continue
        mod = load_mod(source)
        if mod.manifest.get("schedule_yields_to_depth") and mod.manifest.get("depth_schedule"):
            yielding_schedule = yielding_schedule or mod.manifest["depth_schedule"]
        mod_sections.extend((x, len(sets)) for x in getattr(mod, "sections", []))
        sets.extend(mod.sets)
        if settings is None:
            settings = mod.settings
            metadata = {k: v for k, v in mod.manifest.items() if k in _CARRIED_FIELDS}
    routed_import = False
    if settings is None:
        if sets and all(s.imported_from is not None for s in sets):
            settings = imported_settings()
            routed_import = mix != "stack"
        else:
            settings = Settings(depth=1)
    if (
        yielding_schedule is not None
        and import_score(settings, form="input") == "derived"
        and derived_mode() == "full"
        and mix != "stack"
    ):
        settings = dataclasses.replace(
            settings,
            depth=derived_default_depth(),
            path_width=3 if settings.path_width == 1 else settings.path_width,
        )
        yielding_schedule = None
    if mix not in (None, "stack"):
        raise ValueError(f"unknown mix {mix!r}; compose takes None or 'stack'")
    if mix == "stack":
        settings = dataclasses.replace(
            settings,
            depth=1,
            gate_mode="none",
            route_freeze=None,
            softmax_scope="per_set",
            import_score="keys",
        )
    if softmax_scope is None and len(sets) > 1:
        independent = any(routes_alone(s.extra) or s.imported_from is not None for s in sets)
        keyed_imports = all(s.form == "input" for s in sets)
        if independent or (keyed_imports and len({s.router_dim for s in sets}) > 1):
            settings.softmax_scope = "per_set"
    for label, values in (("names", names), ("strengths", strengths), ("priors", priors)):
        if values is not None and len(values) != len(sets):
            raise ValueError(f"{label} has {len(values)} entries for {len(sets)} adapter sets")
    for i, s in enumerate(sets):
        if names is not None:
            s.name = names[i]
        if strengths is not None:
            s.strength = check_strength(strengths[i])
        if priors is not None:
            s.prior = check_prior(priors[i])
    if depth is not None:
        settings.depth = check_depth(depth)
    if softmax_scope is not None:
        settings.softmax_scope = softmax_scope
    if routing_scope is not None:
        settings.routing_scope = routing_scope
    if routing_select is not None:
        settings.routing_select = routing_select
    check_routing(settings.routing_scope, settings.routing_select)
    adapter = KiLoRA(model, sets, settings, metadata)
    extra = [(x, strengths[i] if strengths is not None else 1.0) for x, i in mod_sections]
    extra += list(sections or [])
    if extra:
        tree = section_model if section_model is not None else model
        missing = adapter.add_sections(tree, extra, family)
        if missing:
            raise KiLoRAError(
                f"{len(missing)} section layers match no layer of the model, for example "
                f"{missing[0]!r}"
            )
    if token_depth is not None:
        adapter.set_token_depth(token_depth, token_depth_start, token_depth_base)
    if routed_import and depth is None:
        adapter.depth_schedule = imported_schedule()
    elif yielding_schedule is not None and depth is None:
        from .depth_schedule import compile_schedule

        adapter.depth_schedule = compile_schedule(yielding_schedule)
    return adapter


_CARRIED_FIELDS = ("base_model",)


def _exp(t: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and t.requires_grad:
        return t.exp()
    from .compute import _cache_get, _cache_put, _tensor_key

    key = _tensor_key("exp", [t])
    hit = _cache_get(key)
    if hit is None:
        hit = t.exp()
        _cache_put(key, hit)
    return hit
