# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .sets import AdapterSet
from .subspace_keys import build_subspace_keys

log = logging.getLogger("kilora.keys")


class PromptCache:
    def __init__(self, encode: Callable[[str], torch.Tensor]) -> None:
        self.encode = encode
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, prompt: str) -> torch.Tensor:
        if prompt not in self._cache:
            self._cache[prompt] = self.encode(prompt)
        return self._cache[prompt]

    def __len__(self) -> int:
        return len(self._cache)


def _stack_padded(texts: Sequence[torch.Tensor]) -> torch.Tensor:
    longest = max(t.shape[0] for t in texts)
    return torch.stack([torch.nn.functional.pad(t, (0, 0, 0, longest - t.shape[0])) for t in texts])


def fast_state_keys(
    adapter: Any,
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cache: PromptCache,
    prompts: Sequence[str],
    *,
    latent_shape: tuple[int, ...],
    sigmas: Sequence[float] = (0.8, 0.5),
    m_in: int = 4,
    m_out: int = 4,
    seed: int = 0,
    depth: int = 2,
    set_index: int = 0,
) -> dict:
    if not prompts:
        raise ValueError("fast_state_keys needs at least one prompt")
    clock = _Clock()
    texts = [cache.get(p) for p in prompts]
    clock.lap("encode prompts")
    text = _stack_padded([t for _ in sigmas for t in texts])
    sigma = torch.tensor([float(s) for s in sigmas for _ in prompts])
    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn(len(sigma), *latent_shape, generator=gen)
    batch = (latents, sigma, text)
    return state_keys_from_call(
        adapter,
        lambda: run(*batch),
        set_index,
        m_in=m_in,
        m_out=m_out,
        depth=depth,
        clock=clock,
    )


TIMING = os.environ.get("KILORA_KEY_TIMING", "0") == "1"


class _Clock:
    def __init__(self) -> None:
        self.last = time.perf_counter()
        self.phases: dict[str, float] = {}

    def lap(self, name: str) -> None:
        if TIMING and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.phases[name] = self.phases.get(name, 0.0) + now - self.last
        self.last = now

    def report(self) -> None:
        text = ", ".join(f"{k} {v:.2f} s" for k, v in self.phases.items())
        log.log(logging.INFO if TIMING else logging.DEBUG, "key pass: %s", text)


def state_keys_from_call(
    adapter: Any,
    call: Callable[[], Any],
    set_index: int = 0,
    *,
    m_in: int = 4,
    m_out: int = 4,
    depth: int = 2,
    clock: _Clock | None = None,
) -> Keys:
    clock = clock or _Clock()
    todo = getattr(adapter, "_keys_todo", None)
    if todo:
        adapter._keys_todo = [j for j in todo if j != set_index]
    old = (adapter.current_depth, adapter.depth_schedule, adapter.eval_mode)
    freeze = getattr(adapter.settings, "route_freeze", None)
    adapter.set_depth(depth)
    adapter.depth_schedule = None
    adapter.eval_mode = True
    adapter.settings.route_freeze = None
    start_tau = float(adapter.log_taus[f"s{set_index}"].detach())
    from .compute import eager

    eager_block = eager()
    eager_block.__enter__()
    try:
        states = {}
        queries = record_queries(adapter, call, set_index, states=states)
        clock.lap("forward (temperature)")
        current = {t: (site_keys(adapter, t, set_index)) for t in queries}
        scope = adapter.settings.routing_scope
        set_log_tau(adapter, set_index, calibrate_log_tau(queries, current, scope=scope))
        clock.lap("calibrate")
        keys = build_subspace_keys(
            adapter,
            [None],
            lambda _: call(),
            source="states",
            m_in=m_in,
            m_out=m_out,
            set_index=set_index,
            clock=clock,
        )
        new = {t: (k["keys_in"], k["rows_in"]) for t, k in keys.items() if t in queries}
        from .compute import _share_transition_base, share_transition_gamma, share_transition_kind

        if share_transition_gamma() and share_transition_kind() == "weighted":
            for index, (target, site) in enumerate(
                zip(adapter.targets, adapter.sites, strict=True)
            ):
                if target not in states or target not in keys or set_index not in site.keyed:
                    continue
                layer = adapter._layer_sets(index, site, site.tensors[f"s{set_index}_down"].device)
                transition = _share_transition_base(
                    layer[site.keyed.index(set_index)], states=states[target]
                )
                if transition is not None:
                    keys[target]["share_transition_weighted"] = transition.cpu()
        result = Keys(keys)
        result.log_tau = calibrate_log_tau(queries, new, scope=scope)
        clock.lap("calibrate again")
        clock.report()
        return result
    finally:
        eager_block.__exit__(None, None, None)
        set_log_tau(adapter, set_index, start_tau)
        adapter.set_depth(old[0])
        adapter.depth_schedule, adapter.eval_mode = old[1], old[2]
        adapter.settings.route_freeze = freeze


class Keys(dict):
    log_tau: float = 0.0


TARGET_TOP = 0.6


def key_target() -> float:
    value = os.environ.get("KILORA_KEY_TARGET")
    if not value:
        return TARGET_TOP
    target = float(value)
    if not 0.0 < target < 1.0:
        raise ValueError(f"KILORA_KEY_TARGET must be in (0, 1), got {value}")
    return target


def key_settings() -> dict:
    from .subspace_keys import CONTRAST

    return {"target": key_target(), "contrast": CONTRAST}


def record_queries(
    adapter: Any, call: Callable[[], Any], set_index: int = 0, *, states=None
) -> dict:
    from .compute import key_probes

    out = {}
    with torch.no_grad(), key_probes() as record:
        call()
    for target, site in zip(adapter.targets, adapter.sites, strict=True):
        if set_index not in site.keyed or target not in record:
            continue
        pos = site.keyed.index(set_index)
        first = next(
            (e for e in record[target] if e["k"] in (0, 1) and e["queries"][pos] is not None),
            None,
        )
        if first is not None:
            out[target] = first["queries"][pos].float().cpu()
            if states is not None and first.get("state") is not None:
                states[target] = first["state"]
    return out


def site_keys(adapter: Any, target: str, set_index: int = 0) -> tuple:
    index = adapter.targets.index(target)
    site = adapter.sites[index]
    if adapter._set_info[set_index]["form"] == "state":
        from .compute import derived_keys

        layers = adapter._layer_sets(index, site, site.tensors[f"s{set_index}_down"].device)
        basis, rows, *_ = derived_keys(layers[site.keyed.index(set_index)])
        return basis.detach().cpu(), rows.cpu()
    return site.tensors[f"s{set_index}_keys_in"].detach().cpu(), site.rows[set_index][0].cpu()


CALIBRATION_ROWS = 65536


def calibrate_log_tau(
    queries: dict, keys: dict, target_top: float | None = None, scope: str = "token"
) -> float:
    from .compute import ROUTING_SCOPES, norm_energy, orthonormal_keys, sparsemax

    if scope not in ROUTING_SCOPES:
        raise ValueError(f"unknown routing scope {scope!r}")
    target_top = key_target() if target_top is None else target_top
    rows_of_scores = []
    for target, q in queries.items():
        k, rows = keys[target]
        scores = norm_energy(q, orthonormal_keys(k.float(), rows), rows)
        scores = scores.mean(-2) if scope == "tensor" else scores.flatten(0, -2)
        rows_of_scores.append(scores)
    if not rows_of_scores:
        return 0.0
    scores = torch.cat(rows_of_scores, 0)
    if scores.shape[0] > CALIBRATION_ROWS:
        idx = torch.linspace(0, scores.shape[0] - 1, CALIBRATION_ROWS).round().long()
        scores = scores[idx]

    def top(log_tau: float) -> float:
        return float(sparsemax(math.exp(log_tau) * scores).max(-1).values.mean())

    lo, hi = -5.0, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if top(mid) < target_top else (lo, mid)
    chosen = (lo + hi) / 2
    log.info("routing temperature %.3g: mean top weight %.3f", math.exp(chosen), top(chosen))
    return chosen


def set_log_tau(adapter: Any, set_index: int, log_tau: float) -> None:
    with torch.no_grad():
        adapter.log_taus[f"s{set_index}"].fill_(float(log_tau))


def write_keys(adapter_set: AdapterSet, keys: dict, log_tau: float | None = None) -> AdapterSet:
    unknown = sorted(set(keys) - set(adapter_set.experts))
    if unknown:
        raise ValueError(f"keys for layers that the set does not have: {unknown}")
    experts = {}
    for target, ex in adapter_set.experts.items():
        k = keys.get(target)
        if k is None:
            experts[target] = ex
            continue
        experts[target] = dataclasses.replace(
            ex,
            keys_in=k["keys_in"].detach().float().cpu().clone(),
            keys_out=k["keys_out"].detach().float().cpu().clone(),
            rows_in=k["rows_in"].long().cpu().clone(),
            rows_out=k["rows_out"].long().cpu().clone(),
        )
    extra = {} if log_tau is None else {"log_tau": float(log_tau)}
    return dataclasses.replace(adapter_set, experts=experts, key_source="states", **extra)


__all__ = ["PromptCache", "calibrate_log_tau", "fast_state_keys", "write_keys"]
