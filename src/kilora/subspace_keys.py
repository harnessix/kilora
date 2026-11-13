# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterable
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .compute import key_probes, orthonormal_keys

log = logging.getLogger(__name__)

CONTRAST = os.environ.get("KILORA_KEY_CONTRAST", "0") == "1"
READ_EPS = 1e-12

SOURCES = ("gradients", "states")
PARTS = ("keys_in", "keys_out", "rows_in", "rows_out")
_META = "kilora.subspace_keys"


def build_subspace_keys(
    adapter: Any,
    batches: Iterable[Any],
    loss_fn: Callable[[Any], torch.Tensor],
    *,
    source: str = "gradients",
    m_in: int = 4,
    m_out: int = 4,
    set_index: int = 0,
    clock: Any = None,
    contrast: bool | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    i = set_index
    if adapter._set_info[i]["form"] == "state":
        state_set = adapter.to_sets()[i]
        return {
            target: {name: getattr(expert, name).detach().cpu() for name in PARTS}
            for target, expert in state_set.experts.items()
        }
    sums: dict[tuple[str, str], torch.Tensor] = {}
    routers = {}
    for index, (target, site) in enumerate(zip(adapter.targets, adapter.sites, strict=True)):
        if i in site.keyed:
            pos = site.keyed.index(i)
            routers[target] = (pos, adapter.routers[f"s{i}_{adapter._out[index]}"].detach())
    old = adapter.settings.iteration
    adapter.settings.iteration = "accumulate"
    grads = [(p, p.requires_grad) for p in adapter.parameters()]
    for p, _ in grads:
        p.requires_grad_(False)
    try:
        for batch in batches:
            with key_probes() as record:
                if source == "states":
                    with torch.no_grad():
                        loss_fn(batch)
                else:
                    loss_fn(batch).backward()
            for target, entries in record.items():
                if target not in routers:
                    continue
                pos, _ = routers[target]
                for n, entry in enumerate(entries):
                    if entry["k"] == "output":
                        continue
                    if source == "states":
                        s = entry["queries"][pos]
                    else:
                        out = next((e for e in entries[n:] if e["k"] == "output"), None)
                        if out is None or out["grads"] is None:
                            continue
                        s = out["grads"][pos]
                    if s is None:
                        continue
                    _add_entry(sums, target, entry, pos, s)
            for p in adapter.parameters():
                p.grad = None
    finally:
        adapter.settings.iteration = old
        for p, flag in grads:
            p.requires_grad_(flag)
    if clock is not None:
        clock.lap("forward (states)")
    out: dict[str, dict[str, torch.Tensor]] = {}
    for target, site in zip(adapter.targets, adapter.sites, strict=True):
        if target not in routers:
            continue
        rows_in, rows_out = site.rows[i]
        result = {}
        for part, m, rows in (("keys_in", m_in, rows_in), ("keys_out", m_out, rows_out)):
            now = orthonormal_keys(site.tensors[f"s{i}_{part}"].detach().cpu(), rows.cpu())
            experts, _, d = now.shape
            keys = _start_keys(now, rows, m)
            moment = sums.get((target, part))
            if moment is None and part == "keys_out":
                log.info("%s: no output-key data (depth 1), using the input keys", target)
                result["keys_out"] = _start_keys(result["keys_in"], result["rows_in"], m)
                result["rows_out"] = torch.full_like(result["rows_in"], m)
                continue
            if moment is not None:
                keys = _top_keys(moment, m, keys, contrast)
            result[part] = keys
            result["rows_" + part[5:]] = torch.full((experts,), m, dtype=torch.long)
        out[target] = result
    if clock is not None:
        clock.lap("solve")
    return out


def keys_from_records(
    record: dict[str, list[dict]],
    positions: dict[str, int],
    current: dict[str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]],
    m_in: int = 4,
    m_out: int = 4,
    contrast: bool | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    sums: dict[tuple[str, str], torch.Tensor] = {}
    for target, entries in record.items():
        pos = positions.get(target)
        if pos is None:
            continue
        for entry in entries:
            if entry.get("k") == "output" or entry.get("queries") is None:
                continue
            q = entry["queries"][pos]
            if q is None:
                continue
            _add_entry(sums, target, entry, pos, q)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for target, (kin, kout, rows_in, rows_out) in current.items():
        if target not in positions:
            continue
        result: dict[str, torch.Tensor] = {}
        for part, m, keys0, rows in (
            ("keys_in", m_in, kin, rows_in),
            ("keys_out", m_out, kin if kout is None else kout, rows_out),
        ):
            moment = sums.get((target, part))
            if moment is None and part == "keys_out":
                log.info("%s: no output-key data (depth 1), using the input keys", target)
                result["keys_out"] = _start_keys(result["keys_in"], result["rows_in"], m)
                result["rows_out"] = torch.full_like(result["rows_in"], m)
                continue
            now = orthonormal_keys(keys0.detach().float().cpu(), rows.cpu())
            experts, _, d = now.shape
            keys = _start_keys(now, rows, m)
            if moment is not None:
                keys = _top_keys(moment, m, keys, contrast)
            result[part] = keys
            result["rows_" + part[5:]] = torch.full((experts,), m, dtype=torch.long)
        out[target] = result
    return out


def _start_keys(now: torch.Tensor, rows: torch.Tensor, m: int) -> torch.Tensor:
    import zlib

    experts, have, d = now.shape
    if have == m and bool((rows == m).all()):
        return now.clone()
    out = []
    for e in range(experts):
        seed = zlib.crc32(now[e].contiguous().numpy().tobytes())
        fill = torch.randn(m, d, generator=torch.Generator().manual_seed(seed))
        n = min(m, have)
        fill[:n] = now[e, :n]
        out.append(fill)
    return orthonormal_keys(torch.stack(out))


def _add_entry(sums: dict, target: str, entry: dict, pos: int, signal: torch.Tensor) -> None:
    mask = _mask_at(entry, pos)
    gates, prev, reads = entry.get("gates"), entry.get("prev"), entry.get("reads")
    energy = None if reads is None else reads[pos]
    content = None
    if energy is not None:
        content = _weighted(energy / (energy.sum(-1, keepdim=True) + READ_EPS), mask)
    if gates is not None and gates[pos] is not None:
        w = content if content is not None else _weighted(gates[pos], mask)
        _add(sums, (target, "keys_in"), signal, w)
    if prev is not None and prev[pos] is not None:
        w = content if content is not None else _weighted(prev[pos], mask)
        _add(sums, (target, "keys_out"), signal, w)


def _top_keys(moment: torch.Tensor, m: int, keys: torch.Tensor, contrast: bool | None):
    has = (moment.diagonal(dim1=-2, dim2=-1).sum(-1) > 0).cpu()
    contrast = CONTRAST if contrast is None else contrast
    if contrast and moment.shape[0] > 1:
        moment = moment - moment[has.to(moment.device)].mean(0, keepdim=True)
    _, vectors = torch.linalg.eigh(moment.double())
    top = vectors[..., -m:].flip(-1).transpose(-1, -2).float().cpu()
    return torch.where(has[:, None, None], top, keys)


def _mask_at(entry: dict, pos: int) -> torch.Tensor | None:
    masks = entry.get("masks")
    return None if masks is None else masks[pos]


def _weighted(weights: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return weights if mask is None else weights * mask.to(weights.device, weights.dtype)


def _add(sums: dict, key: tuple[str, str], signal: torch.Tensor, weights: torch.Tensor) -> None:
    s = signal.detach().float().reshape(-1, signal.shape[-1])
    w = weights.detach().float().expand(*signal.shape[:-1], -1).reshape(-1, weights.shape[-1])
    moment = torch.einsum("ne,nr,ns->ers", w.to(s.device), s, s)
    sums[key] = sums[key] + moment if key in sums else moment


def save_subspace_keys(path, keys: dict[str, dict[str, torch.Tensor]], source: str) -> None:
    tensors = {f"{t}/{part}": v[part].contiguous() for t, v in keys.items() for part in PARTS}
    meta = {_META: json.dumps({"source": source, "targets": sorted(keys)})}
    save_file(tensors, str(path), metadata=meta)


def load_subspace_keys(path) -> tuple[dict[str, dict[str, torch.Tensor]], str]:
    with safe_open(str(path), "pt") as f:
        meta = json.loads((f.metadata() or {}).get(_META, "{}"))
    if meta.get("source") not in SOURCES:
        raise ValueError(f"{path}: not a key file")
    flat = load_file(str(path))
    return {t: {p: flat[f"{t}/{p}"] for p in PARTS} for t in meta["targets"]}, meta["source"]


def subspace_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    pa, pb = a.t() @ a, b.t() @ b
    return float((pa - pb).norm() / math.sqrt(2 * a.shape[0]))
