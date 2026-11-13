# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from .recording import ContextStack

_CACHES = ContextStack("kilora_route_freeze_caches")
RECORDS = ContextStack("kilora_route_freeze_records")
_OFF = ContextStack("kilora_route_freeze_off")
_FOLD_COUNTS = ContextStack("kilora_route_freeze_fold_counts")


@contextlib.contextmanager
def fold_counts(counts: dict) -> Iterator[dict]:
    with _FOLD_COUNTS.pushed(counts):
        yield counts


def note_fold(folded: bool, replay: bool = False) -> None:
    if _FOLD_COUNTS:
        counts = _FOLD_COUNTS[-1]
        name = ("replayed" if replay else "folded") if folded else "recording"
        counts[name] = counts.get(name, 0) + 1


@contextlib.contextmanager
def freezing(cache: dict) -> Iterator[dict]:
    with _CACHES.pushed(cache):
        yield cache


def check_route_freeze(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"route_freeze must be a sampling progress in [0, 1], got {value}")
    return value


def threshold(keyed) -> float | None:
    return keyed[0].route_freeze if keyed else None


def active(keyed, progress: float) -> bool:
    t = threshold(keyed)
    from .compute import _IR_ACTIVE_OVERRIDES
    from .ir.dispatch import peek_override

    override = peek_override("freeze") if _IR_ACTIVE_OVERRIDES.get() else None
    if override is not None and override.variant == "default":
        return t is not None and bool(_CACHES) and not _OFF
    return t is not None and bool(_CACHES) and not _OFF and progress >= t


def note_routes(target: str, gates) -> None:
    for rec in RECORDS:
        rec["routes"].setdefault(target, []).append([g.detach() for g in gates])


def note_gate(target: str, i: int, qa: torch.Tensor, qb: torch.Tensor) -> None:
    for rec in RECORDS:
        rec["gates"][(target, i)] = (qa.detach(), qb.detach())


def frozen(target: str, lead: tuple) -> dict | None:
    entry = _CACHES[-1].get(target) if _CACHES else None
    if entry is None or entry["lead"] != lead:
        return None
    return entry


def run(call, x, y0, keyed, positions, target, regions, depth):
    from .compute import _IR_ACTIVE_OVERRIDES, _import_depth_chain, import_fixed_read
    from .ir.dispatch import current_override, peek_override, same_override

    override = peek_override("freeze") if _IR_ACTIVE_OVERRIDES.get() else None
    lead = tuple(y0.shape[:-1])
    entry = frozen(target, lead)
    if entry is not None and entry.get("_import_fixed_read", False) != import_fixed_read():
        entry = None
    if entry is not None and not same_override(entry.get("_ir_freeze_override"), override):
        entry = None
    if entry is not None:
        if override is not None:
            current_override("freeze", override.variant)
        return output(entry, x, y0, regions, [s.expert_mask for s in keyed])
    rec = {"routes": {}, "gates": {}, "fused": len(keyed) == 1}
    with RECORDS.pushed(rec), _OFF.pushed(True):
        y = call()
    built = build(rec, keyed, positions, target, regions, depth, lead)
    if (
        built is not None
        and depth > 1
        and any(s.form == "input" and _import_depth_chain(s) for s in keyed)
    ):
        routes = rec["routes"].get(target, [])
        keeps = rec.get("input_keeps", {})
        if all((target, k) in keeps for k in range(1, len(routes))):
            built["chain_sets"] = keyed
            built["chain_depth"] = depth
            built["chain_target"] = target
            built["chain_routes"] = routes
            built["chain_keeps"] = [
                [_pack_keep(keeps[(target, k)][i]) for i in range(len(keyed))]
                for k in range(1, len(routes))
            ]
    if built is not None and _CACHES:
        built["_ir_freeze_override"] = override
        built["_import_fixed_read"] = import_fixed_read()
        _CACHES[-1][target] = built
    return y


def _pack_keep(m: torch.Tensor):
    from .compute import expect_masks

    if expect_masks() or m.dtype == torch.bool or not m.is_floating_point() or m.ndim == 0:
        return m
    cols = m.shape[-1]
    bits = (m != 0).reshape(-1, cols).to(torch.uint8)
    pad = (-cols) % 8
    if pad:
        bits = F.pad(bits, (0, pad))
    shifts = torch.arange(8, device=m.device, dtype=torch.uint8)
    packed = (bits.view(bits.shape[0], -1, 8) << shifts).sum(-1, dtype=torch.uint8)
    return (packed, tuple(m.shape), m.dtype)


def _unpack_keep(value) -> torch.Tensor:
    if isinstance(value, tuple):
        packed, shape, dtype = value
        shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
        bits = (packed.unsqueeze(-1) >> shifts) & 1
        cols = shape[-1]
        return bits.reshape(packed.shape[0], -1)[:, :cols].to(dtype).reshape(shape)
    return value


class _FrozenProduct(dict):
    def __getitem__(self, key):
        if key == "up":
            return _product_up(self)
        return super().__getitem__(key)

    def __contains__(self, key):
        return key == "up" or super().__contains__(key)

    def get(self, key, default=None):
        if key == "up" or super().__contains__(key):
            return self[key]
        return default


def _product_up(entry) -> torch.Tensor:

    lead = entry["scale"].shape[:-1]
    parts = []
    for up, qb in zip(entry["ups"], entry["up_gates"], strict=True):
        folded = up.float()
        folded = folded * qb.float() if qb is not None else folded.expand(*lead, *folded.shape)
        parts.append(folded)
    return torch.cat(parts, -2)


def _step_weights(n: int, depth) -> list[float]:
    frac = float(depth) - int(depth) if float(depth) > 1 else 0.0
    w = [1.0] * n
    if frac and n:
        w[-1] = frac
    return w


def build(rec, keyed, positions, target, regions, depth, lead) -> dict | None:
    steps = rec["routes"].get(target)
    if not steps or any(len(g) != len(keyed) for g in steps):
        return None
    from .compute import _IR_ACTIVE_OVERRIDES, _IR_INTERPRET_ALL
    from .ir.dispatch import peek_override

    override = peek_override("freeze") if _IR_ACTIVE_OVERRIDES.get() else None
    if _IR_INTERPRET_ALL.get() or override is not None:
        supported = len(lead) == 2 and all(
            route.ndim == 3 and route.dtype == torch.float32 for step in steps for route in step
        )
        if supported:
            from .compute import current_progress
            from .ir.part_driver import freeze_record_ir

            frozen, entry = freeze_record_ir(
                rec,
                keyed,
                positions,
                target,
                regions,
                depth,
                lead,
                current_progress() if override is not None else 0.0,
            )
            return entry if override is None or override.variant != "default" or frozen else None
        from .fused import _note

        if override is not None:
            raise NotImplementedError("freeze override requires float32 three-dimensional routes")
        _note("IR driver refused freeze: unsupported route layout or dtype")
    scales, ups, up_gates, forms, exact = [], [], [], [], []
    for i, s in enumerate(keyed):
        gates = [g[i].float() for g in steps]
        if s.iteration == "path" or s.aggregation == "replace":
            w = gates[-1]
        else:
            weights = _step_weights(len(gates), depth)
            w = sum(f * g for f, g in zip(weights, gates, strict=True))
        mean = w.mean(-2)
        col = mean.repeat_interleave(s.per_expert, dim=-1)
        scale = s.scale.float() if isinstance(s.scale, torch.Tensor) else float(s.scale)
        col = col * scale
        q = rec["gates"].get((target, i)) or rec["gates"].get((f"{target}#input{positions[i]}", 0))
        if q is not None:
            qa, qb = q
            col = col * qa.float().mean(-2)
            exact.append(
                (s, mean * scale * qa.float().mean(-2), qb) if getattr(s, "exact", None) else None
            )
        else:
            qb = None
            exact.append((s, mean * scale, None) if getattr(s, "exact", None) else None)
        scales.append(col)
        ups.append(s.up)
        up_gates.append(None if qb is None else qb.detach().clone())
        forms.append(s.form)
    return _FrozenProduct(
        {
            "lead": lead,
            "forms": forms,
            "downs": [s.down for s in keyed],
            "expert_masks": [s.expert_mask for s in keyed],
            "scale": torch.cat(scales, -1),
            "ups": ups,
            "up_gates": up_gates,
            "regions": list(regions),
            "exact": exact,
        }
    )


def output(entry, x, y0, regions=None, masks=None) -> torch.Tensor:
    if entry.get("chain_sets") is not None:
        import dataclasses

        from .compute import _replay_import_chain

        layers = entry["chain_sets"]
        if masks is not None:
            layers = [
                dataclasses.replace(s, expert_mask=m) for s, m in zip(layers, masks, strict=True)
            ]
        keeps = [[_unpack_keep(m) for m in step] for step in entry["chain_keeps"]]
        stats = {
            "forced_routes_all": entry["chain_routes"],
            "forced_read_keeps_all": keeps,
        }
        if len(layers) == 1:
            stats["forced_routes"] = [step[0] for step in entry["chain_routes"]]
            stats["forced_read_keeps"] = [step[0] for step in keeps]
        return _replay_import_chain(
            x,
            y0,
            layers,
            list(range(len(layers))),
            entry["chain_depth"],
            entry["regions"] if regions is None else regions,
            entry["chain_target"],
            stats,
        )
    dt = y0.dtype
    reads = []
    regions = entry["regions"] if regions is None else regions
    if masks is None:
        masks = entry.get("expert_masks") or [None] * len(entry["downs"])
    for form, down, r, em in zip(entry["forms"], entry["downs"], regions, masks, strict=True):
        src = y0 if form == "state" else x
        c = F.linear(src.to(dt), down.to(dt)).float()
        if em is not None:
            c = c * em.float()
        if r is not None:
            c = c * r.float()
        reads.append(c)
    c = torch.cat(reads, -1) * entry["scale"].unsqueeze(-2)
    up = _product_up(entry) if "ups" in entry else entry["up"]
    delta = torch.matmul(c, up)
    if any(entry.get("exact", ())):
        import dataclasses

        from .compute import _exact_correction

        for item, read, region, mask in zip(entry["exact"], reads, regions, masks, strict=True):
            if item is None:
                continue
            s, weights, channels = item
            s = dataclasses.replace(s, expert_mask=mask)
            extra = _exact_correction(s, x, read, weights.unsqueeze(-2), region, dt)
            if channels is not None:
                extra = extra * channels.to(dt)
            delta = delta + extra
    return y0 + delta.to(dt)
