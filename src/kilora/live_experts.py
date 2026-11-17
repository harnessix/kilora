# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import time

import torch

LIVE_EXPERT_MASS = 0.05
LIVE_EXPERT_TOP = 4
LIVE_EXPERT_HISTORY = 16
LIVE_EXPERT_COPIES = 2

WRITE_MEASURE = "forward"
WRITE_MEASURES = ("forward", "candidates")
FORWARD_DRAWS = 4
FORWARD_SIGMA = 0.05
FORWARD_CHUNK = 8
WRITE_RESOLUTION = 512

LIVE_EXPERT_SETTINGS = {
    "live_expert_mass": LIVE_EXPERT_MASS,
    "live_expert_top": LIVE_EXPERT_TOP,
    "live_expert_history": LIVE_EXPERT_HISTORY,
    "live_expert_copies": LIVE_EXPERT_COPIES,
    "write_measure": WRITE_MEASURE,
    "forward_draws": FORWARD_DRAWS,
    "forward_sigma": FORWARD_SIGMA,
    "forward_chunk": FORWARD_CHUNK,
    "write_resolution": WRITE_RESOLUTION,
    "probe_rows": True,
    "probe_call": "separate",
    "probe_draws": 0,
    "probe_budget": 2,
}

NO_TRAVERSED = "no traversed expert"
NO_MOVES_NOTE = f"live write: no moves ({NO_TRAVERSED})."

_READ_ZERO_KEEP = 1.0


def check_setting(name, value):
    from .errors import KiLoRAError

    if name == "probe_call":
        if value != "separate":
            raise KiLoRAError("probe_call must be separate")
        return value
    if name == "probe_rows":
        if not isinstance(value, bool):
            raise KiLoRAError("probe_rows must be a boolean")
        return value
    if name == "write_measure":
        if value not in WRITE_MEASURES:
            raise KiLoRAError("write_measure must be forward or candidates")
        return str(value)
    if name == "forward_sigma":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KiLoRAError("forward_sigma must be a number above 0 and at most 1")
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise KiLoRAError("forward_sigma must be a number above 0 and at most 1")
        return value
    if name == "live_expert_mass":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KiLoRAError("live_expert_mass must be a number from 0 to 1")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise KiLoRAError("live_expert_mass must be a number from 0 to 1")
        return value
    if name == "write_resolution":
        from .measure_size import check_measure_size

        message = "write_resolution must be 0 (the render resolution) or 256 to 4096 pixels"
        if isinstance(value, bool) or not isinstance(value, int):
            raise KiLoRAError(message)
        if value == 0:
            return 0
        try:
            return check_measure_size(value)
        except ValueError:
            raise KiLoRAError(message) from None
    least = 0 if name in ("live_expert_copies", "probe_draws", "probe_budget") else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise KiLoRAError(f"{name} must be an integer of at least {least}")
    return int(value)


def setting(session, name):
    return getattr(session, name, LIVE_EXPERT_SETTINGS[name])


class RouteRecord:
    def __init__(self):
        self.mass = {}
        self.keeps = {}
        self.ran = set()
        self._counts = {}

    def add_shares(self, target, share):
        share = share.detach().float().reshape(-1, share.shape[-1]).mean(0).cpu()
        count = self._counts.get(target, 0)
        old = self.mass.get(target)
        self.mass[target] = share if old is None else (old * count + share) / (count + 1)
        self._counts[target] = count + 1
        self.ran.add(target)

    def add_mean(self, target, mean, calls):
        mean = mean.detach().float().reshape(-1).cpu()
        count = self._counts.get(target, 0)
        old = self.mass.get(target)
        self.mass[target] = mean if old is None else (old * count + mean * calls) / (count + calls)
        self._counts[target] = count + int(calls)
        self.ran.add(target)

    def add_keeps(self, target, experts, keep):
        if keep.shape[-2] != experts:
            return
        low = keep.detach().float().reshape(-1, experts, keep.shape[-1]).amin(0).cpu()
        old = self.keeps.get(target)
        self.keeps[target] = low if old is None or old.shape != low.shape else old.minimum(low)
        self.ran.add(target)

    @property
    def empty(self):
        return not self.ran


@contextlib.contextmanager
def recording_routes(session, record):
    from .compute import observing_set_share_layers, recording_read_keep_raw

    source = session._source
    set_key = source.extra.get("live_id", source.name)
    names = {session.mod_id, source.name, set_key}
    experts = source.num_experts

    def observe(target, gates, sets):
        for gate, layer in zip(gates, sets or (), strict=False):
            if layer.key == set_key and target in session.state:
                record.add_shares(target, gate)

    def wants(_target, sets):
        return any(layer.key == set_key for layer in sets)

    def keeps(name, _step, key, values):
        hook, sep, _index = name.rpartition("#")
        target = hook if sep else name
        if key in names and target in session.state and values.ndim >= 2:
            record.add_keeps(target, experts, values)

    with observing_set_share_layers(wants, observe), recording_read_keep_raw(keeps):
        yield record


@contextlib.contextmanager
def recording_fused_routes(session, record):
    from .compute import observing_fused_mass
    from .device_pack import DevicePack, nbytes_of

    source = session._source
    set_key = source.extra.get("live_id", source.name)

    sums = {}
    pack = DevicePack()

    def observe(target, gates, sets):
        if target not in session.state:
            return
        for gate, layer in zip(gates, sets or (), strict=False):
            if layer.key == set_key:
                share = gate.detach().float().reshape(-1, gate.shape[-1]).mean(0)
                total = sums.get(target)
                if (
                    total is None
                    or total[0].shape != share.shape
                    or total[0].device != share.device
                ):
                    if not pack.hint:
                        pack.hint = len(session.state) * nbytes_of(share.shape, torch.float32)
                    sums[target] = [pack.put(share), 1]
                else:
                    total[0] += share
                    total[1] += 1

    try:
        with observing_fused_mass(observe):
            yield record
    finally:
        for target, (total, calls) in sums.items():
            record.add_mean(target, (total / calls).cpu(), calls)


def measure_routes(session, record, state, measure=True):
    given = getattr(record, "route_record", None)
    if isinstance(given, RouteRecord) and not given.empty:
        return given
    measure = getattr(record, "measure", None) if measure else None
    if measure is None:
        return None
    from .live_imports import ImportTrial

    routes = RouteRecord()
    with recording_routes(session, routes):
        measure(ImportTrial(session, state, state))
    return routes


def _exact_experts(session, target):
    ex = session._source.experts.get(target)
    found = set(ex.exact) if ex is not None else set()
    found |= {e for t, e in session._section_pairs if t == target}
    return found


def write_target(session, routes, state=None):
    state = session.state if state is None else state
    frac = float(setting(session, "live_expert_mass"))
    top = int(setting(session, "live_expert_top"))
    result = {}
    for target in sorted(state):
        parts = state[target]
        experts = [
            int(key.removeprefix("strength_")) for key in parts if key.startswith("strength_")
        ]
        if not experts:
            continue
        chosen = []
        mass = routes.mass.get(target)
        exact = _exact_experts(session, target)
        if mass is not None and mass.numel():
            total = float(mass.clamp(min=0).sum())
            ranked = sorted(
                (e for e in experts if e < mass.numel() and e not in exact),
                key=lambda e: -float(mass[e]),
            )
            if total > 0:
                chosen = [e for e in ranked if float(mass[e]) > frac * total][:top]
        if target in routes.ran:
            keeps = routes.keeps.get(target)
            for e in sorted(exact & set(experts)):
                silent = keeps is not None and e < keeps.shape[0]
                if silent and not bool((keeps[e] < _READ_ZERO_KEEP).any()):
                    continue
                chosen.append(e)
        if chosen:
            result[target] = sorted(set(chosen))
    return result


def active_rows(session, routes, target, expert, rank):
    ex = session._source.experts[target]
    down = ex.down[expert, 0, :rank].detach().float()
    up = ex.up[expert, 0, :rank].detach().float()
    weights = (down.abs().amax(-1) > 0) & (up.abs().amax(-1) > 0)
    read = weights.clone().cpu()
    keeps = routes.keeps.get(target) if routes is not None else None
    if keeps is not None and expert < keeps.shape[0]:
        row = keeps[expert]
        size = min(rank, int(row.shape[0]))
        recorded = torch.zeros(rank, dtype=torch.bool)
        recorded[:size] = row[:size] < _READ_ZERO_KEEP
        if bool((keeps < _READ_ZERO_KEEP).any()):
            read &= recorded
    return read, weights.cpu()


def restrict_move(move, read, output):
    import dataclasses

    if move.part != "core":
        return move
    direction = move.direction * (output.to(move.direction)[:, None] * read.to(move.direction))
    if not bool(direction.abs().amax() > 0):
        return None
    return dataclasses.replace(move, direction=direction)


def target_moves(session, directions, target, routes):
    kept = []
    masks = {}
    for move in directions:
        experts = target.get(move.target)
        if move.expert < 0:
            kept.append(move)
            continue
        if not experts or move.expert not in experts:
            continue
        key = (move.target, move.expert)
        if move.part == "core":
            rank = int(move.direction.shape[0])
            if key not in masks:
                masks[key] = active_rows(session, routes, move.target, move.expert, rank)
            restricted = restrict_move(move, *masks[key])
            if restricted is not None:
                kept.append(restricted)
        else:
            kept.append(move)
    return kept


def restrict_state(session, before, after, target, routes):
    out = {t: {k: v.clone() for k, v in parts.items()} for t, parts in after.items()}
    for name, parts in out.items():
        experts = set(target.get(name, ()))
        for key in list(parts):
            prefix, sep, number = key.rpartition("_")
            if not sep or prefix not in ("core", "strength") or not number.isdigit():
                continue
            expert = int(number)
            old = before.get(name, {}).get(key)
            if old is None:
                continue
            if expert not in experts:
                parts[key] = old.clone()
            elif prefix == "core":
                read, output = active_rows(session, routes, name, expert, int(old.shape[0]))
                mask = output.to(old)[:, None] * read.to(old)
                parts[key] = old + (parts[key] - old) * mask
    return out


def record_history(session, routes):
    history = list(getattr(session, "route_history", None) or [])
    history.append({t: m.detach().float().cpu().clone() for t, m in routes.mass.items()})
    keep = int(setting(session, "live_expert_history"))
    session.route_history = history[-keep:]


def expert_policy(session, target, routes):
    frac = float(setting(session, "live_expert_mass"))
    history = list(getattr(session, "route_history", None) or [])
    if history and routes is not None and _same_mass(history[-1], routes.mass):
        history = history[:-1]
    result = {}
    for layer, experts in target.items():
        for expert in experts:
            shared = False
            for mass in history:
                row = mass.get(layer)
                if row is None or expert >= row.numel():
                    continue
                total = float(row.clamp(min=0).sum())
                if total > 0 and float(row[expert]) > frac * total:
                    shared = True
                    break
            result[(layer, expert)] = "copy" if shared else "in place"
    return result


def _same_mass(a, b):
    if a.keys() != b.keys():
        return False
    return all(a[k].shape == b[k].shape and torch.equal(a[k], b[k].float().cpu()) for k in a)


def history_tensors(session):
    out = {}
    for i, mass in enumerate(getattr(session, "route_history", None) or []):
        for target, value in mass.items():
            out[f"kilora/routes/{i}/{target}"] = value.clone()
    return out


def history_from_tensors(tensors):
    rows = {}
    for name, value in tensors.items():
        if not name.startswith("kilora/routes/"):
            continue
        index, _, target = name.removeprefix("kilora/routes/").partition("/")
        if index.isdigit() and target:
            rows.setdefault(int(index), {})[target] = value.float().clone()
    return [rows[i] for i in sorted(rows)]


def target_count(target):
    return sum(len(experts) for experts in target.values())


class WriteClock:
    def __init__(self):
        self.start = time.perf_counter()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def line(self, candidates, calls=None, rows=None):
        seconds = time.perf_counter() - self.start
        text = f" Candidates: {candidates}"
        if calls is not None:
            text += f", model calls: {calls}"
        if rows is not None:
            text += f", rows: {rows}"
        text += f", write time {seconds:.2f} s"
        if self.cuda:
            peak = torch.cuda.max_memory_allocated() / 2**30
            text += f", peak GPU memory {peak:.2f} GiB"
        return text + "."


def mask_terms(terms, by_target, target):
    import dataclasses

    masks = {}
    for layer, moves_ in by_target.items():
        experts = set(target.get(layer, ()))
        masks[layer] = torch.tensor(
            [m.expert < 0 or m.expert in experts for m in moves_], dtype=torch.float64
        )
    out = []
    for term in terms:
        mask = masks.get(term.node)
        if mask is None or term.response.shape[-1] != mask.numel():
            out.append(term)
            continue
        out.append(dataclasses.replace(term, response=term.response * mask.to(term.response)))
    return out
