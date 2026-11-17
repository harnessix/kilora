# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import math

import torch

from .device_pack import DevicePack

CACHE_DEVICE = "cpu"


def write_seed(session) -> int:
    count = len(getattr(session, "commits", ()) or ())
    digest = hashlib.sha256(f"{session.mod_id}/{count}".encode()).digest()
    return int.from_bytes(digest[:7], "big")


def _noise_seed(seed, draw, layer, sample, ordinal) -> int:
    mixed = seed * 1_000_003 + draw * 7919 + layer * 104_729 + sample * 31_337 + ordinal * 613
    return mixed % (2**63 - 1)


class ForwardEstimate:
    set_key = None
    replay_all = False

    def __init__(self, session, state, *, draws, sigma, chunk, seed):
        self.session = session
        self.state = state
        self.draws = int(draws)
        self.sigma = float(sigma)
        self.chunk = max(int(chunk), 1)
        self.splits = 0
        self.seed = int(seed)
        self.du = {}
        self.fixed_changes = {}
        source = session._source
        self.index = {target: i for i, target in enumerate(sorted(source.experts))}
        self.cores = {}
        for target, ex in source.experts.items():
            parts = state.get(target) or {}
            found = [
                e
                for e in range(int(ex.up.shape[0]))
                if f"core_{e}" in parts and f"strength_{e}" in parts
            ]
            if found:
                self.cores[target] = found
        self.packs = {kind: DevicePack(n) for kind, n in self._hints(source).items()}
        self.mode = "clean"
        self.copies = 1
        self.draw0 = 0
        self.reads = {}
        self._squares = {}
        self.target = {}
        self.noise = {}
        self.mass = {}
        self.core_sums = {}
        self.strength_sums = {}
        self.losses = []
        self.clean_loss = None
        self.forwards = 0
        self._sample = 0
        self._samples = [0]
        self._ordinal = {}
        self.seen = set()
        self._live_rows = {}
        self._device_parts = {}
        self._device_sums = {}
        self._rows = None
        self._pending = []
        self.accept_before = None
        self.clean_calls = 0
        self.draw_calls = 0

    def _hints(self, source):
        from .device_pack import nbytes_of

        f32 = torch.float32
        parts = squares = live = sums = 0
        for target, experts in self.cores.items():
            ex = source.experts[target]
            width_in, width_out = int(ex.down.shape[-1]), int(ex.up.shape[-1])
            for e in experts:
                rank = int(self.state[target][f"core_{e}"].shape[0])
                parts += nbytes_of((rank, rank), f32) + nbytes_of((rank, width_in), f32)
                parts += nbytes_of((rank, width_out), f32)
                squares += 4 * nbytes_of((2,), f32)
                live += nbytes_of((rank,), torch.bool)
                sums += nbytes_of((self.draws, rank, rank), f32)
                sums += nbytes_of((self.draws,), torch.float64)
        return {"parts": parts, "squares": squares, "live": live, "sums": sums, "pending": 0}

    def delta(self, target, h):
        return None

    def begin_pass(self):
        self._sample = 0
        self._samples = [0]
        self._ordinal = {}

    def begin_sample(self, index):
        self.begin_samples([index])

    def begin_samples(self, indices):
        self._samples = [int(i) for i in indices] or [0]
        self._sample = self._samples[0]
        self._ordinal = {}

    def begin_rows(self, items):
        self._rows = [(int(i), tag) for i, tag in items]
        self._samples = [i for i, _tag in self._rows] or [0]
        self._sample = self._samples[0]
        self._ordinal = {}
        self._pending = []
        self.packs["pending"].clear()

    def end_rows(self):
        for key, draw, core, strength in self._pending:
            target, expert = key
            core_sums, strength_sums = self._sums_on(target, expert, core.device)
            core_sums[draw] += core
            strength_sums[draw] += strength.to(strength_sums.dtype)
        self._pending = []
        self.packs["pending"].clear()

    def drop_rows(self):
        self._pending = []
        self.packs["pending"].clear()

    def _rows_delta(self, target, x):
        rows = self._rows
        if target in self.session._source.experts and any(t == "clean" for _i, t in rows):
            self.seen.add(target)
        experts = self.cores.get(target)
        if not experts:
            return None
        ex = self.session._source.experts[target]
        groups = len(rows)
        if x.shape[-1] != int(ex.down.shape[-1]) or x.shape[0] % groups:
            return None
        ordinal = self._ordinal.get(target, 0)
        self._ordinal[target] = ordinal + 1
        tokens = x.reshape(groups, -1, x.shape[-1])
        for j, (sample, tag) in enumerate(rows):
            if tag == "clean":
                self._clean(target, (target, sample, ordinal), tokens[j], experts)
        draws = [
            (j, sample, tag)
            for j, (sample, tag) in enumerate(rows)
            if isinstance(tag, int) and not isinstance(tag, bool)
        ]
        sigma = self.noise.get(target, 0.0)
        if not draws or target not in self.target or not sigma:
            return None
        width = int(ex.up.shape[-1])
        layer = self.index[target]
        device = x.device
        count = int(tokens.shape[1])
        out = torch.zeros((groups, count, width), device=device)
        pending = self.packs["pending"]
        if not pending.hint:
            from .device_pack import nbytes_of

            one = sum(
                nbytes_of((int(self.state[t][f"core_{e}"].shape[0]),) * 2, torch.float32)
                + nbytes_of((), torch.float32)
                for t, experts in self.target.items()
                for e in experts
            )
            pending.hint = one * max(len(draws), 1)
        for j, sample, draw in draws:
            gen = torch.Generator(device=device)
            paired = getattr(self, "antithetic", False)
            noise_draw = draw // 2 if paired else draw
            gen.manual_seed(_noise_seed(self.seed, noise_draw, layer, sample, ordinal))
            eps = torch.randn((count, width), generator=gen, device=device)
            if paired and draw % 2:
                eps = -eps
            out[j] = eps * sigma
            reads = self.reads.get((target, sample, ordinal))
            if reads is None:
                continue
            for e in self.target[target]:
                c = reads.get(e)
                if c is None:
                    continue
                core, _strength, _down, up = self._parts_on(target, e, device)
                c = c.to(device, non_blocking=True)
                u = eps @ up.T
                self._pending.append(
                    (
                        (target, e),
                        draw,
                        pending.put(u.T @ c),
                        pending.put((u * (c @ core.T)).sum()),
                    )
                )
        return out.reshape(*x.shape[:-1], width).to(x.dtype)

    def _parts_on(self, target, expert, device):
        key = (target, expert, str(device))
        parts = self._device_parts.get(key)
        if parts is None:
            core, strength, down, up = self._parts(target, expert)
            pack = self.packs["parts"]
            parts = (
                pack.put(core, device=device),
                strength,
                pack.put(down, device=device),
                pack.put(up, device=device),
            )
            self._device_parts[key] = parts
        return parts

    def _parts(self, target, expert):
        ex = self.session._source.experts[target]
        parts = self.state[target]
        core = parts[f"core_{expert}"].detach().float()
        rank = int(core.shape[0])
        strength = float(parts[f"strength_{expert}"])
        down = ex.down[expert, 0, :rank].detach().float()
        up = ex.up[expert, 0, :rank].detach().float()
        return core, strength, down, up

    @torch.no_grad()
    def input_delta(self, target, x):
        if self._rows is not None:
            return self._rows_delta(target, x)
        if self.mode == "clean" and target in self.session._source.experts:
            self.seen.add(target)
        experts = self.cores.get(target)
        if not experts:
            return None
        if self.mode != "clean" and target not in self.target:
            return None
        ex = self.session._source.experts[target]
        groups = len(self._samples) if self.mode == "clean" else self.copies
        if x.shape[-1] != int(ex.down.shape[-1]) or x.shape[0] % groups:
            return None
        ordinal = self._ordinal.get(target, 0)
        self._ordinal[target] = ordinal + 1
        tokens = x.reshape(groups, -1, x.shape[-1])
        if self.mode == "clean":
            for j, sample in enumerate(self._samples):
                self._clean(target, (target, sample, ordinal), tokens[j], experts)
            return None
        return self._draw(target, (target, self._sample, ordinal), x, tokens.shape[1])

    def _clean(self, target, key, x, experts):
        reads = {}
        for e in experts:
            core, strength, down, up = self._parts_on(target, e, x.device)
            c = x.float() @ down.T
            y = (c @ core.T) @ up * strength
            entries = self._squares.setdefault((target, e), {})
            old = entries.get(key)
            sums = torch.stack([y.square().sum(), (c @ up).square().sum()])
            entries[key] = (
                self.packs["squares"].put(sums, None if old is None else old[0]),
                int(y.numel()),
            )
            live = (c != 0).any(0)
            old = self._live_rows.get((target, e))
            if old is None or old.shape != live.shape or old.device != live.device:
                self._live_rows[(target, e)] = self.packs["live"].put(live)
            else:
                old |= live
            reads[e] = c.to(CACHE_DEVICE, non_blocking=c.is_cuda)
        self.reads[key] = reads

    def _draw(self, target, key, x, tokens):
        width = int(self.session._source.experts[target].up.shape[-1])
        sigma = self.noise.get(target, 0.0)
        if not sigma:
            return None
        reads = self.reads.get(key)
        layer = self.index[target]
        device = x.device
        noises = []
        for j in range(self.copies):
            gen = torch.Generator(device=device)
            gen.manual_seed(_noise_seed(self.seed, self.draw0 + j, layer, self._sample, key[2]))
            noises.append(torch.randn((tokens, width), generator=gen, device=device))
        eps = torch.stack(noises)
        if reads is not None:
            draws = slice(self.draw0, self.draw0 + self.copies)
            for e in self.target[target]:
                c = reads.get(e)
                if c is None:
                    continue
                core, _strength, _down, up = self._parts_on(target, e, device)
                c = c.to(device, non_blocking=True)
                u = eps @ up.T
                core_sums, strength_sums = self._sums_on(target, e, device)
                core_sums[draws] += torch.einsum("ktr,ts->krs", u, c)
                strength_sums[draws] += (u * (c @ core.T)).sum(dim=(1, 2)).to(strength_sums.dtype)
        return (eps * sigma).reshape(*x.shape[:-1], width).to(x.dtype)

    def _sums_on(self, target, expert, device):
        sums = self._device_sums.get((target, expert))
        if sums is None:
            rank = int(self.state[target][f"core_{expert}"].shape[0])
            wide = torch.float32 if torch.device(device).type == "mps" else torch.float64
            pack = self.packs["sums"]
            sums = (
                pack.zeros((self.draws, rank, rank), torch.float32, device),
                pack.zeros((self.draws,), wide, device),
            )
            self._device_sums[(target, expert)] = sums
        return sums

    def _drop_sums(self, first, copies):
        for core, strength in self._device_sums.values():
            core[first : first + copies] = 0
            strength[first : first + copies] = 0

    def _collect_sums(self):
        for key, (core, strength) in self._device_sums.items():
            self.core_sums[key] = core.cpu()
            self.strength_sums[key] = [float(v) for v in strength.double().cpu()]
        self._device_sums = {}
        self.packs["sums"].clear()

    def set_target(self, target, routes=None):
        chosen = {}
        for layer, experts in self.cores.items():
            keep = experts if target is None else [e for e in experts if e in target.get(layer, ())]
            if keep:
                chosen[layer] = keep
        self.target = chosen
        for layer, experts in chosen.items():
            mass = routes.mass.get(layer) if routes is not None else None
            total = float(mass.clamp(min=0).sum()) if mass is not None and mass.numel() else 0.0
            power = 0.0
            for e in experts:
                g = 1.0
                if total > 0 and e < mass.numel():
                    g = float(mass[e]) / total
                self.mass[(layer, e)] = g
                parts = list((self._squares.get((layer, e)) or {}).values())
                count = max(sum(int(p[1]) for p in parts), 1)
                sums = torch.stack([p[0] for p in parts]).sum(0).cpu() if parts else torch.zeros(2)
                value = float(sums[0]) if float(sums[0]) else float(sums[1])
                power += (g * self.sigma) ** 2 * value / count
                rank = int(self.state[layer][f"core_{e}"].shape[0])
                self.core_sums[(layer, e)] = torch.zeros(self.draws, rank, rank)
                self.strength_sums[(layer, e)] = [0.0] * self.draws
            self.noise[layer] = math.sqrt(power)

    def batches(self):
        return [(s, min(self.chunk, self.draws - s)) for s in range(0, self.draws, self.chunk)]

    def run_clean(self, measure):
        self.mode, self.copies, self.draw0 = "clean", 1, 0
        self.begin_pass()
        rows = _rows_of(measure)
        if rows is not None:
            return self._run_clean_rows(measure, rows)
        batched = getattr(measure, "batched", None)
        with profile_phase(measure, "clean"):
            losses = [batched(self)] if callable(batched) else run_copies(measure, self, 1)
            _wait(self.reads)
        self.forwards += 1
        self.clean_loss = float(losses[0])
        return self.clean_loss

    def _run_clean_rows(self, measure, rows):
        count = int(getattr(measure, "sample_count", 1) or 1)
        held = int(getattr(measure, "accept_count", 0) or 0)
        layout = [(s, "clean") for s in range(count)]
        layout += [(count + a, "plain") for a in range(held)]
        calls0 = _calls(measure)
        try:
            with profile_phase(measure, "clean"):
                losses = rows(self, layout)
                _wait(self.reads)
        finally:
            self._rows = None
        self.clean_calls = _calls(measure) - calls0
        self.forwards += 1
        self.clean_loss = _mean(losses[:count])
        self.accept_before = _mean(losses[count:]) if held else None
        return self.clean_loss

    def _run_draws_rows(self, measure, rows):
        self.mode = "noise"
        count = int(getattr(measure, "sample_count", 1) or 1)
        layout = [(s, k) for k in range(self.draws) for s in range(count)]
        calls0 = _calls(measure)
        try:
            with profile_phase(measure, "draw"):
                losses = rows(self, layout)
        finally:
            self._rows = None
        self.draw_calls = _calls(measure) - calls0
        self.forwards += 1
        self.losses = [_mean(losses[k * count : (k + 1) * count]) for k in range(self.draws)]
        self._collect_sums()
        self.reads = {}
        return all(math.isfinite(v) for v in [self.clean_loss, *self.losses])

    def add_keeps(self, routes):
        source = self.session._source
        for target in sorted(self.seen):
            routes.ran.add(target)
            ex = source.experts[target]
            experts, rows = int(ex.down.shape[0]), int(ex.down.shape[2])
            keep = torch.full((experts, rows), 0.5)
            known = False
            for e in self.cores.get(target, ()):
                live = self._live_rows.get((target, e))
                if live is None:
                    continue
                live = live.cpu()
                keep[e, : live.numel()] = torch.where(live, 0.5, 1.0)
                known = True
            if known:
                routes.add_keeps(target, experts, keep)
        self._live_rows = {}
        self.packs["live"].clear()

    def run_draws(self, measure):
        rows = _rows_of(measure)
        if rows is not None:
            return self._run_draws_rows(measure, rows)
        self.mode = "noise"
        self.losses = []
        first = 0
        while first < self.draws:
            copies = min(self.chunk, self.draws - first)
            self.copies, self.draw0 = copies, first
            self.begin_pass()
            try:
                with profile_phase(measure, "draw"):
                    losses = run_copies(measure, self, copies)
            except out_of_memory():
                if copies == 1:
                    raise
                self._drop_sums(first, copies)
                self.chunk = max(copies // 2, 1)
                self.splits += 1
                free_cached()
                continue
            self.forwards += 1
            self.losses.extend(float(v) for v in losses)
            first += copies
        self._collect_sums()
        self.reads = {}
        return all(math.isfinite(v) for v in [self.clean_loss, *self.losses])

    def gradients(self):
        retained = getattr(self, "mean_gradients", None)
        if retained is not None:
            return retained
        k = len(self.losses)
        if not k:
            return {}
        reward = torch.tensor([self.clean_loss - v for v in self.losses], dtype=torch.float64)
        reward = reward - reward.mean()
        out = {}
        for (layer, e), sums in self.core_sums.items():
            sigma = self.noise.get(layer, 0.0)
            if not sigma:
                continue
            scale = -1.0 / (k * sigma)
            g = self.mass.get((layer, e), 1.0)
            strength = float(self.state[layer][f"strength_{e}"])
            core = scale * g * strength * torch.einsum("k,kij->ij", reward, sums[:k].double())
            sums_s = torch.tensor(self.strength_sums[(layer, e)][:k], dtype=torch.float64)
            amount = scale * g * float((reward * sums_s).sum())
            out[(layer, e)] = (core, amount)
        return out

    def slopes(self, directions):
        grads = self.gradients()
        out = {}
        for move in directions:
            grad = grads.get((move.target, move.expert)) if move.expert >= 0 else None
            part = getattr(move, "part", "")
            if grad is None or part not in ("core", "strength"):
                out[move.id] = 0.0
                continue
            if part == "core":
                out[move.id] = float((grad[0] * move.direction.double().cpu()).sum())
            else:
                out[move.id] = grad[1] * float(move.direction.double().sum())
        return out


def _rows_of(measure):
    try:
        rows = getattr(measure, "rows", None)
    except AttributeError:
        return None
    return rows if callable(rows) else None


def _calls(measure) -> int:
    calls = getattr(measure, "calls", None)
    return int(calls[0]) if isinstance(calls, list) and calls else 0


def _mean(values) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def out_of_memory() -> tuple[type[BaseException], ...]:
    found = [getattr(torch, "OutOfMemoryError", None)]
    found.append(getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None))
    return tuple({t for t in found if isinstance(t, type)}) or (MemoryError,)


def free_cached() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_phase(measure, name):
    import contextlib

    phase = getattr(getattr(measure, "profile", None), "phase", None)
    return phase(name) if callable(phase) else contextlib.nullcontext()


def _wait(reads):
    if torch.cuda.is_available() and reads:
        torch.cuda.synchronize()


def run_copies(measure, trial, copies):
    batched = getattr(measure, "copies", None)
    if batched is not None:
        return list(batched(trial, copies))
    out = []
    first = trial.draw0
    for j in range(copies):
        trial.copies, trial.draw0 = 1, first + j
        trial.begin_pass()
        out.append(float(measure(trial)))
    trial.copies, trial.draw0 = copies, first
    return out


def supports(measure) -> bool:
    return callable(getattr(measure, "copies", None))
