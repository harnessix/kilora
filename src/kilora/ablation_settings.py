# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import json
import os
from contextvars import ContextVar

import torch


def _choice(name, choices):
    value = os.environ.get(name)
    if value is not None and value not in choices:
        raise ValueError(f"{name} must be {' or '.join(choices)} or unset, got {value!r}")
    return value


def read_dropout_off():
    return _choice("KILORA_READ_DROPOUT", ("off",)) == "off"


def carry_steps(default):
    value = _choice("KILORA_CARRY_STEPS", ("on", "off"))
    return default if value is None else value == "on"


def gate_coherence(default):
    return False if _choice("KILORA_GATE_COHERENCE", ("off",)) == "off" else default


def force_loop():
    return read_dropout_off() or bool(os.environ.get("KILORA_STEP_STATS"))


_ACTIVE = ContextVar("kilora_ablation_step_stats", default=None)
_TIMESTEP = ContextVar("kilora_ablation_timestep", default=0)


@contextlib.contextmanager
def timestep(value):
    token = _TIMESTEP.set(value.detach() if isinstance(value, torch.Tensor) else value)
    try:
        yield
    finally:
        _TIMESTEP.reset(token)


@contextlib.contextmanager
def suspend_recording():
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def recording_outputs():
    return _ACTIVE.get() is not None


def note_output(target, step, output, base=None):
    record = _ACTIVE.get()
    if record is not None:
        record["outputs"].append(
            (target, step, output.detach(), None if base is None else base.detach())
        )


def note_transfer(transferred, change, base, per_expert):
    record = _ACTIVE.get()
    if record is not None:
        record["transfers"].append(
            (transferred.detach(), change.detach(), base.detach(), per_expert)
        )


@contextlib.contextmanager
def recording(target, base):
    path = os.environ.get("KILORA_STEP_STATS")
    if not path or _ACTIVE.get() is not None:
        yield
        return
    from .compute import recording_read_keep_raw

    record = {"outputs": [], "keep": [], "transfers": []}
    token = _ACTIVE.set(record)

    def observe(name, step, key, probability):
        record["keep"].append((name, step, key, probability.detach()))

    try:
        with recording_read_keep_raw(observe):
            yield
        with torch.no_grad():
            previous = {}
            changes = []
            for name, step, output, initial in record["outputs"]:
                if step == 1:
                    previous.pop(name, None)
                initial = base.detach() if initial is None else initial
                before = previous.get(name, initial).float()
                ratio = (output.float() - before).norm() / before.norm().clamp_min(1e-12)
                changes.append({"target": name, "step": step, "relative_change": ratio.item()})
                previous[name] = output
            histogram = [0] * 10
            keep_steps = []
            for name, step, key, probability in record["keep"]:
                counts = torch.histc(probability.float(), bins=10, min=0, max=1)
                counts = [int(count.item()) for count in counts]
                histogram = [a + b for a, b in zip(histogram, counts, strict=True)]
                keep_steps.append({"target": name, "step": step, "set": key, "histogram": counts})
            peaks, norms = [], []
            for transferred, change, read, per_expert in record["transfers"]:
                shape = (*read.shape[:-1], -1, per_expert)
                sigma = read.float().reshape(shape).square().mean(-1, keepdim=True).sqrt() + 1e-6
                peaks.append((transferred.float().reshape(shape).abs() / sigma).max().item())
                norms.append(change.float().square().sum())
            time = _TIMESTEP.get()
            time = time.reshape(-1)[0].item() if isinstance(time, torch.Tensor) else time
            line = {
                "target": target,
                "sampler_timestep": time,
                "depth_steps": changes,
                "keep_probability_histogram": histogram,
                "keep_probability_steps": keep_steps,
                "max_transferred_over_sigma": max(peaks) if peaks else None,
                "accumulated_change_norm": (
                    torch.stack(norms).sum().sqrt().item() if norms else None
                ),
            }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, allow_nan=False) + "\n")
    finally:
        _ACTIVE.reset(token)
