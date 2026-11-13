# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import functools
import zlib
from pathlib import Path

import torch

from . import adapter, bake, compute


@contextlib.contextmanager
def capture(dir, steps=(0, 15), tokens=1024):

    if tokens <= 0 or any(step < 0 for step in steps):
        raise ValueError("tokens must be positive and steps must be nonnegative")
    path = Path(dir)
    path.mkdir(parents=True, exist_ok=True)
    wanted = set(steps)
    records = []
    seen = set()
    current_key = None
    step = -1
    active = False
    original = compute.run_layer
    aliases = [(module, module.run_layer) for module in (adapter, bake)]

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal current_key, step, active
        if active:
            return original(*args, **kwargs)
        active = True
        try:
            key = compute._DROPOUT_KEY[-1][1]
            if step < 0 or key != current_key:
                current_key = key
                step += 1
            x = args[0] if args else kwargs["x"]
            y0 = args[1] if len(args) > 1 else kwargs["y0"]
            sets = args[2] if len(args) > 2 else kwargs["sets"]
            depth = args[3] if len(args) > 3 else kwargs["depth"]
            target = kwargs.get("target", "")
            identity = (step, target)
            if step in wanted and identity not in seen:
                flat_x = x.detach().reshape(-1, x.shape[-1])
                flat_y = y0.detach().reshape(-1, y0.shape[-1])
                if flat_x.shape[0] != flat_y.shape[0]:
                    raise ValueError("input and output token counts differ")
                gen = torch.Generator(device="cpu").manual_seed(
                    zlib.crc32(f"{step}:{target}".encode())
                )
                indices = torch.randperm(flat_x.shape[0], generator=gen)[:tokens]
                record = {
                    "target": target,
                    "step": step,
                    "x": flat_x.index_select(0, indices.to(flat_x.device)).float().cpu(),
                    "y0": flat_y.index_select(0, indices.to(flat_y.device)).float().cpu(),
                    "sets": sets,
                    "depth": depth,
                    "progress": float(
                        kwargs.get("t")
                        if kwargs.get("t") is not None
                        else compute.current_progress()
                    ),
                    "dropout_key": compute._DROPOUT_KEY[-1],
                    "has_regions": kwargs.get("regions") is not None
                    or kwargs.get("expert_regions") is not None,
                    "has_schedule": kwargs.get("schedule") is not None,
                }
                seen.add(identity)
                records.append(record)
                torch.save(
                    {k: v for k, v in record.items() if k != "sets"},
                    path / f"capture_{len(records) - 1:05d}.pt",
                )
            return original(*args, **kwargs)
        finally:
            active = False

    compute.run_layer = wrapped
    for module, fn in aliases:
        if fn is original:
            module.run_layer = wrapped
    try:
        yield records
    finally:
        compute.run_layer = original
        for module, fn in aliases:
            module.run_layer = fn
