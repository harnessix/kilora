# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..multi_import import lokr_parts, lycoris_delta


@dataclass(frozen=True)
class ReadFeatures:
    directions: dict[str, torch.Tensor]
    singular_values: dict[str, torch.Tensor]
    summary: torch.Tensor


def _svd_rows(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = min(rank + 8, *matrix.shape)
    if q == min(matrix.shape):
        _, values, directions = torch.linalg.svd(matrix, full_matrices=False)
        return directions[:rank].clone(), values[:rank].clone()
    devices = [matrix.device] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        _, values, right = torch.svd_lowrank(matrix, q=q, niter=2)
    return right[:, :rank].T.contiguous(), values[:rank].clone()


def _full_svd_rows(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, values, directions = torch.linalg.svd(matrix, full_matrices=False)
    return directions[:rank].clone(), values[:rank].clone()


def lokr_directions(
    w1: torch.Tensor, w2: torch.Tensor, rank: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:

    if rank < 1:
        raise ValueError("rank must be positive")
    d1, s1 = _full_svd_rows(w1.float(), rank)
    d2, s2 = _full_svd_rows(w2.float(), rank)
    values = (s1[:, None] * s2[None, :]).flatten()
    order = torch.argsort(values, descending=True, stable=True)[:rank]
    rows = torch.empty((len(order), w1.shape[1] * w2.shape[1]), dtype=d1.dtype, device=d1.device)
    for row, i in zip(rows, order, strict=True):
        torch.outer(d1[i // len(s2)], d2[i % len(s2)], out=row.view(w1.shape[1], w2.shape[1]))
    return rows, values[order]


def _layer_rows(parts: Mapping[str, torch.Tensor], rank: int) -> torch.Tensor:
    if "down" in parts and "up" in parts:
        down, up = parts["down"].float(), parts["up"].float()
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError("LoRA matrices must have shapes [rank, in] and [out, rank]")
        scale = abs(float(parts["alpha"].item()) / down.shape[0]) if "alpha" in parts else 1.0
        return down * (up.norm(dim=0) * scale)[:, None]
    if any(k.startswith("lokr_") for k in parts):
        w1, w2 = lokr_parts(dict(parts))
        directions, values = lokr_directions(w1, w2, rank)
        return directions * values[:, None]
    if any(k.startswith("hada_") for k in parts):
        delta, _ = lycoris_delta(dict(parts))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _, values, right = torch.svd_lowrank(delta.float(), q=min(rank + 8, *delta.shape))
        directions, values = right[:, :rank].T.contiguous(), values[:rank]
        return directions * values[:, None]
    raise ValueError("unrecognized adapter layer")


def extract_content(
    layers: Mapping[str, Mapping[str, torch.Tensor]],
    layer_groups: Mapping[str, str],
    group_widths: Mapping[str, int],
    rank: int = 4,
    group_order: Sequence[str] | None = None,
) -> ReadFeatures:

    if rank < 1:
        raise ValueError("rank must be positive")
    order = list(group_widths) if group_order is None else list(group_order)
    if len(order) != len(group_widths) or set(order) != set(group_widths):
        raise ValueError("group_order must contain every group exactly once")
    grouped: dict[str, list[torch.Tensor]] = {group: [] for group in order}
    lora: dict[str, list[Mapping[str, torch.Tensor]]] = {group: [] for group in order}
    for name in sorted(layers):
        group = layer_groups.get(name)
        if group is None:
            continue
        if group not in grouped:
            raise ValueError(f"unknown group: {group}")
        parts = layers[name]
        if "down" in parts and "up" in parts:
            down, up = parts["down"], parts["up"]
            if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError("LoRA matrices must have shapes [rank, in] and [out, rank]")
            if down.shape[1] != group_widths[group]:
                raise ValueError(f"input width mismatch for group {group}")
            lora[group].append(parts)
        else:
            rows = _layer_rows(parts, rank)
            if rows.ndim != 2 or rows.shape[1] != group_widths[group]:
                raise ValueError(f"input width mismatch for group {group}")
            grouped[group].append(rows)
    directions, values, summary = {}, {}, []
    for group in order:
        width = group_widths[group]
        if width < 1:
            raise ValueError("group widths must be positive")
        if lora[group]:
            batches: dict[tuple, list[Mapping[str, torch.Tensor]]] = {}
            for parts in lora[group]:
                up = parts["up"]
                batches.setdefault((up.shape, up.dtype, up.device), []).append(parts)
            for batch in batches.values():
                weights = torch.stack([parts["up"] for parts in batch]).float().norm(dim=1)
                scales = weights.new_tensor(
                    [
                        abs(float(parts["alpha"].item()) / parts["down"].shape[0])
                        if "alpha" in parts
                        else 1.0
                        for parts in batch
                    ]
                )
                weights.mul_(scales[:, None])
                rows = torch.cat([parts["down"] for parts in batch]).float()
                rows.mul_(weights.reshape(-1, 1))
                grouped[group].append(rows)
        if grouped[group]:
            matrix = grouped[group][0] if len(grouped[group]) == 1 else torch.cat(grouped[group])
            d, s = _svd_rows(matrix, rank)
            signs = torch.sign(d.gather(1, d.abs().argmax(dim=1, keepdim=True)))
            d = d * signs
            summary.append(d[0] * s[0])
        else:
            d, s = torch.empty((0, width)), torch.empty(0)
            summary.append(torch.zeros(width))
        directions[group], values[group] = d, s
    return ReadFeatures(directions, values, torch.cat(summary) if summary else torch.empty(0))
