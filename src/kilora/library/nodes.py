# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from pathlib import Path

import torch

from ..compute import LayerSet
from ..families import anima_never_target_comfy, anima_never_target_diffusers
from .basis import LibraryBasis
from .mounts import basis_mount


class DirectionAdapters(list):
    def __init__(self):
        super().__init__()
        self.skipped_count = 0
        self.skipped_names = []
        self.notes = []

    def report_unused(self, mounts, used):
        never = []
        unmatched = []
        for name in mounts:
            if name in used:
                continue
            if anima_never_target_diffusers(name) or anima_never_target_comfy(name):
                never.append(name)
            else:
                unmatched.append(name)
        self.skipped_count = len(never)
        self.skipped_names = never[:3]
        for names, reason in ((never, "never-target"), (unmatched, "unmatched")):
            if names:
                sample = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
                self.notes.append(f"skipped {len(names)} {reason} mounts ({sample})")


def projected_core(basis: LibraryBasis, mount: str) -> torch.Tensor:

    stored = basis_mount(basis.mounts, mount)
    if stored is None:
        raise KeyError(mount)
    mount = stored
    b = basis.mounts[mount]
    rank = min(b.u.shape[1], b.v.shape[1])
    core = b.u.new_zeros((rank, rank))
    found = False
    for parts in basis._parts.values():
        if mount not in parts:
            continue
        left, right = parts[mount]
        left = left.to(device=b.u.device, dtype=b.u.dtype)
        right = right.to(device=b.v.device, dtype=b.v.dtype)
        core = core + (b.u[:, :rank].T @ left) @ (right.T @ b.v[:, :rank])
        found = True
    if not found:
        raise ValueError(f"{mount}: no stored library directions")
    return core


def direction_set(
    basis: LibraryBasis,
    mount: str,
    *,
    chunk: int = 1,
    strength: float = 1.0,
    fine: bool = False,
    candidates: int = 16,
) -> LayerSet:

    if chunk < 1:
        raise ValueError("chunk must be positive")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
        raise ValueError("candidate count must be a positive integer")
    stored = basis_mount(basis.mounts, mount)
    if stored is None:
        raise KeyError(mount)
    b = basis.mounts[stored]
    rank = min(b.u.shape[1], b.v.shape[1], b.output_values.numel())
    if rank == 0:
        raise ValueError("mount has no directions")
    count = math.ceil(rank / chunk)
    cols = count * chunk
    core = projected_core(basis, stored)
    down = b.v.new_zeros((cols, b.v.shape[0]))
    down[:rank] = b.v[:, :rank].T
    up = b.u.new_zeros((cols, b.u.shape[0]))
    up[:rank] = b.u[:, :rank].T * strength
    coupling = core.new_zeros((cols, rank))
    coupling[:rank] = core
    keys = coupling.reshape(count, chunk, rank).clone()
    keys = torch.nn.functional.normalize(keys, dim=-1)
    rows = torch.tensor(
        [min(chunk, rank - i * chunk) for i in range(count)],
        dtype=torch.long,
        device=b.u.device,
    )
    return LayerSet(
        router=b.v[:, :rank].T,
        keys_in=keys,
        keys_out=keys.clone(),
        rows_in=rows,
        rows_out=rows.clone(),
        down=down,
        up=up,
        per_expert=chunk,
        form="input",
        direction_core=coupling,
        fine=fine,
        direction_candidates=candidates,
        key=mount,
        direction_sources=b.output_sources[:rank],
    )


def direction_adapters(
    basis: LibraryBasis,
    shapes: dict,
    *,
    chunk: int = 1,
    fine: bool = False,
    candidates: int = 16,
):

    from ..sets import AdapterSet, KeyedExperts

    adapters = DirectionAdapters()
    used = set()
    for target, shape in shapes.items():
        mount = basis_mount(basis.mounts, target)
        if mount is None:
            continue
        used.add(mount)
        b = basis.mounts[mount]
        if (b.v.shape[0], b.u.shape[0]) != tuple(shape):
            raise ValueError(f"{target}: library basis does not match model shape")
        layer = direction_set(basis, mount, chunk=chunk, fine=fine, candidates=candidates)
        count = layer.keys_in.shape[0]
        ex = KeyedExperts(
            keys_in=layer.keys_in,
            keys_out=layer.keys_out,
            rows_in=layer.rows_in,
            rows_out=layer.rows_out,
            down=layer.down.reshape(count, 1, chunk, -1),
            up=layer.up.reshape(count, 1, chunk, -1),
        )
        adapters.append(
            AdapterSet(
                name=f"direction-{len(adapters)}",
                part_type=("lora",),
                experts={target: ex},
                routers={shape[1]: layer.router},
                form="input",
                key_source="directions",
                extra={
                    "direction_core": layer.direction_core,
                    "direction_sources": layer.direction_sources,
                    "fine": fine,
                    "candidates": candidates,
                },
            )
        )
    adapters.report_unused(basis.mounts, used)
    return adapters


def direction_file_adapters(path, shapes: dict):

    from ..import_terms import decode_ids, read_direction_terms
    from ..sets import AdapterSet, KeyedExperts

    terms = read_direction_terms(path)
    adapters = DirectionAdapters()
    used = set()
    for target, shape in shapes.items():
        mount = basis_mount(terms, target)
        if mount is None:
            continue
        used.add(mount)
        parts = terms[mount]
        output, input_, core = (parts[k] for k in ("terms_out", "terms_in", "terms_c"))
        if (input_.shape[0], output.shape[0]) != tuple(shape):
            raise ValueError(f"{target}: direction file does not match model shape")
        rank = core.shape[0]
        pack_id = decode_ids(parts["terms_ids"])[0][0].split(":", 1)[0]
        keys = torch.eye(rank).unsqueeze(0)
        rows = torch.tensor([rank], dtype=torch.long)
        experts = KeyedExperts(
            keys_in=keys,
            keys_out=keys.clone(),
            rows_in=rows,
            rows_out=rows.clone(),
            down=input_.T.reshape(1, 1, rank, -1),
            up=output.T.reshape(1, 1, rank, -1),
        )
        adapters.append(
            AdapterSet(
                name=f"direction-file-{pack_id[:40]}",
                part_type=("lora",),
                experts={target: experts},
                routers={shape[1]: input_.T},
                form="input",
                key_source="directions",
                imported_from={"sources": [{"path": str(path), "sha256": pack_id}]},
                extra={
                    "direction_core": core,
                    "direction_sources": [[{"file": str(path), "label": Path(path).name}]],
                    "fine": False,
                    "candidates": 1,
                },
            )
        )
    adapters.report_unused(terms, used)
    return adapters


def node_candidates(x: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:

    if x.ndim < 1 or keys.ndim != 2 or x.shape[-1] != keys.shape[-1]:
        raise ValueError("query and node key widths must match")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("candidate count must be a positive integer")
    if keys.shape[0] == 0:
        raise ValueError("at least one node key is required")
    return torch.topk(x @ keys.transpose(0, 1), min(k, keys.shape[0]), dim=-1).indices
