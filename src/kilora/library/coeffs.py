# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from safetensors import safe_open

from ..lora_import import _scale
from ..multi_import import _group
from .basis import _delta, _resolve_groups


def _project_dense(delta, u, v):
    ul = u.T @ delta
    core = ul @ v
    output_energy = ul.square().sum(dim=1).clamp_min(0)
    input_energy = (delta @ v).square().sum(dim=0).clamp_min(0)
    total = delta.square().sum().clamp_min(0)
    share = core.square().sum() / total if total > 0 else total.new_zeros(())
    return core, output_energy, input_energy, total, share


def _low_rank_parts(file, keys, rank):

    if "down" in keys and "up" in keys:
        up = file.get_tensor(keys["up"]).float()
        down = file.get_tensor(keys["down"]).float()
        if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError("invalid linear LoRA weights")
        alpha = file.get_tensor(keys["alpha"]) if "alpha" in keys else None
        scale = _scale(str(keys), alpha, down.shape[0], None)
        if rank is None:
            return up * scale, down, "linear"
        ql, rl = torch.linalg.qr(up, mode="reduced")
        qr, rr = torch.linalg.qr(down.T, mode="reduced")
        u, s, vh = torch.linalg.svd((rl @ rr.T) * scale, full_matrices=False)
        n = min(rank, s.numel())
        return (ql @ u[:, :n]) * s[:n], (qr @ vh[:n].T).T, "linear"
    delta = _delta(file, keys, "cpu")
    if rank is None:
        return delta, None, "dense"
    singular_u, singular_s, singular_vh = torch.linalg.svd(delta, full_matrices=False)
    n = min(rank, singular_s.numel())
    return singular_u[:, :n] * singular_s[:n], singular_vh[:n], "dense"


def lora_coefficients(path, basis, rank=None):

    if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
        raise ValueError("rank must be a positive integer or None")
    result = {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        all_keys = list(file.keys())
        keys = [k for k in all_keys if not k.startswith(("lora_te", "text_encoder"))]
        groups, skipped, layout = _group(dict.fromkeys(keys), str(path))
        if skipped:
            raise ValueError(f"unsupported keys: {skipped[:5]}")
        for name, parts in _resolve_groups(groups, layout).items():
            if name not in basis.mounts:
                continue
            left, right, kind = _low_rank_parts(file, parts, rank)
            mount = basis.mounts[name]
            u, v = mount.u.float().cpu(), mount.v.float().cpu()
            width = left.shape[1] if right is None else right.shape[1]
            if u.shape[0] != left.shape[0] or v.shape[0] != width:
                raise ValueError(f"{name}: basis and adapter shapes differ")
            if right is None:
                core, output_energy, input_energy, total, share = _project_dense(left, u, v)
            else:
                ul, rv = u.T @ left, right @ v
                core = ul @ rv
                output_energy = ((ul @ (right @ right.T)) * ul).sum(dim=1).clamp_min(0)
                input_energy = ((rv.T @ (left.T @ left)) * rv.T).sum(dim=1).clamp_min(0)
                total = ((left.T @ left) * (right @ right.T)).sum().clamp_min(0)
                share = core.square().sum() / total if total > 0 else total.new_zeros(())
            result[name] = {
                "core": core,
                "output_energy": output_energy,
                "input_energy": input_energy,
                "energy": total,
                "captured_share": share,
                "kind": kind,
            }
    return result
