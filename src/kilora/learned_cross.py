# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


def validate(data):
    if set(data) != {"down", "up", "bias"}:
        raise ValueError("learned cross needs down, up and bias tensors")
    down, up, bias = (data[k] for k in ("down", "up", "bias"))
    if not all(isinstance(t, torch.Tensor) and t.is_floating_point() for t in (down, up, bias)):
        raise ValueError("learned cross tensors must be floating point")
    if down.ndim != 2 or up.ndim != 2 or bias.ndim != 1:
        raise ValueError("learned cross tensor dimensions are invalid")
    rank, width = down.shape
    if rank < 1 or width < 1 or up.shape != (width, rank) or bias.shape != (width,):
        raise ValueError("learned cross tensor shapes do not match")
    if not all(torch.isfinite(t).all().item() for t in (down, up, bias)):
        raise ValueError("learned cross tensors must be finite")
    return data


def cross(x0, x, data):
    validate(data)
    if x.shape != x0.shape or x.shape[-1] != data["bias"].numel():
        raise ValueError("learned cross feature shape does not match")
    down, up, bias = (data[k].to(device=x.device, dtype=x.dtype) for k in ("down", "up", "bias"))
    return x0 * (F.linear(F.linear(x, down), up, bias)) + x
