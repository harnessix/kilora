# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import torch

from .compute import dropout_uniforms


def rotate_pair(
    predecessor_up: torch.Tensor,
    successor_down: torch.Tensor,
    target: str,
    step: int,
    angle: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:

    if predecessor_up.ndim != 2 or successor_down.ndim != 2:
        raise ValueError("path matrices must be matrices")
    n = predecessor_up.shape[0]
    if successor_down.shape[1] != n or predecessor_up.device != successor_down.device:
        raise ValueError("path matrices must share a feature space and device")
    if not math.isfinite(angle):
        raise ValueError("rotation angle must be finite")
    if n < 2 or angle == 0:
        return predecessor_up.clone(), successor_down.clone()
    vectors = dropout_uniforms((2, n), target + "#rotation", step, predecessor_up.device)
    u = vectors[0].double() - 0.5
    v = vectors[1].double() - 0.5
    if float(u.norm()) < 1e-12:
        u = torch.zeros_like(u)
        u[0] = 1
    u = u / u.norm()
    v = v - u * (u @ v)
    if float(v.norm()) < 1e-12:
        v = torch.zeros_like(v)
        v[torch.argmin(u.abs())] = 1
        v = v - u * (u @ v)
    v = v / v.norm()
    c, s = math.cos(angle) - 1, math.sin(angle)

    def apply(x):
        u_, v_ = u.to(x.dtype), v.to(x.dtype)
        ux, vx = u_ @ x, v_ @ x
        return x + u_[:, None] * (c * ux - s * vx) + v_[:, None] * (s * ux + c * vx)

    return apply(predecessor_up), apply(successor_down.T).T
