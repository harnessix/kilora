# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .. import compute, route_freeze
from ..program import interpret
from . import driver
from .default import default_program


def budget_auto_ir(alphas, y0, target):
    if y0.ndim != 3 or y0.dtype != torch.float32:
        raise ValueError("IR automatic budget needs a three-dimensional float32 output")
    weights = {axis: compute._axis_alpha(alpha, y0, target) for axis, alpha in alphas.items()}
    sizes = {"batch": y0.shape[0], "tokens": y0.shape[1]}
    total = torch.tensor(0.0, dtype=torch.float64, device=y0.device)
    tensor_total = False
    for axis, alpha in weights.items():
        if axis in compute.UNBUDGETED_AXES:
            continue
        if isinstance(alpha, torch.Tensor):
            variant = "auto_sum" if tensor_total else "auto_seed_scalar"
            value = alpha[..., 0]
            tensor_total = True
        else:
            variant = "auto_add_scalar" if tensor_total else "scalar_sum"
            value = torch.tensor(alpha, dtype=torch.float64, device=y0.device)
        total = interpret(
            default_program("budget", variant, sizes=sizes), {"previous": total, "alpha": value}, {}
        )["total"]
    result = {}
    for axis, alpha in weights.items():
        if axis in compute.UNBUDGETED_AXES:
            result[axis] = alpha
            continue
        is_tensor = isinstance(alpha, torch.Tensor)
        value = (
            alpha[..., 0]
            if is_tensor
            else torch.tensor(alpha, dtype=torch.float64, device=y0.device)
        )
        variant = "auto" if is_tensor else "scalar"
        shape_total = total if is_tensor else torch.zeros((), dtype=torch.float64, device=y0.device)
        scale = interpret(
            default_program("budget", variant, sizes=sizes),
            {"total": shape_total, "alpha": value},
            {},
        )["scale"]
        result[axis] = scale.unsqueeze(-1) if is_tensor else scale.item()
    return result


def budget_weights_ir(weights):
    from .driver import _run

    budgeted = [name for name in weights if name not in compute.UNBUDGETED_AXES]
    tensors = [weights[name] for name in budgeted if isinstance(weights[name], torch.Tensor)]
    if not tensors or any(
        value.ndim != 3 or value.shape[-1] != 1 or value.dtype != torch.float32 for value in tensors
    ):
        raise NotImplementedError(
            "budget override needs float32 mapped alphas shaped [batch, tokens, 1]"
        )
    first = tensors[0]
    sizes = {"batch": first.shape[0], "tokens": first.shape[1]}
    total = torch.tensor(0.0, dtype=torch.float64, device=first.device)
    tensor_total = False
    for name in budgeted:
        alpha = weights[name]
        if isinstance(alpha, torch.Tensor):
            variant = "auto_sum" if tensor_total else "auto_seed_scalar"
            value = alpha[..., 0]
            tensor_total = True
        else:
            variant = "auto_add_scalar" if tensor_total else "scalar_sum"
            value = torch.tensor(alpha, dtype=torch.float64, device=first.device)
        total = _run("budget", variant, {"previous": total, "alpha": value}, sizes)["total"]
    result = dict(weights)
    for name in budgeted:
        alpha = weights[name]
        value = (
            alpha[..., 0]
            if isinstance(alpha, torch.Tensor)
            else torch.tensor(alpha, dtype=torch.float64, device=first.device)
        )
        variant = "auto" if isinstance(alpha, torch.Tensor) else "auto_scalar"
        result[name] = _run("budget", variant, {"total": total, "alpha": value}, sizes)[
            "scale"
        ].unsqueeze(-1)
    return result


def freeze_record_ir(rec, keyed, positions, target, regions, depth, lead, progress, enabled=True):
    from .driver import _run

    steps = rec["routes"].get(target)
    if not steps or any(len(g) != len(keyed) for g in steps):
        return False, None
    if len(lead) != 2:
        raise ValueError("IR freeze needs batch and token axes")
    scales, ups, forms = [], [], []
    threshold = route_freeze.threshold(keyed)
    enabled = enabled and threshold is not None
    threshold = threshold if threshold is not None else 0.0
    frozen = False
    for i, layer in enumerate(keyed):
        routes = [step[i].float() for step in steps]
        batch, tokens, experts = routes[0].shape
        cols = experts * layer.per_expert
        sizes = {
            "batch": batch,
            "tokens": tokens,
            "experts": experts,
            "per": layer.per_expert,
            "cols": cols,
            "out": layer.up.shape[-1],
        }
        history = torch.zeros_like(routes[0])
        replace = torch.tensor(
            layer.iteration == "path" or layer.aggregation == "replace", device=history.device
        )
        for fraction, route in zip(
            route_freeze._step_weights(len(routes), depth), routes, strict=True
        ):
            history = _run(
                "freeze",
                "step",
                {
                    "previous": history,
                    "route": route,
                    "fraction": torch.tensor(fraction, device=history.device),
                    "replace": replace,
                },
                sizes,
            )["aggregate"]
        gate = rec["gates"].get((target, i)) or rec["gates"].get(
            (f"{target}#input{positions[i]}", 0)
        )
        if gate is None:
            qa = history.new_ones(batch, tokens, 1)
            qb = history.new_ones(batch, 1, layer.up.shape[-1])
        else:
            qa, qb = (value.float() for value in gate)
        strength = torch.as_tensor(layer.scale, dtype=torch.float32, device=history.device).expand(
            batch, cols
        )
        result = driver._run(
            "freeze",
            "default",
            {
                "progress": torch.tensor(progress, dtype=torch.float64, device=history.device),
                "threshold": torch.tensor(threshold, dtype=torch.float64, device=history.device),
                "enabled": torch.tensor(enabled, device=history.device),
                "history": history,
                "last_weights": routes[-1],
                "strength": strength,
                "gate_token": qa,
                "gate_channel": qb,
                "up": layer.up.float(),
            },
            sizes,
        )
        frozen = bool(result["frozen"].item())
        scales.append(result["scales"])
        ups.append(result["up_gated"])
        forms.append(layer.form)
    return frozen, {
        "lead": lead,
        "forms": forms,
        "downs": [s.down for s in keyed],
        "expert_masks": [s.expert_mask for s in keyed],
        "scale": torch.cat(scales, -1),
        "up": torch.cat(ups, -2),
        "regions": list(regions),
    }
