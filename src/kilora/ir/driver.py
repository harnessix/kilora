# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from .. import compute as c
from ..program import interpret
from .defaults import default_bindings, default_program, interpret_key_preparation
from .dispatch import current_override, dispatch_active, interpreting_all, lowering


def _run(part, variant, inputs, sizes=None, *, context=None, _setting=None, **parameters):
    if part == "carry" and variant in ("drop", "dropped") and c._ablation.read_dropout_off():
        return {"state": inputs["change"]}
    bindings = default_bindings(part, variant, **parameters)
    override = current_override(part, variant, _setting)
    program = default_program(part, variant, sizes)
    if override is not None:
        for name, value in override.parameters.items():
            if part == "keep" and name == "DROPOUT_DELTA":
                continue
            if name in bindings:
                bindings[name] = torch.as_tensor(value, dtype=bindings[name].dtype)
            elif (
                part in {"guard", "live_write", "budget", "freeze"}
                and name in inputs
                and inputs[name].ndim == 0
            ):
                inputs = dict(inputs)
                inputs[name] = torch.as_tensor(
                    value, dtype=inputs[name].dtype, device=inputs[name].device
                )
            else:
                raise ValueError(f"unknown parameter {name!r} for {part}.{variant}")
        if override.program is not None:
            program = override.program
            bindings = dict(override.bindings)
    expand_select = None
    if part == "select" and variant.startswith("tensor_") and override and override.program:
        expand_select = inputs["logits"]
        region = inputs["region"].unsqueeze(-1) if bool(inputs["has_region"]) else None
        inputs = {**inputs, "logits": c.region_mean(expand_select, region)}
    if (part, variant) == ("carry", "dropped") and not (override and override.program):
        bindings["one"] = bindings["one"].to(inputs["mask"].dtype)
    selected = (
        lowering(part, variant, program, override.parameters if override else None)
        if dispatch_active()
        and not (override and (override.program is not None or override.parameters))
        else None
    )
    if callable(selected):
        return selected(inputs)
    result = interpret(
        program,
        inputs,
        {k: v.to(next(iter(inputs.values())).device) for k, v in bindings.items()},
        context=context,
    )
    if (
        override is not None
        and override.program is not None
        and part not in {"budget", "freeze", "guard", "live_write"}
    ):
        from .parts import PARTS, check_invariants

        axis_sizes = {}
        for name, spec in PARTS[part].signature.inputs.items():
            for axis, extent in zip(spec.axes, inputs[name].shape, strict=True):
                if axis != "r1":
                    axis_sizes[axis] = extent
        check_invariants(part, result, sizes=axis_sizes)
    if expand_select is not None:
        result["weights"] = result["weights"].expand_as(expand_select)
    return result


def _adaptive_reference(s, z, history, norm, rms, step, active=None):
    with torch.no_grad():
        if c.base_reference():
            first_norm, first_rms = torch.zeros_like(norm), torch.zeros_like(rms)
        else:
            if "first_norm" not in history:
                up = s.up.detach().float()
                dz = z.detach().float()
                if history.get("before") is not None:
                    dz = dz - history["before"].detach().float()
                history["first_norm"], history["first_rms"] = c.change_stats(
                    dz, up, up @ up.t(), active
                )
            first_norm, first_rms = history["first_norm"], history["first_rms"]
    return {
        "first_norm": first_norm,
        "first_rms": first_rms,
        "first_step": torch.tensor(step == 1 and not c.base_reference(), device=z.device),
    }


def _read_mask(variant, inputs, setting):
    if c._ablation.read_dropout_off():
        return torch.ones_like(inputs["change"])
    mask = _run("carry", variant, inputs, _setting=setting)["mask"]
    keep = setting.ir_parts.get("keep")
    if keep is None and not interpreting_all():
        return mask
    if keep is not None and setting.ir_parts.get("carry") is not None:
        raise NotImplementedError("simultaneous keep and carry mask overrides")
    base = inputs["base_read"].detach().float()
    delta = keep.parameters.get("DROPOUT_DELTA", c.DROPOUT_DELTA) if keep else c.DROPOUT_DELTA
    sigma = base.pow(2).mean(-1, keepdim=True).sqrt() + delta
    probability = _run(
        "keep",
        "default",
        {"carried": inputs["change"], "base_scale": sigma},
        _setting=setting,
    )["probability"]
    if setting.ir_parts.get("carry") is not None:
        return mask
    return probability if variant == "mask_expect" else (inputs["uniforms"] < probability).float()


def _prepare_keys(s):
    if s.form == "state" and s.down is not None:
        return c._set_keys(s)
    tensors = [s.keys_in, s.rows_in, s.keys_out, s.rows_out]
    cache_key = c._tensor_key("ir_keys", tensors)
    hit = c._cache_get(cache_key)
    if hit is not None:
        return hit

    def prepare(keys, rows):
        return interpret_key_preparation(keys, rows)

    kin = prepare(s.keys_in, s.rows_in)
    result = (kin, kin if s.keys_out is None else prepare(s.keys_out, s.rows_out))
    c._cache_put(cache_key, result)
    return result


def prepare_layer_ir(sets):

    if torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for s in sets
        if s.form != "state"
        for t in (s.keys_in, s.keys_out, s.rows_in, s.rows_out)
    ):
        raise NotImplementedError("run_layer_ir does not yet support gradients")
    return tuple(_prepare_keys(s) for s in sets)


def _score(query, keys, rows, temperature, setting=None):
    width = query.shape[-1]
    return _run(
        "score",
        "keys",
        {
            "query": query,
            "keys": keys,
            "rows": c.key_rows(keys, rows),
            "key_width": torch.tensor(float(width), device=query.device),
            "temperature": torch.as_tensor(temperature, device=query.device, dtype=torch.float32),
        },
        _setting=setting,
    )["logits"]


def _project_route(query, keys, weights, setting=None):
    batch, tokens, width = query.shape
    experts, per, _ = keys.shape
    projection = _run(
        "score",
        "projection",
        {
            "query": query.reshape(1, batch * tokens, width),
            "keys": keys.reshape(experts * per, width).t().unsqueeze(0),
        },
        _setting=setting,
    )["projection"].reshape(batch, tokens, experts, per)
    weighted = _run(
        "score", "weighted", {"projection": projection, "weights": weights}, _setting=setting
    )["weighted"]
    return _run(
        "score",
        "projection",
        {
            "query": weighted.reshape(1, batch * tokens, experts * per),
            "keys": keys.reshape(1, experts * per, width),
        },
        _setting=setting,
    )["projection"].reshape(batch, tokens, width)


def _joint_derived_select(logits, sets, regions, candidates, noise_scale=0.0):
    means = sets[0].routing_scope == "tensor"
    if means and not any(r is not None for r in regions) and all(m is None for m in candidates):
        joint = torch.cat(
            [
                lg + (s.log_prior if len(sets) > 1 else 0.0)
                for lg, s in zip(logits, sets, strict=True)
            ],
            -1,
        )
        mean = c.region_mean(joint, None)
        noise = c.gumbel(mean, noise_scale) if noise_scale else torch.zeros_like(mean)
        weights = _run(
            "select",
            f"tensor_{sets[0].routing_select}",
            {
                "logits": joint,
                "noise": noise,
                "region": torch.ones_like(joint[..., 0]),
                "has_region": torch.tensor(False, device=joint.device),
            },
            _setting=sets[0],
        )["weights"]
        return list(weights.expand_as(joint).split([lg.shape[-1] for lg in logits], -1))
    parts = []
    for lg, s, region, mask in zip(logits, sets, regions, candidates, strict=True):
        prior = s.log_prior if len(sets) > 1 else 0.0
        if means:
            lg = _run(
                "select",
                "region_mean",
                {
                    "logits": lg,
                    "region": torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                    "has_region": torch.tensor(region is not None, device=lg.device),
                    "prior": torch.as_tensor(prior, dtype=lg.dtype, device=lg.device),
                },
                _setting=s,
            )["mean_logits"]
        else:
            lg = lg + prior
        parts.append(lg if mask is None else lg.masked_fill(~mask, -torch.inf))
    joint = torch.cat(parts, -1)
    noise = c.gumbel(joint, noise_scale) if noise_scale else torch.zeros_like(joint)
    weights = _run(
        "select",
        f"token_{sets[0].routing_select}",
        {"logits": joint, "noise": noise},
        _setting=sets[0],
    )["weights"]
    return [
        gate.expand_as(lg)
        for gate, lg in zip(weights.split([lg.shape[-1] for lg in logits], -1), logits, strict=True)
    ]


def run_layer_ir(
    x,
    y0,
    sets,
    depth,
    *,
    shared=False,
    noise_scale=0.0,
    query_policy=None,
    grad_steps=None,
    target="",
    regions=None,
    schedule=None,
    expert_regions=None,
    index=0,
    count=1,
    kind="other",
    t=None,
    token_depth=None,
    expert_scales=None,
    _apply_alphas=True,
    _input_place=0,
):

    from ..call_values import current_active_axes, current_axis_alphas, current_row_axes

    if not sets:
        return y0
    if any(s.apply_once for s in sets):
        raise NotImplementedError("run_layer_ir does not support sets that run once")
    if expert_regions is not None and not any(expert_regions):
        expert_regions = None
    if expert_scales is not None and not any(v is not None for v in expert_scales):
        expert_scales = None
    if sets[0].iteration == "path" and sets[0].path_width > 1:
        if interpreting_all() or any(s.ir_parts for s in sets):
            raise NotImplementedError("run_layer_ir does not support path width above one")
        return c.run_layer(
            x,
            y0,
            sets,
            depth,
            shared=shared,
            noise_scale=noise_scale,
            query_policy=query_policy,
            grad_steps=grad_steps,
            target=target,
            regions=regions,
            schedule=schedule,
            expert_regions=expert_regions,
            index=index,
            count=count,
            kind=kind,
            t=t,
            token_depth=token_depth,
            expert_scales=expert_scales,
            _apply_alphas=_apply_alphas,
        )
    if sets[0].fixed_depth is not None and all(s.fixed_depth == sets[0].fixed_depth for s in sets):
        depth = sets[0].fixed_depth
        schedule = None
        token_depth = None
    auto = False
    if schedule is not None:
        key = next((s.key for s in sets if s.down is not None and s.form == "state"), sets[0].key)
        progress = c.current_progress() if t is None else float(t)
        depth = float(schedule.depth(index, count, kind, progress, key))
        if depth <= 0:
            return y0
        if getattr(schedule, "is_auto", False):
            depth = min(depth, c.AUTO_DEPTH_CAP)
            auto = True
    td = None
    if token_depth is not None:
        values = c._token_values(token_depth, y0)
        lower, upper = torch.stack((values.min(), values.max())).tolist()
        if lower == upper:
            depth = c._whole(lower)
        else:
            td = c._TokenDepth(values, upper)
            depth = td.steps
    if depth <= 0:
        return y0
    imported = [i for i, s in enumerate(sets) if s.form == "input" and s.down is not None]
    state = [i for i, s in enumerate(sets) if s.form == "state" and s.down is not None]
    if c._derived_state_sets(sets):
        if y0.ndim != 3 or any(s.exact for s in sets):
            return c._run_layer(
                x,
                y0,
                sets,
                depth,
                shared=shared,
                noise_scale=noise_scale,
                query_policy=query_policy,
                grad_steps=grad_steps,
                target=target,
                regions=regions,
                expert_regions=expert_regions,
                expert_scales=expert_scales,
                token_depth=token_depth,
                _apply_alphas=_apply_alphas,
            )
        if expert_regions is not None:
            sets = c._expert_masks(sets, expert_regions, y0)
        if expert_scales is not None:
            sets = c._expert_scaled(sets, expert_scales, y0)
        if regions is not None and len(regions) != len(sets):
            raise ValueError(f"{len(regions)} regions for {len(sets)} sets")
        regs = [
            c._token_region(regions[i] if regions is not None else s.region, y0)
            for i, s in enumerate(sets)
        ]
        positions = sorted(imported + state)
        if td is not None and any(sets[i].iteration == "path" for i in positions):
            raise ValueError("a per-token depth needs accumulate iteration (path mode: later)")
        from .dispatch import interpret_all

        def chain():
            with interpret_all():
                return c._run_input(
                    x,
                    y0,
                    sets,
                    positions,
                    depth,
                    regs,
                    target,
                    td,
                    interpret=True,
                    grad_steps=grad_steps,
                    noise_scale=noise_scale,
                    query_policy=query_policy,
                    shared=shared,
                    stop=True if auto else None,
                )

        progress = c.current_progress() if t is None else float(t)
        keyed = [sets[i] for i in positions]
        if td is None and not torch.is_grad_enabled() and c._route_freeze.active(keyed, progress):
            return c._route_freeze.run(
                chain, x, y0, keyed, positions, target, [regs[i] for i in positions], depth
            )
        return chain()
    if sets[0].iteration == "path" and not (
        len(sets) == 1 and sets[0].form == "input" and c.import_score(sets[0]) == "derived"
    ):
        if td is not None:
            raise ValueError("a per-token depth needs accumulate iteration (path mode: later)")
        if (
            y0.ndim != 3
            or query_policy is not None
            or expert_regions is not None
            or expert_scales is not None
            or not _apply_alphas
            or c._STEP_CARRY
            or c._PROBES
            or c._ROUTES
            or c._SHARES
            or c._SET_SHARES
            or c._SET_SHARE_SCOPES
            or c._GATE_STATS
            or c._READ_KEEP_OBSERVERS
            or c._route_freeze.RECORDS
            or current_active_axes() is not None
            or bool(current_axis_alphas())
            or current_row_axes() is not None
            or any(
                s.form not in ("state", "input")
                or (s.form == "input" and s.direction_core is not None)
                or s.axis != "creative"
                or s.down is None
                or s.up is None
                or s.route_freeze is not None
                or s.expert_offsets is not None
                or s.expert_inputs is not None
                or s.expert_mask is not None
                or 0 < s.candidates < s.keys_in.shape[0]
                or any(s.cross_weights)
                or s.learned_cross is not None
                or s.axis_joint
                for s in sets
            )
        ):
            raise NotImplementedError("run_layer_ir does not yet support these path options")
        if regions is not None and len(regions) != len(sets):
            raise ValueError(f"{len(regions)} regions for {len(sets)} sets")
        regs = [
            c._token_region(regions[i] if regions is not None else s.region, y0)
            for i, s in enumerate(sets)
        ]

        def walk(n):
            return _path_ir(x, y0, sets, n, shared, noise_scale, grad_steps, target, regs, auto)

        if auto:
            return walk(math.ceil(depth))
        low = math.floor(depth)
        if low == depth:
            return walk(low)
        hi = walk(low + 1)
        lo = walk(low) if low else y0
        return lo + (depth - low) * (hi - lo)
    if len(sets) == 1 and sets[0].form == "input":
        if expert_regions is not None:
            sets = c._expert_masks(sets, expert_regions, y0)
        if expert_scales is not None:
            sets = c._expert_scaled(sets, expert_scales, y0)
        if (
            y0.ndim != 3
            or y0.dtype != torch.float32
            or noise_scale
            or query_policy is not None
            or grad_steps is not None
            or not _apply_alphas
            or c._STEP_CARRY
            or c._PROBES
            or c._ROUTES
            or c._SHARES
            or c._SET_SHARES
            or c._SET_SHARE_SCOPES
            or c._GATE_STATS
            or c._READ_KEEP_OBSERVERS
            or c._route_freeze.RECORDS
            or current_active_axes() is not None
            or bool(current_axis_alphas())
            or current_row_axes() is not None
            or not c._fast_path_ok(y0, sets, noise_scale, query_policy, grad_steps)
            or (
                torch.is_grad_enabled()
                and any(
                    isinstance(v, torch.Tensor) and v.requires_grad
                    for v in (
                        x,
                        y0,
                        sets[0].down,
                        sets[0].up,
                        sets[0].router,
                        sets[0].keys_in,
                        sets[0].keys_out,
                    )
                )
            )
        ):
            raise NotImplementedError(
                "run_layer_ir does not yet support these imported layer options"
            )
        if regions is not None and len(regions) != 1:
            raise ValueError(f"{len(regions)} regions for 1 set")
        region = c._token_region(regions[0] if regions is not None else sets[0].region, y0)
        progress = c.current_progress() if t is None else float(t)
        if td is None and not torch.is_grad_enabled() and c._route_freeze.active(sets, progress):
            return c._route_freeze.run(
                lambda: _run_import_ir(x, y0, sets[0], depth, target, region, _input_place),
                x,
                y0,
                sets,
                [0],
                target,
                [region],
                depth,
            )
        return _run_import_ir(x, y0, sets[0], depth, target, region, _input_place)
    steps, low, frac = c._depth_parts(depth)
    grad = torch.is_grad_enabled() and any(
        isinstance(v, torch.Tensor) and v.requires_grad
        for v in [y0]
        + [
            v
            for s in sets
            for v in (
                s.down,
                s.up,
                s.router,
                s.keys_in,
                s.keys_out,
                s.scale,
                s.tau,
                *(s.gate or {}).values(),
            )
        ]
    )
    if (
        y0.ndim != 3
        or y0.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or query_policy is not None
        or grad_steps is not None
        or expert_regions is not None
        or expert_scales is not None
        or c._STEP_CARRY
        or c._PROBES
        or c._ROUTES
        or c._SHARES
        or c._SET_SHARES
        or c._SET_SHARE_SCOPES
        or c._GATE_STATS
        or c._READ_KEEP_OBSERVERS
        or c._route_freeze.RECORDS
        or current_active_axes() is not None
        or bool(current_axis_alphas())
        or current_row_axes() is not None
        or (not grad and not c._fast_path_ok(y0, sets, 0.0, query_policy, grad_steps))
        or (grad and (noise_scale > 0 or td is not None))
        or not _apply_alphas
        or any(s.carry != sets[0].carry for s in sets)
        or any(
            s.form != "state"
            or s.iteration != "accumulate"
            or s.aggregation != "accumulate"
            or s.down is None
            or s.up is None
            or s.routing_scope not in ("token", "tensor")
            or s.axis != "creative"
            or s.route_freeze is not None
            or 0 < s.candidates < s.keys_in.shape[0]
            or any(s.cross_weights)
            or s.learned_cross is not None
            or (not grad and c.learned_kind(s) != "none")
            or s.expert_offsets is not None
            or s.expert_inputs is not None
            or s.expert_mask is not None
            or (isinstance(s.log_prior, torch.Tensor) and s.log_prior.numel() != 1)
            for s in sets
        )
    ):
        raise NotImplementedError("run_layer_ir does not yet support these layer options")
    if regions is not None and len(regions) != len(sets):
        raise ValueError(f"{len(regions)} regions for {len(sets)} sets")
    regs = [
        c._token_region(regions[i] if regions is not None else s.region, y0)
        for i, s in enumerate(sets)
    ]
    if grad and c._fast_path_ok(y0, sets, noise_scale, query_policy, grad_steps):
        with torch.no_grad():
            return run_layer_ir(x, y0, sets, depth, shared=shared, target=target, regions=regions)
    if grad:
        return _run_layer_grad(y0, sets, depth, shared, target, regs)
    if td is not None or noise_scale > 0:
        return _loop_ir(y0, sets, depth, shared, noise_scale, target, regs, td)
    dtype, device = y0.dtype, y0.device
    keys = prepare_layer_ir(sets)
    base = [F.linear(y0, s.down.to(dtype)).float() for s in sets]
    hold = [value * c._scale(s, torch.float32) for value, s in zip(base, sets, strict=True)]
    up, gs, hs = c._lowrank_mats(sets)
    q0 = [F.linear(y0, s.router.to(dtype)).float() for s in sets]
    sizes = [s.up.shape[0] for s in sets]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    zs = [torch.zeros_like(value) for value in base]
    zm = torch.cat(zs, -1)
    masks = [None] * len(sets)
    history = [None] * len(sets)
    before = [None] * len(sets)
    head_rates = [None] * len(sets)
    head_counts = [0] * len(sets)
    references = [{} for _ in sets]
    prev = None
    norm, rms = c._y0_gate_stats(y0)
    snap = y0 if frac and low == 0 else None
    for k in range(1, steps + 1):
        if any(mask is not None for mask in masks):
            qs, coefs = list(q0), list(base)
            for source, off, n, mask in zip(sets, offsets, sizes, masks, strict=True):
                zj = zm[..., off : off + n]
                for i, target_set in enumerate(sets):
                    if mask is None:
                        qs[i] = _run(
                            "iterate",
                            "output",
                            {"base": qs[i], "coefficients": zj, "up": hs[i][off : off + n]},
                        )["output"]
                        coefs[i] = _run(
                            "iterate",
                            "output",
                            {"base": coefs[i], "coefficients": zj, "up": gs[i][off : off + n]},
                        )["output"]
                    else:
                        a, b = mask
                        shared_channel = b.ndim == 2
                        variant = "gated_read_shared" if shared_channel else "gated_read"
                        common = {
                            "token": a,
                            "change": zj,
                            "up": source.up.float()
                            if shared_channel
                            else source.up.float().unsqueeze(0).expand(y0.shape[0], -1, -1),
                            "channel": b.squeeze(0) if shared_channel else b.squeeze(-2),
                        }
                        qs[i] = _run(
                            "iterate",
                            variant,
                            {**common, "base": qs[i], "target": target_set.router.float().t()},
                        )["result"]
                        coefs[i] = _run(
                            "iterate",
                            variant,
                            {**common, "base": coefs[i], "target": target_set.down.float().t()},
                        )["result"]
        else:
            qs = [
                _run("iterate", "output", {"base": q, "coefficients": zm, "up": h})["output"]
                for q, h in zip(q0, hs, strict=True)
            ]
            coefs = [
                _run("iterate", "output", {"base": v, "coefficients": zm, "up": g})["output"]
                for v, g in zip(base, gs, strict=True)
            ]
        logits_list = []
        for i, (s, (kin, kout)) in enumerate(zip(sets, keys, strict=True)):
            q = qs[i]
            v = q if prev is None else _project_route(q, kout, prev[i], s)
            logits_list.append(_score(v, kin, s.rows_in, s.tau, s))

        def select(logits, setting, region):
            noise = torch.zeros_like(logits[:, :1] if setting.routing_scope == "tensor" else logits)
            inputs = {"logits": logits, "noise": noise}
            if setting.routing_scope == "tensor":
                inputs.update(
                    region=torch.ones_like(logits[..., 0]) if region is None else region[..., 0],
                    has_region=torch.tensor(region is not None, device=device),
                )
            return _run(
                "select",
                f"{setting.routing_scope}_{setting.routing_select}",
                inputs,
                _setting=setting,
            )["weights"]

        if (
            shared
            and len(sets) > 1
            and sets[0].routing_scope == "tensor"
            and any(region is not None for region in regs)
        ):
            means = []
            for logits, region, s in zip(logits_list, regs, sets, strict=True):
                inputs = {
                    "logits": logits,
                    "region": torch.ones_like(logits[..., 0]) if region is None else region[..., 0],
                    "has_region": torch.tensor(region is not None, device=device),
                    "prior": torch.as_tensor(s.log_prior, dtype=logits.dtype, device=device),
                }
                means.append(_run("select", "region_mean", inputs)["mean_logits"])
            joint = torch.cat(means, -1)
            inputs = {"logits": joint, "noise": torch.zeros_like(joint)}
            weights = _run("select", f"token_{sets[0].routing_select}", inputs)["weights"]
            gates = [
                g.expand_as(logits)
                for g, logits in zip(
                    torch.split(weights, [lg.shape[-1] for lg in logits_list], dim=-1),
                    logits_list,
                    strict=True,
                )
            ]
        elif shared and len(sets) > 1:
            joint = torch.cat(
                [lg + s.log_prior for lg, s in zip(logits_list, sets, strict=True)], -1
            )
            weights = select(joint, sets[0], None)
            gates = list(torch.split(weights, [lg.shape[-1] for lg in logits_list], dim=-1))
        else:
            gates = [
                select(lg, s, region) for lg, s, region in zip(logits_list, sets, regs, strict=True)
            ]
        prev = gates
        y = y0
        masks = []
        for i, (s, gate) in enumerate(zip(sets, gates, strict=True)):
            coef = coefs[i]
            part = coef * gate.repeat_interleave(s.per_expert, dim=-1) * c._scale(s, torch.float32)
            zs[i] = _run(
                "iterate", "accumulate", {"previous_state": zs[i], "change": part}, _setting=s
            )["state"]
            z = zs[i]
            if c.uses_adaptive(s):
                if s.gate_head_limit is not None and head_counts[i] >= s.gate_head_limit:
                    a, b = head_rates[i]
                else:
                    batch, tokens, cols = z.shape
                    out = y0.shape[-1]
                    zr = z if regs[i] is None else z * regs[i].to(z.dtype)
                    res = _run(
                        "gate",
                        "adaptive",
                        {
                            "coefficients": zr,
                            "before": torch.zeros_like(z) if before[i] is None else before[i],
                            "prior_history": torch.zeros_like(z)
                            if history[i] is None
                            else history[i],
                            "has_history": torch.tensor(history[i] is not None, device=device),
                            "up": s.up.float(),
                            "base_norm": norm,
                            "base_rms": rms,
                            **_adaptive_reference(s, zr, references[i], norm, rms, k),
                        },
                        dict(batch=batch, tokens=tokens, cols=cols, out=out),
                        tokens=tokens,
                        _setting=s,
                        kappa=s.gate_kappa,
                        coherence=s.gate_coherence,
                    )
                    before[i], history[i] = zr, res["history"]
                    a, b = res["token"], res["channel"]
                    if s.gate_head_limit is not None:
                        head_counts[i] += 1
                        head_rates[i] = (a, b)
                if k < steps and not c.expect_masks():
                    n = a.numel()
                    u = c.dropout_uniforms((n + b.numel(),), f"{target}#{i}#adaptive", k, device)
                    mask = (u < torch.cat((a.reshape(-1), b.reshape(-1)))).float()
                    a, b = mask[:n].reshape(a.shape), mask[n:].reshape(b.shape)
                if regs[i] is not None:
                    a = a * regs[i]
                masks.append((a, b))
                y = _run(
                    "iterate",
                    "output_split",
                    {
                        "base": y,
                        "coefficients": (a * z).to(dtype),
                        "up": s.up.to(dtype),
                        "token": torch.ones_like(a, dtype=dtype),
                        "channel": b.to(dtype).squeeze(-2),
                    },
                )["output"]
            elif regs[i] is not None:
                a = regs[i]
                b = torch.ones((1, s.up.shape[1]), device=device)
                masks.append((a, b))
                y = _run(
                    "iterate",
                    "output_split",
                    {
                        "base": y,
                        "coefficients": (a * z).to(dtype),
                        "up": s.up.to(dtype),
                        "token": torch.ones_like(a, dtype=dtype),
                        "channel": b.to(dtype),
                    },
                )["output"]
            else:
                masks.append(None)
                y = _run(
                    "iterate",
                    "output",
                    {"base": y, "coefficients": z.to(dtype), "up": s.up.to(dtype)},
                )["output"]
        if k < steps:
            zs_carried = []
            for i, (s, z) in enumerate(zip(sets, zs, strict=True)):
                shape = (*z.shape[:-1], z.shape[-1] // s.per_expert, s.per_expert)
                u = c.dropout_uniforms(
                    shape, c._set_target(target, s, i), k, device, c.column_keys(s)
                )
                variant = "mask_expect" if c.expect_masks() else "mask_sample"
                mask = _read_mask(
                    variant,
                    {
                        "change": z.detach().float().reshape(shape),
                        "base_read": base[i].detach().reshape(shape),
                        "uniforms": u,
                    },
                    s,
                ).reshape_as(z)
                cz = _run(
                    "carry",
                    s.carry,
                    {
                        "change": z,
                        "carried": hold[i],
                        "mask": mask,
                        "base_read": base[i],
                        "uniforms": torch.zeros_like(z),
                    },
                    _setting=s,
                )["state"]
                if s.carry == "dropped":
                    hold[i] = cz
                zs_carried.append(cz)
            zm = torch.cat(zs_carried, -1)
        if frac and k == low:
            snap = (
                y
                if any(mask is not None for mask in masks)
                else _run(
                    "iterate",
                    "output",
                    {"base": y0, "coefficients": torch.cat(zs, -1).to(dtype), "up": up.to(dtype)},
                )["output"]
            )
    if not any(mask is not None for mask in masks):
        y = _run(
            "iterate",
            "output",
            {"base": y0, "coefficients": torch.cat(zs, -1).to(dtype), "up": up.to(dtype)},
        )["output"]
    return snap + frac * (y - snap) if frac else y


def _loop_ir(y0, sets, depth, shared, noise_scale, target, regs, td=None):
    dtype, device = y0.dtype, y0.device
    keys = [c._set_keys(s) for s in sets]
    base = [F.linear(y0.float(), s.down.float()) for s in sets]
    hold = [F.linear(y0, s.down.to(dtype)) * c._scale(s, dtype) for s in sets]
    zs = [None] * len(sets)
    before, history = [None] * len(sets), [None] * len(sets)
    head_rates, head_counts = [None] * len(sets), [0] * len(sets)
    references = [{} for _ in sets]
    norm, rms = c._y0_gate_stats(y0)
    steps, low, frac = c._depth_parts(depth)
    prev, carried, out = None, y0, y0
    blend_coefficients = td is not None and not any(c.uses_adaptive(s) for s in sets)
    blended_zs = [None] * len(sets)
    snap = y0 if frac and low == 0 else None
    for k in range(1, steps + 1):
        old_zs, old_hold, old_carried = list(zs), list(hold), carried
        logits = []
        for i, (s, (kin, kout)) in enumerate(zip(sets, keys, strict=True)):
            q = F.linear(carried.float(), s.router.float())
            v = q if prev is None else _project_route(q, kout, prev[i], s)
            logits.append(_score(v, kin, s.rows_in, s.tau, s))

        def select(lg, s, region):
            shape = lg[:, :1] if s.routing_scope == "tensor" else lg
            noise = c.gumbel(shape, noise_scale) if noise_scale > 0 else torch.zeros_like(shape)
            inputs = {"logits": lg, "noise": noise}
            if s.routing_scope == "tensor":
                inputs.update(
                    region=torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                    has_region=torch.tensor(region is not None, device=device),
                )
            return _run("select", f"{s.routing_scope}_{s.routing_select}", inputs, _setting=s)[
                "weights"
            ]

        if (
            shared
            and len(sets) > 1
            and sets[0].routing_scope == "tensor"
            and any(r is not None for r in regs)
        ):
            means = []
            for lg, region, s in zip(logits, regs, sets, strict=True):
                inputs = {
                    "logits": lg,
                    "region": torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                    "has_region": torch.tensor(region is not None, device=device),
                    "prior": torch.as_tensor(s.log_prior, dtype=lg.dtype, device=device),
                }
                means.append(_run("select", "region_mean", inputs)["mean_logits"])
            joint = torch.cat(means, -1)
            noise = c.gumbel(joint, noise_scale) if noise_scale > 0 else torch.zeros_like(joint)
            if sets[0].routing_select == "top1" and noise_scale > 0:
                joint, noise = joint + noise, torch.zeros_like(noise)
            weights = _run(
                "select", f"token_{sets[0].routing_select}", {"logits": joint, "noise": noise}
            )["weights"]
            gates = [
                g.expand_as(lg)
                for g, lg in zip(
                    torch.split(weights, [v.shape[-1] for v in logits], -1), logits, strict=True
                )
            ]
        elif shared and len(sets) > 1:
            joint = torch.cat([lg + s.log_prior for lg, s in zip(logits, sets, strict=True)], -1)
            gates = list(
                torch.split(select(joint, sets[0], None), [v.shape[-1] for v in logits], -1)
            )
        else:
            gates = [select(lg, s, r) for lg, s, r in zip(logits, sets, regs, strict=True)]
        prev = gates
        for i, (s, gate) in enumerate(zip(sets, gates, strict=True)):
            coef = F.linear(carried.to(dtype), s.down.to(dtype))
            part = coef * gate.to(dtype).repeat_interleave(s.per_expert, -1) * c._scale(s, dtype)
            zs[i] = (
                part
                if zs[i] is None
                else _run(
                    "iterate", "accumulate", {"previous_state": zs[i], "change": part}, _setting=s
                )["state"]
            )
        y, masks = y0, []
        for i, (s, z, region) in enumerate(zip(sets, zs, regs, strict=True)):
            if c.uses_adaptive(s):
                if s.gate_head_limit is not None and head_counts[i] >= s.gate_head_limit:
                    a, b = head_rates[i]
                else:
                    zr = z if region is None else z * region.to(dtype)
                    if td is not None:
                        zr = zr * td.active(k).to(dtype)
                    batch, tokens, cols = z.shape
                    inputs = {
                        "coefficients": zr.float(),
                        "before": torch.zeros_like(zr, dtype=torch.float32)
                        if before[i] is None
                        else before[i],
                        "prior_history": torch.zeros_like(zr, dtype=torch.float32)
                        if history[i] is None
                        else history[i],
                        "has_history": torch.tensor(history[i] is not None, device=device),
                        "up": s.up.float(),
                        "base_norm": norm,
                        "base_rms": rms,
                        **_adaptive_reference(
                            s,
                            zr,
                            references[i],
                            norm,
                            rms,
                            k,
                            None if td is None else td.active(k),
                        ),
                    }
                    if td is None:
                        res = _run(
                            "gate",
                            "adaptive",
                            inputs,
                            dict(batch=batch, tokens=tokens, cols=cols, out=y0.shape[-1]),
                            tokens=tokens,
                            _setting=s,
                            kappa=s.gate_kappa,
                            coherence=s.gate_coherence,
                        )
                    else:
                        inputs["token_count"] = _run(
                            "gate", "active_count", {"active": td.active(k)}
                        )["token_count"]
                        res = _run(
                            "gate",
                            "adaptive_active",
                            inputs,
                            dict(batch=batch, tokens=tokens, cols=cols, out=y0.shape[-1]),
                            tokens=tokens,
                            _setting=s,
                            kappa=s.gate_kappa,
                            coherence=s.gate_coherence,
                        )
                    before[i], history[i] = zr, res["history"]
                    a, b = res["token"], res["channel"]
                    if s.gate_head_limit is not None:
                        head_counts[i] += 1
                        head_rates[i] = (a, b)
                last = k == steps if td is None else td.last(k)
                if last is not True and not c.expect_masks():
                    n = a.numel()
                    u = c.dropout_uniforms((n + b.numel(),), f"{target}#{i}#adaptive", k, device)
                    mask = (u < torch.cat((a.reshape(-1), b.reshape(-1)))).float()
                    ha, hb = mask[:n].reshape(a.shape), mask[n:].reshape(b.shape)
                    if td is not None:
                        ha, hb = torch.where(last, a, ha), torch.where(last, b, hb)
                    a, b = ha, hb
                if region is not None:
                    a = a * region
                mask = (a.to(dtype), b.to(dtype))
            elif region is not None:
                mask = (region.to(dtype), None)
            else:
                mask = None
            masks.append(mask)
            y = _loop_output(y, z, s.up, mask, dtype)
        if td is not None:
            if blend_coefficients:
                for i, (z, region) in enumerate(zip(zs, regs, strict=True)):
                    value = z if region is None else z * region.to(dtype)
                    old = blended_zs[i]
                    blended_zs[i] = td.blend(
                        torch.zeros_like(value) if old is None else old, value, k
                    )
            else:
                out = td.blend(out, y, k)
            zs = [td.keep(z, old, k) for z, old in zip(zs, old_zs, strict=True)]
        if k < steps:
            carried = y0
            for i, (s, z, gate_mask) in enumerate(zip(sets, zs, masks, strict=True)):
                shape = (*z.shape[:-1], z.shape[-1] // s.per_expert, s.per_expert)
                u = c.dropout_uniforms(
                    shape, c._set_target(target, s, i), k, device, c.column_keys(s)
                )
                variant = "mask_expect" if c.expect_masks() else "mask_sample"
                mask = (
                    _read_mask(
                        variant,
                        {
                            "change": z.detach().float().reshape(shape),
                            "base_read": base[i].reshape(shape),
                            "uniforms": u,
                        },
                        s,
                    )
                    .reshape_as(z)
                    .to(dtype)
                )
                cz = _run(
                    "carry",
                    s.carry,
                    {
                        "change": z,
                        "carried": hold[i],
                        "mask": mask,
                        "base_read": base[i],
                        "uniforms": torch.zeros_like(z),
                    },
                    _setting=s,
                )["state"]
                if s.carry == "dropped":
                    hold[i] = cz
                carried = _loop_output(carried, cz, s.up, gate_mask, dtype)
            if td is not None:
                carried = td.keep(carried, old_carried, k)
                hold = [td.keep(h, old, k) for h, old in zip(hold, old_hold, strict=True)]
            if frac and k == low:
                snap = y
    if td is not None and blend_coefficients:
        out = y0
        for z, s in zip(blended_zs, sets, strict=True):
            out = _loop_output(out, z, s.up, None, dtype)
    return out if td is not None else snap + frac * (y - snap) if frac else y


def _loop_output(base, z, up, mask, dtype):
    if dtype != torch.float32:
        from .. import fused

        if mask is None:
            return base + z.to(dtype) @ up.to(dtype)
        delta = fused.token_gate(z, mask[0], dtype) @ up.to(dtype)
        return base + delta if mask[1] is None else fused.channel_change(delta, mask[1], base)
    if mask is None:
        return _run("iterate", "output", {"base": base, "coefficients": z, "up": up.to(dtype)})[
            "output"
        ]
    if mask[1] is None:
        return _run(
            "iterate", "output", {"base": base, "coefficients": mask[0] * z, "up": up.to(dtype)}
        )["output"]
    if mask[1].shape[-2] != 1:
        return _run(
            "iterate",
            "output_split_tokens",
            {
                "base": base,
                "coefficients": z,
                "up": up.to(dtype),
                "token": mask[0],
                "channel": mask[1],
            },
        )["output"]
    return _run(
        "iterate",
        "output_split",
        {
            "base": base,
            "coefficients": z,
            "up": up.to(dtype),
            "token": mask[0],
            "channel": mask[1].squeeze(-2),
        },
    )["output"]


def _gate_mask_grad(q, uniforms):
    shape = q.shape
    mode = "expect" if c.expect_masks() else "sample"
    mask = _run(
        "gate",
        f"mask_{mode}",
        {"probability": q.reshape(1, 1, -1), "uniforms": uniforms.reshape(1, 1, -1)},
    )["mask"].reshape(shape)
    return mask.detach() if c._HARD_GATES or not q.requires_grad else mask


def _separable_gate_grad(s, z, previous, norm, rms, weights, target, index, step):
    g = s.gate
    qa, qb = c._gate_scores()(
        z.float(),
        None if previous is None else previous.float(),
        s.up.float(),
        norm,
        rms,
        weights.float(),
        *(g[n].float() for n in c._SEPARABLE_PARTS),
    )
    n = qa.numel()
    u = c.dropout_uniforms((n + qb.numel(),), f"{target}#{index}#gate", step, qa.device)
    return _gate_mask_grad(qa, u[:n].reshape(qa.shape)), _gate_mask_grad(
        qb, u[n:].reshape(qb.shape)
    )


def _output_gate_grad(s, delta, y0, previous, weights, rms, target, index, step):
    feats = torch.stack([delta, y0, previous], -1).float() / rms[..., None]
    weights = weights.float()
    used = (weights.detach() > 0).flatten(0, -2).any(0).nonzero().flatten().tolist()
    q = None
    for expert in used:
        part = weights[..., expert : expert + 1] * c.gate_score(
            s.output_gate, s.gate, feats, expert
        )
        q = part if q is None else q + part
    u = c.dropout_uniforms(tuple(q.shape), f"{target}#{index}#gate", step, q.device)
    return _gate_mask_grad(q.to(delta.dtype), u)


def _split_gate_grad(
    s, z, history, norm, rms, weights, target, index, step, region, last, active=None
):
    a = b = None
    if c.learned_kind(s) == "separable":
        a, b = _separable_gate_grad(
            s, z, history.get("prev"), norm, rms, weights, target, index, step
        )
    history["prev"] = z
    if c.uses_adaptive(s):
        limit = s.gate_head_limit
        if limit is not None and history.get("head_count", 0) >= limit:
            qa, qb = history["head_q"]
        else:
            zr = z if region is None else z * region.to(z.dtype)
            if active is not None:
                zr = zr * active.to(zr.dtype)
            batch, tokens, cols = z.shape
            with torch.no_grad():
                result = _run(
                    "gate",
                    "adaptive" if active is None else "adaptive_active",
                    {
                        **(
                            {}
                            if active is None
                            else {
                                "token_count": _run("gate", "active_count", {"active": active})[
                                    "token_count"
                                ]
                            }
                        ),
                        "coefficients": zr.detach(),
                        "before": torch.zeros_like(z)
                        if history.get("before") is None
                        else history["before"].detach(),
                        "prior_history": torch.zeros_like(z)
                        if "m" not in history
                        else history["m"].detach(),
                        "has_history": torch.tensor("m" in history, device=z.device),
                        "up": s.up.detach().float(),
                        "base_norm": norm,
                        "base_rms": rms,
                        **_adaptive_reference(s, zr, history, norm, rms, step, active),
                    },
                    dict(batch=batch, tokens=tokens, cols=cols, out=s.up.shape[-1]),
                    tokens=tokens,
                    _setting=s,
                    kappa=s.gate_kappa,
                    coherence=s.gate_coherence,
                )
            history["before"], history["m"] = zr, result["history"]
            qa, qb = result["token"], result["channel"]
            history["q"] = (qa if region is None else qa * region, qb)
            if limit is not None:
                history["head_count"] = history.get("head_count", 0) + 1
                history["head_q"] = (qa, qb)
        if c._route_freeze.RECORDS:
            c._route_freeze.note_gate(target, index, qa, qb)
        for stats in c._GATE_STATS:
            stats.extend((qa.flatten(), qb.flatten()))
        if last is True or c.expect_masks():
            ha, hb = qa, qb
        else:
            n = qa.numel()
            u = c.dropout_uniforms((n + qb.numel(),), f"{target}#{index}#adaptive", step, qa.device)
            mask = (u < torch.cat((qa.reshape(-1), qb.reshape(-1)))).float()
            ha, hb = mask[:n].reshape(qa.shape), mask[n:].reshape(qb.shape)
            if isinstance(last, torch.Tensor):
                ha, hb = torch.where(last, qa, ha), torch.where(last, qb, hb)
        a, b = (ha if a is None else a * ha), (hb if b is None else b * hb)
    return (a if region is None else a * region), b


def _run_layer_grad(y0, sets, depth, shared, target, regions):
    dtype, device = y0.dtype, y0.device
    steps, low, frac = c._depth_parts(depth)
    keys = [c._set_keys(s) for s in sets]
    with torch.no_grad():
        base = [F.linear(y0.float(), s.down.float()) for s in sets]
    hold = [F.linear(y0, s.down.to(dtype)) * c._scale(s, dtype) for s in sets]
    norm, rms = c._y0_gate_stats(y0)
    history = [{} for _ in sets]
    prev_delta = [torch.zeros_like(y0) for _ in sets]
    with torch.no_grad():
        gate_rms = y0.float().pow(2).mean(-2, keepdim=True).sqrt() + c.DROPOUT_DELTA
    zs = [None] * len(sets)
    carried, prev, snap = y0, None, y0 if frac and low == 0 else None
    for k in range(1, steps + 1):
        logits = []
        for i, (s, (kin, kout)) in enumerate(zip(sets, keys, strict=True)):
            query = F.linear(carried.float(), s.router.float())
            query = query if prev is None else c.out_project(query, kout, prev[i])
            logits.append(s.tau * c.norm_energy(query, kin, s.rows_in))
        if shared and len(sets) > 1:
            gates = c._gates_from_logits(logits, sets, shared, 0.0, regions)
        else:
            gates = []
            for lg, s, region in zip(logits, sets, regions, strict=True):
                noise = torch.zeros_like(lg[:, :1] if s.routing_scope == "tensor" else lg)
                inputs = {"logits": lg, "noise": noise}
                if s.routing_scope == "tensor":
                    inputs.update(
                        region=torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                        has_region=torch.tensor(region is not None, device=device),
                    )
                gate = _run("select", f"{s.routing_scope}_{s.routing_select}", inputs, _setting=s)[
                    "weights"
                ]
                if c._HARD_GATES and s.routing_select == "top1":
                    gate = gate.detach()
                gates.append(gate)
        prev = gates
        for i, (s, gate) in enumerate(zip(sets, gates, strict=True)):
            coef = F.linear(carried.to(dtype), s.down.to(dtype))
            part = (
                coef * gate.to(dtype).repeat_interleave(s.per_expert, dim=-1) * c._scale(s, dtype)
            )
            zs[i] = (
                part
                if zs[i] is None
                else _run(
                    "iterate", "accumulate", {"previous_state": zs[i], "change": part}, _setting=s
                )["state"]
            )
        y, masks = y0, []
        for i, (s, z, region) in enumerate(zip(sets, zs, regions, strict=True)):
            kind = c.learned_kind(s)
            if kind not in ("none", "separable"):
                delta = z @ s.up.to(dtype)
                mask = _output_gate_grad(
                    s, delta, y0, prev_delta[i], gates[i], gate_rms, target, i, k
                )
                if c.uses_adaptive(s):
                    a, b = _split_gate_grad(
                        s, z, history[i], norm, rms, gates[i], target, i, k, region, k == steps
                    )
                    mask = mask * (a * b).to(mask.dtype)
                elif region is not None:
                    mask = mask * region.to(mask.dtype)
                prev_delta[i] = delta
                masks.append(mask)
                y = y + mask * delta
            elif kind == "separable" or c.uses_adaptive(s):
                a, b = _split_gate_grad(
                    s, z, history[i], norm, rms, gates[i], target, i, k, region, k == steps
                )
                masks.append((a.to(dtype), b.to(dtype)))
                y = y + ((a.to(dtype) * z) @ s.up.to(dtype)) * b.to(dtype)
            elif region is not None:
                masks.append((region.to(dtype), None))
                y = y + (region.to(dtype) * z) @ s.up.to(dtype)
            else:
                masks.append(None)
                y = y + z @ s.up.to(dtype)
        if k < steps:
            carried = y0
            for i, (s, z, mask) in enumerate(zip(sets, zs, masks, strict=True)):
                m = c.read_dropout(
                    z, base[i], s.per_expert, c._set_target(target, s, i), k, c.column_keys(s)
                )
                cz = _run(
                    "carry",
                    s.carry,
                    {
                        "change": z,
                        "carried": hold[i],
                        "mask": m,
                        "base_read": base[i],
                        "uniforms": torch.zeros_like(z),
                    },
                    _setting=s,
                )["state"]
                if s.carry == "dropped":
                    hold[i] = cz
                if mask is None:
                    part = cz @ s.up.to(dtype)
                elif not isinstance(mask, tuple):
                    part = mask * (cz @ s.up.to(dtype))
                else:
                    a, b = mask
                    part = (a * cz) @ s.up.to(dtype)
                    if b is not None:
                        part = part * b
                carried = carried + part
        if frac and k == low:
            snap = y
    return snap + frac * (y - snap) if frac else y


def _run_import_ir(x, y0, s, depth, target, region, input_place=0):
    score = c.import_score(s)
    if (
        score not in ("keys", "read_share", "derived")
        or (score in ("read_share", "derived") and c.share_transition_gamma(s))
        or (
            score == "derived"
            and (c.derived_mode() != "keys" or c.derived_diffusion() or c.derived_exclusion())
        )
        or s.iteration not in (("accumulate", "path") if score == "derived" else ("accumulate",))
        or s.aggregation != "accumulate"
        or s.axis != "creative"
        or s.carry not in ("drop", "dropped")
        or s.routing_scope not in ("token", "tensor")
        or c.learned_kind(s) != "none"
        or s.direction_core is not None
        or s.expert_inputs is not None
        or s.expert_offsets is not None
        or 0 < s.candidates < s.keys_in.shape[0]
        or s.down is None
        or s.up is None
        or depth != int(depth)
        or depth < 1
    ):
        raise NotImplementedError("run_layer_ir does not yet support these imported options")
    chain = depth > 1 and c._import_depth_chain(s)
    if chain and s.exact:
        raise NotImplementedError("exact import depth chains use the native eager loop")
    dtype, device = y0.dtype, y0.device
    kin, kout = _prepare_keys(s)
    coef = c._input_read(s, x, dtype).float()
    if region is not None:
        coef = coef * region.float()
    scale = s.scale.float() if isinstance(s.scale, torch.Tensor) else s.scale
    up = s.up.to(dtype)
    read_cross = c._import_read_cross(s, [up]) if chain else None
    q0 = F.linear(y0.float(), s.router.float())
    cross = up.float() @ s.router.float().t()
    prev = None
    z = None
    carried = None
    hold = coef * scale if s.carry == "dropped" else None
    result = y0
    alive = None
    reference = None
    split_reference = None
    step_set = c._import_depth_settings(s) if chain else s
    metric = c._step_metric(s, device) if chain else None
    adaptive = c.uses_adaptive(s)
    history = {}
    if adaptive:
        norm, rms = c._y0_gate_stats(y0)

    def gated(z, weights, step):
        a, b = _split_gate_grad(
            s,
            z,
            history,
            norm,
            rms,
            weights,
            f"{target}#input{input_place}",
            0,
            step,
            region,
            step == int(depth),
        )
        product = _run(
            "iterate",
            "output",
            {"base": torch.zeros_like(y0), "coefficients": (a.float() * z).to(dtype), "up": up},
            _setting=s,
        )["output"]
        return product * b.to(dtype)

    for step in range(1, int(depth) + 1):
        step_coef = (
            c._import_depth_coefs(s, coef, [carried], [up], region, read_cross)
            if chain and step > 1
            else coef
        )
        if score == "keys":
            q = q0 if carried is None else q0 + carried.float() @ cross
            query = q if prev is None else _project_route(q, kout, prev, s)
            logits = _score(query, kin, s.rows_in, s.tau, s)
        elif score == "derived":
            logits = c.pooled_derived_logits(s, x, target)
        else:
            shape = (*coef.shape[:-1], coef.shape[-1] // s.per_expert, s.per_expert)
            logits = _run(
                "score",
                "read_share",
                {
                    "reads": step_coef.reshape(shape),
                    "temperature": torch.as_tensor(s.tau, dtype=torch.float32, device=device),
                },
            )["logits"]
        inputs = {
            "logits": logits,
            "noise": torch.zeros_like(logits[:, :1] if s.routing_scope == "tensor" else logits),
        }
        if s.routing_scope == "tensor":
            inputs.update(
                region=torch.ones_like(logits[..., 0]) if region is None else region[..., 0],
                has_region=torch.tensor(region is not None, device=device),
            )
        gate = _run("select", f"{s.routing_scope}_{s.routing_select}", inputs, _setting=s)[
            "weights"
        ]
        c._note_routes(target, [gate], [s])
        prev = gate
        part = step_coef * gate.float().repeat_interleave(s.per_expert, dim=-1)
        if alive is not None:
            part = part * alive
        if (
            not chain
            and score in ("read_share", "derived")
            and c._import_fold_enabled()
            and not c._route_freeze.RECORDS
            and depth > 1
            and (s.gate_head_limit is None or not adaptive)
        ):
            if adaptive:
                c._import_fold_history(part, scale, int(depth), s.iteration, region, history, s.up)
                count = 1 if s.iteration == "path" else int(depth)
                return y0 + gated(part * count * scale, gate, int(depth))
            count = 1 if s.iteration == "path" else int(depth)
            return y0 + (part * count * scale).to(dtype) @ up
        if chain and z is not None:
            z = c.depth_split_step(
                z,
                part,
                step_set,
                metric,
                split_reference,
                c._weight_of(scale),
                f"{target}#input{input_place}",
            )
        else:
            z = (
                part
                if z is None or s.iteration == "path"
                else _run(
                    "iterate", "accumulate", {"previous_state": z, "change": part}, _setting=s
                )["state"]
            )
        if chain:
            size = c._metric_size(z, metric, c._weight_of(scale))
            if step == 1:
                reference = size
                split_reference = size
            else:
                ratio = (step_set.radial_budget * reference / size.clamp_min(1e-30)).clamp(max=1)
                z = z * ratio[..., None]
        if adaptive:
            result = y0 + gated(z * scale, gate, step)
        else:
            result = (
                y0 + (z.to(dtype) @ up) * scale if depth == 1 else y0 + (z * scale).to(dtype) @ up
            )
        if step < depth:
            scaled = z * scale
            shape = (*z.shape[:-1], z.shape[-1] // s.per_expert, s.per_expert)
            uniforms = c.dropout_uniforms(
                shape, f"{target}#input{input_place}", step, device, c.column_keys(s)
            )
            variant = "mask_expect" if c.expect_masks() else "mask_sample"
            mask = _read_mask(
                variant,
                {
                    "change": scaled.detach().float().reshape(shape),
                    "base_read": coef.detach().reshape(shape),
                    "uniforms": uniforms,
                },
                s,
            ).reshape_as(z)
            for rec in c._route_freeze.RECORDS:
                rec.setdefault("input_keeps", {})[(target, step)] = {input_place: mask.detach()}
            if chain:
                alive = mask if alive is None else alive * mask
            if hold is None:
                carried = mask * scaled
            else:
                hold = _run(
                    "carry",
                    "dropped",
                    {
                        "change": scaled,
                        "carried": hold,
                        "mask": mask,
                        "base_read": coef,
                        "uniforms": torch.zeros_like(z),
                    },
                    _setting=s,
                )["state"]
                carried = hold
    return result


def _path_route_ir(queries, sets, keys, shared, noise_scale, regions):
    gates = [None] * len(sets)
    for form in ("state", "input"):
        indices = [i for i, setting in enumerate(sets) if setting.form == form]
        if not indices:
            continue
        group = [sets[i] for i in indices]
        logits = [
            _score(queries[i], keys[i][0], sets[i].rows_in, sets[i].tau, sets[i]) for i in indices
        ]
        joint = shared and form == "state" and len(group) > 1
        if (
            joint
            and group[0].routing_scope == "tensor"
            and any(regions[i] is not None for i in indices)
        ):
            means = []
            for i in indices:
                lg, region, setting = logits[indices.index(i)], regions[i], sets[i]
                inputs = {
                    "logits": lg,
                    "region": torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                    "has_region": torch.tensor(region is not None, device=lg.device),
                    "prior": torch.as_tensor(setting.log_prior, dtype=lg.dtype, device=lg.device),
                }
                means.append(_run("select", "region_mean", inputs)["mean_logits"])
            mean = torch.cat(means, -1)
            noise = c.gumbel(mean, noise_scale) if noise_scale > 0 else torch.zeros_like(mean)
            weights = _run(
                "select",
                f"token_{group[0].routing_select}",
                {"logits": mean, "noise": noise},
                _setting=group[0],
            )["weights"]
            split = torch.split(weights, [keys[i][0].shape[0] for i in indices], -1)
            for i, part in zip(indices, split, strict=True):
                gates[i] = part.expand_as(logits[indices.index(i)])
            continue
        if joint:
            logits = [
                torch.cat([lg + s.log_prior for lg, s in zip(logits, group, strict=True)], -1)
            ]
            selected = [group[0]]
            selected_regions = [None]
        else:
            selected = group
            selected_regions = [regions[i] for i in indices]
        chosen = []
        for lg, setting, region in zip(logits, selected, selected_regions, strict=True):
            scope = setting.routing_scope
            shape = lg[:, :1] if scope == "tensor" else lg
            noise = (
                c.gumbel(shape, noise_scale)
                if noise_scale > 0 and form == "state"
                else torch.zeros_like(shape)
            )
            inputs = {"logits": lg, "noise": noise}
            if scope == "tensor":
                inputs.update(
                    region=torch.ones_like(lg[..., 0]) if region is None else region[..., 0],
                    has_region=torch.tensor(region is not None, device=lg.device),
                )
            chosen.append(
                _run("select", f"{scope}_{setting.routing_select}", inputs, _setting=setting)[
                    "weights"
                ]
            )
        if joint:
            chosen = list(torch.split(chosen[0], [keys[i][0].shape[0] for i in indices], -1))
        for i, weights in zip(indices, chosen, strict=True):
            gates[i] = weights
    return gates


def _path_ir(x, y0, sets, depth, shared, noise_scale, grad_steps, target, regions, auto):

    dtype = y0.dtype
    keys = [c._set_keys(s) for s in sets]
    reads = [
        F.linear(y0.to(dtype), s.down.to(dtype))
        if s.form == "state"
        else c._input_read(s, x, dtype)
        for s in sets
    ]
    queries = [F.linear(y0.float(), s.router.float()) for s in sets]
    states, previous = [], []
    previous_node = done = gates = None
    frozen = False
    histories = [{} for _ in sets]
    if any(s.gate_head_limit is not None and c.uses_adaptive(s) for s in sets):
        norm, rms = c._y0_gate_stats(y0)
    for step in range(1, depth + 1):
        keep = grad_steps is None or grad_steps(step)
        with contextlib.nullcontext() if keep else torch.no_grad():
            if frozen:
                queries = [q.detach() for q in queries]
            gates = _path_route_ir(queries, sets, keys, shared, noise_scale, regions)
            new = []
            for s, gate, read in zip(sets, gates, reads, strict=True):
                strength = s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale
                value = _run(
                    "iterate",
                    "path_step",
                    {"read": read, "weight": gate.to(dtype).repeat_interleave(s.per_expert, -1)},
                    context={"strength": strength},
                    _setting=s,
                )["state"]
                new.append(value if keep else value.detach())
            node = torch.cat(gates, -1).detach().argmax(-1)
            if (auto or sets[0].fixed_point_stop) and previous_node is not None:
                same = (node == previous_node).all(-1)
                done = same if done is None else done | same
                new = [
                    _run(
                        "iterate",
                        "path_stop",
                        {
                            "stopped": done[..., None, None].expand_as(z),
                            "previous": old,
                            "state": z,
                        },
                        _setting=s,
                    )["result"]
                    for old, z, s in zip(states, new, sets, strict=True)
                ]
            for i, (s, z) in enumerate(zip(sets, new, strict=True)):
                if step == 1 and c.uses_adaptive(s) and not c.base_reference():
                    up = s.up.detach().float()
                    histories[i]["first_norm"], histories[i]["first_rms"] = c.change_stats(
                        z, up, up @ up.t()
                    )
                if s.gate_head_limit is not None and c.uses_adaptive(s):
                    _split_gate_grad(
                        s, z, histories[i], norm, rms, gates[i], target, i, step, None, True
                    )
            previous, states, previous_node = states, new, node
            queries = [
                _project_route(q, kout, gate, setting)
                for q, (_, kout), gate, setting in zip(queries, keys, gates, sets, strict=True)
            ]
        frozen = not keep
        if done is not None and bool(done.all()):
            break
    y = y0
    gate_rms = None
    for i, (s, z) in enumerate(zip(sets, states, strict=True)):
        delta = _run(
            "iterate", "path_delta", {"coefficients": z, "up": s.up.to(dtype)}, _setting=s
        )["delta"]
        kind = c.learned_kind(s)
        if kind in ("none", "separable") and (kind == "separable" or c.uses_adaptive(s)):
            norm, rms = c._y0_gate_stats(y0)
            before = previous[i] if previous else None
            hist = {"prev": before, "before": before}
            hist.update(
                {
                    key: histories[i][key]
                    for key in ("first_norm", "first_rms")
                    if key in histories[i]
                }
            )
            if s.gate_head_limit is not None and c.uses_adaptive(s):
                hist.update(histories[i])
                hist["prev"] = before
                hist["head_count"] = s.gate_head_limit
            a, b = _split_gate_grad(s, z, hist, norm, rms, gates[i], target, i, step, None, True)
            delta = a.to(dtype) * delta * b.to(dtype)
        elif kind != "none":
            if gate_rms is None:
                with torch.no_grad():
                    gate_rms = y0.float().pow(2).mean(-2, keepdim=True).sqrt() + c.DROPOUT_DELTA
            before = previous[i] @ s.up.to(dtype) if previous else torch.zeros_like(delta)
            mask = c.output_gate(s, delta, y0, before, gates[i], gate_rms, target, i, step)
            if c.uses_adaptive(s):
                norm, rms = c._y0_gate_stats(y0)
                hist = {"before": previous[i] if previous else None}
                hist.update(
                    {
                        key: histories[i][key]
                        for key in ("first_norm", "first_rms")
                        if key in histories[i]
                    }
                )
                if s.gate_head_limit is not None:
                    hist.update(histories[i])
                    hist["head_count"] = s.gate_head_limit
                a, b = c.split_gate(s, z, hist, norm, rms, gates[i], target, i, step, None, True)
                mask = mask * (a * b).to(mask.dtype)
            delta = mask * delta
        if regions[i] is not None:
            delta = regions[i].to(dtype) * delta
        y = _run("iterate", "path_add", {"base": y, "delta": delta}, _setting=s)["output"]
    return y.detach() if frozen else y
