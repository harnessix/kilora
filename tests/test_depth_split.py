# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute
from kilora.comfyui.patching import StepCarry
from kilora.compute import (
    depth_split_on,
    depth_split_step,
    depth_step_metric,
    dropout_key,
    metric_inner,
    radial_budget_scale,
    recording_routes,
    run_layer,
    step_carry,
)


def _state_sets(**kw):
    return [dataclasses.replace(s, **kw) for s in _sets(2, 16, 4, 2, 8, seed=3)]


def _input_sets(**kw):
    sets = _sets(2, 16, 4, 2, 8, seed=5, scales=[0.7, -0.4])
    return [dataclasses.replace(s, form="input", **kw) for s in sets]


def _xy(seed=4, batch=2):
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, 7, 16, generator=gen),
        torch.randn(batch, 7, 16, generator=gen),
    )


def _calls(sets, depths, carrier=None, token_depth=None):
    outs, states, routes = [], [], []
    for n, depth in enumerate(depths):
        x, y0 = _xy(seed=4 + n)
        t = torch.tensor([float(len(depths) - n)])
        args = {"input": y0, "timestep": t, "c": {}}
        with torch.no_grad(), dropout_key(0, t), recording_routes() as seen:
            if carrier is None:
                out = run_layer(x, y0, sets, depth, target="layer", token_depth=token_depth)
                outs.append(out)
            else:
                cache = carrier.cache_for(args)
                with step_carry(cache):
                    outs.append(run_layer(x, y0, sets, depth, target="layer"))
                states.append({k: v for k, v in cache.items()})
        routes.append({k: [r.clone() for r in v] for k, v in seen.items()})
    return outs, states, routes


def _close(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


CASES = {
    "depth1": dict(depths=[1, 1, 1]),
    "schedule": dict(depths=[1, 1, 3, 3]),
    "fractional": dict(depths=[2.5, 1.5, 3.25]),
    "carry": dict(depths=[3, 3, 2], carry=True),
    "token_depth": dict(
        depths=[3, 3], token_depth=torch.tensor([[0.0, 1, 2, 3, 2.5, 1, 3]]).expand(2, 7)
    ),
    "output_gate": dict(depths=[3, 2.5], gate_mode="adaptive"),
}


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("case", list(CASES))
def test_off_is_off(case, form, monkeypatch):
    if form == "input":
        monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    spec = dict(CASES[case])
    depths = spec.pop("depths")
    carry = spec.pop("carry", False)
    token_depth = spec.pop("token_depth", None)
    make = _state_sets if form == "state" else _input_sets
    if form == "input" and carry:
        pytest.skip("imported sets alone carry nothing across sampler steps")
    plain = make(**spec)
    marked = [
        dataclasses.replace(
            s,
            radial_step=1.0,
            orthogonal_step=1.0,
            radial_budget=0.0,
            step_metric=depth_step_metric(s.up),
        )
        for s in make(**spec)
    ]
    assert not any(depth_split_on(s) for s in marked)
    want = _calls(plain, depths, StepCarry() if carry else None, token_depth)

    def never(*a, **k):
        raise AssertionError("the split ran with default settings")

    monkeypatch.setattr(compute, "depth_split_step", never)
    got = _calls(marked, depths, StepCarry() if carry else None, token_depth)
    assert _equal(got, want)


def test_off_is_off_with_the_read_dropout():
    sets = _state_sets(carry="drop")
    want = _calls(sets, [3, 3])
    marked = [dataclasses.replace(s, step_metric=depth_step_metric(s.up)) for s in sets]
    got = _calls(marked, [3, 3])
    assert _equal(got, want)


def _random(seed=0, cols=8, width=16):
    gen = torch.Generator().manual_seed(seed)
    up = torch.randn(cols, width, generator=gen) / cols**0.5
    z = torch.randn(3, 5, cols, generator=gen)
    part = torch.randn(3, 5, cols, generator=gen) * 0.5
    return up, z, part


def test_the_metric_is_the_output_inner_product():
    up, a, b = _random(1)
    g = depth_step_metric(up)
    want = ((a @ up) * (b @ up)).sum(-1)
    torch.testing.assert_close(metric_inner(a, b, g), want, rtol=1e-5, atol=1e-5)


def _split(z, part, g):
    c = metric_inner(part, z, g) / metric_inner(z, z, g)
    return c, part - c[..., None] * z


def test_the_split_adds_up_and_is_orthogonal():
    up, z, part = _random(2)
    g = depth_step_metric(up)
    c, o = _split(z, part, g)
    torch.testing.assert_close(c[..., None] * z + o, part, rtol=1e-5, atol=1e-5)
    scale = metric_inner(o, o, g).sqrt() * metric_inner(z, z, g).sqrt()
    assert (metric_inner(o, z, g).abs() <= 1e-5 * scale + 1e-7).all()


def _set(up, **kw):
    s = _sets(1, up.shape[1], up.shape[0] // 2, 2, 8, seed=0)[0]
    return dataclasses.replace(s, up=up, **kw)


def test_radial_step_zero_changes_the_size_by_the_orthogonal_part_only():
    up, z, part = _random(3)
    g = depth_step_metric(up)
    new = depth_split_step(z, part, _set(up, radial_step=0.0), g)
    _, o = _split(z, part, g)
    want = metric_inner(z, z, g) + metric_inner(o, o, g)
    torch.testing.assert_close(metric_inner(new, new, g), want, rtol=1e-4, atol=1e-5)


def test_orthogonal_step_zero_keeps_the_direction():
    up, z, part = _random(4)
    g = depth_step_metric(up)
    new = depth_split_step(z, part, _set(up, orthogonal_step=0.0), g)
    c, _ = _split(z, part, g)
    torch.testing.assert_close(new, (1 + c[..., None]) * z, rtol=1e-5, atol=1e-5)


def test_a_zero_state_token_adds_the_part_as_is():
    up, z, part = _random(5)
    z[0, 0] = 0
    g = depth_step_metric(up)
    new = depth_split_step(z, part, _set(up, radial_step=0.5), g)
    assert torch.equal(new[0, 0], part[0, 0])


def _bisect(c, zz, oo, a_r, a_o, limit, steps=80):
    def size2(t):
        return (1 + t * a_r * c) ** 2 * zz + a_o**2 * oo

    lo, hi = torch.zeros_like(c), torch.ones_like(c)
    for _ in range(steps):
        mid = (lo + hi) / 2
        fits = size2(mid) <= limit**2
        lo, hi = torch.where(fits, mid, lo), torch.where(fits, hi, mid)
    return torch.where(size2(torch.ones_like(c)) <= limit**2, torch.ones_like(c), lo)


def test_the_budget_root_matches_a_bisection():
    gen = torch.Generator().manual_seed(6)
    c = torch.rand(400, generator=gen, dtype=torch.float64) * 6 - 3
    zz = torch.rand(400, generator=gen, dtype=torch.float64) + 0.1
    oo = torch.rand(400, generator=gen, dtype=torch.float64) * 0.05
    limit = (zz.sqrt() * (1 + torch.rand(400, generator=gen, dtype=torch.float64))).clamp_min(
        (zz + oo).sqrt() * 1.01
    )
    assert (1 + 1.3 * c < -1).any()
    assert ((c < 0) & (1 + 1.3 * c >= -1)).any() and (c > 0).any()
    t, over = radial_budget_scale(c, zz, oo, 1.3, 0.8, limit)
    assert not over.any()
    torch.testing.assert_close(t, _bisect(c, zz, oo, 1.3, 0.8, limit), rtol=0, atol=1e-9)
    size = ((1 + t * 1.3 * c) ** 2 * zz + 0.64 * oo).sqrt()
    assert (size <= limit * (1 + 1e-9)).all()
    back = (1 + 1.3 * c < -1) & (t < 1)
    assert back.any()


def test_the_budget_cuts_an_overshooting_negative_step_and_counts_overflow():
    c = torch.tensor([-0.5, -3.0, -0.5, -3.0, -3.0, 2.0, 2.0], dtype=torch.float64)
    zz = torch.tensor([1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    oo = torch.tensor([0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 9.0], dtype=torch.float64)
    limit = torch.tensor([1.5, 1.5, 0.1, 1.0, 1.5, 1.5, 1.5], dtype=torch.float64)
    t, over = radial_budget_scale(c, zz, oo, 1.0, 1.0, limit)
    assert t[0] == 1 and not over[0]
    torch.testing.assert_close(t[1], torch.tensor(2.5 / 3, dtype=torch.float64))
    assert not over[1]
    assert t[2] == 1 and over[2]
    torch.testing.assert_close(t[3], torch.tensor(0.5, dtype=torch.float64))
    assert not over[3]
    torch.testing.assert_close(t[4], torch.tensor(1 / 3, dtype=torch.float64))
    assert over[4]
    assert 0 < t[5] < 1 and not over[5]
    assert t[6] == 0 and over[6]


def test_the_budget_keeps_the_state_inside_the_limit():
    up, z, part = _random(7)
    g = depth_step_metric(up)
    s = _set(up, radial_budget=1.0, orthogonal_step=0.0)
    ref = metric_inner(z, z, g).sqrt()
    new = depth_split_step(z, part * 3, s, g, ref)
    assert (metric_inner(new, new, g).sqrt() <= ref * (1 + 1e-5)).all()


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("steps", [(0.0, 1.0), (1.0, 0.0), (0.5, 1.7)])
def test_depth_one_without_carry_ignores_the_step_sizes(form, steps):
    make = _state_sets if form == "state" else _input_sets
    plain = make()
    split = make(radial_step=steps[0], orthogonal_step=steps[1])
    assert all(depth_split_on(s) for s in split)
    assert _close(_calls(split, [1, 1]), _calls(plain, [1, 1]))


@pytest.mark.parametrize("form", ["state", "input"])
def test_later_steps_split(form):
    make = _state_sets if form == "state" else _input_sets
    plain = _calls(make(), [3])[0][0]
    split = _calls(make(radial_step=0.0, radial_budget=1.0), [3])[0][0]
    assert not torch.equal(plain, split)


def test_the_first_step_splits_with_carry():
    plain = _calls(_state_sets(), [1, 1], StepCarry())[0]
    split = _calls(_state_sets(radial_step=0.0), [1, 1], StepCarry())[0]
    assert _close(plain[0], split[0])
    assert not torch.equal(plain[1], split[1])
