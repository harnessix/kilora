# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute
from kilora.compute import dropout_key, dropout_uniforms, read_dropout, run_layer


def _run(y0, sets, depth, fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, target="layer")


def test_keep_rate_matches_keep_probability():
    torch.manual_seed(0)
    tokens, experts, per = 20000, 3, 8
    base = torch.randn(tokens, experts * per)
    z = base * 3 * torch.rand(tokens, 1)
    keep = read_dropout(z, base, per, "layer#0", 1)
    c, c0 = z.unflatten(-1, (experts, per)), base.unflatten(-1, (experts, per))
    sigma0 = c0.pow(2).mean(-1, keepdim=True).sqrt() + compute.DROPOUT_DELTA
    q = 0.5 ** (c / sigma0).abs()
    m = keep.unflatten(-1, (experts, per))
    assert set(keep.unique().tolist()) <= {0.0, 1.0}
    for lo, hi in ((0.1, 0.3), (0.5, 0.6), (0.7, 0.8), (0.9, 1.0)):
        inside = (q >= lo) & (q < hi)
        assert inside.sum() > 2000
        rate, mean_q = float(m[inside].mean()), float(q[inside].mean())
        assert abs(rate - mean_q) < 0.02, (lo, rate, mean_q)


def test_keep_for_a_standard_normal_component_is_about_0_621():
    x = torch.randn(400000, generator=torch.Generator().manual_seed(1))
    q = compute.keep_probability(x, torch.tensor(1.0))
    want = 2 * math.exp(math.log(2) ** 2 / 2) * 0.5 * math.erfc(math.log(2) / math.sqrt(2))
    assert abs(want - 0.6205) < 1e-3 and abs(float(q.mean()) - want) < 3e-3
    keep = read_dropout(x.reshape(-1, 8), x.reshape(-1, 8) * 0 + 1.0, 8, "layer#0", 1)
    assert abs(float(keep.mean()) - want) < 3e-3


def test_straight_through_pushes_large_components_down():
    base = torch.ones(1, 4)
    z = torch.tensor([[0.5, 1.0, 2.0, 3.0]], requires_grad=True)
    with dropout_key(0, 0):
        m = read_dropout(z, base, 4, "st#0", 1)
    (m * z).sum().backward()
    kept = m.detach() == 1
    assert bool(kept.any())
    q = 0.5 ** z.detach()
    want = m.detach() + z.detach() * (-math.log(2)) * q
    torch.testing.assert_close(z.grad, want)
    assert bool((z.grad[kept] < 1).all())


def test_uniforms_are_keyed_and_replay_bit_identical():
    a = dropout_uniforms((4, 6), "layer#0", 2)
    assert torch.equal(a, dropout_uniforms((4, 6), "layer#0", 2))
    assert bool(((a >= 0) & (a < 1)).all())
    assert not torch.equal(a, dropout_uniforms((4, 6), "layer#0", 3))
    assert not torch.equal(a, dropout_uniforms((4, 6), "layer#1", 2))
    with dropout_key(0, 0.5):
        b = dropout_uniforms((4, 6), "layer#0", 2)
        with dropout_key(0, torch.tensor([0.5, 0.5])):
            assert torch.equal(b, dropout_uniforms((4, 6), "layer#0", 2))
    assert not torch.equal(a, b)
    with dropout_key(7, 0.5):
        assert not torch.equal(b, dropout_uniforms((4, 6), "layer#0", 2))


def test_fast_path_equals_loop_and_replays(monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(1))
    sets = _sets(2, 16, 4, 3, 8, seed=2, modes=["accumulate", "replace"], scales=[1.0, -0.7])
    for depth in (1, 3, 6):
        with dropout_key(3, 0.25):
            loop = _run(y0, sets, depth, False, monkeypatch)
            fast = _run(y0, sets, depth, True, monkeypatch)
            again = _run(y0, sets, depth, True, monkeypatch)
        torch.testing.assert_close(fast, loop, rtol=1e-4, atol=1e-5)
        assert torch.equal(fast, again)
        if depth > 1:
            with dropout_key(3, 0.5):
                assert not torch.equal(_run(y0, sets, depth, True, monkeypatch), fast)


def test_central_difference_pairs_share_the_mask(monkeypatch):
    y0 = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(4))
    sets = _sets(1, 16, 4, 2, 8, seed=5)
    eps = 1e-4
    with dropout_key(0, 0.3):
        plus = _run(y0 + eps, sets, 2, True, monkeypatch)
        minus = _run(y0 - eps, sets, 2, True, monkeypatch)
    assert (plus - minus).abs().max() < 0.05


def _no_dropout(z, *args):
    return torch.ones_like(z)


@pytest.mark.parametrize("fast", [False, True])
def test_depth_one_has_no_dropout(fast, monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(1))
    sets = _sets(2, 16, 4, 3, 8, seed=2, modes=["accumulate", "replace"], scales=[1.0, -0.7])
    out = _run(y0, sets, 1, fast, monkeypatch)
    monkeypatch.setattr(compute, "read_dropout", _no_dropout)
    assert torch.equal(out, _run(y0, sets, 1, fast, monkeypatch))


@pytest.mark.parametrize("fast", [False, True])
def test_the_last_step_is_never_masked(fast, monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(3))
    sets = _sets(1, 16, 4, 3, 8, seed=4)
    seen = []
    real = compute.read_dropout

    def spy(z, base, per, target, k, columns=None):
        seen.append(k)
        return real(z, base, per, target, k, columns)

    monkeypatch.setattr(compute, "read_dropout", spy)
    _run(y0, sets, 4, fast, monkeypatch)
    assert seen == [1, 2, 3]


def test_gradients_flow_through_the_unmasked_output():
    sets = _sets(1, 8, 2, 4, 4, seed=6)
    s = sets[0]
    s.up = s.up.clone().requires_grad_(True)
    y0 = torch.randn(3, 8, generator=torch.Generator().manual_seed(7))
    out = run_layer(y0, y0, sets, 1, target="g")
    out.sum().backward()
    coef = y0 @ s.down.t()
    gate = compute._gates(y0, sets, False, 0.0)[0]
    weight = gate.repeat_interleave(s.per_expert, -1)
    torch.testing.assert_close(s.up.grad, (coef * weight).t() @ torch.ones(3, 8))


def _st_sets(seed=8):
    sets = _sets(1, 16, 2, 4, 8, seed=seed)
    sets[0].up = (sets[0].up * 4.0).requires_grad_(True)
    return sets


def test_straight_through_keeps_the_forward_values(monkeypatch):
    y0 = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(9))
    sets = _st_sets()
    with dropout_key(1, 0.5):
        trained = run_layer(y0, y0, sets, 5, target="layer")
        plain = _run(y0, sets, 5, False, monkeypatch)
    assert torch.equal(trained.detach(), plain)
    m = read_dropout(trained, y0, 4, "x#0", 1)
    with torch.no_grad():
        assert torch.equal(m.detach(), read_dropout(trained, y0, 4, "x#0", 1))


def test_straight_through_adds_the_keep_probability_gradient(monkeypatch):
    y0 = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(10))
    sets = _st_sets()
    real = compute.read_dropout
    grads = {}
    for st in (True, False):
        if not st:
            monkeypatch.setattr(compute, "read_dropout", lambda z, *a: real(z.detach(), *a))
        sets[0].up.grad = None
        with dropout_key(1, 0.5):
            run_layer(y0, y0, sets, 5, target="layer").pow(2).mean().backward()
        grads[st] = sets[0].up.grad.clone()
    assert not torch.allclose(grads[True], grads[False])


def test_straight_through_moves_weights_toward_the_stable_component():
    y0 = torch.randn(1, 64, 8, generator=torch.Generator().manual_seed(0)).abs()
    down = torch.eye(8)[:2]
    real = compute.read_dropout

    def train(st):
        a = torch.zeros(2, requires_grad=True)
        for it in range(60):
            w = torch.tensor([0.8, 1.2]) * a.exp()
            s = compute.LayerSet(
                router=torch.zeros(4, 8),
                keys_in=torch.eye(4)[None, :1],
                per_expert=2,
                down=down,
                up=torch.diag(w) @ down,
            )
            with dropout_key(0, it):
                y = run_layer(y0, y0, [s], 8, target="t")
            loss = (y[..., :2].sum(-1) - 20.0).pow(2).mean()
            (grad,) = torch.autograd.grad(loss, a)
            with torch.no_grad():
                a -= 0.002 * grad
        return float(a[0] - a[1])

    compute.LOWRANK_ENABLED, old = False, compute.LOWRANK_ENABLED
    try:
        st = train(True)
        compute.read_dropout = lambda z, *args: real(z.detach(), *args)
        plain = train(False)
    finally:
        compute.read_dropout, compute.LOWRANK_ENABLED = real, old
    assert st > plain + 0.1, (st, plain)


def test_depth_is_not_strength(model, x):
    from test_adapter import make, randomize_up

    adapter = make(model, depth=3)
    randomize_up(adapter)
    with torch.no_grad():
        deep = model(x)
        adapter.set_depth(1)
        adapter.set_strengths([3.0])
        strong = model(x)
    assert not torch.allclose(deep, strong, rtol=1e-3, atol=1e-4)


def test_state_stays_bounded_as_depth_grows(monkeypatch):
    sets = _drop(_sets(1, 16, 2, 4, 8, seed=8))
    sets[0].up = sets[0].up * 6.0
    y0 = torch.randn(4, 7, 16, generator=torch.Generator().manual_seed(9))
    real = compute.read_dropout

    def run(depth, dropout=True):
        carried = [0.0]

        def spy(z, *args):
            m = real(z, *args) if dropout else torch.ones_like(z)
            carried.append(float(((m * z) @ sets[0].up).norm()))
            return m

        monkeypatch.setattr(compute, "read_dropout", spy)
        out = float((_run(y0, sets, depth, True, monkeypatch) - y0).norm())
        monkeypatch.undo()
        return out, max(carried)

    first = run(1, False)[0]
    for depth in (32, 128, 512):
        for dropout in (False, True):
            change, carried = run(depth, dropout)
            assert change <= compute.IMPORT_DEPTH_BUDGET * first * 1.0001
            assert math.isfinite(carried)


def test_dropped_carry_obeys_the_cumulative_bound(monkeypatch):
    sets = _dropped(_sets(1, 16, 2, 4, 8, seed=8))
    sets[0].up = sets[0].up * 6.0
    y0 = torch.randn(4, 7, 16, generator=torch.Generator().manual_seed(9))
    one, deep = ((_run(y0, sets, d, True, monkeypatch) - y0).norm(dim=-1) for d in (1, 32))
    assert bool((deep <= compute.IMPORT_DEPTH_BUDGET * one + 1e-5).all())


def _dropped(sets):
    import dataclasses

    return [dataclasses.replace(s, carry="dropped") for s in sets]


def _drop(sets):
    import dataclasses

    return [dataclasses.replace(s, carry="drop") for s in sets]


@pytest.mark.parametrize("fast", [False, True])
def test_dropped_carry_depth_one_is_unchanged(fast, monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(21))
    sets = _sets(2, 16, 4, 3, 8, seed=22, modes=["accumulate", "replace"], scales=[1.0, -0.7])
    assert torch.equal(
        _run(y0, _dropped(sets), 1, fast, monkeypatch), _run(y0, _drop(sets), 1, fast, monkeypatch)
    )


def test_dropped_carry_fast_path_equals_loop_and_replays(monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(23))
    sets = _sets(2, 16, 4, 3, 8, seed=24, modes=["accumulate", "replace"], scales=[1.0, -0.7])
    held = _dropped(sets)
    for depth in (2, 4, 7):
        with dropout_key(3, 0.25):
            loop = _run(y0, held, depth, False, monkeypatch)
            fast = _run(y0, held, depth, True, monkeypatch)
            again = _run(y0, held, depth, True, monkeypatch)
            drop = _run(y0, _drop(sets), depth, True, monkeypatch)
        torch.testing.assert_close(fast, loop, rtol=1e-4, atol=1e-5)
        assert torch.equal(fast, again)
        assert not torch.allclose(fast, drop)


def test_dropped_carry_moves_only_dropped_components(monkeypatch):
    y0 = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(25))
    s = _dropped(_drop(_sets(1, 16, 1, 4, 8, seed=26)))[0]
    c0 = y0 @ s.down.t()
    for keep in (0.0, 1.0):
        monkeypatch.setattr(
            compute, "read_dropout", lambda z, *a, keep=keep, **k: torch.full_like(z, keep)
        )
        hold, z, alive = c0.clone(), torch.zeros_like(c0), 1.0
        for k in range(1, 4):
            state = y0 if k == 1 else y0 + hold @ s.up
            z = z + (state @ s.down.t()) * alive
            size = (z @ s.up).norm(dim=-1, keepdim=True)
            if k == 1:
                reference = size
            else:
                ratio = compute.IMPORT_DEPTH_BUDGET * reference / size.clamp_min(1e-30)
                z = z * ratio.clamp(max=1)
            hold = hold + (1 - keep) * (z - hold)
            alive *= keep
        want = y0 + z @ s.up
        for fast in (False, True):
            torch.testing.assert_close(
                _run(y0, [s], 3, fast, monkeypatch), want, rtol=1e-5, atol=1e-6
            )


def test_dropped_carry_has_the_straight_through_gradient():
    y0 = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(27))
    sets = _dropped(_st_sets())
    with dropout_key(1, 0.5):
        out = run_layer(y0, y0, sets, 4, target="layer")
    out.pow(2).mean().backward()
    assert sets[0].up.grad is not None and bool(torch.isfinite(sets[0].up.grad).all())


@pytest.mark.parametrize("fast", [False, True])
def test_detailed_keep_observer_identifies_set_and_step_without_changing_output(fast, monkeypatch):
    y0 = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(72))
    sets = _sets(2, 16, 4, 3, 8, seed=2)
    sets[0].key, sets[1].key = "first", "second"
    expected = _run(y0, sets, 3, fast, monkeypatch)
    observed = []
    with compute.recording_read_keep_detail(lambda *args: observed.append(args)):
        actual = _run(y0, sets, 3, fast, monkeypatch)
    assert torch.equal(actual, expected)
    assert {(name, step, key) for name, step, key, _ in observed} == {
        (f"layer#{i}", step, key) for i, key in enumerate(("first", "second")) for step in (1, 2)
    }
    assert all(torch.all((rates >= 0) & (rates <= 1)) for _, _, _, rates in observed)


def test_detailed_keep_observations_match_fast_and_loop(monkeypatch):
    y0 = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(73))
    sets = _sets(2, 16, 4, 3, 8, seed=3)
    sets[0].key, sets[1].key = "first", "second"
    observations = []
    outputs = []
    for fast in (False, True):
        current = []
        with compute.recording_read_keep_detail(
            lambda *args, current=current: current.append(args)
        ):
            outputs.append(_run(y0, sets, 3, fast, monkeypatch))
        observations.append(current)
    assert torch.allclose(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
    for left, right in zip(*observations, strict=True):
        assert left[:3] == right[:3]
        assert torch.allclose(left[3], right[3], atol=1e-5, rtol=1e-5)
