# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from kilora import compute
from kilora.compute import (
    LayerSet,
    carry_transfer,
    current_progress,
    layer_kind,
    layer_places,
    run_layer,
    sampling_progress,
    step_carry,
)


class _Schedule:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def depth(self, index, count, kind, t, set_key):
        self.calls.append((index, count, kind, t, set_key))
        return float(self.fn(index, count, kind, t, set_key))


def _sets(
    n_sets=2,
    width=32,
    experts=4,
    per_expert=2,
    router_dim=8,
    seed=0,
    carry="dropped",
    modes=None,
    gate=False,
):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_sets):
        cols = experts * per_expert
        out.append(
            LayerSet(
                import_score="keys",
                router=torch.randn(router_dim, width, generator=gen) / width**0.5,
                keys_in=torch.randn(experts, 4, router_dim, generator=gen),
                per_expert=per_expert,
                down=torch.randn(cols, width, generator=gen) / width**0.5,
                up=torch.randn(cols, width, generator=gen) * 0.3 / cols**0.5,
                scale=[1.0, -0.6, 1.3][i],
                aggregation=(modes or ["accumulate"] * n_sets)[i],
                carry=carry,
                key=f"s{i}",
            )
        )
    return out


def _y0(seed=1):
    return torch.randn(2, 9, 32, generator=torch.Generator().manual_seed(seed))


def _run(y0, sets, depth, fast, monkeypatch, **kw):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0, y0, sets, depth, target="blk", **kw)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_no_schedule_and_whole_schedules_are_bit_identical(fast, carry, monkeypatch):
    sets = _sets(carry=carry, modes=["accumulate", "replace"])
    y0 = _y0()
    base = _run(y0, sets, 3, fast, monkeypatch)
    assert torch.equal(base, _run(y0, sets, 3.0, fast, monkeypatch))
    flat = _Schedule(lambda *a: 3)
    assert torch.equal(base, _run(y0, sets, 1, fast, monkeypatch, schedule=flat))
    assert torch.equal(base, _run(y0, sets, 3, fast, monkeypatch, schedule=None))


@pytest.mark.parametrize("fast", [False, True])
def test_depth_zero_gives_the_base_output(fast, monkeypatch):
    sets = _sets()
    y0 = _y0()
    off = _Schedule(lambda *a: 0)
    out = _run(y0, sets, 4, fast, monkeypatch, schedule=off)
    assert torch.equal(out, y0)
    assert torch.equal(_run(y0, sets, 0, fast, monkeypatch), y0)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("d", [0.25, 1.5, 2.75, 4.1])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_fractional_depth_interpolates_floor_and_ceil(fast, d, carry, monkeypatch):
    sets = _sets(carry=carry, modes=["accumulate", "replace"])
    y0 = _y0(seed=3)
    lo = _run(y0, sets, math.floor(d), fast, monkeypatch)
    hi = _run(y0, sets, math.ceil(d), fast, monkeypatch)
    out = _run(y0, sets, d, fast, monkeypatch)
    f = d - math.floor(d)
    assert torch.allclose(out, lo + f * (hi - lo), rtol=1e-5, atol=1e-6)
    sched = _run(y0, sets, 1, fast, monkeypatch, schedule=_Schedule(lambda *a: d))
    assert torch.equal(out, sched)


@pytest.mark.parametrize("d", [0.4, 1.0, 2.5, 3.3])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_fast_path_equals_the_loop_for_fractional_depths(d, carry, monkeypatch):
    sets = _sets(n_sets=3, carry=carry, modes=["accumulate", "replace", "accumulate"])
    y0 = _y0(seed=5)
    loop = _run(y0, sets, d, False, monkeypatch, shared=True)
    fast = _run(y0, sets, d, True, monkeypatch, shared=True)
    assert torch.allclose(loop, fast, rtol=1e-4, atol=1e-5), (loop - fast).abs().max()


def test_path_iteration_blends_its_two_depths(monkeypatch):
    import dataclasses

    sets = [dataclasses.replace(s, iteration="path") for s in _sets()]
    y0 = _y0(seed=7)
    lo = _run(y0, sets, 2, False, monkeypatch)
    hi = _run(y0, sets, 3, False, monkeypatch)
    out = _run(y0, sets, 2.25, False, monkeypatch)
    assert torch.allclose(out, lo + 0.25 * (hi - lo), rtol=1e-5, atol=1e-6)


def _carried_run(sets, depths, fast, monkeypatch, schedule=None):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    cache, outs, states = {}, [], []
    with step_carry(cache):
        for n, d in enumerate(depths):
            y0 = _y0(seed=10 + n)
            with torch.no_grad(), sampling_progress(n / max(len(depths) - 1, 1)):
                outs.append(run_layer(y0, y0, sets, d, target="blk", schedule=schedule))
            entry = cache["blk"]
            states.append(
                (
                    [t.clone() for t in entry["carried"]],
                    [t.clone() for t in entry["change"]],
                    entry["depth"],
                )
            )
    return outs, states


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_carry_across_steps_blends_the_change(fast, carry, monkeypatch):
    sets = _sets(carry=carry)
    _, lo = _carried_run(sets, [2, 1], fast, monkeypatch)
    _, hi = _carried_run(sets, [2, 2], fast, monkeypatch)
    outs, mid = _carried_run(sets, [2, 1.5], fast, monkeypatch)
    for a, b, m in zip(lo[1][1], hi[1][1], mid[1][1], strict=True):
        assert torch.allclose(m, a + 0.5 * (b - a), rtol=1e-5, atol=1e-6)
    y0 = _y0(seed=11)
    for s, z, c in zip(sets, mid[1][1], mid[1][0], strict=True):
        want = carry_transfer(z, y0.float() @ s.down.float().t(), s.per_expert)
        assert torch.allclose(c, want, rtol=1e-5, atol=1e-6)
    assert mid[1][2] == 3.5
    outs_fast, _ = _carried_run(sets, [2, 1.5, 2.5], True, monkeypatch)
    outs_loop, _ = _carried_run(sets, [2, 1.5, 2.5], False, monkeypatch)
    for a, b in zip(outs_fast, outs_loop, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_carry_with_a_schedule_gets_the_depth_per_sampler_step(monkeypatch):
    sets = _sets()
    sched = _Schedule(lambda i, n, kind, t, key: 1 + 2 * t)
    outs, states = _carried_run(sets, [9, 9, 9], False, monkeypatch, schedule=sched)
    ref, ref_states = _carried_run(sets, [1, 2, 3], False, monkeypatch)
    for a, b in zip(outs, ref, strict=True):
        assert torch.equal(a, b)
    assert [s[2] for s in states] == [1, 3, 6]
    assert [c[3] for c in sched.calls] == [0.0, 0.5, 1.0]
    assert {c[4] for c in sched.calls} == {"s0"}


def test_progress_and_layer_places():
    assert current_progress() == 0.0
    with sampling_progress(0.3):
        assert current_progress() == pytest.approx(0.3)
        with sampling_progress(2.0):
            assert current_progress() == 1.0
    targets = [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
        "transformer_blocks.3.norm1.linear",
    ]
    places = layer_places(targets)
    assert places[targets[0]] == (0, 3, "self_attn")
    assert places[targets[1]] == (0, 3, "cross_attn")
    assert places[targets[2]] == (0, 3, "mlp")
    assert places[targets[3]] == (1, 3, "self_attn")
    assert places[targets[4]] == (2, 3, "other")
    assert layer_kind("blocks.2.mlp.fc1") == "mlp"


def test_adapter_passes_place_kind_and_progress(monkeypatch):
    seen = []
    real = compute.run_layer

    def spy(*args, **kw):
        seen.append({k: kw.get(k) for k in ("schedule", "index", "count", "kind")})
        return real(*args, **kw)

    from kilora import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "run_layer", spy)
    model, adapter = _tiny_adapter()
    sched = _Schedule(lambda i, n, kind, t, key: 0 if kind == "mlp" else 1.5)
    adapter.set_depth_schedule(sched)
    model.eval()
    with torch.no_grad(), sampling_progress(0.5):
        model(torch.randn(1, 3, 16))
    assert seen and all(s["schedule"] is sched for s in seen)
    assert {(c[2], c[3]) for c in sched.calls} <= {("self_attn", 0.5), ("mlp", 0.5)}
    assert {c[1] for c in sched.calls} == {2}


def _tiny_adapter():
    from conftest import randomize_up
    from torch import nn

    from kilora import KiLoRAConfig, attach

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = nn.Linear(16, 16)
            self.ff = nn.Linear(16, 16)

        def forward(self, x):
            return self.ff(self.attn1(x))

    torch.manual_seed(0)
    model = nn.Sequential(Block(), Block())
    config = KiLoRAConfig(target_modules=["*.attn1", "*.ff"], num_experts=2, rank=2, router_dim=4)
    adapter = attach(model, config)
    randomize_up(adapter)
    return model, adapter


@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("d", [0.5, 2.25, 3.75])
def test_fractional_depth_with_separable_gates_fast_equals_loop(carry, d, monkeypatch):
    import dataclasses

    from kilora.compute import dropout_key, new_gate

    gen = torch.Generator().manual_seed(4)
    sets = [
        dataclasses.replace(
            s,
            output_gate="separable",
            gate=new_gate("separable", s.keys_in.shape[0], s.up.shape[1], gen, s.up.shape[0]),
        )
        for s in _sets(carry=carry)
    ]
    y0 = _y0(seed=9)
    with dropout_key(1, 0.4):
        loop = _run(y0, sets, d, False, monkeypatch)
        fast = _run(y0, sets, d, True, monkeypatch)
        lo = _run(y0, sets, math.floor(d), False, monkeypatch)
        hi = _run(y0, sets, math.ceil(d), False, monkeypatch)
    torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-5)
    f = d - math.floor(d)
    torch.testing.assert_close(loop, lo + f * (hi - lo), rtol=1e-5, atol=1e-5)


class _Auto(_Schedule):
    is_auto = True

    def __init__(self, value=16):
        super().__init__(lambda *a: value)


def _fixed_point_set():
    import dataclasses

    s = _sets(1, 16, 4, 2, 8, seed=14)[0]
    s = dataclasses.replace(
        s,
        down=torch.cat([torch.eye(8), torch.zeros(8, 8)], -1),
        up=torch.cat([torch.zeros(8, 8), torch.eye(8)], -1) * 0.05,
        tau=200.0,
        iteration="path",
        routing_select="top1",
    )
    y0 = torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(16))
    return s, y0


@pytest.mark.parametrize("fast", [False, True])
def test_auto_stops_early_at_a_fixed_point(fast, monkeypatch):
    from kilora.compute import recording_routes

    s, y0 = _fixed_point_set()
    assert not s.fixed_point_stop
    with recording_routes() as auto:
        out = _run(y0, [s], 1, fast, monkeypatch, schedule=_Auto())
    with recording_routes() as walk:
        full = _run(y0, [s], 16, fast, monkeypatch)
    assert len(walk["blk"]) == 16
    assert len(auto["blk"]) <= 3
    torch.testing.assert_close(out, full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("value", [16, 40, 2.5])
def test_auto_never_runs_more_than_16_steps(value, monkeypatch):
    import dataclasses

    from kilora.compute import AUTO_DEPTH_CAP, recording_routes

    assert AUTO_DEPTH_CAP == 16
    s, y0 = _fixed_point_set()
    s = dataclasses.replace(s, keys_out=None, tau=1.0)
    with recording_routes() as rec:
        _run(y0, [s], 1, True, monkeypatch, schedule=_Auto(value))
    assert 1 <= len(rec["blk"]) <= min(16, math.ceil(value))
    acc = _sets()
    with recording_routes() as rec:
        _run(_y0(), acc, 1, False, monkeypatch, schedule=_Auto(value))
    assert len(rec["blk"]) == min(16, math.ceil(value))


@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_auto_fast_path_equals_the_loop(carry, monkeypatch):
    sets = _sets(carry=carry, modes=["accumulate", "replace"])
    y0 = _y0()
    loop = _run(y0, sets, 1, False, monkeypatch, schedule=_Auto())
    fast = _run(y0, sets, 1, True, monkeypatch, schedule=_Auto())
    torch.testing.assert_close(fast, loop, rtol=1e-4, atol=1e-5)
    assert torch.equal(loop, _run(y0, sets, 16, False, monkeypatch))
    s, p0 = _fixed_point_set()
    a = _run(p0, [s], 1, False, monkeypatch, schedule=_Auto())
    b = _run(p0, [s], 1, True, monkeypatch, schedule=_Auto())
    assert torch.equal(a, b)
