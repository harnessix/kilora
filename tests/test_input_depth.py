# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute
from kilora.compute import _gates, dropout_key, run_layer

COMBOS = [(a, b) for a in ("token", "tensor") for b in ("sparsemax", "top1")]


def _input_sets(n=2, seed=3, **kw):
    sets = _sets(n, 16, 4, 2, 8, seed=seed, scales=[1.0, -0.6][:n])
    return [dataclasses.replace(s, form="input", **kw) for s in sets]


def _xy(seed=4):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 7, 16, generator=gen), torch.randn(2, 7, 16, generator=gen)


def _run(x, y0, sets, depth, fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad(), dropout_key(0, torch.tensor([0.4])):
        return run_layer(x, y0, sets, depth, target="layer")


@pytest.mark.parametrize("fast", [False, True])
def test_depth_one_is_the_single_application_bit_for_bit(fast, monkeypatch):
    sets = _input_sets()
    x, y0 = _xy()
    with torch.no_grad():
        want = y0
        for s, gate in zip(sets, _gates(y0, sets, False, 0.0), strict=True):
            weight = gate.repeat_interleave(s.per_expert, dim=-1)
            want = want + ((torch.nn.functional.linear(x, s.down) * weight) @ s.up) * s.scale
    assert torch.equal(_run(x, y0, sets, 0, fast, monkeypatch), y0)
    assert torch.equal(_run(x, y0, sets, 1, fast, monkeypatch), want)


@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_change_stays_bounded_over_depth_seven(iteration, monkeypatch):
    sets = _input_sets(n=1, iteration=iteration)
    x, y0 = _xy(seed=5)
    s = sets[0]
    with torch.no_grad():
        a = torch.nn.functional.linear(x, s.down).reshape(2, 7, 4, s.per_expert)
        up = s.up.reshape(4, s.per_expert, 16)
        per_expert = torch.einsum("btec,ecw->btew", a, up).norm(dim=-1)
        peak = per_expert.max(-1).values * abs(s.scale)
    for k in range(1, 8):
        y = _run(x, y0, sets, k, False, monkeypatch)
        bound = peak * (k if iteration == "accumulate" else 1)
        assert bool(((y - y0).norm(dim=-1) <= bound * (1 + 1e-5) + 1e-6).all()), k


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("depth", [3, 2.5])
def test_fast_path_equals_the_loop(scope, select, iteration, carry, depth, monkeypatch):
    sets = _input_sets(routing_scope=scope, routing_select=select, iteration=iteration, carry=carry)
    x, y0 = _xy(seed=6)
    loop = _run(x, y0, sets, depth, False, monkeypatch)
    fast = _run(x, y0, sets, depth, True, monkeypatch)
    torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-6)
    if iteration == "accumulate":
        assert not torch.allclose(loop, _run(x, y0, sets, 1, False, monkeypatch))


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("fast", [False, True])
def test_replay_is_bit_identical(scope, select, fast, monkeypatch):
    sets = _input_sets(routing_scope=scope, routing_select=select)
    x, y0 = _xy(seed=7)
    a = _run(x, y0, sets, 4, fast, monkeypatch)
    b = _run(x, y0, sets, 4, fast, monkeypatch)
    assert torch.equal(a, b)


def test_routes_are_recorded_at_every_step():
    sets = _input_sets()
    x, y0 = _xy(seed=8)
    with torch.no_grad(), compute.recording_routes() as routes:
        run_layer(x, y0, sets, 3, target="layer")
    assert len(routes["layer"]) == 3


def _mixed(scope="token", select="sparsemax", carry="dropped", seed=9):
    state = _sets(1, 16, 4, 2, 8, seed=seed)[0]
    (inp,) = _input_sets(n=1, seed=seed + 1)
    return [
        dataclasses.replace(s, routing_scope=scope, routing_select=select, carry=carry)
        for s in (state, inp)
    ]


def test_mixed_layer_at_depth_one_selects_jointly(monkeypatch):
    sets = _mixed()
    x, y0 = _xy(seed=10)
    both = _run(x, y0, sets, 1, False, monkeypatch)
    with torch.no_grad():
        gates = compute._mixed_route(y0, sets, None, None, False, 0.0, [None, None], True, x=x)
        torch.testing.assert_close(sum(g.sum(-1) for g in gates), torch.ones_like(gates[0][..., 0]))
        want = y0
        for s, gate in zip(sets, gates, strict=True):
            read = torch.nn.functional.linear(y0 if s.form == "state" else x, s.down)
            z = read * gate.repeat_interleave(s.per_expert, -1) * s.scale
            want = want + z @ s.up
    torch.testing.assert_close(both, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("depth", [3, 2.5])
def test_mixed_fast_path_equals_the_loop(scope, select, carry, depth, monkeypatch):
    sets = _mixed(scope, select, carry)
    x, y0 = _xy(seed=11)
    loop = _run(x, y0, sets, depth, False, monkeypatch)
    fast = _run(x, y0, sets, depth, True, monkeypatch)
    torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fast", [False, True])
def test_mixed_steps_share_one_carried_state(fast, monkeypatch):
    sets = _mixed(seed=12)
    x, y0 = _xy(seed=13)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    key = torch.tensor([0.4])
    with torch.no_grad(), compute.recording_routes() as with_state, dropout_key(0, key):
        run_layer(x, y0, sets, 4, target="layer")
    stronger = [dataclasses.replace(sets[0], scale=8.0), sets[1]]
    with torch.no_grad(), compute.recording_routes() as with_strong, dropout_key(0, key):
        run_layer(x, y0, stronger, 4, target="layer")
    first = [r[..., 1] for r in with_state["layer"]]
    other = [r[..., 1] for r in with_strong["layer"]]
    assert torch.equal(first[0], other[0])
    assert any(not torch.equal(a, b) for a, b in zip(first[1:], other[1:], strict=True))
    a = _run(x, y0, sets, 4, fast, monkeypatch)
    assert torch.equal(a, _run(x, y0, sets, 4, fast, monkeypatch))


def _path(sets):
    return [dataclasses.replace(s, iteration="path") for s in sets]


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("depth", [1, 3, 2.5])
def test_mixed_path_walks_both_forms_together(scope, select, depth, monkeypatch):
    import math

    sets = _path(_mixed(scope, select))
    x, y0 = _xy(seed=14)
    both = _run(x, y0, sets, depth, False, monkeypatch)
    with torch.no_grad(), dropout_key(0, torch.tensor([0.4])):
        carried = [None, None]
        holds = [
            torch.nn.functional.linear(y0 if s.form == "state" else x, s.down) * s.scale
            for s in sets
        ]
        previous = None
        outputs = [y0]
        for k in range(1, math.ceil(depth) + 1):
            state = y0
            for c, s in zip(carried, sets, strict=True):
                if c is not None:
                    state = state + c @ s.up
            gates = compute._mixed_route(
                state, sets, None, previous, False, 0.0, [None, None], True, x=x
            )
            torch.testing.assert_close(
                sum(g.sum(-1) for g in gates), torch.ones_like(gates[0][..., 0])
            )
            z = [
                torch.nn.functional.linear(state if s.form == "state" else x, s.down)
                * g.repeat_interleave(s.per_expert, -1)
                * s.scale
                for s, g in zip(sets, gates, strict=True)
            ]
            out = y0
            for value, s in zip(z, sets, strict=True):
                out = out + value @ s.up
            outputs.append(out)
            if k < math.ceil(depth):
                for i, (s, value) in enumerate(zip(sets, z, strict=True)):
                    base = torch.nn.functional.linear(y0 if s.form == "state" else x, s.down)
                    name = f"layer#{i}" if s.form == "state" else f"layer#input{i}"
                    keep = compute._read_dropout_for_set(
                        value, base, s.per_expert, name, k, compute.column_keys(s), s.key
                    )
                    holds[i] = holds[i] + (1 - keep) * (value - holds[i])
                    carried[i] = holds[i]
            previous = gates
        low = math.floor(depth)
        want = (
            outputs[low]
            if low == depth
            else outputs[low] + (depth - low) * (outputs[-1] - outputs[low])
        )
    torch.testing.assert_close(both, want, rtol=1e-5, atol=1e-6)
    fast = _run(x, y0, sets, depth, True, monkeypatch)
    torch.testing.assert_close(fast, both, rtol=1e-5, atol=1e-6)
    assert torch.equal(both, _run(x, y0, sets, depth, False, monkeypatch))


def test_mixed_path_routes_are_recorded_for_both_forms():
    sets = _path(_mixed())
    x, y0 = _xy(seed=15)
    with torch.no_grad(), compute.recording_routes() as routes:
        run_layer(x, y0, sets, 3, target="layer")
    assert len(routes["layer"]) == 3 and routes["layer"][0].shape[-1] == 2


def _expert_masks(n_experts, tokens, on):
    m = torch.zeros(2, tokens, 1)
    m[:, on] = 1.0
    return [m for _ in range(n_experts)]


def test_expert_regions_limit_each_expert_to_its_tokens(monkeypatch):
    (s,) = _input_sets(n=1)
    x, y0 = _xy(seed=16)
    with torch.no_grad():
        plain = run_layer(x, y0, [s], 1, target="layer")
        off = run_layer(
            x, y0, [s], 1, target="layer", expert_regions=[_expert_masks(4, 7, slice(0, 0))]
        )
        part = run_layer(
            x, y0, [s], 1, target="layer", expert_regions=[_expert_masks(4, 7, slice(0, 3))]
        )
    torch.testing.assert_close(off, y0)
    torch.testing.assert_close(part[:, 3:], y0[:, 3:])
    torch.testing.assert_close(part[:, :3], plain[:, :3])


@pytest.mark.parametrize("depth", [1, 3])
def test_expert_regions_fast_path_equals_the_loop(depth, monkeypatch):
    sets = _mixed()
    x, y0 = _xy(seed=17)
    regions = [None, [None, None, None, None]]
    regions[1][1] = torch.zeros(2, 7, 1)
    regions[1][1][:, 2:5] = 1.0
    outs = []
    for fast in (False, True):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
        with torch.no_grad(), dropout_key(0, torch.tensor([0.4])):
            outs.append(run_layer(x, y0, sets, depth, target="layer", expert_regions=regions))
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-5, atol=1e-6)


def test_key_probes_count_only_the_tokens_a_set_may_change():
    from kilora.compute import key_probes
    from kilora.subspace_keys import keys_from_records

    (s,) = _input_sets(n=1)
    x, y0 = _xy(seed=18)
    region = torch.zeros(2, 7, 1)
    region[:, :3] = 1.0
    other = y0.clone()
    other[:, 3:] = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(19)) * 5
    found = []
    for base in (y0, other):
        with torch.no_grad(), key_probes() as record:
            run_layer(x, base, [s], 2, target="layer", regions=[region])
        assert record["layer"][0]["masks"][0] is not None
        current = {
            "layer": (
                s.keys_in,
                None,
                torch.full((4,), s.keys_in.shape[1]),
                torch.full((4,), s.keys_in.shape[1]),
            )
        }
        found.append(keys_from_records(record, {"layer": 0}, current, 2, 2)["layer"])
    for part in ("keys_in", "keys_out"):
        torch.testing.assert_close(found[0][part], found[1][part], rtol=1e-4, atol=1e-5)
