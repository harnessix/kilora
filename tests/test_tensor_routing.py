# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from conftest import _sparsemax, reference_layer
from test_adapter import make, ref_sets
from test_lowrank_depth import _sets

from kilora import compute
from kilora.adapter import Settings
from kilora.compute import (
    ROUTING_SCOPES,
    ROUTING_SELECTS,
    dropout_key,
    read_dropout,
    route_shares,
    run_layer,
    sparsemax,
    tensor_gate,
    token_gate,
)

COMBOS = [(a, b) for a in ROUTING_SCOPES for b in ROUTING_SELECTS]


def _routed(sets, scope, select):
    return [dataclasses.replace(s, routing_scope=scope, routing_select=select) for s in sets]


def _run(y0, sets, depth, fast, monkeypatch, shared=False):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, shared=shared, target="layer")


def test_sparsemax_matches_its_definition():
    z = torch.randn(50, 6, generator=torch.Generator().manual_seed(0)) * 2
    p = sparsemax(z)
    torch.testing.assert_close(p, _sparsemax(z), rtol=0, atol=1e-6)
    torch.testing.assert_close(p.sum(-1), torch.ones(50))
    assert (p == 0).any() and (p >= 0).all()
    one = sparsemax(torch.tensor([3.0, 0.5, -1.0, 0.0]))
    assert torch.equal(one, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_tensor_gate_is_the_same_for_every_token_of_a_sample():
    lg = torch.randn(3, 9, 5, generator=torch.Generator().manual_seed(1))
    p = tensor_gate(lg)
    assert torch.equal(p, p[:, :1].expand_as(p))
    torch.testing.assert_close(p[:, 0], sparsemax(lg.mean(1)))
    assert not torch.equal(p[0, 0], p[1, 0])


def test_token_gate_is_sparsemax_per_token():
    lg = torch.randn(3, 9, 5, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(token_gate(lg), sparsemax(lg))
    assert not torch.equal(token_gate(lg)[:, 0], token_gate(lg)[:, 1])


def test_top1_selection_is_the_arg_max_of_the_scope():
    lg = torch.randn(3, 9, 5, generator=torch.Generator().manual_seed(3))
    with compute.hard_gates():
        tensor = tensor_gate(lg, select="top1")
        token = token_gate(lg, select="top1")
    one = torch.nn.functional.one_hot(lg.mean(1).argmax(-1), 5).float()
    assert torch.equal(tensor, one[:, None].expand_as(lg))
    assert torch.equal(token, torch.nn.functional.one_hot(lg.argmax(-1), 5).float())


def _one_hot_sets(seed=4, scope="tensor", select="sparsemax"):
    sets = _sets(1, 16, 4, 3, 8, seed=seed)
    state = dataclasses.replace(sets[0], tau=4000.0, import_score="derived")
    return _routed([state], scope, select)


def _tensor_top1_reference(y0, s, depth, target="layer#0"):
    r = s.per_expert
    base = y0 @ s.down.t()
    z = torch.zeros_like(base)
    carried = y0
    alive = None
    reference = None
    held = None
    for k in range(1, depth + 1):
        lg = compute.derived_logits(s, carried)
        e = lg.mean(-2).argmax(-1)
        part = torch.zeros_like(z)
        for b in range(len(e)):
            cols = slice(int(e[b]) * r, (int(e[b]) + 1) * r)
            part[b, :, cols] = s.scale * (carried[b] @ s.down[cols].t())
        z = z + (part if alive is None else part * alive)
        size = (z @ s.up).float().norm(dim=-1)
        if reference is None:
            reference = size
        else:
            z = z * (2 * reference / size.clamp_min(1e-30)).clamp(max=1)[..., None]
        y = y0 + z @ s.up
        if k < depth:
            m = read_dropout(z, base, r, target, k)
            alive = m if alive is None else alive * m
            if s.carry == "drop":
                carried = y0 + (z * m) @ s.up
            else:
                old = s.scale * base if held is None else held
                held = old + (1 - m) * (z - old)
                carried = y0 + held @ s.up
    return y


@pytest.mark.parametrize("select", ROUTING_SELECTS)
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("fast", [False, True])
def test_derived_one_hot_weights_equal_tensor_top1_chain(fast, carry, select, monkeypatch):
    sets = [dataclasses.replace(s, carry=carry) for s in _one_hot_sets(select=select)]
    y0 = torch.randn(3, 7, 16, generator=torch.Generator().manual_seed(5))
    seen = []
    real = compute.tensor_gate

    def spy(logits, noise_scale=0.0, region=None, select="sparsemax"):
        p = real(logits, noise_scale, region, select)
        seen.append(p)
        return p

    monkeypatch.setattr(compute, "tensor_gate", spy)
    out = _run(y0, sets, 4, fast, monkeypatch)
    assert seen and all(bool(((p == 0) | (p == 1)).all()) for p in seen)
    want = _tensor_top1_reference(y0, sets[0], 4)
    torch.testing.assert_close(out, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("fast", [False, True])
def test_derived_dominant_logit_per_token_equals_top1_chain(fast, carry, monkeypatch):
    soft = _one_hot_sets(scope="token")
    hard = _one_hot_sets(scope="token", select="top1")
    soft, hard = ([dataclasses.replace(s, carry=carry) for s in sets] for sets in (soft, hard))
    y0 = torch.randn(3, 7, 16, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        lg = compute.derived_logits(soft[0], y0)
    p = token_gate(lg)
    assert bool(((p == 0) | (p == 1)).all())
    assert not torch.equal(p[:, :1].expand_as(p), p)
    a = _run(y0, soft, 4, fast, monkeypatch)
    b = _run(y0, hard, 4, fast, monkeypatch)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("modes", [["accumulate", "accumulate"], ["accumulate", "replace"]])
def test_fast_path_equals_the_loop(shared, modes, scope, select, monkeypatch):
    sets = _sets(2, 16, 4, 2, 8, seed=6, modes=modes, scales=[1.0, -0.7])
    sets = _routed(sets, scope, select)
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(7))
    loop = _run(y0, sets, 5, False, monkeypatch, shared)
    fast = _run(y0, sets, 5, True, monkeypatch, shared)
    torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("scope", "select"), COMBOS)
@pytest.mark.parametrize("fast", [False, True])
def test_replay_is_bit_identical(scope, select, fast, monkeypatch):
    sets = _routed(_sets(2, 16, 4, 2, 8, seed=12), scope, select)
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(13))
    runs = []
    for _ in range(2):
        with dropout_key(0, torch.tensor([0.5])):
            runs.append(_run(y0, sets, 3, fast, monkeypatch))
    assert torch.equal(runs[0], runs[1])


def test_each_sample_routes_on_its_own(monkeypatch):
    sets = _sets(1, 16, 4, 2, 8, seed=8)
    y0 = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(9))
    other = y0.clone()
    other[1] = torch.randn(6, 16, generator=torch.Generator().manual_seed(10)) * 3
    a = _run(y0, sets, 3, True, monkeypatch)
    b = _run(other, sets, 3, True, monkeypatch)
    assert torch.equal(a[0], b[0])


@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize(("scope", "select"), COMBOS)
def test_adapter_matches_the_reference(model, scope, select, carry):
    adapter = make(model, depth=3, routing_scope=scope, routing_select=select, carry=carry)
    for site in adapter.sites:
        for key, p in site.tensors.items():
            if key.endswith("_up"):
                with torch.no_grad():
                    gen = torch.Generator().manual_seed(3)
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    module = model.layers[0].feed_forward.w2
    inp = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        with adapter.depth(0):
            y0 = module(inp)
        out = module(inp)
        ref = reference_layer(
            inp,
            y0,
            ref_sets(adapter),
            3,
            target=adapter.targets[0],
            routing_scope=scope,
            routing_select=select,
            carry=carry,
        )
    assert not torch.allclose(out, y0, atol=1e-2)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_route_shares():
    lg = torch.randn(3, 8, 4, generator=torch.Generator().manual_seed(11))
    tensor = route_shares(lg, "tensor")
    torch.testing.assert_close(tensor, sparsemax(lg.mean(1)).sum(0))
    token = route_shares(lg)
    torch.testing.assert_close(token, sparsemax(lg).reshape(-1, 4).sum(0))
    top1 = route_shares(lg, "token", "top1")
    assert float(top1.sum()) == 24
    assert torch.equal(top1, torch.bincount(lg.argmax(-1).flatten(), minlength=4).float())
    whole = route_shares(lg, "tensor", "top1")
    assert torch.equal(whole, torch.bincount(lg.mean(1).argmax(-1), minlength=4).float())


def test_routing_is_saved(model, tmp_path):
    from kilora import load_mod
    from kilora.modfile import save_mod

    adapter = make(model)
    sets = adapter.to_sets()
    adapter.remove()
    assert (Settings().routing_scope, Settings().routing_select) == ("token", "sparsemax")
    for scope, select in COMBOS:
        path = tmp_path / f"{scope}-{select}.kmod.safetensors"
        save_mod(path, sets, Settings(routing_scope=scope, routing_select=select), id="r")
        st = load_mod(path).settings
        assert (st.routing_scope, st.routing_select) == (scope, select)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noisy", [False, True])
def test_top1_is_exact_and_keeps_the_soft_gradient(dtype, noisy):
    gen = torch.Generator().manual_seed(1947)
    logits = torch.randn(2, 127, 7, generator=gen).to(dtype).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    noise = torch.randn(2, 127, 7, generator=gen).to(dtype) if noisy else None
    weights = torch.randn(2, 127, 7, generator=gen).to(dtype)
    gate = compute.top1_gate(logits, noise)
    choice = (reference if noise is None else reference + noise).argmax(-1, keepdim=True)
    hard = torch.zeros_like(reference).scatter(-1, choice, 1.0)
    assert torch.equal(gate, hard)
    actual_grad = torch.autograd.grad((gate * weights).sum(), logits)[0]
    picked = reference.softmax(-1).gather(-1, choice)
    expected_grad = torch.autograd.grad((picked * weights.gather(-1, choice)).sum(), reference)[0]
    assert torch.equal(actual_grad, expected_grad)
