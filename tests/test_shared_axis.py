# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_axis_alphas import _set
from test_depth_forward import _sets, _y0

from kilora import compute
from kilora.call_values import active_axes, axis_alphas
from kilora.compute import run_layer


def _run(y, sets, **kwargs):
    return run_layer(y, y, sets, 1, target="layer", **kwargs)


def test_no_shared_matches_previous_axis_mix_exactly():
    y = _y0()[..., :4]
    a, b = _set("creative"), _set("ip")
    one, two = _run(y, [a]), _run(y, [b])
    expected = y + (one - y)
    expected = expected + (two - y)
    assert torch.equal(_run(y, [a, b]), expected)
    with axis_alphas({"creative": 0.25, "ip": 0.75}):
        expected = y + (one - y) * 0.25
        expected = expected + (two - y) * 0.75
        assert torch.equal(_run(y, [a, b]), expected)


@pytest.mark.parametrize("score", ["keys", "read_share", "derived"])
def test_shared_input_competes_and_alphas_bound_its_change(score):
    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(7))
    shared = dataclasses.replace(
        _set("ip"), axis="shared", import_score=score, routing_select="top1"
    )
    a = dataclasses.replace(shared, axis="creative", up=torch.zeros_like(shared.up))
    b = dataclasses.replace(a, axis="ip")
    solo = _run(y, [shared])
    alpha = torch.linspace(0, 1, 6).expand(2, -1)
    with axis_alphas({"creative": alpha, "ip": 1 - alpha}):
        out = _run(y, [shared, a, b])
    torch.testing.assert_close(out, solo, rtol=1e-5, atol=1e-6)
    assert (out - y).abs().max() > 0
    with active_axes(["ip"]), axis_alphas({"ip": 0.4}):
        selected = _run(y, [shared, a, b])
    torch.testing.assert_close(selected, y + 0.4 * (solo - y))


def test_shared_state_fast_and_loop_agree(monkeypatch):
    y = _y0()
    a, b, c = _sets(3)
    sets = [dataclasses.replace(a, axis="shared"), b, dataclasses.replace(c, axis="ip")]
    with torch.no_grad(), axis_alphas({"creative": 0.3, "ip": 0.7}):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        loop = run_layer(y, y, sets, 2, target="layer")
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = run_layer(y, y, sets, 2, target="layer")
    torch.testing.assert_close(fast, loop, rtol=1e-4, atol=1e-5)


def test_axis_gate_selects_specific_or_shared_expert():
    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(11))
    a = dataclasses.replace(_set("creative"), routing_select="top1")
    shared = dataclasses.replace(_set("ip"), axis="shared", routing_select="top1")
    b = dataclasses.replace(a, axis="ip", up=torch.zeros_like(a.up))
    with active_axes(["creative"]):
        creative = _run(y, [a, shared, b])
    with active_axes(["ip"]):
        ip = _run(y, [a, shared, b])
    torch.testing.assert_close(creative, _run(y, [a]))
    torch.testing.assert_close(ip, _run(y, [shared]))
    with axis_alphas({"creative": 0.2, "ip": 0.8}):
        mixed = _run(y, [a, shared, b])
    torch.testing.assert_close(mixed, y + 0.2 * (creative - y) + 0.8 * (ip - y))


def test_shared_input_fast_and_loop_agree_at_multiple_steps(monkeypatch):
    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(12))
    a = dataclasses.replace(_set("creative"), import_score="derived")
    shared = dataclasses.replace(_set("ip"), axis="shared", import_score="derived")
    with torch.no_grad():
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        loop = run_layer(y, y, [a, shared], 2, target="layer")
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = run_layer(y, y, [a, shared], 2, target="layer")
    torch.testing.assert_close(fast, loop, rtol=1e-4, atol=1e-5)


def test_shared_axis_route_replay_is_identical():
    from kilora import route_freeze
    from kilora.compute import sampling_progress

    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(13))
    a = dataclasses.replace(_set("creative"), route_freeze=0.0)
    shared = dataclasses.replace(_set("ip"), axis="shared", route_freeze=0.0)
    with torch.no_grad(), axis_alphas({"creative": 0.5}):
        plain = _run(y, [a, shared])
        with route_freeze.freezing({}), sampling_progress(0.9):
            first = _run(y, [a, shared])
            replay = _run(y, [a, shared])
    assert torch.equal(first, plain)
    assert torch.equal(replay, plain)


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_joint_import_gate_restricts_each_set_to_candidates(scope, monkeypatch):
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    keys = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]])
    first = dataclasses.replace(
        _set("creative"),
        candidates=1,
        routing_scope=scope,
        import_score="derived",
        routing_select="top1",
        keys_in=keys,
    )
    second = dataclasses.replace(first, axis="shared")
    monkeypatch.setattr(compute, "derived_keys", lambda s: (keys, None))
    monkeypatch.setattr(
        compute, "derived_logits", lambda s, query: query.new_tensor([[[1.0, 10.0], [1.0, 10.0]]])
    )
    monkeypatch.setattr(compute, "derived_exclusion", lambda: 0.0)
    gates = compute._axis_import_gates([first, second], [x, x], [None, None], x, x)
    assert all(torch.count_nonzero(gate[..., 1]) == 0 for gate in gates)
    assert torch.equal(sum(gate.sum(-1) for gate in gates), torch.ones_like(x[..., 0]))
    unrestricted = [dataclasses.replace(s, candidates=0) for s in (first, second)]
    gates = compute._axis_import_gates(unrestricted, [x, x], [None, None], x, x)
    assert sum(torch.count_nonzero(gate[..., 1]) for gate in gates) > 0
