# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch
from test_fused import _set
from test_fused_input import _imported, _inputs

from kilora import compute


@pytest.mark.parametrize("depth", range(1, 8))
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
def test_fold_matches_loop(monkeypatch, depth, iteration, gate):
    x, y0 = _inputs(401, tokens=8)
    s = _imported(402, iteration=iteration, gate_mode=gate, import_score="read_share")
    region = torch.linspace(0.2, 1, 8, device=y0.device).expand(2, -1)
    monkeypatch.setenv("KILORA_FUSED", "0")
    outputs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("KILORA_IMPORT_FOLD", enabled)
        with torch.no_grad(), compute.dropout_key(7, 3):
            outputs.append(compute.run_layer(x, y0, [s], depth, regions=[region]))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth_map", [False, True])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
def test_fractional_and_token_depth(monkeypatch, depth_map, gate):
    x, y0 = _inputs(403, tokens=8)
    s = _imported(404, gate_mode=gate, import_score="read_share")
    td = torch.tensor([[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]], device=y0.device).expand(2, -1)
    monkeypatch.setenv("KILORA_FUSED", "0")
    outputs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("KILORA_IMPORT_FOLD", enabled)
        with torch.no_grad(), compute.dropout_key(7, 3):
            outputs.append(
                compute.run_layer(x, y0, [s], 3.5, token_depth=td if depth_map else None)
            )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("other_form", ["state", "keys"])
def test_mixed_call_uses_loop(monkeypatch, other_form):
    x, y0 = _inputs(405, tokens=8)
    s = _imported(406, import_score="read_share")
    other = (
        dataclasses.replace(_set(407), gate_mode="adaptive")
        if other_form == "state"
        else _imported(407, import_score="keys")
    )
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setattr(compute, "_import_fold_history", lambda *a: pytest.fail("fold used"))
    with torch.no_grad():
        compute.run_layer(x, y0, [s, other], 3)


def test_routing_weights_are_constant(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    x, y0 = _inputs(410, tokens=8)
    s = _imported(411, import_score="read_share")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "0")
    weights = []
    with torch.no_grad(), compute.recording_shares(lambda _, gates: weights.append(gates[0])):
        compute.run_layer(x, y0, [s], 7)
    assert len(weights) == 7
    assert all(torch.equal(weights[0], w) for w in weights[1:])


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="needs Triton interpreter")
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("depth", [2, 3, 7, 2.5])
@pytest.mark.parametrize("base_reference", [False, True])
def test_fused_single_update(monkeypatch, gate, iteration, depth, base_reference):
    from kilora import fused

    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1" if base_reference else "0")
    if iteration == "accumulate":
        monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    if depth == 2.5 and gate == "adaptive":
        pytest.skip("fractional adaptive depths keep the loop")
    x, y0 = _inputs(408, tokens=8)
    s = _imported(409, gate_mode=gate, import_score="read_share", iteration=iteration)
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    calls = []
    kernel = fused._update_kernel

    class Launcher:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(1)
                return kernel[grid](*args, **kwargs)

            return run

    outputs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("KILORA_IMPORT_FOLD", enabled)
        with torch.no_grad(), compute.dropout_key(7, 3):
            outputs.append(compute.run_layer(x, y0, [s], depth))
    monkeypatch.setattr(fused, "_update_kernel", Launcher())
    with torch.no_grad(), compute.dropout_key(7, 3):
        got = compute.run_layer(x, y0, [s], depth)
    assert len(calls) == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(outputs[0], got, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="needs Triton interpreter")
def test_chain_keeps_the_fused_loop_when_fold_is_enabled(monkeypatch):
    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "1")
    x, y0 = _inputs(418, tokens=8, n_in=24)
    s = _imported(419, n_in=24, gate_mode="none", import_score="read_share")
    calls = []
    kernel = fused._update_kernel

    class Launcher:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(1)
                return kernel[grid](*args, **kwargs)

            return run

    monkeypatch.setattr(fused, "_update_kernel", Launcher())
    with torch.no_grad(), compute.dropout_key(7, 3):
        actual = compute.run_layer(x, y0, [s], 3)
        with compute.eager():
            expected = compute.run_layer(x, y0, [s], 3)
    assert calls == [1, 1, 1]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_chain_read_share_routes_follow_the_new_read(monkeypatch):
    from test_share_transition import planted

    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = dataclasses.replace(planted(), carry="drop")
    x = torch.tensor([[[1.0, 0.5]]])
    routes = []
    with torch.no_grad(), compute.recording_shares(lambda _, gs: routes.append(gs[0])):
        compute.run_layer(x, torch.zeros_like(x), [s], 3)
    assert len(routes) == 3
    assert not torch.equal(routes[0], routes[1])
