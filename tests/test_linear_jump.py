# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from kilora import compute, linear_jump


def test_fit_planted_map():
    gen = torch.Generator().manual_seed(42)
    first = torch.randn(256, 6, generator=gen)
    expected = torch.randn(6, 6, generator=gen) * 0.2
    last = first + first @ expected
    actual, residual = linear_jump.fit(first, last)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert residual < 1e-6


def test_fused_jump_and_fallback(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    pytest.importorskip("triton")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs a GPU or the Triton interpreter")
    from test_fused_input import _imported, _inputs

    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0 = _inputs(101, batch=2, tokens=32)
    s = _imported(102, gate_mode="none")
    s.jump_fits = {}
    with torch.no_grad(), linear_jump.recording("before") as records:
        full = compute.run_layer(x, y0, [s], 3, target="test")
    fit = records["layers"][("test", 3)]
    assert fit[1] < 1e-4
    s.jump_fits[("before", 3)] = fit
    trace = []
    fused._TRACE.append(trace)
    try:
        with torch.no_grad():
            shortcut = compute.run_layer(x, y0, [s], 3, target="test")
    finally:
        fused._TRACE.remove(trace)
    assert len(trace) == 1
    with torch.no_grad():
        one = compute.run_layer(x, y0, [s], 1, target="test")
        s.jump_fits[("before", 1)] = (torch.full_like(fit[0], 100), 0.0)
        still_one = compute.run_layer(x, y0, [s], 1, target="test")
        s.jump_fits[("before", 3)] = (torch.full_like(fit[0], 100), 1.0)
        fallback = compute.run_layer(x, y0, [s], 3, target="test")
    monkeypatch.setenv("KILORA_FUSED", "0")
    with torch.no_grad():
        eager = compute.run_layer(x, y0, [s], 3, target="test")
    torch.testing.assert_close(shortcut, full, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(shortcut, eager, rtol=1e-4, atol=1e-5)
    assert torch.equal(one, still_one)
    assert torch.equal(fallback, full)


def test_probe_records_both_bands_and_restores_plans():
    from types import SimpleNamespace

    from kilora.comfyui.state_keys import KeyPass

    layer = SimpleNamespace(route_freeze=0.4, jump_fits={})
    target = SimpleNamespace(name="layer", depth=3, schedule=object(), sets=[layer])
    original_schedule = target.schedule
    plan = SimpleNamespace(parts=[(0, -1, target)], _cache={"stale": 1})
    probe = KeyPass([plan], [], "family", linear_jump=True)
    seen = []
    first = torch.eye(4)

    def forward(_input, _time, **_conditioning):
        seen.append(compute.current_progress())
        assert target.schedule is None
        assert layer.route_freeze is None
        compute._STEP_CARRY[-1]["probe"] = True
        linear_jump.note("layer", 3, first, 2 * first)

    existing_carry = {}
    with compute.step_carry(existing_carry):
        probe._fit_jump(forward, {"input": first, "timestep": torch.tensor(1.0), "c": {}})
        assert compute._STEP_CARRY[-1] is existing_carry
    assert existing_carry == {}
    assert seen == [0.2, 0.8]
    assert set(layer.jump_fits) == {("before", 3), ("after", 3)}
    assert target.schedule is original_schedule
    assert layer.route_freeze == 0.4
    assert plan._cache == {}


@pytest.mark.parametrize("base_reference", [False, True])
def test_adaptive_jump_and_freeze_band(monkeypatch, base_reference):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1" if base_reference else "0")
    pytest.importorskip("triton")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs a GPU or the Triton interpreter")
    from test_fused_input import _imported, _inputs

    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0 = _inputs(103)
    s = _imported(104, gate_mode="adaptive")
    cols = s.down.shape[0]
    zero_map = torch.zeros(cols, cols, device=y0.device)
    s.route_freeze = 0.4
    s.jump_fits = {("before", 3): (zero_map, 0.0), ("after", 3): (zero_map, 0.0)}
    trace = []
    fused._TRACE.append(trace)
    try:
        with torch.no_grad():
            one = compute.run_layer(x, y0, [s], 1, target="test")
            trace.clear()
            before = compute.run_layer(x, y0, [s], 3, target="test")
            assert len(trace) == (1 if base_reference else 3)
            trace.clear()
            with compute.sampling_progress(0.8):
                compute.run_layer(x, y0, [s], 3, target="test")
            assert len(trace) == 3
    finally:
        fused._TRACE.remove(trace)
    if base_reference:
        torch.testing.assert_close(before, one, rtol=1e-4, atol=1e-5)
    else:
        s.jump_fits.clear()
        with torch.no_grad():
            expected = compute.run_layer(x, y0, [s], 3, target="test")
        assert torch.equal(before, expected)


def test_chain_records_jump_and_rejects_a_bad_fit(monkeypatch):
    pytest.importorskip("triton")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs a GPU or the Triton interpreter")
    from test_fused_input import _imported, _inputs

    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y = _inputs(111, n_in=24)
    s = _imported(112, n_in=24, gate_mode="none")
    with torch.no_grad(), linear_jump.recording("before") as records:
        expected = compute.run_layer(x, y, [s], 3, target="chain")
    fitted = records["layers"][("chain", 3)]
    s.jump_fits = {("before", 3): (fitted[0], s.jump_threshold + 1)}
    trace = []
    fused._TRACE.append(trace)
    try:
        with torch.no_grad():
            actual = compute.run_layer(x, y, [s], 3, target="chain")
    finally:
        fused._TRACE.remove(trace)
    assert len(trace) == 3
    assert torch.equal(actual, expected)
