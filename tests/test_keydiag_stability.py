# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_keydiag_state import layer
from test_share_transition import planted

from kilora import compute
from kilora.keydiag_stability import graph_stability, spectral_radius, stability


def test_graph_radius_and_return_mass():
    t = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
    result = graph_stability(t)
    assert result["off_diagonal_radius"] == pytest.approx(0.5)
    assert result["return_mass_2"] == pytest.approx(0.53125)
    assert result["return_mass_3"] == pytest.approx(0.4921875)
    assert spectral_radius(torch.diag(torch.tensor([-2.0, 0.5]))) == pytest.approx(2.0)


def test_resolvent_matches_series_and_rejects_divergence():
    t = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    gamma = 0.7
    power = torch.eye(2)
    expected = torch.zeros_like(t)
    for _ in range(100):
        expected += (1 - gamma) * power
        power = gamma * power @ t
    torch.testing.assert_close(compute.transition_resolvent(t, gamma), expected)
    with pytest.raises(ValueError, match="converge"):
        compute.transition_resolvent(torch.eye(2) * 2, 0.5)
    for value in (0, 1, -1, float("nan")):
        with pytest.raises(ValueError):
            compute.transition_resolvent(t, value)


def test_weighted_cached_transition_applies_resolvent_once(monkeypatch):
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "weighted")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0.6")
    s = planted()
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    base = compute._share_transition_base(s, states=states)
    cached = dataclasses.replace(s, share_transition_weighted=base)
    torch.testing.assert_close(
        compute.share_transition(cached), compute.transition_resolvent(base, 0.6)
    )


def test_option_off_preserves_bits_and_on_changes_output(monkeypatch):
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.delenv("KILORA_SHARE_TRANSITION_RESOLVENT", raising=False)
    with torch.no_grad(), compute.dropout_key(4, 1):
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0")
        assert torch.equal(baseline, compute.run_layer(x, torch.zeros_like(x), [s], 3))
        monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0.6")
        changed = compute.run_layer(x, torch.zeros_like(x), [s], 3)
    assert not torch.equal(baseline, changed)


def test_state_step_radius_measured_and_no_output_change(monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    s = dataclasses.replace(layer(), carry="drop")
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    with torch.no_grad(), compute.eager(), compute.dropout_key(4, 1):
        before = compute.run_layer(x, x, [s], 3)
        result = stability(s, x, x, 3)
        after = compute.run_layer(x, x, [s], 3)
    assert torch.equal(before, after)
    assert 0 < result["depth_step_radius"] <= 0.1
    assert result["off_diagonal_radius"] == 0


def test_fused_resolvent_matches_eager(monkeypatch):
    pytest.importorskip("triton")
    import os

    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0.6")
    s = _imported(3, n_in=24, routing="token", gate_mode="none")
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    expected, used = _run(monkeypatch, False, x, y, [s], 3)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], 3)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_resolvent_cache_tracks_weights_gamma_and_raw(monkeypatch):
    s = planted()
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0.6")
    calls = []
    original = compute.transition_resolvent

    def counted(transition, gamma):
        calls.append(gamma)
        return original(transition, gamma)

    monkeypatch.setattr(compute, "transition_resolvent", counted)
    with torch.inference_mode():
        first = compute.share_transition(s)
        assert compute.share_transition(s) is first
        raw = compute.share_transition(s, _kind="cosine", _raw=True)
        torch.testing.assert_close(
            raw, compute._share_transition_base(s, _kind="cosine", _raw=True)
        )
        assert calls == [0.6]
        monkeypatch.setenv("KILORA_SHARE_TRANSITION_RESOLVENT", "0.7")
        compute.share_transition(s)
    with torch.no_grad():
        s.down.add_(0.1)
    compute.share_transition(s)
    assert calls == [0.6, 0.7, 0.7]
