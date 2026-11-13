# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest
import torch

from kilora import compute, fused


def layer():
    gen = torch.Generator().manual_seed(3)
    return compute.LayerSet(
        router=torch.randn(4, 8, generator=gen),
        keys_in=torch.randn(4, 1, 4, generator=gen),
        down=torch.randn(4, 8, generator=gen),
        up=torch.randn(4, 8, generator=gen),
    )


def test_candidate_decision_and_note_once(monkeypatch, caplog):
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "device_ok", lambda _: True)
    monkeypatch.setattr(compute, "_EAGER", False)
    s = layer()
    y = torch.randn(1, 3, 8)
    s.candidates = 0
    assert fused.fused_ok(y, [s])
    monkeypatch.setattr(fused, "_NOTED", set())
    caplog.set_level(logging.INFO, logger="kilora")
    for count in (2, 2, 3):
        s.candidates = count
        assert not fused.fused_ok(y, [s])
    assert sum("candidate selection" in r.message for r in caplog.records) == 1
    assert fused.LAST_REASON[0] == "candidate selection"


def test_candidate_fallback_matches_loop(monkeypatch):
    if "candidates" not in compute.LayerSet.__dataclass_fields__:
        pytest.skip("requires T202 candidate routing")
    s = layer()
    x = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(5))
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "device_ok", lambda _: True)
    s.candidates = 2
    with torch.no_grad():
        monkeypatch.setenv("KILORA_FUSED", "0")
        expected = compute.run_layer(x, x, [s], 1)
        monkeypatch.setenv("KILORA_FUSED", "1")
        monkeypatch.setattr(fused, "run_fused", lambda *a, **k: pytest.fail("entered fused kernel"))
        actual = compute.run_layer(x, x, [s], 1)
    assert torch.equal(actual, expected)


def test_default_is_bit_identical_to_explicit_zero(monkeypatch):
    s = layer()
    x = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(7))
    monkeypatch.setenv("KILORA_FUSED", "0")
    with torch.no_grad():
        expected = compute.run_layer(x, x, [s], 1)
        s.candidates = 0
        actual = compute.run_layer(x, x, [s], 1)
    assert torch.equal(actual, expected)
