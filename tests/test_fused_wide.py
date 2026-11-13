# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

pytest.importorskip("triton")

from test_fused_input import DEVICE, TARGET, _imported, _inputs
from test_fused_token_depth import _run

from kilora import compute, fused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)


@pytest.mark.parametrize("depth", [1, 3, 1.5])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("mapped", [False, True])
def test_wide_read_share_matches_eager(monkeypatch, depth, gate, routing, mapped):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "0")
    monkeypatch.setattr(fused, "_shared_limit", lambda _: 99 * 1024)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    x, y = _inputs(11, batch=1, tokens=5)
    s = _imported(12, experts=4, per_expert=256, gate_mode=gate, routing=routing)
    region = torch.tensor([[1.0, 0.3, 0.0, 0.7, 1.0]], device=DEVICE)
    depths = torch.tensor([[0.0, 0.5, 1.0, 1.5, 3.0]], device=DEVICE) if mapped else None
    assert fused.fits(y, 1024, 8, tiled=True)
    assert not fused.fits(y, 1024, 8)
    expected, used = _run(monkeypatch, False, x, y, [s], depth, depths, [region])
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], depth, depths, [region])
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("rank", [87, 195, 257])
def test_wide_column_tail_and_dropout_masks(monkeypatch, rank):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setattr(fused, "_shared_limit", lambda _: 99 * 1024)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    x, y = _inputs(21, batch=1, tokens=5)
    s = _imported(22, experts=4, per_expert=rank, gate_mode="none", carry="drop")
    trace = []
    fused._TRACE.append(trace)
    try:
        with torch.no_grad(), compute.dropout_key(7, 3):
            fused.run_fused(y, [s], 2, target=TARGET, name=TARGET, x=x)
            c0 = torch.nn.functional.linear(x, s.down).float()
            z = trace[0]["z"].reshape_as(c0)
            mask = compute.read_dropout(z, c0, rank, TARGET, 1)
        assert 0 < mask.float().mean() < 1
        assert torch.equal(trace[0]["carried"].reshape_as(z), mask * z)
    finally:
        fused._TRACE.pop()
    expected, used = _run(monkeypatch, False, x, y, [s], 3, None)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], 3, None)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_wide_key_scoring_still_obeys_its_limit(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "keys")
    monkeypatch.setattr(fused, "_shared_limit", lambda _: 99 * 1024)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    x, y = _inputs(31, batch=1, tokens=5)
    s = _imported(32, experts=4, per_expert=256, gate_mode="none")
    expected, _ = _run(monkeypatch, False, x, y, [s], 1, None)
    actual, used = _run(monkeypatch, True, x, y, [s], 1, None)
    assert not used
    assert torch.equal(actual, expected)


def test_wide_adaptive_uses_torch_gate(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "1")
    monkeypatch.setattr(fused, "_shared_limit", lambda _: 99 * 1024)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    x, y = _inputs(41, batch=1, tokens=5)
    s = _imported(42, experts=4, per_expert=256, gate_mode="adaptive")
    expected, _ = _run(monkeypatch, False, x, y, [s], 1, None)
    monkeypatch.setattr(fused, "_adaptive_token_kernel", None)
    actual, used = _run(monkeypatch, True, x, y, [s], 1, None)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
