# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from test_fused_input import TARGET, _imported, _inputs

from kilora import adaptive_gate, compute
from kilora.adaptive_gate import adaptive_keep

pytestmark = pytest.mark.skipif(
    adaptive_gate.triton is None
    or (not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"),
    reason="needs CUDA or the Triton interpreter",
)


@pytest.fixture(autouse=True)
def _original_reference(monkeypatch):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")


def _reference(
    z, before, history, up, norm, rms, kappa=1.0, coherence=True, active=None, gram=None
):
    dz = z if before is None else z - before.float()
    gram = up @ up.t()
    step = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
    qa = torch.pow(0.5, step / (kappa * norm + 1e-6))
    tokens = z.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)

    def diag(s):
        return ((s @ up) * up).sum(-2)

    acc = diag(z.transpose(-1, -2) @ z / tokens).clamp(min=0).sqrt()
    qb = torch.pow(0.5, acc / (kappa * rms + 1e-6))
    if coherence and history is not None:
        m = history.float()
        dzt, mt = dz.transpose(-1, -2), m.transpose(-1, -2)
        dot = diag(dzt @ m)
        nd = diag(dzt @ dz).clamp(min=0).sqrt()
        nm = diag(mt @ m).clamp(min=0).sqrt()
        norms = nd * nm
        cos = torch.where(norms > 1e-6, dot / norms.clamp(min=1e-6), torch.ones_like(dot))
        qb = qb * (0.5 + 0.5 * cos).clamp(0, 1)
    old = 0.0 if history is None else 0.5 * history.float()
    return qa[..., None], qb[..., None, :], old + 0.5 * dz


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_rates_against_original(monkeypatch, history, active):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_diff = 0.0
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(2, 17, 12, generator=g).to(device)
        before = torch.randn(2, 17, 12, generator=g).to(device)
        hist = torch.randn(2, 17, 12, generator=g).to(device) if history else None
        up = torch.randn(12, 48, generator=g).to(device)
        norm = torch.rand(2, 17, generator=g).to(device) * 8
        rms = torch.rand(2, 48, generator=g).to(device) * 2
        mask = (torch.rand(2, 17, 1, generator=g) > 0.4).to(device) if active else None
        if mask is not None:
            z = z * mask
        old = _reference(z, before, hist, up, norm, rms, active=mask)
        got = adaptive_keep(z, before, hist, up, norm, rms, active=mask)
        for expected, result in zip(old, got, strict=True):
            max_diff = max(max_diff, (expected - result).abs().max().item())
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    assert max_diff <= 1e-5


def test_sampled_masks_match_original(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_MASKS", "sample")
    s = _imported(91)
    device = s.up.device
    g = torch.Generator().manual_seed(92)
    steps = [torch.randn(2, 8, s.up.shape[0], generator=g).to(device) for _ in range(3)]
    norm = torch.rand(2, 8, generator=g).to(device) * 5
    rms = torch.rand(2, s.up.shape[1], generator=g).to(device) * 2

    def run(keep):
        monkeypatch.setattr(compute, "adaptive_keep", keep)
        hist = {}
        with torch.no_grad(), compute.dropout_key(7, 3):
            return [
                compute.split_gate(s, z, hist, norm, rms, None, TARGET, 0, k, None)
                for k, z in enumerate(steps)
            ]

    old = run(_reference)
    new = run(adaptive_keep)
    for old_parts, new_parts in zip(old, new, strict=True):
        for expected, result in zip(old_parts, new_parts, strict=True):
            assert torch.equal(expected, result)


@pytest.mark.parametrize("masks", ["sample", "expect"])
@pytest.mark.parametrize("depth", [1, 3, 1.5])
def test_layer_against_original_and_eager(monkeypatch, masks, depth):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED_GATE", "torch")
    monkeypatch.setenv("KILORA_MASKS", masks)
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    x, y = _inputs(85, batch=2, tokens=8)
    s = _imported(86, routing="token")

    def run(fused_on):
        monkeypatch.setenv("KILORA_FUSED", "1" if fused_on else "0")
        with torch.no_grad(), compute.dropout_key(7, 3):
            return compute.run_layer(x, y, [s], depth, target=TARGET)

    monkeypatch.setattr(compute, "adaptive_keep", _reference)
    old = run(False)
    monkeypatch.setattr(compute, "adaptive_keep", adaptive_keep)
    eager = run(False)
    fused = run(True)
    torch.testing.assert_close(eager, old, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fused, eager, rtol=1e-4, atol=1e-5)
