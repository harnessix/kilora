# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute, gate_stats
from kilora.compute import dropout_key, run_layer

pytestmark = pytest.mark.skipif(
    gate_stats.triton is None
    or (not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"),
    reason="requires Triton or its interpreter",
)


def _input(shape, dtype):
    return torch.randn(shape, generator=torch.Generator().manual_seed(128)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 19, 3072), (2, 3, 5, 31), (17, 1)])
def test_fused_stats_match_torch_and_are_deterministic(dtype, shape):
    y = _input(shape, dtype)
    sq = y.float().square()
    expected = (
        sq.sum(-1).sqrt() + compute.DROPOUT_DELTA,
        sq.mean(-2).sqrt() + compute.DROPOUT_DELTA,
    )
    a = gate_stats.fused_stats(y, compute.DROPOUT_DELTA, grid_size=1)
    b = gate_stats.fused_stats(y, compute.DROPOUT_DELTA, grid_size=7)
    c = gate_stats.fused_stats(y, compute.DROPOUT_DELTA, grid_size=1)
    for reference, first, second, third in zip(expected, a, b, c, strict=True):
        torch.testing.assert_close(first, reference, rtol=1e-5, atol=1e-6)
        assert first.dtype == reference.dtype and first.shape == reference.shape
        assert torch.equal(first, second)
        assert torch.equal(first, third)


def test_torch_switch_and_fallback(monkeypatch):
    y = _input((2, 19, 31), torch.bfloat16)
    sq = y.float().square()
    expected = (
        sq.sum(-1).sqrt() + compute.DROPOUT_DELTA,
        sq.mean(-2).sqrt() + compute.DROPOUT_DELTA,
    )
    monkeypatch.setenv("KILORA_GATE_STATS", "torch")
    monkeypatch.setattr(gate_stats, "fused_stats", lambda *a, **k: pytest.fail("fused path used"))
    for actual, reference in zip(compute._y0_gate_stats(y), expected, strict=True):
        assert torch.equal(actual, reference)
    assert not gate_stats.supported(y.transpose(-1, -2))
    assert not gate_stats.supported(torch.empty(2, 0, 4))


@pytest.mark.parametrize("mode", ["none", "adaptive"])
def test_stand_in_layer_against_torch(mode, monkeypatch):
    y = _input((2, 19, 16), torch.float32)
    sets = [dataclasses.replace(s, gate_mode=mode) for s in _sets(2, 16, 4, 2, 8, seed=3)]
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    with dropout_key(0, 0.5), torch.no_grad():
        reference = run_layer(y, y, sets, 4, target="layer")
    monkeypatch.setattr(
        compute, "_y0_gate_stats", lambda x: gate_stats.fused_stats(x, compute.DROPOUT_DELTA)
    )
    with dropout_key(0, 0.5), torch.no_grad():
        actual = run_layer(y, y, sets, 4, target="layer")
    difference = (actual - reference).abs().max().item()
    print(f"stand-in gate={mode} max abs difference: {difference:.9g}")
    if mode == "none":
        assert torch.equal(actual, reference)
    else:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_dispatch_and_torch_override(monkeypatch):
    y = _input((2, 19, 3072), torch.bfloat16).cuda()
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.delenv("KILORA_GATE_STATS", raising=False)
    reference = (
        y.float().square().sum(-1).sqrt() + compute.DROPOUT_DELTA,
        y.float().square().mean(-2).sqrt() + compute.DROPOUT_DELTA,
    )
    actual = compute._y0_gate_stats(y)
    for got, want in zip(actual, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    monkeypatch.setenv("KILORA_GATE_STATS", "torch")
    monkeypatch.setattr(gate_stats, "fused_stats", lambda *a, **k: pytest.fail("fused path used"))
    for got, want in zip(compute._y0_gate_stats(y), reference, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("mode", ["none", "adaptive"])
@pytest.mark.parametrize("masks", ["sample", "expect"])
def test_stand_in_final_latents(mode, masks, monkeypatch):
    from conftest import randomize_up

    from kilora import KiLoRAConfig, attach
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser().eval()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=3,
            train_depth=None,
            output_gate="none",
        ),
    )
    randomize_up(adapter, seed=7)
    adapter.settings.gate_mode = mode
    monkeypatch.setenv("KILORA_MASKS", masks)
    original = compute._y0_gate_stats
    initial = _input((2, 19, 4), torch.float32)

    def sample():
        latents = initial.clone()
        with torch.no_grad():
            for step in range(5):
                with dropout_key(step, 0.5):
                    latents = latents - model(latents) * 0.1
        return latents

    monkeypatch.setattr(compute, "_y0_gate_stats", original)
    reference = sample()
    monkeypatch.setattr(
        compute, "_y0_gate_stats", lambda x: gate_stats.fused_stats(x, compute.DROPOUT_DELTA)
    )
    actual = sample()
    difference = (actual - reference).abs().max().item()
    print(f"stand-in final latents gate={mode} masks={masks} max abs difference: {difference:.9g}")
    if mode == "none":
        assert torch.equal(actual, reference)
    else:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,width", [(4096, 3072), (4095, 3072), (4097, 3073)])
def test_gpu_elementwise_against_torch(dtype, tokens, width, monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(128)
    y = torch.randn((2, tokens, width), device="cuda", dtype=dtype, generator=generator)
    monkeypatch.setenv("KILORA_GATE_STATS", "torch")
    expected = compute._y0_gate_stats(y)
    actual = gate_stats.fused_stats(y, compute.DROPOUT_DELTA)
    for name, got, want in zip(("norm", "rms"), actual, expected, strict=True):
        relative = (got - want).abs() / want.abs()
        bad = (~torch.isfinite(relative)) | (relative >= 1e-5)
        if bad.any():
            index = tuple(bad.nonzero()[0].tolist())
            pytest.fail(
                f"{name} first differing index {index}: actual={got[index].item()}, "
                f"expected={want[index].item()}, relative={relative[index].item()}, "
                f"maximum relative={relative.max().item()}"
            )


@pytest.mark.parametrize("tokens", [65, 129])
def test_partial_sum_order_is_independent_of_grid(tokens):
    y = _input((2, tokens, 31), torch.float32)
    one = gate_stats.fused_stats(y, compute.DROPOUT_DELTA, grid_size=1)
    many = gate_stats.fused_stats(y, compute.DROPOUT_DELTA, grid_size=5)
    assert torch.equal(one[0], many[0])
    assert torch.equal(one[1], many[1])


@pytest.mark.parametrize("lowrank", [False, True])
def test_torch_mode_replays_reference_over_steps(lowrank, monkeypatch):
    from conftest import randomize_up

    from kilora import KiLoRAConfig, attach
    from kilora.testing import TinyDenoiser

    monkeypatch.delenv("KILORA_GATE_STATS", raising=False)
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", lowrank)
    model = TinyDenoiser().eval()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=3,
            train_depth=None,
            output_gate="none",
        ),
    )
    randomize_up(adapter, seed=7)
    adapter.settings.gate_mode = "adaptive"
    initial = _input((2, 19, 4), torch.float32)
    original = compute._y0_gate_stats

    def reference(y):
        with torch.no_grad():
            sq = y.float().square()
            return sq.sum(-1).sqrt() + compute.DROPOUT_DELTA, sq.mean(
                -2
            ).sqrt() + compute.DROPOUT_DELTA

    def sample():
        latents = initial.clone()
        with torch.no_grad():
            for step in range(5):
                with dropout_key(step, 0.5):
                    latents = latents - model(latents) * 0.1
        return latents

    monkeypatch.setattr(compute, "_y0_gate_stats", reference)
    expected = sample()
    monkeypatch.setattr(compute, "_y0_gate_stats", original)
    with compute.gate_stats_mode("torch"):
        actual = sample()
    assert torch.equal(actual, expected)
