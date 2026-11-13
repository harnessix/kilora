# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

pytest.importorskip("triton")

from kilora import fused

DEVICE = (
    "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
)
pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs CUDA or the Triton interpreter",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_token", [False, True])
def test_elementwise_chains(dtype, per_token, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    torch.manual_seed(47)
    shape = (2, 7, 19)
    z = torch.randn(shape, device=DEVICE)
    a = torch.rand((2, 7, 1), device=DEVICE)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)
    delta = torch.randn(shape, device=DEVICE, dtype=dtype)
    b = torch.rand((2, 7 if per_token else 1, 19), device=DEVICE)
    mult = torch.rand((2, 1, 1), device=DEVICE, dtype=dtype)
    kernel = fused._elementwise_kernel
    launches = []

    class CountingKernel:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def call(*args, **kwargs):
                launches.append(kwargs["MODE"])
                return launch(*args, **kwargs)

            return call

    monkeypatch.setattr(fused, "_elementwise_kernel", CountingKernel())
    with torch.no_grad():
        assert torch.equal(fused.token_gate(z, a, dtype), (a.float() * z).to(dtype))
        assert torch.equal(fused.add_cast(y, z), y + z.to(dtype))
        assert torch.equal(fused.channel_change(delta, b), delta * b.to(dtype))
        assert torch.equal(fused.channel_change(delta, b, y), y + delta * b.to(dtype))
        for scale in (0.37, mult):
            assert torch.equal(fused.scaled_difference(y, delta, scale), y + scale * (delta - y))
    assert launches == [0, 1, 3, 2, 4, 4]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_many_steps_equal_torch(dtype, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    torch.manual_seed(57)
    base = torch.randn((2, 9, 13), device=DEVICE, dtype=dtype)
    eager = base.clone()
    actual = base.clone()
    with torch.no_grad():
        for _ in range(5):
            z = torch.randn((2, 9, 13), device=DEVICE)
            a = torch.rand((2, 9, 1), device=DEVICE)
            b = torch.rand((2, 1, 13), device=DEVICE)
            delta = (a * z).to(dtype)
            eager = eager + delta * b.to(dtype)
            actual = fused.channel_change(fused.token_gate(z, a, dtype), b, actual)
            assert torch.equal(actual, eager)


@pytest.mark.skipif(DEVICE != "cuda", reason="large GPU shape")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_gpu_layer_shapes(dtype, gated, per_token):
    torch.manual_seed(71)
    z = torch.randn((1, 4096, 512), device=DEVICE)
    a = torch.rand((1, 4096, 1), device=DEVICE)
    y = torch.randn((1, 4096, 3072), device=DEVICE, dtype=dtype)
    delta = torch.randn_like(y)
    b = torch.rand((1, 4096 if per_token else 1, 3072), device=DEVICE)
    with torch.no_grad():
        assert torch.equal(fused.token_gate(z, a, dtype), (z * a).to(dtype))
        got = fused.channel_change(delta, b, y) if gated else fused.add_cast(y, delta)
        expected = y + delta * b.to(dtype) if gated else y + delta
        assert torch.equal(got, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_input_output_matches_original(dtype, gated, per_token, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    torch.manual_seed(123)
    lead = (2, 7)
    cols, width = 5, 13
    z = torch.randn((*lead, cols), device=DEVICE)
    y = torch.randn((*lead, width), dtype=dtype, device=DEVICE)
    up = torch.randn((cols, width), dtype=dtype, device=DEVICE)
    a = torch.rand((*lead, 1), device=DEVICE)
    b = torch.rand((2, 7 if per_token else 1, width), device=DEVICE)
    p = SimpleNamespace(C=cols, OUT=width, up_dt=up)
    with torch.no_grad():
        got = fused._input_output(y, z, (a, b) if gated else None, p, lead, dtype)
        if gated:
            expected = y + ((a.float() * z).to(dtype) @ up) * b.to(dtype)
        else:
            expected = y + z.to(dtype) @ up
        assert torch.equal(got, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stand_in_model_over_steps(dtype, monkeypatch):
    from kilora.fixed_change import fixed_copy
    from kilora.sections import LowRankChange
    from kilora.timeline import Clock, Row, time_scales

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    torch.manual_seed(31)
    base = torch.nn.Linear(11, 11, device=DEVICE, dtype=dtype)
    change = LowRankChange(torch.randn(4, 11), torch.randn(11, 4), 0.3)
    model = fixed_copy(base, [change])
    initial = torch.randn((2, 5, 11), device=DEVICE, dtype=dtype)
    rows = [Row("all", "strength", "0.37")]

    def run(on):
        monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
        x = initial.clone()
        values = []
        with torch.no_grad(), time_scales(rows, Clock()):
            for _ in range(5):
                x = model(x)
                values.append(x)
        return values

    expected = run(False)
    actual = run(True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_unusual_broadcast_and_gradients_use_torch(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    delta = torch.randn((2, 2, 3, 7), device=DEVICE, requires_grad=True)
    gate = torch.randn((1, 2, 1, 7), device=DEVICE)
    y = torch.randn_like(delta)
    got = fused.channel_change(delta, gate, y)
    expected = y + delta * gate.to(delta.dtype)
    assert torch.equal(got, expected)
    got.sum().backward()
    assert delta.grad is not None
    with torch.no_grad():
        assert torch.equal(fused.channel_change(delta, gate, y), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_imported_eager_path_gate_integration(dtype, per_token, depth, monkeypatch):
    from test_fused_input import _imported

    from kilora import compute

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "0")
    torch.manual_seed(911)
    x = torch.randn((2, 7, 20), device=DEVICE, dtype=dtype)
    y = torch.randn((2, 7, 24), device=DEVICE, dtype=dtype)
    st = _imported(3, gate_mode="adaptive", routing="token")
    a = torch.rand((2, 7, 1), device=DEVICE)
    b = torch.rand((2, 7 if per_token else 1, 24), device=DEVICE)
    stats = {"forced_gates": [(a, b)] * depth}

    def run(enabled):
        monkeypatch.setenv("KILORA_FUSED", "1" if enabled else "0")
        with torch.no_grad(), compute.dropout_key(4, 1):
            return compute._run_input(x, y, [st], [0], depth, [None], "test", stats=stats)

    assert torch.equal(run(False), run(True))


def test_empty_output_uses_torch():
    delta = torch.empty((2, 3, 0), device=DEVICE)
    gate = torch.empty((2, 1, 0), device=DEVICE)
    with torch.no_grad():
        assert torch.equal(fused.channel_change(delta, gate, delta), delta + delta * gate)
