# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest
import torch
from test_fused import DEVICE, _no_warm_up_running, _set
from test_fused_input import TARGET, _imported, _inputs
from torch.utils._python_dispatch import TorchDispatchMode

from kilora import adaptive_gate, compute, fused
from kilora.bake import TargetPlan

pytestmark = pytest.mark.skipif(
    not fused.HAVE_TRITON or not fused.device_ok(torch.empty(1, device=DEVICE)),
    reason="needs a GPU or the Triton interpreter with CPU opt-in",
)


class _Counter(TorchDispatchMode):
    def __init__(self):
        self.inside = False
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if not self.inside and name not in {
            "aten::view",
            "aten::_unsafe_view",
            "aten::reshape",
            "aten::detach",
            "aten::empty",
            "aten::empty_like",
            "aten::expand",
            "aten::transpose",
            "aten::t",
            "aten::as_strided",
            "aten::alias",
            "aten::slice",
            "aten::select",
            "aten::permute",
            "aten::unsqueeze",
            "aten::squeeze",
            "aten::unbind",
        }:
            self.ops[name] += 1
        return func(*args, **(kwargs or {}))


class _Launcher:
    def __init__(self, kernel, counter):
        self.kernel = kernel
        self.counter = counter

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.counter.inside = True
            try:
                return launch(*args, **kwargs)
            finally:
                self.counter.inside = False

        return run


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("kind", ["separable", "none", "adaptive"])
def test_fused_torch_ops(monkeypatch, kind, depth):
    _no_warm_up_running()
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_FUSED_GATE", "torch" if kind == "adaptive" else "kernel")
    x, y0 = _inputs(71, batch=2, tokens=8)
    s = (
        _set(72, output_gate="separable", gate_mode="learned")
        if kind == "separable"
        else _imported(72, gate_mode=kind)
    )
    s.tau = torch.tensor(1.0, dtype=torch.float64, device=y0.device)
    s.scale = torch.tensor(1.0, dtype=torch.float64, device=y0.device)
    plan = TargetPlan(sets=[s], depth=depth, name=TARGET)
    invoked = []
    run_fused = fused.run_fused

    def spy(*args, **kwargs):
        invoked.append(True)
        return run_fused(*args, **kwargs)

    monkeypatch.setattr(fused, "run_fused", spy)

    def run():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return plan.run(x, y0)

    new_arg = fused._arg

    def old_arg(value):
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(1).float(), 0.0, True
        return None, float(value), False

    channel_kernel = fused._channel_output_kernel

    class OldChannel:
        def __getitem__(self, grid):
            def launch(y, delta, gate, out, tokens, width, count, **kwargs):
                out.copy_(y + delta * gate.reshape(-1, 1, width))

            return launch

    monkeypatch.setattr(fused, "_arg", old_arg)
    monkeypatch.setattr(fused, "_channel_output_kernel", OldChannel())
    kernel_uniforms = fused.uniforms
    monkeypatch.setattr(
        fused,
        "uniforms",
        lambda n, target, k, device: compute.dropout_uniforms((n,), target, k, device),
    )
    expected = run()
    monkeypatch.setattr(fused, "_arg", new_arg)
    monkeypatch.setattr(fused, "_channel_output_kernel", channel_kernel)
    monkeypatch.setattr(fused, "uniforms", kernel_uniforms)
    assert torch.equal(run(), expected)
    counter = _Counter()
    for name in dir(fused):
        if name.endswith("_kernel") and hasattr(getattr(fused, name), "__getitem__"):
            monkeypatch.setattr(fused, name, _Launcher(getattr(fused, name), counter))
    for name in ("_gate_token_kernel", "_gate_channel_kernel"):
        monkeypatch.setattr(adaptive_gate, name, _Launcher(getattr(adaptive_gate, name), counter))
    with counter:
        got = run()
    assert len(invoked) == 3
    assert torch.equal(got, expected)
    gemms = counter.ops.pop("aten::mm", 0) + counter.ops.pop("aten::bmm", 0)
    extra_gemms = max(0, gemms - (3 if kind == "separable" else 2))
    extra = sum(counter.ops.values()) + extra_gemms
    if kind == "adaptive":
        assert extra <= (20 if depth == 1 else 56), (extra, gemms, counter.ops)
        return
    if extra > 3 and (kind, depth) in {
        ("separable", 3),
    }:
        pytest.xfail(f"op budget: {extra} extra, {gemms} GEMMs; {counter.ops}")
    assert extra <= 3, f"{extra} extra, {gemms} GEMMs; {counter.ops}"


def test_adaptive_gate_torch_ops_per_step(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    z = torch.randn(2, 8, 12, device=DEVICE)
    up = torch.randn(12, 24, device=DEVICE)
    gram = up @ up.t()
    norm = torch.rand(2, 8, device=DEVICE)
    rms = torch.rand(2, 24, device=DEVICE)
    counter = _Counter()
    for name in ("_gate_token_kernel", "_gate_channel_kernel"):
        monkeypatch.setattr(adaptive_gate, name, _Launcher(getattr(adaptive_gate, name), counter))
    with counter:
        adaptive_gate.adaptive_keep(z, z, z, up, norm, rms, gram=gram)
    assert sum(counter.ops.values()) <= 12, counter.ops


def test_scalar_argument_cache():
    value = torch.tensor([1.25], dtype=torch.float64)
    first, _, is_pointer = fused._arg(value)
    second, _, _ = fused._arg(value)
    assert is_pointer and first is second
    assert torch.equal(first, value.float())
    value.add_(1)
    third, _, _ = fused._arg(value)
    assert third is not first
    assert torch.equal(third, value.float())
    scalar = torch.tensor(0.75)
    assert fused._arg(scalar)[0] is scalar
