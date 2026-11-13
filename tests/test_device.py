# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora.compute import dropout_key, new_gate, run_layer, step_carry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _defaults(device):
    s = _sets(2, 16, 4, 2, 8, seed=0, scales=[1.0, -0.5])
    gen = torch.Generator().manual_seed(0)
    out = []
    for x in s:
        gate = new_gate("separable", 4, 16, gen, x.up.shape[0])
        moved = {
            f.name: getattr(x, f.name).to(device)
            for f in dataclasses.fields(x)
            if isinstance(getattr(x, f.name), torch.Tensor)
        }
        moved_gate = {k: v.to(device) for k, v in gate.items()}
        out.append(dataclasses.replace(x, **moved, output_gate="separable", gate=moved_gate))
    return out


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("grad", [False, True])
def test_all_defaults_stay_on_the_input_device(fast, grad, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    sets = _defaults(DEVICE)
    if grad:
        for s in sets:
            s.up.requires_grad_(True)
    y0 = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1)).to(DEVICE)
    trap = torch.device("meta") if DEVICE == "cpu" else torch.device("cpu")
    cache: dict = {}
    with torch.set_grad_enabled(grad), trap, dropout_key(0, 0.5):
        for _ in range(2):
            with step_carry(cache):
                out = run_layer(y0, y0, sets, 3, target="layer")
    assert out.device.type == torch.device(DEVICE).type
    if grad:
        out.sum().backward()
        assert sets[0].up.grad is not None


def test_adapter_tensors_live_on_the_layer_device(model):
    from test_adapter import make

    adapter = make(model)
    device = next(model.parameters()).device
    tensors = list(adapter.parameters()) + list(adapter.buffers())
    assert tensors and all(t.device == device for t in tensors)
    adapter.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_cuda_layer_call_copies_nothing_from_the_host():
    from torch.profiler import ProfilerActivity, profile

    from kilora import KiLoRAConfig, attach
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser(channels=4, width=16, num_layers=2, seed=0).cuda().eval()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.attention.to_out.0"], num_experts=4, routes=2, router_dim=8
        ),
    )
    x = torch.randn(2, 6, 4, device="cuda")
    with torch.no_grad():
        model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(x)
            torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    assert not any("HtoD" in n for n in names), [n for n in names if "HtoD" in n][:5]
    assert sum("cudaStreamSynchronize" in n for n in names) == 0
    adapter.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fast", [False, True])
def test_no_host_sync_inside_layer_calls(fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    sets = _defaults("cuda")
    y0 = torch.randn(2, 5, 16, device="cuda")
    cache: dict = {}
    with torch.no_grad(), dropout_key(0, 1):
        with step_carry(cache):
            run_layer(y0, y0, sets, 3, target="layer")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(2):
                with step_carry(cache):
                    run_layer(y0, y0, sets, 3, target="layer")
        finally:
            torch.cuda.set_sync_debug_mode("default")
