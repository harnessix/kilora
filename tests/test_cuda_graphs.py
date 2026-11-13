# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import sys
import types

import pytest
import torch

from kilora import compute, linear_jump, route_freeze
from kilora.comfyui import cuda_graphs
from kilora.recording import perturbing


@pytest.fixture
def cuda(monkeypatch):
    class Stream:
        def __init__(self, **kwargs):
            pass

        def wait_stream(self, other):
            pass

    class Graph:
        def replay(self):
            if fail[0]:
                raise RuntimeError("replay failed")
            self.output.copy_(self.inputs["input"] * 2)

    fail, current = [False], [None]

    @contextlib.contextmanager
    def capture(graph, **kwargs):
        current[0] = graph
        try:
            yield
        finally:
            current[0] = None

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (100, 100))

    def invoke(args):
        result = args["input"] * 2
        if current[0] is not None:
            current[0].inputs = args
            current[0].output = result
        return result

    return invoke, fail


def args(n=1):
    return {"input": torch.ones(n), "timestep": torch.tensor([1.0]), "c": {}}


def test_buckets_lru_and_replay(cuda):
    invoke, _ = cuda
    graphs = cuda_graphs.Graphs()
    cache = {"target": {}}
    for n in range(1, 5):
        assert torch.equal(graphs.call(args(n), cache, invoke), torch.full((n,), 2.0))
    changed = args(1)
    changed["input"].fill_(3)
    replayed = graphs.call(changed, cache, invoke)
    assert replayed.item() == 6
    changed["input"].fill_(4)
    assert replayed.item() == 6
    graphs.call(args(5), cache, invoke)
    assert len(graphs.buckets) == 4
    sizes = [key[2][1][0][1][1] for key in graphs.buckets]
    assert (2,) not in sizes
    assert (1,) in sizes


def test_failure_disables_bucket_and_new_generation_clears(cuda, caplog):
    invoke, fail = cuda
    graphs = cuda_graphs.Graphs()
    cache = {}
    graphs.generation((0,), cache)
    graphs.call(args(), cache, invoke)
    fail[0] = True
    assert torch.equal(graphs.call(args(), cache, invoke), torch.tensor([2.0]))
    assert len(graphs.failed) == 1 and not graphs.buckets
    graphs.call(args(), cache, invoke)
    assert caplog.text.count("CUDA graph disabled") == 1
    graphs.generation((0,), {})
    assert not graphs.failed


def test_memory_budget_refuses_capture(cuda, monkeypatch):
    invoke, _ = cuda
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1, 100))
    graphs = cuda_graphs.Graphs()
    graphs.call(args(), {}, invoke)
    assert not graphs.buckets


def test_safety_exclusions(monkeypatch):
    mm = types.ModuleType("comfy.model_management")
    mm.VRAMState = types.SimpleNamespace(LOW_VRAM=1, NO_VRAM=2)
    mm.vram_state = 0
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    out = types.SimpleNamespace(model=types.SimpleNamespace(model_lowvram=False))
    call = {"input": types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))}
    frozen, targets = {"layer": {}}, {"layer": 0.4}

    def check(**kwargs):
        return cuda_graphs.safe(
            out,
            call,
            frozen,
            targets,
            kwargs.get("carry", False),
            kwargs.get("progress", 0.5),
            mutable=kwargs.get("mutable", False),
        )

    assert check()
    assert not check(carry=True)
    assert not check(mutable=True)
    assert not check(progress=0.3)
    assert not cuda_graphs.safe(out, call, {}, targets, False, 0.5)
    for state in (1, 2):
        mm.vram_state = state
        assert not check()
    mm.vram_state = 0
    out.model.model_lowvram = True
    assert not check()
    out.model.model_lowvram = False
    for active in (
        compute._PROBES,
        compute._ROUTES,
        compute._SHARES,
        route_freeze.RECORDS,
        route_freeze._OFF,
        linear_jump._RECORDS,
    ):
        if hasattr(active, "pushed"):
            with active.pushed({}):
                assert not check()
            continue
        active.append({})
        try:
            assert not check()
        finally:
            active.pop()
    with perturbing(object()):
        assert not check()
    call["input"].device.type = "cpu"
    assert not check()


def test_disabled_wrapper_and_generation_reset(monkeypatch):
    from kilora.comfyui.patching import StepCarry, _key_mask_by_step

    monkeypatch.delenv("KILORA_CUDA_GRAPHS", raising=False)
    out = types.SimpleNamespace(model_options={})
    _key_mask_by_step(out)
    calls = []

    def model(x, timestep, **conditions):
        calls.append(timestep)
        return x + timestep

    call = {"input": torch.ones(1, 1, 1), "timestep": torch.tensor([2.0]), "c": {}}
    assert torch.equal(
        out.model_options["model_function_wrapper"](model, call), torch.full((1, 1, 1), 3.0)
    )
    assert len(calls) == 1
    carrier = StepCarry()
    first = carrier.cache_for(call)
    call["timestep"] = torch.tensor([1.0])
    assert carrier.cache_for(call) is first
    call["timestep"] = torch.tensor([2.0])
    assert carrier.cache_for(call) is not first


def test_capture_exception_disables_bucket(cuda, monkeypatch, caplog):
    invoke, _ = cuda
    graphs = cuda_graphs.Graphs()
    calls = []

    def broken_graph():
        calls.append(1)
        raise RuntimeError("capture failed")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", broken_graph)
    cache = {}
    assert torch.equal(graphs.call(args(), cache, invoke), torch.tensor([2.0]))
    assert torch.equal(graphs.call(args(), cache, invoke), torch.tensor([2.0]))
    assert len(calls) == 1
    assert caplog.text.count("CUDA graph disabled") == 1


def test_graphs_switch_refuses_in_this_release(monkeypatch, caplog):
    from kilora.comfyui import cuda_graphs
    from kilora.comfyui.patching import _key_mask_by_step

    monkeypatch.setenv("KILORA_CUDA_GRAPHS", "1")
    monkeypatch.setattr(cuda_graphs, "_WARNED", [])
    monkeypatch.setattr(cuda_graphs.Graphs, "call", lambda *a, **k: 1 / 0)
    with caplog.at_level("WARNING", "kilora.cuda_graphs"):
        assert not cuda_graphs.requested() and not cuda_graphs.requested()
        out = types.SimpleNamespace(model_options={})
        _key_mask_by_step(out)
        call = {"input": torch.ones(1, 1, 1), "timestep": torch.tensor([2.0]), "c": {}}
        got = out.model_options["model_function_wrapper"](lambda x, t, **c: x + t, call)
    assert torch.equal(got, torch.full((1, 1, 1), 3.0))
    assert len([r for r in caplog.records if "disabled in this release" in r.getMessage()]) == 1
