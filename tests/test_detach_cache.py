# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import gc
import sys
import types
import weakref

import pytest
import torch
from test_bypass import Patcher
from test_comfyui import FakeBaseModel
from test_families import H, _model
from test_working_copies import routed_chain

from kilora import bake, compute
from kilora.comfyui import live_write, patching
from kilora.comfyui.bypass import ForwardHook, same_patcher_reload


@pytest.mark.parametrize("baseline", [True, False], ids=["baseline", "retained"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_repeated_load_cycle(tmp_path, monkeypatch, baseline, device):
    base = Patcher(FakeBaseModel(_model("flux")))
    out, _ = patching.apply(
        base, routed_chain(tmp_path), 1, joint=True, bypass="on", import_score="keys"
    )
    dtype = torch.float64 if device == "cpu" else torch.float32
    out.model.to(device=device, dtype=dtype)
    out.load_device = torch.device(device)
    out.model.device = out.load_device
    out.model.model_loaded_weight_memory = 1
    callbacks = out.callbacks["on_detach_after"]["kilora"]
    plans = patching.applied(out)["key_pass"].module_plans
    counts = dict(clear=0, on=0, derived=0, inject=0, eject=0)
    original_callback = callbacks[0]

    def callback(*args):
        populated = any(plan._cache for plan in plans)
        result = original_callback(*args)
        counts["clear"] += populated and not any(plan._cache for plan in plans)
        return result

    callbacks[0] = callback
    original_on, original_get = bake.ModulePlan._on, compute._cache_get
    original_inject, original_eject = ForwardHook.inject, ForwardHook.eject

    def on(plan, device, dtype):
        counts["on"] += (device, dtype) not in plan._cache
        return original_on(plan, device, dtype)

    def cache_get(key):
        result = original_get(key)
        if key is not None and result is None:
            counts["derived"] += 1
        return result

    def inject(hook, module):
        counts["inject"] += not hook.injected
        return original_inject(hook, module)

    def eject(hook):
        counts["eject"] += hook.injected
        return original_eject(hook)

    monkeypatch.setattr(bake.ModulePlan, "_on", on)
    monkeypatch.setattr(compute, "_cache_get", cache_get)
    monkeypatch.setattr(ForwardHook, "inject", inject)
    monkeypatch.setattr(ForwardHook, "eject", eject)
    if baseline:
        monkeypatch.setattr("kilora.comfyui.bypass.retaining", lambda patcher: False)
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = [types.SimpleNamespace(model=out)]

    def load(models):
        assert models == [out]
        out.eject()
        for cb in callbacks:
            cb(out, False)
        out.inject()
        out.eject()
        out.inject()

    mm.load_models_gpu = load
    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    handle = live_write.PathHandle(None, None, None, patcher=out)
    x = torch.randn(2, 5, H, dtype=dtype, device=device)
    paths = (
        "diffusion_model.double_blocks.0.img_attn.qkv",
        "diffusion_model.double_blocks.0.img_mod.lin",
    )
    results = []
    compute._DERIVED.clear()
    with torch.no_grad():
        for _ in range(31):
            live_write.ensure_loaded(handle)
            results.append([out.model.get_submodule(path)(x) for path in paths])
    print("CYCLE_COUNTS", device, "baseline" if baseline else "retained", counts)
    assert all(
        torch.equal(a, b) for result in results[1:] for a, b in zip(results[0], result, strict=True)
    )
    assert counts["clear"] == (30 if baseline else 0)
    assert counts["on"] == (31 if baseline else 1)
    assert counts["inject"] == (124 if baseline else 2)
    assert counts["eject"] == (122 if baseline else 0)
    cached = weakref.ref(next(iter(plans[0]._cache.values())))
    with same_patcher_reload(out):
        out.model.model_loaded_weight_memory = 0
        out.eject()
        callbacks[0](out, True)
    gc.collect()
    assert cached() is None and not plans[0]._cache
    out.model.model_loaded_weight_memory = 1
    with torch.no_grad():
        live_write.ensure_loaded(handle)
        after_unload = [out.model.get_submodule(path)(x) for path in paths]
    assert all(torch.equal(a, b) for a, b in zip(results[0], after_unload, strict=True))
    assert counts["on"] == (32 if baseline else 2)
    with monkeypatch.context() as old, torch.no_grad():
        old.setattr("kilora.comfyui.bypass.retaining", lambda patcher: False)
        live_write.ensure_loaded(handle)
        without_retention = [out.model.get_submodule(path)(x) for path in paths]
    assert all(torch.equal(a, b) for a, b in zip(results[0], without_retention, strict=True))
    out.callbacks["on_cleanup"]["kilora"][0](out)
    assert not plans[0]._cache
    assert all("forward" not in out.model.get_submodule(path).__dict__ for path in paths)


def test_switch_patcher_ejects_old_hooks(tmp_path):
    chain = routed_chain(tmp_path)
    base = Patcher(FakeBaseModel(_model("flux")))
    first, _ = patching.apply(base, [chain[0]], 1, bypass="on", gate=("none", None))
    second, _ = patching.apply(base, [chain[1]], 1, bypass="on", gate=("none", None))
    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    layer = base.model.get_submodule(path)
    x = torch.randn(2, H)
    with torch.no_grad():
        plain = layer(x)
        first.inject()
        first_result = layer(x)
        first.eject()
        assert torch.equal(layer(x), plain)
        second.inject()
        second_result = layer(x)
        second.eject()
        assert torch.equal(layer(x), plain)
        second.inject()
        assert torch.equal(layer(x), second_result)
        second.eject()
    assert not torch.equal(first_result, second_result)


def test_other_injections_keep_normal_ejection(tmp_path):
    base = Patcher(FakeBaseModel(_model("flux")))
    out, _ = patching.apply(base, [routed_chain(tmp_path)[0]], 1, bypass="on")
    out.load_device = torch.device("cpu")
    out.model.device = out.load_device
    out.model.model_loaded_weight_memory = 1
    out.inject()
    out.injections["other"] = [object()]
    with same_patcher_reload(out):
        out.injections["kilora"][0].eject(out)
    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    assert "forward" not in out.model.get_submodule(path).__dict__


def test_partial_unload_to_zero_clears_working_copies(tmp_path):
    base = Patcher(FakeBaseModel(_model("flux")))
    out, _ = patching.apply(
        base, routed_chain(tmp_path), 1, joint=True, bypass="on", import_score="keys"
    )
    out.model.double()
    out.model.model_loaded_weight_memory = 1
    plan = patching.applied(out)["key_pass"].module_plans[0]
    layer = out.model.get_submodule("diffusion_model.double_blocks.0.img_attn.qkv")
    out.inject()
    with torch.no_grad():
        layer(torch.randn(2, H, dtype=torch.float64))
    cached = weakref.ref(next(iter(plan._cache.values())))
    out.eject()
    out.model.model_loaded_weight_memory = 0
    out.inject()
    gc.collect()
    assert cached() is None and not plan._cache
    out.eject()


def test_failed_reload_removes_retained_hooks(tmp_path, monkeypatch):
    base = Patcher(FakeBaseModel(_model("flux")))
    out, _ = patching.apply(base, [routed_chain(tmp_path)[0]], 1, bypass="on")
    out.load_device = torch.device("cpu")
    out.model.device = out.load_device
    out.model.model_loaded_weight_memory = 1
    out.inject()
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = [types.SimpleNamespace(model=out)]

    def fail(_models):
        out.eject()
        raise RuntimeError("load failed")

    mm.load_models_gpu = fail
    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    with pytest.raises(RuntimeError, match="load failed"):
        live_write.ensure_loaded(live_write.PathHandle(None, None, None, patcher=out))
    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    assert "forward" not in out.model.get_submodule(path).__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unload_to_cpu_releases_cuda_copies_and_rebuilds(tmp_path):
    base = Patcher(FakeBaseModel(_model("flux")))
    out, _ = patching.apply(
        base, routed_chain(tmp_path), 1, joint=True, bypass="on", import_score="keys"
    )
    plan = patching.applied(out)["key_pass"].module_plans[0]
    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    layer = out.model.get_submodule(path)
    x = torch.randn(2, H)
    out.inject()
    with torch.no_grad():
        cpu_expected = layer(x)
        out.model.cuda()
        layer(x.cuda())
        cached = weakref.ref(next(iter(plan._cache.values())))
        out.load_device = torch.device("cuda")
        out.model.device = out.load_device
        out.model.model_loaded_weight_memory = 0
        with same_patcher_reload(out):
            out.eject()
            out.model.cpu()
            out.model.device = torch.device("cpu")
            out.callbacks["on_detach_after"]["kilora"][0](out, True)
        gc.collect()
        assert cached() is None and not plan._cache
        out.inject()
        assert torch.equal(layer(x), cpu_expected)
        assert next(iter(plan._cache))[0].type == "cpu"
    out.eject()
