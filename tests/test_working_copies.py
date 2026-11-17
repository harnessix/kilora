# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import gc
import weakref

import pytest
import torch
from safetensors.torch import save_file
from test_bypass import Patcher
from test_comfyui import FakeBaseModel, FakePatcher
from test_families import H, _ab, _model
from test_sections import _mod

from kilora.bake import is_baked
from kilora.comfyui import patching
from kilora.fixed_change import fixed_copy, is_fixed
from kilora.sections import LowRankChange, Section


class MemoryPatcher(FakePatcher):
    def clone(self):
        return type(self)(self.model, self.object_patches, self.model_options)

    def memory_required(self, input_shape):
        return 1234


def make_patched(tmp_path, patcher_type=MemoryPatcher, bypass=None):
    down = torch.randn(2, H)
    up = torch.randn(6 * H, 2)
    path, _ = _mod(
        tmp_path, [Section("fixed-lora", "fixed", {"double_blocks.0.img_mod.lin": (down, up)})]
    )
    model = FakeBaseModel(_model("flux"))
    model.get_dtype = lambda: torch.float32
    base = patcher_type(model)
    out, _ = patching.apply(base, [patching.ModEntry("mod", str(path))], 1, bypass=bypass)
    return base, out


def routed_chain(tmp_path):
    chain = []
    for i in range(2):
        gen = torch.Generator().manual_seed(i)
        weights = {}
        for name, size in (("attn.to_q", H), ("norm1.linear", 6 * H)):
            down, up = _ab(gen, H, size)
            prefix = f"transformer.transformer_blocks.0.{name}"
            weights[f"{prefix}.lora_A.weight"] = down
            weights[f"{prefix}.lora_B.weight"] = up
        path = tmp_path / f"routed-{i}.safetensors"
        save_file(weights, str(path))
        chain.append(patching.ModEntry("lora", str(path)))
    return chain


@pytest.mark.parametrize("bypass", ["off", "on"])
def test_released_routed_patcher_needs_no_cyclic_collection(tmp_path, bypass):
    chain = routed_chain(tmp_path)
    base = Patcher(FakeBaseModel(_model("flux")))
    gc.collect()
    gc.disable()
    try:
        out, _ = patching.apply(base, chain, 1, joint=True, bypass=bypass, import_score="keys")
        key_pass = out.model_options[patching.APPLIED_KEY]["key_pass"]
        plans = key_pass.module_plans
        tensor_refs = [
            weakref.ref(s.down)
            for p in plans
            for _, _, t in p.parts
            for s in t.sets
            if s.down is not None
        ]
        copy_refs = [weakref.ref(m) for m in out.object_patches.values()]
        if bypass == "on":
            inject = out.injections["kilora"][0].inject
            hooks = next(
                cell.cell_contents
                for cell in inject.__closure__
                if isinstance(cell.cell_contents, list)
            )
            copy_refs.extend(
                weakref.ref(hook.after.__self__)
                for hook in hooks
                if isinstance(getattr(hook.after, "__self__", None), torch.nn.Module)
            )
            del inject, hooks
            out.inject()
        try:
            module = out.get_model_object("diffusion_model.double_blocks.0.img_attn.qkv")
            mount = out.get_model_object("diffusion_model.double_blocks.0.img_mod.lin")
            key_pass.done = True

            def forward(x, timestep, _module=module, _mount=mount, **kwargs):
                return _module(x), _mount(x)

            with torch.no_grad():
                out.model_options["model_function_wrapper"](
                    forward, {"input": torch.randn(2, H), "timestep": torch.tensor(1), "c": {}}
                )
        finally:
            if bypass == "on":
                out.eject()
        del module, mount, forward, plans, key_pass, out
        assert tensor_refs and all(ref() is None for ref in tensor_refs)
        assert copy_refs and all(ref() is None for ref in copy_refs)
    finally:
        gc.enable()


def test_routed_import_reused_without_changing_output(tmp_path, monkeypatch):
    from kilora import multi_import

    chain = routed_chain(tmp_path)
    base = Patcher(FakeBaseModel(_model("flux")))
    patching._ROUTED_CACHE.clear()
    original = multi_import.import_loras
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(multi_import, "import_loras", counted)
    first, _ = patching.apply(base, chain, 1, joint=True, bypass="off")
    x = torch.randn(2, H)
    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    with torch.no_grad():
        expected = first.get_model_object(path)(x)
        for _, _, target in first.get_model_object(path)._kilora_plan.parts:
            for layer in target.sets:
                if layer.down is not None:
                    layer.down.zero_()
        second, _ = patching.apply(base, chain, 1, joint=True, bypass="off")
        assert torch.equal(expected, second.get_model_object(path)(x))
    assert len(calls) == 1
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    patching.apply(base, chain, 1, joint=True, bypass="off")
    assert len(calls) == 2


def test_unload_clears_caches_and_preserves_masters(tmp_path):
    _, out = make_patched(tmp_path)
    baked = next(m for m in out.object_patches.values() if is_baked(m))
    fixed = next(m for m in out.object_patches.values() if is_fixed(m))
    plan = baked._kilora_plan
    change = fixed.changes[0]
    master = plan.parts[0][2].sets[0].down
    fixed_master = change.down
    inputs = [(baked, torch.randn(2, baked.in_features)), (fixed, torch.randn(2, H))]
    with torch.no_grad():
        before = [m(x) for m, x in inputs]
        assert plan._cache and change._cast
        for module, _ in inputs:
            module.to("cpu")
        assert not plan._cache and not change._cast
        after = [m(x) for m, x in inputs]
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
        for module, _ in inputs:
            module.to("meta")
    assert not plan._cache and not change._cast
    assert plan.parts[0][2].sets[0].down is master
    assert change.down is fixed_master


@pytest.mark.parametrize("kind", ["baked", "fixed"])
def test_copy_move_keeps_base_parameters_and_buffers(tmp_path, kind):
    _, out = make_patched(tmp_path)
    module = next(
        m for m in out.object_patches.values() if (is_baked(m) if kind == "baked" else is_fixed(m))
    )
    base = module.__dict__["_kilora_base"]
    base.register_buffer("test_buffer", torch.ones(1))
    module.register_buffer("test_buffer", base.test_buffer)
    weight, buffer = base.weight, base.test_buffer
    assert module.weight is weight and module.test_buffer is buffer
    x = torch.randn(2, module.in_features)
    with torch.no_grad():
        expected = module(x)
        module.to(torch.float64)
        result = module(x.double())
        plain = torch.nn.functional.linear(x.double(), module.weight, module.bias)
        if kind == "baked":
            reference = module._kilora_plan.run(x.double(), plain)
        else:
            reference = module.changes[0].add(x.double(), plain)
    assert torch.equal(result, reference)
    assert base.weight is weight and base.weight.dtype == torch.float32
    assert base.test_buffer is buffer and buffer.dtype == torch.float32
    assert module.weight.dtype == module.test_buffer.dtype == torch.float64
    torch.testing.assert_close(result, expected.double(), rtol=1e-5, atol=1e-5)


def test_clear_removes_inherited_working_copy_callbacks(tmp_path):
    _, out = make_patched(tmp_path, FakePatcher)
    clone = out.clone()
    assert clone.callbacks["on_detach_after"]["kilora"]
    cleared = patching.clear(out)
    assert "kilora" not in cleared.callbacks["on_detach_after"]
    assert "kilora" not in cleared.callbacks["on_cleanup"]
    assert out.callbacks["on_detach_after"]["kilora"]


def test_stock_weight_replacement_stays_on_patched_layer(tmp_path):
    _, out = make_patched(tmp_path)
    path, module = next((p, m) for p, m in out.object_patches.items() if is_baked(m))
    base = patching._base_module(out, path)
    original = base.weight
    replacement = torch.nn.Parameter(original.detach().clone() + 1)
    module._parameters["weight"] = replacement
    assert patching._base_module(out, path) is base
    assert base.weight is original and module.weight is replacement
    x = torch.randn(2, module.in_features)
    with torch.no_grad():
        plain = torch.nn.functional.linear(x, replacement, module.bias)
        assert torch.equal(module(x), module._kilora_plan.run(x, plain))


@pytest.mark.parametrize("bypass", ["off", "on"])
@pytest.mark.parametrize("event", ["on_detach_after", "on_cleanup"])
def test_callback_clears_all_working_copies(tmp_path, monkeypatch, bypass, event):
    found = {}
    original_plans, original_changes = patching.module_plans, patching._fixed_changes

    def capture_plans(*args):
        found["plans"] = original_plans(*args)
        return found["plans"]

    def capture_changes(*args):
        result = original_changes(*args)
        found["changes"] = [c for group in result.values() for c in group]
        return result

    monkeypatch.setattr(patching, "module_plans", capture_plans)
    monkeypatch.setattr(patching, "_fixed_changes", capture_changes)
    _, out = make_patched(tmp_path, Patcher, bypass)
    if bypass == "on":
        out.inject()
        modules = [out.get_model_object(path) for path in found["plans"]]
        modules += [out.model.diffusion_model.get_submodule("double_blocks.0.img_mod.lin")]
    else:
        modules = list(out.object_patches.values())
    inputs = [torch.randn(2, module.in_features) for module in modules]
    try:
        with torch.no_grad():
            before = [m(x) for m, x in zip(modules, inputs, strict=True)]
            assert all(plan._cache for plan in found["plans"].values())
            assert all(change._cast for change in found["changes"])
            out.callbacks[event]["kilora"][0](out)
            assert all(not plan._cache for plan in found["plans"].values())
            assert all(not change._cast for change in found["changes"])
            if bypass == "on":
                out.inject()
            after = [m(x) for m, x in zip(modules, inputs, strict=True)]
            assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    finally:
        if bypass == "on":
            out.eject()


def test_memory_estimate_survives_clone_and_clear(tmp_path):
    base, out = make_patched(tmp_path)
    count = 0
    for module in out.object_patches.values():
        if is_baked(module):
            for _, _, target in module._kilora_plan.parts:
                for layer in target.sets:
                    for name in (
                        "router",
                        "keys_in",
                        "keys_out",
                        "down",
                        "up",
                        "share_transition_weighted",
                    ):
                        value = getattr(layer, name, None)
                        if isinstance(value, torch.Tensor):
                            count += value.numel()
                    for name in ("rows_in", "rows_out"):
                        value = getattr(layer, name, None)
                        if isinstance(value, torch.Tensor):
                            count += value.numel() * value.element_size() // 4
                    count += sum(t.numel() for t in (layer.gate or {}).values())
        if is_fixed(module):
            count += sum(
                v.numel()
                for c in module.changes
                for v in vars(c).values()
                if isinstance(v, torch.Tensor)
            )
    expected = base.memory_required([1]) + count * 4
    assert out.memory_required([1]) == expected
    assert out.clone().memory_required([1]) == expected
    assert patching.clear(out).memory_required([1]) == base.memory_required([1])
    out.model.memory_required = lambda *args, **kwargs: 300
    original = out.model.memory_required
    wrapper = out.model_options["transformer_options"]["wrappers"]["prepare_sampling"][
        patching._MEMORY_KEY
    ][0]
    assert wrapper(lambda p: p.model.memory_required([1]), out) == 300 + count * 4
    assert out.model.memory_required is original
    with pytest.raises(RuntimeError):
        wrapper(lambda p: (_ for _ in ()).throw(RuntimeError()), out)
    assert out.model.memory_required is original


def test_nested_fixed_and_baked_clear_both_caches(tmp_path):
    _, out = make_patched(tmp_path)
    baked = next(m for m in out.object_patches.values() if is_baked(m))
    change = LowRankChange(torch.randn(2, baked.in_features), torch.randn(baked.out_features, 2))
    combined = fixed_copy(baked, [change])
    with torch.no_grad():
        combined(torch.randn(2, baked.in_features))
    assert combined._kilora_plan._cache and change._cast
    combined.to("cpu")
    assert not combined._kilora_plan._cache and not change._cast


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_unload_releases_working_copies(tmp_path):
    _, out = make_patched(tmp_path)
    modules = list(out.object_patches.values())
    for module in modules:
        module.to("cuda")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    with torch.no_grad():
        for module in modules:
            module(torch.randn(2, module.in_features, device="cuda"))
    out.callbacks["on_detach_after"]["kilora"][0](out)
    for module in modules:
        module.to("cpu")
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() <= baseline + 1024 * 1024


def _load_copies(patcher, device):
    for path, module in patcher.object_patches.items():
        parent_path, _, name = path.rpartition(".")
        parent = patcher.model.get_submodule(parent_path)
        setattr(parent, name, module)
        if not getattr(module, "comfy_patched_weights", False):
            module.to(device)
            module.comfy_patched_weights = True


def test_set_switch_loads_clean_copies_from_live_layers(tmp_path):
    chain = routed_chain(tmp_path)
    base = MemoryPatcher(FakeBaseModel(_model("flux")))
    paths = (
        "diffusion_model.double_blocks.0.img_attn.qkv",
        "diffusion_model.double_blocks.0.img_mod.lin",
    )
    originals = [base.model.get_submodule(path) for path in paths]
    for layer in originals:
        layer.comfy_cast_weights = True
        layer.prev_comfy_cast_weights = False
        layer.comfy_patched_weights = True
        layer.comfy_force_cast_weights = True
        layer.weight_function = [object()]
        layer.bias_function = [object()]
    first, _ = patching.apply(base, [chain[0]], 1, bypass="off")
    assert set(first.object_patches) == set(paths)
    for layer in first.object_patches.values():
        assert not hasattr(layer, "comfy_patched_weights")
        assert not hasattr(layer, "prev_comfy_cast_weights")
        assert not hasattr(layer, "comfy_force_cast_weights")
        assert layer.comfy_cast_weights is False
        assert layer.weight_function == layer.bias_function == []
    _load_copies(first, "meta")
    previous = {p: first.model.get_submodule(p) for p in paths}
    second, _ = patching.apply(base, [chain[1]], 1, bypass="off")
    for path, layer in second.object_patches.items():
        assert layer.weight.device.type == "meta"
        assert layer.weight is not previous[path].weight
        assert not hasattr(layer, "comfy_patched_weights")
        assert layer.weight_function == layer.bias_function == []
    _load_copies(second, "meta")
    assert all(layer.weight.device.type == "meta" for layer in second.object_patches.values())
    assert all(layer.weight.device.type == "cpu" for layer in originals)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_set_switch_matches_fresh_model(tmp_path):
    import copy

    chain = routed_chain(tmp_path)
    original = FakeBaseModel(_model("flux"))
    fresh = MemoryPatcher(copy.deepcopy(original))
    base = MemoryPatcher(original)
    first, _ = patching.apply(base, [chain[0]], 1, bypass="off", gate=("none", None))
    _load_copies(first, "cuda")
    x = torch.randn(2, H, device="cuda")
    with torch.no_grad():
        for layer in first.object_patches.values():
            layer(x)
        second, _ = patching.apply(base, [chain[1]], 1, bypass="off", gate=("none", None))
        _load_copies(second, "cuda")
        expected, _ = patching.apply(fresh, [chain[1]], 1, bypass="off", gate=("none", None))
        _load_copies(expected, "cuda")
        for path, layer in second.object_patches.items():
            assert torch.equal(layer(x), expected.object_patches[path](x))
