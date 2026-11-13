# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_derived_cache_entries_go_with_their_tensors():
    import gc

    from kilora import compute

    a, b = torch.randn(64, 8), torch.randn(8, 64)
    key = compute._tensor_key("test-derived", [a, b])
    compute._cache_put(key, torch.zeros(1024))
    assert compute._cache_get(key) is not None and key[0] in compute._DERIVED
    del a, b
    gc.collect()
    assert key[0] not in compute._DERIVED


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_derived_cache_returns_gpu_memory():
    import gc

    from kilora import compute

    torch.randn(1, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    a = torch.randn(1024, 1024, device="cuda")
    compute._cache_put(compute._tensor_key("test-gpu", [a]), torch.zeros(4096, 1024, device="cuda"))
    del a
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 2**20


def test_a_value_that_holds_its_source_does_not_keep_it_alive():
    import gc
    import weakref

    from kilora import compute

    a = torch.randn(6, 4)
    key = compute._tensor_key("test-holds-source", [a])
    view = a.t()

    class Tables:
        pass

    tables = Tables()
    tables.up, tables.view, tables.other = a, view, torch.ones(3)
    compute._cache_put(key, (tables, {"v": view}))
    got, extra = compute._cache_get(key)
    assert got.up is a and torch.equal(got.view, view) and torch.equal(extra["v"], view)
    assert got.view.stride() == view.stride()
    assert got.other is tables.other
    ref = weakref.ref(a)
    del a, view, tables, got, extra
    gc.collect()
    assert ref() is None and key[0] not in compute._DERIVED


def test_derived_keys_digest_cache_keeps_only_cpu_copies():
    import gc
    import weakref

    from kilora import compute

    s = compute.LayerSet(
        down=torch.randn(8, 12),
        up=torch.randn(8, 10),
        router=torch.randn(4, 10),
        keys_in=torch.randn(2, 1, 4),
        per_expert=4,
    )
    kin = compute.derived_keys(s)[0]
    ref = weakref.ref(kin)
    stored = list(compute._DERIVED_KEYS_BY_DIGEST.values())
    assert stored and all(t.device.type == "cpu" for v in stored for t in v)
    assert all(t is not kin for v in stored for t in v)
    assert compute._derived_keys_bytes() <= compute._DERIVED_KEYS_BUDGET
    del s, kin
    gc.collect()
    assert ref() is None


def test_replaced_routed_sets_release_their_tensors(tmp_path):
    import gc
    import weakref

    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _ab, _model

    from kilora import compute
    from kilora.comfyui import patching

    model = _model("flux")
    generator = torch.Generator().manual_seed(83)
    layers = ("double_blocks.0.img_attn.proj", "single_blocks.1.linear2")

    def make_chain(prefix):
        entries = []
        for i in range(2):
            weights = {}
            for path in layers:
                layer = model.get_submodule(path)
                a, b = _ab(generator, layer.in_features, layer.out_features)
                stem = "lora_unet_" + path.replace(".", "_")
                weights[f"{stem}.lora_down.weight"] = a
                weights[f"{stem}.lora_up.weight"] = b
            file = tmp_path / f"{prefix}{i}.safetensors"
            save_file(weights, str(file))
            entries.append(patching.ModEntry("lora", str(file), 1.0))
        return patching.ModChain(tuple(entries))

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            yield from tensors(vars(value))

    def apply_and_forward(chain):
        patched, _ = patching.apply(
            FakePatcher(FakeBaseModel(model)),
            chain,
            1,
            joint=True,
            gate=("none", None),
            import_score="derived",
        )
        backups = []
        for path, baked in patched.object_patches.items():
            for _, _, plan in baked.__dict__["_kilora_plan"].parts:
                for layer_set in plan.sets:
                    for name, value in vars(layer_set).items():
                        if isinstance(value, torch.Tensor):
                            setattr(layer_set, name, value.clone())
            parent_name, _, name = path.rpartition(".")
            parent = patched.get_model_object(parent_name)
            backups.append((parent, name, getattr(parent, name)))
            setattr(parent, name, baked)
        try:
            with torch.no_grad():
                model()
        finally:
            for parent, name, old in backups:
                setattr(parent, name, old)
        return patched

    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()
    first = apply_and_forward(make_chain("a"))
    sources = [
        tensor
        for baked in first.object_patches.values()
        for _, _, plan in baked.__dict__["_kilora_plan"].parts
        for layer_set in plan.sets
        for tensor in tensors(vars(layer_set))
    ]
    assert sources
    owner_ids = {id(compute._owner(t)) for t in sources}
    entries = [
        entry
        for entry in compute._DERIVED.values()
        if any(ref is not None and ref() is not None and id(ref()) in owner_ids for ref in entry[0])
    ]
    values = [tensor for _, value, _ in entries for tensor in tensors(value)]
    assert entries and values
    assert compute._DERIVED_KEYS_BY_DIGEST
    digest_values = [t for value in compute._DERIVED_KEYS_BY_DIGEST.values() for t in value]
    assert all(t.device.type == "cpu" for t in digest_values)
    assert not {id(t) for t in digest_values} & {id(t) for t in sources + values}
    refs = [weakref.ref(t) for t in sources + values]
    stripped_count = sum(entry[2] for entry in entries)
    print(f"tiny Flux derived entries stripped: {stripped_count}/{len(entries)}")
    del sources, entries, values, digest_values

    second = apply_and_forward(make_chain("b"))
    del first
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert second.object_patches


def test_dtype_view_in_cached_container_does_not_hold_source():
    import gc
    import weakref

    from kilora import compute

    source = torch.randn(8)
    ref = weakref.ref(source)
    key = compute._tensor_key("dtype-view", [source])
    compute._cache_put(key, {"view": source.view(torch.int32)})
    assert torch.equal(compute._cache_get(key)["view"], source.view(torch.int32))
    del source
    gc.collect()
    assert ref() is None
    assert key[0] not in compute._DERIVED


def test_derived_keys_cache_respects_budget_for_one_entry(monkeypatch):
    from kilora import compute

    monkeypatch.setattr(compute, "_DERIVED_KEYS_BUDGET", 1)
    monkeypatch.setattr(compute, "_DERIVED_KEYS_BY_DIGEST", {})
    layer_set = compute.LayerSet(
        down=torch.randn(8, 12),
        up=torch.randn(8, 10),
        router=torch.randn(4, 10),
        keys_in=torch.randn(2, 1, 4),
        per_expert=4,
    )
    compute.derived_keys(layer_set)
    assert compute._derived_keys_bytes() <= 1
