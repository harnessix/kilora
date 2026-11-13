# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_comfyui import denoiser, folders
from test_import_modes import Toy, _lora

from kilora import compose, compute, derived_cache, key_cache
from kilora.bake import ScopedSet, plan_targets
from kilora.multi_import import import_loras


@pytest.fixture
def chain(folders, tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from kilora.comfyui import nodes as N

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "key-cache"))
    out = None
    for i in range(3):
        gen = torch.Generator().manual_seed(20 + i)
        sd = {}
        for layer in ("0", "1"):
            stem = f"lora_unet_layers_{layer}_feed_forward_w2"
            sd[f"{stem}.lora_down.weight"] = torch.randn(2, 32, generator=gen)
            sd[f"{stem}.lora_up.weight"] = torch.randn(16, 2, generator=gen)
        save_file(sd, str(folders["loras"] / f"s{i}.safetensors"))
        (out,) = N.KiLoRAImportLoRA().add(f"s{i}.safetensors", 1.0, sets=out)
    return out


def _clear():
    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()


def _files(tmp_path):
    model = Toy().eval()
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.safetensors"
        _lora(model, path, rank=3, seed=i, layers=["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"])
        paths.append(str(path))
    return paths


def test_batched_bases_match_per_expert_reference(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "3")
    torch.manual_seed(9)
    down = torch.randn(12, 7)
    up = torch.randn(12, 6)
    down[8:] = 0
    layer = compute.LayerSet(
        router=torch.zeros(1, 6),
        keys_in=torch.zeros(3, 1, 1),
        down=down,
        up=up,
        per_expert=4,
        form="input",
    )
    result = compute.derived_keys(layer)
    for weight, bases, counts in ((down, result[0], result[1]), (up, result[2], result[3])):
        for i, block in enumerate(weight.reshape(3, 4, -1)):
            _, singular, vh = torch.linalg.svd(block, full_matrices=False)
            tol = singular.max() * max(block.shape) * torch.finfo(singular.dtype).eps
            count = min(3, int((singular > tol).sum()))
            assert counts[i] == count
            torch.testing.assert_close(bases[i, :count], vh[:count], atol=1e-6, rtol=1e-6)
            assert torch.count_nonzero(bases[i, count:]) == 0


def test_apply_lazy_disk_reuse_and_output(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    paths = _files(tmp_path)
    count = 0
    orig_svd = torch.linalg.svd

    def svd(*args, **kwargs):
        nonlocal count
        count += 1
        return orig_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", svd)
    x = torch.randn(1, 3, 12)
    model = Toy().eval()
    first = compose(model, [import_loras(paths, model).adapter_set])
    assert count == 0
    with torch.no_grad():
        expected = model(x)
    assert count == 4
    first.remove()
    _clear()
    count = 0
    model = Toy().eval()
    second = compose(model, [import_loras(paths, model).adapter_set])
    with torch.no_grad():
        actual = model(x)
    assert count == 0
    assert torch.equal(expected, actual)
    second.remove()

    _lora(
        model,
        tmp_path / "0.safetensors",
        rank=3,
        seed=99,
        layers=["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"],
    )
    _clear()
    count = 0
    model = Toy().eval()
    third = compose(model, [import_loras(paths, model).adapter_set])
    with torch.no_grad():
        model(x)
    assert count == 4
    third.remove()
    _clear()
    count = 0
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "2")
    model = Toy().eval()
    fourth = compose(model, [import_loras(paths, model).adapter_set])
    with torch.no_grad():
        model(x)
    assert count == 4
    fourth.remove()


def test_plan_source_and_transition_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    model = Toy().eval()
    paths = []
    for i in range(3):
        path = tmp_path / f"transition{i}.safetensors"
        _lora(model, path, rank=3, seed=i, layers=["blocks.0.self_attn.q_proj"])
        paths.append(str(path))
    imported = import_loras(paths, model).adapter_set
    target = next(iter(imported.experts))
    layer = plan_targets([ScopedSet(imported)], 2)[target].sets[0]
    assert derived_cache.layer_source(layer)
    layer = (
        plan_targets([ScopedSet(imported)], 2)[target]
        .to(torch.device("cpu"), torch.float32)
        .sets[0]
    )
    assert derived_cache.layer_source(layer)
    first = compute.share_transition(layer)
    _clear()
    again = compute.share_transition(layer)
    assert torch.equal(first, again)
    layer.down.add_(1)
    assert derived_cache.layer_source(layer) is None
    changed = compute.share_transition(layer)
    assert not torch.equal(first, changed)
    ex = imported.experts[target]
    assert derived_cache.source(dataclasses.replace(ex, down=ex.down.clone())) is None


def test_inference_mode_keeps_file_source(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    with torch.inference_mode():
        imported = import_loras(paths, model).adapter_set
        target = next(iter(imported.experts))
        assert derived_cache.source(imported.experts[target])
        layer = (
            plan_targets([ScopedSet(imported)], 2)[target]
            .to(torch.device("cpu"), torch.float32)
            .sets[0]
        )
        assert derived_cache.layer_source(layer)
        adapter = compose(model, [imported])
        model(torch.randn(1, 3, 12))
        adapter.remove()


def test_comfy_apply_reuses_cached_bases(denoiser, chain, monkeypatch, tmp_path):
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _call, _sets

    from kilora.comfyui import patching

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "derived-cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    patching._ROUTED_CACHE.clear()
    base = FakePatcher(FakeBaseModel(denoiser))
    count = 0
    svd = torch.linalg.svd

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    with torch.inference_mode():
        first, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    assert _sets(first) and all(derived_cache.layer_source(s) for s in _sets(first))
    assert count == 0
    x, cond = torch.randn(1, 5, 4), torch.ones(1, 2)
    result = _call(first, x, cond)
    assert count > 0
    _clear()
    count = 0
    with torch.inference_mode():
        second, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    assert all(derived_cache.layer_source(s) for s in _sets(second))
    assert torch.equal(result, _call(second, x, cond))
    assert count == 0


def test_output_matches_previous_per_expert_svd(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    paths = _files(tmp_path)
    model = Toy().eval()
    imported = import_loras(paths, model).adapter_set
    x = torch.randn(1, 3, 12)
    orig_svd = torch.linalg.svd

    def loop_svd(a, *args, **kwargs):
        if a.ndim != 3:
            return orig_svd(a, *args, **kwargs)
        parts = [orig_svd(block, *args, **kwargs) for block in a]
        return tuple(torch.stack(items) for items in zip(*parts, strict=True))

    monkeypatch.setattr(derived_cache, "load", lambda *a, **k: None)
    monkeypatch.setattr(derived_cache, "store", lambda *a, **k: None)
    monkeypatch.setattr(torch.linalg, "svd", loop_svd)
    first = compose(model, [imported])
    with torch.no_grad():
        expected = model(x)
    first.remove()
    _clear()
    monkeypatch.setattr(torch.linalg, "svd", orig_svd)
    second = compose(model, [imported])
    with torch.no_grad():
        actual = model(x)
    second.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_immutable_schedule_maximum_is_reused(monkeypatch):
    from kilora.comfyui import depth

    for expression in ("1 + 3 * (t > 0.4)", "i / n + self + 2 * cross + 3 * mlp + t", "2.00000001"):
        schedule = depth.compile_depth(expression)
        expected = depth._max_depth(schedule, depth.PREVIEW_BLOCKS)
        depth._cached_max_depth.cache_clear()
        assert depth.max_depth(schedule) == expected
        before = depth._cached_max_depth.cache_info()
        assert depth.max_depth(depth.compile_depth(expression)) == expected
        assert depth._cached_max_depth.cache_info().hits == before.hits + 1
        assert depth.max_depth(schedule, blocks=10) == depth._max_depth(schedule, 10)


def test_schedule_maximum_tracks_default_dtype():
    from kilora.comfyui import depth

    schedule = depth.compile_depth("2.00000001")
    original = torch.get_default_dtype()
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            assert depth.max_depth(schedule) == depth._max_depth(schedule, depth.PREVIEW_BLOCKS)
    finally:
        torch.set_default_dtype(original)


def test_custom_reader_cannot_reuse_file_results(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    from kilora import multi_import

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    first = import_loras(paths, model).adapter_set
    calls = 0
    content_keys = multi_import._content_keys

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return content_keys(*args, **kwargs)

    def modified(path):
        weights, metadata, record = multi_import._read_file(path)
        for name in weights:
            if name.endswith("lora_up.weight"):
                weights[name] = weights[name].flip(0)
        return weights, metadata, record

    monkeypatch.setattr(multi_import, "_content_keys", counted)
    second = import_loras(paths, model, read_file=modified).adapter_set
    assert calls == len(first.experts)
    for target, expert in second.experts.items():
        assert derived_cache.source(expert) is None
        assert not torch.equal(expert.up, first.experts[target].up)


def test_content_keys_disk_cache_and_setting_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    original = torch.linalg.eigh
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counted)
    first = import_loras(paths, model).adapter_set
    assert calls == len(first.experts)
    calls = 0
    second = import_loras(paths, model).adapter_set
    assert calls == 0
    for target, expert in first.experts.items():
        assert torch.equal(expert.keys_in, second.experts[target].keys_in)
    import_loras(paths, model, router_dim=16)
    assert calls == len(first.experts)
    calls = 0
    changed_layout = Toy().eval()
    changed_layout.extra_projection = torch.nn.Linear(12, 12)
    layout_result = import_loras(paths, changed_layout).adapter_set
    assert calls == len(first.experts)
    for target, expert in first.experts.items():
        torch.testing.assert_close(expert.keys_in, layout_result.experts[target].keys_in)


def test_modified_layer_does_not_reuse_disk_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    imported = import_loras(paths, model).adapter_set
    target = next(iter(imported.experts))
    layer = plan_targets([ScopedSet(imported)], 2)[target].sets[0]
    before = compute.derived_keys(layer)
    layer.down[0].copy_(layer.down[0].roll(1))
    assert derived_cache.layer_source(layer) is None
    after = compute.derived_keys(layer)
    assert not torch.equal(before[0], after[0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_batched_subspace_matches_per_expert(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "3")
    torch.manual_seed(13)
    down = torch.randn(12, 32, device="cuda")
    up = torch.randn(12, 32, device="cuda")
    layer = compute.LayerSet(
        router=torch.zeros(1, 32, device="cuda"),
        keys_in=torch.zeros(3, 1, 1, device="cuda"),
        down=down,
        up=up,
        per_expert=4,
        form="input",
    )
    bases = compute.derived_keys(layer)
    assert bases[0].device.type == "cuda"
    for weights, got in ((down, bases[0]), (up, bases[2])):
        for index, block in enumerate(weights.reshape(3, 4, -1)):
            _, _, vh = torch.linalg.svd(block.float(), full_matrices=False)
            torch.testing.assert_close(
                got[index].T @ got[index], vh[:3].T @ vh[:3], atol=2e-5, rtol=2e-5
            )


def test_lokr_second_import_and_call_have_no_svd(monkeypatch, tmp_path):
    from test_multi_import import ToyAnima, _lokr, write

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    model = ToyAnima().eval()
    weights, _ = _lokr()
    path = write(tmp_path, "lokr", weights)
    x, ctx = torch.randn(1, 3, 16), torch.randn(1, 3, 8)
    count = 0
    original = torch.linalg.svd

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    first = compose(model, [import_loras([path], model, rank=1).adapter_set])
    with torch.no_grad():
        expected = model(x, ctx)
    assert count > 0
    first.remove()
    _clear()
    count = 0
    second = compose(model, [import_loras([path], model, rank=1).adapter_set])
    with torch.no_grad():
        actual = model(x, ctx)
    assert count == 0
    assert torch.equal(expected, actual)
    second.remove()


def test_individual_preparation_reuses_only_unchanged_experts(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    prepare_mix(paths[:1], model)
    _clear()
    batches = []
    original = torch.linalg.svd

    def counted(tensor, *args, **kwargs):
        if tensor.ndim == 3:
            batches.append(tensor.shape[0])
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    mixed = import_loras(paths[:2], model, device="cpu").adapter_set
    plans = plan_targets([ScopedSet(mixed)], 1)
    for plan in plans.values():
        compute.derived_keys(plan.sets[0])
    assert batches and all(size == 1 for size in batches)
    _clear()
    batches.clear()
    _lora(model, paths[1], rank=3, seed=99, layers=["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"])
    mixed = import_loras(paths[:2], model, device="cpu").adapter_set
    for plan in plan_targets([ScopedSet(mixed)], 1).values():
        compute.derived_keys(plan.sets[0])
    assert batches and all(size == 1 for size in batches)


def test_assembled_bases_match_batched_computation(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    mixed = import_loras(paths, model, device="cpu").adapter_set
    layers = [p.sets[0] for p in plan_targets([ScopedSet(mixed)], 1).values()]
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *a, **k: None)
        patch.setattr(derived_cache, "store", lambda *a, **k: None)
        expected = [compute.derived_keys(layer) for layer in layers]
    _clear()
    for path in paths:
        prepare_mix([path], model)
    _clear()
    mixed = import_loras(paths, model, device="cpu").adapter_set
    layers = [p.sets[0] for p in plan_targets([ScopedSet(mixed)], 1).values()]
    for original, layer in zip(expected, layers, strict=True):
        assert all(
            torch.equal(a, b) for a, b in zip(original, compute.derived_keys(layer), strict=True)
        )


def test_expert_cache_isolated_by_backend(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    prepare_mix(paths[:1], model)
    _clear()
    monkeypatch.setattr(derived_cache, "backend", lambda device: {"device": "stand-in-gpu"})
    imported = import_loras(paths[:1], model, device="cpu").adapter_set
    calls = []
    original = torch.linalg.svd

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    for plan in plan_targets([ScopedSet(imported)], 1).values():
        compute.derived_keys(plan.sets[0])
    assert calls


def test_loha_conversion_is_reused(monkeypatch, tmp_path):
    from test_multi_import import ToyAnima, _loha, write

    from kilora import multi_import

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    weights, _ = _loha()
    path = write(tmp_path, "loha", weights)
    model = ToyAnima().eval()
    first = import_loras([path], model, rank=2, device="cpu").adapter_set
    monkeypatch.setattr(multi_import, "low_rank", lambda *a, **k: pytest.fail("unexpected SVD"))
    second = import_loras([path], model, rank=2, device="cpu").adapter_set
    for name in first.experts:
        assert torch.equal(first.experts[name].down, second.experts[name].down)
        assert torch.equal(first.experts[name].up, second.experts[name].up)


def test_prepared_bases_cover_lower_precision_plan(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    prepare_mix(paths[:1], model)
    _clear()
    imported = import_loras(paths[:1], model, device="cpu").adapter_set
    plans = plan_targets([ScopedSet(imported)], 1)
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    for plan in plans.values():
        rounded = plan.to(torch.device("cpu"), torch.bfloat16)
        compute.derived_keys(rounded.sets[0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_prepared_bases_do_not_override_cuda(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    prepare_mix(paths[:1], model)
    _clear()
    imported = import_loras(paths[:1], model, device="cpu").adapter_set
    plans = plan_targets([ScopedSet(imported)], 1)
    for plan in plans.values():
        layer = plan.to(torch.device("cuda"), torch.float16).sets[0]
        expected = compute._derived_keys_current(layer, fresh=True)
        actual = compute.derived_keys(layer)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_prepared_mix_content_keys_reused_exactly(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *args, **kwargs: None)
        patch.setattr(derived_cache, "store", lambda *args, **kwargs: None)
        expected = import_loras(paths, model, device="cpu").adapter_set
    prepare_mix(paths, model)
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    actual = import_loras(paths, model, device="cpu").adapter_set
    for target in expected.experts:
        assert torch.equal(actual.experts[target].keys_in, expected.experts[target].keys_in)


def test_single_expert_entries_do_not_change_batch_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *args, **kwargs: None)
        patch.setattr(derived_cache, "store", lambda *args, **kwargs: None)
        expected = import_loras(paths, model, device="cpu").adapter_set
    for path in paths:
        import_loras([path], model, device="cpu")
    original = torch.linalg.eigh
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counted)
    actual = import_loras(paths, model, device="cpu").adapter_set
    assert calls and all(size == len(paths) for size in calls)
    for target in expected.experts:
        assert torch.equal(actual.experts[target].keys_in, expected.experts[target].keys_in)


def test_prepared_bases_match_fresh_after_plan_move(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    for path in paths:
        prepare_mix([path], model)
    _clear()
    imported = import_loras(paths, model, device="cpu").adapter_set
    plans = plan_targets([ScopedSet(imported)], 1)
    for plan in plans.values():
        layer = plan.to(torch.device("cpu"), torch.float32).sets[0]
        fresh = compute._derived_keys_current(layer, fresh=True)
        cached = compute.derived_keys(layer)
        for a, b in zip(cached, fresh, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_tied_content_direction_is_not_cached_per_expert():
    from kilora.multi_import import _content_keys

    up = torch.eye(2).reshape(1, 1, 2, 2)
    _, reusable = _content_keys(
        up, torch.eye(2), [{"sha256": "test"}], "layer", device="cpu", return_reusable=True
    )
    assert reusable == [False]


def test_old_format_entries_are_not_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    current = derived_cache._FORMAT
    monkeypatch.setattr(derived_cache, "_FORMAT", current - 1)
    derived_cache.store("source", "keys", {}, ("value",), (torch.ones(1),))
    monkeypatch.setattr(derived_cache, "_FORMAT", current)
    assert derived_cache.load("source", "keys", {}, ("value",)) is None


def test_content_cache_is_backend_specific_in_both_directions(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    backend = {"name": "cpu"}
    monkeypatch.setattr(derived_cache, "backend", lambda device: dict(backend))
    original = torch.linalg.eigh
    calls = []

    def counted(*args, **kwargs):
        calls.append(backend["name"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counted)
    first = import_loras(paths, model, device="cpu").adapter_set
    assert calls
    calls.clear()
    backend["name"] = "stand-in-cuda"
    second = import_loras(paths, model, device="cpu").adapter_set
    assert calls
    calls.clear()
    backend["name"] = "cpu"
    again = import_loras(paths, model, device="cpu").adapter_set
    assert not calls
    for target in first.experts:
        assert torch.equal(first.experts[target].keys_in, again.experts[target].keys_in)
        assert torch.equal(first.experts[target].keys_in, second.experts[target].keys_in)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "reverse"))
    backend["name"] = "stand-in-cuda"
    import_loras(paths, model, device="cpu")
    calls.clear()
    backend["name"] = "cpu"
    import_loras(paths, model, device="cpu")
    assert calls


def test_near_tied_expert_recomputed_with_mix(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    original = torch.linalg.eigh
    batches = []

    def tied(*args, **kwargs):
        result = original(*args, **kwargs)
        batches.append(args[0].shape[0])
        values = result.eigenvalues.clone()
        values[..., -2] = values[..., -1]
        return type(result)((values, result.eigenvectors))

    monkeypatch.setattr(torch.linalg, "eigh", tied)
    for path in paths:
        import_loras([path], model, device="cpu")
    batches.clear()
    import_loras(paths, model, device="cpu")
    assert batches
    assert all(size == len(paths) for size in batches)


def test_per_expert_keys_are_stable_across_mixes_and_ranks(monkeypatch, tmp_path):
    from safetensors.torch import load_file

    from kilora.multi_import import _per_expert_content_keys
    from kilora.sets import make_router

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    model = Toy().eval()
    paths = _files(tmp_path)
    _lora(model, tmp_path / "1.safetensors", rank=5, seed=1)
    for run, indices in enumerate(([0, 1], [2, 0, 1], [1, 0])):
        monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / f"fresh-{run}"))
        mix = import_loras([paths[i] for i in indices], model, device="cpu").adapter_set
        for target, expert in mix.experts.items():
            stem = "lora_unet_" + target.replace(".", "_")
            parts = {}
            for position, i in enumerate(indices):
                state = load_file(paths[i])
                if f"{stem}.lora_down.weight" in state:
                    parts[position] = (
                        state[f"{stem}.lora_down.weight"],
                        state[f"{stem}.lora_up.weight"],
                    )
            router = make_router(32, expert.up.shape[-1], 0)
            expected = _per_expert_content_keys(
                parts,
                mix.imported_from["sources"],
                target,
                router,
                0,
                "cpu",
                cache=False,
                checkpoint=None,
            )
            assert torch.equal(expert.keys_in, expected)


def test_per_expert_single_file_preparation_covers_new_mix(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path)
    model = Toy().eval()
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    stored = []
    original_store = derived_cache.store

    def count_entries(identity, kind, settings, names, values):
        if kind == "expert-content" and settings.get("algorithm") == "per-expert-v3":
            stored.append(identity)
        return original_store(identity, kind, settings, names, values)

    monkeypatch.setattr(derived_cache, "store", count_entries)
    for path in paths:
        prepare_mix([path], model, device="cpu")
    assert len(stored) == len(set(stored)) == len(paths) * 2
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    actual = import_loras(paths, model, device="cpu")
    assert actual.adapter_set.experts
    assert actual.content_key_counts == (0, 2)
    assert len(stored) == len(paths) * 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_single_preparation_covers_new_per_expert_mix(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    model = Toy().eval()
    paths = _files(tmp_path)
    for path in paths:
        prepare_mix([path], model, device="cuda")
    _clear()
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    mixed = import_loras(paths, model, device="cuda").adapter_set
    for plan in plan_targets([ScopedSet(mixed, import_score="derived")], 1).values():
        compute.derived_keys(plan.to(torch.device("cuda"), torch.float16).sets[0])


def test_per_expert_cache_is_backend_specific_and_caches_ties(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    backend = {"name": "cpu"}
    monkeypatch.setattr(derived_cache, "backend", lambda device: dict(backend))
    paths = _files(tmp_path)
    model = Toy().eval()
    original = torch.linalg.eigh
    calls = []

    def tied(*args, **kwargs):
        calls.append(backend["name"])
        result = original(*args, **kwargs)
        values = result.eigenvalues.clone()
        values[..., -2] = values[..., -1]
        return type(result)((values, result.eigenvectors))

    monkeypatch.setattr(torch.linalg, "eigh", tied)
    with caplog.at_level("INFO", logger="kilora.import"):
        first = import_loras(paths, model, device="cpu").adapter_set
    assert calls
    assert "nearly tied content directions" in caplog.text
    calls.clear()
    again = import_loras(list(reversed(paths)), model, device="cpu").adapter_set
    assert calls
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "mix-reference"))
    reference = import_loras(list(reversed(paths)), model, device="cpu").adapter_set
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    for target in first.experts:
        assert torch.equal(again.experts[target].keys_in, reference.experts[target].keys_in)
    calls.clear()
    import_loras(list(reversed(paths)), model, device="cpu")
    assert not calls
    backend["name"] = "other"
    import_loras(paths, model, device="cpu")
    assert calls


def test_per_expert_missing_targets_need_no_eigensolve(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    model = Toy().eval()
    paths = _files(tmp_path)
    _lora(model, tmp_path / "0.safetensors", rank=3, seed=0, layers=["blocks.0.mlp.layer1"])
    for path in paths:
        prepare_mix([path], model, device="cpu")
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    assert import_loras(paths, model, device="cpu").adapter_set.experts


def test_tied_layer_uses_mix_and_other_layers_use_prepared_keys(monkeypatch, tmp_path, caplog):
    from safetensors.torch import load_file, save_file

    from kilora import multi_import
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(multi_import, "make_router", lambda dim, width, seed: torch.eye(dim, width))
    model = Toy().eval()
    paths = _files(tmp_path)[:2]
    target = "blocks.0.mlp.layer1"
    stem = "lora_unet_blocks_0_mlp_layer1.lora_up.weight"
    for path in paths:
        state = load_file(path)
        up = torch.zeros_like(state[stem])
        up[0, 0] = up[1, 1] = 1
        state[stem] = up
        save_file(state, path)
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    for path in paths:
        prepare_mix([path], model, device="cpu")
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    with caplog.at_level("INFO", logger="kilora.import"):
        actual = import_loras(paths, model, device="cpu")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "mix-cache"))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    mix = import_loras(paths, model, device="cpu")
    assert actual.content_key_counts == (1, 1)
    assert "1 layers fell back to mix; 1 used single-expert keys" in actual.report()
    assert "1 layers fell back to mix; 1 used single-expert keys" in caplog.text
    assert torch.equal(
        actual.adapter_set.experts[target].keys_in, mix.adapter_set.experts[target].keys_in
    )
    other = "blocks.0.mlp.layer2"
    expert = actual.adapter_set.experts[other]
    parts = {i: (expert.down[i, 0], expert.up[i, 0].t()) for i in range(2)}
    expected = multi_import._per_expert_content_keys(
        parts,
        actual.sources,
        other,
        actual.adapter_set.routers[expert.up.shape[-1]],
        0,
        "cpu",
        cache=False,
        checkpoint=None,
    )
    assert torch.equal(expert.keys_in, expected)
    again = import_loras(paths, model, device="cpu")
    for name in mix.adapter_set.experts:
        assert torch.equal(
            mix.adapter_set.experts[name].keys_in, again.adapter_set.experts[name].keys_in
        )


def test_comfy_apply_reports_key_counts_and_separates_modes(denoiser, chain, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    patching._ROUTED_CACHE.clear()
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    base = FakePatcher(FakeBaseModel(denoiser))
    with torch.inference_mode():
        _, mix_report = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
        monkeypatch.delenv("KILORA_CONTENT_KEYS")
        _, node_mix = patching.apply(
            base, chain, -1, patching.LiveEnv(), joint=True, content_keys="mix"
        )
        monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
        _, report = patching.apply(
            base, chain, -1, patching.LiveEnv(), joint=True, content_keys="mix"
        )
    assert "Content keys:" not in mix_report
    assert "Content keys:" not in node_mix
    assert "layers fell back to mix" in report
    assert "used single-expert keys" in report


def test_content_key_default_and_overrides(monkeypatch, tmp_path):
    from kilora.comfyui.settings import KiLoRASettings, resolve_settings

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    paths = _files(tmp_path)
    model = Toy().eval()
    assert resolve_settings(None)["content_keys"] == "per_expert"
    assert KiLoRASettings().make("balanced", content_keys="mix")[0]["content_keys"] == "mix"
    assert import_loras(paths, model).content_key_counts is not None
    assert import_loras(paths, model, content_keys="mix").content_key_counts is None
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    assert import_loras(paths, model, content_keys="per_expert").content_key_counts is None
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    assert import_loras(paths, model, content_keys="mix").content_key_counts is not None


TOY_LAYERS = ["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"]


def _ranked_files(tmp_path, ranks):
    model = Toy().eval()
    paths = []
    for i, rank in enumerate(ranks):
        path = tmp_path / f"r{i}.safetensors"
        _lora(model, path, rank=rank, seed=40 + i, layers=TOY_LAYERS)
        paths.append(str(path))
    return paths


def _derived_env(monkeypatch, mode="per_expert"):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", mode)
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")


def _prepare_each(monkeypatch, tmp_path, paths):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "prepared"))
    for path in paths:
        prepare_mix([path], Toy().eval(), device="cpu")
    _clear()


def _toy_run(sources, x):
    from kilora.multi_import import LoRASource

    sources = [LoRASource(path=p, weight=w) for p, w in sources]
    model = Toy().eval()
    adapter_set = import_loras(sources, model, device="cpu").adapter_set
    adapter = compose(model, [adapter_set])
    with torch.no_grad():
        output = model(x)
    adapter.remove()
    return adapter_set, output


def _layer_bases(adapter_set):
    result = []
    plans = plan_targets([ScopedSet(adapter_set, import_score="derived")], 1)
    for target in sorted(plans):
        for layer in plans[target].to(torch.device("cpu"), torch.float32).sets:
            result.append(compute.derived_keys(layer)[:4])
    return result


def _counting(monkeypatch, name):
    calls = []
    original = getattr(torch.linalg, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, name, counted)
    return calls


@pytest.mark.parametrize("weights", [(0.7, 0.7, 0.7), (0.65, 0.8, 0.7)])
@pytest.mark.parametrize("mode", ["per_expert", "mix"])
def test_prepared_single_files_cover_weighted_mix(monkeypatch, tmp_path, weights, mode):
    _derived_env(monkeypatch, mode)
    paths = _ranked_files(tmp_path, (3, 3, 3))
    _prepare_each(monkeypatch, tmp_path, paths)
    x = torch.randn(1, 3, 12)
    sources = list(zip(paths, weights, strict=True))
    with monkeypatch.context() as patch:
        svd = _counting(patch, "svd")
        eigh = _counting(patch, "eigh")
        prepared, actual = _toy_run(sources, x)
    assert svd == []
    if mode == "per_expert":
        assert eigh == []
    _clear()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "fresh"))
    fresh, expected = _toy_run(sources, x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for target, expert in fresh.experts.items():
        torch.testing.assert_close(
            prepared.experts[target].keys_in, expert.keys_in, atol=1e-6, rtol=1e-6
        )


def test_prepared_bases_equal_fresh_bases_bitwise(monkeypatch, tmp_path):
    _derived_env(monkeypatch)
    paths = _ranked_files(tmp_path, (3, 3, 3))
    _prepare_each(monkeypatch, tmp_path, paths)
    x = torch.randn(1, 3, 12)
    sources = [(p, 1.0) for p in paths]
    with monkeypatch.context() as patch:
        svd = _counting(patch, "svd")
        prepared, actual = _toy_run(sources, x)
        prepared_bases = _layer_bases(prepared)
    assert svd == []
    _clear()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "fresh"))
    fresh, expected = _toy_run(sources, x)
    _clear()
    fresh_bases = _layer_bases(fresh)
    assert len(prepared_bases) == len(fresh_bases) > 0
    for got, want in zip(prepared_bases, fresh_bases, strict=True):
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)
    assert torch.equal(actual, expected)


def test_mixed_rank_mix_recomputes_only_other_padding(monkeypatch, tmp_path):
    _derived_env(monkeypatch)
    paths = _ranked_files(tmp_path, (3, 3, 2))
    _prepare_each(monkeypatch, tmp_path, paths)
    stored = []
    original_store = derived_cache.store

    def count_store(identity, kind, settings, names, values):
        if kind == "expert-keys":
            stored.append((identity, settings["per_expert"]))
        return original_store(identity, kind, settings, names, values)

    monkeypatch.setattr(derived_cache, "store", count_store)
    x = torch.randn(1, 3, 12)
    sources = list(zip(paths, (0.65, 0.8, 0.7), strict=True))
    with monkeypatch.context() as patch:
        svd = _counting(patch, "svd")
        eigh = _counting(patch, "eigh")
        prepared, actual = _toy_run(sources, x)
    assert eigh == []
    assert len(stored) == len(TOY_LAYERS)
    assert {padding for _, padding in stored} == {3}
    assert len(svd) == 2 * len(TOY_LAYERS)
    _clear()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "fresh"))
    fresh, expected = _toy_run(sources, x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _anima_files(tmp_path):
    from test_multi_import import ToyAnima, _lokr, lora_weights, to_keys, write

    model = ToyAnima().eval()
    paths = [
        write(tmp_path, f"plain{i}", to_keys(lora_weights(model, rank=rank, seed=i), "kohya"))
        for i, rank in enumerate((2, 4))
    ]
    paths.append(write(tmp_path, "lokr", _lokr()[0]))
    return model, paths


def _per_expert_env(monkeypatch):
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")


def _counting_store(monkeypatch):
    stored = []
    original_store = derived_cache.store

    def count(identity, kind, settings, names, values):
        if kind in ("expert-content", "expert-keys"):
            stored.append((kind, identity))
        return original_store(identity, kind, settings, names, values)

    monkeypatch.setattr(derived_cache, "store", count)
    return stored


def _run_mix(model, sources, x, ctx):
    adapter_set = import_loras(sources, model, rank=4, device="cpu").adapter_set
    adapter = compose(model, [adapter_set])
    with torch.no_grad():
        output = model(x, ctx)
    adapter.remove()
    return adapter_set, output


def test_mix_strengths_share_per_expert_entries(monkeypatch, tmp_path):
    from kilora.multi_import import LoRASource

    _per_expert_env(monkeypatch)
    model, paths = _anima_files(tmp_path)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    stored = _counting_store(monkeypatch)
    x, ctx = torch.randn(1, 3, 16), torch.randn(1, 3, 8)
    _clear()
    _run_mix(model, [LoRASource(path=p, weight=1.0) for p in paths], x, ctx)
    count = len(stored)
    assert {kind for kind, _ in stored} == {"expert-content", "expert-keys"}
    _clear()
    sources = [LoRASource(path=p, weight=0.7) for p in paths]
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
        patch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
        actual_set, actual = _run_mix(model, sources, x, ctx)
    assert len(stored) == count
    _clear()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "fresh"))
    fresh_set, expected = _run_mix(model, sources, x, ctx)
    assert torch.equal(actual, expected)
    for target, expert in fresh_set.experts.items():
        assert torch.equal(actual_set.experts[target].keys_in, expert.keys_in)


def test_negative_strength_keys_equal_fresh(monkeypatch, tmp_path):
    from kilora.multi_import import LoRASource
    from kilora.prepare import prepare_mix

    _per_expert_env(monkeypatch)
    model, paths = _anima_files(tmp_path)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "prepared"))
    for path in paths:
        prepare_mix([path], model, rank=4, device="cpu")
    _clear()
    x, ctx = torch.randn(1, 3, 16), torch.randn(1, 3, 8)
    sources = [LoRASource(path=p, weight=-0.7) for p in paths]
    actual_set, actual = _run_mix(model, sources, x, ctx)
    _clear()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "fresh"))
    fresh_set, expected = _run_mix(model, sources, x, ctx)
    assert torch.equal(actual, expected)
    for target, expert in fresh_set.experts.items():
        assert torch.equal(actual_set.experts[target].keys_in, expert.keys_in)


def test_strength_free_identity_keeps_sign_and_zero_apart():
    down, up = torch.randn(2, 8), torch.randn(4, 2)
    ids = {}
    for weight in (1.0, 0.7, 0.0, -0.7, -1.0):
        record = {"sha256": "ab" * 32, "weight": weight}
        scaled = {0: (down, up * weight)}
        (ids[weight],) = derived_cache.strength_free_identities([record], "t", scaled)
    assert ids[1.0] == ids[0.7]
    assert ids[-1.0] == ids[-0.7]
    assert len({ids[1.0], ids[0.0], ids[-1.0]}) == 3
    other = {0: (down + 1, up)}
    record = {"sha256": "ab" * 32, "weight": 1.0}
    assert derived_cache.strength_free_identities([record], "t", other)[0] != ids[1.0]
    assert derived_cache.strength_free_identities([{"weight": 1.0}], "t", other) is None
