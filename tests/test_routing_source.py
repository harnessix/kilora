# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import collections
import dataclasses
import os

import pytest
import torch
from test_prepared_paddings import (
    ATTENTION,
    LAYERS,
    MLP,
    StandIn,
    _clear,
    _count,
    _files,
    _stores,
)

from kilora import compute, derived_cache, key_cache
from kilora.bake import ScopedSet, plan_targets
from kilora.multi_import import LoRASource, import_loras
from kilora.prepare import (
    PLAN_DTYPES,
    layer_ranks,
    library_paddings,
    padding_dtypes,
    plan_dtype,
    prepare_mix,
    prepare_paddings,
)


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_ROWS", raising=False)
    return _files(tmp_path, (8, 16, 32))


def _edited(adapter_set, scale=0.9):
    experts = {}
    for target, ex in adapter_set.experts.items():
        source = derived_cache.source(ex)
        kept = dataclasses.replace(
            ex, down=ex.down.detach().clone(), up=ex.up.detach().clone(), routing_source=source
        )
        derived_cache.stamp(kept, source)
        experts[target] = dataclasses.replace(
            kept,
            up=kept.up * scale,
            routing_down=kept.down,
            routing_up=kept.up,
        )
    return dataclasses.replace(adapter_set, experts=experts)


def _worker(paths, model, dtype=None):
    _clear()
    ranks = {path: layer_ranks(prepare_mix([path], model)) for path in paths}
    reports = {
        path: prepare_paddings(path, wanted, model=model, dtype=dtype)
        for path, wanted in library_paddings(ranks).items()
    }
    _clear()
    return reports


def _sampling(sources, model, dtypes=(torch.float32, torch.bfloat16), calls=3, seen=None):
    mixed = _edited(import_loras(sources, model, device="cpu").adapter_set)
    plans = plan_targets([ScopedSet(mixed, import_score="derived")], 1)
    out = []
    for _ in range(calls):
        for name in sorted(plans):
            plan = plans[name]
            for layer in plan.sets:
                out.append(compute.derived_keys(layer))
            for dtype in dtypes:
                for layer in plan.to(torch.device("cpu"), dtype).sets:
                    out.append(compute.derived_keys(layer))
    return out


def _fresh(monkeypatch, sources, model, **kwargs):
    _clear()
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *a, **k: None)
        patch.setattr(derived_cache, "store", lambda *a, **k: None)
        result = _sampling(sources, model, **kwargs)
    _clear()
    return result


def _assert_equal(expected, got):
    assert len(expected) == len(got) > 0
    for a, b in zip(expected, got, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _watch(monkeypatch):
    seen = []
    current = compute._derived_keys_current

    def watched(s, *args, **kwargs):
        weights, _, _ = compute.derived_weights(s)
        seen.append(
            (s.routing_down is not None, s.routing_source, derived_cache.layer_source(weights))
        )
        return current(s, *args, **kwargs)

    monkeypatch.setattr(compute, "_derived_keys_current", watched)
    return seen


def test_routed_set_reaches_derived_keys_with_its_source(monkeypatch, library):
    model = StandIn().eval()
    mixed = _edited(import_loras([library[0], library[1]], model, device="cpu").adapter_set)
    for ex in mixed.experts.values():
        assert ex.routing_down is not None
        assert derived_cache.routing_source(ex)
        assert derived_cache.source(ex) is None
    plans = plan_targets([ScopedSet(mixed, import_score="derived")], 1)
    seen = _watch(monkeypatch)
    for name, plan in plans.items():
        for layer in plan.sets:
            assert layer.routing_down is not None
            assert layer.routing_source == derived_cache.routing_source(mixed.experts[name])
            compute.derived_keys(layer)
        for dtype in PLAN_DTYPES:
            for layer in plan.to(torch.device("cpu"), dtype).sets:
                assert layer.routing_source == derived_cache.routing_source(mixed.experts[name])
                compute.derived_keys(layer)
    assert seen
    assert all(routed and routing and source for routed, routing, source in seen)


def test_routing_source_needs_routing_weights(library):
    model = StandIn().eval()
    plain = import_loras([library[0]], model, device="cpu").adapter_set
    for ex in plain.experts.values():
        assert derived_cache.routing_source(ex) is None
        tagged = dataclasses.replace(ex, routing_source=("x", ("y",)))
        assert derived_cache.routing_source(tagged) is None


@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (0.65, 0.8, 0.7)])
@pytest.mark.parametrize("mix", [(0, 1), (0, 2), (0, 1, 2)])
def test_edited_never_seen_mix_needs_no_decomposition(monkeypatch, library, mix, weights):
    model = StandIn().eval()
    sources = [LoRASource(library[i], w) for i, w in zip(mix, weights, strict=False)]
    expected = _fresh(monkeypatch, sources, model)
    _worker(library, model)
    calls = _count(monkeypatch)
    seen = _watch(monkeypatch)
    got = _sampling(sources, model)
    assert calls == {"svd": 0, "eigh": 0}
    assert seen and all(source for _, _, source in seen)
    _assert_equal(expected, got)


def test_plan_dtype_of_a_model():
    class Cast(torch.nn.Module):
        manual_cast_dtype = torch.float16

    class Loaded(torch.nn.Module):
        def get_dtype(self):
            return torch.bfloat16

    assert plan_dtype(None) is None
    assert plan_dtype(Cast()) == torch.float16
    assert plan_dtype(Loaded()) == torch.bfloat16
    assert plan_dtype(torch.nn.Linear(2, 2)) == torch.float32
    assert plan_dtype(torch.nn.Linear(2, 2).to(torch.float64)) is None
    assert padding_dtypes(None) == PLAN_DTYPES
    assert padding_dtypes(torch.bfloat16) == (torch.bfloat16,)


def test_extra_paddings_in_the_plan_dtype_only(monkeypatch, library):
    model = StandIn().eval()
    stores = _stores(monkeypatch)
    _clear()
    ranks = {}
    own = {}
    for path in library:
        before = stores["expert-keys"]
        ranks[path] = layer_ranks(prepare_mix([path], model))
        own[path] = stores["expert-keys"] - before
    for path in library:
        assert own[path] == (len(MLP) + len(ATTENTION)) * len(PLAN_DTYPES)
    stores.clear()
    reports = {
        path: prepare_paddings(path, wanted, model=model, dtype=torch.bfloat16)
        for path, wanted in library_paddings(ranks).items()
    }
    assert stores["expert-keys"] == len(LAYERS) * (2 + 1 + 0)
    for path, extra in zip(library, (2, 1, 0), strict=True):
        assert reports[path]["entries"] == len(LAYERS) * extra
    _clear()


@pytest.mark.parametrize("edited", [False, True])
def test_mix_in_the_plan_dtype_needs_no_decomposition(monkeypatch, library, edited):
    model = StandIn().eval()
    sources = [LoRASource(library[0], 0.65), LoRASource(library[2], 0.8)]
    _worker(library, model, dtype=torch.bfloat16)
    calls = _count(monkeypatch)
    mixed = import_loras(sources, model, device="cpu").adapter_set
    if edited:
        mixed = _edited(mixed)
    plans = plan_targets([ScopedSet(mixed, import_score="derived")], 1)
    for plan in plans.values():
        for layer in plan.to(torch.device("cpu"), torch.bfloat16).sets:
            compute.derived_keys(layer)
    assert calls == {"svd": 0, "eigh": 0}


def test_mix_in_another_dtype_computes_the_extra_entries(monkeypatch, library):
    model = StandIn().eval()
    sources = [LoRASource(library[0], 0.65), LoRASource(library[2], 0.8)]
    expected = _fresh(monkeypatch, sources, model, dtypes=(torch.float16,))
    _worker(library, model, dtype=torch.bfloat16)
    calls = _count(monkeypatch)
    stores = _stores(monkeypatch)
    got = _sampling(sources, model, dtypes=(torch.float16,))
    assert calls["svd"] > 0
    assert stores["expert-keys"] > 0
    _assert_equal(expected, got)
    _clear()
    calls.update({"svd": 0, "eigh": 0})
    again = _sampling(sources, model, dtypes=(torch.float16,))
    assert calls == {"svd": 0, "eigh": 0}
    _assert_equal(expected, again)


def test_startup_plan_dtype_from_the_environment(monkeypatch):
    from kilora.comfyui import startup_prepare

    monkeypatch.setenv("KILORA_PLAN_DTYPE", "bfloat16")
    assert startup_prepare._plan_dtype("cpu") == torch.bfloat16
    monkeypatch.setenv("KILORA_PLAN_DTYPE", "float16")
    assert startup_prepare._plan_dtype("cpu") == torch.float16


def test_startup_worker_passes_the_plan_dtype(monkeypatch, tmp_path):
    from kilora import prepare
    from kilora.comfyui import prepare_node, startup_prepare

    paths = [str(tmp_path / "a"), str(tmp_path / "b")]
    monkeypatch.setattr(prepare_node, "apply_device", lambda model=None: "cpu")
    monkeypatch.setattr(prepare_node, "selected_files", lambda roots, s: (paths, None))
    monkeypatch.setattr(prepare_node, "display_path", lambda p, roots: p)
    monkeypatch.setattr(startup_prepare, "_checkpoint", lambda device: None)
    monkeypatch.setattr(prepare, "prepare_lokr", lambda *a, **k: (0, 0, False))
    ranks = iter([{("blocks.0.mlp.layer1", 4, 6): 2}, {("blocks.0.mlp.layer1", 4, 6): 4}])
    monkeypatch.setattr(prepare, "prepare_file", lambda *a, **k: object())
    monkeypatch.setattr(prepare, "layer_ranks", lambda result: next(ranks))
    got = collections.defaultdict(list)

    def fake(path, wanted, **kwargs):
        got[path].append(kwargs.get("dtype"))
        return {"paddings": [4], "entries": 1, "bytes": 1, "current": 0}

    monkeypatch.setattr(prepare, "prepare_paddings", fake)
    monkeypatch.setattr(startup_prepare, "_plan_dtype", lambda device, library=(): torch.float16)
    startup_prepare._stop.clear()
    startup_prepare._run([str(tmp_path)])
    assert got == {os.path.realpath(paths[0]): [torch.float16]}
