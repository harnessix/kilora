# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import collections

import pytest
import torch
from test_import_modes import _lora
from torch import nn

from kilora import compute, derived_cache, key_cache
from kilora.bake import ScopedSet, plan_targets
from kilora.multi_import import LoRASource, import_loras
from kilora.prepare import (
    PLAN_DTYPES,
    layer_ranks,
    library_paddings,
    prepare_mix,
    prepare_paddings,
)

DIM, HIDDEN, BLOCKS = 48, 64, 2
MLP = [f"blocks.{b}.mlp.{name}" for b in range(BLOCKS) for name in ("layer1", "layer2")]
ATTENTION = ["blocks.0.self_attn.q_proj"]
LAYERS = ATTENTION + MLP


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIM, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, DIM)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()


class StandIn(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([_Block() for _ in range(BLOCKS)])


def _clear():
    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()
    derived_cache.clear_memory()


def _count(monkeypatch):
    calls = {"svd": 0, "eigh": 0}
    svd, eigh = torch.linalg.svd, torch.linalg.eigh

    def counted_svd(*args, **kwargs):
        calls["svd"] += 1
        return svd(*args, **kwargs)

    def counted_eigh(*args, **kwargs):
        calls["eigh"] += 1
        return eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    monkeypatch.setattr(torch.linalg, "eigh", counted_eigh)
    return calls


def _stores(monkeypatch):
    counts = collections.Counter()
    store = derived_cache.store

    def counted(identity, kind, *args, **kwargs):
        counts[kind] += 1
        return store(identity, kind, *args, **kwargs)

    monkeypatch.setattr(derived_cache, "store", counted)
    return counts


def _files(tmp_path, ranks, start=0):
    model = StandIn().eval()
    paths = []
    for i, rank in enumerate(ranks, start):
        path = tmp_path / f"{i}-r{rank}.safetensors"
        _lora(model, path, rank=rank, seed=80 + i, layers=LAYERS)
        paths.append(str(path))
    return paths


def _worker(paths, model, stores):
    _clear()
    written = {}
    ranks = {}
    for path in paths:
        before = stores["expert-keys"]
        ranks[path] = layer_ranks(prepare_mix([path], model))
        written[path] = stores["expert-keys"] - before
    reports = {}
    for path, wanted in library_paddings(ranks).items():
        before = stores["expert-keys"]
        reports[path] = prepare_paddings(path, wanted, model=model)
        written[path] += stores["expert-keys"] - before
    _clear()
    return written, reports


def _sampling(sources, model, calls=3):
    mixed = import_loras(sources, model, device="cpu").adapter_set
    content = [mixed.experts[t].keys_in for t in sorted(mixed.experts)]
    plans = plan_targets([ScopedSet(mixed, import_score="derived")], 1)
    out = []
    for _ in range(calls):
        for name in sorted(plans):
            plan = plans[name]
            for layer in plan.sets:
                out.append(compute.derived_keys(layer))
            for dtype in (torch.float32, torch.bfloat16):
                for layer in plan.to(torch.device("cpu"), dtype).sets:
                    out.append(compute.derived_keys(layer))
    return content, out


def _fresh(monkeypatch, sources, model):
    _clear()
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *a, **k: None)
        patch.setattr(derived_cache, "store", lambda *a, **k: None)
        result = _sampling(sources, model)
    _clear()
    return result


def _assert_equal(expected, got):
    (content_a, keys_a), (content_b, keys_b) = expected, got
    assert len(content_a) == len(content_b) > 0
    for a, b in zip(content_a, content_b, strict=True):
        assert torch.equal(a, b)
    assert len(keys_a) == len(keys_b) > 0
    for a, b in zip(keys_a, keys_b, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_ROWS", raising=False)
    return _files(tmp_path, (8, 16, 32))


def test_store_count_per_file(monkeypatch, library):
    model = StandIn().eval()
    stores = _stores(monkeypatch)
    written, reports = _worker(library, model, stores)
    dtypes = len(PLAN_DTYPES)
    for path, extra in zip(library, (2, 1, 0), strict=True):
        assert written[path] == (len(LAYERS) * (1 + extra)) * dtypes
        assert reports[path]["entries"] == len(LAYERS) * extra * dtypes
    assert reports[library[0]]["paddings"] == [16, 32]
    assert reports[library[1]]["paddings"] == [32]
    assert reports[library[2]] == {"paddings": [], "entries": 0, "bytes": 0, "current": 0}
    rows = compute.derived_rows()
    mlp_entry = 4 * rows * (DIM + HIDDEN) + 2 * 8
    attention_entry = 4 * rows * (DIM + DIM) + 2 * 8
    for path, extra in zip(library[:2], (2, 1), strict=True):
        expected_bytes = (len(MLP) * mlp_entry + len(ATTENTION) * attention_entry) * extra * dtypes
        assert reports[path]["bytes"] == expected_bytes


@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (0.65, 0.8, 0.7)])
@pytest.mark.parametrize("mix", [(0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_never_seen_mix_needs_no_decomposition(monkeypatch, library, mix, weights):
    model = StandIn().eval()
    sources = [LoRASource(library[i], w) for i, w in zip(mix, weights, strict=False)]
    expected = _fresh(monkeypatch, sources, model)
    _worker(library, model, collections.Counter())
    calls = _count(monkeypatch)
    got = _sampling(sources, model)
    assert calls == {"svd": 0, "eigh": 0}
    _assert_equal(expected, got)


def test_padded_entry_equals_mix_entry(monkeypatch, library):
    model = StandIn().eval()
    seen = {}
    store = derived_cache.store

    def keep(identity, kind, settings, names, values):
        if kind == "expert-keys":
            key = (identity, settings["per_expert"], settings["down_dtype"])
            seen.setdefault(key, []).append(tuple(v.detach().clone() for v in values))
        return store(identity, kind, settings, names, values)

    monkeypatch.setattr(derived_cache, "store", keep)
    _worker(library, model, collections.Counter())
    prepared = dict(seen)
    seen.clear()
    monkeypatch.setattr(derived_cache, "load", lambda *a, **k: None)
    _clear()
    mixed = import_loras([library[0], library[2]], model, device="cpu").adapter_set
    plans = plan_targets([ScopedSet(mixed, import_score="derived")], 1)
    for plan in plans.values():
        for dtype in PLAN_DTYPES:
            for layer in plan.to(torch.device("cpu"), dtype).sets:
                compute.derived_keys(layer)
    padded = [key for key in seen if key[1] == 32 and key in prepared]
    assert len(padded) == 2 * len(LAYERS) * len(PLAN_DTYPES)
    for key in padded:
        for a, b in zip(prepared[key][-1], seen[key][-1], strict=True):
            assert torch.equal(a, b)


def test_new_file_pads_existing_files_and_rerun_stores_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    model = StandIn().eval()
    first = _files(tmp_path, (8, 16))
    stores = _stores(monkeypatch)
    _worker(first, model, stores)
    dtypes = len(PLAN_DTYPES)
    added = _files(tmp_path, (32,), start=2)
    stores.clear()
    written, reports = _worker(first + added, model, stores)
    assert written[first[0]] == len(LAYERS) * dtypes
    assert written[first[1]] == len(LAYERS) * dtypes
    assert reports[first[0]]["paddings"] == [32]
    assert reports[first[0]]["current"] == len(LAYERS) * dtypes
    assert written[added[0]] == (len(MLP) + len(ATTENTION)) * dtypes
    stores.clear()
    written, reports = _worker(first + added, model, stores)
    assert sum(stores.values()) == 0
    assert all(report["entries"] == 0 for report in reports.values())


def test_attention_gets_paddings_and_single_rank_layers_do_not():
    ranks = {
        "a": {("blocks.0.self_attn.q_proj", 4, 4): 2, ("blocks.0.mlp.layer1", 4, 6): 2},
        "b": {("blocks.0.self_attn.q_proj", 4, 4): 4, ("blocks.0.mlp.layer1", 4, 6): 4},
        "c": {("blocks.0.mlp.layer1", 4, 6): 4, ("blocks.0.mlp.layer2", 6, 4): 3},
    }
    assert library_paddings(ranks) == {
        "a": {("blocks.0.self_attn.q_proj", 4, 4): [4], ("blocks.0.mlp.layer1", 4, 6): [4]},
        "b": {},
        "c": {},
    }
