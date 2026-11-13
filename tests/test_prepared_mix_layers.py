# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_modes import Toy, _lora

from kilora import compute, derived_cache, key_cache
from kilora.bake import ScopedSet, plan_targets
from kilora.multi_import import LoRASource, import_loras

L1, L2, Q = "blocks.0.mlp.layer1", "blocks.0.mlp.layer2", "blocks.0.self_attn.q_proj"


def _clear():
    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()
    derived_cache.clear_memory()


def _count(monkeypatch):
    calls = {"svd": 0, "eigh": 0, "batch": []}
    svd, eigh = torch.linalg.svd, torch.linalg.eigh

    def counted_svd(tensor, *args, **kwargs):
        calls["svd"] += 1
        if tensor.ndim == 3:
            calls["batch"].append(tensor.shape[0])
        return svd(tensor, *args, **kwargs)

    def counted_eigh(*args, **kwargs):
        calls["eigh"] += 1
        return eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    monkeypatch.setattr(torch.linalg, "eigh", counted_eigh)
    return calls


def _sampling_keys(sources, model, calls=3):
    mixed = import_loras(sources, model, device="cpu").adapter_set
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
    return out


def _files(tmp_path, specs):
    model = Toy().eval()
    paths = []
    for i, (rank, layers) in enumerate(specs):
        path = tmp_path / f"{i}.safetensors"
        _lora(model, path, rank=rank, seed=40 + i, layers=layers)
        paths.append(str(path))
    return paths


def _expected(monkeypatch, sources, model):
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *a, **k: None)
        patch.setattr(derived_cache, "store", lambda *a, **k: None)
        return _sampling_keys(sources, model)


@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (0.65, 0.8, 0.7)])
def test_layers_missing_from_a_file_need_no_svd(monkeypatch, tmp_path, weights):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path, [(3, [L1, L2, Q]), (3, [L1]), (3, [L2, Q])])
    model = Toy().eval()
    sources = [LoRASource(p, w) for p, w in zip(paths, weights, strict=True)]
    _clear()
    expected = _expected(monkeypatch, sources, model)
    _clear()
    for path in paths:
        prepare_mix([path], model)
    _clear()
    compute.DERIVED_STATS.clear()
    calls = _count(monkeypatch)
    got = _sampling_keys(sources, model)
    assert calls["svd"] == 0 and calls["eigh"] == 0
    assert compute.DERIVED_STATS["zero_experts"] == 6
    assert len(got) == len(expected)
    for a, b in zip(expected, got, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_padding_misses_are_counted_exactly(monkeypatch, tmp_path):
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    paths = _files(tmp_path, [(3, [L1, L2]), (5, [L1])])
    model = Toy().eval()
    sources = [LoRASource(paths[0], 0.65), LoRASource(paths[1], 0.8)]
    _clear()
    expected = _expected(monkeypatch, sources, model)
    _clear()
    for path in paths:
        prepare_mix([path], model)
    _clear()
    compute.DERIVED_STATS.clear()
    calls = _count(monkeypatch)
    got = _sampling_keys(sources, model)
    assert calls["batch"] == [1, 1, 1, 1]
    assert calls["svd"] == 4 and calls["eigh"] == 0
    assert compute.DERIVED_STATS["zero_experts"] == 2
    for a, b in zip(expected, got, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_zero_expert_bases_match_the_svd_path():
    gen = torch.Generator().manual_seed(3)
    down = torch.randn(3, 4, 16, generator=gen)
    up = torch.randn(3, 4, 8, generator=gen)
    down[1] = 0
    up[1] = 0
    layer = compute.LayerSet(
        router=torch.randn(8, 8, generator=gen),
        keys_in=torch.randn(3, 2, 8, generator=gen),
        per_expert=4,
        down=down.reshape(12, 16),
        up=up.reshape(12, 8),
    )
    _clear()
    kin, rin, kout, rout, _ = compute._derived_keys_current(layer, fresh=True)
    zero = compute._zero_expert_bases(layer, 8)
    for full, part in zip((kin, rin, kout, rout), zero, strict=True):
        assert full[1].dtype == part.dtype and full[1].shape == part.shape
        assert torch.equal(full[1], part)
