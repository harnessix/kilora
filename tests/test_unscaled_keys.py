# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_modes import Toy, _lora

from kilora import compute, derived_cache, key_cache, multi_import
from kilora.bake import ScopedSet, plan_targets
from kilora.multi_import import LoRASource, import_loras

L1, L2, Q = "blocks.0.mlp.layer1", "blocks.0.mlp.layer2", "blocks.0.self_attn.q_proj"


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


def _files(tmp_path, specs):
    model = Toy().eval()
    paths = []
    for i, (rank, layers) in enumerate(specs):
        path = tmp_path / f"{i}.safetensors"
        _lora(model, path, rank=rank, seed=60 + i, layers=layers)
        paths.append(str(path))
    return paths


def _assert_equal(expected, got):
    (content_a, keys_a), (content_b, keys_b) = expected, got
    assert len(content_a) == len(content_b) > 0
    for a, b in zip(content_a, content_b, strict=True):
        assert torch.equal(a, b)
    assert len(keys_a) == len(keys_b) > 0
    for a, b in zip(keys_a, keys_b, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _prepare(paths, model):
    from kilora.prepare import prepare_mix

    _clear()
    for path in paths:
        prepare_mix([path], model)
    _clear()


@pytest.mark.parametrize(
    "weights", [(1.0, 1.0, 1.0), (0.65, 0.8, 0.7), (-0.7, 0.8, 1.0), (0.7, -1.0, 0.0)]
)
def test_prepared_mix_equals_fresh_bit_for_bit(monkeypatch, tmp_path, weights):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    paths = _files(tmp_path, [(3, [L1, L2, Q])] * 3)
    model = Toy().eval()
    sources = [LoRASource(p, w) for p, w in zip(paths, weights, strict=True)]
    expected = _fresh(monkeypatch, sources, model)
    _prepare(paths, model)
    calls = _count(monkeypatch)
    got = _sampling(sources, model)
    signed = sum(w <= 0 for w in weights)
    layers = 3
    assert calls["eigh"] == signed * layers
    assert calls["svd"] == (2 * 2 * layers if signed else 0)
    _assert_equal(expected, got)


def test_padding_misses_are_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    paths = _files(tmp_path, [(3, [L1, L2]), (5, [L1, L2])])
    model = Toy().eval()
    sources = [LoRASource(paths[0], 0.65), LoRASource(paths[1], 0.8)]
    expected = _fresh(monkeypatch, sources, model)
    _prepare(paths, model)
    calls = _count(monkeypatch)
    got = _sampling(sources, model)
    assert calls["svd"] == 8
    assert calls["eigh"] == 0
    _assert_equal(expected, got)


def test_one_file_at_two_strengths_uses_the_same_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    (path,) = _files(tmp_path, [(3, [L1, L2, Q])])
    model = Toy().eval()
    _prepare([path], model)
    stored = []
    original = derived_cache.store

    def count(identity, kind, settings, names, values):
        if kind in ("expert-content", "expert-keys", "conversion"):
            stored.append((kind, identity))
        return original(identity, kind, settings, names, values)

    monkeypatch.setattr(derived_cache, "store", count)
    calls = _count(monkeypatch)
    at_one = _sampling([LoRASource(path, 1.0)], model)
    _clear()
    at_seven = _sampling([LoRASource(path, 0.7)], model)
    assert stored == []
    assert calls["svd"] == 0 and calls["eigh"] == 0
    _assert_equal(at_one, at_seven)


def test_negative_strength_keys_equal_their_unit_strength(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    (path,) = _files(tmp_path, [(3, [L1, L2, Q])])
    model = Toy().eval()
    expected = _fresh(monkeypatch, [LoRASource(path, -0.7)], model)
    _prepare([path], model)
    _sampling([LoRASource(path, -1.0)], model)
    _clear()
    calls = _count(monkeypatch)
    got = _sampling([LoRASource(path, -0.7)], model)
    assert calls["svd"] == 0 and calls["eigh"] == 0
    _assert_equal(expected, got)


def test_zero_strength_keeps_the_seeded_key_and_zero_bases(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    paths = _files(tmp_path, [(3, [L1, L2, Q])] * 2)
    model = Toy().eval()
    sources = [LoRASource(paths[0], 0.7), LoRASource(paths[1], 0.0)]
    expected = _fresh(monkeypatch, sources, model)
    _prepare(paths, model)
    got = _sampling(sources, model)
    _assert_equal(expected, got)
    imported = import_loras(sources, model, device="cpu")
    mixed = imported.adapter_set
    for target, ex in mixed.experts.items():
        assert not bool(ex.up[1].any())
        router = mixed.routers[ex.up.shape[-1]]
        seeded = multi_import._content_keys(
            torch.zeros_like(ex.up[1:]), router, imported.sources[1:], target, device="cpu"
        )
        assert torch.equal(ex.keys_in[1], seeded[0])
    for _, rin, _, rout, _ in got[1]:
        assert int(rout[1]) == 0
        assert int(rin[1]) > 0


def test_entries_of_the_previous_tags_are_not_read(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    (path,) = _files(tmp_path, [(3, [L1, L2, Q])])
    model = Toy().eval()
    loads = []
    original = derived_cache.load

    def record(identity, kind, settings, names, device="cpu"):
        loads.append((kind, dict(settings)))
        return original(identity, kind, settings, names, device)

    monkeypatch.setattr(derived_cache, "load", record)
    _prepare([path], model)
    _sampling([LoRASource(path, 0.7)], model)
    kinds = {kind for kind, _ in loads}
    assert {"expert-content", "expert-keys", "keys", "content"} & kinds
    for kind, settings in loads:
        if kind == "expert-content":
            assert settings.get("keys_from") == "unscaled-1"
            assert settings.get("algorithm", "per-expert-v3") != "per-expert-v2"
        elif kind == "expert-keys":
            assert settings["identity"] == "strength-free-2"
        elif kind in ("keys", "content"):
            assert settings.get("keys_from") == "unscaled-1"
    assert derived_cache._EXPERT_FORMAT == "strength-free-2"


def test_expert_ids_change_with_the_format(monkeypatch):
    down, up = torch.randn(2, 8), torch.randn(4, 2)
    record = {"sha256": "cd" * 32, "weight": 0.7}
    (new,) = derived_cache.strength_free_identities([record], "t", {0: (down, up)})
    monkeypatch.setattr(derived_cache, "_EXPERT_FORMAT", "strength-free-1")
    (old,) = derived_cache.strength_free_identities([record], "t", {0: (down, up)})
    assert new != old


def test_source_compares_and_serializes_as_its_pair():
    import json
    import pickle

    unit = torch.randn(2, 1, 3, 4)
    value = derived_cache.Source("layer", ("a", "b"), unit)
    assert value == ("layer", ("a", "b")) and hash(value) == hash(("layer", ("a", "b")))
    assert json.dumps(value) == json.dumps(["layer", ["a", "b"]])
    assert derived_cache.unit_up(value) is unit
    copy = pickle.loads(pickle.dumps(value))
    assert torch.equal(derived_cache.unit_up(copy), unit)
    assert derived_cache.unit_up(("layer", ("a", "b"))) is None
