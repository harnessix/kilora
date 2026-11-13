# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_modes import Toy, _lora

from kilora import compose, compute, derived_cache, key_cache
from kilora.multi_import import import_loras
from kilora.prepare import prepare_mix

LAYERS = ["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"]


def _clear():
    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()


def _files(tmp_path, n=3):
    model = Toy().eval()
    paths = []
    for i in range(n):
        path = tmp_path / f"{i}.safetensors"
        _lora(model, path, rank=3, seed=i, layers=LAYERS)
        paths.append(str(path))
    return paths


def _run(paths, x):
    model = Toy().eval()
    handle = compose(model, [import_loras(paths, model).adapter_set])
    with torch.no_grad():
        outputs = [model(x), model(x)]
    handle.remove()
    return outputs


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    derived_cache.clear_memory()
    _clear()
    yield
    derived_cache.clear_memory()
    _clear()


def test_prepared_import_reads_bounded_and_equals_empty_cache(setup, monkeypatch, tmp_path):
    paths = _files(tmp_path)
    x = torch.randn(1, 3, 12)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "empty"))
    expected = _run(paths, x)

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "prepared"))
    for path in paths:
        prepare_mix([path], Toy().eval())
    derived_cache.clear_memory()
    _clear()
    derived_cache.STATS.clear()
    calls = {"svd": 0, "eigh": 0}
    svd, eigh = torch.linalg.svd, torch.linalg.eigh

    def count(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch.linalg, "svd", count("svd", svd))
    monkeypatch.setattr(torch.linalg, "eigh", count("eigh", eigh))
    actual = _run(paths, x)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    assert calls == {"svd": 0, "eigh": 0}
    experts, layers = len(paths), len(LAYERS)
    assert derived_cache.STATS["reads"] <= 2 * experts * layers + 4


def test_second_apply_reads_no_cache_file(setup, monkeypatch, tmp_path):
    paths = _files(tmp_path)
    x = torch.randn(1, 3, 12)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    for path in paths:
        prepare_mix([path], Toy().eval())
    first = _run(paths, x)
    _clear()
    derived_cache.STATS.clear()
    second = _run(paths, x)
    assert derived_cache.STATS["reads"] == 0
    assert derived_cache.STATS["memory_hits"] > 0
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_memory_entries_follow_the_file(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    names = ("a",)
    derived_cache.store("id", "kind", {"s": 1}, names, (torch.ones(3),))
    derived_cache.STATS.clear()
    (value,) = derived_cache.load("id", "kind", {"s": 1}, names)
    assert torch.equal(value, torch.ones(3))
    assert derived_cache.STATS["reads"] == 0
    value.add_(1)
    assert torch.equal(derived_cache.load("id", "kind", {"s": 1}, names)[0], torch.ones(3))
    assert derived_cache.load("id", "kind", {"s": 2}, names) is None
    path = derived_cache._path("id", "kind", {"s": 1})
    from safetensors.torch import save_file

    save_file({"a": torch.full((4,), 2.0)}, path)
    (value,) = derived_cache.load("id", "kind", {"s": 1}, names)
    assert torch.equal(value, torch.full((4,), 2.0))
    assert derived_cache.STATS["reads"] == 1
    import os

    os.remove(path)
    assert derived_cache.load("id", "kind", {"s": 1}, names) is None


def test_old_format_entries_are_not_reused(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    derived_cache.store("id", "kind", {}, ("a",), (torch.ones(2),))
    monkeypatch.setattr(derived_cache, "_FORMAT", derived_cache._FORMAT + 1)
    assert derived_cache.load("id", "kind", {}, ("a",)) is None


def test_changed_lora_file_misses_after_memory_hit(setup, monkeypatch, tmp_path):
    paths = _files(tmp_path)
    x = torch.randn(1, 3, 12)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    _run(paths, x)
    _lora(Toy().eval(), tmp_path / "0.safetensors", rank=3, seed=99, layers=LAYERS)
    _clear()
    count = 0
    svd = torch.linalg.svd

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    _run(paths, x)
    assert count > 0
