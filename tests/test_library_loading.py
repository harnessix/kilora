# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import builtins
import hashlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kilora.library import ByteLRU, LibraryLoader, SelectedFile


def _file(path, value):
    save_file(
        {
            "proj.lora_down.weight": torch.full((2, 3), value),
            "proj.lora_up.weight": torch.ones(4, 2),
        },
        str(path),
    )
    return SelectedFile(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _model():
    model = nn.Module()
    model.proj = nn.Linear(3, 4)
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    return patcher, SimpleNamespace(family="unknown"), model


def test_byte_lru_evicts_least_recent_and_bounds_oversized():
    cache = ByteLRU(8)
    cache.put("a", torch.ones(1))
    cache.put("b", torch.ones(1))
    assert cache.get("a") is not None
    cache.put("c", torch.ones(1))
    assert cache.get("b") is None
    assert cache.get("a") is not None and cache.get("c") is not None
    cache.put("a", torch.ones(3))
    assert cache.get("a") is None and cache.bytes == 4
    assert ByteLRU(0).get("missing") is None


def test_selected_files_only_and_reuse(tmp_path, monkeypatch):
    first = _file(tmp_path / "first.safetensors", 1.0)
    second = _file(tmp_path / "second.safetensors", 2.0)
    unused = _file(tmp_path / "unused.safetensors", 3.0)
    from kilora.library import loading

    calls = []
    opens = []
    imports = []
    original = loading._read_file
    original_open = builtins.open
    original_joint = loading.joint_choice

    def tracked_open(path, *args, **kwargs):
        if str(path).endswith(".safetensors"):
            opens.append(str(path))
        return original_open(path, *args, **kwargs)

    def tracked_joint(*args, **kwargs):
        imports.append(1)
        return original_joint(*args, **kwargs)

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(loading, "_read_file", read)
    monkeypatch.setattr(builtins, "open", tracked_open)
    monkeypatch.setattr(loading, "joint_choice", tracked_joint)
    loader = LibraryLoader([tmp_path])
    patcher, binding, model = _model()
    choice, report = loader.load([second, first], patcher, binding, model)
    assert sorted(calls) == sorted([str(first.path), str(second.path)])
    assert str(unused.path) not in calls
    assert set(opens) == {str(first.path), str(second.path)}
    assert not report.routed_hit
    assert report.read_seconds >= 0 and report.import_seconds >= 0
    assert report.key_setup_seconds >= 0
    again, report = loader.load([first, second], patcher, binding, model)
    assert again is choice and report.routed_hit and len(calls) == 2 and len(imports) == 1
    _, report = loader.load([first], patcher, binding, model)
    assert report.ram_hits == 1 and len(calls) == 2
    assert "key setup" in str(report)
    uncached = LibraryLoader([tmp_path], library_vram_gib=0)
    rebuilt, _ = uncached.load([first, second], patcher, binding, model)
    rebuilt_again, report = uncached.load([first, second], patcher, binding, model)
    assert not report.routed_hit and report.ram_hits == 2
    for name, expert in choice.adapter_sets[0].experts.items():
        for part in ("down", "up", "keys_in"):
            assert torch.equal(
                getattr(expert, part), getattr(rebuilt.adapter_sets[0].experts[name], part)
            )
            assert torch.equal(
                getattr(expert, part), getattr(rebuilt_again.adapter_sets[0].experts[name], part)
            )


def test_changed_and_outside_files_are_rejected(tmp_path):
    folder = tmp_path / "library"
    folder.mkdir()
    selected = _file(folder / "file.safetensors", 1.0)
    outside = _file(tmp_path / "outside.safetensors", 1.0)
    patcher, binding, model = _model()
    loader = LibraryLoader([folder])
    loader.load([selected], patcher, binding, model)
    _file(selected.path, 2.0)
    with pytest.raises(ValueError, match="changed since sync"):
        loader.load([selected], patcher, binding, model)
    with pytest.raises(ValueError, match="outside"):
        loader.load([outside], patcher, binding, model)


def test_routed_cache_does_not_keep_the_model_alive(tmp_path):
    import gc
    import weakref

    first = _file(tmp_path / "first.safetensors", 1.0)
    loader = LibraryLoader([tmp_path])
    patcher, binding, model = _model()
    loader.load([first], patcher, binding, model)
    ref = weakref.ref(model)
    del patcher, model
    gc.collect()
    assert ref() is None
    patcher, binding, model = _model()
    _, report = loader.load([first], patcher, binding, model)
    assert not report.routed_hit
