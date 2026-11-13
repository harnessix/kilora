# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from kilora import modlist
from kilora.adapter import Settings
from kilora.compute import orthonormal_keys
from kilora.modfile import load_mod, save_mod
from kilora.sets import AdapterSet, KeyedExperts, make_router

WIDTH = 8


def _save(path, seed):
    gen = torch.Generator().manual_seed(seed)
    parts = {
        "layers.0.attention.to_q": KeyedExperts(
            keys_in=orthonormal_keys(torch.randn(2, 2, 8, generator=gen)),
            keys_out=orthonormal_keys(torch.randn(2, 2, 8, generator=gen)),
            down=torch.randn(2, 1, 2, WIDTH, generator=gen),
            up=torch.randn(2, 1, 2, WIDTH, generator=gen),
        )
    }
    sets = [
        AdapterSet(
            name="s",
            part_type=("lora",),
            experts=parts,
            routers={WIDTH: make_router(8, WIDTH, seed=11)},
            form="input",
            imported_from={"file": "by-hand"},
        )
    ]
    save_mod(path, sets, Settings(), id="m")


def _down(mod):
    return mod.sets[0].experts["layers.0.attention.to_q"].down


@pytest.fixture(autouse=True)
def _empty_cache(monkeypatch):
    monkeypatch.setattr(modlist, "_CACHE", {})


def test_rewrite_with_same_size_and_mtime_returns_new_mod(tmp_path):
    path = tmp_path / "m.kmod"
    _save(path, 1)
    stat = os.stat(path)
    first = modlist._cached_mod(path)
    assert torch.equal(_down(first), _down(load_mod(path)))
    _save(path, 2)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert os.stat(path).st_size == stat.st_size
    assert os.stat(path).st_mtime_ns == stat.st_mtime_ns
    second = modlist._cached_mod(path)
    assert second is not first
    assert torch.equal(_down(second), _down(load_mod(path)))
    assert not torch.equal(_down(second), _down(first))


def test_same_file_is_a_cache_hit(tmp_path, monkeypatch):
    path = tmp_path / "m.kmod"
    _save(path, 1)
    calls = []

    def counted(p):
        calls.append(p)
        return load_mod(p)

    monkeypatch.setattr(modlist, "load_mod", counted)
    first = modlist._cached_mod(path)
    assert modlist._cached_mod(str(path)) is first
    assert len(calls) == 1


@pytest.mark.parametrize("keep", [4, 20])
def test_truncated_file_raises_like_load_mod_and_is_not_cached(tmp_path, keep):
    path = tmp_path / "m.kmod"
    _save(path, 1)
    data = path.read_bytes()
    path.write_bytes(data[:keep])
    with pytest.raises(Exception) as expected:
        load_mod(path)
    with pytest.raises(type(expected.value)) as got:
        modlist._cached_mod(path)
    assert str(got.value) == str(expected.value)
    assert modlist._CACHE == {}


def test_cache_is_cleared_above_64_entries(tmp_path):
    path = tmp_path / "m.kmod"
    _save(path, 1)
    modlist._CACHE.update({(str(i), 0, 0, 0, ""): None for i in range(65)})
    mod = modlist._cached_mod(path)
    assert list(modlist._CACHE.values()) == [mod]
