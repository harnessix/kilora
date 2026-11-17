# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import route_freeze as R
from kilora.compute import dropout_key, run_layer, sampling_progress

TARGET = "layer"
FREEZE = 0.4


@pytest.fixture(autouse=True)
def _eager_chain(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_MASKS", "sample")


def _layer(n_sets, gate="adaptive", width=16, state=True):
    gen = torch.Generator().manual_seed(400 + n_sets)
    sets = _sets(n_sets, width, 4, 2, 8, seed=400 + n_sets)
    out = [dataclasses.replace(sets[0], route_freeze=FREEZE, gate_mode=gate)] if state else []
    for s in sets[1:] if state else sets:
        out.append(
            dataclasses.replace(
                s,
                form="input",
                down=torch.randn(s.down.shape[0], width, generator=gen) / width**0.5,
                route_freeze=FREEZE,
                gate_mode=gate,
            )
        )
    return out


def _inputs(seed, width=16, tokens=9):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, tokens, width, generator=gen), torch.randn(
        2, tokens, width, generator=gen
    )


def _call(sets, depth, progress, cache, x, y0, seed=3, target=TARGET):
    with (
        torch.no_grad(),
        dropout_key(seed, progress),
        sampling_progress(progress),
        R.freezing(cache),
    ):
        return run_layer(x.clone(), y0.clone(), sets, depth, target=target)


def _storages(tensors) -> set:
    return {(t.device, t.untyped_storage().data_ptr()) for t in tensors if t.numel()}


def _record_tensors(rec) -> dict:
    routes = [g for steps in rec["routes"].values() for step in steps for g in step]
    gates = [q for pair in rec["gates"].values() for q in pair]
    keeps = [
        m[0] if isinstance(m, tuple) else m
        for step in rec.get("input_keeps", {}).values()
        for m in step.values()
    ]
    return {"routes": routes, "keeps": keeps, "gates": gates}


def _entry_tensors(entry) -> list:
    found = []

    def collect(t):
        if isinstance(t, torch.Tensor):
            found.append(t)
        return t

    R._entry_tensors(entry, collect)
    return found


def _census(sets, depth, x, y0, monkeypatch):
    seen = []
    build = R.build

    def spy(rec, *args, **kwargs):
        seen.append({k: (len(v), len(_storages(v))) for k, v in _record_tensors(rec).items()})
        return build(rec, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(R, "build", spy)
        cache = {}
        _call(sets, depth, FREEZE, cache, x, y0)
    assert len(seen) == 1
    owned = _entry_tensors(cache[TARGET])
    return seen[0], (len(owned), len(_storages(owned))), cache


@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
def test_record_keeps_one_buffer_per_kind_and_the_entry_one(n_sets, depth, monkeypatch):
    x, y0 = _inputs(11)
    sets = _layer(n_sets, state=False)
    monkeypatch.setattr(R, "_PACK", False)
    before, entry_before, _ = _census(sets, depth, x, y0, monkeypatch)
    monkeypatch.setattr(R, "_PACK", True)
    after, entry_after, cache = _census(sets, depth, x, y0, monkeypatch)
    print(
        f"record census, {n_sets} sets, depth {depth}: before {before} entry {entry_before}, "
        f"after {after} entry {entry_after} (tensors, storages)"
    )
    assert cache[TARGET].get("chain_routes") is not None
    for kind, (tensors, storages) in after.items():
        assert storages <= 1, kind
        assert tensors == before[kind][0], kind
        assert before[kind][1] >= storages
    assert after["routes"][0] == n_sets * depth
    assert after["keeps"][0] > 0 and after["gates"][0] > 0
    assert entry_after[1] == 1
    assert entry_before[1] > entry_after[1]


def _equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
        assert torch.equal(a, b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for u, v in zip(a, b, strict=True):
            _equal(u, v)
    else:
        assert a == b


def _same_entries(a, b):
    assert dict.keys(a) == dict.keys(b)
    for key in ("lead", "forms", "chain_depth", "chain_target"):
        assert dict.get(a, key) == dict.get(b, key)
    for key in ("scale", "up_gates", "chain_routes", "chain_keeps"):
        _equal(dict.get(a, key), dict.get(b, key))
    for u, v in zip(dict.get(a, "exact") or [], dict.get(b, "exact") or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None:
            _equal(u[1:], v[1:])
    _equal(a["up"], b["up"])


def _renders(sets, depth, x, y0, packed, monkeypatch):
    monkeypatch.setattr(R, "_PACK", packed)
    cache = {}
    outs = [_call(sets, depth, FREEZE, cache, x, y0)]
    for progress, scale in ((0.6, 0.8), (0.9, 1.1)):
        outs.append(_call(sets, depth, progress, cache, x * scale, y0 * 0.9, seed=7))
    return outs, cache


@pytest.mark.parametrize("mode", ["keys", "full"])
@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
def test_packed_record_is_bit_identical_on_the_eager_loop(mode, n_sets, depth, gate, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    x, y0 = _inputs(21)
    sets = _layer(n_sets, gate)
    old, old_cache = _renders(sets, depth, x, y0, False, monkeypatch)
    new, new_cache = _renders(sets, depth, x, y0, True, monkeypatch)
    _same_entries(old_cache[TARGET], new_cache[TARGET])
    for u, v in zip(old, new, strict=True):
        assert torch.equal(u, v)


@pytest.mark.parametrize("masks", ["sample", "expect"])
def test_packed_record_is_bit_identical_with_expected_masks(masks, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", masks)
    x, y0 = _inputs(31)
    sets = _layer(3)
    old, old_cache = _renders(sets, 3.5, x, y0, False, monkeypatch)
    new, new_cache = _renders(sets, 3.5, x, y0, True, monkeypatch)
    _same_entries(old_cache[TARGET], new_cache[TARGET])
    for u, v in zip(old, new, strict=True):
        assert torch.equal(u, v)


def test_pack_copies_values_and_reuses_a_slice():
    pack = R._Pack(hint=64)
    a = torch.arange(6, dtype=torch.float32).view(2, 3)
    b = torch.tensor([True, False, True])
    c = torch.arange(5, dtype=torch.float64)
    pa, pb, pc = pack.put(a), pack.put(b), pack.put(c)
    for got, want in ((pa, a), (pb, b), (pc, c)):
        assert torch.equal(got, want) and got.dtype == want.dtype and got.shape == want.shape
    assert pa.untyped_storage().data_ptr() == pb.untyped_storage().data_ptr()
    assert len(pack.buffers) == 2
    a.add_(1)
    assert torch.equal(pa, a - 1)
    again = pack.put(a, into=pa)
    assert again is pa and torch.equal(pa, a)
    assert len(pack.buffers) == 2


def test_record_buffers_end_with_the_layer_call(monkeypatch):
    x, y0 = _inputs(41)
    sets = _layer(2, state=False)
    held = []
    build = R.build

    def spy(rec, *args, **kwargs):
        held.append(rec)
        return build(rec, *args, **kwargs)

    monkeypatch.setattr(R, "build", spy)
    cache = {}
    _call(sets, 3, FREEZE, cache, x, y0)
    assert held and not held[0]
    assert len(_storages(_entry_tensors(cache[TARGET]))) == 1


@pytest.fixture
def fused_helpers(monkeypatch):
    pytest.importorskip("triton")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs a GPU or the Triton interpreter")
    import test_freeze_record_memory as M

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    return M


@pytest.mark.parametrize("mode", ["keys", "full"])
@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
def test_packed_fused_record_is_bit_identical(mode, n_sets, depth, fused_helpers, monkeypatch):
    M = fused_helpers
    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    x, y0 = M._inputs(1401)
    sets = M._layer(n_sets, "adaptive")
    results = {}
    for packed in (False, True):
        monkeypatch.setattr(R, "_PACK", packed)
        cache = {}
        outs = [M._call(sets, depth, M.FREEZE, cache, x, y0)]
        for progress in (0.6, 0.9):
            outs.append(M._call(sets, depth, progress, cache, x * 0.8, y0 * 0.9, seed=11))
        results[packed] = outs, cache
    (old, old_cache), (new, new_cache) = results[False], results[True]
    for u, v in zip(old, new, strict=True):
        assert torch.equal(u, v)
    assert old_cache.keys() == new_cache.keys()
    if M.TARGET not in new_cache:
        return
    _same_entries(old_cache[M.TARGET], new_cache[M.TARGET])
    owned = _storages(_entry_tensors(new_cache[M.TARGET]))
    assert len(owned) == 1


@pytest.mark.parametrize("n_sets", [2, 7])
def test_fused_record_keeps_one_buffer_per_kind(n_sets, fused_helpers, monkeypatch):
    M = fused_helpers
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    x, y0 = M._inputs(1411)
    sets = M._layer(n_sets, "adaptive")
    seen = []
    build = R.build

    def spy(rec, *args, **kwargs):
        seen.append({k: len(_storages(v)) for k, v in _record_tensors(rec).items()})
        return build(rec, *args, **kwargs)

    monkeypatch.setattr(R, "build", spy)
    M._call(sets, 7, M.FREEZE, {}, x, y0)
    assert len(seen) == 1
    assert all(n <= 1 for n in seen[0].values()), seen[0]
