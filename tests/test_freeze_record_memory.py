# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

pytest.importorskip("triton")

from test_fused import _set
from test_fused_input import _imported, _inputs

from kilora import compute, fused
from kilora import route_freeze as R
from kilora.compute import dropout_key, run_layer, sampling_progress

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)
TARGET = "blk.0.attn1"
RTOL, ATOL = 3e-4, 3e-5
FREEZE = 0.4


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")


def _layer(n_sets, gate_mode="none"):
    sets = [_set(1201, experts=2, import_score="derived", gate_mode=gate_mode)]
    sets += [
        _imported(1202 + i, experts=2, import_score="derived", gate_mode=gate_mode)
        for i in range(n_sets - 1)
    ]
    return [dataclasses.replace(s, route_freeze=FREEZE) for s in sets]


def _joint_calls(monkeypatch):
    seen = []
    original = fused._run_derived_state

    def spy(*args, **kwargs):
        seen.append(bool(R.RECORDS))
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "_run_derived_state", spy)
    return seen


def _call(sets, depth, progress, cache, x, y0, seed=3):
    with (
        torch.no_grad(),
        dropout_key(seed, progress),
        sampling_progress(progress),
        R.freezing(cache),
    ):
        return run_layer(x.clone(), y0.clone(), sets, depth, target=TARGET)


def _record(sets, depth, x, y0, eager):
    cache = {}
    if eager:
        with compute.eager():
            y = _call(sets, depth, FREEZE, cache, x, y0)
    else:
        y = _call(sets, depth, FREEZE, cache, x, y0)
    return y, cache


def _keeps(entry):
    return [[R._unpack_keep(m) for m in step] for step in entry.get("chain_keeps", [])]


@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_fused_record_gives_the_eager_frozen_entry(n_sets, depth, gate_mode, monkeypatch):
    x, y0 = _inputs(1301)
    sets = _layer(n_sets, gate_mode)
    seen = _joint_calls(monkeypatch)
    y_eager, eager = _record(sets, depth, x, y0, eager=True)
    assert not seen
    y_fused, recorded = _record(sets, depth, x, y0, eager=False)
    assert seen == [True]
    torch.testing.assert_close(y_fused, y_eager, rtol=RTOL, atol=ATOL)
    a, b = eager[TARGET], recorded[TARGET]
    torch.testing.assert_close(b["scale"], a["scale"], rtol=RTOL, atol=ATOL)
    for qa, qb in zip(a["up_gates"], b["up_gates"], strict=True):
        assert (qa is None) == (qb is None)
        if qa is not None:
            torch.testing.assert_close(qb, qa, rtol=RTOL, atol=ATOL)
    assert len(b["chain_routes"]) == len(a["chain_routes"]) == depth
    for step_a, step_b in zip(a["chain_routes"], b["chain_routes"], strict=True):
        for ga, gb in zip(step_a, step_b, strict=True):
            assert torch.equal(ga.argmax(-1), gb.argmax(-1))
            torch.testing.assert_close(gb, ga, rtol=RTOL, atol=ATOL)
    keeps_a, keeps_b = _keeps(a), _keeps(b)
    assert len(keeps_b) == len(keeps_a) == depth - 1
    for step_a, step_b in zip(keeps_a, keeps_b, strict=True):
        for ma, mb in zip(step_a, step_b, strict=True):
            torch.testing.assert_close(mb, ma, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
def test_renders_after_the_freeze_match_the_eager_record(n_sets, depth, monkeypatch):
    x, y0 = _inputs(1311)
    sets = _layer(n_sets, "adaptive")
    _, eager = _record(sets, depth, x, y0, eager=True)
    _, recorded = _record(sets, depth, x, y0, eager=False)
    later_x, later_y0 = x * 0.8, y0 * 0.9
    for progress in (0.6, 0.9):
        expected = _call(sets, depth, progress, eager, later_x, later_y0, seed=11)
        actual = _call(sets, depth, progress, recorded, later_x, later_y0, seed=11)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        assert not torch.equal(actual, later_y0)


def test_several_imports_alone_record_on_the_eager_loop_with_a_reason():
    sets = _layer(3)[1:]
    assert not fused.record_fused(sets)
    assert fused.record_fused(sets[:1])
    assert fused.record_fused(_layer(3))


def test_the_pool_is_released_once_after_a_recording_call():
    released = []
    with R.release_after_records(lambda: released.append(1)) as note:
        pass
    assert not released and note["recorded"] == 0
    x, y0 = _inputs(1321)
    sets = _layer(2)
    with R.release_after_records(lambda: released.append(1)) as note:
        _call(sets, 3, FREEZE, {}, x, y0)
        assert not released
    assert note["recorded"] == 1 and released == [1]


class _Peak(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.storages: dict = {}
        self.now = 0
        self.peak = 0
        self.counting = True

    def _update(self):
        live = 0
        for key, (size, refs) in list(self.storages.items()):
            refs[:] = [r for r in refs if r() is not None]
            if not refs:
                del self.storages[key]
            else:
                live += size
        self.now = live
        if self.counting:
            self.peak = max(self.peak, live)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor) and not t.is_meta:
                storage = t.untyped_storage()
                key = (storage.data_ptr(), t.device)
                if storage.nbytes() == 0:
                    continue
                entry = self.storages.setdefault(key, (storage.nbytes(), []))
                entry[1].append(weakref.ref(t))
        self._update()
        return out


def _tensor_bytes(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        key = (storage.data_ptr(), value.device)
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v, seen) for v in value)
    return 0


def _peak_of_record(sets, depth, x, y0, eager=False):
    meter = _Peak()
    records = []
    build, record_fused = R.build, fused.record_fused

    def stop_at_build(rec, *args, **kwargs):
        meter.counting = False
        seen = set()
        records.append(
            _tensor_bytes(rec["routes"], seen)
            + _tensor_bytes(rec["gates"], seen)
            + _tensor_bytes(rec.get("input_keeps", {}), seen)
        )
        return build(rec, *args, **kwargs)

    R.build = stop_at_build
    if eager:
        fused.record_fused = lambda keyed: False
    try:
        with meter:
            _call(sets, depth, FREEZE, {}, x, y0)
    finally:
        R.build, fused.record_fused = build, record_fused
    assert len(records) == 1
    return meter.peak, records[0]


def _peak_without_record(sets, depth, x, y0):
    meter = _Peak()
    with meter:
        _call(sets, depth, 0.1, {}, x, y0)
    return meter.peak


@pytest.mark.parametrize("n_sets", [2, 7])
@pytest.mark.parametrize("depth", [3, 7])
def test_record_step_peak_stays_within_the_fused_call_and_its_record(n_sets, depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "sample")
    x, y0 = _inputs(1331, tokens=64)
    sets = _layer(n_sets)
    seen = _joint_calls(monkeypatch)
    plain = _peak_without_record(sets, depth, x, y0)
    assert seen == [False]
    peak, record = _peak_of_record(sets, depth, x, y0)
    assert seen == [False, True]
    before, _ = _peak_of_record(sets, depth, x, y0, eager=True)
    assert len(seen) == 2
    n = x.shape[0] * x.shape[1]
    workspace = 4 * n * max(s.down.shape[0] for s in sets)
    print(
        f"record step peak, {n_sets} sets, depth {depth}: eager record {before} B, "
        f"fused record {peak} B, fused call {plain} B, record {record} B"
    )
    assert peak <= plain + record + workspace
