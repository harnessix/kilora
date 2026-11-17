# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from kilora.comfyui import patching
from kilora.comfyui.patching import (
    FreezeReserve,
    freeze_reserve_bytes,
    freeze_reserve_shapes,
    reserve_working_memory,
    working_memory_bytes,
)


class _ModelManagement:
    def __init__(self, free=3 << 30):
        self.calls = []
        self.free = free

    def free_memory(self, memory_required, device, keep_loaded=()):
        self.calls.append((memory_required, device))

    def get_free_memory(self, device):
        return self.free


def test_reserve_calls_free_memory_once_with_the_reserve_and_the_device():
    mm = _ModelManagement(free=5 << 30)
    free = reserve_working_memory("cuda:0", 300 << 20, phase="freeze_record", mm=mm)
    assert mm.calls == [(300 << 20, torch.device("cuda:0"))]
    assert free == 5 << 30


def test_reserve_finds_comfyui_in_the_loaded_modules(monkeypatch):
    mm = _ModelManagement(free=7 << 20)
    stand_in = types.ModuleType("comfy.model_management")
    stand_in.free_memory = mm.free_memory
    stand_in.get_free_memory = mm.get_free_memory
    comfy = types.ModuleType("comfy")
    comfy.model_management = stand_in
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", stand_in)
    assert reserve_working_memory("cuda", 64 << 20, phase="freeze_record") == 7 << 20
    assert mm.calls == [(64 << 20, torch.device("cuda"))]


def test_reserve_does_nothing_without_comfyui(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    assert reserve_working_memory("cuda", 64 << 20, phase="freeze_record") == 0


def test_reserve_does_nothing_on_the_cpu_or_without_a_reserve():
    mm = _ModelManagement()
    assert reserve_working_memory("cpu", 64 << 20, phase="freeze_record", mm=mm) == 0
    assert reserve_working_memory(None, 64 << 20, phase="freeze_record", mm=mm) == 0
    assert reserve_working_memory("cuda", 0, phase="freeze_record", mm=mm) == 0
    assert mm.calls == []


def test_live_write_reserves_through_the_same_function():
    mm = _ModelManagement(free=1 << 30)
    reserve = working_memory_bytes("live_write", tokens=4096, width=2048, sets=7, depth=7, rows=1)
    assert reserve_working_memory("cuda", reserve, phase="live_write", mm=mm) == 1 << 30
    assert mm.calls == [(reserve, torch.device("cuda"))]


def test_reserve_errors_leave_the_phase_as_before():
    class Broken(_ModelManagement):
        def free_memory(self, memory_required, device, keep_loaded=()):
            raise RuntimeError("no room")

    assert reserve_working_memory("cuda", 1 << 20, phase="freeze_record", mm=Broken()) == 0


SHAPES = {"width": 32, "sets": 3, "depth": 4, "patch": 2}


def _args(batch=2):
    return {"input": torch.zeros(batch, 4, 8, 8), "timestep": torch.tensor([1.0])}


def _spy(monkeypatch):
    calls = []

    def spy(device, reserve_bytes, *, phase):
        calls.append((device, reserve_bytes, phase))
        return 1

    monkeypatch.setattr(patching, "reserve_working_memory", spy)
    return calls


def test_the_freeze_record_step_reserves_once_per_generation(monkeypatch):
    calls = _spy(monkeypatch)
    reserve = FreezeReserve(0.4, SHAPES)
    steps = [0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 1.0, 1.0]
    for progress in steps + steps:
        reserve.before_call(_args(), progress)
    expected = freeze_reserve_bytes(SHAPES, (2, 4, 8, 8))
    assert expected > 0
    assert calls == [(torch.device("cpu"), expected, "freeze_record")] * 2


def test_the_reserve_comes_at_the_first_step_past_the_freeze_point(monkeypatch):
    calls = _spy(monkeypatch)
    reserve = FreezeReserve(0.5, SHAPES)
    seen = []
    for progress in (0.0, 0.3, 0.6, 0.9):
        before = len(calls)
        reserve.before_call(_args(), progress)
        seen.append(len(calls) - before)
    assert seen == [0, 0, 1, 0]


def test_steps_without_a_freeze_never_reserve(monkeypatch):
    calls = _spy(monkeypatch)
    for reserve in (FreezeReserve(None, SHAPES), FreezeReserve(0.4, None)):
        for progress in (0.0, 0.4, 0.8, 1.0):
            reserve.before_call(_args(), progress)
    assert calls == []


def test_the_reserve_shapes_come_from_the_keyed_freezing_sets():
    def keyed(out, freeze=0.4):
        return SimpleNamespace(down=torch.zeros(4, 8), up=torch.zeros(4, out), route_freeze=freeze)

    plans = {
        "a": SimpleNamespace(sets=[keyed(16), keyed(48), keyed(16)]),
        "b": SimpleNamespace(sets=[keyed(64), SimpleNamespace(down=None, up=None)]),
        "c": SimpleNamespace(sets=[keyed(128, None)]),
    }
    assert freeze_reserve_shapes(plans, 6.5, None) == {
        "width": 64,
        "sets": 3,
        "depth": 7,
        "patch": 2,
    }
    assert freeze_reserve_shapes(plans, 3, 4)["patch"] == 4
    assert freeze_reserve_shapes({"c": plans["c"]}, 3, None) is None
    assert freeze_reserve_shapes(None, 3, None) is None


@pytest.mark.parametrize("phase", ["freeze_record", "live_write"])
@pytest.mark.parametrize("name", ["tokens", "sets", "depth", "width"])
def test_the_reserve_grows_with_each_shape(phase, name):
    shapes = {"tokens": 256, "width": 64, "sets": 3, "depth": 4}
    base = working_memory_bytes(phase, **shapes, rows=8)
    more = working_memory_bytes(phase, **{**shapes, name: shapes[name] * 2}, rows=8)
    assert more > base > 0


def test_the_live_write_reserve_grows_with_the_rows():
    shapes = {"tokens": 256, "width": 64, "sets": 3, "depth": 4}
    one = working_memory_bytes("live_write", **shapes, rows=1)
    many = working_memory_bytes("live_write", **shapes, rows=32)
    assert many > one
    assert working_memory_bytes("live_write", **shapes) == working_memory_bytes(
        "live_write", **shapes, rows=64
    )


def test_an_unknown_phase_is_refused():
    with pytest.raises(ValueError, match="phase"):
        working_memory_bytes("sampling", tokens=1, width=1, sets=1, depth=1)


class _Peak(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.storages: dict = {}
        self.peak = 0

    def _prune(self):
        for key, (_, refs) in list(self.storages.items()):
            refs[:] = [r for r in refs if r() is not None]
            if not refs:
                del self.storages[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._prune()
        for t in pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor) and not t.is_meta:
                storage = t.untyped_storage()
                if storage.nbytes() == 0:
                    continue
                key = (storage.data_ptr(), t.device)
                self.storages.setdefault(key, (storage.nbytes(), []))[1].append(weakref.ref(t))
        self.peak = max(self.peak, sum(size for size, _ in self.storages.values()))
        return out


def _freeze_record_census(tokens, width, sets, depth):
    columns, experts = patching._CENSUS_COLUMNS, patching._CENSUS_EXPERTS
    held = [torch.empty(tokens, width) for _ in range(2 * sets)]
    kept = []
    for _ in range(depth):
        for _ in range(sets):
            read = torch.empty(tokens, columns)
            kept.append(torch.empty(tokens, experts))
            kept.append(torch.empty(tokens, columns))
            change = torch.empty(tokens, width)
            del read, change
    return held, kept


def _live_write_census(tokens, width, sets, depth, rows):
    columns = patching._CENSUS_COLUMNS
    inputs = torch.empty(tokens, width)
    reads = [torch.empty(tokens, columns) for _ in range(sets * depth)]
    old = torch.empty(rows, width)
    change = torch.empty(rows, width)
    new = torch.empty(rows, width)
    return inputs, reads, old, change, new


@pytest.mark.parametrize(("tokens", "width", "sets", "depth"), [(64, 32, 3, 4), (128, 48, 7, 7)])
def test_the_freeze_record_reserve_matches_the_census(tokens, width, sets, depth):
    meter = _Peak()
    with meter:
        _freeze_record_census(tokens, width, sets, depth)
    reserve = working_memory_bytes(
        "freeze_record", tokens=tokens, width=width, sets=sets, depth=depth
    )
    assert abs(reserve - meter.peak) <= 0.1 * meter.peak


@pytest.mark.parametrize(
    ("tokens", "width", "sets", "depth", "rows"), [(64, 32, 3, 4, 8), (128, 48, 7, 7, 1)]
)
def test_the_live_write_reserve_matches_the_census(tokens, width, sets, depth, rows):
    meter = _Peak()
    with meter:
        _live_write_census(tokens, width, sets, depth, rows)
    reserve = working_memory_bytes(
        "live_write", tokens=tokens, width=width, sets=sets, depth=depth, rows=rows
    )
    assert abs(reserve - meter.peak) <= 0.1 * meter.peak


class _Model(torch.nn.Module):
    manual_cast_dtype = torch.float32

    def memory_required(self, input_shape):
        return 100


class _Patcher:
    def __init__(self):
        self.model = _Model()
        self.model_options = {}

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape)


def test_the_model_memory_estimate_includes_the_freeze_reserve():
    patcher = _Patcher()
    patching._working_memory(patcher, {}, {})
    shape = (2, 4, 8, 8)
    assert patcher.memory_required(shape) == 100
    patcher.model_options[patching._RESERVE_KEY] = SHAPES
    reserve = freeze_reserve_bytes(SHAPES, shape)
    assert reserve > 0
    assert patcher.memory_required(shape) == 100 + reserve
    assert patcher.memory_required([1]) == 100
    wrapper = patcher.model_options["transformer_options"]["wrappers"]["prepare_sampling"][
        patching._MEMORY_KEY
    ][0]
    assert wrapper(lambda p, s: p.memory_required(s), patcher, shape) == 100 + reserve


def test_clear_drops_the_freeze_reserve():
    patcher = SimpleNamespace(
        model_options={patching._RESERVE_KEY: SHAPES},
        object_patches={},
    )
    patcher.clone = lambda: SimpleNamespace(
        model_options=dict(patcher.model_options), object_patches={}
    )
    out = patching.clear(patcher)
    assert patching._RESERVE_KEY not in out.model_options
