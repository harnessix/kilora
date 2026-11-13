# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kilora.adapter import compose
from kilora.library import LibraryLoader, SelectedFile
from kilora.library.prefetch import BlockPrefetch


class FakeCopies:
    def __init__(self):
        self.calls = []

    def copy(self, block, source):
        self.calls.append(("copy", block))
        return source, block

    def wait(self, event, _value):
        self.calls.append(("wait", event))


@pytest.mark.parametrize("window", [1, 2])
def test_prefetch_schedule(window):
    backend = FakeCopies()
    scheduler = BlockPrefetch(4, window, lambda block: block, backend)
    for block in range(4):
        assert scheduler.begin(block) == block
        backend.calls.append(("run", block))
        scheduler.end(block)
    scheduler.begin(0)
    calls = backend.calls
    for block in range(4):
        assert (
            calls.index(("copy", block))
            < calls.index(("wait", block))
            < calls.index(("run", block))
        )
    assert calls.index(("copy", 1)) < calls.index(("run", 0))
    if window == 2:
        assert calls.index(("copy", 2)) < calls.index(("run", 0))
    else:
        assert calls.index(("run", 0)) < calls.index(("copy", 2))
    assert calls.count(("copy", 0)) == 2


class TinyBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _selected(tmp_path):
    path = tmp_path / "test.safetensors"
    weights = {}
    for i in range(3):
        weights[f"blocks.{i}.lora_down.weight"] = torch.full((2, 3), 0.1 + i)
        weights[f"blocks.{i}.lora_up.weight"] = torch.full((3, 2), 0.2 + i)
    save_file(weights, str(path))
    return SelectedFile(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _run(tmp_path, device, mode, monkeypatch):
    import kilora.adapter as adapter

    monkeypatch.setattr(adapter, "AUTO_KEYS", False)
    torch.manual_seed(8)
    model = TinyBlocks().to(device).eval()
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    loader = LibraryLoader([tmp_path], device=device, library_vram_gib=0, library_prefetch=mode)
    choice, report = loader.load(
        [_selected(tmp_path)], patcher, SimpleNamespace(family="unknown"), model
    )
    attached = compose(model, choice.adapter_sets)
    with torch.no_grad():
        x = torch.ones(2, 3, device=device)
        outputs = [model(x), model(x)]
    attached.remove()
    return outputs, report


def test_cpu_mode_has_no_effect(tmp_path, monkeypatch):
    off, report = _run(tmp_path, "cpu", "off", monkeypatch)
    auto, _ = _run(tmp_path, "cpu", "auto", monkeypatch)
    assert all(torch.equal(a, b) for a, b in zip(off, auto, strict=True))
    assert "window 1" in str(report)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_prefetch_matches_off(tmp_path, monkeypatch):
    off, _ = _run(tmp_path, "cuda", "off", monkeypatch)
    auto, report = _run(tmp_path, "cuda", "auto", monkeypatch)
    assert all(torch.equal(a, b) for a, b in zip(off, auto, strict=True))
    assert report.prefetch_wait_seconds >= 0


def _run_baked(tmp_path, mode, monkeypatch):
    import kilora.adapter as adapter
    from kilora.bake import ScopedSet, baked_copy, module_plans, plan_targets
    from kilora.library.prefetch import PlanPrefetch

    monkeypatch.setattr(adapter, "AUTO_KEYS", False)
    torch.manual_seed(8)
    model = TinyBlocks().cuda().eval()
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    loader = LibraryLoader([tmp_path], device="cuda", library_vram_gib=0, library_prefetch=mode)
    choice, report = loader.load(
        [_selected(tmp_path)], patcher, SimpleNamespace(family="unknown"), model
    )
    adapter_set = choice.adapter_sets[0]
    targets = plan_targets([ScopedSet(adapter_set)], depth=1)
    placed = module_plans(targets, lambda name: (name, None), lambda _name: 3)
    PlanPrefetch(list(placed.values()), adapter_set.extra["_library_transfer"])
    for i in range(3):
        model.blocks[i] = baked_copy(model.blocks[i], placed[f"blocks.{i}"])
    with torch.no_grad():
        x = torch.ones(2, 3, device="cuda")
        outputs = [model(x), model(x)]
    return outputs, report


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_baked_prefetch_matches_off(tmp_path, monkeypatch):
    off, _ = _run_baked(tmp_path, "off", monkeypatch)
    auto, report = _run_baked(tmp_path, "auto", monkeypatch)
    assert all(torch.equal(a, b) for a, b in zip(off, auto, strict=True))
    assert "Prefetch auto" in str(report)


def test_baked_plans_share_block_schedule(monkeypatch):
    from kilora.library import prefetch

    calls = []

    class FakePlanCopies:
        def __init__(self, device, dtype, report, asynchronous):
            self.device, self.dtype = device, dtype

        def copy(self, block, source):
            calls.append(("copy", block))
            return {id(plan): plan for plan in source}, block

        def wait(self, event, _plans):
            calls.append(("wait", event))

    monkeypatch.setattr(prefetch.PlanPrefetch, "pin", lambda _self: None)
    monkeypatch.setattr(prefetch, "PlanCopies", FakePlanCopies)
    plans = [
        SimpleNamespace(parts=[(0, -1, SimpleNamespace(place=(i // 2, 2, "other")))])
        for i in range(4)
    ]
    config = {"mode": "auto", "window": 1, "report": SimpleNamespace()}
    shared = prefetch.PlanPrefetch(plans, config)
    for index, plan in enumerate(plans):
        assert shared.get(plan, torch.device("cuda"), torch.float32) is plan
        calls.append(("run", index))
        shared.finished(plan)
    assert calls.index(("copy", 1)) < calls.index(("run", 0))
    assert calls.count(("wait", 0)) == calls.count(("wait", 1)) == 1
    assert calls.count(("copy", 0)) == 2
