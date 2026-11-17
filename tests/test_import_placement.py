# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from test_bypass import Patcher
from test_comfyui import FakeBaseModel
from test_families import _ab, _model

from kilora.bake import ModulePlan, ScopedSet, plan_targets
from kilora.comfyui import patching

NAME = "double_blocks.0.img_attn.proj"
RANKS = (2, 30)


def _files(tmp_path, model, dtype=torch.bfloat16):
    layer = model.get_submodule(NAME)
    stem = "lora_unet_" + NAME.replace(".", "_")
    gen = torch.Generator().manual_seed(5)
    entries = []
    for i, rank in enumerate(RANKS):
        a, b = _ab(gen, layer.in_features, layer.out_features, rank=rank)
        path = tmp_path / f"rank{rank}.safetensors"
        save_file(
            {
                f"{stem}.lora_down.weight": (a / rank).to(dtype).contiguous(),
                f"{stem}.lora_up.weight": b.to(dtype).contiguous(),
            },
            str(path),
        )
        entries.append(patching.ModEntry("lora", str(path), 0.7 if i == 0 else 0.4))
    return entries


def _import(tmp_path, monkeypatch, host_dtype):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KILORA_IMPORT_HOST_DTYPE", host_dtype)
    model = _model("flux")
    entries = _files(tmp_path, model)
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries, joint=True)
    return model, load.mods[0].sets[0]


def test_ranks_and_dtype_of_a_joint_import(tmp_path, monkeypatch):
    _, source = _import(tmp_path, monkeypatch, "file")
    expert = source.experts[NAME]
    assert tuple(expert.ranks) == RANKS
    assert expert.down.shape[:3] == (2, 1, max(RANKS))
    assert expert.down.dtype == expert.up.dtype == torch.bfloat16
    assert not torch.count_nonzero(expert.down[0, 0, RANKS[0] :])
    assert not torch.count_nonzero(expert.up[0, 0, RANKS[0] :])
    assert expert.keys_in.dtype == torch.float32


def test_file_dtype_host_copy_gives_the_same_device_copy_and_routing(tmp_path, monkeypatch):
    model, kept = _import(tmp_path / "file", monkeypatch, "file")
    _, wide = _import(tmp_path / "wide", monkeypatch, "float32")
    a, b = kept.experts[NAME], wide.experts[NAME]
    assert b.down.dtype == torch.float32
    assert torch.equal(a.keys_in, b.keys_in)
    if a.keys_out is not None:
        assert torch.equal(a.keys_out, b.keys_out)
    plans = []
    for source in (kept, wide):
        plan = plan_targets([ScopedSet(source, gate_mode="none")], 1)[NAME]
        plans.append(plan.to(torch.device("cpu"), torch.bfloat16))
    assert torch.equal(plans[0].sets[0].down, plans[1].sets[0].down)
    assert torch.equal(plans[0].sets[0].up, plans[1].sets[0].up)
    layer = model.get_submodule(NAME)
    x = torch.randn(2, 3, layer.in_features, generator=torch.Generator().manual_seed(1))
    h = x @ layer.weight.detach().t()
    outputs = []
    for plan in plans:
        placed = ModulePlan(parts=[(0, -1, plan)])
        with torch.no_grad():
            outputs.append(placed.run(x.bfloat16(), h.bfloat16()))
    assert torch.equal(outputs[0], outputs[1])


def _placed(tmp_path, monkeypatch):
    _, source = _import(tmp_path, monkeypatch, "file")
    plans = plan_targets([ScopedSet(source, gate_mode="none")], 1)
    return {target: ModulePlan(parts=[(0, -1, plan)]) for target, plan in plans.items()}


class _FakeMemory:
    def __init__(self, free):
        self.free = free

    def get_free_memory(self, device):
        return self.free

    def minimum_inference_memory(self):
        return 10


def _patcher(placed, other=0):
    placement = patching._Placement(list(placed.values()), 2, other)
    options = {patching._PLACEMENT_KEY: placement, patching._MEMORY_KEY: placement.bytes}
    return SimpleNamespace(
        model_options=options,
        load_device=torch.device("cuda"),
        model_size=lambda: 1000,
        loaded_size=lambda: 0,
        model=SimpleNamespace(memory_required=lambda shape: 100),
    ), placement


def test_placement_keeps_a_copy_that_fits_on_the_device(tmp_path, monkeypatch):
    placed = _placed(tmp_path, monkeypatch)
    patcher, placement = _patcher(placed)
    memory = _FakeMemory(placement.bytes + 1000 + 100 * 1 + 10 + 1)
    assert not patching.decide_placement(patcher, (1, 4, 8, 8), memory)
    assert not placement.host
    assert all(plan._prefetch is None for plan in placed.values())
    assert patcher.model_options[patching._MEMORY_KEY] == placement.bytes


def test_placement_moves_the_parts_to_the_cpu_when_memory_is_short(tmp_path, monkeypatch):
    placed = _placed(tmp_path, monkeypatch)
    patcher, placement = _patcher(placed, other=7)
    plan = placed[NAME]
    in_width = plan.parts[0][2].sets[0].down.shape[-1]
    x = torch.randn(2, 3, in_width, generator=torch.Generator().manual_seed(2))
    out_width = plan.parts[0][2].sets[0].up.shape[-1]
    h = torch.randn(2, 3, out_width, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        before = plan.run(x, h)
    assert plan._cache
    memory = _FakeMemory(1000)
    assert patching.decide_placement(patcher, (1, 4, 8, 8), memory)
    assert placement.host
    for module_plan in placed.values():
        assert module_plan._prefetch is not None
        assert not module_plan._cache
        for _, _, target in module_plan.parts:
            for layer in target.sets:
                for value in vars(layer).values():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == "cpu"
    assert placement.device_bytes() == min(placement.bytes, 3 * placement.block_bytes)
    assert patcher.model_options[patching._MEMORY_KEY] == 7 + placement.device_bytes()
    assert patching.decide_placement(patcher, None, _FakeMemory(10**15))
    with torch.no_grad():
        assert torch.equal(plan.run(x, h), before)


def test_placement_needs_no_decision_on_the_cpu(tmp_path, monkeypatch):
    placed = _placed(tmp_path, monkeypatch)
    patcher, placement = _patcher(placed)
    patcher.load_device = torch.device("cpu")
    assert not patching.decide_placement(patcher, (1, 4, 8, 8), _FakeMemory(0))
    assert not placement.host


def test_a_copy_of_the_options_shares_the_placement(tmp_path, monkeypatch):
    import copy

    placed = _placed(tmp_path, monkeypatch)
    patcher, placement = _patcher(placed)
    options = copy.deepcopy(patcher.model_options)
    assert options[patching._PLACEMENT_KEY] is placement
