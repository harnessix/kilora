# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import gc

import torch
from safetensors.torch import save_file
from test_bypass import Patcher
from test_comfyui import FakeBaseModel
from test_families import _ab, _model
from torch import nn

from kilora import compute, fused
from kilora.bake import ModulePlan, ScopedSet, plan_targets
from kilora.base_weights import BaseWeights, is_quantized_weight
from kilora.comfyui import patching

NAME = "double_blocks.0.img_attn.proj"


class QuantizedWeight(torch.Tensor):
    def dequantize(self):
        return self.as_subclass(torch.Tensor) / 4


def _quantize(model: nn.Module) -> list[tuple[int, int]]:
    shapes = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            stored = (module.weight.detach() * 4).as_subclass(QuantizedWeight)
            del module._parameters["weight"]
            module.register_buffer("weight", stored)
            shapes.append(tuple(stored.shape))
    return shapes


def _dense_float32_copies(model: nn.Module) -> list[torch.Tensor]:

    references = [
        module.weight.dequantize()
        for module in model.modules()
        if isinstance(module, nn.Linear) and is_quantized_weight(module.weight)
    ]
    gc.collect()
    found = []
    for obj in gc.get_objects():
        try:
            if type(obj) is not torch.Tensor or obj.dtype != torch.float32:
                continue
            if any(obj is r for r in references):
                continue
            if any(_same_values(obj, r) for r in references):
                found.append(obj)
        except ReferenceError:
            continue
    del references
    return found


def _same_values(t: torch.Tensor, reference: torch.Tensor) -> bool:
    return (
        t.shape == reference.shape
        and t.device == reference.device
        and not t.is_meta
        and torch.equal(t, reference)
    )


def _two_files(tmp_path, model, dtype):
    layer = model.get_submodule(NAME)
    stem = "lora_unet_" + NAME.replace(".", "_")
    gen = torch.Generator().manual_seed(17)
    entries = []
    for i in range(2):
        a, b = _ab(gen, layer.in_features, layer.out_features, rank=2)
        path = tmp_path / f"set{i}.safetensors"
        save_file(
            {
                f"{stem}.lora_down.weight": a.to(dtype).contiguous(),
                f"{stem}.lora_up.weight": b.to(dtype).contiguous(),
            },
            str(path),
        )
        entries.append(patching.ModEntry("lora", str(path), 0.7))
    return entries


def test_no_float32_copy_of_a_quantized_weight_after_a_forward(tmp_path):
    model = _model("flux")
    entries = _two_files(tmp_path, model, torch.bfloat16)
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries, joint=True)
    source = load.mods[0].sets[0]
    _quantize(model)
    layer = model.get_submodule(NAME)
    assert is_quantized_weight(layer.weight)
    entries_by_name = {
        n: (m, None) for n, m in model.named_modules() if n and isinstance(m, nn.Linear)
    }
    weights = BaseWeights(entries_by_name)
    first = weights[NAME]
    assert type(first) is torch.Tensor and first.dtype == torch.float32
    assert weights[NAME] is not first
    del first
    norms = {name: float(weights[name].norm()) for name in weights}
    assert len(norms) == len(entries_by_name)
    plan = plan_targets([ScopedSet(source, gate_mode="none")], 1)
    placed = {NAME: ModulePlan(parts=[(0, -1, plan[NAME])])}
    x = torch.randn(2, 3, layer.in_features)
    dense = layer.weight.dequantize()
    with torch.no_grad():
        h = x @ dense.t()
        out = placed[NAME].run(x, h)
    del dense
    assert out.shape == h.shape and torch.isfinite(out).all()
    assert not _dense_float32_copies(model)
    dense_weights = [
        m.weight.dequantize()
        for m in model.modules()
        if isinstance(m, nn.Linear) and is_quantized_weight(m.weight)
    ]
    for _, value, _ in compute._DERIVED.values():
        for t in _tensors(value):
            assert not (t.dtype == torch.float32 and any(_same_values(t, w) for w in dense_weights))


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _tensors(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _tensors(v)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for v in vars(value).values():
            yield from _tensors(v)


def test_imported_parts_keep_their_file_dtype(tmp_path):
    model = _model("flux")
    entries = _two_files(tmp_path, model, torch.bfloat16)
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries, joint=True)
    source = load.mods[0].sets[0]
    expert = source.experts[NAME]
    source = dataclasses.replace(
        source,
        experts={
            NAME: dataclasses.replace(
                expert, down=expert.down.to(torch.bfloat16), up=expert.up.to(torch.bfloat16)
            )
        },
    )
    plan = plan_targets([ScopedSet(source, gate_mode="none")], 1)[NAME]
    assert plan.sets[0].down.dtype == torch.bfloat16
    assert plan.sets[0].up.dtype == torch.bfloat16
    moved = plan.to(torch.device("cpu"), torch.bfloat16)
    assert moved.sets[0].down.dtype == torch.bfloat16
    assert moved.sets[0].up.dtype == torch.bfloat16
    assert torch.equal(moved.sets[0].down, plan.sets[0].down)
    assert moved.sets[0].keys_in.dtype == torch.float32


def test_plain_merge_scales_in_float32(tmp_path):
    model = _model("flux")
    entries = _two_files(tmp_path, model, torch.bfloat16)
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries[:1], joint=True)
    source = load.mods[0].sets[0]
    expert = source.experts[NAME]
    single = dataclasses.replace(
        expert,
        down=expert.down[:1].to(torch.bfloat16),
        up=expert.up[:1].to(torch.bfloat16),
        keys_in=expert.keys_in[:1],
        keys_out=None if expert.keys_out is None else expert.keys_out[:1],
        rows_in=None if expert.rows_in is None else expert.rows_in[:1],
        rows_out=None if expert.rows_out is None else expert.rows_out[:1],
    )
    source = dataclasses.replace(source, experts={NAME: single}, strength=0.3)
    scoped = ScopedSet(source, gate_mode="none")
    plan = plan_targets([scoped, scoped], 1, merge_plain=True)[NAME]
    if len(plan.sets) == 1:
        up = single.up.reshape(-1, single.up.shape[-1]).float()
        assert plan.sets[0].up.dtype == torch.float32
        torch.testing.assert_close(plan.sets[0].up, torch.cat([up * 0.3, up * 0.3]))


def test_fused_preparation_keeps_no_read_matrix_without_a_gate(tmp_path):
    model = _model("flux")
    entries = _two_files(tmp_path, model, torch.float32)
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries, joint=True)
    plan = plan_targets(
        [ScopedSet(load.mods[0].sets[0], gate_mode="none", import_score="read_share")], 1
    )[NAME].to(torch.device("cpu"), torch.float32)
    s = plan.sets[0]
    assert s.form == "input" and not compute.uses_adaptive(s)
    p = fused._prepared(s, torch.float32)
    assert p.dr_t.numel() == 0 and p.up_pad.numel() == 0


def test_derived_cache_has_a_byte_budget(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_CACHE_MB", "1")
    saved = dict(compute._DERIVED)
    compute._DERIVED.clear()
    try:
        sources = [torch.zeros(4) for _ in range(6)]
        for i, src in enumerate(sources):
            key = compute._tensor_key(f"budget-test-{i}", [src])
            compute._cache_put(key, torch.zeros(2**18))
            assert compute.derived_bytes() <= 2**20
        assert len(compute._DERIVED) == 1
        last = compute._tensor_key("budget-test-5", [sources[5]])
        assert compute._cache_get(last) is not None
    finally:
        compute._DERIVED.clear()
        compute._DERIVED.update(saved)
