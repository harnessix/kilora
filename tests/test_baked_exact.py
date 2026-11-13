# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from safetensors.torch import save_file
from test_bypass import Patcher
from test_comfyui import FakeBaseModel
from test_families import _ab, _model

from kilora import compose, compute
from kilora.bake import ScopedSet, plan_targets
from kilora.comfyui import patching


@pytest.fixture
def routed(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    monkeypatch.setenv("KILORA_LYCORIS_APPLY", "exact")
    model = _model("flux")
    name = "double_blocks.0.img_attn.proj"
    layer = model.get_submodule(name)
    stem = "lora_unet_" + name.replace(".", "_")
    gen = torch.Generator().manual_seed(321)
    w1 = torch.randn(2, 4, generator=gen)
    w2 = torch.randn(layer.out_features // 2, layer.in_features // 4, generator=gen)
    a, b = _ab(gen, layer.in_features, layer.out_features, rank=1)
    c, d = _ab(gen, layer.in_features, layer.out_features, rank=1)
    weights = [
        {f"{stem}.lokr_w1": w1, f"{stem}.lokr_w2": w2},
        {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b},
        {f"{stem}.lora_down.weight": c, f"{stem}.lora_up.weight": d},
    ]
    entries = []
    for i, sd in enumerate(weights):
        path = tmp_path / f"expert{i}.safetensors"
        save_file(sd, str(path))
        entries.append(patching.ModEntry("lora", str(path), [0.7, -0.4, 0.6][i]))
    x = torch.randn(2, 5, layer.in_features, generator=gen)
    return model, name, entries, x, (torch.kron(w1, w2), b @ a, d @ c)


def adapter_layer(model, entries, name):
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries, joint=True)
    source = load.mods[0].sets[0]
    adapter = compose(model, [source], mix="stack")
    index = adapter.targets.index(name)
    layer = adapter._layer_sets(index, adapter.sites[index], torch.device("cpu"))[0]
    return adapter, dataclasses.replace(layer, gate_mode="none", import_score="read_share")


@pytest.mark.parametrize("mode", ["on", "off"])
def test_routed_exact_apply_matches_dense_reference(routed, monkeypatch, mode):
    model, name, entries, x, matrices = routed
    monkeypatch.setenv("KILORA_BYPASS", mode)
    adapter, reference = adapter_layer(model, entries[:2], name)
    shares = []
    base_layer = model.get_submodule(name)
    with torch.no_grad():
        base = torch.nn.functional.linear(x, base_layer.weight, base_layer.bias)
        with compute.recording_shares(lambda target, gates: shares.append(gates[0].clone())):
            compute.run_layer(x, base, [reference], 1, target=name)
    adapter.remove()
    patched, _ = patching.apply(
        Patcher(FakeBaseModel(model)),
        entries[:2],
        1,
        gate=("none", None),
        joint=True,
        import_score="read_share",
        bypass=mode,
    )
    with torch.no_grad():
        if mode == "on":
            patched.inject()
            try:
                actual = base_layer(x)
            finally:
                patched.eject()
        else:
            actual = patched.object_patches[f"diffusion_model.{name}"](x)
    expected = base + (x @ matrices[0].t()) * 0.7 * shares[0][..., :1]
    expected += (x @ matrices[1].t()) * -0.4 * shares[0][..., 1:2]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_plain_plan_matches_adapter_bits(routed):
    model, name, entries, x, _ = routed
    adapter, reference = adapter_layer(model, entries[1:], name)
    base_layer = model.get_submodule(name)
    with torch.no_grad():
        base = torch.nn.functional.linear(x, base_layer.weight, base_layer.bias)
        expected = compute.run_layer(x, base, [reference], 1, target=name)
    adapter.remove()
    patched, _ = patching.apply(
        Patcher(FakeBaseModel(model)),
        entries[1:],
        1,
        gate=("none", None),
        joint=True,
        import_score="read_share",
        bypass="off",
    )
    with torch.no_grad():
        actual = patched.object_patches[f"diffusion_model.{name}"](x)
    assert torch.equal(actual, expected)


def test_plan_copy_preserves_exact_components(routed):
    model, name, entries, _, _ = routed
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries[:2], joint=True)
    source = load.mods[0].sets[0]
    plan = plan_targets([ScopedSet(source)], 1)[name]
    assert plan.sets[0].exact
    moved = plan.to(torch.device("cpu"), torch.bfloat16)
    for index, parts in plan.sets[0].exact.items():
        assert all(
            new.dtype == torch.bfloat16 and torch.equal(new, old.to(torch.bfloat16))
            for old, new in zip(parts, moved.sets[0].exact[index], strict=True)
        )


def test_plan_dispatch_passes_exact_to_fused(routed, monkeypatch):
    from types import SimpleNamespace

    model, name, entries, x, _ = routed
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries[:2], joint=True)
    plan = plan_targets(
        [ScopedSet(load.mods[0].sets[0], gate_mode="none", import_score="read_share")], 1
    )[name].to(torch.device("cpu"), torch.float32)
    seen = []

    def run_fused(y0, sets, *args, **kwargs):
        seen.append(sets[0].exact)
        return y0

    monkeypatch.setattr(compute, "_fast_path_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        compute,
        "_fused_module",
        lambda: SimpleNamespace(fused_ok=lambda *a, **k: True, run_fused=run_fused),
    )
    with torch.no_grad():
        plan.run(x, model.get_submodule(name)(x))
    assert seen and seen[0]


def test_plain_merge_leaves_exact_sets_intact(routed):
    model, name, entries, _, _ = routed
    load, _ = patching.plan_chain(Patcher(FakeBaseModel(model)), entries[:2], joint=True)
    source = load.mods[0].sets[0]
    expert = source.experts[name]
    single = dataclasses.replace(
        expert,
        down=expert.down[:1],
        up=expert.up[:1],
        keys_in=expert.keys_in[:1],
        keys_out=expert.keys_out[:1],
        rows_in=expert.rows_in[:1],
        rows_out=expert.rows_out[:1],
    )
    source = dataclasses.replace(source, experts={name: single})
    scoped = ScopedSet(source, gate_mode="none")
    plan = plan_targets([scoped, scoped], 1, merge_plain=True)[name]
    assert len(plan.sets) == 2
    assert all(s.exact for s in plan.sets)
