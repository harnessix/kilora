# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora import compute
from kilora.comfyui import nodes as N
from kilora.comfyui.patching import LiveEnv, ModChain, ModEntry, expand_chain
from kilora.compute import LayerSet
from kilora.entry_window import active_layer_sets
from kilora.fixed_change import fixed_copy
from kilora.sections import Section, layer_changes


def toy(experts=1, **kw):
    return LayerSet(
        router=torch.eye(2),
        keys_in=torch.ones(experts, 1, 2),
        keys_out=torch.ones(experts, 1, 2),
        rows_in=torch.ones(experts),
        rows_out=torch.ones(experts),
        down=torch.eye(2)[:experts],
        up=torch.eye(2)[:experts],
        form="input",
        import_score="read_share",
        gate_mode="none",
        **kw,
    )


def run(s, progress, depth=1):
    x = torch.ones(1, 3, 2)
    y = torch.zeros_like(x)
    with torch.no_grad(), compute.sampling_progress(progress):
        return compute.run_layer(x, y, [s], depth)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("progress", [0.0, 0.2, 0.7, 1.0])
def test_full_window_is_bit_identical(eager, progress):
    baseline = toy(2)
    try:
        compute.use_depth_loop(eager)
        expected = run(baseline, progress, 3)
        explicit = dataclasses.replace(baseline, entry_windows=((0.0, 1.0),) * 2)
        assert torch.equal(run(explicit, progress, 3), expected)
        selected, _ = active_layer_sets([explicit], progress)
        assert selected[0] is explicit
    finally:
        compute.use_depth_loop(False)


@pytest.mark.parametrize("eager", [False, True])
def test_entry_starts_after_early_steps(eager):
    s = toy(entry_windows=((0.5, 0.8),))
    try:
        compute.use_depth_loop(eager)
        assert torch.equal(run(s, 0.2), torch.zeros(1, 3, 2))
        assert torch.equal(run(s, 0.7), run(dataclasses.replace(s, entry_windows=()), 0.7))
        assert torch.count_nonzero(run(s, 0.7)) > 0
        assert torch.equal(run(s, 0.9), torch.zeros(1, 3, 2))
        assert torch.count_nonzero(run(s, 0.5)) > 0
        assert torch.count_nonzero(run(s, 0.8)) > 0
    finally:
        compute.use_depth_loop(False)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_other_expert_fills_the_routed_mix(eager, scope):
    s = toy(2, entry_windows=((0.5, 1.0), (0.0, 1.0)), routing_scope=scope)
    seen = []
    try:
        compute.use_depth_loop(eager)
        with compute.recording_shares(lambda _, gates: seen.append(gates[0])):
            outside = run(s, 0.2)
        assert seen
        assert torch.equal(seen[-1][..., 0], torch.zeros(1, 3))
        assert torch.equal(seen[-1][..., 1], torch.ones(1, 3))
        assert torch.equal(outside, torch.tensor([0.0, 1.0]).expand(1, 3, 2))
        seen.clear()
        with compute.recording_shares(lambda _, gates: seen.append(gates[0])):
            run(s, 0.7)
        assert torch.equal(seen[-1], torch.full((1, 3, 2), 0.5))
    finally:
        compute.use_depth_loop(False)


def write_lora(folder, name="window.safetensors"):
    path = folder / name
    stem = "lora_unet_layers_0_attention_to_q"
    gen = torch.Generator().manual_seed(708)
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
        },
        str(path),
    )
    return path


def test_recipe_round_trips_window(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "hashes.json"))
    path = write_lora(folders["loras"])
    (chain,) = N.KiLoRAImportLoRA().add(path.name, 0.7, start=0.5, end=0.8)
    N.KiLoRASaveRecipe().save(chain, "window", "Window", settings={"mix": "stack"})
    recipe = folders["kilora"] / "window.kmod"
    selected = ModChain((ModEntry("mod", str(recipe)),))
    restored, _ = expand_chain(selected, LiveEnv(search_dirs=(str(folders["loras"]),)))
    assert (restored[0].start, restored[0].end) == (0.5, 0.8)
    base = FakePatcher(FakeBaseModel(denoiser))
    direct, report, _ = N.KiLoRAApply().apply(base, chain, 1, settings={"mix": "stack"})
    loaded, _, _ = N.KiLoRAApply().apply(base, selected, 1)
    assert "0.5" in report and "0.8" in report
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(708))
    for progress in (0.2, 0.7, 0.9):
        with torch.no_grad(), compute.sampling_progress(progress):
            assert torch.equal(direct.run(x), loaded.run(x))
            if progress != 0.7:
                assert torch.equal(direct.run(x), base.run(x))


def test_fixed_import_change_uses_window():
    layer = torch.nn.Linear(2, 2, bias=False)
    section = Section("fixed-delta", "fixed", {"proj": torch.eye(2)})
    section.meta["entry_window"] = (0.5, 0.8)
    change = layer_changes([section], 1.0)["proj"]
    patched = fixed_copy(layer, change)
    x = torch.ones(1, 2)
    with compute.sampling_progress(0.2):
        assert torch.equal(patched(x), layer(x))
    with compute.sampling_progress(0.7):
        assert torch.equal(patched(x), layer(x) + x)


@pytest.mark.parametrize("bounds", [(-0.1, 1), (0, 1.1), (float("nan"), 1)])
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        ModEntry("lora", "unused", start=bounds[0], end=bounds[1])


def test_reversed_interval_is_empty():
    assert torch.equal(run(toy(entry_windows=((0.8, 0.2),)), 0.5), torch.zeros(1, 3, 2))


def test_default_window_is_bit_identical_on_triton(monkeypatch):
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("needs Triton interpreter for CPU kernels")
    pytest.importorskip("triton")
    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    s = toy(2)
    seen = []
    original = fused.run_fused

    def record(*args, **kwargs):
        seen.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "run_fused", record)
    baseline = run(s, 0.7, 3)
    assert seen
    full = dataclasses.replace(s, entry_windows=((0.0, 1.0),) * 2)
    assert torch.equal(run(full, 0.7, 3), baseline)


def test_import_inputs_are_appended(monkeypatch):
    monkeypatch.setattr(N.FOLDERS, "list", lambda *a, **k: [])
    optional = N.KiLoRAImportLoRA.INPUT_TYPES()["optional"]
    assert list(optional) == ["sets", "enabled", "lora_stack", "axis", "size_norm", "start", "end"]
    assert optional["start"][1]["default"] == 0.0
    assert optional["end"][1]["default"] == 1.0


def test_import_stack_keeps_each_window(folders):
    a = write_lora(folders["loras"], "a.safetensors")
    b = write_lora(folders["loras"], "b.safetensors")
    (chain,) = N.KiLoRAImportLoRA().add(
        a.name,
        1.0,
        lora_stack=[(a.name, 0.7, 1.0), (b.name, 0.4, 1.0)],
        start=0.3,
        end=0.9,
    )
    assert len(chain) == 2
    assert [(e.start, e.end) for e in chain] == [(0.3, 0.9), (0.3, 0.9)]


def test_stack_does_not_merge_different_windows(folders):
    from test_families import _model

    gen = torch.Generator().manual_seed(708)
    stem = "diffusion_model.double_blocks.0.img_mlp.0"
    for name in ("a.safetensors", "b.safetensors"):
        save_file(
            {
                f"{stem}.lora_down.weight": torch.randn(2, 8, generator=gen),
                f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
            },
            str(folders["loras"] / name),
        )
    (chain,) = N.KiLoRAImportLoRA().add("a.safetensors", 0.7, axis="functional", start=0.5)
    (chain,) = N.KiLoRAImportLoRA().add("b.safetensors", 0.4, axis="functional", sets=chain)
    base = FakePatcher(FakeBaseModel(_model("flux")))
    both, _, _ = N.KiLoRAApply().apply(base, chain, 1, settings={"mix": "stack"})
    only, _, _ = N.KiLoRAApply().apply(base, ModChain((chain[1],)), 1, settings={"mix": "stack"})
    with torch.no_grad(), compute.sampling_progress(0.2):
        assert torch.equal(both.run(2), only.run(2))


def test_window_metadata_reaches_routed_choices(denoiser, folders):
    from kilora.comfyui.patching import plan_chain

    a = write_lora(folders["loras"], "a.safetensors")
    b = write_lora(folders["loras"], "b.safetensors")
    (chain,) = N.KiLoRAImportLoRA().add(a.name, 0.7, axis="creative", start=0.5)
    (chain,) = N.KiLoRAImportLoRA().add(b.name, 0.4, axis="creative", sets=chain)
    plan, _ = plan_chain(FakePatcher(FakeBaseModel(denoiser)), chain, joint=True)
    assert len(plan.mods) == 1
    assert plan.mods[0].sets[0].extra["entry_windows"] == ((0.5, 1.0), (0.0, 1.0))
