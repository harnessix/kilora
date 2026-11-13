# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora.adapter import Settings
from kilora.comfyui import nodes as N
from kilora.comfyui.settings import resolve_settings
from kilora.modfile import save_mod
from kilora.modlist import ModChoice, load_choice
from kilora.references import MissingReferenceError
from kilora.sections import Section


def make_recipe(path, files):
    sections = []
    for file, strength in files:
        sections.append(
            Section(
                "weight-ref",
                file.stem,
                {},
                {
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "filename": file.name,
                    "size": file.stat().st_size,
                    "form": "lora",
                    "transforms": {
                        "strength": strength,
                        "axis": "auto",
                        "roles": "auto",
                        "region": None,
                        "timeline": [],
                        "enabled": True,
                    },
                },
            )
        )
    settings = {
        **resolve_settings({"mix": "stack"}),
        "axis_alpha": "off",
        "library": False,
        "output_gate": "none",
        "gate_kappa": 0.0,
        "path_width": 0,
        "path_select": "mod",
        "route_freeze": 0.0,
    }
    sections.append(Section("logic", "settings", {}, {"version": 1, "settings": settings}))
    save_mod(path, [], Settings(), sections=sections, id="recipe", name="recipe")


def test_recipe_matches_original_chain(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    stem = "lora_unet_layers_0_attention_to_q"
    gen = torch.Generator().manual_seed(3)
    plain, kron = (folders["loras"] / name for name in ("plain.safetensors", "kron.safetensors"))
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
        },
        str(plain),
    )
    save_file(
        {
            f"{stem}.lokr_w1": torch.randn(4, 4, generator=gen),
            f"{stem}.lokr_w2": torch.randn(4, 4, generator=gen),
        },
        str(kron),
    )
    recipe = folders["kilora"] / "recipe.kmod"
    make_recipe(recipe, [(plain, 0.7), (kron, -0.3)])
    loaded = load_choice(ModChoice(path=recipe, strength=0.5), 0, [folders["loras"]])
    assert [r.path for r in loaded.references] == [str(plain), str(kron)]
    assert [r.transforms["strength"] for r in loaded.references] == [0.35, -0.15]
    assert loaded.logic["mix"] == "stack"
    (chain,) = N.KiLoRAImportLoRA().add(plain.name, 0.35)
    (chain,) = N.KiLoRAImportLoRA().add(kron.name, -0.15, sets=chain)
    base = FakePatcher(FakeBaseModel(denoiser))
    want, _, _ = N.KiLoRAApply().apply(base, chain, -1, settings=resolve_settings({"mix": "stack"}))
    (recipe_chain,) = N.KiLoRASelect().select(recipe.name, 0.5, 1.0)
    got, _, _ = N.KiLoRAApply().apply(base, recipe_chain, -1)
    x = torch.randn(1, 5, 4, generator=gen)
    assert torch.equal(got.run(x), want.run(x))
    loaded_node, _ = N.KiLoRALoadMod().load(base, recipe.name, 0.5)
    assert torch.equal(loaded_node.run(x), want.run(x))
    plain.unlink()
    with pytest.raises(MissingReferenceError):
        N.KiLoRALoadMod().load(base, recipe.name, 1.0)


def test_save_recipe_round_trip(denoiser, folders, tmp_path, monkeypatch):
    from dataclasses import replace

    from safetensors import safe_open

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    stem = "lora_unet_layers_0_attention_to_q"
    gen = torch.Generator().manual_seed(123)
    file = folders["loras"] / "style.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
        },
        str(file),
    )
    (chain,) = N.KiLoRAImportLoRA().add(file.name, 0.7)
    from test_modlist import write_mod

    mod = write_mod(denoiser, folders["kilora"], "style-mod", seed=8)
    (chain,) = N.KiLoRASelect().select(mod.name, -0.4, 1.0, sets=chain)
    settings = resolve_settings({"mix": "stack", "lycoris_apply": "exact"})
    (report,) = N.KiLoRASaveRecipe().save(chain, "saved", "My recipe", settings=settings)
    assert hashlib.sha256(file.read_bytes()).hexdigest() in report
    path = folders["kilora"] / "saved.kmod"
    assert (
        load_choice(ModChoice(path=path), 0, [folders["loras"], folders["kilora"]]).logic[
            "lycoris_apply"
        ]
        == "exact"
    )
    with safe_open(path, framework="pt") as contents:
        assert list(contents.keys()) == []
    base = FakePatcher(FakeBaseModel(denoiser))
    want, _, _ = N.KiLoRAApply().apply(base, chain, -1, settings=settings)
    direct, _ = N.KiLoRALoadMod().load(base, path.name, 1.0)
    (selected,) = N.KiLoRASelect().select(path.name, 1.0, 1.0)
    via_list, _, _ = N.KiLoRAApply().apply(base, selected, -1)
    x = torch.randn(1, 5, 4, generator=gen)
    assert torch.equal(want.run(x), direct.run(x))
    assert torch.equal(want.run(x), via_list.run(x))
    (report,) = N.KiLoRASaveRecipe().save(
        N.ModChain((replace(chain[0], region=torch.ones(1, 2, 2)),)), "region", "Region"
    )
    assert "region was not saved" in report


def test_flux_recipe_matches_chain_in_fast_and_loop_paths(folders, tmp_path, monkeypatch):
    from test_families import _model

    from kilora import compute

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    gen = torch.Generator().manual_seed(17)
    stem = "diffusion_model.double_blocks.0.img_attn.proj"
    plain, kron = (folders["loras"] / n for n in ("flux-lora.safetensors", "flux-kron.safetensors"))
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 8, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(8, 2, generator=gen),
        },
        str(plain),
    )
    save_file(
        {
            f"{stem}.lokr_w1": torch.randn(2, 2, generator=gen),
            f"{stem}.lokr_w2": torch.randn(4, 4, generator=gen),
        },
        str(kron),
    )
    path = folders["kilora"] / "flux-recipe.kmod"
    make_recipe(path, [(plain, 0.7), (kron, -0.3)])
    (chain,) = N.KiLoRAImportLoRA().add(plain.name, 0.7)
    (chain,) = N.KiLoRAImportLoRA().add(kron.name, -0.3, sets=chain)
    (recipe,) = N.KiLoRASelect().select(path.name, 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(_model("flux")))
    for loop in (False, True):
        compute.use_depth_loop(loop)
        try:
            expected, _, _ = N.KiLoRAApply().apply(base, chain, -1, settings={"mix": "stack"})
            actual, _, _ = N.KiLoRAApply().apply(base, recipe, -1)
            assert torch.equal(actual.run(2), expected.run(2))
        finally:
            compute.use_depth_loop(False)


def test_recipe_files_are_hashed_once_per_process(tmp_path, monkeypatch):
    import os

    import kilora.references as references

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    monkeypatch.setattr(references, "_VERIFIED", {})
    file = tmp_path / "a.safetensors"
    save_file({"x": torch.ones(4)}, str(file))
    meta = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "filename": file.name}
    calls = []
    real = references._sha256
    monkeypatch.setattr(references, "_sha256", lambda p: calls.append(p) or real(p))
    for _ in range(3):
        assert references.resolve(meta, [tmp_path]) == file
    assert len(calls) == 1
    stat = file.stat()
    save_file({"x": torch.zeros(4)}, str(file))
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(references.KiLoRAError, match="Hash mismatch"):
        references.resolve(meta, [tmp_path])


def test_core_resolve_refuses_a_recipe(folders, tmp_path, monkeypatch):
    from kilora.modlist import resolve

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    lora = folders["loras"] / "plain.safetensors"
    save_file({"lora_unet_x.lora_down.weight": torch.ones(1, 4)}, str(lora))
    recipe = folders["kilora"] / "recipe.kmod"
    make_recipe(recipe, [(lora, 1.0)])
    plan = resolve([ModChoice(path=recipe)], search_dirs=[folders["loras"]])
    assert any("a recipe" in p for p in plan.problems)


def test_shared_entry_recipe_and_output(denoiser, folders, tmp_path, monkeypatch):
    from kilora.comfyui import patching

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    stem = "lora_unet_layers_0_attention_to_q"
    file = folders["loras"] / "shared.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.ones(2, 16),
            f"{stem}.lora_up.weight": torch.ones(16, 2),
        },
        str(file),
    )
    ip_file = folders["loras"] / "ip-zero.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.ones(2, 16),
            f"{stem}.lora_up.weight": torch.zeros(16, 2),
        },
        str(ip_file),
    )
    (shared,) = N.KiLoRAImportLoRA().add(file.name, 0.7, axis="shared")
    (creative,) = N.KiLoRAImportLoRA().add(file.name, 0.7, axis="creative")
    assert shared[0].axis == "shared"
    N.KiLoRASaveRecipe().save(shared, "shared", "Shared")
    path = folders["kilora"] / "shared.kmod"
    loaded = load_choice(ModChoice(path=path), 0, [folders["loras"], folders["kilora"]])
    assert loaded.references[0].transforms["axis"] == "shared"
    expanded, _ = patching.expand_chain(
        [patching.ModEntry("mod", str(path))],
        patching.LiveEnv(search_dirs=(str(folders["loras"]),)),
    )
    assert expanded[0].axis == "shared"
    base = FakePatcher(FakeBaseModel(denoiser))
    x = torch.randn(1, 5, 4)
    from kilora.call_values import axis_alphas

    (shared,) = N.KiLoRAImportLoRA().add(ip_file.name, 1.0, sets=shared, axis="ip")
    (creative,) = N.KiLoRAImportLoRA().add(ip_file.name, 1.0, sets=creative, axis="ip")
    actual, _, _ = N.KiLoRAApply().apply(base, shared, -1)
    expected, _, _ = N.KiLoRAApply().apply(base, creative, -1)
    sets = [
        s
        for module in actual.object_patches.values()
        for _, _, target in module._kilora_plan.parts
        for s in target.sets
        if s.axis == "shared"
    ]
    assert sets
    with axis_alphas({"creative": 0.0, "ip": 1.0}):
        assert not torch.equal(actual.run(x), expected.run(x))
