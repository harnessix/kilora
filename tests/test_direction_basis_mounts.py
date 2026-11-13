# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher
from torch import nn

from kilora.bindings import get_binding
from kilora.comfyui.patching import (
    ModChain,
    ModEntry,
    apply,
    choices_for,
    detect_family,
    diffusers_shapes,
    shape_model,
)
from kilora.errors import KiLoRAError
from kilora.library import LibraryIndex
from kilora.library.basis import build_basis, load_basis
from kilora.library.nodes import direction_adapters, direction_set
from kilora.presets import anima_name


def model_patcher(indices=(0, 1)):
    model = nn.Module()
    model.kilora_family = "anima"
    model.blocks = nn.ModuleDict()
    for i in indices:
        block = nn.Module()
        block.cross_attn = nn.Module()
        block.cross_attn.q_proj = nn.Linear(4, 3, bias=False)
        block.cross_attn.k_proj = nn.Linear(4, 3, bias=False)
        model.blocks[str(i)] = block
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=model))


@pytest.fixture
def library(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    generator = torch.Generator().manual_seed(12)
    for name in ("first", "second"):
        tensors = {}
        for i in range(2):
            for projection in ("q", "k"):
                stem = f"lora_unet_blocks_{i}_cross_attn_{projection}_proj"
                tensors[stem + ".lora_down.weight"] = torch.randn(2, 4, generator=generator)
                tensors[stem + ".lora_up.weight"] = torch.randn(3, 2, generator=generator)
        save_file(tensors, str(folder / f"{name}.safetensors"))
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    basis = build_basis(index, rank=2, top=2)
    assert not basis.skipped
    assert len(basis.files) == 2
    assert len(basis.mounts) == 4
    return index, basis


def test_comfyui_anima_shapes_match_library_mounts(library):
    _, basis = library
    patcher = model_patcher()
    binding = detect_family(patcher)
    assert binding == get_binding("anima")
    shapes = diffusers_shapes(patcher, binding)
    assert len(shapes) == 4
    assert set(basis.mounts) == {anima_name(target) for target in shapes}
    assert not set(shapes).intersection(basis.mounts)

    adapters = direction_adapters(basis, shapes)
    assert len(adapters) == len(basis.mounts)
    assert {target for adapter in adapters for target in adapter.experts} == set(shapes)
    for adapter in adapters:
        target, expert = next(iter(adapter.experts.items()))
        layer = direction_set(basis, anima_name(target))
        torch.testing.assert_close(expert.down.flatten(0, 2), layer.down)
        torch.testing.assert_close(expert.up.flatten(0, 2), layer.up)
        torch.testing.assert_close(adapter.extra["direction_core"], layer.direction_core)


def test_saved_original_mount_names_load_and_match(library):
    index, original = library
    manifest = (index.path / "basis.json").read_bytes()
    tensors = (index.path / "basis.safetensors").read_bytes()
    loaded = load_basis(index)
    assert set(loaded.mounts) == set(original.mounts)
    assert all(name.startswith("blocks.") for name in loaded.mounts)
    patcher = model_patcher()
    binding = detect_family(patcher)
    shapes = diffusers_shapes(patcher, binding)
    adapters = direction_adapters(loaded, shapes)
    assert len(adapters) == 4
    choices = choices_for(patcher, [ModEntry("direction", str(index.path))], binding)
    assert len(choices) == 1
    assert {target for adapter in choices[0].adapter_sets for target in adapter.experts} == set(
        shapes
    )
    assert set(loaded.mounts) == set(original.mounts)
    assert (index.path / "basis.json").read_bytes() == manifest
    assert (index.path / "basis.safetensors").read_bytes() == tensors


@pytest.mark.parametrize(
    ("mount", "target"),
    [
        ("transformer_blocks.0.attn2.to_q", "transformer_blocks.0.attn2.to_q"),
        ("blocks.0.adaln_modulation_cross_attn.1", "transformer_blocks.0.norm2.linear_1"),
        ("final_layer.linear", "proj_out"),
        ("layer", "layer"),
    ],
)
def test_target_names_and_other_anima_names_match(library, mount, target):
    _, basis = library
    original = next(iter(basis.mounts))
    renamed = replace(
        basis,
        mounts={mount: basis.mounts[original]},
        _parts={name: {mount: parts[original]} for name, parts in basis._parts.items()},
    )
    adapters = direction_adapters(renamed, {target: (4, 3)})
    assert len(adapters) == 1
    assert list(adapters[0].experts) == [target]


@pytest.mark.parametrize("reverse", [False, True])
def test_two_mounts_for_one_target_raise_with_both_names(library, reverse):
    _, basis = library
    target = "transformer_blocks.0.attn2.to_q"
    original = anima_name(target)
    entries = [(original, basis.mounts[original]), (target, basis.mounts[original])]
    duplicate = replace(basis, mounts=dict(reversed(entries) if reverse else entries))
    with pytest.raises(ValueError) as error:
        direction_adapters(duplicate, {target: (4, 3)})
    assert original in str(error.value)
    assert target in str(error.value)


def test_shape_check_uses_model_target_name(library):
    _, basis = library
    target = "transformer_blocks.0.attn2.to_q"
    with pytest.raises(ValueError, match=rf"{target}: library basis does not match model shape"):
        direction_adapters(basis, {target: (3, 4)})


def test_no_match_error_lists_first_three_mounts_and_targets(library):
    index, _ = library
    patcher = model_patcher(indices=(10, 11))
    binding = detect_family(patcher)
    targets = list(diffusers_shapes(patcher, binding))
    mounts = list(load_basis(index).mounts)
    with pytest.raises(KiLoRAError) as error:
        choices_for(patcher, [ModEntry("direction", str(index.path))], binding)
    message = str(error.value)
    assert "the stored basis has no mounts matching this model" in message
    assert f"basis mounts: {mounts[:3]}" in message
    assert f"model targets: {targets[:3]}" in message
    assert mounts[3] not in message
    assert targets[3] not in message


@pytest.fixture
def image_library(tmp_path):
    names = []
    for i in range(2):
        for attention in ("self_attn", "cross_attn"):
            names.extend(
                f"blocks.{i}.{attention}.{projection}"
                for projection in ("q_proj", "k_proj", "v_proj", "output_proj")
            )
        names.extend(f"blocks.{i}.mlp.layer{j}" for j in (1, 2))
        names.extend(
            f"blocks.{i}.adaln_modulation_{part}.{j}"
            for part in ("self_attn", "cross_attn", "mlp")
            for j in (1, 2)
        )
    model = shape_model(dict.fromkeys(names, (4, 3)))
    model.kilora_family = "anima"
    folder = tmp_path / "loras"
    folder.mkdir()
    generator = torch.Generator().manual_seed(29)
    for label in ("first", "second"):
        tensors = {}
        for name in names:
            stem = "lora_unet_" + name.replace(".", "_")
            tensors[stem + ".lora_down.weight"] = torch.randn(2, 4, generator=generator)
            tensors[stem + ".lora_up.weight"] = torch.randn(3, 2, generator=generator)
        save_file(tensors, str(folder / f"{label}.safetensors"))
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    basis = build_basis(index, rank=2, top=2)
    assert not basis.skipped
    assert set(basis.mounts) == set(names)
    return index, basis, FakePatcher(FakeBaseModel(model))


def test_real_image_mounts_report_every_never_target(image_library):
    index, basis, patcher = image_library
    manifest = (index.path / "basis.json").read_bytes()
    tensors = (index.path / "basis.safetensors").read_bytes()
    binding = detect_family(patcher)
    shapes = diffusers_shapes(patcher, binding)
    never = [name for name in basis.mounts if ".adaln_modulation_" in name]
    image = set(basis.mounts).difference(never)
    adapters = direction_adapters(basis, shapes)
    assert len(adapters) == len(image) == 20
    assert {anima_name(target) for adapter in adapters for target in adapter.experts} == image
    assert adapters.skipped_count == len(never) == 12
    assert adapters.skipped_names == never[:3]
    line = f"skipped 12 never-target mounts ({', '.join(never[:3])}, ...)"
    assert adapters.notes == [line]

    entries = [ModEntry("direction", str(index.path))]
    choice = choices_for(patcher, entries, binding)[0]
    assert choice.notes == [line]
    _, report = apply(patcher, ModChain(entries), depth=0)
    assert report.splitlines().count(line) == 1
    assert (index.path / "basis.json").read_bytes() == manifest
    assert (index.path / "basis.safetensors").read_bytes() == tensors


def test_empty_adapters_report_never_target_mounts(image_library):
    index, basis, patcher = image_library
    never = {name: mount for name, mount in basis.mounts.items() if ".adaln_modulation_" in name}
    basis = replace(basis, mounts=never)
    adapters = direction_adapters(basis, {})
    assert not adapters
    assert adapters.skipped_count == len(never)
    assert len(adapters.skipped_names) == 3
    assert "never-target mounts" in adapters.notes[0]

    patcher.model.diffusion_model = model_patcher(indices=(10, 11)).model.diffusion_model
    with pytest.raises(KiLoRAError) as error:
        choices_for(patcher, [ModEntry("direction", str(index.path))], detect_family(patcher))
    assert adapters.notes[0] in str(error.value)


def test_no_skips_leave_report_empty(library):
    _, basis = library
    patcher = model_patcher()
    adapters = direction_adapters(basis, diffusers_shapes(patcher, detect_family(patcher)))
    assert adapters.skipped_count == 0
    assert adapters.skipped_names == []
    assert adapters.notes == []


def test_unmatched_image_mounts_also_have_a_report_line(library):
    _, basis = library
    adapters = direction_adapters(basis, {})
    assert not adapters
    assert adapters.skipped_count == 0
    assert adapters.skipped_names == []
    sample = ", ".join(list(basis.mounts)[:3])
    assert adapters.notes == [f"skipped 4 unmatched mounts ({sample}, ...)"]
