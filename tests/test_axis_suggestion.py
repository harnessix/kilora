# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_comfyui_dropin import comfy, loras

from kilora.comfyui import nodes as N
from kilora.roles import suggest_axis


def test_sources_and_precedence(tmp_path):
    path = tmp_path / "character.safetensors"

    def write(meta):
        save_file({"x": torch.zeros(1)}, str(path), metadata=meta)

    write({"kilora_axis": "functional", "modelspec.trigger_phrase": "character"})
    assert suggest_axis(path) == ("functional", "metadata kilora_axis")
    write({"modelspec.trigger_phrase": "tataru (final fantasy)"})
    assert suggest_axis(path)[0] == "ip"
    assert "trigger" in suggest_axis(path)[1]
    write({"tags": "character"})
    assert suggest_axis(path)[0] == "ip"
    write({"civitai": json.dumps({"type": "Character"})})
    assert suggest_axis(path) == ("ip", "metadata civitai")
    write({})
    assert suggest_axis(path) == ("ip", "file name: character.safetensors")
    path.rename(tmp_path / "landscape.safetensors")
    assert suggest_axis(tmp_path / "landscape.safetensors") == ("creative", "no clear axis signal")


@pytest.mark.parametrize(
    ("name", "tags", "images", "expected"),
    [
        ("model", {"tataru taru": 60, "1girl": 60, "solo": 59, "beret": 60}, 60, "ip"),
        (
            "model",
            {
                "bangboo is a white mascot": 8,
                "bangboo is a red mascot": 5,
                "bangboo is a light blue mascot": 4,
            },
            38,
            "ip",
        ),
        ("model", {"1girl": 171, "idoly pride": 171, "solo": 171}, 171, "ip"),
        ("model", {"@aniart": 238, "solo": 232}, 238, "creative"),
        ("model", {"@anlwrsckhst": 100}, 100, "creative"),
        ("model", {"realstyle2": 213, "1girl": 213}, 213, "creative"),
        ("model", {"pc98gal_style": 87}, 87, "creative"),
        ("model", {"monochrome": 180, "greyscale": 180, "simple background": 180}, 180, "creative"),
        ("ij__anima-humanize-composition", {"1boy": 476, "solo": 476}, 784, "creative"),
        ("attributes", {"long hair": 95}, 100, "creative"),
        ("model", {"a character in a room": 10}, 100, "creative"),
    ],
)
def test_training_metadata(tmp_path, name, tags, images, expected):
    path = tmp_path / f"{name}.safetensors"
    save_file(
        {"x": torch.zeros(1)},
        str(path),
        metadata={
            "ss_tag_frequency": json.dumps({"set": tags}),
            "ss_dataset_dirs": json.dumps({"set": {"img_count": images}}),
        },
    )
    assert suggest_axis(path)[0] == expected


def test_dropin_auto_resolution(denoiser, folders, loras, comfy):
    from test_comfyui_dropin import FakeClip, _patcher, sample

    from kilora.comfyui import dropin

    path = folders["loras"] / loras[0]
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    save_file(tensors, str(path), metadata={"kilora_axis": "ip"})
    model, _ = dropin.KiLoRALoRALoader().load_lora(
        _patcher(denoiser), FakeClip(), loras[0], 1.0, 0.0
    )
    state = dropin.state_of(model)
    assert state.entries[0].axis == "auto"
    sample(model, torch.randn(1, 5, 4))
    assert "axis ip (metadata kilora_axis)" in state.cache[2]


def test_apply_override_and_report(denoiser, folders):
    from test_axes import _save

    _save(folders, "character", 30)
    node = N.KiLoRAImportLoRA()
    (auto,) = node.add("character.safetensors", 1.0)
    (explicit,) = node.add("character.safetensors", 1.0, axis="functional")
    model = FakePatcher(FakeBaseModel(denoiser))
    _, report, _ = N.KiLoRAApply().apply(model, auto, 1)
    assert "axis ip (file name:" in report
    _, report, _ = N.KiLoRAApply().apply(model, explicit, 1)
    assert "axis functional (node selection)" in report


def test_palette_saves_resolved_axis(folders, tmp_path, monkeypatch):
    from test_axes import _save

    from kilora import palette

    _save(folders, "character", 31)
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    (chain,) = N.KiLoRAImportLoRA().add("character.safetensors", 1.0)
    N.KiLoRASavePalette().save(chain, "axes", "axes")
    saved = palette.load_palette(tmp_path / "axes.kpalette.json")
    assert saved.entries[0].axis == "ip"
