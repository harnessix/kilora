# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import torch
from safetensors.torch import save_file
from test_comfyui import folders

from kilora.comfyui import nodes as N
from kilora.comfyui.patching import ModChain, ModEntry


def _save(dirs, name, metadata=None):
    save_file({"x": torch.zeros(1)}, str(dirs["loras"] / name), metadata=metadata)


def test_lora_stack_matches_three_chained_nodes(folders):
    specs = [
        ("red.safetensors", 0.5, 0.2),
        ("green.safetensors", 0.8, 1.0),
        ("blue.safetensors", -1.25, 0.0),
    ]
    for name, _model, _clip in specs:
        _save(folders, name)

    chained = None
    for name, model_strength, _clip in specs:
        (chained,) = N.KiLoRAImportLoRA().add(name, model_strength, sets=chained)

    (stacked,) = N.KiLoRAImportLoRA().add("red.safetensors", 0.1, lora_stack=specs)
    assert list(stacked) == list(chained)
    assert [entry.strength for entry in stacked] == [0.5, 0.8, -1.25]
    assert all(entry.kind == "lora" for entry in stacked)

    (off,) = N.KiLoRAImportLoRA().add("red.safetensors", 1.0, enabled=False, lora_stack=specs)
    assert not any(entry.enabled for entry in off)

    (single,) = N.KiLoRAImportLoRA().add("red.safetensors", 0.5, lora_stack=[])
    assert [(entry.path, entry.strength) for entry in single] == [
        (chained[0].path, 0.5),
    ]


def test_trigger_words_from_metadata_tags_and_sidecar(folders):
    _save(
        folders,
        "style.safetensors",
        metadata={
            "modelspec.trigger_phrase": "red hat, blue coat",
            "ss_tag_frequency": json.dumps(
                {
                    "people": {"red hat": 10, "zebra": 8, "extra": 3, "tail": 1},
                    "places": {"solo": 5, "smile": 4, "a": 1, "b": 1},
                }
            ),
        },
    )
    sidecar = folders["loras"] / "style.safetensors.civitai.info"
    sidecar.write_text(
        json.dumps({"trainedWords": ["blue coat", "civitai word"]}),
        encoding="utf-8",
    )
    (words,) = N.KiLoRATriggerWords().trigger_words("style.safetensors")
    assert words == "red hat, blue coat, zebra, extra, solo, smile, a, civitai word"


def test_trigger_words_include_the_lora_stack(folders):
    _save(folders, "one.safetensors", metadata={"modelspec.trigger_phrase": "alpha, beta"})
    _save(folders, "two.safetensors", metadata={"modelspec.trigger_phrase": "beta, gamma"})
    (words,) = N.KiLoRATriggerWords().trigger_words(
        "one.safetensors", lora_stack=[("two.safetensors", 1.0, 0.0)]
    )
    assert words == "alpha, beta, gamma"


def test_trigger_words_read_a_json_sidecar(folders):
    _save(folders, "plain.safetensors")
    (folders["loras"] / "plain.safetensors.json").write_text(
        json.dumps({"trainedWords": "ink, wash"}),
        encoding="utf-8",
    )
    (words,) = N.KiLoRATriggerWords().trigger_words("plain.safetensors")
    assert words == "ink, wash"


def test_civitai_sidecar_wins_over_json(folders):
    _save(folders, "both.safetensors")
    folder = folders["loras"]
    (folder / "both.safetensors.civitai.info").write_text(
        json.dumps({"trainedWords": ["from civitai"]}),
        encoding="utf-8",
    )
    (folder / "both.safetensors.json").write_text(
        json.dumps({"trainedWords": ["from json"]}),
        encoding="utf-8",
    )
    (words,) = N.KiLoRATriggerWords().trigger_words("both.safetensors")
    assert words == "from civitai"


class _SEG:
    def __init__(self, cropped_mask, crop_region):
        self.cropped_mask = cropped_mask
        self.crop_region = crop_region


def test_segs_become_the_region_mask():
    height, width = 4, 5
    first = torch.tensor([[1.0, 0.0], [0.5, 0.25]])
    second = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    segs = (
        (height, width),
        [
            _SEG(first, (0, 0, 2, 2)),
            _SEG(second, (1, 1, 3, 3)),
        ],
    )
    chain = ModChain(
        (
            ModEntry("lora", "a.safetensors", 1.0),
            ModEntry("lora", "b.safetensors", 0.5),
        )
    )
    ones = torch.ones(1, height, width)
    (out,) = N.KiLoRARegion().region(chain, ones, segs=segs)

    canvas = torch.zeros(height, width, dtype=torch.uint8)
    canvas[0:2, 0:2] |= (first * 255).to(torch.uint8)
    canvas[1:3, 1:3] |= (second * 255).to(torch.uint8)
    expected = (canvas.float() / 255.0).unsqueeze(0)

    assert out[0].region.shape == (1, height, width)
    assert torch.equal(out[0].region, expected)
    assert torch.equal(out[1].region, expected)
    assert not torch.equal(out[0].region, ones)
    assert (expected[0, 1, 0] * 255).round().item() == 127
    assert expected[0, 1, 1].item() == 1.0

    (mask_only,) = N.KiLoRARegion().region(chain, ones)
    assert torch.equal(mask_only[0].region, ones)

    (last,) = N.KiLoRARegion().region(chain, ones, apply_to="last mod", segs=segs)
    assert last[0].region is None
    assert torch.equal(last[1].region, expected)
