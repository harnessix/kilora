# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from kilora.comfyui.nodes import FOLDERS, NODE_CLASS_MAPPINGS

ROOT = Path(__file__).resolve().parents[1]
CORE = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "EmptySD3LatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "SolidMask",
    "MaskComposite",
    "PreviewImage",
    "EmptyLatentImage",
    "ConditioningCombine",
}
WORKFLOWS = sorted((ROOT / "example_workflows").glob("*.json"))


def _object_info(monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    info = {k: {"input": c.INPUT_TYPES()} for k, c in NODE_CLASS_MAPPINGS.items()}
    info.update({k: {"input": {"required": {}}} for k in CORE})
    return info


def test_the_use_cases_are_there():
    names = {p.name for p in WORKFLOWS}
    assert {
        "kilora-static-mod-depth1.json",
        "kilora-two-mods.json",
        "kilora-live-mode-stub.json",
        "kilora-lora-mix.json",
        "kilora-character-style-background.json",
        "kilora-depth-schedule.json",
        "kilora-palette-shuffle.json",
        "kilora-depth-map.json",
        "kilora-layers.json",
        "kilora-learn-from-masked-images.json",
    } <= names


def test_imported_loras_use_recommended_strength():
    for path in WORKFLOWS:
        graph = json.loads(path.read_text())
        for node in graph["nodes"]:
            if node["type"] == "KiLoRAImportLoRA":
                assert node["widgets_values"][1] == 0.7, (path.name, node["id"])


def _wiring(name):
    g = json.loads((ROOT / "example_workflows" / name).read_text())
    by_id = {n["id"]: n for n in g["nodes"]}
    links = {ln[0]: ln for ln in g["links"]}

    def source(node, inp):
        i = next(i for i in node["inputs"] if i["name"] == inp)
        return by_id[links[i["link"]][1]]

    return g, source


def test_lora_mix_routes_three_loras_into_the_sampler():
    g, source = _wiring("kilora-lora-mix.json")
    ks = next(n for n in g["nodes"] if n["type"] == "KSampler")
    apply = source(ks, "model")
    assert apply["type"] == "KiLoRAApply"
    chain, node = [], source(apply, "sets")
    while node["type"] == "KiLoRAImportLoRA":
        chain.append(node)
        linked = next(i for i in node["inputs"] if i["name"] == "sets")["link"]
        node = source(node, "sets") if linked is not None else None
        if node is None:
            break
    assert len(chain) == 3


def test_character_region_is_on_the_character_lora_only():
    g, source = _wiring("kilora-character-style-background.json")
    region = next(n for n in g["nodes"] if n["type"] == "KiLoRARegion")
    assert region["widgets_values"] == ["last mod"]
    assert "character" in source(region, "sets")["title"]
    assert source(region, "mask")["type"] == "MaskComposite"
    ks = next(n for n in g["nodes"] if n["type"] == "KSampler")
    assert source(ks, "model")["type"] == "KiLoRAApply"
    assert not any(n["type"] == "KiLoRARegionAttention" for n in g["nodes"])


def test_depth_schedule_feeds_apply():
    g, source = _wiring("kilora-depth-schedule.json")
    apply = next(n for n in g["nodes"] if n["type"] == "KiLoRAApply")
    assert source(apply, "depth_schedule")["type"] == "KiLoRADepthSchedule"


def test_live_workflow_records_before_and_learns_after_the_sampler():
    g = json.loads((ROOT / "example_workflows" / "kilora-live-mode-stub.json").read_text())
    by_id = {n["id"]: n for n in g["nodes"]}
    links = {ln[0]: ln for ln in g["links"]}

    def source(node, name):
        inp = next(i for i in node["inputs"] if i["name"] == name)
        return by_id[links[inp["link"]][1]]["type"]

    ks = next(n for n in g["nodes"] if n["type"] == "KSampler")
    write = next(n for n in g["nodes"] if n["type"] == "KiLoRALiveWrite")
    dec = next(n for n in g["nodes"] if n["type"] == "VAEDecode")
    assert source(ks, "model") == "KiLoRALiveRecord"
    assert source(write, "path") == "KiLoRALiveRecord" and source(write, "samples") == "KSampler"
    assert source(dec, "samples") == "KiLoRALiveWrite"


@pytest.mark.parametrize(
    "name",
    [
        "kilora-lora-mix.json",
        "kilora-character-style-background.json",
        "kilora-depth-schedule.json",
        "kilora-depth-map.json",
    ],
)
def test_new_examples_use_anima_like_comfyuis_template(name):
    g = json.loads((ROOT / "example_workflows" / name).read_text())
    by_type = {n["type"]: n for n in g["nodes"]}
    assert by_type["UNETLoader"]["widgets_values"][0] == "anima-base-v1.0.safetensors"
    assert by_type["CLIPLoader"]["widgets_values"][:2] == [
        "qwen_3_06b_base.safetensors",
        "stable_diffusion",
    ]
    assert by_type["VAELoader"]["widgets_values"] == ["qwen_image_vae.safetensors"]
    _, _, steps, cfg, sampler, scheduler, _ = by_type["KSampler"]["widgets_values"]
    assert (steps, cfg, sampler, scheduler) == (30, 4.0, "er_sde", "simple")
    assert "EmptyLatentImage" in by_type


def test_palette_example_feeds_apply_and_keeps_save_bypassed():
    g, source = _wiring("kilora-palette-shuffle.json")
    apply = next(n for n in g["nodes"] if n["type"] == "KiLoRAApply")
    assert source(apply, "sets")["type"] == "KiLoRAPalette"
    assert source(apply, "settings")["type"] == "KiLoRAPalette"
    pal = next(n for n in g["nodes"] if n["type"] == "KiLoRAPalette")
    assert pal["widgets_values"][1] is True
    save = next(n for n in g["nodes"] if n["type"] == "KiLoRASavePalette")
    assert save["mode"] == 4 and source(save, "sets")["type"] == "KiLoRAPalette"


def test_depth_map_example_feeds_apply_from_a_mask():
    g, source = _wiring("kilora-depth-map.json")
    apply = next(n for n in g["nodes"] if n["type"] == "KiLoRAApply")
    dmap = source(apply, "depth_map")
    assert dmap["type"] == "KiLoRADepthMap"
    assert source(dmap, "mask")["type"] == "MaskComposite"
    low, high, invert, *_ = dmap["widgets_values"]
    assert (low, high, invert) == (2.0, 5.0, True)


def test_imported_workflow_records_masks_writes_and_exports_in_order():
    g, source = _wiring("kilora-imported-live-edit.json")
    types = {n["type"]: n for n in g["nodes"]}
    assert sum(n["type"] == "KiLoRAImportLoRA" for n in g["nodes"]) == 2
    record = types["KiLoRALiveRecord"]
    assert source(record, "model")["type"] == "KiLoRAApply"
    assert source(types["KSampler"], "model") is record
    write = types["KiLoRALiveWrite"]
    assert source(write, "path") is record
    assert source(write, "samples")["type"] == "KSampler"
    assert source(write, "mask")["type"] == "MaskComposite"
    export = types["KiLoRALiveExport"]
    assert source(export, "model") is record
    assert source(export, "new_id") is write
    assert export["widgets_values"][-1] == "lora"


def test_layers_combine_global_creative_and_masked_ip_with_plain_negative():
    g, source = _wiring("kilora-layers.json")
    ks = next(n for n in g["nodes"] if n["type"] == "KSampler")
    combine = source(ks, "positive")
    assert combine["type"] == "ConditioningCombine"
    layers = [source(combine, name) for name in ("conditioning_1", "conditioning_2")]
    assert all(n["type"] == "KiLoRALayerConditioning" for n in layers)
    by_layer = {n["widgets_values"][0]: n for n in layers}
    assert set(by_layer) == {"creative", "ip"}
    creative_mask = next(i for i in by_layer["creative"]["inputs"] if i["name"] == "mask")
    assert creative_mask["link"] is None
    mask = source(by_layer["ip"], "mask")
    assert mask["type"] == "MaskComposite"
    assert source(mask, "destination")["type"] == "SolidMask"
    assert source(mask, "source")["type"] == "SolidMask"
    for layer in layers:
        assert source(layer, "conditioning")["type"] == "CLIPTextEncode"
        assert not any(i["name"] == "compose" for i in layer["inputs"])
        assert layer["widgets_values"][-2:] == ["off", False]
    negative = source(ks, "negative")
    assert negative["type"] == "CLIPTextEncode"
    assert "nsfw, nude, naked" in negative["widgets_values"][0]
    apply = source(ks, "model")
    assert apply["type"] == "KiLoRAApply"
    creative_import = source(apply, "sets")
    ip_import = source(creative_import, "sets")
    assert creative_import["type"] == ip_import["type"] == "KiLoRAImportLoRA"
    assert creative_import["widgets_values"][-1] == "creative"
    assert ip_import["widgets_values"][-1] == "ip"
    save = next(n for n in g["nodes"] if n["type"] == "SaveImage")
    decode = source(save, "images")
    assert decode["type"] == "VAEDecode"
    assert source(decode, "samples") is ks


def test_learn_character_from_images_uses_a_mask():
    g, source = _wiring("kilora-learn-from-masked-images.json")
    solve = next(n for n in g["nodes"] if n["type"] == "KiLoRASolveFromImages")
    assert source(solve, "masks")["type"] == "BiRefNetRMBG"
    assert source(solve, "images")["type"] == "ImageBatch"
    note = next(n for n in g["nodes"] if n["type"] == "Note")
    text = note["widgets_values"][0]
    assert "ComfyUI-RMBG" in text
    assert "optional" in text and "installed separately" in text
    for name in ("rembg", "BiRefNet", "SAM"):
        assert name in text
