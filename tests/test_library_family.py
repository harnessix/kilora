# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch
from safetensors.torch import save_file

from kilora.library import LibraryIndex
from kilora.library.index import _family, _metadata
from kilora.presets import PRESETS


def _keys(stems):
    return [f"{stem}.lora_{part}.weight" for stem in stems for part in ("down", "up")]


@pytest.mark.parametrize("form", ["kohya", "diffusers", "comfy"])
@pytest.mark.parametrize(
    ("family", "paths"),
    [
        ("z-image", ["layers.*.feed_forward.w1"]),
        ("moss-image", ["layers.*.proj.ffn_up_gate"]),
        ("qwen-image", ["transformer_blocks.*.img_mlp.net.0.proj"]),
        ("wan", ["blocks.*.self_attn.q"]),
        ("anima", ["transformer_blocks.*.attn1.to_q", "transformer_blocks.*.ff.net.2"]),
        ("ace-step15", ["decoder.layers.*.mlp.gate_proj"]),
        ("flux3", ["content_mode_blocks.image.*.mlp_in"]),
    ],
)
def test_family_from_preset_names(family, paths, form):
    preset = PRESETS[family]
    import fnmatch

    assert all(
        any(fnmatch.fnmatchcase(p, pattern) for pattern in preset.full_target_modules)
        for p in paths
    )
    stems = [path.replace("*", "7") for path in paths]
    if family == "anima" and form != "diffusers":
        stems = [preset.rename(stem) for stem in stems]
    if form == "kohya":
        stems = ["lora_unet_" + stem.replace(".", "_") for stem in stems]
    elif form == "diffusers":
        stems = ["transformer." + stem for stem in stems]
    else:
        stems = ["model.diffusion_model." + stem for stem in stems]
    expected = "unknown" if family == "anima" and form == "diffusers" else family
    assert _family(_keys(stems)) == expected


def test_exact_ties_and_matching_key_share():
    assert (
        _family(_keys(["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"])) == "unknown"
    )
    assert _family(_keys(["double_blocks.0.img_attn.qkv"])) == "unknown"
    assert _family(_keys(["transformer_blocks.0.attn1.to_q"])) == "unknown"
    keys = _keys(["layers.0.feed_forward.w1", "layers.0.proj.ffn_up_gate"])
    keys.append("layers.0.feed_forward.w1.alpha")
    assert _family(keys) == "z-image"
    assert _family(_keys(["no.such.layer"])) == "unknown"


def test_functional_words_from_preset_name_and_triggers(tmp_path, monkeypatch):
    from dataclasses import replace

    key = {"layers.0.feed_forward.w1.lora_down.weight": torch.ones(1, 1)}
    captions = tmp_path / "style.safetensors"
    save_file(key, str(captions), metadata={"ss_tag_frequency": '{"set": {"special": 3}}'})
    trigger = tmp_path / "other.safetensors"
    save_file(key, str(trigger), metadata={"modelspec.trigger_phrase": "special"})
    named = tmp_path / "special-fix.safetensors"
    save_file(key, str(named))
    monkeypatch.setitem(
        PRESETS, "z-image", replace(PRESETS["z-image"], functional_words=("special",))
    )
    assert _metadata(captions)["axis"] == "creative"
    assert _metadata(trigger)["axis"] == "functional"
    assert _metadata(named)["axis"] == "functional"


def test_sync_500_files_without_hashing(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    tensor = torch.ones(1, 1)
    for i in range(500):
        save_file(
            {"layers.0.feed_forward.w1.lora_down.weight": tensor}, str(folder / f"{i}.safetensors")
        )
    index = LibraryIndex.open(tmp_path / "index")
    monkeypatch.setattr(index.hash_index, "hash", lambda path: path.stem)
    start = time.monotonic()
    report = index.sync([folder])
    elapsed = time.monotonic() - start
    assert (report.added, report.skipped) == (500, 0)
    assert elapsed < 10, f"sync took {elapsed:.2f}s"


@pytest.mark.parametrize("family", sorted(name for name in PRESETS if name != "qwen-text"))
def test_all_preset_targets_match(family):
    from kilora.families import FAMILIES
    from kilora.library.index import _FAMILY_PATTERNS

    preset = PRESETS[family]
    paths = [p.replace("*", "7") for p in preset.full_target_modules]
    if family in ("sd15", "sdxl"):
        paths = [
            p.replace("7.transformer_blocks", "down_blocks.1.attentions.0.transformer_blocks", 1)
            for p in paths
        ]
    if family == "flux3":
        paths = [p.replace("content_mode_blocks.7.", "content_mode_blocks.image.") for p in paths]
    shapes = {path: (4, 4) for path in paths}
    spec = FAMILIES[family]
    aliases = [alias for alias, (target, _) in spec.lora_names(shapes).items() if target in shapes]
    aliases.extend(preset.aliases(paths))
    for stem in {*paths, *aliases}:
        for form in (stem, stem.replace(".", "_")):
            assert _FAMILY_PATTERNS[family].fullmatch(form), (family, form)
