# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import sys
import types

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import folders

from kilora.comfyui import nodes as N
from kilora.comfyui.patching import KiLoRAError, ModChain, ModEntry
from kilora.library import bake, contrast


class FakeClip:
    def __init__(self):
        self.texts = []

    def tokenize(self, value):
        return {"t5xxl": value}

    def encode_from_tokens_scheduled(self, tokens):
        value = tokens["t5xxl"]
        self.texts.append(value)
        codes = torch.tensor([float(ord(c)) for c in value] or [0.0])
        return [[codes.reshape(1, -1, 1), {"pooled_output": codes.sum().reshape(1, 1)}]]


def _lora(folder, name, triggers=()):
    weights = {
        "lora_unet_block.lora_down.weight": torch.ones(2, 4),
        "lora_unet_block.lora_up.weight": torch.ones(4, 2),
    }
    metadata = {"modelspec.license": "Apache-2.0"}
    if triggers:
        metadata["ss_tag_frequency"] = json.dumps(
            {"a": {w: 10 - i for i, w in enumerate(triggers)}}
        )
    path = folder / name
    save_file(weights, str(path), metadata=metadata)
    return path


def _bake(source, model_sha="toy-base"):
    _, source_sha, _, key, _ = bake._request(
        source, model_sha, contrast.Settings(), bake.DEFAULT_PROMPTS
    )
    path = bake._default_path() / "bakes" / f"{key}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"lora_unet_block.lora_down.weight": torch.zeros(2, 4)},
        str(path),
        metadata={
            "kilora_bake_key": key,
            "kilora_bake_source_sha": source_sha,
            "kilora_bake_model_sha": model_sha,
        },
    )
    return path


@pytest.fixture
def library(folders, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "library"))
    loras = folders["loras"]
    files = {
        "sketch": _lora(loras, "Sketch.safetensors", ("inkcat", "linework")),
        "cat": _lora(loras, "cat.safetensors", ("whiskers",)),
        "plain": _lora(loras, "plain.safetensors"),
    }
    return files


def _encode(value, sets=None, layer="global"):
    clip = FakeClip()
    out = N.KiLoRAMentionPrompt().encode(clip, value, sets=sets, layer=layer)
    return clip, out


def test_mentions_parse_with_strengths_and_spans():
    value = "@sketch a cat, @Cat:0.5 by the window"
    mentions = N.parse_mentions(value)
    assert [(m.name, m.strength) for m in mentions] == [("sketch", 0.7), ("Cat", 0.5)]
    assert [value[slice(*m.span)] for m in mentions] == ["@sketch", "@Cat:0.5"]
    assert N.strip_mentions(value, mentions) == "a cat, by the window"


@pytest.mark.parametrize("value", ["mail me at me@sketch.com", "a@b", "x@@sketch", "word@cat:0.5"])
def test_emails_are_not_mentions(value):
    assert N.parse_mentions(value) == []


def test_unknown_name_names_it_and_the_closest_stems(library):
    with pytest.raises(KiLoRAError) as error:
        _encode("a cat, @sketchy")
    assert "@sketchy" in str(error.value)
    assert "Sketch" in str(error.value)


def test_baked_file_keeps_mentions_and_triggers_out_of_the_encoder(library):
    baked = _bake(library["sketch"])
    clip, (conditioning, chain, report) = _encode("@sketch:0.5 a cat on a street")
    assert clip.texts == ["a cat on a street"]
    assert all(w not in clip.texts[0] for w in ("inkcat", "linework", "@"))
    entries = list(chain)
    assert [(e.kind, e.path, e.strength, e.axis) for e in entries] == [
        ("lora", str(baked), 0.5, "auto")
    ]
    assert "baked" in report
    record = conditioning[0][1]["kilora_mentions"][0]
    assert record["baked"] is True
    assert record["path"] == str(library["sketch"])
    assert record["trigger_words"] == ["inkcat", "linework"]


def test_missing_bake_falls_back_to_source_and_triggers(library):
    clip, (conditioning, chain, report) = _encode("a cat on a street, @Sketch")
    assert clip.texts == ["inkcat, linework, a cat on a street"]
    entries = list(chain)
    assert [(e.path, e.strength) for e in entries] == [(str(library["sketch"]), 0.7)]
    assert "inkcat, linework" in report
    assert "no baked file" in report
    assert conditioning[0][1]["kilora_mentions"][0]["baked"] is False


def test_chain_entries_mix_baked_and_source_files(library):
    baked = _bake(library["cat"])
    previous = ModChain().add(ModEntry("lora", str(library["plain"]), 1.0))
    _, (_, chain, _) = _encode("@sketch:1.2 and @cat:0.25 together", sets=previous)
    entries = list(chain)
    assert [(e.path, e.strength) for e in entries] == [
        (str(library["plain"]), 1.0),
        (str(library["sketch"]), 1.2),
        (str(baked), 0.25),
    ]


@pytest.mark.parametrize("layer", ["global", *N.AXES])
def test_layer_sets_the_entry_axis(library, layer):
    _, (conditioning, chain, _) = _encode("@cat a portrait", layer=layer)
    assert [e.axis for e in chain] == ["auto" if layer == "global" else layer]
    assert conditioning[0][1]["kilora_mention_layer"] == layer


def test_metadata_holds_spans_of_the_user_text(library):
    value = "a portrait of @cat:0.9, ink by @plain"
    _, (conditioning, _, _) = _encode(value)
    records = conditioning[0][1]["kilora_mentions"]
    assert [value[slice(*r["span"])] for r in records] == ["@cat:0.9", "@plain"]
    assert [r["name"] for r in records] == ["cat", "plain"]
    assert [r["strength"] for r in records] == [0.9, 0.7]


def test_text_without_mentions_matches_clip_text_encode(library):
    value = "a cat at me@example.com, by the window"
    reference = FakeClip()
    expected = reference.encode_from_tokens_scheduled(reference.tokenize(value))
    previous = ModChain().add(ModEntry("lora", str(library["plain"]), 1.0))
    clip, (conditioning, chain, report) = _encode(value, sets=previous)
    assert clip.texts == [value]
    assert chain is previous
    assert report == ""
    assert len(conditioning) == len(expected)
    for (value_a, meta_a), (value_b, meta_b) in zip(conditioning, expected, strict=True):
        assert torch.equal(value_a, value_b)
        assert set(meta_a) == set(meta_b)
        assert all(torch.equal(meta_a[k], meta_b[k]) for k in meta_a)


def _hooks(monkeypatch):
    hooks = types.ModuleType("comfy.hooks")
    hooks.set_mask_for_conditioning = lambda conds, mask, area, strength: [
        [c, dict(meta)] for c, meta in conds
    ]
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)


def test_layer_node_keeps_mentions_and_warns_on_another_layer(library, monkeypatch, caplog):
    _hooks(monkeypatch)
    _, (conditioning, _, _) = _encode("@cat a portrait", layer="ip")
    node = N.KiLoRALayerConditioning()
    with caplog.at_level(logging.WARNING, logger="kilora"):
        same = node.layer(conditioning, "ip")[0]
    assert not caplog.records
    assert same[0][1]["kilora_mentions"] == conditioning[0][1]["kilora_mentions"]
    with caplog.at_level(logging.WARNING, logger="kilora"):
        node.layer(conditioning, "creative")
    assert len(caplog.records) == 1
    assert "creative" in caplog.records[0].getMessage()
    assert "ip" in caplog.records[0].getMessage()


def test_layer_node_without_mentions_does_not_warn(monkeypatch, caplog):
    _hooks(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="kilora"):
        N.KiLoRALayerConditioning().layer([[torch.ones(1), {}]], "ip")
    assert not caplog.records


def _solved(folder, filename, concept, mode="single", cue="inkcue"):
    result = contrast.ContrastResult(
        {
            "transformer_blocks.0.attn1.to_q": contrast.MountSolution(
                torch.ones(2, 4), torch.ones(2, 4), 0.1, 1.0
            )
        },
        {
            "concept": concept,
            "pairs": [{"target": "a cat, ink", "student": f"a cat, {cue}", "cue": cue}],
            "prompt_solve": {"mode": mode, "version": 1},
        },
    )
    path = folder / "kilora-solved" / filename
    path.parent.mkdir(exist_ok=True)
    contrast.write_lora(path, result)
    return path


@pytest.fixture
def solved_library(library, folders, monkeypatch):
    loras = folders["loras"]
    files = {
        "ink": _solved(loras, "prompt-solve-ink.safetensors", "ink"),
        "joint": _solved(loras, "prompt-solve.safetensors", "joint", mode="joint"),
        "cat": _solved(loras, "prompt-solve-cat.safetensors", "cat"),
    }

    def listing(folder, extension=None):
        names = sorted(p.relative_to(loras).as_posix() for p in loras.rglob("*") if p.is_file())
        return [n for n in names if extension is None or n.endswith(extension)]

    monkeypatch.setattr(N.FOLDERS, "list", listing)
    return files


def test_solved_concept_file_answers_to_its_label(solved_library):
    clip, (conditioning, chain, report) = _encode("@ink a cat on a street")
    assert [(e.path, e.strength) for e in chain] == [(str(solved_library["ink"]), 0.7)]
    assert clip.texts == ["inkcue, a cat on a street"]
    record = conditioning[0][1]["kilora_mentions"][0]
    assert record["name"] == "prompt-solve-ink"
    assert record["trigger_words"] == ["inkcue"]
    _, (_, chain, _) = _encode("@prompt-solve-ink a cat")
    assert [e.path for e in chain] == [str(solved_library["ink"])]


def test_file_stem_comes_before_a_solved_label(solved_library, library):
    _, (_, chain, _) = _encode("@cat a cat")
    assert [e.path for e in chain] == [str(library["cat"])]


def test_joint_files_have_no_mention_name(solved_library):
    with pytest.raises(KiLoRAError) as error:
        _encode("@joint a cat")
    assert "@joint" in str(error.value)


def test_two_solved_files_with_one_label_are_ambiguous(solved_library, folders):
    _solved(folders["loras"], "other-ink.safetensors", "ink")
    with pytest.raises(KiLoRAError) as error:
        _encode("@ink a cat")
    assert "other-ink" in str(error.value) and "prompt-solve-ink" in str(error.value)
