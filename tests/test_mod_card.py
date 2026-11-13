# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from conftest import randomize_up
from safetensors import safe_open

from kilora import KiLoRAConfig, attach
from kilora.card import CARD_KEY, card_fields, card_markdown, embed_card, read_card
from kilora.cli import main
from kilora.modfile import load_mod, read_manifest

TARGET = "layers.*.attention.to_out.0"


@pytest.fixture
def mod(model, tmp_path):
    adapter = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=2, depth=2))
    randomize_up(adapter)
    path = tmp_path / "ink.kmod"
    adapter.save(
        path,
        id="ink",
        name="Ink",
        license="CC-BY-4.0",
        trigger_words=["in ink style"],
        strength={"recommended": 0.8, "min": -1.0, "max": 1.5},
        base_model={"family": "z-image"},
        data_source={
            "name": "ink",
            "images": 24,
            "generators": ["openai gpt-image"],
            "license": "CC0-1.0",
            "note": "Generated images.",
        },
    )
    adapter.remove()
    return path


def test_fields_and_markdown(mod):
    f = card_fields(read_manifest(mod))
    assert f["trigger"] == ["in ink style"] and f["strength"]["recommended"] == 0.8
    assert f["hooks"] == {"count": 2, "patterns": [TARGET]}
    assert f["base_model"]["family"] == "z-image" and f["license"] == "CC-BY-4.0"
    assert f["source_line"] == (
        "24 images from ink, made with openai gpt-image, data license CC0-1.0. Generated images."
    )
    md = card_markdown(f)
    assert "| Trigger | `in ink style` |" in md and "| Recommended strength | 0.8" in md
    assert "| Hooks | 2 layers: layers.*.attention.to_out.0 |" in md


def test_embedded_card_keeps_the_mod_and_its_digest(mod):
    before = load_mod(mod)
    fields = embed_card(mod)
    after = load_mod(mod)
    assert after.manifest["digest"] == before.manifest["digest"]
    with safe_open(str(mod), framework="pt") as f:
        assert json.loads(f.metadata()[CARD_KEY]) == fields
    assert read_card(mod) == fields
    first = mod.read_bytes()
    embed_card(mod)
    assert mod.read_bytes() == first


def test_missing_source_gives_a_plain_line(model, tmp_path):
    adapter = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=2))
    adapter.save(tmp_path / "a.kmod", id="a")
    adapter.remove()
    assert read_card(tmp_path / "a.kmod")["source_line"] == "The training data is not described."


def test_cli_short_card_and_embed(mod, tmp_path):
    out = tmp_path / "card.md"
    assert main(["card", str(mod), "--short", "--embed", "--output", str(out)]) == 0
    assert out.read_text().startswith("# Ink")
    with safe_open(str(mod), framework="pt") as f:
        assert CARD_KEY in f.metadata()
    assert main(["card", str(mod), "--output", str(tmp_path / "README.md")]) == 0
    assert "| Training data | 24 images from ink" in (tmp_path / "README.md").read_text()
