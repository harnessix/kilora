# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import re
from pathlib import Path

import pytest
from test_comfyui import denoiser, folders

from kilora import i18n
from kilora.comfyui import help as help_texts
from kilora.comfyui.nodes import FOLDERS, NODE_CLASS_MAPPINGS

ROOT = Path(__file__).resolve().parents[1]
I18N = Path(i18n.__file__).parent


def _fields(s):
    return sorted(re.findall(r"\{(\w+)\}", s))


def _raw(lang):
    with open(I18N / f"{lang}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_every_locale_file_has_every_key(lang):
    en = _raw("en")
    texts = _raw(lang)
    assert set(texts) == set(en)
    for key, value in texts.items():
        assert value.strip(), key
        assert _fields(value) == _fields(en[key]), key


def test_locale_list_matches_files():
    files = {p.stem for p in I18N.glob("*.json")}
    assert files == set(i18n.LOCALES)
    assert set(i18n.MACHINE_MADE) == set(i18n.LOCALES) - {"en"}


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_shipped_node_defs_are_current(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    with open(ROOT / "locales" / lang / "nodeDefs.json", encoding="utf-8") as f:
        shipped = json.load(f)
    assert shipped == help_texts.node_defs(lang, NODE_CLASS_MAPPINGS), (
        "run: python -m kilora.comfyui.help ."
    )


def test_node_defs_cover_every_node_input_and_output(monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    defs = help_texts.node_defs("de", NODE_CLASS_MAPPINGS)
    assert set(defs) == set(NODE_CLASS_MAPPINGS)
    for key, cls in NODE_CLASS_MAPPINGS.items():
        names = {n for group in cls.INPUT_TYPES().values() for n in group}
        assert set(defs[key]["inputs"]) == names
        if getattr(cls, "OUTPUT_TOOLTIPS", None):
            assert len(defs[key]["outputs"]) == len(cls.RETURN_TYPES)


def test_messages_follow_kilora_lang(monkeypatch):
    monkeypatch.setenv("KILORA_LANG", "ja")
    ja = i18n.text("err.strength", low=-2, high=2)
    assert "-2" in ja and ja != i18n.text("err.strength", lang="en", low=-2, high=2)
    monkeypatch.setenv("KILORA_LANG", "xx")
    assert i18n.text("err.strength", low=0, high=1) == "Strength must be a number from 0 to 1."


def test_every_product_node_input_has_its_own_strings(monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    for key, cls in NODE_CLASS_MAPPINGS.items():
        names = {n for group in cls.INPUT_TYPES().values() for n in group}
        assert names == set(help_texts.SPEC[key][1]), key


def test_keep_previous_input(denoiser, folders):
    from live_helpers import build_live_mod
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    session = patching.sessions_of(patched)[0]
    types = N.KiLoRALiveOptions.INPUT_TYPES()
    assert list(types["required"]) == ["model", "mod", "learning", "speed"]
    assert types["optional"]["keep_previous"][1]["default"] == "unchanged"
    N.KiLoRALiveOptions().set(patched, session.mod_id, True, -1.0, keep_previous="on")
    assert session.keep_previous
    N.KiLoRALiveOptions().set(patched, session.mod_id, True, -1.0)
    assert session.keep_previous
    N.KiLoRALiveOptions().set(patched, session.mod_id, True, -1.0, keep_previous="off")
    assert not session.keep_previous


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_gate_stats_help(lang):
    tip = _raw(lang)["tip.gate_stats"]
    assert "fused" in tip and "faster" in tip
    assert "torch" in tip and "before this release" in tip
    assert "KILORA_GATE_STATS" in tip


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_imported_live_help_is_shipped_in_every_locale(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    texts = _raw(lang)
    defs = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)
    options = defs["KiLoRALiveOptions"]["inputs"]
    assert options["edits"]["tooltip"] == texts["tip.import_edits"]
    assert options["controller"]["tooltip"] == texts["tip.import_controller"]
    assert "off" in options["edits"]["tooltip"]
    assert "unchanged" in options["edits"]["tooltip"]
    assert "r1" in options["controller"]["tooltip"]
    assert "rprop" in options["controller"]["tooltip"]
    record = defs["KiLoRALiveRecord"]
    assert record["inputs"]["editor"]["tooltip"] == texts["tip.import_editor"]
    assert record["description"] == texts["node.KiLoRALiveRecord.desc"]


def test_english_import_help_explains_edit_scope_and_source_safety():
    texts = _raw("en")
    tip = texts["tip.import_edits"]
    for phrase in (
        "core matrix",
        "strength per layer",
        "pack terms",
        "Source files are never written",
        "off restores the imported LoRAs exactly",
    ):
        assert phrase in tip
    assert "without a live mod" in texts["node.KiLoRALiveRecord.desc"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_layer_compose_help_is_shipped_in_every_locale(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    texts = _raw(lang)
    inputs = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)["KiLoRALayerConditioning"]["inputs"]
    assert inputs["compose"]["name"] == texts["in.compose"]
    assert inputs["compose"]["tooltip"] == texts["tip.compose"]
    assert "single" in texts["tip.compose"] and "crop" in texts["tip.compose"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_size_norm_has_strings(lang):
    texts = _raw(lang)
    for key in ("in.size_norm", "tip.size_norm", "tip.apply_size_norm"):
        assert texts[key].strip(), key
    assert help_texts.SPEC["KiLoRAImportLoRA"][1]["size_norm"] == ("in.size_norm", "tip.size_norm")
    assert help_texts.SPEC["KiLoRAApply"][1]["size_norm"] == ("in.size_norm", "tip.apply_size_norm")


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_depth_step_inputs_have_strings_in_every_locale(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    texts = _raw(lang)
    defs = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)["KiLoRAApply"]["inputs"]
    for name in ("radial_step", "orthogonal_step", "radial_budget"):
        assert texts[f"in.{name}"].strip() and texts[f"tip.{name}"].strip()
        assert defs[name] == {"name": texts[f"in.{name}"], "tooltip": texts[f"tip.{name}"]}
    assert "0.5" in texts["tip.radial_step"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_prompt_solve_help_covers_nodes_and_file_location(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    definitions = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)
    assert "KiLoRAPromptSet" in definitions
    solve = definitions["KiLoRASolveFromPrompts"]
    assert "models/loras/kilora-solved/" in solve["description"]
    assert "0.7" in solve["outputs"]["1"]["tooltip"]
    assert "single" in solve["inputs"]["mode"]["tooltip"]
    assert "joint" in solve["inputs"]["mode"]["tooltip"]
    for name, auto in (("steps", "0"), ("size", "0"), ("cfg", "-1")):
        assert f"{auto} uses the model family's default" in solve["inputs"][name]["tooltip"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_pack_help(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    definitions = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)
    assert definitions["KiLoRAPack"]["inputs"]["copy_mods"]["tooltip"]
    assert definitions["KiLoRASolveFromPrompts"]["inputs"]["output"]["tooltip"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_build_editor_pack_strings(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    definitions = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)
    node = definitions["KiLoRABuildEditorPack"]
    assert node["display_name"] and node["description"]
    assert set(node["inputs"]) == {
        "model",
        "clip",
        "name",
        "prompts",
        "base_family",
        "draws",
        "common",
        "family",
        "loras",
        "orthogonalize",
        "size",
        "steps",
        "seed",
    }
    assert all(entry["tooltip"] for entry in node["inputs"].values())
    assert set(node["outputs"]) == {"0", "1"}
    texts = i18n.catalog(lang)
    for key in (
        "err.pack_plugin_missing",
        "err.pack_prompts",
        "err.pack_name",
        "pack_build.report",
        "pack_build.events",
        "pack_build.no_estimate",
        "pack_build.found",
    ):
        assert texts[key]
    assert "{plugin}" in texts["err.pack_plugin_missing"]


@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_image_solve_help_covers_inputs_and_rights(lang, monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    node = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)["KiLoRASolveFromImages"]
    assert "models/loras/kilora-solved/" in node["description"]
    assert set(node["inputs"]) == {
        "model",
        "clip",
        "vae",
        "name",
        "cue",
        "rank",
        "steps",
        "cfg",
        "size",
        "seeds",
        "ignore_guard",
        "overwrite",
        "images",
        "captions",
        "folder",
    }
    assert all(value["tooltip"] for value in node["inputs"].values())
    assert node["outputs"]["2"]["tooltip"]
    if lang == "en":
        assert "responsible for the rights" in node["description"]
        assert "responsible for the rights" in node["inputs"]["images"]["tooltip"]
