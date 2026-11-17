# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora import KiLoRAError
from kilora.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from kilora.comfyui import nodes as N


def test_every_input_has_a_tooltip_and_every_node_a_description(folders):
    for key, cls in NODE_CLASS_MAPPINGS.items():
        assert cls.DESCRIPTION and cls.CATEGORY.startswith("KiLoRA"), key
        for group_name, group in cls.INPUT_TYPES().items():
            if group_name == "hidden":
                continue
            for name, spec in group.items():
                assert len(spec) > 1 and spec[1].get("tooltip"), f"{key}.{name}"


def test_display_names_and_categories(folders):
    for key in NODE_CLASS_MAPPINGS:
        name = NODE_DISPLAY_NAME_MAPPINGS[key]
        assert name.startswith("KiLoRA ")
    live = [k for k, c in NODE_CLASS_MAPPINGS.items() if c.CATEGORY == "KiLoRA/live mode"]
    assert {"KiLoRALiveRecord", "KiLoRALiveWrite"} <= set(live)
    assert all(NODE_DISPLAY_NAME_MAPPINGS[k].startswith("KiLoRA Live: ") for k in live)
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRALiveRecord"].startswith("KiLoRA Live: 1.")
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRALiveWrite"].startswith("KiLoRA Live: 2.")


def test_no_node_is_hidden(folders):
    for cls in NODE_CLASS_MAPPINGS.values():
        assert not hasattr(cls, "DEPRECATED") and not hasattr(cls, "DEV_ONLY")


def test_input_names_and_order_are_unchanged(folders):
    assert list(N.KiLoRALoadMod.INPUT_TYPES()["required"]) == ["model", "file", "strength"]
    assert list(N.KiLoRASelect.INPUT_TYPES()["required"]) == ["file", "strength", "prior"]
    assert list(N.KiLoRAApply.INPUT_TYPES()["required"]) == ["model", "sets", "depth"]
    assert list(N.KiLoRALiveRecord.INPUT_TYPES()["required"]) == ["model", "editor"]
    assert list(N.KiLoRALiveWrite.INPUT_TYPES()["required"]) == ["path", "samples"]
    assert list(N.KiLoRASuggestions.INPUT_TYPES()["optional"]) == [
        "model_path",
        "suggest_by",
    ]
    assert list(N.KiLoRASimilarLoRAs.INPUT_TYPES()["optional"]) == ["similar_space", "model_path"]
    assert N.KiLoRALoadMod.INPUT_TYPES()["required"]["strength"][1]["default"] == 1.0


def test_missing_editor_message(denoiser, folders):
    with pytest.raises(KiLoRAError, match="No live-mode editor found"):
        N.KiLoRALiveRecord().record(FakePatcher(FakeBaseModel(denoiser)), "")


def test_editor_mismatch_messages(denoiser, folders, tmp_path):
    from kilora_editor_delta import check_editor
    from test_path_write import _setup

    from kilora.live import SESSIONS

    SESSIONS.clear()
    try:
        _, _, _, session, editor, _ = _setup(denoiser, folders, tmp_path)
        editor.manifest["base_models"] = [{"family": "sdxl"}]
        with pytest.raises(KiLoRAError, match="Wrong base model"):
            check_editor(editor, session)
        editor.manifest["base_models"] = [session.mod.manifest["base_model"]]
        first = next(iter(editor.v0))
        del editor.v0[first]
        with pytest.raises(KiLoRAError, match="does not fit this mod"):
            check_editor(editor, session)
    finally:
        SESSIONS.clear()


def test_palette_optional_input_order_preserves_saved_workflows(folders):
    optional = N.KiLoRAPalette.INPUT_TYPES()["optional"]
    assert list(optional) == [
        "sets",
        "mask_a",
        "mask_b",
        "mode",
        "reset_state",
        "reward",
        "embedding_model_path",
    ]
