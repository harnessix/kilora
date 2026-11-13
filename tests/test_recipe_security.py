# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import builtins

import pytest
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_modfile import rewrite_manifest
from test_sections import _logic_meta, _reference_meta

from kilora import Settings, load_mod, save_mod
from kilora.comfyui import nodes as N
from kilora.errors import ModFormatError
from kilora.sections import Section


def crafted(tmp_path, kind, meta, change):
    original = tmp_path / "original.kmod"
    result = tmp_path / "crafted.kmod"
    save_mod(original, [], Settings(), sections=[Section(kind, "entry", {}, meta)])
    rewrite_manifest(original, result, change)
    return result


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__",
        "__import__('builtins').setattr(__import__('builtins'), '_kilora_recipe_executed', True)",
        "(lambda: 1)()",
    ],
)
def test_schedule_expressions_are_not_executed(tmp_path, monkeypatch, expression):
    monkeypatch.setattr(builtins, "_kilora_recipe_executed", False, raising=False)
    path = crafted(
        tmp_path,
        "logic",
        _logic_meta(),
        lambda m: m["sections"][0]["meta"]["settings"].update(depth_schedule=expression),
    )
    with pytest.raises(ModFormatError, match="depth_schedule"):
        load_mod(path)
    assert builtins._kilora_recipe_executed is False


@pytest.mark.parametrize(
    "key,value",
    [("unknown_setting", 1), ("route_freeze", 2.0), ("depth", 99)],
)
def test_bad_logic_settings_are_refused(tmp_path, key, value):
    path = crafted(
        tmp_path,
        "logic",
        _logic_meta(),
        lambda m: m["sections"][0]["meta"]["settings"].update({key: value}),
    )
    with pytest.raises(ModFormatError, match=key):
        load_mod(path)


@pytest.mark.parametrize("key,value", [("sha256", "z" * 64), ("region", "subject")])
def test_bad_reference_is_refused(tmp_path, key, value):
    def change(manifest):
        meta = manifest["sections"][0]["meta"]
        (meta["transforms"] if key == "region" else meta)[key] = value

    path = crafted(tmp_path, "weight-ref", _reference_meta(), change)
    with pytest.raises(ModFormatError, match=key):
        load_mod(path)


def test_unknown_section_is_refused(tmp_path):
    path = crafted(
        tmp_path,
        "logic",
        _logic_meta(),
        lambda m: m["sections"][0].update(kind="unknown-kind"),
    )
    with pytest.raises(ModFormatError, match="unknown-kind"):
        load_mod(path)


def test_load_mod_node_refuses_bad_logic(denoiser, folders):
    path = crafted(
        folders["kilora"],
        "logic",
        _logic_meta(),
        lambda m: m["sections"][0]["meta"]["settings"].update(route_freeze=2.0),
    )
    with pytest.raises(ModFormatError, match="route_freeze"):
        N.KiLoRALoadMod().load(FakePatcher(FakeBaseModel(denoiser)), path.name, 1.0)
