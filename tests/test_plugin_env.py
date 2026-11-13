# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
import types
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from kilora import editors, trainers

EDITORS = ["kilora_editor_chunk", "kilora_pullback", "kilora_editor_delta"]
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "kilora_editor_chunk": "plugins/kilora-editor-chunk/src",
    "kilora_pullback": "examples/editor-pullback/src",
    "kilora_editor_delta": "examples/editor-delta/src",
    "kilora_trainer_example": "examples/trainer-example/src",
}


def _load(monkeypatch, name):
    try:
        return importlib.import_module(name)
    except ImportError:
        monkeypatch.syspath_prepend(str(ROOT / SOURCES[name]))
        return importlib.import_module(name)


def _fresh(monkeypatch, module, env, points):
    monkeypatch.setattr(module, "_PLUGINS", {})
    if hasattr(module, "_SOURCES"):
        monkeypatch.setattr(module, "_SOURCES", {})
    if hasattr(module, "DEFAULT_PLUGINS"):
        monkeypatch.setattr(module, "DEFAULT_PLUGINS", ())
    monkeypatch.setattr(module, "_DISCOVERED", False)
    monkeypatch.setitem(sys.modules, "kilora._site", types.ModuleType("kilora._site"))
    monkeypatch.setattr(module, "_entry_points", lambda group: list(points))
    monkeypatch.setenv(editors.PLUGINS_ENV, env)


@pytest.mark.parametrize("name", EDITORS)
@pytest.mark.parametrize("form", ["module", "module:attribute"])
def test_env_variable_registers_the_editor(monkeypatch, name, form):
    mod = _load(monkeypatch, name)
    env = name if form == "module" else f"{name}:PLUGIN"
    _fresh(monkeypatch, editors, f" {env}, ", [])
    assert editors.plugin_for_kind(mod.PLUGIN.kind) is mod.PLUGIN
    assert editors._SOURCES[mod.PLUGIN.kind].startswith(editors.PLUGINS_ENV)


@pytest.mark.parametrize("name", EDITORS)
def test_entry_point_registers_the_editor(monkeypatch, name):
    mod = _load(monkeypatch, name)
    ep = EntryPoint(mod.PLUGIN.kind, f"{name}:PLUGIN", editors.ENTRY_POINT_GROUP)
    _fresh(monkeypatch, editors, "", [ep])
    assert editors.plugin_for_kind(mod.PLUGIN.kind) is mod.PLUGIN


def test_one_variable_lists_editors_and_trainers(monkeypatch):
    pullback = _load(monkeypatch, "kilora_pullback")
    example = _load(monkeypatch, "kilora_trainer_example")
    env = "kilora_pullback,kilora_trainer_example:TRAINERS"
    _fresh(monkeypatch, editors, env, [])
    _fresh(monkeypatch, trainers, env, [])
    assert set(editors.available_editors()) == {pullback.PLUGIN.kind}
    got = trainers.available_trainers()
    assert {p.name for p in example.TRAINERS} <= set(got)


@pytest.mark.parametrize("form", ["module", "module:attribute"])
def test_env_variable_registers_the_trainers(monkeypatch, form):
    example = _load(monkeypatch, "kilora_trainer_example")
    env = "kilora_trainer_example" + ("" if form == "module" else ":TRAINERS")
    _fresh(monkeypatch, trainers, env, [])
    got = trainers.available_trainers()
    for p in example.TRAINERS:
        assert got[p.name] is p


def test_entry_point_registers_the_trainers(monkeypatch):
    example = _load(monkeypatch, "kilora_trainer_example")
    ep = EntryPoint("example", "kilora_trainer_example:TRAINERS", trainers.ENTRY_POINT_GROUP)
    _fresh(monkeypatch, trainers, "", [ep])
    got = trainers.available_trainers()
    for p in example.TRAINERS:
        assert got[p.name] is p


def test_a_missing_attribute_warns(monkeypatch):
    _load(monkeypatch, "kilora_pullback")
    _fresh(monkeypatch, editors, "kilora_pullback:NOPE", [])
    with pytest.warns(UserWarning, match="no attribute NOPE"):
        editors.available_editors()


def test_a_module_that_registers_itself_still_works(monkeypatch, tmp_path):
    (tmp_path / "self_reg_plugin.py").write_text(
        "from kilora.editors import Editor, EditorPlugin, register_editor\n"
        "class E(Editor):\n"
        "    def __init__(self, path): pass\n"
        "register_editor(EditorPlugin('self-reg', E), 'import')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _fresh(monkeypatch, editors, "self_reg_plugin", [])
    assert "self-reg" in editors.available_editors()
    assert editors._SOURCES["self-reg"] == "import"
