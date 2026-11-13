# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from kilora import editors, trainers
from kilora.editors import (
    Editor,
    EditorPlugin,
    EditorState,
    EditorSupport,
    check_mod,
    editor_for,
    register_editor,
    unregister_editor,
)
from kilora.errors import KiLoRAError
from kilora.trainers import TrainerPlugin, editor_main, register_trainer


def _editor_file(path, kind):
    manifest = {"format": "kilora-editor", "kind": kind, "training": {"state": "trained"}}
    save_file({"x": torch.zeros(1)}, str(path), metadata={"kilora.manifest": json.dumps(manifest)})
    return path


class _Fixed(Editor):
    def __init__(self, path):
        self.path = path

    def next_state(self, state, record):
        return None


def test_no_editor_is_built_in_and_an_installed_example_registers():
    assert editors.DEFAULT_PLUGINS == ()
    import kilora_editor_delta

    plugins = editors.available_editors()
    assert plugins["path-write"] is kilora_editor_delta.PLUGIN
    assert plugins["path-write"].inspect is not None


def test_register_and_load_by_kind(tmp_path):
    register_editor(EditorPlugin("test-fixed", _Fixed, title="Fixed"))
    try:
        ed = editor_for(_editor_file(tmp_path / "a.kedit.safetensors", "test-fixed"))
        assert isinstance(ed, _Fixed)
    finally:
        unregister_editor("test-fixed")


def test_later_registration_upgrades_a_kind(tmp_path):
    class _Newer(_Fixed):
        pass

    register_editor(EditorPlugin("test-up", _Fixed))
    register_editor(EditorPlugin("test-up", _Newer))
    try:
        assert type(editor_for(_editor_file(tmp_path / "b.kedit.safetensors", "test-up"))) is _Newer
    finally:
        unregister_editor("test-up")


def test_unknown_kind_is_refused(tmp_path):
    with pytest.raises(KiLoRAError):
        editor_for(_editor_file(tmp_path / "c.kedit.safetensors", "no-such-kind"))


def test_other_major_api_is_refused():
    with pytest.raises(KiLoRAError, match="API"):
        register_editor(EditorPlugin("test-old", _Fixed, api_version="0.9"))
    with pytest.raises(KiLoRAError, match="API"):
        register_trainer(TrainerPlugin("old", "editor", run=lambda a: 0, api_version="2.0"))


def test_env_variable_imports_plugin_modules(monkeypatch, tmp_path):
    (tmp_path / "my_kilora_plugin.py").write_text(
        "from kilora.editors import Editor, EditorPlugin, register_editor\n"
        "class E(Editor):\n"
        "    def __init__(self, path): pass\n"
        "register_editor(EditorPlugin('test-env', E))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(editors.PLUGINS_ENV, "my_kilora_plugin")
    monkeypatch.setattr(editors, "_DISCOVERED", False)
    try:
        assert "test-env" in editors.available_editors()
    finally:
        unregister_editor("test-env")


def test_support_declaration_round_trip_and_checks():
    s = EditorSupport(2, 8, 4, 64, ["a"])
    assert EditorSupport.from_manifest(s.to_manifest()) == s
    assert check_mod(s, {"a": 4}, {"a": 16}) == []
    problems = check_mod(s, {"a": 16, "b": 4}, {"a": 128})
    assert len(problems) == 3


def test_editor_state_protocol():
    class S:
        def state_tensors(self):
            return {}

        def load_tensors(self, tensors):
            pass

        def reset(self):
            pass

    assert isinstance(S(), EditorState)


def test_static_trainer_is_built_in():
    assert "static" in trainers.available_trainers("mod")


def test_editor_main_dispatch(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(trainers, "_PLUGINS", {})
    monkeypatch.setattr(trainers, "_DISCOVERED", True)
    assert editor_main([]) == 1
    assert "No editor trainer" in capsys.readouterr().err
    register_trainer(TrainerPlugin("one", "editor", run=lambda a: seen.append(a) or 0))
    assert editor_main(["train", "--x"]) == 0 and seen == [["train", "--x"]]
    register_trainer(TrainerPlugin("two", "editor", run=lambda a: seen.append(("two", a))))
    assert editor_main(["two", "eval"]) == 0 and seen[-1] == ("two", ["eval"])
    assert editor_main(["eval"]) == 2
    assert editor_main(["--list"]) == 0
    assert "one" in capsys.readouterr().out
