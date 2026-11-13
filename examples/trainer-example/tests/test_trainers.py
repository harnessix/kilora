# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import kilora_trainer_example as ex
import pytest

from kilora import load_mod, trainers
from kilora.check import check_file
from kilora.trainers import editor_main, register_trainer


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(trainers, "_PLUGINS", {})
    monkeypatch.setattr(trainers, "_DISCOVERED", True)
    for plugin in ex.TRAINERS:
        register_trainer(plugin, "test")


def test_static_trainer_writes_a_mod_that_learned(tmp_path, registered):
    out = tmp_path / "static.kmod.safetensors"
    plugin = trainers.get_trainer("example-static", "mod")
    assert plugin.run(["--out", str(out), "--steps", "40"]) == 0
    mod = load_mod(out)
    assert mod.manifest["trained_with"]["tool"] == "kilora-trainer-example"
    report = json.loads((tmp_path / "static.kmod.safetensors.report.json").read_text())
    assert report["last_loss"] < report["first_loss"]
    assert check_file(out).ok


def test_editor_trainer_through_kilora_train_editor(tmp_path, registered):
    out = tmp_path / "pullback.kedit.safetensors"
    assert editor_main(["pullback-rate", "--out", str(out), "--episodes", "2"]) == 0
    report = json.loads((tmp_path / "pullback.kedit.safetensors.report.json").read_text())
    assert 0.0 < report["rate"] <= 0.3
    assert check_file(out).kind == "editor (pullback)"


def test_editor_trainer_is_deterministic(tmp_path, registered):
    a, b = tmp_path / "a.kedit.safetensors", tmp_path / "b.kedit.safetensors"
    for out in (a, b):
        editor_main(["pullback-rate", "--out", str(out), "--episodes", "2", "--seed", "3"])
    ra = json.loads((tmp_path / "a.kedit.safetensors.report.json").read_text())
    rb = json.loads((tmp_path / "b.kedit.safetensors.report.json").read_text())
    assert ra["rate"] == rb["rate"] and ra["scores"] == rb["scores"]


def test_listed_by_kilora_train_editor(capsys, registered):
    assert editor_main(["--list"]) == 0
    assert "pullback-rate" in capsys.readouterr().out
