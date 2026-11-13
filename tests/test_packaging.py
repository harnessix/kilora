# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import re
from pathlib import Path

import pytest

from kilora import __version__
from kilora.comfyui import help as help_texts
from kilora.comfyui.nodes import FOLDERS, NODE_CLASS_MAPPINGS

tomllib = pytest.importorskip("tomllib")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_semantic_and_matches_the_package(project):
    version = project["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert version == __version__


def test_comfy_registry_fields(project):
    p = project["project"]
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]*([._-][A-Za-z0-9]+)*", p["name"])
    assert len(p["name"]) < 100 and "comfyui" not in p["name"].lower()
    assert p["urls"]["Repository"].startswith("https://")
    assert p["license"] == {"file": "LICENSE"}
    assert project["tool"]["comfy"]["PublisherId"]
    assert p["description"] and p["requires-python"]


def test_requirements_match_the_dependencies(project):
    deps = {re.split(r"[<>=!~ ]", d)[0]: d for d in project["project"]["dependencies"]}
    lines = (ROOT / "requirements.txt").read_text().split()
    for line in lines:
        name = re.split(r"[<>=!~ ]", line)[0]
        assert deps.get(name) == line, line
    assert not any(line.startswith("torch") for line in lines)


def test_node_list_is_current(monkeypatch):
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    shipped = json.loads((ROOT / "node_list.json").read_text(encoding="utf-8"))
    assert shipped == help_texts.node_list(), "run: python -m kilora.comfyui.help ."
    assert set(shipped) == set(NODE_CLASS_MAPPINGS)
    settings = shipped["KiLoRASettings"]
    assert "faster" in settings and "torch reproduces images made before this release" in settings


def test_manager_entry_draft(project):
    entry = json.loads((ROOT / "packaging" / "comfyui-manager-entry.json").read_text())
    assert entry["install_type"] == "git-clone"
    assert entry["reference"] == project["project"]["urls"]["Repository"]
    assert entry["files"] == [entry["reference"]]
    assert entry["author"] == project["tool"]["comfy"]["PublisherId"]
