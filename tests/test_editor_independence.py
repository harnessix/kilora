# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import importlib
import json

import pytest
import torch
from live_helpers import build_live_mod
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_path_write import WrapPatcher, fresh_sessions

from kilora import editors
from kilora.check import check_file
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.errors import KiLoRAError


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr(editors, "_PLUGINS", {})
    monkeypatch.setattr(editors, "_SOURCES", {})
    monkeypatch.setattr(editors, "_DISCOVERED", True)


def _session(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod", updater={"id": "not-installed"})
    base = FakePatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(base, chain, 1)
    return patched, patching.sessions_of(patched)[0]


def _editor_file(folders, name, kind):
    manifest = {"format": "kilora-editor", "kind": kind, "training": {"state": "trained"}}
    path = folders["kilora"] / name
    save_file({"x": torch.zeros(1)}, str(path), metadata={"kilora.manifest": json.dumps(manifest)})
    return name


def test_live_host_runs_with_no_editor(denoiser, folders, tmp_path, no_plugins):
    patched, session = _session(denoiser, folders)
    assert session.pairing.ready
    assert "Learning on" in session.set_options(learning=True)
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    with pytest.raises(KiLoRAError, match="No live-mode editor found"):
        N.KiLoRALiveRecord().record(wp, "")
    name = _editor_file(folders, "other.kedit.safetensors", "someone-elses-kind")
    with pytest.raises(KiLoRAError, match="is not a live-mode editor"):
        N.KiLoRALiveRecord().record(wp, name)
    assert session.save_point("start").startswith("save-")
    assert "Reset" in session.reset() or session.commits == []
    session.undo()
    report = check_file(folders["kilora"] / "ink.kmod")
    assert report.ok and report.kind == "live mod"
    assert "no installed editor plugin" in check_file(folders["kilora"] / name).problems[0]


PLUGIN_SOURCE = '''
from kilora.editors import Editor, EditorPlugin


class DoubleUp(Editor):
    """A third-party editor: doubles the live up parts."""

    def __init__(self, path):
        self.path = path

    def next_state(self, state, record):
        return {t: {**parts, "up": parts["up"] * 2.0} for t, parts in state.items()}


PLUGIN = EditorPlugin("third-party-double", DoubleUp, title="Third-party example")
'''

ENTRY_POINTS = "[kilora.editors]\nthird-party-double = third_party_editor:PLUGIN\n"


@pytest.fixture
def third_party(monkeypatch, tmp_path):
    site = tmp_path / "site"
    (site / "third_party_editor").mkdir(parents=True)
    (site / "third_party_editor" / "__init__.py").write_text(PLUGIN_SOURCE)
    info = site / "third_party_editor-1.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: third-party-editor\n")
    (info / "entry_points.txt").write_text(ENTRY_POINTS)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    monkeypatch.setattr(editors, "_PLUGINS", dict(editors._PLUGINS))
    monkeypatch.setattr(editors, "_SOURCES", dict(editors._SOURCES))
    monkeypatch.setattr(editors, "_DISCOVERED", False)
    yield
    import sys

    sys.modules.pop("third_party_editor", None)


def test_third_party_editor_through_the_entry_point(denoiser, folders, third_party):
    plugins = editors.available_editors()
    assert "third-party-double" in plugins
    assert editors._SOURCES["third-party-double"] == "entry point third-party-double"
    patched, session = _session(denoiser, folders)
    session.learning = True
    name = _editor_file(folders, "double.kedit.safetensors", "third-party-double")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, name)
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(0))
    model.sample(x)
    before = {t: p["up"].clone() for t, p in session.state.items()}
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": x})["result"]
    assert "1 commits" in report
    for t, up in before.items():
        assert torch.allclose(session.state[t]["up"], up * 2.0)
    session.undo()
    assert all(torch.equal(session.state[t]["up"], up) for t, up in before.items())
    report = check_file(folders["kilora"] / name)
    assert report.kind == "editor (third-party-double)"


def test_a_broken_plugin_does_not_stop_the_host(monkeypatch, tmp_path):
    site = tmp_path / "site"
    (site / "broken_editor").mkdir(parents=True)
    (site / "broken_editor" / "__init__.py").write_text("raise RuntimeError('broken')\n")
    info = site / "broken_editor-1.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: broken-editor\n")
    (info / "entry_points.txt").write_text("[kilora.editors]\nbroken = broken_editor:PLUGIN\n")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    monkeypatch.setattr(editors, "_PLUGINS", {})
    monkeypatch.setattr(editors, "_DISCOVERED", False)
    with pytest.warns(UserWarning, match="broken"):
        found = editors.available_editors()
    assert "broken" not in found
