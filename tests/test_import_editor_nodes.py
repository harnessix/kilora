# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest
import torch
from test_comfyui import FakeBaseModel
from test_families import _model
from test_import_edit_dimensions import _session
from test_import_modes import DIM, Toy
from test_path_write import WrapPatcher
from test_working_copies import routed_chain

from kilora import compose
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.errors import KiLoRAError
from kilora.i18n import text
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor, ImportedLiveSession


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    SESSIONS.clear()
    monkeypatch.setenv("KILORA_LIVE_ASYNC", "0")
    yield
    SESSIONS.clear()


def test_record_node_opens_imports_and_settings_reach_that_session(tmp_path, monkeypatch):
    monkeypatch.setattr(N.FOLDERS, "path", lambda *args: "")
    base = WrapPatcher(FakeBaseModel(_model("flux")))
    chain = routed_chain(tmp_path)
    applied, _ = patching.apply(base, chain, 1, joint=True, bypass="off")
    assert not patching.sessions_of(applied)
    recorded, handle = N.KiLoRALiveRecord().record(applied, text("choice.import_editor"))
    session = handle.session
    assert isinstance(session, ImportedLiveSession)
    assert isinstance(handle.rule, ImportEditor)
    assert patching.sessions_of(recorded)[0] is session
    assert session.controller == "r1"
    sources = {
        s["path"]: hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() for s in session.sources
    }

    out, _ = N.KiLoRALiveOptions().set(
        recorded, session.mod_id, True, -1.0, edits="off", controller="rprop"
    )
    assert session.edits == "off" and session.controller == "rprop"
    out, _ = N.KiLoRALiveOptions().set(out, session.mod_id, True, -1.0)
    assert session.edits == "off" and session.controller == "rprop"
    out, _ = N.KiLoRALiveOptions().set(out, "", True, -1.0, edits="on", controller="r1")
    _, again = N.KiLoRALiveRecord().record(out, text("choice.import_editor"))
    assert again.session is session
    assert session.edits == "on" and session.controller == "r1"
    assert sources == {
        s["path"]: hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() for s in session.sources
    }


def test_edits_off_restores_exact_source_outputs_through_options(tmp_path, monkeypatch):
    session = _session(tmp_path)
    marker = object()
    monkeypatch.setattr(patching, "sessions_of", lambda model: [session])
    monkeypatch.setattr(patching, "reapply", lambda model: (model, ""))
    model = Toy().eval()
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(73))

    def output(sets):
        adapter = compose(model, sets)
        try:
            with torch.no_grad():
                return model(x).clone()
        finally:
            adapter.remove()

    expected = output([session._source])
    target = next(iter(session.state))
    session.state[target]["strength_0"].fill_(2.0)
    session.state[target]["core_0"][0, 0] = 1.5
    assert not torch.equal(expected, output(session.sets()))
    N.KiLoRALiveOptions().set(marker, session.mod_id, True, -1.0, edits="off", controller="r1")
    assert torch.equal(expected, output(session.sets()))
    N.KiLoRALiveOptions().set(marker, session.mod_id, True, -1.0)
    assert torch.equal(expected, output(session.sets()))
    N.KiLoRALiveOptions().set(marker, session.mod_id, True, -1.0, edits="on")
    assert not torch.equal(expected, output(session.sets()))


def test_invalid_choices_do_not_change_import_options(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(patching, "sessions_of", lambda model: [session])
    for choice in ({"edits": "invalid"}, {"controller": "invalid"}):
        with pytest.raises(KiLoRAError):
            N.KiLoRALiveOptions().set(object(), session.mod_id, False, 0.0, **choice)
        assert session.edits == "on" and session.controller == "r1"


def test_empty_id_refuses_multiple_sessions(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(patching, "sessions_of", lambda model: [session, session])
    with pytest.raises(KiLoRAError, match="session ID"):
        N.KiLoRALiveOptions().set(object(), "", True, -1.0, edits="off")


def test_export_resolves_the_only_imported_session_with_empty_id(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(patching, "sessions_of", lambda model: [session])
    monkeypatch.setattr(N.FOLDERS, "first_dir", lambda folder: str(tmp_path))
    calls = []

    def export(path, **options):
        calls.append((path, options))
        return {"note": "Routing is not kept.", "relative_svd_residual": {}}

    monkeypatch.setattr(session, "export", export)
    result = N.KiLoRALiveExport().export(object(), "", True, "write report", format="lora")
    assert calls == [
        (
            str(tmp_path / "exports" / f"{session.mod_id}-share.safetensors"),
            {"format": "lora", "include_sources": True},
        )
    ]
    assert "Routing is not kept" in result["result"][0]
