# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest
import torch
from safetensors import safe_open
from test_import_modes import DIM, LAYERS, Toy, _lora

from kilora import compose
from kilora.live import (
    SESSIONS,
    LearnResult,
)
from kilora.live_imports import imported_session_for
from kilora.modfile import load_mod, read_manifest
from kilora.multi_import import import_loras


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_imported_session(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = [digest(p) for p in paths]
    sources = [
        dict(sha256=h, path=str(p), strength=1.0, axis="both")
        for p, h in zip(paths, hashes, strict=True)
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")
    assert imported_session_for(routed, list(reversed(sources)), "toy", shapes) is session
    changed = [dict(s) for s in sources]
    changed[0]["strength"] = 0.5
    assert imported_session_for(routed, changed, "toy", shapes) is not session
    changed[0]["sha256"] = "other"
    assert imported_session_for(routed, changed, "toy", shapes) is not session

    x = torch.randn(1, 3, DIM)
    base = Toy().eval()
    original = compose(base, [routed])
    with torch.no_grad():
        expected = base(x)
    original.remove()
    for option in ("on", "off"):
        session.set_options(edits=option)
        adapter = compose(base, session.sets())
        with torch.no_grad():
            assert torch.equal(expected, base(x))
        adapter.remove()

    target = next(iter(session.state))
    session.pending = {t: dict(parts) for t, parts in session.state.items()}
    session.pending[target]["dummy"] = torch.tensor([2.0])
    session._pending_info = LearnResult(1, 0.1, 0.0, False, False, "test")
    session.commit()
    saved = session.save_point("one")
    session.undo()
    assert "dummy" not in session.state[target]
    session.load(saved)
    assert torch.equal(session.state[target]["dummy"], torch.tensor([2.0]))
    mod_path = tmp_path / "saves" / session.mod_id / "edits.kmod"
    assert read_manifest(mod_path)["edits"]["sources"] == sources
    assert load_mod(mod_path).sets == []
    with safe_open(str(mod_path), framework="pt") as f:
        assert f"kilora/edit/{target}/dummy" in list(f.keys())
    assert [digest(p) for p in paths] == hashes


def test_tuple_shapes_save(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    path = tmp_path / "0.safetensors"
    _lora(model, path, rank=4, seed=0, alpha=4.0, layers=LAYERS)
    routed = import_loras([str(path)], model).adapter_set
    shapes = {t: tuple(e.down.shape) for t, e in routed.experts.items()}
    source = dict(sha256="t", path=str(path), strength=1.0, axis="both")
    session = imported_session_for(routed, [source], "toy", shapes, tmp_path / "saves")
    out = tmp_path / "edits.kmod"
    session.save_edits(out)
    assert read_manifest(out)["edits"]["shapes"] == {t: list(s) for t, s in shapes.items()}
