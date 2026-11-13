# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from test_chunk_live import _packs

from kilora.editors import editor_for
from kilora.live import open_session
from kilora.recording import GenerationRecord
from kilora.testing import make_live_mod


def _setup(tmp_path):
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod")
    rule = editor_for(_packs(session, tmp_path))
    rule.check(session)
    chunks = session.editor_state.chunks
    before = chunks.snapshot()
    chunks.coef["common"][0, 0] += 0.01
    merged = chunks.snapshot()
    chunks.restore(before)
    return session, rule, (before, merged, [{"step": 1}])


def test_guarded_merge_source_and_undo(tmp_path):
    session, rule, staged = _setup(tmp_path)
    old = {h: p["up"].clone() for h, p in session.state.items()}
    report = rule.commit_path_merge(session, [GenerationRecord(images=1)], staged, source="distill")
    assert "source=distill" in report
    assert session.commits[-1].source == "distill"
    assert any(not torch.equal(old[h], p["up"]) for h, p in session.state.items())
    assert session.editor_state.forest.branches["main"].paths == [{"step": 1}]
    session.undo()
    assert all(torch.equal(old[h], p["up"]) for h, p in session.state.items())
    assert not session.editor_state.forest.branches["main"].paths
    assert not session.commits


def test_guard_refusal_restores_state_and_paths(tmp_path, monkeypatch):
    session, rule, staged = _setup(tmp_path)
    old = {h: p["up"].clone() for h, p in session.state.items()}
    monkeypatch.setattr("kilora.editors.guard_check", lambda *args: (True, "refused"))
    report = rule.commit_path_merge(session, [GenerationRecord(images=1)], staged, source="distill")
    assert "refused" in report
    assert all(torch.equal(old[h], p["up"]) for h, p in session.state.items())
    assert not session.editor_state.forest.branches["main"].paths
    assert not session.commits
