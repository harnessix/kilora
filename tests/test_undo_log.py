# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from kilora_editor_delta import delta_write, recorder_for
from test_comfyui import denoiser, folders
from test_path_write import _setup, fresh_sessions

from kilora.errors import KiLoRAError
from kilora.live import UndoLog

MB = 1024 * 1024


def _state(value, n=1024):
    return {"t": {"down": torch.ones(4, 4), "up": torch.full((n,), float(value))}}


def test_entry_keeps_only_changed_tensors():
    log = UndoLog(max_mb=1)
    log.push(_state(0), _state(1), [])
    assert log.nbytes == 1024 * 4
    restored, commits = log.pop(_state(1))
    assert torch.equal(restored["t"]["up"], _state(0)["t"]["up"])
    assert torch.equal(restored["t"]["down"], torch.ones(4, 4)) and commits == []


def test_log_drops_oldest_entries_over_the_size():
    log = UndoLog(max_mb=1)
    n = MB // 4 // 4
    for i in range(10):
        log.push(_state(i, n), _state(i + 1, n), [])
    assert len(log) == 4 and log.dropped == 6 and log.nbytes <= MB
    state = _state(10, n)
    for i in reversed(range(6, 10)):
        state, _ = log.pop(state)
        assert float(state["t"]["up"][0]) == i
    assert len(log) == 0


def test_newest_entry_stays_even_when_too_large():
    log = UndoLog(max_mb=0)
    log.push(_state(0), _state(1), [])
    log.push(_state(1), _state(2), [])
    assert len(log) == 1
    state, _ = log.pop(_state(2))
    assert float(state["t"]["up"][0]) == 1


def test_smaller_size_trims_at_once_and_negative_size_is_refused():
    log = UndoLog(max_mb=10)
    for i in range(8):
        log.push(_state(i, MB // 4), _state(i + 1, MB // 4), [])
    assert len(log) == 8
    log.set_size(2.5)
    assert len(log) == 2
    with pytest.raises(KiLoRAError, match="0 MB or more"):
        log.set_size(-1)


def test_size_from_environment(monkeypatch, denoiser, folders, tmp_path):
    monkeypatch.setenv("KILORA_UNDO_LOG_MB", "3")
    _, _, _, session, _, _ = _setup(denoiser, folders, tmp_path)
    assert session.undo_log.max_mb == 3.0


def _write(session, editor, patched, xs):
    rec = recorder_for(session, editor)
    with rec.recording():
        for x in xs:
            patched.run(x)
    return delta_write(session, editor, rec.finish())


def test_long_undo_over_many_writes(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    history = [{t: p["up"].clone() for t, p in session.state.items()}]
    for _ in range(5):
        _write(session, editor, patched, xs)
        history.append({t: p["up"].clone() for t, p in session.state.items()})
    entry = session.undo_log._entries[-1]
    assert all(set(parts) == {"up"} for parts in entry.parts.values())
    assert len(session.undo_log) == 5
    for back in reversed(history[:-1]):
        report = session.undo()
        assert all(torch.equal(back[t], p["up"]) for t, p in session.state.items())
    assert "0 more undo steps" in report and session.commits == []
    with pytest.raises(KiLoRAError, match="nothing to undo"):
        session.undo()


def test_bounded_session_keeps_the_newest_writes(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    _write(session, editor, patched, xs)
    per_write = session.undo_log.nbytes
    session.set_options(undo_log_mb=2.5 * per_write / MB)
    states = []
    for _ in range(4):
        states.append({t: p["up"].clone() for t, p in session.state.items()})
        _write(session, editor, patched, xs)
    assert len(session.undo_log) == 2 and session.undo_log.dropped == 3
    session.undo()
    session.undo()
    assert all(torch.equal(states[-2][t], p["up"]) for t, p in session.state.items())
    assert len(session.commits) == 3
    with pytest.raises(KiLoRAError, match="nothing to undo"):
        session.undo()
