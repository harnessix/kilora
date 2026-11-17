# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path

import pytest
import torch
from test_import_edit_dimensions import _session
from test_import_measure_budget import _quadratic, _setup
from test_import_modes import DIM, Toy

from kilora import compose
from kilora.branches import BranchStore
from kilora.comfyui.live_write import QueuedWrite
from kilora.editors import commit_write
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor, ImportHistory, session_for_shared_edits
from kilora.live_queue import WriteQueue
from kilora.recording import GenerationRecord


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _latent(session):
    model = Toy().eval()
    latent = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(721))
    handle = compose(model, session.sets(), depth=1, mix="stack")
    try:
        with torch.no_grad():
            for _ in range(3):
                latent = model(latent)
        return latent
    finally:
        handle.remove()


def _record(session, directions):
    slopes = torch.linspace(-1.0, 1.0, len(directions), dtype=torch.float64)
    return _quadratic(session, directions, slopes)


def _session_with_empty_target(tmp_path):
    session = _session(tmp_path)
    target = next(iter(session._source.experts))
    session._source.experts[target].up.zero_()
    session.learning, session.guard = True, False
    session.set_options(core_dims=0, measure="compressed", measure_combos=8)
    editor = ImportEditor(session)
    assert session.state[target] == {}
    assert len(session.state) > 1
    return session, editor, session.moves(), target


def _assert_complete_sets(session):
    targets = set(session._source.experts)
    assert targets <= set(session.state)
    for branch in session.editor_state.forest.branches.values():
        assert targets <= set(branch.state.edits)
    for state in (session.pending, session._pinned_state):
        if state is not None:
            assert targets <= set(state)
    for pending in (False, True):
        assert set(session.sets(use_pending=pending)[0].experts) == targets


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("draw", [False, True])
def test_refused_import_write_preserves_empty_targets(tmp_path, pinned, draw):
    session, editor, directions, target = _session_with_empty_target(tmp_path)
    if pinned:
        session.pin()
    before = copy.deepcopy(session.state)
    expected = _latent(session)
    expected_sets = session.sets()
    record = _record(session, directions)
    calls = []

    def held(trial):
        calls.append(trial)
        return 1.0

    record.measure.held = held
    if draw:
        record.extra["draw"] = {"seed": 17, "temperature": 0.3, "depth": 2}
    commit_write(session, editor, record)
    assert len(calls) == 3
    assert "the step did not lower the measured loss" in editor._refused_note
    assert not session.commits and session.pending is None
    _assert_complete_sets(session)
    assert session.state[target] == {}
    for name, parts in before.items():
        assert set(session.state[name]) == set(parts)
        for part, value in parts.items():
            assert torch.equal(session.state[name][part], value)
    for old, new in zip(expected_sets, session.sets(), strict=True):
        for name, expert in old.experts.items():
            assert torch.equal(expert.down, new.experts[name].down)
            assert torch.equal(expert.up, new.experts[name].up)
    assert torch.equal(_latent(session), expected)
    session.unpin()
    _assert_complete_sets(session)


def test_import_history_round_trip_preserves_empty_targets(tmp_path):
    session, editor, directions, target = _session_with_empty_target(tmp_path)
    history = session.editor_state
    branch = history.fork()
    editor.checkout_branch(branch)
    restored = ImportHistory(())
    restored.load_tensors(session.editor_snapshot())
    assert restored.clean[target] == {}
    for state in restored.forest.branches.values():
        assert set(state.state.edits) == set(session.state)
        assert state.state.edits[target] == {}
    restored.reset()
    assert set(restored.expand_live({})) == set(session.state)


def test_import_accept_undo_checkout_and_pin_keep_empty_targets(tmp_path, monkeypatch):
    session, editor, directions, target = _session_with_empty_target(tmp_path)
    initial = _latent(session)
    branch = session.editor_state.fork()
    editor.checkout_branch(branch)
    commit = session.commit

    def checked_commit(*args, **kwargs):
        assert session.pending is not None
        _assert_complete_sets(session)
        return commit(*args, **kwargs)

    monkeypatch.setattr(session, "commit", checked_commit)
    commit_write(session, editor, _record(session, directions))
    assert len(session.commits) == 1
    _assert_complete_sets(session)
    changed = _latent(session)
    assert not torch.equal(changed, initial)
    session.undo()
    _assert_complete_sets(session)
    assert session.state[target] == {}
    assert torch.equal(_latent(session), initial)
    commit_write(session, editor, _record(session, directions))
    editor.checkout_branch("main")
    _assert_complete_sets(session)
    assert torch.equal(_latent(session), initial)
    editor.checkout_branch(branch)
    _assert_complete_sets(session)
    session.pin()
    pinned = _latent(session)
    commit_write(session, editor, _record(session, directions))
    _assert_complete_sets(session)
    assert torch.equal(_latent(session), pinned)
    snapshot = session._saved_editor_snapshot()
    session.unpin()
    session._restore_pin(snapshot)
    _assert_complete_sets(session)
    assert session._pinned_state[target] == {}
    assert torch.equal(_latent(session), pinned)
    saved = session.save_point("empty-pinned")
    session.unpin()
    session.load(saved)
    _assert_complete_sets(session)
    assert torch.equal(_latent(session), pinned)
    session.unpin()
    session.reset()
    _assert_complete_sets(session)
    assert torch.equal(_latent(session), initial)


def test_import_write_commits_on_current_branch(tmp_path):
    session, editor, directions = _setup(tmp_path)
    history = session.editor_state
    assert isinstance(history.forest, BranchStore)
    branch = history.fork(seed=31, temperature=0.2, depth=3)
    editor.checkout_branch(branch)
    before = copy.deepcopy(session.state)
    commit_write(session, editor, _record(session, directions))
    assert history.forest.current == branch
    assert history.forest.state.events == 1
    assert session.undo_log._entries[-1].branch == branch
    assert len(session.commits) == 1
    assert any(
        not torch.equal(v, session.state[t][p])
        for t, parts in before.items()
        for p, v in parts.items()
    )
    assert history.forest.branches["main"].state.events == 0


def test_pin_preserves_final_latent_while_imported_writes_continue(tmp_path):
    session, editor, directions = _setup(tmp_path)
    session.pin()
    expected = _latent(session)
    for _ in range(3):
        commit_write(session, editor, _record(session, directions))
        assert torch.equal(_latent(session), expected)
    forest = session.editor_state.forest
    branch = session.pinned_write_branch
    assert forest.branches[branch].parent == "main"
    assert forest.branches[branch].state.events == 3
    assert forest.branches["main"].state.events == 0
    assert len(session.commits) == 3
    session.unpin()
    assert not torch.equal(_latent(session), expected)


def test_promote_undo_and_reset_are_branch_local(tmp_path):
    session, editor, directions = _setup(tmp_path)
    initial = copy.deepcopy(session.state)
    session.pin()
    commit_write(session, editor, _record(session, directions))
    branch = session.pinned_write_branch
    editor.checkout_branch(branch, promote=True)
    assert session.pinned is None
    assert session.editor_state.forest.head == branch
    changed = _latent(session)
    session.undo()
    assert all(
        torch.equal(v, session.state[t][p]) for t, ps in initial.items() for p, v in ps.items()
    )
    commit_write(session, editor, _record(session, directions))
    session.reset()
    assert all(
        torch.equal(v, session.state[t][p]) for t, ps in initial.items() for p, v in ps.items()
    )
    session.undo()
    assert torch.equal(_latent(session), changed)
    editor.checkout_branch("main")
    assert session.editor_state.forest.state.events == 0


class StepEditor(ImportEditor):
    finish_write = None

    def next_state(self, state, record):
        new = copy.deepcopy(state)
        for parts in new.values():
            for key in parts:
                if key.startswith("strength_"):
                    parts[key] = parts[key] + record.images * 0.01
        return new


@pytest.mark.parametrize("action", ["undo", "reset", "load"])
def test_waiting_import_writes_survive_on_side_branch(tmp_path, action):
    session = _session(tmp_path)
    editor = StepEditor(session)
    commit_write(session, editor, GenerationRecord(images=1))
    saved = session.save_point("before")
    original = copy.deepcopy(session.state)
    queue = WriteQueue(max_pending=1)
    queue.begin_generation()
    try:
        for count in (2, 3, 4):
            queue.submit(QueuedWrite(session, editor, GenerationRecord(images=count), None))
        with queue.exclusive():
            [(branch, numbers)] = queue.keep_pending(session.editor_state, action)
            assert numbers == [1, 2, 3]
            if action == "load":
                session.load(saved)
            else:
                getattr(session, action)()
        visible = _latent(session)
        queue.end_generation()
        assert queue.wait_idle(30)
        assert not queue.dropped
        assert len(queue.reports) == 1
        assert "combined=1" in queue.reports[0][1]
        assert torch.equal(_latent(session), visible)
        assert session.editor_state.forest.branches[branch].state.events == 4
        assert sum(entry.branch == branch for entry in session.undo_log._entries) == 3
        editor.checkout_branch(branch, promote=True)
        for t, parts in original.items():
            for key, value in parts.items():
                if key.startswith("strength_"):
                    expected = value
                    for count in (2, 3, 4):
                        expected = expected + count * 0.01
                    assert torch.equal(session.state[t][key], expected)
        session.undo()
        assert session.editor_state.forest.state.events == 3
    finally:
        queue.end_generation()
        queue.close()


@pytest.mark.parametrize("embedded", [False, True])
def test_export_load_keeps_forest_pin_and_render(tmp_path, embedded):
    session, editor, directions = _setup(tmp_path)
    session.pin()
    expected = _latent(session)
    for _ in range(2):
        commit_write(session, editor, _record(session, directions))
    branch = session.pinned_write_branch
    path = tmp_path / "forest.kmod"
    session.export(path, include_sources=embedded)
    if embedded:
        for source in session.sources:
            Path(source["path"]).unlink()
    SESSIONS.clear()
    loaded = session_for_shared_edits(path, Toy().eval())
    assert loaded is not None
    assert loaded.editor_state.forest.list() == session.editor_state.forest.list()
    assert loaded.pinned == session.pinned
    assert loaded.pinned_write_branch == branch
    assert len(loaded.commits) == len(session.commits)
    assert torch.equal(_latent(loaded), expected)
    ImportEditor(loaded).checkout_branch(branch, promote=True)
    session.unpin()
    assert torch.equal(_latent(loaded), _latent(session))


def test_save_load_adds_forest_and_undo_restores_it(tmp_path):
    session, editor, directions = _setup(tmp_path)
    commit_write(session, editor, _record(session, directions))
    name = session.save_point("one")
    saved_render = _latent(session)
    commit_write(session, editor, _record(session, directions))
    expected = _latent(session)
    before = session.editor_state.forest.list()
    session.load(name)
    assert torch.equal(_latent(session), saved_render)
    assert len(session.editor_state.forest.branches) == 2
    session.undo()
    assert session.editor_state.forest.list() == before
    assert torch.equal(_latent(session), expected)


def test_fresh_session_loads_aligned_pinned_save_point(tmp_path):
    session, editor, directions = _setup(tmp_path)
    session.pin()
    commit_write(session, editor, _record(session, directions))
    expected = _latent(session)
    name = session.save_point("pinned")
    branch = session.pinned_write_branch
    SESSIONS.clear()
    restored = _session(tmp_path)
    assert not restored._aligned_keys
    restored.load(name)
    assert restored.pinned == session.pinned
    assert restored.pinned_write_branch == branch
    assert torch.equal(_latent(restored), expected)
    session.unpin()
    restored.unpin()
    assert torch.equal(_latent(restored), _latent(session))


def test_waiting_write_keeps_revision_after_history_rollback(tmp_path):
    session = _session(tmp_path)
    editor = StepEditor(session)
    queue = WriteQueue()
    queue.begin_generation()
    try:
        snapshot = session.editor_snapshot()
        queue.submit(QueuedWrite(session, editor, GenerationRecord(images=2), None))
        session.editor_state.load_tensors(snapshot)
        with queue.exclusive():
            kept = queue.keep_pending(session.editor_state, "reset")
            assert kept is not None
            [(branch, numbers)] = kept
            assert numbers == [1]
            session.reset()
        visible = _latent(session)
        queue.end_generation()
        assert queue.wait_idle(30)
        assert torch.equal(_latent(session), visible)
        assert session.editor_state.forest.branches[branch].state.events == 1
    finally:
        queue.end_generation()
        queue.close()


def test_edits_off_export_keeps_forest_and_can_enable_saved_edits(tmp_path):
    session, editor, directions = _setup(tmp_path)
    commit_write(session, editor, _record(session, directions))
    edited = _latent(session)
    session.set_options(edits="off")
    expected = _latent(session)
    path = tmp_path / "off.kmod"
    session.export(path)
    restored = session_for_shared_edits(path, Toy().eval())
    assert restored.edits == "off"
    assert restored.editor_state.forest.list() == session.editor_state.forest.list()
    assert torch.equal(_latent(restored), expected)
    restored.set_options(edits="on")
    assert torch.equal(_latent(restored), edited)


def test_imported_draw_forks_and_records_draw_metadata(tmp_path):
    session, editor, directions = _setup(tmp_path)
    record = _record(session, directions)
    record.extra["draw"] = {"seed": 51, "temperature": 0.3, "depth": 4}
    commit_write(session, editor, record)
    forest = session.editor_state.forest
    branch = forest.branches[forest.current]
    assert branch.id != "main" and branch.parent == "main"
    assert (branch.seed, branch.temperature, branch.depth) == (51, 0.3, 4)
    assert branch.state.events == 1
    assert forest.branches["main"].state.events == 0


def test_export_retains_terms_on_an_inactive_branch(tmp_path):
    from test_import_pack_terms import _one_term

    session = _session(tmp_path)
    editor = StepEditor(session)
    branch = session.editor_state.fork()
    editor.checkout_branch(branch)
    target = next(iter(session.state))
    _one_term(session, target)
    commit_write(session, editor, GenerationRecord(images=1))
    edited = _latent(session)
    terms = {k: v.clone() for k, v in session.state[target].items() if k.startswith("terms_")}
    editor.checkout_branch("main")
    expected = _latent(session)
    path = tmp_path / "inactive-terms.kmod"
    session.export(path)
    restored = session_for_shared_edits(path, Toy().eval())
    assert torch.equal(_latent(restored), expected)
    ImportEditor(restored).checkout_branch(branch, promote=True)
    for key, value in terms.items():
        assert torch.equal(restored.state[target][key], value)
    assert torch.equal(_latent(restored), edited)


def test_waiting_writes_survive_undo_of_a_forest_load(tmp_path, monkeypatch):
    from kilora.comfyui import patching

    session = _session(tmp_path)
    editor = StepEditor(session)
    commit_write(session, editor, GenerationRecord(images=1))
    name = session.save_point("one")
    commit_write(session, editor, GenerationRecord(images=2))
    session.load(name)
    assert session.undo_log._entries[-1].whole_editor
    queue = session.write_queue = WriteQueue(max_pending=1)
    monkeypatch.setattr(patching, "sessions_of", lambda _: [session])
    monkeypatch.setattr(patching, "reapply", lambda _: (None, "applied"))
    queue.begin_generation()
    try:
        for count in (3, 4):
            queue.submit(QueuedWrite(session, editor, GenerationRecord(images=count), None))
        patching.live_action(object(), "undo", "", session.mod_id)
        branch = next(
            b for b in session.editor_state.forest.branches.values() if b.action == "undo"
        )
        assert branch.state.events == 1
        visible = _latent(session)
        queue.end_generation()
        assert queue.wait_idle(30)
        assert torch.equal(_latent(session), visible)
        assert session.editor_state.forest.branches[branch.id].state.events == 3
        assert not queue.dropped
        editor.checkout_branch(branch.id, promote=True)
        session.undo()
        assert session.editor_state.forest.state.events == 2
    finally:
        queue.end_generation()
        queue.close()


def test_loading_pinned_save_remaps_pin_and_continues_its_write_branch(tmp_path):
    session, editor, directions = _setup(tmp_path)
    session.pin()
    commit_write(session, editor, _record(session, directions))
    name = session.save_point("pinned")
    pinned = _latent(session)
    session.unpin()
    commit_write(session, editor, _record(session, directions))
    session.load(name)
    forest = session.editor_state.forest
    assert session.pinned["branch"] in forest.branches
    assert session.pinned_write_branch in forest.branches
    assert session.pinned_write_branch == forest.current
    assert torch.equal(_latent(session), pinned)
    commit_write(session, editor, _record(session, directions))
    assert forest.state.events == 2
    assert torch.equal(_latent(session), pinned)


def test_loading_edit_export_retains_waiting_writes(tmp_path):
    session = _session(tmp_path)
    editor = StepEditor(session)
    commit_write(session, editor, GenerationRecord(images=1))
    path = tmp_path / "saved.kmod"
    session.export(path)
    saved = _latent(session)
    commit_write(session, editor, GenerationRecord(images=2))
    queue = session.write_queue = WriteQueue()
    queue.begin_generation()
    try:
        queue.submit(QueuedWrite(session, editor, GenerationRecord(images=3), None))
        session.load_edits(path)
        branch = next(
            b for b in session.editor_state.forest.branches.values() if b.action == "load"
        )
        assert branch.state.events == 2
        assert torch.equal(_latent(session), saved)
        queue.end_generation()
        assert queue.wait_idle(30)
        assert session.editor_state.forest.branches[branch.id].state.events == 3
        assert torch.equal(_latent(session), saved)
        assert not queue.dropped
        editor.checkout_branch(branch.id, promote=True)
        session.undo()
        assert session.editor_state.forest.state.events == 2
    finally:
        queue.end_generation()
        queue.close()
