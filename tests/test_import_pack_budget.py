# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_pack_terms import (
    NO_PACK_MOVES,
    PLATEAU,
    STRONG,
    _gated_record,
    _pack_editor,
)
from test_import_rprop_integration import record as coordinate_record

from kilora.editors import commit_write
from kilora.import_terms import PACK_TERM_MOVES
from kilora.live import SESSIONS
from kilora.live_imports import ImportTrial
from kilora.recording import GenerationRecord

COMBOS = 4
PLATEAU_GAINS = [1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]
STRONG_GAINS = [1.0] * 6


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _editor(tmp_path, controller, measure="compressed"):
    SESSIONS.clear()
    session, editor, move, target = _pack_editor(tmp_path, controller)
    session.set_options(measure=measure, measure_combos=COMBOS)
    return session, editor, move, target


def _counted(base):
    calls = []

    def measure(trial):
        calls.append(1)
        return base.measure(trial)

    return GenerationRecord(measure=measure, guard_score=base.guard_score), calls


def _clone(state):
    return {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}


def _equal_states(a, b):
    assert a.keys() == b.keys()
    for target in a:
        assert a[target].keys() == b[target].keys()
        for key, value in a[target].items():
            assert torch.equal(value, b[target][key]), (target, key)


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_event_with_pack_moves_costs_two_k_evaluations(tmp_path, controller):
    session, editor, move, target = _editor(tmp_path, controller)
    session.editor_state.gains = torch.tensor(PLATEAU_GAINS, dtype=torch.float64)
    rec, calls = _counted(_gated_record(session, move, STRONG))
    new = editor.next_state(session.state, rec)
    assert len(calls) == 2 * COMBOS + 2
    assert f"Pack moves: {PACK_TERM_MOVES} measured" in editor.note
    assert new is not None and "terms_c" in new[target]
    assert bool(new[target]["terms_c"].count_nonzero())
    ids = session.editor_state.compressed["ids"]
    assert move.id in ids
    assert sum(1 for key in ids if ":term:" in key) == PACK_TERM_MOVES


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_gate_reads_earlier_events_only(tmp_path, controller):
    session, editor, move, target = _editor(tmp_path, controller)
    session.editor_state.gains = torch.tensor(STRONG_GAINS, dtype=torch.float64)
    rec, calls = _counted(_gated_record(session, move, PLATEAU))
    editor.next_state(session.state, rec)
    assert NO_PACK_MOVES in editor.note
    assert len(calls) == 2 * COMBOS + 2
    assert not any(":term:" in key for key in session.editor_state.compressed["ids"])


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_stored_decrease_is_the_measured_decrease(tmp_path, controller):
    session, editor, move, _target = _editor(tmp_path, controller)
    before = _clone(session.state)
    rec = _gated_record(session, move, STRONG)
    commit_write(session, editor, rec)
    store = session.editor_state
    assert store.gains.shape[0] == 1
    step = coordinate_record(session, move).measure(ImportTrial(session, before, session.state))
    assert step != 0.0
    expected = rec.measure(ImportTrial(session, before, before)) - rec.measure(
        ImportTrial(session, before, session.state)
    )
    assert expected > 0
    assert float(store.gains[0]) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_estimated_plateau_opens_the_gate_and_strong_keeps_it_shut(tmp_path, controller):
    session, editor, move, target = _editor(tmp_path, controller)
    notes = []
    for _ in range(6):
        commit_write(session, editor, _gated_record(session, move, STRONG))
        notes.append(editor.note)
    gains = session.editor_state.gains
    assert gains.shape[0] == 6 and bool((gains > 0).all())
    assert all(NO_PACK_MOVES in note for note in notes)
    assert "terms_c" not in session.state[target]
    opened = None
    for event in range(30):
        commit_write(session, editor, _gated_record(session, move, PLATEAU))
        if NO_PACK_MOVES not in editor.note:
            opened = event
            break
    assert opened is not None and opened >= 1
    assert f"Pack moves: {PACK_TERM_MOVES} measured" in editor.note
    assert bool(session.state[target]["terms_c"].count_nonzero())


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_fewer_than_three_estimated_events_give_no_pack_moves(tmp_path, controller):
    session, editor, move, target = _editor(tmp_path, controller)
    for _ in range(3):
        commit_write(session, editor, _gated_record(session, move, PLATEAU))
        assert NO_PACK_MOVES in editor.note
    assert session.editor_state.gains.shape[0] == 3
    assert "terms_c" not in session.state[target]


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_no_packs_and_shut_gate_give_the_same_dimension_one_and_two_output(tmp_path, controller):
    results = []
    for packs in (False, True):
        folder = tmp_path / str(packs)
        folder.mkdir()
        session, editor, move, _target = _editor(folder, controller)
        if not packs:
            session.editor_packs = []
        for _ in range(3):
            commit_write(session, editor, _gated_record(session, move, STRONG))
            assert NO_PACK_MOVES in editor.note or not packs
        results.append((session.state, session.editor_state.compressed))
    (state_a, est_a), (state_b, est_b) = results
    _equal_states(state_a, state_b)
    assert est_a["ids"] == est_b["ids"]
    for name in ("slopes", "cov", "noise"):
        assert torch.equal(est_a[name], est_b[name])


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_full_measure_keeps_per_move_pack_measurement(tmp_path, controller):
    session, editor, move, target = _editor(tmp_path, controller, measure="full")
    session.editor_state.gains = torch.tensor(PLATEAU_GAINS[:5], dtype=torch.float64)
    rec, calls = _counted(_gated_record(session, move, PLATEAU))
    new = editor.next_state(session.state, rec)
    directions = session.moves()
    assert len(calls) == 2 * len(directions) + 2 + 2 * PACK_TERM_MOVES
    assert f"Pack moves: {PACK_TERM_MOVES} measured" in editor.note
    assert session.editor_state.compressed is None

    SESSIONS.clear()
    folder = tmp_path / "manual"
    folder.mkdir()
    other, other_editor, other_move, _ = _editor(folder, controller, measure="full")
    other.editor_state.gains = torch.tensor(PLATEAU_GAINS[:5], dtype=torch.float64)
    other_rec = _gated_record(other, other_move, PLATEAU)
    state = other.state
    rule = other_editor._rprop if controller == "rprop" else other_editor._r1
    base = rule(state, other_rec, other.moves())
    base = state if base is None else base
    gain = other_rec.measure(ImportTrial(other, state, state)) - other_rec.measure(
        ImportTrial(other, state, base)
    )
    other.editor_state.add_gain(float(gain))
    candidates, _ = other_editor.pack_moves()
    manual, _, _ = other_editor._pack_step(state, base, other_rec, candidates)
    _equal_states(new, manual)
    assert torch.equal(session.editor_state.gains, other.editor_state.gains)


def test_pack_moves_set_change_keeps_dimension_prior(tmp_path):
    session, editor, move, target = _editor(tmp_path, "rprop")
    commit_write(session, editor, _gated_record(session, move, STRONG))
    prior = session.editor_state.compressed
    assert prior["ids"] == [move.id]
    session.editor_state.gains = torch.tensor(PLATEAU_GAINS, dtype=torch.float64)
    editor.next_state(session.state, _gated_record(session, move, STRONG))
    ids = session.editor_state.compressed["ids"]
    assert ids[0] == move.id and len(ids) == 1 + PACK_TERM_MOVES
