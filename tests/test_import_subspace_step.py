# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_measure_budget import _coordinate, _quadratic, _setup

from kilora import live_imports
from kilora.editors import commit_write
from kilora.import_terms import TERM_FIXED
from kilora.live import SESSIONS
from kilora.live_imports import (
    IMPORT_DECAY,
    IMPORT_EPS,
    NOT_WRITTEN,
    ImportTrial,
    between,
    cap_step,
)
from kilora.recording import GenerationRecord

SPAN_COMBOS = 3
COMBOS = 16


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _clone(state):
    return {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}


def _slopes(directions, scale=1.0):
    return scale * torch.linspace(-1.0, 1.5, len(directions), dtype=torch.float64)


def _scripted(session, directions, slopes, script, calls):
    base = _quadratic(session, directions, slopes)

    def measure(trial):
        calls.append(1)
        if len(calls) <= 2 * COMBOS:
            return base.measure(trial)
        return script[len(calls) - 2 * COMBOS - 1]

    return GenerationRecord(measure=measure, guard_score=base.guard_score)


def _steep(session, directions, slopes, curvature, calls):
    def measure(trial):
        calls.append(1)
        x = torch.tensor([_coordinate(session, trial, m) for m in directions], dtype=torch.float64)
        return float(slopes @ x + curvature * x.square().sum())

    return GenerationRecord(measure=measure, guard_score=lambda _: {"gap": 1.0, "spread": 1.0})


def _hand_norm(session, old, new):
    trial = ImportTrial(session, old, new)
    total = trial.exact_norm + sum(float(v.float().norm()) ** 2 for v in trial.du.values())
    return (total + trial.terms_norm**2) ** 0.5


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_raw_step_is_the_least_norm_gradient_in_the_measured_span(tmp_path, controller):
    session, editor, directions = _setup(tmp_path, controller=controller, combos=SPAN_COMBOS)
    assert len(directions) > SPAN_COMBOS
    slopes = _slopes(directions)
    record = _quadratic(session, directions, slopes)
    assert editor._measure_subspace(session.state, record, directions)
    ids, psi, meas = editor._subspace_samples
    order = [[m.id for m in directions].index(key) for key in ids]
    gradient = slopes[order]
    assert torch.allclose(meas, psi @ gradient, rtol=1e-3, atol=1e-6)
    raw = -torch.tensor([editor._subspace[key] for key in ids], dtype=torch.float64)
    expected = -psi.T @ torch.linalg.pinv(psi @ psi.T) @ psi @ gradient
    assert torch.allclose(raw, expected, rtol=1e-3, atol=1e-6)
    coefficients = torch.linalg.lstsq(psi.T, raw.unsqueeze(1)).solution
    assert torch.allclose(psi.T @ coefficients.squeeze(1), raw, atol=1e-9)


def test_cap_step_clips_to_the_probe_size():
    assert cap_step(5.0) == IMPORT_EPS
    assert cap_step(-5.0) == -IMPORT_EPS
    assert cap_step(0.5 * IMPORT_EPS) == 0.5 * IMPORT_EPS


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_no_move_step_exceeds_the_probe_size(tmp_path, controller, monkeypatch):
    monkeypatch.setattr(live_imports, "IMPORT_WRITE_FRAC", 10.0)
    monkeypatch.setattr(live_imports, "IMPORT_CAP_FRAC", 1e9)
    session, editor, directions = _setup(tmp_path, controller=controller, combos=COMBOS)
    assert len(directions) <= COMBOS
    record = _quadratic(session, directions, _slopes(directions, 100.0))
    assert editor.next_state(session.state, record) is not None
    steps = [abs(step) for step in editor._dim_steps.values()]
    assert steps and max(steps) <= IMPORT_EPS * (1 + 1e-12)
    if controller == "r1":
        assert max(steps) == pytest.approx(IMPORT_EPS)


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_step_is_halved_once_when_the_loss_rises(tmp_path, controller):
    session, editor, directions = _setup(tmp_path, controller=controller, combos=COMBOS)
    slopes = _slopes(directions)
    snapshot = session.editor_snapshot()
    calls = []
    proposed = editor.next_state(
        session.state, _scripted(session, directions, slopes, [0.0, -1.0], calls)
    )
    assert proposed is not None and len(calls) == 2 * COMBOS + 2
    session.editor_state.load_tensors(snapshot)
    calls.clear()
    halved = editor.next_state(
        session.state, _scripted(session, directions, slopes, [0.0, 1.0, -1.0], calls)
    )
    assert halved is not None and len(calls) == 2 * COMBOS + 3
    expected = between(session.state, proposed, 0.5)
    for target, parts in expected.items():
        for key, value in parts.items():
            assert torch.allclose(halved[target][key], value, rtol=1e-6, atol=1e-9), (target, key)
    assert editor._accepted["halved"] is True
    assert "after halving the step" in editor.note
    assert float(session.editor_state.gains[-1]) == 1.0


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_no_write_when_the_halved_step_also_raises_the_loss(tmp_path, controller):
    session, editor, directions = _setup(tmp_path, controller=controller, combos=COMBOS)
    calls = []
    record = _steep(session, directions, _slopes(directions), 1e9, calls)
    before = _clone(session.state)
    gains = session.editor_state.gains.clone()
    commit_write(session, editor, record)
    assert len(calls) == 2 * COMBOS + 3
    for target, parts in before.items():
        for key, value in parts.items():
            assert torch.equal(session.state[target][key], value)
    assert torch.equal(session.editor_state.gains, gains)
    assert NOT_WRITTEN in editor.write_note()
    calls.clear()
    scripted = _scripted(session, directions, _slopes(directions), [0.0, 1.0, 1.0], calls)
    assert editor.next_state(session.state, scripted) is None
    assert len(calls) == 2 * COMBOS + 3
    assert NOT_WRITTEN in editor.write_note()


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_write_stores_the_measured_decrease(tmp_path, controller):
    session, editor, directions = _setup(tmp_path, controller=controller, combos=COMBOS)
    calls = []
    record = _quadratic(session, directions, _slopes(directions), calls)
    state = _clone(session.state)
    new = editor.next_state(session.state, record)
    assert new is not None and len(calls) == 2 * COMBOS + 2
    before = record.measure(ImportTrial(session, state, state))
    after = record.measure(ImportTrial(session, state, new))
    assert before - after > 0
    assert session.editor_state.gains.shape[0] == 1
    assert float(session.editor_state.gains[-1]) == pytest.approx(before - after, rel=1e-9)
    assert editor._accepted["halved"] is False
    assert "step not halved" in editor.note
    assert f"measured loss {before:.6g} before, {after:.6g} after" in editor.note


def test_report_splits_the_change_into_decay_and_moves(tmp_path):
    session, editor, directions = _setup(tmp_path, controller="r1", combos=COMBOS)
    record = _quadratic(session, directions, _slopes(directions))
    commit_write(session, editor, record)
    first = _clone(session.state)
    commit_write(session, editor, _quadratic(session, directions, _slopes(directions)))
    assert editor._accepted is not None and editor._accepted["halved"] is False
    neutral = session._neutral(first)
    decayed = _clone(first)
    for target, parts in decayed.items():
        for part in parts:
            if part not in TERM_FIXED:
                change = first[target][part] - neutral[target][part]
                parts[part] = neutral[target][part] + change * (1 - IMPORT_DECAY)
    decay = _hand_norm(session, first, decayed)
    moves = _hand_norm(session, decayed, session.state)
    assert decay > 0 and moves > 0
    assert editor._accepted["decay"] == pytest.approx(decay, rel=1e-5)
    assert editor._accepted["moves"] == pytest.approx(moves, rel=1e-5)
    split = editor._accepted
    assert f"Change: decay {split['decay']:.4g}, moves {split['moves']:.4g}." in editor.note


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_each_event_costs_two_k_plus_two_or_three_evaluations(tmp_path, controller):
    session, editor, directions = _setup(tmp_path, controller=controller, combos=COMBOS)
    for _ in range(3):
        calls = []
        commit_write(session, editor, _quadratic(session, directions, _slopes(directions), calls))
        assert len(calls) in (2 * COMBOS + 2, 2 * COMBOS + 3)


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_full_measure_never_reaches_the_new_rules(tmp_path, controller, monkeypatch):
    session, editor, directions = _setup(tmp_path, controller=controller, measure="full")
    record = _quadratic(session, directions, _slopes(directions))
    snapshot = session.editor_snapshot()
    first = editor.next_state(session.state, record)
    first_rules = {k: v.clone() for k, v in session.editor_state.rules.items()}
    first_history = {k: v.clone() for k, v in session.editor_state.history.items()}
    assert first is not None
    session.editor_state.load_tensors(snapshot)

    def fail(*_args, **_kwargs):
        raise AssertionError("the full path reached a rule of measure = compressed")

    for name in ("cap_step", "between", "subspace_gradient"):
        monkeypatch.setattr(live_imports, name, fail)
    for name in ("_accept", "_measure_subspace", "_halve_steps"):
        monkeypatch.setattr(live_imports.ImportEditor, name, fail)
    again = editor.next_state(session.state, record)
    assert again is not None
    for target, parts in first.items():
        for key, value in parts.items():
            assert torch.equal(again[target][key], value), (target, key)
    assert session.editor_state.rules.keys() == first_rules.keys()
    assert all(torch.equal(session.editor_state.rules[k], v) for k, v in first_rules.items())
    for key, value in first_history.items():
        again_row = session.editor_state.history[key]
        assert torch.equal(torch.isnan(again_row), torch.isnan(value))
        assert torch.equal(again_row.nan_to_num(0.0), value.nan_to_num(0.0))
    assert editor._accepted is None
    assert "Accept check" not in editor.note
