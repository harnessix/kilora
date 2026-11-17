# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import pytest
import torch
from test_import_edit_dimensions import _session

from kilora.forward_estimate import ForwardEstimate
from kilora.live import SESSIONS, LearnResult
from kilora.live_imports import ImportEditor
from kilora.write_estimates import (
    accumulate_write_estimate,
    copy_write_state,
    reset_write_estimates,
)


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _estimate(state, gradients):
    return SimpleNamespace(
        state=copy_write_state(state),
        target={
            target: [expert for t, expert in gradients if t == target] for target, _ in gradients
        },
        gradients=lambda: gradients,
    )


def test_mean_accepts_a_learnable_direction_after_individual_estimates_fail(monkeypatch, caplog):
    import kilora.live_imports as imports

    state = {"layer": {"core_0": torch.zeros(1, 2, dtype=torch.float64)}}
    session = SimpleNamespace(state=state, editor_state=SimpleNamespace(add_gain=lambda _: None))
    editor = object.__new__(ImportEditor)
    editor.session = session
    editor._forward = None
    editor._decayed = None
    editor._change_norm = lambda old, new: float(
        (new["layer"]["core_0"] - old["layer"]["core_0"]).norm()
    )
    monkeypatch.setattr(imports, "ImportTrial", lambda _session, _old, new: new)

    def measure(candidate):
        x, y = candidate["layer"]["core_0"][0]
        return float((x + 1).square() + 100 * y.square())

    record = SimpleNamespace(measure=measure)
    caplog.set_level(logging.INFO)
    with torch.no_grad():
        for value in (3.0, -3.0):
            gradient = torch.tensor([[1.0, value]], dtype=torch.float64)
            raw = {"layer": {"core_0": -0.1 * gradient}}
            editor._estimate_note = ""
            assert editor._accept(state, record, raw, {}) is None
            estimate = _estimate(state, {("layer", 0): (gradient, 0.0)})
            note = accumulate_write_estimate(session, estimate)
        editor._forward = estimate
        editor._estimate_note = note
        averaged = {"layer": {"core_0": -0.1 * estimate.mean_gradients[("layer", 0)][0]}}
        accepted = editor._accept(state, record, averaged, {})
    assert accepted is not None and measure(accepted) < measure(state)
    assert "over 2 estimates" in note and "floor" in note
    assert "retried at a scale of 0.5" in caplog.text
    assert "candidate refused" in caplog.text and "candidate accepted" in caplog.text
    assert "loss 1 ->" in caplog.text


def test_mean_is_unweighted_per_target_and_does_not_count_a_record_twice():
    state = {"a": {"core_0": torch.zeros(1, 1)}, "b": {"core_0": torch.zeros(1, 1)}}
    session = SimpleNamespace(state=state)
    for gradients in (
        {("a", 0): (torch.tensor([[2.0]]), 2.0), ("b", 0): (torch.tensor([[8.0]]), 8.0)},
        {("a", 0): (torch.tensor([[6.0]]), 6.0)},
        {("b", 0): (torch.tensor([[4.0]]), 4.0)},
    ):
        estimate = _estimate(state, gradients)
        accumulate_write_estimate(session, estimate)
        accumulate_write_estimate(session, estimate)
    assert session._write_estimates["a"][1] == session._write_estimates["b"][1] == 2
    core, strength = estimate.mean_gradients[("b", 0)]
    assert core.item() == strength == 6
    assert all(
        not value[0].requires_grad and value[0].device.type == "cpu"
        for value in estimate.mean_gradients.values()
    )


@pytest.mark.parametrize("change", ["weights", "revision", "branch", "signature"])
def test_mean_resets_when_its_state_or_target_changes(change):
    state = {"layer": {"core_0": torch.zeros(1, 1)}}
    forest = SimpleNamespace(current="a")
    session = SimpleNamespace(state=state, revision=1, editor_state=SimpleNamespace(forest=forest))
    gradients = {("layer", 0): (torch.ones(1, 1), 1.0)}
    accumulate_write_estimate(session, _estimate(state, gradients))
    if change == "weights":
        state["layer"]["core_0"] += 1
    elif change == "revision":
        session.revision += 1
    elif change == "branch":
        forest.current = "b"
    else:
        gradients = {("layer", 1): (torch.ones(1, 1), 1.0)}
    accumulate_write_estimate(session, _estimate(state, gradients))
    assert session._write_estimates["layer"][1] == 1


def test_commit_and_undo_clear_the_mean_even_when_weights_return(tmp_path):
    session = _session(tmp_path)
    ImportEditor(session)
    target = next(iter(session.state))
    gradients = {(target, 0): (torch.ones_like(session.state[target]["core_0"]), 1.0)}
    accumulate_write_estimate(session, _estimate(session.state, gradients))
    session.pending = copy_write_state(session.state)
    session._pending_info = LearnResult(1, 1.0, 0.0, False, False, "test write")
    session.commit()
    assert session._write_estimates == {} and session._write_estimate_state is None
    accumulate_write_estimate(session, _estimate(session.state, gradients))
    session.undo()
    assert session._write_estimates == {} and session._write_estimate_state is None
    accumulate_write_estimate(session, _estimate(session.state, gradients))
    assert session._write_estimates[target][1] == 1


def test_live_writes_use_shared_mean_before_projecting_slopes(tmp_path, monkeypatch):
    from test_forward_estimate import LinearMeasure, _Record

    session = _session(tmp_path)
    session.set_options(core_dims=1, forward_draws=4)
    editor = ImportEditor(session)
    observed = []
    slopes = ForwardEstimate.slopes

    def inspect(estimate, directions):
        assert hasattr(estimate, "mean_gradients")
        observed.append(estimate.mean_response_note)
        return slopes(estimate, directions)

    monkeypatch.setattr(ForwardEstimate, "slopes", inspect)
    with torch.no_grad():
        for _ in range(2):
            editor._next_unmerged(session.state, _Record(LinearMeasure(session)))
    assert "over 1 estimates" in observed[0]
    assert "over 2 estimates" in observed[1]
    reset_write_estimates(session)
    assert session._write_estimates == {}


@pytest.mark.parametrize("controller", ["r1", "rprop"])
@pytest.mark.parametrize("measure", ["full", "compressed"])
def test_imported_candidate_measurements_also_use_the_shared_mean(tmp_path, controller, measure):
    from test_import_measure_budget import _quadratic, _setup

    session, editor, directions = _setup(tmp_path, controller=controller, measure=measure)
    session.set_options(write_measure="candidates")
    slopes = torch.linspace(-1.0, 1.5, len(directions), dtype=torch.float64)
    snapshot = session.editor_snapshot()
    with torch.no_grad():
        for attempt in range(2):
            editor._next_unmerged(session.state, _quadratic(session, directions, slopes))
            assert session._write_estimates
            assert all(row[1] == attempt + 1 for row in session._write_estimates.values())
            session.editor_state.load_tensors(snapshot)
