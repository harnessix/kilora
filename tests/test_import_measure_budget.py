# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_edit_dimensions import _session

from kilora.editors import commit_write
from kilora.import_controller import combinations, compressed_update
from kilora.live import SESSIONS
from kilora.live_imports import IMPORT_EPS, ImportEditor, ImportHistory, ImportTrial
from kilora.recording import GenerationRecord


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _setup(tmp_path, controller="r1", measure="compressed", combos=8):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=0, controller=controller, measure=measure, measure_combos=combos)
    directions = session.moves()
    assert len(directions) >= 2
    editor = ImportEditor(session)
    return session, editor, directions


def _coordinate(session, trial, move):
    ex = session._source.experts[move.target]
    delta = trial.du.get(move.target)
    if delta is None:
        return 0.0
    source = ex.up[move.expert, 0].float()
    shifted = delta.reshape_as(ex.up)[move.expert, 0].float()
    return float((shifted * source).sum() / source.square().sum())


def _quadratic(session, directions, slopes, calls=None):
    def measure(trial):
        if calls is not None:
            calls.append(1)
        x = torch.tensor([_coordinate(session, trial, m) for m in directions], dtype=torch.float64)
        return float(slopes @ x + 0.5 * x.square().sum())

    return GenerationRecord(measure=measure, guard_score=lambda _: {"gap": 1.0, "spread": 1.0})


def _relative_error(estimated, directions, slopes):
    est = torch.tensor([estimated[m.id] for m in directions], dtype=torch.float64)
    return float((est - slopes).norm() / slopes.norm())


def test_estimate_approaches_exact_slopes_as_combinations_grow(tmp_path):
    errors = {}
    for combos in (4, 64):
        SESSIONS.clear()
        folder = tmp_path / str(combos)
        folder.mkdir()
        session, editor, directions = _setup(folder, combos=combos)
        slopes = torch.linspace(-1.0, 1.5, len(directions), dtype=torch.float64)
        record = _quadratic(session, directions, slopes)
        first = editor._measure_compressed(session.state, record, directions)
        errors[combos] = _relative_error(first, directions, slopes)
        for _ in range(3):
            estimated = editor._measure_compressed(session.state, record, directions)
        errors[f"{combos}-later"] = _relative_error(estimated, directions, slopes)
    assert errors[64] < errors[4]
    assert errors["64-later"] < 1e-2


def test_estimator_on_many_moves():
    torch.manual_seed(0)
    n = 32
    slopes = torch.randn(n, dtype=torch.float64)
    ids = [f"m{i}" for i in range(n)]
    errors = {}
    for combos in (4, 64):
        prior = None
        for event in range(4):
            psi = combinations(combos, n, 1000 + event)
            assert torch.allclose(psi.norm(dim=1), torch.ones(combos, dtype=torch.float64))
            assert torch.allclose(psi.abs() * n**0.5, torch.ones_like(psi))
            est, prior = compressed_update(ids, psi, psi @ slopes, prior)
        errors[combos] = float((est - slopes).norm() / slopes.norm())
    assert errors[64] < errors[4]
    assert errors[64] < 1e-2


@pytest.mark.parametrize("controller", ["r1", "rprop"])
@pytest.mark.parametrize("pack", [False, True])
def test_write_costs_two_k_evaluations_and_the_accept_check(tmp_path, controller, pack):
    combos = 5
    session, editor, directions = _setup(tmp_path, controller=controller, combos=combos)
    calls = []
    slopes = torch.linspace(-1.0, 1.0, len(directions), dtype=torch.float64)
    record = _quadratic(session, directions, slopes, calls)
    if not pack:
        assert editor.next_state(session.state, record) is not None
        assert len(calls) == 2 * combos + 2
        return

    class PackMove:
        def __init__(self, key):
            self.id = key
            self.applied = []

        def apply(self, state, delta):
            self.applied.append(delta)

    extra = [PackMove(f"pack:{i}") for i in range(7)]
    estimated = editor._measure_compressed(session.state, record, [*directions, *extra])
    assert len(calls) == 2 * combos
    assert set(estimated) == {m.id for m in [*directions, *extra]}
    assert all(len(m.applied) == 2 * combos for m in extra)
    calls.clear()
    assert editor.next_state(session.state, record) is not None
    assert len(calls) == 2 * combos + 2


def test_full_measure_keeps_the_per_move_path(tmp_path):
    session, editor, directions = _setup(tmp_path, measure="full", combos=3)
    calls = []
    slopes = torch.linspace(-1.0, 1.0, len(directions), dtype=torch.float64)
    record = _quadratic(session, directions, slopes, calls)
    before = session.editor_snapshot()
    first = editor.next_state(session.state, record)
    assert len(calls) == 2 * len(directions)
    assert session.editor_state.compressed is None
    assert session.editor_state.measure_events == 0

    def clone(state):
        return {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}

    for move in directions:
        plus, minus = clone(session.state), clone(session.state)
        move.apply(plus, IMPORT_EPS)
        move.apply(minus, -IMPORT_EPS)
        slope = (
            record.measure(ImportTrial(session, session.state, plus))
            - record.measure(ImportTrial(session, session.state, minus))
        ) / (2 * IMPORT_EPS)
        assert torch.equal(session.editor_state.history[move.id][-1], torch.tensor([slope])[0])
    session.editor_state.load_tensors(before)
    session.editor_state.compressed = {
        "ids": [m.id for m in directions],
        "slopes": torch.ones(len(directions), dtype=torch.float64),
        "cov": torch.eye(len(directions), dtype=torch.float64),
        "noise": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    session.set_options(measure_combos=50)
    again = editor.next_state(session.state, record)
    assert all(torch.equal(v, again[t][k]) for t, p in first.items() for k, v in p.items())


def _compressed_equal(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a["ids"] == b["ids"]
    for name in ("slopes", "cov", "noise"):
        assert torch.equal(a[name].double(), b[name].double())


def test_estimator_state_round_trips_save_load_and_undo(tmp_path):
    session, editor, directions = _setup(tmp_path, controller="rprop")
    slopes = torch.linspace(-1.0, 1.0, len(directions), dtype=torch.float64)
    record = _quadratic(session, directions, slopes)
    commit_write(session, editor, record)
    store = session.editor_state
    assert store.compressed is not None and store.measure_events == 1
    copy = ImportHistory(directions)
    copy.load_tensors(session.editor_snapshot())
    _compressed_equal(copy.compressed, store.compressed)
    assert copy.measure_events == store.measure_events

    first = {
        "ids": list(store.compressed["ids"]),
        **{n: store.compressed[n].clone() for n in ("slopes", "cov", "noise")},
    }
    name = session.save_point("budget")
    commit_write(session, editor, record)
    assert session.editor_state.measure_events == 2
    session.undo()
    assert session.editor_state.measure_events == 1
    _compressed_equal(session.editor_state.compressed, first)
    commit_write(session, editor, record)
    session.load(name)
    assert session.editor_state.measure_events == 1
    _compressed_equal(session.editor_state.compressed, first)
    session.editor_state.reset()
    assert session.editor_state.compressed is None
    assert session.editor_state.measure_events == 0


def test_measure_settings_are_checked(tmp_path):
    session = _session(tmp_path)
    assert session.measure == "compressed"
    assert session.measure_combos == 8
    with pytest.raises(Exception, match="measure must be"):
        session.set_options(measure="partly")
    with pytest.raises(Exception, match="measure_combos"):
        session.set_options(measure_combos=0)
