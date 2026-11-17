# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch
from test_import_edit_dimensions import _session

from kilora import live_imports
from kilora.editors import write_outcome
from kilora.live import SESSIONS
from kilora.live_experts import (
    LIVE_EXPERT_SETTINGS,
    NO_TRAVERSED,
    RouteRecord,
    active_rows,
    expert_policy,
    history_from_tensors,
    history_tensors,
    mask_terms,
    record_history,
    restrict_state,
    target_moves,
    write_target,
)
from kilora.live_imports import ImportEditor, session_for_shared_edits
from kilora.prediction_write import Term


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _routes(session, mass_by_expert, keeps=None):
    routes = RouteRecord()
    for target in session.state:
        routes.mass[target] = torch.tensor(mass_by_expert, dtype=torch.float32)
        routes.ran.add(target)
        if keeps is not None:
            routes.keeps[target] = keeps(target)
    return routes


def _changed(old, new):
    pairs = set()
    for target, parts in new.items():
        for key, value in parts.items():
            before = old.get(target, {}).get(key)
            if before is not None and not torch.equal(before, value):
                pairs.add((target, int(key.rpartition("_")[2])))
    return pairs


def _blank(state):
    strengths = [
        v for parts in state.values() for k, v in parts.items() if k.startswith("strength_")
    ]
    return bool(strengths) and all(not bool(v.any()) for v in strengths)


def _counting(monkeypatch):
    seen = []
    real = live_imports.ImportTrial

    class Counting(real):
        def __init__(self, session, old, new):
            super().__init__(session, old, new)
            if _blank(old):
                return
            seen.append(_changed(old, new))

    monkeypatch.setattr(live_imports, "ImportTrial", Counting)
    return seen


def _record(routes, calls):
    def loss(_trial):
        calls.append(1)
        return float(len(calls)) * 1e-3

    return types.SimpleNamespace(
        measure=loss,
        route_record=routes,
        stats=None,
        guard_score=None,
    )


def test_settings_have_their_defaults(tmp_path):
    session = _session(tmp_path)
    assert LIVE_EXPERT_SETTINGS == {
        "live_expert_mass": 0.05,
        "live_expert_top": 4,
        "live_expert_history": 16,
        "live_expert_copies": 2,
        "write_measure": "forward",
        "forward_draws": 4,
        "forward_sigma": 0.05,
        "forward_chunk": 8,
        "write_resolution": 512,
        "probe_rows": True,
        "probe_call": "separate",
        "probe_draws": 0,
        "probe_budget": 2,
    }
    for name, value in LIVE_EXPERT_SETTINGS.items():
        assert getattr(session, name) == value
    session.set_options(live_expert_mass=0.2, live_expert_top=1, live_expert_copies=0)
    assert (session.live_expert_mass, session.live_expert_top) == (0.2, 1)
    assert session.live_expert_copies == 0
    with pytest.raises(Exception, match="live_expert_top"):
        session.set_options(live_expert_top=0)
    session.set_options(write_measure="candidates", forward_draws=8, forward_sigma=0.1)
    assert (session.write_measure, session.forward_draws) == ("candidates", 8)
    assert session.forward_sigma == 0.1
    for bad in ({"write_measure": "full"}, {"forward_sigma": 0.0}, {"forward_chunk": 0}):
        with pytest.raises(Exception, match=next(iter(bad))):
            session.set_options(**bad)
    session.set_options(write_resolution=0)
    assert session.write_resolution == 0
    session.set_options(write_resolution=770)
    assert session.write_resolution == 768
    for bad in (64, 5000, -1, 1.5, True):
        with pytest.raises(Exception, match="write_resolution"):
            session.set_options(write_resolution=bad)


def test_write_target_keeps_the_traversed_experts(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    routes = _routes(session, [0.98, 0.02])
    target = write_target(session, routes)
    assert target and all(experts == [0] for experts in target.values())
    session.set_options(live_expert_mass=0.01)
    assert all(experts == [0, 1] for experts in write_target(session, routes).values())
    session.set_options(live_expert_top=1)
    assert all(experts == [0] for experts in write_target(session, routes).values())


def test_untraversed_moves_are_never_measured(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.set_options(measure="full", core_dims=1)
    editor = ImportEditor(session)
    routes = _routes(session, [1.0, 0.0])
    seen = _counting(monkeypatch)
    calls = []
    editor._next_unmerged(session.state, _record(routes, calls))
    measured = [pairs for pairs in seen if pairs]
    assert measured
    assert all(expert == 0 for pairs in measured for _target, expert in pairs)
    expected = [m for m in session.moves() if m.expert == 0]
    assert len(calls) == 2 * len(expected)
    assert f"Candidates: {len(expected)}" in editor.note


def test_zero_read_rows_are_never_candidates(tmp_path):
    session = _session(tmp_path)
    session.set_options(core_dims=4)
    session.align_for_moves()

    def keeps(target):
        rank = session._source.experts[target].down.shape[2]
        row = torch.full((2, rank), 0.5)
        row[0, 1:] = 1.0
        return row

    routes = _routes(session, [1.0, 0.0], keeps)
    target = write_target(session, routes)
    kept = target_moves(session, session.moves(), target, routes)
    cores = [m for m in kept if m.part == "core"]
    assert cores
    for move in cores:
        read, _output = active_rows(
            session, routes, move.target, move.expert, move.direction.shape[0]
        )
        assert read.sum() == 1
        assert torch.count_nonzero(move.direction[:, 1:]) == 0


def test_zero_down_rows_are_never_read(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    target = next(iter(session.state))
    ex = session._source.experts[target]
    rank = ex.down.shape[2]
    read, output = active_rows(session, None, target, 0, rank)
    down = ex.down[0, 0, :rank].abs().amax(-1) > 0
    assert torch.equal(read, down & output)


def test_an_empty_target_ends_with_no_moves_and_the_reason(tmp_path, monkeypatch):
    session = _session(tmp_path)
    editor = ImportEditor(session)
    routes = RouteRecord()
    for target in session.state:
        routes.mass[target] = torch.zeros(2)
        routes.ran.add(target)
    calls = []
    seen = _counting(monkeypatch)
    assert editor._next_unmerged(session.state, _record(routes, calls)) is None
    assert not calls and not seen
    assert editor.note.startswith("live write: no moves (no traversed expert).")
    line = write_outcome(editor, {"kind": "no_write"})
    assert line.startswith(f"live write: no moves ({NO_TRAVERSED}).")
    assert "Candidates: 0" in line


def test_route_history_and_policy(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    session.set_options(live_expert_history=2)
    first = _routes(session, [0.0, 1.0])
    record_history(session, first)
    this = _routes(session, [1.0, 0.0])
    target = write_target(session, this)
    policy = expert_policy(session, target, this)
    assert set(policy.values()) == {"in place"}
    shared = _routes(session, [0.5, 0.5])
    record_history(session, shared)
    policy = expert_policy(session, target, this)
    assert set(policy.values()) == {"copy"}
    record_history(session, this)
    assert len(session.route_history) == 2


def test_history_survives_export_and_load(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    session.set_options(live_expert_mass=0.1, live_expert_top=3)
    record_history(session, _routes(session, [0.25, 0.75]))
    record_history(session, _routes(session, [1.0, 0.0]))
    tensors = history_tensors(session)
    back = history_from_tensors(tensors)
    assert len(back) == 2
    target = next(iter(session.state))
    session.state[target]["strength_0"] = torch.full((1,), 1.25)
    path = tmp_path / "out.kmod"
    session.export(path)
    SESSIONS.clear()
    loaded = session_for_shared_edits(path, None, build=lambda _sources: _session(tmp_path))
    assert loaded is not session
    assert torch.equal(loaded.state[target]["strength_0"], torch.full((1,), 1.25))
    assert len(loaded.route_history) == 2
    for a, b in zip(loaded.route_history, session.route_history, strict=True):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert loaded.live_expert_mass == 0.1 and loaded.live_expert_top == 3


def test_path_merge_folds_only_the_target(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    before = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    after = {t: {k: v + 0.1 for k, v in p.items()} for t, p in before.items()}
    routes = _routes(session, [1.0, 0.0])
    target = write_target(session, routes)
    merged = restrict_state(session, before, after, target, routes)
    changed = _changed(before, merged)
    assert changed and all(expert == 0 for _target, expert in changed)


def test_prediction_frames_use_only_the_target(tmp_path):
    session = _session(tmp_path)
    editor = ImportEditor(session)
    weights = {
        target: torch.randn(ex.up.shape[-1], ex.down.shape[-1])
        for target, ex in session._source.experts.items()
    }
    live_imports.import_source_frames(session, weights)
    routes = _routes(session, [1.0, 0.0])
    target = write_target(session, routes)
    editor._routes = routes
    by_target, responses, _p_out, _priors = editor.prediction_setup(session.state, target)
    frames = live_imports.import_source_frames(session)
    for target_name, node in responses:
        assert frames[target_name][node][0] == 0
    assert all(m.expert in (-1, 0) for moves_ in by_target.values() for m in moves_)
    full, _r, _p, _q = editor.prediction_setup(session.state)
    terms = [
        Term("x", layer, torch.ones(3), torch.ones(3, len(moves_)), 1.0)
        for layer, moves_ in full.items()
    ]
    masked = mask_terms(terms, full, target)
    for term, layer in zip(masked, full, strict=True):
        for column, move in enumerate(full[layer]):
            keep = move.expert < 0 or move.expert in target.get(layer, ())
            assert bool(term.response[:, column].abs().sum() > 0) == keep


def test_a_zero_write_render_keeps_the_imported_set(tmp_path):
    session = _session(tmp_path)
    editor = ImportEditor(session)
    routes = _routes(session, [1.0, 0.0])
    editor._next_unmerged(session.state, _record(routes, []))
    for target, ex in session.sets()[0].experts.items():
        assert ex is session._plain[target]


def test_the_clean_read_cache_gives_the_read_keeps(tmp_path):
    from kilora.forward_estimate import ForwardEstimate

    session = _session(tmp_path)
    session.set_options(core_dims=4)
    session.align_for_moves()
    target = next(iter(session.state))
    ex = session._source.experts[target]
    rank = int(session.state[target]["core_0"].shape[0])
    estimate = ForwardEstimate(session, session.state, draws=4, sigma=0.05, chunk=4, seed=1)
    estimate.seen = {target}
    live = torch.zeros(rank, dtype=torch.bool)
    live[0] = True
    estimate._live_rows[(target, 0)] = live
    routes = _routes(session, [1.0, 0.0])
    estimate.add_keeps(routes)
    keeps = routes.keeps[target]
    assert keeps.shape == (int(ex.down.shape[0]), int(ex.down.shape[2]))
    assert float(keeps[0, 0]) == 0.5 and bool((keeps[0, 1:rank] == 1.0).all())
    read, _output = active_rows(session, routes, target, 0, rank)
    assert not bool(read[1:].any())
    assert target in routes.ran
