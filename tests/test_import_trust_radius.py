# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import torch
from test_import_rprop_integration import record, setup_editor

from kilora import live_imports
from kilora.editors import commit_write
from kilora.import_controller import TRUST_RATIO
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor, moves


def _clone(state):
    return {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}


def _random_event(session, seed):
    gen = torch.Generator().manual_seed(seed)
    target = next(iter(session._source.experts))
    ex = session._source.experts[target]
    session._source.experts[target] = dataclasses.replace(
        ex, exact={**ex.exact, 1: (torch.eye(2), torch.eye(2))}
    )
    session.state[target].pop("core_1", None)
    state = _clone(session.state)
    for n, (_target, parts) in enumerate(sorted(state.items())):
        for key in list(parts):
            if key.startswith("core_"):
                rank = parts[key].shape[0]
                size = 0.6 if n % 3 == 0 else 0.03
                parts[key] = torch.eye(rank) + size * torch.randn(rank, rank, generator=gen)
            elif key.startswith("strength_"):
                parts[key] = 1 + 0.05 * torch.randn(1, generator=gen)
    new = _clone(state)
    for n, (_target, parts) in enumerate(sorted(new.items())):
        for key in list(parts):
            if key.startswith("core_"):
                rank = parts[key].shape[0]
                parts[key] = parts[key] + 0.5 * torch.randn(rank, rank, generator=gen)
            elif key.startswith("strength_") and n % 2 == 0:
                parts[key] = parts[key] + 0.8 * torch.randn(1, generator=gen)
    return state, new, target


def _radius(editor, state, new, rows, all_moves, bisect, monkeypatch):
    state, new, rows = _clone(state), _clone(new), {k: v.clone() for k, v in rows.items()}
    with monkeypatch.context() as patch:
        if bisect:
            patch.setattr(live_imports, "_trust_root", lambda coefficients: None)
        counts = editor._trust_radius(
            state, new, rows, all_moves, lambda t, e: editor._pair_norm(state, t, e)
        )
    return new, rows, counts


def test_closed_form_matches_bisection(tmp_path, monkeypatch):
    for seed in range(4):
        folder = tmp_path / str(seed)
        folder.mkdir()
        session, editor, _ = setup_editor(folder)
        state, new, exact_target = _random_event(session, seed)
        all_moves = {m.id: m for m in moves(session, 2)}
        gen = torch.Generator().manual_seed(100 + seed)
        rows = {
            key: torch.tensor([1.0, 0.1, 0.1, 0.0, float(torch.randn(1, generator=gen))])
            for key in all_moves
        }
        session.editor_state.rules = {k: v.clone() for k, v in rows.items()}
        outside = [
            (t, e)
            for t, e in {(m.target, m.expert) for m in all_moves.values()}
            if (pair := editor._pair_norm(state, t, e)) is not None
            and pair[0] > TRUST_RATIO * pair[1]
        ]
        assert outside, "some pairs start outside the radius"
        closed_new, closed_rows, (closed, bisected) = _radius(
            editor, state, new, rows, all_moves, False, monkeypatch
        )
        assert closed > 0 and bisected == 0
        ref_new, ref_rows, (none, all_bisected) = _radius(
            editor, state, new, rows, all_moves, True, monkeypatch
        )
        assert none == 0 and all_bisected == closed
        for target, parts in ref_new.items():
            for key, value in parts.items():
                got = closed_new[target][key]
                scale = max(float(value.double().norm()), 1.0)
                assert float((got.double() - value.double()).norm()) <= 1e-6 * scale, key
        for key, row in ref_rows.items():
            assert abs(float(closed_rows[key][4]) - float(row[4])) <= 1e-6 * max(
                abs(float(row[4])), 1.0
            )
        for target, expert in {(m.target, m.expert) for m in all_moves.values()}:
            pair = editor._pair_norm(closed_new, target, expert)
            if pair is not None and pair[1] > 0:
                assert pair[0] / pair[1] <= TRUST_RATIO + 1e-6
        strength_only = editor._pair_norm(closed_new, exact_target, 1)
        assert strength_only is not None and strength_only[1] == 1.0


def test_trust_root_quadratic_and_degenerate():
    assert abs(live_imports._trust_root([-0.5, 1.75, 1.0, 0.0, 0.0]) - 0.25) < 1e-12
    assert abs(live_imports._trust_root([-1.0, 4.0, 0.0, 0.0, 0.0]) - 0.25) < 1e-12
    assert live_imports._trust_root([0.0, 0.0, 0.0, 0.0, 0.0]) is None
    assert live_imports._trust_root([-1.0, 0.0, 0.0, 0.0, 0.0]) is None
    assert live_imports._trust_root([float("nan"), 1.0, 0.0, 0.0, 0.0]) is None
    assert live_imports._trust_root([-1.0, 0.5, 0.0, 0.0, 0.0]) is None


def _core_and_strength(session):
    session.set_options(core_dims=1)
    choices = moves(session, core_dims=1)
    core = next(move for move in choices if move.part == "core")
    strength = next(
        move
        for move in choices
        if move.part == "strength" and (move.target, move.expert) == (core.target, core.expert)
    )
    session.moves = lambda: [core, strength]
    return core, strength


def test_grams_build_once_per_session(tmp_path, monkeypatch):
    calls = {}
    build = live_imports._build_grams

    def counted(ex, expert, rank):
        key = (id(ex), expert, rank)
        calls[key] = calls.get(key, 0) + 1
        return build(ex, expert, rank)

    monkeypatch.setattr(live_imports, "_build_grams", counted)
    session, editor, _ = setup_editor(tmp_path)
    core, _ = _core_and_strength(session)
    for _ in range(4):
        commit_write(session, editor, record(session, core))
        for row in session.editor_state.rules.values():
            row[1:3] = torch.tensor([1.0, 1.0])
        for target, ex in session._source.experts.items():
            for expert in range(ex.down.shape[0]):
                editor._pair_norm(session.state, target, expert)
    assert calls
    assert all(n == 1 for n in calls.values()), calls
    change, source = editor._pair_norm(session.state, core.target, core.expert)
    assert change / source <= TRUST_RATIO + 1e-6


def test_timing_note_splits_the_event(tmp_path, monkeypatch):
    session, editor, _ = setup_editor(tmp_path)
    core, _ = _core_and_strength(session)
    monkeypatch.delenv("KILORA_IMPORT_TIMING", raising=False)
    editor.next_state(session.state, record(session, core))
    assert "Controller:" not in editor.note
    monkeypatch.setenv("KILORA_IMPORT_TIMING", "1")
    editor = ImportEditor(session)
    editor.next_state(session.state, record(session, core))
    assert "Controller: measurement" in editor.note
    assert "step" in editor.note and "trust radius" in editor.note
    SESSIONS.clear()
