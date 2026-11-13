# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from test_import_edit_dimensions import _session

from kilora.editors import commit_write
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor
from kilora.recording import GenerationRecord


def setup_editor(tmp_path):
    SESSIONS.clear()
    session = _session(tmp_path)
    assert session.controller == "r1"
    for target, source in session._source.experts.items():
        assert torch.equal(session.sets()[0].experts[target].up, source.up)
    session.learning = True
    session.set_options(core_dims=0, controller="rprop", measure="full")
    choices = session.moves()
    session.moves = lambda: choices[:1]
    editor = ImportEditor(session)
    return session, editor, choices


def record(session, move, sign=1, refused=False):
    ex = session._source.experts[move.target]
    source = ex.up[move.expert, 0].float()

    def measure(trial):
        delta = trial.du.get(move.target)
        if delta is None:
            return 0.0
        shifted = delta.reshape_as(ex.up)[move.expert, 0].float()
        return sign * float((shifted * source).sum() / source.square().sum())

    def guard(trial):
        return {"gap": 0.0 if refused and trial is not None else 1.0, "spread": 1.0}

    return GenerationRecord(measure=measure, guard_score=guard)


def test_sign_guard_save_and_undo(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    move = choices[0]
    safe = record(session, move)
    commit_write(session, editor, safe)
    first = session.editor_state.rules[move.id].clone()
    commit_write(session, editor, safe)
    second = session.editor_state.rules[move.id].clone()
    assert abs(float(second[1] / first[1]) - 1.2) < 1e-6
    name = session.save_point("rprop")
    before = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    commit_write(session, editor, record(session, move, sign=-1))
    flipped = session.editor_state.rules[move.id]
    assert abs(float(flipped[1] / second[1]) - 0.5) < 1e-6
    assert all(torch.equal(v, session.state[t][k]) for t, p in before.items() for k, v in p.items())
    session.undo()
    assert torch.equal(session.editor_state.rules[move.id], second)
    commit_write(session, editor, record(session, move, refused=True))
    assert all(torch.equal(v, session.state[t][k]) for t, p in before.items() for k, v in p.items())
    assert abs(float(session.editor_state.rules[move.id][1] / second[1]) - 0.5) < 1e-6
    session.load(name)
    assert torch.equal(session.editor_state.rules[move.id], second)
    session.editor_state = None
    session.load(name)
    ImportEditor(session)
    assert torch.equal(session.editor_state.rules[move.id], second)


def test_decay_and_trust(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    first, other = choices[:2]
    commit_write(session, editor, record(session, first))
    session.editor_state.rules[first.id][1:3] = torch.tensor([0.6, 0.075])
    for _ in range(6):
        commit_write(session, editor, record(session, first))
        change, source = editor._pair_norm(session.state, first.target, first.expert)
        assert change / source <= 0.5 + 1e-6
    assert (
        abs(
            float(session.editor_state.rules[first.id][1])
            - 8 * float(session.editor_state.rules[first.id][2])
        )
        < 1e-6
    )
    amount = float(session.editor_state.rules[first.id][4])
    session.moves = lambda: [other]
    for _ in range(5):
        commit_write(session, editor, record(session, other))
    assert abs(float(session.editor_state.rules[first.id][4]) - amount * 0.9) < 1e-5


def test_default_matches_explicit_r1(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    session.controller = "r1"
    before = session.editor_snapshot()
    default = editor.next_state(session.state, record(session, choices[0]))
    session.editor_state.load_tensors(before)
    session.set_options(controller="r1")
    explicit = editor.next_state(session.state, record(session, choices[0]))
    assert all(torch.equal(v, explicit[t][k]) for t, p in default.items() for k, v in p.items())


def test_strength_only_exact_and_fixed(tmp_path):
    import dataclasses
    from pathlib import Path

    from kilora.sections import Section

    for kind in ("exact", "fixed"):
        folder = tmp_path / kind
        folder.mkdir()
        session, editor, choices = setup_editor(folder)
        move = choices[0]
        target, expert = move.target, move.expert
        del session.state[target][f"core_{expert}"]
        if kind == "exact":
            ex = session._source.experts[target]
            session._source.experts[target] = dataclasses.replace(
                ex, exact={expert: (torch.eye(2), torch.eye(2))}
            )

            def measure(trial, target=target, expert=expert):
                return float(trial.exact[target][expert][0].sum())
        else:
            session._section_pairs.add((target, expert))
            session._sections.append(
                Section(
                    "fixed-delta",
                    Path(session.sources[expert]["path"]).stem,
                    {target: torch.eye(2)},
                    {"lycoris": True},
                )
            )

            def measure(trial, key=(session.mod_id, target, expert)):
                return trial.fixed_changes[key]

        commit_write(session, editor, GenerationRecord(measure=measure))
        assert float(session.state[target][f"strength_{expert}"]) < 1.0
        change, source = editor._pair_norm(session.state, target, expert)
        assert change / source <= 0.5 + 1e-6


def test_core_and_strength_share_trust_radius(tmp_path):
    from kilora.live_imports import ImportTrial

    session, editor, _ = setup_editor(tmp_path)
    session.set_options(core_dims=1)
    from kilora.live_imports import moves

    choices = moves(session, core_dims=1)
    core = next(move for move in choices if move.part == "core")
    strength = next(
        move
        for move in choices
        if move.part == "strength" and (move.target, move.expert) == (core.target, core.expert)
    )
    session.moves = lambda: [core, strength]
    state = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    core.apply(state, 0.01)
    probe = ImportTrial(session, session.state, state).du[core.target]

    def measure(trial):
        delta = trial.du.get(core.target)
        return 0.0 if delta is None else float((delta * probe).sum())

    record = GenerationRecord(measure=measure)
    commit_write(session, editor, record)
    for row in session.editor_state.rules.values():
        row[1:3] = torch.tensor([1.0, 1.0])
    for _ in range(4):
        commit_write(session, editor, record)
        ex = session._source.experts[core.target]
        matrix = session.state[core.target][f"core_{core.expert}"]
        strength_value = session.state[core.target][f"strength_{core.expert}"]
        rank = matrix.shape[0]
        a = ex.down[core.expert, 0, :rank].float()
        b = ex.up[core.expert, 0, :rank].float().T
        change = b @ (strength_value * matrix - torch.eye(rank)) @ a
        source = b @ a
        assert float(change.norm() / source.norm()) <= 0.5 + 1e-6


def test_old_baseline_history_loads(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    session.set_options(controller="r1")
    commit_write(session, editor, record(session, choices[0]))
    before = {key: value.clone() for key, value in session.editor_state.history.items()}
    old = {key: value.clone() for key, value in before.items()}
    session.editor_state.reset()
    assert session._load_editor_state(old, "import-r1") == ""
    for key in before:
        torch.testing.assert_close(session.editor_state.history[key], before[key], equal_nan=True)


def test_idle_high_dimension_survives_option_change(tmp_path):
    from kilora.live_imports import ImportTrial, moves

    session, editor, _ = setup_editor(tmp_path)
    session.set_options(core_dims=3)
    core = next(m for m in moves(session, 3) if m.id.endswith(":core:2:2"))
    session.moves = lambda: [core]
    trial_state = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    core.apply(trial_state, 0.01)
    probe = ImportTrial(session, session.state, trial_state).du[core.target]

    def measure(trial):
        delta = trial.du.get(core.target)
        return 0.0 if delta is None else float((delta * probe).sum())

    commit_write(session, editor, GenerationRecord(measure=measure))
    amount = float(session.editor_state.rules[core.id][4])
    assert amount != 0.0
    session.set_options(core_dims=0)
    other = next(m for m in moves(session, 0) if m.expert != core.expert)
    session.moves = lambda: [other]
    for _ in range(5):
        commit_write(session, editor, record(session, other))
    assert abs(float(session.editor_state.rules[core.id][4]) - amount * 0.9) < 1e-6


def test_default_forward_is_bit_identical(tmp_path):
    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets

    SESSIONS.clear()
    session = _session(tmp_path)
    target = next(iter(session.state))
    outputs = []
    for setting in (session._source, session.sets()[0]):
        plan = plan_targets([ScopedSet(setting, import_score="derived")], 2)[target]
        layer = plan.to(torch.device("cpu"), torch.float32).sets[0]
        torch.manual_seed(42)
        x = torch.randn(1, 3, layer.down.shape[-1])
        y = torch.randn(1, 3, layer.up.shape[-1])
        with torch.no_grad():
            outputs.append(compute.run_layer(x, y, [layer], 1, target=target))
    assert torch.equal(*outputs)
