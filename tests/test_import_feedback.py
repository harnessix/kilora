# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_edit_dimensions import _session

from kilora.bake import ScopedSet, plan_targets
from kilora.editors import commit_write
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor
from kilora.recording import GenerationRecord


@pytest.fixture(autouse=True)
def clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_feedback_and_guard(tmp_path):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=0)
    move = session.moves()[0]
    session.moves = lambda: [move]
    target, expert = move.target, move.expert
    source = session._source.experts[target].up[expert, 0].float()
    editor = ImportEditor(session)

    def loss(trial):
        delta = trial.du.get(target)
        shift = (
            0.0
            if delta is None
            else float(
                (delta.reshape_as(session._source.experts[target].up)[expert, 0] * source).sum()
                / source.square().sum()
            )
        )
        strength = float(session.state[target][f"strength_{expert}"]) + shift
        return (strength - 1.5) ** 2

    safe = GenerationRecord(measure=loss, guard_score=lambda _: {"gap": 1.0, "spread": 1.0})
    before = float(session.state[target][f"strength_{expert}"])
    commit_write(session, editor, safe)
    assert before < float(session.state[target][f"strength_{expert}"]) < 1.5
    note = editor.write_note()
    assert "Moves: 1 measured, 1 written (strength 1/1)." in note
    assert f"Largest: {move.target} LoRA {move.expert} strength = " in note
    before = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    history = session.editor_snapshot()
    refused = GenerationRecord(
        measure=loss,
        guard_score=lambda trial: {"gap": 1.0 if trial is None else 0.0, "spread": 1.0},
    )
    assert "Refused" in commit_write(session, editor, refused)
    assert all(torch.equal(v, session.state[t][k]) for t, p in before.items() for k, v in p.items())
    for key, value in history.items():
        torch.testing.assert_close(value, session.editor_snapshot()[key], equal_nan=True)
    assert not editor.write_note()
    session.reset()
    assert torch.isnan(session.editor_state.history[move.id]).all()
    session.undo()
    for key, value in history.items():
        torch.testing.assert_close(value, session.editor_snapshot()[key], equal_nan=True)


def test_baked_clean_routing_and_settings(tmp_path):
    session = _session(tmp_path)
    assert session.mod.settings.import_score == "derived"
    target = next(iter(session.state))
    clean = session.sets()[0]
    clean_plan = plan_targets([ScopedSet(clean, import_score="derived")], 2)[target]
    session.state[target]["core_0"][0, 0] = 1.4
    session.state[target]["strength_0"][0] = 1.3
    edited = session.sets()[0]
    plan = plan_targets([ScopedSet(edited, import_score="derived")], 2)[target]
    moved = plan.to(torch.device("cpu"), torch.float32)
    assert torch.equal(moved.sets[0].routing_down, clean_plan.sets[0].down)
    assert torch.equal(moved.sets[0].routing_up, clean_plan.sets[0].up)
    assert not torch.equal(moved.sets[0].up, clean_plan.sets[0].up)
    from kilora import compute

    x = torch.randn(1, 3, moved.sets[0].down.shape[-1])
    y = torch.randn(1, 3, moved.sets[0].up.shape[-1])
    routes = []
    outputs = []
    for layer in (clean_plan.sets[0], moved.sets[0]):
        row = []
        with (
            torch.no_grad(),
            compute.recording_shares(lambda _, gates, row=row: row.append(gates[0].clone())),
        ):
            outputs.append(compute.run_layer(x, y, [layer], 1, target=target))
        routes.append(row)
    assert routes[0] and len(routes[0]) == len(routes[1])
    assert all(torch.equal(a, b) for a, b in zip(*routes, strict=True))
    assert not torch.equal(*outputs)


def test_record_opens_and_reuses_imported_session(tmp_path, monkeypatch):
    from test_comfyui import FakeBaseModel
    from test_families import _model
    from test_path_write import WrapPatcher
    from test_working_copies import routed_chain

    from kilora.comfyui import live_write, patching
    from kilora.live_imports import ImportedLiveSession

    monkeypatch.setenv("KILORA_LIVE_ASYNC", "0")
    base = WrapPatcher(FakeBaseModel(_model("flux")))
    chain = routed_chain(tmp_path)
    applied, _ = patching.apply(base, chain, 1, joint=True, bypass="off")
    assert not patching.sessions_of(applied)
    recorded, handle = live_write.record(applied, None)
    assert isinstance(handle.session, ImportedLiveSession)
    assert isinstance(handle.rule, ImportEditor)
    assert patching.sessions_of(recorded)[0] is handle.session
    _, again = live_write.record(applied, None)
    assert again.session is handle.session


def test_fixed_strength_trial(tmp_path):
    from kilora.fixed_change import fixed_copy
    from kilora.live_imports import ImportTrial
    from kilora.recording import perturbing
    from kilora.sections import DenseChange

    session = _session(tmp_path)
    target = next(iter(session.state))
    session._section_pairs.add((target, 0))
    old = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    new = {t: {k: v.clone() for k, v in p.items()} for t, p in old.items()}
    new[target]["strength_0"] += 0.25
    delta = torch.eye(3)
    change = DenseChange(delta)
    change._kilora_import_clean = DenseChange(delta)
    change._kilora_import_key = (session.mod_id, target, 0)
    layer = fixed_copy(torch.nn.Linear(3, 3, bias=False), [change])
    x = torch.randn(2, 3)
    before = layer(x)
    with perturbing(ImportTrial(session, old, new)):
        after = layer(x)
    torch.testing.assert_close(after - before, 0.25 * x)
