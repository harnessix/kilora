# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from test_import_edit_dimensions import _session
from test_import_rprop_integration import record, setup_editor

from kilora.editors import commit_write
from kilora.import_controller import TRUST_RATIO
from kilora.import_terms import (
    PACK_TERM_MOVES,
    PackTerm,
    pack_directions,
    pack_moves_start,
    set_terms,
)
from kilora.live import SESSIONS
from kilora.live_imports import IMPORT_WRITE_FRAC, ImportEditor, ImportTrial, export_low_rank
from kilora.recording import GenerationRecord


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _pack(session, target, basis=True, pack_id="pk"):
    ex = session._source.experts[target]
    out, n_in = ex.up.shape[-1], ex.down.shape[-1]
    gen = torch.Generator().manual_seed(3)
    tensors = {"q": torch.randn(2, out * 2, generator=gen)}
    if basis:
        tensors[f"input_basis.{target}"] = torch.randn(n_in, 2, generator=gen)
    manifest = SimpleNamespace(
        id=pack_id, raw={"chunk": {"hooks": [target], "shapes": {target: [out, 2]}}}
    )
    return _Pack((manifest, tensors), pack_id)


class _Pack(tuple):
    def __new__(cls, value, pack_id):
        pack = super().__new__(cls, value)
        pack.id, pack.version, pack.digest = pack_id, "1", pack_id
        pack.ref = SimpleNamespace(
            to_manifest=lambda: {"id": pack_id, "version": "1", "digest": pack_id}
        )
        return pack


def _one_term(session, target, strength=0.5):
    manifest, tensors = _pack(session, target)
    outputs, inputs = pack_directions(
        [target],
        manifest.raw["chunk"]["shapes"],
        tensors["q"],
        {target: tensors[f"input_basis.{target}"]},
        target,
        "pk",
    )
    set_terms(session.state[target], [PackTerm("pk:out:0:1", "pk:in:1", strength)], outputs, inputs)
    expected = torch.outer(strength * outputs["pk:out:0:1"], tensors[f"input_basis.{target}"][:, 1])
    return expected


def _forward(setting, target, x):
    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets

    plan = plan_targets([ScopedSet(setting, import_score="derived")], 2)[target]
    layer = plan.to(torch.device("cpu"), torch.float32).sets[0]
    y = torch.zeros(1, 3, layer.up.shape[-1])
    with torch.no_grad():
        return compute.run_layer(x, y, [layer], 1, target=target)


def test_no_terms_is_unchanged_and_edits_off_drops_terms(tmp_path):
    session = _session(tmp_path)
    target = next(iter(session.state))
    sets = session.sets()
    assert len(sets) == 1
    x = torch.randn(1, 3, session._source.experts[target].down.shape[-1])
    assert torch.equal(_forward(session._source, target, x), _forward(sets[0], target, x))
    _one_term(session, target)
    assert [s.name for s in session.sets()][1:] == ["edit-terms"]
    session.set_options(edits="off")
    assert len(session.sets()) == 1


def test_one_term_reproduces_outer_product(tmp_path):
    session = _session(tmp_path)
    target = next(iter(session.state))
    before = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    expected = _one_term(session, target)
    ex = session.sets()[1].experts[target]
    assert torch.equal(ex.up[0, 0].T @ ex.down[0, 0], expected)
    trial = ImportTrial(session, before, session.state)
    x = torch.randn(3, ex.down.shape[-1])
    torch.testing.assert_close(trial.input_delta(target, x), x @ expected.T)
    assert not trial.du


def test_old_pack_without_input_basis_gives_no_moves(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    target = choices[0].target
    session.editor_packs = [_pack(session, target, basis=False)]
    session.host_norms = {t: 1.0 for t in session.state}
    moves, skipped = editor.pack_moves()
    assert moves == [] and skipped == [f"pk:{target}"]
    editor.next_state(session.state, record(session, choices[0]))
    assert "Pack moves: 0 measured, 0 written." in editor.note
    assert f"Pack hooks without input_basis: pk:{target}." in editor.note


def _gated_record(session, move, dims_weight):
    base = record(session, move)
    ex = session._source.experts[move.target]
    probe = torch.ones(ex.up.shape[-1], ex.down.shape[-1])

    def measure(trial):
        loss = 1.0 + dims_weight * (1.0 - base.measure(trial)) ** 2
        change = trial.terms.get(move.target)
        if change is not None:
            loss -= float(((change[0] @ change[1]) * probe).sum())
        return loss

    return GenerationRecord(measure=measure, guard_score=base.guard_score)


def _pack_editor(tmp_path, controller):
    session, editor, choices = setup_editor(tmp_path)
    session.set_options(controller=controller)
    target = choices[0].target
    session.editor_packs = [_pack(session, target)]
    session.host_norms = {t: 1.0 for t in session.state}
    assert len(editor.pack_moves()[0]) == PACK_TERM_MOVES
    return session, editor, choices[0], target


def _event(session, editor, move, dims_weight):
    commit_write(session, editor, _gated_record(session, move, dims_weight))
    return editor.note


NO_PACK_MOVES = "Pack moves: 0 measured, 0 written."
STRONG, PLATEAU = 100.0, 1e-3


@pytest.mark.parametrize("controller", ["r1", "rprop"])
@pytest.mark.parametrize("plateau", [False, True])
def test_pack_moves_follow_gain(tmp_path, controller, plateau):
    session, editor, move, target = _pack_editor(tmp_path, controller)
    weights = [STRONG] * 3 + ([PLATEAU] * 3 if plateau else [STRONG] * 3)
    notes = [_event(session, editor, move, w) for w in weights]
    gains = session.editor_state.gains
    assert gains.shape[0] == 6 and bool((gains[:3] > 0).all())
    if not plateau:
        assert all(NO_PACK_MOVES in note for note in notes)
        assert "terms_c" not in session.state[target]
        return
    assert all(NO_PACK_MOVES in note for note in notes[:5])
    written = f"Pack moves: {PACK_TERM_MOVES} measured, {PACK_TERM_MOVES} written."
    assert written in notes[5]
    ex = session.state[target]
    assert bool(ex["terms_c"].count_nonzero())
    change = ex["terms_out"] @ ex["terms_c"] @ ex["terms_in"].T
    assert float(change.norm()) <= TRUST_RATIO * IMPORT_WRITE_FRAC * 1.0 + 1e-6


@pytest.mark.parametrize("controller", ["r1", "rprop"])
def test_fewer_than_three_events_give_no_pack_moves(tmp_path, controller):
    session, editor, move, target = _pack_editor(tmp_path, controller)
    for _ in range(2):
        assert NO_PACK_MOVES in _event(session, editor, move, PLATEAU)
    assert session.editor_state.gains.shape[0] == 2
    assert "terms_c" not in session.state[target]


def test_start_rule():
    assert not pack_moves_start([])
    assert not pack_moves_start([1.0, 0.0])
    assert not pack_moves_start([1.0, 1.0, 1.0])
    assert not pack_moves_start([1.0, 1.0, 1.0, 1.0, 1.0, 0.7])
    assert pack_moves_start([1.0, 1.0, 1.0, 0.2, 0.2, 0.3])
    assert not pack_moves_start([0.0, 0.0, 0.0, 0.0])
    assert not pack_moves_start([1.0, 1.0, 1.0, float("nan"), 0.0, 0.0])


def test_plateau_state_save_load_and_undo(tmp_path):
    session, editor, move, _target = _pack_editor(tmp_path, "rprop")
    for _ in range(3):
        _event(session, editor, move, STRONG)
    saved = session.editor_state.gains.clone()
    assert saved.shape[0] == 3
    name = session.save_point("plateau")
    _event(session, editor, move, PLATEAU)
    assert session.editor_state.gains.shape[0] == 4
    session.undo()
    assert torch.equal(session.editor_state.gains, saved)
    _event(session, editor, move, PLATEAU)
    session.load(name)
    assert torch.equal(session.editor_state.gains, saved)
    session.editor_state = None
    session.load(name)
    ImportEditor(session)
    assert torch.equal(session.editor_state.gains, saved)


def test_add_gain_keeps_first_and_last_events(tmp_path):
    session, _editor, _move, _target = _pack_editor(tmp_path, "r1")
    store = session.editor_state
    for gain in range(1, 10):
        store.add_gain(float(gain))
    assert store.gains.tolist() == [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    store.reset()
    assert store.gains.numel() == 0


def test_write_note_counts_pack_hooks_without_import(tmp_path):
    session, editor, choices = setup_editor(tmp_path)
    target = choices[0].target
    ex = session._source.experts[target]
    out, n_in = ex.up.shape[-1], ex.down.shape[-1]
    gen = torch.Generator().manual_seed(4)
    hooks = [target, "nowhere.a", "nowhere.b"]
    manifest = SimpleNamespace(
        id="pk", raw={"chunk": {"hooks": hooks, "shapes": {h: [out, 2] for h in hooks}}}
    )
    tensors = {
        "q": torch.randn(2, out * 2 * len(hooks), generator=gen),
        f"input_basis.{target}": torch.randn(n_in, 2, generator=gen),
    }
    session.editor_packs = [(manifest, tensors)]
    session.host_norms = {t: 1.0 for t in session.state}
    editor.next_state(session.state, record(session, choices[0]))
    assert editor.uncovered == 2
    assert "Pack hooks skipped without an import: 2." in editor.note


def test_save_load_undo_and_export_terms(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    session = _session(tmp_path)
    target = next(iter(session.state))
    expected = _one_term(session, target)
    saved = {k: v.clone() for k, v in session.state[target].items()}
    path = tmp_path / "edits.kmod"
    session.save_edits(path)
    session.state = session._reset_state()
    session.load_edits(path)
    for key, value in saved.items():
        assert torch.equal(session.state[target][key], value)
    session.undo()
    assert "terms_c" not in session.state[target]
    session.load_edits(path)

    kmod = tmp_path / "out.kmod"
    session.export(kmod, format="kmod")
    with safe_open(str(kmod), framework="pt") as file:
        assert f"kilora/edit/{target}/terms_c" in set(file.keys())

    def product(file):
        state = load_file(str(file))
        a = state[f"transformer.{target}.lora_A.weight"]
        b = state[f"transformer.{target}.lora_B.weight"]
        return b @ a

    with_terms = tmp_path / "with.safetensors"
    note = session.export(with_terms, format="lora")
    assert note["relative_svd_residual"][target] < 1e-5
    session.state = session._reset_state()
    without = tmp_path / "without.safetensors"
    session.export(without, format="lora")
    torch.testing.assert_close(
        product(with_terms) - product(without), expected, atol=1e-5, rtol=1e-4
    )


def test_single_term_plain_export_is_exact(tmp_path):
    from safetensors.torch import load_file

    session = _session(tmp_path)
    target = next(iter(session.state))
    expected = _one_term(session, target)
    path = tmp_path / "one.safetensors"
    note = session.export(path, format="lora")
    assert max(note["relative_svd_residual"].values()) < 1e-5
    state = load_file(str(path))
    product = (
        state[f"transformer.{target}.lora_B.weight"].double()
        @ state[f"transformer.{target}.lora_A.weight"].double()
    )
    ex = session._source.experts[target]
    source = sum(ex.up[e, 0].double().T @ ex.down[e, 0].double() for e in range(ex.down.shape[0]))
    total = source + expected.double()
    assert float((product - total).norm() / total.norm()) < 1e-5


def test_export_low_rank_is_exact_and_falls_back(monkeypatch):
    from kilora import live_imports

    gen = torch.Generator().manual_seed(5)
    delta = torch.randn(12, 3, generator=gen, dtype=torch.float64) @ torch.randn(
        3, 9, generator=gen, dtype=torch.float64
    )
    down, up, error = export_low_rank(delta, 3)
    assert down.shape == (3, 9) and up.shape == (12, 3) and error < 1e-12
    calls = []

    def fake(d, rank, device=None):
        calls.append((tuple(d.shape), rank, device))
        return torch.zeros(rank, 9), torch.zeros(12, rank), 1.0

    monkeypatch.setattr("kilora.multi_import.low_rank", fake)
    monkeypatch.setattr(live_imports, "EXPORT_EXACT_SVD_MAX", 10)
    assert export_low_rank(delta, 3)[2] == 1.0
    assert calls == [((12, 9), 3, "cpu")]
    monkeypatch.setattr(live_imports, "EXPORT_EXACT_SVD_MAX", 10**6)
    tried = []
    exact = torch.linalg.svd

    def fail_once(d, full_matrices=True):
        tried.append(d.device.type)
        if len(tried) == 1:
            raise torch.linalg.LinAlgError("no convergence")
        return exact(d, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", fail_once)
    down, up, error = export_low_rank(delta, 3, device="meta")
    assert tried == ["meta", "cpu"] and error < 1e-12 and len(calls) == 1


def _layer_output(sets, target, x, y, depth):
    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets

    plan = plan_targets([ScopedSet(s, import_score="derived") for s in sets], depth)[target]
    layers = plan.to(torch.device("cpu"), torch.float32).sets
    with torch.no_grad():
        return compute.run_layer(x, y.clone(), layers, depth, target=target)


def test_term_runs_once_at_depth_three(tmp_path):
    session = _session(tmp_path)
    before = {t: {k: v.clone() for k, v in p.items()} for t, p in session.state.items()}
    targets = list(session.state)
    for target in targets:
        _one_term(session, target)
    sets = session.sets()
    assert [s.name for s in sets][1:] == ["edit-terms"]
    assert sets[1].extra["apply_once"] is True
    trial = ImportTrial(session, before, session.state)
    gen = torch.Generator().manual_seed(6)
    for target in targets:
        ex = session._source.experts[target]
        x = torch.randn(1, 3, ex.down.shape[-1], generator=gen)
        y = torch.randn(1, 3, ex.up.shape[-1], generator=gen)
        committed = _layer_output(sets, target, x, y, 3)
        replay = _layer_output(sets[:1], target, x, y, 3) + trial.input_delta(target, x)
        torch.testing.assert_close(committed, replay, atol=1e-6, rtol=1e-6)
