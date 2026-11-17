# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import kilora_editor_chunk.plugin as cl
import pytest
import torch
from kilora_editor_chunk import Chunk, ChunkEditor

from kilora.compute import router_logits
from kilora.editors import (
    available_editors,
    commit_write,
    editor_for,
    register_editor,
    unregister_editor,
)
from kilora.errors import KiLoRAError
from kilora.live import open_session
from kilora.recording import GenerationRecord
from kilora.testing import TINY_FAMILY, make_live_mod


def _packs(session, folder, base_family=TINY_FAMILY):

    names = list(session.state)
    shapes = {t: [int(session.state[t]["up"].shape[-1]), 4] for t in names}
    dim = sum(o * r for o, r in shapes.values())
    q, _ = torch.linalg.qr(torch.randn(dim, 12, generator=torch.Generator().manual_seed(1)))
    rows = q.t()
    ed = ChunkEditor(
        names,
        shapes,
        {"to_q": 1.0, "to_k": 1.0, "to_v": 1.0, "to_out.0": 1.0},
        [Chunk("common", rows[:8], torch.ones(8)), Chunk("retro_game", rows[8:], torch.ones(4))],
    )
    files = ed.save_packs(folder, "1.0.0", base_family, "Apache-2.0")
    path = folder / "chunk.kedit.safetensors"
    cl.save_chunk_editor(path, list(files.values()), base_family, write_norm=0.05, per_event=6)
    return path


def _record(session, target):
    def measure(trial):
        if trial is None:
            return 0.0
        return float(sum(((trial.du[t] - target[t]) ** 2).sum() for t in trial.du))

    return GenerationRecord(images=1, measure=measure)


@pytest.fixture
def setup(tmp_path):
    before = available_editors().get(cl.KIND)
    register_editor(cl.PLUGIN, "test")
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    path = _packs(session, tmp_path)
    g = torch.Generator().manual_seed(5)
    target = {
        t: torch.randn(int(p["up"].shape[-1]), 4, generator=g) for t, p in session.state.items()
    }
    yield session, path, target
    unregister_editor(cl.KIND)
    if before is not None:
        register_editor(before, "test")


def _write(session, path, target):
    editor = editor_for(path)
    editor.check(session)
    return commit_write(session, editor, _record(session, target))


def _ups(session):
    return {t: p["up"].clone() for t, p in session.state.items()}


def test_found_by_kind(setup):
    assert "chunk" in available_editors()
    _, path, _ = setup
    assert isinstance(editor_for(path), cl.ChunkLive)


@pytest.mark.parametrize("form", ["state", "input"])
def test_route_coordinates_match_set_form(setup, form):
    from dataclasses import replace

    session, path, _ = setup
    name = session.mod.manifest["live"]["live_set"]
    live = next(s for s in session.mod.sets if s.name == name)
    if form == "input":
        live = replace(
            live,
            form=form,
            experts={
                t: replace(
                    ex,
                    keys_in=ex.keys_in[..., : live.router_dim].clone(),
                    keys_out=ex.keys_out[..., : live.router_dim].clone(),
                )
                for t, ex in live.experts.items()
            },
        )
        session.mod.sets = [live if s.name == name else s for s in session.mod.sets]
    editor = editor_for(path)
    editor.check(session)
    for t, (router, keys) in editor.routes.items():
        width = live.experts[t].down.shape[-1]
        h = torch.randn(2, width, generator=torch.Generator().manual_seed(7))
        if form == "state":
            assert live.router_for(t).shape[0] != width
            assert torch.equal(router, torch.eye(width))
            expected = router_logits(h, live.experts[t].keys_in, editor.tau)
        else:
            assert torch.equal(router, live.router_for(t))
            expected = router_logits(
                torch.nn.functional.linear(h, live.router_for(t)), keys, editor.tau
            )
        actual = router_logits(torch.nn.functional.linear(h, router), keys, editor.tau)
        assert torch.equal(actual, expected)


def test_write_undo_save_point(setup):
    session, path, target = setup
    before = _ups(session)
    assert "1 commits" in _write(session, path, target)
    one = _ups(session)
    assert any(not torch.equal(one[t], before[t]) for t in one)
    assert session.editor_state.chunks.events == 1
    name = session.save_point("one")
    _write(session, path, target)
    assert session.editor_state.chunks.events == 2
    session.undo()
    assert all(torch.equal(session.state[t]["up"], one[t]) for t in one)
    assert session.editor_state.chunks.events == 1
    _write(session, path, target)
    current = _ups(session)
    session.load(name)
    assert all(torch.equal(session.state[t]["up"], current[t]) for t in current)
    assert session.editor_state.chunks.events == 2
    assert len(session.editor_state.forest.branches) == 2


def test_deterministic_and_pack_off(setup):
    session, path, target = setup
    _write(session, path, target)
    first = _ups(session)
    session.undo()
    _write(session, path, target)
    assert all(torch.equal(session.state[t]["up"], first[t]) for t in first)
    st = session.editor_state
    st.chunks.set_enabled("retro_game", False)
    assert st.chunks.enabled == ["common"]


def test_wrong_base_model(setup, tmp_path):
    session, _, _ = setup
    other = _packs(session, tmp_path / "z", base_family="z-image")
    with pytest.raises(KiLoRAError, match="This editor is for z-image"):
        editor_for(other).check(session)


def test_no_gradients(setup):
    session, path, target = setup
    for p in session.state.values():
        p["up"].requires_grad_(True)
    _write(session, path, target)
    assert all(not p["up"].requires_grad for p in session.state.values())


def test_unrouted_expert_gets_no_write(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    assert editor.experts >= 2
    g = torch.Generator().manual_seed(9)
    hs = {}
    for t, (router, keys) in editor.routes.items():
        h = torch.randn(64, router.shape[1], generator=g)
        q = torch.nn.functional.linear(h, router)
        pick = router_logits(q, keys).argmax(-1) == 0
        assert int(pick.sum()) > 0
        hs[t] = h[pick]

    def measure(trial):
        for t in hs:
            trial.delta(t, hs[t])
        return float(sum(((trial.du[t] - target[t]) ** 2).sum() for t in trial.du))

    before = _ups(session)
    commit_write(session, editor, GenerationRecord(images=1, measure=measure))
    for t, parts in session.state.items():
        up = parts["up"].reshape(-1, parts["up"].shape[-1])
        old = before[t].reshape(up.shape)
        rows1 = editor.col_expert[t] != 0
        rows0 = editor.col_expert[t] == 0
        assert torch.equal(up[rows1], old[rows1])
        assert not torch.equal(up[rows0], old[rows0])


def test_user_options_override_the_manifest(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    calls, times = [0], []

    def measure(trial, time=None):
        calls[0] += 1
        times.append(time)
        return float(sum(((trial.du[t] - target[t]) ** 2).sum() for t in trial.du))

    editor.configure({"packs": ["common"], "diff": "one-sided", "time_bins": 2, "measure": "full"})
    commit_write(session, editor, GenerationRecord(images=1, measure=measure))
    st = session.editor_state.chunks
    assert st.enabled == ["common"]
    assert float(st.coef["retro_game"].abs().sum()) == 0.0
    assert calls[0] == 6 * 2 + 2
    assert sorted(set(times)) == [0.7, 0.95]
    editor.configure({})
    calls[0], times[:] = 0, []
    commit_write(session, editor, GenerationRecord(images=1, measure=lambda trial: measure(trial)))
    assert session.editor_state.chunks.enabled == ["common", "retro_game"]
    assert calls[0] == 2 * 3 and set(times) == {None}
    editor.configure({"measure": "full"})
    calls[0] = 0
    commit_write(session, editor, GenerationRecord(images=1, measure=lambda trial: measure(trial)))
    assert calls[0] == 2 * 6


def test_missing_packs_say_where_to_get_them(setup, tmp_path):
    _, path, _ = setup
    pack = next(tmp_path.glob("*.kedit-pack.safetensors"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pack.rename(elsewhere / pack.name)
    with pytest.raises(FileNotFoundError, match="direction packs") as err:
        editor_for(path)
    text = str(err.value)
    assert pack.name in text and "same folder" in text and "Build packs" in text


def _change(session, start):
    return sum(float((session.state[t]["up"] - start[t]).pow(2).sum()) for t in start) ** 0.5


def test_live_speed_scales_the_write(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    full_speed = session.spec.speed
    assert editor._writer([None]).write_norm == pytest.approx(0.05)
    session.speed = full_speed / 2
    assert editor._writer([None]).write_norm == pytest.approx(0.025)
    start = _ups(session)
    session.speed = full_speed
    commit_write(session, editor, _record(session, target))
    full = _change(session, start)
    session.undo()
    session.speed = full_speed / 2
    commit_write(session, editor, _record(session, target))
    assert _change(session, start) == pytest.approx(full / 2, rel=1e-4)
    session.speed = 0.0
    before = _ups(session)
    commit_write(session, editor, _record(session, target))
    assert all(torch.equal(session.state[t]["up"], before[t]) for t in before)


def test_default_total_cap_bounds_the_state(setup):
    session, path, target = setup
    m = cl.read_manifest(path)
    assert m["writer"]["total_cap"] == pytest.approx(0.05 * cl.DEFAULT_TOTAL_CAP_WRITES)
    editor = editor_for(path)
    editor.check(session)
    w = editor._writer([None])
    st = session.editor_state.chunks
    for _ in range(25):
        commit_write(session, editor, _record(session, target))
    total = w._norm({f: st.coef[f] for f in st.enabled})
    assert editor.state_norm() == pytest.approx(total)
    assert editor.writer.total_cap == pytest.approx(m["writer"]["total_cap"])
    assert total <= m["writer"]["total_cap"] + 1e-5
    assert total > 0.9 * m["writer"]["total_cap"]


def test_files_without_a_cap_get_the_default(setup, tmp_path):
    session, path, _ = setup
    m = cl.read_manifest(path)
    packs = [path.parent / n for n in m["packs"]]
    old = tmp_path / "old.kedit.safetensors"
    cl.save_chunk_editor(old, packs, TINY_FAMILY, write_norm=0.05, total_cap_writes=None)
    assert cl.read_manifest(old)["writer"]["total_cap"] is None
    editor = editor_for(old)
    editor.check(session)
    assert editor._writer([None]).total_cap == pytest.approx(0.05 * cl.DEFAULT_TOTAL_CAP_WRITES)


def test_guard_trial_is_the_routed_write(setup):
    from kilora.compute import hard_gates, router_logits, tensor_gate, token_gate

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    new = editor.next_state(session.state, _record(session, target))
    trial = editor.guard_trial(session.state, new)
    assert isinstance(trial, cl.RoutedWrite)
    routing = getattr(session.mod.settings, "routing", "tensor")
    for t in editor.targets:
        a = cl._down(session.state[t])
        up = session.state[t]["up"]
        db = (new[t]["up"].float() - up.float()).reshape(-1, up.shape[-1]).t()
        h = torch.randn(1, 5, a.shape[1], generator=torch.Generator().manual_seed(2))
        c = h @ a.t()
        if t in editor.routes:
            router, keys = editor.routes[t]
            lg = router_logits(torch.nn.functional.linear(h, router), keys, editor.tau)
            with hard_gates():
                w = tensor_gate(lg) if routing == "tensor" else token_gate(lg)
            ref = sum(
                (c * w[..., e : e + 1] * (editor.col_expert[t] == e).float()) @ db.t()
                for e in range(editor.experts)
            )
        else:
            ref = c @ db.t()
        torch.testing.assert_close(trial.delta(t, h), ref)


def test_guard_refuses_and_restores_a_chunk_write(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    before = _ups(session)

    def bad(trial):
        return {"gap": 1.0 if trial is None else 0.1, "spread": 1.0}

    rec = _record(session, target)
    rec.guard_score = bad
    report = commit_write(session, editor, rec)
    assert report.startswith("Refused by the guard")
    assert all(torch.equal(session.state[t]["up"], before[t]) for t in before)
    assert session.editor_state.chunks.events == 0
    rec = _record(session, target)
    rec.guard_score = lambda trial: {"gap": 1.0, "spread": 1.0}
    assert "Guard: passed." in commit_write(session, editor, rec)
    assert session.editor_state.chunks.events == 1


def _frac_file(session, path, tmp_path, **kw):
    m = cl.read_manifest(path)
    out = tmp_path / "frac.kedit.safetensors"
    packs = [path.parent / n for n in m["packs"]]
    cl.save_chunk_editor(out, packs, TINY_FAMILY, write_norm=0.05, per_event=6, **kw)
    editor = editor_for(out)
    editor.check(session)
    return editor


def test_write_and_cap_are_fractions_of_the_clean_state(setup, tmp_path):
    session, path, target = setup
    editor = _frac_file(session, path, tmp_path, write_frac=0.1, cap_frac=0.25)
    s = sum(float(session.clean[t]["up"].pow(2).sum()) for t in session.clean) ** 0.5
    assert editor.scale == pytest.approx(s) and s > 0
    w = editor._writer([None])
    assert w.write_norm == pytest.approx(0.1 * s) and w.total_cap == pytest.approx(0.25 * s)
    start = _ups(session)
    commit_write(session, editor, _record(session, target))
    assert _change(session, start) == pytest.approx(0.1 * s, rel=1e-4)
    for _ in range(6):
        commit_write(session, editor, _record(session, target))
    assert _change(session, start) <= 0.25 * s * (1 + 1e-4)
    assert editor.state_norm() == pytest.approx(_change(session, start), rel=1e-4)


def test_files_without_fractions_keep_their_absolute_values(setup):
    session, path, _ = setup
    editor = editor_for(path)
    editor.check(session)
    w = editor._writer([None])
    assert w.write_norm == pytest.approx(0.05) and w.total_cap == pytest.approx(0.5)


def test_per_hook_cap_is_off_unless_asked(setup, tmp_path):
    session, path, target = setup
    off = _frac_file(session, path, tmp_path, write_frac=0.5, cap_frac=0.05)
    assert off._per_hook_cap() is None
    on = _frac_file(session, path, tmp_path, write_frac=0.5, cap_frac=0.05, per_hook_cap=True)
    assert on._per_hook_cap() == pytest.approx(0.05)
    new = on.next_state(session.state, _record(session, target))
    for t in on.targets:
        d = (new[t]["up"].float() - session.state[t]["up"].float()).norm()
        assert float(d) <= 0.05 * on.hook_scale[t] * (1 + 1e-4)
    on.configure({"per_hook_cap": False})
    assert on._per_hook_cap() is None


def test_guard_calibration_scores_the_scaled_state(setup):
    from kilora.editors import guard_calibration

    session, path, _ = setup
    editor = editor_for(path)
    editor.check(session)
    seen = []

    def score(trial):
        if trial is None:
            return {"gap": 1.0, "spread": 2.0}
        t = editor.targets[0]
        h = torch.ones(1, 1, cl._down(session.state[t]).shape[1])
        seen.append(float(trial.delta(t, h).norm()))
        return {"gap": 1.0 / (1 + seen[-1]), "spread": 2.0 + seen[-1]}

    from kilora.recording import GenerationRecord as R

    out = guard_calibration(editor, session, R(guard_score=score), ks=(2, 3, 5, 10))
    assert [r["k"] for r in out["rows"]] == [2, 3, 5, 10]
    assert seen[1] == pytest.approx(2 * seen[0], rel=1e-4)
    assert seen[3] == pytest.approx(9 * seen[0], rel=1e-4)
    assert out["rows"][0]["spread_ratio"] == pytest.approx((2.0 + seen[0]) / 2.0)
    assert all(torch.equal(session.state[t]["up"], session.clean[t]["up"]) for t in session.state)


def _fresh(session, tmp_path):
    again = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    assert again is not session and again.editor_state is None
    again.learning = True
    return again


def _same_editor(a, b):
    ta, tb = a.editor_state.state_tensors(), b.editor_state.state_tensors()
    assert set(ta) == set(tb)
    for k in ta:
        torch.testing.assert_close(ta[k], tb[k], rtol=0, atol=0, equal_nan=True, msg=k)


def test_load_before_the_editor_attaches_continues_the_editor(setup, tmp_path):
    session, path, target = setup
    for _ in range(3):
        _write(session, path, target)
    name = session.save_point("three")
    again = _fresh(session, tmp_path)
    assert "starts from zero" not in again.load(name)
    editor_for(path).check(again)
    _same_editor(session, again)
    _write(session, path, target)
    _write(again, path, target)
    assert again.editor_state.chunks.events == 4 and len(again.commits) == 4
    _same_editor(session, again)
    for t in session.state:
        torch.testing.assert_close(again.state[t]["up"], session.state[t]["up"], rtol=0, atol=0)


def test_undo_after_load_restarts_the_editor_on_the_earlier_state(setup, tmp_path):
    session, path, target = setup
    clean = _ups(session)
    _write(session, path, target)
    name = session.save_point("one")
    again = _fresh(session, tmp_path)
    again.load(name)
    editor = editor_for(path)
    editor.check(again)
    assert again.editor_state.chunks.events == 1
    again.undo()
    assert again.editor_state.chunks.events == 0
    for t, up in clean.items():
        torch.testing.assert_close(again.editor_state.base[t], up.float(), rtol=0, atol=0)
    third = _fresh(session, tmp_path)
    third.load(name)
    third.undo()
    editor_for(path).check(third)
    assert third.editor_state.chunks.events == 0


def test_a_save_without_a_fitting_editor_state_starts_from_zero_on_it(setup, tmp_path):
    session, path, target = setup
    _write(session, path, target)
    name = session.save_point("one")
    again = _fresh(session, tmp_path)
    again.load(name)
    again._pending_editor = (None, cl.KIND)
    note = again.attach_editor_state(cl.ChunkEditorState(editor_for(path).editor, {}, 1), cl.KIND)
    assert "starts from zero" in note
    for t, p in again.state.items():
        torch.testing.assert_close(again.editor_state.base[t], p["up"].float(), rtol=0, atol=0)


def test_the_host_weights_set_the_scale_when_the_host_gives_them(setup, tmp_path):
    session, path, _ = setup
    session.host_norms = {t: 2.0 + i for i, t in enumerate(session.state)}
    editor = _frac_file(session, path, tmp_path, write_frac=0.01, cap_frac=0.1)
    assert editor.scale_reference == "host"
    assert editor.hook_scale == session.host_norms
    s = sum(v * v for v in session.host_norms.values()) ** 0.5
    assert editor._writer([None]).write_norm == pytest.approx(0.01 * s)
    session.host_norms = {}
    editor.check(session)
    assert editor.scale_reference == "live state"


def _tensor_change(session):
    base = session.editor_state.base
    return (
        sum(float((session.state[t]["up"].float() - base[t].float()).pow(2).sum()) for t in base)
        ** 0.5
    )


def test_writes_past_the_cap_keep_the_state_at_the_cap(setup, tmp_path):
    session, path, target = setup
    session.host_norms = {t: 1.0 for t in session.state}
    editor = _frac_file(session, path, tmp_path, write_frac=0.05, cap_frac=0.12)
    cap = 0.12 * editor.scale
    norms = []
    for _ in range(16):
        report = commit_write(session, editor, _record(session, target))
        assert "commits" in report
        norms.append(_tensor_change(session))
        assert editor.state_norm() == pytest.approx(norms[-1], rel=1e-4)
    assert max(norms) <= cap * (1 + 1e-4)
    assert norms[-1] == pytest.approx(cap, rel=1e-3)


@pytest.mark.parametrize("cap_effect", [True, False])
def test_the_effect_bound_stops_a_turn_toward_high_gain(setup, tmp_path, cap_effect):
    session, path, target = setup
    session.host_norms = {t: 1.0 for t in session.state}
    editor = _frac_file(session, path, tmp_path, write_frac=0.05, cap_frac=0.12)
    editor.configure({"cap_effect": cap_effect})
    loud = editor.editor.chunks[1]
    loud.mean_keys = {h: torch.eye(4) * 25.0 for h in editor.editor.names}
    quiet = editor.editor.chunks[0]
    quiet.mean_keys = {h: torch.eye(4) for h in editor.editor.names}
    rows = editor.editor.per_hook(loud.q)
    target = {h: 3.0 * d.sum(0) for h, d in rows.items()}
    editor._gain = None
    editor.configure({"cap_effect": True})
    w = editor._writer([None])
    bound = w.total_cap * editor._gain
    editor.configure({"cap_effect": cap_effect})
    for _ in range(16):
        commit_write(session, editor, _record(session, target))
    eff = editor.state_effect_norm()
    assert editor.state_norm() <= w.total_cap * (1 + 1e-4)
    if cap_effect:
        assert editor.writer.effect_cap == pytest.approx(bound)
        assert eff <= bound * (1 + 1e-4)
    else:
        assert editor.writer.effect_cap is None and eff > bound


def test_the_write_note_names_the_capped_quantity(setup, tmp_path):
    session, path, target = setup
    session.host_norms = {t: 1.0 for t in session.state}
    editor = _frac_file(session, path, tmp_path, write_frac=0.05, cap_frac=0.12)
    assert editor.write_note() == ""
    commit_write(session, editor, _record(session, target))
    note = editor.write_note()
    assert note.startswith("State: change norm ") and " of cap " in note
    assert "effect norm" in note and "effect cap" not in note
    assert "eps_steps=" not in note and "eps_nodes=" not in note


def test_write_note_counts_distinct_prediction_steps_and_nodes(setup, tmp_path, monkeypatch):
    from types import SimpleNamespace

    session, path, target = setup
    monkeypatch.setenv("KILORA_PREDICTION_DEBUG", "1")
    import kilora.recording as recording

    monkeypatch.setattr(recording, "prediction_report", lambda *a, **k: "")
    editor = _frac_file(session, path, tmp_path, write_frac=0.05, cap_frac=0.12)
    entry = lambda progress, node: SimpleNamespace(
        progress=progress, node=node, error=torch.tensor([1.0])
    )
    errors = {
        "first": [entry(p, n) for p in (0.25, 0.75) for n in (0, 1)],
        "second": [entry(p, n) for p in (0.25, 0.75) for n in (0, 1)],
    }
    session.editor_state.prediction_errors = errors
    assert editor.write_note() == ""
    commit_write(session, editor, _record(session, target))
    session.editor_state.prediction_errors = errors
    note = editor.write_note()
    assert "eps_steps=2 eps_nodes=4" in note
    assert "first: mean_abs_error=1" in note


def test_path_merge_projection_weight_rank_and_imports():
    from types import SimpleNamespace

    from kilora_editor_chunk import ChunkState

    from kilora.path_graph import PathGraphCache
    from kilora.sets import KeyedExperts

    name = "block.to_q"
    directions = torch.eye(16).reshape(16, -1)
    ed = ChunkEditor(
        [name],
        {name: [4, 4]},
        {"to_q": 1.0},
        [
            Chunk("common", directions, torch.ones(16)),
        ],
    )
    keys = torch.eye(2)[:, None, :]
    ex = KeyedExperts(keys_in=keys)
    live = SimpleNamespace(experts={name: ex}, router_for=lambda _: torch.eye(4)[:2])
    editor = object.__new__(cl.ChunkLive)
    editor.editor, editor.experts = ed, 2
    editor.targets, editor.live_set = [name], live
    editor.session = SimpleNamespace(mod=SimpleNamespace(settings=SimpleNamespace(depth=2)))
    editor.v0 = {name: torch.eye(4)}
    editor.col_expert = {name: torch.tensor([0, 0, 1, 1])}
    editor.path_graph = PathGraphCache()
    imported = torch.randn(3, 4)
    state = {name: {"down": torch.eye(4), "up": torch.randn(4, 4)}, "imported": imported}
    before_import = imported.clone()
    chunks = ChunkState.new(ed)
    chunks.coef["common"][:] = torch.eye(4).reshape(-1) * 0.02
    start = chunks.expanded_u(ed)[name].clone()
    editor._path_merge(chunks, state, {}, {(name, 0): [torch.full((2,), 1e-20)]})
    assert torch.allclose(chunks.expanded_u(ed)[name], start, atol=1e-12)

    snapshot = chunks.snapshot()
    editor._path_merge(chunks, state, {}, {(name, 0): [torch.ones(2)]})
    result = chunks.expanded_u(ed)[name]
    replay = ChunkState.new(ed)
    replay.restore(snapshot)
    editor._path_merge(replay, state, {}, {(name, 0): [torch.ones(2)]})
    assert torch.equal(replay.expanded_u(ed)[name], result)
    for e in range(2):
        mask = (editor.col_expert[name] == e).float()
        expected = start[e] + start[e] @ torch.diag(mask) @ start[e]
        assert torch.allclose(result[e], expected, atol=1e-6)
    assert chunks.node_coef
    for _ in range(8):
        editor._path_merge(chunks, state, {}, {(name, 0): [torch.ones(2)]})
    assert all(torch.linalg.matrix_rank(x) <= 4 for x in chunks.expanded_u(ed)[name])
    assert torch.equal(imported, before_import)


def test_node_coefficients_follow_pack_and_survive_state_roundtrips(setup, tmp_path):
    from kilora_editor_chunk import ChunkState

    session, path, _ = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state.chunks
    h = editor.targets[0]
    st.coef["retro_game"].fill_(0.1)
    before = st.expanded_u(editor.editor)
    st.node_coef[("retro_game", h)] = torch.full_like(st.coef["retro_game"], 0.2)
    merged = st.expanded_u(editor.editor)
    assert not torch.equal(before[h], merged[h])
    st.set_enabled("retro_game", False)
    without = st.expanded_u(editor.editor)
    saved_delta = st.node_coef.pop(("retro_game", h))
    assert torch.equal(without[h], st.expanded_u(editor.editor)[h])
    st.node_coef[("retro_game", h)] = saved_delta
    assert st.undo()
    assert torch.equal(st.expanded_u(editor.editor)[h], merged[h])
    st.save(tmp_path / "node.safetensors", editor.editor)
    loaded = ChunkState.load(tmp_path / "node.safetensors")
    assert torch.equal(loaded.expanded_u(editor.editor)[h], merged[h])
    tensors = session.editor_state.state_tensors()
    session.editor_state.reset()
    session.editor_state.load_tensors(tensors)
    assert torch.equal(session.editor_state.chunks.expanded_u(editor.editor)[h], merged[h])


def test_next_state_path_merge_switch_and_measured_keep(setup, caplog):
    import logging

    from kilora.compute import read_dropout

    caplog.set_level(logging.DEBUG, logger="kilora.editor.chunk")

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    session.mod.settings.depth = 2
    session.mod.sets = [editor.live_set]
    before = session.editor_state.state_tensors()
    original = {h: {k: v.clone() for k, v in parts.items()} for h, parts in session.state.items()}
    generator = torch.Generator().manual_seed(43)
    activations = {
        h: torch.randn(3, router.shape[1], generator=generator)
        for h, (router, _) in editor.routes.items()
    }

    def measure(p):
        value = 0.0
        for h, x in activations.items():
            p.delta(h, x)
            columns = editor.col_expert[h].numel()
            read_dropout(
                torch.ones(3, columns),
                torch.ones(3, columns),
                columns // editor.experts,
                f"{h}#0",
                1,
                set_key=session.mod_id,
            )
            value += float(((p.du[h] - target[h]) ** 2).sum())
        return value

    record = GenerationRecord(images=1, measure=measure)
    editor.manifest["writer"]["diff"] = "one-sided"
    default = editor.next_state(session.state, record)
    default_tensors = session.editor_state.state_tensors()
    assert any("path graph recomputed nodes=" in record.message for record in caplog.records)
    session.editor_state.load_tensors(before)
    editor.configure({"path_merge": False})
    off = editor.next_state(session.state, record)
    assert default_tensors.keys() != session.editor_state.state_tensors().keys()
    assert any(not torch.equal(default[h]["up"], off[h]["up"]) for h in editor.targets)
    session.editor_state.load_tensors(before)
    editor.configure({"path_merge": True})
    on = editor.next_state(session.state, record)
    assert session.editor_state.chunks.node_coef
    assert all(torch.equal(default[h]["up"], on[h]["up"]) for h in editor.targets)
    paths = session.editor_state.forest.branches["main"].paths
    assert paths and all({"predecessor", "successor", "p"} <= set(step) for step in paths)
    assert any(not torch.equal(on[h]["up"], off[h]["up"]) for h in editor.targets)
    assert all(
        torch.equal(session.state[h][k], v)
        for h, parts in original.items()
        for k, v in parts.items()
    )


def test_write_versions_and_freezes_graph_keys(setup, monkeypatch):
    from kilora.compute import LayerSet, derived_keys

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"path_merge": True})
    hook = editor.targets[0]
    down = cl._down(session.state[hook]).clone()
    up = session.state[hook]["up"].float().reshape(down.shape[0], -1).clone()
    layer = LayerSet(
        router=editor.live_set.router_for(hook),
        keys_in=editor.live_set.experts[hook].keys_in,
        down=down,
        up=up,
        per_expert=down.shape[0] // editor.experts,
    )
    before = derived_keys(layer)
    captured = []
    monkeypatch.setattr(
        editor,
        "_path_merge",
        lambda chunks, state, shares, keeps, keys: captured.append(keys[hook]),
    )
    original = _record(session, target)
    mutated = False

    def measure(pert):
        nonlocal mutated
        if not mutated:
            session.state[hook]["up"].zero_()
            mutated = True
        return original.measure(pert)

    new = editor.next_state(session.state, GenerationRecord(images=1, measure=measure))
    assert editor.weight_version == 1
    assert not editor.derived_key_versions._keys
    for actual, expected in zip(captured[0], before, strict=True):
        torch.testing.assert_close(actual, expected)
    editor.next_state(new, _record(session, target))
    assert editor.weight_version == 2
    assert not editor.derived_key_versions._keys
    assert not torch.equal(captured[0][2], captured[1][2])


def test_cross_hook_merge_uses_both_measured_shares():
    from types import SimpleNamespace

    from kilora_editor_chunk import ChunkState

    from kilora.path_graph import PathGraphCache
    from kilora.sets import KeyedExperts

    names = ["first.to_q", "second.to_q"]
    ed = ChunkEditor(
        names,
        {h: [4, 4] for h in names},
        {"to_q": 1.0},
        [Chunk("common", torch.eye(32), torch.ones(32))],
    )
    keys = torch.eye(2)[:, None, :]
    live = SimpleNamespace(
        experts={h: KeyedExperts(keys_in=keys) for h in names},
        router_for=lambda _: torch.eye(4)[:2],
    )
    editor = object.__new__(cl.ChunkLive)
    editor.editor, editor.experts, editor.targets, editor.live_set = ed, 2, names, live
    editor.v0 = {h: torch.eye(4) for h in names}
    editor.col_expert = {h: torch.tensor([0, 0, 1, 1]) for h in names}
    editor.path_graph = PathGraphCache()
    editor.options = {"path_angle": 0.0}
    state = {h: {"down": torch.eye(4)} for h in names}
    chunks = ChunkState.new(ed)
    chunks.coef["common"][0, :16] = 0.1 * torch.eye(4).reshape(-1)
    chunks.coef["common"][0, 16:] = 0.2 * torch.eye(4).reshape(-1)
    before = chunks.expanded_u(ed)
    shares = {names[0]: [0.8, 0.0], names[1]: [0.25, 0.0]}
    editor._path_merge(chunks, state, shares, {})
    after = chunks.expanded_u(ed)
    expected = before[names[0]][0] + 0.2 * (
        before[names[1]][0] @ torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])) @ before[names[0]][0]
    )
    assert torch.allclose(after[names[0]][0], expected, atol=1e-6)
    assert torch.equal(after[names[1]], before[names[1]])
    assert torch.equal(after[names[0]][1], before[names[0]][1])
    spreading = ChunkState.new(ed)
    spreading.coef["common"] = chunks.coef["common"].clone()
    editor._write_target = {h: torch.zeros_like(v) for h, v in before.items()}
    editor._write_target[names[1]][0] = torch.eye(4)
    editor._path_merge(spreading, state, shares, {})
    assert not torch.equal(spreading.expanded_u(ed)[names[0]], after[names[0]])
    editor._write_target = None
    from kilora.compute import dropout_key

    rotated = ChunkState.new(ed)
    rotated.coef["common"] = chunks.coef["common"].clone()
    replay = ChunkState.new(ed)
    replay.restore(rotated.snapshot())
    baseline = ChunkState.new(ed)
    baseline.restore(rotated.snapshot())
    editor._path_merge(baseline, state, shares, {})
    editor.options["path_angle"] = 0.05
    with dropout_key(42, 3):
        editor._path_merge(rotated, state, shares, {})
    with dropout_key(42, 3):
        editor._path_merge(replay, state, shares, {})
    assert torch.equal(rotated.expanded_u(ed)[names[0]], replay.expanded_u(ed)[names[0]])
    out = rotated.expanded_u(ed)
    ref = baseline.expanded_u(ed)
    read = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    residual = (
        out[names[1]][0] @ read @ out[names[0]][0] - ref[names[1]][0] @ read @ ref[names[0]][0]
    ).norm()
    relative = residual / (ref[names[1]][0] @ read @ ref[names[0]][0]).norm().clamp_min(1e-30)
    assert float(relative) == pytest.approx(editor.rotation_residual, abs=1e-7)
    assert editor.rotation_residual_mean == pytest.approx(editor.rotation_residual)
    assert float(relative) < 0.0052


def test_path_rotation_order_and_self_loop(monkeypatch):
    from types import SimpleNamespace

    from kilora_editor_chunk import ChunkState

    from kilora.compute import dropout_key
    from kilora.path_graph import PathGraphCache, PathNode, PathStep
    from kilora.sets import KeyedExperts

    name = "block.to_q"
    ed = ChunkEditor(
        [name],
        {name: [4, 4]},
        {"to_q": 1.0},
        [Chunk("common", torch.eye(16), torch.ones(16))],
    )
    keys = torch.eye(2)[:, None, :]
    editor = object.__new__(cl.ChunkLive)
    editor.editor, editor.experts, editor.targets = ed, 2, [name]
    editor.live_set = SimpleNamespace(
        experts={name: KeyedExperts(keys_in=keys)}, router_for=lambda _: torch.eye(4)[:2]
    )
    editor.v0 = {name: torch.eye(4)}
    editor.col_expert = {name: torch.tensor([0, 0, 1, 1])}
    editor.path_graph, editor.options = PathGraphCache(), {"path_angle": 0.05}
    state = {name: {"down": torch.eye(4)}}
    initial = ChunkState.new(ed, experts=2)
    initial.coef["common"][0] = 0.1 * torch.eye(4).reshape(-1)
    initial.coef["common"][1] = 0.2 * torch.eye(4).reshape(-1)
    nodes = [PathNode(name, depth, expert) for depth in range(2) for expert in range(2)]
    steps = [
        PathStep(nodes[0], nodes[2], 0.5),
        PathStep(nodes[0], nodes[3], 0.5),
        PathStep(nodes[1], nodes[2], 0.5),
    ]
    keeps = {(name, 0): [torch.ones(2)]}

    def run(edges, angle):
        chunks = ChunkState.new(ed, experts=2)
        chunks.restore(initial.snapshot())
        editor.options["path_angle"] = angle
        monkeypatch.setattr(editor.path_graph, "get", lambda *args: edges)
        with dropout_key(42, 3):
            editor._path_merge(chunks, state, {}, keeps)
        return chunks.expanded_u(ed)[name]

    assert torch.equal(run(steps, 0.05), run(list(reversed(steps)), 0.05))
    assert torch.equal(run(steps[:1], 0.05), run(steps[:1], 0.0))


def test_path_rotation_projection_residual_full_span():
    from types import SimpleNamespace

    from kilora_editor_chunk import ChunkState

    from kilora.compute import dropout_key
    from kilora.path_graph import PathGraphCache
    from kilora.sets import KeyedExperts

    names = ["first.to_q", "second.to_q"]
    ed = ChunkEditor(
        names,
        {h: [4, 4] for h in names},
        {"to_q": 1.0},
        [Chunk("common", torch.eye(32), torch.ones(32))],
    )
    editor = object.__new__(cl.ChunkLive)
    editor.editor, editor.experts, editor.targets = ed, 1, names
    editor.live_set = SimpleNamespace(
        experts={h: KeyedExperts(keys_in=torch.ones(1, 1, 4)) for h in names},
        router_for=lambda _: torch.eye(4),
    )
    editor.v0 = {h: torch.eye(4) for h in names}
    editor.col_expert = {h: torch.zeros(4, dtype=torch.long) for h in names}
    editor.path_graph, editor.options = PathGraphCache(), {"path_angle": 0.05}
    state = {h: {"down": torch.eye(4)} for h in names}
    initial = ChunkState.new(ed, experts=1)
    initial.coef["common"][0, :16] = 0.1 * torch.eye(4).reshape(-1)
    initial.coef["common"][0, 16:] = 0.2 * torch.eye(4).reshape(-1)

    def run(angle):
        chunks = ChunkState.new(ed, experts=1)
        chunks.restore(initial.snapshot())
        editor.options["path_angle"] = angle
        with dropout_key(42, 3):
            editor._path_merge(chunks, state, {h: [1.0] for h in names}, {})
        return chunks.expanded_u(ed)

    baseline = run(0.0)
    rotated = run(0.05)
    expected = baseline[names[1]][0] @ baseline[names[0]][0]
    actual = rotated[names[1]][0] @ rotated[names[0]][0]
    relative = float((actual - expected).norm() / expected.norm())
    assert relative < 1e-5
    assert editor.rotation_residual == pytest.approx(relative, abs=1e-7)


def test_perturbation_records_actual_token_routing_separately():
    from kilora.compute import route_shares
    from kilora.recording import Perturbation

    name = "block.to_q"
    keys = torch.eye(2)[:, None, :]
    router = torch.eye(2)
    h = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    base = Perturbation({name: torch.eye(2)}, {name: torch.eye(2)})
    perturbation = cl.RoutedPerturbation(
        base, {name: (router, keys)}, observed_routing="token", observed_select="top1"
    )
    perturbation.delta(name, h)
    expected = route_shares(router_logits(h, keys, 1.0), "token", "top1")
    assert torch.equal(perturbation.observed_counts[name], expected.double())
    assert torch.equal(perturbation.observed_counts[name], torch.tensor([1.0, 2.0]).double())


def test_branch_store_fork_write_promote_and_save(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state
    original = st.chunks
    branch = st.fork(seed=17, temperature=0.8, depth=4)
    child = st.forest.branches[branch].state
    assert all(child.coef[k].data_ptr() == v.data_ptr() for k, v in original.coef.items())
    editor.checkout_branch(branch)
    editor.options["packs"] = ["common"]
    commit_write(session, editor, _record(session, target))
    expected = _ups(session)
    assert child.coef["common"].data_ptr() != original.coef["common"].data_ptr()
    assert child.coef["retro_game"].data_ptr() == original.coef["retro_game"].data_ptr()
    editor.checkout_branch("main")
    assert all(torch.equal(session.state[t]["up"], st.base[t]) for t in st.base)
    editor.checkout_branch(branch, promote=True)
    assert all(torch.equal(session.state[t]["up"], expected[t]) for t in expected)
    with pytest.raises(ValueError):
        st.delete(branch)
    name = session.save_point("branches")
    editor.checkout_branch("main")
    session.load(name)
    restored = session.editor_state
    assert restored.forest.head == branch and restored.forest.current == "main"
    assert set(restored.forest.branches) == {"main", branch, "branch-2", "branch-3"}
    imported = restored.forest.branches["branch-3"]
    assert (imported.parent, imported.seed, imported.temperature, imported.depth) == (
        "branch-2",
        17,
        0.8,
        4,
    )
    session.undo()
    assert set(restored.forest.branches) == {"main", branch}
    assert restored.forest.head == branch and restored.forest.current == "main"


def test_branch_undo_and_reset_preserve_forest(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    commit_write(session, editor, _record(session, target))
    st = session.editor_state
    main = {k: v.clone() for k, v in st.chunks.coef.items()}
    branch = st.fork()
    editor.checkout_branch(branch)
    commit_write(session, editor, _record(session, target))
    commit_write(session, editor, _record(session, target))
    session.undo()
    assert set(st.forest.branches) == {"main", branch}
    assert all(torch.equal(st.forest.branches["main"].state.coef[k], v) for k, v in main.items())
    assert st.forest.current == branch
    session.reset()
    assert all(not torch.count_nonzero(v) for v in st.chunks.coef.values())
    assert all(torch.equal(st.forest.branches["main"].state.coef[k], v) for k, v in main.items())
    session.undo()
    assert any(torch.count_nonzero(v) for v in st.chunks.coef.values())
    assert set(st.forest.branches) == {"main", branch}


def test_draw_and_node_coefficient_fork_roundtrip(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state
    key = ("common", next(iter(session.state)))
    st.chunks.node_coef[key] = torch.ones_like(st.chunks.coef["common"])
    original = st.chunks.node_coef[key]
    record = _record(session, target)
    record.extra["draw"] = {"seed": 41, "temperature": 0.7, "depth": 5}
    commit_write(session, editor, record)
    branch = st.forest.current
    assert branch != st.forest.head == "main"
    b = st.forest.branches[branch]
    assert (b.seed, b.temperature, b.depth) == (41, 0.7, 5)
    assert b.state.node_coef[key].data_ptr() == original.data_ptr()
    b.state.node_coef[key] = b.state.node_coef[key].clone()
    b.state.node_coef[key].add_(1)
    assert torch.equal(st.forest.branches["main"].state.node_coef[key], original)
    saved = st.state_tensors()
    st.load_tensors(saved)
    assert torch.equal(st.forest.branches[branch].state.node_coef[key], original + 1)
    assert torch.equal(st.forest.branches["main"].state.node_coef[key], original)
    session.undo()
    assert set(st.forest.branches) == {"main", branch}
    assert st.forest.current == branch


def test_alternating_branch_writes_and_undo(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state
    branch = st.fork()
    commit_write(session, editor, _record(session, target))
    main_one = _ups(session)
    editor.checkout_branch(branch)
    commit_write(session, editor, _record(session, target))
    child_one = _ups(session)
    editor.checkout_branch("main")
    commit_write(session, editor, _record(session, target))
    editor.checkout_branch(branch)
    commit_write(session, editor, _record(session, target))
    session.undo()
    assert all(torch.equal(session.state[t]["up"], child_one[t]) for t in child_one)
    editor.checkout_branch("main")
    session.undo()
    assert all(torch.equal(session.state[t]["up"], main_one[t]) for t in main_one)
    editor.checkout_branch(branch)
    assert all(torch.equal(session.state[t]["up"], child_one[t]) for t in child_one)


def test_load_undo_restores_exact_forest(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    commit_write(session, editor, _record(session, target))
    name = session.save_point("forest")
    st = session.editor_state
    child = st.fork(seed=9, temperature=0.5, depth=3)
    editor.checkout_branch(child, promote=True)
    commit_write(session, editor, _record(session, target))
    before = session.editor_snapshot()
    ups = _ups(session)
    session.load(name)
    assert len(st.forest.branches) == 3
    assert st.forest.current == st.forest.head == child
    session.undo()
    after = session.editor_snapshot()
    assert before.keys() == after.keys()
    assert all(torch.allclose(before[k], after[k], equal_nan=True) for k in before)
    assert all(torch.equal(session.state[t]["up"], ups[t]) for t in ups)


def test_zero_temperature_writes_current_branch(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    record = _record(session, target)
    record.extra["draw"] = {"seed": 41, "temperature": 0, "depth": 5}
    commit_write(session, editor, record)
    assert list(session.editor_state.forest.branches) == ["main"]
    assert session.editor_state.chunks.events == 1


def test_pristine_load_of_forest_can_be_undone(setup, tmp_path):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    branch = session.editor_state.fork(seed=11, temperature=1, depth=4)
    editor.checkout_branch(branch, promote=True)
    commit_write(session, editor, _record(session, target))
    name = session.save_point("tree")
    again = _fresh(session, tmp_path)
    other = editor_for(path)
    other.check(again)
    before = again.editor_snapshot()
    again.load(name)
    assert again.editor_state.forest.current == branch
    assert again.undo_log._entries[-1].branch == "main"
    again.undo()
    assert again.editor_state.forest.current == "main"
    after = again.editor_snapshot()
    assert before.keys() == after.keys()
    assert all(torch.allclose(before[k], after[k], equal_nan=True) for k in before)


def test_draw_forks_from_current_branch(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state
    first = st.fork()
    editor.checkout_branch(first)
    record = _record(session, target)
    record.extra["draw"] = {"seed": 12, "temperature": 0.6, "depth": 3}
    commit_write(session, editor, record)
    assert st.forest.branches[st.forest.current].parent == first
    assert st.forest.head == "main"


def test_spread_target_and_budget_in_current_branch(setup):
    from kilora.compute import read_dropout

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"path_angle": 0.0, "per_hook_cap": True, "diff": "one-sided"})
    st = session.editor_state
    branch = st.fork(seed=7, temperature=1.0, depth=2)
    st.checkout(branch)
    head = st.forest.branches[st.forest.head].state.snapshot()
    before = st.state_tensors()
    original = {h: parts["up"].clone() for h, parts in session.state.items()}
    editor.manifest["writer"]["total_cap"] = 0.01
    editor.manifest["writer"]["cap_frac"] = 0.002
    activations = {h: torch.ones(2, route.shape[1]) for h, (route, _) in editor.routes.items()}

    def measure(p):
        for h, x in activations.items():
            p.delta(h, x)
            n = editor.col_expert[h].numel()
            read_dropout(
                torch.ones(2, n),
                torch.ones(2, n),
                n // editor.experts,
                f"{h}#0",
                1,
                set_key=session.mod_id,
            )
        return float(sum(((p.du[h] - target[h]) ** 2).sum() for h in p.du))

    out = editor.next_state(session.state, GenerationRecord(images=1, measure=measure))
    assert st.forest.current == branch and st.forest.head != branch
    assert all(
        torch.equal(head["coef"][k], v)
        for k, v in st.forest.branches[st.forest.head].state.coef.items()
    )
    assert st.chunks.node_coef
    assert editor.path_report[0] > 0
    assert "Path merge:" in editor.write_note()
    sizes = {h: float((out[h]["up"] - original[h]).norm()) for h in editor.targets}
    assert (
        sum(n * n for n in sizes.values()) ** 0.5
        <= min(editor.writer.write_norm, editor.writer.total_cap) + 1e-6
    )
    for h, n in sizes.items():
        assert n <= editor._per_hook_cap() * editor.hook_scale[h] + 1e-6
    saved = st.state_tensors()
    st.restore_branch(before)
    assert not st.chunks.node_coef
    st.load_tensors(saved)
    for k, v in saved.items():
        assert torch.allclose(st.state_tensors()[k], v, rtol=0, atol=0, equal_nan=True)


def test_single_node_path_keeps_writer_result(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    before = session.editor_state.state_tensors()
    monkeypatch.setattr(editor.path_graph, "get", lambda *args: [])
    on = editor.next_state(session.state, _record(session, target))
    session.editor_state.load_tensors(before)
    editor.configure({"path_merge": False})
    off = editor.next_state(session.state, _record(session, target))
    assert all(torch.equal(on[h]["up"], off[h]["up"]) for h in editor.targets)


def test_path_graph_invalidation_on_branch_checkout(setup):
    session, path, _ = setup
    editor = editor_for(path)
    editor.check(session)
    st = session.editor_state
    branch = st.fork(seed=4, temperature=1.0, depth=2)
    editor.checkout_branch(branch)
    family = next(iter(st.chunks.coef))
    hook = editor.targets[0]
    delta = torch.zeros_like(st.chunks.coef[family])
    delta[:2, 0] = 0.1
    st.chunks.node_coef[(family, hook)] = delta
    keeps = {(hook, 0): [torch.zeros(editor.experts)] for hook in editor.targets}
    editor._path_merge(st.chunks, session.state, {}, keeps)
    editor.checkout_branch("main")
    editor._path_merge(st.chunks, session.state, {}, keeps)
    changed, total, transitions, all_transitions = editor.path_graph.stats
    assert changed == 2
    assert total == len(editor.targets) * editor.experts
    assert 0 < transitions < all_transitions


def _rendered_ups(session):
    return {t: ex.up.clone() for t, ex in session.sets()[session._live_index].experts.items()}


def test_pin_keeps_render_while_writes_accumulate_and_unpin_follows(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    rendered = _rendered_ups(session)
    session.pin()
    for _ in range(3):
        commit_write(session, editor, _record(session, target))
        assert all(torch.equal(v, _rendered_ups(session)[t]) for t, v in rendered.items())
    forest = session.editor_state.forest
    branch = session.pinned_write_branch
    assert forest.branches[branch].parent == "main"
    assert forest.branches[branch].state.events == 3
    assert len(session.commits) == 3
    assert "pinned=main@" in editor.write_note()
    latest = _ups(session)
    assert any(not torch.equal(latest[t], rendered[t]) for t in latest)
    session.unpin()
    assert all(torch.equal(v, _rendered_ups(session)[t]) for t, v in latest.items())


def test_pin_save_load_and_promote(setup, tmp_path):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    session.pin()
    pinned = _rendered_ups(session)
    commit_write(session, editor, _record(session, target))
    branch = session.pinned_write_branch
    latest = _ups(session)
    name = session.save_point("pinned")
    restored = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    restored.learning = True
    restored.load(name)
    other = editor_for(path)
    other.check(restored)
    assert restored.pinned == session.pinned
    assert restored.pinned_write_branch == branch
    assert all(torch.equal(v, _rendered_ups(restored)[t]) for t, v in pinned.items())
    commit_write(restored, other, _record(restored, target))
    assert restored.pinned_write_branch == branch
    assert restored.editor_state.forest.branches[branch].state.events == 2
    other.checkout_branch("main", promote=True)
    assert restored.pinned is None
    assert all(torch.equal(v, _rendered_ups(restored)[t]) for t, v in pinned.items())
    other.checkout_branch(branch, promote=True)
    assert any(not torch.equal(v, _rendered_ups(restored)[t]) for t, v in pinned.items())
    assert any(not torch.equal(v, latest[t]) for t, v in _rendered_ups(restored).items())


def test_merge_units_hand_example():
    history = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    equivalent, repeat, new = cl.ChunkLive._merge_units(torch.tensor([3.0, 4.0, 12.0]), history)
    assert equivalent == pytest.approx(13 / 3.5)
    assert repeat == pytest.approx(5 / 13)
    assert new == pytest.approx(12 / 13)


def test_merge_measurement_has_same_forward_count(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    initial = session.editor_snapshot()
    calls = []
    record = _record(session, target)
    measure = record.measure

    def count(trial):
        calls.append(1)
        return measure(trial)

    record.measure = count
    editor.configure({"path_merge": False})
    editor.next_state(session.state, record)
    off = len(calls)
    session.editor_state.load_tensors(initial)
    calls.clear()
    editor.configure({"path_merge": True})
    editor.next_state(session.state, record)
    print(f"measurement forwards: off={off}, on={len(calls)}")
    assert len(calls) == off
    assert off > 0


def test_merge_guard_is_separate_and_one_undo_takes_back_write_and_merge(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)

    def merge(chunks, state, shares, keeps, frozen_keys):
        family = editor.editor.chunks[0].family
        hook = editor.targets[0]
        chunks.node_coef[(family, hook)] = torch.ones_like(chunks.coef[family]) * 0.001
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    record = _record(session, target)
    trials = []

    def score(trial):
        if trial is not None:
            trials.append(trial)
        return {"gap": 1.0 if len(trials) < 2 else 0.0, "spread": 1.0}

    record.guard_score = score
    report = commit_write(session, editor, record)
    assert "Merge refused" in report
    assert len(session.commits) == 1
    assert not session.editor_state.chunks.node_coef

    record.guard_score = lambda trial: {"gap": 1.0, "spread": 1.0}
    before_write = {t: parts["up"].clone() for t, parts in session.state.items()}
    report = commit_write(session, editor, record)
    assert "source=merge" in report
    assert "Merge refused" not in report
    assert len(session.commits) == 2
    assert session.commits[-1].source.endswith("+merge")
    assert session.editor_state.chunks.node_coef
    session.undo()
    assert len(session.commits) == 1
    assert all(torch.equal(session.state[t]["up"], v) for t, v in before_write.items())
    assert not session.editor_state.chunks.node_coef


def test_backlog_commits_each_write_with_its_merge(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)

    def merge(chunks, state, shares, keeps, frozen_keys):
        family = editor.editor.chunks[0].family
        hook = editor.targets[0]
        key = (family, hook)
        chunks.node_coef[key] = (
            chunks.node_coef.get(key, torch.zeros_like(chunks.coef[family])) + 0.001
        )
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    records = [_record(session, target), _record(session, target)]
    report = commit_write(session, editor, records[0], records=records)
    assert "source=merge" in report
    assert len(session.commits) == 2
    assert all(commit.source.endswith("+merge") for commit in session.commits)
    assert session.editor_state.chunks.events == 2
    assert session.editor_state.chunks.node_coef
    session.undo()
    assert len(session.commits) == 1
    assert session.editor_state.chunks.node_coef
    session.undo()
    assert not session.editor_state.chunks.node_coef
