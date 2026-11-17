# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import kilora_editor_chunk.plugin as cl
import pytest
import torch
from kilora_editor_chunk import Chunk, ChunkEditor

from kilora.compute import router_logits
from kilora.editors import available_editors, commit_write, editor_for
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
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    path = _packs(session, tmp_path)
    g = torch.Generator().manual_seed(5)
    target = {
        t: torch.randn(int(p["up"].shape[-1]), 4, generator=g) for t, p in session.state.items()
    }
    return session, path, target


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


@pytest.mark.parametrize("merge_on", [True, False])
def test_write_in_inference_mode(setup, merge_on):
    session, path, target = setup
    with torch.inference_mode():
        editor = editor_for(path)
        editor.check(session)
        editor.configure({"path_merge": merge_on})
        for expert in editor.live_set.experts.values():
            for field in ("keys_in", "keys_out", "rows_in", "rows_out"):
                value = getattr(expert, field)
                if value is not None:
                    setattr(expert, field, value.clone())
                    assert getattr(expert, field).is_inference()
        weight = torch.randn(4, 4)
        inputs = torch.randn(1, 4)

        def measure(trial):
            torch.nn.functional.linear(inputs, weight)
            return float(sum(((trial.du[t] - target[t]) ** 2).sum() for t in trial.du))

        assert "1 commits" in commit_write(
            session, editor, GenerationRecord(images=1, measure=measure)
        )


def test_path_merge_accepts_frozen_and_live_keys(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    commit_write(session, editor, _record(session, target))
    chunks = session.editor_state.chunks
    editor._path_merge(chunks, session.state, {}, {})


@pytest.mark.parametrize("measure", ["compressed", "full"])
def test_live_direction_estimates_share_the_mean_until_commit(setup, measure):
    from kilora.write_estimates import copy_write_state

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"measure": measure, "path_merge": False})
    before = copy_write_state(session.state)
    history = session.editor_snapshot()
    with torch.no_grad():
        for attempt in range(2):
            proposed = editor.next_state(session.state, _record(session, target))
            assert proposed is not None
            assert session._write_estimates
            assert all(row[1] == attempt + 1 for row in session._write_estimates.values())
            assert f"over {attempt + 1} estimates" in editor.write_note()
            session.editor_state.load_tensors(history)
    assert all(torch.equal(parts["up"], before[t]["up"]) for t, parts in session.state.items())
    assert "1 commits" in commit_write(session, editor, _record(session, target))
    assert session._write_estimates == {}
    session.undo()
    assert session._write_estimates == {}
