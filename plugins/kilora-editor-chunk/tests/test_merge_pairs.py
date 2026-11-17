# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from test_chunk_plugin import _record, setup

from kilora.editors import commit_write, editor_for
from kilora.live import read_save


def test_merge_pairs_save_load_and_undo(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    session.set_options(editor_options={"keep_merge_pairs": True})
    editor.configure(session.editor_options)

    def merge(chunks, state, shares, keeps, frozen_keys):
        family, hook = editor.editor.chunks[0].family, editor.targets[0]
        key = (family, hook)
        chunks.node_coef[key] = (
            chunks.node_coef.get(key, torch.zeros_like(chunks.coef[family])) + 0.001
        )
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    for _ in range(2):
        assert "source=merge" in commit_write(session, editor, _record(session, target))
    branch = session.editor_state.forest.branches["main"]
    assert [pair[0] for pair in branch.merge_pairs] == [1, 2]
    name = session.save_point("pairs")
    saved = read_save(session.saves_dir / session.mod_id / f"{name}.ksave")
    session.editor_state.load_tensors(saved.editor_state)
    assert len(session.editor_state.forest.branches["main"].merge_pairs) == 2
    assert all(torch.equal(session.editor_state.merge_v0[k], v) for k, v in editor.v0.items())
    session.undo()
    assert len(session.editor_state.forest.branches["main"].merge_pairs) == 1


def test_default_off_has_no_new_state_tensors(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    commit_write(session, editor, _record(session, target))
    before = session.editor_snapshot()
    session.undo()
    editor.configure({"keep_merge_pairs": False})
    commit_write(session, editor, _record(session, target))
    after = session.editor_snapshot()
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, equal_nan=True)


def test_merge_pair_limit_and_undo(setup, monkeypatch):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"keep_merge_pairs": True})
    branch = session.editor_state.forest.branches["main"]
    snap = branch.state.snapshot()
    branch.merge_pairs = [(i, snap, snap) for i in range(32)]

    def merge(chunks, state, shares, keeps, frozen_keys):
        family, hook = editor.editor.chunks[0].family, editor.targets[0]
        chunks.node_coef[(family, hook)] = torch.ones_like(chunks.coef[family]) * 0.001
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    assert "source=merge" in commit_write(session, editor, _record(session, target))
    pairs = session.editor_state.forest.branches["main"].merge_pairs
    assert len(pairs) == 32 and pairs[0][0] == 1
    session.undo()
    pairs = session.editor_state.forest.branches["main"].merge_pairs
    assert len(pairs) == 32 and pairs[0][0] == 0
