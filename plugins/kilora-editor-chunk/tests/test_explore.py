# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from kilora_editor_chunk.editor import ChunkWriter
from kilora_editor_chunk.plugin import ChunkEditorState
from test_chunk_writer import NAMES, SHAPES, _editor, _loss_fn


def test_score_prior_counts_and_cap():
    chunk = _editor().chunks[0]
    chunk.eigenvalues = torch.tensor([4.0, 3.0] + [1.0] * 10)
    counts = torch.tensor([0.0, 0.0] + [1.0] * 10)
    scale = ChunkWriter.exploration_scale(chunk, counts, 100.0)
    assert scale.max() <= 3
    scale = ChunkWriter.exploration_scale(chunk, counts, 0.5)
    assert scale[0] > scale[1]
    counts[0] = 100
    scale = ChunkWriter.exploration_scale(chunk, counts, 0.5)
    assert scale[0] < scale[1]


def test_writer_parity_counts_and_roundtrip():
    ed = _editor()
    base = {h: torch.zeros(*SHAPES[h]) for h in NAMES}
    a = ChunkEditorState(ed, base, 2)
    b = ChunkEditorState(ed, base, 2)
    target = {h: torch.ones(*SHAPES[h]) for h in NAMES}
    common = dict(times=[0.7], weights=[1.0], eps=[0.1], write_norm=0.2, measure="full")
    ChunkWriter(ed, **common).step(a.chunks, _loss_fn(target), {}, 1)
    ChunkWriter(ed, **common, explore=1, draw=False).step(b.chunks, _loss_fn(target), {}, 1)
    for family in a.chunks.coef:
        assert torch.equal(a.chunks.coef[family], b.chunks.coef[family])
    saved = b.state_tensors()
    restored = ChunkEditorState(ed, base, 2)
    restored.load_tensors(saved)
    for key, value in b.chunks.extra.items():
        assert torch.equal(value, restored.chunks.extra[key])
    old = ChunkEditorState(ed, base, 2)
    old.load_tensors(a.state_tensors())
    assert not old.chunks.extra
    parent = {k: v.clone() for k, v in b.chunks.coef.items()}
    b.checkout(b.fork(seed=1, temperature=1.0, depth=3))
    before = b.chunks.extra["explore_count/common"].clone()
    for history in b.chunks.hist.values():
        history.fill_(float("nan"))
    ChunkWriter(ed, **common, explore=1, draw=True, decay=0).step(
        b.chunks, lambda p, t, s: 0.0, {}, 1
    )
    torch.testing.assert_close(b.chunks.extra["explore_count/common"], before * 0.9)
    b.checkout("main")
    assert all(torch.equal(b.chunks.coef[k], v) for k, v in parent.items())


def test_zero_explore_bypasses_bonus_for_draw_and_normal(monkeypatch):
    ed = _editor()
    base = {h: torch.zeros(*SHAPES[h]) for h in NAMES}
    target = {h: torch.ones(*SHAPES[h]) for h in NAMES}
    common = dict(times=[0.7], weights=[1.0], eps=[0.1], write_norm=0.2, measure="full")
    states = []
    for draw in (False, True):
        state = ChunkEditorState(ed, base, 2)
        monkeypatch.setattr(
            ChunkWriter,
            "exploration_scale",
            lambda *args: (_ for _ in ()).throw(AssertionError("bonus called")),
        )
        ChunkWriter(ed, **common, draw=draw).step(state.chunks, _loss_fn(target), {}, 1)
        states.append(state.state_tensors())
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(torch.nan_to_num(states[0][key]), torch.nan_to_num(states[1][key]))
