# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from kilora_editor_chunk.editor import Chunk, ChunkEditor, ChunkState, ChunkWriter
from kilora_editor_chunk.terms import DirectionPool, TermState


def test_terms_norm_combine_compact_and_save(tmp_path):
    pool = DirectionPool()
    pool.add("x", torch.tensor([1.0, 0.0]), "chunk")
    pool.add("y", torch.tensor([0.6, 0.8]), "chunk")
    pool.add("a", torch.tensor([1.0, 0.0]), "v0")
    pool.add("b", torch.tensor([0.6, 0.8]), "v0")
    state = TermState(pool)
    state.add("hook", 0, "x", "a", 1)
    state.add("hook", 0, "x", "a", 2)
    state.add("hook", 0, "y", "b", -0.5)
    state.add("hook", 0, "x", "b", 0.3)
    assert len(state.terms["hook", 0]) == 3
    up, down = state.stacked("hook")
    dense = up @ down
    assert abs(state.change_norm() - float(dense.norm())) < 1e-6
    old = state.snapshot()
    state.add("hook", 0, "y", "a", 0.1)
    assert list(state.delta(old)) == [("hook", 0)]
    state.restore(old)
    state.compact(2)
    up, down = state.stacked("hook")
    assert len(state.terms["hook", 0]) == 2
    assert torch.allclose(up @ down, dense, atol=1e-5)
    path = tmp_path / "terms.pt"
    state.save(path)
    loaded = TermState.load(path)
    up, down = loaded.stacked("hook")
    assert torch.allclose(up @ down, dense, atol=1e-5)


def test_chunk_norm():
    hook = "layer.to_q"
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]])
    editor = ChunkEditor([hook], {hook: [2, 2]}, {"to_q": 1.0}, [Chunk("common", q, torch.ones(2))])
    state = ChunkState.new(editor, experts=1)
    state.coef["common"][0] = torch.tensor([1.0, 2.0])
    writer = ChunkWriter(editor, [0.0], [1.0], [0.1], 1.0)
    assert abs(writer._norm(state.coef) - float(state.expanded_u(editor)[hook].norm())) < 1e-6
    assert abs(editor.coherence() - 0.6) < 1e-6


def test_pool_near_duplicate():
    pool = DirectionPool()
    pool.add("first", torch.tensor([1.0, 0.0]), "chunk")
    name, scale = pool.add("second", torch.tensor([-2.0, 0.0]), "chunk", merge=True)
    assert name == "first" and scale == -2.0


def test_loader_accepts_overlapping_chunks(tmp_path):
    hook = "layer.to_q"
    editor = ChunkEditor(
        [hook],
        {hook: [2, 2]},
        {"to_q": 1.0},
        [
            Chunk("common", torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.ones(1)),
            Chunk("family", torch.tensor([[0.6, 0.8, 0.0, 0.0]]), torch.ones(1)),
        ],
    )
    files = editor.save_packs(tmp_path, "1.0.0", "z-image", "Apache-2.0", storage="float32")
    loaded = ChunkEditor.load_packs([files["common"], files["family"]])
    assert abs(loaded.coherence() - 0.6) < 0.002


def test_term_delta_is_small_and_replays():
    pool = DirectionPool()
    pool.add("out", torch.tensor([1.0, 0.0]), "chunk")
    pool.add("in", torch.tensor([0.0, 1.0]), "v0")
    state = TermState(pool)
    state.add("h", 0, "out", "in", 3.0)
    previous = state.snapshot()
    state.add("h", 0, "out", "in", 2.0)
    delta = state.delta(previous)
    assert len(delta["h", 0]) == 1 and delta["h", 0][0].strength == 2.0
    replay = TermState(pool)
    replay.restore(previous)
    replay.apply_delta(delta)
    assert replay.terms["h", 0][0].strength == 5.0


def test_inline_direction_delta_does_not_repeat_unchanged_terms():
    state = TermState(DirectionPool())
    state.add("h", 0, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 2.0)
    previous = state.snapshot()
    assert state.delta(previous) == {}
    state.terms["h", 0][0].strength = 3.0
    delta = state.delta(previous)
    assert len(delta["h", 0]) == 1 and delta["h", 0][0].strength == 1.0
    replay = TermState(state.pool)
    replay.restore(previous)
    replay.apply_delta(delta)
    assert len(replay.terms["h", 0]) == 1 and replay.terms["h", 0][0].strength == 3.0
