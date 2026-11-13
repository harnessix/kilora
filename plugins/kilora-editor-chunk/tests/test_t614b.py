# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from kilora_editor_chunk.editor import BranchStore, Chunk, ChunkEditor, ChunkState, ChunkWriter
from kilora_editor_chunk.plugin import ChunkEditorState, ChunkLive
from path_merge_reference import reference_path_merge

from kilora.path_graph import PathGraphCache
from kilora.sets import KeyedExperts


def stand_in(mode):
    names = ["first.to_q", "second.to_q"]
    q = torch.zeros(2, 32)
    q[0, 0] = 1
    q[1, 16 + 4] = 1
    ed = ChunkEditor(
        names, {h: [4, 4] for h in names}, {"to_q": 1.0}, [Chunk("common", q, torch.ones(2))]
    )
    live = object.__new__(ChunkLive)
    live.editor, live.targets, live.experts = ed, names, 1
    live.options = {"path_angle": 0.0, "merge_residual": mode}
    live.v0 = {h: torch.eye(4) for h in names}
    live.col_expert = {h: torch.zeros(4, dtype=torch.long) for h in names}
    live.live_set = SimpleNamespace(
        experts={h: KeyedExperts(keys_in=torch.ones(1, 1, 1)) for h in names},
        router_for=lambda _: torch.ones(1, 4),
    )
    live.path_graph = PathGraphCache()
    chunks = ChunkState.new(ed, experts=1)
    chunks.coef["common"][0] = torch.tensor([1.0, 1.0])
    state = {h: {"down": torch.eye(4)} for h in names}
    return live, chunks, state


def test_merge_keeps_outside_direction_and_drop_replays_reference():
    keep, state, weights = stand_in("keep")
    drop, dropped, _ = stand_in("drop")
    before = state.expanded_u(keep.editor)
    shares = {h: [1.0] for h in keep.targets}
    keep._path_merge(state, weights, shares, {})
    drop._path_merge(dropped, weights, shares, {})
    change = state.expanded_u(keep.editor)[keep.targets[0]][0] - before[keep.targets[0]][0]
    expected = torch.zeros(4, 4)
    expected[1, 0] = 1
    torch.testing.assert_close(change, expected, atol=1e-5, rtol=0)
    assert (
        torch.count_nonzero(
            dropped.expanded_u(drop.editor)[drop.targets[0]][0] - before[drop.targets[0]][0]
        )
        == 0
    )
    assert keep.merge_residual_share > 0.999
    assert drop.merge_residual_share == 0
    assert torch.equal(
        state.node_coef["common", keep.targets[0]], dropped.node_coef["common", drop.targets[0]]
    )


def test_drop_matches_previous_merge_bit_for_bit():
    current, chunks, state = stand_in("drop")
    previous, old, _ = stand_in("drop")
    shares = {h: [1.0] for h in current.targets}
    reference_path_merge(previous, old, state, shares, {})
    current._path_merge(chunks, state, shares, {})
    assert chunks.node_coef.keys() == old.node_coef.keys()
    for key in chunks.node_coef:
        assert torch.equal(chunks.node_coef[key], old.node_coef[key])
    for hook, value in chunks.expanded_u(current.editor).items():
        assert torch.equal(value, old.expanded_u(previous.editor)[hook])


def test_residual_snapshot_fork_compact_and_editor_serialization():
    live, chunks, _ = stand_in("keep")
    hook = live.targets[0]
    delta = torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    chunks.add_residual(hook, 0, delta)
    before = chunks.expanded_u(live.editor)[hook].clone()
    forest = BranchStore(chunks)
    branch = forest.fork()
    forest.checkout(branch)
    forest.state.add_residual(hook, 0, delta)
    torch.testing.assert_close(chunks.expanded_u(live.editor)[hook], before)
    forest.state.compact_residual(1)
    torch.testing.assert_close(
        forest.state.expanded_u(live.editor)[hook], before + delta, atol=1e-5, rtol=0
    )
    st = ChunkEditorState(live.editor, {h: torch.zeros(4, 4) for h in live.targets}, 1)
    st.forest = forest
    st._branch_regions["main"] = st.regions
    st._branch_regions[branch] = st.regions
    saved = st.state_tensors()
    restored = ChunkEditorState(live.editor, st.base0, 1)
    restored.load_tensors(saved)
    torch.testing.assert_close(
        restored.chunks.expanded_u(live.editor)[hook], forest.state.expanded_u(live.editor)[hook]
    )
    restored.checkout("main")
    torch.testing.assert_close(restored.chunks.expanded_u(live.editor)[hook], before)


def test_cap_uses_real_residual_change():
    live, chunks, _ = stand_in("keep")
    live.options["per_hook_cap"] = False
    live.manifest = {"writer": {}}
    before = chunks.snapshot()
    chunks.add_residual(live.targets[0], 0, torch.eye(4))
    live.path_report = None
    writer = ChunkWriter(live.editor, [0.0], [1.0], [0.1], write_norm=0.5, total_cap=None)
    live._bound_path_write(chunks, before, writer)
    old = ChunkState.new(live.editor, 1)
    old.restore(before)
    change = {
        h: chunks.expanded_u(live.editor)[h] - old.expanded_u(live.editor)[h] for h in live.targets
    }
    assert abs(live._expanded_norm(change) - 0.5) < 1e-6


def test_residual_state_file_round_trip(tmp_path):
    live, chunks, _ = stand_in("keep")
    hook = live.targets[0]
    chunks.add_residual(hook, 0, torch.eye(4) * 0.25)
    path = tmp_path / "state.safetensors"
    chunks.save(path, live.editor)
    restored = ChunkState.load(path)
    torch.testing.assert_close(
        restored.expanded_u(live.editor)[hook], chunks.expanded_u(live.editor)[hook]
    )
