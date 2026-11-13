# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace

import torch
from kilora_editor_chunk import Chunk, ChunkEditor, ChunkState
from kilora_editor_chunk.plugin import ChunkLive
from path_merge_reference import reference_next_state, reference_path_merge
from test_chunk_plugin import _record, setup

from kilora.compute import dropout_key
from kilora.path_graph import PathGraphCache
from kilora.sets import KeyedExperts


def stand_in():
    names = [f"block{i}.to_q" for i in range(4)]
    dim = 16 * len(names)
    ed = ChunkEditor(
        names,
        {h: [4, 4] for h in names},
        {"to_q": 1.0},
        [Chunk("common", torch.eye(dim), torch.ones(dim))],
    )
    keys = torch.eye(2)[:, None, :]
    editor = object.__new__(ChunkLive)
    editor.editor, editor.experts, editor.targets = ed, 2, names
    editor.live_set = SimpleNamespace(
        experts={h: KeyedExperts(keys_in=keys) for h in names},
        router_for=lambda _: torch.eye(4)[:2],
    )
    editor.v0 = {h: torch.eye(4) for h in names}
    editor.col_expert = {h: torch.tensor([0, 0, 1, 1]) for h in names}
    editor.path_graph, editor.options = PathGraphCache(), {"path_angle": 0.05}
    editor.weight_version = 0
    branch = SimpleNamespace(paths=[])
    forest = SimpleNamespace(branches={"main": branch}, current="main")
    editor.session = SimpleNamespace(editor_state=SimpleNamespace(forest=forest))
    editor._write_target = {h: torch.full((2, 4, 4), 0.001) for h in names}
    chunks = ChunkState.new(ed, experts=2)
    generator = torch.Generator().manual_seed(8)
    chunks.coef["common"] = torch.randn(2, dim, generator=generator) * 0.02
    state = {h: {"down": torch.eye(4)} for h in names}
    shares = {h: [0.6, 0.4] for h in names}
    keeps = {(h, 0): [torch.ones(2)] for h in names}
    return editor, chunks, state, shares, keeps


def test_five_writes_match_reference_and_record_cost():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old, old_chunks, state, shares, keeps = stand_in()
        new, new_chunks, _, _, _ = stand_in()
        old_times, new_times = [], []
        for i in range(5):
            for chunks in (old_chunks, new_chunks):
                chunks.coef["common"] = chunks.coef["common"] + 0.0001
            with dropout_key(42, i):
                started = time.perf_counter()
                reference_path_merge(old, old_chunks, state, shares, keeps)
                old_times.append((time.perf_counter() - started, dict(old.path_merge_timings)))
            with dropout_key(42, i):
                started = time.perf_counter()
                new._path_merge(new_chunks, state, shares, keeps)
                new_times.append((time.perf_counter() - started, dict(new.path_merge_timings)))
            for hook, actual in new_chunks.expanded_u(new.editor).items():
                torch.testing.assert_close(
                    actual, old_chunks.expanded_u(old.editor)[hook], rtol=1e-4, atol=1e-6
                )
        print("before:", old_times, "after:", new_times, flush=True)
        first = min(t[0] for t in new_times[:2])
        last = min(t[0] for t in new_times[-2:])
        assert last <= 1.5 * first
    finally:
        torch.set_num_threads(previous_threads)


def test_off_is_bit_identical_to_original(setup):
    from kilora.editors import editor_for

    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"path_merge": False})
    before = session.editor_state.state_tensors()
    expected = reference_next_state(editor, session.state, _record(session, target))
    expected_state = session.editor_state.state_tensors()
    session.editor_state.load_tensors(before)
    actual = editor.next_state(session.state, _record(session, target))
    actual_state = session.editor_state.state_tensors()
    assert expected_state.keys() == actual_state.keys()
    for key in expected_state:
        torch.testing.assert_close(
            expected_state[key], actual_state[key], rtol=0, atol=0, equal_nan=True
        )
    for hook in expected:
        for key in expected[hook]:
            assert torch.equal(expected[hook][key], actual[hook][key])
