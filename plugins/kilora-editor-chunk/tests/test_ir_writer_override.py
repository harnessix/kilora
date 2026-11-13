# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from kilora_editor_chunk import ChunkState, ChunkWriter
from test_chunk_writer import NAMES, SHAPES, _editor, _loss_fn

from kilora import ir


def test_live_write_override_matches_zero_native_write():
    editor = _editor()
    original = ChunkState.new(editor, experts=2)
    altered = ChunkState.new(editor, experts=2)
    baseline = ChunkState.new(editor, experts=2)
    common = dict(per_event=4, measure="full")
    native = ChunkWriter(editor, [0.7], [1.0], [0.1], 0.0, **common)
    modified = ChunkWriter(
        editor,
        [0.7],
        [1.0],
        [0.1],
        0.2,
        ir_parts={"live_write": ir.PartOverride("default", parameters={"write_scale": 0.0})},
        **common,
    )
    unchanged = ChunkWriter(editor, [0.7], [1.0], [0.1], 0.2, **common)
    loss = _loss_fn({name: torch.ones(*SHAPES[name]) for name in NAMES})
    for writer, state in ((native, original), (modified, altered), (unchanged, baseline)):
        writer.step(state, loss, {"a": [0.4, 0.6]}, 4)
    assert all(torch.equal(original.coef[name], altered.coef[name]) for name in original.coef)
    assert any(not torch.equal(baseline.coef[name], altered.coef[name]) for name in baseline.coef)
