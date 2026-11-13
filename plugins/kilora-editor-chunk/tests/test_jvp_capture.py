# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from types import SimpleNamespace

import torch
from kilora_editor_chunk.plugin import ChunkLive
from test_attention_inner_link import A, B, C, IdentityBinding, Patcher, Toy
from test_chunk_plugin import _packs

from kilora.comfyui import live_write, patching
from kilora.cross_layer import sequential_links
from kilora.live import open_session
from kilora.recording import PathRecorder
from kilora.testing import make_live_mod


def _record(recorder, targets, outputs):
    with recorder.recording():
        for target, output in zip(targets, outputs, strict=True):
            recorder.observe(target, output)
    return recorder.finish()


def _bytes(stats):
    return sum(
        t.numel() * t.element_size()
        for rows in stats.pair_inputs.values()
        for pair in rows
        for t in pair
    )


def test_anima_records_no_raw_rows_but_keeps_links(tmp_path, monkeypatch):
    from kilora.live_sets import LIVE_PRESETS

    monkeypatch.setitem(
        LIVE_PRESETS, "light", dataclasses.replace(LIVE_PRESETS["light"], target_stride=1)
    )
    model = Toy()
    targets = (A, B, C)
    make_live_mod(tmp_path / "live.kmod", model=model, targets=targets)
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    session.editor_options = {"cross_layer": "jvp"}
    editor = ChunkLive(str(_packs(session, tmp_path)))
    monkeypatch.setattr(patching, "sessions_of", lambda _: [session])
    monkeypatch.setattr(patching, "detect_family", lambda _: IdentityBinding())
    monkeypatch.setattr(live_write, "async_enabled", lambda: False)
    _, handle = live_write.record(Patcher(model), editor)
    assert not session.cross_layer_blocks
    assert set(session.cross_layer_inner) == {(A, B)}
    outputs = [torch.randn(300, 8) for _ in targets]
    old = PathRecorder(handle.recorder.a, editor.v0, pair_targets=targets, capture_pair_inputs=True)
    before = _bytes(_record(old, targets, outputs))
    stats = _record(handle.recorder, targets, outputs)
    after = _bytes(stats)
    print(f"pair input bytes: before={before}, after={after}")
    assert before == 2 * 2 * 256 * 8 * 4
    assert after == 0 and not stats.pair_inputs
    assert set(stats.pairs) == {(A, B), (B, C)}
    editor._update_links(SimpleNamespace(stats=stats))
    editor.begin_write()
    assert editor._cross_link((A, B)).residual_variance == 0
    editor._cross_link((B, C))
    assert "cross_layer_jvp=1/2" in editor.finish_write(session, [])


def test_sequential_replay_keeps_rows(tmp_path, monkeypatch):
    import kilora_editor_chunk.editor as chunk_editor

    from kilora.live_sets import LIVE_PRESETS

    monkeypatch.setitem(
        LIVE_PRESETS, "light", dataclasses.replace(LIVE_PRESETS["light"], target_stride=1)
    )
    monkeypatch.setattr(chunk_editor, "hook_kind", lambda name: "to_v")
    model = torch.nn.Module()
    model.seq = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    targets = ("seq.0", "seq.1")
    make_live_mod(tmp_path / "live.kmod", model=model, targets=targets)
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    editor = ChunkLive(str(_packs(session, tmp_path)))
    editor.check(session)
    editor.configure({"cross_layer": "jvp"})
    session.cross_layer_blocks = sequential_links(model, editor.targets)
    session.cross_layer_inner = {}
    pair = tuple(editor.targets)
    x = torch.randn(20, 8)
    stats = _record(editor.recorder(session), pair, (x, session.cross_layer_blocks[pair](x)))
    assert set(stats.pair_inputs) == {pair}
    assert set(stats.pairs) == {pair}
    editor._update_links(SimpleNamespace(stats=stats))
    editor.begin_write()
    assert editor._cross_link(pair) is not session.editor_state.links[pair]
    assert "cross_layer_jvp=1/1" in editor.finish_write(session, [])


def test_boolean_and_selected_pairs():
    targets = ("a", "b", "c", "d")
    basis = {target: torch.eye(3) for target in targets}
    outputs = [torch.randn(5, 3) for _ in targets]
    for capture, expected in (
        (True, {("a", "b"), ("b", "c"), ("c", "d")}),
        (False, set()),
        ({("b", "c")}, {("b", "c")}),
    ):
        stats = _record(
            PathRecorder(basis, basis, pair_targets=targets, capture_pair_inputs=capture),
            targets,
            outputs,
        )
        assert set(stats.pair_inputs) == expected
        assert set(stats.pairs) == {("a", "b"), ("b", "c"), ("c", "d")}
