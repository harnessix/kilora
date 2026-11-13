# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from kilora_editor_chunk.plugin import ChunkLive
from test_chunk_plugin import _packs, _record

from kilora.cross_layer import sequential_links
from kilora.editors import commit_write
from kilora.live import open_session
from kilora.recording import PathRecorder
from kilora.testing import make_live_mod


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_v = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(8, 8))


class AnimaToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.transformer_blocks[0].attn1 = Attention()


@pytest.mark.parametrize(
    "topology,mode",
    [("attention", "jvp"), ("sequence", "jvp"), ("sequence", "rls"), ("sequence", "off")],
)
def test_write_reports_pair_coverage(tmp_path, topology, mode, monkeypatch):
    from kilora.live_sets import LIVE_PRESETS

    monkeypatch.setitem(
        LIVE_PRESETS, "light", dataclasses.replace(LIVE_PRESETS["light"], target_stride=1)
    )
    if topology == "attention":
        model = AnimaToy()
        targets = ("transformer_blocks.0.attn1.to_v", "transformer_blocks.0.attn1.to_out.0")
    else:
        model = torch.nn.Module()
        model.seq = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        targets = ("seq.0", "seq.1")
        import kilora_editor_chunk.editor as chunk_editor

        monkeypatch.setattr(chunk_editor, "hook_kind", lambda name: "to_v")
    make_live_mod(tmp_path / "live.kmod", model=model, targets=targets)
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    path = _packs(session, tmp_path)
    editor = ChunkLive(str(path))
    editor.check(session)
    editor.configure({"cross_layer": mode})
    session.cross_layer_blocks = sequential_links(model, editor.targets)
    pair = tuple(editor.targets)
    if topology == "sequence":
        assert pair in session.cross_layer_blocks
    else:
        assert not session.cross_layer_blocks
    recorder = PathRecorder(
        {h: session.state[h]["down"].reshape(-1, 8) for h in editor.targets},
        editor.v0,
        pair_targets=editor.targets,
        capture_pair_inputs=True,
    )
    x = torch.randn(5, 8)
    y = session.cross_layer_blocks[pair](x) if topology == "sequence" else x
    with recorder.recording():
        recorder.observe(pair[0], x)
        recorder.observe(pair[1], y)
    target = {h: torch.randn(int(p["up"].shape[-1]), 4) for h, p in session.state.items()}
    record = _record(session, target)
    record.stats = recorder.finish()
    report = commit_write(session, editor, record)
    warning = "Cross-layer jvp is not available for these layers. RLS links are used."
    if mode == "jvp":
        assert f"cross_layer_jvp={0 if topology == 'attention' else 1}/1" in report
        if topology == "attention":
            assert 'cross_layer_fallback="no isolated replay or paired activations"' in report
            assert warning in report
        else:
            assert "cross_layer_fallback=" not in report
            assert warning not in report
    else:
        assert "cross_layer_jvp=" not in report
        assert warning not in report


def test_partial_fallback_and_per_write_reset():
    from types import SimpleNamespace

    editor = object.__new__(ChunkLive)
    editor.options = {"cross_layer": "jvp"}
    editor.session = SimpleNamespace(
        editor_state=SimpleNamespace(links={}),
        cross_layer_blocks={
            ("a", "b"): lambda x: x,
            ("b", "c"): lambda x: x + 3,
        },
        state={name: {"down": torch.eye(4)} for name in ("a", "b", "c")},
    )
    editor.v0 = {name: torch.eye(4) for name in ("a", "b", "c")}
    x = torch.randn(3, 4)
    editor._jvp_samples = {pair: [(x, x)] for pair in editor.session.cross_layer_blocks}
    editor.begin_write()
    for _ in range(2):
        editor._cross_link(("a", "b"))
        editor._cross_link(("b", "c"))
    report = editor.finish_write(editor.session, [])
    assert (
        'cross_layer_jvp=1/2 cross_layer_fallback="link replay does not match recorded output"'
        in report
    )
    assert "RLS links are used." not in report
    editor.begin_write()
    editor._cross_link(("a", "b"))
    assert "cross_layer_jvp=1/1" in editor.finish_write(editor.session, [])
    assert "cross_layer_fallback=" not in editor._jvp_write_note()
