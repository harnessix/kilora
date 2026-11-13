# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import kilora_editor_chunk.plugin as cl
import torch
from safetensors.torch import load_file
from test_chunk_plugin import _record
from test_chunk_plugin import setup as chunk_setup

from kilora.compute import DROPOUT_DELTA, keep_probability, read_dropout
from kilora.recording import GenerationRecord


def test_central_keeps_drive_depth_merge_without_extra_forwards(chunk_setup, monkeypatch):
    session, path, target = chunk_setup
    editor = cl.ChunkLive(str(path))
    editor.check(session)
    editor.configure({"diff": "central", "path_angle": 0.0})
    initial = session.editor_snapshot()
    calls = []
    captured = []
    original_merge = editor._path_merge

    def merge(chunks, state, shares, keeps, keys):
        assert not shares
        original_merge(chunks, state, shares, keeps, keys)
        captured.append((keeps, editor.path_report, dict(chunks.node_coef)))

    monkeypatch.setattr(editor, "_path_merge", merge)
    hook = editor.targets[0]
    columns = editor.col_expert[hook].numel()
    base = torch.ones(3, columns)
    unperturbed = torch.full_like(base, 0.4)

    def measure(trial):
        calls.append(1)
        perturbation = float(trial.du[hook].sum()) * 0.001
        read_dropout(
            unperturbed + perturbation,
            base,
            columns // editor.experts,
            f"{hook}#0",
            1,
            set_key=session.mod_id,
        )
        return float(sum(((trial.du[t] - target[t]) ** 2).sum() for t in trial.du))

    record = GenerationRecord(images=1, measure=measure)
    editor.configure({"path_merge": False})
    editor.next_state(session.state, record)
    off = len(calls)
    session.editor_state.load_tensors(initial)
    calls.clear()
    editor.configure({"path_merge": True})
    editor.next_state(session.state, record)
    assert len(calls) == off and off > 0
    keeps, report, coefficients = captured[0]
    expected = keep_probability(torch.tensor(0.4), torch.tensor(1.0 + DROPOUT_DELTA))
    observed = torch.stack(keeps[(hook, 0)]).mean(0)
    torch.testing.assert_close(observed, expected.expand_as(observed), atol=1e-3, rtol=0)
    assert report[0] > 0 and report[1] > 0
    assert any(torch.count_nonzero(value) for value in coefficients.values())
    paths = session.editor_state.forest.branches["main"].paths
    assert any(p["predecessor"][0] == p["successor"][0] for p in paths)


def test_merge_off_matches_t607d(chunk_setup):
    session, path, target = chunk_setup
    expected = load_file(
        str(Path(__file__).parent / "data" / "t607d_merge_off_reference.safetensors")
    )
    current = cl.ChunkLive(str(path))
    current.check(session)
    current.configure({"path_merge": False})
    result = current.next_state(session.state, _record(session, target))
    actual_state = {
        f"editor/{key}": value for key, value in session.editor_state.state_tensors().items()
    }
    actual_result = {
        f"result/{hook}/{key}": value
        for hook, parts in result.items()
        for key, value in parts.items()
    }
    assert expected.keys() == (actual_state | actual_result).keys()
    for key, value in (actual_state | actual_result).items():
        torch.testing.assert_close(value, expected[key], rtol=1e-5, atol=1e-8, equal_nan=True)
