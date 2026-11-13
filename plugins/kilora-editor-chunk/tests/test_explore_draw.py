# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from test_chunk_plugin import _record, setup

from kilora.compute import read_dropout
from kilora.editors import commit_write, editor_for
from kilora.recording import GenerationRecord


def _draw_record(session, editor, target):
    generator = torch.Generator().manual_seed(43)
    activations = {
        h: torch.randn(3, router.shape[1], generator=generator)
        for h, (router, _) in editor.routes.items()
    }

    def measure(trial):
        value = 0.0
        for h, x in activations.items():
            trial.delta(h, x)
            columns = editor.col_expert[h].numel()
            read_dropout(
                torch.ones(3, columns),
                torch.ones(3, columns),
                columns // editor.experts,
                f"{h}#0",
                1,
                set_key=session.mod_id,
            )
            value += float(((trial.du[h] - target[h]) ** 2).sum())
        return value

    return GenerationRecord(
        images=1, measure=measure, extra={"entry_temperature": 1.0, "seed": 42, "depth": 2}
    )


def test_explore_draw_span_and_report(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    session.mod.settings.depth = 2
    session.mod.sets = [editor.live_set]
    editor.configure({"explore": 1.0, "merge_residual": "keep", "path_angle": 0.2})
    editor.manifest["writer"]["diff"] = "one-sided"
    report = commit_write(session, editor, _draw_record(session, editor, target))
    assert "(explore draw: no residuals kept)" in report
    chunks = session.editor_state.chunks
    assert not chunks.residual
    expanded = chunks.expanded_u(editor.editor)
    assert sum(float(value.norm()) for value in expanded.values()) > 0
    for hook, values in expanded.items():
        basis = (
            torch.cat([editor.editor.per_hook(c.q)[hook] for c in editor.editor.chunks], dim=0)
            .reshape(-1, values[0].numel())
            .T.double()
        )
        for value in values:
            vector = value.reshape(-1, 1).double()
            remainder = vector - basis @ torch.linalg.lstsq(basis, vector).solution
            assert float(remainder.norm()) < 1e-6 * max(float(vector.norm()), 1e-20)


def test_next_state_explore_forks_without_changing_parent(setup):
    session, path, target = setup
    editor = editor_for(path)
    editor.check(session)
    editor.configure({"explore": 1.0, "merge_residual": "keep"})
    state = session.editor_state
    for history in state.chunks.hist.values():
        history.zero_()
    parent = {key: value.clone() for key, value in state.state_tensors().items()}
    record = _record(session, target)
    record.extra = {"entry_temperature": 1.0, "seed": 42, "depth": 2}
    result = editor.next_state(session.state, record)
    branch = state.forest.current
    assert branch != "main"
    assert result is not None
    assert any(torch.count_nonzero(v) for v in state.chunks.coef.values())
    state.checkout("main")
    after = state.state_tensors()
    for key, value in parent.items():
        if not key.startswith("forest/"):
            assert key in after and torch.equal(value, after[key]), key
    assert all(torch.count_nonzero(v) == 0 for v in state.chunks.coef.values())
    state.checkout(branch)
    assert any(torch.count_nonzero(v) for v in state.chunks.coef.values())
