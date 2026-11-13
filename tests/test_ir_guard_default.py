# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora import editors
from kilora.ir.defaults import default_bindings, default_program
from kilora.program import interpret


@pytest.mark.parametrize(
    ("gap0", "gap1", "spread0", "spread1", "diffs"),
    [
        (1.0, 0.9, 1.0, 1.0, []),
        (1.0, 0.79, 1.0, 1.0, []),
        (1.0, 0.79, 1.0, 1.0, [1.0]),
        (1.0, 1.0, 1.0, 0.94, []),
        (1.0, 1.0, 1.0, 1.06, []),
        (1.0, 1.0, 0.0, float("nan"), []),
        (float("nan"), 1.0, 1.0, 1.0, []),
        (1.0, float("nan"), 1.0, 1.0, []),
        (1.0, 1.0, 1.0, float("nan"), []),
        (1.0, 0.8, 1.0, 0.95, []),
    ],
)
def test_guard_matches_native(gap0, gap1, spread0, spread1, diffs):
    before = {"gap": gap0, "spread": spread0}
    after = {"gap": gap1, "spread": spread1}
    session = SimpleNamespace(state=None, guard=True, guard_gap_diffs=diffs.copy())
    rule = SimpleNamespace(guard_trial=lambda state, new: object())
    record = SimpleNamespace(
        guard_score=lambda trial, before=before, after=after: before if trial is None else after
    )
    expected, _ = editors.guard_check(rule, session, record, None)
    values = {
        "gap_before": gap0,
        "gap_after": gap1,
        "spread_before": spread0,
        "spread_after": spread1,
        "noise": session.guard_last["gap_noise"],
        "gap_drop": editors.GUARD_GAP_DROP,
        "gap_noise_k": editors.GUARD_GAP_NOISE_K,
        "spread_low": editors.GUARD_SPREAD[0],
        "spread_high": editors.GUARD_SPREAD[1],
    }
    program, bindings = default_program("guard"), default_bindings("guard")
    inputs = {name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()}
    assert bool(interpret(program, inputs, bindings)["refused"]) is expected


def test_seeded_guard_scores_and_paired_noise():
    generator = torch.Generator().manual_seed(119)
    program, bindings = default_program("guard"), default_bindings("guard")
    for _ in range(100):
        scores = torch.randn(6, generator=generator, dtype=torch.float64).tolist()
        before = {"gap": scores[0], "gap_alt": scores[1], "spread": abs(scores[2])}
        after = {"gap": scores[3], "gap_alt": scores[4], "spread": abs(scores[5])}
        session = SimpleNamespace(state=None, guard=True, guard_gap_diffs=[])
        rule = SimpleNamespace(guard_trial=lambda state, new: object())
        record = SimpleNamespace(
            guard_score=lambda trial, before=before, after=after: before if trial is None else after
        )
        expected, _ = editors.guard_check(rule, session, record, None)
        values = {
            "gap_before": session.guard_last["gap0"],
            "gap_after": session.guard_last["gap1"],
            "spread_before": before["spread"],
            "spread_after": after["spread"],
            "noise": session.guard_last["gap_noise"],
            "gap_drop": editors.GUARD_GAP_DROP,
            "gap_noise_k": editors.GUARD_GAP_NOISE_K,
            "spread_low": editors.GUARD_SPREAD[0],
            "spread_high": editors.GUARD_SPREAD[1],
        }
        inputs = {name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()}
        assert bool(interpret(program, inputs, bindings)["refused"]) is expected
