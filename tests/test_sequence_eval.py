# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.sequence_eval import DirectionEvaluation, evaluate_directions


def direction(name, outcome="keep", coefficients=None):
    return {
        "type": "direction",
        "basis_id": name,
        "coefficients": [1.0] if coefficients is None else coefficients,
        "outcome": outcome,
    }


def test_baselines_on_planted_sessions():
    training = [[direction(name) for name in "aaaaabbbbcccddef"]]
    held_out = [
        [direction("a"), direction("a"), direction("f"), direction("f")],
        [direction("b"), direction("b"), direction("c")],
    ]
    result = evaluate_directions(training, held_out)
    assert result == DirectionEvaluation(targets=5, repeat_hits=3, frequency_hits=3)
    assert result.repeat_hit_rate_at_5 == 0.6
    assert result.frequency_hit_rate_at_5 == 0.6


def test_discard_prompt_edits_session_boundaries_and_unseen_directions():
    prompt = {"type": "prompt_edit", "edit": "change", "outcome": "keep"}
    training = [[direction("a"), direction("b", "discard"), prompt]]
    held_out = [
        [direction("a"), direction("b", "discard"), prompt, direction("a")],
        [direction("a"), direction("a", coefficients=[2.0])],
        [direction("a")],
    ]
    result = evaluate_directions(training, held_out)
    assert result == DirectionEvaluation(targets=2, repeat_hits=1, frequency_hits=1)


def test_frequency_ties_have_stable_top_five_and_do_not_fit_on_test():
    training = [[direction(name) for name in "fedcba"]]
    held_out = [[direction("f"), direction("f"), direction("a")]]
    result = evaluate_directions(training, held_out)
    assert result == DirectionEvaluation(targets=2, repeat_hits=1, frequency_hits=1)


def test_empty_targets_are_unmeasured():
    result = evaluate_directions([], [[direction("a")], []])
    assert result.targets == 0
    assert result.repeat_hit_rate_at_5 is None
    assert result.frequency_hit_rate_at_5 is None


def test_offline_evaluation_does_not_change_model_output(model, x):
    with torch.no_grad():
        before = model(x).clone()
        evaluate_directions([[direction("a")]], [[direction("a"), direction("a")]])
        after = model(x)
    assert torch.equal(before, after)
