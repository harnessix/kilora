# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.sequence_model import evaluate_model, train_directions


def direction(number, outcome="keep"):
    return {"type": "direction", "basis_id": str(number), "coefficients": [1.0], "outcome": outcome}


def test_sequence_model_learns_held_out_pattern_and_stays_off(model, x):
    training = [
        [direction((start + i) % 12) for i in range(7)] for start in range(12) for _ in range(3)
    ]
    held_out = [[direction((start + i) % 12) for i in range(7)] for start in range(12)]
    with torch.no_grad():
        before = model(x).clone()
    predictor = train_directions(training, epochs=35, seed=7)
    result = evaluate_model(predictor, training, held_out)
    print(f"held-out synthetic result: {result}")
    assert result.accepted, result
    assert result.model_hits / result.targets >= result.repeat_hits / result.targets + 0.05
    assert result.model_hits / result.targets >= result.frequency_hits / result.targets + 0.05
    with torch.no_grad():
        assert torch.equal(before, model(x))
    assert not predictor.model.training
    assert len(predictor.model.encoder.layers) == 2
    assert predictor.model.items.embedding_dim == 64


def test_unknown_directions_are_not_recommended():
    training = [[direction(i) for i in range(8)] for _ in range(2)]
    predictor = train_directions(training, epochs=1)
    unknown = ("unknown", (1.0,))
    assert unknown not in predictor.predict([unknown])
    result = evaluate_model(predictor, training, [[direction(0), direction(99)]])
    assert result.targets == 1 and result.model_hits == 0


def test_attention_does_not_see_future_items():
    from kilora.sequence_model import SequenceModel

    torch.manual_seed(2)
    model = SequenceModel(8).eval()
    with torch.no_grad():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 7, 8]]))
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
