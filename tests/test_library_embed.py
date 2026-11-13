# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.library.embed import LayerChain, clip_loss, supervised_infonce


def _row(vector):
    return {
        "mounts": {
            "layer": {
                "core": vector[:4].reshape(2, 2),
                "output_energy": vector[4:6],
                "input_energy": vector[6:8],
            }
        }
    }


def test_training_generalizes_and_shuffled_control():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(11)
    centers = torch.randn(3, 8, generator=generator) * 2
    train, held, labels = [], [], []
    for group in range(3):
        for index in range(8):
            row = _row(centers[group] + 0.35 * torch.randn(8, generator=generator))
            (train if index < 6 else held).append(row)
            if index < 6:
                labels.append(group)

    def fit(training_labels, held_labels):
        torch.manual_seed(7)
        model = LayerChain(["layer"], 2, d=16, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        for _ in range(200):
            optimizer.zero_grad()
            loss = supervised_infonce(model(train), training_labels)
            loss.backward()
            optimizer.step()
        vectors = model.encode(held)
        scores = vectors @ vectors.T
        scores.fill_diagonal_(-2)
        nearest = scores.argmax(dim=1).tolist()
        return sum(held_labels[i] == held_labels[j] for i, j in enumerate(nearest)) / len(held)

    good = fit(labels, [0, 0, 1, 1, 2, 2])
    shuffled = [labels[i] for i in torch.randperm(len(labels), generator=generator)]
    control = fit(shuffled, [0, 1, 2, 0, 1, 2])
    assert good >= 0.8
    assert control <= 0.6


def test_dropout_missing_and_roundtrip(tmp_path):
    torch.manual_seed(4)
    model = LayerChain(["layer", "missing"], 2, d=8, semantic_dim=5, dropout=1)
    rows = [_row(torch.randn(8)), {"mounts": {}}]
    assert torch.isfinite(model(rows)).all()
    semantic = model.encode_semantic(torch.randn(2, 5))
    assert torch.isfinite(clip_loss(model(rows), semantic))
    before = model.encode(rows)
    model.save(tmp_path)
    restored = LayerChain.load(tmp_path)
    assert torch.equal(before, restored.encode(rows))
    assert model.training
