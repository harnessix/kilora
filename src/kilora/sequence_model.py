# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kilora.sequence_eval import Direction, Session, _kept_directions, evaluate_directions


class SequenceModel(nn.Module):
    def __init__(self, count: int, max_length: int = 32):
        super().__init__()
        self.max_length = max_length
        self.items = nn.Embedding(count + 2, 64, padding_idx=0)
        self.positions = nn.Embedding(max_length, 64)
        layer = nn.TransformerEncoderLayer(
            64, 2, dim_feedforward=256, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, count)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length < 1 or length > self.max_length:
            raise ValueError("sequence length outside model limits")
        positions = torch.arange(length, device=ids.device)
        x = self.items(ids) + self.positions(positions)
        mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        return self.output(self.norm(self.encoder(x, mask=mask, src_key_padding_mask=ids == 0)))


@dataclass
class DirectionPredictor:
    model: SequenceModel
    directions: tuple[Direction, ...]

    @torch.no_grad()
    def predict(self, prefix: list[Direction]) -> tuple[Direction, ...]:
        if not prefix:
            return ()
        lookup = {direction: i + 1 for i, direction in enumerate(self.directions)}
        ids = [lookup.get(d, len(self.directions) + 1) for d in prefix[-self.model.max_length :]]
        self.model.eval()
        logits = self.model(torch.tensor([ids], dtype=torch.long))[0, -1]
        indexes = logits.topk(min(5, len(self.directions))).indices.tolist()
        return tuple(self.directions[i] for i in indexes)


def train_directions(
    sessions: list[Session], *, epochs: int = 30, seed: int = 0, max_length: int = 32
) -> DirectionPredictor:
    sequences = [list(_kept_directions(session)) for session in sessions]
    directions = tuple(sorted({item for seq in sequences for item in seq}))
    if not directions or not any(len(seq) > 1 for seq in sequences):
        raise ValueError("training requires at least one pair of kept directions")
    if epochs < 1 or max_length < 1:
        raise ValueError("epochs and max_length must be positive")
    lookup = {direction: i + 1 for i, direction in enumerate(directions)}
    examples = []
    for seq in sequences:
        ids = [lookup[item] for item in seq]
        for end in range(1, len(ids)):
            examples.append((ids[max(0, end - max_length) : end], ids[end] - 1))
    width = max(len(prefix) for prefix, _ in examples)
    x = torch.zeros(len(examples), width, dtype=torch.long)
    for row, (prefix, _) in enumerate(examples):
        x[row, : len(prefix)] = torch.tensor(prefix)
    lengths = torch.tensor([len(prefix) - 1 for prefix, _ in examples])
    targets = torch.tensor([target for _, target in examples])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = SequenceModel(len(directions), max_length)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(x)[torch.arange(len(examples)), lengths]
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
    model.eval()
    return DirectionPredictor(model, directions)


@dataclass(frozen=True)
class ModelEvaluation:
    targets: int
    model_hits: int
    repeat_hits: int
    frequency_hits: int

    @property
    def accepted(self) -> bool:
        return (
            self.targets > 0
            and (self.model_hits - max(self.repeat_hits, self.frequency_hits)) / self.targets
            >= 0.05
        )


def evaluate_model(
    predictor: DirectionPredictor, training: list[Session], held_out: list[Session]
) -> ModelEvaluation:
    baseline = evaluate_directions(training, held_out)
    hits = 0
    for session in held_out:
        prefix: list[Direction] = []
        for direction in _kept_directions(session):
            if prefix:
                hits += direction in predictor.predict(prefix)
            prefix.append(direction)
    return ModelEvaluation(baseline.targets, hits, baseline.repeat_hits, baseline.frequency_hits)
