# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


class LayerChain(nn.Module):
    def __init__(self, mounts, k, d=64, semantic_dim=None, dropout=0.0):
        super().__init__()
        self.mounts = tuple(mounts)
        if not self.mounts or len(set(self.mounts)) != len(self.mounts):
            raise ValueError("mounts must be nonempty and unique")
        if k < 1 or d < 1 or not 0 <= dropout <= 1:
            raise ValueError("invalid dimensions or dropout")
        if semantic_dim is not None and semantic_dim < 1:
            raise ValueError("invalid semantic dimension")
        self.k, self.d, self.semantic_dim, self.dropout = k, d, semantic_dim, dropout
        self.initial = nn.Parameter(torch.zeros(d))
        self.projections = nn.ModuleList(nn.Linear(k * k + 2 * k, d) for _ in mounts)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d)) for _ in mounts
        )
        self.semantic = nn.Linear(semantic_dim, d) if semantic_dim is not None else None

    def forward(self, rows):
        device, dtype = self.initial.device, self.initial.dtype
        h = self.initial.expand(len(rows), -1)
        for name, projection, block in zip(self.mounts, self.projections, self.blocks, strict=True):
            inputs = []
            for row in rows:
                item = row.get("mounts", {}).get(name)
                if item is None:
                    x = torch.zeros(self.k * self.k + 2 * self.k, device=device, dtype=dtype)
                else:
                    parts = [
                        torch.as_tensor(item[key], device=device, dtype=dtype).reshape(-1)
                        for key in ("core", "output_energy", "input_energy")
                    ]
                    if [p.numel() for p in parts] != [self.k * self.k, self.k, self.k]:
                        raise ValueError(f"invalid coefficients for {name}")
                    x = torch.cat(parts)
                inputs.append(x)
            x = torch.stack(inputs)
            if self.training and self.dropout:
                keep = torch.rand((len(rows), 1), device=device) >= self.dropout
                x = x * keep
            h = h + block(torch.cat((h, projection(x)), dim=-1))
        return F.normalize(h, dim=-1)

    def encode(self, rows):
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self(rows)
        finally:
            self.train(was_training)

    def encode_semantic(self, vectors):
        if self.semantic is None:
            raise ValueError("semantic head is disabled")
        return F.normalize(self.semantic(vectors), dim=-1)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "mounts": self.mounts,
            "k": self.k,
            "d": self.d,
            "semantic_dim": self.semantic_dim,
            "dropout": self.dropout,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(path / "model.safetensors"),
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        model = cls(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        model.load_state_dict(load_file(str(path / "model.safetensors")))
        return model


def supervised_infonce(embeddings, labels, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(labels) != len(embeddings):
        raise ValueError("labels and embeddings differ in length")
    n = len(labels)
    mask = torch.tensor(
        [[a == b for b in labels] for a in labels], dtype=torch.bool, device=embeddings.device
    )
    mask.fill_diagonal_(False)
    valid = mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0
    logits = embeddings @ embeddings.T / temperature
    logits = logits.masked_fill(torch.eye(n, device=logits.device, dtype=torch.bool), -torch.inf)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probs.masked_fill(~mask, 0).sum(dim=1)[valid] / mask.sum(dim=1)[valid]).mean()


def clip_loss(embeddings, semantic_embeddings, temperature=0.1):
    if temperature <= 0 or embeddings.shape != semantic_embeddings.shape:
        raise ValueError("invalid temperature or embedding shapes")
    logits = embeddings @ semantic_embeddings.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
