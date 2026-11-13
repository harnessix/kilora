# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora import KiLoRAConfig, attach
from kilora.testing import TinyLM, TinyLMBackend, tiny_texts

TARGETS = ["layers.*.attention.to_q", "layers.*.attention.to_v", "layers.*.feed_forward.w1"]


def test_keyed_experts_train_a_language_model():
    model = TinyLM(seed=0)
    model.requires_grad_(False)
    backend = TinyLMBackend(model)
    texts = [t for ts in tiny_texts(4, 6, seed=2).values() for t in ts]
    adapter = attach(
        model, KiLoRAConfig(target_modules=TARGETS, num_experts=4, router_dim=8, train_depth=(1, 3))
    )
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    model.train()
    losses = []
    for step in range(60):
        adapter.start_step(step)
        pred, target = backend.forward(texts[:8], backend.sample_events(8, gen), gen)
        loss = adapter.wrap_loss(backend.loss(pred, target))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert sum(losses[-10:]) < sum(losses[:10])
    adapter.remove()
