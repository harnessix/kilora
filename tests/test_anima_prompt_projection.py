# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from torch import nn

from kilora import compute
from kilora.routing_features import call_features, prompt_projection, routing_features


class Tree(nn.Module):
    def __init__(self, naming, with_key=True):
        super().__init__()
        block = nn.Module()
        attn = nn.Module()
        if with_key:
            attn.add_module("k_proj" if naming == "comfy" else "to_k", nn.Linear(3, 2))
        block.add_module("cross_attn" if naming == "comfy" else "attn2", attn)
        self.add_module(
            "blocks" if naming == "comfy" else "transformer_blocks", nn.ModuleList([block])
        )


@pytest.mark.parametrize("naming", ["comfy", "diffusers"])
def test_projection_lookup(naming):
    model = Tree(naming)
    module = model.get_submodule(
        "blocks.0.cross_attn.k_proj" if naming == "comfy" else "transformer_blocks.0.attn2.to_k"
    )
    projection = prompt_projection(model, "anima")
    prompt, x = torch.randn(1, 4, 3), torch.randn(1, 5, 2)
    for target in ("transformer_blocks.0.attn1.to_q", "blocks.0.self_attn.q_proj"):
        torch.testing.assert_close(projection(target, prompt, x), module(prompt.mean(1)))
    assert projection("transformer_blocks.1.attn1.to_q", prompt, x) is None
    assert (
        prompt_projection(Tree(naming, with_key=False), "anima")(
            "transformer_blocks.0.attn1.to_q", prompt, x
        )
        is None
    )
    assert prompt_projection(model, "flux") is None


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
@pytest.mark.parametrize("fast", [False, True])
def test_projection_applies_cross_and_missing_note(device, fast, monkeypatch, caplog):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    monkeypatch.setattr(compute, "_cross_missing_noted", set())
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Tree("diffusers")
    with torch.no_grad():
        model.transformer_blocks[0].attn2.to_k.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        model.transformer_blocks[0].attn2.to_k.bias.zero_()
    identity = torch.eye(2, device=device, dtype=dtype)
    layer = compute.LayerSet(
        router=identity,
        keys_in=identity.unsqueeze(1),
        down=identity,
        up=identity,
        per_expert=1,
        tau=1.0,
    )
    crossed = dataclasses.replace(layer, cross_weights=(0, 0.4, 0))
    x = torch.ones(1, 4, 2, device=device, dtype=dtype)
    prompt = torch.tensor([[[1.0, 0.0, 0.0]]], device=device, dtype=dtype)
    target = "transformer_blocks.0.attn1.to_q"
    applied = []
    real = compute._cross_logits

    def observe(*args):
        result = real(*args)
        applied.append(not torch.equal(result, args[0]))
        return result

    monkeypatch.setattr(compute, "_cross_logits", observe)
    with torch.no_grad(), routing_features():
        with call_features(prompt, None):
            base = compute.run_layer(x, x, [layer], 1, target=target)
            first = compute.run_layer(x, x, [crossed], 1, target=target)
            second = compute.run_layer(x, x, [crossed], 1, target=target)
        assert torch.equal(base, first) and torch.equal(first, second)
        assert not any(applied)
        applied.clear()
        with call_features(prompt, None, prompt_projection(model, "anima")):
            result = compute.run_layer(x, x, [crossed], 1, target=target)
        assert any(applied)
        assert not torch.equal(result, base)
    assert caplog.text.count("prompt projection is missing (needed for Anima)") == 1
