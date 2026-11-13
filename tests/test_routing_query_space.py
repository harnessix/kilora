# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from torch import nn

from kilora import compute
from kilora.routing_features import (
    call_features,
    current_score_features,
    layer_features,
    prompt_projection,
    routing_features,
)


def _layer(width_in, width_out):
    router = torch.eye(width_out)[:2]
    return compute.LayerSet(
        router=router,
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(width_in)[:2],
        up=torch.eye(width_out)[:2],
        per_expert=1,
        tau=1.0,
        form="input" if width_in != width_out else "state",
    )


def _observe(monkeypatch):
    applied, seen = [], []
    real_cross, real_gate = compute._cross_logits, compute._gates_from_logits

    def cross(*args):
        result = real_cross(*args)
        applied.append(not torch.equal(result, args[0]))
        return result

    def gate(*args, **kwargs):
        result = real_gate(*args, **kwargs)
        seen.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(compute, "_cross_logits", cross)
    monkeypatch.setattr(compute, "_gates_from_logits", gate)
    return applied, seen


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(("width_in", "width_out"), [(8, 16), (16, 8)])
def test_non_square_layer_applies_prompt_cross(width_in, width_out, fast, monkeypatch, caplog):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    monkeypatch.setattr(compute, "_cross_missing_noted", set())
    applied, seen = _observe(monkeypatch)
    layer = _layer(width_in, width_out)
    crossed = dataclasses.replace(layer, cross_weights=(0, 0.4, 0))
    x = torch.ones(1, 4, width_in)
    y0 = torch.ones(1, 4, width_out)
    prompt = torch.zeros(1, 1, width_out)
    prompt[..., 0] = 1
    with torch.no_grad(), call_features(prompt, None), routing_features():
        base = compute.run_layer(x, y0, [layer], 1)
        baseline = seen[-1][..., 0].mean()
        features = []
        real = compute._gates_from_logits

        def record(*args, **kwargs):
            features.append(current_score_features())
            return real(*args, **kwargs)

        monkeypatch.setattr(compute, "_gates_from_logits", record)
        result = compute.run_layer(x, y0, [crossed], 1)
        changed = seen[-1][..., 0].mean()
        replay = compute.run_layer(x, y0, [crossed], 1)
    f = features[0][0]
    assert f.token.shape == f.region.shape == f.prompt.shape == (1, 4, width_out)
    assert any(applied)
    assert "is unavailable" not in caplog.text
    assert changed > baseline
    assert not torch.equal(result, base)
    assert torch.equal(result, replay)


@pytest.mark.parametrize(("width_in", "width_out"), [(8, 16), (16, 8)])
def test_features_follow_layer_output(width_in, width_out):
    layer = [_layer(width_in, width_out)]
    x = torch.randn(2, 5, width_in)
    y0 = torch.randn(2, 5, width_out)
    region = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]])
    with call_features(None, None), layer_features(x, y0, layer, [region]):
        f = current_score_features()[0]
    assert torch.equal(f.token, y0)
    torch.testing.assert_close(f.region[:, 0], torch.stack((y0[0, :2].mean(0), y0[1, 2:4].mean(0))))


def test_joint_text_rows_come_from_layer_output():
    from kilora.family_spec import TokenLayout
    from kilora.region import token_layouts

    x = torch.randn(1, 3, 8)
    y0 = torch.randn(1, 3, 16)
    with (
        token_layouts(lambda _: TokenLayout("image2d", text="before"), 2),
        call_features(torch.ones(1, 1, 5), None),
        layer_features(x, y0, [_layer(8, 16)], None, "target"),
    ):
        torch.testing.assert_close(current_score_features()[0].prompt[0, 2], y0[0, :2].mean(0))
    with (
        token_layouts(lambda _: TokenLayout("image2d", text="end"), 2),
        call_features(torch.ones(1, 1, 5), None),
        layer_features(x, y0, [_layer(8, 16)], None, "target"),
    ):
        torch.testing.assert_close(current_score_features()[0].prompt[0, 0], y0[0, -2:].mean(0))


class AnimaLike(nn.Module):
    def __init__(self, out=8, hidden=16, text=3):
        super().__init__()
        block = nn.Module()
        attn = nn.Module()
        attn.add_module("to_k", nn.Linear(text, out))
        block.add_module("attn2", attn)
        ff = nn.Module()
        ff.add_module(
            "net", nn.ModuleList([nn.Linear(out, hidden), nn.GELU(), nn.Linear(hidden, out)])
        )
        block.add_module("ff", ff)
        self.add_module("transformer_blocks", nn.ModuleList([block]))


@pytest.mark.parametrize("fast", [False, True])
def test_anima_like_feed_forward_gets_prompt_feature(fast, monkeypatch, caplog):
    from kilora.family_spec import TokenLayout
    from kilora.region import token_layouts

    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    monkeypatch.setattr(compute, "_cross_missing_noted", set())
    out, hidden, text = 8, 16, 3
    model = AnimaLike(out, hidden, text)
    key = model.transformer_blocks[0].attn2.to_k
    with torch.no_grad():
        key.weight.zero_()
        key.weight[0, 0] = 1.0
        key.bias.zero_()
    projection = prompt_projection(model, "anima")
    prompt = torch.zeros(1, 2, text)
    prompt[..., 0] = 1
    x = torch.ones(1, 4, hidden)
    y0 = torch.ones(1, 4, out)
    target = "transformer_blocks.0.ff.net.2"
    feature = projection(target, prompt, y0)
    assert feature.shape == (1, out)
    torch.testing.assert_close(feature, key(prompt.mean(1)))
    assert projection(target, prompt, x) is None
    assert (
        projection("transformer_blocks.0.ff.net.0.proj", prompt, torch.ones(1, 4, hidden)) is None
    )

    applied, seen = _observe(monkeypatch)
    layer = _layer(hidden, out)
    crossed = dataclasses.replace(layer, cross_weights=(0, 0.4, 0))
    with (
        torch.no_grad(),
        token_layouts(lambda _: TokenLayout("video3d")),
        call_features(prompt, None, projection),
        routing_features(),
    ):
        base = compute.run_layer(x, y0, [layer], 1, target=target)
        baseline = seen[-1][..., 0].mean()
        result = compute.run_layer(x, y0, [crossed], 1, target=target)
        changed = seen[-1][..., 0].mean()
    assert any(applied)
    assert "is unavailable" not in caplog.text
    assert changed > baseline
    assert not torch.equal(result, base)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(("width_in", "width_out"), [(8, 16), (16, 8)])
def test_zero_weights_are_bit_identical(width_in, width_out, fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    layer = _layer(width_in, width_out)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, width_in, generator=gen)
    y0 = torch.randn(1, 4, width_out, generator=gen)
    prompt = torch.randn(1, 2, width_out, generator=gen)
    with torch.no_grad(), call_features(prompt, torch.randn(1, 4, generator=gen)):
        base = compute.run_layer(x, y0, [layer], 1)
        with routing_features():
            featured = compute.run_layer(x, y0, [layer], 1)
    assert torch.equal(base, featured)
