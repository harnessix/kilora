# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest
import torch

from kilora.axis_owner import AxisOwner
from kilora.region_attention import (
    CallState,
    RegionAttention,
    comfy_override,
    kind_patches,
    owner_bias,
    soft_owner_bias,
)


def maps():
    return {
        "ip": torch.tensor([[[1.0, 0.7, 0.0, 0.0]]]),
        "other": torch.tensor([[[0.0, 0.3, 1.0, 0.0]]]),
        "creative": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
    }


def test_soft_bias_is_finite_and_excludes_same_far_and_unconditional():
    bias = soft_owner_bias(
        maps(), 4, (1, 4), 2, torch.tensor([True, False]), torch.float32, "cpu", 2.0
    )
    assert torch.isfinite(bias).all()
    assert torch.equal(bias[1], torch.zeros_like(bias[1]))
    assert bias[0, 0, 0, 0] == 0
    assert bias[0, 0, 0, 3] == 0
    assert bias[0, 0, 1, 2] == pytest.approx(-1.4)
    assert bias[0, 0, 0, 1] == 0
    assert bias[0, 0, 2, 1] == pytest.approx(-1.4)
    assert bias[0, 0, 2, 0] == -2


def test_large_bias_matches_hard_probabilities_for_owned_keys():
    owners = {
        "ip": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "other": torch.tensor([[[0.0, 1.0, 0.0]]]),
        "creative": torch.tensor([[[0.0, 0.0, 1.0]]]),
    }
    hard = owner_bias(owners, 3, (1, 3), 1, None, torch.float32, "cpu")
    soft = soft_owner_bias(owners, 3, (1, 3), 1, None, torch.float32, "cpu", 1e4)
    logits = torch.tensor([[[[0.2, 1.2, 0.7], [1.0, 0.3, 0.4], [0.3, 0.7, 0.1]]]])
    assert torch.allclose((logits + hard).softmax(-1), (logits + soft).softmax(-1))


def test_zero_bias_preserves_attention_and_cross_attention(monkeypatch):
    monkeypatch.setenv("KILORA_AXIS_BIAS", "0")
    state = CallState(latent=(1, 4), owners=maps(), owner_mode="soft")
    override = comfy_override((), RegionAttention(), state)
    q = torch.randn(1, 4, 2)
    calls = []

    def attention(q, k, v, heads, **kwargs):
        calls.append(kwargs)
        return q

    kind_patches(state)["attn1_patch"](q, q, q)
    assert torch.equal(override(attention, q, q, q, 1), q)
    kind_patches(state)["attn2_patch"](q, q, q)
    assert torch.equal(override(attention, q, q, q, 1), q)
    assert calls == [{}, {}]


def test_inhibition_sharpens_overlap_and_preserves_sum(monkeypatch, caplog):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", "soft")
    owner = AxisOwner((3, 9))
    left = torch.zeros(1, 3, 9)
    left[:, :, :4] = 1
    owner.masks = {"ip": left, "creative": 1 - left}
    score = torch.full((1, 3, 9), 0.5)
    owner.scores = {("ip", "a", 0): score, ("creative", "b", 0): score}
    monkeypatch.setenv("KILORA_AXIS_INHIBIT", "0")
    no_inhibition = owner.finish_step()
    owner.ema = {}
    owner.masks = {"ip": left, "creative": 1 - left}
    owner.scores = {("ip", "a", 0): score, ("creative", "b", 0): score}
    monkeypatch.setenv("KILORA_AXIS_INHIBIT", "0.3")
    with caplog.at_level(logging.INFO, logger="kilora.axis_owner"):
        inhibited = owner.finish_step()

    def overlap(m):
        return ((m["ip"] > 0.25) & (m["creative"] > 0.25)).float().mean()

    assert overlap(inhibited) < overlap(no_inhibition)
    assert torch.allclose(inhibited["ip"] + inhibited["creative"], torch.ones_like(score))
    assert "overlap=" in caplog.text and "mean bias=" in caplog.text


def test_zero_inhibition_matches_single(monkeypatch):
    def run(mode):
        monkeypatch.setenv("KILORA_AXIS_ZBUFFER", mode)
        owner = AxisOwner((2, 3))
        owner.masks = {"ip": torch.ones(1, 2, 3), "creative": torch.zeros(1, 2, 3)}
        owner.scores = {
            ("ip", "a", 0): torch.full((1, 2, 3), 0.51),
            ("creative", "b", 0): torch.full((1, 2, 3), 0.49),
        }
        return owner.finish_step()

    monkeypatch.setenv("KILORA_AXIS_INHIBIT", "0")
    single, soft = run("single"), run("soft")
    assert all(torch.equal(single[name], soft[name]) for name in single)


def test_soft_single_pass_uses_previous_masks_and_installs_bias(monkeypatch):
    import types

    from test_axis_owner import make_set

    from kilora.bake import TargetPlan
    from kilora.call_values import current_axis_alphas
    from kilora.comfyui.patching import _key_mask_by_step

    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", "soft")
    monkeypatch.setenv("KILORA_AXIS_INHIBIT", "0")
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher)
    plan = TargetPlan(sets=[make_set("ip", [1, 0]), make_set("creative", [0, 1])])
    seen = []
    y = torch.tensor([[[1.0, 0.0]] * 4])

    def model(x, timestep, **c):
        seen.append((dict(current_axis_alphas()), c["transformer_options"]))
        return plan.run(y, y)

    wrapper = patcher.model_options["model_function_wrapper"]
    for step in (2, 1):
        wrapper(
            model,
            {
                "input": torch.zeros(1, 2, 2, 2),
                "timestep": torch.tensor([step]),
                "c": {"transformer_options": {"cond_or_uncond": [0]}},
            },
        )
    assert seen[0][0] == {}
    assert torch.all(seen[1][0]["ip"] == 1)
    assert torch.all(seen[1][0]["creative"] == 0)
    assert seen[1][0]["functional"] == 1
    options = seen[1][1]
    assert "optimized_attention_override" in options
    q = torch.ones(1, 4, 1)
    options["patches"]["attn1_patch"][-1](q, q, q)
    captured = {}

    def attention(q, k, v, heads, **kwargs):
        captured.update(kwargs)
        return q

    options["optimized_attention_override"](attention, q, q, q, 1)
    assert torch.equal(captured["mask"], torch.zeros(1, 1, 4, 4))
