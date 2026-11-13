# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch
from test_axis_owner import make_set

from kilora.bake import TargetPlan
from kilora.call_values import current_axis_alphas
from kilora.comfyui.patching import _key_mask_by_step
from kilora.region_attention import CallState, RegionAttention, comfy_override, kind_patches


def attention(q, k, v, heads, mask=None, **kwargs):
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask[:, 0]
    return logits.softmax(-1) @ v


def test_owner_attention_matches_manual_and_keeps_unconditional():
    owners = {
        "ip": torch.tensor([[[1.0, 0.0, 0.0, 0.5]]]),
        "creative": torch.tensor([[[0.0, 1.0, 0.0, 0.5]]]),
        "other": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
    }
    state = CallState(latent=(1, 4), owners=owners, cond_rows=torch.tensor([True, False]))
    override = comfy_override((), RegionAttention(), state)
    q = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]).repeat(2, 1, 1)
    kind_patches(state)["attn1_patch"](q, q, q)
    actual = override(attention, q, q, q, 1)
    bias = torch.tensor(
        [
            [0, 0, -torch.inf, -torch.inf],
            [-torch.inf, 0, -torch.inf, -torch.inf],
            [-torch.inf, 0, 0, -torch.inf],
            [0, 0, 0, 0],
        ],
        dtype=q.dtype,
    )
    expected = torch.stack(
        [((q[0] @ q[0].T) + bias).softmax(-1) @ q[0], (q[1] @ q[1].T).softmax(-1) @ q[1]]
    )
    assert torch.allclose(actual, expected)
    kind_patches(state)["attn2_patch"](q, q, q)
    assert torch.equal(override(attention, q, q, q, 1), attention(q, q, q, 1))


def test_previous_step_gating_and_auto_precedence(monkeypatch):
    from kilora.call_values import axis_alphas

    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", "single")
    patcher = types.SimpleNamespace(model_options={})
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    plan = TargetPlan(sets=sets, name="layer")
    seen = []

    class Auto:
        def call(self, args, invoke):
            with axis_alphas({"ip": 0.25, "creative": 0.75}):
                return invoke(args)

    _key_mask_by_step(patcher, auto_alpha=Auto())
    wrapper = patcher.model_options["model_function_wrapper"]
    y = torch.tensor([[[1.0, 0.0]] * 4])

    def model(x, timestep, **c):
        seen.append((dict(current_axis_alphas()), c.get("transformer_options", {})))
        return plan.run(y, y)

    def call(step):
        return wrapper(
            model,
            {
                "input": torch.zeros(1, 2, 2, 2),
                "timestep": torch.tensor([step]),
                "c": {"transformer_options": {"cond_or_uncond": [0]}},
            },
        )

    call(2)
    assert seen[0][0] == {"ip": 0.25, "creative": 0.75}
    assert "optimized_attention_override" not in seen[0][1]
    call(1)
    assert torch.all(seen[1][0]["ip"] == 1)
    assert torch.all(seen[1][0]["creative"] == 0)
    assert seen[1][0]["functional"] == 1
    assert "optimized_attention_override" in seen[1][1]
    call(3)
    assert seen[2][0] == {"ip": 0.25, "creative": 0.75}


@pytest.mark.parametrize("mode", ["", "layer"])
def test_other_modes_do_not_install_single_owner(monkeypatch, mode):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", mode)
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher)
    x = torch.randn(1, 2, 2, 2)
    wrapper = patcher.model_options["model_function_wrapper"]
    out = wrapper(lambda x, t, **c: x * 2, {"input": x, "timestep": torch.tensor([1.0]), "c": {}})
    assert torch.equal(out, x * 2)


@pytest.mark.parametrize("depth_path", ["loop", "fast"])
def test_gating_suppresses_other_axis(monkeypatch, depth_path):
    from kilora.call_values import axis_alphas
    from kilora.region import latent_size

    monkeypatch.setenv("KILORA_DEPTH_PATH", depth_path)
    y = torch.ones(1, 4, 2)
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    plan = TargetPlan(sets=sets, name="layer")
    ip = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    creative = 1 - ip
    with latent_size((2, 2)), axis_alphas({"ip": ip, "creative": creative}):
        both = plan.run(y, y)
    with latent_size((2, 2)), axis_alphas({"ip": ip, "creative": torch.zeros_like(ip)}):
        only_ip = plan.run(y, y)
    with latent_size((2, 2)), axis_alphas({"ip": torch.zeros_like(ip), "creative": creative}):
        only_creative = plan.run(y, y)
    assert torch.equal(ip + creative, torch.ones_like(ip))
    assert torch.allclose(both, only_ip + only_creative - y)


def test_functional_scores_do_not_take_ownership():
    from kilora.axis_owner import AxisOwner
    from kilora.region import latent_size

    owner = AxisOwner((2, 2), axes=("ip", "creative"))
    sets = [make_set("ip", [1, 0]), make_set("functional", [1, 0])]
    with latent_size((2, 2)):
        owner.record("layer", torch.tensor([[[1.0, 0.0]] * 4]), sets)
    masks = owner.finish_step()
    assert "functional" not in masks
    assert torch.all(masks["ip"] + masks["creative"] == 1)
