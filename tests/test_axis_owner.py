# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import torch

from kilora.axis_owner import AxisOwner, owner_masks, recording
from kilora.bake import TargetPlan
from kilora.comfyui.layer_alpha import install_layer_alpha
from kilora.compute import LayerSet
from kilora.family_spec import TokenLayout
from kilora.region import latent_size, token_layouts


def make_set(axis, key):
    return LayerSet(
        router=torch.eye(2),
        keys_in=torch.tensor(key, dtype=torch.float32).reshape(1, 1, 2),
        down=torch.ones(1, 2),
        up=torch.ones(1, 2),
        form="input",
        axis=axis,
        gate_mode="none",
    )


def collect(scale=1):
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    for s in sets:
        s.keys_in *= scale
    y = torch.zeros(1, 4, 8, 2)
    y[:, :, :4, 0] = 1
    y[:, :, 4:, 1] = 1
    owner = AxisOwner((4, 8))
    with latent_size((4, 8)), token_layouts(lambda _: TokenLayout("image2d", (1, 1))):
        owner.record("layer", y, sets)
    return owner.finish_step()


def test_spatial_owner_feathers_and_sums():
    masks = collect()
    assert torch.allclose(masks["ip"][:, :, :3], torch.ones(1, 4, 3))
    assert torch.allclose(masks["creative"][:, :, 5:], torch.ones(1, 4, 3))
    assert torch.all((masks["ip"][:, :, 3:5] > 0) & (masks["ip"][:, :, 3:5] < 1))
    assert torch.allclose(masks["ip"] + masks["creative"], torch.ones(1, 4, 8))
    assert all(torch.equal(masks[k], collect(1000)[k]) for k in masks)


def test_margin_and_missing_scores():
    masks = owner_masks(
        {"ip": torch.full((1, 3, 3), 0.51), "creative": torch.full((1, 3, 3), 0.49)}
    )
    assert torch.allclose(masks["ip"], torch.full((1, 3, 3), 0.51))
    zero = owner_masks({"ip": torch.zeros(1, 3, 3)})
    assert torch.all(zero["creative"] == 1)
    assert torch.all(AxisOwner((3, 3)).finish_step()["creative"] == 1)


def test_text_and_unconditional_rows_are_skipped():
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    owner = AxisOwner((2, 2))
    plan = TargetPlan(sets=sets, name="layer")
    y = torch.ones(1, 4, 2)
    with recording(owner, [1]), latent_size((2, 2)):
        plan.run(y, y)
    assert not owner.scores
    with recording(owner), latent_size((2, 2)), token_layouts(lambda _: TokenLayout("text1d")):
        plan.run(y, y)
    assert not owner.scores


def test_layer_sampler_uses_previous_map_and_off_switch(monkeypatch):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", "layer")
    patcher = types.SimpleNamespace(model_options={})
    seen = []
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    plan = TargetPlan(sets=sets, name="layer")
    y = torch.zeros(1, 4, 8, 2)
    y[:, :, :4, 0] = 1
    y[:, :, 4:, 1] = 1

    def model(x, timestep, **c):
        with latent_size((4, 8)):
            return plan.run(y, y)

    def calc(model_arg, conds, x, sigma, options):
        seen.append(conds)
        options["model_function_wrapper"](
            model,
            {
                "input": x,
                "timestep": sigma,
                "c": {"transformer_options": {"cond_or_uncond": [0], "kilora_layer": "ip"}},
            },
        )
        return x

    install_layer_alpha(patcher, None, {}, 0, calc)
    hook = patcher.model_options["sampler_calc_cond_batch_function"]
    original = torch.full((1, 4, 8), 0.25)
    conds = [
        [
            {"kilora_layer": "ip", "mask": original},
            {"kilora_layer": "creative", "mask": 1 - original},
            {"kilora_layer": "functional", "mask": original},
        ]
    ]
    args = {
        "model": None,
        "conds": conds,
        "input": torch.zeros(1, 4, 4, 8),
        "sigma": torch.tensor([2.0]),
        "model_options": patcher.model_options,
    }
    hook(args)
    assert seen[0][0][0]["mask"] is original
    hook({**args, "sigma": torch.tensor([1.0])})
    assert torch.allclose(seen[1][0][0]["mask"], collect()["ip"])
    assert torch.allclose(seen[1][0][1]["mask"], collect()["creative"])
    assert torch.all(seen[1][0][2]["mask"] == 1)
    monkeypatch.delenv("KILORA_AXIS_ZBUFFER")
    hook({**args, "sigma": torch.tensor([0.5])})
    assert seen[2][0][0]["mask"] is original


def test_ranking_uses_relative_order_and_aggregates_each_expert():
    ids = [("ip", "a", 0), ("ip", "b", 0), ("creative", "c", 0)]

    def run(scale):
        owner = AxisOwner((2, 2))
        first = torch.tensor([0.003, 0.001, 0.002]).repeat(1, 4, 1) * scale
        second = torch.tensor([0.001, 0.003, 0.002]).repeat(1, 4, 1) * scale
        owner.record_scores(first, ids, (2, 2))
        owner.record_scores(second, ids, (2, 2))
        return owner.finish_step()

    small = run(1)
    large = run(1000)
    assert all(torch.equal(small[k], large[k]) for k in small)
    assert torch.allclose(small["ip"], torch.full((1, 2, 2), 0.5))


def test_ema_and_joint_text_layout():
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    owner = AxisOwner((2, 2))
    layout = TokenLayout("image2d", (1, 1), text="before")
    y = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    with latent_size((2, 2)), token_layouts(lambda _: layout, text_tokens=2):
        owner.record("layer", y, sets)
        first = owner.finish_step()
        assert torch.all(first["ip"] == 1)
        y[:, 2:] = torch.tensor([0.0, 1.0])
        owner.record("layer", y, sets)
        second = owner.finish_step()
    assert torch.allclose(second["ip"], torch.full((1, 2, 2), 0.5))
    assert torch.allclose(second["creative"], torch.full((1, 2, 2), 0.5))


def test_unset_switch_has_no_recording_and_identical_output(monkeypatch):
    import kilora.axis_owner as owner_module

    monkeypatch.delenv("KILORA_AXIS_ZBUFFER", raising=False)
    plan = TargetPlan(sets=[make_set("ip", [1, 0]), make_set("creative", [0, 1])])
    y = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(12))
    expected = plan.run(y, y)
    calls = []
    monkeypatch.setattr(AxisOwner, "record", lambda *args: calls.append(args))
    patcher = types.SimpleNamespace(model_options={})

    def calc(model_arg, conds, x, sigma, options):
        return options["model_function_wrapper"](
            lambda x, t, **c: plan.run(y, y),
            {"input": x, "timestep": sigma, "c": {"transformer_options": {}}},
        )

    install_layer_alpha(patcher, None, {}, 0, calc)
    hook = patcher.model_options["sampler_calc_cond_batch_function"]
    args = {
        "model": None,
        "conds": [[{"kilora_layer": "ip"}]],
        "input": y,
        "sigma": torch.tensor([1.0]),
        "model_options": patcher.model_options,
    }
    assert torch.equal(hook(args), expected)
    assert not calls
    assert owner_module._current.get() is None


def test_positive_batch_rows_and_multiple_conditions():
    owner = AxisOwner((2, 2), batch=1)
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    positive = torch.tensor([1.0, 0.0]).repeat(2, 4, 1)
    negative = torch.tensor([0.0, 1.0]).repeat(2, 4, 1)
    plan = TargetPlan(sets=sets, name="layer")
    y = torch.cat([positive, negative])
    with recording(owner, [0, 1]), latent_size((2, 2)):
        plan.run(y, y)
    masks = owner.finish_step()
    assert masks["ip"].shape == (1, 2, 2)
    assert torch.all(masks["ip"] == 1)


def test_zero_expert_scores_choose_creative():
    owner = AxisOwner((2, 2))
    ids = [("ip", "one", 0), ("creative", "two", 0)]
    owner.record_scores(torch.zeros(1, 4, 2), ids, (2, 2))
    masks = owner.finish_step()
    assert torch.all(masks["creative"] == 1)
    assert torch.all(masks["ip"] == 0)
