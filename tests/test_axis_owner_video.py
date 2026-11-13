# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import types

import pytest
import torch
from test_axis_owner import make_set

from kilora.axis_owner import AxisOwner
from kilora.bake import TargetPlan
from kilora.call_values import current_axis_alphas
from kilora.comfyui import patching
from kilora.comfyui.layer_alpha import install_layer_alpha
from kilora.family_spec import TokenLayout
from kilora.region import latent_size, token_layouts


def inputs():
    y = torch.zeros(1, 4, 6, 2)
    y[:, :, :3, 0] = 1
    y[:, :, 3:, 1] = 1
    return y.reshape(1, 24, 2)


def test_video_layout_records_single_frame_and_skips_multiple_frames(caplog):
    owner = AxisOwner((8, 12))
    layout = TokenLayout("video3d", (1, 2, 2))
    sets = [make_set("ip", [1, 0]), make_set("creative", [0, 1])]
    with latent_size((1, 8, 12)), token_layouts(lambda _: layout):
        owner.record("layer", inputs(), sets)
    assert owner.scores
    masks = owner.finish_step()
    assert masks["ip"][0, 0, 0] == 1
    assert masks["creative"][0, 0, -1] == 1
    with (
        caplog.at_level(logging.INFO, logger="kilora.axis_owner"),
        latent_size((2, 8, 12)),
        token_layouts(lambda _: layout),
    ):
        owner.record("layer", inputs().repeat(1, 2, 1), sets)
        owner.record("layer", inputs().repeat(1, 2, 1), sets)
    assert not owner.scores
    assert caplog.text.count("disabled for multiple frames") == 1


@pytest.mark.parametrize("mode", ["single", "soft"])
def test_single_frame_wrapper_matches_4d_and_installs_attention(monkeypatch, mode):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", mode)
    monkeypatch.setenv("KILORA_AXIS_INHIBIT", "0")

    def run(video):
        layout = TokenLayout("video3d", (1, 2, 2)) if video else TokenLayout("image2d", (2, 2))
        monkeypatch.setattr(patching, "layout_resolver", lambda *_: lambda _: layout)
        patcher = types.SimpleNamespace(model_options={})
        patching._key_mask_by_step(patcher)
        plan = TargetPlan(sets=[make_set("ip", [1, 0]), make_set("creative", [0, 1])], name="layer")
        seen = []

        def model(x, timestep, **c):
            seen.append((dict(current_axis_alphas()), c["transformer_options"]))
            return plan.run(inputs(), inputs())

        wrapper = patcher.model_options["model_function_wrapper"]
        outputs = []
        for step in (2, 1):
            shape = (1, 16, 1, 8, 12) if video else (1, 16, 8, 12)
            outputs.append(
                wrapper(
                    model,
                    {
                        "input": torch.zeros(shape),
                        "timestep": torch.tensor([step]),
                        "c": {"transformer_options": {"cond_or_uncond": [0]}},
                    },
                )
            )
        assert seen[0][0] == {}
        alphas, options = seen[1]
        assert alphas["ip"].shape == (1, 8, 12)
        assert alphas["ip"][0, 0, 0] == 1
        assert alphas["creative"][0, 0, -1] == 1
        q = torch.ones(1, 24, 1)
        options["patches"]["attn1_patch"][-1](q, q, q)
        captured = {}

        def attention(q, k, v, heads, **kwargs):
            captured.update(kwargs)
            return q

        options["optimized_attention_override"](attention, q, q, q, 1)
        assert captured["mask"].shape == (1, 1, 24, 24)
        return outputs, alphas, captured["mask"]

    image, video = run(False), run(True)
    assert all(torch.equal(a, b) for a, b in zip(image[0], video[0], strict=True))
    assert all(torch.equal(image[1][key], video[1][key]) for key in ("ip", "creative"))
    assert torch.equal(image[2], video[2])


@pytest.mark.parametrize("mode", ["single", "soft", "layer"])
def test_multiple_frames_disable_owner_once(monkeypatch, caplog, mode):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", mode)
    patcher = types.SimpleNamespace(model_options={})
    seen = []

    def model(x, timestep, **c):
        seen.append((dict(current_axis_alphas()), c.get("transformer_options", {})))
        return x

    if mode == "layer":

        def calc(model_arg, conds, x, sigma, options):
            assert "mask" not in conds[0][0]
            return options["model_function_wrapper"](
                model, {"input": x, "timestep": sigma, "c": {}}
            )

        install_layer_alpha(patcher, None, {}, 0, calc)
        hook = patcher.model_options["sampler_calc_cond_batch_function"]

        def call(step):
            return hook(
                {
                    "model": None,
                    "conds": [[{"kilora_layer": "ip"}]],
                    "input": torch.zeros(1, 16, 2, 8, 12),
                    "sigma": torch.tensor([step]),
                    "model_options": patcher.model_options,
                }
            )
    else:
        patching._key_mask_by_step(patcher)
        wrapper = patcher.model_options["model_function_wrapper"]

        def call(step):
            return wrapper(
                model,
                {"input": torch.zeros(1, 16, 2, 8, 12), "timestep": torch.tensor([step]), "c": {}},
            )

    with caplog.at_level(logging.INFO):
        call(2)
        call(1)
    assert caplog.text.count("disabled for multiple frames") == 1
    assert all(
        not alphas and "optimized_attention_override" not in options for alphas, options in seen
    )


@pytest.mark.parametrize("mode", ["single", "soft"])
def test_batch_size_change_resets_owner(monkeypatch, mode):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", mode)
    patcher = types.SimpleNamespace(model_options={})
    patching._key_mask_by_step(patcher)
    wrapper = patcher.model_options["model_function_wrapper"]
    plan = TargetPlan(sets=[make_set("ip", [1, 0]), make_set("creative", [0, 1])])
    seen = []

    def model(x, timestep, **c):
        seen.append(dict(current_axis_alphas()))
        y = inputs().repeat(x.shape[0], 1, 1)
        return plan.run(y, y)

    for batch, step in ((1, 3), (1, 2), (2, 1), (2, 0)):
        wrapper(
            model,
            {
                "input": torch.zeros(batch, 16, 1, 8, 12),
                "timestep": torch.tensor([step]),
                "c": {"transformer_options": {"cond_or_uncond": [0]}},
            },
        )
    assert seen[1]["ip"].shape == (1, 8, 12)
    assert seen[2] == {}
    assert seen[3]["ip"].shape == (2, 8, 12)
