# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from kilora.comfyui.layer_crop import CropHistory, crop_box


@pytest.fixture(autouse=True)
def comfy_conds(monkeypatch):
    import sys
    import types

    class Cond:
        def __init__(self, cond):
            self.cond = cond

    comfy = types.ModuleType("comfy")
    module = types.ModuleType("comfy.conds")
    module.CONDRegular = Cond
    comfy.conds = module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", module)


@pytest.fixture(autouse=True)
def no_ring(monkeypatch):

    from kilora.comfyui import layer_crop

    monkeypatch.setattr(layer_crop, "ring_width", lambda bounds, progress, **schedule: 0)


def test_crop_margin_grid_and_edges():
    assert crop_box((40, 40, 48, 48), (128, 128)) == (32, 32, 32, 32)
    assert crop_box((0, 0, 8, 8), (128, 128)) == (16, 16, 0, 0)
    h, w, y, x = crop_box((20, 50, 100, 100), (160, 160), patch=4)
    assert h % 16 == w % 16 == y % 4 == x % 4 == 0
    assert y <= 10 and y + h >= 110 and x <= 42 and x + w >= 108
    assert crop_box((0, 0, 120, 120), (128, 128)) == (128, 128, 0, 0)


def test_crop_history_keeps_union_for_four_steps():
    history = CropHistory()
    assert history.update((1, 2, 4, 5)) == (1, 2, 4, 5)
    for _ in range(3):
        assert history.update((5, 6, 8, 9)) == (1, 2, 8, 9)
    assert history.update((5, 6, 8, 9)) == (5, 6, 8, 9)
    with pytest.raises(ValueError):
        CropHistory(3)


def test_sampler_crop_composes_and_keeps_history():
    import types

    import torch

    from kilora.comfyui.layer_crop import install_layer_crop

    seen = []

    def calc(model, groups, x, sigma, options):
        cond = groups[0][0]
        seen.append(cond.get("area"))
        area = cond.get("area", (*x.shape[-2:], 0, 0))
        h, w, y, left = area
        output = torch.zeros_like(x)
        window = x[..., y : y + h, left : left + w]
        output[..., y : y + h, left : left + w] = window * 2
        return [output]

    patcher = types.SimpleNamespace(model_options={})
    install_layer_crop(patcher, 2, calc=calc)
    sampler = patcher.model_options["sampler_calc_cond_batch_function"]
    x = torch.randn(1, 16, 2, 128, 128)
    mask = torch.zeros(1, 2, 128, 128)
    mask[..., 40:48, 40:48] = 1
    cond = {"kilora_crop": "auto", "kilora_alpha_source": "mask", "mask": mask}

    def call(sigma):
        return sampler(
            {
                "input": x,
                "sigma": torch.tensor([sigma]),
                "conds": [[cond]],
                "model": None,
                "model_options": patcher.model_options,
            }
        )[0]

    result = call(5.0)
    assert seen[-1] == (32, 32, 32, 32)
    assert torch.equal(result * mask.unsqueeze(1), 2 * x * mask.unsqueeze(1))
    mask.zero_()
    mask[..., 80:88, 80:88] = 1
    for sigma in (4.0, 3.0, 2.0):
        call(sigma)
        h, w, top, left = seen[-1]
        assert top <= 40 and left <= 40 and top + h >= 88 and left + w >= 88
    call(1.0)
    assert seen[-1] == (32, 32, 72, 72)


def test_crop_exclusions_and_chained_sampler():
    import types

    import torch

    from kilora.comfyui.layer_crop import install_layer_crop

    seen = []
    patcher = types.SimpleNamespace(
        model_options={"sampler_calc_cond_batch_function": lambda a: seen.append(a)}
    )
    install_layer_crop(patcher)
    mask = torch.zeros(1, 128, 128)
    mask[..., 40:48, 40:48] = 1
    base = {"kilora_crop": "auto", "kilora_alpha_source": "mask", "mask": mask}
    conds = [
        base,
        {**base, "control": object()},
        {**base, "kilora_crop": "off"},
        {**base, "kilora_alpha_source": "attention"},
        {**base, "area": (8, 8, 0, 0)},
    ]
    args = {
        "input": torch.zeros(1, 1, 128, 128),
        "sigma": torch.tensor([1.0]),
        "conds": [conds],
        "model_options": patcher.model_options,
    }
    patcher.model_options["sampler_calc_cond_batch_function"](args)
    actual = seen[-1]["conds"][0]
    assert actual[0]["area"] == (32, 32, 32, 32)
    assert all(actual[i].get("area") == conds[i].get("area") for i in range(1, len(conds)))
    assert "area" not in base
    args["model_options"] = {"transformer_options": {"patches": {"attn1_patch": [object()]}}}
    patcher.model_options["sampler_calc_cond_batch_function"](args)
    assert "area" not in seen[-1]["conds"][0][0]


def test_local_coordinates_can_shift_composition():
    import torch

    def model(x):
        return torch.arange(x.shape[-1], dtype=x.dtype).expand_as(x)

    x = torch.zeros(1, 1, 32, 64)
    full = model(x)[..., 8:24, 24:40]
    cropped = model(x[..., 8:24, 24:40])
    assert torch.equal(full - cropped, torch.full_like(full, 24))


def test_crop_safe_gate_and_large_masks():
    import types

    import torch

    from kilora.comfyui.layer_crop import install_layer_crop

    mask = torch.ones(1, 64, 64)
    cond = {"kilora_crop": "auto", "kilora_alpha_source": "mask", "mask": mask}
    args = {
        "input": torch.zeros(1, 1, 64, 64),
        "sigma": torch.tensor([1.0]),
        "conds": [[cond]],
        "model_options": {},
    }
    seen = []
    patcher = types.SimpleNamespace(model_options={"sampler_calc_cond_batch_function": seen.append})
    install_layer_crop(patcher)
    patcher.model_options["sampler_calc_cond_batch_function"](args)
    assert "area" not in seen[-1]["conds"][0][0]
    mask.zero_()
    mask[..., 8:12, 8:12] = 1
    patcher = types.SimpleNamespace(model_options={"sampler_calc_cond_batch_function": seen.append})
    install_layer_crop(patcher, safe=False)
    patcher.model_options["sampler_calc_cond_batch_function"](args)
    assert "area" not in seen[-1]["conds"][0][0]


def test_control_chain_windows_and_cache():
    import types

    import torch

    from kilora.comfyui.layer_crop import install_layer_crop

    class Control:
        def __init__(self, hint, previous=None):
            self.cond_hint_original = hint
            self.cond_hint = hint
            self.previous_controlnet = previous

        def copy(self):
            return Control(self.cond_hint_original, self.previous_controlnet)

    hint = torch.arange(256 * 384).reshape(1, 1, 256, 384)
    control = Control(hint, Control(hint + 1))
    mask = torch.zeros(1, 128, 128)
    mask[..., 40:48, 40:48] = 1
    cond = {"kilora_crop": "auto", "kilora_alpha_source": "mask", "mask": mask, "control": control}
    seen = []
    patcher = types.SimpleNamespace(model_options={"sampler_calc_cond_batch_function": seen.append})
    install_layer_crop(patcher)
    args = {
        "input": torch.zeros(1, 1, 128, 128),
        "sigma": torch.tensor([2.0]),
        "conds": [[cond]],
        "model_options": {},
    }
    sampler = patcher.model_options["sampler_calc_cond_batch_function"]
    sampler(args)
    cropped = seen[-1]["conds"][0][0]["control"]
    assert torch.equal(cropped.cond_hint_original, hint[..., 64:128, 96:192])
    assert torch.equal(
        cropped.previous_controlnet.cond_hint_original, hint[..., 64:128, 96:192] + 1
    )
    assert cropped.cond_hint is None and cropped.previous_controlnet.cond_hint is None
    import torch.nn.functional as F

    full_output = F.avg_pool2d(hint.float(), (2, 3))
    crop_output = F.avg_pool2d(cropped.cond_hint_original.float(), (2, 3))
    assert torch.equal(crop_output, full_output[..., 32:64, 32:64])
    assert torch.equal(
        crop_output * mask[..., 32:64, 32:64],
        full_output[..., 32:64, 32:64] * mask[..., 32:64, 32:64],
    )
    args["sigma"] = torch.tensor([1.0])
    sampler(args)
    assert seen[-1]["conds"][0][0]["control"] is cropped
    assert torch.equal(control.cond_hint_original, hint)


def test_per_row_windows_map_regions_alphas_and_depth():
    import torch

    from kilora.bake import TargetPlan
    from kilora.call_values import crop_window
    from kilora.compute import _map_axis_alpha
    from kilora.depth_map import DepthMap
    from kilora.region import latent_size

    image = torch.arange(64 * 64).reshape(1, 64, 64).float() / 4096
    boxes = torch.tensor([[0, 0, 64, 64], [32, 32, 64, 64]])
    expected = torch.cat((image[:, :32, :32], image[:, 32:, 32:])).reshape(2, -1)
    h = torch.zeros(2, 32 * 32, 1)
    plan = TargetPlan(name="target", regions=[image], token_depth=DepthMap(image, 0, 1, 0, 0))
    with latent_size((32, 32)), crop_window(boxes):
        region = plan._token_regions(h)[0]
        alpha = _map_axis_alpha(image, h, "target").squeeze(-1)
        depth = plan._token_depth(h)
    assert torch.equal(region, expected)
    assert torch.allclose(alpha, expected)
    assert torch.equal(depth, expected)
    with latent_size((64, 64)):
        full = plan._token_regions(torch.zeros(1, 64 * 64, 1))[0]
    assert torch.equal(full.reshape(1, 64, 64), image)


def test_batched_crop_windows_reach_model_and_match_full_maps():
    import types

    import torch

    from kilora.bake import TargetPlan
    from kilora.comfyui.layer_crop import install_layer_crop
    from kilora.comfyui.patching import _key_mask_by_step
    from kilora.compute import _map_axis_alpha
    from kilora.depth_map import DepthMap

    image = torch.arange(128 * 128).reshape(1, 128, 128).float() / (128 * 128)
    plan = TargetPlan(name="target", regions=[image], token_depth=DepthMap(image, 0, 1, 0, 0))
    calls = []
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher)
    patcher.model_options.pop("sampler_calc_cond_batch_function")

    def model(x, timestep, **kwargs):
        assert "kilora_crop_window" not in kwargs
        calls.append(x.shape)
        h = x.flatten(2).transpose(1, 2)
        alpha = _map_axis_alpha(image, h, "target").squeeze(-1)
        region = plan._token_regions(h)[0]
        depth = plan._token_depth(h)
        return (alpha + region + depth).reshape_as(x)

    def calc(model_arg, groups, x, sigma, options):
        entries = groups[0]
        areas = [c["area"] for c in entries]
        crops = torch.cat([x[..., top : top + h, left : left + w] for h, w, top, left in areas])
        windows = torch.cat([c["model_conds"]["kilora_crop_window"].cond for c in entries])
        output = options["model_function_wrapper"](
            model, {"input": crops, "timestep": sigma, "c": {"kilora_crop_window": windows}}
        )
        return output, areas

    install_layer_crop(patcher, calc=calc)
    masks = torch.zeros(2, 128, 128)
    masks[0, 40:48, 40:48] = 1
    masks[1, 80:88, 80:88] = 1
    conds = [{"kilora_crop": "auto", "kilora_alpha_source": "mask", "mask": mask} for mask in masks]
    result, areas = patcher.model_options["sampler_calc_cond_batch_function"](
        {
            "input": torch.zeros(1, 1, 128, 128),
            "sigma": torch.tensor([1.0]),
            "conds": [conds],
            "model": None,
            "model_options": patcher.model_options,
        }
    )
    assert len(calls) == 1 and calls[0][0] == 2
    for i, (height, width, top, left) in enumerate(areas):
        expected = image[..., top : top + height, left : left + width] * 3
        assert torch.allclose(result[i], expected)


def test_panel_tokens_use_each_crop_position():
    import torch

    from kilora.call_values import crop_window
    from kilora.family_spec import TokenLayout
    from kilora.roles import panel_tokens

    layout = TokenLayout(kind="image2d", patch=(1, 1))
    with crop_window(torch.tensor([[0, 0, 64, 64], [0, 32, 64, 64]])):
        panels = panel_tokens(layout, (32, 32), ("left", 0.5))
    assert panels.shape == (2, 1024)
    assert panels[0].all() and not panels[1].any()
