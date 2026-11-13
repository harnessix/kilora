# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.call_values import current_axis_alphas
from kilora.comfyui.patching import _mask_alpha_wrapper
from kilora.compute import LayerSet, run_layer
from kilora.region import latent_size


class Patcher:
    def __init__(self):
        self.model_options = {}


def _call(patcher, mask, feather_px=0):
    _mask_alpha_wrapper(patcher, mask, feather_px)
    args = {"input": torch.zeros(1, 4, 8, 8), "timestep": 0, "c": {}}
    return patcher.model_options["model_function_wrapper"](
        lambda *args, **kwargs: dict(current_axis_alphas()), args
    )


def test_mask_maps_to_grid_and_complement():
    mask = torch.zeros(1, 32, 32)
    mask[:, :, :16] = 1
    alphas = _call(Patcher(), mask)
    assert torch.equal(alphas["ip"][0, :, :4], torch.ones(8, 4))
    assert torch.equal(alphas["ip"][0, :, 4:], torch.zeros(8, 4))
    assert torch.equal(alphas["creative"], 1 - alphas["ip"])
    assert current_axis_alphas() == {}


def test_feather_only_softens_edge():
    mask = torch.zeros(1, 32, 32)
    mask[:, :, :16] = 1
    soft = _call(Patcher(), mask, 4)["ip"][0]
    assert torch.equal(soft[:, :2], torch.ones(8, 2))
    assert torch.equal(soft[:, 6:], torch.zeros(8, 2))
    assert ((soft[:, 3:5] > 0) & (soft[:, 3:5] < 1)).all()


def test_mask_scales_layer_and_matches_plain_loop(monkeypatch):
    import kilora.compute as compute

    mask = torch.zeros(1, 32, 32)
    mask[:, :, :16] = 1
    alphas = _call(Patcher(), mask)
    y = torch.randn(1, 64, 4, generator=torch.Generator().manual_seed(3))
    layer = LayerSet(
        router=torch.eye(4),
        keys_in=torch.ones(1, 1, 4),
        down=torch.ones(2, 4),
        up=torch.ones(2, 4),
        form="input",
        axis="ip",
        gate_mode="none",
        per_expert=2,
    )
    from kilora.call_values import axis_alphas

    def run():
        return run_layer(y, y, [layer], 1, target="layer")

    baseline = run()
    with latent_size((8, 8)), axis_alphas(alphas):
        result = run()
        monkeypatch.setattr(compute, "_fast_path_ok", lambda *args, **kwargs: False)
        loop = run()
    expected = y + alphas["ip"].reshape(1, 64, 1) * (baseline - y)
    assert torch.allclose(result, expected)
    assert torch.allclose(result, loop)
