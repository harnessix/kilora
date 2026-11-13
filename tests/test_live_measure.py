# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.comfyui.live_write import PathHandle, image_loss


def test_loss_is_the_mean_of_both_conditionings():
    x0 = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(0))
    seen = []

    def apply_model(x_t, t, c_crossattn):
        seen.append((x_t.clone(), t.clone()))
        rows = c_crossattn.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))
        return torch.zeros_like(x_t) + rows

    handle = PathHandle(
        session=None,
        rule=None,
        recorder=None,
        timesteps=[0.2, 0.5, 0.8],
        conditioning={"c_crossattn": torch.full((2, 3), 2.0)},
        empty_conditioning={"c_crossattn": torch.full((2, 3), -1.0)},
        apply_model=apply_model,
    )
    value = image_loss(handle, x0, event=3)(None)
    full = float((2.0 - x0).pow(2).mean())
    empty = float((-1.0 - x0).pow(2).mean())
    assert abs(value - (full + empty) / 2) < 1e-6
    ((x, t),) = seen
    xa, xb = x.chunk(2)
    ta, tb = t.chunk(2)
    assert torch.equal(xa, xb) and torch.equal(ta, tb)


def test_loss_is_taken_in_the_models_latent_space():
    x0 = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(1))
    handle = PathHandle(
        session=None,
        rule=None,
        recorder=None,
        timesteps=[0.5],
        conditioning={"c_crossattn": torch.zeros(1, 2)},
        empty_conditioning={"c_crossattn": torch.zeros(1, 2)},
        apply_model=lambda x_t, t, c_crossattn: torch.zeros_like(x_t),
        process_in=lambda x: 2.0 * x + 1.0,
    )
    value = image_loss(handle, x0, event=0)(None)
    assert abs(value - float((2.0 * x0 + 1.0).pow(2).mean())) < 1e-6


def _row_model(calls):
    def apply_model(x_t, t, c_crossattn, **other):
        calls.append(x_t.shape[0])
        rows = c_crossattn.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))
        return torch.zeros_like(x_t) + rows

    return apply_model


def _measure_handle(apply_model, caption, empty):
    return PathHandle(
        session=None,
        rule=None,
        recorder=None,
        timesteps=[0.5],
        conditioning=caption,
        empty_conditioning=empty,
        apply_model=apply_model,
    )


def test_one_model_call_per_forward_and_the_guard_reuses_the_base_pass():
    calls = []
    x0 = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(4))
    cap, emp = {"c_crossattn": torch.full((1, 3, 2), 2.0)}, {"c_crossattn": torch.zeros(1, 3, 2)}
    measure = image_loss(_measure_handle(_row_model(calls), cap, emp), x0, event=0)
    measure(None)
    assert calls == [2]
    score = measure.guard_score(None)
    assert calls == [2, 2]
    assert abs(score["gap"] - (float(x0.pow(2).mean()) - float((2.0 - x0).pow(2).mean()))) < 1e-6
    assert "gap_alt" in score
    measure.guard_score(object())
    assert calls == [2, 2, 2, 2] and measure.calls == [4]


def test_text_lengths_are_joined_as_comfyui_does_and_give_the_same_loss():
    x0 = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(5))
    cap = {"c_crossattn": torch.randn(1, 3, 2, generator=torch.Generator().manual_seed(6))}
    emp = {"c_crossattn": torch.randn(1, 2, 2, generator=torch.Generator().manual_seed(7))}
    joined, apart = [], []
    one = image_loss(_measure_handle(_row_model(joined), cap, emp), x0, event=0)(None)
    other = {**emp, "y": None}
    cap2 = {**cap, "y": 1}
    two = image_loss(_measure_handle(_row_model(apart), cap2, other), x0, event=0)(None)
    assert joined == [2] and apart == [1, 1]
    assert abs(one - two) < 1e-6


def test_conditions_join_only_when_safe():
    from kilora.comfyui.live_write import batch_conditions

    a = {"c_crossattn": torch.ones(1, 4, 2), "opts": {"k": 1}}
    assert batch_conditions(a, {"c_crossattn": torch.zeros(1, 4, 2), "opts": {"k": 1}}, 1)
    assert batch_conditions(a, {"c_crossattn": torch.zeros(1, 4, 2), "opts": {"k": 2}}, 1) is None
    assert batch_conditions(a, {"c_crossattn": torch.zeros(1, 4, 3), "opts": {"k": 1}}, 1) is None
    five = {"c_crossattn": torch.ones(1, 5, 2), "opts": a["opts"]}
    long = {"c_crossattn": torch.zeros(1, 7, 2), "opts": a["opts"]}
    assert batch_conditions(five, long, 1) is None
    j = batch_conditions(a, {"c_crossattn": torch.zeros(1, 2, 2), "opts": a["opts"]}, 1)
    assert j["c_crossattn"].shape == (2, 4, 2) and j["opts"] is a["opts"]
