# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import types

import pytest
import torch

from kilora.call_values import compose_layout, current_compose_layout
from kilora.comfyui import layer_compose as lc
from kilora.comfyui.layer_batch import install_layer_batch
from kilora.comfyui.layer_compose import ComposedLayout, Segment, compose, token_weights


class Cross:
    def __init__(self, cond):
        self.cond = cond


class Regular:
    def __init__(self, cond):
        self.cond = cond


@pytest.fixture
def comfy_conds(monkeypatch):
    comfy = types.ModuleType("comfy")
    module = types.ModuleType("comfy.conds")
    module.CONDRegular = Regular
    module.CONDCrossAttn = Cross
    comfy.conds = module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", module)
    return module


def layout(*masks, strengths=(1.0,), axes=None):
    segments = [Segment("global", None, 4, None, s) for s in strengths]
    for i, mask in enumerate(masks):
        axis = "ip" if axes is None else axes[i]
        segments.append(Segment("layer", axis, 3, mask, 1.0))
    return ComposedLayout(tuple(segments))


def test_hard_mask_weights_and_a_token_outside_every_mask():
    mask = torch.zeros(1, 2, 2)
    mask[0, 0, 0] = 1.0
    w = token_weights(layout(mask), 4, (2, 2), 0.0, for_text=True)
    assert w.shape == (2, 1, 4)
    assert torch.allclose(w.sum(0), torch.ones(1, 4))
    assert torch.allclose(w[1, 0], torch.tensor([1 - lc.BETA, 0.0, 0.0, 0.0]))
    assert torch.allclose(w[0, 0], torch.tensor([lc.BETA, 1.0, 1.0, 1.0]))


def test_soft_mask_weights_scale_with_membership():
    mask = torch.full((1, 2, 2), 0.5)
    w = token_weights(layout(mask), 4, (2, 2), 0.0, for_text=True)
    assert torch.allclose(w[1], torch.full((1, 4), 0.5 * (1 - lc.BETA)))
    assert torch.allclose(w[0], 1 - w[1])


def test_overlaps_never_sum_above_one():
    a = torch.ones(1, 2, 2)
    b = torch.ones(1, 2, 2)
    w = token_weights(layout(a, b), 4, (2, 2), 0.0, for_text=True)
    assert torch.allclose(w[1], w[2])
    assert torch.allclose(w[1] + w[2], torch.full((1, 4), 1 - lc.BETA))
    assert torch.allclose(w.sum(0), torch.ones(1, 4))


def test_global_segments_share_by_strength():
    w = token_weights(layout(strengths=(1.0, 3.0)), 4, (2, 2), 0.0, for_text=True)
    assert torch.allclose(w[0], torch.full((1, 4), 0.25))
    assert torch.allclose(w[1], torch.full((1, 4), 0.75))


def test_text_end_gives_the_global_prompt_all_weight_but_not_the_mounts():
    mask = torch.ones(1, 2, 2)
    late = token_weights(layout(mask), 4, (2, 2), lc.TEXT_END, for_text=True)
    assert torch.equal(late[1], torch.zeros(1, 4))
    assert torch.equal(late[0], torch.ones(1, 4))
    mounts = token_weights(layout(mask), 4, (2, 2), 0.9, for_text=False)
    assert torch.allclose(mounts[1], torch.full((1, 4), 1 - lc.BETA))


def test_constants_are_read_at_call_time(monkeypatch):
    mask = torch.ones(1, 2, 2)
    monkeypatch.setattr(lc, "BETA", 0.0)
    w = token_weights(layout(mask), 4, (2, 2), 0.0, for_text=True)
    assert torch.equal(w[1], torch.ones(1, 4))
    monkeypatch.setattr(lc, "TEXT_END", 1.5)
    assert torch.equal(token_weights(layout(mask), 4, (2, 2), 0.9, for_text=True)[1], w[1])


def test_shared_mount_weights_give_ip_and_creative_equal_shares_in_a_box(monkeypatch):
    mask = torch.zeros(1, 2, 2)
    mask[0, 1, 1] = 1.0
    monkeypatch.setattr(lc, "MOUNT_WEIGHTS", "shared")
    w = token_weights(layout(mask), 4, (2, 2), 0.0, for_text=False)
    assert torch.allclose(w[1, 0], torch.tensor([0.0, 0.0, 0.0, 0.5]))
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 1.0, 1.0, 0.5]))
    text = token_weights(layout(mask), 4, (2, 2), 0.0, for_text=True)
    assert torch.allclose(text[1, 0, 3], torch.tensor(1 - lc.BETA))


def test_no_grid_is_an_error():
    with pytest.raises(ValueError):
        token_weights(layout(torch.ones(1, 2, 2)), 7, (2, 2), 0.0, for_text=True)


def cond(length, value, **extra):
    context = torch.full((1, length, 2), float(value))
    return {"model_conds": {"c_crossattn": Cross(context), "y": "keep"}, **extra}


def layer(axis, length, value, mask=None, **extra):
    extra = {"kilora_layer": axis, "kilora_compose": "single", **extra}
    if mask is not None:
        extra["mask"] = mask
    return cond(length, value, **extra)


def box():
    mask = torch.zeros(1, 4, 4)
    mask[:, 1:3, 1:3] = 1.0
    return mask


def two_layers(negatives=None):
    positive = [layer("creative", 4, 1.0), layer("ip", 6, 2.0, box())]
    return [positive, [cond(4, -1.0)] if negatives is None else negatives]


def test_two_layer_sample_with_a_plain_negative():
    groups, (pos, neg) = compose(two_layers(), family="anima")
    assert [s.role for s in pos.segments] == ["global", "layer"]
    assert [s.axis for s in pos.segments] == ["creative", "ip"]
    assert pos.segments[0].mask is None and pos.segments[1].mask is not None
    assert pos.lengths == neg.lengths == (4, 12)
    (p,), (n,) = groups
    assert p["strength"] == 1.0 and "mask" not in p and "kilora_layer" not in p
    assert p["model_conds"]["y"] == "keep"
    context = p["model_conds"]["c_crossattn"].cond
    negative = n["model_conds"]["c_crossattn"].cond
    assert context.shape == negative.shape == (1, 16, 2)
    assert context[0, :4, 0].tolist() == [1.0] * 4
    assert context[0, 4:, 0].tolist() == [2.0] * 12
    assert negative[0, :, 0].tolist() == [-1.0] * 16
    view = pos.caption_layout(torch.tensor([True]))
    assert view.prompt == 4 and view.lengths == (12,) and view.total == 16


def test_a_negative_routed_through_a_layer_replaces_its_segment():
    routed = layer("ip", 6, -5.0)
    conds = two_layers([cond(4, -1.0), routed])
    groups, (pos, neg) = compose(conds, family="cosmos")
    negative = groups[1][0]["model_conds"]["c_crossattn"].cond
    assert len(groups[1]) == 1
    assert negative[0, :4, 0].tolist() == [-1.0] * 4
    assert negative[0, 4:, 0].tolist() == [-5.0] * 6
    assert pos.lengths == neg.lengths == (4, 6)


def test_a_full_layer_mask_counts_as_global():
    positive = [layer("creative", 4, 1.0, torch.ones(1, 4, 4)), layer("ip", 4, 2.0, box())]
    _, (pos, _) = compose([positive, [cond(4, 0.0)]], family="anima")
    assert [s.role for s in pos.segments] == ["global", "layer"]


@pytest.mark.parametrize(
    "change, cause",
    [
        (lambda c: c[0][1].update(kilora_compose="separate"), "a layer uses compose separate"),
        (lambda c: c[0][1].update(control=object()), "a layer has its own control"),
        (lambda c: c[0][1].update(hooks=object()), "a layer has its own hooks"),
        (lambda c: c[0][1].update(timestep_start=0.5), "a layer has its own timestep range"),
        (lambda c: c[0][1].update(area=(1, 1, 0, 0)), "a layer has an area instead of a mask"),
        (
            lambda c: c[0][1].update(kilora_alpha_source="attention"),
            "a layer uses alpha_source attention",
        ),
        (lambda c: c[0].pop(0), "no global prompt"),
        (lambda c: c[1].append(cond(4, -2.0)), "several plain negatives"),
    ],
)
def test_fallback_causes(change, cause):
    conds = two_layers()
    change(conds)
    groups, info = compose(conds, family="anima")
    assert groups is None
    assert info.startswith(cause)


def test_fallback_causes_from_the_model():
    groups, info = compose(two_layers(), options={"kilora_layer_filter": True}, family="anima")
    assert (groups, info) == (None, "a KiLoRA Layer Filter is connected")
    groups, info = compose(two_layers(), options={"kilora_alpha_mask": True}, family="anima")
    assert (groups, info) == (None, "KiLoRA Apply has an alpha mask")
    groups, info = compose(two_layers(), family="anima", carry=True)
    assert (groups, info) == (None, "the mods carry state across steps")
    groups, info = compose(two_layers(), family="sdxl")
    assert (groups, info) == (None, "the model family is not Anima or Cosmos")


def test_compose_layout_rows():
    pos = layout()
    assert current_compose_layout() is None
    with compose_layout((pos, pos), torch.tensor([0, 1])):
        layouts, rows = current_compose_layout()
        assert layouts == (pos, pos) and rows.tolist() == [0, 1]
    assert current_compose_layout() is None


def sample(conds, monkeypatch, *, ready):
    monkeypatch.setattr(lc, "SINGLE_READY", ready)
    seen = []

    def previous(args):
        seen.append(args["conds"])
        return "result"

    patcher = types.SimpleNamespace(model_options={"sampler_calc_cond_batch_function": previous})
    install_layer_batch(patcher, family="anima")
    result = patcher.model_options["sampler_calc_cond_batch_function"](
        {"conds": conds, "model_options": patcher.model_options, "sigma": torch.ones(1)}
    )
    return result, seen


def test_single_runs_as_separate_while_not_ready(monkeypatch, comfy_conds, caplog):
    def build(mode):
        conds = two_layers()
        for c in conds[0]:
            c["kilora_compose"] = mode
            c["kilora_layer_batch"] = False
        return conds

    hooks = types.ModuleType("comfy.hooks")

    class Group:
        def add(self, hook):
            pass

        @staticmethod
        def combine_all_hooks(groups):
            return "hooks"

    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = lambda options, hook_scope: options
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    sys.modules["comfy"].hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)

    skip = ("kilora_compose", "model_conds", "mask")

    def strip(groups):
        return [[{k: v for k, v in c.items() if k not in skip} for c in g] for g in groups]

    expected, main = sample(build("separate"), monkeypatch, ready=False)
    with caplog.at_level(logging.INFO, logger="kilora.comfyui.layer_batch"):
        result, single = sample(build("single"), monkeypatch, ready=False)
    assert result == expected
    assert strip(single[0]) == strip(main[0])
    for ga, gb in zip(single[0], main[0], strict=True):
        for a, b in zip(ga, gb, strict=True):
            ma, mb = a.get("mask"), b.get("mask")
            assert (ma is None and mb is None) or torch.equal(ma, mb)
    assert not any("not available yet" in r.getMessage() for r in caplog.records)
    assert not any("compose single" in r.getMessage() for r in caplog.records)


def test_single_is_ready_by_default():
    assert lc.SINGLE_READY is True


def test_ready_single_sample_carries_the_layout_id(monkeypatch, comfy_conds):
    _, seen = sample(two_layers(), monkeypatch, ready=True)
    positive, negative = seen[0]
    assert len(positive) == len(negative) == 1
    key = positive[0]["model_conds"]["kilora_compose_id"].cond
    assert negative[0]["model_conds"]["kilora_compose_id"].cond.tolist() == key.tolist()
    pos, neg = lc.lookup(int(key[0]))
    assert pos.lengths == neg.lengths == (4, 12)


def test_an_alpha_mask_marks_the_model_options():
    from kilora.comfyui.patching import _mask_alpha_wrapper

    out = types.SimpleNamespace(model_options={})
    _mask_alpha_wrapper(out, torch.ones(1, 8, 8))
    assert out.model_options["kilora_alpha_mask"] is True
    groups, info = compose(two_layers(), options=out.model_options, family="anima")
    assert (groups, info) == (None, "KiLoRA Apply has an alpha mask")
