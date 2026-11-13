# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.compute import token_gate
from kilora.routing_crosses import cross_scores
from kilora.routing_features import RoutingFeatures


def test_zero_weights_leave_logits_untouched():
    logits = torch.tensor([[[0.2, 0.4]]])
    features = RoutingFeatures(torch.ones(1, 1, 2), torch.ones(1, 1, 2), None, None)
    result = cross_scores(logits, torch.eye(2), features, {"token_prompt": 0.0})
    assert result is logits
    assert torch.equal(token_gate(result, 0.0), token_gate(logits, 0.0))


def test_prompt_cross_raises_trigger_weight():
    logits = torch.zeros(1, 3, 2)
    token = torch.ones(1, 3, 2)
    prompt = torch.tensor([[[1.0, 0.0]]]).expand_as(token)
    features = RoutingFeatures(token, token, prompt, None)
    base = token_gate(logits, 0.0)
    crossed = token_gate(cross_scores(logits, torch.eye(2), features, {"token_prompt": 0.4}), 0.0)
    assert torch.all(crossed[..., 0] > base[..., 0])


def test_cross_refuses_unaligned_features():
    f = RoutingFeatures(torch.ones(1, 1, 2), torch.ones(1, 1, 2), torch.ones(1, 1, 3), None)
    with pytest.raises(ValueError, match="content-key space"):
        cross_scores(torch.zeros(1, 1, 2), torch.eye(2), f, {"token_prompt": 1.0})


def _layer():
    from kilora.compute import LayerSet

    return LayerSet(
        router=torch.eye(2),
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(2),
        up=torch.eye(2),
        per_expert=1,
        tau=1.0,
    )


@pytest.mark.parametrize("fast", [False, True])
def test_layer_crosses_and_zero_parity(fast, monkeypatch):
    import dataclasses

    from kilora import compute
    from kilora.routing_features import call_features, routing_features

    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    layer = _layer()
    x = torch.ones(1, 4, 2)
    prompt = torch.tensor([[[1.0, 0.0]]])
    seen = []
    real = compute._gates_from_logits

    def record(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(compute, "_gates_from_logits", record)
    with torch.no_grad(), call_features(prompt, None):
        base = compute.run_layer(x, x, [layer], 1)
        with routing_features():
            zero = compute.run_layer(x, x, [layer], 1)
            seen.clear()
            crossed = compute.run_layer(
                x, x, [dataclasses.replace(layer, cross_weights=(0, 0.4, 0))], 1
            )
            weight = seen[-1][..., 0].mean()
            replay = compute.run_layer(
                x, x, [dataclasses.replace(layer, cross_weights=(0, 0.4, 0))], 1
            )
    assert torch.equal(base, zero)
    assert torch.equal(crossed, replay)
    assert weight > 0.6
    assert not torch.equal(crossed, zero)


def test_mismatched_prompt_is_skipped_once(monkeypatch, caplog):
    import dataclasses

    from kilora import compute
    from kilora.routing_features import call_features, routing_features

    monkeypatch.setattr(compute, "_cross_missing_noted", set())
    layer = dataclasses.replace(_layer(), cross_weights=(0, 1, 0))
    x = torch.ones(1, 3, 2)
    with torch.no_grad(), call_features(torch.ones(1, 1, 3), None), routing_features():
        base = compute.run_layer(x, x, [_layer()], 1)
        first = compute.run_layer(x, x, [layer], 1)
        second = compute.run_layer(x, x, [layer], 1)
    assert torch.equal(base, first) and torch.equal(first, second)
    assert caplog.text.count("Routing cross token_prompt is unavailable") == 1


def test_joint_layer_prompt_and_settings():
    from test_sections import _logic_meta

    from kilora.comfyui.settings import KiLoRASettings, resolve_settings
    from kilora.family_spec import TokenLayout
    from kilora.reference_sections import validate
    from kilora.region import token_layouts
    from kilora.routing_features import call_features, current_score_features, layer_features

    layer = _layer()
    x = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [9.0, 9.0]]])
    with (
        token_layouts(lambda target: TokenLayout("image2d", text="before"), 2),
        call_features(torch.ones(1, 1, 5), None),
        layer_features(x, x, [layer], None, "target"),
    ):
        torch.testing.assert_close(current_score_features()[0].prompt[0, 2], torch.ones(2))
    (settings,) = KiLoRASettings().make("balanced", cross_token_prompt=0.4)
    assert settings["cross_token_prompt"] == 0.4
    for bad in (float("nan"), 17, True):
        with pytest.raises(ValueError, match="cross_token_prompt"):
            resolve_settings({"cross_token_prompt": bad})
    meta = _logic_meta()
    validate("logic", meta, {})
    meta["settings"]["cross_token_prompt"] = 0.4
    validate("logic", meta, {})
    meta["settings"]["cross_token_prompt"] = 17
    with pytest.raises(ValueError, match="cross_token_prompt"):
        validate("logic", meta, {})


def test_target_plan_copy_preserves_cross_weights():
    import dataclasses

    from kilora.bake import TargetPlan

    layer = dataclasses.replace(_layer(), cross_weights=(0.1, 0.2, 0.3))
    copied = TargetPlan(sets=[layer], name="test").to(torch.device("cpu"), torch.float32)
    assert copied.sets[0].cross_weights == layer.cross_weights


def test_cross_scores_best_valid_key_row():
    keys = torch.tensor([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    x = torch.tensor([[[1.0, 0.0]]])
    features = RoutingFeatures(x, x, x, None)
    result = cross_scores(
        torch.zeros(1, 1, 2), keys, features, {"token_prompt": 1}, torch.tensor([2, 1])
    )
    torch.testing.assert_close(result, torch.tensor([[[1.0, 0.0]]]))


@pytest.mark.parametrize("fast", [False, True])
def test_joint_tree_cross_uses_text_rows(fast, monkeypatch):
    import dataclasses

    from test_families import _model

    from kilora import compute, routing_crosses
    from kilora.bake import TargetPlan
    from kilora.family_spec import TokenLayout
    from kilora.region import token_layouts
    from kilora.routing_features import call_features, routing_features

    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    model = _model("flux")
    block = model.get_submodule("single_blocks.0")
    width = block.linear1.in_features
    block.add_module("routed", torch.nn.Linear(width, width))
    base = block.routed
    with torch.no_grad():
        base.weight.copy_(torch.eye(width))
        base.bias.zero_()
    keys = torch.zeros(2, 1, width)
    keys[0, 0, 0] = keys[1, 0, 1] = 1
    layer = dataclasses.replace(
        _layer(),
        router=torch.eye(width),
        keys_in=keys,
        down=torch.eye(width)[:2],
        up=torch.eye(width)[:2],
    )
    x = torch.zeros(1, 4, width)
    x[:, :, :2] = 1
    x[:, 0, 1] = 0
    seen, applied = [], []
    real_gate, real_cross = compute._gates_from_logits, routing_crosses.cross_scores

    def gate(*args, **kwargs):
        result = real_gate(*args, **kwargs)
        seen.append(result[0].detach().clone())
        return result

    def cross(*args, **kwargs):
        applied.append(1)
        return real_cross(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", gate)
    monkeypatch.setattr(routing_crosses, "cross_scores", cross)
    with (
        torch.no_grad(),
        token_layouts(lambda _: TokenLayout("image2d", text="before"), 1),
        call_features(torch.ones(1, 1, 3), None),
        routing_features(),
    ):
        TargetPlan(sets=[layer], name="joint").run(x, base(x))
        baseline = seen[-1][..., 0].mean()
        TargetPlan(sets=[dataclasses.replace(layer, cross_weights=(0, 0.4, 0))], name="joint").run(
            x, base(x)
        )
        changed = seen[-1][..., 0].mean()
    assert applied and changed > baseline


def test_missing_projection_disables_cross_even_at_matching_width(monkeypatch, caplog):
    import dataclasses

    from kilora import compute
    from kilora.family_spec import TokenLayout
    from kilora.region import token_layouts
    from kilora.routing_features import call_features, routing_features

    monkeypatch.setattr(compute, "_cross_missing_noted", set())
    layer = _layer()
    crossed = dataclasses.replace(layer, cross_weights=(0, 1, 0))
    x = torch.ones(1, 3, 2)
    with (
        torch.no_grad(),
        token_layouts(lambda _: TokenLayout("image2d")),
        call_features(torch.ones(1, 1, 2), None, lambda *_: None),
        routing_features(),
    ):
        base = compute.run_layer(x, x, [layer], 1)
        first = compute.run_layer(x, x, [crossed], 1)
        second = compute.run_layer(x, x, [crossed], 1)
    assert torch.equal(base, first) and torch.equal(first, second)
    assert caplog.text.count("Routing cross token_prompt is unavailable") == 1
