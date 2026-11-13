# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute
from kilora.routing_features import call_features, current_score_features, routing_features


@pytest.mark.parametrize("fast", [False, True])
def test_score_features_and_output(fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    sets = _sets(1, 8, 3, 2, 4, seed=7)
    x = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8) / 10
    y = x * 0.5
    region = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]])
    prompt = torch.randn(2, 3, 6)
    timestep = torch.randn(2, 4)
    observed = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        observed.append(current_score_features())
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", score)
    with torch.no_grad(), call_features(prompt, timestep):
        base = compute.run_layer(x, y, sets, 2, regions=[region])
        assert all(item is None for item in observed)
        observed.clear()
        with routing_features():
            result = compute.run_layer(x, y, sets, 2, regions=[region])
            replay = compute.run_layer(x, y, sets, 2, regions=[region])
    assert torch.equal(base, result)
    assert torch.equal(result, replay)
    assert observed and all(item is not None for item in observed)
    f = observed[0][0]
    assert f.token.shape == f.region.shape == (2, 5, 8)
    assert f.prompt.shape == (2, 5, 6)
    assert f.timestep_embedding.shape == (2, 5, 4)
    torch.testing.assert_close(f.region[:, 0], torch.stack((y[0, :2].mean(0), y[1, 2:4].mean(0))))
    torch.testing.assert_close(f.prompt[:, 0], prompt.mean(1))
    torch.testing.assert_close(f.timestep_embedding[:, 0], timestep)
    assert current_score_features() is None


def test_missing_values_are_explicit():
    sets = _sets(1, 8, 2, 2, 4, seed=1)
    x = torch.randn(2, 5, 8)
    seen = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        seen.append(current_score_features())
        return real(*args, **kwargs)

    from unittest.mock import patch

    with patch.object(compute, "_gates_from_logits", score), routing_features():
        compute.run_layer(x, x, sets, 1)
    assert seen[0][0].prompt is None
    assert seen[0][0].timestep_embedding is None


def test_tiny_flux_target_receives_features(monkeypatch):
    from test_families import _model

    from kilora.bake import TargetPlan

    model = _model("flux")
    base = model.get_submodule("double_blocks.0.img_attn.proj")
    plan = TargetPlan(sets=_sets(1, 8, 2, 2, 4, seed=2), name="flux")

    class Routed(torch.nn.Module):
        def forward(self, x):
            return plan.run(x, base(x))

    layer = Routed()
    x = torch.randn(2, 6, 8)
    seen = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        seen.append(current_score_features())
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", score)
    prompt = torch.randn(2, 3, 6)
    timestep = torch.randn(2, 4)
    with torch.no_grad(), call_features(prompt, timestep):
        base_output = layer(x)
        with routing_features():
            output = layer(x)
    assert torch.equal(base_output, output)
    f = next(item[0] for item in seen if item is not None)
    assert f.token.shape == f.region.shape == (2, 6, 8)
    assert f.prompt.shape == (2, 6, 6)
    assert f.timestep_embedding.shape == (2, 6, 4)


def test_row_selection_uses_matching_conditioning(monkeypatch):
    from kilora.call_values import row_axes

    sets = _sets(1, 8, 2, 2, 4, seed=3)
    x = torch.randn(3, 5, 8)
    prompt = torch.arange(18, dtype=torch.float32).reshape(3, 1, 6)
    seen = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        seen.append(current_score_features()[0].prompt[:, 0].clone())
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", score)
    with (
        torch.no_grad(),
        call_features(prompt, None),
        routing_features(),
        row_axes(torch.tensor([-1, 0, -1])),
    ):
        compute.run_layer(x, x, sets, 1)
    assert any(torch.equal(item, prompt[[0, 2], 0]) for item in seen)


@pytest.mark.parametrize("pooled", [True, False])
def test_comfy_wrapper_passes_available_call_values(monkeypatch, pooled):
    from types import SimpleNamespace

    from test_families import _model

    from kilora.bake import TargetPlan
    from kilora.comfyui.patching import _key_mask_by_step

    patcher = SimpleNamespace(model_options={})
    from kilora.comfyui.settings import KiLoRASettings
    from kilora.routing_features import timestep_features

    (settings,) = KiLoRASettings().make("balanced", routing_features=True)
    _key_mask_by_step(patcher, collect_routing_features=settings["routing_features"])
    wrapper = patcher.model_options["model_function_wrapper"]
    sets = _sets(1, 8, 2, 2, 4, seed=4)
    base_layer = _model("flux").get_submodule("double_blocks.0.img_attn.proj")
    plan = TargetPlan(sets=sets, name="flux")
    prompt = torch.randn(2, 3, 6)
    prompt_y = torch.randn(2, 7)
    x = torch.randn(2, 5, 8)
    seen = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        seen.append(current_score_features())
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", score)
    args = {
        "input": x,
        "timestep": torch.tensor([1.0, 1.0]),
        "c": {"c_crossattn": prompt, **({"y": prompt_y} if pooled else {})},
    }

    def apply_model(value, sigma, **kwargs):
        return plan.run(value, base_layer(value))

    with torch.no_grad():
        output = wrapper(apply_model, args)
        base = apply_model(x, args["timestep"])
    assert torch.equal(output, base)
    f = seen[0][0]
    assert f.token.shape == f.region.shape == (2, 5, 8)
    assert f.prompt.shape == (2, 5, 7 if pooled else 6)
    assert f.timestep_embedding.shape == (2, 5, 256)
    torch.testing.assert_close(f.prompt[:, 0], prompt_y if pooled else prompt.mean(1))
    torch.testing.assert_close(f.timestep_embedding[:, 0], timestep_features(args["timestep"]))


def test_spatial_inputs_and_missing_input_note(monkeypatch, caplog):
    sets = _sets(1, 8, 2, 2, 4, seed=5)
    x = torch.randn(2, 3, 4, 8)
    seen = []
    real = compute._gates_from_logits

    def score(*args, **kwargs):
        seen.append(current_score_features())
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", score)
    monkeypatch.setattr(compute, "_missing_feature_input_noted", False)
    with torch.no_grad():
        base = compute.run_layer(x, x, sets, 1)
        with routing_features():
            output = compute.run_layer(x, x, sets, 1)
            compute.run_layer(None, x, sets, 1)
            compute.run_layer(None, x, sets, 1)
    assert torch.equal(base, output)
    features = next(item[0] for item in seen if item is not None)
    assert features.token.shape == features.region.shape == (2, 12, 8)
    assert caplog.text.count("Routing features are unavailable") == 1


def test_anima_wrapper_projects_context_per_block(monkeypatch):
    from test_multi_import import ToyAnima

    from kilora.family_spec import TokenLayout
    from kilora.region import token_layouts
    from kilora.routing_features import layer_features, prompt_projection

    model = ToyAnima()
    projection = prompt_projection(model, "anima")
    layer = _sets(1, 8, 2, 2, 4, seed=3)
    module = model.blocks[0].cross_attn.k_proj
    prompt = torch.randn(1, 3, module.in_features)
    x = torch.randn(1, 4, module.out_features)
    with (
        call_features(prompt, None, projection),
        token_layouts(lambda _: TokenLayout("video3d")),
        layer_features(x, x, layer, None, "transformer_blocks.0.attn1.to_q"),
    ):
        actual = current_score_features()[0].prompt[:, 0]
    torch.testing.assert_close(actual, module(prompt.mean(1)))
    assert projection("transformer_blocks.0.ff.net.2", prompt, torch.randn(1, 4, 99)) is None
