# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from test_import_attention_plain import ATTN, MLP

from kilora import KiLoRAConfig, attach, compute
from kilora.bake import ScopedSet, plan_targets
from kilora.testing import TinyDenoiser


def state_plan(target):
    model = TinyDenoiser(seed=0).eval()
    adapter = attach(
        model,
        KiLoRAConfig(target_modules=[target], num_experts=3, routes=1, router_dim=8, seed=12),
        name="state",
    )
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for site in adapter.sites:
            up = site.tensors.get("s0_up")
            if up is not None:
                up.copy_(torch.randn(up.shape, generator=gen) * 0.3)
    state = adapter.to_sets()[0]
    adapter.remove()
    plan = plan_targets([ScopedSet(state, route_freeze=0.0)], 3)[target]
    x = torch.randn(
        2, 5, model.get_submodule(target).in_features, generator=torch.Generator().manual_seed(7)
    )
    return model, plan, x


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_state_attention_uses_the_same_depth_chain(depth, iteration, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    model, plan, x = state_plan(ATTN)
    sets = [replace(s, iteration=iteration, route_freeze=None) for s in plan.sets]
    base = model.get_submodule(ATTN)(x)
    with torch.no_grad(), compute.eager():
        actual = compute.run_layer(x, base, sets, depth, target=ATTN)
        expected = compute.run_layer(x, base, sets, depth, target=MLP)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("iteration", ["accumulate", "path", "mix"])
def test_training_attention_uses_drawn_depth_and_iteration(iteration, monkeypatch):
    import importlib

    adapter_module = importlib.import_module("kilora.adapter")
    model = TinyDenoiser(seed=0).train()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=[ATTN],
            num_experts=3,
            routes=1,
            router_dim=8,
            train_depth=(0, 0),
            train_iteration=iteration,
            output_gate="none",
        ),
    )
    with torch.no_grad():
        adapter.sites[0].tensors["s0_up"].normal_()
    layer = model.get_submodule(ATTN)
    x = torch.randn(2, 5, layer.in_features)
    calls = []
    original = adapter_module.run_layer

    def record(x, h, sets, depth, **kwargs):
        calls.append((depth, sets[0].iteration))
        return original(x, h, sets, depth, **kwargs)

    monkeypatch.setattr(adapter_module, "run_layer", record)
    adapter.start_step(0)
    base = torch.nn.functional.linear(x, layer.weight, layer.bias)
    zero_draw = layer(x)
    assert torch.equal(zero_draw, base)
    assert not calls
    adapter._step_depth = 3
    three_draw = layer(x)
    modes = ["path", "accumulate"] if iteration == "mix" else [iteration]
    assert calls == [(3, mode) for mode in modes]
    assert not torch.equal(three_draw, base)
    three_draw.sum().backward()
    assert adapter.sites[0].tensors["s0_up"].grad is not None
    adapter.remove()


@pytest.mark.parametrize("kind", ["import", "trained", "live"])
@pytest.mark.parametrize("depth", [1, 3])
def test_attention_and_feed_forward_match_in_adapter_and_bake(kind, depth, monkeypatch):
    import copy

    from kilora import compose, import_lora
    from kilora.bake import bake
    from kilora.live_sets import LivePreset, make_live_set

    monkeypatch.setenv("KILORA_MASKS", "expect")
    model = TinyDenoiser(seed=0).eval()
    model.layers[0].feed_forward.w1 = copy.deepcopy(model.layers[0].attention.to_q)
    module = model.get_submodule(ATTN)
    gen = torch.Generator().manual_seed(71)
    if kind == "import":
        source = import_lora(
            {
                f"{ATTN}.lora_A.weight": torch.randn(2, module.in_features, generator=gen),
                f"{ATTN}.lora_B.weight": torch.randn(module.out_features, 2, generator=gen),
            },
            model,
            router_dim=8,
        ).adapter_set
    else:
        adapter = attach(
            model,
            KiLoRAConfig(target_modules=[ATTN], num_experts=3, routes=1, router_dim=8),
        )
        with torch.no_grad():
            adapter.sites[0].tensors["s0_up"].normal_(generator=gen)
        source = adapter.to_sets()[0]
        adapter.remove()
        if kind == "live":
            source, _ = make_live_set(source, LivePreset("test", 3, 1, ()), "test")
    expert = source.experts[ATTN]
    source = replace(source, experts={ATTN: expert, MLP: copy.deepcopy(expert)})
    x = torch.randn(2, 5, module.in_features, generator=gen)
    hooked = compose(model, [source], depth=depth)
    with torch.no_grad(), compute.eager():
        attention = model.get_submodule(ATTN)(x)
        ordinary = model.get_submodule(MLP)(x)
    assert torch.equal(attention, ordinary)
    settings = hooked.settings
    hooked.remove()
    baked = bake(
        model,
        [
            ScopedSet(
                source,
                import_score=settings.import_score,
                gate_mode=settings.gate_mode,
                route_freeze=settings.route_freeze,
            )
        ],
        depth=depth,
    )
    with torch.no_grad(), compute.eager():
        attention_baked = model.get_submodule(ATTN)(x)
        ordinary_baked = model.get_submodule(MLP)(x)
    assert torch.equal(attention_baked, ordinary_baked)
    torch.testing.assert_close(attention_baked, attention, rtol=1e-5, atol=1e-5)
    baked.remove()
