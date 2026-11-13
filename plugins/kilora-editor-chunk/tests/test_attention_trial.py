# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import torch
from kilora_editor_chunk.plugin import RoutedPerturbation, RoutedWrite

from kilora import KiLoRAConfig, attach
from kilora.bake import ModulePlan, ScopedSet, plan_targets
from kilora.recording import Perturbation, perturbing
from kilora.testing import TinyDenoiser

ATTN = "layers.0.attention.to_q"


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


def test_attention_trial_replays_the_changed_live_set():
    model, target, x = state_plan(ATTN)
    layer = target.sets[0]
    target.sets[0] = replace(layer, key="live")
    layer = target.sets[0]
    base = model.get_submodule(ATTN)(x)
    gen = torch.Generator().manual_seed(10)
    v = torch.randn(2, layer.down.shape[0], generator=gen)
    du = torch.randn(layer.up.shape[1], 2, generator=gen) * 0.1
    delta_up = (du @ v).t()
    p = RoutedPerturbation(
        Perturbation({ATTN: v @ layer.down}, {ATTN: du}),
        {},
        trial_up={ATTN: delta_up},
        set_key="live",
    )
    module = ModulePlan([(0, -1, target)])
    with torch.no_grad():
        before = module.run(x, base)
        with perturbing(p):
            measured = module.run(x, base)
        changed = replace(layer, up=layer.up + delta_up)
        expected = target.run(x, base, trial=lambda _name, _sets: [changed])
        assert not torch.equal(before, expected)
        torch.testing.assert_close(measured - before, expected - before)
        guard = RoutedWrite({ATTN: layer.down}, {ATTN: delta_up.t()}, {}, {}, set_key="live")
        with perturbing(guard):
            guarded = module.run(x, base)
        torch.testing.assert_close(guarded, expected)


def test_input_trial_keeps_additive_measurement_without_replay_opt_in():
    model, target, x = state_plan(ATTN)
    layer = replace(target.sets[0], key="live", form="input", import_score="keys")
    target.sets[0] = layer
    base = model.get_submodule(ATTN)(x)
    gen = torch.Generator().manual_seed(10)
    v = torch.randn(2, layer.down.shape[0], generator=gen)
    du = torch.randn(layer.up.shape[1], 2, generator=gen) * 0.1
    delta_up = (du @ v).t()
    trial = RoutedPerturbation(
        Perturbation({ATTN: v @ layer.down}, {ATTN: du}),
        {},
        trial_up={ATTN: delta_up},
        set_key="live",
    )
    module = ModulePlan([(0, -1, target)])
    with torch.no_grad():
        before = module.run(x, base)
        with perturbing(trial):
            measured = module.run(x, base)
        torch.testing.assert_close(measured, before + trial.delta(ATTN, base))
        trial.replay_all = True
        with perturbing(trial):
            replayed = module.run(x, base)
        changed = replace(layer, up=layer.up + delta_up)
        expected = target.run(x, base, trial=lambda _name, _sets: [changed])
        torch.testing.assert_close(replayed, expected)
