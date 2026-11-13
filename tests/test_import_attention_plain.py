# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import import_lora
from kilora.adapter import IMPORTED_DEPTH_SCHEDULE, imported_settings
from kilora.bake import ScopedSet, plan_targets
from kilora.compute import run_layer
from kilora.depth_schedule import compile_schedule
from kilora.route_freeze import freezing
from kilora.testing import TinyDenoiser

ATTN = "layers.0.attention.to_q"
MLP = "layers.0.feed_forward.w1"


def make_plan(target=ATTN, depth=3, schedule=None):
    model = TinyDenoiser(seed=0).eval()
    g = torch.Generator().manual_seed(31)
    module = model.get_submodule(target)
    down = torch.randn(3, module.in_features, generator=g)
    up = torch.randn(module.out_features, 3, generator=g)
    state = {f"{target}.lora_A.weight": down, f"{target}.lora_B.weight": up}
    adapter = import_lora(state, model, weight=0.7, router_dim=8).adapter_set
    settings = imported_settings()
    scoped = ScopedSet(
        adapter, route_freeze=settings.route_freeze, import_score=settings.import_score
    )
    return model, down, up, plan_targets([scoped], depth, schedule)[target]


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("scheduled", [False, True])
def test_import_attention_uses_the_same_depth_and_freeze(depth, scheduled, monkeypatch):
    from dataclasses import replace

    from kilora import compute

    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    schedule = compile_schedule(IMPORTED_DEPTH_SCHEDULE) if scheduled else None
    model, down, _, plan = make_plan(depth=depth, schedule=schedule)
    sets = [replace(s, route_freeze=0.4) for s in plan.sets]
    x = torch.randn(2, 5, down.shape[1], generator=torch.Generator().manual_seed(4))
    base = model.get_submodule(ATTN)(x)
    caches = [{}, {}]
    for t in (0.0, 0.5, 0.9):
        with torch.no_grad(), compute.eager(), freezing(caches[0]):
            attention = run_layer(x, base, sets, depth, target=ATTN, schedule=schedule, t=t)
        with torch.no_grad(), compute.eager(), freezing(caches[1]):
            feed_forward = run_layer(x, base, sets, depth, target=MLP, schedule=schedule, t=t)
        assert torch.equal(attention, feed_forward)
    assert caches[0] and caches[1]


def test_mlp_unchanged(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    model, down, _, plan = make_plan(target=MLP)
    x = torch.randn(2, 4, down.shape[1], generator=torch.Generator().manual_seed(6))
    base = model.get_submodule(MLP)(x)
    with torch.no_grad():
        actual = run_layer(x, base, plan.sets, 3, target=MLP)
    baseline = torch.tensor(
        [
            8.90075874,
            0.21966764,
            -5.75855064,
            -6.89744473,
            -4.11472416,
            -2.69580412,
            -5.36809826,
            27.80084038,
            11.18918133,
            -16.84566116,
            12.65908527,
            -2.41103506,
        ]
    )
    torch.testing.assert_close(actual.flatten()[:12], baseline, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("target", [ATTN, MLP])
def test_import_dispatch_uses_the_bounded_chain(target, monkeypatch):
    from kilora import compute

    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    model, down, _, plan = make_plan(target=target)
    x = torch.randn(2, 4, down.shape[1], generator=torch.Generator().manual_seed(6))
    base = model.get_submodule(target)(x)
    with torch.no_grad(), compute.eager(), compute.dropout_key(7, 3):
        actual = run_layer(x, base, plan.sets, 3, target=target)
        expected = compute._run_input(x, base, plan.sets, [0], 3, [None], target)
    assert torch.equal(actual, expected)
