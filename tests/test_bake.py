# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from conftest import TARGET, randomize_up

from kilora import KiLoRAConfig, attach, compose, import_lora
from kilora.bake import ScopedSet, bake, is_baked
from kilora.bindings import get_binding


def keyed(model, name, seed, **kw):
    kw.setdefault("router_seed", 0)
    target_modules = kw.pop("target_modules", [TARGET])
    a = attach(
        model,
        KiLoRAConfig(
            target_modules=target_modules, num_experts=4, routes=2, router_dim=8, seed=seed, **kw
        ),
        name=name,
    )
    randomize_up(a, seed=seed)
    s = a.to_sets()[0]
    a.remove()
    return s


def imported(model, name, seed):
    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for n in ("layers.0.attention.to_q", "layers.1.attention.to_out.0"):
        m = model.get_submodule(n)
        sd[f"{n}.lora_A.weight"] = torch.randn(2, m.in_features, generator=gen) * 0.3
        sd[f"{n}.lora_B.weight"] = torch.randn(m.out_features, 2, generator=gen) * 0.3
    return import_lora(sd, model, name=name, router_dim=8).adapter_set


@pytest.mark.parametrize("scope", ["shared", "per_set"])
@pytest.mark.parametrize("score", [None, "keys", "derived"], ids=["default", "keys", "derived"])
@pytest.mark.parametrize("balance", ["0", "1"])
@pytest.mark.parametrize("mode", ["keys", "full"])
def test_baked_output_matches_hooked_adapter(model, x, scope, score, balance, mode, monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.setenv("KILORA_FILE_BALANCE", balance)
    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    a, b = keyed(model, "a", 1), keyed(model, "b", 2)
    a.prior, b.prior = 1.0, 2.0
    imp = imported(model, "imp", 3)
    hooked = compose(model, [a, b, imp], strengths=[1.0, -0.7, 0.5], depth=3, softmax_scope=scope)
    if score is not None:
        hooked.settings.import_score = score
    want = model(x)
    sets = hooked.to_sets()
    hooked.remove()
    base = model(x)
    scoped = [ScopedSet(s, scope) for s in sets]
    if score is not None:
        scoped = [dataclasses.replace(s, import_score=score) for s in scoped]
    assert all(s.import_score == hooked.settings.import_score for s in scoped)
    baked = bake(model, scoped, depth=3)
    assert is_baked(model.layers[0].attention.to_out[0])
    torch.testing.assert_close(model(x), want, rtol=1e-5, atol=1e-5)
    baked.remove()
    assert not is_baked(model.layers[0].attention.to_out[0])
    assert torch.equal(model(x), base)


def test_baking_leaves_the_base_module_unchanged(model, x):
    module = model.layers[0].attention.to_out[0]
    weight = module.weight
    baked = bake(model, [keyed(model, "a", 1)], depth=2)
    copy = model.layers[0].attention.to_out[0]
    assert copy is not module and copy.weight is weight
    assert not is_baked(module)
    inp = torch.randn(3, 16)
    assert not torch.allclose(copy(inp), module(inp))
    baked.remove()


def test_sets_at_a_layer_add_their_own_changes(model, x):
    target = ["layers.*.feed_forward.w2"]
    a = keyed(model, "a", 1, router_seed=0, target_modules=target)
    b = keyed(model, "b", 2, router_seed=5, target_modules=target)
    base = model(x)

    def target():
        return model.layers[0].feed_forward.w2

    inp = torch.randn(4, 32, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        both = bake(
            model,
            [
                ScopedSet(a, "per_set"),
                ScopedSet(b, "per_set"),
            ],
            depth=1,
        )
        got = target()(inp)
        both.remove()
        only_a = bake(model, [ScopedSet(a)], depth=1)
        h_a = target()(inp)
        only_a.remove()
        off = dataclasses.replace(a, strength=0.0)
        only_b = bake(
            model,
            [
                ScopedSet(off, "per_set"),
                ScopedSet(b, "per_set"),
            ],
            depth=1,
        )
        h_b = target()(inp)
        only_b.remove()
        y0 = target()(inp)
    torch.testing.assert_close(got, y0 + (h_a - y0) + (h_b - y0), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(got, h_a)
    assert torch.equal(model(x), base)


def test_z_image_binding_names():
    b = get_binding("z-image")
    assert b.hook_point("layers.12.attention.to_out.0") == "base::block.12.attn.out"
    assert b.hook_point("noise_refiner.1.attention.to_k") == "base::noise_block.1.attn.k"
    place = b.to_comfy("layers.3.attention.to_v")
    assert place.path == "diffusion_model.layers.3.attention.qkv" and place.part == (2, 3)
    assert (
        b.to_comfy("layers.3.attention.to_out.0").path == "diffusion_model.layers.3.attention.out"
    )
    assert b.from_comfy("layers.3.attention.qkv") == [
        ("layers.3.attention.to_q", (0, 3)),
        ("layers.3.attention.to_k", (1, 3)),
        ("layers.3.attention.to_v", (2, 3)),
    ]
    assert b.matches_comfy(["layers.0.attention.qkv", "layers.0.feed_forward.w3"])
    assert not b.matches_comfy(["layers.0.attention.to_q"])


def test_baked_copy_owns_weight_table(model):
    a = keyed(model, "a", 1)
    baked = bake(model, [a], depth=1)
    copy = model.layers[0].attention.to_out[0]
    base = copy.__dict__["_kilora_base"]
    base_weight = base.weight
    new_weight = torch.nn.Parameter(copy.weight.detach().clone() * 2)
    copy._parameters["weight"] = new_weight
    assert base.weight is base_weight
    assert copy.weight is new_weight
    copy.comfy_cast_weights = True
    assert not hasattr(base, "comfy_cast_weights")
    baked.remove()


def test_bake_prepares_host_norm_and_base_transition(model, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "base")
    imp = imported(model, "imp", 3)
    target = "layers.0.attention.to_q"
    norm = model.get_submodule(target).weight.detach().float().norm()
    baked = bake(model, [imp], depth=2)
    try:
        plan = model.get_submodule(target).__dict__["_kilora_plan"]
        layer_set = plan.parts[0][2].sets[0]
        assert layer_set.host_weight_norm == pytest.approx(norm.item())
        assert layer_set.share_transition_base is not None
        moved = plan.parts[0][2].to(torch.device("cpu"), torch.float32).sets[0]
        assert moved.host_weight_norm == layer_set.host_weight_norm
        torch.testing.assert_close(moved.share_transition_base, layer_set.share_transition_base)
    finally:
        baked.remove()


def test_hook_prepares_host_norm_and_base_transition(model, x, monkeypatch):
    from kilora import adapter

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "base")
    imp = imported(model, "imp", 3)
    seen = []

    def record(x, h, sets, depth, **kwargs):
        seen.extend(sets)
        return h

    monkeypatch.setattr(adapter, "run_layer", record)
    hooked = compose(model, [imp], depth=2)
    try:
        model(x)
        assert seen
        assert all(s.host_weight_norm and s.share_transition_base is not None for s in seen)
    finally:
        hooked.remove()
