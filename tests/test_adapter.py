# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from conftest import randomize_up, reference_layer

from kilora import KiLoRAConfig, attach, compose
from kilora.schedule import draw_depth


def make(model, **kw):
    kw.setdefault("target_modules", ["layers.*.feed_forward.w2"])
    kw.setdefault("num_experts", 4)
    kw.setdefault("routes", 2)
    kw.setdefault("rank", 1)
    kw.setdefault("router_dim", 8)
    kw.setdefault("output_gate", "none")
    return attach(model, KiLoRAConfig(**kw))


def test_new_adapter_keeps_base_output(model, x):
    base = model(x)
    adapter = make(model)
    assert torch.equal(model(x), base)
    randomize_up(adapter)
    assert not torch.allclose(model(x), base)
    adapter.remove()
    assert torch.equal(model(x), base)


def test_depth_zero_is_base_model(model, x):
    base = model(x)
    adapter = make(model)
    randomize_up(adapter)
    with adapter.depth(0):
        assert torch.equal(model(x), base)
    assert adapter.current_depth == 1


def ref_sets(adapter, target_index=0, strengths=None):
    site = adapter.sites[target_index]
    out = []
    for n, i in enumerate(site.keyed):
        t = site.tensors
        width = t[f"s{i}_up"].shape[-1]
        out.append(
            {
                "router": adapter.routers[f"s{i}_{width}"],
                "keys": t[f"s{i}_keys_in"],
                "keys_out": t[f"s{i}_keys_out"],
                "tau": adapter.log_taus[f"s{i}"].exp(),
                "down": t[f"s{i}_down"],
                "up": t[f"s{i}_up"],
                "strength": 1.0 if strengths is None else strengths[n],
                "mode": adapter._set_info[i]["aggregation"],
                "log_prior": adapter.priors[i].log(),
                "score": "derived"
                if adapter._set_info[i]["form"] == "state"
                else adapter.settings.import_score,
            }
        )
    return out


@pytest.mark.parametrize("mode", ["accumulate", "replace"])
def test_iteration_matches_reference(model, mode):
    adapter = make(model, depth=3, aggregation=mode)
    randomize_up(adapter)
    module = model.layers[0].feed_forward.w2
    inp = torch.randn(3, 32, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        with adapter.depth(0):
            y0 = module(inp)
        out = module(inp)
        ref = reference_layer(inp, y0, ref_sets(adapter), 3, target=adapter.targets[0])
    torch.testing.assert_close(out, ref, rtol=3e-5, atol=5e-6)


@pytest.mark.parametrize("mode", ["accumulate", "replace"])
def test_depth_one_is_the_lora_form_on_the_state(model, mode):
    adapter = make(model, depth=1, aggregation=mode, num_experts=3, routing_select="top1")
    randomize_up(adapter)
    t = adapter.sites[0].tensors
    with torch.no_grad():
        t["s0_down"].copy_(t["s0_down"][:1].expand_as(t["s0_down"]))
        t["s0_up"].copy_(t["s0_up"][:1].expand_as(t["s0_up"]))
    module = model.layers[0].feed_forward.w2
    inp = torch.randn(5, 32, generator=torch.Generator().manual_seed(2))
    a = t["s0_down"][0].reshape(-1, 16)
    b = t["s0_up"][0].reshape(-1, 16)
    with torch.no_grad():
        with adapter.depth(0):
            y0 = module(inp)
        torch.testing.assert_close(module(inp), y0 + (y0 @ a.t()) @ b, rtol=1e-5, atol=1e-6)
        adapter.set_depth(3)
        three = module(inp)
        ref = reference_layer(
            inp, y0, ref_sets(adapter), 3, target=adapter.targets[0], routing_select="top1"
        )
    torch.testing.assert_close(three, ref, rtol=1e-4, atol=1e-5)


def test_derived_routing_scores_follow_the_read_state(model):
    adapter = make(model, num_experts=8, depth=1)
    randomize_up(adapter, scale=2.0)
    module = model.layers[0].feed_forward.w2
    layer = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))[0]
    from kilora.compute import derived_logits

    inp = torch.randn(64, 32, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        with adapter.depth(0):
            h0 = module(inp)
        logits0 = derived_logits(layer, h0)
        h1 = module(inp)
        logits1 = derived_logits(layer, h1)
    assert (logits0.argmax(-1) != logits1.argmax(-1)).any()


def test_strength_is_signed_and_linear_at_depth_one(model, x):
    base = model(x)
    adapter = make(model, depth=1)
    randomize_up(adapter)
    module = model.layers[0].feed_forward.w2
    inp = torch.randn(5, 32, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        with adapter.depth(0):
            plain = module(inp)
        pos = module(inp) - plain
        adapter.set_strengths([-1.0])
        neg = module(inp) - plain
        adapter.set_strengths({"main": 0.0})
        zero = module(inp) - plain
    torch.testing.assert_close(neg, -pos)
    assert torch.count_nonzero(zero) == 0
    assert torch.equal(model(x), base)


def test_bad_controls_are_refused(model):
    adapter = make(model)
    with pytest.raises(ValueError):
        adapter.set_strengths([float("nan")])
    with pytest.raises(ValueError):
        adapter.set_priors([-1.0])
    with pytest.raises(ValueError):
        adapter.set_priors([0.0])
    assert float(adapter.priors[0]) == 1.0
    with pytest.raises(KeyError):
        adapter.set_strengths({"missing": 1.0})
    with pytest.raises(ValueError):
        adapter.set_depth(-1)


def test_compose_two_sets_shared_softmax(model, x):
    a = make(model, seed=1)
    randomize_up(a, seed=1)
    set_a = a.to_sets()[0]
    a.remove()
    b = make(model, seed=2)
    randomize_up(b, seed=2)
    set_b = b.to_sets()[0]
    b.remove()
    set_a.name, set_b.name = "a", "b"
    combined = compose(model, [set_a, set_b], strengths=[1.0, -1.0], depth=2)
    out = model(x)
    module = model.layers[0].feed_forward.w2
    inp = torch.randn(4, 32, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        with combined.depth(0):
            y0 = module(inp)
        ref = reference_layer(
            inp,
            y0,
            ref_sets(combined, strengths=[1.0, -1.0]),
            2,
            shared=True,
            target=combined.targets[0],
        )
        torch.testing.assert_close(module(inp), ref, rtol=1e-5, atol=1e-5)
    combined.set_priors({"b": 0.0})
    combined.set_strengths({"a": 0.0})
    torch.testing.assert_close(model(x), combined_base(model, combined, x))
    assert out.shape == x.shape


def combined_base(model, adapter, x):
    with adapter.depth(0):
        return model(x)


def test_compose_per_set_softmax_runs_each_set(model, x):
    a = make(model, seed=1)
    randomize_up(a, seed=1)
    set_a = a.to_sets()[0]
    a.remove()
    set_a.name = "a"
    set_b = type(set_a)(**{**set_a.__dict__, "name": "b"})
    base = model(x)
    alone = compose(model, [set_a], softmax_scope="per_set")
    want = model(x)
    alone.remove()
    combined = compose(model, [set_a, set_b], strengths=[1.0, 0.0], softmax_scope="per_set")
    torch.testing.assert_close(model(x), want, rtol=1e-5, atol=1e-5)
    combined.set_strengths([1.0, 1.0])
    assert not torch.allclose(model(x), base) and not torch.allclose(model(x), want)


def test_sets_with_different_targets_and_shapes_compose(model, x):
    a = make(model, router_seed=0).to_sets()[0]
    b = make(model, router_seed=1, rank=2, target_modules=["layers.0.attention.to_q"])
    randomize_up(b)
    b = b.to_sets()[0]
    b.name = "other"
    adapter = compose(model, [a, b])
    assert adapter.targets[-1] == "layers.0.attention.to_q"
    assert model(x).shape == x.shape
    c = make(model, router_dim=4).to_sets()[0]
    c.name = "c"
    c.routers = {width: torch.zeros(4, width) for width in c.routers}
    for expert in c.experts.values():
        expert.keys_in = expert.keys_in[..., :4]
        expert.keys_out = expert.keys_out[..., :4]
        expert.rows_in = expert.rows_in.clamp(max=4)
        expert.rows_out = expert.rows_out.clamp(max=4)
    adapter.remove()
    shared = compose(model, [a, c], softmax_scope="shared")
    assert shared.settings.softmax_scope == "shared"
    shared.remove()
    auto = compose(model, [a, c])
    assert auto.settings.softmax_scope == "shared"
    auto.remove()


def test_random_depth_draw_is_stateless():
    draws = [draw_depth(7, step, 1, 4) for step in range(200)]
    assert draws == [draw_depth(7, step, 1, 4) for step in range(200)]
    assert set(draws) == {1, 2, 3, 4}
    assert draws != [draw_depth(8, step, 1, 4) for step in range(200)]


def test_start_step_and_wrap_loss_at_depth_zero(model, x):
    model.train()
    adapter = make(model, train_depth=(0, 0))
    assert adapter.start_step(0) == 0
    loss = adapter.wrap_loss(model(x).pow(2).mean())
    loss.backward()
    grads = [p.grad for p in adapter.trainable_parameters()]
    assert all(g is not None and torch.count_nonzero(g) == 0 for g in grads)


def test_training_uses_drawn_depth_and_gradients_reach_experts(model, x):
    model.train()
    adapter = make(model, train_depth=(2, 2), depth=5)
    randomize_up(adapter)
    assert adapter.start_step(3) == 2
    loss = adapter.wrap_loss(model(x).pow(2).mean())
    loss.backward()
    t = adapter.sites[0].tensors
    assert t["s0_keys_in"].grad is None
    assert t["s0_keys_out"].grad is None
    assert t["s0_down"].grad is not None and t["s0_down"].grad.abs().sum() > 0
    assert t["s0_up"].grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        adapter.set_depth(2)
        two = model(x)
        adapter.set_depth(5)
        five = model(x)
    assert not torch.allclose(two, five)


def test_last_step_training_matches_detached_reference(model):
    adapter = make(model, depth=3, train_depth=(3, 3), last_step_training=True)
    randomize_up(adapter)
    module = model.layers[0].feed_forward.w2
    module.train()
    adapter.start_step(0)
    inp = torch.randn(4, 32, generator=torch.Generator().manual_seed(8))
    out = module(inp)
    out.sum().backward()
    t = adapter.sites[0].tensors
    got = t["s0_up"].grad.clone()
    for p in adapter.parameters():
        p.grad = None
    with torch.no_grad(), adapter.depth(0):
        module.eval()
        y0 = module(inp)
        module.train()
    ref = reference_layer(
        inp, y0, ref_sets(adapter), 3, grad_last_only=True, target=adapter.targets[0]
    )
    ref.sum().backward()
    torch.testing.assert_close(got, t["s0_up"].grad)
    torch.testing.assert_close(out.detach(), ref.detach(), rtol=1e-5, atol=1e-5)


def test_routing_noise_only_in_training(model, x):
    adapter = make(model, routing_noise=5.0, num_experts=8, train_depth=None)
    randomize_up(adapter)
    model.eval()
    assert torch.equal(model(x), model(x))


def test_no_matching_module_is_an_error(model):
    from kilora import KiLoRAError

    with pytest.raises(KiLoRAError, match="no nn.Linear"):
        make(model, target_modules=["nothing.*"])


def test_control_buffers_follow_the_model_device():
    from kilora import import_lora
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser(seed=0).to("meta")
    sd = {
        "layers.0.attention.to_q.lora_A.weight": torch.zeros(2, 16),
        "layers.0.attention.to_q.lora_B.weight": torch.zeros(16, 2),
    }
    adapter = compose(model, [import_lora(sd, model).adapter_set])
    assert adapter.strengths.device.type == "meta" and adapter.priors.device.type == "meta"
