# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from diffusers import ZImageTransformer2DModel

from kilora import compose, get_preset, import_lora
from kilora.diffusers import attach_to_pipeline, compose_into_pipeline, denoiser_of

TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]


def tiny_transformer(seed=0):
    torch.manual_seed(seed)
    m = ZImageTransformer2DModel(
        in_channels=4,
        dim=32,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=16,
        axes_dims=[4, 6, 6],
        axes_lens=[64, 32, 32],
    )
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return m.eval()


def inputs():
    gen = torch.Generator().manual_seed(5)
    x = [torch.randn(4, 1, 8, 8, generator=gen)]
    cap = [torch.randn(5, 16, generator=gen)]
    return x, torch.tensor([0.4]), cap


def run(model):
    x, t, cap = inputs()
    with torch.no_grad():
        return model(x, t, cap, return_dict=False)[0][0]


def random_lora(model, seed, rank=4):
    gen = torch.Generator().manual_seed(seed)
    sd = {}
    for name, module in model.named_modules():
        if not name.startswith("layers.") or not any(name.endswith("." + t) for t in TARGETS):
            continue
        sd[f"transformer.{name}.lora_A.weight"] = (
            torch.randn(rank, module.in_features, generator=gen) * 0.2
        )
        sd[f"transformer.{name}.lora_B.weight"] = (
            torch.randn(module.out_features, rank, generator=gen) * 0.2
        )
    return sd


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.lora_parity
def test_one_lora_matches_diffusers_step():
    sd = random_lora(tiny_transformer(), seed=1)
    reference = tiny_transformer()
    reference.load_lora_adapter(dict(sd), prefix="transformer", adapter_name="one")
    reference.set_adapters(["one"], weights=[0.8])
    want = run(reference)

    model = tiny_transformer()
    base = run(model)
    compose(model, [import_lora(sd, model, weight=0.8).adapter_set], mix="stack")
    got = run(model)
    assert rel(want, base) > 1e-2
    assert rel(got, want) < 1e-4


@pytest.mark.lora_parity
def test_two_loras_match_diffusers_multi_adapter():
    m0 = tiny_transformer()
    sd1, sd2 = random_lora(m0, seed=1), random_lora(m0, seed=2, rank=2)
    reference = tiny_transformer()
    reference.load_lora_adapter(dict(sd1), prefix="transformer", adapter_name="a")
    reference.load_lora_adapter(dict(sd2), prefix="transformer", adapter_name="b")
    reference.set_adapters(["a", "b"], weights=[0.6, -0.9])
    want = run(reference)

    model = tiny_transformer()
    a = import_lora(sd1, model, name="a").adapter_set
    b = import_lora(sd2, model, name="b").adapter_set
    compose(model, [a, b], strengths=[0.6, -0.9], mix="stack")
    assert rel(run(model), want) < 1e-4


def test_preset_targets_attention_layers_and_pipeline_helpers():
    model = tiny_transformer()
    pipe = types.SimpleNamespace(transformer=model)
    assert denoiser_of(pipe) is model
    base = run(model)
    preset = get_preset("z-image")
    adapter = attach_to_pipeline(pipe, preset.config(num_experts=4, router_dim=8))
    assert adapter.targets == [
        f"layers.{i}.attention.{p}" for i in range(2) for p in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    assert torch.equal(run(model), base)
    with torch.no_grad():
        for site in adapter.sites:
            site.tensors["s0_up"].normal_()
    changed = run(model)
    assert not torch.allclose(changed, base)
    sets = adapter.to_sets()
    adapter.remove()
    again = compose_into_pipeline(pipe, sets, depth=1)
    torch.testing.assert_close(run(model), changed)
    again.remove()
    assert torch.equal(run(model), base)


def test_training_step_on_tiny_transformer():
    model = tiny_transformer().train()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = attach_to_pipeline(
        types.SimpleNamespace(transformer=model),
        get_preset("z-image").config(num_experts=4, router_dim=8, train_depth=(1, 3)),
    )
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-2)
    x, t, cap = inputs()
    target = torch.randn_like(x[0])
    losses = []
    for step in range(5):
        adapter.start_step(step)
        out = model(x, t, cap, return_dict=False)[0][0]
        loss = adapter.wrap_loss((out - target).pow(2).mean())
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
