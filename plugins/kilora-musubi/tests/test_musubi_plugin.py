# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import importlib
import types

import pytest
import torch
from safetensors.torch import load_file

from kilora import compose, load_mod
from kilora.plugin_support import UnsupportedTrainerVersion
from kilora.testing import TinyDenoiser


@pytest.fixture
def musubi_version(monkeypatch):
    import kilora_musubi.network as network

    monkeypatch.setattr(network, "installed_version", lambda: "0.3.6")
    return network


def run_trainer(model, out, network_args, steps=4, dim=2, alpha=4):
    module = importlib.import_module("kilora_musubi.zimage")
    net_kwargs = dict(a.split("=") for a in network_args)
    net = module.create_arch_network(
        1.0, dim, alpha, None, None, model, neuron_dropout=None, **net_kwargs
    )
    net.prepare_network(types.SimpleNamespace())
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    net.enable_gradient_checkpointing()
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-2)
    assert len(params) == len(descriptions)
    opt = torch.optim.AdamW(params)
    net.prepare_grad_etc(model)
    net.on_epoch_start(model)
    gen = torch.Generator().manual_seed(0)
    x, target = torch.randn(2, 5, 4, generator=gen), torch.randn(2, 5, 4, generator=gen)
    for _ in range(steps):
        net.on_step_start()
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.get_trainable_params(), 1.0)
        opt.step()
        opt.zero_grad()
        assert net.apply_max_norm_regularization(1.0, "cpu") == (0, 0.0, 0.0)
    net.save_weights(str(out), torch.float32, {"ss_steps": str(steps)})
    return net, x


def test_network_module_trains_a_static_mod(musubi_version, tmp_path):
    model = TinyDenoiser(seed=0)
    base = model(torch.zeros(1, 2, 4)).detach()
    net, x = run_trainer(
        model,
        tmp_path / "zi.safetensors",
        [
            "mod_id=musubi-style",
            "license=CC0-1.0",
            "seed=3",
            "trigger_words=ink style|soft lines",
            "recommended_strength=0.6",
        ],
    )
    with torch.no_grad():
        trained = model(x)
    mod = load_mod(tmp_path / "zi.kmod.safetensors")
    m = mod.manifest
    assert m["id"] == "musubi-style" and m["trained_with"] == {
        "tool": "musubi-tuner",
        "version": "0.3.6",
    }
    assert m["base_model"] == {"family": "z-image"} and "live" not in m
    assert m["trigger_words"] == ["ink style", "soft lines"]
    assert m["strength"]["recommended"] == 0.6
    net.remove()
    assert torch.equal(model(torch.zeros(1, 2, 4)), base)
    adapter = compose(model, [tmp_path / "zi.kmod.safetensors"])
    with torch.no_grad():
        torch.testing.assert_close(model(x), trained, rtol=1e-5, atol=1e-6)
    adapter.remove()

    module = importlib.import_module("kilora_musubi.zimage")
    weights = load_file(str(tmp_path / "zi.safetensors"))
    again = module.create_arch_network_from_weights(1, weights, unet=model)
    with torch.no_grad():
        torch.testing.assert_close(model(x), trained, rtol=1e-5, atol=1e-6)
    assert again.rank == 2
    again.remove()


def test_network_args_and_errors(musubi_version, tmp_path):
    model = TinyDenoiser(seed=0)
    with pytest.raises(Exception, match="unknown KiLoRA network arguments"):
        run_trainer(model, tmp_path / "a.safetensors", ["bogus=1"])
    net, _ = run_trainer(
        model,
        tmp_path / "b.safetensors",
        ["target_modules=layers.0.attention.to_out.0", "exclude_patterns=[]"],
    )
    assert net.adapter.targets == ["layers.0.attention.to_out.0"]
    net.remove()


def test_unsupported_or_missing_musubi_stops(monkeypatch):
    import kilora_musubi.network as network

    monkeypatch.setattr(network, "installed_version", lambda: "0.4.1")
    with pytest.raises(UnsupportedTrainerVersion, match="musubi-tuner >=0.3.6, <0.4"):
        importlib.import_module("kilora_musubi.zimage").create_arch_network(
            1.0, 4, 4, None, None, TinyDenoiser()
        )
    monkeypatch.setattr(network, "installed_version", lambda: None)
    with pytest.raises(UnsupportedTrainerVersion, match="cannot find"):
        importlib.import_module("kilora_musubi.zimage").create_arch_network(
            1.0, 4, 4, None, None, TinyDenoiser()
        )
