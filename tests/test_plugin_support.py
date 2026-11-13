# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import compose, load_mod
from kilora.plugin_support import TrainerNetwork, UnsupportedTrainerVersion, check_trainer_version


def test_version_check():
    assert check_trainer_version("tool", "0.3.6", ">=0.3.6, <0.4") == "0.3.6"
    with pytest.raises(UnsupportedTrainerVersion, match="This plugin supports tool >=0.3.6, <0.4"):
        check_trainer_version("tool", "0.4.0", ">=0.3.6, <0.4")
    with pytest.raises(UnsupportedTrainerVersion, match="cannot find"):
        check_trainer_version("tool", None, ">=1")


def network(target_modules=None):
    return TrainerNetwork(
        family="z-image",
        target_modules=target_modules,
        rank=2,
        alpha=4,
        trained_with={"tool": "stub", "version": "1.0"},
        manifest={"license": "CC0-1.0"},
    )


def test_train_save_and_load_as_mod(model, x, tmp_path):
    base = model(x).detach()
    net = network(["layers.*.feed_forward.w2"])
    net.attach(model)
    assert net.adapter.targets == [f"layers.{i}.feed_forward.w2" for i in range(2)]
    groups = net.parameter_groups(1e-2)
    assert groups[1]["lr"] == pytest.approx(2e-2)
    assert len(groups) == 2
    before = [p.detach().clone() for group in groups for p in group["params"]]
    assert before
    assert all(
        not p.requires_grad
        for site in net.adapter.sites
        for key, p in site.tensors.items()
        if "_keys_" in key
    )
    opt = torch.optim.AdamW(groups)
    for _ in range(5):
        loss = (model(x) - 1).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    trained = model(x).detach()
    assert not torch.allclose(trained, base)
    active = [p for group in groups for p in group["params"]]
    assert any(not torch.equal(a, b) for a, b in zip(before, active, strict=True))
    net.set_multiplier(0.0)
    torch.testing.assert_close(model(x), base)
    net.set_multiplier(1.0)
    net.set_enabled(False)
    torch.testing.assert_close(model(x), base)
    net.set_enabled(True)
    kmod = net.save(tmp_path / "style.safetensors")
    assert kmod.endswith("style.kmod.safetensors")
    net.remove()
    mod = load_mod(kmod)
    assert mod.manifest["trained_with"] == {"tool": "stub", "version": "1.0"}
    assert mod.manifest["id"] == "style" and mod.manifest["license"] == "CC0-1.0"
    assert mod.settings.depth == 1 and mod.sets[0].num_experts == 16
    adapter = compose(model, [kmod])
    torch.testing.assert_close(model(x), trained, rtol=1e-5, atol=1e-6)
    adapter.remove()
    again = network(["layers.*.feed_forward.w2"])
    again.attach(model)
    assert again.load(tmp_path / "style.safetensors") == []
    torch.testing.assert_close(model(x), trained, rtol=1e-5, atol=1e-6)


def test_static_mods_need_two_or_more_experts():
    from kilora.errors import KiLoRAError
    from kilora.plugin_support import TrainerNetwork

    with pytest.raises(KiLoRAError, match="at least 2 experts per layer"):
        TrainerNetwork(family="z-image", rank=2, num_experts=1)
    net = TrainerNetwork(family="z-image", rank=2, num_experts=2)
    assert net.num_experts == 2
