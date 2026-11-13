# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from aitk_stub import NetworkConfig, StubLoRANetwork, install, uninstall
from kilora_aitoolkit.cli import install as install_shim
from kilora_aitoolkit.extension import AI_TOOLKIT_EXTENSIONS

from kilora import compose, load_mod
from kilora.errors import KiLoRAError
from kilora.plugin_support import UnsupportedTrainerVersion
from kilora.testing import TinyDenoiser


@pytest.fixture
def stub():
    base = install()
    yield base
    uninstall()


def data():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(2, 5, 4, generator=gen), torch.randn(2, 5, 4, generator=gen)


def kilora_config():
    return NetworkConfig(
        type="kilora",
        linear=2,
        linear_alpha=2,
        network_kwargs={"mod_id": "aitk-style", "license": "CC0-1.0", "seed": 1},
    )


def test_extension_trains_a_static_mod(stub, tmp_path):
    process_class = AI_TOOLKIT_EXTENSIONS[0].get_process()
    assert AI_TOOLKIT_EXTENSIONS[0].uid == "kilora_trainer"
    model = TinyDenoiser(seed=0)
    x, target = data()
    base = model(x).detach()
    process = process_class(kilora_config(), model, (x, target), str(tmp_path / "run.safetensors"))
    net = process.run()
    assert stub.LoRASpecialNetwork is StubLoRANetwork
    torch.testing.assert_close(process.prior, base)
    torch.testing.assert_close(process.off, base)
    with torch.no_grad(), net:
        trained = model(x)
    assert not torch.allclose(trained, base)
    mod = load_mod(tmp_path / "run.kmod.safetensors")
    m = mod.manifest
    assert m["id"] == "aitk-style" and m["license"] == "CC0-1.0"
    assert m["trained_with"] == {"tool": "ai-toolkit", "version": "0.13.39"}
    assert m["base_model"] == {"family": "z-image"}
    assert m["requires_features"] == ["experts.lora"] and "live" not in m
    net.remove()
    adapter = compose(model, [tmp_path / "run.kmod.safetensors"])
    with torch.no_grad():
        torch.testing.assert_close(model(x), trained, rtol=1e-5, atol=1e-6)
    adapter.remove()


def test_other_network_types_use_ai_toolkit_unchanged(stub, tmp_path):
    process_class = AI_TOOLKIT_EXTENSIONS[0].get_process()
    process = process_class(NetworkConfig(type="lora"), TinyDenoiser(), data(), "x")
    assert isinstance(process.run(), StubLoRANetwork)


def test_unsupported_version_stops(tmp_path):
    install(version="0.14.2")
    try:
        process_class = AI_TOOLKIT_EXTENSIONS[0].get_process()
        process = process_class(kilora_config(), TinyDenoiser(), data(), str(tmp_path / "r"))
        with pytest.raises(UnsupportedTrainerVersion, match="supports AI Toolkit >=0.13.39, <0.14"):
            process.run()
    finally:
        uninstall()


def test_install_shim(tmp_path):
    root = tmp_path / "ai-toolkit"
    (root / "extensions").mkdir(parents=True)
    (root / "version.py").write_text('VERSION = "0.13.39"\n')
    target = install_shim(root)
    assert "from kilora_aitoolkit.extension import AI_TOOLKIT_EXTENSIONS" in target.read_text()
    with pytest.raises(SystemExit, match="exists"):
        install_shim(root)
    with pytest.raises(SystemExit, match="does not look like"):
        install_shim(tmp_path)


def test_mixed_strengths_per_batch_item_stop_with_a_message(stub, tmp_path):
    process_class = AI_TOOLKIT_EXTENSIONS[0].get_process()
    model = TinyDenoiser(seed=0)
    process = process_class(kilora_config(), model, data(), str(tmp_path / "run.safetensors"))
    net = process.run()
    net.multiplier = [0.5, 0.5]
    assert net.multiplier == 0.5
    with pytest.raises(KiLoRAError, match="per batch item"):
        net.multiplier = [1.0, 0.0]
