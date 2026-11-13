# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_families import _ab, _delta, _model, _save
from torch import nn


def _checkpoint(tmp_path, family, prefix):
    from safetensors.torch import save_file

    sd = {}
    for name, m in _model(family).named_modules():
        if isinstance(m, nn.Linear):
            sd[f"{prefix}{name}.weight"] = m.weight.detach().clone()
            sd[f"{prefix}{name}.bias"] = m.bias.detach().clone()
    path = tmp_path / f"{family}-base.safetensors"
    save_file(sd, str(path))
    return str(path)


@pytest.mark.parametrize(
    ("family", "prefix", "stem", "target"),
    [
        (
            "wan",
            "model.diffusion_model.",
            "transformer.blocks.0.attn1.to_q",
            "blocks.0.self_attn.q",
        ),
        ("flux", "", "lora_unet_single_blocks_1_linear2", "single_blocks.1.linear2"),
        (
            "hunyuan-video",
            "diffusion_model.",
            "transformer.double_blocks.0.img_attn_proj",
            "double_blocks.0.img_attn.proj",
        ),
        ("qwen-text", "model.", "lora_te_layers_1_mlp_down_proj", "layers.1.mlp.down_proj"),
    ],
)
def test_cli_import_finds_the_family_from_the_checkpoint(tmp_path, family, prefix, stem, target):
    import argparse

    from kilora.cli import _import
    from kilora.modfile import load_mod

    base = _checkpoint(tmp_path, family, prefix)
    m = _model(family).get_submodule(target)
    gen = torch.Generator().manual_seed(2)
    a, b = _ab(gen, m.in_features, m.out_features)
    kohya = stem.startswith("lora_")
    names = (
        (".lora_down.weight", ".lora_up.weight") if kohya else (".lora_A.weight", ".lora_B.weight")
    )
    lora = _save(tmp_path, "style", {stem + names[0]: a, stem + names[1]: b})
    out = tmp_path / "style.kmod.safetensors"
    args = argparse.Namespace(
        loras=[lora],
        model=base,
        output=str(out),
        family=None,
        mode="routed",
        weights=None,
        names=None,
        source_info=None,
        id=None,
        license=None,
    )
    assert _import(args) == 0
    mod = load_mod(out)
    assert mod.manifest["base_model"]["family"] == family
    (adapter_set,) = mod.sets
    assert list(adapter_set.experts) == [target]
    torch.testing.assert_close(_delta(adapter_set, target), b @ a)


def test_cli_import_refuses_an_unknown_checkpoint(tmp_path, capsys):
    import argparse

    from safetensors.torch import save_file

    from kilora.cli import _import

    base = tmp_path / "other.safetensors"
    save_file({"foo.bar.weight": torch.zeros(2, 2)}, str(base))
    lora = _save(tmp_path, "x", {"foo.bar.lora_A.weight": torch.zeros(1, 2)})
    args = argparse.Namespace(
        loras=[lora],
        model=str(base),
        output=str(tmp_path / "x.kmod"),
        family=None,
        mode="routed",
        weights=None,
        names=None,
        source_info=None,
        id=None,
        license=None,
    )
    assert _import(args) == 1
    assert "give --family" in capsys.readouterr().err
