# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import argparse

import torch
from safetensors.torch import save_file

from kilora import compose
from kilora.cli import _import_lora
from kilora.modfile import load_mod
from kilora.testing import TinyDenoiser


def _lora_file(model, path, rank=2):
    gen = torch.Generator().manual_seed(4)
    sd = {}
    for name, m in model.named_modules():
        if name.endswith("attention.to_out.0") or name.endswith("attention.to_v"):
            sd[f"transformer.{name}.lora_A.weight"] = torch.randn(
                rank, m.in_features, generator=gen
            )
            sd[f"transformer.{name}.lora_B.weight"] = torch.randn(
                m.out_features, rank, generator=gen
            )
    save_file(sd, str(path), metadata={"modelspec.license": "CC-BY-4.0"})
    return sd


def _args(lora, out, *, training=False, weight=1.0):
    return argparse.Namespace(
        lora=str(lora),
        output=str(out),
        model="tiny",
        family="z-image",
        weight=weight,
        experts=2,
        for_training=training,
        id=None,
        license=None,
    )


def test_untrained_import_is_plain_lora_at_depth_one(tmp_path, capsys):
    model = TinyDenoiser(seed=0).eval()
    sd = _lora_file(model, tmp_path / "ink.safetensors")
    assert _import_lora(_args(tmp_path / "ink.safetensors", tmp_path / "ink.kmod"), model) == 0
    assert "Form: input" in capsys.readouterr().out
    mod = load_mod(tmp_path / "ink.kmod")
    (s,) = mod.sets
    assert s.form == "input" and mod.manifest["license"] == "CC-BY-4.0"
    assert s.extra.get("trained_with_kilora") is False
    x = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        modules = dict(model.named_modules())
        handles = []
        for key, a in sd.items():
            if not key.endswith("lora_A.weight"):
                continue
            name = key[len("transformer.") : -len(".lora_A.weight")]
            b = sd[key.replace("lora_A", "lora_B")]

            def hook(m, inp, out, a=a, b=b):
                return out + (inp[0] @ a.t()) @ b.t()

            handles.append(modules[name].register_forward_hook(hook))
        want = model(x)
        for h in handles:
            h.remove()
        adapter = compose(model, mod.sets, mix="stack")
        adapter.set_depth(1)
        torch.testing.assert_close(model(x), want, rtol=1e-5, atol=1e-5)
        adapter.remove()


def test_import_for_training_converts_with_the_pseudo_inverse(tmp_path, capsys):
    model = TinyDenoiser(seed=0).eval()
    _lora_file(model, tmp_path / "ink.safetensors")
    args = _args(tmp_path / "ink.safetensors", tmp_path / "ink-t.kmod", training=True)
    assert _import_lora(args, model) == 0
    assert "Converted for training" in capsys.readouterr().out
    (s,) = load_mod(tmp_path / "ink-t.kmod").sets
    plain = load_mod_input(model, tmp_path)
    assert s.form == "state"
    modules = dict(model.named_modules())
    for target, ex in s.experts.items():
        w = modules[target].weight.detach().float()
        gram = w @ w.t()
        lam = 1e-4 * float(gram.diagonal().mean())
        pinv = w.t() @ torch.linalg.inv(gram + lam * torch.eye(gram.shape[0]))
        a_plain = plain.experts[target].down[0, 0]
        torch.testing.assert_close(ex.down[0, 0], a_plain @ pinv, rtol=1e-2, atol=1e-3)
        assert torch.equal(ex.up, plain.experts[target].up)


def test_conversion_is_exact_with_a_small_ridge(tmp_path):
    from kilora.lora_import import import_to_mod

    model = TinyDenoiser(seed=0).eval()
    _lora_file(model, tmp_path / "ink.safetensors")
    import_to_mod(
        tmp_path / "ink.safetensors", model, tmp_path / "e.kmod", state_form=True, ridge=1e-9
    )
    plain = load_mod_input(model, tmp_path)
    modules = dict(model.named_modules())
    gen = torch.Generator().manual_seed(7)
    for target, ex in load_mod(tmp_path / "e.kmod").sets[0].experts.items():
        w = modules[target].weight.detach()
        x = torch.randn(5, w.shape[1], generator=gen)
        got = (x @ w.t()) @ ex.down[0, 0].t()
        want = x @ plain.experts[target].down[0, 0].t()
        assert float((got - want).norm() / want.norm()) < 1e-3


def load_mod_input(model, tmp_path):
    _import_lora(_args(tmp_path / "ink.safetensors", tmp_path / "ink-i.kmod"), model)
    return load_mod(tmp_path / "ink-i.kmod").sets[0]


def test_bad_file_gives_a_plain_error(tmp_path, capsys):
    model = TinyDenoiser(seed=0).eval()
    save_file({"something.else": torch.zeros(2)}, str(tmp_path / "bad.safetensors"))
    assert _import_lora(_args(tmp_path / "bad.safetensors", tmp_path / "x.kmod"), model) == 1
    assert capsys.readouterr().err.startswith("error:")
