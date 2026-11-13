# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher
from test_families import _ab, _model

from kilora.comfyui import patching


@pytest.fixture
def inputs(tmp_path):
    model = _model("flux")
    layer = "double_blocks.0.img_attn.proj"
    module = model.get_submodule(layer)
    stem = "lora_unet_" + layer.replace(".", "_")
    gen = torch.Generator().manual_seed(932)
    w1 = torch.randn(2, 4, generator=gen)
    w2 = torch.randn(module.out_features // 2, module.in_features // 4, generator=gen)
    a, b = _ab(gen, module.in_features, module.out_features, rank=1)
    weights = {
        "lokr": {f"{stem}.lokr_w1": w1, f"{stem}.lokr_w2": w2},
        "plain": {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b},
    }
    paths = {}
    for name, sd in weights.items():
        path = tmp_path / f"{name}.safetensors"
        save_file(sd, str(path))
        paths[name] = str(path)
    x = torch.randn(2, 3, module.in_features, generator=gen)
    return model, layer, paths, x, torch.kron(w1, w2)


def _apply(model, entries):
    return patching.apply(
        FakePatcher(FakeBaseModel(model)),
        patching.ModChain(tuple(entries)),
        1,
        gate=("none", None),
        joint=True,
    )


def test_single_creative_lokr_is_exact(inputs):
    model, layer, paths, x, delta = inputs
    entry = patching.ModEntry("lora", paths["lokr"], 0.7, axis="creative")
    out, report = _apply(model, [entry])
    with torch.no_grad():
        actual = out.object_patches[f"diffusion_model.{layer}"](x)
        expected = model.get_submodule(layer)(x) + 0.7 * x @ delta.t()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert "energy kept by the SVD cut" not in report
    assert "single." in report
    assert patching.joined_names([entry], True) == frozenset()


def test_single_creative_loha_is_exact(tmp_path):
    model = _model("flux")
    layer = "double_blocks.0.img_attn.proj"
    module = model.get_submodule(layer)
    stem = "lora_unet_" + layer.replace(".", "_")
    gen = torch.Generator().manual_seed(933)
    a, b = _ab(gen, module.in_features, module.out_features, rank=2)
    c, d = _ab(gen, module.in_features, module.out_features, rank=2)
    path = tmp_path / "loha.safetensors"
    save_file(
        {
            f"{stem}.hada_w1_a": b,
            f"{stem}.hada_w1_b": a,
            f"{stem}.hada_w2_a": d,
            f"{stem}.hada_w2_b": c,
        },
        str(path),
    )
    entry = patching.ModEntry("lora", str(path), 0.7, axis="creative")
    out, report = _apply(model, [entry])
    x = torch.randn(2, 3, module.in_features, generator=gen)
    with torch.no_grad():
        actual = out.object_patches[f"diffusion_model.{layer}"](x)
        expected = module(x) + 0.7 * x @ ((b @ a) * (d @ c)).t()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert "energy kept by the SVD cut" not in report
    assert "single." in report
    assert not patching.joined_names([entry], True)


@pytest.mark.parametrize("names", [("plain",), ("plain", "lokr")])
def test_routed_output_matches_previous_group_selection(inputs, monkeypatch, names):
    model, layer, paths, x, _ = inputs
    entries = [patching.ModEntry("lora", paths[name], 0.7, axis="creative") for name in names]
    actual, actual_report = _apply(model, entries)
    assert patching.joined_names(entries, True) == frozenset(names)

    def previous_groups(chain, joint):
        groups = {
            axis: [e for e in chain if patching._joint_lora(e, axis)]
            for axis in patching.JOINT_SETS
        }
        return {
            axis: es for axis, es in groups.items() if joint and (axis == "creative" or len(es) > 1)
        }

    monkeypatch.setattr(patching, "_routed_groups", previous_groups)
    expected, expected_report = _apply(model, entries)
    with torch.no_grad():
        actual_output = actual.object_patches[f"diffusion_model.{layer}"](x)
        expected_output = expected.object_patches[f"diffusion_model.{layer}"](x)
    assert torch.equal(actual_output, expected_output)
    assert actual_report == expected_report
