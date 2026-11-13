# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("diffusers")
from test_anima import _tiny

from kilora.adapter import compose
from kilora.families import anima_never_target_comfy, anima_never_target_diffusers
from kilora.multi_import import import_loras

W = 32


def _shapes(model):
    return {
        n: (m.in_features, m.out_features)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    }


BLOCK = {
    f"adaln_modulation_{attn}_{j}": f"transformer_blocks.0.{norm}.linear_{j}"
    for attn, norm in (("self_attn", "norm1"), ("cross_attn", "norm2"), ("mlp", "norm3"))
    for j in (1, 2)
}
OUTSIDE = {
    "final_layer_adaln_modulation_1": "norm_out.linear_1",
    "final_layer_adaln_modulation_2": "norm_out.linear_2",
    "final_layer_linear": "proj_out",
    "t_embedder_1_linear_1": "time_embed.t_embedder.linear_1",
    "t_embedder_1_linear_2": "time_embed.t_embedder.linear_2",
    "x_embedder_proj_1": "patch_embed.proj",
}


def _layout(model, lokr: bool, gen):
    shapes = _shapes(model)
    stems = {f"blocks_0_{k}": v for k, v in BLOCK.items()}
    if lokr:
        stems.update(OUTSIDE)
    sd, want = {}, {}
    for stem, layer in stems.items():
        n_in, n_out = shapes[layer]
        key = f"lora_unet_{stem}"
        if lokr and n_in % 2 == 0 and n_out % 2 == 0:
            w1, w2 = (
                torch.randn(2, 2, generator=gen),
                torch.randn(n_out // 2, n_in // 2, generator=gen),
            )
            sd[f"{key}.lokr_w1"], sd[f"{key}.lokr_w2"] = w1, w2
            sd[f"{key}.alpha"] = torch.tensor(1.0)
            want[layer] = torch.kron(w1, w2)
        else:
            a, b = torch.randn(2, n_in, generator=gen), torch.randn(n_out, 2, generator=gen)
            sd[f"{key}.lora_down.weight"], sd[f"{key}.lora_up.weight"] = a, b
            sd[f"{key}.alpha"] = torch.tensor(1.0)
            want[layer] = b @ a * 0.5
    a, b = torch.randn(2, W, generator=gen), torch.randn(W, 2, generator=gen)
    sd["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"] = a
    sd["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"] = b
    extra = []
    if lokr:
        stem = "lora_unet_llm_adapter_blocks_0_mlp_0"
        sd[f"{stem}.lokr_w1"] = torch.randn(2, 2, generator=gen)
        sd[f"{stem}.lokr_w2"] = torch.randn(4, 4, generator=gen)
        sd[f"{stem}.alpha"] = torch.tensor(1.0)
        extra = [f"{stem}.alpha", f"{stem}.lokr_w1", f"{stem}.lokr_w2"]
    return sd, want, extra


def _save(tmp_path, name, sd):
    from safetensors.torch import save_file

    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return str(path)


def test_name_map_round_trip():
    for diffusers in [*BLOCK.values(), *OUTSIDE.values()]:
        comfy = anima_never_target_comfy(diffusers)
        assert comfy is not None and anima_never_target_diffusers(comfy) == diffusers


@pytest.mark.parametrize("lokr", [True, False], ids=["aniart-like", "sketch-like"])
def test_never_target_keys_resolve_and_apply_exactly(lokr, tmp_path):
    model = _tiny(1)
    gen = torch.Generator().manual_seed(4)
    sd, want, extra = _layout(model, lokr, gen)
    result = import_loras([_save(tmp_path, "style", sd)], model, family="anima")
    assert sorted(result.unmatched["style"]) == sorted(extra)
    changed = {}
    for section in result.sections:
        for path in section.data:
            changed[anima_never_target_diffusers(path)] = section
    assert sorted(changed) == sorted(want)
    adapter = compose(
        model, [result.adapter_set], sections=[(s, 0.5) for s in result.sections], family="anima"
    )
    try:
        modules = dict(model.named_modules())
        for layer, delta in want.items():
            module = modules[layer]
            x = torch.randn(3, module.in_features, generator=gen)
            with torch.no_grad():
                got = module(x)
                base = torch.nn.functional.linear(x, module.weight, module.bias)
            torch.testing.assert_close(got, base + 0.5 * x @ delta.t(), rtol=1e-4, atol=1e-4)
    finally:
        adapter.remove()
    with torch.no_grad():
        x = torch.randn(3, W, generator=gen)
        layer = modules["transformer_blocks.0.norm1.linear_1"]
        assert torch.equal(layer(x), torch.nn.functional.linear(x, layer.weight, layer.bias))


def _conditioner():
    from diffusers.models.condition_embedders.condition_embedder_anima import (
        AnimaTextConditioner,
    )

    torch.manual_seed(1)
    return AnimaTextConditioner(
        source_dim=16,
        target_dim=16,
        model_dim=16,
        num_attention_heads=2,
        num_layers=1,
        mlp_ratio=2.0,
        target_vocab_size=50,
        use_self_attention=True,
        use_layer_norm=False,
    ).eval()


def _adapter_keys(gen):
    sd, want = {}, {}
    for path, (n_in, n_out) in {
        "llm_adapter.blocks.0.mlp.0": (16, 32),
        "llm_adapter.blocks.0.self_attn.o_proj": (16, 16),
    }.items():
        key = "lora_unet_" + path.replace(".", "_")
        w1, w2 = torch.randn(2, 2, generator=gen), torch.randn(n_out // 2, n_in // 2, generator=gen)
        sd[f"{key}.lokr_w1"], sd[f"{key}.lokr_w2"] = w1, w2
        sd[f"{key}.alpha"] = torch.tensor(1.0)
        want[path] = torch.kron(w1, w2)
    a, b = torch.randn(2, 16, generator=gen), torch.randn(16, 2, generator=gen)
    sd["lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.lora_down.weight"] = a
    sd["lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.lora_up.weight"] = b
    sd["lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.alpha"] = torch.tensor(1.0)
    want["llm_adapter.blocks.0.cross_attn.o_proj"] = b @ a * 0.5
    return sd, want


def test_llm_adapter_keys_apply_to_the_text_conditioner(tmp_path):
    from kilora.families import diffusers_tree

    model, conditioner = _tiny(1), _conditioner()
    tree = diffusers_tree(model, llm_adapter=conditioner)
    gen = torch.Generator().manual_seed(6)
    sd = _layout(model, True, gen)[0]
    adapter_sd, adapter_want = _adapter_keys(gen)
    sd = {k: v for k, v in sd.items() if "llm_adapter" not in k}
    sd.update(adapter_sd)
    result = import_loras([_save(tmp_path, "aniart", sd)], model, family="anima", full_model=tree)
    assert result.unmatched["aniart"] == []
    paths = {p for s in result.sections for p in s.data}
    assert set(adapter_want) <= paths
    adapter = compose(
        model,
        [result.adapter_set],
        sections=[(s, 0.7) for s in result.sections],
        family="anima",
        section_model=tree,
    )
    try:
        for path, delta in adapter_want.items():
            layer = tree.get_submodule(path)
            x = torch.randn(3, layer.in_features, generator=gen)
            with torch.no_grad():
                base = torch.nn.functional.linear(x, layer.weight, layer.bias)
                torch.testing.assert_close(
                    layer(x), base + 0.7 * x @ delta.t(), rtol=1e-4, atol=1e-4
                )
    finally:
        adapter.remove()


def test_llm_adapter_keys_apply_in_comfyui(tmp_path, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _tree

    from kilora.comfyui import nodes as N
    from kilora.fixed_change import is_fixed

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    layers = {f"blocks.0.self_attn.{p}": (W, W) for p in ("q_proj", "k_proj", "v_proj")}
    layers.update(
        {
            "blocks.0.self_attn.output_proj": (W, W),
            "blocks.0.mlp.layer1": (W, 4 * W),
            "blocks.0.mlp.layer2": (4 * W, W),
            "blocks.0.cross_attn.output_proj": (W, W),
            "llm_adapter.blocks.0.mlp.0": (16, 32),
            "llm_adapter.blocks.0.self_attn.o_proj": (16, 16),
            "llm_adapter.blocks.0.cross_attn.o_proj": (16, 16),
            "llm_adapter.out_proj": (16, 16),
        }
    )
    model = _tree("Anima", layers)
    gen = torch.Generator().manual_seed(7)
    sd, want = _adapter_keys(gen)
    a, b = torch.randn(2, W, generator=gen), torch.randn(W, 2, generator=gen)
    sd["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"] = a
    sd["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"] = b
    _save(tmp_path, "aniart", sd)
    (chain,) = N.KiLoRAImportLoRA().add("aniart.safetensors", 0.5)
    patched, report, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, output_gate="none"
    )
    for path, delta in want.items():
        layer = patched.object_patches[f"diffusion_model.{path}"]
        assert is_fixed(layer)
        x = torch.randn(3, layer.in_features, generator=gen)
        with torch.no_grad():
            base = model.get_submodule(path)(x)
            torch.testing.assert_close(layer(x), base + 0.5 * x @ delta.t(), rtol=1e-4, atol=1e-4)
