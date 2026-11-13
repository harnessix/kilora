# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import CosmosTransformer3DModel

from kilora import adapter as adapter_module
from kilora import compute as C
from kilora.adapter import attach
from kilora.bindings import get_binding
from kilora.presets import ANIMA_NOTICE, anima_name, get_preset

TEXT = 12


def _tiny(blocks=2):
    from kilora.anima import ensure_mask_resize

    ensure_mask_resize()
    torch.manual_seed(0)
    return CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=blocks,
        mlp_ratio=4.0,
        text_embed_dim=TEXT,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    ).eval()


def _inputs(seed=1, batch=2):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        hidden_states=torch.randn(batch, 16, 1, 8, 8, generator=gen),
        timestep=torch.full((batch,), 0.6),
        encoder_hidden_states=torch.randn(batch, 5, TEXT, generator=gen),
        padding_mask=torch.zeros(1, 1, 64, 64),
        return_dict=False,
    )


def _attached(model):
    adapter = attach(model, get_preset("anima").config(num_experts=4, routes=2, router_dim=8))
    with torch.no_grad():
        for site in adapter.sites:
            site.tensors["s0_up"].normal_(0, 0.2)
    return adapter


def test_preset_finds_ten_targets_per_block():
    model = _tiny(blocks=3)
    adapter = _attached(model)
    assert len(adapter.targets) == 30
    per_block = [t for t in adapter.targets if t.startswith("transformer_blocks.1.")]
    attn = [
        f"{a}.{p}"
        for a in ("self_attn", "cross_attn")
        for p in ("q_proj", "k_proj", "v_proj", "output_proj")
    ]
    got = sorted(anima_name(t).split(".", 2)[2] for t in per_block)
    assert got == sorted([*attn, "mlp.layer1", "mlp.layer2"])
    assert not any("norm" in t or "adaln" in t for t in adapter.targets)
    s = adapter.settings
    assert (s.routing_scope, s.routing_select, s.carry, s.output_gate, s.carry_across_steps) == (
        "token",
        "sparsemax",
        "dropped",
        "separable",
        False,
    )


def test_fast_path_equals_the_loop_with_all_defaults(monkeypatch):
    model = _tiny()
    _attached(model)
    real = C.run_layer

    def generation(fast):
        outs = []

        def keep(*args, **kw):
            out = real(*args, **kw)
            outs.append(out)
            return out

        monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
        monkeypatch.setattr(adapter_module, "run_layer", keep)
        with torch.no_grad(), C.generation(model, carry=True):
            for step in range(3):
                inputs = _inputs()
                inputs["timestep"] = torch.full((2,), 0.9 - 0.3 * step)
                model(**inputs)
        return outs

    loop, fast = generation(False), generation(True)
    assert len(loop) == len(fast) == 3 * 20
    for a, b in zip(loop, fast, strict=True):
        assert float((a - b).abs().max() / a.abs().max()) < 1e-5


def test_flow_loss_trains_the_adapter():
    from kilora.anima import flow_loss

    model = _tiny()
    adapter = _attached(model)
    model.train()
    gen = torch.Generator().manual_seed(2)
    batch = [
        types.SimpleNamespace(
            latent=torch.randn(16, 8, 8, generator=gen), caption=torch.randn(n, TEXT, generator=gen)
        )
        for n in (5, 3)
    ]
    loss = flow_loss(model, batch, torch.Generator().manual_seed(3), torch.float32)
    loss.backward()
    assert torch.isfinite(loss)
    assert adapter.sites[0].tensors["s0_up"].grad.abs().sum() > 0


def test_comfyui_binding_and_notice():
    binding = get_binding("anima")
    model = _tiny()
    targets = _attached(model).targets
    comfy = [anima_name(t) for t in targets] + ["llm_adapter.out_proj"]
    assert binding.matches_comfy(comfy)
    points = set()
    for t in targets:
        place = binding.to_comfy(t)
        assert place.path == "diffusion_model." + anima_name(t) and place.part is None
        assert binding.from_comfy(anima_name(t)) == [(t, None)]
        points.add(binding.hook_point(t))
    assert len(points) == len(targets)
    assert not get_binding("z-image").matches_comfy(comfy)
    preset = get_preset("anima")
    assert (preset.prediction, preset.latent_format, preset.flow_shift) == ("flow", "wan21", 3.0)
    assert (preset.resolution, preset.steps, preset.guidance) == (1024, 30, 4.0)
    assert "CircleStone Labs Non-Commercial License" in ANIMA_NOTICE


def test_card_carries_the_notice():
    from kilora.card import card_fields, card_markdown

    card = card_markdown(card_fields({"id": "a", "license_notice": ANIMA_NOTICE}))
    assert "## License notice" in card and ANIMA_NOTICE in card


def test_mod_costs_at_most_twice_the_base_call():
    import time

    from kilora.anima import ensure_mask_resize

    ensure_mask_resize()
    torch.manual_seed(0)
    model = CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=4,
        attention_head_dim=64,
        num_layers=4,
        mlp_ratio=4.0,
        text_embed_dim=128,
        adaln_lora_dim=16,
        max_size=(4, 128, 128),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    ).eval()
    inputs = dict(
        hidden_states=torch.randn(1, 16, 1, 96, 96),
        timestep=torch.full((1,), 0.5),
        encoder_hidden_states=torch.randn(1, 64, 128),
        padding_mask=torch.zeros(1, 1, 768, 768),
        return_dict=False,
    )

    adapter = attach(model, get_preset("anima").config(num_experts=4, routes=2, router_dim=32))
    with torch.no_grad():
        for site in adapter.sites:
            site.tensors["s0_up"].normal_(0, 0.02)
    assert adapter.settings.output_gate == "separable"

    def once():
        start = time.perf_counter()
        model(**inputs)
        return time.perf_counter() - start

    base, with_mod = [], []
    with torch.no_grad():
        model(**inputs)
        for _ in range(6):
            with adapter.depth(0):
                base.append(once())
            with_mod.append(once())
    adapter.remove()
    base, with_mod = min(base), min(with_mod)
    assert with_mod < 2.0 * base, (with_mod, base)


def test_kohya_anima_lora_with_adaln_keys_imports():
    from kilora.lora_import import import_lora

    model = _tiny(1)
    gen = torch.Generator().manual_seed(2)
    width = 32
    sd = {}
    for name, (n_in, n_out) in {
        "self_attn_q_proj": (width, width),
        "cross_attn_k_proj": (TEXT, width),
        "mlp_layer1": (width, 4 * width),
        "adaln_modulation_cross_attn_1": (width, 4),
        "adaln_modulation_mlp_2": (4, 3 * width),
    }.items():
        stem = f"lora_unet_blocks_0_{name}"
        sd[f"{stem}.lora_down.weight"] = torch.randn(2, n_in, generator=gen)
        sd[f"{stem}.lora_up.weight"] = torch.randn(n_out, 2, generator=gen)
        sd[f"{stem}.alpha"] = torch.tensor(2.0)
    result = import_lora(sd, model, family="anima")
    (section,) = result.sections
    assert sorted(section.data) == [
        "blocks.0.adaln_modulation_cross_attn.1",
        "blocks.0.adaln_modulation_mlp.2",
    ]
    imported = result.adapter_set
    assert sorted(imported.experts) == [
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn2.to_k",
        "transformer_blocks.0.ff.net.0.proj",
    ]
