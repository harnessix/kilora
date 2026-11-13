# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from kilora import adapter as adapter_module
from kilora import compute as C
from kilora.adapter import attach
from kilora.bindings import BINDINGS
from kilora.families import FAMILIES, detect, detect_names, get_family, normalize_for
from kilora.family_spec import discover_linear, prediction_of
from kilora.multi_import import import_loras
from kilora.presets import get_preset

H = 8
MLP = 16


def _tree(cls_name: str, layers: dict[str, tuple[int, int]]) -> nn.Module:
    class Fake(nn.Module):
        def forward(self, batch=2, tokens=6):
            out = 0.0
            for i, (_name, m) in enumerate(self.named_modules()):
                if isinstance(m, nn.Linear):
                    gen = torch.Generator().manual_seed(i)
                    x = torch.randn(batch, tokens, m.in_features, generator=gen)
                    out = out + m(x).square().mean()
            return out

    Fake.__name__ = Fake.__qualname__ = cls_name
    root = Fake()
    torch.manual_seed(0)
    for path, (fan_in, fan_out) in layers.items():
        parent = root
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(parts[-1], nn.Linear(fan_in, fan_out))
    return root.eval()


def _double_stream(blocks=2, singles=2, hunyuan=False) -> dict[str, tuple[int, int]]:
    layers = {"img_in": (4, H), "time_in.in_layer": (16, H), "time_in.out_layer": (H, H)}
    if hunyuan:
        layers["txt_in.input_embedder"] = (6, H)
        layers["txt_in.individual_token_refiner.blocks.0.self_attn_qkv"] = (H, 3 * H)
    else:
        layers["txt_in"] = (6, H)
    for i in range(blocks):
        for s in ("img", "txt"):
            b = f"double_blocks.{i}.{s}"
            layers.update(
                {
                    f"{b}_mod.lin": (H, 6 * H),
                    f"{b}_attn.qkv": (H, 3 * H),
                    f"{b}_attn.proj": (H, H),
                    f"{b}_mlp.0": (H, MLP),
                    f"{b}_mlp.2": (MLP, H),
                }
            )
    for i in range(singles):
        layers.update(
            {
                f"single_blocks.{i}.linear1": (H, 3 * H + MLP),
                f"single_blocks.{i}.linear2": (H + MLP, H),
                f"single_blocks.{i}.modulation.lin": (H, 3 * H),
            }
        )
    layers.update({"final_layer.linear": (H, 16), "final_layer.adaLN_modulation.1": (H, 2 * H)})
    return layers


def _qwen_image(blocks=2):
    layers = {
        "img_in": (4, H),
        "txt_in": (6, H),
        "proj_out": (H, 16),
        "norm_out.linear": (H, 2 * H),
    }
    layers["time_text_embed.timestep_embedder.linear_1"] = (16, H)
    for i in range(blocks):
        b = f"transformer_blocks.{i}"
        for p in ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj"):
            layers[f"{b}.attn.{p}"] = (H, H)
        layers[f"{b}.attn.to_add_out"] = (H, H)
        for s in ("img", "txt"):
            layers[f"{b}.{s}_mod.1"] = (H, 6 * H)
            layers[f"{b}.{s}_mlp.net.0.proj"] = (H, MLP)
            layers[f"{b}.{s}_mlp.net.2"] = (MLP, H)
    return layers


def _wan(blocks=2):
    layers = {
        "text_embedding.0": (6, H),
        "text_embedding.2": (H, H),
        "time_embedding.0": (16, H),
        "time_projection.1": (H, 6 * H),
        "head.head": (H, 16),
    }
    for i in range(blocks):
        for a in ("self_attn", "cross_attn"):
            for p in "qkvo":
                layers[f"blocks.{i}.{a}.{p}"] = (H, H)
        layers[f"blocks.{i}.ffn.0"] = (H, MLP)
        layers[f"blocks.{i}.ffn.2"] = (MLP, H)
    return layers


def _ltxv(blocks=2):
    layers = {
        "patchify_proj": (4, H),
        "adaln_single.linear": (H, 6 * H),
        "caption_projection.linear_1": (6, H),
        "proj_out": (H, 4),
    }
    for i in range(blocks):
        for a in ("attn1", "attn2"):
            for p in ("to_q", "to_k", "to_v", "to_out.0"):
                layers[f"transformer_blocks.{i}.{a}.{p}"] = (H, H)
        layers[f"transformer_blocks.{i}.ff.net.0.proj"] = (H, MLP)
        layers[f"transformer_blocks.{i}.ff.net.2"] = (MLP, H)
    return layers


def _qwen_text(blocks=2):
    layers = {}
    for i in range(blocks):
        for p in "qkvo":
            layers[f"layers.{i}.self_attn.{p}_proj"] = (H, H)
        layers[f"layers.{i}.mlp.gate_proj"] = (H, MLP)
        layers[f"layers.{i}.mlp.up_proj"] = (H, MLP)
        layers[f"layers.{i}.mlp.down_proj"] = (MLP, H)
    return layers


def _hunyuan3d():
    layers = _double_stream()
    for name in ("img_in", "txt_in"):
        del layers[name]
    layers.update({"latent_in": (4, H), "cond_in": (6, H)})
    return layers


def _ace_step15(blocks=2):
    layers = {"decoder.condition_embedder": (6, H), "decoder.time_embed.linear_1": (16, H)}
    for i in range(2):
        layers[f"encoder.lyric_encoder.layers.{i}.self_attn.q_proj"] = (H, H)
    for i in range(blocks):
        for a in ("self_attn", "cross_attn"):
            for p in "qkvo":
                layers[f"decoder.layers.{i}.{a}.{p}_proj"] = (H, H)
        layers[f"decoder.layers.{i}.mlp.gate_proj"] = (H, MLP)
        layers[f"decoder.layers.{i}.mlp.up_proj"] = (H, MLP)
        layers[f"decoder.layers.{i}.mlp.down_proj"] = (MLP, H)
    return layers


def _moss_image(blocks=2, fused=True):
    layers = {
        "video_embedder": (4, H),
        "time_embedding.0": (16, H),
        "shared_block_modulation": (H, 6 * H),
        "final_linear_video": (H, 4),
        "video_head_modulation": (H, 2 * H),
        "text_embedder.proj": (6, H),
        "text_embedder.blocks.0.qkv": (H, 3 * H),
    }
    for i in range(blocks):
        names = ("qkv", "o") if fused else ("q", "k", "v", "o")
        for p in names:
            layers[f"layers.{i}.proj.{p}"] = (H, 3 * H if p == "qkv" else H)
        layers[f"layers.{i}.proj.ffn_up_gate"] = (H, 2 * MLP)
        layers[f"layers.{i}.proj.ffn_down"] = (MLP, H)
    return layers


def _flux3(blocks=2):
    layers = {"emb_in.image": (4, H), "emb_in.image_cond": (4, H), "txt_in": (6, H)}
    layers["single_stream_modulations.image.lin"] = (H, 6 * H)
    stacks = ["content_mode_blocks.image", "content_mode_blocks.image_cond", "txt_mode_blocks"]
    for stack in [*stacks, "single_blocks"]:
        for i in range(blocks):
            b = f"{stack}.{i}"
            layers.update(
                {
                    f"{b}.qkv": (H, 3 * H),
                    f"{b}.attn_out": (H, H),
                    f"{b}.mlp_in": (H, 2 * MLP),
                    f"{b}.mlp_out": (MLP, H),
                }
            )
    return layers


def _chroma():
    layers = {
        k: v for k, v in _double_stream().items() if "_mod." not in k and "modulation" not in k
    }
    layers["distilled_guidance_layer.in_proj"] = (16, H)
    layers["distilled_guidance_layer.layers.0.in_layer"] = (H, H)
    return layers


MODELS = {
    "flux": ("Flux", lambda: _double_stream(), 2 * 8 + 2 * 2),
    "qwen-image": ("QwenImageTransformer2DModel", lambda: _qwen_image(), 2 * 12),
    "wan": ("WanModel", lambda: _wan(), 2 * 10),
    "ltxv": ("LTXVModel", lambda: _ltxv(), 2 * 10),
    "hunyuan-video": ("HunyuanVideo", lambda: _double_stream(hunyuan=True), 2 * 8 + 2 * 2),
    "qwen-text": ("Llama2_", lambda: _qwen_text(), 2 * 7),
    "hunyuan3d": ("Hunyuan3Dv2", lambda: _hunyuan3d(), 2 * 8 + 2 * 2),
    "moss-image": ("MOSSImage", lambda: _moss_image(), 2 * 4),
    "flux3": ("Flux3", lambda: _flux3(), 8 * 4),
    "ace-step15": ("AceStepConditionGenerationModel", lambda: _ace_step15(), 2 * 11),
}


def _model(family):
    cls, layers, _ = MODELS[family]
    return _tree(cls, layers())


@pytest.mark.parametrize("family", sorted(MODELS))
def test_discovery_finds_every_target(family):
    model = _model(family)
    spec = get_family(family)
    targets = spec.targets(model)
    assert len(targets) == MODELS[family][2]
    assert not any(spec.is_excluded(t) for t in targets)
    auto = discover_linear(model, exclude=spec.exclude)
    assert set(targets) <= set(auto) | set()
    adapter = attach(model, get_preset(family).config(num_experts=2, routes=1, router_dim=4))
    assert sorted(adapter.targets) == sorted(targets)
    binding = BINDINGS[family]
    assert all(binding.from_comfy(t) == [(t, None)] for t in targets)
    assert binding.to_comfy(targets[0]).path == "diffusion_model." + targets[0]
    adapter.remove()


@pytest.mark.parametrize("family", sorted(MODELS))
def test_detection_by_class_and_by_names(family):
    model = _model(family)
    root = get_family(family).root
    assert detect(model, root).family == family
    paths = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert detect_names(paths, root).family == family


@pytest.mark.parametrize("family", sorted(MODELS))
def test_fast_path_equals_the_loop(family, monkeypatch):
    model = _model(family)
    adapter = attach(model, get_preset(family).config(num_experts=4, routes=2, router_dim=8))
    with torch.no_grad():
        for site in adapter.sites:
            site.tensors["s0_up"].normal_(0, 0.2)
    real = C.run_layer

    def outputs(fast):
        outs = []

        def keep(*args, **kw):
            out = real(*args, **kw)
            outs.append(out)
            return out

        monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
        monkeypatch.setattr(adapter_module, "run_layer", keep)
        with torch.no_grad():
            model()
        return outs

    loop, fast = outputs(False), outputs(True)
    assert len(loop) == len(fast) == MODELS[family][2]
    for a, b in zip(loop, fast, strict=True):
        assert float((a - b).abs().max() / a.abs().max()) < 1e-5


def test_prediction_comes_from_model_sampling():
    class EPS: ...

    class V_PREDICTION(EPS): ...

    class CONST: ...

    class ModelSamplingDiscrete: ...

    def model(*bases):
        sampling = type("Sampling", (*bases, ModelSamplingDiscrete), {})()
        return type("Model", (), {"model_sampling": sampling})()

    assert prediction_of(model(EPS)) == "eps"
    assert prediction_of(model(V_PREDICTION)) == "v"
    assert prediction_of(model(CONST)) == "flow"
    assert prediction_of(object()) == ""


def _delta(adapter_set, target, expert=0):
    ex = adapter_set.experts[target]
    return ex.up[expert, 0].t() @ ex.down[expert, 0]


def _ab(gen, n_in, n_out, rank=2):
    return torch.randn(rank, n_in, generator=gen), torch.randn(n_out, rank, generator=gen)


def _save(tmp_path, name, sd):
    from safetensors.torch import save_file

    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return str(path)


def test_flux_diffusers_lora_lands_on_fused_slices(tmp_path):
    model = _model("flux")
    gen = torch.Generator().manual_seed(3)
    sd, want = {}, {}
    qkv = torch.zeros(3 * H, H)
    for j, proj in enumerate(("to_q", "to_v")):
        a, b = _ab(gen, H, H)
        sd[f"transformer.transformer_blocks.0.attn.{proj}.lora_A.weight"] = a
        sd[f"transformer.transformer_blocks.0.attn.{proj}.lora_B.weight"] = b
        rows = (0 if j == 0 else 2) * H
        qkv[rows : rows + H] = b @ a
    want["double_blocks.0.img_attn.qkv"] = qkv
    a, b = _ab(gen, H, MLP)
    sd["transformer.single_transformer_blocks.1.proj_mlp.lora_A.weight"] = a
    sd["transformer.single_transformer_blocks.1.proj_mlp.lora_B.weight"] = b
    lin1 = torch.zeros(3 * H + MLP, H)
    lin1[3 * H :] = b @ a
    want["single_blocks.1.linear1"] = lin1
    a, b = _ab(gen, H, H)
    sd["transformer.transformer_blocks.1.norm1.linear.lora_A.weight"] = torch.randn(2, H)
    sd["transformer.transformer_blocks.1.norm1.linear.lora_B.weight"] = torch.randn(6 * H, 2)
    path = _save(tmp_path, "flux-diffusers", sd)
    result = import_loras([path], model, family="flux")
    assert sorted(result.adapter_set.experts) == sorted(want)
    for target, delta in want.items():
        torch.testing.assert_close(_delta(result.adapter_set, target), delta)
    assert result.unmatched["flux-diffusers"] == []
    (section,) = result.sections
    assert section.kind == "fixed-lora" and list(section.data) == ["double_blocks.1.img_mod.lin"]

    a, b = _ab(gen, MLP, H, rank=4)
    kohya = {
        "lora_unet_double_blocks_0_txt_mlp_2.lora_down.weight": a,
        "lora_unet_double_blocks_0_txt_mlp_2.lora_up.weight": b,
        "lora_unet_double_blocks_0_txt_mlp_2.alpha": torch.tensor(2.0),
    }
    result = import_loras([_save(tmp_path, "flux-kohya", kohya)], model, family="flux")
    torch.testing.assert_close(
        _delta(result.adapter_set, "double_blocks.0.txt_mlp.2"), (b @ a) * 0.5
    )


def test_sliced_alpha_folds_into_the_rows(tmp_path):
    model = _model("flux")
    gen = torch.Generator().manual_seed(4)
    a, b = _ab(gen, H, H, rank=4)
    sd = {
        "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight": a,
        "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight": b,
        "transformer.single_transformer_blocks.0.attn.to_k.alpha": torch.tensor(8.0),
    }
    result = import_loras([_save(tmp_path, "k", sd)], model, family="flux")
    want = torch.zeros(3 * H + MLP, H)
    want[H : 2 * H] = (b @ a) * 2.0
    torch.testing.assert_close(_delta(result.adapter_set, "single_blocks.0.linear1"), want)


FORMATS = {
    "qwen-image": [
        ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_q", None),
        (
            "transformer.transformer_blocks.1.img_mlp.net.2",
            "transformer_blocks.1.img_mlp.net.2",
            None,
        ),
        (
            "lycoris_transformer_blocks_0_attn_add_k_proj",
            "transformer_blocks.0.attn.add_k_proj",
            1.0,
        ),
    ],
    "wan": [
        ("transformer.blocks.0.attn1.to_q", "blocks.0.self_attn.q", None),
        ("transformer.blocks.1.ffn.net.0.proj", "blocks.1.ffn.0", None),
        ("diffusion_model.blocks.0.cross_attn.o", "blocks.0.cross_attn.o", 1.0),
        ("lora_unet_blocks_1_self_attn_v", "blocks.1.self_attn.v", 4.0),
    ],
    "ltxv": [
        (
            "diffusion_model.transformer_blocks.0.attn1.to_q",
            "transformer_blocks.0.attn1.to_q",
            None,
        ),
        ("transformer_blocks.1.ff.net.2", "transformer_blocks.1.ff.net.2", 2.0),
    ],
    "hunyuan-video": [
        ("transformer.double_blocks.0.img_attn_qkv", "double_blocks.0.img_attn.qkv", None),
        ("diffusion_model.double_blocks.1.txt_mlp.fc1", "double_blocks.1.txt_mlp.0", None),
        ("diffusion_model.single_blocks.0.linear1", "single_blocks.0.linear1", 1.0),
    ],
    "ace-step15": [
        ("base_model.model.layers.0.self_attn.q_proj", "decoder.layers.0.self_attn.q_proj", None),
        ("lycoris_layers_1_mlp_down_proj", "decoder.layers.1.mlp.down_proj", 1.0),
    ],
    "flux3": [
        ("diffusion_model.single_blocks.1.attn_out", "single_blocks.1.attn_out", None),
        ("lora_unet_txt_mode_blocks_0_mlp_in", "txt_mode_blocks.0.mlp_in", 2.0),
    ],
    "moss-image": [
        ("layers.1.proj.ffn_down", "layers.1.proj.ffn_down", None),
        ("diffusion_model.layers.0.proj.o", "layers.0.proj.o", 1.0),
    ],
    "hunyuan3d": [
        ("lora_unet_double_blocks_0_img_attn_qkv", "double_blocks.0.img_attn.qkv", None),
        ("transformer.single_transformer_blocks.0.proj_out", "single_blocks.0.linear2", None),
    ],
    "qwen-text": [
        ("lora_te_layers_0_self_attn_q_proj", "layers.0.self_attn.q_proj", 2.0),
        (
            "text_encoders.qwen3_4b.transformer.model.layers.1.mlp.up_proj",
            "layers.1.mlp.up_proj",
            None,
        ),
        ("base_model.model.model.layers.1.self_attn.o_proj", "layers.1.self_attn.o_proj", None),
    ],
}


@pytest.mark.parametrize("family", sorted(FORMATS))
def test_import_each_family_key_format(family, tmp_path):
    model = _model(family)
    modules = dict(model.named_modules())
    gen = torch.Generator().manual_seed(5)
    sd, want = {}, {}
    for stem, target, alpha in FORMATS[family]:
        m = modules[target]
        a, b = _ab(gen, m.in_features, m.out_features)
        kohya = stem.startswith(("lora_", "lycoris_"))
        down, up = (
            (".lora_down.weight", ".lora_up.weight")
            if kohya
            else (".lora_A.weight", ".lora_B.weight")
        )
        sd[stem + down], sd[stem + up] = a, b
        if alpha is not None:
            sd[stem + ".alpha"] = torch.tensor(alpha)
        want[target] = (b @ a) * (alpha / 2 if alpha is not None else 1.0)
    if family == "qwen-text":
        sd["lora_unet_blocks_0_q.lora_down.weight"] = torch.randn(2, H)
        sd["lora_unet_blocks_0_q.lora_up.weight"] = torch.randn(H, 2)
    result = import_loras([_save(tmp_path, family, sd)], model, family=family)
    assert sorted(result.adapter_set.experts) == sorted(want)
    for target, delta in want.items():
        torch.testing.assert_close(_delta(result.adapter_set, target), delta)


def test_older_families_keep_their_import():
    sd = {"transformer.layers.0.attention.to_q.lora_A.weight": torch.zeros(1, 2)}
    out, skipped, *_ = normalize_for("z-image", sd, nn.Linear(2, 2))
    assert out == sd and skipped == []
    out, skipped, *_ = normalize_for(None, sd, nn.Linear(2, 2))
    assert out == sd and skipped == []


def test_every_family_has_a_preset_and_a_binding():
    for family, spec in FAMILIES.items():
        assert family in BINDINGS
        assert get_preset(family).router_seed == spec.router_seed


def test_moss_image_q_k_v_keys_land_on_the_fused_qkv(tmp_path):
    model = _model("moss-image")
    gen = torch.Generator().manual_seed(6)
    a, b = _ab(gen, H, H)
    sd = {
        "lora_unet_layers_1_proj_k.lora_down.weight": a,
        "lora_unet_layers_1_proj_k.lora_up.weight": b,
    }
    result = import_loras([_save(tmp_path, "moss", sd)], model, family="moss-image")
    want = torch.zeros(3 * H, H)
    want[H : 2 * H] = b @ a
    torch.testing.assert_close(_delta(result.adapter_set, "layers.1.proj.qkv"), want)


def test_flux3_projection_keys_land_on_the_fused_qkv(tmp_path):
    model = _model("flux3")
    gen = torch.Generator().manual_seed(8)
    a, b = _ab(gen, H, H)
    stem = "diffusion_model.content_mode_blocks.image.1.v_proj"
    sd = {f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b}
    result = import_loras([_save(tmp_path, "f3", sd)], model, family="flux3")
    want = torch.zeros(3 * H, H)
    want[2 * H :] = b @ a
    torch.testing.assert_close(_delta(result.adapter_set, "content_mode_blocks.image.1.qkv"), want)


def test_chroma_uses_the_flux_family():
    from test_families import _tree as tree

    model = tree("Chroma", _chroma())
    spec = detect(model)
    assert spec.family == "flux"
    targets = spec.targets(model)
    assert len(targets) == 20 and not any("distilled" in t for t in targets)


def _klein(blocks=2, singles=2):
    layers = {"emb_in.image": (4, H), "emb_in.image_cond": (4, H), "txt_in": (6, H)}
    for stream in ("early_stream", "single_stream"):
        for mode in ("image", "image_cond", "txt"):
            layers[f"{stream}_modulations.{mode}.lin"] = (H, 3 * H)
    layers["final_layer.image.linear"] = (H, 4)
    layers["final_layer.image.adaLN_modulation.1"] = (H, 2 * H)
    stacks = [f"content_mode_blocks.image.{i}" for i in range(blocks)]
    stacks += [f"content_mode_blocks.image_cond.{i}" for i in range(blocks)]
    stacks += [f"txt_mode_blocks.{i}" for i in range(blocks)]
    stacks += [f"single_blocks.{i}" for i in range(singles)]
    for b in stacks:
        for p in ("q_proj", "k_proj", "v_proj", "attn_out"):
            layers[f"{b}.{p}"] = (H, H)
        layers[f"{b}.mlp_in"] = (H, 2 * MLP)
        layers[f"{b}.mlp_out"] = (MLP, H)
    return layers


def test_klein_targets_and_diffusers_lora_names(tmp_path):
    from test_families import _tree as tree

    model = tree("Flux3", _klein())
    spec = detect(model)
    assert spec.family == "flux3"
    assert len(spec.targets(model)) == 8 * 6
    gen = torch.Generator().manual_seed(9)
    names = {
        "transformer.mode_blocks.image.1.attn.to_k": "content_mode_blocks.image.1.k_proj",
        "transformer.mode_blocks.image_cond.0.ff.linear_in": (
            "content_mode_blocks.image_cond.0.mlp_in"
        ),
        "transformer.single_transformer_blocks.1.attn.to_out.0": "single_blocks.1.attn_out",
        "transformer.txt_mode_blocks.0.ff.linear_out": "txt_mode_blocks.0.mlp_out",
    }
    sd, want = {}, {}
    for stem, target in names.items():
        m = model.get_submodule(target)
        a, b = _ab(gen, m.in_features, m.out_features)
        sd[f"{stem}.lora_A.weight"], sd[f"{stem}.lora_B.weight"] = a, b
        want[target] = b @ a
    for stem in ("transformer.early_stream_modulations.txt.linear", "transformer.norm_out.linear"):
        sd[f"{stem}.lora_A.weight"] = torch.randn(2, H, generator=gen)
        sd[f"{stem}.lora_B.weight"] = torch.randn(
            3 * H if "modulations" in stem else 2 * H, 2, generator=gen
        )
    result = import_loras([_save(tmp_path, "tarot", sd)], model, family="flux3")
    assert sorted(result.adapter_set.experts) == sorted(want)
    for target, delta in want.items():
        torch.testing.assert_close(_delta(result.adapter_set, target), delta)
    assert sorted(path for section in result.sections for path in section.data) == [
        "early_stream_modulations.txt.lin",
        "final_layer.image.adaLN_modulation.1",
    ]
    assert {section.meta["mount_kind"] for section in result.sections} == {"other", "final"}
    assert result.unmatched["tarot"] == []


@pytest.mark.parametrize("family", ["moss-image", "z-image"])
def test_single_stream_image_rows(family):
    from kilora.family_spec import map_region

    spec = get_family(family)
    target = "layers.0.proj.qkv" if family == "moss-image" else "layers.0.attention.to_q"
    layout = spec.layout(target)
    latent = (4, 4)
    image_count = 16 if family == "moss-image" else 4
    image = torch.arange(image_count * H).reshape(image_count, H).float()
    selected = []
    for text_count in (2, 7):
        padded_count = ((image_count + layout.pad - 1) // layout.pad) * layout.pad
        padding = torch.full((padded_count - image_count, H), -2.0)
        text = torch.full((text_count, H), -1.0)
        parts = [text, image, padding] if layout.text == "before" else [image, padding, text]
        packed = torch.cat(parts)
        rows = spec.image_token_rows(len(packed), latent)
        selected.append(packed[rows])
        region = map_region(torch.ones(latent), len(packed), layout, latent)
        torch.testing.assert_close(region[0, rows], torch.ones(image_count))
        assert int(region.sum()) == image_count
    for rows in selected:
        torch.testing.assert_close(rows, image)
    assert selected[0].shape == selected[1].shape


def test_image_rows_without_layout():
    from kilora.family_spec import FamilySpec

    assert FamilySpec("unknown").image_token_rows(10, (4, 4)) is None


@pytest.mark.parametrize("family", ["flux", "flux3", "qwen-image", "wan", "anima"])
def test_image_rows_require_one_block_stream(family):
    assert get_family(family).image_token_rows(20, (4, 4)) is None
