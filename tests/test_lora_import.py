# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file

from kilora import LoRAImportError, compose, import_lora, load_mod
from kilora.testing import TinyDenoiser

pytestmark = pytest.mark.lora_parity

MODULES = ["layers.0.attention.to_q", "layers.1.attention.to_out.0", "layers.0.feed_forward.w1"]


def lora_weights(model, rank=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name in MODULES:
        m = model.get_submodule(name)
        out[name] = (
            torch.randn(rank, m.in_features, generator=gen) * 0.3,
            torch.randn(m.out_features, rank, generator=gen) * 0.3,
        )
    return out


def peft_keys(f, prefix="transformer."):
    sd = {}
    for name, (a, b) in f.items():
        sd[f"{prefix}{name}.lora_A.weight"] = a
        sd[f"{prefix}{name}.lora_B.weight"] = b
    return sd


def kohya_keys(f, alpha):
    sd = {}
    for name, (a, b) in f.items():
        stem = "lora_unet_" + name.replace(".", "_")
        sd[f"{stem}.lora_down.weight"] = a
        sd[f"{stem}.lora_up.weight"] = b
        sd[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return sd


def expected_outputs(model, f, scale, inputs):
    out = {}
    for name, (a, b) in f.items():
        m = model.get_submodule(name)
        x = inputs[name]
        out[name] = m(x) + scale * (x @ a.t() @ b.t())
    return out


def module_inputs(model, dtype=torch.float32):
    gen = torch.Generator().manual_seed(9)
    return {
        n: torch.randn(2, 5, model.get_submodule(n).in_features, generator=gen).to(dtype)
        for n in MODULES
    }


def check_modules(model, adapter, f, scale, dtype=torch.float32, tol=1e-5):
    inputs = module_inputs(model, dtype)
    with adapter.depth(0), torch.no_grad():
        model_ref = TinyDenoiser(seed=0).to(dtype)
        expected = expected_outputs(
            model_ref, {k: (a.to(dtype), b.to(dtype)) for k, (a, b) in f.items()}, scale, inputs
        )
    with torch.no_grad():
        for name in MODULES:
            got = model.get_submodule(name)(inputs[name])
            want = expected[name]
            err = ((got.float() - want.float()).norm() / want.float().norm()).item()
            assert err < tol, (name, err)


@pytest.fixture
def model():
    return TinyDenoiser(seed=0).eval()


@pytest.mark.parametrize("prefix", ["transformer.", "", "unet."])
def test_peft_layout_parity(model, prefix):
    f = lora_weights(model)
    result = import_lora(peft_keys(f, prefix), model)
    assert result.layout == "peft" and result.adapter_set.strength == 1.0
    adapter = compose(model, [result.adapter_set], mix="stack")
    assert adapter.current_depth == 1
    check_modules(model, adapter, f, 1.0)


def test_peft_layout_with_adapter_config_scale(model, tmp_path):
    f = lora_weights(model, rank=4)
    save_file(peft_keys(f), str(tmp_path / "lora.safetensors"))
    (tmp_path / "adapter_config.json").write_text('{"r": 4, "lora_alpha": 2}')
    result = import_lora(tmp_path / "lora.safetensors", model, weight=0.5)
    assert result.scale == 0.5 and result.adapter_set.strength == 0.25
    adapter = compose(model, [result.adapter_set], mix="stack")
    check_modules(model, adapter, f, 0.25)


def test_old_diffusers_layout_parity(model):
    f = lora_weights(model)
    sd = {}
    for name, (a, b) in f.items():
        sd[f"unet.{name}.lora.down.weight"] = a
        sd[f"unet.{name}.lora.up.weight"] = b
    adapter = compose(model, [import_lora(sd, model).adapter_set], mix="stack")
    check_modules(model, adapter, f, 1.0)


def test_kohya_layout_parity_with_alpha(model):
    f = lora_weights(model, rank=4)
    result = import_lora(kohya_keys(f, alpha=2.0), model)
    assert result.layout == "kohya" and result.scale == 0.5
    adapter = compose(model, [result.adapter_set], mix="stack")
    check_modules(model, adapter, f, 0.5)


def test_comfyui_layout_parity(model):
    f = lora_weights(model)
    sd = peft_keys(f, prefix="diffusion_model.")
    result = import_lora(sd, model)
    assert result.layout == "comfyui"
    check_modules(model, compose(model, [result.adapter_set], mix="stack"), f, 1.0)


def test_fused_qkv_and_out_names_from_the_original_module_tree(model):
    gen = torch.Generator().manual_seed(4)
    a = torch.randn(2, 16, generator=gen)
    b = torch.randn(48, 2, generator=gen)
    a_out = torch.randn(2, 16, generator=gen)
    b_out = torch.randn(16, 2, generator=gen)
    sd = {
        "diffusion_model.layers.0.attention.qkv.lora_A.weight": a,
        "diffusion_model.layers.0.attention.qkv.lora_B.weight": b,
        "diffusion_model.layers.0.attention.out.lora_A.weight": a_out,
        "diffusion_model.layers.0.attention.out.lora_B.weight": b_out,
    }
    with pytest.raises(LoRAImportError, match="no module"):
        import_lora(sd, model)
    result = import_lora(sd, model, family="z-image")
    assert sorted(result.adapter_set.experts) == [
        "layers.0.attention.to_k",
        "layers.0.attention.to_out.0",
        "layers.0.attention.to_q",
        "layers.0.attention.to_v",
    ]
    compose(model, [result.adapter_set], mix="stack")
    x = torch.randn(3, 16)
    attn = model.layers[0].attention
    fused = x @ a.t() @ b.t()
    for i, proj in enumerate([attn.to_q, attn.to_k, attn.to_v]):
        want = torch.nn.functional.linear(x, proj.weight) + fused[:, 16 * i : 16 * (i + 1)]
        torch.testing.assert_close(proj(x), want, rtol=1e-5, atol=1e-5)
    kohya = {
        "lora_unet_layers_0_attention_qkv.lora_down.weight": a,
        "lora_unet_layers_0_attention_qkv.lora_up.weight": b,
    }
    assert len(import_lora(kohya, model, family="z-image").adapter_set.experts) == 3


def test_bf16_parity(model):
    f = lora_weights(model)
    model = model.to(torch.bfloat16)
    adapter = compose(model, [import_lora(peft_keys(f), model).adapter_set], mix="stack")
    check_modules(model, adapter, f, 1.0, dtype=torch.bfloat16, tol=1e-2)


def test_two_loras_add_with_their_weights(model):
    f1, f2 = lora_weights(model, seed=1), lora_weights(model, seed=2)
    one = import_lora(peft_keys(f1), model, name="one", weight=0.7).adapter_set
    two = import_lora(peft_keys(f2), model, name="two", weight=-0.4).adapter_set
    compose(model, [one, two], mix="stack")
    inputs = module_inputs(model)
    ref = TinyDenoiser(seed=0)
    with torch.no_grad():
        for name in MODULES:
            x = inputs[name]
            a1, b1 = f1[name]
            a2, b2 = f2[name]
            want = (
                ref.get_submodule(name)(x)
                + 0.7 * (x @ a1.t() @ b1.t())
                - 0.4 * (x @ a2.t() @ b2.t())
            )
            torch.testing.assert_close(model.get_submodule(name)(x), want, rtol=1e-5, atol=1e-5)


def test_imported_set_defaults_to_depth_one_and_is_trainable(model):
    result = import_lora(peft_keys(lora_weights(model)), model)
    imported = result.adapter_set
    assert not imported.frozen and imported.extra["trained_with_kilora"] is False
    assert imported.aggregation == "accumulate" and imported.num_experts == 4
    assert "Train it before sharing" in result.report()
    adapter = compose(model, [imported], mix="stack")
    assert adapter.current_depth == 1
    assert any(p.requires_grad for p in adapter.parameters())
    adapter.set_depth(4)
    assert adapter.current_depth == 4


def test_text_encoder_keys_are_skipped_and_reported(model):
    sd = peft_keys(lora_weights(model))
    sd["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = torch.zeros(2, 4)
    result = import_lora(sd, model)
    assert len(result.skipped_keys) == 1
    assert "text encoder" in result.report()


@pytest.mark.parametrize(
    "key, match",
    [
        ("transformer.layers.0.attention.to_q.hada_w1_a", "not a plain LoRA"),
        ("transformer.layers.0.attention.to_q.dora_scale", "not a plain LoRA"),
    ],
)
def test_non_plain_formats_are_refused(model, key, match):
    sd = peft_keys(lora_weights(model))
    sd[key] = torch.zeros(2)
    with pytest.raises(LoRAImportError, match=match):
        import_lora(sd, model)


def test_wrong_shapes_and_conv_are_refused(model):
    sd = {
        "transformer.layers.0.attention.to_q.lora_A.weight": torch.zeros(2, 7),
        "transformer.layers.0.attention.to_q.lora_B.weight": torch.zeros(16, 2),
    }
    with pytest.raises(LoRAImportError, match="do not fit"):
        import_lora(sd, model)
    sd = {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": torch.zeros(2, 16, 1, 1),
        "lora_unet_layers_0_attention_to_q.lora_up.weight": torch.zeros(16, 2, 1, 1),
    }
    with pytest.raises(LoRAImportError, match="linear layers"):
        import_lora(sd, model)


def test_pickle_lora_is_refused(model, tmp_path):
    torch.save(peft_keys(lora_weights(model)), tmp_path / "lora.safetensors")
    with pytest.raises(LoRAImportError, match="not a safetensors file"):
        import_lora(tmp_path / "lora.safetensors", model)


def test_import_record_and_mod_round_trip(model, tmp_path):
    f = lora_weights(model)
    sd = kohya_keys(f, alpha=3.0)
    save_file(
        sd,
        str(tmp_path / "My Style.safetensors"),
        {"modelspec.license": "CC-BY-4.0", "ss_dataset_dirs": "/secret/path"},
    )
    result = import_lora(tmp_path / "My Style.safetensors", model)
    adapter = compose(model, [result.adapter_set], mix="stack")
    adapter.save(tmp_path / "style.kmod", id="style")
    record = load_mod(tmp_path / "style.kmod").sets[0].imported_from
    assert record["format"] == "kohya"
    assert record["original_name"] == "My Style.safetensors"
    assert record["original_license"] == "CC-BY-4.0"
    assert record["scale"] == 1.0
    assert record["digest"].startswith("sha256-")
    assert "secret" not in (tmp_path / "style.kmod").read_bytes().decode("latin-1")
    mod = load_mod(tmp_path / "style.kmod")
    assert mod.manifest["requires_features"] == ["experts.lora"]
    assert mod.manifest["depth"]["recommended"] == 1
    adapter.remove()
    again = compose(model, [tmp_path / "style.kmod"], mix="stack")
    check_modules(model, again, f, 1.0)


def test_imported_set_composes_with_keyed_set(model):
    from conftest import TARGET, randomize_up

    from kilora import KiLoRAConfig, attach

    x = torch.randn(1, 4, 4)
    keyed = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=4, router_dim=8))
    randomize_up(keyed)
    keyed_set = keyed.to_sets()[0]
    keyed.remove()
    f = lora_weights(model)
    imported = import_lora(peft_keys(f), model, name="imp").adapter_set
    both = compose(model, [keyed_set, imported], depth=2)
    assert both.current_depth == 2
    assert both.settings.softmax_scope == "per_set"
    model(x)


@pytest.mark.parametrize("fixed_read", [False, True])
def test_untrained_import_takes_depth_without_an_output_gate(monkeypatch, fixed_read):
    from kilora import compose
    from kilora.testing import TinyDenoiser

    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1" if fixed_read else "0")
    model = TinyDenoiser(seed=0).eval()
    gen = torch.Generator().manual_seed(1)
    target = next(n for n, m in model.named_modules() if n.endswith("feed_forward.w1"))
    module = dict(model.named_modules())[target]
    a = torch.randn(2, module.in_features, generator=gen)
    b = torch.randn(module.out_features, 2, generator=gen)
    sd = {f"{target}.lora_A.weight": a, f"{target}.lora_B.weight": b}
    imported = import_lora(sd, model).adapter_set
    assert imported.form == "input"
    x = torch.randn(3, module.in_features, generator=gen)
    with torch.no_grad():
        base = module(x)
        change = (x @ a.t()) @ b.t()
        for iteration, times in (("accumulate", [1, 2, 3]), ("path", [1, 1, 1])):
            adapter = compose(model, [imported], mix="stack")
            adapter.settings.iteration = iteration
            for depth, k in zip((1, 2, 3), times, strict=True):
                adapter.set_depth(depth)
                actual = module(x)
                if fixed_read or iteration == "path" or depth == 1:
                    torch.testing.assert_close(actual, base + k * change, rtol=1e-5, atol=1e-5)
                else:
                    assert (actual - base).norm() <= 2 * change.norm() + 1e-4
            adapter.remove()


def test_conversion_to_state_form_is_exact_for_square_and_tall_layers():
    from kilora.lora_import import to_state_form

    gen = torch.Generator().manual_seed(2)
    model = torch.nn.Module()
    model.square = torch.nn.Linear(8, 8, bias=False)
    model.tall = torch.nn.Linear(6, 10, bias=False)
    model.wide = torch.nn.Linear(10, 6, bias=False)
    sd = {}
    for name, m in (("square", model.square), ("tall", model.tall), ("wide", model.wide)):
        sd[f"{name}.lora_A.weight"] = torch.randn(2, m.in_features, generator=gen)
        sd[f"{name}.lora_B.weight"] = torch.randn(m.out_features, 2, generator=gen)
    imported = import_lora(sd, model, experts=1).adapter_set
    state, lost = to_state_form(imported, model, ridge=1e-8)
    assert state.form == "state"
    assert lost["square"] < 1e-3 and lost["tall"] < 1e-3
    assert lost["wide"] > 0.05
    for name in ("square", "tall"):
        m = getattr(model, name)
        x = torch.randn(4, m.in_features, generator=gen)
        a_state = state.experts[name].down[0, 0]
        a_plain = imported.experts[name].down[0, 0]
        torch.testing.assert_close((m(x) @ a_state.t()), x @ a_plain.t(), rtol=1e-3, atol=1e-3)
