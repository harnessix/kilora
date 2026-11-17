# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_families import MLP, H, _ab, _model


def test_comfyui_applies_a_diffusers_flux_lora_exactly(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    model = _model("flux")
    gen = torch.Generator().manual_seed(1)
    sd, weights = {}, {}
    for proj in ("to_q", "to_v"):
        a, b = _ab(gen, H, H)
        weights[proj] = (a, b)
        sd[f"transformer.transformer_blocks.1.attn.{proj}.lora_A.weight"] = a
        sd[f"transformer.transformer_blocks.1.attn.{proj}.lora_B.weight"] = b
    save_file(sd, str(tmp_path / "flux.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("flux.safetensors", 1.0)
    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, output_gate="none"
    )
    assert list(patched.object_patches) == ["diffusion_model.double_blocks.1.img_attn.qkv"]
    qkv = patched.object_patches["diffusion_model.double_blocks.1.img_attn.qkv"]
    x = torch.randn(2, 3, H)
    with torch.no_grad():
        got = qkv(x)
        want = model.get_submodule("double_blocks.1.img_attn.qkv")(x).clone()
        a, b = weights["to_q"]
        want[..., 0:H] += x @ a.t() @ b.t()
        a, b = weights["to_v"]
        want[..., 2 * H : 3 * H] += x @ a.t() @ b.t()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_comfyui_applies_modulation_keys_as_a_fixed_plain_lora(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui.patching import clear
    from kilora.fixed_change import is_fixed

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    model = _model("flux")
    gen = torch.Generator().manual_seed(2)
    a1, b1 = _ab(gen, H, 6 * H)
    a2, b2 = _ab(gen, H, 6 * H, rank=4)
    a3, b3 = _ab(gen, H, H)
    sd = {
        "transformer.transformer_blocks.0.norm1.linear.lora_A.weight": a1,
        "transformer.transformer_blocks.0.norm1.linear.lora_B.weight": b1,
        "lora_unet_double_blocks_1_txt_mod_lin.lora_down.weight": a2,
        "lora_unet_double_blocks_1_txt_mod_lin.lora_up.weight": b2,
        "lora_unet_double_blocks_1_txt_mod_lin.alpha": torch.tensor(2.0),
        "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": a3,
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": b3,
    }
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "mod.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("mod.safetensors", 0.5)
    patcher = FakePatcher(FakeBaseModel(model))
    patched, report, _ = N.KiLoRAApply().apply(patcher, chain, -1)
    fixed = {k for k, v in patched.object_patches.items() if is_fixed(v)}
    assert fixed == {
        "diffusion_model.double_blocks.0.img_mod.lin",
        "diffusion_model.double_blocks.1.txt_mod.lin",
    }
    assert "diffusion_model.double_blocks.0.img_attn.proj" in patched.object_patches
    x = torch.randn(2, H)
    with torch.no_grad():
        got = patched.object_patches["diffusion_model.double_blocks.0.img_mod.lin"](x)
        base = model.get_submodule("double_blocks.0.img_mod.lin")
        want = base(x) + 0.5 * x @ a1.t() @ b1.t()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        got = patched.object_patches["diffusion_model.double_blocks.1.txt_mod.lin"](x)
        base = model.get_submodule("double_blocks.1.txt_mod.lin")
        want = base(x) + 0.5 * 0.5 * x @ a2.t() @ b2.t()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    wrapper = patched.object_patches["diffusion_model.double_blocks.0.img_mod.lin"]
    assert wrapper.base is model.get_submodule("double_blocks.0.img_mod.lin")
    assert not any(is_fixed(v) for v in clear(patched).object_patches.values())


def test_anima_adaln_keys_of_a_kohya_lora_apply_as_a_fixed_change(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _tree

    from kilora.comfyui import nodes as N
    from kilora.fixed_change import is_fixed
    from kilora.lora_import import import_lora

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    layers = {f"blocks.0.self_attn.{p}": (H, H) for p in ("q_proj", "k_proj", "v_proj")}
    layers["blocks.0.self_attn.output_proj"] = (H, H)
    layers["blocks.0.mlp.layer1"] = (H, MLP)
    layers["blocks.0.mlp.layer2"] = (MLP, H)
    layers["blocks.0.adaln_modulation_cross_attn.1"] = (H, 3 * H)
    model = _tree("Anima", layers)
    gen = torch.Generator().manual_seed(3)
    a1, b1 = _ab(gen, H, H)
    a2, b2 = _ab(gen, H, 3 * H)
    sd = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": a1,
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": b1,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_down.weight": a2,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_up.weight": b2,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.alpha": torch.tensor(1.0),
    }
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "style.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("style.safetensors", 0.5)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(model)), chain, -1)
    path = "diffusion_model.blocks.0.adaln_modulation_cross_attn.1"
    assert is_fixed(patched.object_patches[path])
    x = torch.randn(2, H)
    with torch.no_grad():
        got = patched.object_patches[path](x)
        base = model.get_submodule("blocks.0.adaln_modulation_cross_attn.1")
        want = base(x) + 0.5 * 0.5 * x @ a2.t() @ b2.t()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    alone = import_lora(sd, _tree("Anima", {k: v for k, v in layers.items() if "adaln" not in k}))
    assert not alone.sections


@pytest.mark.parametrize("form", ["comfy", "diffusers"])
def test_klein_lora_formats_match_a_plain_merge(form, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _klein, _tree

    from kilora.comfyui import nodes as N

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    model = _tree("Flux3", _klein())
    gen = torch.Generator().manual_seed(11)
    pairs = {
        "content_mode_blocks.image.0.q_proj": "mode_blocks.image.0.attn.to_q",
        "content_mode_blocks.image.1.attn_out": "mode_blocks.image.1.attn.to_out.0",
        "txt_mode_blocks.0.mlp_in": "txt_mode_blocks.0.ff.linear_in",
        "single_blocks.1.mlp_out": "single_transformer_blocks.1.ff.linear_out",
    }
    if form == "diffusers":
        pairs["single_stream_modulations.image.lin"] = "single_stream_modulations.image.linear"
        pairs["final_layer.image.adaLN_modulation.1"] = "norm_out.linear"
    sd, want = {}, {}
    for comfy, diffusers in pairs.items():
        m = model.get_submodule(comfy)
        a, b = _ab(gen, m.in_features, m.out_features, rank=4)
        stem = f"diffusion_model.{comfy}" if form == "comfy" else f"transformer.{diffusers}"
        sd[f"{stem}.lora_A.weight"], sd[f"{stem}.lora_B.weight"] = a, b
        sd[f"{stem}.alpha"] = torch.tensor(2.0)
        want[comfy] = b @ a * 0.5
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "k.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("k.safetensors", 1.0)
    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, output_gate="none"
    )
    assert sorted(p.removeprefix("diffusion_model.") for p in patched.object_patches) == sorted(
        want
    )
    for comfy, delta in want.items():
        layer = patched.object_patches[f"diffusion_model.{comfy}"]
        x = torch.randn(2, 3, layer.in_features, generator=gen)
        with torch.no_grad():
            base = model.get_submodule(comfy)(x)
            torch.testing.assert_close(layer(x), base + x @ delta.t(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("plain_too", [False, True], ids=["lokr-only", "lokr-and-lora"])
def test_plain_stack_applies_lokr_layers_exactly(plain_too, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    model = _model("flux")
    gen = torch.Generator().manual_seed(21)
    sd, want = {}, {}
    for path in ("double_blocks.0.img_attn.proj", "single_blocks.1.linear2"):
        m = model.get_submodule(path)
        w1 = torch.randn(2, 4, generator=gen)
        w2 = torch.randn(m.out_features // 2, m.in_features // 4, generator=gen)
        stem = "lora_unet_" + path.replace(".", "_")
        sd[f"{stem}.lokr_w1"], sd[f"{stem}.lokr_w2"] = w1, w2
        sd[f"{stem}.alpha"] = torch.tensor(1.0)
        want[path] = torch.kron(w1, w2)
    if plain_too:
        m = model.get_submodule("double_blocks.1.txt_mlp.0")
        a, b = _ab(gen, m.in_features, m.out_features)
        sd["lora_unet_double_blocks_1_txt_mlp_0.lora_down.weight"] = a
        sd["lora_unet_double_blocks_1_txt_mlp_0.lora_up.weight"] = b
        want["double_blocks.1.txt_mlp.0"] = b @ a
    path = tmp_path / "lokr.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    chain = patching.ModChain((patching.ModEntry("lora", str(path), 0.7),))
    out, _ = patching.apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, gate=("none", None), joint=False
    )
    assert sorted(p.removeprefix("diffusion_model.") for p in out.object_patches) == sorted(want)
    for layer_path, delta in want.items():
        layer = out.object_patches[f"diffusion_model.{layer_path}"]
        x = torch.randn(2, 3, layer.in_features, generator=gen)
        with torch.no_grad():
            base = model.get_submodule(layer_path)(x)
            torch.testing.assert_close(layer(x), base + 0.7 * x @ delta.t(), rtol=1e-4, atol=1e-4)


def test_kron_change_equals_the_full_kronecker_product():
    from kilora.sections import KronChange

    gen = torch.Generator().manual_seed(3)
    w1, w2 = torch.randn(3, 2, generator=gen), torch.randn(5, 7, generator=gen)
    x = torch.randn(4, 6, 14, generator=gen)
    got = KronChange(w1, w2, 0.5).add(x, torch.zeros(4, 6, 15))
    torch.testing.assert_close(got, 0.5 * x @ torch.kron(w1, w2).t(), rtol=1e-5, atol=1e-5)


def test_plain_stack_runs_one_set_per_layer(tmp_path):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    model = _model("flux")
    gen = torch.Generator().manual_seed(31)
    paths, want = [], {}
    for k, strength in enumerate((0.8, -0.3)):
        sd = {}
        for layer in ("double_blocks.0.img_attn.proj", "single_blocks.1.linear2"):
            m = model.get_submodule(layer)
            a, b = _ab(gen, m.in_features, m.out_features, rank=2 + k)
            stem = "lora_unet_" + layer.replace(".", "_")
            sd[f"{stem}.lora_down.weight"], sd[f"{stem}.lora_up.weight"] = a, b
            want[layer] = want.get(layer, 0) + strength * (b @ a)
        path = tmp_path / f"l{k}.safetensors"
        save_file(sd, str(path))
        paths.append((str(path), strength))
    chain = patching.ModChain(tuple(patching.ModEntry("lora", p, s) for p, s in paths))
    out, _ = patching.apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, gate=("none", None), joint=False
    )
    for layer, delta in want.items():
        baked = out.object_patches[f"diffusion_model.{layer}"]
        plans = [p for _, _, p in baked.__dict__["_kilora_plan"].parts]
        assert len(plans) == 1 and len(plans[0].sets) == 1
        x = torch.randn(2, 3, baked.in_features, generator=gen)
        with torch.no_grad():
            base = model.get_submodule(layer)(x)
            torch.testing.assert_close(baked(x), base + x @ delta.t(), rtol=1e-4, atol=1e-4)


def test_routed_lokr_reports_the_energy_kept_in_the_apply_report(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = _model("flux")
    gen = torch.Generator().manual_seed(41)
    layer = "double_blocks.0.img_attn.proj"
    m = model.get_submodule(layer)
    stem = "lora_unet_" + layer.replace(".", "_")
    a, b = _ab(gen, m.in_features, m.out_features, rank=1)
    plain = {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b}
    lokr = {
        f"{stem}.lokr_w1": torch.randn(2, 4, generator=gen),
        f"{stem}.lokr_w2": torch.randn(m.out_features // 2, m.in_features // 4, generator=gen),
        f"{stem}.alpha": torch.tensor(1.0),
    }
    entries = []
    for name, sd in (("plain", plain), ("lokr", lokr)):
        save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / f"{name}.st"))
        entries.append(patching.ModEntry("lora", str(tmp_path / f"{name}.st"), 1.0))
    _, report = patching.apply(
        FakePatcher(FakeBaseModel(model)), patching.ModChain(tuple(entries)), 1, joint=True
    )
    assert "lycoris_apply=truncated" in report and "energy kept by the SVD cut" in report
    assert "Warning" in report and "lycoris_apply=exact applies the whole change" in report
    _, report = patching.apply(
        FakePatcher(FakeBaseModel(model)),
        patching.ModChain(tuple(entries)),
        1,
        joint=True,
        lycoris_apply="exact",
    )
    assert "lycoris_apply=exact" in report and "energy kept 100% (exact change)" in report
    assert "energy kept by the SVD cut" not in report and "Warning" not in report


def test_plain_stack_keeps_a_lora_and_a_lokr_on_the_same_layers(tmp_path):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    model = _model("flux")
    gen = torch.Generator().manual_seed(51)
    layers = ("double_blocks.0.img_attn.proj", "single_blocks.1.linear2")
    plain, lokr, want = {}, {}, {}
    for layer in layers:
        m = model.get_submodule(layer)
        stem = "lora_unet_" + layer.replace(".", "_")
        a, b = _ab(gen, m.in_features, m.out_features)
        plain[f"{stem}.lora_down.weight"], plain[f"{stem}.lora_up.weight"] = a, b
        w1 = torch.randn(2, 4, generator=gen)
        w2 = torch.randn(m.out_features // 2, m.in_features // 4, generator=gen)
        lokr[f"{stem}.lokr_w1"], lokr[f"{stem}.lokr_w2"] = w1, w2
        lokr[f"{stem}.alpha"] = torch.tensor(1.0)
        want[layer] = 0.9 * (b @ a) + 0.6 * torch.kron(w1, w2)
    entries = []
    for name, sd, strength in (("plain", plain, 0.9), ("lokr", lokr, 0.6)):
        path = tmp_path / f"{name}.safetensors"
        save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
        entries.append(patching.ModEntry("lora", str(path), strength))
    out, _ = patching.apply(
        FakePatcher(FakeBaseModel(model)),
        patching.ModChain(tuple(entries)),
        1,
        gate=("none", None),
        joint=False,
    )
    for layer, delta in want.items():
        patched = out.object_patches[f"diffusion_model.{layer}"]
        x = torch.randn(2, 3, patched.in_features, generator=gen)
        with torch.no_grad():
            base = model.get_submodule(layer)(x)
            torch.testing.assert_close(patched(x), base + x @ delta.t(), rtol=1e-4, atol=1e-4)
    again, _ = patching.apply(out, patching.ModChain(tuple(entries)), 1, gate=("none", None))
    patched = again.object_patches[f"diffusion_model.{layers[0]}"]
    x = torch.randn(2, 3, patched.in_features, generator=gen)
    with torch.no_grad():
        base = model.get_submodule(layers[0])(x)
        torch.testing.assert_close(patched(x), base + x @ want[layers[0]].t(), rtol=1e-4, atol=1e-4)


def test_joint_apply_scales_global_mounts_and_reports_keys(tmp_path):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _tree

    from kilora.comfyui import patching

    layers = {f"blocks.0.self_attn.{p}": (H, H) for p in ("q_proj", "k_proj", "v_proj")}
    layers["blocks.0.self_attn.output_proj"] = (H, H)
    layers["blocks.0.mlp.layer1"] = (H, MLP)
    layers["blocks.0.mlp.layer2"] = (MLP, H)
    layers["blocks.0.adaln_modulation_cross_attn.1"] = (H, 3 * H)
    layers["llm_adapter.proj"] = (H, H)
    model = _tree("Anima", layers)
    gen = torch.Generator().manual_seed(77)
    paths, states = [], []
    for i in range(2):
        state = {}
        for path, dim in [
            ("blocks_0_self_attn_q_proj", H),
            ("blocks_0_adaln_modulation_cross_attn_1", 3 * H),
            ("llm_adapter_proj", H),
        ]:
            a, b = _ab(gen, H, dim)
            state[f"lora_unet_{path}.lora_down.weight"] = a.contiguous()
            state[f"lora_unet_{path}.lora_up.weight"] = b.contiguous()
        path = tmp_path / f"{i}.safetensors"
        save_file(state, str(path))
        states.append(state)
        paths.append(str(path))
    chain = patching.ModChain(tuple(patching.ModEntry("lora", p, axis="creative") for p in paths))
    base = FakePatcher(FakeBaseModel(model))
    routed, report = patching.apply(base, chain, 1, gate=("none", None), joint=True)
    stacked, _ = patching.apply(base, chain, 1, gate=("none", None), joint=False)
    x = torch.randn(2, H)
    for path in ("blocks.0.adaln_modulation_cross_attn.1", "llm_adapter.proj"):
        key = "diffusion_model." + path
        original = model.get_submodule(path)(x)
        stacked_output = stacked.object_patches[key](x)
        routed_output = routed.object_patches[key](x)
        torch.testing.assert_close(routed_output, original + 0.5 * (stacked_output - original))
    assert "previous-step mean share, 1/2 before the first step" in report
    from kilora.multi_import import import_loras

    result = import_loras(paths, model, family="anima", full_model=model)
    for state, source in zip(states, result.sources, strict=True):
        assert set().union(*map(set, source["coverage"].values())) == set(state)
        assert all(key not in report for key in state)


def test_flux_joint_share_samples_one_representative_layer(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora import compute
    from kilora.call_values import current_mount_share
    from kilora.comfyui import patching

    monkeypatch.setenv("KILORA_DERIVED_CARRY", "0")
    model = _model("flux")
    gen = torch.Generator().manual_seed(79)
    paths = []
    for i in range(2):
        state = {}
        for stem, dim in (("attn.to_out.0", H), ("norm1.linear", 6 * H)):
            a, b = _ab(gen, H, dim)
            prefix = f"transformer.transformer_blocks.0.{stem}"
            state[prefix + ".lora_A.weight"] = a.contiguous()
            state[prefix + ".lora_B.weight"] = b.contiguous()
        path = tmp_path / f"flux{i}.safetensors"
        save_file(state, str(path))
        paths.append(str(path))
    chain = patching.ModChain(tuple(patching.ModEntry("lora", p, axis="creative") for p in paths))
    patched, report = patching.apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, gate=("none", None), joint=True
    )
    assert "adaln 2" in report
    layer = patched.object_patches["diffusion_model.double_blocks.0.img_attn.proj"]
    assert "diffusion_model.double_blocks.0.img_mod.lin" in patched.object_patches
    wrapper = patched.model_options["model_function_wrapper"]
    x = torch.randn(2, 4, H, generator=gen)
    all_gates = []
    with (
        torch.no_grad(),
        compute.dropout_key(0, 2.0),
        compute.recording_set_shares(lambda target, gates, sets: all_gates.append(gates[0])),
    ):
        for _ in range(8):
            layer(x)
    expected = torch.stack(all_gates).mean((0, 2))[:, 0]
    observed = []
    original = compute.layer_set_share_observation

    def track(target, sets):
        from contextlib import contextmanager

        @contextmanager
        def scoped():
            with original(target, sets):
                observed.append(bool(compute._SET_SHARES))
                yield

        return scoped()

    monkeypatch.setattr(compute, "layer_set_share_observation", track)

    def apply_model(inp, timestep, **kwargs):
        share = current_mount_share(patching.JOINT_SET, 0, 2)
        for _ in range(8):
            layer(x)
        return share

    args = {"input": torch.zeros(2, 4, 4, 4), "timestep": 2.0, "c": {}}
    with torch.no_grad():
        assert wrapper(apply_model, args) == 0.5
        args["timestep"] = 1.0
        actual = wrapper(apply_model, args)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert sum(observed[:8]) <= 4 and sum(observed[8:]) <= 4
