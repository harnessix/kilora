# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_modlist import write_mod

from kilora import compose, import_lora
from kilora.bake import is_baked
from kilora.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from kilora.comfyui import nodes as N
from kilora.modlist import LoadOrderError
from kilora.testing import TinyDenoiser


class FakeBaseModel(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.diffusion_model = denoiser


class FakePatcher:
    def __init__(self, model, object_patches=None, model_options=None):
        self.model = model
        self.object_patches = dict(object_patches or {})
        self.model_options = dict(model_options or {})
        self.callbacks = {}

    def clone(self):
        out = FakePatcher(self.model, self.object_patches, self.model_options)
        out.callbacks = {
            event: {key: list(items) for key, items in groups.items()}
            for event, groups in self.callbacks.items()
        }
        return out

    def add_callback_with_key(self, event, key, fn):
        self.callbacks.setdefault(event, {}).setdefault(key, []).append(fn)

    def remove_callbacks_with_key(self, event, key):
        self.callbacks.get(event, {}).pop(key, None)

    def kilora_token_map(self, latent):
        from kilora.backends import TokenMap

        return TokenMap()

    def kilora_forward(self, patcher, latent, positive, sigma):
        x = latent["samples"]
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
        return patcher.run((1 - sigma) * x + sigma * noise)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        obj = self.model
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def run(self, x):
        backup = {}
        for name, obj in self.object_patches.items():
            parent_name, _, attr = name.rpartition(".")
            parent = self.get_model_object(parent_name) if parent_name else self.model
            backup[name] = (parent, attr, getattr(parent, attr))
            setattr(parent, attr, obj)
        try:
            with torch.no_grad():
                return self.model.diffusion_model(x)
        finally:
            for parent, attr, old in backup.values():
                setattr(parent, attr, old)


@pytest.fixture
def denoiser():
    m = TinyDenoiser(seed=0).eval()
    m.kilora_family = "kilora-tiny-1"
    return m


@pytest.fixture
def folders(tmp_path, monkeypatch):
    dirs = {"kilora": tmp_path / "kilora", "loras": tmp_path / "loras"}
    for d in dirs.values():
        d.mkdir()

    def listing(folder, extension=None):
        names = sorted(p.name for p in dirs[folder].iterdir())
        return [n for n in names if extension is None or n.endswith(extension)]

    monkeypatch.setattr(N.FOLDERS, "list", listing)
    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(dirs[folder] / name))
    monkeypatch.setattr(N.FOLDERS, "first_dir", lambda folder: str(dirs[folder]))
    monkeypatch.setattr(N.FOLDERS, "dirs", lambda folder: [str(dirs[folder])])
    return dirs


def test_node_names_are_stable():
    assert list(NODE_CLASS_MAPPINGS) == [
        "KiLoRASelect",
        "KiLoRAImportLoRA",
        "KiLoRATriggerWords",
        "KiLoRAPalette",
        "KiLoRASavePalette",
        "KiLoRASuggestions",
        "KiLoRALoadOrder",
        "KiLoRAApply",
        "KiLoRAClearHistory",
        "KiLoRAClear",
        "KiLoRALiveSave",
        "KiLoRALiveOptions",
        "KiLoRALivePacks",
        "KiLoRAFreeze",
        "KiLoRAModInfo",
        "KiLoRALiveExport",
        "KiLoRALoadMod",
        "KiLoRALiveRecord",
        "KiLoRALiveWrite",
        "KiLoRALiveLearnFromImages",
        "KiLoRARegion",
        "KiLoRADepthSchedule",
        "KiLoRARegionAttention",
        "KiLoRASimple",
        "KiLoRASettings",
        "KiLoRALoRALoader",
        "KiLoRARouteLoaded",
        "KiLoRARouteRecord",
        "KiLoRARouteMap",
        "KiLoRATokenRoles",
        "KiLoRATwoPanels",
        "KiLoRARegionIsolation",
        "KiLoRATimeline",
        "KiLoRADepthMap",
        "KiLoRALayerConditioning",
        "KiLoRALayerFilter",
        "KiLoRASaveRecipe",
        "KiLoRADirectionNodes",
        "KiLoRALoRALibrary",
        "KiLoRASimilarLoRAs",
        "KiLoRAPrepareLoRAs",
        "KiLoRAMentionPrompt",
        "KiLoRAPromptSet",
        "KiLoRASolveFromPrompts",
        "KiLoRASolveFromImages",
        "KiLoRAPack",
        "KiLoRABuildEditorPack",
    ]
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRAApply"] == "KiLoRA Apply Mod List"
    assert N.KiLoRAApply.RETURN_TYPES == ("MODEL", "STRING", "CLIP")
    assert N.KiLoRAApply.RETURN_NAMES == ("model", "report", "clip")
    assert N.KiLoRASelect.RETURN_NAMES == ("sets",)


def test_input_names_are_stable(folders):
    select = N.KiLoRASelect.INPUT_TYPES()
    assert list(select["required"]) == ["file", "strength", "prior"]
    assert list(select["optional"]) == ["sets", "enabled"]
    assert select["required"]["strength"][1]["min"] == -2.0
    assert list(N.KiLoRAApply.INPUT_TYPES()["required"]) == ["model", "sets", "depth"]
    assert list(N.KiLoRAImportLoRA.INPUT_TYPES()["required"]) == ["lora_name", "strength"]


def test_select_apply_clear(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(2))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    write_mod(denoiser, folders["kilora"], "parchment", seed=2)
    base_patcher = FakePatcher(FakeBaseModel(denoiser))
    base = base_patcher.run(x)

    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    (chain,) = N.KiLoRASelect().select("parchment-1.0.0.kmod", -0.5, 1.0, sets=chain)
    patched, report, _ = N.KiLoRAApply().apply(base_patcher, chain, 2, radial_step=1.0)
    assert "1. ink" in report and "2. parchment" in report and "Depth: 2" in report
    assert base_patcher.object_patches == {}
    assert all(is_baked(v) for v in patched.object_patches.values())
    assert set(patched.object_patches) == {
        "diffusion_model.layers.0.attention.to_out.0",
        "diffusion_model.layers.1.attention.to_out.0",
    }

    reference = compose(
        denoiser,
        [folders["kilora"] / "ink-1.0.0.kmod", folders["kilora"] / "parchment-1.0.0.kmod"],
        strengths=[1.0, -0.5],
        depth=2,
    )
    with torch.no_grad():
        want = denoiser(x)
    reference.remove()
    torch.testing.assert_close(patched.run(x), want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(base_patcher.run(x), base)

    (one,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    again, _, _ = N.KiLoRAApply().apply(patched, one, 2)
    single, _, _ = N.KiLoRAApply().apply(base_patcher, one, 2)
    torch.testing.assert_close(again.run(x), single.run(x))

    (cleared,) = N.KiLoRAClear().clear(patched)
    assert cleared.object_patches == {}
    torch.testing.assert_close(cleared.run(x), base)


def test_disabled_mod_and_depth_from_mods(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0, enabled=False)
    patched, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, -1)
    assert patched.object_patches == {} and "off: ink-1.0.0.kmod" in report
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, -1)
    assert "Depth: 1" in report
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 9)
    assert "depth above trained range (1 to 4)" in report


def test_conflicts_stop_apply(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1, requires=["parchment >=2"])
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    patcher = FakePatcher(FakeBaseModel(denoiser))
    with pytest.raises(LoadOrderError, match="ink requires parchment >=2"):
        N.KiLoRAApply().apply(patcher, chain, 2)
    result = N.KiLoRALoadOrder().show(patcher, chain)
    assert "ink requires parchment >=2" in result["result"][0]


def test_import_lora_node_matches_plain_lora(denoiser, folders):
    gen = torch.Generator().manual_seed(3)
    sd = {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": torch.randn(2, 16, generator=gen),
        "lora_unet_layers_0_attention_to_q.lora_up.weight": torch.randn(16, 2, generator=gen),
        "lora_unet_layers_0_attention_to_q.alpha": torch.tensor(1.0),
    }
    save_file(sd, str(folders["loras"] / "style.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("style.safetensors", 0.8)
    apply = N.KiLoRAApply().apply
    patched, report, _ = apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1, output_gate="none")
    assert "Depth: 1" in report
    x = torch.randn(1, 5, 4)
    imported = import_lora(sd, denoiser, weight=0.8).adapter_set
    hooked = compose(denoiser, [imported], mix="stack")
    with torch.no_grad():
        want = denoiser(x)
    hooked.remove()
    actual = patched.run(x)
    torch.testing.assert_close(actual, want, rtol=1e-5, atol=1e-6)
    from kilora.comfyui.expert_text import conditioning_values

    with conditioning_values({}):
        assert torch.equal(patched.run(x), actual)


def test_palette_nodes_save_and_load_mods(denoiser, folders, tmp_path, monkeypatch):
    from kilora import palette

    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    write_mod(denoiser, folders["kilora"], "parchment", seed=2)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 0.7, 1.0)
    (chain,) = N.KiLoRASelect().select("parchment-1.0.0.kmod", -0.3, 1.0, sets=chain)
    saved = N.KiLoRASavePalette().save(chain, "Two mods", "two")
    assert "Saved two.kpalette.json with 2 entries." in saved["result"][0]
    loaded, words, settings, report = N.KiLoRAPalette().load("two.kpalette.json")
    assert [(e.strength, e.path.endswith(".kmod")) for e in loaded] == [(0.7, True), (-0.3, True)]
    assert "2 of 2 entries" in report and settings["mix"] == "balanced"


class FakeZImageAttention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = torch.nn.Linear(width, 3 * width, bias=False)
        self.out = torch.nn.Linear(width, width, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        w = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
        return self.out(w @ v)


class FakeZImageBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = FakeZImageAttention(width)
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w1 = torch.nn.Linear(width, 2 * width, bias=False)
        self.feed_forward.w2 = torch.nn.Linear(2 * width, width, bias=False)
        self.feed_forward.w3 = torch.nn.Linear(width, 2 * width, bias=False)

    def forward(self, x):
        x = x + self.attention(x)
        ff = self.feed_forward
        return x + ff.w2(torch.nn.functional.silu(ff.w1(x)) * ff.w3(x))


class FakeZImage(torch.nn.Module):
    def __init__(self, width=12, n=2):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([FakeZImageBlock(width) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_z_image_names_and_fused_qkv_lora(folders):
    model = FakeZImage()
    patcher = FakePatcher(FakeBaseModel(model))
    gen = torch.Generator().manual_seed(1)
    weights = {}
    sd = {}
    for proj in ("to_q", "to_v"):
        a, b = torch.randn(2, 12, generator=gen), torch.randn(12, 2, generator=gen)
        weights[proj] = (a, b)
        sd[f"transformer.layers.1.attention.{proj}.lora_A.weight"] = a
        sd[f"transformer.layers.1.attention.{proj}.lora_B.weight"] = b
    save_file(sd, str(folders["loras"] / "zi.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("zi.safetensors", 1.0)
    patched, report, _ = N.KiLoRAApply().apply(patcher, chain, 1, output_gate="none")
    assert list(patched.object_patches) == ["diffusion_model.layers.1.attention.qkv"]
    qkv = patched.object_patches["diffusion_model.layers.1.attention.qkv"]
    base = model.layers[1].attention.qkv
    x = torch.randn(3, 12)
    with torch.no_grad():
        got = qkv(x)
        want = base(x).clone()
        a, b = weights["to_q"]
        want[:, 0:12] += x @ a.t() @ b.t()
        a, b = weights["to_v"]
        want[:, 24:36] += x @ a.t() @ b.t()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_unknown_model_is_refused(folders):
    write_mod(TinyDenoiser(), folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    from kilora import KiLoRAError

    with pytest.raises(KiLoRAError, match="does not support this base model"):
        N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(TinyDenoiser())), chain, 1)


def test_keyed_z_image_mod_lands_on_attention_out(folders):
    from kilora import Settings, save_mod
    from kilora.sets import new_keyed_set

    widths = {f"layers.{i}.attention.to_out.0": (12, 12) for i in range(2)}
    s = new_keyed_set(
        "zi", widths, num_experts=3, routes=2, rank=1, router_dim=4, router_seed=0, seed=0
    )
    for ex in s.experts.values():
        ex.up.normal_()
    save_mod(
        folders["kilora"] / "zi.kmod",
        [s],
        Settings(depth=2),
        id="zi",
        base_model={"family": "z-image"},
    )
    model = FakeZImage()
    patcher = FakePatcher(FakeBaseModel(model))
    (chain,) = N.KiLoRASelect().select("zi.kmod", 1.0, 1.0)
    patched, report, _ = N.KiLoRAApply().apply(patcher, chain, -1)
    assert sorted(patched.object_patches) == [
        "diffusion_model.layers.0.attention.out",
        "diffusion_model.layers.1.attention.out",
    ]
    x = torch.randn(1, 4, 12)
    assert not torch.allclose(patched.run(x), patcher.run(x))


def test_mod_info_node(denoiser, folders):
    write_mod(
        denoiser,
        folders["kilora"],
        "ink",
        seed=1,
        trigger_words=["ink style"],
        example_prompts=["a boat, ink style"],
        description="Soft ink lines.",
        strength={"recommended": 0.7, "min": -1.0, "max": 1.5},
    )
    out = N.KiLoRAModInfo().info("ink-1.0.0.kmod")
    strength, words, prompt, depth, info = out["result"]
    assert out["ui"]["text"] == [info]
    assert (strength, words, prompt, depth) == (0.7, "ink style", "a boat, ink style", 1)
    assert "Soft ink lines." in info and "Recommended strength: 0.7 (range -1 to 1.5)" in info
    assert "Trigger: ink style" in info and "Training data: not described" in info
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", strength, 1.0)
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, -1)
    assert "trigger words: ink style" in report and "strength +0.7" in report


def test_mod_info_card_in_every_locale(denoiser, folders, monkeypatch):
    from kilora import i18n

    write_mod(
        denoiser,
        folders["kilora"],
        "ink",
        seed=1,
        trigger_words=["ink style"],
        data_source={"name": "ink", "images": 24, "license": "CC0-1.0"},
    )
    seen = set()
    for lang in i18n.LOCALES:
        monkeypatch.setenv("KILORA_LANG", lang)
        info = N.KiLoRAModInfo().info("ink-1.0.0.kmod")["result"][4]
        assert "ink style" in info and "24" in info and "CC0-1.0" in info
        assert "{" not in info
        seen.add(info)
    assert len(seen) == len(i18n.LOCALES)


def test_load_mod_node_names_are_stable(folders):
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRALoadMod"] == "KiLoRA Load Mod"
    assert N.KiLoRALoadMod.RETURN_TYPES == ("MODEL", "STRING")
    assert N.KiLoRALoadMod.RETURN_NAMES == ("model", "report")
    types = N.KiLoRALoadMod.INPUT_TYPES()
    assert list(types["required"]) == ["model", "file", "strength"]
    assert list(types["optional"]) == [
        "depth",
        "iteration",
        "routing_scope",
        "routing_select",
    ]


def test_load_mod_matches_select_and_apply(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(3))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    base_patcher = FakePatcher(FakeBaseModel(denoiser))
    base = base_patcher.run(x)

    loaded, report = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", 0.7, 2)
    assert "1. ink" in report and "Depth: 2" in report
    assert base_patcher.object_patches == {}
    assert all(is_baked(v) for v in loaded.object_patches.values())

    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 0.7, 1.0)
    reference, _, _ = N.KiLoRAApply().apply(base_patcher, chain, 2)
    torch.testing.assert_close(loaded.run(x), reference.run(x))
    assert not torch.allclose(loaded.run(x), base)
    torch.testing.assert_close(base_patcher.run(x), base)


def test_load_mod_strength_control(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(4))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    base_patcher = FakePatcher(FakeBaseModel(denoiser))
    base = base_patcher.run(x)

    off, _ = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", 0.0, 1)
    torch.testing.assert_close(off.run(x), base, rtol=1e-5, atol=1e-6)

    low, _ = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", 0.5, 1)
    full, _ = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", 1.0, 1)
    neg, _ = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", -1.0, 1)
    d_low = (low.run(x) - base).flatten()
    d_full = (full.run(x) - base).flatten()
    d_neg = (neg.run(x) - base).flatten()
    assert 0 < d_low.norm() < d_full.norm()
    assert torch.nn.functional.cosine_similarity(d_low, d_full, dim=0) > 0.95
    assert torch.nn.functional.cosine_similarity(d_neg, d_full, dim=0) < -0.9


def test_load_mod_default_depth_and_clear(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(5))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    base_patcher = FakePatcher(FakeBaseModel(denoiser))
    base = base_patcher.run(x)
    loaded, report = N.KiLoRALoadMod().load(base_patcher, "ink-1.0.0.kmod", 1.0)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    reference, _, _ = N.KiLoRAApply().apply(base_patcher, chain, -1)
    torch.testing.assert_close(loaded.run(x), reference.run(x))
    (cleared,) = N.KiLoRAClear().clear(loaded)
    torch.testing.assert_close(cleared.run(x), base)


def test_routing_dropdowns_offer_scope_and_select(folders):
    for cls in (N.KiLoRAApply, N.KiLoRALoadMod):
        optional = cls.INPUT_TYPES()["optional"]
        options, spec = optional["routing_scope"]
        assert options == ["mod", "token", "tensor"] and spec["default"] == "mod"
        options, spec = optional["routing_select"]
        assert options == ["mod", "sparsemax", "top1"] and spec["default"] == "mod"
    assert N._from_mod("mod") is None and N._from_mod("top1") == "top1"


def test_add_lora_reads_a_lokr_file(denoiser, folders):
    gen = torch.Generator().manual_seed(4)
    stem = "lora_unet_layers_0_attention_to_q"
    w1 = torch.randn(4, 4, generator=gen)
    w2_a, w2_b = torch.randn(4, 2, generator=gen), torch.randn(2, 4, generator=gen)
    sd = {
        f"{stem}.lokr_w1": w1,
        f"{stem}.lokr_w2_a": w2_a,
        f"{stem}.lokr_w2_b": w2_b,
        f"{stem}.alpha": torch.tensor(2.0),
    }
    save_file(sd, str(folders["loras"] / "kron.safetensors"))
    delta = torch.kron(w1, w2_a @ w2_b) * (2.0 / 2)
    (chain,) = N.KiLoRAImportLoRA().add("kron.safetensors", 0.8)
    patched, report, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, 1, output_gate="none"
    )
    module = denoiser.layers[0].attention.to_q
    handle = module.register_forward_hook(lambda m, i, o: o + 0.8 * i[0] @ delta.t())
    x = torch.randn(1, 5, 4)
    with torch.no_grad():
        want = denoiser(x)
    handle.remove()
    torch.testing.assert_close(patched.run(x), want, rtol=1e-4, atol=1e-5)
