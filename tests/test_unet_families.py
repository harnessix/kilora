# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import UNet2DConditionModel
from live_helpers import write

from kilora import adapter as adapter_module
from kilora import compose, get_preset, import_lora, load_mod
from kilora import compute as C
from kilora.adapter import Settings, attach
from kilora.bake import bake
from kilora.bindings import get_binding
from kilora.live import LiveSpec, open_session
from kilora.live_basis import columns_of, default_v0
from kilora.live_sets import LIVE_PRESETS, make_live_set
from kilora.modfile import save_mod
from kilora.presets import ldm_unet_name

CONTEXT = 16


def _sd15():
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(8, 8, 16, 16),
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        layers_per_block=2,
        cross_attention_dim=CONTEXT,
        attention_head_dim=4,
        norm_num_groups=4,
    ).eval()


def _sdxl():
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(8, 16, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,
        transformer_layers_per_block=(1, 2, 2),
        cross_attention_dim=CONTEXT,
        attention_head_dim=4,
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=CONTEXT + 6 * 4,
    ).eval()


class Wrap(torch.nn.Module):
    def __init__(self, unet, family):
        super().__init__()
        self.unet, self.family = unet, family
        gen = torch.Generator().manual_seed(1)
        self.ctx = torch.randn(1, 5, CONTEXT, generator=gen)
        self.extra = None
        if family == "sdxl":
            self.extra = {
                "text_embeds": torch.randn(1, CONTEXT, generator=gen),
                "time_ids": torch.randn(1, 6, generator=gen),
            }

    def forward(self, x):
        kw = {"added_cond_kwargs": self.extra} if self.extra else {}
        return self.unet(x, 500, self.ctx, **kw).sample


@pytest.fixture(params=["sd15", "sdxl"])
def unet(request):
    family = request.param
    unet = _sd15() if family == "sd15" else _sdxl()
    return Wrap(unet, family).eval(), family


def _x():
    return torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2))


def _trained(model, family, experts=4, depth_seed=3):
    preset = get_preset(family)
    config = preset.config(
        num_experts=experts,
        routes=2,
        rank=2,
        router_dim=8,
    )
    a = attach(model, config)
    gen = torch.Generator().manual_seed(depth_seed)
    with torch.no_grad():
        for site in a.sites:
            t = site.tensors["s0_up"]
            t.copy_(torch.randn(t.shape, generator=gen) * 0.3)
    sets = a.to_sets()
    a.remove()
    return sets[0]


def test_presets_target_every_attention_layer(unet):
    model, family = unet
    main = _trained(model, family)
    blocks = {
        n for n, _ in model.named_modules() if n.rsplit(".", 2)[-2:-1] == ["transformer_blocks"]
    }
    assert len(main.target_modules) == 8 * len(blocks)
    assert all(".attn1." in t or ".attn2." in t for t in main.target_modules)
    k = next(t for t in main.target_modules if t.endswith("attn2.to_k"))
    assert dict(model.named_modules())[k].in_features == CONTEXT
    assert main.experts[k].out_features == dict(model.named_modules())[k].out_features


def test_compose_depth_fast_path_and_bake(unet, monkeypatch):
    model, family = unet
    main = _trained(model, family)
    x = _x()
    with torch.no_grad():
        base = model(x)
    out = {}
    for fast in (False, True):
        monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
        adapter = compose(model, [main], depth=3)
        with torch.no_grad():
            out[fast] = model(x)
        adapter.remove()
    assert not torch.allclose(out[True], base)
    real, worst = C.run_layer, []

    def both(*args, **kw):
        monkeypatch.setattr(C, "LOWRANK_ENABLED", False)
        loop = real(*args, **kw)
        monkeypatch.setattr(C, "LOWRANK_ENABLED", True)
        fast = real(*args, **kw)
        worst.append(float((fast - loop).abs().max() / loop.abs().max().clamp(min=1e-6)))
        return fast

    monkeypatch.setattr(adapter_module, "run_layer", both)
    adapter = compose(model, [main], depth=3)
    with torch.no_grad():
        model(x)
    adapter.remove()
    monkeypatch.undo()
    assert worst and max(worst) < 2e-5
    assert not torch.allclose(out[False], base)
    handle = bake(model, [main], depth=3)
    with torch.no_grad():
        baked = model(x)
    handle.remove()
    torch.testing.assert_close(baked, out[True], rtol=1e-4, atol=1e-4)
    with torch.no_grad():
        torch.testing.assert_close(model(x), base)


@pytest.mark.parametrize("preset", ["light", "standard"])
def test_live_state_save_undo_reload_bit_identical(unet, preset, tmp_path):
    model, family = unet
    main = _trained(model, family)
    live, layout = make_live_set(main, LIVE_PRESETS[preset], family)
    assert layout.id == f"{family}-{preset}-1"
    if preset == "light":
        assert all(t.endswith(("attn1.to_v", "attn1.to_out.0")) for t in layout.targets)
    else:
        assert len(layout.targets) == len(main.target_modules)
    cols = {t: columns_of(live.experts[t].down) for t in layout.targets}
    v0 = default_v0(layout.targets, cols, seed=5, rank=8)
    for t in layout.targets:
        torch.testing.assert_close(v0[t] @ v0[t].t(), torch.eye(8), atol=1e-5, rtol=0)
    spec = LiveSpec(layout=layout, updater="plugin", live_set=live.name)
    path = tmp_path / f"ink.{family}.kmod.safetensors"
    save_mod(
        path,
        [main, live],
        Settings(depth=2, softmax_scope="per_set"),
        id="ink",
        mod_family="style",
        base_model={"family": family},
        live=spec.to_manifest(),
    )
    s = open_session(path, saves_dir=tmp_path / "saves")
    s.learning = True
    x = _x()

    def run(sets):
        adapter = compose(model, sets, depth=2, softmax_scope="per_set")
        with torch.no_grad():
            out = model(x)
        adapter.remove()
        return out

    clean = run(s.sets())
    write(s, size=0.05)
    one = {t: p["up"].clone() for t, p in s.state.items()}
    after_one = run(s.sets())
    assert not torch.allclose(after_one, clean)
    name = s.save_point("one")
    write(s, seed=1, size=0.05)
    s.undo()
    assert all(torch.equal(s.state[t]["up"], one[t]) for t in one)
    assert torch.equal(run(s.sets()), after_one)
    again = open_session(path, saves_dir=tmp_path / "saves")
    again.load(name)
    assert all(torch.equal(again.state[t]["up"], one[t]) for t in one)
    assert torch.equal(run(again.sets()), after_one)
    assert load_mod(path).manifest["base_model"]["family"] == family


def _lora(model, family, rank=2):
    gen = torch.Generator().manual_seed(7)
    out = {}
    for n, m in model.unet.named_modules():
        if isinstance(m, torch.nn.Linear) and ".transformer_blocks." in n and ".attn" in n:
            out[n] = (
                torch.randn(rank, m.in_features, generator=gen) * 0.2,
                torch.randn(m.out_features, rank, generator=gen) * 0.2,
            )
    return out


def _keys(lora, layout):
    sd = {}
    for n, (down, up) in lora.items():
        if layout == "peft":
            sd[f"unet.{n}.lora_A.weight"], sd[f"unet.{n}.lora_B.weight"] = down, up
        elif layout == "kohya":
            stem = "lora_unet_" + n.replace(".", "_")
            sd[f"{stem}.lora_down.weight"], sd[f"{stem}.lora_up.weight"] = down, up
            sd[f"{stem}.alpha"] = torch.tensor(2.0)
        elif layout == "kohya-ldm":
            stem = "lora_unet_" + ldm_unet_name(n).replace(".", "_")
            sd[f"{stem}.lora_down.weight"], sd[f"{stem}.lora_up.weight"] = down, up
            sd[f"{stem}.alpha"] = torch.tensor(2.0)
        else:
            stem = "diffusion_model." + ldm_unet_name(n)
            sd[f"{stem}.lora_A.weight"], sd[f"{stem}.lora_B.weight"] = down, up
    return sd


@pytest.mark.parametrize("layout", ["peft", "kohya", "kohya-ldm", "comfyui"])
def test_lora_import_common_layouts(unet, layout):
    model, family = unet
    lora = _lora(model, family)
    x = _x()
    scale = 1.0
    merged = {}
    with torch.no_grad():
        mods = dict(model.unet.named_modules())
        for n, (down, up) in lora.items():
            merged[n] = mods[n].weight.clone()
            mods[n].weight += scale * up @ down
        want = model(x)
        for n, w in merged.items():
            mods[n].weight.copy_(w)
    result = import_lora(_keys(lora, layout), model.unet, family=family)
    assert len(result.adapter_set.experts) == len(lora)
    adapter = compose(model.unet, [result.adapter_set], mix="stack")
    with torch.no_grad():
        got = model(x)
    adapter.remove()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_ldm_names_need_the_family(unet):
    model, family = unet
    with pytest.raises(Exception, match="no module with this name"):
        import_lora(_keys(_lora(model, family), "comfyui"), model.unet)


def test_comfyui_binding(unet):
    model, family = unet
    binding = get_binding(family)
    names = [n for n, _ in model.unet.named_modules() if n]
    ldm = [ldm_unet_name(n) or n for n in names]
    if family == "sdxl":
        ldm.append("label_emb.0.0")
    assert binding.matches_comfy(ldm)
    other = get_binding("sdxl" if family == "sd15" else "sd15")
    assert not other.matches_comfy(ldm)
    targets = _trained(model, family).target_modules
    points = set()
    for t in targets:
        inner = t.removeprefix("unet.")
        place = binding.to_comfy(inner)
        assert place.path == "diffusion_model." + ldm_unet_name(inner) and place.part is None
        assert binding.from_comfy(ldm_unet_name(inner)) == [(inner, None)]
        points.add(binding.hook_point(inner))
    assert len(points) == len(targets) and all(p.startswith("base::") for p in points)
