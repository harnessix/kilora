# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import struct

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora.bake import TargetPlan
from kilora.compute import dropout_key
from kilora.families import get_family
from kilora.family_spec import TokenLayout
from kilora.region import latent_size, token_layouts
from kilora.roles import (
    file_policy,
    first_policy,
    looks_like_edit,
    parse_policy,
    reference_shapes,
    role_mask,
    role_spans,
    shapes_of,
)

WIDTH = 16


def test_policies():
    assert parse_policy(None) == "auto" and parse_policy("") == "auto"
    assert parse_policy("Text + target") == "target+text"
    assert parse_policy("reference:1,text") == "reference:1+text"
    assert parse_policy("ALL") == "all"
    for bad in ("style", "target:1", "reference:x"):
        with pytest.raises(ValueError):
            parse_policy(bad)
    assert first_policy("auto", None, "reference", "target") == "reference"
    assert first_policy("auto", "") == "auto"
    assert looks_like_edit("loras/Qwen-Edit-2509-Multiple-angles.safetensors")
    assert looks_like_edit("klein_consistency_v2.safetensors")
    assert not looks_like_edit("anime_style.safetensors")


def test_policy_from_lora_metadata(tmp_path):
    def write(name, meta):
        header = json.dumps({"__metadata__": meta}).encode()
        path = tmp_path / name
        path.write_bytes(struct.pack("<Q", len(header)) + header)
        return path

    assert file_policy(write("a.safetensors", {"token_roles": "reference"})) == "reference"
    assert file_policy(write("b.safetensors", {"other": "1"})) == "auto"
    assert file_policy(tmp_path / "missing.safetensors") == "auto"
    assert file_policy(None) == "auto"


def test_reference_shapes_from_comfy_conditioning():
    refs = [torch.zeros(1, 16, 8, 6), torch.zeros(1, 16, 4, 4)]
    assert shapes_of({"ref_latents": refs}) == ((8, 6), (4, 4))
    assert shapes_of({"ref_shapes": [(8, 8)]}) == ((8, 8),)
    assert shapes_of({}) == ()


def _spans(family, target, tokens, latent, refs, text_tokens=None):
    layout = get_family(family).layout(target)
    return role_spans(layout, tokens, latent, refs, text_tokens)


def test_klein_index_layout():
    refs = [(8, 8), (4, 8)]
    n, r0, r1 = 16, 16, 8
    spans = _spans("flux", "double_blocks.0.img_attn.qkv", n + r0 + r1, (8, 8), refs)
    assert spans == {"target": [(0, 16)], "reference:0": [(16, 32)], "reference:1": [(32, 40)]}
    spans = _spans("flux", "single_blocks.0.linear1", 5 + n + r0 + r1, (8, 8), refs)
    assert spans["text"] == [(0, 5)] and spans["target"] == [(5, 21)]
    assert spans["reference:1"] == [(37, 45)]
    assert _spans("flux", "double_blocks.0.txt_attn.qkv", 5, (8, 8), refs) == {"text": [(0, 5)]}
    assert _spans("flux", "double_blocks.0.img_attn.qkv", 41, (8, 8), refs) is None


def test_qwen_edit_layout():
    spans = _spans("qwen-image", "transformer_blocks.0.attn.to_q", 16 + 16, (8, 8), [(8, 8)])
    assert spans == {"target": [(0, 16)], "reference:0": [(16, 32)]}
    text = _spans("qwen-image", "transformer_blocks.0.attn.add_q_proj", 7, (8, 8), [(8, 8)])
    assert text == {"text": [(0, 7)]}


def test_moss_and_flux3_layouts():
    spans = _spans("moss-image", "layers.0.proj.q", 4 + 3 + 4, (2, 2), [(2, 2)])
    assert spans == {"target": [(0, 4)], "text": [(4, 7)], "reference:0": [(7, 11)]}
    spans = _spans("flux3", "single_blocks.0.qkv", 4 + 4 + 3, (2, 2), [(2, 2)])
    assert spans == {"target": [(0, 4)], "reference:0": [(4, 8)], "text": [(8, 11)]}
    spans = _spans("flux3", "content_mode_blocks.image_cond.0.qkv", 4, (2, 2), [])
    assert spans == {"reference": [(0, 4)]}


def test_masks():
    layout = TokenLayout("image2d", (2, 2), text="before")
    refs = [(4, 4)]
    tokens = 3 + 4 + 4

    def mask(policy, refs=refs):
        m = role_mask(policy, layout, tokens, (4, 4), refs)
        return None if m is None else m[0].tolist()

    assert mask("auto") == [0] * 3 + [1] * 4 + [0] * 4
    assert mask("reference") == [0] * 7 + [1] * 4
    assert mask("reference:0+text") == [1] * 3 + [0] * 4 + [1] * 4
    assert mask("all") is None
    assert mask("auto", []) is None and mask("target", []) is None
    assert mask("reference", []) == [0] * tokens
    ref_layer = TokenLayout("image2d", (1, 1), role="reference")
    assert role_mask("auto", ref_layer, 4, (2, 2), [])[0].tolist() == [0] * 4
    assert role_mask("reference", ref_layer, 4, (2, 2), []) is None
    assert role_mask("auto", None, tokens, (4, 4), refs, name="x") is None


def _plan(target, policies, form="state", n=2):
    sets = [dataclasses.replace(s, form=form) for s in _sets(n, WIDTH, 4, 2, 8, seed=3)]
    return TargetPlan(sets=sets, depth=2, name=target, regions=[None] * n, roles=list(policies))


def _call(plan, h, family, latent, refs, text_tokens=None, fast=True):
    resolve = get_family(family).layout
    with (
        torch.no_grad(),
        dropout_key(1, 0.4),
        latent_size(latent),
        token_layouts(resolve, text_tokens),
        reference_shapes(refs),
    ):
        return plan.run(h, h)


def _h(tokens, seed=5):
    return torch.randn(2, tokens, WIDTH, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("fast", [True, False])
def test_target_policy_keeps_references_on_base(form, fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    target, refs = "double_blocks.0.img_attn.qkv", [(8, 8)]
    h = _h(32)
    for policy in ("target", "auto"):
        y = _call(_plan(target, [policy] * 2, form), h, "flux", (8, 8), refs)
        assert torch.equal(y[:, 16:], h[:, 16:])
        assert (y[:, :16] - h[:, :16]).abs().sum(-1).min() > 0
    y = _call(_plan(target, ["reference"] * 2, form), h, "flux", (8, 8), refs)
    assert torch.equal(y[:, :16], h[:, :16])
    assert (y[:, 16:] - h[:, 16:]).abs().sum(-1).min() > 0
    mixed = _call(_plan(target, ["target", "reference"], form), h, "flux", (8, 8), refs)
    assert mixed.shape == h.shape


@pytest.mark.parametrize("policy", ["auto", "target", "target+text", "all"])
def test_without_references_bit_identical(policy):
    target = "single_blocks.0.linear1"
    h = _h(5 + 16)
    plain = _plan(target, [])
    plain.roles = []
    want = _call(plain, h, "flux", (8, 8), None)
    got = _call(_plan(target, [policy] * 2), h, "flux", (8, 8), None)
    assert torch.equal(want, got)
    with torch.no_grad(), dropout_key(1, 0.4):
        assert torch.equal(_plan(target, [policy] * 2).run(h, h), want)


def test_qwen_edit_layer_on_tiny_model():
    target = "transformer_blocks.0.img_mlp.net.2"
    refs = [(4, 8)]
    h = _h(16 + 8)
    y = _call(_plan(target, ["auto"] * 2), h, "qwen-image", (8, 8), refs)
    assert torch.equal(y[:, 16:], h[:, 16:]) and not torch.equal(y[:, :16], h[:, :16])
    text_layer = "transformer_blocks.0.txt_mlp.net.2"
    t = _h(7)
    assert torch.equal(_call(_plan(text_layer, ["target"] * 2), t, "qwen-image", (8, 8), refs), t)
    assert not torch.equal(
        _call(_plan(text_layer, ["target+text"] * 2), t, "qwen-image", (8, 8), refs), t
    )


def test_roles_multiply_a_region():
    target = "double_blocks.0.img_attn.qkv"
    refs = [(8, 8)]
    h = _h(32)
    region = torch.zeros(1, 8, 8)
    region[..., :4] = 1
    plan = _plan(target, ["all", "all"])
    plan.regions = [region, region]
    with_region = _call(plan, h, "flux", (8, 8), refs)
    plan = _plan(target, ["auto", "auto"])
    plan.regions = [region, region]
    both = _call(plan, h, "flux", (8, 8), refs)
    assert torch.equal(both, with_region)
    assert torch.equal(both[:, 16:], h[:, 16:])


def test_mod_file_keeps_the_policy(model, tmp_path):
    from kilora import KiLoRAConfig, attach, load_mod
    from kilora.modfile import save_mod

    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=1,
            train_depth=None,
            output_gate="none",
        ),
    )
    assert adapter.settings.token_roles == "auto"
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    save_mod(tmp_path / "a.kmod.safetensors", sets, settings, id="a")
    assert load_mod(tmp_path / "a.kmod.safetensors").settings.token_roles == "auto"
    settings.token_roles = "reference"
    save_mod(tmp_path / "b.kmod.safetensors", sets, settings, id="b")
    assert load_mod(tmp_path / "b.kmod.safetensors").settings.token_roles == "reference"


def test_node_and_policy_order(tmp_path):
    from types import SimpleNamespace

    from kilora.adapter import Settings
    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching as P

    header = json.dumps({"__metadata__": {"token_roles": "reference"}}).encode()
    lora = tmp_path / "omini_edit.safetensors"
    lora.write_bytes(struct.pack("<Q", len(header)) + header)
    chain = P.ModChain((P.ModEntry("mod", "/m/a.kmod"), P.ModEntry("lora", str(lora))))
    (out,) = N.KiLoRATokenRoles().set_roles(chain, "target", apply_to="last mod")
    assert [e.roles for e in out] == ["auto", "target"]
    (custom,) = N.KiLoRATokenRoles().set_roles(chain, "auto", custom="reference:0 + text")
    assert {e.roles for e in custom} == {"reference:0+text"}

    def mod(path, label, roles="auto"):
        return SimpleNamespace(
            path=path, label=label, sets=[1], settings=Settings(token_roles=roles)
        )

    class Plan:
        mods = [mod("/m/a.kmod", "a.kmod", "target+text"), mod(None, lora.name)]

        def scoped_sets(self):
            from kilora.bake import ScopedSet

            return [ScopedSet(adapter_set=None), ScopedSet(adapter_set=None)]

    scoped = P._with_regions(Plan(), chain)
    assert [s.roles for s in scoped] == ["target+text", "reference"]
    scoped = P._with_regions(Plan(), out)
    assert [s.roles for s in scoped] == ["target+text", "target"]
    plain = P.ModChain((P.ModEntry("lora", str(tmp_path / "klein_edit.safetensors")),))
    Plan.mods = [mod(None, "klein_edit.safetensors")]
    assert P._role_hints(Plan(), plain) and not P._role_hints(Plan(), chain[:1])


def test_apply_wrapper_passes_the_reference_sizes():
    from types import SimpleNamespace

    from kilora.comfyui import patching as P
    from kilora.roles import current_references

    out = SimpleNamespace(model_options={})
    P._key_mask_by_step(out, family="flux")
    wrapper = out.model_options["model_function_wrapper"]
    seen = []

    def apply_model(x, t, **c):
        seen.append(current_references())
        return x

    x = torch.zeros(1, 16, 8, 8)
    refs = [torch.zeros(1, 16, 8, 6)]
    wrapper(apply_model, {"input": x, "timestep": torch.tensor([0.5]), "c": {"ref_latents": refs}})
    wrapper(apply_model, {"input": x, "timestep": torch.tensor([0.5]), "c": {}})
    assert seen == [((8, 6),), ()]


@pytest.mark.parametrize("mlp", [False, True])
def test_anima_cosmos_reference_frame(mlp):
    refs = shapes_of({"ref_latents": [torch.zeros(1, 16, 8, 8)]})
    target = "transformer_blocks.0.ff.net.0.proj" if mlp else "transformer_blocks.0.attn1.to_q"
    g = torch.Generator().manual_seed(9)
    h = (
        torch.randn(2, 2, 4, 4, WIDTH, generator=g)
        if mlp
        else torch.randn(2, 32, WIDTH, generator=g)
    )
    y = _call(_plan(target, ["auto"] * 2), h, "anima", (1, 8, 8), refs)
    flat_y, flat_h = y.reshape(2, 32, WIDTH), h.reshape(2, 32, WIDTH)
    assert torch.equal(flat_y[:, 16:], flat_h[:, 16:])
    assert (flat_y[:, :16] - flat_h[:, :16]).abs().sum(-1).min() > 0
    edit = _call(_plan(target, ["reference"] * 2), h, "anima", (1, 8, 8), refs)
    assert torch.equal(edit.reshape(2, 32, WIDTH)[:, :16], flat_h[:, :16])
    one = h[:, :1] if mlp else h[:, :16]
    plain = _plan(target, [])
    assert torch.equal(
        _call(_plan(target, ["auto"] * 2), one, "anima", (1, 8, 8), None),
        _call(plain, one, "anima", (1, 8, 8), None),
    )


def test_panel_tokens_and_masks():
    from kilora.roles import check_panels, panel_tokens

    layout = TokenLayout("image2d", (1, 1))
    left = panel_tokens(layout, (2, 4), check_panels("left"))
    assert left.reshape(2, 4).tolist() == [[True, True, False, False]] * 2
    top = panel_tokens(layout, (4, 2), check_panels("top", 0.25)).reshape(4, 2)
    assert top[0].all() and not top[1:].any()
    video = panel_tokens(TokenLayout("video3d", (1, 1, 1)), (2, 2, 4), check_panels("right"))
    assert video.reshape(2, 2, 4)[:, :, 2:].all() and not video.reshape(2, 2, 4)[:, :, :2].any()
    for bad in (("middle", 0.5), ("left", 0.0), ("left", 1.0)):
        with pytest.raises(ValueError):
            check_panels(*bad)
    joint = TokenLayout("image2d", (1, 1), text="before")
    panels = ("left", 0.5)
    m = role_mask("auto", joint, 3 + 8, (2, 4), (), panels=panels)[0].tolist()
    assert m == [0] * 3 + [0, 0, 1, 1] * 2
    m = role_mask("reference:0", joint, 11, (2, 4), (), panels=panels)[0].tolist()
    assert m == [0] * 3 + [1, 1, 0, 0] * 2
    assert role_mask("all", joint, 11, (2, 4), (), panels=panels) is None
    assert role_mask("target+reference+text", joint, 11, (2, 4), (), panels=panels) is None


def test_two_panel_canvas_on_a_layer():
    target = "double_blocks.0.img_attn.qkv"
    h = _h(16)
    plan = _plan(target, ["auto"] * 2)
    plan.panels = [("left", 0.5)] * 2
    y = _call(plan, h, "flux", (8, 8), None).reshape(2, 4, 4, WIDTH)
    hh = h.reshape(2, 4, 4, WIDTH)
    assert torch.equal(y[:, :, :2], hh[:, :, :2])
    assert (y[:, :, 2:] - hh[:, :, 2:]).abs().sum(-1).min() > 0
    plan = _plan(target, ["all"] * 2)
    plan.panels = [("left", 0.5)] * 2
    assert (_call(plan, h, "flux", (8, 8), None) - h).abs().sum(-1).min() > 0


def test_reference_layer_without_a_layout_context():
    from kilora.families import get_family

    target = "content_mode_blocks.image_cond.0.qkv"
    h = _h(16)
    plan = _plan(target, ["auto"] * 2)
    plan.layout = get_family("flux3").layout(target)
    with torch.no_grad(), dropout_key(1, 0.4):
        assert torch.equal(plan.run(h, h), h)
        plan.roles = ["reference"] * 2
        assert not torch.equal(plan.run(h, h), h)


def test_klein_patch_follows_the_model():
    from kilora.families import layout_resolver

    resolve = layout_resolver("flux", 1)
    assert resolve("double_blocks.0.img_attn.qkv").patch == (1, 1)
    assert resolve("single_blocks.0.linear1").text == "before"
    assert resolve("double_blocks.0.txt_attn.qkv").kind == "text1d"
    assert layout_resolver("flux", None)("double_blocks.0.img_attn.qkv").patch == (2, 2)
    assert layout_resolver("qwen-image", 1)("transformer_blocks.0.attn.to_q").patch == (2, 2)
    refs = [(8, 8)]
    spans = role_spans(resolve("double_blocks.0.img_attn.qkv"), 64 + 64, (8, 8), refs)
    assert spans == {"target": [(0, 64)], "reference:0": [(64, 128)]}


def test_two_panels_node_marks_every_entry():
    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching as P

    chain = P.ModChain((P.ModEntry("mod", "/m/a.kmod"), P.ModEntry("lora", "/l/b.safetensors")))
    (out,) = N.KiLoRATwoPanels().panels(chain, "right", 0.4)
    assert [e.panels for e in out] == [("right", 0.4)] * 2
    with pytest.raises(P.KiLoRAError):
        N.KiLoRATwoPanels().panels(P.ModChain(), "left")
    with pytest.raises(P.KiLoRAError):
        N.KiLoRATwoPanels().panels(chain, "middle")


def _joint_plan(target, expert_roles):
    (s,) = [dataclasses.replace(s, form="input") for s in _sets(1, WIDTH, 4, 2, 8, seed=4)]
    plan = TargetPlan(sets=[s], depth=1, name=target, regions=[None], roles=["all"])
    plan.expert_roles = [expert_roles]
    return plan


@pytest.mark.parametrize("fast", [True, False])
def test_joint_import_roles_per_expert(fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    target, refs = "double_blocks.0.img_attn.qkv", [(8, 8)]
    h = _h(32)
    y = _call(_joint_plan(target, ["target"] * 4), h, "flux", (8, 8), refs)
    assert torch.equal(y[:, 16:], h[:, 16:]) and not torch.equal(y[:, :16], h[:, :16])
    y = _call(_joint_plan(target, ["reference"] * 4), h, "flux", (8, 8), refs)
    assert torch.equal(y[:, :16], h[:, :16]) and not torch.equal(y[:, 16:], h[:, 16:])
    plain = _joint_plan(target, None)
    plain.expert_roles = []
    want = _call(plain, h, "flux", (8, 8), refs)
    assert torch.equal(_call(_joint_plan(target, ["all"] * 4), h, "flux", (8, 8), refs), want)
    mixed = _call(_joint_plan(target, ["target", "target", "all", "all"]), h, "flux", (8, 8), refs)
    assert not torch.equal(mixed, want) and not torch.equal(mixed[:, 16:], h[:, 16:])
    assert torch.equal(
        _call(_joint_plan(target, ["target"] * 4), h, "flux", (8, 8), None),
        _call(plain, h, "flux", (8, 8), None),
    )


def test_joint_import_region_and_roles_per_expert():
    target, refs = "double_blocks.0.img_attn.qkv", [(8, 8)]
    h = _h(32)
    left = torch.zeros(1, 8, 8)
    left[..., :4] = 1
    plan = _joint_plan(target, ["all"] * 4)
    plan.expert_regions = [[left] * 4]
    y = _call(plan, h, "flux", (8, 8), refs)
    grid = (y[:, :16] - h[:, :16]).abs().sum(-1).reshape(2, 4, 4)
    assert grid[:, :, 2:].max() == 0 and grid[:, :, :2].min() > 0
    assert torch.equal(y[:, 16:], h[:, 16:])


def test_joint_scoped_sets_take_the_entries_roles(tmp_path):
    from types import SimpleNamespace

    from kilora.adapter import Settings
    from kilora.bake import ScopedSet
    from kilora.comfyui import patching as P

    header = json.dumps({"__metadata__": {"token_roles": "reference"}}).encode()
    omini = tmp_path / "omini.safetensors"
    omini.write_bytes(struct.pack("<Q", len(header)) + header)
    chain = P.ModChain(
        (
            P.ModEntry("lora", str(tmp_path / "extend.safetensors"), roles="all", axis="creative"),
            P.ModEntry("lora", str(tmp_path / "style.safetensors"), axis="creative"),
            P.ModEntry("lora", str(omini), axis="creative"),
        )
    )

    class Plan:
        mods = [SimpleNamespace(path=None, label="routed LoRAs: x", sets=[1], settings=Settings())]

        def scoped_sets(self):
            return [ScopedSet(adapter_set=SimpleNamespace(name=P.JOINT_SET))]

    (s,) = P._with_regions(Plan(), chain)
    assert s.roles == "all" and s.expert_roles == ["all", "auto", "reference"]
    hints = P._role_hints(Plan(), P.ModChain((P.ModEntry("lora", "/x/klein_edit.safetensors"),)))
    assert len(hints) == 1


def test_outpaint_chain_on_parallel_axes(tmp_path):
    from types import SimpleNamespace

    from kilora.adapter import Settings
    from kilora.bake import ScopedSet
    from kilora.comfyui import patching as P

    border = torch.ones(1, 8, 8)
    border[:, 2:6, 2:6] = 0
    chain = P.ModChain(
        (
            P.ModEntry("lora", str(tmp_path / "extend.safetensors"), axis="functional"),
            P.ModEntry("lora", str(tmp_path / "style.safetensors"), region=border, axis="creative"),
        )
    )

    class Plan:
        mods = [
            SimpleNamespace(path=None, label="extend.safetensors", sets=[1], settings=Settings()),
            SimpleNamespace(path=None, label="routed LoRAs: style", sets=[1], settings=Settings()),
        ]

        def scoped_sets(self):
            return [
                ScopedSet(adapter_set=SimpleNamespace(name="extend")),
                ScopedSet(adapter_set=SimpleNamespace(name=P.JOINT_SET)),
            ]

    edit, style = P._with_regions(Plan(), chain)
    assert (edit.axis, edit.gate_mode, edit.fixed_depth, edit.roles) == (
        "functional",
        "none",
        1,
        "auto",
    )
    assert edit.region is None and edit.expert_regions is None
    assert style.axis == "creative" and style.roles == "all"
    assert style.expert_roles == ["auto"] and style.expert_regions[0] is border


def test_several_token_axes_run_as_one_token_axis():
    from kilora import route_freeze as R
    from kilora.compute import sampling_progress

    (s,) = [
        dataclasses.replace(s, form="input", gate_mode="adaptive", route_freeze=0.5)
        for s in _sets(1, WIDTH, 4, 2, 8, seed=6)
    ]
    target = "transformer_blocks.0.ff.net.0.proj"
    g = torch.Generator().manual_seed(7)
    h = torch.randn(2, 2, 4, 4, WIDTH, generator=g)
    left = torch.zeros(1, 8, 8)
    left[..., :4] = 1
    plan = TargetPlan(sets=[s], depth=3, name=target, regions=[None], roles=["all"])
    plan.expert_regions = [[left] * 4]
    cache, flat_cache = {}, {}
    for p in (0.2, 0.6, 0.8):
        with (
            torch.no_grad(),
            dropout_key(1, p),
            sampling_progress(p),
            latent_size((2, 8, 8)),
            token_layouts(get_family("anima").layout),
        ):
            with R.freezing(cache):
                y = plan.run(h, h)
            masks = [m.reshape(m.shape[0], -1, 1) for m in plan._expert_masks(h)[0]]
            with R.freezing(flat_cache):
                want = C.run_layer(
                    h.flatten(1, -2),
                    h.flatten(1, -2),
                    [s],
                    3,
                    target=target,
                    expert_regions=[masks],
                )
        assert y.shape == h.shape
        assert torch.equal(y, want.reshape(h.shape))
    entry = cache[target]
    cols = s.down.shape[0]
    assert entry["up"].numel() <= 2 * cols * WIDTH
    d = (y - h).abs().sum(-1)
    assert d[..., 2:].max() == 0 and d[..., :2].min() > 0
