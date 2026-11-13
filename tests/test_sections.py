# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher
from test_families import H, _ab, _model, _save

from kilora import Settings, load_mod, save_mod
from kilora.errors import UnsupportedFeatureError
from kilora.modfile import FEATURES
from kilora.sections import (
    SECTION_KINDS,
    LayerChange,
    Section,
    SectionKind,
    layer_changes,
    register_section,
)
from kilora.sets import new_keyed_set


def _mod(tmp_path, sections, name="m"):
    s = new_keyed_set(
        "s",
        {"single_blocks.0.linear2": (H, H)},
        num_experts=2,
        routes=1,
        rank=1,
        router_dim=4,
        router_seed=0,
        seed=0,
    )
    path = tmp_path / f"{name}.kmod"
    manifest = save_mod(
        path, [s], Settings(depth=1), id=name, base_model={"family": "flux"}, sections=sections
    )
    return path, manifest


def test_fixed_lora_round_trip(tmp_path):
    gen = torch.Generator().manual_seed(0)
    a, b = _ab(gen, H, 6 * H)
    data = {"double_blocks.0.img_mod.lin": (a, b)}
    path, manifest = _mod(tmp_path, [Section("fixed-lora", "style", data, {"layers": 1})])
    assert "section.fixed-lora" in manifest["requires_features"]
    (back,) = load_mod(path).sections
    assert (back.kind, back.name, back.meta) == ("fixed-lora", "style", {"layers": 1})
    d, u = back.data["double_blocks.0.img_mod.lin"]
    assert torch.equal(d, a) and torch.equal(u, b)


def test_fixed_delta_round_trip(tmp_path):
    delta = torch.randn(2 * H, H, generator=torch.Generator().manual_seed(1))
    path, manifest = _mod(tmp_path, [Section("fixed-delta", "lokr", {"final_layer.linear": delta})])
    assert "section.fixed-delta" in manifest["requires_features"]
    (back,) = load_mod(path).sections
    assert torch.equal(back.data["final_layer.linear"], delta)
    x = torch.randn(3, H)
    (change,) = layer_changes([back], 0.5)["final_layer.linear"]
    torch.testing.assert_close(change.add(x, torch.zeros(3, 2 * H)), 0.5 * x @ delta.t())


class BiasChange(LayerChange):
    def __init__(self, bias, strength):
        self.bias, self.strength = bias, strength

    def add(self, x, y):
        return y + self.strength * self.bias


def test_a_new_kind_needs_no_core_change(tmp_path):
    handler = SectionKind(
        kind="test-bias",
        since="0.1.1",
        encode=lambda data: {f"{p}/bias": b for p, b in data.items()},
        decode=lambda tensors: {k.rsplit("/", 1)[0]: v for k, v in tensors.items()},
        changes=lambda data, s: {p: [BiasChange(b, s)] for p, b in data.items()},
        scale=lambda data, w: {p: b * w for p, b in data.items()},
    )
    register_section(handler)
    try:
        bias = torch.arange(4.0)
        path, manifest = _mod(tmp_path, [Section("test-bias", "b", {"img_in": bias})])
        assert "section.test-bias" in manifest["requires_features"]
        (back,) = load_mod(path).sections
        (change,) = layer_changes([back], 2.0)["img_in"]
        assert torch.equal(change.add(None, torch.zeros(4)), 2 * bias)
    finally:
        SECTION_KINDS.pop("test-bias")
        FEATURES.pop("section.test-bias")
    with pytest.raises(UnsupportedFeatureError, match="section.test-bias"):
        load_mod(path)


def test_kilora_import_keeps_the_changes_and_apply_loads_them(tmp_path, monkeypatch):
    from test_family_import_cli import _checkpoint

    from kilora.cli import _import
    from kilora.comfyui import nodes as N
    from kilora.fixed_change import is_fixed

    gen = torch.Generator().manual_seed(4)
    a, b = _ab(gen, H, 6 * H)
    w1, w2 = torch.randn(3, 1, generator=gen), torch.randn(H, H, generator=gen)
    t_a, t_b = _ab(gen, H, H)
    sd = {
        "transformer.transformer_blocks.0.norm1.linear.lora_A.weight": a,
        "transformer.transformer_blocks.0.norm1.linear.lora_B.weight": b,
        "lora_unet_single_blocks_1_modulation_lin.lokr_w1": w1,
        "lora_unet_single_blocks_1_modulation_lin.lokr_w2": w2,
        "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": t_a,
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": t_b,
    }
    lora = _save(tmp_path, "style", sd)
    out = tmp_path / "style.kmod.safetensors"
    base = _checkpoint(tmp_path, "flux", "model.diffusion_model.")
    args = argparse.Namespace(
        loras=[lora],
        model=base,
        output=str(out),
        family=None,
        mode="routed",
        weights=None,
        names=None,
        source_info=None,
        id=None,
        license=None,
    )
    assert _import(args) == 0
    mod = load_mod(out)
    kinds = {s.kind: s for s in mod.sections}
    assert set(kinds) == {"fixed-lora", "fixed-delta"}
    lokr = torch.kron(w1, w2)
    assert torch.equal(kinds["fixed-delta"].data["single_blocks.1.modulation.lin"], lokr)

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    model = _model("flux")
    (chain,) = N.KiLoRASelect().select(out.name, 0.5, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(model)), chain, -1)
    fixed = {k for k, v in patched.object_patches.items() if is_fixed(v)}
    assert fixed == {
        "diffusion_model.double_blocks.0.img_mod.lin",
        "diffusion_model.single_blocks.1.modulation.lin",
    }
    v = torch.randn(2, H)
    with torch.no_grad():
        got = patched.object_patches["diffusion_model.double_blocks.0.img_mod.lin"](v)
        want = model.get_submodule("double_blocks.0.img_mod.lin")(v) + 0.5 * v @ a.t() @ b.t()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        got = patched.object_patches["diffusion_model.single_blocks.1.modulation.lin"](v)
        want = model.get_submodule("single_blocks.1.modulation.lin")(v) + 0.5 * v @ lokr.t()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_fixed_copy_follows_the_model_when_it_moves():
    from kilora.fixed_change import fixed_copy, is_fixed
    from kilora.sections import LowRankChange

    torch.manual_seed(0)
    base = torch.nn.Linear(4, 3)
    change = LowRankChange(torch.randn(2, 4), torch.randn(3, 2), strength=0.5)
    parent = torch.nn.Sequential(fixed_copy(base, [change]))
    assert is_fixed(parent[0])
    assert set(parent.state_dict()) == {"0.weight", "0.bias"}
    x = torch.randn(5, 4)
    expected = base(x) + 0.5 * x @ change.down.T @ change.up.T
    torch.testing.assert_close(parent(x), expected)
    parent.double()
    assert parent[0].weight.dtype == torch.float64
    torch.testing.assert_close(parent(x.double()), expected.double(), rtol=1e-5, atol=1e-5)


def _reference_meta():
    return {
        "sha256": "a" * 64,
        "form": "lora",
        "transforms": {
            "strength": 1.0,
            "axis": "auto",
            "roles": "auto",
            "region": None,
            "timeline": [],
            "enabled": True,
        },
    }


def _logic_meta():
    from kilora.comfyui.settings import DEFAULTS

    return {
        "version": 1,
        "settings": {
            **DEFAULTS,
            "axis_alpha": "off",
            "library": False,
            "mix": "stack",
            "depth": 1,
            "depth_schedule": None,
            "iteration": "accumulate",
            "routing_scope": "token",
            "routing_select": "sparsemax",
            "output_gate": "none",
            "gate_kappa": 0.0,
            "path_width": 0,
            "path_select": "converged",
            "route_freeze": 0.0,
        },
    }


@pytest.mark.parametrize("kind,meta", [("weight-ref", _reference_meta()), ("logic", _logic_meta())])
def test_data_section_round_trip_and_old_feature_set(tmp_path, kind, meta, monkeypatch):
    path, manifest = _mod(tmp_path, [Section(kind, "entry", {}, meta)])
    assert f"section.{kind}" in manifest["requires_features"]
    assert load_mod(path).sections[0].meta == meta
    monkeypatch.delitem(FEATURES, f"section.{kind}")
    with pytest.raises(UnsupportedFeatureError, match=f"section.{kind}"):
        load_mod(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("strength", 3),
        ("axis", "bad"),
        ("roles", "bad"),
        ("region", "named"),
        ("enabled", 1),
        ("timeline", [{}]),
    ],
)
def test_reference_transform_validation(tmp_path, key, value):
    meta = _reference_meta()
    meta["transforms"][key] = value
    with pytest.raises(ValueError, match=key):
        _mod(tmp_path, [Section("weight-ref", "entry", {}, meta)])


@pytest.mark.parametrize(
    "key,value",
    [
        ("mix", "bad"),
        ("depth", 17),
        ("iteration", "bad"),
        ("routing_scope", "bad"),
        ("routing_select", "bad"),
        ("output_gate", "bad"),
        ("gate_kappa", -1),
        ("path_width", 9),
        ("path_select", "bad"),
        ("route_freeze", 2),
        ("depth_schedule", "__import__('os')"),
        ("derived", {}),
    ],
)
def test_logic_settings_validation(tmp_path, key, value):
    meta = _logic_meta()
    meta["settings"][key] = value
    with pytest.raises(ValueError, match=key):
        _mod(tmp_path, [Section("logic", "settings", {}, meta)])


@pytest.mark.parametrize(
    "key,value",
    [("sha256", "bad"), ("size", -1), ("filename", 1), ("source_url", 1), ("form", "derived")],
)
def test_reference_metadata_validation(tmp_path, key, value):
    meta = _reference_meta()
    meta[key] = value
    with pytest.raises(ValueError, match=key):
        _mod(tmp_path, [Section("weight-ref", "entry", {}, meta)])


def test_data_sections_reject_tensors_and_duplicate_logic(tmp_path):
    with pytest.raises(ValueError, match="tensors"):
        _mod(tmp_path, [Section("weight-ref", "entry", {"x": torch.ones(1)}, _reference_meta())])
    section = Section("logic", "settings", {}, _logic_meta())
    with pytest.raises(ValueError, match="logic"):
        _mod(tmp_path, [section, section])


@pytest.mark.parametrize("value", [None, "keys", "read_share"])
def test_logic_import_score_is_checked(tmp_path, value):
    meta = _logic_meta()
    meta["settings"]["import_score"] = value
    _mod(tmp_path, [Section("logic", "settings", {}, meta)])


def test_logic_import_score_refuses_unknown(tmp_path):
    meta = _logic_meta()
    meta["settings"]["import_score"] = "bad"
    with pytest.raises(ValueError, match="import_score"):
        _mod(tmp_path, [Section("logic", "settings", {}, meta)])


@pytest.mark.parametrize(
    "key",
    [
        "import_score",
        "axis_alpha",
        "library",
        "candidates",
        "candidate_summary",
        "routing_features",
        "cross_token_region",
        "cross_token_prompt",
        "cross_token_timestep",
    ],
)
def test_logic_requires_newer_settings(tmp_path, key):
    meta = _logic_meta()
    del meta["settings"][key]
    with pytest.raises(ValueError, match=f"{key} is required"):
        _mod(tmp_path, [Section("logic", "settings", {}, meta)])


@pytest.mark.parametrize("key,value", [("attention_route", "add"), ("attention_route_cap", 3.0)])
def test_logic_refuses_removed_attention_settings(tmp_path, key, value):
    meta = _logic_meta()
    meta["settings"][key] = value
    with pytest.raises(ValueError, match=key):
        _mod(tmp_path, [Section("logic", "settings", {}, meta)])
