# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from kilora_editor_chunk.plugin import ChunkLive
from test_chunk_plugin import _packs

from kilora.comfyui import live_write, patching
from kilora.cross_layer import inner_attention_links
from kilora.live import open_session
from kilora.testing import make_live_mod


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_k = torch.nn.Linear(8, 8)
        self.to_v = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(8, 8))

    def forward(self, x, delta=None):
        q = self.to_q(x).reshape(-1, 2, 4).transpose(0, 1)
        k = self.to_k(x).reshape(-1, 2, 4).transpose(0, 1)
        v = self.to_v(x)
        if delta is not None:
            v = v + delta
        v = v.reshape(-1, 2, 4).transpose(0, 1)
        mixed = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(0, 1)
        return self.to_out(mixed.reshape(-1, 8))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for block in self.transformer_blocks:
            block.attn1 = Attention()


class IdentityBinding:
    def from_comfy(self, name):
        return [(name, None)]


class Patcher:
    model_options = {}

    def __init__(self, model):
        self.model = SimpleNamespace(diffusion_model=model)

    def clone(self):
        return Patcher(self.model.diffusion_model)

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper


A = "transformer_blocks.0.attn1.to_v"
B = "transformer_blocks.0.attn1.to_out.0"
C = "transformer_blocks.1.attn1.to_v"


def test_attention_response_matches_finite_difference():
    torch.manual_seed(4)
    model = Toy().double()
    attention = model.transformer_blocks[0].attn1
    x = torch.randn(19, 8, dtype=torch.float64)
    ua = torch.randn(3, 8, dtype=torch.float64)
    ub = torch.randn(4, 8, dtype=torch.float64)
    weight = attention.to_out[0].weight.detach()
    pair = inner_attention_links([A, B], {B: weight}, {A: (8, 8)})
    response = ub @ pair[(A, B)] @ ua.T
    eps = 1e-5
    columns = []
    for direction in ua:
        plus = attention(x, eps * direction)
        minus = attention(x, -eps * direction)
        columns.append(((plus - minus) / (2 * eps) @ ub.T).mean(0))
    torch.testing.assert_close(response, torch.stack(columns, dim=1), rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("mode", ["jvp", "rls", "off"])
@pytest.mark.parametrize("missing", [False, True])
def test_session_setup_and_reporting(tmp_path, monkeypatch, mode, missing):
    from kilora.live_sets import LIVE_PRESETS

    monkeypatch.setitem(
        LIVE_PRESETS, "light", dataclasses.replace(LIVE_PRESETS["light"], target_stride=1)
    )
    model = Toy()
    make_live_mod(tmp_path / "live.kmod", model=model, targets=(A, B, C))
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    session.editor_options = {"cross_layer": mode}
    editor = ChunkLive(str(_packs(session, tmp_path)))
    monkeypatch.setattr(patching, "sessions_of", lambda _: [session])
    monkeypatch.setattr(patching, "detect_family", lambda _: IdentityBinding())
    monkeypatch.setattr(live_write, "async_enabled", lambda: False)
    if missing:
        model.transformer_blocks[0].attn1.to_out[0].weight = None
    live_write.record(Patcher(model), editor)
    fallback = SimpleNamespace(weights=torch.eye(8), residual_variance=1.0)
    session.editor_state.links = {(A, B): fallback, (B, C): fallback}
    editor.begin_write()
    inner = editor._cross_link((A, B))
    outer = editor._cross_link((B, C))
    note = editor.finish_write(session, [])
    assert outer is fallback
    if mode == "jvp":
        assert f"cross_layer_jvp={0 if missing else 1}/2" in note
        assert 'cross_layer_fallback="no isolated replay or paired activations"' in note
        assert ("RLS links are used." in note) == missing
        if missing:
            assert inner is fallback
        else:
            assert inner is not fallback and inner.residual_variance == 0
            bases = [editor._jvp_bases[h][1] for h in (A, B)]
            expected = bases[1] @ model.transformer_blocks[0].attn1.to_out[0].weight @ bases[0].T
            torch.testing.assert_close(inner.weights, expected.T)
            assert editor._cross_link((A, B)) is inner
    else:
        assert inner is fallback
        assert "cross_layer_jvp=" not in note
        assert not hasattr(session, "cross_layer_inner")


def test_host_weight_fused_slice_and_incompatible_shapes(monkeypatch):
    model = torch.nn.Module()
    model.fused = torch.nn.Linear(4, 12, bias=False)
    binding = SimpleNamespace(from_comfy=lambda name: [(A, (2, 3))] if name == "fused" else [])
    monkeypatch.setattr(patching, "detect_family", lambda _: binding)
    weights = patching.host_weights(Patcher(model), [A])
    torch.testing.assert_close(weights[A], model.fused.weight[8:12])
    assert inner_attention_links([A, B], {B: torch.zeros(8, 9)}, {A: (8, 8)}) == {}


def test_anima_host_mapping_uses_base_output_projection():
    model = torch.nn.Module()
    model.kilora_family = "anima"
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    attn = model.blocks[0].self_attn = torch.nn.Module()
    attn.v_proj = torch.nn.Linear(6, 8)
    attn.output_proj = torch.nn.Linear(8, 5)
    patcher = Patcher(model)
    binding = patching.detect_family(patcher)
    shapes = patching.diffusers_shapes(patcher, binding)
    weights = patching.host_weights(patcher, [A, B])
    assert shapes[A] == (6, 8)
    assert shapes[B] == (8, 5)
    assert inner_attention_links([A, B], weights, shapes)[(A, B)] is weights[B]
    torch.testing.assert_close(weights[B], attn.output_proj.weight.detach())
