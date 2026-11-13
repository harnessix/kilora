# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from types import SimpleNamespace

import torch
from kilora_editor_chunk.plugin import ChunkLive
from test_attention_inner_link import A, B, C, IdentityBinding, Patcher, Toy
from test_chunk_plugin import _packs

from kilora.comfyui import live_write, patching
from kilora.cross_layer import inner_attention_links
from kilora.live import open_session
from kilora.testing import make_live_mod


def _shares_storage(view, weight):
    start = weight.untyped_storage().data_ptr()
    return start <= view.data_ptr() < start + weight.untyped_storage().nbytes()


def _bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def test_host_weights_selects_detached_views_including_fused_rows(monkeypatch):
    model = Toy().to(dtype=torch.bfloat16)
    model.fused = torch.nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)

    def mapping(name):
        return [("fused.part", (2, 3))] if name == "fused" else [(name, None)]

    monkeypatch.setattr(patching, "detect_family", lambda _: SimpleNamespace(from_comfy=mapping))
    weights = patching.host_weights(Patcher(model), [B, "fused.part"])
    assert set(weights) == {B, "fused.part"}
    for target, weight in (
        (B, model.transformer_blocks[0].attn1.to_out[0].weight),
        ("fused.part", model.fused.weight),
    ):
        view = weights[target]
        assert view.dtype == weight.dtype and view.device == weight.device
        assert not view.requires_grad and _shares_storage(view, weight)
    torch.testing.assert_close(weights["fused.part"], model.fused.weight[8:12])


def test_session_retains_model_dtype_and_reduces_setup_bytes(tmp_path, monkeypatch):
    from kilora.live_sets import LIVE_PRESETS

    monkeypatch.setitem(
        LIVE_PRESETS, "light", dataclasses.replace(LIVE_PRESETS["light"], target_stride=1)
    )
    model = Toy().to(dtype=torch.bfloat16)
    targets = (A, B, C)
    make_live_mod(tmp_path / "live.kmod", model=model, targets=targets)
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    session.editor_options = {"cross_layer": "jvp"}
    editor = ChunkLive(str(_packs(session, tmp_path)))
    monkeypatch.setattr(patching, "sessions_of", lambda _: [session])
    monkeypatch.setattr(patching, "detect_family", lambda _: IdentityBinding())
    monkeypatch.setattr(live_write, "async_enabled", lambda: False)
    original = patching.host_weights
    seen = []

    def capture(patcher, requested):
        weights = original(patcher, requested)
        seen.append((list(requested), weights))
        return weights

    monkeypatch.setattr(patching, "host_weights", capture)
    patcher = Patcher(model)
    live_write.record(patcher, editor)
    assert len(seen) == 1 and seen[0][0] == [B]
    assert set(session.cross_layer_inner) == {(A, B)}
    assert all(t.dtype == torch.bfloat16 for t in session.cross_layer_inner.values())
    editor.begin_write()
    link = editor._cross_link((A, B))
    bases = [editor._jvp_bases[h][1] for h in (A, B)]
    expected = bases[1] @ model.transformer_blocks[0].attn1.to_out[0].weight.float() @ bases[0].T
    torch.testing.assert_close(link.weights, expected.T)
    assert all(t.dtype == torch.bfloat16 for t in session.cross_layer_inner.values())

    old_weights = {
        name: tensor.float().clone() for name, tensor in original(patcher, targets).items()
    }
    shapes = patching.diffusers_shapes(patcher, IdentityBinding())
    old_inner = inner_attention_links(targets, old_weights, shapes)
    before = _bytes(old_weights.values()) + _bytes(old_inner.values())
    after = _bytes(seen[0][1].values()) + _bytes(session.cross_layer_inner.values())
    print(f"attention setup tensor bytes: before={before}, after={after}")
    assert after < before
