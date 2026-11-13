# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import uuid

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, FakeZImage

from kilora.bake import is_baked
from kilora.bindings import BINDINGS
from kilora.comfyui.takeover import KiLoRARouteLoaded, group_patches, group_state


class Adapter:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights


class PatchPatcher(FakePatcher):
    def __init__(self, model, object_patches=None, model_options=None, patches=None):
        super().__init__(model, object_patches, model_options)
        self.patches = {k: list(v) for k, v in (patches or {}).items()}
        self.patches_uuid = uuid.uuid4()

    def clone(self):
        return PatchPatcher(self.model, self.object_patches, self.model_options, self.patches)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        for key, value in patches.items():
            self.patches.setdefault(key, []).append(
                (strength_patch, value, strength_model, None, None)
            )
        self.patches_uuid = uuid.uuid4()


class FakeClip:
    def __init__(self, patches=()):
        self.patches = list(patches)

    def clone(self):
        return FakeClip(self.patches)


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    new_model = model.clone() if model is not None else None
    if new_model is not None:
        new_model.add_patches(lora, strength_model)
    new_clip = None
    if clip is not None:
        new_clip = clip.clone()
        new_clip.patches.append(("te", strength_clip))
    return new_model, new_clip


SIZES = {
    "attention.qkv": (12, 36),
    "attention.out": (12, 12),
    "feed_forward.w1": (12, 24),
    "feed_forward.w2": (24, 12),
    "feed_forward.w3": (12, 24),
}


def make_lora(layers, rank, seed, kind="lora", alpha=None):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for layer in layers:
        block, name = layer.split(".", 1)
        n_in, n_out = SIZES[name]
        key = f"diffusion_model.layers.{block}.{name}.weight"
        a = None if alpha is None else torch.tensor(float(alpha))
        if kind == "lora":
            up = torch.randn(n_out, rank, generator=gen)
            down = torch.randn(rank, n_in, generator=gen)
            out[key] = Adapter("lora", (up, down, a, None, None, None))
        elif kind == "lokr":
            w1 = torch.randn(2, 2, generator=gen)
            w2_a = torch.randn(n_out // 2, rank, generator=gen)
            w2_b = torch.randn(rank, n_in // 2, generator=gen)
            out[key] = Adapter("lokr", (w1, None, a, None, None, w2_a, w2_b, None, None))
        else:
            w = [
                torch.randn(*s, generator=gen)
                for s in ((n_out, rank), (rank, n_in), (n_out, rank), (rank, n_in))
            ]
            out[key] = Adapter("loha", (w[0], w[1], a, w[2], w[3], None, None, None))
    return out


def full_delta(adapter):
    kind, w = adapter.name, adapter.weights
    if kind == "lora":
        scale = 1.0 if w[2] is None else float(w[2]) / w[1].shape[0]
        return w[0] @ w[1] * scale
    if kind == "lokr":
        scale = 1.0 if w[2] is None else float(w[2]) / w[6].shape[0]
        return torch.kron(w[0], w[5] @ w[6]) * scale
    scale = 1.0 if w[2] is None else float(w[2]) / w[1].shape[0]
    return (w[0] @ w[1]) * (w[3] @ w[4]) * scale


def base_patcher():
    return PatchPatcher(FakeBaseModel(FakeZImage()))


def assert_matches(load, patches, lora, cut=8):
    assert set(load.entries) == set(lora)
    for key, i in load.entries.items():
        assert patches[key][i][1] is lora[key]
    state, not_targets, _ = group_state(load, patches, BINDINGS["z-image"], cut)
    assert not_targets == []
    for key, adapter in lora.items():
        path = key.removeprefix("diffusion_model.").removesuffix(".weight")
        targets = BINDINGS["z-image"].from_comfy(path)
        got = torch.cat(
            [
                state[f"transformer.{t}.lora_B.weight"] @ state[f"transformer.{t}.lora_A.weight"]
                for t, _ in targets
            ]
        )
        torch.testing.assert_close(got, full_delta(adapter), rtol=1e-4, atol=1e-4)


def check_takeover(patcher, loras, strengths):
    out, report = KiLoRARouteLoaded().route(patcher)
    assert f"Found {len(loras)} LoRA loads" in report and "Ambiguous" not in report
    assert out.patches == {}
    assert patcher.patches
    assert out.object_patches and all(is_baked(v) for v in out.object_patches.values())
    for n, s in enumerate(strengths):
        assert f"LoRA {n + 1} (strength {s:g}" in report
    x = torch.randn(1, 4, 12)
    assert not torch.allclose(out.run(x), patcher.run(x))
    return out, report


def test_lora_loader_twice():
    a = make_lora(["0.attention.qkv", "0.attention.out", "1.attention.qkv"], 4, 1, alpha=2)
    b = make_lora(["0.attention.qkv", "0.feed_forward.w1", "1.feed_forward.w2"], 2, 2)
    model, clip = load_lora_for_models(base_patcher(), FakeClip(), a, 0.8, 0.5)
    model, clip = load_lora_for_models(model, clip, b, 1.0, 1.0)
    grouping = group_patches(model.patches)
    assert [ld.strength for ld in grouping.loads] == [0.8, 1.0]
    assert not grouping.ambiguous
    assert_matches(grouping.loads[0], model.patches, a)
    assert_matches(grouping.loads[1], model.patches, b)
    out, _ = check_takeover(model, [a, b], [0.8, 1.0])
    assert clip.patches == [("te", 0.5), ("te", 1.0)]


def test_lora_loader_model_only_twice_same_signature():
    layers = ["0.attention.qkv", "1.attention.out", "1.feed_forward.w3"]
    a, b = make_lora(layers, 4, 1), make_lora(layers, 4, 2)
    model, _ = load_lora_for_models(base_patcher(), None, a, 1.0, 0)
    model, _ = load_lora_for_models(model, None, b, 1.0, 0)
    grouping = group_patches(model.patches)
    assert len(grouping.loads) == 2 and not grouping.ambiguous
    assert_matches(grouping.loads[0], model.patches, a)
    assert_matches(grouping.loads[1], model.patches, b)
    check_takeover(model, [a, b], [1.0, 1.0])


def test_power_lora_loop_three_kinds_on_layer_subsets():
    a = make_lora(["0.attention.qkv", "0.attention.out", "1.attention.qkv"], 4, 1)
    b = make_lora(["0.attention.out", "1.attention.out", "1.feed_forward.w1"], 2, 2, "lokr", 1)
    c = make_lora(["1.attention.qkv", "0.feed_forward.w3", "1.feed_forward.w1"], 2, 3, "loha")
    model, clip = base_patcher(), FakeClip()
    for lora, s in ((a, 1.0), (b, 0.6), (c, -0.5)):
        model, clip = load_lora_for_models(model, clip, lora, s, s)
    grouping = group_patches(model.patches)
    assert [ld.strength for ld in grouping.loads] == [1.0, 0.6, -0.5]
    for load, lora in zip(grouping.loads, (a, b, c), strict=True):
        assert_matches(load, model.patches, lora, cut=64)
    _, report = check_takeover(model, [a, b, c], [1.0, 0.6, -0.5])
    assert "Cut to rank 4 by SVD" in report


def test_same_signature_on_overlapping_subsets_is_reported_not_guessed():
    a = make_lora(["0.attention.qkv", "0.attention.out"], 4, 1)
    b = make_lora(["0.attention.out", "1.attention.out"], 4, 2)
    model, _ = load_lora_for_models(base_patcher(), None, a, 1.0, 0)
    model, _ = load_lora_for_models(model, None, b, 1.0, 0)
    out, report = KiLoRARouteLoaded().route(model)
    assert "Ambiguous" in report and "Found 0 LoRA loads" in report
    assert {k: len(v) for k, v in out.patches.items()} == {
        k: len(v) for k, v in model.patches.items()
    }
    assert out.object_patches == {}


def test_same_signature_split_by_another_lora_is_reported():
    a = make_lora(["0.attention.qkv"], 4, 1)
    b = make_lora(["0.attention.out"], 2, 2)
    c = make_lora(["1.attention.out"], 4, 3)
    model = base_patcher()
    for lora in (a, b, c):
        model, _ = load_lora_for_models(model, None, lora, 1.0, 0)
    out, report = KiLoRARouteLoaded().route(model)
    assert "Ambiguous" in report
    assert out.object_patches == {}


def test_only_the_ambiguous_loras_stay_plain():
    a = make_lora(["0.attention.qkv", "0.attention.out"], 4, 1)
    b = make_lora(["0.attention.out", "1.attention.out"], 4, 2)
    c = make_lora(["1.feed_forward.w2"], 2, 3)
    model = base_patcher()
    for lora, s in ((a, 1.0), (b, 1.0), (c, 0.7)):
        model, _ = load_lora_for_models(model, None, lora, s, 0)
    out, report = KiLoRARouteLoaded().route(model)
    assert "Ambiguous" in report and "Found 1 LoRA loads" in report
    assert "diffusion_model.layers.1.feed_forward.w2.weight" not in out.patches
    assert len(out.patches["diffusion_model.layers.0.attention.out.weight"]) == 2


def test_no_patches_gives_the_model_back():
    out, report = KiLoRARouteLoaded().route(base_patcher())
    assert "nothing to take over" in report and out.object_patches == {}


def test_same_signature_next_to_each_other_on_disjoint_layers_is_reported():
    from kilora.comfyui.takeover import MERGE_NOTE

    a = make_lora(["0.attention.qkv"], 4, 1)
    b = make_lora(["1.attention.out"], 4, 2)
    model = base_patcher()
    for lora in (a, b):
        model, _ = load_lora_for_models(model, None, lora, 1.0, 0)
    out, report = KiLoRARouteLoaded().route(model)
    assert "Found 1 LoRA loads" in report
    assert MERGE_NOTE in report


def test_separate_block_ranges_get_a_warning():
    from kilora.comfyui.takeover import _blocks, _ranges

    keys = [f"diffusion_model.layers.{b}.attention.qkv.weight" for b in (0, 1, 2, 7, 8, 12)]
    assert _blocks(keys) == [0, 1, 2, 7, 8, 12]
    assert _ranges(_blocks(keys)) == [(0, 2), (7, 8), (12, 12)]


def test_settings_go_through_resolve_settings(monkeypatch):
    from kilora.comfyui import patching

    seen = {}
    real = patching.apply

    def spy(model, chain, depth, env, iteration, schedule, scope, select, **kw):
        seen.update(depth=depth, scope=scope, select=select)
        return real(model, chain, depth, env, iteration, schedule, scope, select, **kw)

    monkeypatch.setattr(patching, "apply", spy)
    a = make_lora(["0.attention.qkv"], 4, 1)
    model, _ = load_lora_for_models(base_patcher(), None, a, 1.0, 0)
    KiLoRARouteLoaded().route(model, {"mix": "keep_each_strong"})
    assert seen == {"depth": -1, "scope": "token", "select": "top1"}
    out, report = KiLoRARouteLoaded().route(model, {"mix": "stack"})
    assert "stack" in report and out.patches == model.patches
    with pytest.raises(ValueError, match="unknown KiLoRA settings"):
        KiLoRARouteLoaded().route(model, {"depht": 2})
