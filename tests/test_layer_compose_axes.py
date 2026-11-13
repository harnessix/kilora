# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import fnmatch

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora.call_values import ComposedAlpha, axis_alphas, current_axis_alphas
from kilora.comfyui import layer_compose as lc
from kilora.comfyui.layer_compose import ComposedLayout, Segment, axis_maps, segment_axes
from kilora.comfyui.patching import _compose_axis_scope, _compose_scope
from kilora.compute import run_layer
from kilora.family_spec import TokenLayout
from kilora.region import latent_size, token_layouts

LATENT = (4, 4)
GLOBAL_TOKENS = 4
LAYER_TOKENS = 3
TEXT = GLOBAL_TOKENS + LAYER_TOKENS
TO_K = "blocks.0.attn2.to_k"
IMAGE = "blocks.0.mlp.layer1"


def _box():
    mask = torch.zeros(1, *LATENT)
    mask[0, :2, :2] = 1.0
    return mask


def _bench(global_axis="creative"):
    return ComposedLayout(
        (
            Segment("global", global_axis, GLOBAL_TOKENS, None, 1.0),
            Segment("layer", "ip", LAYER_TOKENS, _box(), 1.0),
        )
    )


def _inside():
    return _box()[0].reshape(-1).bool()


def _resolve(target):
    if fnmatch.fnmatch(target, "*.attn2.to_k") or fnmatch.fnmatch(target, "*.attn2.to_v"):
        return TokenLayout("text1d")
    return TokenLayout("image2d", (1, 1))


def _axis_set(axis, seed, **kw):
    (s,) = _sets(1, 16, 4, 2, 8, seed=seed)
    return dataclasses.replace(s, axis=axis, key=f"{axis}{seed}", **kw)


def _args(per=1):
    layout = _bench()
    key = lc.register((layout, layout))
    return {
        "input": torch.zeros(2 * per, 4, *LATENT),
        "timestep": torch.ones(2 * per),
        "c": {"transformer_options": {"cond_or_uncond": [0, 1]}},
    }, key


def test_segment_axes_follow_the_layer_axes():
    assert segment_axes(Segment("global", None, 1, None, 1.0)) == C.AXES
    assert segment_axes(Segment("global", "creative", 1, None, 1.0)) == ("creative", "functional")
    assert segment_axes(Segment("layer", "ip", 1, None, 1.0)) == ("ip",)
    assert segment_axes(Segment("layer", "functional", 1, None, 1.0)) == ("functional",)


def test_bench_numbers_inside_and_outside_the_ip_box():
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    inside = _inside()
    ip = maps["ip"].image.reshape(2, -1)
    creative = maps["creative"].image.reshape(2, -1)
    functional = maps["functional"].image.reshape(2, -1)
    assert pytest.approx(0.1) == lc.BETA
    assert torch.allclose(ip[:, inside], torch.full_like(ip[:, inside], 0.9))
    assert torch.allclose(creative[:, inside], torch.full_like(creative[:, inside], 0.1))
    assert torch.allclose(ip[:, ~inside], torch.zeros_like(ip[:, ~inside]))
    assert torch.allclose(creative[:, ~inside], torch.ones_like(creative[:, ~inside]))
    assert torch.allclose(functional, creative)


def test_shared_mount_weights_give_equal_shares_in_the_box(monkeypatch):
    monkeypatch.setattr(lc, "MOUNT_WEIGHTS", "shared")
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    inside = _inside()
    ip = maps["ip"].image.reshape(2, -1)
    creative = maps["creative"].image.reshape(2, -1)
    assert torch.allclose(ip[:, inside], torch.full_like(ip[:, inside], 0.5))
    assert torch.allclose(creative[:, inside], torch.full_like(creative[:, inside], 0.5))
    assert torch.allclose(creative[:, ~inside], torch.ones_like(creative[:, ~inside]))


def test_mount_weights_ignore_the_text_end():
    layout = _bench()
    early = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    late = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.9)
    assert lc.TEXT_END < 0.9
    for a in C.AXES:
        assert torch.equal(early[a].image, late[a].image)


def test_text_segments_run_the_axes_of_their_layer():
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    g, n = [1.0] * GLOBAL_TOKENS, [0.0] * LAYER_TOKENS
    expected = {
        "creative": g + n,
        "functional": g + n,
        "ip": [0.0] * GLOBAL_TOKENS + [1.0] * LAYER_TOKENS,
    }
    for a, values in expected.items():
        assert torch.equal(maps[a].text, torch.tensor([values, values]))
    plain = _bench(global_axis=None)
    maps = axis_maps((plain, plain), torch.tensor([0]), LATENT, 0.0)
    for a in C.AXES:
        assert torch.equal(maps[a].text[0, :GLOBAL_TOKENS], torch.ones(GLOBAL_TOKENS))


def test_the_model_call_records_the_same_weights_in_both_branches():
    records = []

    def stand_in(args):
        alphas = current_axis_alphas()
        assert set(alphas) == set(C.AXES)
        assert all(isinstance(v, ComposedAlpha) for v in alphas.values())
        batch = args["input"].shape[0]
        image = torch.zeros(batch, LATENT[0] * LATENT[1], 16)
        text = torch.zeros(batch, TEXT, 16)
        with latent_size(LATENT), token_layouts(_resolve, text_tokens=TEXT):
            records.append(
                {
                    a: (
                        C._axis_alpha(alphas[a], image, IMAGE),
                        C._axis_alpha(alphas[a], text, TO_K),
                    )
                    for a in C.AXES
                }
            )

    for per in (1, 2):
        args, key = _args(per)
        for progress in (0.0, 0.5, 1.0):
            with _compose_scope(torch.tensor([key]), args):
                scope = _compose_axis_scope(args, progress)
                assert scope is not None
                with scope:
                    stand_in(args)
    assert len(records) == 6
    inside = _inside()
    for record in records:
        for _a, (image, text) in record.items():
            mid = image.shape[0] // 2
            assert torch.equal(image[:mid], image[mid:])
            assert torch.equal(text[:mid], text[mid:])
        ip_image, ip_text = record["ip"]
        assert torch.allclose(ip_image[:, inside, 0], torch.full((ip_image.shape[0], 4), 0.9))
        assert torch.equal(ip_text[0, :, 0], torch.tensor([0.0] * 4 + [1.0] * 3))
        creative_image, creative_text = record["creative"]
        rows = creative_image.shape[0]
        assert torch.allclose(creative_image[:, ~inside, 0], torch.ones(rows, 12))
        assert torch.equal(creative_text[0, :, 0], torch.tensor([1.0] * 4 + [0.0] * 3))


def test_no_layout_gives_no_axis_scope():
    args, _ = _args()
    assert _compose_axis_scope(args, 0.0) is None


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("target", [TO_K, "blocks.0.attn2.to_v"])
def test_a_key_mount_changes_its_own_segment_only(form, target):
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    kw = {"form": form}
    ip = _axis_set("ip", 3, **kw)
    creative = _axis_set("creative", 4, **kw)
    context = torch.randn(2, TEXT, 16, generator=torch.Generator().manual_seed(9))
    with torch.no_grad(), latent_size(LATENT), token_layouts(_resolve, text_tokens=TEXT):
        full_ip = run_layer(context, context, [ip], 1, target=target)
        full_creative = run_layer(context, context, [creative], 1, target=target)
        with axis_alphas(maps):
            got_ip = run_layer(context, context, [ip], 1, target=target)
            got_creative = run_layer(context, context, [creative], 1, target=target)
    g = GLOBAL_TOKENS
    assert not torch.allclose(full_ip[:, g:], context[:, g:])
    assert torch.equal(got_ip[:, :g], context[:, :g])
    assert torch.allclose(got_ip[:, g:], full_ip[:, g:], atol=1e-6)
    assert torch.allclose(got_creative[:, :g], full_creative[:, :g], atol=1e-6)
    assert torch.equal(got_creative[:, g:], context[:, g:])


def test_a_text_layer_of_another_length_runs_at_full_weight():
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0]), LATENT, 0.0)
    context = torch.zeros(1, TEXT + 5, 16)
    with latent_size(LATENT), token_layouts(_resolve, text_tokens=TEXT + 5):
        assert C._axis_alpha(maps["ip"], context, TO_K) == 1.0


def test_tensor_scope_routing_averages_over_the_axis_tokens(monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", False)
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    ip = _axis_set("ip", 6, routing_scope="tensor")
    y = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(2)) * 3
    seen = []
    real = C.region_mean

    def recording(logits, region):
        seen.append(None if region is None else region.detach().clone())
        return real(logits, region)

    monkeypatch.setattr(C, "region_mean", recording)
    with torch.no_grad(), latent_size(LATENT), token_layouts(_resolve):
        with axis_alphas(maps):
            got = run_layer(y, y, [ip], 1, target=IMAGE)
        assert seen and all(r is not None for r in seen)
        gate = _inside().float()
        for region in seen:
            rows = region.shape[0]
            assert torch.equal(region.reshape(rows, -1), gate.expand(rows, -1))
        seen.clear()
        routed = run_layer(
            y, y, [dataclasses.replace(ip, region=gate[None].expand(2, -1))], 1, target=IMAGE
        )
        whole = run_layer(y, y, [ip], 1, target=IMAGE)
    alpha = maps["ip"].image.reshape(2, -1, 1)
    assert torch.allclose(got, y + alpha * (routed - y), atol=1e-6)
    assert not torch.allclose(got, y + alpha * (whole - y), atol=1e-6)


def test_token_scope_routing_keeps_its_region():
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    ip = _axis_set("ip", 6)
    y = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(2))
    weight = torch.ones(2, 16, 1)
    assert C._composed_regions([ip], None, weight, y) is None
    assert C._composed_regions([ip], None, 1.0, y) is None
    with torch.no_grad(), latent_size(LATENT), token_layouts(_resolve):
        whole = run_layer(y, y, [ip], 1, target=IMAGE)
        with axis_alphas(maps):
            got = run_layer(y, y, [ip], 1, target=IMAGE)
    alpha = maps["ip"].image.reshape(2, -1, 1)
    assert torch.allclose(got, y + alpha * (whole - y), atol=1e-6)


def test_tensor_scope_region_multiplies_with_the_set_region():
    tensor = _axis_set("ip", 6, routing_scope="tensor")
    own = torch.zeros(2, 16)
    own[:, 0] = 1.0
    own[:, 15] = 1.0
    weight = _inside().float().reshape(1, 16, 1).expand(2, -1, -1)
    y = torch.zeros(2, 16, 16)
    (region,) = C._composed_regions([dataclasses.replace(tensor, region=own)], None, weight, y)
    expected = torch.zeros(2, 16)
    expected[:, 0] = 1.0
    assert torch.equal(region[..., 0], expected)


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("axis", ["ip", "creative"])
def test_a_depth_above_one_scales_the_change_once(depth, axis):
    layout = _bench()
    maps = axis_maps((layout, layout), torch.tensor([0, 1]), LATENT, 0.0)
    s = _axis_set(axis, 3)
    y = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(5))
    with torch.no_grad(), latent_size(LATENT), token_layouts(_resolve):
        full = run_layer(y, y, [s], depth, target=IMAGE)
        one = run_layer(y, y, [s], 1, target=IMAGE)
        with axis_alphas(maps):
            got = run_layer(y, y, [s], depth, target=IMAGE)
    assert not torch.allclose(full, one)
    alpha = maps[axis].image.reshape(2, -1, 1)
    assert torch.allclose(got, y + alpha * (full - y), atol=1e-5)
