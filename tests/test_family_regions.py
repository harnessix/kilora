# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.families import get_family
from kilora.family_spec import TokenLayout, map_region
from kilora.region import latent_size, layer_region, token_layouts


def _left_side(h=8, w=8, batch=1):
    m = torch.zeros(batch, h, w)
    m[:, :, : w // 2] = 1
    return m


def test_image2d_grid_with_patch():
    layout = TokenLayout("image2d", (2, 2))
    r = map_region(_left_side(), 4 * 4, layout, (8, 8))
    assert r.shape == (1, 16)
    assert r.reshape(4, 4)[:, :2].eq(1).all() and r.reshape(4, 4)[:, 2:].eq(0).all()
    r = map_region(_left_side(), 16 + 5, layout, (8, 8))
    assert r[0, 16:].eq(0).all()
    with pytest.raises(ValueError):
        map_region(_left_side(), 15, layout, (8, 8))


def test_video3d_one_mask_for_every_frame():
    layout = TokenLayout("video3d", (1, 2, 2))
    r = map_region(_left_side(), 3 * 16, layout, (3, 8, 8))
    frames = r.reshape(3, 4, 4)
    assert torch.equal(frames[0], frames[1]) and torch.equal(frames[1], frames[2])
    assert frames[:, :, :2].eq(1).all() and frames[:, :, 2:].eq(0).all()


def test_video3d_mask_per_frame():
    layout = TokenLayout("video3d", (1, 2, 2))
    masks = torch.zeros(3, 8, 8)
    masks[1] = 1
    r = map_region(masks, 3 * 16, layout, (3, 8, 8))
    frames = r.reshape(3, 16)
    assert frames[0].eq(0).all() and frames[1].eq(1).all() and frames[2].eq(0).all()
    pixel = torch.zeros(6, 8, 8)
    pixel[2:4] = 1
    r = map_region(pixel, 3 * 16, layout, (3, 8, 8)).reshape(3, 16)
    assert r[1].eq(1).all() and r[0].eq(0).all()
    r = map_region(masks[None].repeat(2, 1, 1, 1), 48, layout, (3, 8, 8))
    assert r.shape == (2, 48)


def test_video_of_one_frame_keeps_batch_rows():
    layout = TokenLayout("video3d", (1, 2, 2))
    masks = torch.stack([_left_side()[0], 1 - _left_side()[0]])
    r = map_region(masks, 16, layout, (1, 8, 8))
    assert r.shape == (2, 16) and torch.equal(r[0], 1 - r[1])


def test_audio1d():
    layout = TokenLayout("audio1d", (2,))
    mask = torch.zeros(16)
    mask[:8] = 1
    r = map_region(mask, 8, layout, (16,))
    assert r.tolist() == [[1.0] * 4 + [0.0] * 4]


def test_joint_sequence_text_span_is_never_masked():
    layout = TokenLayout("image2d", (2, 2), text="before")
    r = map_region(_left_side(), 5 + 16, layout, (8, 8))
    assert r[0, :5].eq(0).all()
    assert torch.equal(
        r[0, 5:], map_region(_left_side(), 16, TokenLayout("image2d", (2, 2)), (8, 8))[0]
    )
    r = map_region(_left_side(), 5 + 16 + 3, layout, (8, 8), text_tokens=5, text_value=1.0)
    assert r[0, :5].eq(1).all() and r[0, 21:].eq(0).all()
    after = TokenLayout("image2d", (2, 2), text="after")
    r = map_region(_left_side(), 16 + 4 + 3, after, (8, 8), text_tokens=4, text_value=1.0)
    assert torch.equal(
        r[0, :16], map_region(_left_side(), 16, TokenLayout("image2d", (2, 2)), (8, 8))[0]
    )
    assert r[0, 16:20].eq(1).all() and r[0, 20:].eq(0).all()


def test_joint_sequence_with_padded_image_tokens():
    layout = TokenLayout("image2d", (2, 2), text="before", pad=32)
    r = map_region(_left_side(), 7 + 32, layout, (8, 8), text_tokens=4, text_value=1.0)
    assert r[0, :7].eq(1).all() and r[0, 7 + 16 :].eq(0).all()


def test_text_layers_have_no_region():
    off = map_region(_left_side(), 77, TokenLayout("text1d"), (8, 8))
    assert off.shape == (1, 77) and off.eq(0).all()
    assert map_region(_left_side(), 40, TokenLayout("set"), (8, 8)) is None


@pytest.mark.parametrize(
    ("family", "target", "tokens", "latent", "kind"),
    [
        ("flux", "double_blocks.0.img_attn.qkv", 16, (8, 8), "image"),
        ("flux", "double_blocks.0.txt_attn.qkv", 7, (8, 8), "off"),
        ("flux", "single_blocks.3.linear1", 7 + 16, (8, 8), "joint"),
        ("hunyuan-video", "single_blocks.0.linear2", 7 + 32, (2, 8, 8), "joint"),
        ("wan", "blocks.0.self_attn.q", 2 * 16, (2, 8, 8), "image"),
        ("wan", "blocks.0.cross_attn.k", 9, (2, 8, 8), "off"),
        ("ltxv", "transformer_blocks.0.attn1.to_q", 2 * 64, (2, 8, 8), "image"),
        ("qwen-image", "transformer_blocks.0.attn.to_q", 16, (8, 8), "image"),
        ("qwen-image", "transformer_blocks.0.attn.add_q_proj", 7, (8, 8), "off"),
        ("qwen-text", "layers.0.self_attn.q_proj", 12, (8, 8), "off"),
        ("hunyuan3d", "single_blocks.0.linear1", 40, (8, 8), "none"),
        ("ace-step15", "decoder.layers.0.self_attn.q_proj", 4, (8,), "audio"),
        ("ace-step15", "decoder.layers.0.cross_attn.k_proj", 9, (8,), "off"),
        ("anima", "transformer_blocks.0.attn1.to_q", 16, (1, 8, 8), "image"),
        ("anima", "transformer_blocks.0.attn2.to_k", 9, (1, 8, 8), "off"),
        ("z-image", "layers.0.attention.to_q", 7 + 32, (8, 8), "joint"),
        ("moss-image", "layers.0.proj.qkv", 64 + 7, (8, 8), "image"),
        ("flux3", "single_blocks.0.qkv", 64 + 7, (8, 8), "image"),
        ("flux3", "txt_mode_blocks.0.qkv", 7, (8, 8), "off"),
        ("flux3", "content_mode_blocks.image_cond.0.qkv", 64, (8, 8), "off"),
        ("flux3", "content_mode_blocks.image.0.qkv", 64, (8, 8), "image"),
        ("sd15", "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k", 77, (8, 8), "off"),
    ],
)
def test_family_layouts_through_layer_region(family, target, tokens, latent, kind):
    spec = get_family(family)
    mask = torch.tensor([[1.0] * 4 + [0.0] * 4]) if kind == "audio" else _left_side()
    with latent_size(latent), token_layouts(spec.layout, text_tokens=7):
        r = layer_region(mask, tokens, target=target)
    if kind == "off":
        assert r.shape == (1, tokens) and r.eq(0).all()
        return
    if kind == "none":
        assert r is None
        return
    assert r.shape == (1, tokens)
    if kind == "joint":
        assert r[0, :7].eq(0).all()
    assert r.max() == 1 and r.min() == 0


def test_layer_region_without_layouts_is_unchanged():
    mask = _left_side(12, 16)
    with latent_size((12, 16)):
        old = layer_region(mask, 3 * 4)
        new = layer_region(mask, 3 * 4, target="anything")
    assert torch.equal(old, new)
    with latent_size((1, 12, 16)):
        assert torch.equal(layer_region(mask, 3 * 4), old)


def test_text_at_the_end_after_references():
    layout = TokenLayout("image2d", (2, 2), text="end")
    r = map_region(_left_side(), 16 + 5 + 4, layout, (8, 8), text_tokens=4, text_value=1.0)
    plain = map_region(_left_side(), 16, TokenLayout("image2d", (2, 2)), (8, 8))
    assert torch.equal(r[0, :16], plain[0])
    assert r[0, 16:21].eq(0).all() and r[0, 21:].eq(1).all()
