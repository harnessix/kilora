# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import compute as C
from kilora import region_attention as ra
from kilora.call_values import compose_layout
from kilora.comfyui import layer_compose as lc
from kilora.comfyui.layer_compose import ComposedLayout, Segment, axis_maps
from kilora.region_attention import CallState, RegionAttention, comfy_override

LATENT = (8, 8)
TOKENS = 16
DIM, HEADS = 8, 2
MASKS: list = []


def attn(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    MASKS.append(mask)
    b = q.shape[0]
    q, k, v = (t.reshape(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask
    return (logits.softmax(-1) @ v).transpose(1, 2).flatten(2)


@pytest.fixture(autouse=True)
def plain_backend(monkeypatch):
    monkeypatch.setattr(ra, "_masked_backend", lambda func: func)


def box_mask():
    m = torch.zeros(1, *LATENT)
    m[..., :4, :4] = 1.0
    return m


def in_box():
    return box_mask()[0].reshape(4, 2, 4, 2).amax((1, 3)).reshape(-1) > 0


def layout(mask=None, global_axis="creative"):
    return ComposedLayout(
        (
            Segment("global", global_axis, 4, None, 1.0),
            Segment("layer", "ip", 3, box_mask() if mask is None else mask, 1.0),
        )
    )


def hidden(batch=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, TOKENS, DIM, generator=g)


def self_call(layouts, rows, h, progress=0.0, override=None, state=None):
    state = state or CallState(progress=progress, latent=LATENT, compose=True)
    override = override or comfy_override((), RegionAttention(), state)
    state.kind = "self"
    with compose_layout(layouts, rows):
        return override(attn, h, h, h, HEADS)


def test_a_query_in_the_box_gets_the_penalty_on_keys_outside():
    one = layout()
    MASKS.clear()
    self_call((one, one), None, hidden())
    (mask,) = MASKS
    assert mask.shape == (1, 1, TOKENS, TOKENS)
    inside = in_box()
    bias = mask[0, 0]
    penalty = torch.full((int(inside.sum()), int((~inside).sum())), -lc.SELF_PENALTY)
    assert torch.equal(bias[inside][:, ~inside], penalty)
    assert torch.equal(bias[inside][:, inside], torch.zeros_like(bias[inside][:, inside]))
    assert torch.equal(bias[~inside][:, inside], -lc.SELF_PENALTY * torch.ones_like(penalty.T))
    assert torch.equal(bias[~inside][:, ~inside], torch.zeros_like(bias[~inside][:, ~inside]))


def test_the_penalty_reads_its_strength_at_call_time(monkeypatch):
    monkeypatch.setattr(lc, "SELF_PENALTY", 0.2)
    one = layout()
    MASKS.clear()
    self_call((one, one), None, hidden())
    inside = in_box()
    bias = MASKS[0][0, 0]
    assert torch.allclose(bias[inside][:, ~inside], torch.tensor(-0.2))


@pytest.mark.parametrize(
    "settings, progress",
    [
        ({}, 0.5),
        ({"TEXT_END": 0.2}, 0.3),
        ({"SELF_END": 0.1}, 0.2),
        ({"SELF_PENALTY": 0.0}, 0.0),
    ],
)
def test_outside_the_window_or_without_penalty_is_bit_identical(monkeypatch, settings, progress):
    for name, value in settings.items():
        monkeypatch.setattr(lc, name, value)
    one = layout()
    h = hidden()
    MASKS.clear()
    out = self_call((one, one), None, h, progress=progress)
    assert MASKS == [None]
    assert torch.equal(out, attn(h, h, h, HEADS))


def test_self_end_replaces_the_text_end(monkeypatch):
    monkeypatch.setattr(lc, "SELF_END", 0.6)
    assert lc.self_end() == 0.6
    one = layout()
    MASKS.clear()
    self_call((one, one), None, hidden(), progress=0.5)
    assert MASKS[0] is not None
    monkeypatch.setattr(lc, "SELF_END", None)
    assert lc.self_end() == lc.TEXT_END


def test_positive_and_negative_rows_get_the_same_bias():
    mask = box_mask()
    pos, neg = layout(mask), layout(mask)
    MASKS.clear()
    self_call((pos, neg), torch.tensor([0, 0, 1, 1]), hidden(batch=4))
    (bias,) = MASKS
    assert bias.shape == (4, 1, TOKENS, TOKENS)
    for b in range(1, 4):
        assert torch.equal(bias[b], bias[0])
    assert bool((bias[0] != 0).any())


def test_negative_rows_use_their_own_layout():
    other = torch.zeros(1, *LATENT)
    other[..., 4:, 4:] = 1.0
    MASKS.clear()
    self_call((layout(), layout(other)), torch.tensor([0, 1]), hidden(batch=2))
    (bias,) = MASKS
    assert not torch.equal(bias[0], bias[1])


def test_the_bias_is_computed_once_per_model_call(monkeypatch):
    calls = []
    inner = ra.region_bias

    def counting(*a, **kw):
        calls.append(1)
        return inner(*a, **kw)

    monkeypatch.setattr(ra, "region_bias", counting)
    mask = box_mask()
    layouts = (layout(mask), layout(mask))
    state = CallState(progress=0.0, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    h = hidden(batch=2)
    MASKS.clear()
    for _ in range(3):
        self_call(layouts, torch.tensor([0, 1]), h, override=override, state=state)
    assert len(calls) == 1
    assert all(m is MASKS[0] for m in MASKS)
    self_call(layouts, torch.tensor([0, 1]), h)
    assert len(calls) == 2


def test_a_region_attention_bias_inside_adds_to_the_penalty():
    seen = []

    def previous(func, *args, **kwargs):
        seen.append(kwargs.get("mask"))
        return func(*args, **kwargs)

    one = layout()
    state = CallState(progress=0.0, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state, previous)
    self_call((one, one), None, hidden(), override=override, state=state)
    (mask,) = seen
    assert mask is not None
    assert bool((mask != 0).any())


def test_without_a_composed_layout_self_attention_is_bit_identical():
    h = hidden()
    state = CallState(latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "self"
    assert torch.equal(override(attn, h, h, h, HEADS), attn(h, h, h, HEADS))
    MASKS.clear()
    assert torch.equal(self_call(None, None, h, state=state), attn(h, h, h, HEADS))
    plain = CallState(latent=LATENT)
    one = layout()
    assert torch.equal(self_call((one, one), None, h, state=plain), attn(h, h, h, HEADS))
    assert all(m is None for m in MASKS)


def test_a_cross_attention_call_gets_no_self_penalty():
    one = layout()
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, TOKENS, DIM, generator=g)
    k = torch.randn(1, one.total, DIM, generator=g)
    state = CallState(progress=0.0, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "cross"
    MASKS.clear()
    with compose_layout((one, one), None):
        override(attn, q, k, k, HEADS)
    assert all(m is None for m in MASKS)


AXIS_LATENT = (4, 4)


def axis_layout():
    mask = torch.zeros(1, *AXIS_LATENT)
    mask[0, :2, :2] = 1.0
    return ComposedLayout(
        (
            Segment("global", "creative", 4, None, 1.0),
            Segment("layer", "ip", 3, mask, 1.0),
        )
    )


def test_cross_kv_scope_takes_the_layers_off_the_image_tokens(monkeypatch):
    one = axis_layout()
    rows = torch.tensor([0, 1])
    main = axis_maps((one, one), rows, AXIS_LATENT, 0.0)
    monkeypatch.setattr(lc, "MOUNT_SCOPE", "cross_kv")
    scoped = axis_maps((one, one), rows, AXIS_LATENT, 0.0)
    assert torch.equal(scoped["ip"].image, torch.zeros_like(scoped["ip"].image))
    assert torch.equal(scoped["creative"].image, main["creative"].image)
    for a in C.AXES:
        assert torch.equal(scoped[a].text, main[a].text)
    assert torch.equal(scoped["ip"].text[:, 4:], torch.ones(2, 3))


def test_all_scope_keeps_the_maps(monkeypatch):
    one = axis_layout()
    rows = torch.tensor([0, 1])
    assert lc.MOUNT_SCOPE == "all"
    main = axis_maps((one, one), rows, AXIS_LATENT, 0.0)
    inside = one.segments[1].mask[0].reshape(-1).bool()
    ip = main["ip"].image.reshape(2, -1)
    assert torch.allclose(ip[:, inside], torch.full_like(ip[:, inside], 1.0 - lc.BETA))
    assert torch.equal(ip[:, ~inside], torch.zeros_like(ip[:, ~inside]))
    monkeypatch.setattr(lc, "MOUNT_SCOPE", "all")
    again = axis_maps((one, one), rows, AXIS_LATENT, 0.0)
    for a in C.AXES:
        assert torch.equal(again[a].image, main[a].image)
        assert torch.equal(again[a].text, main[a].text)


def test_the_settings_note_names_the_penalty_its_end_and_the_scope(monkeypatch):
    monkeypatch.setattr(lc, "SELF_PENALTY", 0.2)
    monkeypatch.setattr(lc, "MOUNT_SCOPE", "cross_kv")
    note = lc.settings_note()
    assert "penalty 0.2" in note
    assert f"until progress {lc.TEXT_END:g}" in note
    assert "mount scope cross_kv" in note


def test_the_defaults():
    assert lc.SELF_PENALTY == 1.0
    assert lc.SELF_END is None
    assert lc.MOUNT_SCOPE == "all"
