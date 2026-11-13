# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest
import torch

from kilora.call_values import compose_layout
from kilora.comfyui import layer_compose as lc
from kilora.comfyui import patching
from kilora.comfyui.layer_compose import ComposedLayout, Segment, compose
from kilora.region_attention import (
    CallState,
    RegionAttention,
    comfy_override,
    composed_weights,
    softmax_bias,
    split_attention,
)

LATENT = (8, 8)
TOKENS = 16
DIM, HEADS = 8, 2
PROBS: list = []


def attn(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    b = q.shape[0]
    q, k, v = (t.reshape(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask
    probs = logits.softmax(-1)
    PROBS.append(probs)
    return (probs @ v).transpose(1, 2).flatten(2)


def left_mask():
    m = torch.zeros(1, *LATENT)
    m[..., :4] = 1.0
    return m


def top_right_mask():
    m = torch.zeros(1, *LATENT)
    m[:, :4, 4:] = 1.0
    return m


def token_set(mask):
    return mask[0].reshape(4, 2, 4, 2).amax((1, 3)).reshape(-1) > 0


LENGTHS = (4, 3, 5)


def two_layer_layout(a=None, b=None):
    return ComposedLayout(
        (
            Segment("global", None, LENGTHS[0], None, 1.0),
            Segment("layer", "ip", LENGTHS[1], left_mask() if a is None else a, 1.0),
            Segment("layer", "creative", LENGTHS[2], top_right_mask() if b is None else b, 1.0),
        )
    )


def qkv(batch=1, keys=None, seed=0):
    keys = sum(LENGTHS) if keys is None else keys
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, TOKENS, DIM, generator=g)
    k = torch.randn(batch, keys, DIM, generator=g)
    v = torch.randn(batch, keys, DIM, generator=g)
    return q, k, v


def per_segment(q, k, v, lengths=LENGTHS):
    out, start = [], 0
    for n in lengths:
        out.append(attn(q, k[:, start : start + n], v[:, start : start + n], HEADS))
        start += n
    return out


def run(layouts, rows, q, k, v, progress=0.0, state=None):
    state = state or CallState(progress=progress, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "cross"
    with compose_layout(layouts, rows):
        return override(attn, q, k, v, HEADS)


def test_hard_masks_with_beta_zero_read_their_layer_text_only(monkeypatch):
    monkeypatch.setattr(lc, "BETA", 0.0)
    layout = two_layer_layout()
    q, k, v = qkv()
    out = run((layout, layout), torch.zeros(1, dtype=torch.long), q, k, v)
    a_g, a_a, a_b = per_segment(q, k, v)
    in_a, in_b = token_set(left_mask()), token_set(top_right_mask())
    outside = ~(in_a | in_b)
    assert outside.any()
    torch.testing.assert_close(out[:, in_a], a_a[:, in_a], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:, in_b], a_b[:, in_b], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:, outside], a_g[:, outside], atol=1e-6, rtol=0)


def test_beta_mixes_the_global_text_into_a_region(monkeypatch):
    monkeypatch.setattr(lc, "BETA", 0.1)
    layout = two_layer_layout()
    q, k, v = qkv()
    out = run((layout, layout), None, q, k, v)
    a_g, a_a, a_b = per_segment(q, k, v)
    in_a, in_b = token_set(left_mask()), token_set(top_right_mask())
    outside = ~(in_a | in_b)
    expect_a = 0.9 * a_a + 0.1 * a_g
    expect_b = 0.9 * a_b + 0.1 * a_g
    torch.testing.assert_close(out[:, in_a], expect_a[:, in_a], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:, in_b], expect_b[:, in_b], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:, outside], a_g[:, outside], atol=1e-6, rtol=0)


@pytest.mark.parametrize("end", [0.35, 0.2])
def test_after_the_text_end_every_token_reads_the_global_text(monkeypatch, end):
    monkeypatch.setattr(lc, "TEXT_END", end)
    layout = two_layer_layout()
    q, k, v = qkv()
    PROBS.clear()
    out = run((layout, layout), None, q, k, v, progress=end)
    (a_g, *_) = per_segment(q, k, v)
    torch.testing.assert_close(out, a_g, atol=1e-6, rtol=0)
    assert [p.shape[-1] for p in PROBS] == [LENGTHS[0], *LENGTHS]


def test_text_end_follows_the_sampling_clock_of_the_timeline(monkeypatch):
    monkeypatch.setattr(lc, "SINGLE_READY", True)
    layout = two_layer_layout()
    key = lc.register((layout, layout))
    q, k, v = qkv()
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    outs = {}
    for sigma in (0.8, 0.4):
        topts = {"sample_sigmas": sigmas, "cond_or_uncond": [0]}
        args = {
            "input": torch.zeros(1, 4, *LATENT),
            "timestep": torch.tensor([sigma]),
            "c": {"c_crossattn": k, "transformer_options": topts},
        }
        with patching._compose_scope(torch.tensor([key]), args):
            installed = args["c"]["transformer_options"]
            assert installed is not topts and "optimized_attention_override" not in topts
            for patch in installed["patches"]["attn2_patch"]:
                patch(q, k, v)
            outs[sigma] = installed["optimized_attention_override"](attn, q, k, v, HEADS)
    (a_g, *_) = per_segment(q, k, v)
    assert not torch.allclose(outs[0.8], a_g)
    torch.testing.assert_close(outs[0.4], a_g, atol=1e-6, rtol=0)


def test_overlapping_masks_give_weights_that_sum_to_one():
    a = torch.zeros(1, *LATENT)
    a[..., :6] = 1.0
    b = torch.zeros(1, *LATENT)
    b[..., 2:] = 0.7
    layout = two_layer_layout(a, b)
    w = composed_weights([layout], TOKENS, LATENT, 0.0, torch.device("cpu"))
    assert w.shape == (3, 1, TOKENS)
    torch.testing.assert_close(w.sum(0), torch.ones(1, TOKENS))
    assert (w >= 0).all()
    assert ((w[1] > 0) & (w[2] > 0)).any()


def test_positive_and_negative_rows_use_their_own_layouts(monkeypatch):
    monkeypatch.setattr(lc, "BETA", 0.0)
    pos = two_layer_layout()
    neg = two_layer_layout(top_right_mask(), left_mask())
    q, k, v = qkv(batch=2)
    out = run((pos, neg), torch.tensor([0, 1]), q, k, v)
    a_g, a_a, a_b = per_segment(q, k, v)
    in_a = token_set(left_mask())
    torch.testing.assert_close(out[0, in_a], a_a[0, in_a], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[1, in_a], a_b[1, in_a], atol=1e-6, rtol=0)


def test_equal_layouts_share_one_computation():
    layout = two_layer_layout()
    q, k, v = qkv(batch=2)
    PROBS.clear()
    run((layout, layout), torch.tensor([0, 1]), q, k, v)
    assert len(PROBS) == len(LENGTHS)
    assert all(p.shape[0] == 2 for p in PROBS)


@pytest.mark.parametrize("settings, progress", [({}, 0.5), ({"SELF_PENALTY": 0.0}, 0.0)])
def test_self_attention_does_not_change_without_the_penalty(monkeypatch, settings, progress):
    for name, value in settings.items():
        monkeypatch.setattr(lc, name, value)
    layout = two_layer_layout()
    g = torch.Generator().manual_seed(1)
    h = torch.randn(1, TOKENS, DIM, generator=g)
    state = CallState(progress=progress, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "self"
    with compose_layout((layout, layout), None):
        assert torch.equal(override(attn, h, h, h, HEADS), attn(h, h, h, HEADS))


def test_no_composed_layout_is_bit_identical():
    q, k, v = qkv()
    state = CallState(latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "cross"
    assert torch.equal(override(attn, q, k, v, HEADS), attn(q, k, v, HEADS))
    with compose_layout(None, None):
        state.kind = "cross"
        assert torch.equal(override(attn, q, k, v, HEADS), attn(q, k, v, HEADS))


def test_an_unknown_compose_id_installs_nothing():
    topts = {"cond_or_uncond": [0]}
    args = {
        "input": torch.zeros(1, 4, *LATENT),
        "timestep": torch.tensor([1.0]),
        "c": {"transformer_options": topts},
    }
    with patching._compose_scope(torch.tensor([10**9]), args):
        assert args["c"]["transformer_options"] is topts
    assert topts == {"cond_or_uncond": [0]}


def test_split_attention_keeps_the_layout_of_skip_reshape():
    q, k, v = qkv()
    w = torch.rand(3, 1, TOKENS)

    def heads_first(q, k, v, heads, skip_reshape=False, skip_output_reshape=False, **kw):
        assert skip_reshape and skip_output_reshape
        probs = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).softmax(-1)
        return probs @ v

    split = [t.reshape(1, -1, HEADS, DIM // HEADS).transpose(1, 2) for t in (q, k, v)]
    kwargs = {"skip_reshape": True, "skip_output_reshape": True}
    out = split_attention(heads_first, (*split, HEADS), kwargs, LENGTHS, w)
    flat = split_attention(lambda *a, **kw: attn(*a), (q, k, v, HEADS), {}, LENGTHS, w)
    torch.testing.assert_close(out.transpose(1, 2).flatten(2), flat, atol=1e-6, rtol=0)


def test_one_softmax_gives_padding_tokens_no_weight(monkeypatch):
    monkeypatch.setattr(lc, "ATTENTION", "softmax")
    layout = two_layer_layout()
    q, k, v = qkv()
    real = torch.ones(1, sum(LENGTHS), dtype=torch.bool)
    padding = [2, 3, 6, 10, 11]
    real[0, padding] = False
    k[0, padding] = 0.0
    v[0, padding] = 0.0
    state = CallState(latent=LATENT, compose=True, real=real)
    PROBS.clear()
    out = run((layout, layout), None, q, k, v, state=state)
    assert out.shape == q.shape
    (probs,) = PROBS
    assert probs[..., padding].abs().max() == 0
    torch.testing.assert_close(probs.sum(-1), torch.ones(probs.shape[:-1]))
    outside = ~(token_set(left_mask()) | token_set(top_right_mask()))
    assert probs[0][:, outside][..., LENGTHS[0] :].abs().max() == 0


def test_one_softmax_bias_is_the_log_of_the_weights():
    w = torch.tensor([[[0.25, 1.0]], [[0.75, 0.0]]])
    bias = softmax_bias((1, 2), w, None, torch.float32)
    assert bias.shape == (1, 1, 2, 3)
    torch.testing.assert_close(bias[0, 0, 0], torch.tensor([0.25, 0.75, 0.75]).log())
    assert bias[0, 0, 1, 0] == 0 and torch.isinf(bias[0, 0, 1, 1:]).all()


class Cross:
    def __init__(self, cond):
        self.cond = cond


@pytest.fixture
def comfy_conds(monkeypatch):
    comfy = types.ModuleType("comfy")
    module = types.ModuleType("comfy.conds")
    module.CONDRegular = Cross
    module.CONDCrossAttn = Cross
    comfy.conds = module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", module)
    return module


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.proj = torch.randn(4, DIM, generator=g)

    def preprocess_text_embeds(self, embeds, ids, t5xxl_weights=None):
        out = embeds @ self.proj + ids[..., None].float()
        pad = torch.zeros(out.shape[0], 512 - out.shape[1], out.shape[2])
        return torch.cat([out, pad], 1)


class Model:
    def __init__(self):
        self.diffusion_model = Denoiser()


def test_layer_texts_take_the_adapter_path_of_the_main_prompt(comfy_conds):
    model = Model()
    g = torch.Generator().manual_seed(8)
    texts = [torch.randn(1, n, 4, generator=g) for n in (6, 9, 5)]
    ids = [torch.arange(1, t.shape[1] + 1) for t in texts]

    def adapted(t, i):
        return model.diffusion_model.preprocess_text_embeds(t, i[None])

    def cond(context, **extra):
        return {"model_conds": {"c_crossattn": Cross(context)}, **extra}

    box = torch.zeros(1, *LATENT)
    box[:, 2:6, 2:6] = 1.0
    positive = [
        cond(adapted(texts[0], ids[0])),
        cond(adapted(texts[1], ids[1]), kilora_layer="ip", kilora_compose="single", mask=box),
    ]
    negative = [cond(adapted(texts[2], ids[2]))]
    groups, (pos, neg) = compose([positive, negative], family="anima")
    assert pos.lengths == neg.lengths == (512, 512)
    context = groups[0][0]["model_conds"]["c_crossattn"].cond
    assert context.shape == (1, 1024, DIM)

    caption = [[texts[1], {"t5xxl_ids": ids[1]}]]
    encoded = patching.encode_region_caption(model, caption, "cpu", torch.float32)
    torch.testing.assert_close(context[:, 512:], encoded)
    torch.testing.assert_close(context[:, :512], adapted(texts[0], ids[0]))

    args = {
        "input": torch.zeros(1, 4, *LATENT),
        "timestep": torch.tensor([1.0]),
        "c": {"c_crossattn": context, "transformer_options": {"cond_or_uncond": [0]}},
    }
    patching._install_compose_attention(args)
    override = args["c"]["transformer_options"]["optimized_attention_override"]
    real = None
    for cell in override.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, CallState):
            real = value.real
    assert real is not None
    assert int(real[0, :512].sum()) == 6 and int(real[0, 512:].sum()) == 9
