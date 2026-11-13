# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.comfyui import patching
from kilora.region_attention import (
    CallState,
    CaptionLayout,
    RegionAttention,
    caption_weights,
    comfy_override,
    cross_weight,
    diffusers_region_attention,
    memberships,
)

LATENT = (8, 8)
GRID = 16
DIM, HEADS = 8, 2


def _left(batch=1):
    m = torch.zeros(batch, 32, 32)
    m[..., :16] = 1
    return m


def _right(batch=1):
    return 1 - _left(batch)


def _attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options=None, **kw):
    if not skip_reshape:
        b, s, _ = q.shape
        q, k, v = (t.reshape(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask
    probs = logits.softmax(-1)
    PROBS.append(probs)
    return (probs @ v).transpose(1, 2).flatten(2)


PROBS: list = []


def _optimized(q, k, v, heads, **kwargs):
    topts = kwargs.get("transformer_options") or {}
    if "optimized_attention_override" in topts:
        kwargs["_inside_attn_wrapper"] = True
        return topts["optimized_attention_override"](_attention, q, k, v, heads, **kwargs)
    return _attention(q, k, v, heads, **kwargs)


class Toy(torch.nn.Module):
    def __init__(self, layers=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = [torch.randn(4, DIM, DIM, generator=g) / DIM**0.5 for _ in range(layers)]
        self.text = torch.randn(1, 7, DIM, generator=g)

    def forward(self, x, topts):
        for wq, wk, wv, wc in self.w:
            heads = lambda t: t.reshape(t.shape[0], t.shape[1], HEADS, -1).transpose(1, 2)
            q, k, v = heads(x @ wq), heads(x @ wk), heads(x @ wv)
            x = x + _optimized(q, k, v, HEADS, skip_reshape=True, transformer_options=topts)
            t = self.text.expand(x.shape[0], -1, -1)
            x = x + _optimized(x @ wc, t, t, HEADS, transformer_options=topts)
        return x


def _x(batch=1, seed=1):
    return torch.randn(batch, GRID, DIM, generator=torch.Generator().manual_seed(seed))


def _topts(mode, progress=0.4, masks=None, strength=2.0, outside=False):
    masks = masks if masks is not None else [_left(), _right()]
    cfg = RegionAttention(mode, strength=strength, until=0.6, outside_region=outside, start=0.3)
    return {
        "optimized_attention_override": comfy_override(
            masks, cfg, CallState(progress=progress, latent=LATENT)
        )
    }


def _left_tokens():
    return (torch.arange(GRID) % 4) < 2


def test_off_is_bit_identical_to_no_override():
    toy, x = Toy(), _x()
    assert torch.equal(toy(x, {}), toy(x, _topts("off")))


@pytest.mark.parametrize("mode", ["soft", "hard"])
@pytest.mark.parametrize("progress", [0.0, 0.29, 0.6, 0.9])
def test_outside_the_window_equals_off(mode, progress):
    toy, x = Toy(), _x()
    assert torch.equal(toy(x, {}), toy(x, _topts(mode, progress=progress)))


@pytest.mark.parametrize("progress", [0.3, 0.45, 0.59])
def test_inside_the_window_changes_the_output(progress):
    toy, x = Toy(), _x()
    assert not torch.equal(toy(x, {}), toy(x, _topts("hard", progress=progress)))


def test_window_defaults_and_bounds():
    cfg = RegionAttention("soft")
    assert (cfg.start, cfg.until, cfg.strength) == (0.3, 0.6, 0.5)
    assert [cfg.active(p) for p in (0.0, 0.299, 0.3, 0.5, 0.6, 1.0)] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert not RegionAttention("off").active(0.4)


def test_hard_blocks_cross_region_attention_early():
    PROBS.clear()
    Toy(layers=1)(_x(), _topts("hard", masks=[_left()]))
    probs = PROBS[0][0]
    left = _left_tokens()
    assert probs[:, left][:, :, ~left].abs().max() == 0
    assert probs[:, ~left][:, :, left].sum() > 0
    assert torch.isfinite(PROBS[1]).all()


def test_hard_region_output_ignores_tokens_outside():
    toy, x = Toy(layers=1), _x()
    y = x.clone()
    y[:, ~_left_tokens()] += 5.0
    topts = _topts("hard", masks=[_left()])
    left = _left_tokens()
    a = toy.forward(x, topts)[:, left] - x[:, left]
    b = toy.forward(y, topts)[:, left] - y[:, left]
    assert torch.allclose(a, b, atol=1e-5)


def test_soft_reduces_cross_region_attention():
    def mass(topts):
        PROBS.clear()
        Toy(layers=1)(_x(), topts)
        left = _left_tokens()
        return float(PROBS[0][0][:, left][:, :, ~left].sum())

    off, soft, hard = mass({}), mass(_topts("soft")), mass(_topts("hard"))
    assert hard < soft < off


def _leak(topts, seed=1):
    toy, x = Toy(seed=seed), _x(seed=seed)
    left = _left_tokens()
    delta = torch.zeros_like(x)
    delta[:, left] = torch.randn(
        1, int(left.sum()), DIM, generator=torch.Generator().manual_seed(9)
    )
    change = (toy(x + delta, topts) - toy(x, topts)).abs().mean(-1)[0]
    return float(change[~left].mean() / change[left].mean())


def test_leak_ratio_drops_with_soft_and_hard_for_two_regions():
    off = _leak({})
    soft = _leak(_topts("soft", strength=4.0))
    hard = _leak(_topts("hard"))
    assert hard < soft < off
    assert hard < 0.05 * off


def test_outside_region_keeps_a_single_region_change_in():
    masks = [_left()]
    free = _leak(_topts("hard", masks=masks))
    closed = _leak(_topts("hard", masks=masks, outside=True))
    soft = _leak(_topts("soft", masks=masks, strength=4.0, outside=True))
    assert closed < soft < free
    assert closed < 0.05 * _leak({})


def test_cross_weight_soft_and_overlapping_memberships():
    a = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.5]])
    b = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]])
    w = cross_weight(torch.stack([a, b]))[0]
    assert torch.equal(w.diagonal(), torch.zeros(5))
    assert w[3].abs().max() == 0
    assert w[0, 1] == 0 and w[1, 0] == 0 and w[1, 2] == 0
    assert w[0, 2] == 1 and w[0, 3] == 1 and w[2, 0] == 1
    assert w[4, 3] == 0.5 and w[0, 4] == 0.5


def test_memberships_repeat_for_conditional_and_unconditional_halves():
    m = memberships([_left(2)], GRID, LATENT, 4)
    assert m.shape == (1, 4, GRID)
    assert memberships([_left()], 15, LATENT, 1) is None
    with pytest.raises(ValueError):
        memberships([_left(3)], GRID, LATENT, 4)


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        RegionAttention("full")


def test_comfy_patcher_installs_the_override_for_regions():
    from test_comfyui import FakePatcher

    chain = patching.ModChain(
        [
            patching.ModEntry("mod", "a.kilora", region=_left()),
            patching.ModEntry("mod", "b.kilora", region=_right()),
        ]
    )
    p = FakePatcher(None, model_options={patching.APPLIED_KEY: {"chain": chain}})
    seen = {}

    def apply_model(x, t, transformer_options=None, **kw):
        seen["topts"] = transformer_options
        return Toy()(_x(), transformer_options)

    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    def args(sigma):
        topts = {"sample_sigmas": sigmas}
        c = {"transformer_options": topts}
        return {"input": torch.zeros(1, 16, *LATENT), "timestep": torch.tensor([sigma]), "c": c}

    off, n = patching.apply_region_attention(p, RegionAttention("off"))
    assert n == 0 and "model_function_wrapper" not in off.model_options
    on, n = patching.apply_region_attention(p, RegionAttention("hard"))
    assert n == 2
    wrapper = on.model_options["model_function_wrapper"]
    y = wrapper(apply_model, args(0.6))
    assert "optimized_attention_override" in seen["topts"]
    assert not torch.equal(y, Toy()(_x(), {}))
    for sigma in (1.0, 0.8, 0.4):
        assert torch.equal(wrapper(apply_model, args(sigma)), Toy()(_x(), {}))
    empty, n = patching.apply_region_attention(FakePatcher(None), RegionAttention("soft"))
    assert n == 0 and "model_function_wrapper" not in empty.model_options


def test_diffusers_processor_wrapper_blocks_cross_region_attention():
    pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0

    class Patched(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.proj = torch.nn.Linear(4 * 4, DIM)
            self.attn1 = Attention(
                DIM,
                heads=HEADS,
                dim_head=DIM // HEADS,
                qk_norm="rms_norm",
                processor=CosmosAttnProcessor2_0(),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            t = (
                x.reshape(b, c, h // 2, 2, w // 2, 2)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(b, -1, c * 4)
            )
            return self.attn1(self.proj(t))

    model, x = Patched(), torch.randn(1, 4, *LATENT)
    y = x.clone()
    y[..., 4:] += 3.0
    left = _left_tokens()
    cfg = RegionAttention("hard")
    with diffusers_region_attention(model, [_left()], cfg, progress=lambda: 0.4):
        a, b = model(x)[:, left], model(y)[:, left]
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(model(x)[:, left], model(y)[:, left], atol=1e-3)
    assert isinstance(model.attn1.processor, CosmosAttnProcessor2_0)
    base = model(x)
    for outside in (0.0, 0.9):
        with diffusers_region_attention(model, [_left()], cfg, progress=lambda p=outside: p):
            assert torch.equal(model(x), base)


class CaptionToy(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(3, DIM, DIM, generator=g) / DIM**0.5

    def forward(self, x, t, c_crossattn=None, transformer_options=None, **kw):
        topts = transformer_options or {}
        patches = topts.get("patches", {})
        h = x.flatten(2).transpose(1, 2)[:, :GRID, :DIM]
        for p in patches.get("attn1_patch", []):
            p(h, h, h, extra_options=topts)
        h = h + _optimized(h @ self.w[0], h, h, HEADS, transformer_options=topts)
        for p in patches.get("attn2_patch", []):
            p(h, c_crossattn, c_crossattn, extra_options=topts)
        ctx = c_crossattn
        PROBS.clear()
        return h + _optimized(h @ self.w[1], ctx @ self.w[2], ctx, HEADS, transformer_options=topts)


def _caption_patcher(captions=True, mode="off"):
    from test_comfyui import FakePatcher

    g = torch.Generator().manual_seed(3)
    cap = [[torch.randn(1, 5, DIM, generator=g), {}]]
    chain = patching.ModChain(
        [
            patching.ModEntry("mod", "a.kilora", region=_left(), region_caption=cap),
            patching.ModEntry("mod", "b.kilora", region=_right()),
        ]
    )
    p = FakePatcher(None, model_options={patching.APPLIED_KEY: {"chain": chain}})
    cfg = RegionAttention(mode, captions=captions)
    return patching.apply_region_attention(p, cfg)[0], cap


def _caption_args(sigma, cond_or_uncond=(0,)):
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    b = len(cond_or_uncond)
    topts = {"sample_sigmas": sigmas, "cond_or_uncond": list(cond_or_uncond)}
    context = torch.randn(b, 7, DIM, generator=torch.Generator().manual_seed(4))
    x = torch.randn(b, GRID * DIM // 64 * 4, *LATENT, generator=torch.Generator().manual_seed(5))
    c = {"c_crossattn": context, "transformer_options": topts}
    return {"input": x, "timestep": torch.tensor([sigma]), "c": c}


def _run(patched, args):
    toy = CaptionToy()
    seen = {}

    def apply_model(x, t, **c):
        seen.update(c)
        return toy(x, t, **c)

    if "model_function_wrapper" in patched.model_options:
        return patched.model_options["model_function_wrapper"](apply_model, args), seen
    return apply_model(args["input"], args["timestep"], **args["c"]), seen


def _masked_reference(args, seen, cond_or_uncond):
    context = seen["c_crossattn"]
    b, keys = context.shape[0], context.shape[1]
    left = _left_tokens()
    allowed = torch.zeros(b, GRID, keys, dtype=torch.bool)
    allowed[:, :, :7] = True
    for row, kind in enumerate(cond_or_uncond):
        if kind == 0:
            allowed[row, left, :7] = False
            allowed[row, left, 7:] = True
    mask = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))[:, None]

    def reference(func, q, k, v, heads, **kwargs):
        if k.shape[1] != q.shape[1]:
            kwargs["mask"] = mask
        return func(q, k, v, heads, **kwargs)

    c = {**seen, "transformer_options": {"optimized_attention_override": reference}}
    return CaptionToy()(args["input"], args["timestep"], **c)


@pytest.mark.parametrize("cond_or_uncond", [(0,), (0, 1), (1, 0)])
def test_captions_on_the_split_path_equal_the_masked_softmax(cond_or_uncond):
    patched, _ = _caption_patcher()
    args = _caption_args(0.6, cond_or_uncond=cond_or_uncond)
    out, seen = _run(patched, args)
    assert seen["c_crossattn"].shape[1] == 7 + 5
    expected = _masked_reference(args, seen, cond_or_uncond)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_region_queries_attend_to_their_caption_only():
    patched, _ = _caption_patcher()
    _run(patched, _caption_args(0.6))
    prompt, caption = PROBS
    assert prompt.shape[-1] == 7 and caption.shape[-1] == 5


def test_unconditional_rows_never_use_captions():
    patched, _ = _caption_patcher()
    args = _caption_args(0.6, cond_or_uncond=(1,))
    _run(patched, args)
    (prompt,) = PROBS
    assert prompt.shape[-1] == 7


@pytest.mark.parametrize("sigma", [1.0, 0.8, 0.4])
def test_captions_outside_the_window_are_bit_identical(sigma):
    patched, _ = _caption_patcher()
    plain, _ = _caption_patcher(captions=False)
    a, seen = _run(patched, _caption_args(sigma))
    b, _ = _run(plain, _caption_args(sigma))
    assert seen["c_crossattn"].shape[1] == 7
    assert torch.equal(a, b)
    assert "model_function_wrapper" not in plain.model_options


def test_caption_weights_layout_and_feathered_edges():
    m = torch.zeros(1, 32, 32)
    m[..., :16] = 1
    m[..., 16:24] = 0.5
    layout = CaptionLayout(7, (3,), (m,), torch.tensor([True, False]))
    assert layout.segments == (7, 3)
    w = caption_weights(layout, GRID, LATENT, torch.device("cpu"))
    assert w.shape == (2, 2, GRID)
    inside = [c <= 2 for c in (torch.arange(GRID) % 4).tolist()]
    assert (w[1, 0] == 1).tolist() == inside
    assert (w[1, 1] == 0).all()
    torch.testing.assert_close(w.sum(0), torch.ones(2, GRID))
    assert caption_weights(layout, GRID, None, torch.device("cpu")) is None


def test_region_caption_goes_through_the_text_adapter():
    class Denoiser(torch.nn.Module):
        def preprocess_text_embeds(self, embeds, ids, t5xxl_weights=None):
            self.got = (embeds.shape, ids.shape, t5xxl_weights.shape)
            return torch.zeros(1, 512, 4)

    class Model:
        diffusion_model = Denoiser()

    cap = [
        [
            torch.ones(1, 6, 4),
            {"t5xxl_ids": torch.ones(9, dtype=torch.long), "t5xxl_weights": torch.ones(9)},
        ]
    ]
    out = patching.encode_region_caption(Model(), cap, "cpu", torch.float32)
    assert out.shape == (1, 512, 4)
    assert Model.diffusion_model.got == ((1, 6, 4), (1, 9), (1, 9, 1))
    plain = [[torch.ones(1, 6, 4), {}]]
    assert patching.encode_region_caption(Model(), plain, "cpu", torch.float32).shape == (1, 6, 4)


def test_region_node_keeps_the_caption():
    from kilora.comfyui import nodes as N

    chain = patching.ModChain([patching.ModEntry("mod", "a.kilora")])
    cap = [[torch.zeros(1, 2, 4), {}]]
    (out,) = N.KiLoRARegion().region(chain, _left()[0].reshape(1, 32, 32), caption=cap)
    assert out[0].region_caption is cap
