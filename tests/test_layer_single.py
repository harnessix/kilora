# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import types

import pytest
import torch
from test_layer_negative import layered, masks, stand_in
from test_layer_negative import run as run_separate

from kilora.call_values import ComposedAlpha, compose_layout, current_axis_alphas
from kilora.comfyui import layer_compose as lc
from kilora.comfyui import patching
from kilora.comfyui.layer_batch import install_layer_batch
from kilora.comfyui.layer_compose import compose
from kilora.region_attention import CallState, RegionAttention, comfy_override

LATENT = (8, 8)
TOKENS = 16
DIM, HEADS = 8, 2
CFG = 4.0
CALLS: list = []


class Cross:
    def __init__(self, cond):
        self.cond = cond


class Regular:
    def __init__(self, cond):
        self.cond = cond


@pytest.fixture
def comfy_stub(monkeypatch):
    comfy = types.ModuleType("comfy")
    conds = types.ModuleType("comfy.conds")
    conds.CONDRegular = Regular
    conds.CONDCrossAttn = Cross
    hooks = types.ModuleType("comfy.hooks")

    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = lambda options, hook_scope: types.SimpleNamespace(
        options=options
    )
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy.conds, comfy.hooks = conds, hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", conds)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    return comfy


def attn(q, k, v, heads, mask=None, skip_reshape=False, **kw):
    CALLS.append(k.shape[1])
    b = q.shape[0]
    q, k, v = (t.reshape(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        logits = logits + mask
    return (logits.softmax(-1) @ v).transpose(1, 2).flatten(2)


def text(length, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, length, DIM, generator=g)


def cond(context, **extra):
    return {"model_conds": {"c_crossattn": Cross(context)}, **extra}


def box():
    mask = torch.zeros(1, *LATENT)
    mask[:, 2:6, 2:6] = 1.0
    return mask


def single_layers(negatives=None, mode="single"):
    positive = [
        cond(text(4, 1), kilora_layer="creative", kilora_compose=mode),
        cond(text(4, 2), kilora_layer="ip", kilora_compose=mode, mask=box()),
    ]
    if negatives is None:
        negatives = [cond(text(4, 3))]
    return [positive, negatives]


def query(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, TOKENS, DIM, generator=g).repeat(batch, 1, 1)


def cross(layouts, rows, q, context):
    state = CallState(progress=0.0, latent=LATENT, compose=True)
    override = comfy_override((), RegionAttention(), state)
    state.kind = "cross"
    with compose_layout(layouts, rows):
        return override(attn, q, context, context, HEADS)


def reference(layout, q, context, row):
    from kilora.region_attention import composed_weights

    w = composed_weights([layout], TOKENS, LATENT, 0.0, torch.device("cpu"))
    out, start = 0.0, 0
    for s, n in enumerate(layout.lengths):
        part = context[row : row + 1, start : start + n]
        out = out + attn(q[row : row + 1], part, part, HEADS) * w[s, 0, :, None]
        start += n
    return out


def test_negative_segments_hold_the_plain_negative_with_the_positive_weights():
    groups, (pos, neg) = compose(single_layers(), family="anima")
    negative = groups[1][0]["model_conds"]["c_crossattn"].cond
    plain = text(4, 3)
    assert torch.equal(negative[:, :4], plain) and torch.equal(negative[:, 4:], plain)
    assert pos is not neg
    for p, n in zip(pos.segments, neg.segments, strict=True):
        assert (p.role, p.axis, p.length, p.strength) == (n.role, n.axis, n.length, n.strength)
        assert p.mask is n.mask
    assert pos.segments[0].text != pos.segments[1].text
    assert neg.segments[0].text == neg.segments[1].text


def test_the_mask_of_a_negative_is_not_used():
    masked = cond(text(4, 3), mask=torch.zeros(1, *LATENT), mask_strength=1.0)
    groups, (pos, neg) = compose(single_layers([masked]), family="anima")
    (n,) = groups[1]
    assert "mask" not in n and "mask_strength" not in n and n["strength"] == 1.0
    assert all(a.mask is b.mask for a, b in zip(neg.segments, pos.segments, strict=True))


def test_equal_negative_segments_run_one_attention_call():
    _, (pos, neg) = compose(single_layers(), family="anima")
    groups, _ = compose(single_layers(), family="anima")
    context = groups[1][0]["model_conds"]["c_crossattn"].cond
    q = query(1)
    CALLS.clear()
    out = cross((pos, neg), torch.tensor([1]), q, context)
    assert CALLS == [4]
    CALLS.clear()
    expected = reference(neg, q, context, 0)
    assert len(CALLS) == 2
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_equal_positive_segments_run_one_attention_call():
    same = text(4, 1)
    positive = [
        cond(same, kilora_layer="creative", kilora_compose="single"),
        cond(same.clone(), kilora_layer="ip", kilora_compose="single", mask=box()),
    ]
    groups, (pos, neg) = compose([positive, [cond(text(4, 3))]], family="anima")
    assert pos.segments[0].text == pos.segments[1].text
    context = groups[0][0]["model_conds"]["c_crossattn"].cond
    q = query(1)
    CALLS.clear()
    out = cross((pos, neg), torch.tensor([0]), q, context)
    assert CALLS == [4]
    torch.testing.assert_close(out, reference(pos, q, context, 0), atol=1e-6, rtol=1e-5)


def test_a_mount_on_text_tokens_keeps_every_segment():
    _, (pos, neg) = compose(single_layers(), family="anima")
    groups, _ = compose(single_layers(), family="anima")
    context = groups[1][0]["model_conds"]["c_crossattn"].cond.clone()
    context[:, 4:] += 0.5
    q = query(1)
    CALLS.clear()
    out = cross((pos, neg), torch.tensor([1]), q, context)
    assert CALLS == [4, 4]
    torch.testing.assert_close(out, reference(neg, q, context, 0), atol=1e-6, rtol=1e-5)


def test_a_routed_negative_replaces_its_segment_and_the_plain_one_is_not_evaluated_there():
    routed = cond(text(4, 9), kilora_layer="ip", kilora_compose="single")
    conds = single_layers([cond(text(4, 3)), routed])
    groups, (pos, neg) = compose(conds, family="anima")
    assert len(groups[1]) == 1
    negative = groups[1][0]["model_conds"]["c_crossattn"].cond
    assert torch.equal(negative[:, :4], text(4, 3))
    assert torch.equal(negative[:, 4:], text(4, 9))
    assert neg.segments[0].text != neg.segments[1].text
    q = query(1)
    CALLS.clear()
    out = cross((pos, neg), torch.tensor([1]), q, negative)
    assert CALLS == [4, 4]
    torch.testing.assert_close(out, reference(neg, q, negative, 0), atol=1e-6, rtol=1e-5)


def sampler(patcher, conds, sigma, x=None):
    x = torch.zeros(1, DIM, *LATENT) if x is None else x
    return patcher.model_options["sampler_calc_cond_batch_function"](
        {
            "conds": conds,
            "input": x,
            "sigma": torch.tensor([sigma]),
            "model": None,
            "model_options": patcher.model_options,
        }
    )


@pytest.mark.parametrize(
    "change, options, cause",
    [
        (lambda c: c[0][1].update(kilora_compose="separate"), {}, "a layer uses compose separate"),
        (lambda c: c[0][1].update(control=object()), {}, "a layer has its own control"),
        (lambda c: c[0][1].update(hooks=object()), {}, "a layer has its own hooks"),
        (lambda c: c[0][1].update(timestep_start=0.5), {}, "a layer has its own timestep range"),
        (lambda c: c[0][1].update(area=(1, 1, 0, 0)), {}, "a layer has an area instead"),
        (
            lambda c: c[0][1].update(kilora_alpha_source="attention"),
            {},
            "a layer uses alpha_source attention",
        ),
        (lambda c: c[0].pop(0), {}, "no global prompt"),
        (None, {"kilora_layer_filter": True}, "a KiLoRA Layer Filter is connected"),
        (None, {"kilora_alpha_mask": True}, "KiLoRA Apply has an alpha mask"),
        (None, {"family": "sdxl"}, "the model family is not Anima or Cosmos"),
        (None, {"separate": True}, "the mods carry state across steps"),
    ],
)
def test_each_fallback_cause_runs_separate_with_one_info_line(
    change, options, cause, comfy_stub, caplog
):
    seen = []

    def previous(args):
        seen.append(args["conds"])
        return "result"

    options = dict(options)
    family = options.pop("family", "anima")
    separate = options.pop("separate", False)
    patcher = types.SimpleNamespace(
        model_options={"sampler_calc_cond_batch_function": previous, **options}
    )
    install_layer_batch(patcher, separate=separate, family=family)
    with caplog.at_level(logging.INFO, logger="kilora.comfyui.layer_batch"):
        for sigma in (2.0, 1.0):
            conds = single_layers()
            if change is not None:
                change(conds)
            sampler(patcher, conds, sigma)
    lines = [r.getMessage() for r in caplog.records if "falls back to separate" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].startswith(f"KiLoRA: compose single falls back to separate: {cause}")
    for groups in seen:
        assert any(c.get("kilora_layer") for c in groups[0])
        assert not any("kilora_compose_id" in c["model_conds"] for c in groups[0])


def _separate_conds(mode):
    conds = layered(masks())
    for c in conds[0]:
        if mode is not None:
            c["kilora_compose"] = mode
    return conds


def test_compose_separate_is_bit_identical_to_main(comfy_stub):
    results = {}
    for mode in (None, "separate"):
        calls = []
        patcher = types.SimpleNamespace(model_options={})
        install_layer_batch(patcher, calc=stand_in(calls), family="anima")
        x, out = run_separate(patcher, _separate_conds(mode))
        results[mode] = (
            out,
            [(c["kind"], c["layer"], c["text"], c["area"], id(c["mask"])) for c in calls],
        )
    (a, calls_a), (b, calls_b) = results[None], results["separate"]
    assert len(calls_a) == len(calls_b) == 4
    assert [c[:4] for c in calls_a] == [c[:4] for c in calls_b]
    for ya, yb in zip(a, b, strict=True):
        assert torch.equal(ya, yb)


def model_wrapper():
    def wrapper(apply_model, args):
        c = dict(args["c"])
        ids = c.pop("kilora_compose_id")
        args = {**args, "c": c}
        with patching._compose_scope(ids, args):
            scope = patching._compose_axis_scope(args, 0.0)
            assert scope is not None
            with scope:
                return apply_model(args["input"], args["timestep"], **args["c"])

    return wrapper


def denoiser(records):
    def apply_model(x, t, c_crossattn=None, transformer_options=None, **kw):
        alphas = current_axis_alphas()
        assert all(isinstance(alphas[a], ComposedAlpha) for a in alphas)
        records.append(
            {a: (alphas[a].image.clone(), alphas[a].text.clone()) for a in alphas},
        )
        batch = x.shape[0]
        q = query(batch) + x.mean((1, 2, 3))[:, None, None]
        for patch in transformer_options["patches"]["attn2_patch"]:
            patch(q, c_crossattn, c_crossattn)
        out = transformer_options["optimized_attention_override"](
            attn, q, c_crossattn, c_crossattn, HEADS
        )
        grid = out.mean(-1).reshape(batch, 1, 4, 4)
        return x + grid.repeat_interleave(2, -1).repeat_interleave(2, -2)

    return apply_model


def comfy_calc(sigmas, records):
    def calc(model, conds, x, sigma, options):
        wrapper = options.get("model_function_wrapper")
        items = [(kind, c) for kind, group in enumerate(conds) for c in group or ()]
        assert len({c["model_conds"]["c_crossattn"].cond.shape for _, c in items}) == 1
        kinds = [kind for kind, _ in items]
        c = {
            "c_crossattn": torch.cat([c["model_conds"]["c_crossattn"].cond for _, c in items]),
            "kilora_compose_id": torch.cat(
                [c["model_conds"]["kilora_compose_id"].cond for _, c in items]
            ),
            "transformer_options": {"cond_or_uncond": kinds, "sample_sigmas": sigmas},
        }
        args = {
            "input": x.repeat(len(items), 1, 1, 1),
            "timestep": sigma.repeat(len(items)),
            "c": c,
            "cond_or_uncond": kinds,
        }
        y = wrapper(denoiser(records), args)
        out = []
        for kind in range(len(conds)):
            rows = [i for i, (k, _) in enumerate(items) if k == kind]
            for i in rows:
                assert "mask" not in items[i][1] and items[i][1]["strength"] == 1.0
            out.append(y[rows].mean(0, keepdim=True))
        return out

    return calc


def single_run(caplog, conds=None):
    sigmas = torch.tensor([2.0, 1.0, 0.0])
    records = []
    patcher = types.SimpleNamespace(model_options={"model_function_wrapper": model_wrapper()})
    install_layer_batch(patcher, calc=comfy_calc(sigmas, records), family="anima")
    x = torch.linspace(0, 1, DIM * 64).reshape(1, DIM, *LATENT)
    outs = []
    with caplog.at_level(logging.INFO, logger="kilora.comfyui.layer_batch"):
        for sigma in (2.0, 1.0):
            outs.append(sampler(patcher, conds or single_layers(), sigma, x))
    return outs, records


def test_two_layers_with_single_give_two_rows_per_step(comfy_stub, caplog):
    outs, records = single_run(caplog)
    assert len(records) == 2
    lines = [r.getMessage() for r in caplog.records if "batch rows per step" in r.getMessage()]
    assert lines == [
        "KiLoRA: layered sample, 1 model calls and 2 batch rows per step for 2 conds "
        "(0 negatives paired with a layer, compose single)."
    ]
    rows = records[0]["ip"][0].shape[0]
    assert rows <= 2


def test_single_guided_result_reads_the_same_weights_in_both_branches(comfy_stub, caplog):
    (first, _), records = single_run(caplog)
    cond, uncond = first
    guided = uncond + CFG * (cond - uncond)
    assert torch.isfinite(guided).all()
    assert not torch.allclose(cond, uncond)
    for record in records:
        for _axis, (image, words) in record.items():
            assert image.shape[0] == words.shape[0] == 2
            assert torch.equal(image[0], image[1])
            assert torch.equal(words[0], words[1])
    image, words = records[0]["ip"]
    inside = box()[0] > 0
    assert torch.allclose(image[0][inside], torch.full((16,), 1 - lc.BETA))
    assert torch.equal(image[0][~inside], torch.zeros(48))
    assert torch.equal(words[0], torch.tensor([0.0] * 4 + [1.0] * 4))
