# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

from kilora import bake
from kilora.comfyui import patching
from kilora.region_isolation import BaseBranch, RegionIsolation, isolate, outside_weight

H = W = 16


def _left(px=H * 8):
    m = torch.zeros(1, px, px)
    m[..., : px // 2] = 1
    return m


def test_outside_weight_is_zero_inside_and_strength_outside():
    w = outside_weight([_left()], (1, 16, 1, H, W), feather=0, strength=0.7)
    assert w.shape == (1, 1, 1, H, W)
    assert w[..., : W // 2].abs().max() == 0
    assert torch.allclose(w[..., W // 2 :], torch.full((), 0.7))


def test_feather_softens_only_the_edge():
    w = outside_weight([_left()], (1, 16, H, W), feather=16, strength=1.0)[0, 0, 0]
    assert w[: W // 2 - 2].abs().max() == 0 and torch.equal(w[W // 2 + 2 :], torch.ones(W // 2 - 2))
    edge = w[W // 2 - 2 : W // 2 + 2]
    assert ((edge > 0) & (edge < 1)).all()


def test_union_of_masks_and_no_mask():
    right = 1 - _left()
    w = outside_weight([_left(), right], (1, 4, H, W), feather=0, strength=1.0)
    assert w.abs().max() == 0
    assert torch.equal(outside_weight([], (1, 4, H, W), 0, 1.0), torch.ones(1, 1, H, W))


def test_base_branch_follows_the_flow_ode_with_guidance():
    b = BaseBranch()
    x0 = torch.full((1, 1, 2, 2), 4.0)
    assert torch.equal(b.latent_at(x0, 1.0), x0)
    b.outputs, b.scale = [torch.full_like(x0, 2.0), torch.full_like(x0, 1.0)], 3.0
    x = b.latent_at(torch.zeros_like(x0), 0.5)
    assert torch.allclose(x, 4.0 + 0.5 * (4.0 - 4.0) * torch.ones_like(x0))
    b.outputs = [torch.zeros_like(x0), None]
    assert torch.allclose(b.latent_at(x0, 0.25), torch.full_like(x0, 2.0))
    fresh = torch.full_like(x0, 9.0)
    assert torch.equal(b.latent_at(fresh, 0.9), fresh)


def test_isolate_keeps_the_inside_and_takes_the_base_outside():
    w = outside_weight([_left()], (1, 1, H, W), 0, 1.0)
    x = torch.randn(1, 1, H, W, generator=torch.Generator().manual_seed(0))
    seen = []

    def calc(latent, base):
        seen.append((latent.clone(), base))
        return [latent * 2 + (0.0 if base else 1.0), None]

    out = isolate(calc, x, 1.0, w, BaseBranch())
    assert [b for _, b in seen] == [True, False]
    assert torch.equal(out[0][..., : W // 2], x[..., : W // 2] * 2 + 1)
    assert torch.allclose(out[0][..., W // 2 :], x[..., W // 2 :] * 2)
    assert out[1] is None


def _patcher():
    from test_comfyui import FakePatcher

    chain = patching.ModChain([patching.ModEntry("mod", "a.kilora", region=_left())])
    return FakePatcher(None, model_options={patching.APPLIED_KEY: {"chain": chain}})


def _toy_calc(log):
    region = F.adaptive_avg_pool2d(_left()[None], (H, W))[0]

    def calc(model, conds, x, sigma, options):
        base = bool(bake._BASE_ONLY)
        log.append(base)
        d = F.avg_pool2d(x.flatten(0, 1)[:, None], 9, 1, 4, count_include_pad=False)
        d = d.view_as(x) * 0.95
        if not base:
            d = d + 0.5 * region
        return [d, d * 0.5]

    return calc


def _sample(patched, steps=10, calc=None):
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    options = dict(patched.model_options)
    options["transformer_options"] = {"sample_sigmas": sigmas}
    x = torch.randn(1, 1, H, W, generator=torch.Generator().manual_seed(1))
    hook = options.get("sampler_calc_cond_batch_function")
    post = options.get("sampler_post_cfg_function", [])
    for i in range(steps):
        s, s_next = sigmas[i], sigmas[i + 1]
        args = {
            "conds": [1, 0],
            "input": x,
            "sigma": s.view(1),
            "model": None,
            "model_options": options,
        }
        cond, uncond = hook(args) if hook else calc(None, None, x, s.view(1), options)
        d = uncond + 2.0 * (cond - uncond)
        for fn in post:
            d = fn({"denoised": d, "cond_scale": 2.0})
        x = d + (s_next / s) * (x - d)
    return x


def _leak(changed, base):
    change = (changed - base).abs()[0, 0]
    return float(change[:, W // 2 + 2 :].mean() / change[:, : W // 2 - 2].mean())


def test_isolation_cuts_a_leak_that_builds_over_the_steps():
    log = []
    calc = _toy_calc(log)

    def plain_base(model, conds, x, sigma, options):
        with bake.base_only():
            return calc(model, conds, x, sigma, options)

    p = _patcher()
    base = _sample(p, calc=plain_base)
    off = _sample(p, calc=calc)
    full = _sample(patching.apply_region_isolation(p, RegionIsolation(1.0, 1.01, 0), calc)[0])
    window = _sample(patching.apply_region_isolation(p, RegionIsolation(1.0, 0.6, 0), calc)[0])
    assert _leak(off, base) > 0.05
    assert _leak(full, base) < 0.05 * _leak(off, base)
    assert _leak(window, base) < _leak(off, base)


def test_after_until_one_plain_call_and_bit_identical():
    log = []
    calc = _toy_calc(log)
    patched, n = patching.apply_region_isolation(_patcher(), RegionIsolation(1.0, 0.6, 0), calc)
    assert n == 1
    _sample(patched, steps=10)
    assert log == [True, False] * 6 + [False] * 4
    sigmas = torch.linspace(1.0, 0.0, 11)
    options = {**patched.model_options, "transformer_options": {"sample_sigmas": sigmas}}
    x = torch.randn(1, 1, H, W)
    args = {
        "conds": [1, 0],
        "input": x,
        "sigma": sigmas[8].view(1),
        "model": None,
        "model_options": options,
    }
    hooked = options["sampler_calc_cond_batch_function"](args)
    plain = calc(None, None, x, sigmas[8], options)
    assert all(torch.equal(a, b) for a, b in zip(hooked, plain, strict=True))


@pytest.mark.parametrize("strength", [0.0])
def test_strength_zero_or_no_region_changes_nothing(strength):
    from test_comfyui import FakePatcher

    p, n = patching.apply_region_isolation(_patcher(), RegionIsolation(strength))
    assert n == 0 and "sampler_calc_cond_batch_function" not in p.model_options
    q, n = patching.apply_region_isolation(FakePatcher(None), RegionIsolation(1.0))
    assert n == 0 and "sampler_calc_cond_batch_function" not in q.model_options


def test_base_only_returns_the_base_output_of_baked_modules():
    lin = torch.nn.Linear(4, 4)

    class Plan:
        def run(self, x, h):
            return h + 1

    baked = bake.baked_copy(lin, Plan())
    x = torch.randn(2, 4)
    assert torch.allclose(baked(x), lin(x) + 1)
    with bake.base_only():
        assert torch.equal(baked(x), lin(x))


def test_baked_plan_without_parts_ignores_conditionings():
    from kilora.comfyui.expert_text import conditioning_values

    lin = torch.nn.Linear(4, 4)

    class Plan:
        def run(self, x, h):
            return h + x

    plan = Plan()
    baked = bake.baked_copy(lin, plan)
    x = torch.randn(2, 4)
    with conditioning_values({"base": x + 5, "expert_first": x + 10}):
        actual = baked(x)
    assert torch.equal(actual, plan.run(x, lin(x)))


def test_isolation_takes_its_regions_from_a_mod_list():
    from test_comfyui import FakePatcher

    from kilora.comfyui import nodes as N

    chain = patching.ModChain([patching.ModEntry("mod", "a.kilora", region=_left())])
    plain = FakePatcher(None)
    out, n = patching.apply_region_isolation(plain, RegionIsolation(1.0), chain=chain)
    assert n == 1 and "sampler_calc_cond_batch_function" in out.model_options
    model, report = N.KiLoRARegionIsolation().isolate(plain, 1.0, sets=chain)
    assert "sampler_calc_cond_batch_function" in model.model_options
    empty = patching.ModChain([patching.ModEntry("mod", "a.kilora")])
    _, n = patching.apply_region_isolation(plain, RegionIsolation(1.0), chain=empty)
    assert n == 0
