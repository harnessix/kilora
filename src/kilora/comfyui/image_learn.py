# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib

import torch
from torch.nn import functional as F

from .. import families
from ..library import contrast
from . import patching
from . import prompt_solve as P

SEEDS = (811, 1811)
UNRELATED_PROMPTS = (
    "a city street at night",
    "a bowl of fruit on a wooden table",
    "a mountain lake at dawn",
)


def settings_for(patcher, t_min: float = 0.0, t_max: float = 1.0) -> contrast.Settings:
    from ..image_terms import sampled_steps

    spec = families.get_family(patching.detect_family(patcher).family)
    sampling, _ = P._solve_sampling(spec)
    steps = sampling["steps"]
    settings = contrast.Settings(
        seeds=SEEDS,
        size=sampling["size"],
        steps=steps,
        cfg=sampling["cfg"],
        sample_steps=sampled_steps(steps, t_min, t_max),
    )
    settings.validate()
    return settings


def _sigmas(patcher, settings):
    import comfy.samplers

    spec = families.get_family(patching.detect_family(patcher).family)
    sampling, _ = P._solve_sampling(spec)
    return comfy.samplers.calculate_sigmas(
        patcher.model.model_sampling, sampling["scheduler"], settings.steps
    ).to(patcher.load_device)


def _denoised(patcher, conditioning, x, sigma, options, trial):
    import comfy.samplers
    from comfy.sampler_helpers import convert_cond

    from ..recording import perturbing

    cond = comfy.samplers.encode_model_conds(
        patcher.model.extra_conds, convert_cond(conditioning), x, x.device, "positive"
    )
    with contextlib.ExitStack() as stack:
        if trial is not None:
            stack.enter_context(perturbing(trial))
        stack.enter_context(contrast._without_malloc_graph())
        return comfy.samplers.calc_cond_batch(patcher.model, [cond], x, sigma, options)[0]


def _options(patcher):
    options = dict(patcher.model_options)
    options.pop("model_function_wrapper", None)
    return options


@torch.inference_mode()
def held_out_losses(patcher, clip, items, cue, settings, trial):
    import comfy.sample

    from .image_solve import _prompt

    sigmas = _sigmas(patcher, settings)
    options = _options(patcher)
    losses = []
    patcher.pre_run()
    try:
        for item in items:
            conditioning = P._encode(clip, _prompt(item.caption, cue))
            z = item.latent.to(patcher.load_device)
            for seed in settings.seeds:
                noise = comfy.sample.prepare_noise(torch.zeros_like(item.latent), seed)
                noise = noise.to(z.device)
                for step in settings.sample_steps:
                    sigma = sigmas[step : step + 1]
                    x = patcher.model.model_sampling.noise_scaling(sigma, noise, z)
                    denoised = _denoised(patcher, conditioning, x, sigma, options, trial)
                    losses.append(float(F.mse_loss(denoised.float(), z.float())))
    finally:
        patcher.cleanup()
    return losses


@torch.inference_mode()
def unrelated_drift(patcher, clip, settings, trial):
    import comfy.sample

    sigmas = _sigmas(patcher, settings)
    options = _options(patcher)
    latent = P.empty_latent(patcher, settings.size)
    noise = comfy.sample.prepare_noise(latent, settings.seeds[0]).to(patcher.load_device)
    zero = patcher.model.process_latent_in(latent).to(patcher.load_device)
    step = settings.sample_steps[len(settings.sample_steps) // 2]
    sigma = sigmas[step : step + 1]
    x = patcher.model.model_sampling.noise_scaling(sigma, noise, zero)
    change, base = 0.0, 0.0
    patcher.pre_run()
    try:
        for prompt in UNRELATED_PROMPTS:
            conditioning = P._encode(clip, prompt)
            without = _denoised(patcher, conditioning, x, sigma, options, None).float()
            with_change = _denoised(patcher, conditioning, x, sigma, options, trial).float()
            change += float((with_change - without).square().sum())
            base += float(without.square().sum())
    finally:
        patcher.cleanup()
    return (change / max(base, 1e-30)) ** 0.5
