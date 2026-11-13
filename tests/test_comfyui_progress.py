# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_depth_map import chain

from kilora.comfyui import nodes as N
from kilora.compute import current_progress


def _karras(n, sigma_min=0.03, sigma_max=14.6, rho=7.0):
    ramp = torch.linspace(0, 1, n)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sigmas = (hi + ramp * (lo - hi)) ** rho
    return torch.cat([sigmas, torch.zeros(1)])


def _sample(patched, sigmas, outer=None):
    wrapper = patched.model_options["model_function_wrapper"]
    if outer is not None:
        inner = wrapper

        def wrapper(apply_model, args):
            return outer(inner, apply_model, args)

    seen = []

    def apply_model(x, t, **c):
        seen.append(current_progress())
        return x

    x = torch.randn(2, 16, 4)
    for sigma in sigmas[:-1]:
        timestep = torch.full((2,), float(sigma))
        c = {
            "transformer_options": {
                "sample_sigmas": sigmas,
                "sigmas": timestep[:1],
                "cond_or_uncond": [0, 1],
            }
        }
        wrapper(apply_model, {"input": x, "timestep": timestep, "c": c, "cond_or_uncond": [0, 1]})
    return seen


def test_progress_follows_the_sigma_schedule(denoiser, chain):
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 3)
    steps = 20
    seen = _sample(patched, _karras(steps))
    keys = len(seen) - steps
    assert keys >= 0 and set(seen[:keys]) <= {0.0}
    seen = seen[keys:]
    assert seen == [i / (steps - 1) for i in range(steps)]
    outside = []

    def outer(inner, apply_model, args):
        outside.append(current_progress())
        return inner(apply_model, args)

    assert _sample(patched, _karras(steps), outer) == seen
    assert set(outside) == {0.0}
    assert math.isclose(seen[8], 8 / 19)
