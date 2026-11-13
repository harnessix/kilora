# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import logging
import types

import pytest
import torch
from test_axis_alphas import _set

from kilora.call_values import axis_alphas
from kilora.comfyui import patching
from kilora.comfyui.layer_alpha import install_layer_alpha
from kilora.compute import run_layer


@pytest.mark.parametrize("inference", [False, True])
def test_model_call_switch_and_budget(monkeypatch, caplog, inference):
    monkeypatch.delenv("KILORA_AXIS_ZBUFFER", raising=False)
    patcher = types.SimpleNamespace(model_options={})
    patching._key_mask_by_step(patcher)
    wrapper = patcher.model_options["model_function_wrapper"]
    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(7))
    ip, creative = _set("ip"), _set("creative")
    functional = dataclasses.replace(_set("ip"), axis="functional")

    def run(sets, sigma, alphas=None):
        def model(x, timestep, **c):
            with axis_alphas(alphas or {}):
                return run_layer(y, y, sets, 1, target="layer")

        with torch.set_grad_enabled(not inference):
            return wrapper(model, {"input": y, "timestep": torch.tensor([sigma]), "c": {}})

    monkeypatch.delenv("KILORA_AXIS_IP_EARLY", raising=False)
    normal = run([ip, creative], 0.9)
    ip_only = run([ip], 0.9)
    functional_only = run([functional], 0.9)
    creative_only = run([creative], 0.9)
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "")
    assert torch.equal(run([ip, creative], 0.9), normal)
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0")
    assert torch.equal(run([ip, creative], 0.9), normal)
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0.5")
    with caplog.at_level(logging.INFO, logger="kilora"):
        torch.testing.assert_close(run([ip, creative], 0.9), ip_only)
        torch.testing.assert_close(run([ip, creative], 0.9, {"ip": 0.1, "creative": 0.9}), ip_only)
        assert torch.equal(run([ip, creative], 0.5), normal)
        assert torch.equal(run([ip, creative], 0.4), normal)
    assert caplog.text.count("axis ip early: sigma=") == 3
    assert torch.equal(run([creative], 0.9), creative_only)
    torch.testing.assert_close(
        run([ip, creative, functional], 0.9), y + (ip_only - y) + (functional_only - y)
    )
    torch.testing.assert_close(
        run([ip, creative, functional], 0.9, {"functional": 0.25}),
        y + (ip_only - y) + 0.25 * (functional_only - y),
    )


@pytest.mark.parametrize("value", ["", "bad", "-1", "1.1", "nan"])
def test_invalid_or_empty_threshold_is_off(monkeypatch, value):
    from kilora.call_values import ip_early_threshold

    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", value)
    assert ip_early_threshold() == 0


def test_mounted_ip_suppresses_creative_only_targets(monkeypatch):
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0.5")
    ip, creative = _set("ip"), _set("creative")
    plans = {"ip_target": types.SimpleNamespace(sets=[ip])}
    patcher = types.SimpleNamespace(model_options={})
    patching._key_mask_by_step(patcher, plans=plans)
    y = torch.randn(1, 4, 4)

    def model(x, timestep, **c):
        return run_layer(y, y, [creative], 1, target="creative_target")

    wrapper = patcher.model_options["model_function_wrapper"]
    args = {"input": y, "timestep": torch.tensor([0.9]), "c": {}}
    assert torch.equal(wrapper(model, args), y)
    args["timestep"] = torch.tensor([0.5])
    assert torch.equal(wrapper(model, args), model(y, args["timestep"]))
    unmounted = types.SimpleNamespace(model_options={})
    patching._key_mask_by_step(
        unmounted, plans={"creative_target": types.SimpleNamespace(sets=[creative])}
    )
    args["timestep"] = torch.tensor([0.9])
    assert torch.equal(unmounted.model_options["model_function_wrapper"](model, args), model(y, 0))


def test_layer_owner_early_masks_and_missing_ip(monkeypatch):
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", "layer")
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0.5")
    patcher = types.SimpleNamespace(model_options={})
    seen = []

    def calc(model, conds, x, sigma, options):
        seen.append(conds)
        return x

    install_layer_alpha(patcher, None, {}, 0, calc)
    hook = patcher.model_options["sampler_calc_cond_batch_function"]
    mask = torch.full((1, 4, 8), 0.25)
    conds = [[{"kilora_layer": a, "mask": mask} for a in ("ip", "creative", "functional")]]
    args = {
        "model": None,
        "conds": conds,
        "input": torch.zeros(1, 4, 4, 8),
        "sigma": torch.tensor([0.9]),
        "model_options": patcher.model_options,
    }
    hook(args)
    assert torch.all(seen[-1][0][0]["mask"] == 1)
    assert torch.all(seen[-1][0][1]["mask"] == 0)
    assert seen[-1][0][2]["mask"] is mask
    hook({**args, "sigma": torch.tensor([0.5])})
    at_threshold = seen[-1]
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0")
    baseline = types.SimpleNamespace(model_options={})
    install_layer_alpha(baseline, None, {}, 0, calc)
    baseline_hook = baseline.model_options["sampler_calc_cond_batch_function"]
    baseline_hook(args)
    baseline_hook({**args, "sigma": torch.tensor([0.5])})
    assert all(
        torch.equal(a["mask"], b["mask"]) for a, b in zip(at_threshold[0], seen[-1][0], strict=True)
    )
    monkeypatch.setenv("KILORA_AXIS_IP_EARLY", "0.5")
    fresh = types.SimpleNamespace(model_options={})
    install_layer_alpha(fresh, None, {}, 0, calc)
    fresh.model_options["sampler_calc_cond_batch_function"]({**args, "conds": [conds[0][1:]]})
    assert seen[-1][0][0]["mask"] is mask
    fresh = types.SimpleNamespace(model_options={})
    install_layer_alpha(fresh, None, {}, 0, calc, ip_mounted=False)
    fresh.model_options["sampler_calc_cond_batch_function"](args)
    assert seen[-1][0][0]["mask"] is mask
