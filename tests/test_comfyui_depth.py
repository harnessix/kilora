# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pickle

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_modlist import write_mod

from kilora import KiLoRAError
from kilora.comfyui import NODE_DISPLAY_NAME_MAPPINGS, patching
from kilora.comfyui import depth as D
from kilora.comfyui import nodes as N


def _build(text="", preset="none", **kw):
    out = N.KiLoRADepthSchedule().build(text, preset, **kw)
    return out["result"], out["ui"]["text"][0]


def test_node_interface(folders):
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRADepthSchedule"] == "KiLoRA Depth Schedule"
    assert N.KiLoRADepthSchedule.RETURN_TYPES == ("KILORA_DEPTH", "IMAGE")
    assert N.KiLoRADepthSchedule.RETURN_NAMES == ("depth_schedule", "preview")
    types = N.KiLoRADepthSchedule.INPUT_TYPES()
    assert list(types["required"]) == ["schedule", "preset"]
    assert types["required"]["preset"][0][0] == "none"
    assert {"flat", "late-deep", "mid-deep", "early-steps", "late-steps"} <= set(
        types["required"]["preset"][0]
    )
    apply = N.KiLoRAApply.INPUT_TYPES()
    assert list(apply["required"]) == ["model", "sets", "depth"]
    assert list(apply["optional"]) == [
        "iteration",
        "depth_schedule",
        "routing_scope",
        "routing_select",
        "output_gate",
        "gate_kappa",
        "path_width",
        "path_select",
        "clip",
        "route_freeze",
        "timeline",
        "settings",
        "depth_map",
        "linear_jump",
        "jump_threshold",
        "axis_alpha",
        "alpha_mask",
        "gate_history",
        "expert_text",
        "variation",
        "record_history",
        "candidates",
        "attention_route",
        "attention_route_cap",
        "size_norm",
        "radial_step",
        "orthogonal_step",
        "radial_budget",
    ]
    spec = apply["optional"]["depth_schedule"]
    assert spec[0] == "KILORA_DEPTH" and spec[1]["tooltip"]


def test_list_schedule_and_heatmap():
    (schedule, image), report = _build("0, 2, 4")
    assert schedule.depth(0, 28, "self_attn", 0.0, "") == 0
    assert schedule.depth(1, 28, "mlp", 0.5, "") == 2
    assert schedule.depth(27, 28, "other", 1.0, "") == 4
    assert pickle.loads(pickle.dumps(schedule)).depth(5, 28, "mlp", 0.0, "") == 4
    assert image.shape == (1, D.PREVIEW_STEPS * D.CELL, D.PREVIEW_BLOCKS * D.CELL, 3)
    assert image.dtype == torch.float32 and image.min() >= 0 and image.max() <= 1
    first = image[0, 0, 0]
    assert torch.all(first == 0)
    assert image[0, 0, D.CELL].sum() < image[0, 0, 2 * D.CELL].sum()
    assert "Depth 0 to 4" in report


def test_preset_progress_and_text_wins():
    (schedule, image), _ = _build("", "early-steps")
    assert schedule.depth(3, 28, "self_attn", 0.0, "") > schedule.depth(3, 28, "self_attn", 1.0, "")
    rows = image[0, :: D.CELL, 0].sum(-1)
    assert rows[0] > rows[-1]
    (text_wins, _), _ = _build("2", "early-steps")
    assert text_wins.depth(0, 28, "mlp", 0.0, "") == 2
    (none, image), report = _build("", "none")
    assert none is None and torch.all(image == 0) and "none" in report


def test_preview_size_follows_blocks():
    (_, image), _ = _build("1, 2", blocks=12)
    assert image.shape == (1, D.PREVIEW_STEPS * D.CELL, 12 * D.CELL, 3)


def test_bad_schedule_is_a_clear_error():
    with pytest.raises(KiLoRAError, match="depth"):
        _build("1, banana(")


def test_max_depth():
    assert D.max_depth(D.compile_depth("1, 2.5")) == 3
    assert D.max_depth(D.compile_depth("auto")) == 16
    assert D.max_depth(D.compile_depth("0")) == 0


def test_apply_without_schedule_is_unchanged(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(2))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    old, old_report, _ = N.KiLoRAApply().apply(base, chain, 2)
    new, new_report, _ = N.KiLoRAApply().apply(base, chain, 2, depth_schedule=None)
    assert old_report == new_report and "Depth schedule" not in new_report
    assert torch.equal(old.run(x), new.run(x))
    assert patching.applied(new)["depth_schedule"] is None


def test_apply_takes_the_schedule(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    (schedule, _), _ = _build("1, 3")
    patched, report, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, 1, depth_schedule=schedule
    )
    assert "Depth: 3" in report and "Depth schedule:" in report
    assert patching.applied(patched)["depth_schedule"] is schedule
    again, _ = patching.reapply(patched)
    assert patching.applied(again)["depth_schedule"] is schedule


def test_mod_default_and_workflow_override(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1, depth_schedule="2, 2")
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    patched, report, _ = N.KiLoRAApply().apply(base, chain, -1)
    used = patching.applied(patched)["depth_schedule"]
    assert used.depth(0, 28, "mlp", 0.0, "") == 2
    assert "from ink" in report and "Depth: 2" in report
    (override, _), _ = _build("1")
    patched, report, _ = N.KiLoRAApply().apply(base, chain, -1, depth_schedule=override)
    assert patching.applied(patched)["depth_schedule"] is override
    assert "from ink" not in report and "Depth: 1" in report
    loaded, _ = N.KiLoRALoadMod().load(base, "ink-1.0.0.kmod", 1.0)
    assert patching.applied(loaded)["depth_schedule"] is not None


def test_stored_schedule_round_trip():
    schedule = D.compile_depth("0.5, 1.5")
    import json

    back = D.from_manifest(json.loads(schedule.to_json()))
    assert back.depth(1, 28, "mlp", 0.0, "") == 1.5


def test_schedule_reaches_the_forward_pass(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(3))
    write_mod(
        denoiser, folders["kilora"], "ink", seed=1, target_modules=["layers.*.feed_forward.w2"]
    )
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    plain, _, _ = N.KiLoRAApply().apply(base, chain, 2)
    again, _, _ = N.KiLoRAApply().apply(base, chain, 2, depth_schedule=None)
    assert torch.equal(plain.run(x), again.run(x))
    (deep, _), _ = _build("1, 4")
    scheduled, _, _ = N.KiLoRAApply().apply(base, chain, 2, depth_schedule=deep)
    assert not torch.allclose(plain.run(x), scheduled.run(x))
    deepest, _, _ = N.KiLoRAApply().apply(base, chain, 4)
    assert not torch.allclose(deepest.run(x), scheduled.run(x))
    (flat, _), _ = _build("2")
    same, _, _ = N.KiLoRAApply().apply(base, chain, 1, depth_schedule=flat)
    assert torch.allclose(plain.run(x), same.run(x), atol=1e-6)
    (off, _), _ = _build("0, 0")
    none, _, _ = N.KiLoRAApply().apply(base, chain, 2, depth_schedule=off)
    assert torch.equal(none.run(x), base.run(x))


def test_mod_default_reaches_the_forward_pass(denoiser, folders):
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(4))
    write_mod(
        denoiser, folders["kilora"], "ink", seed=1, target_modules=["layers.*.feed_forward.w2"]
    )
    write_mod(
        denoiser,
        folders["kilora"],
        "deep",
        seed=1,
        depth_schedule="1, 4",
        target_modules=["layers.*.feed_forward.w2"],
    )
    base = FakePatcher(FakeBaseModel(denoiser))
    (plain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    (stored,) = N.KiLoRASelect().select("deep-1.0.0.kmod", 1.0, 1.0)
    a, _, _ = N.KiLoRAApply().apply(base, plain, 1)
    b, _, _ = N.KiLoRAApply().apply(base, stored, 1)
    assert not torch.allclose(a.run(x), b.run(x))
    c, _, _ = N.KiLoRAApply().apply(base, plain, 4)
    assert not torch.allclose(c.run(x), b.run(x))


def test_progress_from_the_sampler_sigmas():
    sigmas = torch.tensor([14.6, 7.0, 3.0, 1.0, 0.0])
    args = {"timestep": torch.tensor([3.0]), "c": {"transformer_options": {}}}
    assert patching._progress(args) == 0.0
    args["c"]["transformer_options"]["sample_sigmas"] = sigmas
    assert patching._progress(args) == pytest.approx(2 / 3)
    args["timestep"] = torch.tensor([14.6])
    assert patching._progress(args) == 0.0
    args["timestep"] = torch.tensor([1.0])
    assert patching._progress(args) == 1.0


def test_reapply_keeps_alpha_inputs_and_off_mask_is_inert(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    mask = torch.zeros(1, 8, 8)
    plain, _, _ = N.KiLoRAApply().apply(base, chain, 2)
    off, _, _ = N.KiLoRAApply().apply(base, chain, 2, alpha_mask=mask, axis_alpha="off")
    x = torch.randn(1, 5, 4)
    assert torch.equal(plain.run(x), off.run(x))
    auto, _, _ = N.KiLoRAApply().apply(base, chain, 2, alpha_mask=mask, axis_alpha="auto")
    from kilora.call_values import current_axis_alphas

    args = {"input": torch.zeros(1, 1, 2, 2), "timestep": torch.tensor([1.0]), "c": {}}
    values = auto.model_options["model_function_wrapper"](
        lambda *args, **kwargs: dict(current_axis_alphas()), args
    )
    assert torch.equal(values["ip"], torch.zeros(1, 2, 2))
    assert torch.equal(values["creative"], torch.ones(1, 2, 2))
    again, _ = patching.reapply(auto)
    assert patching.applied(again)["axis_alpha"] == "auto"
    assert patching.applied(again)["alpha_mask"] is mask


def test_reapply_replaces_layer_alpha_hooks_and_keeps_earlier_hooks(denoiser, folders):
    import sys

    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    calls = []

    def earlier_model(fn, args):
        calls.append("earlier model")
        return fn(args["input"], args["timestep"], **args["c"])

    def earlier_sampler(args):
        calls.append("earlier sampler")
        return args["model_options"]["model_function_wrapper"](
            lambda x, timestep, **c: x,
            {"input": args["input"], "timestep": args["sigma"], "c": {}},
        )

    base = FakePatcher(FakeBaseModel(denoiser))
    base.model_options.update(
        model_function_wrapper=earlier_model, sampler_calc_cond_batch_function=earlier_sampler
    )
    model, _, _ = N.KiLoRAApply().apply(base, chain, 2)
    model, _ = patching.reapply(model)
    model, _ = patching.reapply(model)
    options = model.model_options
    layer_calls = []

    def profile(frame, event, arg):
        name = frame.f_code.co_filename
        if (
            event == "call"
            and name.endswith(("/comfyui/layer_alpha.py", "/comfyui/layer_crop.py"))
            and frame.f_code.co_name in ("sampler", "model_wrapper")
        ):
            prefix = "crop " if name.endswith("layer_crop.py") else ""
            layer_calls.append(prefix + frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        options["sampler_calc_cond_batch_function"](
            {
                "conds": [[], None],
                "input": torch.ones(1, 1, 2, 2),
                "sigma": torch.tensor([1.0]),
                "model_options": options,
            }
        )
    finally:
        sys.setprofile(None)
    assert layer_calls == ["crop sampler", "sampler", "model_wrapper"]
    assert calls == ["earlier sampler", "earlier model"]
