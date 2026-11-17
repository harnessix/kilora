# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect

import pytest

from kilora.comfyui import nodes as N
from kilora.comfyui import settings as S
from kilora.i18n import catalog

MAIN = {
    "mix": "balanced",
    "depth": -1,
    "iteration": None,
    "routing_scope": None,
    "routing_select": None,
    "depth_schedule": None,
    "import_score": None,
    "lycoris_apply": "truncated",
    "candidates": 0,
    "candidate_summary": "keys",
    "routing_features": False,
    "cross_token_region": 0.0,
    "cross_token_prompt": 0.0,
    "cross_token_timestep": 0.0,
    "content_keys": "per_expert",
    "gate_stats": "fused",
}

DOCUMENTED = {
    "balanced": {},
    "keep_each_strong": {"routing_scope": "token", "routing_select": "top1"},
    "subtle": {"depth": 1, "routing_scope": "tensor", "routing_select": "sparsemax"},
    "stack": {"depth": 1},
}


@pytest.mark.parametrize("missing", [None, {}])
def test_no_settings_are_the_main_defaults(missing):
    assert S.resolve_settings(missing) == MAIN


@pytest.mark.parametrize("mix", list(DOCUMENTED))
def test_each_mix_maps_as_documented(mix):
    want = {**MAIN, **DOCUMENTED[mix], "mix": mix}
    assert S.resolve_settings({"mix": mix}) == want
    assert S.KiLoRASimple().make(mix) == (want,)
    assert S.KiLoRASettings().make(mix) == (want,)
    tip = catalog("en")["tip.mix"]
    assert f"{mix}:" in tip
    for key, value in DOCUMENTED[mix].items():
        assert str(value) in tip, (mix, key)


def test_the_simple_choices_are_the_documented_mixes():
    assert S.KiLoRASimple.INPUT_TYPES()["required"]["mix"][0] == list(DOCUMENTED)


def test_resolve_is_idempotent():
    for mix in DOCUMENTED:
        (s,) = S.KiLoRASettings().make(mix, routing_select="mod", iteration="path")
        assert S.resolve_settings(s) == s
        assert s["routing_select"] is None and s["iteration"] == "path"


def test_advanced_values_replace_the_mix():
    (s,) = S.KiLoRASettings().make("subtle", depth=3, routing_scope="token")
    assert s["depth"] == 3 and s["routing_scope"] == "token"
    assert s["routing_select"] == "sparsemax"
    (s,) = S.KiLoRASettings().make("subtle", depth=-1, routing_scope="mod")
    assert s["depth"] == -1 and s["routing_scope"] is None


def test_unknown_values_are_errors():
    with pytest.raises(ValueError, match="mix"):
        S.resolve_settings({"mix": "loud"})
    with pytest.raises(ValueError, match="routing_select"):
        S.resolve_settings({"routing_select": "top2"})
    with pytest.raises(ValueError, match="strength"):
        S.resolve_settings({"strength": 1.0})


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_apply(model, chain, depth, env, iteration, schedule, scope, select, **kw):
        seen.append(
            {
                "depth": depth,
                "iteration": iteration,
                "depth_schedule": schedule,
                "routing_scope": scope,
                "routing_select": select,
                "import_score": kw.get("import_score"),
                "lycoris_apply": kw.get("lycoris_apply"),
                "candidates": kw.get("candidates"),
                "candidate_summary": kw.get("candidate_summary"),
                "content_keys": kw.get("content_keys"),
                "gate_stats": kw.get("gate_stats"),
                "routing_features": kw.get("routing_features_enabled"),
                **dict(zip(S.CROSS_SETTINGS, kw.get("cross_weights", (0, 0, 0)), strict=True)),
            }
        )
        return model, ""

    monkeypatch.setattr(N.patching, "apply", fake_apply)
    monkeypatch.setattr(N, "_chain", lambda sets: sets)
    monkeypatch.setattr(N, "live_env", lambda: None)
    return seen


def test_apply_takes_jump_alpha_and_variation_inputs_last():
    types = N.KiLoRAApply.INPUT_TYPES()
    assert list(types["optional"])[-17:] == [
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
    assert types["optional"]["settings"][0] == S.SETTINGS
    choices, options = types["optional"]["size_norm"]
    assert choices == ["off", "weight"] and options["default"] == "off"


def test_apply_without_settings_is_unchanged(calls):
    N.KiLoRAApply().apply("m", [], -1)
    assert calls[-1] == {k: v for k, v in MAIN.items() if k != "mix"}


def test_apply_uses_the_settings(calls):
    (s,) = S.KiLoRASimple().make("subtle")
    N.KiLoRAApply().apply("m", [], -1, settings=s)
    assert calls[-1] == {k: v for k, v in s.items() if k != "mix"}


def test_apply_inputs_win_over_the_settings(calls):
    (s,) = S.KiLoRASimple().make("subtle")
    N.KiLoRAApply().apply("m", [], 4, iteration="path", routing_select="top1", settings=s)
    assert calls[-1]["depth"] == 4
    assert calls[-1]["iteration"] == "path"
    assert calls[-1]["routing_select"] == "top1"
    assert calls[-1]["routing_scope"] == "tensor"


def test_candidates_setting_and_apply_override(calls):
    (settings,) = S.KiLoRASettings().make("balanced", candidates=3)
    N.KiLoRAApply().apply("m", [], -1, settings=settings)
    assert calls[-1]["candidates"] == 3
    N.KiLoRAApply().apply("m", [], -1, settings=settings, candidates=2)
    assert calls[-1]["candidates"] == 2
    with pytest.raises(ValueError, match="candidates"):
        S.resolve_settings({"candidates": -1})


def test_candidate_summary_setting_and_apply(calls):
    (settings,) = S.KiLoRASettings().make("balanced", candidates=2, candidate_summary="read_map")
    N.KiLoRAApply().apply("m", [], -1, settings=settings)
    assert calls[-1]["candidate_summary"] == "read_map"
    with pytest.raises(ValueError, match="candidate_summary"):
        S.resolve_settings({"candidate_summary": "unknown"})


def test_candidate_summary_logic_is_required_and_checked():
    from kilora.reference_sections import validate

    settings = {
        **S.resolve_settings(None),
        "output_gate": "none",
        "gate_kappa": 0.0,
        "path_width": 0,
        "path_select": "mod",
        "route_freeze": 0.0,
        "axis_alpha": "off",
        "library": False,
    }
    validate("logic", {"version": 1, "settings": settings}, {})
    del settings["candidate_summary"]
    with pytest.raises(ValueError, match="candidate_summary"):
        validate("logic", {"version": 1, "settings": settings}, {})
    settings["candidate_summary"] = "read_map"
    validate("logic", {"version": 1, "settings": settings}, {})
    settings["candidate_summary"] = "unknown"
    with pytest.raises(ValueError, match="candidate_summary"):
        validate("logic", {"version": 1, "settings": settings}, {})


def test_routing_features_setting(calls):
    (settings,) = S.KiLoRASettings().make("balanced", routing_features=True)
    N.KiLoRAApply().apply("m", [], -1, settings=settings)
    assert calls[-1]["routing_features"] is True
    with pytest.raises(ValueError, match="routing_features"):
        S.resolve_settings({"routing_features": 1})


def test_cross_settings_reach_apply(calls):
    (settings,) = S.KiLoRASettings().make("balanced", cross_token_prompt=0.7)
    N.KiLoRAApply().apply("m", [], -1, settings=settings)
    assert calls[-1]["cross_token_prompt"] == 0.7


def test_gate_stats_setting_and_environment(monkeypatch, calls):
    from kilora import compute

    monkeypatch.delenv("KILORA_GATE_STATS", raising=False)
    (settings,) = S.KiLoRASettings().make("balanced", gate_stats="torch")
    N.KiLoRAApply().apply("m", [], -1, settings=settings)
    assert calls[-1]["gate_stats"] == "torch"
    with compute.gate_stats_mode("torch"):
        assert compute._GATE_STATS_MODE.get() == "torch"
        with compute.gate_stats_mode("fused"):
            assert compute._GATE_STATS_MODE.get() == "torch"
    assert compute._GATE_STATS_MODE.get() is None
    monkeypatch.setenv("KILORA_GATE_STATS", "fused")
    with compute.gate_stats_mode("torch"):
        assert compute._GATE_STATS_MODE.get() == "fused"
    monkeypatch.setenv("KILORA_GATE_STATS", "invalid")
    with pytest.raises(ValueError, match="gate_stats"):
        compute.resolve_gate_stats("torch")
    with pytest.raises(ValueError, match="gate_stats"):
        S.resolve_settings({"gate_stats": "invalid"})


def test_gate_stats_wrapper_reaches_computation(monkeypatch):
    import torch

    from kilora import compute
    from kilora.comfyui.patching import _gate_stats_wrapper

    class Model:
        model_options = {}

    monkeypatch.delenv("KILORA_GATE_STATS", raising=False)
    model = Model()
    _gate_stats_wrapper(model, "torch")
    seen = []

    def forward(x, t, **kwargs):
        seen.append(compute._GATE_STATS_MODE.get())
        return compute._y0_gate_stats(x)

    y = torch.randn(2, 3, 4)
    args = {"input": y, "timestep": 0, "c": {}}
    result = model.model_options["model_function_wrapper"](forward, args)
    assert seen == ["torch"]
    sq = y.float().square()
    assert torch.equal(result[0], sq.sum(-1).sqrt() + compute.DROPOUT_DELTA)
    monkeypatch.setenv("KILORA_GATE_STATS", "fused")
    model.model_options["model_function_wrapper"](forward, args)
    assert seen[-1] == "fused"


def test_import_edit_settings_are_last_and_default_to_unchanged():
    types = N.KiLoRALiveOptions.INPUT_TYPES()
    assert list(types["required"]) == ["model", "mod", "learning", "speed"]
    assert list(types["optional"]) == [
        "keep_previous",
        "measure_size",
        "reduce_with",
        "pin",
        "controller",
        "measure",
        "measure_combos",
        "edits",
        "live_expert_mass",
        "live_expert_top",
        "live_expert_history",
        "live_expert_copies",
        "write_measure",
        "forward_draws",
        "forward_sigma",
        "forward_chunk",
        "write_resolution",
        "probe_rows",
        "probe_call",
        "probe_budget",
        "probe_draws",
        "measure_samples",
        "learn",
    ]
    assert types["optional"]["write_measure"][0] == ["unchanged", "forward", "candidates"]
    assert types["optional"]["write_measure"][1]["default"] == "unchanged"
    assert types["optional"]["learn"][0] == ["unchanged", "measure", "prediction"]
    assert types["optional"]["learn"][1]["default"] == "unchanged"
    assert types["optional"]["edits"][0] == ["unchanged", "on", "off"]
    assert types["optional"]["controller"][0] == ["unchanged", "r1", "rprop"]
    for name in ("edits", "controller"):
        assert types["optional"][name][1]["default"] == "unchanged"
    assert types["optional"]["write_resolution"][0] == "INT"
    spec = types["optional"]["write_resolution"][1]
    assert (spec["default"], spec["min"], spec["max"]) == (-1, -1, 4096)
    assert inspect.signature(N.KiLoRALiveOptions.set).parameters["write_resolution"].default == -1


def test_layer_compose_input_defaults_to_separate():
    types = N.KiLoRALayerConditioning.INPUT_TYPES()
    assert list(types["optional"])[-3:] == ["compose", "region_penalty", "mount_scope"]
    assert types["optional"]["compose"][0] == ["separate", "single"]
    assert types["optional"]["compose"][1]["default"] == "separate"


def test_layer_region_penalty_and_mount_scope_inputs():
    optional = N.KiLoRALayerConditioning.INPUT_TYPES()["optional"]
    kind, spec = optional["region_penalty"]
    assert kind == "FLOAT"
    assert {k: v for k, v in spec.items() if k != "tooltip"} == {
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    }
    choices, spec = optional["mount_scope"]
    assert choices == ["cross_kv", "all"]
    assert spec["default"] == "all"
    params = inspect.signature(N.KiLoRALayerConditioning.layer).parameters
    assert params["region_penalty"].default is None
    assert params["mount_scope"].default is None


def test_size_norm_off_leaves_the_chain_alone(calls, monkeypatch):
    def never(*a, **k):
        raise AssertionError("size_norm=off must not measure files")

    monkeypatch.setattr(N, "size_normed_chain", never)
    N.KiLoRAApply().apply("m", [], -1)
    N.KiLoRAApply().apply("m", [], -1, size_norm="off")
    assert calls[-1] == calls[-2] == {k: v for k, v in MAIN.items() if k != "mix"}


def test_size_norm_weight_scales_lora_entries_only(calls, monkeypatch):
    seen = []

    def fake(model, chain, every_lora=False, radial_budget=0.0):
        seen.append((every_lora, radial_budget))
        return chain, ["a: size_norm=weight, size S_f 0.01, scale c_f 1.785."]

    monkeypatch.setattr(N, "size_normed_chain", fake)
    N.KiLoRAApply().apply("m", [], -1, size_norm="weight")
    N.KiLoRAApply().apply("m", [], -1, size_norm="weight", radial_budget=2.0)
    assert seen == [(True, 0.0), (True, 0.0)]
    with pytest.raises(N.patching.KiLoRAError):
        N.KiLoRAApply().apply("m", [], -1, size_norm="output")


def test_import_lora_node_has_size_norm(monkeypatch):
    monkeypatch.setattr(N.FOLDERS, "list", lambda *a, **k: [])
    types = N.KiLoRAImportLoRA.INPUT_TYPES()
    assert types["required"]["strength"][1]["default"] == 0.7
    choices, options = types["optional"]["size_norm"]
    assert choices == ["off", "weight"] and options["default"] == "off"


def test_apply_carries_the_depth_step_inputs(monkeypatch):
    optional = N.KiLoRAApply.INPUT_TYPES()["optional"]
    for name, default, high in (
        ("radial_step", 0.5, 2.0),
        ("orthogonal_step", 1.0, 2.0),
        ("radial_budget", 0.0, 4.0),
    ):
        kind, spec = optional[name]
        assert kind == "FLOAT"
        assert (spec["default"], spec["min"], spec["max"]) == (default, 0.0, high)
    seen = []

    def fake_apply(*args, **kw):
        seen.append(kw.get("depth_steps"))
        return args[0], ""

    monkeypatch.setattr(N.patching, "apply", fake_apply)
    monkeypatch.setattr(N, "_chain", lambda sets: sets)
    monkeypatch.setattr(N, "live_env", lambda: None)
    N.KiLoRAApply().apply("m", [], -1)
    assert seen[-1] == (0.5, 1.0, 0.0)
    N.KiLoRAApply().apply("m", [], -1, radial_step=0.0, orthogonal_step=1.5, radial_budget=1.0)
    assert seen[-1] == (0.0, 1.5, 1.0)


def test_apply_without_radial_step_builds_the_comfyui_default(monkeypatch):
    seen = []

    def fake_apply(*args, **kw):
        seen.append(kw.get("depth_steps"))
        return args[0], ""

    monkeypatch.setattr(N.patching, "apply", fake_apply)
    monkeypatch.setattr(N, "_chain", lambda sets: sets)
    monkeypatch.setattr(N, "live_env", lambda: None)
    N.KiLoRAApply().apply("m", [], -1)
    assert seen == [(0.5, 1.0, 0.0)]


def test_depth_step_values_are_checked():
    from kilora.compute import check_depth_step

    assert check_depth_step("radial_budget", 4.0) == 4.0
    with pytest.raises(ValueError, match="radial_step"):
        check_depth_step("radial_step", 2.5)
    with pytest.raises(ValueError, match="orthogonal_step"):
        check_depth_step("orthogonal_step", -0.1)


def test_prompt_solve_node_defaults_allow_family_sampling():
    from kilora.comfyui.prompt_solve import KiLoRASolveFromPrompts
    from kilora.library.contrast import Settings

    inputs = KiLoRASolveFromPrompts.INPUT_TYPES()["required"]
    defaults = Settings()
    for name in ("rank", "negative_prompt"):
        assert inputs[name][1]["default"] == getattr(defaults, name)
    assert tuple(map(int, inputs["seeds"][1]["default"].split())) == defaults.seeds
    for name, auto in (("steps", 0), ("size", 0), ("cfg", -1.0)):
        assert inputs[name][1]["default"] == auto
        assert inputs[name][1]["min"] == auto
    assert inputs["mode"][1]["default"] == "single"
    assert inputs["overwrite"][1]["default"] is False


def test_pack_publication_defaults():
    from kilora.comfyui.pack_nodes import KiLoRAPack
    from kilora.comfyui.prompt_solve import KiLoRASolveFromPrompts

    assert KiLoRASolveFromPrompts.INPUT_TYPES()["required"]["output"][1]["default"] == "lora"
    inputs = KiLoRAPack.INPUT_TYPES()["required"]
    assert inputs["strength"][1]["default"] == 0.7
    assert inputs["depth"][1]["default"] == -1
    assert inputs["copy_mods"][1]["default"] is False


def test_build_editor_pack_inputs_order_and_defaults():
    from kilora.comfyui.nodes import NODE_CLASS_MAPPINGS, KiLoRABuildEditorPack

    assert list(NODE_CLASS_MAPPINGS)[-1] == "KiLoRABuildEditorPack"
    types = KiLoRABuildEditorPack.INPUT_TYPES()
    assert list(types["required"]) == [
        "model",
        "clip",
        "name",
        "prompts",
        "base_family",
        "draws",
        "common",
        "family",
    ]
    assert list(types["optional"]) == ["loras", "orthogonalize", "size", "steps", "seed"]
    required, optional = types["required"], types["optional"]
    assert required["draws"][1]["default"] == 2
    assert required["common"][1]["default"] == 48
    assert required["family"][1]["default"] == 16
    assert required["base_family"][1]["default"] == "auto"
    assert optional["orthogonalize"][1]["default"] is False
    assert optional["size"][1]["default"] == 0 and optional["steps"][1]["default"] == 0


def test_apply_ignores_deprecated_attention_widgets(calls):
    node = N.KiLoRAApply()
    node.apply("m", [], -1)
    expected = calls[-1]
    node.apply("m", [], -1, attention_route="add", attention_route_cap=9.0)
    assert calls[-1] == expected
    assert "attention_route" not in calls[-1]
    assert "attention_route_cap" not in calls[-1]
    tips = catalog("en")
    assert "ignored" in tips["tip.attention_route"]
    assert "ignored" in tips["tip.attention_route_cap"]


@pytest.mark.parametrize("mode", [None, "chain", "1"])
@pytest.mark.parametrize("form", ["input", "state"])
def test_ip_depth_defaults_to_one_with_optional_chain(mode, form, monkeypatch):
    from types import SimpleNamespace

    import torch
    from test_learned_cross import _set

    from kilora import Settings, compute
    from kilora.bake import ScopedSet, plan_targets
    from kilora.comfyui.patching import ModEntry, _with_regions

    if mode is None:
        monkeypatch.delenv("KILORA_IP_DEPTH", raising=False)
    else:
        monkeypatch.setenv("KILORA_IP_DEPTH", mode)
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    entries = [
        ModEntry(kind="mod", path="style.kmod", axis="creative"),
        ModEntry(kind="mod", path="character.kmod", axis="ip"),
    ]
    mods = [
        SimpleNamespace(path=e.path, label=e.path, settings=Settings(), sets=[_set()])
        for e in entries
    ]
    for mod in mods:
        mod.sets[0].form = form
    scoped = [ScopedSet(m.sets[0], gate_mode="adaptive", import_score="derived") for m in mods]
    plan = SimpleNamespace(mods=mods, scoped_sets=lambda: scoped)
    creative, ip = _with_regions(plan, entries)
    assert ip.gate_mode == creative.gate_mode
    assert ip.route_freeze is None
    assert ip.fixed_depth == (None if mode == "chain" else 1)
    layers = [plan_targets([s], 1)["layer"].sets[0] for s in (creative, ip)]
    y0 = torch.tensor([[[1.0, 0.4, 0.2, 0.1], [0.4, 1.0, 0.2, 0.1]]])
    with torch.no_grad(), compute.eager():
        for depth in range(1, 9):
            schedule = SimpleNamespace(depth=lambda *_, depth=depth: depth)
            first = compute.run_layer(y0, y0, layers[:1], 16, schedule=schedule)
            character = compute.run_layer(y0, y0, layers[1:], 16, schedule=schedule)
            expected = first if mode == "chain" else compute.run_layer(y0, y0, layers[:1], 1)
            torch.testing.assert_close((character - y0).norm(dim=-1), (expected - y0).norm(dim=-1))


def test_ip_depth_rejects_unknown_switch(monkeypatch):
    from types import SimpleNamespace

    from test_learned_cross import _set

    from kilora import Settings
    from kilora.bake import ScopedSet
    from kilora.comfyui.patching import ModEntry, _with_regions

    monkeypatch.setenv("KILORA_IP_DEPTH", "other")
    entry = ModEntry(kind="mod", path="character.kmod", axis="ip")
    mod = SimpleNamespace(path=entry.path, label=entry.path, settings=Settings(), sets=[_set()])
    plan = SimpleNamespace(mods=[mod], scoped_sets=lambda: [ScopedSet(mod.sets[0])])
    with pytest.raises(ValueError, match="KILORA_IP_DEPTH"):
        _with_regions(plan, [entry])


def test_image_solve_defaults_match_prompt_family_sampling():
    from kilora.comfyui.image_solve import KiLoRASolveFromImages
    from kilora.comfyui.prompt_solve import KiLoRASolveFromPrompts

    inputs = KiLoRASolveFromImages.INPUT_TYPES()["required"]
    prompt = KiLoRASolveFromPrompts.INPUT_TYPES()["required"]
    for name in ("rank", "steps", "cfg", "size", "seeds", "overwrite"):
        assert inputs[name][0] == prompt[name][0]
        assert inputs[name][1]["default"] == prompt[name][1]["default"]
    assert inputs["ignore_guard"][1]["default"] is False
    assert set(KiLoRASolveFromImages.INPUT_TYPES()["optional"]) == {
        "images",
        "captions",
        "folder",
        "contrast_components",
        "masks",
        "mask_contrast",
        "two_stage",
        "mask_start",
        "mask_samples",
        "refit_rounds",
        "guard_rule",
        "trajectory_check",
    }


def test_depth_map_source_is_last_with_the_auto_inputs():
    types = N.KiLoRADepthMap.INPUT_TYPES()
    assert list(types["required"]) == [
        "min_depth",
        "max_depth",
        "invert",
        "gamma",
        "feather",
        "start",
        "base",
        "source",
    ]
    source = types["required"]["source"]
    assert list(source[0]) == ["auto", "mask", "image"]
    assert source[1]["default"] == "mask" and source[1]["tooltip"]
    optional = types["optional"]
    assert list(optional) == [
        "mask",
        "image",
        "mean_depth",
        "detail_weight",
        "change_weight",
        "disagreement_weight",
        "smooth",
    ]
    defaults = {name: optional[name][1]["default"] for name in list(optional)[2:]}
    assert defaults == {
        "mean_depth": -1.0,
        "detail_weight": 1.0,
        "change_weight": 1.0,
        "disagreement_weight": 0.0,
        "smooth": 2.0,
    }
    assert optional["smooth"][1].get("advanced") is True
    assert all(optional[name][1]["tooltip"] for name in defaults)
    params = inspect.signature(N.KiLoRADepthMap.build).parameters
    assert params["source"].default == "mask"
    assert {name: params[name].default for name in defaults} == defaults


def test_image_contrast_components_advanced_input_and_help():
    from kilora.comfyui.help import node_defs
    from kilora.comfyui.image_solve import KiLoRASolveFromImages

    spec = KiLoRASolveFromImages.INPUT_TYPES()["optional"]["contrast_components"]
    assert spec[0] == "INT"
    assert spec[1]["default"] == 4
    assert spec[1]["min"] == 0 and spec[1]["max"] == 4
    assert spec[1]["advanced"] is True
    tip = catalog("en")["tip.image_solve_contrast_components"]
    assert "principal directions" in tip
    assert "0 writes the previous mean-only weights" in tip
    assert spec[1]["tooltip"] == tip
    definitions = node_defs("en", {"KiLoRASolveFromImages": KiLoRASolveFromImages})
    assert definitions["KiLoRASolveFromImages"]["inputs"]["contrast_components"] == {
        "name": catalog("en")["in.image_solve_contrast_components"],
        "tooltip": tip,
    }


def test_image_solve_measurement_switches_are_advanced_and_keep_defaults():
    from kilora.comfyui.image_solve import KiLoRASolveFromImages

    inputs = KiLoRASolveFromImages.INPUT_TYPES()["optional"]
    expected = {
        "mask_start": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.95, "advanced": True}),
        "mask_samples": ("STRING", {"default": "", "advanced": True}),
        "refit_rounds": ("INT", {"default": 0, "min": 0, "max": 8, "advanced": True}),
        "guard_rule": (["loss", "drift"], {"default": "loss", "advanced": True}),
        "trajectory_check": ("BOOLEAN", {"default": False, "advanced": True}),
    }
    for name, (expected_kind, expected_options) in expected.items():
        kind, options = inputs[name]
        assert kind == expected_kind
        assert {key: options[key] for key in expected_options} == expected_options
