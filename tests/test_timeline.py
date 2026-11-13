# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora.bake import TargetPlan
from kilora.comfyui import patching
from kilora.compute import dropout_key
from kilora.timeline import (
    Clock,
    Row,
    Timeline,
    TimelineError,
    check_row,
    clock_of,
    parse_rows,
    preview,
    scale_for,
    time_scales,
)

WIDTH = 16


def test_keyframes_linear_and_step():
    r = Row("all", "strength", "0:1, 0.5:0, 1:2")
    assert r.value(Clock(t=0.25)) == pytest.approx(0.5)
    assert r.value(Clock(t=0.75)) == pytest.approx(1.0)
    assert r.value(Clock(t=-1)) == 1 and r.value(Clock(t=2)) == 2
    s = dataclasses.replace(r, interpolation="step")
    assert s.value(Clock(t=0.49)) == 1 and s.value(Clock(t=0.5)) == 0


def test_expressions_see_every_variable():
    r = Row("all", "strength", "lerp(1, 0, t) + step / steps * 0 + (sigma > 0.5)")
    assert r.value(Clock(t=0.25, step=1, steps=4, sigma=0.9)) == pytest.approx(1.75)


@pytest.mark.parametrize("curve", ["__import__('os')", "t.real", "open", "[t]", "x + 1", "t if"])
def test_unsafe_or_unknown_expressions_are_errors(curve):
    with pytest.raises(TimelineError):
        Row("all", "strength", curve).value(Clock())


def test_clocks_progress_step_and_sigma():
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    c = clock_of(0.5, sigmas)
    assert (c.step, c.steps, c.t, c.sigma) == (2, 4, pytest.approx(2 / 3), 0.5)
    assert Row("all", "active", "0:1, 2:0", "step", "step").value(c) == 0
    assert Row("all", "active", "0:0, 0.6:1", "sigma", "step").value(c) == 0
    assert Row("all", "active", "0:0, 0.4:1", "sigma", "step").value(c) == 1
    assert clock_of(0.3, None) == Clock(sigma=0.3)


def test_check_row_errors_are_clear():
    with pytest.raises(TimelineError, match="unknown quantity"):
        check_row(Row("all", "speed", "1"))
    with pytest.raises(TimelineError, match="unknown target"):
        check_row(Row("style", "strength", "1"), ["mod-a"])
    with pytest.raises(TimelineError, match="range"):
        check_row(Row("all", "active", "0:0, 1:2"))
    with pytest.raises(TimelineError, match="unknown clock"):
        check_row(Row("all", "strength", "1", clock="minutes"))
    check_row(Row("functional", "strength", "1 - t"), ["mod-a"])
    check_row(Row("mod-a", "route_freeze", "0.4"), ["mod-a"])


def test_parse_rows_and_chaining():
    rows = parse_rows(
        "# comment\nall | strength | 1 - t\nmod-a | active | 0:1, 0.5:0 | progress | step\n"
    )
    assert rows[1] == Row("mod-a", "active", "0:1, 0.5:0", "progress", "step")
    with pytest.raises(TimelineError, match="row 1"):
        parse_rows("all | strength")
    t = Timeline().add(rows[0]).add(rows[1])
    assert isinstance(t, Timeline) and len(t) == 2 and t.rows_for("active") == [rows[1]]


def test_scale_for_matches_all_axis_and_entry():
    rows = [
        Row("all", "strength", "0.5"),
        Row("functional", "strength", "0"),
        Row("mod-a", "active", "0:1, 0.5:0", interpolation="step"),
    ]
    assert scale_for(rows, Clock(t=0.1), "mod-a", "creative") == 0.5
    assert scale_for(rows, Clock(t=0.6), "mod-a", "creative") == 0
    assert scale_for(rows, Clock(t=0.1), "mod-b", "functional") == 0
    assert scale_for(rows, Clock(t=0.9), "mod-b", "creative") == 0.5


def _plan(axes=("creative",), tags=("mod-a",)):
    sets = [
        dataclasses.replace(s, axis=a)
        for s, a in zip(_sets(len(axes), WIDTH, 4, 2, 8, seed=3), axes, strict=True)
    ]
    return TargetPlan(sets=sets, depth=1, name="layer", tags=list(tags))


def _run(plan, rows=None, t=0.0):
    x = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(1))
    h = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(2))
    with torch.no_grad(), dropout_key(1, 0.5):
        if rows is None:
            return plan.run(x, h), h
        with time_scales(rows, Clock(t=t)):
            return plan.run(x, h), h


def test_no_timeline_is_bit_identical_and_keeps_the_sets():
    plan = _plan()
    assert plan._sets() is plan.sets
    a, _ = _run(plan)
    b, _ = _run(plan, [Row("all", "strength", "1")])
    assert torch.equal(a, b)
    with time_scales([Row("all", "route_freeze", "0.4")], Clock()):
        assert plan._sets() is plan.sets


def test_active_off_equals_the_base_output_and_strength_scales_the_change():
    plan = _plan()
    off, h = _run(plan, [Row("mod-a", "active", "0:1, 0.5:0", interpolation="step")], t=0.7)
    assert torch.equal(off, h)
    on, _ = _run(plan, [Row("mod-a", "active", "0:1, 0.5:0", interpolation="step")], t=0.2)
    full, _ = _run(plan)
    assert torch.equal(on, full)
    scaled, _ = _run(plan, [Row("all", "strength", "0.5")])
    assert torch.allclose(scaled - h, (full - h) * 0.5, atol=1e-5)


def test_an_axis_row_leaves_the_other_axis_unchanged():
    both = _plan(("creative", "functional"), ("mod-a", "mod-b"))
    only_creative = TargetPlan(sets=both.sets[:1], depth=1, name="layer", tags=["mod-a"])
    a, _ = _run(both, [Row("functional", "strength", "0")])
    b, _ = _run(only_creative)
    assert torch.allclose(a, b, atol=1e-6)


def test_a_step_curve_through_the_model_wrapper_switches_the_mod_off_at_that_step():
    from test_comfyui import FakePatcher

    plan = _plan()
    timeline = Timeline([Row("mod-a", "active", "0:1, 2:0", clock="step", interpolation="step")])
    p = FakePatcher(None)
    patching._key_mask_by_step(p, timeline=timeline)
    wrapper = p.model_options["model_function_wrapper"]
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    x = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(1))
    h = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(2))

    def apply_model(inp, t, **c):
        with torch.no_grad():
            return plan.run(x, h)

    outs = []
    for sigma in sigmas[:-1]:
        args = {
            "input": torch.zeros(1, 4, 8, 8),
            "timestep": sigma.view(1),
            "c": {"transformer_options": {"sample_sigmas": sigmas}},
        }
        outs.append(wrapper(apply_model, args))
    assert not torch.equal(outs[0], h) and not torch.equal(outs[1], h)
    assert torch.equal(outs[2], h) and torch.equal(outs[3], h)


def test_timeline_overrides_point_values_and_accepts_joint_entries(monkeypatch):
    monkeypatch.setattr(patching, "_is_lycoris", lambda path: False)
    chain = [
        patching.ModEntry("lora", "/x/mod-a.safetensors", axis="creative"),
        patching.ModEntry("mod", "/x/m.kmod"),
    ]
    t = Timeline([Row("m", "route_freeze", "0.4"), Row("all", "gate_kappa", "2")])
    assert patching._timeline_overrides(t, chain, joint=False) == {
        "route_freeze": 0.4,
        "gate_kappa": 2.0,
    }
    assert (
        patching._timeline_overrides(Timeline([Row("mod-a", "strength", "0.5")]), chain, joint=True)
        == {}
    )
    assert patching.joined_names(chain, joint=True) == frozenset({"mod-a"})
    assert patching.joined_names(chain, joint=False) == frozenset()
    with pytest.raises(TimelineError, match="unknown target"):
        patching._timeline_overrides(Timeline([Row("nope", "strength", "1")]), chain, joint=False)
    assert patching._timeline_overrides(None, chain, joint=False) == {}
    assert patching.entry_name(chain[0]) == "mod-a" and patching.entry_name(chain[1]) == "m"


def test_preview_size_and_node_chaining():
    from kilora.comfyui import nodes as N

    img = preview(Timeline([Row("all", "strength", "1 - t"), Row("mod-a", "active", "0:1, 0.5:0")]))
    assert img.shape == (1, 360, 640, 3) and float(img.min()) >= 0 and float(img.max()) <= 1
    t1, p1 = N.KiLoRATimeline().row("all", "strength", "1 - t")
    t2, p2 = N.KiLoRATimeline().row(
        "mod-a", "active", "0:1, 0.5:0", timeline=t1, more_rows="functional | strength | 0.5"
    )
    assert len(t2) == 3 and p2.shape == (1, 360, 640, 3)
    with pytest.raises(patching.KiLoRAError):
        N.KiLoRATimeline().row("all", "strength", "9")


def test_region_attention_rows_resolve_per_call_like_the_node_window():
    from kilora.region_attention import RegionAttention

    node = RegionAttention("soft", strength=0.5, start=0.3, until=0.6, captions=True)
    rows = (
        Row("all", "region_attention", "0:0, 0.3:0.5, 0.6:0", interpolation="step"),
        Row("all", "caption_window", "0:0, 0.3:1, 0.6:0", interpolation="step"),
    )
    timed = dataclasses.replace(RegionAttention("soft", captions=True), timeline=rows)
    for t in (0.0, 0.29, 0.3, 0.45, 0.59, 0.6, 0.9):
        cfg = timed.at(Clock(t=t))
        assert cfg.active(t) == node.active(t) and cfg.captions_active(t) == node.captions_active(t)
        if cfg.active(t):
            assert cfg.strength == node.strength
    assert node.at(Clock(t=0.5)) is node


def test_region_isolation_rows_equal_the_node_window_through_a_sampler():
    from test_region_isolation import _patcher, _sample, _toy_calc

    from kilora.region_isolation import RegionIsolation

    log_a, log_b = [], []
    node, _ = patching.apply_region_isolation(
        _patcher(), RegionIsolation(1.0, 0.6, 0), _toy_calc(log_a)
    )
    p = _patcher()
    p.model_options[patching.APPLIED_KEY]["timeline"] = Timeline(
        [Row("all", "region_isolation", "0:1, 0.6:0", interpolation="step")]
    )
    timed, _ = patching.apply_region_isolation(p, RegionIsolation(0.0, feather=0), _toy_calc(log_b))
    a, b = _sample(node), _sample(timed)
    assert log_a == log_b and torch.equal(a, b)


def test_region_isolation_step_clock_row():
    from test_region_isolation import _patcher, _sample, _toy_calc

    from kilora.region_isolation import RegionIsolation

    log = []
    p = _patcher()
    p.model_options[patching.APPLIED_KEY]["timeline"] = Timeline(
        [Row("all", "region_isolation", "0:0, 3:1, 5:0", clock="step", interpolation="step")]
    )
    timed, _ = patching.apply_region_isolation(p, RegionIsolation(0.0), _toy_calc(log))
    _sample(timed, steps=8)
    assert log == [False] * 3 + [True, False] * 2 + [False] * 3


def test_fixed_sections_follow_the_entry_rows():
    from kilora.fixed_change import fixed_copy
    from kilora.sections import LowRankChange

    g = torch.Generator().manual_seed(0)
    base = torch.nn.Linear(6, 6)
    change = LowRankChange(torch.randn(2, 6, generator=g), torch.randn(6, 2, generator=g))
    change.tag, change.axis = "mod-a", "creative"
    fixed = fixed_copy(base, [change])
    x = torch.randn(3, 6, generator=g)
    with torch.no_grad():
        full, plain = fixed(x), base(x)
        with time_scales([Row("mod-a", "active", "0")], Clock()):
            assert torch.equal(fixed(x), plain)
        with time_scales([Row("creative", "strength", "0.5")], Clock()):
            assert torch.allclose(fixed(x) - plain, (full - plain) * 0.5, atol=1e-6)
        with time_scales([Row("other", "strength", "0")], Clock()):
            assert torch.equal(fixed(x), full)


def test_text_encoder_parts_take_the_scale_at_the_start(monkeypatch):
    from types import SimpleNamespace

    from kilora import text_encoder_lora

    mods = [SimpleNamespace(path="/x/mod-a.safetensors", label=None, strength=1.0)]
    seen = {}
    monkeypatch.setattr(patching, "plan_chain", lambda *a, **k: (SimpleNamespace(mods=mods), None))
    monkeypatch.setattr(
        dataclasses, "replace", lambda m, **kw: SimpleNamespace(**{**vars(m), **kw})
    )
    monkeypatch.setattr(
        text_encoder_lora,
        "apply_to_clip",
        lambda clip, ms: (seen.setdefault("s", [m.strength for m in ms]), []),
    )
    chain = [patching.ModEntry("lora", "/x/mod-a.safetensors")]
    t = Timeline([Row("mod-a", "strength", "0:0.25, 1:1")])
    patching.apply_clip(None, "clip", chain, None, t)
    assert seen["s"] == [0.25]


def _calls(timeline, joined=frozenset(), probe=None):
    from test_comfyui import FakePatcher

    p = FakePatcher(None)
    patching._key_mask_by_step(p, timeline=timeline, joined=joined)
    wrapper = p.model_options["model_function_wrapper"]
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    out = []
    for sigma in sigmas[:-1]:
        args = {
            "input": torch.zeros(1, 4, 8, 8),
            "timestep": sigma.view(1),
            "c": {"transformer_options": {"sample_sigmas": sigmas}},
        }
        out.append(wrapper(lambda *a, **k: probe(), args))
    return out


def test_depth_rows_set_the_depth_map_values_per_call():
    from kilora.call_values import current_token_depth

    t = Timeline(
        [
            Row("all", "depth_map", "0:0, 0.5:1", interpolation="step"),
            Row("all", "depth_base", "1 + step"),
        ]
    )
    assert _calls(t, probe=current_token_depth) == [
        (False, 1.0),
        (False, 2.0),
        (True, 3.0),
        (True, 4.0),
    ]
    assert (
        _calls(Timeline([Row("all", "strength", "1")]), probe=current_token_depth)
        == [(None, None)] * 4
    )
    with pytest.raises(TimelineError, match="target all"):
        check_row(Row("creative", "depth_map", "1"))


def test_entry_rows_inside_the_routed_set_become_expert_scales():
    from kilora.call_values import expert_scale_vector

    t = Timeline(
        [
            Row("mod-a", "strength", "0:1, 1:0"),
            Row("mod-b", "active", "0:1, 0.5:0", interpolation="step"),
        ]
    )
    joined = frozenset({"mod-a", "mod-b"})
    got = _calls(t, joined, probe=lambda: expert_scale_vector(["mod-a", "mod-b", "mod-c"]))
    assert got[0] == [1.0, 1.0, 1.0] and got[3] == [0.0, 0.0, 1.0]
    assert got[1][1] == 1.0 and got[2][1] == 0.0
    assert _calls(t, frozenset(), probe=lambda: expert_scale_vector(["mod-a"])) == [None] * 4
