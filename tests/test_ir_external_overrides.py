# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora import compute, editors, ir, route_freeze
from kilora.ir.dispatch import active_overrides


def test_budget_override_matches_native_alpha():
    weights = {"a": 0.8, "b": 0.4}
    override = ir.PartOverride("scalar", parameters={"alpha": 0.5})
    with active_overrides({"budget": override}):
        actual = compute.axis_budget(weights)
    assert actual == {key: 0.5 for key in weights}
    assert actual != compute.axis_budget(weights)


def test_freeze_step_override_matches_native_fraction():
    routes = [torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])] * 2
    layer = SimpleNamespace(
        route_freeze=0.2,
        iteration="accumulate",
        aggregation="accumulate",
        per_expert=1,
        scale=1.0,
        up=torch.eye(2),
        down=torch.eye(2),
        form="state",
        expert_mask=None,
    )
    rec = {"routes": {"layer": [[route] for route in routes]}, "gates": {}}
    reduced = {"routes": {"layer": [[route * 0.5] for route in routes]}, "gates": {}}
    expected = route_freeze.build(reduced, [layer], [0], "layer", [None], 2, (1, 2))
    original = route_freeze.build(rec, [layer], [0], "layer", [None], 2, (1, 2))
    override = ir.PartOverride("step", parameters={"fraction": 0.5})
    with active_overrides({"freeze": override}):
        actual = route_freeze.build(rec, [layer], [0], "layer", [None], 2, (1, 2))
    assert torch.equal(actual["scale"], expected["scale"])
    assert not torch.equal(actual["scale"], original["scale"])


def test_guard_override_matches_native_threshold(monkeypatch):
    scores = [{"gap": 1.0, "spread": 1.0}, {"gap": 0.7, "spread": 1.0}]
    record = SimpleNamespace(guard_score=lambda trial: scores[trial is not None])
    rule = SimpleNamespace(guard_trial=lambda state, new: object())
    session = SimpleNamespace(state=None, guard=True)
    assert editors.guard_check(rule, session, record, None)[0]
    monkeypatch.setattr(editors, "GUARD_GAP_DROP", 0.4)
    expected = editors.guard_check(rule, session, record, None)[0]
    monkeypatch.setattr(editors, "GUARD_GAP_DROP", 0.2)
    live_set = SimpleNamespace(
        name="live", ir_parts={"guard": ir.PartOverride("default", parameters={"gap_drop": 0.4})}
    )
    session.mod = SimpleNamespace(manifest={"live": {"live_set": "live"}}, sets=[live_set])
    assert editors.guard_check(rule, session, record, None)[0] is expected is False
    session.guard = False
    with pytest.raises(NotImplementedError, match="guard"):
        editors.guard_check(rule, session, record, None)


def test_mismatched_shared_layer_overrides_are_refused():
    a = compute.LayerSet(router=None, keys_in=None, ir_parts={"budget": ir.PartOverride("scalar")})
    b = compute.LayerSet(router=None, keys_in=None)
    with ir.override_scope(), pytest.raises(NotImplementedError, match="budget"):
        compute.run_layer(None, torch.zeros(1, 2, 3), [a, b], 1, shared=True)


def test_layer_scopes_budget_and_nested_dispatch(monkeypatch):
    setting = compute.LayerSet(
        router=None,
        keys_in=None,
        ir_parts={"budget": ir.PartOverride("scalar", parameters={"alpha": 0.5})},
    )

    def run(*args, **kwargs):
        with active_overrides({}):
            return compute.axis_budget({"a": 0.8, "b": 0.4})

    monkeypatch.setattr(compute, "_run_layer", run)
    with ir.override_scope():
        result = compute.run_layer(None, torch.zeros(1, 1, 1), [setting], 1)
    assert result == {key: 0.5 for key in ("a", "b")}


def test_unused_layer_budget_is_named(monkeypatch):
    setting = compute.LayerSet(
        router=None, keys_in=None, ir_parts={"budget": ir.PartOverride("scalar")}
    )
    monkeypatch.setattr(compute, "_run_layer", lambda *a, **kw: None)
    with ir.override_scope(), pytest.raises(NotImplementedError, match="budget"):
        compute.run_layer(None, torch.zeros(1, 1, 1), [setting], 1)


def test_freeze_decision_parameter_matches_native_threshold():
    route = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
    layer = SimpleNamespace(
        route_freeze=0.8,
        iteration="accumulate",
        aggregation="accumulate",
        per_expert=1,
        scale=1.0,
        up=torch.eye(2),
        down=torch.eye(2),
        form="state",
        expert_mask=None,
    )
    rec = {"routes": {"layer": [[route]]}, "gates": {}}
    override = ir.PartOverride("default", parameters={"threshold": 0.3})
    with route_freeze.freezing({}), compute.sampling_progress(0.5):
        assert not route_freeze.active([layer], 0.5)
        with active_overrides({"freeze": override}):
            assert route_freeze.active([layer], 0.5)
            actual = route_freeze.build(rec, [layer], [0], "layer", [None], 1, (1, 2))
        layer.route_freeze = 0.3
        assert route_freeze.active([layer], 0.5)
        expected = route_freeze.build(rec, [layer], [0], "layer", [None], 1, (1, 2))
    assert torch.equal(actual["scale"], expected["scale"])


def test_mapped_budget_replacement_scales_independent_alphas():
    from kilora.ir.defaults import _record, default_program
    from kilora.program import Signature, program_identity, validate

    record = _record("budget", "auto")
    record["instructions"][0]["attributes"]["value"] = 0.4
    record["identity"] = program_identity(record)
    default = default_program("budget", "auto", {"batch": 2, "tokens": 2})
    signature = Signature(
        {name: default.value_types[name] for name in record["inputs"]},
        {name: default.value_types[name] for name in record["outputs"]},
    )
    program = validate(record, signature)
    a = torch.full((2, 2, 1), 0.75)
    b = torch.full((2, 2, 1), 0.5)
    weights = {"a": a, "b": b}
    override = ir.PartOverride("auto", program=program)
    from kilora.ir.dispatch import interpreted_part

    interpreted_part([SimpleNamespace(ir_parts={"budget": override})])
    with active_overrides({"budget": override}):
        actual = compute.axis_budget(weights)
    expected = {name: value * 0.4 for name, value in weights.items()}
    assert all(torch.equal(actual[name], expected[name]) for name in weights)
    assert not torch.equal(actual["a"], compute.axis_budget(weights)["a"])


def test_target_plan_activates_override_scope(monkeypatch):
    from kilora.bake import TargetPlan

    setting = compute.LayerSet(
        router=torch.ones(1, 1),
        keys_in=torch.ones(1, 1, 1),
        ir_parts={"budget": ir.PartOverride("scalar", parameters={"alpha": 0.5})},
    )
    plan = TargetPlan(sets=[setting])
    monkeypatch.setattr(plan, "_with_roles", lambda output, regions: regions)
    monkeypatch.setattr(compute, "_run_layer", lambda *a, **kw: compute.axis_budget({"a": 0.8}))
    result = plan.run(None, torch.zeros(1, 1, 1))
    assert result == {"a": 0.5}
    assert compute._IR_ACTIVE_OVERRIDES.get() == 0


def test_save_refuses_unserializable_overrides(tmp_path):
    from kilora.modfile import save_mod

    setting = SimpleNamespace(ir_parts={"guard": ir.PartOverride("default")})
    with pytest.raises(NotImplementedError, match="guard"):
        save_mod(tmp_path / "mod.kmod", [setting], None)


def test_default_routed_override_is_not_silently_unused():
    from test_ir_dispatch import _layer

    output, setting = _layer()
    setting.ir_parts = {"carry": ir.PartOverride("mask_expect")}
    with (
        torch.no_grad(),
        compute.eager(),
        ir.override_scope(),
        pytest.raises(NotImplementedError, match="carry"),
    ):
        compute.run_layer(output, output, [setting], 1)


def test_cached_freeze_marks_override_as_used(monkeypatch):
    override = ir.PartOverride("default", parameters={"threshold": 0.3})
    entry = {"lead": (1, 2), "_ir_freeze_override": override}
    monkeypatch.setattr(route_freeze, "output", lambda *args: "cached")
    setting = SimpleNamespace(expert_mask=None)
    with route_freeze.freezing({"layer": entry}), active_overrides({"freeze": override}):
        result = route_freeze.run(
            lambda: None, None, torch.zeros(1, 2, 3), [setting], [0], "layer", [None], 1
        )
    assert result == "cached"


def test_plan_device_copy_preserves_overrides():
    from kilora.bake import TargetPlan

    override = ir.PartOverride("scalar", parameters={"alpha": 0.5})
    setting = compute.LayerSet(
        router=torch.ones(1, 1),
        keys_in=torch.ones(1, 1, 1),
        ir_parts={"budget": override},
    )
    copied = TargetPlan(sets=[setting]).to(torch.device("cpu"), torch.float32)
    assert copied.sets[0].ir_parts["budget"] is override
