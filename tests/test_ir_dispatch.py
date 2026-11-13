# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import compute, fused, ir
from kilora.ir.dispatch import lowering


@pytest.fixture(autouse=True)
def _override_scope():
    with ir.override_scope():
        yield


def _layer():
    generator = torch.Generator().manual_seed(1190)
    y = torch.randn(2, 4, 6, generator=generator)
    setting = compute.LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 6, generator=generator) * 0.1,
        per_expert=2,
        gate_mode="adaptive",
    )
    return y, setting


def test_default_and_forced_interpreter_parity():
    y, setting = _layer()
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 3, target="dispatch")
        with ir.interpret_all():
            assert lowering("carry", "mask_expect") is None
            result = compute.run_layer(y, y, [setting], 3, target="dispatch")
        assert lowering("carry", "mask_expect") == "native"
    assert torch.equal(result, native)


def test_parameter_override_interprets_and_refuses_fusion(monkeypatch):
    y, setting = _layer()
    monkeypatch.setenv("KILORA_MASKS", "expect")
    setting.ir_parts = {"carry": ir.PartOverride("mask_expect", parameters={"delta": 0.2})}
    assert lowering("carry", "mask_expect", parameters={"delta": 0.2}) is None
    assert not fused.fused_ok(y, [setting])
    assert fused.LAST_REASON[0] == "interpreted part carry"
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        modified = compute.run_layer(y, y, [setting], 3, target="dispatch")
        setting.ir_parts = {}
        native = compute.run_layer(y, y, [setting], 3, target="dispatch")
    assert not torch.equal(modified, native)


def test_force_context_restores_after_exception():
    with pytest.raises(RuntimeError), ir.interpret_all():
        raise RuntimeError("test")
    assert lowering("carry", "mask_expect") == "native"


def test_lowering_is_selected_for_default_program_and_disabled_in_context(monkeypatch):
    from kilora.ir import driver
    from kilora.ir.dispatch import active_overrides

    previous = torch.randn(1, 2, 3)
    change = torch.randn(1, 2, 3)
    inputs = {"previous_state": previous, "change": change}
    native = lowering("iterate", "accumulate")
    assert callable(native)
    expected = native(inputs)
    original = driver.interpret

    def fail(*args, **kwargs):
        raise AssertionError("default program should use its lowering")

    monkeypatch.setattr(driver, "interpret", fail)
    with active_overrides({}):
        actual = driver._run("iterate", "accumulate", inputs)
    assert torch.equal(actual["state"], expected["state"])
    monkeypatch.setattr(driver, "interpret", original)
    with ir.interpret_all():
        interpreted = driver._run("iterate", "accumulate", inputs)
    assert torch.equal(interpreted["state"], expected["state"])
    assert torch.equal(interpreted["output"], expected["output"])


def test_uninvoked_override_is_not_silently_ignored():
    y, setting = _layer()
    setting.ir_parts = {"carry": ir.PartOverride("mask_expect", parameters={"delta": 0.2})}
    with torch.no_grad(), compute.eager(), pytest.raises(NotImplementedError, match="not invoked"):
        compute.run_layer(y, y, [setting], 1)


@pytest.mark.parametrize(
    ("part", "variant", "inputs"),
    [
        (
            "score",
            "weighted",
            {"projection": torch.randn(2, 3, 2, 4), "weights": torch.rand(2, 3, 2)},
        ),
        (
            "carry",
            "drop",
            {
                "change": torch.randn(2, 3, 4),
                "carried": torch.randn(2, 3, 4),
                "mask": torch.rand(2, 3, 4),
                "base_read": torch.randn(2, 3, 4),
                "uniforms": torch.rand(2, 3, 4),
            },
        ),
        (
            "carry",
            "dropped",
            {
                "change": torch.randn(2, 3, 4),
                "carried": torch.randn(2, 3, 4),
                "mask": torch.rand(2, 3, 4),
                "base_read": torch.randn(2, 3, 4),
                "uniforms": torch.rand(2, 3, 4),
            },
        ),
    ],
)
def test_part_lowering_matches_interpreter(part, variant, inputs):
    from kilora.ir import driver
    from kilora.ir.dispatch import active_overrides

    with active_overrides({}):
        actual = driver._run(part, variant, inputs)
    with ir.interpret_all():
        expected = driver._run(part, variant, inputs)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in actual)


def test_changed_identity_cannot_select_a_native_lowering():
    from dataclasses import replace

    original = ir.default_program("iterate", "accumulate")
    changed = replace(original, identity="a" * 64)
    assert callable(lowering("iterate", "accumulate", original))
    assert lowering("iterate", "accumulate", changed) is None


@pytest.mark.parametrize("case", ["path", "input", "token_depth"])
def test_interpret_all_supported_layer_parity(case):
    y, setting = _layer()
    setting.gate_mode = "none"
    kwargs = {}
    if case == "path":
        setting.iteration = "path"
    elif case == "input":
        setting.form = "input"
    else:
        kwargs["token_depth"] = torch.tensor([[1, 2, 3, 2], [2, 3, 1, 2]])
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 3, target="dispatch", **kwargs)
        with ir.interpret_all():
            result = compute.run_layer(y, y, [setting], 3, target="dispatch", **kwargs)
    assert torch.equal(result, native)


def test_interpret_all_candidates_has_one_dispatch_note(monkeypatch):
    y, setting = _layer()
    setting.candidates = 1
    notes = []
    monkeypatch.setattr(fused, "_note", notes.append)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 2)
        notes.clear()
        with ir.interpret_all():
            result = compute.run_layer(y, y, [setting], 2)
    assert torch.equal(result, native)
    assert len(notes) == 1
    assert notes[0].startswith("interpreted part")


def test_native_identity_table_matches_shipped_records():
    from kilora.ir.defaults import _record
    from kilora.ir.identities import NATIVE_IDENTITIES
    from kilora.program import program_identity

    for (part, variant), identity in NATIVE_IDENTITIES.items():
        record = _record(part, variant)
        assert identity == record["identity"] == program_identity(record), (part, variant)


def test_distinct_gate_overrides_use_each_set(monkeypatch):
    import copy

    monkeypatch.setenv("KILORA_MASKS", "expect")
    y, first = _layer()
    second = copy.deepcopy(first)
    first.ir_parts = {"gate": ir.PartOverride("adaptive", parameters={"kappa": 0.2})}
    second.ir_parts = {"gate": ir.PartOverride("adaptive", parameters={"kappa": 2.0})}
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        result = compute.run_layer(y, y, [first, second], 3)
        first.ir_parts = {}
        second.ir_parts = {}
        first.gate_kappa, second.gate_kappa = 0.2, 2.0
        native = compute.run_layer(y, y, [first, second], 3)
    assert torch.equal(result, native)


def test_shared_routing_refuses_different_select_overrides():
    import copy

    y, first = _layer()
    second = copy.deepcopy(first)
    first.ir_parts = {"select": ir.PartOverride("token_top1")}
    second.ir_parts = {"select": ir.PartOverride("token_sparsemax")}
    with pytest.raises(NotImplementedError, match="different select overrides"), ir.interpret_all():
        compute.run_layer(y, y, [first, second], 2, shared=True)


def test_distinct_carry_overrides_use_each_set(monkeypatch):
    import copy

    monkeypatch.setenv("KILORA_MASKS", "expect")
    y, first = _layer()
    second = copy.deepcopy(first)
    first.gate_mode = second.gate_mode = "none"
    first.ir_parts = {"carry": ir.PartOverride("mask_expect", parameters={"delta": 0.2})}
    second.ir_parts = {"carry": ir.PartOverride("mask_expect", parameters={"delta": 0.8})}
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        actual = compute.run_layer(y, y, [first, second], 3, target="carry")
        second.ir_parts = {"carry": ir.PartOverride("mask_expect", parameters={"delta": 0.2})}
        changed = compute.run_layer(y, y, [first, second], 3, target="carry")
    assert not torch.equal(actual, changed)


def test_interpret_all_gradient_parity():
    import copy

    y, setting = _layer()
    setting.gate_mode = "none"
    setting.up.requires_grad_()
    other = copy.deepcopy(setting)
    with compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 3, target="grad")
        native.sum().backward()
        with ir.interpret_all():
            result = compute.run_layer(y, y, [other], 3, target="grad")
        result.sum().backward()
    assert torch.equal(result, native)
    assert torch.equal(other.up.grad, setting.up.grad)


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("selection", ["top1", "sparsemax"])
@pytest.mark.parametrize("path_select", ["converged", "score", "mix"])
def test_interpret_all_path_choices(scope, selection, path_select):
    y, setting = _layer()
    setting.gate_mode = "none"
    setting.iteration = "path"
    setting.routing_scope = scope
    setting.routing_select = selection
    setting.path_select = path_select
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 3, target="path-choice")
        with ir.interpret_all():
            result = compute.run_layer(y, y, [setting], 3, target="path-choice")
    assert torch.equal(result, native)


def test_keep_override_changes_only_its_set(monkeypatch):
    import copy

    monkeypatch.setenv("KILORA_MASKS", "expect")
    y, first = _layer()
    second = copy.deepcopy(first)
    first.gate_mode = second.gate_mode = "none"
    first.ir_parts = {"keep": ir.PartOverride("default", parameters={"decay_base": 0.8})}
    second.ir_parts = {"keep": ir.PartOverride("default", parameters={"decay_base": 0.2})}
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        actual = compute.run_layer(y, y, [first, second], 3, target="keep")
        second.ir_parts = {"keep": ir.PartOverride("default", parameters={"decay_base": 0.8})}
        changed = compute.run_layer(y, y, [first, second], 3, target="keep")
    assert not torch.equal(actual, changed)


def test_forced_path_calls_entry_select_and_iterate(monkeypatch):
    from kilora.ir import driver

    y, setting = _layer()
    setting.gate_mode = "none"
    setting.iteration = "path"
    used = set()
    original = driver._run

    def record(part, *args, **kwargs):
        used.add(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    with torch.no_grad(), compute.eager(), ir.interpret_all():
        compute.run_layer(y, y, [setting], 3)
    assert {"entry", "select", "iterate"} <= used


def test_forced_accumulate_invokes_keep(monkeypatch):
    from kilora.ir import driver

    y, setting = _layer()
    setting.gate_mode = "none"
    used = set()
    original = driver._run

    def record(part, *args, **kwargs):
        used.add(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    with torch.no_grad(), compute.eager(), ir.interpret_all():
        compute.run_layer(y, y, [setting], 3)
    assert {"entry", "select", "keep", "carry", "iterate"} <= used


def test_override_tracking_covers_in_place_edits_and_cleanup():
    import gc

    before = compute._IR_ACTIVE_OVERRIDES.get()
    _, setting = _layer()
    setting.ir_parts["keep"] = ir.PartOverride("default")
    setting.ir_parts.update({"gate": ir.PartOverride("adaptive")})
    setting.ir_parts.pop("keep")
    setting.ir_parts.clear()
    del setting
    gc.collect()
    assert before == compute._IR_ACTIVE_OVERRIDES.get()


def test_shared_routing_accepts_identical_select_overrides():
    import copy

    y, first = _layer()
    second = copy.deepcopy(first)
    first.gate_mode = second.gate_mode = "none"
    first.routing_select = second.routing_select = "top1"
    first.ir_parts = {"select": ir.PartOverride("token_top1", parameters={"one": 0.9})}
    second.ir_parts = {"select": ir.PartOverride("token_top1", parameters={"one": 0.9})}
    with torch.no_grad(), compute.eager():
        result = compute.run_layer(y, y, [first, second], 3, shared=True)
    assert result.shape == y.shape


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_interpret_all_adaptive_path_parity(limit):
    y, setting = _layer()
    setting.iteration = "path"
    setting.gate_head_limit = limit
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(y, y, [setting], 3, target="path-adaptive")
        with ir.interpret_all():
            result = compute.run_layer(y, y, [setting], 3, target="path-adaptive")
    assert torch.equal(result, native)


def test_imported_set_runs_keep_program(monkeypatch):
    from kilora.ir import driver

    y, setting = _layer()
    setting.form = "input"
    setting.gate_mode = "none"
    used = set()
    original = driver._run

    def record(part, *args, **kwargs):
        used.add(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    with torch.no_grad(), compute.eager(), ir.interpret_all():
        compute.run_layer(y, y, [setting], 3)
    assert "keep" in used


def test_adaptive_path_uses_gate_override(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    y, setting = _layer()
    setting.iteration = "path"
    setting.ir_parts = {"gate": ir.PartOverride("adaptive", parameters={"kappa": 0.2})}
    with torch.no_grad(), compute.eager():
        result = compute.run_layer(y, y, [setting], 3)
        setting.ir_parts = {}
        setting.gate_kappa = 0.2
        native = compute.run_layer(y, y, [setting], 3)
    assert torch.equal(result, native)


@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("selection", ["top1", "sparsemax"])
def test_interpret_all_noise_parity(iteration, selection):
    y, setting = _layer()
    setting.gate_mode = "none"
    setting.iteration = iteration
    setting.routing_select = selection
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        torch.manual_seed(919)
        native = compute.run_layer(y, y, [setting], 3, noise_scale=0.3)
        torch.manual_seed(919)
        with ir.interpret_all():
            result = compute.run_layer(y, y, [setting], 3, noise_scale=0.3)
    assert torch.equal(result, native)


def test_gate_parameter_override_refuses_fusion(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    y, setting = _layer()
    setting.ir_parts = {"gate": ir.PartOverride("adaptive", parameters={"kappa": 0.2})}
    assert not fused.fused_ok(y, [setting])
    assert fused.LAST_REASON[0] == "interpreted part gate"
    with torch.no_grad(), compute.eager():
        changed = compute.run_layer(y, y, [setting], 3)
        setting.ir_parts = {}
        setting.gate_kappa = 0.2
        expected = compute.run_layer(y, y, [setting], 3)
    assert torch.equal(changed, expected)


def test_cosine_transition_forced_interpretation_is_exact(monkeypatch):
    from kilora.ir import driver

    monkeypatch.delenv("KILORA_SHARE_TRANSITION_RESOLVENT", raising=False)
    monkeypatch.delenv("KILORA_SHARE_TRANSITION_KIND", raising=False)
    _, setting = _layer()
    setting.form = "input"
    seen = []
    original = driver._run

    def record(part, *args, **kwargs):
        seen.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    native = compute.share_transition(setting)
    with ir.interpret_all():
        result = compute.share_transition(setting)
    assert torch.equal(result, native)
    assert "transition" in seen


def test_derived_entry_from_input_uses_interpreter(monkeypatch):
    from kilora.ir import driver

    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    y, setting = _layer()
    setting.form = "input"
    seen = []
    original = driver._run

    def record(part, *args, **kwargs):
        seen.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    native = compute.derived_logits(setting, y)
    with ir.interpret_all():
        result = compute.derived_logits(setting, y)
    assert torch.equal(result, native)
    assert "entry" in seen


@pytest.mark.parametrize("values", [[0.1], [0.1, 0.2, 0.3], [0.73, 0.81, 0.32]])
def test_scalar_budget_interpreter_matches_python_exactly(values, monkeypatch):
    from kilora.ir import driver

    weights = {str(i): value for i, value in enumerate(values)}
    weights["functional"] = 0.6
    seen = []
    original = driver._run

    def record(part, *args, **kwargs):
        seen.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    native = compute.axis_budget(weights)
    with ir.interpret_all():
        actual = compute.axis_budget(weights)
    assert actual == native == weights
    assert "budget" in seen


def test_freeze_record_uses_program_and_matches_native(monkeypatch):
    from kilora import route_freeze
    from kilora.ir import driver

    y, setting = _layer()
    setting.gate_mode = "none"
    routes = [torch.softmax(torch.randn(2, 4, 2), dim=-1) for _ in range(3)]
    rec = {"routes": {"freeze": [[route] for route in routes]}, "gates": {}}
    args = (rec, [setting], [0], "freeze", [None], 2.5, tuple(y.shape[:-1]))
    native = route_freeze.build(*args)
    seen = []
    original = driver._run

    def record(part, *args, **kwargs):
        seen.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    with ir.interpret_all():
        actual = route_freeze.build(*args)
    assert torch.equal(actual["scale"], native["scale"])
    assert torch.equal(actual["up"], native["up"])
    assert "freeze" in seen


@pytest.mark.parametrize("gap,spread", [(0.9, 1.0), (0.7, 1.0), (1.0, 0.9)])
def test_guard_decision_uses_program(gap, spread, monkeypatch):
    from types import SimpleNamespace

    from kilora import editors
    from kilora.ir import driver

    before, after = {"gap": 1.0, "spread": 1.0}, {"gap": gap, "spread": spread}
    session = SimpleNamespace(state=None, guard=True, guard_gap_diffs=[])
    rule = SimpleNamespace(guard_trial=lambda state, new: object())
    record = SimpleNamespace(guard_score=lambda trial: before if trial is None else after)
    expected = editors.guard_check(rule, session, record, None)
    seen = []
    original = driver._run

    def observe(part, *args, **kwargs):
        seen.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", observe)
    with ir.interpret_all():
        actual = editors.guard_check(rule, session, record, None)
    assert actual == expected
    assert "guard" in seen


@pytest.mark.parametrize("capped", [False, True])
def test_editor_write_interprets_and_matches_native(monkeypatch, capped):
    import copy
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "plugins/kilora-editor-chunk/src")
    )
    from kilora_editor_chunk import Chunk, ChunkEditor, ChunkState, ChunkWriter

    from kilora.ir import defaults

    names = ["layers.0.attention.to_q"]
    shapes = {names[0]: [3, 2]}
    rows = torch.eye(6)
    editor = ChunkEditor(names, shapes, {"to_q": 1.0}, [Chunk("common", rows, torch.ones(6))])
    writer = ChunkWriter(
        editor,
        [0.7],
        [1.0],
        [0.1],
        write_norm=0.2,
        per_event=4,
        measure="full",
        decay=0.2,
        chunk_cap=0.05 if capped else None,
        total_cap=0.3 if capped else None,
    )
    native = ChunkState.new(editor)
    for value in native.coef.values():
        value.fill_(0.01)
    forced = copy.deepcopy(native)

    def loss(pert, t, seed):
        return sum(float((v - 0.2).square().sum()) for v in pert.values())

    seen = []
    original = defaults.interpret_live_write

    def observe(*args, **kwargs):
        seen.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(defaults, "interpret_live_write", observe)
    native.push()
    writer.step(native, loss, {names[0]: [0.4, 0.6]}, seed=1)
    forced.push()
    with ir.interpret_all():
        writer.step(forced, loss, {names[0]: [0.4, 0.6]}, seed=1)
    assert seen
    assert all(torch.equal(native.coef[name], forced.coef[name]) for name in native.coef)
