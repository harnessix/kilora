# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_ir_dispatch import _layer
from test_modfile import trained

from kilora import Settings, compute, load_mod, save_mod
from kilora.ir.dispatch import active_overrides, override_scope
from kilora.ir.driver import _run
from kilora.program import program_identity
from kilora.sections import Section


def select_program(op="softmax"):
    axes = ["batch", "tokens", "experts"]
    typ = {"axes": axes, "dtype": "float32"}
    p = {
        "part": "select",
        "scope": "*",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"logits": typ, "noise": typ},
        "outputs": {"weights": typ},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": op,
                "args": ["logits"],
                "result": "weights",
                "attributes": {"axis": "experts"}
                if op == "softmax"
                else {"value": 0.0, "by_strength": False},
            }
        ],
    }
    p["identity"] = program_identity(p)
    return Section("program", "select", {}, p)


def load_with_section(model, tmp_path, section):
    adapter = trained(model)
    sets = adapter.to_sets()
    adapter.remove()
    path = tmp_path / "mod.kmod"
    save_mod(path, sets, Settings(), sections=[section], id="test")
    return load_mod(path)


def test_select_load_and_run(model, tmp_path):
    mod = load_with_section(model, tmp_path, select_program())
    override = mod.sets[0].ir_parts["select"]
    inputs = {"logits": torch.randn(2, 3, 4), "noise": torch.zeros(2, 3, 4)}
    with active_overrides({"select": override}):
        weights = _run("select", "token_sparsemax", inputs)["weights"]
    assert torch.equal(weights, torch.softmax(inputs["logits"], -1))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))


def test_bad_simplex_refused_at_load(model, tmp_path):
    with pytest.raises(ValueError, match="part select: simplex weights"):
        load_with_section(model, tmp_path, select_program("scale"))


def test_first_call_checks_invariants(model, tmp_path):
    section = select_program()
    section.meta["instructions"][0]["args"] = ["noise"]
    section.meta["identity"] = program_identity(section.meta)
    mod = load_with_section(model, tmp_path, section)
    inputs = {"logits": torch.zeros(2, 3, 4), "noise": torch.full((2, 3, 4), float("nan"))}
    with (
        active_overrides({"select": mod.sets[0].ir_parts["select"]}),
        pytest.raises(ValueError, match="part select: simplex weights"),
    ):
        _run("select", "token_sparsemax", inputs)


def test_gate_kappa_matches_native(model, tmp_path):
    mod = load_with_section(
        model,
        tmp_path,
        Section("params", "gate", {}, {"part": "gate", "scope": "*", "params": {"kappa": 0.2}}),
    )
    y, setting = _layer()
    setting.ir_parts = mod.sets[0].ir_parts
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        with override_scope():
            actual = compute.run_layer(y, y, [setting], 3)
        setting.ir_parts = {}
        setting.gate_kappa = 0.2
        expected = compute.run_layer(y, y, [setting], 3)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "meta, message",
    [
        ({"part": "invalid", "scope": "*", "params": {}}, "unknown part"),
        ({"part": "gate", "scope": "*", "params": {"oops": 1}}, "unknown parameter"),
        ({"part": "gate", "scope": "*", "params": {"kappa": -1}}, "kappa"),
        ({"part": "guard", "scope": "*", "params": {}}, "guard.*not yet supported"),
    ],
)
def test_bad_params_refused(model, tmp_path, meta, message):
    with pytest.raises(ValueError, match=message):
        load_with_section(model, tmp_path, Section("params", "params", {}, meta))


def test_select_replacement_supports_tensor_and_top1(model, tmp_path):
    mod = load_with_section(model, tmp_path, select_program())
    override = mod.sets[0].ir_parts["select"]
    logits = torch.randn(2, 3, 4)
    inputs = {
        "logits": logits,
        "noise": torch.zeros(2, 1, 4),
        "region": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        "has_region": torch.tensor(True),
    }
    with active_overrides({"select": override}):
        weights = _run("select", "tensor_top1", inputs)["weights"]
    mean = compute.region_mean(logits, inputs["region"].unsqueeze(-1))
    torch.testing.assert_close(weights, mean.softmax(-1).expand_as(logits))


def test_replacement_embedded_binding(model, tmp_path):
    section = select_program()
    section.meta["bindings"] = {
        "temperature": {"kind": "embedded", "dtype": "float32", "shape": [], "axes": []}
    }
    section.data["temperature"] = torch.tensor(2.0)
    section.meta["instructions"] = [
        {
            "id": 0,
            "op": "div",
            "args": ["logits", "temperature"],
            "result": "scaled",
            "attributes": {},
        },
        {
            "id": 1,
            "op": "softmax",
            "args": ["scaled"],
            "result": "weights",
            "attributes": {"axis": "experts"},
        },
    ]
    section.meta["identity"] = program_identity(section.meta)
    mod = load_with_section(model, tmp_path, section)
    inputs = {"logits": torch.randn(2, 3, 4), "noise": torch.zeros(2, 3, 4)}
    with active_overrides({"select": mod.sets[0].ir_parts["select"]}):
        result = _run("select", "token_sparsemax", inputs)["weights"]
    assert torch.equal(result, torch.softmax(inputs["logits"] / 2, -1))


def test_scope_is_preserved_through_adapter(model, tmp_path):
    from kilora import compose

    mod = load_with_section(model, tmp_path, select_program())
    adapter = compose(model, mod.sets)
    layer = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))[0]
    assert (
        layer.ir_parts["select"].program.identity == mod.sets[0].ir_parts["select"].program.identity
    )
    adapter.remove()


def test_keep_delta_matches_reference(model, tmp_path):
    from kilora.ir.driver import _read_mask

    section = Section(
        "params", "keep", {}, {"part": "keep", "scope": "*", "params": {"DROPOUT_DELTA": 0.2}}
    )
    mod = load_with_section(model, tmp_path, section)
    _, setting = _layer()
    setting.ir_parts = mod.sets[0].ir_parts
    inputs = {
        "change": torch.rand(2, 3, 2, 4),
        "base_read": torch.rand(2, 3, 2, 4),
        "uniforms": torch.rand(2, 3, 2, 4),
    }
    with active_overrides({}, [setting]):
        actual = _read_mask("mask_expect", inputs, setting)
    scale = inputs["base_read"].pow(2).mean(-1, keepdim=True).sqrt() + 0.2
    expected = compute.keep_probability(inputs["change"], scale)
    assert torch.equal(actual, expected)


def test_carry_replacement_uses_part_signature(model, tmp_path):
    axes = ["batch", "tokens", "cols"]
    typ = {"axes": axes, "dtype": "float32"}
    program = {
        "part": "carry",
        "scope": "*",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {name: typ for name in ("change", "carried", "mask", "base_read", "uniforms")},
        "outputs": {"state": typ},
        "bindings": {},
        "instructions": [
            {"id": 0, "op": "mul", "args": ["change", "mask"], "result": "state", "attributes": {}}
        ],
    }
    program["identity"] = program_identity(program)
    mod = load_with_section(model, tmp_path, Section("program", "carry", {}, program))
    _, setting = _layer()
    setting.ir_parts = mod.sets[0].ir_parts
    inputs = {name: torch.rand(2, 3, 4) for name in program["inputs"]}
    with active_overrides({}, [setting]):
        result = _run("carry", "dropped", inputs, _setting=setting)["state"]
    assert torch.equal(result, inputs["change"] * inputs["mask"])


def test_score_replacement_uses_part_signature(model, tmp_path):
    typ = lambda axes: {"axes": axes, "dtype": "float32"}
    program = {
        "part": "score",
        "scope": "*",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {
            "query": typ(["batch", "tokens", "key"]),
            "keys": typ(["experts", "per", "key"]),
            "temperature": typ([]),
        },
        "outputs": {"logits": typ(["batch", "tokens", "experts"])},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "mean",
                "args": ["keys"],
                "result": "average",
                "attributes": {"axis": "per", "keep": False},
            },
            {
                "id": 1,
                "op": "matmul",
                "args": ["query", "average"],
                "result": "scores",
                "attributes": {"transpose_b": True},
            },
            {
                "id": 2,
                "op": "mul",
                "args": ["scores", "temperature"],
                "result": "logits",
                "attributes": {},
            },
        ],
    }
    program["identity"] = program_identity(program)
    mod = load_with_section(model, tmp_path, Section("program", "score", {}, program))
    _, setting = _layer()
    setting.ir_parts = mod.sets[0].ir_parts
    query, keys = torch.randn(2, 3, 5), torch.randn(4, 2, 5)
    inputs = {
        "query": query,
        "keys": keys,
        "temperature": torch.tensor(2.0),
        "rows": torch.ones(4),
        "key_width": torch.tensor(5.0),
    }
    with active_overrides({}, [setting]):
        result = _run("score", "keys", inputs, _setting=setting)["logits"]
    assert torch.equal(result, (query @ keys.mean(1).t()) * 2)


def test_shared_selection_applies_to_all_sets(model, tmp_path):
    import copy

    mod = load_with_section(model, tmp_path, select_program())
    _, first = _layer()
    second = copy.deepcopy(first)
    first.ir_parts = dict(mod.sets[0].ir_parts)
    second.ir_parts = dict(mod.sets[0].ir_parts)
    inputs = {"logits": torch.randn(2, 3, 4), "noise": torch.zeros(2, 3, 4)}
    with active_overrides({}, [first, second], shared=True):
        weights = _run("select", "token_sparsemax", inputs)["weights"]
    assert torch.equal(weights, inputs["logits"].softmax(-1))


def test_baked_plan_keeps_part_overrides(model, tmp_path):
    from kilora.bake import ScopedSet, plan_targets

    mod = load_with_section(model, tmp_path, select_program())
    plans = plan_targets([ScopedSet(mod.sets[0])], depth=2, merge_plain=True)
    for plan in plans.values():
        assert "select" in plan.sets[0].ir_parts
        moved = plan.to(torch.device("cpu"), torch.float32)
        assert "select" in moved.sets[0].ir_parts


def test_baked_model_executes_select_override(model, x, tmp_path):
    from kilora.bake import ScopedSet, bake

    mod = load_with_section(model, tmp_path, select_program())
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        modified = bake(model, [ScopedSet(mod.sets[0], scope="per_set", gate_mode="none")], depth=2)
        actual = model(x)
        modified.remove()
        plain_set = mod.sets[0]
        plain_set.ir_parts = {}
        baseline = bake(model, [ScopedSet(plain_set, scope="per_set", gate_mode="none")], depth=2)
        expected = model(x)
        baseline.remove()
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, expected)


def test_gate_kappa_with_token_depth(model, tmp_path):
    section = Section(
        "params", "gate", {}, {"part": "gate", "scope": "*", "params": {"kappa": 0.2}}
    )
    mod = load_with_section(model, tmp_path, section)
    y, setting = _layer()
    setting.ir_parts = mod.sets[0].ir_parts
    depths = torch.tensor([[1, 2, 3, 2], [2, 3, 1, 2]])
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        with override_scope():
            actual = compute.run_layer(y, y, [setting], 3, token_depth=depths)
        setting.ir_parts = {}
        setting.gate_kappa = 0.2
        expected = compute.run_layer(y, y, [setting], 3, token_depth=depths)
    assert torch.equal(actual, expected)
