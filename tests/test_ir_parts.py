# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import adaptive_gate, compute
from kilora.ir import PARTS, check_invariants, probe_inputs
from kilora.program import ProgramError, program_identity, validate

SIZES = {"batch": 2, "tokens": 3, "experts": 4, "per": 2, "cols": 5, "key": 6, "out": 7, "axes": 3}


def outputs_for(part):
    result = {}
    for name, typ in part.sized_signature(SIZES).outputs.items():
        result[name] = torch.zeros(typ.shape, dtype=getattr(torch, typ.dtype))
    if "weights" in result:
        result["weights"].fill_(1 / SIZES["experts"])
    if "probability" in result:
        result["probability"].fill_(0.5)
    if "temperature" in result:
        result["temperature"].fill_(1)
    return result


@pytest.mark.parametrize("name", list(PARTS))
def test_signature_validates(name):
    part = PARTS[name]
    signature = part.sized_signature(SIZES)
    bindings, instructions = {}, []
    for index, (output, value) in enumerate(signature.outputs.items()):
        binding = f"constant_{index}"
        bindings[binding] = {
            "kind": "embedded",
            "dtype": "float32",
            "shape": list(value.shape),
            "axes": list(value.axes),
        }
        if value.dtype == "bool":
            op, args, attrs = "eq", [binding, binding], {}
        else:
            op, args, attrs = "scale", [binding], {"value": 1.0, "by_strength": False}
        instructions.append(
            {"id": index, "op": op, "args": args, "result": output, "attributes": attrs}
        )
    program = {
        "name": name,
        "part": name,
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {
            n: {"axes": list(v.axes), "dtype": v.dtype} for n, v in signature.inputs.items()
        },
        "outputs": {
            n: {"axes": list(v.axes), "dtype": v.dtype} for n, v in signature.outputs.items()
        },
        "bindings": bindings,
        "instructions": instructions,
    }
    program["identity"] = program_identity(program)
    validate(program, signature)


def test_part_registry():
    assert set(PARTS) == {
        "score",
        "select",
        "keep",
        "carry",
        "gate",
        "transition",
        "entry",
        "budget",
        "freeze",
        "iterate",
        "live_write",
        "guard",
    }


@pytest.mark.parametrize("name", list(PARTS))
def test_probe_deterministic_and_output_shapes(name):
    part = PARTS[name]
    a, b = probe_inputs(part, SIZES, 17), probe_inputs(name, SIZES, 17)
    assert all(torch.equal(a[k], b[k]) for k in a)
    result = outputs_for(part)
    check_invariants(part, result, sizes=SIZES)
    key = next(iter(result))
    result[key] = torch.zeros(2, 2, 2, 2, 2)
    with pytest.raises(ProgramError, match=f"{name}.*output shapes"):
        check_invariants(part, result, sizes=SIZES)


@pytest.mark.parametrize(
    ("name", "key", "bad", "message"),
    [
        ("score", "logits", float("nan"), "finite logits"),
        ("select", "weights", -0.1, "simplex weights"),
        ("select", "weights", 0.1, "simplex weights"),
        ("keep", "probability", 0.0, "keep probability"),
        ("keep", "probability", 1.1, "keep probability"),
        ("gate", "token", -0.1, "gate parts"),
        ("gate", "channel", 1.1, "gate parts"),
        ("transition", "weights", 0.0, "simplex weights"),
        ("entry", "logits", float("inf"), "finite logits"),
        ("entry", "temperature", 0.0, "temperature"),
        ("budget", "scales", -0.1, "axis weights"),
        ("budget", "scales", float("inf"), "axis weights"),
        ("freeze", "weights", 0.0, "simplex weights"),
        ("live_write", "update", float("nan"), "finite update"),
    ],
)
def test_bad_values(name, key, bad, message):
    result = outputs_for(PARTS[name])
    result[key].fill_(bad)
    with pytest.raises(ProgramError, match=f"{name}.*{message}"):
        check_invariants(name, result)


def test_native_probe_outputs():
    score = probe_inputs("score", SIZES, 12)
    logits = compute.router_logits(score["query"], score["keys"], score["temperature"])
    check_invariants("score", {"logits": logits}, sizes=SIZES)
    sel = probe_inputs("select", SIZES, 12)
    check_invariants("select", {"weights": compute.sparsemax(sel["logits"])}, sizes=SIZES)
    keep = probe_inputs("keep", SIZES, 12)
    check_invariants(
        "keep",
        {"probability": compute.keep_probability(keep["carried"], keep["base_scale"])},
        sizes=SIZES,
    )
    gate = probe_inputs("gate", SIZES, 12)
    token, channel, history = adaptive_gate.adaptive_keep(
        gate["coefficients"],
        gate["before"],
        gate["prior_history"],
        gate["up"],
        gate["base_norm"],
        gate["base_rms"],
    )
    check_invariants("gate", {"token": token, "channel": channel, "history": history}, sizes=SIZES)
    budget = probe_inputs("budget", SIZES, 12)["weights"].abs()
    scales = compute.axis_budget({str(i): budget[..., i] for i in range(budget.shape[-1])})
    check_invariants("budget", {"scales": torch.stack(list(scales.values()), dim=-1)}, sizes=SIZES)


def test_native_carry_and_entry():
    sizes = {**SIZES, "cols": SIZES["experts"] * SIZES["per"]}
    inputs = probe_inputs("carry", sizes, 31)
    state = compute.carry_transfer(inputs["change"], inputs["base_read"], sizes["per"])
    check_invariants("carry", {"state": state}, sizes=sizes)

    layer = compute.LayerSet(
        router=torch.eye(2),
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(2),
        up=torch.eye(2),
        per_expert=1,
        form="input",
        import_score="derived",
        tau=1.0,
    )
    inputs = probe_inputs("entry", {"batch": 2, "tokens": 3, "experts": 2, "per": 2}, 31)
    query = inputs["reads"][..., 0, :]
    logits = compute.derived_logits(layer, query)
    check_invariants(
        "entry",
        {"logits": logits, "temperature": inputs["temperature_in"]},
        sizes={"batch": 2, "tokens": 3, "experts": 2},
    )


def test_native_transition_and_freeze():
    from types import SimpleNamespace

    from kilora import route_freeze

    layer = compute.LayerSet(
        router=torch.eye(2),
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(2),
        up=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        per_expert=1,
        form="input",
        import_score="read_share",
        tau=1.0,
    )
    sizes = {"batch": 2, "tokens": 3, "experts": 2}
    inputs = probe_inputs("transition", sizes, 33)
    matrix = compute.share_transition(layer)
    check_invariants("transition", {"weights": inputs["last_weights"] @ matrix}, sizes=sizes)
    freeze = probe_inputs("freeze", sizes, 33)
    with route_freeze.freezing({}):
        frozen = route_freeze.active([SimpleNamespace(route_freeze=0.5)], 0.75)
    check_invariants(
        "freeze",
        {
            "frozen": torch.tensor(frozen),
            "weights": freeze["last_weights"].mean(-2),
            "scales": torch.zeros(2, 1),
            "up_gated": torch.zeros(2, 1, 1),
        },
        sizes=sizes,
    )


def test_native_guard():
    from types import SimpleNamespace

    from kilora import editors

    inputs = probe_inputs("guard", {}, 49)
    before = {"gap": float(inputs["gap_before"]), "spread": float(inputs["spread_before"])}
    after = {"gap": float(inputs["gap_after"]), "spread": float(inputs["spread_after"])}
    record = SimpleNamespace(guard_score=lambda trial: before if trial is None else after)
    rule = SimpleNamespace(guard_trial=lambda state, new: object())
    refused, _ = editors.guard_check(rule, SimpleNamespace(state=None, guard=True), record, None)
    check_invariants("guard", {"refused": torch.tensor(refused)})
