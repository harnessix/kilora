# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import pytest

from kilora.program import (
    ProgramError,
    Signature,
    ValueType,
    check_program_count,
    layer_change_signature,
    program_identity,
    validate,
)


def signed(p):
    p["identity"] = program_identity(p)
    return p


def lora(silu=False):
    ops = [
        ("matmul", ["x", "A"], "h", {"transpose_b": True}),
        *(([("silu", ["h"], "g", {})]) if silu else []),
        ("matmul", ["g" if silu else "h", "B"], "d", {"transpose_b": True}),
        ("scale", ["d"], "e", {"value": 1.0, "by_strength": True}),
        ("output_add", ["e"], "out", {}),
    ]
    return signed(
        {
            "name": "test",
            "op_set": "kilora.ops",
            "op_set_version": 1,
            "hooks": ["hook"],
            "inputs": {
                "x": {"axes": ["batch", "tokens", "in"]},
                "y": {"axes": ["batch", "tokens", "out"]},
            },
            "bindings": {
                "A": {"kind": "embedded", "dtype": "float32", "shape": [8, 16]},
                "B": {"kind": "embedded", "dtype": "float32", "shape": [32, 8]},
            },
            "instructions": [
                {"id": i, "op": op, "args": args, "result": result, "attributes": attrs}
                for i, (op, args, result, attrs) in enumerate(ops)
            ],
        }
    )


def selection():
    t = ValueType(("batch", "tokens", "experts"), "float32", (1, 4096, 4))
    sig = Signature({"logits": t}, {"weights": t})
    p = signed(
        {
            "name": "select",
            "op_set": "kilora.ops",
            "op_set_version": 1,
            "inputs": {"logits": {"axes": list(t.axes), "dtype": t.dtype}},
            "outputs": {"weights": {"axes": list(t.axes), "dtype": t.dtype}},
            "bindings": {},
            "instructions": [
                {
                    "id": 0,
                    "op": "scale",
                    "args": ["logits"],
                    "result": "scaled",
                    "attributes": {"value": 1, "by_strength": False},
                },
                {"id": 1, "op": "relu", "args": ["scaled"], "result": "weights", "attributes": {}},
            ],
        }
    )
    return p, sig


@pytest.mark.parametrize("silu", [False, True])
def test_lora(silu):
    p = lora(silu)
    result = validate(p, layer_change_signature((16, 32)))
    assert result.identity == p["identity"]
    assert result.value_types["out"].shape == (1, 4096, 32)


def test_selection_and_outputs():
    p, sig = selection()
    validate(p, sig)
    p["outputs"] = {}
    with pytest.raises(ProgramError, match="outputs"):
        validate(p, sig)
    p, sig = selection()
    p["outputs"]["weights"]["axes"] = ["batch", "tokens", "cols"]
    with pytest.raises(ProgramError, match="output weights"):
        validate(p, sig)
    p, sig = selection()
    p["instructions"][-1]["result"] = "other"
    with pytest.raises(ProgramError, match="output weights"):
        validate(p, sig)


@pytest.mark.parametrize(
    "change, error",
    [
        (lambda p: p["instructions"][0].update(op="bad"), "unknown op"),
        (lambda p: p["instructions"][0].update(args=["x"]), "arity"),
        (lambda p: p["instructions"][0].update(attributes={}), "attributes"),
        (lambda p: p["instructions"][2]["attributes"].update(value=65), "attribute"),
        (lambda p: p["instructions"][0].update(id=3), "ids"),
        (lambda p: p["instructions"][0].update(result="x"), "SSA"),
        (lambda p: p["instructions"][0].update(args=["h", "A"]), "SSA"),
        (lambda p: p["bindings"]["A"].update(shape=[8, 15]), "shape mismatch"),
        (lambda p: p["bindings"]["A"].update(kind="derived"), "derived"),
        (lambda p: p.update(identity="wrong"), "identity"),
        (lambda p: p["instructions"].pop(), "output_add"),
    ],
)
def test_rejections(change, error):
    p = lora()
    change(p)
    with pytest.raises(ProgramError, match=error) as exc:
        validate(p, layer_change_signature((16, 32)))
    assert "program test" in str(exc.value)
    assert "instruction" in str(exc.value) and "op " in str(exc.value)


def test_unknown_axis_and_duplicate_output():
    p, sig = selection()
    t = ValueType(("batch", "tokens", "unknown"), shape=(1, 4096, 4))
    with pytest.raises(ProgramError, match="axis"):
        validate(p, Signature({"logits": t}, {"weights": t}))
    p = lora()
    p["instructions"].append(
        {"id": 4, "op": "output_add", "args": ["e"], "result": "out2", "attributes": {}}
    )
    with pytest.raises(ProgramError, match="exactly one"):
        validate(p, layer_change_signature((16, 32)))


def test_preflight_and_instruction_limit():
    p, sig = selection()
    t = ValueType(("batch", "tokens", "experts"), shape=(1, 4096, 16384))
    with pytest.raises(ProgramError, match="working-bytes"):
        validate(p, Signature({"logits": t}, {"weights": t}))
    p["instructions"] = [{}] * 10000
    with pytest.raises(ProgramError, match="limit 96"):
        validate(p, sig)


def test_identity_and_count():
    p = lora()
    reordered = dict(reversed(list(deepcopy(p).items())))
    assert program_identity(p) == program_identity(reordered)
    check_program_count(16)
    with pytest.raises(ProgramError, match="count"):
        check_program_count(17)


def test_rank_can_equal_output_width():
    p = lora(silu=True)
    p["bindings"]["A"]["shape"] = [32, 16]
    p["bindings"]["B"]["shape"] = [32, 32]
    signed(p)
    result = validate(p, layer_change_signature((16, 32)))
    assert result.value_types["h"].axes == ("batch", "tokens", "rank")


def test_malformed_binding_reports_program_error():
    p = lora()
    p["bindings"]["A"]["dtype"] = []
    with pytest.raises(ProgramError, match="program test.*invalid program data"):
        validate(p, layer_change_signature((16, 32)))
