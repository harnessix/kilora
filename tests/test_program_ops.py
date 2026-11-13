# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from pathlib import Path

import pytest

from kilora.errors import KiLoRAError
from kilora.program import op_set_digest, op_table, register_ops


def test_op_set_lock():
    table = op_table("kilora.ops", 1)
    assert op_set_digest(table) == (Path(__file__).parent / "data/op_set.lock").read_text().strip()
    changed = deepcopy(table)
    changed["scale"]["attributes"]["value"]["max"] = 63
    assert op_set_digest(changed) != op_set_digest(table)


def test_core_op_schemas():
    table = op_table("kilora.ops", 1)
    groups = {
        1: "scale silu gelu tanh sigmoid relu clamp mask output_add abs exp log sqrt neg "
        "sum mean max min softmax sort cumsum argmax one_hot cast transpose rename_axis unsqueeze "
        "clamp_min select detach vector_norm sum_pair flatten unflatten svd_values svd_vh narrow "
        "pow_int arange_axis repeat_interleave",
        2: "matmul add mul sub div pow lt le gt ge eq gather batch_matmul key_projection",
        3: "where",
        4: "stack",
        -1: "stack_rows",
    }
    assert set(table) == {name for names in groups.values() for name in names.split()}
    for arity, names in groups.items():
        for name in names.split():
            assert table[name]["name"] == name
            assert table[name]["arity"] == arity
            assert table[name]["since_version"] == 1
            assert table[name]["type_rule"]
            assert all(attr["required"] for attr in table[name]["attributes"].values())
    attrs = {name: set(op["attributes"]) for name, op in table.items()}
    expected = {
        "matmul": {"transpose_b"},
        "scale": {"value", "by_strength"},
        "gelu": {"approximate"},
        "clamp": {"low", "high"},
        "mask": {"source"},
        "narrow": {"axis", "new_axis", "size_of"},
        "svd_vh": {"axis"},
        "svd_values": {"axis"},
        "unflatten": {"axis", "axis1", "axis2", "size_of"},
        "flatten": {"axis1", "axis2", "new_axis"},
        "sum_pair": {"axis1", "axis2"},
        "vector_norm": {"axis", "keep"},
        "sum": {"axis", "keep"},
        "mean": {"axis", "keep"},
        "max": {"axis", "keep"},
        "min": {"axis", "keep"},
        "softmax": {"axis"},
        "sort": {"axis", "descending"},
        "cumsum": {"axis"},
        "argmax": {"axis"},
        "one_hot": {"axis", "size_of"},
        "gather": {"axis"},
        "cast": {"dtype"},
        "transpose": {"axis1", "axis2"},
        "rename_axis": {"axis", "new_axis"},
        "unsqueeze": {"axis", "before", "after"},
        "clamp_min": {"low"},
        "stack": {"axis", "before"},
        "stack_rows": {"axis", "before"},
        "select": {"axis", "index"},
        "pow_int": {"exponent"},
        "arange_axis": {"axis"},
        "repeat_interleave": {"axis", "new_axis", "repeats"},
    }
    assert attrs == {name: expected.get(name, set()) for name in table}
    assert table["gelu"]["attributes"]["approximate"]["values"] == ["none", "tanh"]
    assert table["scale"]["attributes"]["value"]["min"] == -64
    assert table["clamp"]["attributes"]["high"]["greater_than_attribute"] == "low"


def test_plugin_registration_and_errors():
    ops = {
        "identity": {
            "name": "identity",
            "arity": 1,
            "attributes": {},
            "type_rule": "same",
            "since_version": 1,
        }
    }
    register_ops("test_plugin.ops", 1, ops)
    assert op_table("test_plugin.ops", 1) == ops
    ops["identity"]["arity"] = 9
    returned = op_table("test_plugin.ops", 1)
    assert returned["identity"]["arity"] == 1
    returned.clear()
    assert op_table("test_plugin.ops", 1)
    with pytest.raises(KiLoRAError, match=r"other\.ops"):
        op_table("other.ops", 1)
    with pytest.raises(KiLoRAError, match="update KiLoRA"):
        op_table("kilora.ops", 2)
    with pytest.raises(KiLoRAError, match=r"cannot replace.*kilora\.ops"):
        register_ops("kilora.ops", 1, {})
    with pytest.raises(KiLoRAError, match="already registered"):
        register_ops("test_plugin.ops", 1, {})


def test_stack_and_select_validate_and_execute():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    value = ValueType(("batch", "cols"), shape=(2, 3))
    signature = Signature({name: value for name in ("a", "b", "c", "d")}, {"out": value})
    program = {
        "name": "stack_select",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {name: {"axes": list(value.axes)} for name in signature.inputs},
        "outputs": {"out": {"axes": list(value.axes)}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "stack",
                "args": ["a", "b", "c", "d"],
                "result": "stacked",
                "attributes": {"axis": "r3", "before": "cols"},
            },
            {
                "id": 1,
                "op": "select",
                "args": ["stacked"],
                "result": "out",
                "attributes": {"axis": "r3", "index": 2},
            },
        ],
    }
    program["identity"] = program_identity(program)
    inputs = {name: torch.full((2, 3), float(i)) for i, name in enumerate(signature.inputs)}
    assert torch.equal(interpret(validate(program, signature), inputs, {})["out"], inputs["c"])

    for attrs, error in [
        ({"axis": "r3", "index": 4}, "out of bounds"),
        ({"axis": "r2", "index": 0}, "axis absent"),
        ({"axis": "r3", "index": True}, "attribute index"),
    ]:
        bad = deepcopy(program)
        bad["instructions"][1]["attributes"] = attrs
        with pytest.raises(ProgramError, match=error):
            validate(bad, signature)
    bad = deepcopy(program)
    bad["instructions"][0]["attributes"]["axis"] = "cols"
    with pytest.raises(ProgramError, match="invalid stack axis"):
        validate(bad, signature)
    bad = deepcopy(program)
    bad["inputs"]["d"]["dtype"] = "bool"
    bad_signature = Signature(
        {**signature.inputs, "d": ValueType(("batch", "cols"), "bool", (2, 3))},
        signature.outputs,
    )
    with pytest.raises(ProgramError, match="stack input types or shapes differ"):
        validate(bad, bad_signature)


def test_vector_norm_matches_torch_and_validates():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {"x": ValueType(("batch", "tokens", "key"), shape=(2, 3, 7))},
        {"out": ValueType(("batch", "tokens"), shape=(2, 3))},
    )
    program = {
        "name": "norm",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["batch", "tokens", "key"]}},
        "outputs": {"out": {"axes": ["batch", "tokens"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "vector_norm",
                "args": ["x"],
                "result": "out",
                "attributes": {"axis": "key", "keep": False},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(2, 3, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["out"],
        torch.linalg.vector_norm(x, dim=-1, keepdim=False, dtype=torch.float32),
    )
    program["instructions"][0]["attributes"]["axis"] = "per"
    with pytest.raises(ProgramError, match="axis is absent"):
        validate(program, signature)


def test_sum_pair_matches_torch_and_rejects_bad_axes():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {"x": ValueType(("experts", "per", "key"), shape=(3, 4, 7))},
        {"out": ValueType(("experts",), shape=(3,))},
    )
    program = {
        "name": "sum_pair",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["experts", "per", "key"]}},
        "outputs": {"out": {"axes": ["experts"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "sum_pair",
                "args": ["x"],
                "result": "out",
                "attributes": {"axis1": "per", "axis2": "key"},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(3, 4, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["out"], x.sum((-2, -1))
    )
    program["instructions"][0]["attributes"]["axis2"] = "per"
    with pytest.raises(ProgramError, match="two distinct input axes"):
        validate(program, signature)


def test_flatten_matches_torch_and_rejects_nonadjacent_axes():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {"x": ValueType(("experts", "per", "key"), shape=(3, 4, 7))},
        {"out": ValueType(("cols", "key"), shape=(12, 7))},
    )
    program = {
        "name": "flatten",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["experts", "per", "key"]}},
        "outputs": {"out": {"axes": ["cols", "key"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "flatten",
                "args": ["x"],
                "result": "out",
                "attributes": {"axis1": "experts", "axis2": "per", "new_axis": "cols"},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(3, 4, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["out"], x.flatten(0, 1)
    )
    program["instructions"][0]["attributes"]["axis2"] = "key"
    with pytest.raises(ProgramError, match="adjacent axes"):
        validate(program, signature)


def test_unflatten_matches_torch_and_rejects_invalid_size():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {
            "x": ValueType(("batch", "cols"), shape=(2, 12)),
            "sizes": ValueType(("experts", "per"), shape=(3, 4)),
        },
        {"out": ValueType(("batch", "experts", "per"), shape=(2, 3, 4))},
    )
    program = {
        "name": "unflatten",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["batch", "cols"]}, "sizes": {"axes": ["experts", "per"]}},
        "outputs": {"out": {"axes": ["batch", "experts", "per"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "unflatten",
                "args": ["x"],
                "result": "out",
                "attributes": {
                    "axis": "cols",
                    "axis1": "experts",
                    "axis2": "per",
                    "size_of": "sizes",
                },
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(2, 12)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x, "sizes": torch.zeros(3, 4)}, {})["out"],
        x.unflatten(-1, (3, 4)),
    )
    program["instructions"][0]["attributes"]["axis2"] = "experts"
    with pytest.raises(ProgramError, match="invalid unflatten axes"):
        validate(program, signature)


def test_svd_values_matches_torch():
    import torch

    from kilora.program import Signature, ValueType, interpret, program_identity, validate

    signature = Signature(
        {"x": ValueType(("per", "key"), shape=(4, 7))}, {"out": ValueType(("r1",), shape=(4,))}
    )
    program = {
        "name": "svd_values",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["per", "key"]}},
        "outputs": {"out": {"axes": ["r1"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "svd_values",
                "args": ["x"],
                "result": "out",
                "attributes": {"axis": "r1"},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(4, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["out"],
        torch.linalg.svd(x, full_matrices=False).S,
    )


def test_svd_vh_matches_torch():
    import torch

    from kilora.program import Signature, ValueType, interpret, program_identity, validate

    signature = Signature(
        {"x": ValueType(("per", "key"), shape=(4, 7))},
        {"out": ValueType(("r1", "key"), shape=(4, 7))},
    )
    program = {
        "name": "svd_vh",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["per", "key"]}},
        "outputs": {"out": {"axes": ["r1", "key"]}},
        "bindings": {},
        "instructions": [
            {"id": 0, "op": "svd_vh", "args": ["x"], "result": "out", "attributes": {"axis": "r1"}}
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(4, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["out"],
        torch.linalg.svd(x, full_matrices=False).Vh,
    )


def test_narrow_matches_torch_and_rejects_large_size():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {"x": ValueType(("r1", "key"), shape=(4, 7)), "sizes": ValueType(("r2",), shape=(3,))},
        {"out": ValueType(("r2", "key"), shape=(3, 7))},
    )
    program = {
        "name": "narrow",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["r1", "key"]}, "sizes": {"axes": ["r2"]}},
        "outputs": {"out": {"axes": ["r2", "key"]}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "narrow",
                "args": ["x"],
                "result": "out",
                "attributes": {"axis": "r1", "new_axis": "r2", "size_of": "sizes"},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.randn(4, 7)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x, "sizes": torch.zeros(3)}, {})["out"], x[:3]
    )
    too_large = Signature(
        {"x": signature.inputs["x"], "sizes": ValueType(("r2",), shape=(5,))},
        {"out": ValueType(("r2", "key"), shape=(5, 7))},
    )
    with pytest.raises(ProgramError, match="narrow exceeds"):
        validate(program, too_large)


def test_repeat_interleave_validation_and_execution():
    import torch

    from kilora.program import (
        ProgramError,
        Signature,
        ValueType,
        interpret,
        program_identity,
        validate,
    )

    signature = Signature(
        {"x": ValueType(("batch", "experts"), shape=(2, 3))},
        {"y": ValueType(("batch", "cols"), shape=(2, 6))},
    )
    program = {
        "name": "repeat",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": ["batch", "experts"], "dtype": "float32"}},
        "outputs": {"y": {"axes": ["batch", "cols"], "dtype": "float32"}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "repeat_interleave",
                "args": ["x"],
                "result": "y",
                "attributes": {"axis": "experts", "new_axis": "cols", "repeats": 2},
            }
        ],
    }
    program["identity"] = program_identity(program)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.equal(
        interpret(validate(program, signature), {"x": x}, {})["y"],
        torch.repeat_interleave(x, 2, dim=-1),
    )
    bad = deepcopy(program)
    bad["instructions"][0]["attributes"]["repeats"] = 3
    with pytest.raises(ProgramError, match="repeat axis shape mismatch"):
        validate(bad, signature)
