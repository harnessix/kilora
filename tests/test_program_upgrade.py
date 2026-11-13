# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import pytest
import torch

import kilora.program as program_module
from kilora.errors import KiLoRAError
from kilora.program import (
    LOWERINGS,
    ProgramError,
    Signature,
    ValueType,
    interpret,
    program_identity,
    register_upgrade,
    upgrade_program,
    validate,
)


def sample():
    value_type = ValueType(("batch", "tokens", "experts"), shape=(1, 4096, 4))
    data = {
        "name": "example",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": list(value_type.axes), "dtype": "float32"}},
        "outputs": {"out": {"axes": list(value_type.axes), "dtype": "float32"}},
        "bindings": {},
        "instructions": [
            {
                "id": 0,
                "op": "scale",
                "args": ["x"],
                "result": "out",
                "attributes": {"value": 2.0, "by_strength": False},
            }
        ],
    }
    data["identity"] = program_identity(data)
    return data, Signature({"x": value_type}, {"out": value_type})


def test_upgrade_validates_copy_and_keeps_saved_section(monkeypatch):
    saved, signature = sample()
    saved["instructions"][0]["result"] = "scaled"
    saved["instructions"][0]["attributes"]["by_strength"] = True
    saved["instructions"].append(
        {"id": 1, "op": "relu", "args": ["scaled"], "result": "out", "attributes": {}}
    )
    saved["identity"] = program_identity(saved)
    original = deepcopy(saved)
    x = torch.randn(1, 3, 4)
    context = {"strength": 0.7, "timeline": 0.6}
    expected = interpret(validate(saved, signature), {"x": x}, {}, context)["out"]
    table = program_module.op_table("kilora.ops", 1)
    attrs = table["scale"]["attributes"]
    attrs["use_strength"] = attrs.pop("by_strength")
    table["scale"]["since_version"] = 2
    monkeypatch.setitem(program_module._TABLES["kilora.ops"], 2, table)
    monkeypatch.setattr(program_module, "_UPGRADES", {})

    def rename(data):
        attrs = data["instructions"][0]["attributes"]
        attrs["use_strength"] = attrs.pop("by_strength")

    register_upgrade("kilora.ops", 1, rename)
    upgraded = upgrade_program(saved)
    assert upgraded["op_set_version"] == 2
    assert upgraded["instructions"][0]["attributes"]["use_strength"] is True
    assert upgraded["identity"] == program_identity(upgraded)
    validated = validate(saved, signature)
    assert validated.identity == saved["identity"]
    assert validated.upgraded_identity == upgraded["identity"]
    with pytest.raises(ValueError, match="scale version 2"):
        interpret(validated, {"x": x}, {})
    monkeypatch.setitem(
        LOWERINGS,
        ("scale", 2, "torch_eager"),
        lambda a, k, e: (
            a[0] * (k["value"] * e["strength"] * e["timeline"] if k["use_strength"] else k["value"])
        ),
    )
    assert torch.equal(interpret(validated, {"x": x}, {}, context)["out"], expected)
    assert saved == original
    with pytest.raises(KiLoRAError, match="downgraded"):
        upgrade_program(upgraded, 1)
    saved["instructions"][0]["attributes"]["value"] = 3
    with pytest.raises(ProgramError, match="identity mismatch"):
        validate(saved, signature)


def test_newer_and_missing_upgrade_are_refused(monkeypatch):
    saved, signature = sample()
    saved["op_set_version"] = 2
    with pytest.raises(ProgramError, match="update KiLoRA"):
        validate(saved, signature)
    saved, _ = sample()
    monkeypatch.setitem(
        program_module._TABLES["kilora.ops"], 2, program_module.op_table("kilora.ops", 1)
    )
    monkeypatch.setattr(program_module, "_UPGRADES", {})
    with pytest.raises(KiLoRAError, match="No upgrade"):
        upgrade_program(saved)


def test_identity_uses_canonical_program_fields():
    saved, _ = sample()
    reordered = deepcopy(saved)
    reordered["instructions"][0]["attributes"] = dict(
        reversed(list(reordered["instructions"][0]["attributes"].items()))
    )
    assert program_identity(dict(reversed(list(reordered.items())))) == saved["identity"]
    reordered["instructions"][0]["attributes"]["value"] = 3
    assert program_identity(reordered) != saved["identity"]
