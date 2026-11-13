# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F
from test_program_validate import lora, selection

from kilora.program import (
    LOWERINGS,
    ProgramError,
    interpret,
    layer_change_signature,
    op_table,
    program_identity,
    validate,
)
from kilora.sections import (
    LowRankChange,
    ProgramChange,
    Section,
    decode_sections,
    encode_sections,
    layer_changes,
)


def test_lowering_coverage():
    from kilora.program import _TABLES

    for version in _TABLES["kilora.ops"]:
        table = op_table("kilora.ops", version)
        missing = [
            op
            for op, schema in table.items()
            if (op, schema["since_version"], "torch_eager") not in LOWERINGS
        ]
        assert missing == [], (version, missing)


def test_lora_and_silu():
    torch.manual_seed(31)
    x, y = torch.randn(2, 3, 16), torch.randn(2, 3, 32)
    a, b = torch.randn(8, 16), torch.randn(32, 8)
    for silu in (False, True):
        change = ProgramChange(lora(silu), {"A": a, "B": b})
        actual = change.add(x, y)
        expected = y + F.linear(F.silu(F.linear(x, a)) if silu else F.linear(x, a), b)
        assert torch.equal(actual, expected)
        if not silu:
            assert torch.equal(actual, LowRankChange(a, b).add(x, y))


def test_generic_selection_and_sort():
    p, sig = selection()
    p["instructions"] = [
        {
            "id": 0,
            "op": "softmax",
            "args": ["logits"],
            "result": "weights",
            "attributes": {"axis": "experts"},
        }
    ]
    p["identity"] = program_identity(p)
    x = torch.randn(2, 7, 4)
    assert torch.equal(
        interpret(validate(p, sig), {"logits": x}, {})["weights"], torch.softmax(x, -1)
    )
    p["instructions"] = [
        {
            "id": 0,
            "op": "sort",
            "args": ["logits"],
            "result": "sorted",
            "attributes": {"axis": "experts", "descending": True},
        },
        {
            "id": 1,
            "op": "cumsum",
            "args": ["sorted"],
            "result": "weights",
            "attributes": {"axis": "experts"},
        },
    ]
    p["identity"] = program_identity(p)
    assert torch.equal(
        interpret(validate(p, sig), {"logits": x}, {})["weights"],
        torch.sort(x, -1, descending=True).values.cumsum(-1),
    )


def test_section_roundtrip_and_strength():
    p = lora()
    a, b = torch.randn(8, 16), torch.randn(32, 8)
    sections = [Section("program", "example", {"A": a, "B": b}, p)]
    tensors, records = encode_sections(sections)
    decoded = decode_sections(records, tensors)
    change = layer_changes(decoded, 0.5)["hook"][0]
    x, y = torch.randn(1, 2, 16), torch.randn(1, 2, 32)
    assert torch.equal(change.add(x, y), y + 0.5 * F.linear(F.linear(x, a), b))
    assert change.validate_for(16, 32) == validate(p, layer_change_signature((16, 32)))


def test_timeline_and_strength():
    from kilora.fixed_change import fixed_copy
    from kilora.timeline import Clock, Row, time_scales

    torch.manual_seed(4)
    base = torch.nn.Linear(16, 32)
    a, b = torch.randn(8, 16), torch.randn(32, 8)
    change = ProgramChange(lora(), {"A": a, "B": b}, 0.25)
    change.tag, change.axis = "mod-a", "creative"
    layer = fixed_copy(base, [change])
    x = torch.randn(1, 2, 16)
    plain = base(x)
    with time_scales([Row("mod-a", "strength", "0.5")], Clock()):
        actual = layer(x)
    assert torch.allclose(actual, plain + 0.125 * F.linear(F.linear(x, a), b), atol=1e-6)


def test_referenced_bindings(tmp_path):
    import hashlib

    from safetensors.torch import save_file

    from kilora.sections import resolve_program_bindings

    a, b = torch.randn(8, 16), torch.randn(32, 8)
    path = tmp_path / "weights.safetensors"
    save_file({"other": b}, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    p = lora()
    p["bindings"]["B"].update(kind="ref", sha256=digest, tensor_key="other")
    p["identity"] = program_identity(p)
    ref = Section("weight-ref", "weight", {}, {"sha256": digest})
    section = Section("program", "program", {"A": a}, p)
    resolved = resolve_program_bindings([ref, section], [tmp_path])[1]
    x, y = torch.randn(1, 2, 16), torch.randn(1, 2, 32)
    assert torch.equal(
        layer_changes([resolved], 1)["hook"][0].add(x, y), y + F.linear(F.linear(x, a), b)
    )


def test_program_disables_fused_path(monkeypatch):
    from kilora import compute, fused

    reasons = []
    monkeypatch.setattr(fused, "_note", reasons.append)
    token = compute._PROGRAM_LAYER.set(True)
    try:
        assert not compute._fast_path_ok(torch.zeros(1), [], 0, None, None)
        assert not fused.fused_ok(torch.zeros(1), [])
    finally:
        compute._PROGRAM_LAYER.reset(token)
    assert reasons == ["a layer-change program"]


def test_scaled_section():
    from kilora.sections import scaled

    a, b = torch.randn(8, 16), torch.randn(32, 8)
    section = scaled(Section("program", "program", {"A": a, "B": b}, lora()), 0.25)
    x, y = torch.randn(1, 2, 16), torch.randn(1, 2, 32)
    expected = y + F.linear(F.linear(x, a), b) * 0.25
    assert torch.equal(layer_changes([section], 1)["hook"][0].add(x, y), expected)
    tensors, records = encode_sections([section])
    decoded = decode_sections(records, tensors)
    assert torch.equal(layer_changes(decoded, 1)["hook"][0].add(x, y), expected)


def test_mask_rejected_at_hook_but_allowed_for_other_parts():
    from kilora.fixed_change import fixed_copy

    p = lora()
    p["instructions"].insert(
        3,
        {
            "id": 3,
            "op": "mask",
            "args": ["e"],
            "result": "masked",
            "attributes": {"source": "region"},
        },
    )
    p["instructions"][-1].update(id=4, args=["masked"])
    p["identity"] = program_identity(p)
    change = ProgramChange(p, {"A": torch.randn(8, 16), "B": torch.randn(32, 8)})
    with pytest.raises(
        ProgramError, match="program test.*instruction 3, op mask.*regions in programs come later"
    ):
        change.validate_for(16, 32)
    with pytest.raises(ProgramError, match="instruction 3, op mask"):
        fixed_copy(torch.nn.Linear(16, 32), [change])
    generic, sig = selection()
    generic["instructions"][0].update(op="mask", attributes={"source": "roles"})
    generic["identity"] = program_identity(generic)
    validate(generic, sig)


def test_reference_without_digest():
    from kilora.sections import resolve_program_bindings

    ref = Section("weight-ref", "legacy", {}, {})
    sections = [ref]
    assert resolve_program_bindings(sections, []) is sections
    p = lora()
    p["bindings"]["B"].update(kind="ref", sha256="missing", tensor_key="b")
    section = Section("program", "program", {}, p)
    with pytest.raises(ValueError, match="program binding B: missing weight-ref missing"):
        resolve_program_bindings([ref, section], [])


@pytest.mark.parametrize("path_kind", ["after", "full_after"])
def test_apply_hook_program_disables_fused(tmp_path, monkeypatch, caplog, path_kind):
    import logging

    from test_bypass import Patcher
    from test_comfyui import FakeBaseModel
    from test_families import H, _model
    from test_working_copies import routed_chain

    from kilora import bake, compute, fused
    from kilora.comfyui import patching

    path = "diffusion_model.double_blocks.0.img_attn.qkv"
    p = lora()
    p["bindings"]["A"]["shape"] = [8, H]
    p["bindings"]["B"]["shape"] = [3 * H, 8]
    p["identity"] = program_identity(p)
    change = ProgramChange(p, {"A": torch.randn(8, H), "B": torch.randn(3 * H, 8)})
    monkeypatch.setattr(patching, "_fixed_changes", lambda *args: {path: [change]})
    calls = []

    def run(self, x, y):
        calls.append(compute._PROGRAM_LAYER.get())
        assert not fused.fused_ok(y, [])
        return y

    def run_forward(plan, original, x, *args, **kwargs):
        return run(plan, x, original(x, *args, **kwargs))

    monkeypatch.setattr(bake.ModulePlan, "run", run)
    monkeypatch.setattr(bake, "run_plan_forward", run_forward)
    monkeypatch.setattr(fused, "_NOTED", set())
    out, _ = patching.apply(
        Patcher(FakeBaseModel(_model("flux"))),
        routed_chain(tmp_path),
        1,
        joint=True,
        bypass="on",
        import_score="keys",
    )
    injection = out.injections["kilora"][0].inject
    hooks = next(
        cell.cell_contents for cell in injection.__closure__ if isinstance(cell.cell_contents, list)
    )
    hook = next(h for h in hooks if h.path == path)
    base = out.get_model_object(path)
    x = torch.randn(1, 2, H)
    with caplog.at_level(logging.INFO, logger="kilora"):
        for _ in range(2):
            if path_kind == "after":
                hook.after(x, base(x))
            else:
                hook.full_after(x, base.forward)
    assert calls == [True, True]
    assert not compute._PROGRAM_LAYER.get()
    assert sum("a layer-change program" in record.message for record in caplog.records) == 1
