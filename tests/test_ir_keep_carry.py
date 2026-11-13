# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import compute
from kilora.ir import check_invariants
from kilora.ir.defaults import default_bindings, default_program
from kilora.program import interpret


def run(part, variant, inputs, delta=None):
    bindings = default_bindings(part, variant)
    if delta is not None:
        bindings["delta"] = torch.tensor(delta)
    return interpret(default_program(part, variant), inputs, bindings)


def test_keep_probability():
    gen = torch.Generator().manual_seed(100)
    z = torch.randn(2, 3, 4, 5, generator=gen)
    sigma = torch.rand(2, 3, 4, 1, generator=gen) + 0.5
    out = run("keep", "default", {"carried": z, "base_scale": sigma})
    assert torch.equal(out["probability"], compute.keep_probability(z, sigma))
    check_invariants("keep", out)


@pytest.mark.parametrize("expect", [False, True])
def test_dropout_mask_and_parameter(expect, monkeypatch):
    gen = torch.Generator().manual_seed(101)
    z = torch.randn(2, 3, 12, generator=gen)
    base = torch.randn(2, 3, 12, generator=gen)
    shape = (2, 3, 3, 4)
    uniforms = compute.dropout_uniforms(shape, "test", 2, z.device, None)
    inputs = {"change": z.reshape(shape), "base_read": base.reshape(shape), "uniforms": uniforms}
    variant = "mask_expect" if expect else "mask_sample"
    out = run("carry", variant, inputs)["mask"].reshape_as(z)
    monkeypatch.setenv("KILORA_MASKS", "expect" if expect else "sample")
    native = compute.read_dropout(z, base, 4, "test", 2)
    assert torch.equal(out, native)
    delta = 0.25
    actual = run("carry", variant, inputs, delta)["mask"]
    sigma = base.reshape(shape).pow(2).mean(-1, keepdim=True).sqrt() + delta
    q = compute.keep_probability(z.reshape(shape), sigma)
    assert torch.equal(actual, q if expect else (uniforms < q).float())


@pytest.mark.parametrize("variant", ["drop", "dropped"])
def test_carry_kinds(variant):
    gen = torch.Generator().manual_seed(102)
    change = torch.randn(2, 3, 12, generator=gen)
    carried = torch.randn(2, 3, 12, generator=gen)
    mask = torch.rand(2, 3, 12, generator=gen)
    inputs = {
        "change": change,
        "carried": carried,
        "mask": mask,
        "base_read": change,
        "uniforms": mask,
    }
    out = run("carry", variant, inputs)
    native = mask * change if variant == "drop" else carried + (1 - mask) * (change - carried)
    assert torch.equal(out["state"], native)
    check_invariants("carry", out)


def test_sampler_transfer():
    gen = torch.Generator().manual_seed(103)
    z = torch.randn(2, 3, 12, generator=gen)
    base = torch.randn(2, 3, 12, generator=gen)
    shape = (2, 3, 3, 4)
    out = run("carry", "transfer", {"change": z.reshape(shape), "base_read": base.reshape(shape)})
    assert torch.equal(out["state"].reshape_as(z), compute.carry_transfer(z, base, 4))


def test_keep_program_validates_with_distinct_singleton_axis():
    from kilora.ir.defaults import _record
    from kilora.ir.parts import PARTS
    from kilora.program import validate

    signature = PARTS["keep"].sized_signature({"batch": 2, "tokens": 3, "experts": 4, "per": 5})
    validate(_record("keep", "default"), signature)
