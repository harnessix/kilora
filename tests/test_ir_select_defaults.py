# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.compute import tensor_gate, token_gate
from kilora.ir.defaults import default_bindings, default_program
from kilora.ir.parts import check_invariants
from kilora.program_eager import interpret


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("selection", ["sparsemax", "top1"])
def test_select_default_parity(scope, selection):
    generator = torch.Generator().manual_seed(4242)
    logits = torch.randn(2, 7, 5, generator=generator)
    variant = f"{scope}_{selection}"
    program = default_program("select", variant)
    bindings = default_bindings("select", variant)
    regions = (None, torch.rand(2, 7, 1, generator=generator), torch.zeros(2, 7, 1))
    for region in regions if scope == "tensor" else (None,):
        inputs = {
            "logits": logits,
            "noise": torch.zeros_like(logits[:, :1] if scope == "tensor" else logits),
        }
        if scope == "tensor":
            inputs.update(
                region=(torch.ones_like(logits[..., 0]) if region is None else region[..., 0]),
                has_region=torch.tensor(region is not None),
            )
            expected = tensor_gate(logits, region=region, select=selection)
        else:
            expected = token_gate(logits, select=selection)
        outputs = interpret(program, inputs, bindings)
        assert torch.equal(outputs["weights"], expected)
        check_invariants("select", outputs, sizes={"batch": 2, "tokens": 7, "experts": 5})


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("selection", ["sparsemax", "top1"])
def test_select_default_supplied_noise(scope, selection):
    from kilora.compute import region_mean, sparsemax, top1_gate

    generator = torch.Generator().manual_seed(814)
    logits = torch.randn(2, 7, 5, generator=generator)
    noise = torch.randn(2, 1 if scope == "tensor" else 7, 5, generator=generator)
    variant = f"{scope}_{selection}"
    inputs = {"logits": logits, "noise": noise}
    if scope == "tensor":
        region = torch.rand(2, 7, 1, generator=generator)
        inputs.update(region=region[..., 0], has_region=torch.tensor(True))
        noisy = region_mean(logits, region) + noise
        expected = (sparsemax(noisy) if selection == "sparsemax" else top1_gate(noisy)).expand_as(
            logits
        )
    else:
        expected = (
            sparsemax(logits + noise) if selection == "sparsemax" else top1_gate(logits, noise)
        )
    outputs = interpret(
        default_program("select", variant), inputs, default_bindings("select", variant)
    )
    assert torch.equal(outputs["weights"], expected)
    check_invariants("select", outputs, sizes={"batch": 2, "tokens": 7, "experts": 5})


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_top1_default_has_exact_values_and_native_gradients(scope):
    from kilora.compute import region_mean

    gen = torch.Generator().manual_seed(1948)
    logits = torch.randn(2, 127, 7, generator=gen).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    variant = f"{scope}_top1"
    inputs = {
        "logits": logits,
        "noise": torch.zeros_like(logits[:, :1] if scope == "tensor" else logits),
    }
    if scope == "tensor":
        inputs.update(region=torch.ones_like(logits[..., 0]), has_region=torch.tensor(False))
        query = region_mean(reference, None)
        expected = tensor_gate(reference, select="top1").expand_as(reference)
    else:
        query = reference
        expected = token_gate(reference, select="top1")
    actual = interpret(
        default_program("select", variant), inputs, default_bindings("select", variant)
    )["weights"]
    hard = torch.zeros_like(query).scatter(-1, query.argmax(-1, keepdim=True), 1.0)
    assert torch.equal(actual, hard.expand_as(logits))
    assert torch.equal(actual, expected)
    weights = torch.randn(2, 127, 7, generator=gen)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), reference)[0]
    assert torch.equal(actual_grad, expected_grad)
