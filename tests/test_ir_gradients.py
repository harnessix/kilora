# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.compute import LayerSet, dropout_key, eager, run_layer
from kilora.ir.driver import run_layer_ir


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("select", ["top1", "sparsemax"])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("mask_mode", ["sample", "expect"])
def test_layer_gradients_match_native(
    scope, depth, select, adaptive, carry, mask_mode, monkeypatch
):
    monkeypatch.setenv("KILORA_MASKS", mask_mode)
    gen = torch.Generator().manual_seed(873)
    y = torch.randn(2, 4, 6, generator=gen)
    values = [
        torch.randn(4, 6, generator=gen) * 0.1,
        torch.randn(4, 6, generator=gen) * 0.1,
        torch.randn(3, 6, generator=gen),
        torch.randn(2, 2, 3, generator=gen),
        torch.randn(2, 2, 3, generator=gen),
        torch.tensor(0.8),
        torch.tensor(1.3),
    ]

    def run(fn):
        params = [v.clone().requires_grad_() for v in values]
        s = LayerSet(
            down=params[0],
            up=params[1],
            router=params[2],
            keys_in=params[3],
            keys_out=params[4],
            scale=params[5],
            tau=params[6],
            per_expert=2,
            routing_select=select,
            routing_scope=scope,
            gate_mode="adaptive" if adaptive else "none",
            carry=carry,
        )
        with eager(), dropout_key(17, 4):
            out = fn(y, y, [s], depth, target="grad")
        loss = out.square().sum()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return out, loss, grads

    expected, loss, grads = run(run_layer)
    actual, actual_loss, actual_grads = run(run_layer_ir)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_loss, loss)
    for index, (a, b) in enumerate(zip(actual_grads, grads, strict=True)):
        assert (a is None and b is None) or (
            a is not None and b is not None and torch.equal(a, b)
        ), index


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("adaptive", [False, True])
def test_scorer_gradients_match_native(kind, depth, adaptive):
    from kilora.compute import new_gate

    gen = torch.Generator().manual_seed(925)
    y = torch.randn(2, 4, 6, generator=gen)
    weights = [
        torch.randn(4, 6, generator=gen) * 0.1,
        torch.randn(4, 6, generator=gen) * 0.1,
        torch.randn(3, 6, generator=gen),
        torch.randn(2, 1, 3, generator=gen),
    ]
    gate = new_gate(kind, 2, 6, gen, 4)

    def run(fn):
        params = [v.clone().requires_grad_() for v in weights]
        scorer = {k: v.clone().requires_grad_() for k, v in gate.items()}
        s = LayerSet(
            down=params[0],
            up=params[1],
            router=params[2],
            keys_in=params[3],
            per_expert=2,
            routing_select="top1",
            output_gate=kind,
            gate=scorer,
            gate_mode="both" if adaptive else "learned",
        )
        with eager(), dropout_key(17, 4):
            out = fn(y, y, [s], depth, target="scorer")
        gradients = torch.autograd.grad(
            out.square().sum(), params + list(scorer.values()), allow_unused=True
        )
        return out, gradients

    expected, grads = run(run_layer)
    actual, actual_grads = run(run_layer_ir)
    assert torch.equal(actual, expected)
    for index, (a, b) in enumerate(zip(actual_grads, grads, strict=True)):
        assert (a is None and b is None) or (
            a is not None and b is not None and torch.equal(a, b)
        ), index


@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_gate_mask_program_gradient(mode):
    from kilora.ir import default_bindings, default_program
    from kilora.program import interpret

    q = torch.tensor([[[0.2, 0.7, 0.4]]], requires_grad=True)
    u = torch.tensor([[[0.1, 0.9, 0.3]]])
    output = interpret(
        default_program("gate", f"mask_{mode}"),
        {"probability": q, "uniforms": u},
        default_bindings("gate", f"mask_{mode}"),
    )["mask"]
    hard = (u < q).float() if mode == "sample" else q.detach()
    assert torch.equal(output, hard)
    assert torch.equal(torch.autograd.grad(output.sum(), q)[0], torch.ones_like(q))


def test_detach_op_removes_gradient():
    from kilora.program import Signature, ValueType, interpret, program_identity, validate

    typ = ValueType(("batch", "tokens", "cols"))
    sig = Signature({"x": typ}, {"y": typ})
    record = {
        "name": "detach-test",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"x": {"axes": list(typ.axes)}},
        "outputs": {"y": {"axes": list(typ.axes)}},
        "bindings": {},
        "instructions": [{"id": 0, "op": "detach", "args": ["x"], "result": "y", "attributes": {}}],
    }
    record["identity"] = program_identity(record)
    x = torch.randn(1, 2, 3, requires_grad=True)
    result = interpret(validate(record, sig), {"x": x}, {})["y"]
    assert torch.equal(result, x)
    assert not result.requires_grad


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("depth", [1, 2.5, 3])
def test_multiple_sets_and_input_gradient(shared, depth):
    gen = torch.Generator().manual_seed(918)
    y0 = torch.randn(2, 4, 6, generator=gen)
    templates = []
    for _ in range(2):
        templates.append(
            [
                torch.randn(2, 6, generator=gen) * 0.1,
                torch.randn(2, 6, generator=gen) * 0.1,
                torch.randn(3, 6, generator=gen),
                torch.randn(2, 2, 3, generator=gen),
            ]
        )

    def run(fn):
        y = y0.clone().requires_grad_()
        params = [[t.clone().requires_grad_() for t in template] for template in templates]
        sets = [
            LayerSet(down=a, up=b, router=r, keys_in=k, gate_mode="adaptive")
            for a, b, r, k in params
        ]
        with eager(), dropout_key(17, 4):
            out = fn(y, y, sets, depth, target="multi", shared=shared)
        grads = torch.autograd.grad(
            out.square().sum(), [y] + [t for group in params for t in group[:2]]
        )
        return out, grads

    expected, grads = run(run_layer)
    actual, actual_grads = run(run_layer_ir)
    assert torch.equal(actual, expected)
    for a, b in zip(actual_grads, grads, strict=True):
        assert torch.equal(a, b)


def test_scale_only_gradient():
    gen = torch.Generator().manual_seed(509)
    y = torch.randn(2, 4, 6, generator=gen)
    down, up, router = (torch.randn(*shape, generator=gen) for shape in ((2, 6), (2, 6), (3, 6)))
    keys = torch.randn(2, 1, 3, generator=gen)

    def run(fn):
        scale = torch.tensor(0.7, requires_grad=True)
        s = LayerSet(down=down, up=up, router=router, keys_in=keys, scale=scale, gate_mode="none")
        with eager(), dropout_key(17, 4):
            result = fn(y, y, [s], 3, target="scale")
        gradient = torch.autograd.grad(result.square().sum(), scale)[0]
        return result, gradient

    expected, gradient = run(run_layer)
    actual, actual_gradient = run(run_layer_ir)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, gradient)
    assert gradient.isfinite() and gradient.abs() > 0


@pytest.mark.xfail(strict=True, reason="tensor-base pow differs from native scalar-base torch.pow")
@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_read_mask_program_gradient(mode, monkeypatch):
    from kilora.compute import read_dropout
    from kilora.ir import default_bindings, default_program
    from kilora.program import interpret

    monkeypatch.setenv("KILORA_MASKS", mode)
    gen = torch.Generator().manual_seed(75)
    value = torch.randn(2, 4, 4, generator=gen)
    base = torch.randn(2, 4, 4, generator=gen, requires_grad=True)
    gradient = torch.randn(2, 4, 4, generator=gen)

    def run(native):
        z = value.clone().requires_grad_()
        with dropout_key(17, 4):
            if native:
                mask = read_dropout(z, base, 2, "part", 2)
            else:
                from kilora.compute import dropout_uniforms

                shape = (2, 4, 2, 2)
                uniforms = dropout_uniforms(shape, "part", 2, z.device)
                mask = interpret(
                    default_program("carry", f"mask_{mode}"),
                    {
                        "change": z.reshape(shape),
                        "base_read": base.reshape(shape),
                        "uniforms": uniforms,
                    },
                    default_bindings("carry", f"mask_{mode}"),
                )["mask"].reshape_as(z)
        return mask, torch.autograd.grad(mask, (z, base), gradient, allow_unused=True)

    expected, grad = run(True)
    actual, actual_grad = run(False)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad[0], grad[0])
    assert actual_grad[1] is None and grad[1] is None
