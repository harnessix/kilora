# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.compute import LayerSet, dropout_key, eager, run_layer
from kilora.ir.driver import run_layer_ir


@pytest.fixture(autouse=True)
def _keys_arm(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")


@pytest.mark.parametrize("path_select", ["converged", "score", "mix"])
@pytest.mark.parametrize("depth", [0.5, 1, 2, 3.5, 5])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_path_matches_eager(path_select, depth, fixed, adaptive):
    gen = torch.Generator().manual_seed(1172)
    x = torch.randn(2, 4, 6, generator=gen)
    y = torch.randn(2, 4, 6, generator=gen)
    layer = LayerSet(
        router=torch.randn(3, 6, generator=gen),
        keys_in=torch.randn(3, 1, 3, generator=gen),
        keys_out=torch.randn(3, 1, 3, generator=gen),
        down=torch.randn(6, 6, generator=gen) * 0.1,
        up=torch.randn(6, 6, generator=gen) * 0.1,
        per_expert=2,
        iteration="path",
        path_width=1,
        path_select=path_select,
        fixed_point_stop=fixed,
        gate_mode="adaptive" if adaptive else "none",
        region=torch.rand(2, 4, generator=gen),
    )
    with torch.no_grad(), eager(), dropout_key(12, 3):
        expected = run_layer(x, y, [layer], depth, target="path")
        actual = run_layer_ir(x, y, [layer], depth, target="path")
    assert torch.equal(actual, expected), (actual - expected).abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_path_multiple_sets_and_precision(dtype, shared, scope):
    gen = torch.Generator().manual_seed(1173)
    y = torch.randn(2, 5, 6, generator=gen).to(dtype)
    sets = [
        LayerSet(
            router=torch.randn(3, 6, generator=gen),
            keys_in=torch.randn(3, 1, 3, generator=gen),
            down=torch.randn(3, 6, generator=gen) * 0.1,
            up=torch.randn(3, 6, generator=gen) * 0.1,
            iteration="path",
            routing_scope=scope,
            path_width=1,
            scale=0.37,
            region=torch.rand(2, 5, generator=gen),
        )
        for _ in range(2)
    ]
    with torch.no_grad(), eager(), dropout_key(12, 3):
        expected = run_layer(y, y, sets, 4, target="multiple", shared=shared)
        actual = run_layer_ir(y, y, sets, 4, target="multiple", shared=shared)
    assert torch.equal(actual, expected), (actual - expected).abs().max()


def test_wide_path_uses_native(monkeypatch):
    from kilora import compute

    layer = LayerSet(
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
        iteration="path",
        path_width=2,
    )
    result = torch.ones(1, 2, 3)
    monkeypatch.setattr(compute, "run_layer", lambda *args, **kwargs: result)
    assert run_layer_ir(result, result, [layer], 2) is result


@pytest.mark.parametrize("noise", [0.0, 0.2])
@pytest.mark.parametrize("fractional", [False, True])
def test_mixed_input_and_state_paths(noise, fractional):
    gen = torch.Generator().manual_seed(1174)
    x = torch.randn(2, 4, 5, generator=gen)
    y = torch.randn(2, 4, 6, generator=gen)
    sets = [
        LayerSet(
            router=torch.randn(3, 6, generator=gen),
            keys_in=torch.randn(3, 1, 3, generator=gen),
            down=torch.randn(3, 5 if form == "input" else 6, generator=gen) * 0.1,
            up=torch.randn(3, 6, generator=gen) * 0.1,
            iteration="path",
            path_width=1,
            form=form,
        )
        for form in ("state", "input")
    ]
    depth = 2.5 if fractional else 3
    with torch.no_grad(), eager(), dropout_key(12, 3):
        torch.manual_seed(72)
        expected = run_layer(x, y, sets, depth, target="mixed", noise_scale=noise)
        torch.manual_seed(72)
        actual = run_layer_ir(x, y, sets, depth, target="mixed", noise_scale=noise)
    assert torch.equal(actual, expected)


def test_path_auto_schedule():
    class Schedule:
        is_auto = True

        def depth(self, *args):
            return 3.4

    gen = torch.Generator().manual_seed(1175)
    y = torch.randn(2, 3, 6, generator=gen)
    layer = LayerSet(
        router=torch.randn(3, 6, generator=gen),
        keys_in=torch.randn(3, 1, 3, generator=gen),
        down=torch.randn(3, 6, generator=gen),
        up=torch.randn(3, 6, generator=gen),
        iteration="path",
    )
    with torch.no_grad(), eager():
        expected = run_layer(y, y, [layer], 1, schedule=Schedule())
        actual = run_layer_ir(y, y, [layer], 1, schedule=Schedule())
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("depth", [2.5, 3.5, 5.5])
def test_fractional_path_keeps_early_stopped_output(depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    y = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    layer = LayerSet(
        router=torch.ones(1, 2),
        keys_in=torch.ones(1, 1, 1),
        down=0.2 * torch.eye(2),
        up=0.1 * torch.eye(2),
        per_expert=2,
        iteration="path",
        fixed_point_stop=True,
        gate_mode="none",
    )
    with torch.no_grad(), eager(), dropout_key(12, 3):
        expected = run_layer(y, y, [layer], 2, target="stopped")
        actual = run_layer(y, y, [layer], depth, target="stopped")
        interpreted = run_layer_ir(y, y, [layer], depth, target="stopped")
    assert not torch.equal(expected, y)
    assert torch.equal(actual, expected)
    assert torch.equal(interpreted, expected)


def test_path_gradients():
    gen = torch.Generator().manual_seed(1176)
    y = torch.randn(2, 4, 6, generator=gen, requires_grad=True)
    down = (torch.randn(3, 6, generator=gen) * 0.1).requires_grad_()
    up = (torch.randn(3, 6, generator=gen) * 0.1).requires_grad_()
    layer = LayerSet(
        router=torch.randn(3, 6, generator=gen),
        keys_in=torch.randn(3, 1, 3, generator=gen),
        down=down,
        up=up,
        iteration="path",
        scale=0.37,
    )
    with eager():
        expected = run_layer(y, y, [layer], 3)
        expected_grad = torch.autograd.grad(expected.square().sum(), (y, down, up))
        actual = run_layer_ir(y, y, [layer], 3)
        actual_grad = torch.autograd.grad(actual.square().sum(), (y, down, up))
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad, strict=True))


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
def test_imported_derived_chain_scores_shared_input(scope, square, carry, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    gen = torch.Generator().manual_seed(1193)
    x = torch.randn(2, 5, 8, generator=gen)
    width = 8 if square else 6
    y = torch.randn(2, 5, width, generator=gen)
    layer = LayerSet(
        router=torch.randn(3, width, generator=gen),
        keys_in=torch.randn(3, 1, 3, generator=gen),
        down=torch.randn(6, 8, generator=gen) * 0.3,
        up=torch.randn(6, width, generator=gen) * 0.3,
        per_expert=2,
        form="input",
        import_score="derived",
        routing_scope=scope,
        carry=carry,
        gate_mode="none",
    )
    with torch.no_grad(), eager(), dropout_key(12, 3):
        expected = run_layer(x, y, [layer], 4, target="derived-chain")
        actual = run_layer_ir(x, y, [layer], 4, target="derived-chain")
    assert torch.equal(actual, expected), (actual - expected).abs().max()
