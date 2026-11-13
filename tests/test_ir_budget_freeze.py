# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora import route_freeze
from kilora.compute import axis_budget, budget_share
from kilora.ir import check_invariants
from kilora.ir.default import default_program
from kilora.program import interpret


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_budget_default(count):
    axes = [f"axis{i}" for i in range(count)]
    weights = torch.ones(2, 5, count)
    result = interpret(
        default_program("budget", sizes={"batch": 2, "tokens": 5, "axes": count}),
        {"weights": weights},
        {},
    )
    native = axis_budget({**dict.fromkeys(axes, 1.0), "functional": 1.0})
    expected = torch.tensor([native[axis] for axis in axes]).expand_as(weights)
    assert torch.equal(result["scales"], expected)
    assert torch.equal(result["scales"], weights)
    assert native["functional"] == 1.0
    assert native[axes[0]] == budget_share([*axes, "functional"])
    check_invariants("budget", result)


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_budget_auto_mapped_alpha(count):
    generator = torch.Generator().manual_seed(45)
    alphas = [torch.rand(2, 5, generator=generator) * 1.5 for _ in range(count)]
    sizes = {"batch": 2, "tokens": 5}
    sum_program = default_program("budget", "auto_sum", sizes=sizes)
    share_program = default_program("budget", "auto", sizes=sizes)
    total = torch.zeros_like(alphas[0])
    for alpha in alphas:
        total = interpret(sum_program, {"previous": total, "alpha": alpha}, {})["total"]
    result = [
        interpret(share_program, {"total": total, "alpha": alpha}, {})["scale"] for alpha in alphas
    ]
    native = axis_budget({**{str(i): a for i, a in enumerate(alphas)}, "functional": 0.73})
    assert all(torch.equal(value, native[str(i)]) for i, value in enumerate(result))
    assert native["functional"] == 0.73
    check_invariants("budget", {"scales": torch.stack(result, dim=-1)})


@pytest.mark.parametrize("progress,enabled", [(0.2, True), (0.5, True), (0.9, True), (0.9, False)])
@pytest.mark.parametrize(
    "mode,depth,gated",
    [
        ("accumulate", 1, False),
        ("accumulate", 3, True),
        ("accumulate", 2.5, True),
        ("replace", 3, True),
        ("path", 3, False),
    ],
)
def test_freeze_native(progress, enabled, mode, depth, gated):
    gen = torch.Generator().manual_seed(81)
    count = int(depth) + (depth % 1 != 0)
    routes = [torch.softmax(torch.randn(2, 5, 4, generator=gen), dim=-1) for _ in range(count)]
    qa = torch.rand(2, 5, 1, generator=gen) if gated else torch.ones(2, 5, 1)
    qb = torch.rand(2, 1, 6, generator=gen) if gated else torch.ones(2, 1, 6)
    up = torch.randn(8, 6, generator=gen)
    sizes = {"batch": 2, "tokens": 5, "experts": 4, "per": 2, "cols": 8, "out": 6}
    step = default_program("freeze", "step", sizes=sizes)
    history = torch.zeros_like(routes[0])
    for route, fraction in zip(routes, route_freeze._step_weights(count, depth), strict=True):
        history = interpret(
            step,
            {
                "previous": history,
                "route": route,
                "fraction": torch.tensor(fraction),
                "replace": torch.tensor(mode != "accumulate"),
            },
            {},
        )["aggregate"]
    inputs = {
        "progress": torch.tensor(progress, dtype=torch.float64),
        "threshold": torch.tensor(0.5, dtype=torch.float64),
        "enabled": torch.tensor(enabled),
        "history": history,
        "last_weights": routes[-1],
        "strength": torch.full((2, 8), 0.7),
        "gate_token": qa,
        "gate_channel": qb,
        "up": up,
    }
    result = interpret(default_program("freeze", sizes=sizes), inputs, {})
    keyed = [
        SimpleNamespace(
            route_freeze=0.5,
            iteration="path" if mode == "path" else "accumulate",
            aggregation="replace" if mode == "replace" else "accumulate",
            per_expert=2,
            scale=0.7,
            up=up,
            down=torch.eye(8),
            form="state",
            expert_mask=None,
        )
    ]
    rec = {"routes": {"layer": [[r] for r in routes]}, "gates": {}}
    if gated:
        rec["gates"][("layer", 0)] = (qa, qb)
    entry = route_freeze.build(rec, keyed, [0], "layer", [None], depth, (2, 5))
    with route_freeze.freezing({}):
        assert result["frozen"].item() == (enabled and route_freeze.active(keyed, progress))
    assert torch.equal(result["weights"], routes[-1].mean(-2))
    assert torch.equal(result["scales"], entry["scale"])
    assert torch.equal(result["up_gated"], entry["up"])
    check_invariants("freeze", result, sizes=sizes)


def test_budget_auto_native_mapped_inputs():
    from kilora.compute import _axis_alpha
    from kilora.family_spec import TokenLayout
    from kilora.region import latent_size, token_layouts

    y = torch.zeros(1, 4, 3)
    source = torch.linspace(0, 1, 16).reshape(1, 16)
    with latent_size((4, 4)), token_layouts(lambda _: TokenLayout("image2d", (2, 2))):
        maps = [_axis_alpha(source, y, "layer"), _axis_alpha(1 - source, y, "layer")]
    expected = axis_budget({"a": maps[0], "b": maps[1]})
    programs = [
        default_program("budget", name, sizes={"batch": 1, "tokens": 4})
        for name in ("auto_sum", "auto")
    ]
    total = torch.zeros(1, 4)
    for alpha in maps:
        total = interpret(programs[0], {"previous": total, "alpha": alpha[..., 0]}, {})["total"]
    for key, alpha in zip(("a", "b"), maps, strict=True):
        actual = interpret(programs[1], {"total": total, "alpha": alpha[..., 0]}, {})["scale"]
        assert torch.equal(actual, expected[key][..., 0])


@pytest.mark.parametrize("values", [[0.1], [0.1, 0.2, 0.3], [0.73, 0.81, 0.32]])
def test_budget_auto_scalar_native(values):
    sums = default_program("budget", "scalar_sum")
    shares = default_program("budget", "scalar")
    total = torch.tensor(0.0, dtype=torch.float64)
    for value in values:
        total = interpret(
            sums, {"previous": total, "alpha": torch.tensor(value, dtype=torch.float64)}, {}
        )["total"]
    native = axis_budget({str(i): value for i, value in enumerate(values)})
    for i, value in enumerate(values):
        result = interpret(
            shares, {"total": total, "alpha": torch.tensor(value, dtype=torch.float64)}, {}
        )
        assert result["scale"].item() == native[str(i)]


def test_freeze_progress_uses_native_float_precision():
    sizes = {"batch": 1, "tokens": 1, "experts": 1, "out": 1}
    inputs = {
        "progress": torch.tensor(0.5, dtype=torch.float64),
        "threshold": torch.tensor(0.5000000001, dtype=torch.float64),
        "enabled": torch.tensor(True),
        "history": torch.ones(1, 1, 1),
        "last_weights": torch.ones(1, 1, 1),
        "strength": torch.ones(1, 1),
        "gate_token": torch.ones(1, 1, 1),
        "gate_channel": torch.ones(1, 1, 1),
        "up": torch.ones(1, 1),
    }
    assert not interpret(default_program("freeze", sizes=sizes), inputs, {})["frozen"].item()


def test_budget_auto_mixed_scalar_and_mapped_values():
    sizes = {"batch": 2, "tokens": 5}
    mapped = torch.rand(2, 5, generator=torch.Generator().manual_seed(19))
    values = [0.1, 0.2, mapped, 0.7, mapped * 0.9]
    total = torch.tensor(0.0, dtype=torch.float64)
    tensor_total = False
    for value in values:
        if isinstance(value, torch.Tensor):
            variant = "auto_sum" if tensor_total else "auto_seed_scalar"
            alpha = value
            tensor_total = True
        else:
            variant = "auto_add_scalar" if tensor_total else "scalar_sum"
            alpha = torch.tensor(value, dtype=torch.float64)
        total = interpret(
            default_program("budget", variant, sizes=sizes), {"previous": total, "alpha": alpha}, {}
        )["total"]
    native = axis_budget({str(i): value for i, value in enumerate(values)})
    for i, value in enumerate(values):
        variant = "auto" if isinstance(value, torch.Tensor) else "auto_scalar"
        alpha = (
            value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float64)
        )
        actual = interpret(
            default_program("budget", variant, sizes=sizes), {"total": total, "alpha": alpha}, {}
        )["scale"]
        expected = native[str(i)]
        if not isinstance(expected, torch.Tensor):
            expected = torch.full_like(actual, expected)
        assert torch.equal(actual, expected)


def test_drivers_map_alphas_and_build_multiple_frozen_sets():
    from kilora.compute import _axis_alpha
    from kilora.ir.part_driver import budget_auto_ir, freeze_record_ir

    gen = torch.Generator().manual_seed(718)
    y = torch.zeros(2, 5, 4)
    alphas = {"a": 0.2, "b": torch.rand(2, 5, generator=gen), "c": 0.7, "functional": 0.6}
    expected_budget = axis_budget({k: _axis_alpha(v, y, "layer") for k, v in alphas.items()})
    actual_budget = budget_auto_ir(alphas, y, "layer")
    for axis, expected in expected_budget.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual_budget[axis], expected)
        else:
            assert actual_budget[axis] == expected

    routes = [
        [torch.softmax(torch.randn(2, 5, n, generator=gen), -1) for n in (2, 3)] for _ in range(3)
    ]
    layers = [
        SimpleNamespace(
            route_freeze=0.3 if i == 0 else None,
            iteration="accumulate",
            aggregation="accumulate",
            per_expert=2,
            scale=torch.rand(2, n * 2, generator=gen),
            up=torch.randn(n * 2, 4, generator=gen),
            down=torch.randn(n * 2, 4),
            form="input",
            expert_mask=None,
        )
        for i, n in enumerate((2, 3))
    ]
    qa, qb = torch.rand(2, 5, 1, generator=gen), torch.rand(2, 1, 4, generator=gen)
    rec = {"routes": {"layer": routes}, "gates": {("layer#input1", 0): (qa, qb)}}
    expected = route_freeze.build(rec, layers, [0, 1], "layer", [None, None], 2.5, (2, 5))
    frozen, actual = freeze_record_ir(rec, layers, [0, 1], "layer", [None, None], 2.5, (2, 5), 0.8)
    assert frozen
    assert torch.equal(actual["scale"], expected["scale"])
    assert torch.equal(actual["up"], expected["up"])
    frozen, _ = freeze_record_ir(rec, layers, [0, 1], "layer", [None, None], 2.5, (2, 5), 0.2)
    assert not frozen


def test_freeze_shared_routing_slices_preserve_native_scales():
    from kilora.ir.part_driver import freeze_record_ir

    gen = torch.Generator().manual_seed(241)
    joint = [torch.softmax(torch.randn(1, 3, 4, generator=gen), -1) for _ in range(2)]
    routes = [[step[..., :2], step[..., 2:]] for step in joint]
    layers = [
        SimpleNamespace(
            route_freeze=0.2,
            iteration="accumulate",
            aggregation="accumulate",
            per_expert=1,
            scale=1.0,
            up=torch.eye(2),
            down=torch.eye(2),
            form="state",
            expert_mask=None,
        )
        for _ in range(2)
    ]
    rec = {"routes": {"layer": routes}, "gates": {}}
    expected = route_freeze.build(rec, layers, [0, 1], "layer", [None, None], 2, (1, 3))
    frozen, actual = freeze_record_ir(rec, layers, [0, 1], "layer", [None, None], 2, (1, 3), 0.7)
    assert frozen and torch.equal(actual["scale"], expected["scale"])
    assert not torch.allclose(routes[-1][0].sum(-1), torch.ones(1, 3))
    assert torch.allclose(torch.cat(routes[-1], -1).sum(-1), torch.ones(1, 3))
