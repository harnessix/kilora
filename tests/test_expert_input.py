# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from test_layer_batch import make_set
from test_lowrank_depth import _sets

from kilora import compute
from kilora.call_values import active_axes, axis_alphas, row_axes
from kilora.compute import AXES, _input_read, run_layer


def test_each_expert_reads_its_input_and_missing_uses_common():
    s = dataclasses.replace(_sets(1, 4, 2, 2, 4, seed=4)[0], form="input")
    x = torch.randn(2, 3, 4)
    own = torch.randn(2, 3, 4)
    s = dataclasses.replace(s, expert_inputs=(own, None))
    expected = torch.cat((F.linear(own, s.down[:2]), F.linear(x, s.down[2:])), -1)
    assert torch.equal(_input_read(s, x, x.dtype), expected)
    with pytest.raises(ValueError, match="expert input count"):
        _input_read(dataclasses.replace(s, expert_inputs=(own,)), x, x.dtype)


def test_input_reads_match_loop_and_replay(monkeypatch):
    s = dataclasses.replace(
        _sets(1, 4, 2, 2, 4, seed=7)[0],
        form="input",
        gate_mode="none",
        expert_inputs=(torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
    )
    x = torch.randn(2, 3, 4)
    y0 = torch.randn_like(x)
    outputs = []
    with torch.no_grad():
        for fast in (False, True):
            monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
            actual = run_layer(x, y0, [s], 3)
            assert torch.equal(actual, run_layer(x, y0, [s], 3))
            outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[1])


@pytest.mark.parametrize("fast", [False, True])
def test_batched_axes_match_separate_calls(fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    x = torch.randn(2, 2, 3, 4)
    own = torch.randn_like(x)
    sets = [dataclasses.replace(make_set("ip", 1), expert_inputs=(own,)), make_set("creative", 2)]
    alphas = {"ip": 0.6, "creative": 0.4}
    with torch.no_grad(), axis_alphas(alphas):
        expected = []
        for i, axis in enumerate(("ip", "creative")):
            row_sets = [dataclasses.replace(sets[0], expert_inputs=(own[i : i + 1],)), sets[1]]
            with active_axes((axis,)):
                expected.append(run_layer(x[i : i + 1], x[i : i + 1], row_sets, 1))
        with row_axes(torch.tensor([AXES.index("ip"), AXES.index("creative")])):
            actual = run_layer(x, x, sets, 1)
    assert torch.equal(actual, torch.cat(expected))


def _text_module(two=False, layout=None, depth=1):
    from kilora.bake import ModulePlan, TargetPlan, baked_copy
    from kilora.families import TEXT

    count = 2 if two else 1
    s = dataclasses.replace(
        _sets(1, 4, count, 2, 4, seed=4)[0],
        form="input",
        gate_mode="none",
        import_score="keys",
        scale=0.5,
    )
    layer = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4))
        layer.bias.fill_(0.25)
    plan = TargetPlan(
        sets=[s],
        depth=depth,
        layout=TEXT if layout is None else layout,
        tags=["first"],
        expert_names=[["first", "second"]] if two else [None],
    )
    return baked_copy(layer, ModulePlan([(0, -1, plan)])), s


def test_text_projection_uses_expert_conditioning():
    from kilora.comfyui.expert_text import conditioning_values

    layer, s = _text_module()
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 8
    own = base + 1
    with conditioning_values({"base": base, "expert_first": own}):
        actual = layer(base * 7)
    expected = F.linear(own, layer.weight, layer.bias) + 0.5 * (F.linear(own, s.down) @ s.up)
    torch.testing.assert_close(actual, expected)


def test_routed_offsets_and_missing_expert_read():
    from kilora.comfyui.expert_text import conditioning_values

    layer, s = _text_module(two=True)
    base = torch.randn(2, 3, 4)
    own = torch.randn_like(base)
    shares = []
    with (
        compute.recording_shares(lambda target, gates: shares.append(gates)),
        conditioning_values({"base": base, "expert_first": own}),
    ):
        actual = layer(base)
    gates = shares[0][0]
    expected = F.linear(base, layer.weight, layer.bias)
    expected = expected + gates[..., :1] * F.linear(own - base, layer.weight)
    for i, value in enumerate((own, base)):
        delta = F.linear(value, s.down[i * 2 : (i + 1) * 2]) @ s.up[i * 2 : (i + 1) * 2]
        expected = expected + 0.5 * gates[..., i : i + 1] * delta
    torch.testing.assert_close(actual, expected)


def test_text_rows_match_individual_calls_and_nontext_unchanged():
    from kilora.comfyui.expert_text import conditioning_values
    from kilora.family_spec import TokenLayout

    layer, _ = _text_module()
    other, _ = _text_module(layout=TokenLayout("set"))
    base = torch.randn(2, 3, 4)
    own = torch.randn_like(base)
    with conditioning_values({"base": base, "expert_first": own}):
        with row_axes(torch.tensor([AXES.index("creative"), -1])):
            actual = layer(base * 2)
        nontext = other(base * 2)
    expected = []
    for i in range(2):
        with conditioning_values({"base": base[i : i + 1], "expert_first": own[i : i + 1]}):
            expected.append(layer(base[i : i + 1] * 2))
    torch.testing.assert_close(actual, torch.cat(expected))
    assert torch.equal(nontext, other(base * 2))


def test_offsets_are_applied_once_at_depth_and_axis_weight():
    s = dataclasses.replace(
        make_set("ip", 1),
        expert_offsets=(torch.full((2, 2, 2, 3, 4), 2.0),),
    )
    x = torch.randn(2, 2, 2, 3, 4)
    without = dataclasses.replace(s, expert_offsets=None)
    with axis_alphas({"ip": 0.25}):
        actual = run_layer(x, x, [s], 3)
        expected = run_layer(x, x, [without], 3) + 0.5
    torch.testing.assert_close(actual, expected)
