# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from unittest.mock import patch

import torch
from test_input_depth import _input_sets, _xy

from kilora import compute, route_freeze


def test_routed_import_freeze_scalar_conversions(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive", route_freeze=0.4)
    x, y = _xy()
    region = torch.ones(y.shape[:-1])
    scales = [[1.0] * (s.down.shape[0] // s.per_expert)]

    def call(cache, *, region_arg, scale_arg):
        with (
            torch.no_grad(),
            compute.dropout_key(7, 3),
            compute.sampling_progress(0.5),
            route_freeze.freezing(cache),
        ):
            return compute.run_layer(
                x,
                y,
                [s],
                3,
                target="layer",
                regions=region_arg,
                expert_scales=scale_arg,
            )

    reference_cache = {}
    reference = [call(reference_cache, region_arg=None, scale_arg=None) for _ in range(2)]
    counts = []
    originals = {
        name: getattr(torch.Tensor, name) for name in ("item", "__bool__", "__float__", "__int__")
    }

    def wrapper(name):
        def counted(tensor, *args, **kwargs):
            counts.append(name)
            return originals[name](tensor, *args, **kwargs)

        return counted

    cache = {}
    with patch.multiple(torch.Tensor, **{name: wrapper(name) for name in originals}):
        actual = []
        for _ in range(2):
            counts.clear()
            actual.append(call(cache, region_arg=[region], scale_arg=scales))
            assert len(counts) <= 2, counts
    for got, want in zip(actual, reference, strict=True):
        assert torch.equal(got, want)


def test_depth_map_gate_keeps_original_output(monkeypatch):
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive")
    x, y = _xy()
    depth = torch.tensor([[2, 3, 2.5, 3, 2, 3, 2]]).expand(2, -1)
    original = compute.split_gate

    def old_gate(*args, **kwargs):
        args = list(args)
        last = args[10] if len(args) > 10 else kwargs.get("last", False)
        if isinstance(last, torch.Tensor):
            last = True if bool(last.all()) else False if not bool(last.any()) else last
            if len(args) > 10:
                args[10] = last
            else:
                kwargs["last"] = last
        return original(*args, **kwargs)

    def call():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return compute.run_layer(x, y, [s], 3, target="layer", token_depth=depth)

    monkeypatch.setenv("KILORA_FUSED", "0")
    for mode in ("sample", "expect"):
        monkeypatch.setenv("KILORA_MASKS", mode)
        monkeypatch.setattr(compute, "split_gate", old_gate)
        expected = call()
        monkeypatch.setattr(compute, "split_gate", original)
        assert torch.equal(call(), expected)


def test_region_and_depth_map_have_no_scalar_conversions_after_first_call(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive", route_freeze=0.4)
    x, y = _xy()
    region = torch.tensor([[1, 1, 0.5, 0, 1, 0.25, 1]]).expand(2, -1).clone()
    depth = torch.tensor([[2, 3, 2.5, 3, 2, 3, 2]]).expand(2, -1).clone()

    def call():
        with torch.no_grad(), compute.dropout_key(7, 3), compute.sampling_progress(0.5):
            return compute.run_layer(
                x, y, [s], 3, target="layer", regions=[region], token_depth=depth
            )

    expected = call()
    originals = {
        name: getattr(torch.Tensor, name)
        for name in ("item", "__bool__", "__float__", "__int__", "tolist")
    }
    counts = []

    def wrapper(name):
        def counted(tensor, *args, **kwargs):
            counts.append(name)
            return originals[name](tensor, *args, **kwargs)

        return counted

    with patch.multiple(torch.Tensor, **{name: wrapper(name) for name in originals}):
        actual = call()
        assert not counts, counts
    assert torch.equal(actual, expected)

    region.fill_(1)
    depth.fill_(1)
    changed = call()
    with torch.no_grad(), compute.dropout_key(7, 3):
        reference = compute.run_layer(x, y, [s], 1, target="layer")
    assert torch.equal(changed, reference)


def test_plan_omits_unit_expert_scales():
    from kilora.bake import TargetPlan
    from kilora.call_values import expert_scales

    plan = TargetPlan(expert_names=[["a", "b"]])
    with expert_scales({"a": 1.0, "b": 1.0}):
        assert plan._expert_scales() is None
    with expert_scales({"a": 0.5}):
        assert plan._expert_scales() == [[0.5, 1.0]]


def test_plan_reuses_depth_map_and_invalidates_it_on_change(monkeypatch):
    from kilora.bake import TargetPlan
    from kilora.depth_map import make_depth_map

    depth_map = make_depth_map(torch.tensor([[0.0, 0.5], [1.0, 0.2]]), start=0, base=1)
    plan = TargetPlan(token_depth=depth_map)
    monkeypatch.setattr(plan, "_token_region", lambda tensor, h: tensor.reshape(1, -1))
    h = torch.zeros(1, 4, 8)
    first = plan._token_depth(h)
    assert plan._token_depth(h) is first
    other = TargetPlan(token_depth=depth_map)
    monkeypatch.setattr(other, "_token_region", lambda tensor, h: tensor.reshape(1, -1))
    assert other._token_depth(h) is first
    depth_map.values.add_(0.1)
    changed = plan._token_depth(h)
    assert changed is not first
    assert not torch.equal(changed, first)


def test_inference_tensor_metadata_is_not_reused_after_mutation():
    with torch.inference_mode():
        region = torch.ones(1, 4)
        y = torch.zeros(1, 4, 8)
        assert compute._token_region(region, y) is None
        region[0, 0] = 0
        assert compute._token_region(region, y) is not None


def test_cached_region_and_depth_match_uncached_outputs(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive")
    x, y = _xy()
    region = torch.tensor([[1, 0, 0.5, 1, 1, 0.2, 1]]).expand(2, -1).clone()
    for values in ([2, 3, 2.5, 3, 2, 3, 2], [1.5] * 7):
        depth = torch.tensor([values]).expand(2, -1).clone()
        for mode in ("sample", "expect"):
            monkeypatch.setenv("KILORA_MASKS", mode)

            def call(depth=depth):
                with torch.no_grad(), compute.dropout_key(7, 3):
                    return compute.run_layer(
                        x, y, [s], 3, target="layer", regions=[region], token_depth=depth
                    )

            original = compute._tensor_metadata
            with monkeypatch.context() as patcher:
                patcher.setattr(compute, "_tensor_metadata", lambda tensor, name, fn: fn())
                expected = call()
            assert compute._tensor_metadata is original
            assert torch.equal(call(), expected)
            assert torch.equal(call(), expected)


def test_plan_inference_mode_has_no_warm_scalar_conversions(monkeypatch):
    from kilora.bake import TargetPlan
    from kilora.depth_map import make_depth_map
    from kilora.region import latent_size

    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive")
    x, y = _xy()
    region = torch.tensor([[[1, 0, 0.5, 1, 1, 0.2, 1]]])
    m = make_depth_map(torch.tensor([[[0, 1, 0.5, 1, 0, 1, 0]]]), start=0, base=1)
    plan = TargetPlan(
        sets=[s], depth=3, name="layer", regions=[region], roles=["all"], token_depth=m
    )
    originals = {
        name: getattr(torch.Tensor, name)
        for name in ("item", "__bool__", "__float__", "__int__", "tolist")
    }
    counts = []

    def wrapper(name):
        def counted(tensor, *args, **kwargs):
            counts.append(name)
            return originals[name](tensor, *args, **kwargs)

        return counted

    with torch.inference_mode(), latent_size((1, 7)), compute.dropout_key(7, 3):
        expected = plan.run(x, y)
        with patch.multiple(torch.Tensor, **{name: wrapper(name) for name in originals}):
            actual = plan.run(x, y)
            assert not counts, counts
    assert torch.equal(actual, expected)


def test_metadata_cache_for_multiple_token_axes(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_input_sets(n=1)[0], gate_mode="adaptive")
    x, y = _xy()
    x = x[:, :6].reshape(2, 2, 3, -1)
    y = y[:, :6].reshape(2, 2, 3, -1)
    region = torch.tensor([[[1, 0, 0.5], [1, 1, 0.2]]]).expand(2, -1, -1).clone()
    depth = torch.tensor([[[2, 3, 2.5], [3, 2, 3]]]).expand(2, -1, -1).clone()

    def call():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return compute.run_layer(
                x, y, [s], 3, target="layer", regions=[region], token_depth=depth
            )

    expected = call()
    originals = {
        name: getattr(torch.Tensor, name)
        for name in ("item", "__bool__", "__float__", "__int__", "tolist")
    }
    counts = []

    def wrapper(name):
        def counted(tensor, *args, **kwargs):
            counts.append(name)
            return originals[name](tensor, *args, **kwargs)

        return counted

    with patch.multiple(torch.Tensor, **{name: wrapper(name) for name in originals}):
        actual = call()
        assert not counts, counts
    assert torch.equal(actual, expected)
