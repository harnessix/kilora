# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets
from test_output_gate import _gated

from kilora import compute as C
from kilora.compute import dropout_key, run_layer, tensor_gate
from kilora.region import grid_for, latent_size, layer_region, tokens_from_image

TOKENS = 9


def _y0(seed=1):
    return torch.randn(2, TOKENS, 16, generator=torch.Generator().manual_seed(seed))


def _run(sets, depth, fast, monkeypatch, regions=None):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    with torch.no_grad(), dropout_key(1, 0.4):
        return run_layer(_y0(), _y0(), sets, depth, target="layer", regions=regions)


def _plain(n=2, seed=2, **kw):
    return [dataclasses.replace(s, **kw) for s in _sets(n, 16, 4, 2, 8, seed=seed)]


def _sides():
    left = torch.zeros(2, TOKENS)
    left[:, :4] = 1
    return left, 1 - left


@pytest.mark.parametrize("gate", ["none", "separable", "neuron"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("fast", [True, False])
def test_full_region_equals_no_region_bit_for_bit(gate, iteration, fast, monkeypatch):
    sets = _plain(iteration=iteration)
    if gate != "none":
        sets = _gated(gate, sets)
    ones = [dataclasses.replace(s, region=torch.ones(2, TOKENS)) for s in sets]
    for depth in (1, 3):
        assert torch.equal(
            _run(ones, depth, fast, monkeypatch), _run(sets, depth, fast, monkeypatch)
        )


@pytest.mark.parametrize("gate", ["none", "separable", "neuron"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("fast", [True, False])
def test_zero_region_equals_the_base_output(gate, iteration, fast, monkeypatch):
    sets = _plain(iteration=iteration)
    if gate != "none":
        sets = _gated(gate, sets)
    zero = [dataclasses.replace(s, region=torch.zeros(2, TOKENS)) for s in sets]
    out = _run(zero, 3, fast, monkeypatch)
    assert torch.isfinite(out).all()
    assert torch.equal(out, _y0())


@pytest.mark.parametrize("gate", ["none", "separable"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_fast_path_equals_the_loop_with_regions(gate, shared, carry, monkeypatch):
    sets = _plain(carry=carry)
    if gate != "none":
        sets = _gated(gate, sets)
    soft = torch.rand(2, TOKENS, generator=torch.Generator().manual_seed(5))
    regions = [_sides()[0], soft]
    assert C._fast_path_ok(_y0(), sets, 0.0, None, None)
    for depth in (1, 3, 5):
        monkeypatch.setattr(C, "LOWRANK_ENABLED", True)
        with torch.no_grad(), dropout_key(1, 0.4):
            fast = run_layer(_y0(), _y0(), sets, depth, shared=shared, regions=regions)
        monkeypatch.setattr(C, "LOWRANK_ENABLED", False)
        with torch.no_grad(), dropout_key(1, 0.4):
            loop = run_layer(_y0(), _y0(), sets, depth, shared=shared, regions=regions)
        torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("shared", [False, True])
def test_disjoint_regions_change_only_their_own_tokens(fast, shared, monkeypatch):
    a, b = _plain(seed=4)
    left, right = _sides()
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    with torch.no_grad(), dropout_key(1, 0.4):
        both = run_layer(_y0(), _y0(), [a, b], 3, shared=shared, regions=[left, right])
        only_a = run_layer(_y0(), _y0(), [a, b], 3, shared=shared, regions=[left, 0 * right])
        only_b = run_layer(_y0(), _y0(), [a, b], 3, shared=shared, regions=[0 * left, right])
    y0 = _y0()
    assert not torch.equal(both[:, :4], y0[:, :4])
    assert not torch.equal(both[:, 4:], y0[:, 4:])
    assert torch.equal(only_a[:, 4:], y0[:, 4:])
    assert torch.equal(only_b[:, :4], y0[:, :4])
    if shared:
        return
    torch.testing.assert_close(both[:, :4], only_a[:, :4], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(both[:, 4:], only_b[:, 4:], rtol=1e-6, atol=1e-6)


def test_masked_mean_routing_differs_from_full_mean():
    logits = torch.randn(2, TOKENS, 4, generator=torch.Generator().manual_seed(7)) * 3
    left, right = _sides()
    full = tensor_gate(logits)
    in_left = tensor_gate(logits, region=left[..., None])
    in_right = tensor_gate(logits, region=right[..., None])
    assert not torch.equal(in_left, full) and not torch.equal(in_left, in_right)
    expect = C.sparsemax(logits[:, :4].mean(-2, keepdim=True)).expand_as(logits)
    torch.testing.assert_close(in_left, expect)


def test_regions_route_each_set_by_its_own_tokens():
    (s,) = _plain(n=1, seed=6, routing_scope="tensor")
    left, right = _sides()
    q = _y0() * 3
    with torch.no_grad():
        g_left, g_right = C._gates(
            q, [s, s], False, 0.0, regions=[left[..., None], right[..., None]]
        )
        (g_full,) = C._gates(q, [s], False, 0.0)
    assert not torch.equal(g_left, g_right)
    assert not torch.equal(g_left, g_full)


def test_batch_rows_repeat_for_a_doubled_batch(monkeypatch):
    (s,) = _plain(n=1)
    left, _ = _sides()
    out_one = _run([s], 2, False, monkeypatch, regions=[left[:1]])
    out_two = _run([s], 2, False, monkeypatch, regions=[left])
    assert torch.equal(out_one, out_two)


def test_image_masks_map_to_the_token_grid():
    mask = torch.zeros(1, 64, 96)
    mask[:, :, :48] = 1
    tokens = tokens_from_image(mask, (2, 3))
    assert tokens.tolist() == [[1.0, 0.5, 0.0, 1.0, 0.5, 0.0]]
    padded = tokens_from_image(mask, (2, 3), before=2)
    assert padded.shape == (1, 8) and padded[0, :2].tolist() == [0.0, 0.0]
    assert grid_for(6 * 8, (12, 16)) == (6, 8)
    assert grid_for(3 * 3, (5, 5)) == (3, 3)
    with latent_size((12, 16)):
        assert layer_region(mask, 3 * 4).shape == (1, 12)
    with pytest.raises(ValueError):
        layer_region(mask, 7, (12, 16))


def test_regions_on_activations_with_several_token_axes():
    from kilora.bake import TargetPlan

    plan = TargetPlan.__new__(TargetPlan)
    plan.name = "blocks.0.mlp.layer1"
    mask = torch.zeros(1, 32, 32)
    mask[..., :16] = 1
    plan.regions = [mask]
    h = torch.randn(1, 1, 4, 4, 16, generator=torch.Generator().manual_seed(3))
    with latent_size((1, 4, 4)):
        (region,) = plan._token_regions(h)
    assert region.shape == (1, 1, 4, 4)
    assert torch.equal(region[0, 0], torch.tensor([[1.0, 1, 0, 0]] * 4))
    sets = _plain(1)
    with torch.no_grad(), dropout_key(1, 0.4):
        y = run_layer(h, h, sets, 2, target="layer", regions=[region])
    delta = (y - h).abs().sum(-1)[0, 0]
    assert delta[:, 2:].max() == 0 and delta[:, :2].min() > 0


def _derived_region_set():
    return C.LayerSet(
        form="input",
        import_score="derived",
        router=torch.eye(4),
        keys_in=torch.zeros(2, 1, 4),
        keys_out=torch.zeros(2, 1, 4),
        down=torch.tensor([[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0]]).repeat(2, 1),
        up=torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.1],
            ]
        ),
        per_expert=2,
        gate_mode="none",
        carry="drop",
    )


def _derived_region_inputs():
    x = torch.tensor(
        [[[1.0, 0.7, 0.2, 0.3], [0.7, 1.0, 0.3, 0.2], [0.8, 0.9, 0.2, 0.1], [0.9, 0.8, 0.1, 0.2]]]
    )
    left = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    right = 1 - left
    return x, torch.zeros_like(x), [left, right]


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("engine", ["eager", "ir"])
def test_derived_expert_regions_select_only_the_expert_that_reads(depth, engine, monkeypatch):
    from kilora import ir

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0, masks = _derived_region_inputs()
    routes = []
    with (
        torch.no_grad(),
        C.eager(),
        ir.interpret_all() if engine == "ir" else contextlib.nullcontext(),
        C.recording_shares(lambda _, gates: routes.append(gates[0].clone())),
    ):
        result = run_layer(x, y0, [_derived_region_set()], depth, expert_regions=[masks])
    expected = torch.cat(masks, -1)
    assert len(routes) == depth
    assert all(torch.equal(gate, expected) for gate in routes)
    assert torch.count_nonzero(result[..., :2]) > 0
    assert torch.count_nonzero(result[..., 2:]) > 0


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("width", [1, 3])
def test_derived_region_path_walks_never_start_outside_the_region(depth, width, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "0")
    x, y0, masks = _derived_region_inputs()
    layer = dataclasses.replace(_derived_region_set(), path_width=width)
    routes = []
    original = C._derived_path_note

    def record(stats, gates, logits, active=None):
        routes.append(gates.clone())
        return original(stats, gates, logits, active)

    monkeypatch.setattr(C, "_derived_path_note", record)
    with torch.no_grad(), C.eager():
        run_layer(x, y0, [layer], depth, expert_regions=[masks])
    assert routes
    expected = torch.cat(masks, -1)
    for gate in routes:
        assert torch.equal(gate == 0, expected == 0)
        torch.testing.assert_close(gate, expected, rtol=0, atol=1e-7)


def test_derived_region_scores_preserve_inside_and_fully_masked_tokens(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_POOL", "1,0,0")
    layer = _derived_region_set()
    x, _, _ = _derived_region_inputs()
    before = C.derived_logits(layer, x)
    mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.4], [0.2, 0.8]]])
    masked = dataclasses.replace(layer, expert_mask=mask.repeat_interleave(2, -1))
    after = C.derived_logits(masked, x)
    assert torch.equal(after[:, :2], before[:, :2])
    assert torch.equal(after[:, 3:], before[:, 3:])
    assert after[0, 2, 0] == -1e9
    assert after[0, 2, 1] == before[0, 2, 1]
    assert torch.equal(C.pooled_derived_logits(masked, x, "layer"), after)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("engine", ["eager", "ir"])
def test_derived_output_without_regions_matches_original_scores_bit_for_bit(
    depth, engine, monkeypatch
):
    from kilora import ir

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0, _ = _derived_region_inputs()
    layer = _derived_region_set()
    with (
        torch.no_grad(),
        C.eager(),
        ir.interpret_all() if engine == "ir" else contextlib.nullcontext(),
    ):
        actual = run_layer(x, y0, [layer], depth)
        full = dataclasses.replace(layer, expert_mask=torch.ones(1, 4, 4))
        assert torch.equal(run_layer(x, y0, [full], depth), actual)
        monkeypatch.setattr(C, "derived_logits", C._derived_logits)
        expected = run_layer(x, y0, [layer], depth)
    assert torch.equal(actual, expected)


def test_pooled_derived_regions_do_not_spread_excluded_scores(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0.5,0.25,0.25")
    x, _, masks = _derived_region_inputs()
    layer = _derived_region_set()
    masked = dataclasses.replace(layer, expert_mask=torch.cat(masks, -1).repeat_interleave(2, -1))
    with latent_size((2, 2)):
        before = C.pooled_derived_logits(layer, x, "layer")
        after = C.pooled_derived_logits(masked, x, "layer")
    active = torch.cat(masks, -1).bool()
    assert torch.equal(after[active], before[active])
    assert (after[~active] == -1e9).all()
