# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.ir import default_bindings, default_program
from kilora.program import interpret


def test_accumulate_and_output_match_eager():
    generator = torch.Generator().manual_seed(117)
    previous = torch.randn(2, 3, 4, generator=generator)
    change = torch.randn(2, 3, 4, generator=generator)
    base = torch.randn(2, 3, 5, generator=generator)
    up = torch.randn(4, 5, generator=generator)
    result = interpret(
        default_program("iterate", "accumulate"),
        {"previous_state": previous, "change": change},
        default_bindings("iterate", "accumulate"),
    )
    expected = previous + change
    assert torch.equal(result["state"], expected)
    assert torch.equal(result["output"], expected)
    output = interpret(
        default_program("iterate", "output"),
        {"base": base, "coefficients": result["state"], "up": up},
        default_bindings("iterate", "output"),
    )
    assert torch.equal(output["output"], base + expected @ up)


def test_shared_channel_read_preserves_native_matrix_shapes():
    generator = torch.Generator().manual_seed(1171)
    token = torch.rand(2, 7, 1, generator=generator)
    change = torch.randn(2, 7, 4, generator=generator)
    up = torch.randn(4, 6, generator=generator)
    channel = torch.ones(6)
    target = torch.randn(6, 3, generator=generator)
    base = torch.randn(2, 7, 3, generator=generator)
    actual = interpret(
        default_program("iterate", "gated_read_shared"),
        dict(token=token, change=change, up=up, channel=channel, target=target, base=base),
        {},
    )["result"]
    assert torch.equal(actual, base + (token * change) @ ((up * channel) @ target))


@pytest.mark.parametrize("mask_mode", ["sample", "expect"])
@pytest.mark.parametrize("region", [False, True])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("selection", ["sparsemax", "top1"])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
def test_driver_matches_eager_at_each_depth(
    scope, selection, carry, region, mask_mode, monkeypatch
):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.setenv("KILORA_MASKS", mask_mode)
    generator = torch.Generator().manual_seed(119)
    x = torch.randn(2, 4, 6, generator=generator)
    y0 = torch.randn(2, 4, 6, generator=generator)
    for adaptive in (False, True):
        s = LayerSet(
            router=torch.randn(3, 6, generator=generator),
            keys_in=torch.randn(2, 1, 3, generator=generator),
            per_expert=2,
            down=torch.randn(4, 6, generator=generator) * 0.1,
            up=torch.randn(4, 6, generator=generator) * 0.1,
            gate_mode="adaptive" if adaptive else "none",
            routing_scope=scope,
            routing_select=selection,
            carry=carry,
            region=torch.rand(2, 4, generator=generator) if region else None,
        )
        for depth in (0.5, 1, 2.5, 3, 5):
            with torch.no_grad(), eager(), dropout_key(17, 4):
                expected = run_layer(x, y0, [s], depth, target="test")
                actual = run_layer_ir(x, y0, [s], depth, target="test")
            assert torch.equal(actual, expected), (adaptive, depth, (actual - expected).abs().max())


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("regions", [False, True])
def test_driver_multiple_sets_and_schedule(shared, scope, regions):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    class Schedule:
        is_auto = False

        def depth(self, index, count, kind, progress, key):
            return 2.5

    generator = torch.Generator().manual_seed(433)
    x = torch.randn(2, 5, 6, generator=generator)
    y0 = torch.randn(2, 5, 6, generator=generator)
    sets = [
        LayerSet(
            router=torch.randn(3, 6, generator=generator),
            keys_in=torch.randn(2, 1, 3, generator=generator),
            down=torch.randn(2, 6, generator=generator) * 0.1,
            up=torch.randn(2, 6, generator=generator) * 0.1,
            gate_mode="adaptive" if i else "none",
            routing_scope=scope,
            region=(torch.zeros(2, 5) if i == 0 else torch.rand(2, 5, generator=generator))
            if regions
            else None,
        )
        for i in range(2)
    ]
    with torch.no_grad(), eager(), dropout_key(92, 5):
        expected = run_layer(x, y0, sets, 1, target="two", schedule=Schedule(), shared=shared)
        actual = run_layer_ir(x, y0, sets, 1, target="two", schedule=Schedule(), shared=shared)
    assert torch.equal(actual, expected)


def test_driver_supports_derived_native_loop_options(monkeypatch):
    from kilora import compute
    from kilora.compute import LayerSet
    from kilora.ir.driver import run_layer_ir

    s = LayerSet(
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
    )
    y = torch.ones(1, 2, 3)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    with torch.no_grad(), compute.dropout_key(3, 1):
        expected = compute.run_layer(y, y, [s], 2)
        actual = run_layer_ir(y, y, [s], 2)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("limit", [1, 2])
def test_driver_gate_history_limit(limit):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(973)
    y = torch.randn(2, 4, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
        gate_mode="adaptive",
        gate_head_limit=limit,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 5, target="head")
        actual = run_layer_ir(y, y, [s], 5, target="head")
    assert torch.equal(actual, expected)


def test_driver_fixed_depth_overrides_layer_depth():
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(219)
    y = torch.randn(2, 4, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
        fixed_depth=2.5,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 1, target="fixed")
        actual = run_layer_ir(y, y, [s], 1, target="fixed")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("region", [False, True])
def test_driver_low_precision(dtype, adaptive, region):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(1029)
    y = torch.randn(2, 5, 6, generator=generator).to(dtype)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 6, generator=generator) * 0.1,
        per_expert=2,
        gate_mode="adaptive" if adaptive else "none",
        region=torch.rand(2, 5, generator=generator) if region else None,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 5, target="precision")
        actual = run_layer_ir(y, y, [s], 5, target="precision")
    assert torch.equal(actual, expected)


def test_driver_constant_token_depth():
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(220)
    y = torch.randn(2, 4, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
    )
    depth = torch.full((2, 4), 2.5)
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 1, target="tokens", token_depth=depth)
        actual = run_layer_ir(y, y, [s], 1, target="tokens", token_depth=depth)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("scenario", ["depth", "noise", "both"])
@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_driver_token_depth_and_noise(adaptive, scenario, scope):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(2204)
    y = torch.randn(2, 5, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 6, generator=generator) * 0.1,
        per_expert=2,
        gate_mode="adaptive" if adaptive else "none",
        routing_scope=scope,
    )
    depths = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1.0]])
    kwargs = dict(
        target="tokens",
        token_depth=depths if scenario != "noise" else None,
        noise_scale=0.3 if scenario != "depth" else 0.0,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        torch.manual_seed(301)
        expected = run_layer(y, y, [s], 5, **kwargs)
        torch.manual_seed(301)
        actual = run_layer_ir(y, y, [s], 5, **kwargs)
    assert torch.equal(actual, expected), (actual - expected).abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_driver_mixed_depth_noise_region(dtype, adaptive, carry, mode, monkeypatch):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.setenv("KILORA_MASKS", mode)
    generator = torch.Generator().manual_seed(2205)
    y = torch.randn(2, 5, 6, generator=generator).to(dtype)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 6, generator=generator) * 0.1,
        per_expert=2,
        gate_mode="adaptive" if adaptive else "none",
        carry=carry,
        region=torch.rand(2, 5, generator=generator),
    )
    depths = torch.tensor([[0, 1.5, 2, 4, 5], [5, 4, 3, 2.5, 1]])
    with torch.no_grad(), eager(), dropout_key(17, 4):
        torch.manual_seed(301)
        expected = run_layer(y, y, [s], 5, target="mixed", token_depth=depths, noise_scale=0.3)
        torch.manual_seed(301)
        actual = run_layer_ir(y, y, [s], 5, target="mixed", token_depth=depths, noise_scale=0.3)
    assert torch.equal(actual, expected), (actual - expected).abs().max()


def test_driver_supports_derived_query_policy_and_step_carry():
    from kilora.compute import LayerSet, dropout_key, run_layer, step_carry
    from kilora.ir.driver import run_layer_ir

    s = LayerSet(
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
    )
    y = torch.ones(1, 2, 3)
    with torch.no_grad(), dropout_key(3, 1):
        expected = run_layer(y, y, [s], 2, query_policy=lambda x: x)
        actual = run_layer_ir(y, y, [s], 2, query_policy=lambda x: x)
    assert torch.equal(actual, expected)
    caches = []
    outputs = []
    for run in (run_layer, run_layer_ir):
        cache = {}
        with torch.no_grad(), dropout_key(3, 1), step_carry(cache):
            outputs.append(run(y, y, [s], 2, target="carry"))
        caches.append(cache)
    assert torch.equal(*outputs)
    assert set(caches[0]) == set(caches[1]) == {"carry"}


@pytest.mark.parametrize("selection", ["sparsemax", "top1"])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("shared", [False, True])
def test_driver_noise_with_multiple_sets(selection, scope, shared):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(777)
    y = torch.randn(2, 5, 6, generator=generator)
    sets = [
        LayerSet(
            router=torch.randn(3, 6, generator=generator),
            keys_in=torch.randn(2, 1, 3, generator=generator),
            down=torch.randn(2, 6, generator=generator) * 0.1,
            up=torch.randn(2, 6, generator=generator) * 0.1,
            gate_mode="adaptive" if i else "none",
            routing_scope=scope,
            routing_select=selection,
            region=torch.zeros(2, 5) if i == 0 else torch.rand(2, 5, generator=generator),
        )
        for i in range(2)
    ]
    with torch.no_grad(), eager(), dropout_key(17, 4):
        torch.manual_seed(201)
        expected = run_layer(y, y, sets, 4, target="noise", noise_scale=0.3, shared=shared)
        torch.manual_seed(201)
        actual = run_layer_ir(y, y, sets, 4, target="noise", noise_scale=0.3, shared=shared)
    assert torch.equal(actual, expected), (actual - expected).abs().max()


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_driver_mixed_depth_gate_history_limit(limit, mode, monkeypatch):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.setenv("KILORA_MASKS", mode)
    generator = torch.Generator().manual_seed(1211)
    y = torch.randn(2, 5, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
        gate_mode="adaptive",
        gate_head_limit=limit,
    )
    depths = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 5, target="depth-head", token_depth=depths)
        actual = run_layer_ir(y, y, [s], 5, target="depth-head", token_depth=depths)
    assert torch.equal(actual, expected), (actual - expected).abs().max()


def test_driver_does_not_delegate_routing_or_iteration_to_native(monkeypatch):
    from kilora import compute
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(221)
    y = torch.randn(1, 4, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
        gate_mode="adaptive",
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(y, y, [s], 3, target="parts")

        def forbidden(*args, **kwargs):
            raise AssertionError("the native routing or loop was called")

        for name in ("norm_energy", "out_project", "_run_loop", "_run_lowrank", "_gated_reads"):
            monkeypatch.setattr(compute, name, forbidden)
        actual = run_layer_ir(y, y, [s], 3, target="parts")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("score", ["keys", "read_share"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("fold", ["0", "1"])
def test_imported_driver_parity(score, depth, carry, fold, monkeypatch):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.setenv("KILORA_IMPORT_FOLD", fold)
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    generator = torch.Generator().manual_seed(619)
    x = torch.randn(2, 4, 6, generator=generator)
    y0 = torch.randn(2, 4, 5, generator=generator)
    s = LayerSet(
        form="input",
        import_score=score,
        carry=carry,
        per_expert=2,
        router=torch.randn(3, 5, generator=generator),
        keys_in=torch.randn(2, 2, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 5, generator=generator) * 0.1,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(x, y0, [s], depth, target="import")
        actual = run_layer_ir(x, y0, [s], depth, target="import")
    assert torch.equal(actual, expected), (actual - expected).abs().max()


@pytest.mark.parametrize("score", ["keys", "read_share"])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("mask_mode", ["sample", "expect"])
def test_imported_driver_with_region_and_override(score, scope, mask_mode, monkeypatch):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    monkeypatch.setenv("KILORA_MASKS", mask_mode)
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "0")
    generator = torch.Generator().manual_seed(908)
    x = torch.randn(2, 4, 6, generator=generator)
    y0 = torch.randn(2, 4, 5, generator=generator)
    s = LayerSet(
        form="input",
        import_score="derived",
        routing_scope=scope,
        per_expert=2,
        router=torch.randn(3, 5, generator=generator),
        keys_in=torch.randn(2, 2, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 5, generator=generator) * 0.1,
        region=torch.rand(2, 4, generator=generator),
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(x, y0, [s], 3, target="region")
        actual = run_layer_ir(x, y0, [s], 3, target="region")
    assert torch.equal(actual, expected), (actual - expected).abs().max()


def test_derived_import_default_interpreted_and_nondefault_native(monkeypatch):
    from kilora import compute, fused, ir
    from kilora.ir import driver

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    y = torch.ones(1, 2, 3)
    layer = compute.LayerSet(
        form="input",
        import_score="derived",
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
    )
    calls = []
    original = driver._run

    def record(part, *args, **kwargs):
        calls.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    notes = []
    monkeypatch.setattr(fused, "_note", notes.append)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        expected = compute.run_layer(y, y, [layer], 3, target="derived")
        with ir.interpret_all():
            actual = compute.run_layer(y, y, [layer], 3, target="derived")
        assert torch.equal(actual, expected)
        assert "select" in calls
        monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0.1")
        expected = compute.run_layer(y, y, [layer], 3, target="derived")
        calls.clear()
        with ir.interpret_all():
            actual = compute.run_layer(y, y, [layer], 3, target="derived")
        assert torch.equal(actual, expected)
        assert any("IR driver refused" in note for note in notes)


@pytest.mark.parametrize("depth", [1, 3])
def test_derived_import_native_parity(depth, monkeypatch):
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    generator = torch.Generator().manual_seed(624)
    x = torch.randn(2, 4, 6, generator=generator)
    y = torch.randn(2, 4, 5, generator=generator)
    layer = LayerSet(
        form="input",
        import_score="derived",
        per_expert=2,
        router=torch.randn(3, 5, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 5, generator=generator) * 0.1,
    )
    with torch.no_grad(), eager(), dropout_key(17, 4):
        expected = run_layer(x, y, [layer], depth, target="derived")
        actual = run_layer_ir(x, y, [layer], depth, target="derived")
    assert torch.equal(actual, expected)


def test_imported_driver_refuses_native_loop(monkeypatch):
    from kilora import compute
    from kilora.ir.driver import run_layer_ir

    layer = compute.LayerSet(
        form="input",
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
    )
    y = torch.ones(1, 2, 3)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    with pytest.raises(NotImplementedError, match="imported layer options"):
        run_layer_ir(y, y, [layer], 3)


@pytest.mark.parametrize("case", ["noise", "depth"])
def test_driver_full_width_loop_does_not_delegate(case, monkeypatch):
    from kilora import compute
    from kilora.compute import LayerSet, dropout_key, eager, run_layer
    from kilora.ir.driver import run_layer_ir

    generator = torch.Generator().manual_seed(2210)
    y = torch.randn(1, 5, 6, generator=generator)
    s = LayerSet(
        router=torch.randn(3, 6, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(2, 6, generator=generator) * 0.1,
        up=torch.randn(2, 6, generator=generator) * 0.1,
        gate_mode="adaptive",
    )
    kwargs = {"noise_scale": 0.3} if case == "noise" else {"token_depth": torch.arange(1, 6)[None]}
    with torch.no_grad(), eager(), dropout_key(17, 4):
        torch.manual_seed(221)
        expected = run_layer(y, y, [s], 5, target="parts", **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("the native routing or loop was called")

        for name in ("norm_energy", "out_project", "_run_loop", "_run_lowrank", "adaptive_keep"):
            monkeypatch.setattr(compute, name, forbidden)
        torch.manual_seed(221)
        actual = run_layer_ir(y, y, [s], 5, target="parts", **kwargs)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("form", ["state", "input"])
def test_driver_refuses_candidate_selection(form):
    from kilora import compute
    from kilora.ir.driver import run_layer_ir

    layer = compute.LayerSet(
        form=form,
        candidates=1,
        router=torch.ones(2, 3),
        keys_in=torch.ones(2, 1, 2),
        down=torch.ones(2, 3),
        up=torch.ones(2, 3),
    )
    y = torch.ones(1, 2, 3)
    if form == "input":
        with torch.no_grad(), pytest.raises(NotImplementedError):
            run_layer_ir(y, y, [layer], 3)
    else:
        with torch.no_grad(), compute.dropout_key(4, 2):
            expected = compute.run_layer(y, y, [layer], 3)
            actual = run_layer_ir(y, y, [layer], 3)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_derived_import_freeze_interpreter_parity(iteration, monkeypatch):
    from kilora import compute, ir, route_freeze
    from kilora.ir import driver

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "sample")
    generator = torch.Generator().manual_seed(812)
    x = torch.randn(2, 4, 6, generator=generator)
    y = torch.randn(2, 4, 5, generator=generator)
    layer = compute.LayerSet(
        form="input",
        import_score="derived",
        iteration=iteration,
        route_freeze=0.4,
        gate_mode="adaptive",
        gate_kappa=0.5,
        per_expert=2,
        router=torch.randn(3, 5, generator=generator),
        keys_in=torch.randn(2, 1, 3, generator=generator),
        down=torch.randn(4, 6, generator=generator) * 0.1,
        up=torch.randn(4, 5, generator=generator) * 0.1,
    )
    parts = []
    original = driver._run

    def record(part, *args, **kwargs):
        parts.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        with compute.sampling_progress(0.6), route_freeze.freezing({}) as cache:
            expected = [compute.run_layer(x, y, [layer], 3, target="freeze") for _ in range(2)]
            assert "freeze" in cache
        with compute.sampling_progress(0.6), route_freeze.freezing({}) as cache:
            with ir.interpret_all():
                actual = [compute.run_layer(x, y, [layer], 3, target="freeze") for _ in range(2)]
            assert "freeze" in cache
    assert {"select", "gate", "freeze"} <= set(parts)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("depth", [1, 3])
def test_import_ir_regions_select_the_expert_that_can_read(depth, monkeypatch):
    from test_region import _derived_region_inputs, _derived_region_set

    from kilora import compute
    from kilora.ir import driver

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0, masks = _derived_region_inputs()
    layer = _derived_region_set()
    selected = []
    original = driver._run

    def record(part, *args, **kwargs):
        result = original(part, *args, **kwargs)
        if part == "select" and "weights" in result:
            selected.append(result["weights"].clone())
        return result

    monkeypatch.setattr(driver, "_run", record)
    with torch.no_grad(), compute.eager():
        expected = compute.run_layer(x, y0, [layer], depth, expert_regions=[masks])
        actual = driver.run_layer_ir(x, y0, [layer], depth, expert_regions=[masks])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert len(selected) == depth
    assert all(torch.equal(gates, torch.cat(masks, -1)) for gates in selected)
