# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute, route_freeze
from kilora.bake import ModulePlan, TargetPlan
from kilora.comfyui.patching import _key_mask_by_step


@pytest.fixture(autouse=True)
def full_derived_mode(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")


def test_fold_counts_distinguish_recording_and_replay():
    counts = {}
    with route_freeze.fold_counts(counts):
        route_freeze.note_fold(False)
        route_freeze.note_fold(True)
        route_freeze.note_fold(True, replay=True)
    assert counts == {"recording": 1, "folded": 1, "replayed": 1}


def test_nofold_replay_is_counted_separately(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MACRO", "0")
    monkeypatch.setenv("KILORA_DERIVED_FREEZE_NOFOLD", "1")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    x = torch.ones(1, 3, 2)
    counts = {}
    with (
        torch.no_grad(),
        route_freeze.freezing({}),
        route_freeze.fold_counts(counts),
        compute.sampling_progress(0.5),
    ):
        for step in (1, 2):
            with compute.dropout_key(4, step), compute.derived_call():
                compute.run_layer(x, x, [s], 3, target="layer")
    assert counts == {"recording": 1, "replayed": 1}


def test_bfloat16_fold_product_accumulates_in_float32(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    x = torch.tensor([[[0.7, 0.3], [0.4, 0.8]]], dtype=torch.bfloat16)
    y = torch.zeros_like(x)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(4, 1), compute.derived_call():
            compute.run_layer(x, y, [s], 3, target="layer")
        saved = cache[("derived_macro", "layer", 0)]
        coef = compute._input_read(s, x, y.dtype).float()
        coef = coef * saved["weight"].repeat_interleave(s.per_expert, -1) * s.scale
        expected = y + (coef @ s.up.float()).to(y.dtype)
        with compute.dropout_key(7, 2), compute.derived_call():
            actual = compute.run_layer(x, y, [s], 3, target="layer")
    assert torch.equal(actual, expected)


def test_comfy_wrapper_folds_inference_mode_weights(monkeypatch, caplog):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    plan = ModulePlan([(0, -1, TargetPlan(sets=[s], depth=2, name="layer"))])

    class Model:
        model_options = {}

    model = Model()
    _key_mask_by_step(model)
    wrapper = model.model_options["model_function_wrapper"]
    x = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    sigmas = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])

    def apply(x, t, **kwargs):
        return plan.run(x, x)

    with caplog.at_level("INFO", logger="kilora"), torch.inference_mode():
        for sigma in sigmas[:-1]:
            args = {
                "input": x,
                "timestep": sigma[None],
                "c": {"transformer_options": {"sample_sigmas": sigmas, "cond_or_uncond": [0]}},
            }
            wrapper(apply, args)
    assert "1 folded, 0 replayed unfolded, 1 recording calls" in caplog.text
    moved = plan._on(x.device, x.dtype).parts[0][2].sets[0]
    assert compute._tensor_key("macro", [moved.down, moved.up]) is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_derived_logits_match_float_reference(monkeypatch, dtype):
    torch.manual_seed(42)
    s = dataclasses.replace(planted(), down=torch.randn(8, 64), up=torch.randn(8, 64), per_expert=4)
    x = torch.randn(3, 7, 64).to(dtype)
    kin, rows, _, _, _ = compute.derived_keys(s)
    share = compute.norm_energy(x.float(), kin, rows.clamp_min(1)) * rows / kin.shape[-1]
    rank = rows.float()
    width = kin.shape[-1]
    variance = 2 * rank * (width - rank) / (width * width * (width + 2))
    expected = s.tau * (share - rank / width) / variance.sqrt()
    torch.testing.assert_close(compute.derived_logits(s, x), expected, rtol=0.012, atol=0.004)


@pytest.mark.parametrize("depth", [1, 2, 3.5, 7])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("fused", [False, True])
def test_frozen_paths_match_recorded_output(monkeypatch, depth, iteration, fused):
    if fused:
        pytest.importorskip("triton")
        import os

        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires Triton interpreter")
        monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1" if fused else "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MACRO", "1")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "0.5")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = dataclasses.replace(planted(), path_width=2, iteration=iteration, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]])
    y = torch.tensor([[[0.2, -0.3], [0.1, 0.4], [0.8, -0.5]]])
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(4, 1), compute.derived_call():
            expected = compute.run_layer(x, y, [s], depth, target="layer")
        assert ("derived_macro", "layer", 0) in cache

        def no_loop(*args, **kwargs):
            raise AssertionError("frozen path ran the depth loop")

        monkeypatch.setattr(compute, "_run_input_fused", no_loop)
        with compute.dropout_key(9, 2), compute.derived_call():
            actual = compute.run_layer(x, y, [s], depth, target="layer")
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_macro_disabled_and_keys_mode_fall_back(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    x = torch.ones(1, 3, 2)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        for switch in ("off", "keys"):
            monkeypatch.setenv("KILORA_DERIVED_MACRO", "0" if switch == "off" else "1")
            monkeypatch.setenv("KILORA_DERIVED_MODE", "keys" if switch == "keys" else "full")
            with compute.dropout_key(1, 1):
                compute.run_layer(x, x, [s], 2, target="layer")
            assert ("derived_macro", "layer", 0) not in cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("depth", [2, 3.5])
def test_default_frozen_fold_matches_recording(monkeypatch, dtype, gate, depth):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.delenv("KILORA_DERIVED_MACRO", raising=False)
    s = dataclasses.replace(planted(), gate_mode=gate, path_width=2, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.5], [0.25, 1.0]]], dtype=dtype)
    y = torch.tensor([[[2.0, -1.0], [1.0, 2.0]]], dtype=dtype)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(4, 1), compute.derived_call():
            expected = compute.run_layer(x, y, [s], depth, target="layer")
        assert ("derived_macro", "layer", 0) in cache
        with compute.dropout_key(7, 2), compute.derived_call():
            actual = compute.run_layer(x, y, [s], depth, target="layer")
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-4, atol=1e-5)


def test_carry_before_freeze_and_disabled_after(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "0.5")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "1")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    x = torch.ones(1, 3, 2)
    priors = []
    original = compute.derived_entries

    def observe(*args, **kwargs):
        priors.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "derived_entries", observe)
    with torch.no_grad(), route_freeze.freezing({}), compute.derived_call():
        with compute.sampling_progress(0.2), compute.dropout_key(1, 1):
            compute.run_layer(x, x, [s], 2, target="one")
            compute.run_layer(x, x, [s], 2, target="two", index=1)
        with compute.sampling_progress(0.5), compute.dropout_key(1, 2):
            compute.run_layer(x, x, [s], 2, target="three")
    assert priors[0] is None and priors[1] is not None and priors[2] is None


def test_score_off_is_unchanged(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    s = dataclasses.replace(planted(), route_freeze=None)
    x = torch.tensor([[[1.0, 0.5], [0.25, 1.0]]])
    with torch.no_grad(), compute.dropout_key(1, 1):
        monkeypatch.setenv("KILORA_DERIVED_MACRO", "0")
        baseline = compute.run_layer(x, x, [s], 3)
        monkeypatch.delenv("KILORA_DERIVED_MACRO")
        actual = compute.run_layer(x, x, [s], 3)
    assert torch.equal(actual, baseline)


def test_frozen_fold_one_read_one_write(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountProducts(TorchDispatchMode):
        def __init__(self):
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
                self.count += 1
            return func(*args, **(kwargs or {}))

    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), gate_mode="adaptive", path_width=2, route_freeze=0.4)
    x = torch.ones(1, 3, 2)
    original_gate = compute.split_gate

    def agreeing_gate(*args, **kwargs):
        a, b = original_gate(*args, **kwargs)
        return a, torch.ones_like(b)

    monkeypatch.setattr(compute, "split_gate", agreeing_gate)
    with torch.no_grad(), route_freeze.freezing({}), compute.sampling_progress(0.5):
        with compute.dropout_key(1, 1):
            compute.run_layer(x, x, [s], 3, target="layer")
        counter = CountProducts()
        with counter:
            compute.run_layer(x, x, [s], 3, target="layer")
    assert counter.count == 2


@pytest.mark.parametrize("agree", [False, True])
@pytest.mark.parametrize("depth", [3, 3.5])
def test_adaptive_fold_stores_gates_and_matches_scaled_weights(monkeypatch, agree, depth):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), gate_mode="adaptive", path_width=2, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]])
    y = torch.tensor([[[0.2, -0.3], [0.1, 0.4], [0.8, -0.5]]])
    original_gate = compute.split_gate
    calls = []

    def controlled_gate(*args, **kwargs):
        a, b = original_gate(*args, **kwargs)
        calls.append(1)
        value = 1.0 if agree else 0.5 + 0.03 * len(calls)
        return a, torch.full_like(b, value)

    monkeypatch.setattr(compute, "split_gate", controlled_gate)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(4, 1), compute.derived_call():
            compute.run_layer(x, y, [s], depth, target="layer")
        saved = cache[("derived_macro", "layer", 0)]
        assert "up" not in saved
        cached_tensors = [saved["gates"], *saved["weight"], *saved["output_gates"]]
        assert all(t.shape[-2:] != s.up.shape for t in cached_tensors)
        gates = saved["output_gates"]
        assert len(gates) == (1 if agree else 2 * (2 if depth % 1 else 1))
        assert all(b.shape == (1, 1, s.up.shape[-1]) for b in gates)
        coef = compute._input_read(s, x, y.dtype).float()
        old_coef = torch.cat([coef * w * s.scale for w in saved["weight"]], dim=-1)
        old_up = torch.cat([s.up.float() * b.float() for b in gates], dim=-2).to(y.dtype)
        expected = y + old_coef.to(y.dtype) @ old_up
        with compute.dropout_key(7, 2), compute.derived_call():
            actual = compute.run_layer(x, y, [s], depth, target="layer")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("fused", [False, True])
def test_freeze_without_fold_replays_paths(monkeypatch, dtype, gate, fused):
    if fused:
        pytest.importorskip("triton")
        import os

        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires Triton interpreter")
        monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1" if fused else "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = dataclasses.replace(planted(), gate_mode=gate, path_width=2, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]], dtype=dtype)
    y = torch.tensor([[[0.2, -0.3], [0.1, 0.4], [0.8, -0.5]]], dtype=dtype)
    outputs = []
    for nofold in (False, True):
        monkeypatch.setenv("KILORA_DERIVED_MACRO", "0" if nofold else "1")
        monkeypatch.setenv("KILORA_DERIVED_FREEZE_NOFOLD", "1" if nofold else "0")
        with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
            with compute.dropout_key(4, 1), compute.derived_call():
                compute.run_layer(x, y, [s], 3, target="layer")
            assert ("derived_nofold" if nofold else "derived_macro", "layer", 0) in cache
            with compute.dropout_key(9, 2), compute.derived_call():
                outputs.append(compute.run_layer(x * 0.7, y, [s], 3, target="layer"))
    tol = (0.01, 0.008) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(outputs[0].float(), outputs[1].float(), rtol=tol[0], atol=tol[1])


def test_nofold_inactive_before_freeze_and_when_disabled(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MACRO", "0")
    s = dataclasses.replace(planted(), path_width=2, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]])
    for progress in (0.2, 0.5):
        results = []
        for setting in ("0", "1" if progress < 0.4 else "0"):
            monkeypatch.setenv("KILORA_DERIVED_FREEZE_NOFOLD", setting)
            with (
                torch.no_grad(),
                route_freeze.freezing({}) as cache,
                compute.sampling_progress(progress),
            ):
                with compute.dropout_key(4, 1), compute.derived_call():
                    results.append(compute.run_layer(x, x, [s], 3, target="layer"))
                assert not cache
        assert torch.equal(*results)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("depth", [3.5, 4])
def test_batched_walks_match_per_path_loop(
    monkeypatch, dtype, gate, fused, later, scope, iteration, depth
):
    if fused:
        pytest.importorskip("triton")
        import os

        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires Triton interpreter")
        monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1" if fused else "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = dataclasses.replace(
        planted(),
        gate_mode=gate,
        path_width=2,
        route_freeze=0.4,
        routing_scope=scope,
        iteration=iteration,
        scale=0.65,
    )
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]], dtype=dtype)
    y = torch.tensor([[[0.2, -0.3], [0.1, 0.4], [0.8, -0.5]]], dtype=dtype)
    batched = compute._derived_batched_paths

    def loop(s, x, y, region, target, entries, depth):
        from kilora.path_walks import select_weights

        states, paths = [], []
        for entry in entries:
            stats = {}
            paths.append(
                compute._run_input_fused(
                    x, y, [s], [0], depth, [region], "layer", None, entry, stats
                )
            )
            stats.setdefault("converged", torch.zeros_like(stats["score"], dtype=torch.bool))
            states.append(stats)
        weights = select_weights(states, s.path_select)
        change = sum((p - y) * weights[..., n : n + 1].to(p.dtype) for n, p in enumerate(paths))
        return change, states, weights

    outputs = []
    for implementation in (loop, batched):
        monkeypatch.setattr(compute, "_derived_batched_paths", implementation)
        with (
            torch.no_grad(),
            compute.derived_call(),
            compute.sampling_progress(0.2),
            compute.dropout_key(4, 1),
        ):
            region = [torch.tensor([[1.0, 0.3, 0.0]])] if scope == "tensor" else None
            if later:
                compute.run_layer(x, y, [s], depth, target="layer", index=0, regions=region)
            outputs.append(
                compute.run_layer(
                    x, y, [s], depth, target="layer", index=int(later), regions=region
                )
            )
    tol = (0.03, 0.02) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(outputs[0].float(), outputs[1].float(), rtol=tol[0], atol=tol[1])


@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_batched_one_read_one_write_and_up_cache(monkeypatch, gate_mode):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(
        planted(), up=planted().up.to(torch.bfloat16), gate_mode=gate_mode, path_width=2
    )
    x = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    entries = [torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]])]
    original_read = compute._input_read
    original_gate = compute.split_gate
    original_matmul = torch.Tensor.__matmul__
    counts = {"read": 0, "write": 0}

    def read(*args):
        counts["read"] += 1
        return original_read(*args)

    def gate(*args, **kwargs):
        a, b = original_gate(*args, **kwargs)
        return a, torch.ones_like(b)

    up = compute._as(s.up, torch.float32)

    def matmul(a, b):
        if b is up:
            counts["write"] += 1
        return original_matmul(a, b)

    monkeypatch.setattr(compute, "_input_read", read)
    monkeypatch.setattr(compute, "split_gate", gate)
    monkeypatch.setattr(torch.Tensor, "__matmul__", matmul)
    with torch.no_grad():
        compute._derived_batched_paths(s, x, x, None, "layer#input0", entries, 3)
        assert counts == {"read": 1, "write": 1}
        assert compute._as(s.up, torch.float32) is up
        s.up.add_(1)
        assert compute._as(s.up, torch.float32) is not up


def test_batched_reuses_up_conversion(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountCasts(TorchDispatchMode):
        def __init__(self, weight):
            self.weight = weight
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._to_copy.default and args[0] is self.weight:
                self.count += 1
            return func(*args, **(kwargs or {}))

    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), up=planted().up.to(torch.bfloat16), gate_mode="adaptive")
    x = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    entries = [torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]])]
    with torch.no_grad():
        compute.derived_keys(s)
        counter = CountCasts(s.up)
        with counter:
            for _ in range(2):
                compute._derived_batched_paths(s, x, x, None, "layer#input0", entries, 2)
            assert counter.count <= 4
            s.up.add_(1)
            compute._derived_batched_paths(s, x, x, None, "layer#input0", entries, 2)
        assert counter.count <= 6


def test_frozen_fold_reuses_up_conversion(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountCasts(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._to_copy.default and args[0] is s.up:
                self.count += 1
            return func(*args, **(kwargs or {}))

    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), up=planted().up.to(torch.bfloat16), route_freeze=0.4)
    x = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    with torch.no_grad(), route_freeze.freezing({}), compute.sampling_progress(0.5):
        with compute.dropout_key(1, 1):
            compute.run_layer(x, x, [s], 3, target="layer")
        key = compute._tensor_key(f"as-{torch.float32}", [s.up])
        compute._DERIVED.pop(key[0], None)
        count = CountCasts()
        with count:
            for _ in range(2):
                compute.run_layer(x, x, [s], 3, target="layer")
        assert count.count <= 1


def test_batched_distinct_channel_gates_write_separately(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = dataclasses.replace(planted(), gate_mode="adaptive", path_width=2)
    x = torch.ones(1, 3, 2)
    entries = [torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]])]
    original_gate = compute.split_gate
    original_matmul = torch.Tensor.__matmul__
    up = compute._as(s.up, torch.float32)
    calls = {"gates": 0, "writes": 0}

    def gate(*args, **kwargs):
        a, b = original_gate(*args, **kwargs)
        calls["gates"] += 1
        return a, torch.full_like(b, 0.5 if calls["gates"] % 2 else 1.0)

    def matmul(a, b):
        if b is up:
            calls["writes"] += 1
        return original_matmul(a, b)

    monkeypatch.setattr(compute, "split_gate", gate)
    monkeypatch.setattr(torch.Tensor, "__matmul__", matmul)
    with torch.no_grad():
        compute._derived_batched_paths(s, x, x, None, "layer#input0", entries, 3)
    assert calls["writes"] == 2


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "keys",
            [
                [
                    [1.3538450002670288, -0.18461239337921143],
                    [3.0999999046325684, 0.4000000059604645],
                    [0.8494386076927185, 1.5567412376403809],
                ]
            ],
        ),
        (
            "read_share",
            [
                [
                    [0.365517258644104, 2.2862069606781006],
                    [2.8522934913635254, 0.4743119478225708],
                    [2.159292221069336, 0.41061943769454956],
                ]
            ],
        ),
    ],
)
def test_other_import_modes_match_starting_commit(monkeypatch, mode, expected):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", mode)
    s = planted()
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]])
    y = torch.tensor([[[0.2, -0.3], [0.1, 0.4], [0.8, -0.5]]])
    with torch.no_grad(), compute.dropout_key(4, 1):
        actual = compute.run_layer(x, y, [s], 3, target="layer")
    assert torch.equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_batched_fused_preserves_token_entries(monkeypatch, scope):
    import os

    pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter")
    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    s = dataclasses.replace(planted(), routing_scope=scope)
    reads = torch.tensor([[[[1.0, 0.4], [0.3, 1.0], [0.7, 0.8]]]])
    entry = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    region = torch.tensor([[[1.0], [0.3], [0.0]]])
    gates, actual = fused.derived_batched_step(s, reads, entry * 20, region, None, entry=entry)
    assert torch.equal(actual, reads * entry)
    normalized = entry / entry.sum(-1, keepdim=True)
    expected = normalized if scope == "token" else normalized[..., :1, :].expand_as(entry)
    assert torch.equal(gates, expected)


def test_batched_fused_launches_once_per_step(monkeypatch):
    import os

    pytest.importorskip("triton")
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter")
    from kilora import fused

    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "0")
    s = dataclasses.replace(planted(), path_width=2)
    x = torch.ones(1, 3, 2)
    original = fused.derived_batched_step
    calls = []

    def step(*args, **kwargs):
        calls.append(args[2].shape[0])
        return original(*args, **kwargs)

    def no_loop(*args, **kwargs):
        raise AssertionError("per-path loop used")

    monkeypatch.setattr(fused, "derived_batched_step", step)
    monkeypatch.setattr(compute, "_run_input_fused", no_loop)
    with torch.no_grad(), compute.dropout_key(1, 1):
        compute.run_layer(x, x, [s], 3, target="layer")
    assert calls == [2, 2, 2]


def test_derived_batched_switch_uses_reference_loop(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    s = dataclasses.replace(planted(), path_width=2)
    x = torch.ones(1, 3, 2)

    def fail(*args, **kwargs):
        raise AssertionError("batched path used with switch off")

    monkeypatch.setattr(compute, "_derived_batched_paths", fail)
    with torch.no_grad(), compute.dropout_key(4, 1):
        result = compute.run_layer(x, x, [s], 3, target="layer")
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("mode", ["keys", "read_share", "derived"])
def test_chain_freeze_replays_routes_and_keeps_the_bound(monkeypatch, dtype, gate, mode):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", mode)
    s = dataclasses.replace(planted(), gate_mode=gate, path_width=2, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.4], [0.3, 1.0]]], dtype=dtype)
    y = torch.zeros_like(x)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(4, 1), compute.derived_call():
            first = compute.run_layer(x, y, [s], 3, target="chain")
        assert cache
        with compute.dropout_key(9, 2), compute.derived_call():
            replay = compute.run_layer(x, y, [s], 3, target="chain")
        torch.testing.assert_close(replay, first, rtol=1e-4, atol=1e-5)
        if mode == "derived":
            saved = cache[("derived_macro", "chain", 0)]
            assert saved["chain"] and len(saved["paths"]) == 2
            assert all(len(path["read_keeps"]) == 2 for path in saved["paths"])
        else:
            assert len(cache["chain"]["chain_keeps"]) == 2
