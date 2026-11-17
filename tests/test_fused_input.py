# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
import threading

import pytest
import torch

pytest.importorskip("triton")

from test_fused import DEVICE, ROUTES, _routing, _set, _to

from kilora import compute, fused
from kilora.compute import LayerSet, run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)
TARGET = "blk.0.attn1"


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")


def _imported(seed, experts=4, per_expert=2, width=24, n_in=20, router_dim=8, **kw):
    gen = torch.Generator().manual_seed(seed)
    cols = experts * per_expert
    kw.setdefault("gate_mode", "adaptive")
    s = LayerSet(
        router=torch.randn(router_dim, width, generator=gen) / width**0.5,
        keys_in=torch.randn(experts, 3, router_dim, generator=gen),
        keys_out=torch.randn(experts, 2, router_dim, generator=gen),
        per_expert=per_expert,
        down=torch.randn(cols, n_in, generator=gen) / n_in**0.5,
        up=torch.randn(cols, width, generator=gen) * 0.6 / cols**0.5,
        form="input",
        **_routing(kw),
    )
    return _to(s, DEVICE)


def _inputs(seed, batch=2, tokens=37, width=24, n_in=20):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, tokens, n_in, generator=gen).to(DEVICE)
    y0 = torch.randn(batch, tokens, width, generator=gen).to(DEVICE)
    return x, y0


def _run(monkeypatch, on, x, y0, sets, depth, regions=None):
    monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
    seen = []
    real = fused.run_fused
    real_step = fused.derived_batched_step

    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            seen.append(1)
        return real(*a, **k)

    def spy_step(*a, **k):
        if threading.get_ident() == me:
            seen.append(1)
        return real_step(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    monkeypatch.setattr(fused, "derived_batched_step", spy_step)
    with torch.no_grad(), compute.dropout_key(7, 3):
        y = run_layer(x, y0, sets, depth, target=TARGET, regions=regions)
    return y, bool(seen)


def _same(monkeypatch, x, y0, sets, depth, regions=None, fused_expected=True):
    eager, used = _run(monkeypatch, False, x, y0, sets, depth, regions)
    assert not used
    got, used = _run(monkeypatch, True, x, y0, sets, depth, regions)
    assert used == fused_expected
    assert torch.allclose(eager.float(), got.float(), rtol=1e-4, atol=1e-5), (
        (eager - got).abs().max()
    )
    return eager, got


@pytest.mark.parametrize("routing", list(ROUTES))
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_imported_set_matches_the_eager_loop(routing, depth, gate, carry, monkeypatch):
    x, y0 = _inputs(1)
    s = _imported(2, routing=routing, gate_mode=gate, carry=carry)
    eager, _ = _same(monkeypatch, x, y0, [s], depth)
    assert not torch.equal(eager, y0)


@pytest.mark.parametrize("masks", ["sample", "expect"])
def test_imported_set_masks_match(masks, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", masks)
    x, y0 = _inputs(3)
    s = _imported(4, routing="token")
    _same(monkeypatch, x, y0, [s], 3)


@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_imported_set_with_a_region(routing, monkeypatch):
    x, y0 = _inputs(5)
    s = _imported(6, routing=routing)
    region = (torch.arange(37) < 20).float()[None].expand(2, 37).to(DEVICE)
    _same(monkeypatch, x, y0, [s], 3, regions=[region])


def test_mixed_layer_runs_eager(monkeypatch):
    x, y0 = _inputs(7)
    imported = _imported(8)
    from test_fused import _set

    state = dataclasses.replace(_set(9), gate_mode="adaptive")
    _same(monkeypatch, x, y0, [state, imported], 3, fused_expected=False)


@pytest.mark.parametrize("depth", [1.5, 2.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_imported_set_at_a_fractional_depth(depth, gate, routing, monkeypatch):
    x, y0 = _inputs(10)
    _same(monkeypatch, x, y0, [_imported(11, gate_mode=gate, routing=routing)], depth)


@pytest.mark.parametrize("policy", ["auto", "reference"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_imported_set_with_token_roles(policy, routing, monkeypatch):
    from kilora.family_spec import TokenLayout
    from kilora.roles import role_mask

    x, y0 = _inputs(8)
    s = _imported(9, routing=routing)
    layout = TokenLayout("image2d", (1, 1))
    mask = role_mask(policy, layout, 37, (5, 5), [(3, 4)])
    eager, got = _same(monkeypatch, x, y0, [s], 2, regions=[mask.to(DEVICE)])
    off = (mask[0] == 0).to(y0.device)
    for y in (eager, got):
        assert torch.equal(y[:, off], y0[:, off])
        assert (y[:, ~off] - y0[:, ~off]).abs().sum(-1).min() > 0


@pytest.mark.parametrize("form", ["input", "state"])
@pytest.mark.parametrize("score", ["keys", "read_share"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("gate", ["none", "adaptive", "adaptive_kernel"])
@pytest.mark.parametrize("depth", [1, 2.5])
def test_route_freeze_on_the_fused_path(monkeypatch, form, score, routing, gate, depth):
    from kilora import route_freeze as R
    from kilora.compute import sampling_progress

    x, y0 = _inputs(12, tokens=7)
    make = _imported if form == "input" else _set
    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "1" if gate == "adaptive_kernel" else "0")
    mode = "adaptive" if gate == "adaptive_kernel" else gate
    s = dataclasses.replace(make(13, routing=routing, gate_mode=mode), route_freeze=0.5)
    region = torch.linspace(0.2, 1, y0.shape[-2], device=DEVICE).expand(y0.shape[0], -1)
    real = fused.run_fused
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fused, "run_fused", spy)
    entries, outputs = {}, {}
    for on in (False, True):
        monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
        cache = {}
        ys = []
        for progress in (0.5, 0.8):
            with (
                torch.no_grad(),
                compute.dropout_key(7, progress),
                sampling_progress(progress),
                R.freezing(cache),
            ):
                ys.append(run_layer(x, y0, [s], depth, target=TARGET, regions=[region]))
        entries[on], outputs[on] = cache[TARGET], ys
        assert bool(calls) == on
    for key in ("scale", "up"):
        torch.testing.assert_close(entries[True][key], entries[False][key], rtol=1e-4, atol=1e-5)
    for eager, actual in zip(outputs[False], outputs[True], strict=True):
        torch.testing.assert_close(actual, eager, rtol=1e-4, atol=1e-5)


def test_route_freeze_multiple_imports_remain_eager(monkeypatch):
    from kilora import route_freeze as R

    x, y0 = _inputs(20, tokens=7)
    sets = [dataclasses.replace(_imported(i), route_freeze=0.0) for i in (21, 22)]
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setattr(fused, "run_fused", lambda *a, **k: pytest.fail("unexpected fused call"))
    cache = {}
    with torch.no_grad(), R.freezing(cache):
        run_layer(x, y0, sets, 2, target=TARGET)
    assert len(cache[TARGET]["downs"]) == 2


@pytest.mark.parametrize("n_in,width", [(96, 24), (56, 24), (24, 96)])
@pytest.mark.parametrize("per_expert", [4, 8])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
def test_imported_set_with_other_input_and_output_widths(
    n_in, width, per_expert, routing, gate, monkeypatch
):
    x, y0 = _inputs(14, width=width, n_in=n_in)
    s = _imported(
        15, per_expert=per_expert, width=width, n_in=n_in, routing=routing, gate_mode=gate
    )
    for depth in (1, 3):
        _same(monkeypatch, x, y0, [s], depth)


def test_imported_sets_of_two_ranks_at_two_layers(monkeypatch):
    a_x, a_y = _inputs(16, width=16, n_in=64)
    b_x, b_y = _inputs(17, width=64, n_in=16)
    a = _imported(18, experts=2, per_expert=16, width=16, n_in=64)
    b = _imported(19, experts=2, per_expert=32, width=64, n_in=16)
    for _ in range(2):
        _same(monkeypatch, a_x, a_y, [a], 3)
        _same(monkeypatch, b_x, b_y, [b], 3)


def test_warm_up_compiles_an_imported_set():
    t = _imported(20, per_expert=4)
    assert fused.warm_up(DEVICE, templates=[(t, 1), (t, 3)], dtypes=(torch.float32,)) is True
    assert not compute._STEP_CARRY


@pytest.mark.parametrize("routing", ["token", "tensor", "token_top1"])
@pytest.mark.parametrize("depth", [1, 3, 2.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_imported_set_in_path_mode(routing, depth, gate, carry, monkeypatch):
    x, y0 = _inputs(21)
    s = _imported(22, routing=routing, gate_mode=gate, carry=carry, iteration="path")
    eager, _ = _same(monkeypatch, x, y0, [s], depth)
    assert not torch.equal(eager, y0)


def test_imported_set_with_several_walks_runs_eager(monkeypatch):
    x, y0 = _inputs(23)
    s = _imported(24, iteration="path", path_width=2)
    _same(monkeypatch, x, y0, [s], 3, fused_expected=False)


def _expert_mask(experts, per_expert, batch=2, tokens=37, ones_for=()):
    t = torch.arange(tokens)
    cols = []
    for e in range(experts):
        m = torch.ones(tokens) if e == 0 or e in ones_for else (t % experts == e).float()
        cols.append(m[:, None].expand(tokens, per_expert))
    return torch.cat(cols, -1)[None].expand(batch, tokens, experts * per_expert).to(DEVICE)


@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("depth", [1, 3, 2.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
def test_imported_set_with_per_lora_regions(routing, depth, gate, monkeypatch):
    x, y0 = _inputs(25)
    s = _imported(26, routing=routing, gate_mode=gate)
    s = dataclasses.replace(s, expert_mask=_expert_mask(4, s.per_expert))
    eager, _ = _same(monkeypatch, x, y0, [s], depth)
    plain, _ = _run(monkeypatch, False, x, y0, [dataclasses.replace(s, expert_mask=None)], depth)
    assert not torch.equal(eager, plain)


def test_per_lora_regions_with_a_set_region_and_path_mode(monkeypatch):
    x, y0 = _inputs(27)
    s = _imported(28, iteration="path")
    s = dataclasses.replace(s, expert_mask=_expert_mask(4, s.per_expert, ones_for=(2,)))
    region = (torch.arange(37) < 25).float()[None].expand(2, 37).to(DEVICE)
    _same(monkeypatch, x, y0, [s], 3, regions=[region])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("variant", ["accumulate", "path", "expert_mask"])
def test_imported_set_in_a_low_precision_model(dtype, routing, variant, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    worst = 0.0
    for seed in range(8):
        x, y0 = _inputs(30 + seed, tokens=64)
        x, y0 = x.to(dtype), y0.to(dtype)
        s = _imported(40 + seed, routing=routing)
        if variant == "path":
            s = dataclasses.replace(s, iteration="path")
        if variant == "expert_mask":
            mask = _expert_mask(4, s.per_expert, tokens=64).to(dtype)
            s = dataclasses.replace(s, expert_mask=mask)
        eager, _ = _run(monkeypatch, False, x, y0, [s], 3)
        got, used = _run(monkeypatch, True, x, y0, [s], 3)
        assert used
        rel = float((eager.float() - got.float()).norm() / ((eager.float() - y0.float()).norm()))
        worst = max(worst, rel)
    assert worst < 0.005, worst


def test_parallel_axes_on_the_fused_path(monkeypatch):
    x, y0 = _inputs(29)
    creative = _imported(30, routing="token")
    functional = dataclasses.replace(
        _imported(31, experts=1, per_expert=3),
        axis="functional",
        gate_mode="none",
        fixed_depth=1,
    )
    eager, _ = _run(monkeypatch, False, x, y0, [creative, functional], 3)
    got, used = _run(monkeypatch, True, x, y0, [creative, functional], 3)
    assert used
    assert torch.allclose(eager.float(), got.float(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_several_token_axes_are_one_token_axis(routing, monkeypatch):
    x, y0 = _inputs(16, tokens=2 * 3 * 4)
    s = _imported(17, routing=routing)
    x5, y5 = x.reshape(2, 2, 3, 4, -1), y0.reshape(2, 2, 3, 4, -1)
    eager, got = _same(monkeypatch, x5, y5, [s], 2)
    assert got.shape == y5.shape
    flat, _ = _run(monkeypatch, True, x, y0, [s], 2)
    assert torch.equal(got.reshape(flat.shape), flat)


def test_kernels_that_do_not_fit_the_shared_memory_run_eager(monkeypatch, caplog):
    from test_fused import _set

    monkeypatch.setattr(fused, "_shared_limit", lambda t: 101376)
    assert fused.fits(torch.zeros(1), 64) and not fused.fits(torch.zeros(1), 160)
    big = _set(20, experts=4, per_expert=40)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    assert not fused.fused_ok(_y0_like(big), [big], 2)
    monkeypatch.setattr(fused, "_shared_limit", lambda t: None)
    x, y0 = _inputs(18)
    s = _imported(19)
    eager, _ = _run(monkeypatch, False, x, y0, [s], 2)

    class OutOfResources(Exception):
        pass

    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise OutOfResources("out of resource: shared memory")

    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setattr(fused, "run_fused", fail)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    caplog.clear()
    with torch.no_grad(), compute.dropout_key(7, 3), caplog.at_level("WARNING", "kilora"):
        got = run_layer(x, y0, [s], 2, target=TARGET)
        again = run_layer(x, y0, [s], 2, target=TARGET)
    assert torch.allclose(got.float(), eager.float(), rtol=1e-4, atol=1e-5)
    assert torch.equal(got, again) and calls == [1]
    warnings = [r for r in caplog.records if "shared memory" in r.getMessage()]
    assert len(warnings) == 1

    def other(*a, **k):
        raise ValueError("a real bug")

    monkeypatch.setattr(fused, "run_fused", other)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    with pytest.raises(ValueError), torch.no_grad(), compute.dropout_key(7, 3):
        run_layer(x, y0, [s], 2, target=TARGET)


def test_axis_alpha_on_fused_path(monkeypatch):
    from kilora.call_values import axis_alphas

    x, y0 = _inputs(71)
    s = _imported(72, gate_mode="none")
    alpha = torch.linspace(0, 1, y0.shape[1], device=DEVICE)[None].expand(y0.shape[0], -1)
    with axis_alphas({"creative": alpha}):
        _same(monkeypatch, x, y0, [s], 2)


def _permuted(s, order):
    p = s.per_expert
    cols = torch.cat([torch.arange(e * p, (e + 1) * p) for e in order]).to(s.down.device)
    idx = torch.tensor(order, device=s.keys_in.device)
    return dataclasses.replace(
        s,
        down=s.down[cols],
        up=s.up[cols],
        keys_in=s.keys_in[idx],
        keys_out=None if s.keys_out is None else s.keys_out[idx],
        expert_ids=None if s.expert_ids is None else s.expert_ids[list(order)],
    )


@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_dropout_by_expert_id_matches_eager_and_ignores_the_order(carry, monkeypatch):
    x, y0 = _inputs(3)
    s = _imported(4, carry=carry)
    s = dataclasses.replace(s, expert_ids=torch.tensor([91, 17, 55, 3]))
    eager, got = _same(monkeypatch, x, y0, [s], 3)
    order = [2, 0, 3, 1]
    again, _ = _same(monkeypatch, x, y0, [_permuted(s, order)], 3)
    assert torch.allclose(again.float(), got.float(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("routing", list(ROUTES))
@pytest.mark.parametrize("depth", [1, 3, 1.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("with_region", [False, True])
def test_read_share_matches_eager(routing, depth, gate, with_region, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    x, y0 = _inputs(51)
    s = _imported(52, routing=routing, gate_mode=gate)
    region = (torch.arange(37) < 20).float()[None].expand(2, 37).to(DEVICE)
    _same(monkeypatch, x, y0, [s], depth, regions=[region] if with_region else None)


def test_read_share_zero_reads_and_expert_regions(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    x, y0 = _inputs(53)
    x[:, :5] = 0
    s = _imported(54, routing="tensor_top1")
    s = dataclasses.replace(s, expert_mask=_expert_mask(4, s.per_expert))
    _same(monkeypatch, x, y0, [s], 3)


def test_depth_one_read_share_core_operations(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    x, y0 = _inputs(55)
    s = _imported(56, gate_mode="none")
    eager, _ = _run(monkeypatch, False, x, y0, [s], 1)
    _run(monkeypatch, True, x, y0, [s], 1)
    calls = []
    linear, matmul = fused.F.linear, torch.matmul

    def count_linear(*args, **kwargs):
        calls.append("read")
        return linear(*args, **kwargs)

    def count_matmul(*args, **kwargs):
        calls.append("write")
        return matmul(*args, **kwargs)

    class Launcher:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                calls.append("kernel")
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(fused.F, "linear", count_linear)
    monkeypatch.setattr(torch, "matmul", count_matmul)
    for name in ("_read_share_kernel", "_update_kernel"):
        monkeypatch.setattr(fused, name, Launcher(getattr(fused, name)))
    got, used = _run(monkeypatch, True, x, y0, [s], 1)
    assert used
    assert torch.allclose(eager.float(), got.float(), rtol=1e-4, atol=1e-5)
    assert calls == ["read", "kernel", "kernel", "write"]


def _count_fused_ops(monkeypatch, run):
    from torch.utils._python_dispatch import TorchDispatchMode

    calls = []
    inside = [False]

    class Launcher:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                calls.append("triton")
                inside[0] = True
                try:
                    return launch(*args, **kwargs)
                finally:
                    inside[0] = False

            return run

    class CountOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name
            if not inside[0] and name not in {
                "aten::view",
                "aten::_unsafe_view",
                "aten::reshape",
                "aten::detach",
                "aten::empty",
                "aten::empty_like",
                "aten::expand",
                "aten::transpose",
                "aten::t",
                "aten::as_strided",
                "aten::alias",
                "aten::slice",
                "aten::select",
            }:
                calls.append(name)
            return func(*args, **(kwargs or {}))

    for name, kernel in list(vars(fused).items()):
        if name.endswith("_kernel") and hasattr(kernel, "__getitem__"):
            monkeypatch.setattr(fused, name, Launcher(kernel))
    with CountOps():
        result = run()
    return result, calls


@pytest.mark.parametrize("form", ["input", "state"])
def test_depth_one_adaptive_launch_count(form, monkeypatch):
    from test_fused import _no_warm_up_running

    _no_warm_up_running()
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "1")

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    x, y0 = _inputs(57)
    if form == "input":
        s = _imported(58, gate_mode="adaptive")

        def run(on):
            return _run(monkeypatch, on, x, y0, [s], 1)
    else:
        from test_fused import _set

        s = _set(58, gate_mode="adaptive")

        def run(on):
            monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
            with torch.no_grad(), compute.dropout_key(7, 3):
                return run_layer(y0, y0, [s], 1, target=TARGET), on

    eager, _ = run(False)
    run(True)
    (got, used), calls = _count_fused_ops(monkeypatch, lambda: run(True))
    assert used
    assert torch.allclose(eager.float(), got.float(), rtol=1e-4, atol=1e-5)
    assert calls.count("triton") == 4, calls
    assert len(calls) <= 7, calls


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("gate", ["none", "separable"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_derived_state_launch_budget(depth, gate, dtype, monkeypatch):
    from test_fused import _no_warm_up_running

    _no_warm_up_running()
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    s = _set(
        58,
        import_score="derived",
        gate_mode="learned" if gate == "separable" else "none",
        output_gate=gate,
    )
    _, y0 = _inputs(57)
    y0 = y0.to(dtype)

    def run():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return fused.run_fused(y0, [s], depth, target=TARGET)

    run()
    _, calls = _count_fused_ops(monkeypatch, run)
    budgets = {
        (torch.float32, "none"): [(3, 5), (9, 23), (13, 32)],
        (torch.float32, "separable"): [(4, 112), (10, 230), (14, 339)],
        (torch.bfloat16, "none"): [(2, 6), (8, 28), (12, 39)],
        (torch.bfloat16, "separable"): [(4, 113), (10, 235), (14, 346)],
    }
    launches, total = budgets[dtype, gate][depth - 1]
    if depth == 1:
        total += 1
    if DEVICE != "cpu" and dtype == torch.bfloat16 and gate == "none":
        launches += 1
    assert calls.count("triton") <= launches, calls
    assert len(calls) <= total, calls


@pytest.mark.parametrize("rank", [1, 3, 8, 32])
def test_read_dropout_reduction_preserves_seeded_masks(rank):
    generator = torch.Generator().manual_seed(rank)
    base = torch.randn(2, 7, 4 * rank, generator=generator).to(DEVICE)
    change = torch.randn(2, 7, 4 * rank, generator=generator).to(DEVICE)
    shape = (2, 7, 4, rank)
    sigma = base.float().reshape(shape).pow(2).mean(-1, keepdim=True).sqrt()
    sigma += compute.DROPOUT_DELTA
    with torch.no_grad(), compute.dropout_key(7, 3):
        uniform = compute.dropout_uniforms(shape, TARGET, 1, base.device)
        expected = (uniform < compute.keep_probability(change.reshape(shape), sigma)).reshape(
            change.shape
        )
        actual = compute.read_dropout(change, base, rank, TARGET, 1)
    assert torch.equal(actual.bool(), expected)


def test_wide_imported_sets_fit_without_a_read_matrix(monkeypatch):
    monkeypatch.setattr(fused, "_shared_limit", lambda t: 101376)
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    x, y0 = _inputs(25)
    wide = _imported(26, experts=4, per_expert=64)
    assert fused.fits(y0, 256, wide.router.shape[0]) and not fused.fits(y0, 256)
    _same(monkeypatch, x, y0, [wide], 2)


def _y0_like(s):
    return torch.zeros(2, 5, s.up.shape[1], device=s.up.device)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("form", ["input", "state"])
def test_attention_mount_matches_eager_and_fused(depth, form, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setitem(globals(), "TARGET", "blocks.0.attn1.to_q")
    sets = [_imported(307, import_score="derived")] if form == "input" else [_set(307)]
    width = sets[0].up.shape[-1]
    n_in = sets[0].down.shape[-1]
    x, y = _inputs(308, width=width, n_in=n_in)
    _same(monkeypatch, x, y, sets, depth)


@pytest.mark.parametrize("target", [TARGET, "blk.0.feed_forward.w1"])
@pytest.mark.parametrize("imported_first", [False, True])
@pytest.mark.parametrize("masks", ["sample", "expect"])
def test_derived_mixed_layer_uses_joint_chain(target, imported_first, masks, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", masks)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "1")
    x, y0 = _inputs(731)
    imported = _imported(732, import_score="derived")
    state = _set(733, import_score="derived")
    sets = [imported, state] if imported_first else [state, imported]
    calls = []
    original = fused.run_fused
    thread = threading.get_ident()

    def record(y, settings, *args, **kwargs):
        if threading.get_ident() == thread:
            calls.append([s.form for s in settings])
        return original(y, settings, *args, **kwargs)

    monkeypatch.setattr(fused, "run_fused", record)
    with torch.no_grad(), compute.dropout_key(7, 3):
        with compute.eager():
            expected = run_layer(x, y0, sets, 3, target=target)
        actual = run_layer(x, y0, sets, 3, target=target)
    assert not calls
    with torch.no_grad(), compute.dropout_key(7, 3):
        direct = original(y0, sets, 3, target=target, x=x)
    torch.testing.assert_close(direct, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_fused_scores_each_read_and_preserves_the_bound(carry, dtype, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    s = _set(814, carry=carry, gate_mode="none")
    _, y0 = _inputs(815, batch=1, tokens=5)
    y0 = y0.to(dtype)
    assert fused.fused_ok(y0, [s], 3)
    trace = []
    fused._TRACE.append(trace)
    try:
        with torch.no_grad(), compute.dropout_key(7, 3):
            actual = fused.run_fused(y0, [s], 3, target=TARGET)
            with compute.eager():
                expected = run_layer(y0, y0, [s], 3, target=TARGET)
    finally:
        fused._TRACE.pop()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-4, atol=tolerance)
    read_state = y0
    reference = None
    metric = compute._step_metric(s, y0.device)
    for step in trace:
        logits = compute.derived_logits(s, read_state)
        weights = compute._gate(logits, s, 0.0)
        actual_weights = step["p"][..., : weights.shape[-1]].reshape_as(weights)
        torch.testing.assert_close(
            actual_weights,
            weights,
            rtol=1e-4,
            atol=5e-2 if dtype == torch.bfloat16 else 1e-5,
        )
        assert torch.equal(actual_weights.argmax(-1), weights.argmax(-1))
        size = compute._metric_size(step["z"].reshape(*y0.shape[:-1], -1), metric)
        if reference is None:
            reference = size
        assert bool((size <= compute.IMPORT_DEPTH_BUDGET * reference + 1e-5).all())
        if step["carried"] is not None:
            carried = step["carried"].reshape(*y0.shape[:-1], -1)
            read_state = y0 + carried.to(dtype) @ s.up.to(dtype)
    assert len(trace) == 3
    changed = dataclasses.replace(
        s,
        router=torch.zeros_like(s.router),
        keys_in=torch.zeros_like(s.keys_in),
        keys_out=torch.zeros_like(s.keys_out),
    )
    with torch.no_grad(), compute.dropout_key(7, 3):
        again = fused.run_fused(y0, [changed], 3, target=TARGET)
    assert torch.equal(actual, again)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("mode", ["keys", "full"])
@pytest.mark.parametrize("path_width", [1, 3])
def test_derived_expert_regions_keep_full_share_on_fused_steps(
    depth, mode, path_width, monkeypatch
):
    from test_region import _derived_region_inputs, _derived_region_set

    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    monkeypatch.setenv("KILORA_DERIVED_LIFT_AFTER", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    x, y0, masks = _derived_region_inputs()
    x, y0 = x.to(DEVICE), y0.to(DEVICE)
    layer = dataclasses.replace(_to(_derived_region_set(), DEVICE), path_width=path_width)
    expected = torch.cat(masks, -1).to(DEVICE)
    layer = dataclasses.replace(layer, expert_mask=expected.repeat_interleave(2, -1))
    seen = []
    original = compute._mask_derived_logits

    def record(setting, logits):
        result = original(setting, logits)
        if setting.expert_mask is not None:
            seen.append(compute.sparsemax(result).clone())
        return result

    monkeypatch.setattr(compute, "_mask_derived_logits", record)
    _same(monkeypatch, x, y0, [layer], depth)
    assert seen
    assert all(torch.equal(gates, expected.expand_as(gates)) for gates in seen)


@pytest.mark.parametrize("depth", [1, 3])
def test_derived_fused_output_without_regions_keeps_original_scores(depth, monkeypatch):
    from test_region import _derived_region_inputs, _derived_region_set

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0, _ = _derived_region_inputs()
    x, y0 = x.to(DEVICE), y0.to(DEVICE)
    layer = _to(_derived_region_set(), DEVICE)
    actual, used = _run(monkeypatch, True, x, y0, [layer], depth)
    assert used
    full = dataclasses.replace(layer, expert_mask=torch.ones(1, 4, 4, device=DEVICE))
    inside, used = _run(monkeypatch, True, x, y0, [full], depth)
    assert used and torch.equal(inside, actual)
    monkeypatch.setattr(compute, "derived_logits", compute._derived_logits)
    expected, used = _run(monkeypatch, True, x, y0, [layer], depth)
    assert used and torch.equal(actual, expected)


@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("depth", [2, 4])
def test_derived_keys_chain_uses_shared_logits(routing, depth, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = _imported(1239, n_in=24, routing=routing, gate_mode="none", path_width=1)
    x, y0 = _inputs(1240, n_in=24)
    scores = []
    original = compute.pooled_derived_logits

    def score(layer, value, target):
        scores.append(value.clone())
        return original(layer, value, target)

    monkeypatch.setattr(compute, "pooled_derived_logits", score)
    _same(monkeypatch, x, y0, [s], depth)
    assert scores
    assert all(torch.equal(value, x) for value in scores)
