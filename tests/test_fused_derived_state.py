# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os

import pytest
import torch

pytest.importorskip("triton")

from test_fused import DEVICE, _set
from test_fused_input import _imported, _inputs

from kilora import compute, fused
from kilora.compute import run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)
TARGET = "blk.0.attn1"
RTOL, ATOL = 3e-4, 3e-5


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")


def _state(seed, gate_mode="none", **kw):
    return _set(seed, import_score="derived", gate_mode=gate_mode, **kw)


def _calls(monkeypatch):
    seen = []
    original = fused.run_fused

    def spy(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "run_fused", spy)
    return seen


def _compare(monkeypatch, x, y0, sets, depth, regions=None):
    monkeypatch.setenv("KILORA_FUSED", "1")
    with torch.no_grad(), compute.dropout_key(7, 3):
        with compute.eager():
            expected = run_layer(x, y0, sets, depth, target=TARGET, regions=regions)
        before = fused.COUNTS["fused"]
        actual = run_layer(x, y0, sets, depth, target=TARGET, regions=regions)
        after = fused.COUNTS["fused"]
    return expected, actual, after - before


def _record_routes(monkeypatch):
    seen = []
    original = compute._gates_from_logits

    def spy(*args, **kwargs):
        gates = original(*args, **kwargs)
        seen.append(torch.cat(gates, -1).clone())
        return gates

    monkeypatch.setattr(compute, "_gates_from_logits", spy)
    return seen


@pytest.mark.parametrize("depth", [0.5, 1, 2, 3, 2.5])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
def test_state_set_alone_matches_eager(depth, gate_mode, carry, monkeypatch):
    x, y0 = _inputs(901)
    sets = [_state(902, gate_mode=gate_mode, carry=carry)]
    seen = _calls(monkeypatch)
    expected, actual, fused_calls = _compare(monkeypatch, x, y0, sets, depth)
    assert fused_calls >= 1
    assert len(seen) == 1
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("imports", [1, 2])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_state_set_with_imports_matches_eager(imports, routing, gate_mode, monkeypatch):
    x, y0 = _inputs(911)
    sets = [_state(912, routing=routing, gate_mode=gate_mode)] + [
        _imported(913 + i, import_score="derived", gate_mode=gate_mode, routing=routing)
        for i in range(imports)
    ]
    region = torch.linspace(0, 1, y0.shape[-2], device=DEVICE).expand(*y0.shape[:-1])
    regions = [region, region.flip(-1)] + ([None] if imports == 2 else [])
    seen = _calls(monkeypatch)
    routes = _record_routes(monkeypatch)
    expected, actual, fused_calls = _compare(monkeypatch, x, y0, sets, 3, regions)
    assert fused_calls == 1
    assert len(seen) == 1
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert len(routes) == 6
    for eager_gates, fused_gates in zip(routes[:3], routes[3:], strict=True):
        assert torch.equal(eager_gates.argmax(-1), fused_gates.argmax(-1))
        assert torch.equal(eager_gates == 0, fused_gates == 0)
        torch.testing.assert_close(fused_gates, eager_gates, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("mask", ["region", "expert"])
def test_state_masks_and_routes(mask, routing, monkeypatch):
    x, y0 = _inputs(941)
    s = _state(942, routing=routing)
    regions = None
    if mask == "region":
        region = torch.linspace(0, 1, y0.shape[-2], device=DEVICE).expand(*y0.shape[:-1])
        regions = [region]
    else:
        expert_mask = torch.ones((*y0.shape[:-1], s.down.shape[0]), device=DEVICE)
        expert_mask[:, :17, s.per_expert : 2 * s.per_expert] = 0
        expert_mask[:, 17:, 2 * s.per_expert :] = 0
        s.expert_mask = expert_mask
    routes = _record_routes(monkeypatch)
    trace = []
    fused._TRACE.append(trace)
    try:
        expected, actual, calls = _compare(monkeypatch, x, y0, [s], 3, regions)
    finally:
        fused._TRACE.pop()
    assert calls == 1
    assert len(routes) == len(trace) == 3
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for eager_gates, step in zip(routes, trace, strict=True):
        fused_gates = step["p"][..., : s.keys_in.shape[0]].expand_as(eager_gates)
        assert torch.equal(eager_gates.argmax(-1), fused_gates.argmax(-1))
        assert torch.equal(eager_gates == 0, fused_gates == 0)
        torch.testing.assert_close(fused_gates, eager_gates, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "state_mode,import_mode,output_gate",
    [
        ("adaptive", "none", "none"),
        ("none", "auto", "none"),
    ],
)
def test_different_gate_modes_stay_eager(state_mode, import_mode, output_gate, monkeypatch):
    x, y0 = _inputs(921)
    sets = [
        _state(922, gate_mode=state_mode, output_gate=output_gate),
        _imported(923, import_score="derived", gate_mode=import_mode),
    ]
    seen = _calls(monkeypatch)
    reason = "derived state routing with different output gate modes"
    before = fused.REASONS.get(reason, 0)
    expected, actual, fused_calls = _compare(monkeypatch, x, y0, sets, 3)
    assert fused_calls == 0
    assert not seen
    assert fused.REASONS.get(reason, 0) > before
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("balance", ["0", "1"])
@pytest.mark.parametrize("depth", [1, 2.5, 3])
def test_joint_column_settings_and_masks(balance, depth, monkeypatch):
    monkeypatch.setenv("KILORA_FILE_BALANCE", balance)
    x, y0 = _inputs(951, n_in=24)
    layers = [
        _imported(
            952,
            experts=2,
            per_expert=3,
            n_in=24,
            import_score="derived",
            gate_mode="none",
            scale=0.4,
            tau=1.3,
            log_prior=-0.2,
            carry="drop",
        ),
        _state(953, experts=3, per_expert=2, scale=-0.2, tau=0.7, log_prior=0.1),
        _imported(
            954,
            experts=2,
            per_expert=1,
            n_in=24,
            import_score="derived",
            gate_mode="none",
            scale=torch.tensor(0.3),
            tau=torch.tensor(1.1),
        ),
    ]
    layers[0].expert_ids = torch.tensor([431, 523], device=DEVICE)
    layers[2].expert_ids = torch.tensor([617, 739], device=DEVICE)
    mask = torch.ones((*y0.shape[:-1], layers[0].down.shape[0]), device=DEVICE)
    mask[:, :17, : layers[0].per_expert] = 0
    layers[0].expert_mask = mask
    regions = [torch.ones_like(y0[..., 0]), None, torch.zeros_like(y0[..., 0])]
    routes = _record_routes(monkeypatch)
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, depth, regions)
    assert calls == 1
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    steps = int(depth) + (depth % 1 != 0)
    assert len(routes) == 2 * steps
    for a, b in zip(routes[:steps], routes[steps:], strict=True):
        assert torch.equal(a.argmax(-1), b.argmax(-1))
        assert torch.equal(a == 0, b == 0)


def test_different_import_depth_settings_have_a_reason(monkeypatch):
    x, y0 = _inputs(961)
    layers = [
        _state(962),
        _imported(963, import_score="derived", gate_mode="none"),
        _imported(964, import_score="derived", gate_mode="none", radial_step=0.4),
    ]
    reason = "joint derived state routing with different import depth settings"
    before = fused.REASONS.get(reason, 0)
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, 3)
    assert calls == 0
    assert fused.REASONS.get(reason, 0) == before + 1
    assert fused.LAST_REASON[0] == reason
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_common_import_depth_settings_run_fused(monkeypatch):
    x, y0 = _inputs(965, n_in=24)
    layers = [_state(966)] + [
        _imported(
            seed,
            n_in=24,
            import_score="derived",
            gate_mode="none",
            radial_step=0.6,
            orthogonal_step=0.9,
            radial_budget=1.3,
        )
        for seed in (967, 968)
    ]
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, 3)
    assert calls == 1
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_changed_stored_weights_refresh_low_rank_maps(monkeypatch):
    x, y0 = _inputs(971)
    layers = [_state(972), _imported(973, import_score="derived", gate_mode="none")]
    expected, before, calls = _compare(monkeypatch, x, y0, layers, 3)
    assert calls == 1
    torch.testing.assert_close(before, expected, rtol=RTOL, atol=ATOL)
    with torch.no_grad():
        layers[0].down.add_(0.1)
        layers[1].up.mul_(0.7)
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, 3)
    assert calls == 1
    assert not torch.equal(actual, before)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@contextlib.contextmanager
def _no_host_reads():
    def refuse(*args, **kwargs):
        raise AssertionError("a host read during the fused depth loop")

    saved = {n: getattr(torch.Tensor, n) for n in ("item", "tolist", "__bool__")}
    try:
        for n in saved:
            setattr(torch.Tensor, n, refuse)
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.Tensor, n, f)


@pytest.mark.parametrize("imports", [0, 1, 2])
def test_no_host_reads_in_the_depth_loop(imports, monkeypatch):
    x, y0 = _inputs(931)
    sets = [_state(932)] + [
        _imported(933 + i, import_score="derived", gate_mode="none") for i in range(imports)
    ]
    lora = [s for s in sets if s.down is not None]
    regs = [None] * len(lora)
    with torch.no_grad(), compute.dropout_key(7, 3):
        expected = fused.run_fused(y0, lora, 3, target=TARGET, regions=regs, x=x)
        with _no_host_reads():
            actual = fused.run_fused(y0, lora, 3, target=TARGET, regions=regs, x=x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == torch.device(DEVICE).type


@pytest.mark.parametrize("depth", [1, 2.5, 3])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("state_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_separable_state_with_ungated_imports(depth, routing, state_first, dtype, monkeypatch):
    x, y0 = _inputs(971)
    x, y0 = x.to(dtype), y0.to(dtype)
    state = _state(
        972, output_gate="separable", gate_mode="learned", routing=routing, scale=-0.2, tau=0.7
    )
    imported = _imported(
        973, import_score="derived", gate_mode="none", routing=routing, scale=0.4, tau=1.3
    )
    layers = [state, imported] if state_first else [imported, state]
    region = torch.linspace(0, 1, y0.shape[-2], device=DEVICE).expand(*y0.shape[:-1])
    regions = [region, region.flip(-1)]
    seen = _calls(monkeypatch)
    routes = _record_routes(monkeypatch)
    column_gates = []
    kernel = fused._derived_column_gate_kernel

    class Launcher:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(values, gates, *args, **kwargs):
                column_gates.append(gates.clone())
                return launch(values, gates, *args, **kwargs)

            return run

    monkeypatch.setattr(fused, "_derived_column_gate_kernel", Launcher())
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, depth, regions)
    assert calls == len(seen) == 1
    tolerance = 2e-2 if dtype == torch.bfloat16 else ATOL
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=tolerance)
    steps = 1 if depth == 1 else 3
    assert len(routes) == 2 * steps
    for eager, fast in zip(routes[:steps], routes[steps:], strict=True):
        torch.testing.assert_close(
            fast, eager, rtol=RTOL, atol=5e-2 if dtype == torch.bfloat16 else ATOL
        )
        assert torch.equal(fast.argmax(-1), eager.argmax(-1))
        if dtype == torch.float32:
            assert torch.equal(fast.argmax(-1), eager.argmax(-1))
            assert torch.equal(fast == 0, eager == 0)
    assert column_gates
    start = state.up.shape[0] if state_first else 0
    for gates in column_gates:
        ungated = gates[..., start : start + imported.up.shape[0]]
        assert torch.equal(ungated, torch.ones_like(ungated))


@pytest.mark.parametrize("tensor_values", [False, True])
def test_state_prepared_scalars_follow_changes(tensor_values, monkeypatch):
    s = _state(981, tau=0.7, scale=-0.2)
    if tensor_values:
        s.tau = torch.tensor(s.tau, device=DEVICE)
        s.scale = torch.tensor(s.scale, device=DEVICE)
    with torch.no_grad():
        before = fused._prepared(s, torch.bfloat16)
        assert fused._prepared(s, torch.bfloat16).state_keys is before.state_keys
        if tensor_values:
            s.tau.add_(0.5)
            s.scale.add_(0.5)
        else:
            s.tau += 0.5
            s.scale += 0.5
        after = fused._prepared(s, torch.bfloat16)
        assert after.state_keys is not before.state_keys
        torch.testing.assert_close(after.state_tau, torch.tensor(1.2, device=DEVICE))
        torch.testing.assert_close(after.state_scale, torch.tensor(0.3, device=DEVICE))
        _, y0 = _inputs(982)
        y0 = y0.to(torch.bfloat16)
        fused.run_fused(y0, [s], 3, target=TARGET)
        original = torch.as_tensor

        def no_python_scalar(value, *args, **kwargs):
            assert isinstance(value, torch.Tensor)
            return original(value, *args, **kwargs)

        monkeypatch.setattr(torch, "as_tensor", no_python_scalar)
        fused.run_fused(y0, [s], 3, target=TARGET)


def _adaptive_mix(dtype, routing, imports, head_limit=None, state_mode="learned"):
    x, y0 = _inputs(1011)
    x, y0 = x.to(dtype), y0.to(dtype)
    state = _state(
        1012, output_gate="separable", gate_mode=state_mode, routing=routing, scale=-0.2, tau=0.7
    )
    extra = [
        _imported(
            1013 + i,
            import_score="derived",
            gate_mode="auto",
            routing=routing,
            scale=0.4 - 0.1 * i,
            tau=1.3,
            gate_head_limit=head_limit,
        )
        for i in range(imports)
    ]
    for s in extra:
        assert compute.uses_adaptive(s) and compute.learned_kind(s) == "none"
    assert compute.learned_kind(state) == "separable"
    region = torch.linspace(0, 1, y0.shape[-2], device=DEVICE).expand(*y0.shape[:-1])
    regions = [region] + [region.flip(-1) * (0.5 + 0.5 * (i % 2)) for i in range(imports)]
    return x, y0, [state, *extra], regions


def _check_adaptive_mix(monkeypatch, dtype, depth, x, y0, layers, regions):
    seen = _calls(monkeypatch)
    routes = _record_routes(monkeypatch)
    expected, actual, calls = _compare(monkeypatch, x, y0, layers, depth, regions)
    assert calls == len(seen) == 1
    tolerance = 2e-2 if dtype == torch.bfloat16 else ATOL
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=tolerance)
    steps = 1 if depth == 1 else 3
    assert len(routes) == 2 * steps
    for eager, fast in zip(routes[:steps], routes[steps:], strict=True):
        assert torch.equal(fast.argmax(-1), eager.argmax(-1))
        if dtype == torch.float32:
            assert torch.equal(fast == 0, eager == 0)


@pytest.mark.parametrize("adaptive_kernel", ["0", "1"])
@pytest.mark.parametrize("imports", [1, 2])
@pytest.mark.parametrize("depth", [1, 2.5, 3])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_separable_state_with_adaptive_imports(
    imports, depth, routing, dtype, adaptive_kernel, monkeypatch
):
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", adaptive_kernel)
    x, y0, layers, regions = _adaptive_mix(dtype, routing, imports)
    launched = []
    kernel = fused._adaptive_token_kernel

    class Launcher:
        def __getitem__(self, grid):
            launched.append(1)
            return kernel[grid]

    monkeypatch.setattr(fused, "_adaptive_token_kernel", Launcher())
    _check_adaptive_mix(monkeypatch, dtype, depth, x, y0, layers, regions)
    steps = 1 if depth == 1 else 3
    assert len(launched) == (imports * steps if adaptive_kernel == "1" else 0)


@pytest.mark.parametrize("adaptive_kernel", ["0", "1"])
@pytest.mark.parametrize("depth", [2.5, 3])
def test_separable_state_with_adaptive_imports_draws_masks(depth, adaptive_kernel, monkeypatch):
    monkeypatch.delenv("KILORA_MASKS")
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", adaptive_kernel)
    x, y0, layers, regions = _adaptive_mix(torch.float32, "token", 2)
    _check_adaptive_mix(monkeypatch, torch.float32, depth, x, y0, layers, regions)


@pytest.mark.parametrize("head_limit", [1, 2])
@pytest.mark.parametrize("depth", [1, 2.5, 3])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_separable_state_with_head_limited_adaptive_imports(
    head_limit, depth, routing, dtype, monkeypatch
):
    x, y0, layers, regions = _adaptive_mix(dtype, routing, 2, head_limit=head_limit)
    for s in layers[1:]:
        assert compute.gate_fixed_after(s) == head_limit
    _check_adaptive_mix(monkeypatch, dtype, depth, x, y0, layers, regions)


@pytest.mark.parametrize("depth", [1, 3])
def test_both_mode_state_with_adaptive_imports(depth, monkeypatch):
    x, y0, layers, regions = _adaptive_mix(torch.float32, "token", 2, state_mode="both")
    assert compute.uses_adaptive(layers[0])
    _check_adaptive_mix(monkeypatch, torch.float32, depth, x, y0, layers, regions)


@pytest.mark.parametrize("adaptive_kernel", ["0", "1"])
@pytest.mark.parametrize("head_limit", [None, 1])
@pytest.mark.parametrize("imports", [1, 2])
def test_no_host_reads_with_adaptive_imports(imports, head_limit, adaptive_kernel, monkeypatch):
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", adaptive_kernel)
    x, y0, lora, regs = _adaptive_mix(torch.float32, "token", imports, head_limit=head_limit)
    regs = [compute._token_region(r, y0) for r in regs]
    with torch.no_grad(), compute.dropout_key(7, 3):
        expected = fused.run_fused(y0, lora, 3, target=TARGET, regions=regs, x=x)
        with _no_host_reads():
            actual = fused.run_fused(y0, lora, 3, target=TARGET, regions=regs, x=x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _census(monkeypatch, run):
    from test_fused_input import _count_fused_ops

    from kilora import adaptive_gate

    rates = []
    with monkeypatch.context() as m:
        for name in ("_gate_token_kernel", "_gate_channel_kernel"):
            kernel = getattr(adaptive_gate, name)

            class Launcher:
                def __init__(self, kernel):
                    self.kernel = kernel

                def __getitem__(self, grid):
                    rates.append("triton")
                    return self.kernel[grid]

            m.setattr(adaptive_gate, name, Launcher(kernel))
        _, calls = _count_fused_ops(m, run)
    return calls + rates


@pytest.mark.parametrize("adaptive_kernel", ["0", "1"])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("imports", [1, 2])
def test_adaptive_import_launch_census(imports, depth, adaptive_kernel, monkeypatch):
    from test_fused import _no_warm_up_running

    _no_warm_up_running()
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", adaptive_kernel)
    x, y0, lora, regs = _adaptive_mix(torch.float32, "token", imports)
    regs = [compute._token_region(r, y0) for r in regs]

    def fast():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return fused.run_fused(y0, lora, depth, target=TARGET, regions=regs, x=x)

    def eager():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return compute._run_input(x, y0, lora, list(range(len(lora))), depth, regs, TARGET)

    fast()
    eager()
    fast_calls = _census(monkeypatch, fast)
    eager_calls = _census(monkeypatch, eager)
    assert len(fast_calls) <= len(eager_calls), (fast_calls, eager_calls)
