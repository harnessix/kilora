# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
import threading

import pytest
import torch

pytest.importorskip("triton")

from kilora import compute, fused
from kilora.compute import LayerSet, run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")


ROUTES = {
    "token": ("token", "sparsemax"),
    "token_top1": ("token", "top1"),
    "tensor": ("tensor", "sparsemax"),
    "tensor_top1": ("tensor", "top1"),
}


def _set(seed, experts=4, per_expert=2, width=24, router_dim=8, **kw):
    gen = torch.Generator().manual_seed(seed)
    cols = experts * per_expert
    kw.setdefault("import_score", "keys")
    s = LayerSet(
        router=torch.randn(router_dim, width, generator=gen) / width**0.5,
        keys_in=torch.randn(experts, 3, router_dim, generator=gen),
        keys_out=torch.randn(experts, 2, router_dim, generator=gen),
        per_expert=per_expert,
        down=torch.randn(cols, width, generator=gen) / width**0.5,
        up=torch.randn(cols, width, generator=gen) * 0.6 / cols**0.5,
        **_routing(kw),
    )
    if s.output_gate == "separable":
        s.gate = compute.new_gate("separable", experts, width, gen, cols)
        s.gate["tb2"] = torch.zeros(experts)
        s.gate["cb2"] = torch.zeros(experts, width)
    return _to(s, DEVICE)


def _routing(kw):
    name = kw.pop("routing", None)
    if name is not None:
        kw["routing_scope"], kw["routing_select"] = ROUTES[name]
    return kw


def _to(s, dev):
    for name in ("router", "keys_in", "keys_out", "down", "up"):
        setattr(s, name, getattr(s, name).to(dev))
    if s.gate is not None:
        s.gate = {k: v.to(dev) for k, v in s.gate.items()}
    if isinstance(s.tau, torch.Tensor):
        s.tau = s.tau.to(dev)
    if isinstance(s.scale, torch.Tensor):
        s.scale = s.scale.to(dev)
    return s


def _y0(seed, batch=2, tokens=37, width=24, dtype=torch.float32):
    return torch.randn(batch, tokens, width, generator=torch.Generator().manual_seed(seed)).to(
        DEVICE, dtype
    )


def _run(monkeypatch, on, y0, sets, depth, calls=1, **kw):
    monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
    seen = []
    real = fused.run_fused

    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    cache: dict = {}
    outs = []
    with torch.no_grad(), compute.step_carry(cache):
        for step in range(calls):
            with compute.dropout_key(7, step):
                outs.append(run_layer(y0, y0, sets, depth, target="blk.0.attn1", **kw))
    assert bool(seen) == on
    return outs, cache


def _same(monkeypatch, y0, sets, depth, calls=2, atol=1e-5, **kw):
    a, ca = _run(monkeypatch, False, y0, sets, depth, calls, **kw)
    b, cb = _run(monkeypatch, True, y0, sets, depth, calls, **kw)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x.float(), y.float(), rtol=1e-4, atol=atol), (x - y).abs().max()
    ea, eb = ca["blk.0.attn1"], cb["blk.0.attn1"]
    for key in ("carried", "change"):
        for x, y in zip(ea[key], eb[key], strict=True):
            assert x.shape == y.shape
            assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), (key, (x - y).abs().max())
    ga, gb = ea.get("gates"), eb.get("gates")
    assert (ga is None) == (gb is None)
    for x, y in zip(ga or [], gb or [], strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            fa = [x["m"], *x["q"]] if x["m"] is not None else list(x["q"])
            fb = [y["m"], *y["q"]] if y["m"] is not None else list(y["q"])
            for u, v in zip(fa, fb, strict=True):
                assert torch.allclose(u.float(), v.float(), rtol=1e-4, atol=1e-5)
    return a, b


def test_uniforms_are_bit_identical():
    for target, k, n in (("blk.3.ff#0", 1, 1000), ("x#1#gate", 4, 77)):
        with compute.dropout_key(123, 0.5):
            ref = compute.dropout_uniforms((n,), target, k, DEVICE)
            got = fused.uniforms(n, target, k, DEVICE)
        assert torch.equal(ref, got)


@pytest.mark.parametrize("depth", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_depth_and_carry(depth, carry, monkeypatch):
    _same(monkeypatch, _y0(1), [_set(1, carry=carry)], depth)


@pytest.mark.parametrize("routing", list(ROUTES))
@pytest.mark.parametrize("depth", [1, 3])
def test_separable_gate(routing, depth, monkeypatch):
    s = _set(2, output_gate="separable", routing=routing)
    _same(monkeypatch, _y0(2), [s], depth)


@pytest.mark.parametrize("mode", ["learned", "adaptive", "both"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("depth", [1, 3, 1.5])
def test_split_gate_on_the_fused_path(mode, routing, depth, monkeypatch):
    if mode == "learned":
        monkeypatch.setenv("KILORA_FUSED_GATE", "torch")
    s = _set(40, output_gate="separable", routing=routing, gate_mode=mode)
    region = (torch.rand(2, 37, generator=torch.Generator().manual_seed(40)) > 0.3).float()
    _same(monkeypatch, _y0(40), [s], depth)
    _same(monkeypatch, _y0(41), [s], depth, regions=[region.to(DEVICE)])


@pytest.mark.parametrize("depth", [1, 1.5, 3])
@pytest.mark.parametrize("masks", ["sample", "expect"])
@pytest.mark.parametrize("with_region", [False, True])
@pytest.mark.parametrize("kernel", ["0", "1"])
def test_adaptive_gate_masks_and_history(depth, masks, with_region, kernel, monkeypatch):
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", kernel)
    monkeypatch.setenv("KILORA_MASKS", masks)
    s = _set(71, gate_mode="adaptive", routing="token")
    region = torch.linspace(0, 1, 37, device=DEVICE).expand(2, 37)
    _same(monkeypatch, _y0(71), [s], depth, regions=[region] if with_region else None)


@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_open_adaptive_gate_is_no_gate_bit_for_bit(routing, monkeypatch):
    import dataclasses
    import math

    monkeypatch.setenv("KILORA_FUSED", "1")
    plain = _set(43, routing=routing, gate_mode="none")
    open_gate = dataclasses.replace(
        plain, gate_mode="adaptive", gate_kappa=math.inf, gate_coherence=False
    )
    y0 = _y0(43)
    for depth, exact in ((1, True), (3, False)):
        with torch.no_grad(), compute.dropout_key(1, 0.5):
            a = run_layer(y0, y0, [open_gate], depth, target="blk.0.attn1")
            b = run_layer(y0, y0, [plain], depth, target="blk.0.attn1")
        if exact:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gate", ["none", "separable", "adaptive"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("depth", [1, 3])
def test_expected_masks_match_eager(gate, routing, depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    kw = {"gate_mode": "adaptive"} if gate == "adaptive" else {"output_gate": gate}
    if gate == "none":
        kw["gate_mode"] = "none"
    s = _set(44, routing=routing, **kw)
    _same(monkeypatch, _y0(44), [s], depth)


def test_mask_mode_setting():
    assert compute.mask_mode() == "sample"
    compute.set_masks("expect")
    try:
        assert compute.expect_masks()
    finally:
        compute.set_masks("sample")
    with pytest.raises(ValueError):
        compute.set_masks("soft")


def test_adaptive_gate_without_learned_weights(monkeypatch):
    s = _set(42, routing="token", gate_mode="adaptive")
    _same(monkeypatch, _y0(42), [s], 2, calls=3)


@pytest.mark.parametrize("depth", [1, 2.5])
def test_separable_gate_fractional(depth, monkeypatch):
    _same(monkeypatch, _y0(3), [_set(3, output_gate="separable")], depth)


@pytest.mark.parametrize("gate", ["none", "separable"])
def test_region(gate, monkeypatch):
    s = _set(4, output_gate=gate)
    region = (torch.rand(2, 37, generator=torch.Generator().manual_seed(4)) > 0.4).float()
    _same(monkeypatch, _y0(4), [s], 2, regions=[region.to(DEVICE)])


def test_replace_tensor_strength_and_temperature(monkeypatch):
    s = _set(5, aggregation="replace", scale=torch.tensor(-0.7), tau=torch.tensor(1.7))
    _same(monkeypatch, _y0(5), [s], 3)


@pytest.mark.parametrize("routing", list(ROUTES))
@pytest.mark.parametrize("gate", ["none", "separable"])
def test_scope_and_selection(routing, gate, monkeypatch):
    s = _set(30, experts=7, per_expert=3, routing=routing, output_gate=gate)
    region = (torch.rand(2, 37, generator=torch.Generator().manual_seed(30)) > 0.3).float()
    _same(monkeypatch, _y0(30), [s], 2, regions=[region.to(DEVICE)])
    _same(monkeypatch, _y0(31), [s], 1.5)


@pytest.mark.parametrize("routing", ["token", "token_top1"])
def test_token_routing_more_experts(routing, monkeypatch):
    s = _set(6, experts=11, per_expert=3, routing=routing, width=40, router_dim=20)
    _same(monkeypatch, _y0(6, width=40), [s], 2)


@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reduced_precision_types(dtype, routing, monkeypatch):
    if DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 products are slow or missing on a CPU")
    s = _set(7, output_gate="separable", routing=routing)
    y0 = _y0(7, dtype=dtype)
    a, _ = _run(monkeypatch, False, y0, [s], 1, calls=2)
    b, _ = _run(monkeypatch, True, y0, [s], 1, calls=2)
    reference, _ = _run(monkeypatch, False, y0.float(), [s], 1, calls=2)
    for x, y, full in zip(a, b, reference, strict=True):
        assert y.dtype == dtype
        diff = (x.float() - y.float()).abs()
        assert float(diff.max()) <= 0.05 * float(x.float().abs().max())
        far = diff > 2e-2 + 2e-2 * x.float().abs()
        assert float(far.float().mean()) < 1e-3
        if s.form == "state":
            eager_error = (x.float() - full).square().mean().sqrt()
            fused_error = (y.float() - full).square().mean().sqrt()
            assert fused_error <= 1.1 * eager_error + 1e-3
        else:
            assert float((diff == 0).float().mean()) > 0.99


@pytest.mark.parametrize("per_expert", [2, 5, 16])
def test_masks_are_bit_identical_on_the_same_inputs(per_expert):
    s = _set(11 + per_expert, per_expert=per_expert, width=40, router_dim=16)
    y0 = _y0(11, width=40)
    c0 = torch.nn.functional.linear(y0, s.down).float()
    hold = c0 * s.scale
    with torch.no_grad(), compute.dropout_key(7, 0):
        carry: dict = {}
        fused.run_fused(y0, [s], 1, target="blk.0.attn1", carry=carry, _drop_last=True)
        entry = carry["blk.0.attn1"]
        z = entry["change"][0]
        m = compute.read_dropout(z, c0, s.per_expert, "blk.0.attn1#0", 1)
        carry = {}
        fused.run_fused(y0, [s], 1, target="blk.0.attn1", carry=carry)
        ends = carry["blk.0.attn1"]["carried"][0]
    assert 0 < float(m.mean()) < 1
    assert torch.equal(entry["dropped"], hold + (1 - m) * (z - hold))
    expected = compute.carry_transfer(z, c0, s.per_expert)
    if DEVICE == "cuda":
        assert torch.equal(ends, expected)
    else:
        assert torch.allclose(ends, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_channel_output_rounding(dtype):
    if DEVICE == "cpu" and dtype != torch.float32:
        pytest.skip("the Triton interpreter truncates low-precision stores")
    gen = torch.Generator().manual_seed(311)
    y = torch.randn(2, 37, 24, generator=gen).to(DEVICE, dtype)
    delta = torch.randn(2, 37, 24, generator=gen).to(DEVICE, dtype)
    gate = torch.rand(2, 24, generator=gen).to(DEVICE)
    out = torch.empty_like(y)
    fused._channel_output_kernel[(fused.triton.cdiv(y.numel(), 256),)](
        y, delta, gate, out, 37, 24, y.numel(), BLOCK=256, enable_fp_fusion=False
    )
    assert torch.equal(out, y + delta * gate[:, None, :].to(dtype))
    fused._add_output_kernel[(fused.triton.cdiv(y.numel(), 256),)](
        y, delta, out, y.numel(), BLOCK=256
    )
    assert torch.equal(out, y + delta)


@pytest.mark.parametrize("strided", [False, True])
def test_gated_read_preserves_multiplication_order(strided):
    s = _set(312, output_gate="separable")
    p = fused._prepared(s, torch.float32)
    gate = torch.rand(2, p.OUT * 2, generator=torch.Generator().manual_seed(313)).to(DEVICE)
    gate = gate[:, ::2] if strided else gate[:, : p.OUT].contiguous()
    expected = torch.matmul(p.up_pad[None] * gate[:, None, :], p.dr_t)
    assert torch.equal(fused._gated_read(p, gate), expected)


def test_routing_mode_reads_scope_and_selection_fields():
    from types import SimpleNamespace

    for scope in ("token", "tensor"):
        for select in ("sparsemax", "top1"):
            s = SimpleNamespace(routing_scope=scope, routing_select=select)
            assert fused.routing_mode(s) == (scope == "token", select == "top1")
    assert fused.routing_mode(SimpleNamespace(routing_scope="row", routing_select="top1")) is None


def test_fallbacks(monkeypatch):
    s = _set(9)
    y0 = _y0(9)
    monkeypatch.setenv("KILORA_FUSED", "1")
    assert fused.fused_ok(y0, [s])
    assert not fused.fused_ok(y0, [s, _set(10)])
    assert not fused.fused_ok(y0.to(torch.float64), [s])
    with compute.recording_routes():
        assert not fused.fused_ok(y0, [s])
    monkeypatch.setenv("KILORA_FUSED", "0")
    assert not fused.fused_ok(y0, [s])


def _no_warm_up_running():
    import threading

    for t in threading.enumerate():
        if t.name == "kilora-fused-warm-up":
            t.join(300)


def test_warm_up_never_raises_and_keeps_shared_state(monkeypatch):
    _no_warm_up_running()
    assert fused.warm_up(DEVICE) is True
    assert not compute._STEP_CARRY
    assert fused.warm_up("no-such-device") is False
    monkeypatch.setenv("KILORA_FUSED", "0")
    assert fused.warm_up(DEVICE) is False
    assert fused.start_warm_up() is None


def test_warm_up_per_shape_compiles_each_shape_once(monkeypatch):
    _no_warm_up_running()
    calls = []
    real = fused.run_fused
    monkeypatch.setattr(fused, "run_fused", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = _set(20, experts=3, per_expert=5)
    b = _set(21, experts=3, per_expert=5)
    c = _set(22, experts=6, per_expert=2, routing="token", output_gate="separable")
    two = [a, _set(23)]
    one = (torch.float32,)
    thread = fused.warm_up_sets([([a], 1), ([b], 1), ([c], 2.5), (two, 1)], DEVICE, one)
    thread.join()
    assert len(calls) == 2 * 2
    assert fused.shape_key(a) == fused.shape_key(b) != fused.shape_key(c)
    calls.clear()
    assert fused.warm_up_sets([([b], 1)], DEVICE, one) is None
    assert calls == []
    monkeypatch.setenv("KILORA_FUSED", "0")
    assert fused.warm_up_sets([([a], 1)], DEVICE) is None


def test_apply_warms_up_the_mod_shapes_in_the_model_type(monkeypatch):
    from types import SimpleNamespace

    from kilora.bake import TargetPlan
    from kilora.comfyui import patching

    seen = []
    monkeypatch.setattr(
        fused, "warm_up_sets", lambda pairs, dtypes=None: seen.append((pairs, dtypes))
    )
    model = SimpleNamespace(manual_cast_dtype=None, get_dtype=lambda: torch.bfloat16)
    plan = TargetPlan(sets=[_set(24)], depth=2)
    patching._warm_up_shapes(SimpleNamespace(model=model), {"t": plan})
    assert seen == [([(plan.sets, 2)], (torch.bfloat16,))]
    monkeypatch.setattr(fused, "warm_up_sets", lambda *a, **k: 1 / 0)
    patching._warm_up_shapes(SimpleNamespace(), {"t": plan})


def test_plugin_load_starts_no_work_without_a_gpu(monkeypatch):
    import importlib

    import kilora.comfyui as plugin

    started = []
    monkeypatch.setattr(fused, "start_warm_up", lambda: started.append(1))
    importlib.reload(plugin)
    assert started == [1]
    monkeypatch.setattr(fused, "start_warm_up", lambda: 1 / 0)
    importlib.reload(plugin)


@pytest.mark.parametrize("policy", ["auto", "reference"])
def test_token_roles(policy, monkeypatch):
    from kilora.family_spec import TokenLayout
    from kilora.roles import role_mask

    layout = TokenLayout("image2d", (1, 1))
    mask = role_mask(policy, layout, 37, (5, 5), [(3, 4)])
    y0 = _y0(50)
    a, b = _same(monkeypatch, y0, [_set(50)], 2, regions=[mask.to(DEVICE)])
    off = mask[0] == 0
    for y in (a[0], b[0]):
        assert torch.equal(y[:, off.to(y.device)], y0[:, off.to(y0.device)])


def test_warm_up_skips_shapes_that_do_not_fit(monkeypatch):
    _no_warm_up_running()
    monkeypatch.setattr(fused, "_WARMED", set())
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    calls = []
    real = fused.run_fused

    class OutOfResources(Exception):
        pass

    def spy(y0, sets, *a, **k):
        calls.append(sets[0].down.shape[0])
        if sets[0].down.shape[0] == 12:
            raise OutOfResources("out of resource: shared memory")
        return real(y0, sets, *a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    monkeypatch.setattr(fused, "_shared_limit", lambda t: 4 * (64 * 64 + 32 * 64))
    big, odd, small = (
        _set(30, experts=4, per_expert=40),
        _set(31, experts=3, per_expert=4),
        _set(32, per_expert=8),
    )
    one = (torch.float32,)
    assert fused.warm_up(DEVICE, [(big, 1), (odd, 1), (small, 1)], one) is True
    assert 160 not in calls and calls.count(12) == 1 and 32 in calls


def test_adaptive_gate_runs_in_torch_by_default(monkeypatch):
    monkeypatch.delenv("KILORA_ADAPTIVE_KERNEL", raising=False)
    seen = []
    real = fused._torch_gate
    monkeypatch.setattr(fused, "_torch_gate", lambda *a, **k: seen.append(1) or real(*a, **k))
    s = _set(72, gate_mode="adaptive", routing="token")
    _same(monkeypatch, _y0(72), [s], 2)
    assert seen and not fused.adaptive_kernel()
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "1")
    assert fused.adaptive_kernel()


def test_full_history_fused_is_unchanged(monkeypatch):
    import dataclasses

    y0 = _y0(31)
    s = _set(31, gate_mode="adaptive")
    baseline, _ = _run(monkeypatch, True, y0, [s], 3)
    explicit, _ = _run(monkeypatch, True, y0, [dataclasses.replace(s, gate_head_limit=None)], 3)
    assert all(torch.equal(a, b) for a, b in zip(baseline, explicit, strict=True))
    assert not fused.fused_ok(y0, [dataclasses.replace(s, gate_head_limit=2)], 3)


def test_torch_gate_stats_replays_reference_with_fused_interpreter(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_GATE", "torch")
    monkeypatch.delenv("KILORA_GATE_STATS", raising=False)
    original = compute._y0_gate_stats

    def reference(y):
        with torch.no_grad():
            sq = y.float().square()
            return sq.sum(-1).sqrt() + compute.DROPOUT_DELTA, sq.mean(
                -2
            ).sqrt() + compute.DROPOUT_DELTA

    sets = [_set(4, gate_mode="adaptive")]
    y = _y0(12)
    monkeypatch.setattr(compute, "_y0_gate_stats", reference)
    expected = _run(monkeypatch, True, y, sets, 3, calls=5)
    monkeypatch.setattr(compute, "_y0_gate_stats", original)
    with compute.gate_stats_mode("torch"):
        actual = _run(monkeypatch, True, y, sets, 3, calls=5)
    for a, b in zip(actual[0], expected[0], strict=True):
        assert torch.equal(a, b)
