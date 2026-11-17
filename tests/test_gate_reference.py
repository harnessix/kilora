# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch
from test_adaptive_gate import _toy

from kilora import adaptive_gate, compute, fused
from kilora.adaptive_gate import adaptive_keep, change_stats


@pytest.fixture(autouse=True)
def _new_rule(monkeypatch):
    monkeypatch.delenv("KILORA_GATE_BASE_REFERENCE", raising=False)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")


def _stats(value):
    return value.norm(dim=-1), value.square().mean(-2).sqrt()


def _keep(z, before=None, history=None, first=None, **kwargs):
    base = torch.ones_like(z)
    reference = (
        {} if first is None else dict(zip(("first_norm", "first_rms"), _stats(first), strict=True))
    )
    return adaptive_keep(
        z, before, history, torch.eye(z.shape[-1]), *_stats(base), **reference, **kwargs
    )


def test_full_first_application_and_base_comparison(monkeypatch):
    z = torch.ones(2, 7, 4)
    qa, qb, history = _keep(z)
    assert torch.equal(qa, torch.ones_like(qa))
    assert torch.equal(qb, torch.ones_like(qb))
    assert torch.equal(history, z * 0.5)
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    old_a, old_b, old_history = _keep(z)
    torch.testing.assert_close(old_a * old_b, torch.full_like(z, 0.25), rtol=0, atol=1e-6)
    assert torch.equal(history, old_history)


def test_growth_by_the_first_change_halves_again():
    first = torch.ones(2, 7, 4) * 3
    qa, qb, _ = _keep(2 * first, first, first=first)
    torch.testing.assert_close(qa, torch.full_like(qa, 0.5), rtol=0, atol=1e-4)
    torch.testing.assert_close(qb, torch.full_like(qb, 0.5), rtol=0, atol=1e-4)


@pytest.mark.parametrize("size", [1e-7, 0.01, 0.05])
def test_small_changes_keep_base_token_scale_and_remove_first_channel_change(size, monkeypatch):
    first = torch.ones(1, 8, 2) * size
    qa, qb, _ = _keep(first)
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    old_a, old_b, _ = _keep(first)
    assert (qa - old_a).abs().max() < 0.05
    assert (qb - old_b).abs().max() < 0.05
    for k in range(2, 5):
        monkeypatch.delenv("KILORA_GATE_BASE_REFERENCE")
        qa, qb, _ = _keep(k * first, (k - 1) * first, first=first)
        monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
        old_a, old_b, _ = _keep(k * first, (k - 1) * first)
        torch.testing.assert_close(qa, old_a, rtol=0, atol=1e-6)
        expected = old_b * torch.pow(0.5, torch.tensor(-size / (1 + 1e-6)))
        torch.testing.assert_close(qb, expected, rtol=0, atol=1e-6)
        if size == 1e-7:
            torch.testing.assert_close(qb, old_b, rtol=0, atol=1e-6)


def test_direction_still_damps_an_oscillating_channel(monkeypatch):
    first = torch.ones(1, 8, 2)
    before = first.clone()
    dz = first.clone()
    dz[..., 1] *= -1
    history = first * 0.5
    qa, qb, new = _keep(before + dz, before, history, first=first)
    assert torch.equal(qb[..., 1], torch.zeros_like(qb[..., 1]))
    assert (qb[..., 0] > 0).all()
    _, size_only, _ = _keep(before + dz, before, history, first=first, coherence=False)
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    _, old_b, old_hist = _keep(before + dz, before, history)
    _, old_size, _ = _keep(before + dz, before, history, coherence=False)
    assert torch.equal(qb / size_only, old_b / old_size)
    assert torch.equal(new, old_hist)
    assert (qa > 0).all()


def test_reference_uses_change_after_region_and_active_mask_and_resets_per_call(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = _toy([1.0, 1.0])
    z = torch.ones(1, 4, 2) * 3
    before = torch.ones_like(z)
    region = torch.tensor([1.0, 0.5, 0.0, 1.0])[None, :, None]
    active = torch.tensor([True, True, True, False])[None, :, None]
    hist = compute.gate_history(None, 0, before, region)
    norm, rms = _stats(torch.ones(1, 4, 8))
    compute.split_gate(s, z, hist, norm, rms, None, "test", 0, 1, region, active=active)
    dz = z * region * active - before * region
    want = change_stats(dz, s.up, s.up @ s.up.t(), active)
    assert torch.equal(hist["first_norm"], want[0])
    assert torch.equal(hist["first_rms"], want[1])
    ends = compute._gate_ends([s], [hist])
    assert set(ends[0]) == {"m", "q"}
    restored = compute.gate_history({"gates": ends}, 0, z, region)
    assert "first_norm" not in restored and "first_rms" not in restored
    assert restored["q"] is hist["q"]
    compute.split_gate(s, z + 0.2, restored, norm, rms, None, "test", 0, 1, region)
    assert torch.equal(restored["q"][0], region)
    assert torch.equal(restored["q"][1], torch.ones_like(restored["q"][1]))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("mode", ["adaptive", "none", "learned"])
def test_depth_one_retains_full_output_and_rates_across_sampler_steps(mode, fast, monkeypatch):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    y = torch.ones(1, 4, 8)
    s = dataclasses.replace(_toy([1.0]), gate_mode=mode)
    plain = dataclasses.replace(s, gate_mode="none")
    cache = {}
    with torch.no_grad(), compute.dropout_key(3, 0.5):
        expected = compute.run_layer(y, y, [plain], 1, target="layer")
        with compute.step_carry(cache):
            actual = compute.run_layer(y, y, [s], 1, target="layer")
    assert torch.equal(actual, expected)
    if mode == "adaptive":
        for rate in cache["layer"]["gates"][0]["q"]:
            assert torch.equal(rate, torch.ones_like(rate))


def _parent_keep(
    z, z_before, history, up, y0_norm, y0_rms, kappa=1.0, coherence=True, active=None, gram=None
):
    z, up = z.detach().float(), up.detach().float()
    dz = z if z_before is None else z - z_before.detach().float()
    gram = up @ up.t() if gram is None else gram
    step = ((dz @ gram) * dz).sum(-1).clamp(min=0).sqrt()
    qa = torch.pow(0.5, step / (kappa * y0_norm + 1e-6))
    tokens = z.shape[-2] if active is None else active.float().sum(-2, keepdim=True).clamp(min=1)

    def diag(moment):
        return ((moment @ up) * up).sum(-2)

    if coherence and history is not None:
        m = history.float()
        dzt, mt = dz.transpose(-1, -2), m.transpose(-1, -2)
        moments = torch.stack((z.transpose(-1, -2) @ z / tokens, dzt @ m, dzt @ dz, mt @ m), dim=-3)
        acc, dot, nd, nm = diag(moments).unbind(-2)
        acc = acc.clamp(min=0).sqrt()
        qb = torch.pow(0.5, acc / (kappa * y0_rms + 1e-6))
        norms = nd.clamp(min=0).sqrt() * nm.clamp(min=0).sqrt()
        cos = torch.where(norms > 1e-6, dot / norms.clamp(min=1e-6), torch.ones_like(dot))
        qb = qb * (0.5 + 0.5 * cos).clamp(0, 1)
    else:
        acc = diag(z.transpose(-1, -2) @ z / tokens).clamp(min=0).sqrt()
        qb = torch.pow(0.5, acc / (kappa * y0_rms + 1e-6))
    old = 0.0 if history is None else 0.5 * history.float()
    return qa[..., None], qb[..., None, :], old + 0.5 * dz


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("depth", [1, 3, 2.5])
def test_base_switch_outputs_equal_parent(depth, carry, fast, monkeypatch):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setattr(adaptive_gate, "triton", None)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    s = dataclasses.replace(_toy([0.6, -1.6]), carry=carry)
    y = torch.randn(2, 9, 8, generator=torch.Generator().manual_seed(718)) + 1

    def sample():
        cache = {}
        outputs = []
        for step in range(3):
            with torch.no_grad(), compute.step_carry(cache), compute.dropout_key(step, 0.5):
                outputs.append(compute.run_layer(y, y, [s], depth, target="layer"))
        return outputs

    actual = sample()
    monkeypatch.setattr(compute, "adaptive_keep", _parent_keep)
    for a, b in zip(actual, sample(), strict=True):
        assert torch.equal(a, b)


_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.mark.skipif(
    not _INTERPRET or adaptive_gate.triton is None, reason="needs TRITON_INTERPRET=1"
)
@pytest.mark.parametrize("coherence", [False, True])
@pytest.mark.parametrize("base_reference", [False, True])
def test_small_triton_kernels_match_torch(coherence, base_reference, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    if base_reference:
        monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    generator = torch.Generator().manual_seed(718)
    up = torch.randn(4, 11, generator=generator)
    first = torch.randn(2, 7, 4, generator=generator) * 3
    first_norm, first_rms = change_stats(first, up, up @ up.t())
    norm, rms = _stats(torch.randn(2, 7, 11, generator=generator))
    history = None
    for k in range(1, 5):
        before = None if k == 1 else (k - 1) * first
        args = (k * first, before, history, up, norm, rms)
        kwargs = dict(
            first_norm=first_norm, first_rms=first_rms, first_step=k == 1, coherence=coherence
        )
        actual = adaptive_keep(*args, **kwargs)
        with monkeypatch.context() as m:
            m.setattr(adaptive_gate, "triton", None)
            expected = adaptive_keep(*args, **kwargs)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
        history = expected[2]


@pytest.mark.skipif(not _INTERPRET or not fused.HAVE_TRITON, reason="needs TRITON_INTERPRET=1")
@pytest.mark.parametrize("base_reference", [False, True])
@pytest.mark.parametrize("has_carry", [False, True])
def test_fused_adaptive_kernels_match_eager_rates(base_reference, has_carry, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    if base_reference:
        monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    batch, tokens, cols, width = 2, 7, 4, 8
    generator = torch.Generator().manual_seed(718)
    up = torch.randn(cols, width, generator=generator) * 0.3
    up_pad = torch.zeros(32, width)
    up_pad[:cols] = up
    y = torch.randn(batch, tokens, width, generator=generator) + 1
    z1 = torch.randn(batch, tokens, cols, generator=generator) * 2
    region = torch.tensor([1, 0.5, 1, 0, 1, 0.75, 1], dtype=torch.float32).repeat(batch, 1)
    before = torch.randn(batch, tokens, cols, generator=generator) if has_carry else None
    history = torch.randn(batch, tokens, cols, generator=generator) if has_carry else None
    first_norm = torch.empty(batch, tokens)
    first_rms = torch.empty(batch, width)
    ref_norm = ref_rms = None
    dummy = torch.empty(1)
    for k, scale in enumerate((1.0, 2.0, 1.0, 3.0), 1):
        z = z1 * scale
        zr = z * region[..., None]
        if k == 1:
            dz = zr if before is None else zr - before
            ref_norm, ref_rms = change_stats(dz, up, up @ up.t())
        expected = adaptive_keep(
            zr,
            before,
            history,
            up,
            *_stats(y),
            first_norm=ref_norm,
            first_rms=ref_rms,
            first_step=k == 1,
        )
        qa, raw_qa, a = (torch.empty(batch, tokens) for _ in range(3))
        qb, b = (torch.empty(batch, width) for _ in range(2))
        next_before, next_history = torch.empty_like(z), torch.empty_like(z)
        common = dict(
            HAS_BEFORE=before is not None,
            HAS_HIST=history is not None,
            HAS_REGION=True,
            EXPECT=True,
            BASE_REFERENCE=base_reference,
            FIRST_STEP=k == 1,
            USE_LIBDEVICE=False,
            BT=32,
            CP=32,
            BO=32,
        )
        fused._adaptive_token_kernel[(batch, 1)](
            z,
            before if before is not None else dummy,
            history if history is not None else dummy,
            up_pad,
            y,
            region,
            qa,
            raw_qa,
            first_norm,
            a,
            next_before,
            next_history,
            dummy,
            tokens,
            cols,
            width,
            width,
            1.0,
            0,
            RECORD_QA=True,
            WRITE_ZA=False,
            **common,
        )
        fused._adaptive_channel_kernel[(batch, 1)](
            z,
            before if before is not None else dummy,
            history if history is not None else dummy,
            up_pad,
            y,
            region,
            qb,
            first_rms,
            b,
            dummy,
            dummy,
            tokens,
            cols,
            width,
            width,
            1.0,
            0,
            COHERENCE=True,
            WRITE_OUT=False,
            **common,
        )
        torch.testing.assert_close(raw_qa[..., None], expected[0], rtol=0, atol=1e-5)
        torch.testing.assert_close(qb[..., None, :], expected[1], rtol=0, atol=1e-5)
        torch.testing.assert_close(next_history, expected[2], rtol=0, atol=1e-5)
        if not base_reference:
            torch.testing.assert_close(first_norm, ref_norm, rtol=0, atol=1e-5)
            torch.testing.assert_close(first_rms, ref_rms, rtol=0, atol=1e-5)
        before, history = next_before, next_history


def test_new_channel_growth_stays_bounded_after_the_first_change():
    first = torch.ones(1, 8, 2) * 3
    peak = 3 / (torch.exp(torch.tensor(1.0)) * torch.log(torch.tensor(2.0)))
    for size in (0.0, 0.1, 1.0, 3.0, 30.0):
        z = first + size
        _, qb, _ = _keep(z, first, first=first, coherence=False)
        assert (qb * size <= peak * (1 + 1e-5)).all()


@pytest.mark.parametrize("depth", [1, 3])
def test_fused_gate_closure_uses_call_local_reference(depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = _toy([1.0, 1.0])
    gate = fused._torch_gate(s, None, None, None, None)
    z = torch.ones(1, 4, 2) * 3
    norm, rms = _stats(torch.ones(1, 4, 8))
    hist = {}
    for k in range(1, depth + 1):
        expected = compute.split_gate(s, k * z, hist, norm, rms, None, "test", 0, k, None, True)
        actual = gate(k * z, (norm, rms), None, "test", k, True)
        for a, b in zip(actual, expected, strict=True):
            assert torch.equal(a, b)
    assert "first_norm" in gate.hist


@pytest.mark.parametrize("form", ["state", "input"])
def test_depth_one_matches_plain_application(form, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = dataclasses.replace(_toy([1.0, -1.6]), form=form)
    y = torch.ones(1, 4, 8)
    with torch.no_grad(), compute.dropout_key(3, 0.5):
        actual = compute.run_layer(y, y, [s], 1, target="layer")
        plain = compute.run_layer(
            y, y, [dataclasses.replace(s, gate_mode="none")], 1, target="layer"
        )
    assert torch.equal(actual, plain)


def test_import_step_folding_retains_first_change_rates(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(_toy([1.0, 0.6]), form="input")
    y = torch.ones(1, 4, 8)

    def run(fold):
        monkeypatch.setenv("KILORA_IMPORT_FOLD", fold)
        with torch.no_grad(), compute.dropout_key(3, 0.5):
            return compute.run_layer(y, y, [s], 3, target="layer")

    assert torch.equal(run("1"), run("0"))


@pytest.mark.parametrize("width", [1, 2])
def test_path_gates_retain_the_first_change_reference(width, monkeypatch):
    from test_lowrank_depth import _sets

    s = dataclasses.replace(
        _sets(1, 16, 4, 2, 8, seed=718)[0],
        iteration="path",
        path_width=width,
        gate_mode="adaptive",
    )
    y = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(719))
    calls = []
    original = compute.adaptive_keep

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "adaptive_keep", record)
    with torch.no_grad(), compute.dropout_key(3, 0.5):
        compute.run_layer(y, y, [s], 3, target="layer")
    assert calls
    assert all("first_norm" in call and "first_rms" in call for call in calls)
    assert len(calls) == 3 * width
    for start in range(0, len(calls), 3):
        first, *later = calls[start : start + 3]
        assert first["first_step"]
        assert all(not call["first_step"] for call in later)
        for call in later:
            assert torch.equal(call["first_norm"], first["first_norm"])
            assert torch.equal(call["first_rms"], first["first_rms"])
