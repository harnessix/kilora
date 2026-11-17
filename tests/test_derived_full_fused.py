# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import threading

import pytest
import torch

pytest.importorskip("triton")

from test_fused_input import DEVICE, TARGET, _imported, _inputs

from kilora import compute, fused, path_walks
from kilora.compute import run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")


def _set(gate_mode):
    s = _imported(3, n_in=24, gate_mode=gate_mode)
    s.path_width = 3
    return s


def _region(tokens):
    return (torch.arange(tokens) % 3 != 0).float().view(1, tokens, 1).to(DEVICE)


def _spy(monkeypatch):
    calls, weights = [], []
    real = fused.run_fused
    real_select = path_walks.select_weights
    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            calls.append(k.get("walks"))
        return real(*a, **k)

    def select(*a, **k):
        out = real_select(*a, **k)
        if threading.get_ident() == me:
            weights.append(out.detach().clone())
        return out

    monkeypatch.setattr(fused, "run_fused", spy)
    monkeypatch.setattr(path_walks, "select_weights", select)
    return calls, weights


def _call(monkeypatch, fused_on, x, y, s, depth, regions=None, stacked=True):
    monkeypatch.setenv("KILORA_FUSED", "1" if fused_on else "0")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
    if not fused_on:
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    else:
        monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)
    with torch.no_grad(), compute.dropout_key(7, 3):
        return run_layer(x, y, [s], depth, target=TARGET, regions=regions)


@pytest.mark.parametrize("depth", [1, 2, 4, 2.5])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("with_region", [False, True])
def test_stacked_walks_match_eager_walks(monkeypatch, depth, gate_mode, with_region):
    s = _set(gate_mode)
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    regions = [_region(5)] if with_region else None
    calls, weights = _spy(monkeypatch)
    expected = _call(monkeypatch, False, x, y, s, depth, regions)
    eager_weights = list(weights)
    assert not any(w is not None for w in calls)
    calls.clear()
    weights.clear()
    actual = _call(monkeypatch, True, x, y, s, depth, regions)
    assert [w is not None for w in calls] == [True]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert len(weights) == len(eager_weights) == 1
    assert torch.equal(weights[0].argmax(-1), eager_weights[0].argmax(-1))
    torch.testing.assert_close(weights[0], eager_weights[0], rtol=1e-4, atol=1e-5)


def test_one_fused_chain_call_per_full_mode_call(monkeypatch):
    s = _set("adaptive")
    x, y = _inputs(5, batch=2, tokens=6, n_in=24)
    calls, _ = _spy(monkeypatch)
    before = dict(fused.COUNTS)
    _call(monkeypatch, True, x, y, s, 4)
    assert len(calls) == 1
    assert calls[0] is not None and calls[0].shape[0] == 3
    assert fused.COUNTS.get("walks_fused", 0) == before.get("walks_fused", 0) + 1
    assert fused.COUNTS["import_fused"] == before["import_fused"] + 1


def test_turned_off_keeps_the_walks_and_counts_the_reason(monkeypatch):
    s = _set("none")
    x, y = _inputs(6, batch=1, tokens=5, n_in=24)
    calls, _ = _spy(monkeypatch)
    before = fused.COUNTS.get("walks_eager", 0)
    _call(monkeypatch, True, x, y, s, 4, stacked=False)
    assert all(w is None for w in calls)
    assert fused.COUNTS.get("walks_eager", 0) == before + 1
    assert fused.REASONS.get("stacked walks are turned off", 0) >= 1


def test_no_host_read_in_the_depth_loop(monkeypatch):
    s = _set("adaptive")
    x, y = _inputs(7, batch=1, tokens=5, n_in=24)
    reads = []
    names = {"run_fused", "_derived_stacked_walks"}

    def counted(method):
        real = getattr(torch.Tensor, method)

        def wrapper(self, *a, **k):
            frame = sys._getframe(1)
            if frame.f_code.co_name in names:
                reads.append((method, frame.f_code.co_name, frame.f_lineno))
            return real(self, *a, **k)

        return wrapper

    for method in ("cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, counted(method))
    _call(monkeypatch, True, x, y, s, 4)
    assert reads == []


def test_expert_ids_are_read_once_per_tensor(monkeypatch):
    s = _set("none")
    s.expert_ids = torch.arange(4, device=DEVICE)
    x, y = _inputs(8, batch=1, tokens=5, n_in=24)
    monkeypatch.setenv("KILORA_FUSED", "1")
    compute._EXPERT_ID_KEYS.clear()
    start = compute.EXPERT_ID_READS[0]
    with torch.no_grad(), compute.dropout_key(7, 3), compute.derived_call():
        for index in range(3):
            compute.run_layer(x, y, [s], 4, target=f"blk.{index}.attn1")
    assert compute.EXPERT_ID_READS[0] == start + 1
    assert compute._expert_id_key(s.expert_ids) == (0, 1, 2, 3)
    assert compute.EXPERT_ID_READS[0] == start + 1
    s.expert_ids.add_(0)
    assert compute._expert_id_key(s.expert_ids) == (0, 1, 2, 3)
    assert compute.EXPERT_ID_READS[0] == start + 2


def test_expert_id_key_takes_inference_tensors():
    with torch.inference_mode():
        ids = torch.arange(4, device=DEVICE)
    compute._EXPERT_ID_KEYS.clear()
    start = compute.EXPERT_ID_READS[0]
    assert compute._expert_id_key(ids) == (0, 1, 2, 3)
    assert compute._expert_id_key(ids) == (0, 1, 2, 3)
    assert compute.EXPERT_ID_READS[0] == start + 1


def test_full_mode_runs_with_inference_expert_ids(monkeypatch):
    s = _set("none")
    with torch.inference_mode():
        s.expert_ids = torch.arange(4, device=DEVICE)
    x, y = _inputs(8, batch=1, tokens=5, n_in=24)
    with torch.inference_mode(), compute.dropout_key(7, 3), compute.derived_call():
        out = compute.run_layer(x, y, [s], 4, target="blk.0.attn1")
    assert torch.isfinite(out).all()


def test_keys_mode_does_not_use_stacked_walks(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    s = _set("adaptive")
    x, y = _inputs(9, batch=1, tokens=5, n_in=24)
    calls, _ = _spy(monkeypatch)
    before = (fused.COUNTS.get("walks_fused", 0), fused.COUNTS.get("walks_eager", 0))
    on = _call(monkeypatch, True, x, y, s, 4, stacked=True)
    off = _call(monkeypatch, True, x, y, s, 4, stacked=False)
    assert torch.equal(on, off)
    assert all(w is None for w in calls)
    after = (fused.COUNTS.get("walks_fused", 0), fused.COUNTS.get("walks_eager", 0))
    assert after == before


def _write_calls(monkeypatch, s, stacked, calls=None):
    from types import SimpleNamespace

    from kilora import route_freeze
    from kilora.call_profile import CallProfile, profiling
    from kilora.live_experts import RouteRecord, recording_fused_routes

    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_MACRO", "1")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
    monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)
    session = SimpleNamespace(_source=SimpleNamespace(extra={}, name=s.key), state={TARGET: {}})
    x, y = _inputs(5, batch=2, tokens=6, n_in=24)
    record, outs, per_call, profile = RouteRecord(), [], [], CallProfile()
    with (
        torch.no_grad(),
        route_freeze.freezing({}),
        compute.sampling_progress(0.5),
        recording_fused_routes(session, record),
    ):
        for step, key in ((1, 4), (2, 9), (3, 13)):
            if calls is not None:
                calls.clear()
            with profiling(profile), compute.dropout_key(key, step), compute.derived_call():
                outs.append(run_layer(x, y, [s], 4, target=TARGET))
            if calls is not None:
                per_call.append(list(calls))
    return outs, record.mass.get(TARGET), per_call, profile


@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_the_write_calls_use_the_stacked_walks_with_the_same_loss(monkeypatch, gate_mode):
    s = _set(gate_mode)
    s.route_freeze = 0.4
    calls, _weights = _spy(monkeypatch)
    expected, mass_one, per_call_one, profile_one = _write_calls(monkeypatch, s, False, calls)
    actual, mass, per_call, profile = _write_calls(monkeypatch, s, True, calls)
    assert all(len(c) == 3 and all(w is None for w in c) for c in per_call_one)
    assert [len(c) for c in per_call] == [1, 1, 1]
    assert all(c[0] is not None and c[0].shape[0] == 3 for c in per_call)
    assert profile_one.walks == {0: 9}
    assert profile.walks == {3: 3}
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
    assert mass is not None and mass_one is not None
    assert mass.shape == mass_one.shape


def _depth_spy(monkeypatch):
    calls = []
    real = fused.run_fused
    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            calls.append((k.get("walks"), a[2] if len(a) > 2 else k.get("depth")))
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    return calls


def _write_model_call(monkeypatch, s, fused_on, scope_on):
    from kilora.comfyui import live_write

    monkeypatch.setenv("KILORA_FUSED", "1" if fused_on else "0")
    monkeypatch.setenv(live_write.CALL_SCOPE_ENV, "1" if scope_on else "0")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1")
    if fused_on:
        monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)
    else:
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    x, y = _inputs(5, batch=4, tokens=6, n_in=24)
    with torch.no_grad(), compute.dropout_key(7, 3), live_write.write_call_scope():
        return [run_layer(x, y, [s], 4, target=TARGET, index=index) for index in range(3)]


@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_a_write_call_takes_the_stacked_walks_with_the_carry_of_a_render_call(
    monkeypatch, gate_mode
):
    s = _set(gate_mode)
    calls = _depth_spy(monkeypatch)
    expected = _write_model_call(monkeypatch, s, False, True)
    assert all(w is None for w, _depth in calls)
    calls.clear()
    actual = _write_model_call(monkeypatch, s, True, True)
    assert len(calls) == 3
    assert all(w is not None and w.shape[0] == 3 for w, _depth in calls)
    assert calls[0][1] == 4
    assert all(depth < 4 for _w, depth in calls[1:])
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


def test_without_the_call_scope_every_layer_runs_as_a_first_block(monkeypatch):
    s = _set("none")
    calls = _depth_spy(monkeypatch)
    _write_model_call(monkeypatch, s, True, False)
    assert [depth for _w, depth in calls] == [4, 4, 4]


def test_the_single_walks_make_no_route_mass_note_per_depth_step(monkeypatch):
    from types import SimpleNamespace

    from kilora.live_experts import RouteRecord, recording_fused_routes

    s = _set("none")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "0")
    monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    notes = []
    real = compute._note_fused_mass_body

    def body(target, gates, sets, walks=False):
        notes.append(walks)
        return real(target, gates, sets, walks)

    monkeypatch.setattr(compute, "_note_fused_mass_body", body)
    calls = _depth_spy(monkeypatch)
    session = SimpleNamespace(_source=SimpleNamespace(extra={}, name=s.key), state={TARGET: {}})
    record = RouteRecord()
    x, y = _inputs(5, batch=2, tokens=6, n_in=24)
    with (
        torch.no_grad(),
        compute.dropout_key(7, 3),
        compute.derived_call(),
        recording_fused_routes(session, record),
    ):
        run_layer(x, y, [s], 4, target=TARGET)
    assert len(calls) == 3 and all(w is None for w, _depth in calls)
    assert notes == [True]
    assert record.mass[TARGET].shape[-1] == s.keys_in.shape[0]
