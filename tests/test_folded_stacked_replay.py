# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
import threading

import pytest
import torch

pytest.importorskip("triton")

from test_fused_input import _imported, _inputs

from kilora import compute, fused, route_freeze
from kilora.compute import run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

TARGET = "blk.0.attn1"
KEY = ("derived_macro", TARGET, 0)


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_DERIVED_MACRO", "1")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)


def _set(gate_mode):
    s = _imported(11, experts=7, n_in=24, gate_mode=gate_mode)
    s.path_width = 3
    s.route_freeze = 0.4
    return s


def _spy(monkeypatch):
    calls = []
    real = fused.run_fused
    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            calls.append(k.get("walks"))
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    return calls


def _frozen_calls(monkeypatch, s, depth, stacked, calls=None):
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
    x, y = _inputs(5, batch=1, tokens=6, n_in=24)
    outs, per_call = [], []
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        for step, key in ((1, 4), (2, 9), (3, 13)):
            if calls is not None:
                calls.clear()
            with compute.dropout_key(key, step), compute.derived_call():
                outs.append(run_layer(x, y, [s], depth, target=TARGET))
            if calls is not None:
                per_call.append(list(calls))
            assert KEY in cache and cache[KEY]["chain"]
    return outs, per_call


@pytest.mark.parametrize("depth", [4, 7])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_folded_record_on_the_stacked_chain_matches_the_walks_one_by_one(
    monkeypatch, depth, gate_mode
):
    s = _set(gate_mode)
    expected, _ = _frozen_calls(monkeypatch, s, depth, stacked=False)
    actual, _ = _frozen_calls(monkeypatch, s, depth, stacked=True)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
    for a in actual[1:]:
        torch.testing.assert_close(a, actual[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth", [4, 7])
def test_folded_record_on_the_stacked_chain_matches_the_eager_loop(monkeypatch, depth):
    s = _set("adaptive")
    actual, _ = _frozen_calls(monkeypatch, s, depth, stacked=True)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    expected, _ = _frozen_calls(monkeypatch, s, depth, stacked=True)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth", [4, 7])
def test_calls_after_the_freeze_count_the_folded_path(monkeypatch, depth):
    s = _set("adaptive")
    calls = _spy(monkeypatch)

    before = dict(fused.COUNTS)
    _, per_call = _frozen_calls(monkeypatch, s, depth, stacked=False, calls=calls)
    assert [len(c) for c in per_call] == [3, 3, 3]
    assert all(w is None for c in per_call for w in c)
    assert fused.COUNTS.get("folded_eager", 0) == before.get("folded_eager", 0) + 2
    assert fused.COUNTS.get("folded_fused", 0) == before.get("folded_fused", 0)
    assert fused.REASONS.get("folded replay with the stacked walks turned off", 0) >= 2

    before = dict(fused.COUNTS)
    _, per_call = _frozen_calls(monkeypatch, s, depth, stacked=True, calls=calls)
    assert [len(c) for c in per_call] == [1, 1, 1]
    assert all(c[0] is not None and c[0].shape[0] == 3 for c in per_call)
    assert fused.COUNTS.get("folded_fused", 0) == before.get("folded_fused", 0) + 2
    assert fused.COUNTS.get("folded_eager", 0) == before.get("folded_eager", 0)
    assert fused.COUNTS.get("walks_fused", 0) == before.get("walks_fused", 0) + 1
    assert fused.COUNTS["import_fused"] == before["import_fused"] + 3


def test_stacked_record_keeps_each_walk_record(monkeypatch):
    s = _set("adaptive")
    x, y = _inputs(6, batch=1, tokens=6, n_in=24)
    records = []
    for stacked in (False, True):
        monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
        with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
            with compute.dropout_key(4, 1), compute.derived_call():
                run_layer(x, y, [s], 4, target=TARGET)
            records.append(cache[KEY])
    one, both = records
    assert len(one["paths"]) == len(both["paths"]) == 3
    for a, b in zip(one["paths"], both["paths"], strict=True):
        assert len(a["routes"]) == len(b["routes"]) == 4
        assert len(a["read_keeps"]) == len(b["read_keeps"]) == 3
        assert len(a["output_gates"]) == len(b["output_gates"]) > 0
        for (aa, ab), (ba, bb) in zip(a["output_gates"], b["output_gates"], strict=True):
            torch.testing.assert_close(ba, aa, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(bb, ab, rtol=1e-4, atol=1e-5)
        for ra, rb in zip(a["routes"], b["routes"], strict=True):
            torch.testing.assert_close(rb, ra, rtol=1e-4, atol=1e-5)
        for ka, kb in zip(a["read_keeps"], b["read_keeps"], strict=True):
            assert torch.equal(kb, ka)
    torch.testing.assert_close(both["path_weights"], one["path_weights"])


def test_keys_mode_is_unchanged(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")

    def refuse(*a, **k):
        raise AssertionError("keys mode replayed a full-mode folded record")

    s = _set("adaptive")
    monkeypatch.setattr(compute, "_folded_stacked_replay", refuse)
    outs = []
    for stacked in (False, True):
        outs.append(_frozen_calls_keys(monkeypatch, s, stacked))
    for a, b in zip(*outs, strict=True):
        assert torch.equal(a, b)


def _frozen_calls_keys(monkeypatch, s, stacked):
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
    x, y = _inputs(5, batch=1, tokens=6, n_in=24)
    outs = []
    with torch.no_grad(), route_freeze.freezing({}), compute.sampling_progress(0.5):
        for step, key in ((1, 4), (2, 9)):
            with compute.dropout_key(key, step), compute.derived_call():
                outs.append(run_layer(x, y, [s], 4, target=TARGET))
    return outs


def test_replay_stacks_the_record_once(monkeypatch):
    s = _set("adaptive")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1")
    x, y = _inputs(8, batch=1, tokens=6, n_in=24)
    seen = []
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        for step, key in ((1, 4), (2, 9), (3, 13)):
            with compute.dropout_key(key, step), compute.derived_call():
                run_layer(x, y, [s], 4, target=TARGET)
            seen.append(cache[KEY].get("stacked_forced"))
        saved = cache[KEY]
    assert seen[0] is None
    assert seen[1] is not None and seen[2] is seen[1]
    routes = seen[1]["forced_routes"]
    for n, state in enumerate(saved["paths"]):
        for step, route in zip(routes, state["routes"], strict=True):
            assert route.data_ptr() == step[n].data_ptr()
