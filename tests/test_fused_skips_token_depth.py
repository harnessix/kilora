# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
import threading

import pytest
import torch

pytest.importorskip("triton")

from test_fused import DEVICE
from test_fused_input import _imported, _inputs
from test_token_depth_cost import _layer_map
from test_token_depth_rows import _record

from kilora import compute, fused, segments
from kilora.compute import run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

TARGET = "blk.0.attn1"
EXPERTS = 7
KINDS = ["mask", "mask_soft", "auto2", "auto3"]
REFUSAL = "depth skips with a per-token depth"
GATHERED = "depth skips on the gathered rows"


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_DEPTH_ROWS", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.delenv("KILORA_SEGMENTS", raising=False)


def _table(seed=0, rate=0.5):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand((EXPERTS, EXPERTS, EXPERTS), generator=gen) < rate).to(DEVICE)


def _every_relay():
    return torch.ones((EXPERTS, EXPERTS, EXPERTS), dtype=torch.bool, device=DEVICE)


def _skips(monkeypatch, table):
    monkeypatch.setattr(segments, "skip_decisions", lambda s: table)


def _set(seed, gate_mode="adaptive", routing="token_top1", width=1, **kw):
    s = _imported(seed, experts=EXPERTS, n_in=24, gate_mode=gate_mode, routing=routing, **kw)
    s.path_width = width
    return s


def _spy(monkeypatch):
    calls = []
    real = fused.run_fused
    me = threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            calls.append(k.get("skips") is not None)
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    return calls


def _call(monkeypatch, on, x, y0, s, d, regions=None):
    monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
    with torch.no_grad(), compute.dropout_key(7, 3):
        return run_layer(x, y0, [s], 3, target=TARGET, regions=regions, token_depth=d)


def _close(eager, got):
    assert torch.allclose(eager, got, rtol=1e-4, atol=1e-5), (eager - got).abs().max()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_every_imported_call_with_skips_and_a_map_runs_fused(monkeypatch, kind, width, gate_mode):
    _skips(monkeypatch, _table())
    s = _set(61, gate_mode, width=width)
    x, y0 = _inputs(62, n_in=24)
    d = _layer_map(kind)
    calls = _spy(monkeypatch)
    eager_before = fused.COUNTS["import_eager"]
    fused_before = fused.COUNTS["import_fused"]
    reasons_before = dict(fused.IMPORT_REASONS)
    _call(monkeypatch, True, x, y0, s, d)
    added = {
        k: v - reasons_before.get(k, 0)
        for k, v in fused.IMPORT_REASONS.items()
        if v != reasons_before.get(k, 0)
    }
    assert calls == [True] * width
    assert fused.COUNTS["import_fused"] - fused_before == width
    assert fused.COUNTS["import_eager"] == eager_before
    assert not added
    assert REFUSAL not in added


def test_skips_with_a_map_take_no_refusal():
    s = _imported(63, experts=EXPERTS, n_in=24)
    td = compute._TokenDepth(_layer_map("auto2"), 5.0)
    assert fused.import_skip_reason(s, 5, td) is None
    assert fused.import_skip_reason(s, 4.5) == "depth skips with a fractional depth"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_skips_with_a_map_match_the_token_loop(monkeypatch, kind, gate_mode, width, seed):
    _skips(monkeypatch, _table(seed))
    s = _set(64 + seed, gate_mode, width=width)
    x, y0 = _inputs(66 + seed, n_in=24)
    d = _layer_map(kind)
    eager = _call(monkeypatch, False, x, y0, s, d)
    calls = _spy(monkeypatch)
    got = _call(monkeypatch, True, x, y0, s, d)
    assert calls == [True] * width
    _close(eager, got)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("width", [1, 3])
def test_every_relay_with_a_map_matches_the_token_loop(monkeypatch, kind, gate_mode, width):
    s = _set(68, gate_mode, width=width)
    x, y0 = _inputs(69, n_in=24)
    d = _layer_map(kind)
    _skips(monkeypatch, None)
    unskipped = _call(monkeypatch, False, x, y0, s, d)
    _skips(monkeypatch, _every_relay())
    eager = _call(monkeypatch, False, x, y0, s, d)
    got = _call(monkeypatch, True, x, y0, s, d)
    assert not torch.allclose(eager, unskipped)
    _close(eager, got)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("table", ["every", "planted"])
def test_the_gated_chain_runs_the_skips_on_the_gathered_rows(monkeypatch, kind, table):
    table = _every_relay() if table == "every" else _table(5)
    _skips(monkeypatch, table)
    s = _set(70, "adaptive")
    x, y0 = _inputs(71, n_in=24)
    d = _layer_map(kind)
    td = compute._TokenDepth(compute._token_values(d, y0), float(d.max()), d)
    with torch.no_grad(), compute.dropout_key(2, 1):
        monkeypatch.setenv("KILORA_FUSED", "0")
        eager = compute._run_input(x, y0, [s], [0], td.steps, [None], TARGET, td)
        _skips(monkeypatch, None)
        unskipped = compute._run_input(x, y0, [s], [0], td.steps, [None], TARGET, td)
        _skips(monkeypatch, table)
        monkeypatch.setenv("KILORA_FUSED", "1")
        seen = _record(monkeypatch)
        got = fused.run_fused(
            y0,
            [s],
            td.steps,
            False,
            TARGET,
            [None],
            x=x,
            name=f"{TARGET}#input0",
            td=td,
            skips=table,
        )
    flat = d.reshape(-1)
    for key in ("update", "bound", "carry"):
        assert len(seen[key]) == td.steps, key
        for k, rows in enumerate(seen[key], start=1):
            assert rows == sorted(torch.nonzero(flat > k - 1).reshape(-1).tolist()), (key, k)
    if table is not None and bool(table.all()):
        assert not torch.allclose(eager, unskipped)
    _close(eager, got)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("table", ["every", "planted"])
def test_a_call_with_skips_on_the_gathered_rows_runs_fused(monkeypatch, kind, table):
    table = _every_relay() if table == "every" else _table(5)
    _skips(monkeypatch, table)
    s = _set(70, "adaptive")
    x, y0 = _inputs(71, n_in=24)
    d = _layer_map(kind)
    td = compute._TokenDepth(compute._token_values(d, y0), float(d.max()), d)
    with torch.no_grad(), compute.dropout_key(2, 1):
        monkeypatch.setenv("KILORA_FUSED", "0")
        eager = compute._run_input(x, y0, [s], [0], td.steps, [None], TARGET, td)
        monkeypatch.setenv("KILORA_FUSED", "1")
        calls = _spy(monkeypatch)
        reasons_before = dict(fused.IMPORT_REASONS)
        got = compute._run_input_fused(x, y0, [s], [0], td.steps, [None], TARGET, td)
    added = {
        k: v - reasons_before.get(k, 0)
        for k, v in fused.IMPORT_REASONS.items()
        if v != reasons_before.get(k, 0)
    }
    assert GATHERED not in added
    assert not added
    assert calls == [compute._import_skips(s, td.steps) is not None]
    assert fused.import_skip_reason(s, td.steps, td) is None
    _close(eager, got)


@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_other_routings_with_skips_and_a_map_match_the_token_loop(monkeypatch, routing, gate_mode):
    _skips(monkeypatch, _table(2))
    s = _set(72, gate_mode, routing=routing)
    x, y0 = _inputs(73, n_in=24)
    d = _layer_map("auto2")
    _close(_call(monkeypatch, False, x, y0, s, d), _call(monkeypatch, True, x, y0, s, d))


@pytest.mark.parametrize("kind", ["mask", "auto2", "auto3"])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("width", [1, 3])
def test_skips_with_a_map_under_the_step_carry_match_the_token_loop(
    monkeypatch, kind, gate_mode, width
):
    _skips(monkeypatch, _table(3))
    s = _set(74, gate_mode, width=width)
    x, y0 = _inputs(75, n_in=24)
    d = _layer_map(kind)
    outputs = {}
    for on in (False, True):
        with compute.step_carry({}):
            first = _call(monkeypatch, on, x, y0, s, d)
            second = _call(monkeypatch, on, x, y0, s, d)
        outputs[on] = (first, second)
    _close(outputs[False][0], outputs[True][0])
    _close(outputs[False][1], outputs[True][1])


@pytest.mark.parametrize("kind", ["mask_soft", "auto2", "auto3"])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("width", [1, 3])
def test_skips_with_a_map_and_frozen_routes_match_the_token_loop(
    monkeypatch, kind, gate_mode, width
):
    _skips(monkeypatch, _every_relay())
    s = dataclasses.replace(_set(76, gate_mode, width=width), route_freeze=0.0)
    x, y0 = _inputs(77, n_in=24)
    d = _layer_map(kind)
    _close(_call(monkeypatch, False, x, y0, s, d), _call(monkeypatch, True, x, y0, s, d))


def test_shared_input_skip_decisions_match_token_loop(monkeypatch):
    _skips(monkeypatch, _table(0))
    s = _set(64, "none", width=1)
    x, y0 = _inputs(66, n_in=24)
    d = _layer_map("mask")
    td = compute._TokenDepth(compute._token_values(d, y0), float(d.max()), d)
    snapshots = {False: [], True: []}
    original = segments.skip_mask
    backend = False

    def record(table, previous, current, following):
        mask = original(table, previous, current, following)
        snapshots[backend].append(
            (previous.clone(), current.clone(), following.clone(), mask.clone())
        )
        return mask

    monkeypatch.setattr(segments, "skip_mask", record)
    with torch.no_grad(), compute.dropout_key(7, 3):
        entry = compute.derived_entries(s, x, f"{TARGET}#input0", 1)[0]
        with compute.eager():
            expected = compute._run_input(
                x, y0, [s], [0], td.steps, [None], TARGET, td, entry=entry
            )
        backend = True
        actual = fused.run_fused(
            y0,
            [s],
            td.steps,
            False,
            TARGET,
            [None],
            x=x,
            name=f"{TARGET}#input0",
            td=td,
            entry=entry,
            skips=_table(0),
        )
    assert len(snapshots[False]) == len(snapshots[True])
    for step, (eager, got) in enumerate(zip(snapshots[False], snapshots[True], strict=True), 2):
        if not torch.equal(eager[-1], got[-1]):
            row = int((eager[-1] != got[-1]).reshape(-1).nonzero()[0])
            details = {
                label: [value.reshape(-1, EXPERTS)[row].tolist() for value in record[:3]]
                for label, record in (("eager", eager), ("fused", got))
            }
            pytest.fail(f"skip decisions differ at step {step}, row {row}: {details}")
    _close(expected, actual)
