# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
import threading

import pytest
import torch

pytest.importorskip("triton")

from test_fused_input import DEVICE, TARGET, _imported, _inputs

from kilora import compute, fused, segments
from kilora.compute import run_layer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

EXPERTS = 7


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.delenv("KILORA_SEGMENTS", raising=False)


def _table(seed=0, rate=0.5):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand((EXPERTS, EXPERTS, EXPERTS), generator=gen) < rate).to(DEVICE)


def _skips(monkeypatch, table):
    monkeypatch.setattr(segments, "skip_decisions", lambda s: table)


def _set(routing="token_top1", gate_mode="none"):
    s = _imported(11, experts=EXPERTS, n_in=24, gate_mode=gate_mode, routing=routing)
    s.path_width = 3
    return s


def _region(tokens):
    return (torch.arange(tokens) % 3 != 0).float().view(1, tokens, 1).to(DEVICE)


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


def _call(monkeypatch, fused_on, x, y, s, depth, regions=None, stacked=True):
    monkeypatch.setenv("KILORA_FUSED", "1" if fused_on else "0")
    monkeypatch.setenv("KILORA_DERIVED_STACKED", "1" if stacked else "0")
    if not fused_on:
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    else:
        monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)
    with torch.no_grad(), compute.dropout_key(7, 3):
        return run_layer(x, y, [s], depth, target=TARGET, regions=regions)


def _eager_counts():
    return fused.COUNTS["import_eager"], dict(fused.IMPORT_REASONS)


@pytest.mark.parametrize("depth", [4, 7])
@pytest.mark.parametrize("stacked", [True, False])
def test_depth_skip_calls_leave_the_eager_fallback(monkeypatch, depth, stacked):
    _skips(monkeypatch, _table())
    s = _set()
    x, y = _inputs(3, batch=1, tokens=6, n_in=24)
    calls = _spy(monkeypatch)
    eager_before, reasons_before = _eager_counts()
    fused_before = fused.COUNTS["import_fused"]
    walks_before = fused.COUNTS.get("walks_fused", 0)
    _call(monkeypatch, True, x, y, s, depth, stacked=stacked)
    eager_after, reasons_after = _eager_counts()
    added = {
        k: v - reasons_before.get(k, 0)
        for k, v in reasons_after.items()
        if v != reasons_before.get(k, 0)
    }
    print(
        f"depth {depth}: before {s.path_width} eager (depth skips), "
        f"after {eager_after - eager_before} eager by reason {added}"
    )
    fused_calls = 1 if stacked else s.path_width
    assert calls == [True] * fused_calls
    assert fused.COUNTS["import_fused"] - fused_before == fused_calls
    assert fused.COUNTS.get("walks_fused", 0) - walks_before == (1 if stacked else 0)
    assert eager_after == eager_before
    assert not added


@pytest.mark.parametrize("depth", [4, 7])
@pytest.mark.parametrize("with_skips", [False, True])
@pytest.mark.parametrize("with_region", [False, True])
@pytest.mark.parametrize("routing", ["token_top1", "token"])
@pytest.mark.parametrize("stacked", [True, False])
def test_fused_skips_match_eager_walks(
    monkeypatch, depth, with_skips, with_region, routing, stacked
):
    if with_skips:
        _skips(monkeypatch, _table(seed=depth))
    else:
        monkeypatch.setenv("KILORA_SEGMENTS", "0")
    s = _set(routing)
    x, y = _inputs(4, batch=1, tokens=6, n_in=24)
    regions = [_region(6)] if with_region else None
    expected = _call(monkeypatch, False, x, y, s, depth, regions)
    calls = _spy(monkeypatch)
    actual = _call(monkeypatch, True, x, y, s, depth, regions, stacked=stacked)
    if with_skips and with_region:
        assert calls == []
    elif with_skips or stacked:
        assert len(calls) == (1 if stacked else s.path_width)
    assert all(c == with_skips for c in calls)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth", [3, 4, 7])
@pytest.mark.parametrize("with_region", [False, True])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_fused_chain_with_skips_matches_eager_loop(monkeypatch, depth, with_region, gate_mode):
    table = torch.ones((EXPERTS, EXPERTS, EXPERTS), dtype=torch.bool, device=DEVICE)
    _skips(monkeypatch, table)
    s = _set("token_top1", gate_mode)
    x, y = _inputs(5, batch=2, tokens=5, n_in=24)
    region = _region(5) if with_region else None
    entry = torch.zeros((*x.shape[:-1], EXPERTS), device=DEVICE)
    entry[..., 2] = 1
    with torch.no_grad(), compute.dropout_key(2, 1):
        monkeypatch.setenv("KILORA_FUSED", "0")
        expected = compute._run_input(x, y, [s], [0], depth, [region], TARGET, entry=entry)
        _skips(monkeypatch, None)
        unskipped = compute._run_input(x, y, [s], [0], depth, [region], TARGET, entry=entry)
        _skips(monkeypatch, table)
        monkeypatch.setenv("KILORA_FUSED", "1")
        if with_region:
            reason = "depth skips with a region"
            assert fused.import_skip_reason(s, depth, None, region) == reason
            before = fused.IMPORT_REASONS.get(reason, 0)
            actual = compute._run_input_fused(x, y, [s], [0], depth, [region], TARGET, entry=entry)
            assert fused.IMPORT_REASONS.get(reason, 0) == before + 1
            with pytest.raises(ValueError, match="import_skip_reason"):
                full = region.expand(*y.shape[:-1], 1).contiguous()
                fused.run_fused(
                    y, [s], depth, False, TARGET, [full], x=x, name="n", entry=entry, skips=table
                )
            assert not torch.allclose(expected, unskipped)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            return
        actual = fused.run_fused(
            y,
            [s],
            depth,
            False,
            TARGET,
            [region],
            x=x,
            name=f"{TARGET}#input0",
            entry=entry,
            skips=table,
        )
    assert not torch.allclose(expected, unskipped)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_skips_outside_the_chain_name_their_reason(monkeypatch):
    _skips(monkeypatch, _table())
    s = _set()
    assert fused.import_skip_reason(s, 4) is None
    assert fused.import_skip_reason(s, 4.5) == "depth skips with a fractional depth"
    td = object()
    assert fused.import_skip_reason(s, 4, td) is None
    region = _region(6)
    assert fused.import_skip_reason(s, 4, None, region) == "depth skips with a region"
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    assert fused.import_skip_reason(s, 4) == "depth skips without the import depth chain"
    x, y = _inputs(3, batch=1, tokens=6, n_in=24)
    before = fused.IMPORT_REASONS.get("depth skips without the import depth chain", 0)
    _call(monkeypatch, True, x, y, s, 4)
    after = fused.IMPORT_REASONS.get("depth skips without the import depth chain", 0)
    assert after == before + 3
    with pytest.raises(ValueError, match="import_skip_reason"), torch.no_grad():
        fused.run_fused(y, [s], 4, False, TARGET, [None], x=x, name="n", skips=_table())


def test_keys_mode_takes_no_skips_and_stays_bit_identical(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    _skips(monkeypatch, _table())
    s = _set("token")
    s.path_width = 1
    x, y = _inputs(6, batch=1, tokens=6, n_in=24)
    calls = _spy(monkeypatch)
    monkeypatch.setenv("KILORA_FUSED", "1")
    with torch.no_grad(), compute.dropout_key(1, 1):
        dispatched = compute._run_input_fused(x, y, [s], [0], 4, [None], TARGET)
        direct = fused.run_fused(y, [s], 4, False, TARGET, [None], x=x, name=f"{TARGET}#input0")
    assert calls == [False, False]
    assert torch.equal(dispatched, direct)
