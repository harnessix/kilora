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
from test_fused_token_depth import _map, _run, _same
from test_token_depth_cost import _layer_map

from kilora import compute, fused, segments

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

KINDS = ["mask", "mask_soft", "auto2", "auto3", "zeros"]


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_DEPTH_ROWS", "1")


def _depth(kind, batch=2, tokens=37):
    if kind == "zeros":
        t = torch.arange(tokens, dtype=torch.float32)
        d = torch.where(t % 4 == 0, torch.zeros_like(t), 0.5 + (t % 5) * 0.9)
        return d[None].expand(batch, tokens).contiguous().to(DEVICE)
    if kind == "per_sample":
        a = _layer_map("mask", 1, tokens)
        b = _layer_map("auto2", 1, tokens)
        return torch.cat((a, b.flip(-1))).contiguous()
    return _layer_map(kind, batch, tokens)


def _expected(d, k):
    return sorted(torch.nonzero(d.reshape(-1) > k - 1).reshape(-1).tolist())


@pytest.mark.parametrize("kind", [*KINDS, "per_sample"])
def test_each_step_gathers_the_tokens_it_runs(kind):
    d = _depth(kind)
    T = d.shape[-1]
    td = compute._TokenDepth(d, float(d.max()), d)
    plan = td.gather(T)
    assert td.gather(T) is plan
    flat = td.d.reshape(-1)
    last = torch.ceil(flat).long()
    for k in range(1, td.steps + 1):
        rows, entries, counts, widest, active = plan.step(k)
        assert sorted(rows.tolist()) == _expected(d, k)
        assert active == len(_expected(d, k))
        local = plan.perm_local.long()
        for b in range(plan.samples):
            mine = (local[b, : int(counts[b])] + b * T).tolist()
            assert sorted(mine) == sorted(r for r in _expected(d, k) if r // T == b)
        assert widest == int(counts.max())
        rows_l = rows.tolist()
        for row, entry in zip(rows_l, entries.tolist(), strict=True):
            if int(last[row]) == k:
                assert entry == int(plan.high[row])
            elif int(last[row]) == k + 1 and float(flat[row]) < k + 1:
                assert entry == int(plan.low[row])
            else:
                assert entry == -1
        picked = plan.picked(k)
        assert set(picked.tolist()) <= set(rows_l)
    high = plan.high
    assert bool(((high >= 0) == (flat > 0)).all())
    fractional = (flat > 1) & (flat != torch.floor(flat))
    assert bool(((plan.low >= 0) == fractional).all())
    assert plan.total == int((flat > 0).sum()) + int(fractional.sum())
    assert sorted(plan.high[flat > 0].tolist() + plan.low[fractional].tolist()) == list(
        range(plan.total)
    )
    assert bool((plan.code[flat == 0] == -2).all())


def _record(monkeypatch):
    seen = {"update": [], "bound": [], "carry": []}
    me = threading.get_ident()

    def wrap(name, key):
        real = getattr(fused, name)

        class Launcher:
            def __getitem__(self, grid):
                launch = real[grid]

                def go(*args, **kwargs):
                    if threading.get_ident() == me and kwargs.get("GATHER"):
                        if key == "update":
                            perm = kwargs["perm_ptr"].long()
                            counts = kwargs["count_ptr"].tolist()
                            T = perm.shape[1]
                            rows = [
                                b * T + t
                                for b, n in enumerate(counts)
                                for t in perm[b, :n].tolist()
                            ]
                        else:
                            rows = kwargs["rows_ptr"].tolist()
                        seen[key].append(sorted(rows))
                    return launch(*args, **kwargs)

                return go

        monkeypatch.setattr(fused, name, Launcher())

    wrap("_update_kernel", "update")
    wrap("_import_bound_kernel", "bound")
    wrap("_import_carry_kernel", "carry")
    return seen


@pytest.mark.parametrize("kind", [*KINDS, "per_sample"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_the_kernels_run_the_tokens_of_each_step(kind, routing, monkeypatch):
    x, y0 = _inputs(41)
    s = _imported(42, routing=routing, gate_mode="adaptive")
    d = _depth(kind)
    seen = _record(monkeypatch)
    _, used = _run(monkeypatch, True, x, y0, [s], 3, d)
    assert used
    steps = int(torch.ceil(d.max()))
    for key in ("update", "bound", "carry"):
        assert len(seen[key]) == steps, key
        for k, rows in enumerate(seen[key], start=1):
            assert rows == _expected(d, k), (key, k)


def test_uniform_depth_gathers_nothing(monkeypatch):
    x, y0 = _inputs(43)
    s = _imported(44, gate_mode="adaptive")
    seen = _record(monkeypatch)
    _run(monkeypatch, True, x, y0, [s], 3, None)
    _run(monkeypatch, True, x, y0, [s], 3, torch.full((2, 37), 3.0, device=DEVICE))
    assert seen == {"update": [], "bound": [], "carry": []}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("routing", ["token", "tensor", "token_top1"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_gathered_rows_match_the_token_loop(kind, routing, carry, monkeypatch):
    x, y0 = _inputs(45)
    s = _imported(46, routing=routing, gate_mode="adaptive", carry=carry)
    _same(monkeypatch, x, y0, [s], _depth(kind))


@pytest.mark.parametrize("kind", ["steps", "fractional", "zeros"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_gathered_rows_with_every_token_active_match_the_token_loop(
    kind, routing, carry, monkeypatch
):
    x, y0 = _inputs(1)
    s = _imported(2, routing=routing, gate_mode="adaptive", carry=carry)
    seen = _record(monkeypatch)
    _same(monkeypatch, x, y0, [s], _map(kind))
    assert seen["update"]


def test_each_sample_its_own_map_matches_the_token_loop(monkeypatch):
    x, y0 = _inputs(47)
    s = _imported(48, gate_mode="adaptive")
    _same(monkeypatch, x, y0, [s], _depth("per_sample"))


@pytest.mark.parametrize("kind", ["mask", "mask_soft", "auto2", "auto3"])
def test_gathered_rows_with_frozen_routes_match_the_token_loop(kind, monkeypatch):
    x, y0 = _inputs(49)
    s = dataclasses.replace(_imported(50, gate_mode="adaptive"), route_freeze=0.0)
    _same(monkeypatch, x, y0, [s], _depth(kind))


@pytest.mark.parametrize("kind", ["mask_soft", "auto2"])
def test_gathered_rows_with_a_region_match_the_token_loop(kind, monkeypatch):
    x, y0 = _inputs(51)
    s = _imported(52, gate_mode="adaptive")
    region = (torch.arange(37) % 3 != 0).float().view(1, 37, 1).to(DEVICE)
    _same(monkeypatch, x, y0, [s], _depth(kind), regions=[region])


@pytest.mark.parametrize("kind", ["mask", "auto3"])
def test_gathered_rows_with_derived_scores_match_the_token_loop(kind, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    x, y0 = _inputs(53)
    s = _imported(54, gate_mode="adaptive")
    _same(monkeypatch, x, y0, [s], _depth(kind))


@pytest.mark.parametrize("kind", ["mask", "auto2"])
def test_gathered_rows_under_the_step_carry_match_the_token_loop(kind, monkeypatch):
    x, y0 = _inputs(55)
    s = _imported(56, gate_mode="adaptive")
    d = _depth(kind)
    with compute.step_carry({}):
        first = _same(monkeypatch, x, y0, [s], d)
    with compute.step_carry({}):
        again = _same(monkeypatch, x, y0, [s], d)
    assert torch.equal(first, again)


@pytest.mark.parametrize("kind", ["mask", "auto2"])
def test_depth_skips_with_a_map_match_the_token_loop(kind, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    experts = 7
    gen = torch.Generator().manual_seed(0)
    table = (torch.rand((experts,) * 3, generator=gen) < 0.5).to(DEVICE)
    monkeypatch.setattr(segments, "skip_decisions", lambda s: table)
    x, y0 = _inputs(57, n_in=24)
    s = _imported(58, experts=experts, n_in=24, gate_mode="adaptive", routing="token_top1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    with torch.no_grad(), compute.dropout_key(7, 3):
        eager = compute.run_layer(x, y0, [s], 3, target="blk.0.attn1", token_depth=_depth(kind))
    monkeypatch.setenv("KILORA_FUSED", "1")
    eager_before = fused.COUNTS["import_eager"]
    with torch.no_grad(), compute.dropout_key(7, 3):
        got = compute.run_layer(x, y0, [s], 3, target="blk.0.attn1", token_depth=_depth(kind))
    assert fused.COUNTS["import_eager"] == eager_before
    assert torch.allclose(eager, got, rtol=1e-4, atol=1e-5), (eager - got).abs().max()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["mask_soft", "zeros"])
def test_gathered_rows_in_model_types_match_the_token_loop(dtype, kind, monkeypatch):
    x, y0 = _inputs(59)
    s = _imported(60, gate_mode="adaptive")
    _same(monkeypatch, x.to(dtype), y0.to(dtype), [s], _depth(kind))
