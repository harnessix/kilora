# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
import threading

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


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")


def _map(kind, batch=2, tokens=37):
    t = torch.arange(tokens, dtype=torch.float32)
    if kind == "constant":
        d = torch.full((tokens,), 3.0)
    elif kind == "steps":
        d = 1.0 + (t * 3 // tokens)
    elif kind == "fractional":
        d = 0.5 + t * 3.0 / tokens
    else:
        d = torch.where(t % 4 == 0, torch.zeros_like(t), 1.0 + (t % 3))
    return d[None].expand(batch, tokens).to(DEVICE)


def _run(monkeypatch, on, x, y0, sets, depth, token_depth, regions=None):
    monkeypatch.setenv("KILORA_FUSED", "1" if on else "0")
    seen, real, me = [], fused.run_fused, threading.get_ident()

    def spy(*a, **k):
        if threading.get_ident() == me:
            seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    with torch.no_grad(), compute.dropout_key(7, 3):
        y = run_layer(x, y0, sets, depth, target=TARGET, regions=regions, token_depth=token_depth)
    return y, bool(seen)


def _same(monkeypatch, x, y0, sets, token_depth, regions=None):
    eager, used = _run(monkeypatch, False, x, y0, sets, 3, token_depth, regions)
    assert not used
    got, used = _run(monkeypatch, True, x, y0, sets, 3, token_depth, regions)
    assert used
    rtol = 1e-4 if y0.dtype == torch.float32 else 4 * torch.finfo(y0.dtype).eps
    atol = 1e-5 if y0.dtype == torch.float32 else rtol
    assert torch.allclose(eager.float(), got.float(), rtol=rtol, atol=atol), (
        (eager - got).abs().max()
    )
    return eager


@pytest.mark.parametrize("kind", ["steps", "fractional", "zeros"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_imported_set_with_a_depth_per_token(kind, routing, gate, carry, monkeypatch):
    x, y0 = _inputs(1)
    s = _imported(2, routing=routing, gate_mode=gate, carry=carry)
    _same(monkeypatch, x, y0, [s], _map(kind))


@pytest.mark.parametrize("kind", ["steps", "fractional", "zeros"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_trained_set_with_a_depth_per_token(kind, routing, carry, monkeypatch):
    _, y0 = _inputs(3)
    s = dataclasses.replace(_set(4, routing=routing, carry=carry), gate_mode="none")
    _same(monkeypatch, y0, y0, [s], _map(kind))


def test_imported_set_with_a_region_and_token_depth(monkeypatch):
    x, y0 = _inputs(5)
    s = _imported(6, routing="tensor")
    region = (torch.arange(37) < 25).float()[None].expand(2, 37).to(DEVICE)
    _same(monkeypatch, x, y0, [s], _map("fractional"), regions=[region])


@pytest.mark.parametrize("form", ["input", "state"])
def test_constant_map_is_the_uniform_depth(form, monkeypatch):
    x, y0 = _inputs(7)
    if form == "input":
        s = _imported(8)
    else:
        s, x = dataclasses.replace(_set(8), gate_mode="none"), y0
    a, _ = _run(monkeypatch, True, x, y0, [s], 3, _map("constant"))
    b, _ = _run(monkeypatch, True, x, y0, [s], 3, None)
    assert torch.equal(a, b)


def test_a_gated_trained_set_with_token_depth_runs_eager(monkeypatch):
    _, y0 = _inputs(9)
    s = dataclasses.replace(_set(10), gate_mode="adaptive")
    _, used = _run(monkeypatch, True, y0, y0, [s], 3, _map("steps"))
    assert not used


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("form", ["input", "state"])
def test_ungated_depth_map_uses_one_write_and_no_torch_depth_bookkeeping(monkeypatch, dtype, form):
    x, y0 = _inputs(21)
    if form == "state":
        x = y0
    x, y0 = x.to(dtype), y0.to(dtype)
    s = (
        _imported(22, gate_mode="none")
        if form == "input"
        else dataclasses.replace(_set(22), gate_mode="none")
    )
    depth = _map("fractional")
    _run(monkeypatch, True, x, y0, [s], 3, depth)
    calls = []
    matmul = torch.matmul

    def count_matmul(*args, **kwargs):
        calls.append(1)
        return matmul(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("depth bookkeeping must run in the kernels")

    monkeypatch.setattr(torch, "matmul", count_matmul)
    monkeypatch.setattr(compute._TokenDepth, "full", forbidden)
    monkeypatch.setattr(compute._TokenDepth, "blend", forbidden)
    _, used = _run(monkeypatch, True, x, y0, [s], 3, depth)
    assert used
    assert len(calls) == 1


def test_noncontiguous_depth_map(monkeypatch):
    x, y0 = _inputs(23)
    s = _imported(24, gate_mode="none")
    depth = _map("fractional").transpose(0, 1).contiguous().transpose(0, 1)
    assert not depth.is_contiguous()
    _same(monkeypatch, x, y0, [s], depth)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("form", ["input", "state"])
def test_fractional_depth_map_model_types_match_eager(monkeypatch, dtype, form):
    x, y0 = _inputs(1)
    if form == "input":
        s = _imported(2, gate_mode="none")
    else:
        s, x = dataclasses.replace(_set(2), gate_mode="none"), y0
    _same(monkeypatch, x.to(dtype), y0.to(dtype), [s], _map("fractional"))


def test_bfloat16_coefficient_blend_improves_seeded_precision(monkeypatch):
    x, y0 = _inputs(1)
    s = _imported(2, gate_mode="none")
    depth = _map("fractional")
    reference, _ = _run(monkeypatch, False, x, y0, [s], 3, depth)
    legacy = dataclasses.replace(s, expert_offsets=[None] * s.keys_in.shape[0])
    old, _ = _run(monkeypatch, False, x.bfloat16(), y0.bfloat16(), [legacy], 3, depth)
    new, used = _run(monkeypatch, True, x.bfloat16(), y0.bfloat16(), [s], 3, depth)
    assert used
    assert (new.float() - reference).square().mean() <= (old.float() - reference).square().mean()
