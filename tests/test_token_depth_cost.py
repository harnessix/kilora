# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

pytest.importorskip("triton")

from test_fused import DEVICE
from test_fused_input import _imported, _inputs
from test_fused_token_depth import _run, _same

from kilora import compute, fused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

_VIEWS = {
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::detach",
    "aten::expand",
    "aten::transpose",
    "aten::t",
    "aten::as_strided",
    "aten::alias",
    "aten::slice",
    "aten::select",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::permute",
}


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")


def _census(run):
    from torch.utils._python_dispatch import TorchDispatchMode

    launches = [0]
    size = [0]
    inside = [False]

    class Launcher:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def go(*args, **kwargs):
                launches[0] += 1
                inside[0] = True
                try:
                    return launch(*args, **kwargs)
                finally:
                    inside[0] = False

            return go

    class Bytes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if not inside[0] and func._schema.name not in _VIEWS:
                for t in out if isinstance(out, (tuple, list)) else (out,):
                    if isinstance(t, torch.Tensor):
                        size[0] += t.numel() * t.element_size()
            return out

    kernels = {
        name: kernel
        for name, kernel in vars(fused).items()
        if name.endswith("_kernel") and hasattr(kernel, "__getitem__")
    }
    for name, kernel in kernels.items():
        setattr(fused, name, Launcher(kernel))
    try:
        with Bytes():
            result = run()
    finally:
        for name, kernel in kernels.items():
            setattr(fused, name, kernel)
    return result, launches[0], size[0]


def _layer_map(kind, batch=2, tokens=37):
    t = torch.arange(tokens, dtype=torch.float32)
    if kind == "mask":
        d = 1.0 + (t * 5 // tokens)
    elif kind == "mask_soft":
        d = (1.0 + t * 4.0 / (tokens - 1)).clamp(1, 5)
    elif kind == "auto2":
        d = 1.0 + 4.0 * ((t * 0.37) % 1.0) ** 3
    else:
        d = 1.0 + 4.0 * ((t * 0.61) % 1.0) ** 0.8
    return d[None].expand(batch, tokens).contiguous().to(DEVICE)


@pytest.mark.parametrize("kind", ["mask", "mask_soft"])
def test_depth_map_costs_no_more_than_uniform_depth(kind, monkeypatch):
    from test_fused import _no_warm_up_running

    _no_warm_up_running()
    x, y0 = _inputs(31, width=96)
    s = _imported(32, width=96, gate_mode="adaptive")
    depth = _layer_map(kind)
    assert float(depth.max()) == 5.0

    def uniform():
        return _run(monkeypatch, True, x, y0, [s], 5, None)

    def varying():
        return _run(monkeypatch, True, x, y0, [s], 5, depth)

    uniform()
    varying()
    (_, used), uniform_launches, uniform_bytes = _census(uniform)
    assert used
    (_, used), varying_launches, varying_bytes = _census(varying)
    assert used
    assert varying_launches <= uniform_launches, (varying_launches, uniform_launches)
    assert varying_bytes <= uniform_bytes, (varying_bytes, uniform_bytes)


@pytest.mark.parametrize("kind", ["mask", "mask_soft", "auto2", "auto3"])
def test_each_step_writes_the_rows_of_its_tokens(kind):
    d = _layer_map(kind)[..., None]
    td = compute._TokenDepth(d, float(d.max()), d)
    plan = td.rows()
    assert td.rows() is plan
    flat = td.d.reshape(-1)
    last = torch.ceil(flat).long()
    written = torch.zeros_like(last)
    for k in range(1, td.steps + 1):
        rows, weights, is_last, computed = plan.step(k)
        ends = rows[is_last.bool()]
        assert set(ends.tolist()) == set(torch.nonzero(last == k).reshape(-1).tolist())
        lower = rows[:computed][~is_last[:computed].bool()]
        expect = torch.nonzero((last == k + 1) & (flat < k + 1)).reshape(-1)
        assert set(lower.tolist()) == set(expect.tolist())
        assert bool((flat[rows[:computed]] > k - 1).all())
        written.index_add_(0, ends, torch.ones_like(ends))
        later = torch.cat([plan.step(j)[0][plan.step(j)[2].bool()] for j in range(k, td.steps + 1)])
        assert set(later.tolist()) == set(torch.nonzero(flat > k - 1).reshape(-1).tolist())
        assert int(td.active(k).sum()) == later.numel()
    assert bool((written == (flat > 0).long()).all())


@pytest.mark.parametrize("kind", ["mask", "mask_soft", "auto2", "auto3"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("routing", ["token", "tensor"])
def test_depth_maps_with_the_adaptive_gate_match_the_token_loop(kind, carry, routing, monkeypatch):
    x, y0 = _inputs(33)
    s = _imported(34, routing=routing, gate_mode="adaptive", carry=carry)
    _same(monkeypatch, x, y0, [s], _layer_map(kind))


@pytest.mark.parametrize("kind", ["mask_soft", "auto2"])
def test_depth_maps_with_frozen_routes_match_the_token_loop(kind, monkeypatch):
    import dataclasses

    x, y0 = _inputs(35)
    s = dataclasses.replace(_imported(36, gate_mode="adaptive"), route_freeze=0.0)
    _same(monkeypatch, x, y0, [s], _layer_map(kind))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_depth_maps_in_model_types_match_the_token_loop(dtype, monkeypatch):
    x, y0 = _inputs(37)
    s = _imported(38, gate_mode="adaptive")
    _same(monkeypatch, x.to(dtype), y0.to(dtype), [s], _layer_map("auto3"))


def test_uniform_depth_is_unchanged_by_the_row_plan(monkeypatch):
    x, y0 = _inputs(39)
    s = _imported(40, gate_mode="adaptive")
    constant = torch.full((2, 37), 3.0, device=DEVICE)
    a, _ = _run(monkeypatch, True, x, y0, [s], 3, constant)
    b, _ = _run(monkeypatch, True, x, y0, [s], 3, None)
    assert torch.equal(a, b)
