# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
from collections import Counter

import pytest
import torch

pytest.importorskip("triton")

from test_fused import DEVICE, _no_warm_up_running
from test_fused_input import TARGET, _imported, _inputs
from torch.utils._python_dispatch import TorchDispatchMode

from kilora import adaptive_gate, compute, fused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)

LAYER_BUDGET = 112
GATE_BUDGET = 6


class _Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.inside:
            self.ops[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


class _Launcher:
    def __init__(self, kernel, counter):
        self.kernel = kernel
        self.counter = counter

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.counter.inside = True
            try:
                return launch(*args, **kwargs)
            finally:
                self.counter.inside = False

        return run


def _wrap_kernels(monkeypatch, counter):
    for module in (fused, adaptive_gate):
        for name, kernel in list(vars(module).items()):
            if name.endswith("_kernel") and hasattr(kernel, "__getitem__"):
                monkeypatch.setattr(module, name, _Launcher(kernel, counter))


@pytest.fixture
def _setup(monkeypatch):
    _no_warm_up_running()
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_MASKS", "sample")
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.delenv("KILORA_GATE_BASE_REFERENCE", raising=False)


def test_layer_call_budget(_setup, monkeypatch):
    x, y0 = _inputs(31, batch=2, tokens=64)
    s = _imported(32)

    def run():
        with torch.no_grad(), compute.dropout_key(7, 3):
            return compute.run_layer(x, y0, [s], 4, target=TARGET)

    run()
    counter = _Count()
    _wrap_kernels(monkeypatch, counter)
    with counter:
        run()
    total = sum(counter.ops.values())
    print(f"layer call ops: {total} {dict(counter.ops.most_common())}")
    assert total <= LAYER_BUDGET, (total, counter.ops)


def test_gate_call_budget(_setup, monkeypatch):
    s = _imported(33)
    g = torch.Generator().manual_seed(34)
    cols, out = s.up.shape
    steps = [torch.randn(2, 64, cols, generator=g).to(DEVICE) for _ in range(2)]
    norm = (torch.rand(2, 64, generator=g) + 0.1).to(DEVICE)
    rms = (torch.rand(2, out, generator=g) + 0.1).to(DEVICE)

    def run(counter=None):
        hist = {}
        with torch.no_grad(), compute.dropout_key(7, 3):
            compute.split_gate(s, steps[0], hist, norm, rms, None, TARGET, 0, 1, None)
            if counter is None:
                return compute.split_gate(s, steps[1], hist, norm, rms, None, TARGET, 0, 2, None)
            with counter:
                return compute.split_gate(s, steps[1], hist, norm, rms, None, TARGET, 0, 2, None)

    run()
    counter = _Count()
    _wrap_kernels(monkeypatch, counter)
    run(counter)
    total = sum(counter.ops.values())
    print(f"gate call ops: {total} {dict(counter.ops.most_common())}")
    assert total <= GATE_BUDGET, (total, counter.ops)
