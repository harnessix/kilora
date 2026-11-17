# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os

import pytest
import torch

pytest.importorskip("triton")

from test_fused_input import DEVICE, _imported, _inputs, _run, _same

from kilora import compute, fused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU or the Triton interpreter",
)


@pytest.fixture(autouse=True)
def _cpu_opt_in(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.delenv("KILORA_IMPORT_FIXED_READ", raising=False)


def _radial(seed, **kw):
    s = dataclasses.replace(_imported(seed, **kw), radial_step=0.5)
    assert compute.depth_split_on(s)
    return s


def _region(y0):
    return torch.linspace(0.2, 1, y0.shape[-2], device=DEVICE).expand(y0.shape[0], -1)


@pytest.mark.parametrize("depth", [1, 3, 2.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
@pytest.mark.parametrize("with_region", [False, True])
def test_radial_default_takes_the_fused_path(depth, gate, with_region, monkeypatch):
    x, y0 = _inputs(31)
    s = _radial(32, gate_mode=gate)
    regions = [_region(y0)] if with_region else None
    before = fused.COUNTS["import_fused"]
    eager, _ = _same(monkeypatch, x, y0, [s], depth, regions=regions)
    assert fused.COUNTS["import_fused"] == before + 1
    assert not torch.equal(eager, y0)


@pytest.mark.parametrize("steps", [(0.5, 1.0, 0.0), (0.0, 1.7, 1.3), (1.4, 0.6, 3.0)])
def test_other_split_settings_take_the_fused_path(steps, monkeypatch):
    x, y0 = _inputs(33)
    radial, orthogonal, budget = steps
    s = dataclasses.replace(
        _imported(34), radial_step=radial, orthogonal_step=orthogonal, radial_budget=budget
    )
    _same(monkeypatch, x, y0, [s], 3)


def test_fixed_reads_with_a_split_stay_eager(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    x, y0 = _inputs(35)
    s = _radial(36)
    reason = "a depth split without the import depth chain"
    before = fused.REASONS.get(reason, 0)
    before_eager = fused.COUNTS["import_eager"]
    eager, _ = _run(monkeypatch, False, x, y0, [s], 3)
    got, used = _run(monkeypatch, True, x, y0, [s], 3)
    assert not used
    assert fused.REASONS.get(reason, 0) == before + 1
    assert fused.COUNTS["import_eager"] == before_eager + 1
    assert torch.equal(eager, got)


def test_split_reason_names_the_uncovered_settings(monkeypatch):
    s = _radial(37)
    assert fused.import_split_reason([s], 3) is None
    assert fused.import_split_reason([s], 1) is None
    plain = _imported(37)
    assert fused.import_split_reason([plain], 3) is None
    td = object()
    assert fused.import_split_reason([s], 1, td) == (
        "a depth split with a per-token depth at depth 1 or less"
    )
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    assert fused.import_split_reason([s], 3) == "a depth split without the import depth chain"


@pytest.mark.parametrize("depth", [1, 3, 2.5])
@pytest.mark.parametrize("gate", ["adaptive", "none"])
def test_sets_without_a_split_keep_their_gate(depth, gate, monkeypatch):
    x, y0 = _inputs(38)
    s = dataclasses.replace(_imported(39, gate_mode=gate), radial_step=1.0)
    assert not compute.depth_split_on(s)
    with torch.no_grad():
        assert compute._import_fast_ok(y0, [s], depth) == compute._fast_path_ok(
            y0, [s], 0.0, None, None
        )
    before = fused.COUNTS["import_fused"]
    _same(monkeypatch, x, y0, [s], depth)
    assert fused.COUNTS["import_fused"] == before + 1


def test_replayed_chain_with_the_radial_default_is_fused(monkeypatch):
    x, y0 = _inputs(40)
    s = _radial(41)
    monkeypatch.setenv("KILORA_FUSED", "1")
    calls = []
    real = fused.run_fused

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fused, "run_fused", spy)
    with torch.no_grad(), compute.dropout_key(7, 3):
        compute._replay_import_chain(x, y0, [s], [0], 3, [None], "blk.0.attn1", None)
    assert calls
