# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
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
def _derived(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.delenv("KILORA_IMPORT_FIXED_READ", raising=False)


def _radial(gate_mode="adaptive", radial_step=0.5):
    s = dataclasses.replace(_imported(3, n_in=24, gate_mode=gate_mode), radial_step=radial_step)
    s.path_width = 3
    assert compute.depth_split_on(s) == (radial_step != 1.0)
    return s


def _region(tokens):
    return (torch.arange(tokens) % 3 != 0).float().view(1, tokens, 1).to(DEVICE)


def _spy_paths(monkeypatch):
    walks, chains = [], []
    real_walks = compute._run_derived_walks
    real_chain = compute._run_input_fused
    me = threading.get_ident()

    def spy_walks(x, y0, sets, once, depth, *a, **k):
        if threading.get_ident() == me:
            walks.append(depth)
        return real_walks(x, y0, sets, once, depth, *a, **k)

    def spy_chain(x, y0, sets, once, depth, *a, **k):
        out = real_chain(x, y0, sets, once, depth, *a, **k)
        if threading.get_ident() == me:
            chains.append((list(once), depth, a[3] if len(a) > 3 else k.get("entry"), out))
        return out

    monkeypatch.setattr(compute, "_run_derived_walks", spy_walks)
    monkeypatch.setattr(compute, "_run_input_fused", spy_chain)
    return walks, chains


def _call(monkeypatch, fused_on, x, y, s, depth, regions=None, scales=None):
    monkeypatch.setenv("KILORA_FUSED", "1" if fused_on else "0")
    if not fused_on:
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    else:
        monkeypatch.delenv("KILORA_DERIVED_BATCHED", raising=False)
    with torch.no_grad(), compute.dropout_key(7, 3):
        return run_layer(x, y, [s], depth, target=TARGET, regions=regions, expert_scales=scales)


@pytest.mark.parametrize("depth", [1, 2, 4, 7])
@pytest.mark.parametrize("radial_step", [0.5, 1.0])
@pytest.mark.parametrize("fused_on", [False, True])
def test_full_mode_with_the_split_reaches_the_walks(monkeypatch, depth, radial_step, fused_on):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial(radial_step=radial_step)
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    walks, chains = _spy_paths(monkeypatch)
    out = _call(monkeypatch, fused_on, x, y, s, depth)
    assert walks == [depth]
    assert all(entry is not None for _, _, entry, _ in chains)
    assert not torch.equal(out, y)


@pytest.mark.parametrize("depth", [1, 2, 4, 7])
@pytest.mark.parametrize("radial_step", [0.5, 1.0])
def test_keys_mode_keeps_the_split_chain(monkeypatch, depth, radial_step):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    s = _radial(radial_step=radial_step)
    x, y = _inputs(5, batch=1, tokens=5, n_in=24)
    walks, chains = _spy_paths(monkeypatch)
    out = _call(monkeypatch, True, x, y, s, depth)
    assert walks == []
    assert len(chains) == 1
    once, chain_depth, entry, chain_out = chains[0]
    assert once == [0] and chain_depth == depth and entry is None
    assert torch.equal(out, chain_out)


def test_keys_mode_is_the_split_chain_output(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "1")
    s = _radial()
    x, y = _inputs(6, batch=1, tokens=5, n_in=24)
    with torch.no_grad(), compute.dropout_key(7, 3):
        out = run_layer(x, y, [s], 4, target=TARGET)
    with torch.no_grad(), compute.dropout_key(7, 3):
        direct = compute._run_input_fused(x, y, [s], [0], 4, [None], TARGET)
    assert torch.equal(out, direct)


@pytest.mark.parametrize("depth", [2, 4, 7])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("with_region", [False, True])
def test_stacked_walks_with_the_split_match_eager_walks(monkeypatch, depth, gate_mode, with_region):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial(gate_mode)
    x, y = _inputs(8, batch=1, tokens=5, n_in=24)
    regions = [_region(5)] if with_region else None
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
    expected = _call(monkeypatch, False, x, y, s, depth, regions)
    eager_weights = list(weights)
    assert not any(w is not None for w in calls)
    calls.clear()
    weights.clear()
    before = fused.COUNTS.get("walks_fused", 0)
    actual = _call(monkeypatch, True, x, y, s, depth, regions)
    assert [w is not None for w in calls] == [True]
    assert fused.COUNTS.get("walks_fused", 0) == before + 1
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert len(weights) == len(eager_weights) == 1
    assert torch.equal(weights[0].argmax(-1), eager_weights[0].argmax(-1))
    torch.testing.assert_close(weights[0], eager_weights[0], rtol=1e-4, atol=1e-5)


def test_split_changes_the_walks(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial()
    plain = dataclasses.replace(s, radial_step=compute.DEPTH_STEP_DEFAULTS[0])
    x, y = _inputs(9, batch=1, tokens=5, n_in=24)
    split = _call(monkeypatch, True, x, y, s, 4)
    unsplit = _call(monkeypatch, True, x, y, plain, 4)
    assert not torch.equal(split, unsplit)


def test_split_without_the_depth_chain_keeps_eager_walks(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    s = _radial()
    x, y = _inputs(10, batch=1, tokens=5, n_in=24)
    walks, chains = _spy_paths(monkeypatch)
    reason = "stacked walks with a depth split without the import depth chain"
    before = fused.REASONS.get(reason, 0)
    _call(monkeypatch, True, x, y, s, 4)
    assert walks == [4]
    assert fused.REASONS.get(reason, 0) == before + 1
    assert len(chains) == s.path_width
    assert all(entry is not None for _, _, entry, _ in chains)


@pytest.mark.parametrize("fused_on", [False, True])
def test_full_mode_at_depth_one_keeps_the_walks(monkeypatch, fused_on):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial()
    x, y = _inputs(11, batch=1, tokens=5, n_in=24)
    walks, _ = _spy_paths(monkeypatch)
    actual = _call(monkeypatch, fused_on, x, y, s, 1)
    plain = dataclasses.replace(s, radial_step=1.0)
    expected = _call(monkeypatch, fused_on, x, y, plain, 1)
    assert walks == [1, 1]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("fused_on", [False, True])
def test_full_walks_with_zero_expert_scales_keep_the_base(monkeypatch, fused_on):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial()
    x, y = _inputs(12, batch=2, tokens=5, n_in=24)
    walks, _ = _spy_paths(monkeypatch)
    actual = _call(monkeypatch, fused_on, x, y, s, 4, scales=[[0, 0, 0, 0]])
    assert walks == [4]
    assert torch.equal(actual, y)


def test_full_walks_with_partial_expert_scales_match_eager(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial()
    x, y = _inputs(13, batch=2, tokens=5, n_in=24)
    scales = [[0, 0.5, 0, 2]]
    expected = _call(monkeypatch, False, x, y, s, 4, scales=scales)
    actual = _call(monkeypatch, True, x, y, s, 4, scales=scales)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, y)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fused_on", [False, True])
def test_full_walks_keep_tokens_with_all_experts_masked(monkeypatch, fused_on):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    s = _radial()
    x, y = _inputs(14, batch=2, tokens=5, n_in=24)
    mask = torch.ones(2, 5, s.down.shape[0], device=DEVICE)
    mask[:, ::2] = 0
    s = dataclasses.replace(s, expert_mask=mask)
    actual = _call(monkeypatch, fused_on, x, y, s, 4)
    assert torch.equal(actual[:, ::2], y[:, ::2])
    assert torch.isfinite(actual).all()
