# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from test_import_attention_plain import ATTN, MLP

from kilora import compute


def routed_set(form="input", **kwargs):
    eye = torch.eye(4)
    return compute.LayerSet(
        form=form,
        router=eye,
        keys_in=eye[:, None],
        keys_out=eye[:, None],
        down=eye * 0.1,
        up=eye * 0.2,
        per_expert=1,
        gate_mode="none",
        import_score="derived",
        **kwargs,
    )


@pytest.mark.parametrize("form", ["input", "state"])
def test_attention_routes_each_token(form, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    x = torch.eye(4)[None]
    s = routed_set(form)
    seen = []
    with (
        torch.no_grad(),
        compute.eager(),
        compute.recording_shares(lambda target, gates: seen.append(gates[0])),
    ):
        actual = compute.run_layer(x, x, [s], 1, target=ATTN)
    gates = seen[0]
    assert not torch.equal(gates[:, :1], gates[:, 1:2])
    torch.testing.assert_close(gates.sum(-1), torch.ones_like(gates[..., 0]))
    expected = x + ((x @ s.down.T) * gates) @ s.up
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("depth", [1, 3, 16])
def test_attention_import_bound_and_carried_read(depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    s = routed_set()
    x = torch.eye(4)[None]
    with torch.no_grad(), compute.eager():
        first = compute.run_layer(x, x, [s], 1, target=ATTN)
        actual = compute.run_layer(x, x, [s], depth, target=ATTN)
        expected = compute.run_layer(x, x, [s], depth, target=MLP)
    assert torch.equal(actual, expected)
    assert (actual - x).norm() <= compute.IMPORT_DEPTH_BUDGET * (first - x).norm() + 1e-6
    if depth > 1:
        assert not torch.equal(actual, first)


def test_attention_import_read_keep_scales_later_additions(monkeypatch):
    from test_import_depth import _set

    s = _set(rectangular=True)
    x = torch.ones(1, 2, 4)
    y = torch.zeros(1, 2, 3)
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    with torch.no_grad(), compute.eager():
        first = compute.run_layer(x, y, [s], 1, target=ATTN)
        monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **kw: torch.ones_like(z))
        full = compute.run_layer(x, y, [s], 2, target=ATTN) - first
        monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **kw: torch.full_like(z, 0.5))
        kept = compute.run_layer(x, y, [s], 2, target=ATTN) - first
    torch.testing.assert_close(kept, full * 0.5)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("fixed_read", ["0", "1"])
def test_attention_fold_matches_eager(depth, fixed_read, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", fixed_read)
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = replace(routed_set(), import_score="read_share")
    x = torch.eye(4)[None]
    with torch.no_grad(), compute.eager():
        monkeypatch.setenv("KILORA_IMPORT_FOLD", "0")
        loop = compute.run_layer(x, x, [s], depth, target=ATTN)
        monkeypatch.setenv("KILORA_IMPORT_FOLD", "1")
        folded = compute.run_layer(x, x, [s], depth, target=ATTN)
        ordinary = compute.run_layer(x, x, [s], depth, target=MLP)
    torch.testing.assert_close(folded, loop, rtol=1e-5, atol=1e-6)
    assert torch.equal(folded, ordinary)
