# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute, keydiag_conflict


def sample():
    return dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        up=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        per_expert=1,
        keys_in=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]),
        import_score="derived",
    )


def test_overlaps_match_dense_and_cache_invalidates():
    s = sample()
    result = keydiag_conflict.conflict(s)
    assert result["write_write"][0][1] == pytest.approx(-1)
    assert result["write_read"][0][1] == pytest.approx(-1)
    assert result["most_negative_pair"] == [0, 1]
    first = compute.derived_conflict(s)
    assert compute.derived_conflict(s) is first
    s.up[1].neg_()
    assert compute.derived_conflict(s)[0, 1] == 0
    torch.manual_seed(5)
    s = dataclasses.replace(s, down=torch.randn(6, 5), up=torch.randn(6, 5), per_expert=2)
    ww, wr = keydiag_conflict.overlaps(s)
    w = [b.T @ a for a, b in zip(s.down.reshape(3, 2, 5), s.up.reshape(3, 2, 5), strict=True)]
    for e in range(3):
        for f in range(3):
            norm = torch.linalg.norm(w[e]) * torch.linalg.norm(w[f])
            torch.testing.assert_close(ww[e, f], (w[e] * w[f]).sum() / norm)
            torch.testing.assert_close(wr[e, f], torch.trace(w[f] @ w[e]) / norm)


def test_entry_excludes_opposite_pair_and_zero_unchanged(monkeypatch):
    s = sample()
    x = torch.tensor([[[1.0, 0.1]]]).expand(1, 32, 2)

    def draw():
        with compute.dropout_key(3, 1):
            return torch.stack(compute.derived_entries(s, x, "layer", 2), dim=-2)

    monkeypatch.delenv("KILORA_DERIVED_EXCLUSION", raising=False)
    baseline = draw()
    monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "0")
    assert torch.equal(baseline, draw())
    monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "100")
    selected = draw().sum(-2)
    assert not ((selected[..., 0] > 0) & (selected[..., 1] > 0)).any()


def test_prior_and_lowrank_agree(monkeypatch):
    s = sample()
    x = torch.tensor([[[1.0, 0.2], [0.4, 1.0]]])
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "2")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    with torch.no_grad():
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        plain = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, torch.zeros_like(x), [s], 3)
    torch.testing.assert_close(fast, plain)
    logits = compute.derived_logits(s, x)
    prior = [torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])]
    gate = compute._import_gates([s], [x], [None], x, prev=prior)[0]
    assert gate[0, 0, 1] < compute.sparsemax(logits)[0, 0, 1]
    monkeypatch.delenv("KILORA_DERIVED_EXCLUSION")
    with torch.no_grad():
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "0")
        assert torch.equal(baseline, compute.run_layer(x, torch.zeros_like(x), [s], 3))


def test_overlaps_keep_intermediates_in_rank_space(monkeypatch):
    s = dataclasses.replace(sample(), down=torch.randn(6, 7), up=torch.randn(6, 7), per_expert=2)
    original = torch.einsum
    shapes = []

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "einsum", checked)
    keydiag_conflict.overlaps(s)
    assert (7, 7) not in shapes
    assert all(max(shape, default=0) <= 3 for shape in shapes)
    nonsquare = dataclasses.replace(s, up=torch.randn(6, 5))
    assert keydiag_conflict.conflict(nonsquare)["write_read"] is None
