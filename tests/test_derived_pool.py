# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from test_share_transition import planted

from kilora import compute, region
from kilora.family_spec import TokenLayout


def test_default_scores_are_bit_identical(monkeypatch):
    s = planted()
    x = torch.randn(2, 16, 2)
    monkeypatch.delenv("KILORA_DERIVED_POOL", raising=False)
    with region.latent_size((4, 4)):
        assert torch.equal(compute.derived_logits(s, x), compute.pooled_derived_logits(s, x, "a"))


def test_constant_and_planted_region(monkeypatch):
    scores = torch.ones(1, 25, 2)
    monkeypatch.setattr(compute, "derived_logits", lambda s, x: scores)
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0,1,0")
    with region.latent_size((5, 5)):
        assert torch.equal(compute.pooled_derived_logits(None, scores, "a"), scores)
        scores.zero_()
        scores[..., 1] = 0.1
        scores[0, 12, 0] = 9
        pooled = compute.pooled_derived_logits(None, scores, "a")[0].argmax(-1).reshape(5, 5)
        assert torch.equal(pooled[1:4, 1:4], torch.zeros(3, 3, dtype=torch.long))
        assert pooled.sum() == 16


def test_joint_layout_preserves_nonimage_and_batches(monkeypatch):
    scores = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    monkeypatch.setattr(compute, "derived_logits", lambda s, x: scores)
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0,0,1")
    layout = TokenLayout("image2d", (1, 1), text="before")
    with region.latent_size((2, 2)), region.token_layouts(lambda _: layout, text_tokens=2):
        result = compute.pooled_derived_logits(None, scores, "a")
    assert torch.equal(result[:, :2], scores[:, :2])
    torch.testing.assert_close(result[:, 2:], scores[:, 2:].mean(1, keepdim=True).expand(-1, 4, -1))


def test_text_and_unknown_grids_are_unchanged(monkeypatch):
    scores = torch.randn(1, 4, 2)
    monkeypatch.setattr(compute, "derived_logits", lambda s, x: scores)
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0,1,0")
    with region.latent_size((2, 2)), region.token_layouts(lambda _: TokenLayout("text1d")):
        assert torch.equal(compute.pooled_derived_logits(None, scores, "a"), scores)
    assert torch.equal(compute.pooled_derived_logits(None, scores, "a"), scores)


def test_fused_matches_eager_with_pooling(monkeypatch):
    pytest.importorskip("triton")
    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0.5,0.3,0.2")
    s = _imported(3, n_in=24, routing="token", gate_mode="none")
    x, y = _inputs(4, batch=2, tokens=16, n_in=24)
    with region.latent_size((4, 4)):
        expected, used = _run(monkeypatch, False, x, y, [s], 2)
        assert not used
        actual, used = _run(monkeypatch, True, x, y, [s], 2)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_pooling_matches_plain_and_lowrank(monkeypatch):
    s = planted()
    x = torch.randn(2, 16, 2)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_POOL", "0.5,0.3,0.2")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.4")
    with torch.no_grad(), region.latent_size((4, 4)), compute.dropout_key(7, 3):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        plain = compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")
    torch.testing.assert_close(fast, plain)


@pytest.mark.parametrize("value", ["1,2", "1,nan,0", "bad"])
def test_invalid_weights(monkeypatch, value):
    monkeypatch.setenv("KILORA_DERIVED_POOL", value)
    with pytest.raises(ValueError, match="KILORA_DERIVED_POOL"):
        compute.pooled_derived_logits(planted(), torch.ones(1, 4, 2), "a")
