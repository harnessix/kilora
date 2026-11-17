# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute
from kilora.adapter import imported_settings


def chain():
    return dataclasses.replace(
        planted(),
        down=torch.eye(3),
        up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        per_expert=1,
        import_score="derived",
    )


def test_diffusion_path_and_projector_lift(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0.5")
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = chain()
    rout, rin = compute.derived_diffusion_matrices(s)
    assert rout.shape == rin.shape == (3, 3)
    assert compute.derived_diffusion_matrices(s)[0] is rout
    raw = compute.share_transition(s, _raw=True)
    diffused = rout @ raw @ rin.T
    assert diffused[0, 2] > raw[0, 2]
    expected = diffused.clamp_min(0)
    expected /= expected.sum(-1, keepdim=True).clamp_min(1e-30)
    torch.testing.assert_close(compute.share_transition(s), expected)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    kin, rows, *_ = compute.derived_keys(s)
    projectors = torch.stack([k.T @ k for k in kin])
    mixed = torch.einsum("ef,fij->eij", rin, projectors)
    shares = torch.einsum("bi,eij,bj->be", x, mixed, x)
    shares /= x.square().sum(-1, keepdim=True) + compute.SCORE_EPS
    rank = rows.float()
    width = x.shape[-1]
    variance = 2 * rank * (width - rank) / (width * width * (width + 2))
    expected_lifts = (shares - (rin @ (rank / width))) / variance.sqrt()
    torch.testing.assert_close(compute.derived_logits(s, x), s.tau * expected_lifts)
    assert rows.tolist() == [1, 1, 1]


def test_zero_is_bit_identical_and_defaults(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    s, x = chain(), torch.tensor([[1.0, 2.0, 3.0]])
    baseline = compute.derived_logits(s, x)
    transition = compute.share_transition(s)
    assert imported_settings().depth == 4
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    assert torch.equal(baseline, compute.derived_logits(s, x))
    assert torch.equal(transition, compute.share_transition(s))
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "1")
    assert compute.derived_diffusion() == 0.5
    assert imported_settings().depth == 2
    assert imported_settings(depth=7).depth == 7
    assert compute.share_transition_gamma(s) == 0.5
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.7")
    assert compute.share_transition_gamma(s) == 0.7
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    assert imported_settings().depth == 3


@pytest.mark.parametrize("value", ["-0.1", "1.2", "nan"])
def test_invalid_strength(monkeypatch, value):
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", value)
    with pytest.raises(ValueError):
        compute.derived_diffusion()


def test_resolvents_are_expert_sized(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0.3")
    s = dataclasses.replace(chain(), down=torch.randn(3, 13), up=torch.randn(3, 13))
    solve = torch.linalg.solve
    shapes = []

    def checked(a, b):
        shapes.append((a.shape, b.shape))
        return solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", checked)
    compute.derived_diffusion_matrices(s)
    assert shapes == [(torch.Size([3, 3]), torch.Size([3, 3]))] * 2


def test_diffused_plain_and_lowrank_agree(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0.5")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    with torch.no_grad(), compute.dropout_key(7, 3):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        plain = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, torch.zeros_like(x), [s], 3)
    torch.testing.assert_close(fast, plain)


def test_diffused_fused_matches_eager(monkeypatch):
    pytest.importorskip("triton")
    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu":
        pytest.skip("requires a GPU")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0.5")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = _imported(3, n_in=24, routing="token", gate_mode="none")
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    expected, used = _run(monkeypatch, False, x, y, [s], 2)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], 2)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
