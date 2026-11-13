# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from kilora import compute, keydiag_paths
from kilora.keydiag_paths import compare_paths


def record(down, up, x=None, depth=3):
    width = down.shape[1]
    n = down.shape[0]
    s = compute.LayerSet(
        router=torch.eye(width),
        keys_in=torch.eye(width)[:n, None],
        down=down,
        up=up,
        per_expert=1,
        routing_select="top1",
        gate_mode="none",
    )
    x = torch.eye(width) if x is None else x
    return {"sets": [s], "x": x, "y0": x, "depth": depth, "target": "layer"}


def test_four_matrices_direction_and_covariance(monkeypatch):
    monkeypatch.delenv("KILORA_KEY_GRAPH", raising=False)
    r = record(
        torch.eye(2),
        torch.tensor([[1.0, 2.0], [3.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )
    result = compare_paths(r, 0)
    assert result["energy"]["matrix"][0] == pytest.approx([0.2, 0.8])
    assert result["cosine"]["matrix"][0] == pytest.approx([1 / 3, 2 / 3])
    assert result["angles"]["matrix"][0] == pytest.approx([0.2, 0.8])
    assert result["weighted"]["matrix"] == [[1.0, 0.0], [1.0, 0.0]]
    assert result["energy"]["diagonal_share"] == pytest.approx(0.1)
    assert result["cosine"]["one_hop_tv"] is None
    assert result["path_equivalence"]["median"] is None
    json.dumps(result, allow_nan=False)


def test_path_equivalence_and_zero_rows(monkeypatch):
    monkeypatch.delenv("KILORA_KEY_GRAPH", raising=False)
    r = record(torch.eye(3), torch.eye(3))
    result = compare_paths(r, 0)
    assert result["path_equivalence"] == {"median": 0.0, "share_below_0_1": 1.0, "triples": 6}
    r["sets"][0].up.zero_()
    result = compare_paths(r, 0)
    assert result["energy"]["matrix"] == torch.eye(3).tolist()
    json.dumps(result, allow_nan=False)


def test_observed_one_hop_and_unchanged_output(monkeypatch):
    monkeypatch.setenv("KILORA_KEY_GRAPH", "1")
    r = record(torch.eye(2), torch.eye(2) * 0.1)
    s = r["sets"][0]
    x = r["x"].unsqueeze(0)
    with torch.no_grad():
        before = compute.run_layer(x, x, [s], 3, target="layer")
        result = compare_paths(r, 0)
        after = compute.run_layer(x, x, [s], 3, target="layer")
    assert result["energy"]["one_hop_tv"] == pytest.approx([0.0, 0.0])
    assert torch.equal(before, after)


def test_non_square_skipped():
    r = record(torch.eye(2), torch.ones(2, 3))
    assert compare_paths(r, 0) == {"skipped": "not square"}


def test_one_hop_uses_each_observed_mix(monkeypatch):
    r = record(torch.eye(2), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    weights = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]])]
    monkeypatch.setattr(keydiag_paths, "_trace", lambda record, index: weights)
    result = compare_paths(r, 0)
    assert result["energy"]["one_hop_tv"] == pytest.approx([0.0, 1.0])


def test_non_equivalent_paths(monkeypatch):
    monkeypatch.delenv("KILORA_KEY_GRAPH", raising=False)
    r = record(torch.eye(3), torch.ones(3, 3))
    result = compare_paths(r, 0)["path_equivalence"]
    assert result["median"] == pytest.approx(1 / 3**0.5 - 1 / 3)
    assert result["share_below_0_1"] == 0.0


def test_operator_matrices_match_direct_products_and_use_rank_blocks(monkeypatch):
    generator = torch.Generator().manual_seed(18)
    experts, rank, width = 3, 2, 5
    a = torch.randn(experts, rank, width, generator=generator)
    b = torch.randn(experts, rank, width, generator=generator)
    x = torch.randn(7, width, generator=generator)
    r = record(a.flatten(0, 1), b.flatten(0, 1), x)
    r["sets"][0].per_expert = rank
    original_einsum = torch.einsum

    def checked_einsum(*args, **kwargs):
        value = original_einsum(*args, **kwargs)
        assert value.ndim < 2 or value.shape[-2:] != (width, width)
        return value

    original_matmul = torch.Tensor.__matmul__

    def checked_matmul(left, right):
        value = original_matmul(left, right)
        assert value.ndim < 2 or value.shape[-2:] != (width, width)
        return value

    monkeypatch.setattr(torch, "einsum", checked_einsum)
    monkeypatch.setattr(torch.Tensor, "__matmul__", checked_matmul)
    result = compare_paths(r, 0)
    monkeypatch.undo()
    operators = b.transpose(1, 2) @ a
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    norms = torch.linalg.matrix_norm(operators)
    weighted_norms = torch.stack(
        [(w @ covariance @ w.T).trace().clamp_min(0).sqrt() for w in operators]
    )
    raw_trace = torch.empty(experts, experts)
    raw_transmission = torch.empty_like(raw_trace)
    raw_weighted_trace = torch.empty_like(raw_trace)
    raw_weighted_transmission = torch.empty_like(raw_trace)
    for e in range(experts):
        for f in range(experts):
            product = operators[f] @ operators[e]
            raw_trace[e, f] = product.trace() / (norms[e] * norms[f])
            raw_transmission[e, f] = product.norm() / (norms[e] * norms[f])
            raw_weighted_trace[e, f] = (product @ covariance).trace() / (
                weighted_norms[e] * weighted_norms[f]
            )
            raw_weighted_transmission[e, f] = (product @ covariance @ product.T).trace().clamp_min(
                0
            ).sqrt() / (weighted_norms[e] * norms[f])
    for name, matrix in {
        "trace": raw_trace.clamp_min(0),
        "transmission": raw_transmission,
        "weighted_trace": raw_weighted_trace.clamp_min(0),
        "weighted_transmission": raw_weighted_transmission,
    }.items():
        assert torch.tensor(result[name]["matrix"]) == pytest.approx(
            keydiag_paths._normalize(matrix).numpy(), abs=1e-6
        )
    assert result["trace"]["negative_share"] == pytest.approx((raw_trace < 0).float().mean().item())
    assert result["weighted_trace"]["negative_share"] == pytest.approx(
        (raw_weighted_trace < 0).float().mean().item()
    )
    for prefix in ("", "weighted_"):
        for rank in (4, 8, 16):
            assert (
                result[f"{prefix}embed_r{rank}"]["matrix"]
                == result[f"{prefix}transmission"]["matrix"]
            )
    assert result["transmission_path_equivalence"] == pytest.approx(
        keydiag_paths._equivalence(raw_transmission), abs=1e-6
    )


def test_operator_matrices_ignore_positive_expert_scale():
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(6, 5, generator=generator)
    b = torch.randn(6, 5, generator=generator)
    r = record(a, b, torch.randn(7, 5, generator=generator))
    r["sets"][0].per_expert = 2
    before = compare_paths(r, 0)
    r["sets"][0].up[2:4] *= 10
    after = compare_paths(r, 0)
    for name in (
        "trace",
        "transmission",
        "weighted_trace",
        "weighted_transmission",
        "embed_r4",
        "weighted_embed_r4",
    ):
        assert torch.tensor(after[name]["matrix"]) == pytest.approx(
            torch.tensor(before[name]["matrix"]).numpy(), abs=1e-6
        )


def test_truncated_embeddings_match_best_rank_four():
    generator = torch.Generator().manual_seed(13)
    matrix = torch.rand(6, 6, generator=generator)
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    expected = (u[:, :4] * singular[:4]) @ vh[:4]
    assert torch.allclose(
        keydiag_paths._embeddings(matrix)["embed_r4"], expected.clamp_min(0), atol=1e-6
    )
