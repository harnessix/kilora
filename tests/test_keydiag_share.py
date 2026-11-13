# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from kilora.compute import LayerSet
from kilora.keydiag_share import share_bias


def layer(blocks):
    blocks = torch.as_tensor(blocks, dtype=torch.float32)
    count, rows, width = blocks.shape
    return LayerSet(
        router=torch.eye(width),
        keys_in=torch.zeros(count, 1, width),
        down=blocks.reshape(count * rows, width),
        per_expert=rows,
    )


def test_rank_bias_and_padding():
    blocks = torch.zeros(2, 32, 1)
    blocks[0, :4] = 1
    blocks[1] = 1
    result = share_bias(layer(blocks), torch.tensor([[1.0], [-2.0]]))
    low, high = result["experts"]
    assert [low["effective_rank"], high["effective_rank"]] == [4, 32]
    assert low["mean_share"] == pytest.approx(1 / 9)
    assert high["mean_share"] == pytest.approx(8 / 9)
    assert low["normalized_mean_share"] == pytest.approx(0.5)
    assert high["normalized_mean_share"] == pytest.approx(0.5)
    assert low["stable_rank"] == pytest.approx(1)
    assert result["largest_share_is_largest_norm"]


def test_neutral_seed_and_global_random_state():
    s = layer([[[1, 0]], [[0, 1]]])
    x = torch.tensor([[0.0, 0], [1, 3], [2, 4], [4, 1]])
    torch.manual_seed(10)
    before = torch.random.get_rng_state()
    first = share_bias(s, x, neutral_seed=17)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first == share_bias(s, x, neutral_seed=17)
    assert first["experts"] != share_bias(s, x, neutral_seed=18)["experts"]


def test_correlations_with_ties_and_zero_expert():
    s = layer([[[1], [0]], [[2], [0]], [[3], [3]], [[0], [0]]])
    result = share_bias(s, torch.ones(2, 1))
    assert result["share_vs_frobenius_norm"]["spearman"] == pytest.approx(1)
    assert result["share_vs_effective_rank"]["spearman"] == pytest.approx(0.948683298)
    assert result["share_vs_effective_rank"]["pearson"] == pytest.approx(0.8809358116)
    assert result["experts"][3]["stable_rank"] == 0
    assert result["experts"][3]["normalized_mean_share"] == 0
    json.dumps(result, allow_nan=False)


def test_undefined_correlations_and_single_token():
    result = share_bias(layer([[[1]], [[1]]]), torch.tensor([[2.0]]))
    assert result["share_vs_effective_rank"] == {"pearson": None, "spearman": None}
    assert result["share_vs_frobenius_norm"] == {"pearson": None, "spearman": None}
    json.dumps(result, allow_nan=False)
