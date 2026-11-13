# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora.keydiag_landing import landing, pairs


def _set(down, up, rank):
    return SimpleNamespace(down=down, up=up, per_expert=rank)


def test_planted_own_landing_and_dependent_rows():
    a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    b = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    result = landing(_set(a, b, 2), _set(a, None, 2), torch.randn(100, 4))
    for key in ("activation", "weight_only"):
        assert result[key]["own"] == pytest.approx([0.5, 0.5])
        assert result[key]["best_other"] == pytest.approx([0.0, 0.0])


def test_identical_subspaces_have_unit_weight_overlap():
    a = torch.eye(4)
    result = landing(_set(a, a, 2), _set(a, None, 2), torch.randn(50, 4))
    assert result["weight_only"]["own"] == pytest.approx([1.0, 1.0])
    assert result["weight_only"]["best_other"] == pytest.approx([0.0, 0.0])
    assert result["activation"]["own"] == pytest.approx([0.5, 0.5])


def test_random_overlap_matches_rank_over_width():
    gen = torch.Generator().manual_seed(123)
    width, rank, experts = 128, 8, 12
    a = torch.randn(experts * rank, width, generator=gen)
    b = torch.randn(experts * rank, width, generator=gen)
    x = torch.randn(100, width, generator=gen)
    result = landing(_set(a, b, rank), _set(a, None, rank), x)
    values = torch.tensor(result["weight_only"]["matrix"])
    assert values.mean().item() == pytest.approx(rank / width, abs=0.004)


def test_zero_vectors_are_json_safe():
    import json

    a = torch.zeros(2, 4)
    result = landing(_set(a, a, 1), _set(a, None, 1), torch.zeros(3, 4))
    assert result["activation"]["matrix"] == [[0.0, 0.0], [0.0, 0.0]]
    assert result["weight_only"]["matrix"] == [[0.0, 0.0], [0.0, 0.0]]
    json.dumps(result, allow_nan=False)


def test_pairs_anima_names():
    names = [
        "diffusion_model.blocks.0.self_attn.output_proj",
        "diffusion_model.blocks.0.cross_attn.q_proj",
        "diffusion_model.blocks.0.cross_attn.k_proj",
        "diffusion_model.blocks.0.cross_attn.output_proj",
        "diffusion_model.blocks.0.mlp.layer1",
        "diffusion_model.blocks.0.mlp.layer2",
        "diffusion_model.blocks.1.self_attn.q_proj",
        "diffusion_model.blocks.1.self_attn.k_proj",
        "diffusion_model.blocks.1.self_attn.v_proj",
        "transformer_blocks.3.attn1.to_out.0",
        "transformer_blocks.3.attn2.to_q",
        "transformer_blocks.3.attn2.to_out.0",
        "transformer_blocks.3.ff.net.0.proj",
        "transformer_blocks.3.ff.net.2",
        "transformer_blocks.4.attn1.to_v",
    ]
    expected = [
        (names[0], names[1]),
        (names[3], names[4]),
        (names[5], names[6]),
        (names[5], names[7]),
        (names[5], names[8]),
        (names[9], names[10]),
        (names[11], names[12]),
        (names[13], names[14]),
    ]
    assert pairs(names + names) == sorted(expected)
    assert pairs([]) == []
