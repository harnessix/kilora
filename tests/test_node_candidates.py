# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.library.nodes import node_candidates


def test_candidates_match_top_dot_products():
    x = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    indices = node_candidates(x, keys, 2)
    assert indices.tolist() == [[1, 0], [2, 1]]


def test_all_candidates_and_invalid_arguments():
    x = torch.tensor([[1.0, 2.0]])
    keys = torch.eye(2)
    assert sorted(node_candidates(x, keys, 16)[0].tolist()) == [0, 1]
    with pytest.raises(ValueError):
        node_candidates(x, keys, 0)
    with pytest.raises(ValueError):
        node_candidates(x, torch.empty(0, 2), 1)
