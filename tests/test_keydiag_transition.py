# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from kilora.compute import LayerSet
from kilora.keydiag_transition import depth_transition


def layer(down, up, **kwargs):
    experts = down.shape[0]
    width = down.shape[1]
    return LayerSet(
        router=torch.eye(width),
        keys_in=torch.eye(width)[:experts, None, :],
        down=down,
        up=up,
        per_expert=1,
        tau=0.0,
        **kwargs,
    )


def test_block_diagonal_and_off_diagonal():
    x = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    s = layer(torch.eye(2), torch.eye(2))
    result = depth_transition(s, x)
    assert result["diagonal_share"] == 1.0
    assert result["self_share"] == [1.0, 1.0]
    assert result["weight_only"]["diagonal_share"] == 1.0
    s.up = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    result = depth_transition(s, x)
    assert result["matrix"][1] == pytest.approx([0.8, 0.2])
    assert result["diagonal_share"] == pytest.approx(1 / 3)
    json.dumps(result, allow_nan=False)


def test_imported_non_square_skips_but_state_is_allowed():
    down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.ones(2, 3)
    assert depth_transition(layer(down, up, form="input"), torch.ones(1, 2)) == {
        "skipped": "not square"
    }
    assert "matrix" in depth_transition(layer(down, torch.eye(2)), torch.ones(1, 2))


def test_isotropic_coefficients_match_weight_only():
    s = layer(torch.eye(2), torch.tensor([[1.0, 2.0], [3.0, 1.0]]), scale=2.0)
    x = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    result = depth_transition(s, x)
    torch.testing.assert_close(
        torch.tensor(result["matrix"]), torch.tensor(result["weight_only"]["matrix"])
    )
    assert result["diagonal_share"] == pytest.approx(result["weight_only"]["diagonal_share"])


def test_read_share_weights_and_zero_energy():
    s = layer(torch.eye(2), torch.eye(2), form="input", import_score="read_share")
    s.tau = 1.0
    result = depth_transition(s, torch.tensor([[2.0, 1.0]]))
    assert result["diagonal_share"] == 1.0
    assert result["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    s.up = torch.ones(2, 2)
    result = depth_transition(s, torch.tensor([[2.0, 1.0]]))
    assert result["matrix"][0] == pytest.approx([64 / 65, 1 / 65])
    s.up.zero_()
    result = depth_transition(s, torch.zeros(1, 2))
    assert result["matrix"] == [[0.0, 0.0], [0.0, 0.0]]
    assert result["diagonal_share"] == 0.0
    json.dumps(result, allow_nan=False)
