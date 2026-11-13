# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.library.basis import LibraryBasis, MountBasis
from kilora.library.nodes import projected_core


def test_projection_uses_all_file_directions():
    u = torch.eye(3)[:, :2]
    v = torch.eye(4)[:, :2]
    left = torch.tensor([[1.0], [2.0], [0.0]])
    right = torch.tensor([[3.0], [4.0], [0.0], [0.0]])
    mount = MountBasis(u, v, torch.ones(2), torch.ones(2), [], [])
    basis = LibraryBasis(
        {"layer": mount},
        {},
        [],
        [],
        2,
        1,
        _parts={"a": {"layer": (left, right)}, "b": {"layer": (left, right)}},
    )
    torch.testing.assert_close(projected_core(basis, "layer"), 2 * u.T @ left @ right.T @ v)


def test_missing_directions_rejected():
    mount = MountBasis(torch.eye(2), torch.eye(2), torch.ones(2), torch.ones(2), [], [])
    basis = LibraryBasis({"layer": mount}, {}, [], [], 2, 1)
    with pytest.raises(ValueError, match="no stored library directions"):
        projected_core(basis, "layer")
