# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute
from kilora.path_rotation import rotate_pair


def _projector(rows):
    return rows.T @ rows


def test_pair_preserves_composite_and_changes_feature_keys(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "2")
    gen = torch.Generator().manual_seed(31)
    a = torch.randn(2, 5, generator=gen)
    b = torch.randn(5, 2, generator=gen)
    f = torch.randn(3, 5, generator=gen)
    layer = dataclasses.replace(planted(), down=a, up=b.T, per_expert=2)
    original = compute.derived_keys(layer)
    rank_q, _ = torch.linalg.qr(torch.randn(2, 2, generator=gen))
    rank_layer = dataclasses.replace(layer, down=rank_q @ a, up=rank_q @ b.T)
    rank_keys = compute.derived_keys(rank_layer)
    torch.testing.assert_close(_projector(original[2][0]), _projector(rank_keys[2][0]))
    torch.testing.assert_close(_projector(original[0][0]), _projector(rank_keys[0][0]))
    with compute.dropout_key(42, 3):
        rotated_b, rotated_f = rotate_pair(b, f, "hook", 1)
        repeat_b, repeat_f = rotate_pair(b, f, "hook", 1)
    assert torch.equal(rotated_b, repeat_b) and torch.equal(rotated_f, repeat_f)
    torch.testing.assert_close(rotated_f @ rotated_b @ a, f @ b @ a, atol=1e-6, rtol=1e-5)
    changed = compute.derived_keys(dataclasses.replace(layer, up=rotated_b.T))
    assert not torch.allclose(_projector(original[2][0]), _projector(changed[2][0]))


def test_versioned_keys_freeze_entire_write(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    first, second = planted(), planted()
    cache = compute.DerivedKeyVersions()
    with cache.write_window(0, [first, second]):
        keys = compute.derived_keys(first)
        assert len(keys) == 5
        assert torch.equal(keys[4], compute._derived_keys_current(first)[4])
        first.down.data[0] = torch.tensor([0.0, 2.0])
        second.down[0] = torch.tensor([0.0, 2.0])
        assert compute.derived_keys(first) is keys
        torch.testing.assert_close(
            compute.derived_keys(second)[0][0, 0].abs(), torch.tensor([1.0, 0.0])
        )
        with pytest.raises(ValueError, match="not registered"):
            compute.derived_keys(planted())
    with cache.write_window(0, [first]):
        assert compute.derived_keys(first) is keys
    with cache.write_window(1, [first]):
        updated = compute.derived_keys(first)
        assert updated is not keys
        torch.testing.assert_close(updated[0][0, 0].abs(), torch.tensor([0.0, 1.0]))
