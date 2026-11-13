# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora import compute
from kilora.compute import run_layer
from kilora.library.basis import LibraryBasis, MountBasis
from kilora.library.nodes import direction_adapters, direction_set, projected_core


def basis_for_mount():
    u = torch.eye(3)[:, :2]
    v = torch.eye(4)[:, :2]
    left = u[:, :1] * 2
    right = v[:, 1:2]
    mount = MountBasis(u, v, torch.ones(2), torch.ones(2), [], [])
    return LibraryBasis(
        {"layer": mount},
        {},
        [],
        [],
        2,
        1,
        _parts={"file": {"layer": (left, right)}},
    )


def test_direction_set_and_projection_accept_model_target_names():
    basis = basis_for_mount()
    stored = "blocks.0.cross_attn.q_proj"
    target = "transformer_blocks.0.attn2.to_q"
    basis.mounts[stored] = basis.mounts.pop("layer")
    basis._parts["file"][stored] = basis._parts["file"].pop("layer")
    expected = direction_set(basis, stored)
    actual = direction_set(basis, target)
    for field in ("router", "keys_in", "keys_out", "down", "up", "direction_core"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    assert actual.key == target
    torch.testing.assert_close(projected_core(basis, target), projected_core(basis, stored))


def test_direction_routes_non_square_and_writes_coupled_direction():
    layer = direction_set(basis_for_mount(), "layer")
    layer.routing_select = "top1"
    x = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    y = torch.zeros(1, 1, 3)
    torch.testing.assert_close(run_layer(x, y, [layer], 1), torch.tensor([[[2.0, 0.0, 0.0]]]))


def test_shared_projection_once_across_depth(monkeypatch):
    layer = direction_set(basis_for_mount(), "layer")
    layer.routing_select = "top1"
    original = compute.F.linear
    reads = []

    def count(input, weight, *args, **kwargs):
        if weight.shape == (2, 4):
            reads.append(weight)
        return original(input, weight, *args, **kwargs)

    monkeypatch.setattr(compute.F, "linear", count)
    x = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    with torch.no_grad():
        result = run_layer(x, torch.zeros(1, 1, 3), [layer], 3)
    assert len(reads) == 1
    assert result.shape == (1, 1, 3)


def test_adapter_retains_shared_coupling_and_base_unchanged():
    adapter = direction_adapters(basis_for_mount(), {"layer": (4, 3)})[0]
    assert adapter.extra["direction_core"].shape == (2, 2)
    assert adapter.key_source == "directions"
    x = torch.randn(1, 2, 4)
    y = torch.randn(1, 2, 3)
    assert torch.equal(run_layer(x, y, [], 2), y)


def test_direction_chunks_pad_last_expert():
    layer = direction_set(basis_for_mount(), "layer", chunk=3)
    assert layer.down.shape == (3, 4)
    assert layer.rows_in.tolist() == [2]
    assert torch.count_nonzero(layer.down[-1]) == 0


def test_baked_plan_keeps_direction_reads_and_does_not_run_svd(monkeypatch):
    from kilora.bake import ScopedSet, plan_targets

    adapter = direction_adapters(basis_for_mount(), {"layer": (4, 3)})[0]
    plan = plan_targets([ScopedSet(adapter, scope="independent")], 1)["layer"]
    plan = plan.to(torch.device("cpu"), torch.float32)
    layer = plan.sets[0]
    layer.routing_select = "top1"

    def no_svd(*args, **kwargs):
        raise AssertionError("direction routing must not compute an SVD")

    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    x = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    with torch.no_grad():
        result = run_layer(x, torch.zeros(1, 1, 3), [layer], 1)
    torch.testing.assert_close(result, torch.tensor([[[2.0, 0.0, 0.0]]]))
