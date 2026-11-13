# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import logging
import re
import time

import pytest
import torch

from kilora import compute
from kilora.compute import LayerSet, run_layer


def direction(count=24, fine=False, candidates=16):
    torch.manual_seed(32)
    width = 8
    keys = torch.nn.functional.normalize(torch.randn(count, 1, width), dim=-1)
    return LayerSet(
        router=torch.eye(width),
        keys_in=keys,
        keys_out=keys.clone(),
        rows_in=torch.ones(count, dtype=torch.long),
        down=torch.randn(count, width),
        up=torch.randn(count, width) * 0.01,
        direction_core=torch.randn(count, width) * 0.1,
        form="input",
        fine=fine,
        direction_candidates=candidates,
        gate_mode="none",
    )


def test_depth_limits_and_disabled_identity():
    x = torch.randn(1, 3, 8)
    y = torch.randn_like(x)
    layer = direction()
    with pytest.raises(ValueError, match="limit of 16"):
        run_layer(x, y, [layer], 17)
    plain = dataclasses.replace(layer, direction_core=None, fine=True)
    assert torch.isfinite(run_layer(x, y, [plain], 17)).all()
    assert torch.equal(
        run_layer(x, y, [layer], 4),
        run_layer(x, y, [dataclasses.replace(layer, direction_candidates=1)], 4),
    )
    with pytest.raises(ValueError, match="limit of 64"):
        run_layer(x, y, [dataclasses.replace(layer, fine=True)], 65)


def test_fine_depth_and_logging(caplog):
    x = torch.randn(1, 3, 8)
    with torch.no_grad(), caplog.at_level(logging.INFO, logger="kilora.compute"):
        output = run_layer(x, torch.zeros_like(x), [direction(fine=True)], 64)
    assert torch.isfinite(output).all()
    match = re.search(r"last-step mean keep probability ([0-9.]+)", caplog.text)
    assert match and float(match.group(1)) < 1


@pytest.mark.parametrize("fast", [True, False])
def test_all_candidates_equal_exact(monkeypatch, fast):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    x = torch.randn(1, 3, 8)
    y = torch.randn_like(x)
    layer = direction(count=24)
    with torch.no_grad():
        exact = run_layer(x, y, [layer], 5)
        actual = run_layer(
            x, y, [dataclasses.replace(layer, fine=True, direction_candidates=24)], 5
        )
    assert torch.equal(actual, exact)


def test_candidate_exact_scoring_is_bounded(monkeypatch):
    seen = []
    original = torch.einsum

    def capture(equation, *args):
        if equation == "...r,...kmr->...km":
            seen.append(args[1].shape[-3])
        return original(equation, *args)

    monkeypatch.setattr(torch, "einsum", capture)
    x = torch.randn(1, 2, 8)
    with torch.no_grad():
        run_layer(x, torch.zeros_like(x), [direction(256, True, 16)], 3)
    assert seen and set(seen) == {16}


def test_depth_step_benchmark():
    torch.set_num_threads(1)
    x = torch.randn(1, 4, 8)
    layer = direction(256, True)
    with torch.no_grad():
        start = time.perf_counter()
        output = run_layer(x, torch.zeros_like(x), [layer], 64)
        seconds = (time.perf_counter() - start) / 64
    print(f"256 nodes: {seconds * 1000:.3f} ms/depth step")
    assert torch.isfinite(output).all()


def test_candidate_routing_matches_selected_exact_scores():
    layer = direction(count=24, fine=True, candidates=5)
    q = torch.randn(1, 3, 8)
    keys = compute._set_keys(layer)
    actual = compute._route([q], [layer], [keys], None, False, 0.0)[0]
    from kilora.library.nodes import node_candidates

    indices = node_candidates(q, keys[0][:, 0, :], 5)
    exact = compute.norm_energy(q, keys[0], layer.rows_in).gather(-1, indices)
    expected = torch.zeros_like(actual).scatter(-1, indices, compute.sparsemax(exact))
    torch.testing.assert_close(actual, expected)


def test_fine_fast_and_eager_agree(monkeypatch):
    x = torch.randn(1, 3, 8)
    y = torch.randn_like(x)
    layer = direction(fine=True)
    with torch.no_grad():
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = run_layer(x, y, [layer], 8)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        eager = run_layer(x, y, [layer], 8)
    torch.testing.assert_close(fast, eager)


def test_fine_survives_adapter_baking():
    from kilora.bake import ScopedSet, plan_targets
    from kilora.library.basis import LibraryBasis, MountBasis
    from kilora.library.nodes import direction_adapters

    u = torch.eye(3)
    v = torch.eye(4)[:, :3]
    basis = LibraryBasis(
        {"layer": MountBasis(u, v, torch.ones(3), torch.ones(3), [], [])},
        {},
        [],
        [],
        3,
        1,
        _parts={"file": {"layer": (u, v)}},
    )
    adapter = direction_adapters(basis, {"layer": (4, 3)}, fine=True, candidates=2)[0]
    plan = plan_targets([ScopedSet(adapter, scope="independent")], 64)["layer"]
    layer = plan.to(torch.device("cpu"), torch.float32).sets[0]
    assert layer.fine and layer.direction_candidates == 2
    with torch.no_grad():
        result = run_layer(torch.randn(1, 2, 4), torch.zeros(1, 2, 3), [layer], 64)
    assert torch.isfinite(result).all()


def test_single_step_logs_keep_rate(caplog):
    x = torch.randn(1, 2, 8)
    with torch.no_grad(), caplog.at_level(logging.INFO, logger="kilora.compute"):
        run_layer(x, torch.zeros_like(x), [direction(fine=True)], 1)
    assert "last-step mean keep probability" in caplog.text
