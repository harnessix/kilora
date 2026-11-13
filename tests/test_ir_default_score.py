# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora.compute import read_energy
from kilora.ir import check_invariants, default_bindings, default_program
from kilora.program import ProgramError
from kilora.program_eager import interpret


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_read_share_matches_native(seed):
    generator = torch.Generator().manual_seed(seed)
    reads = torch.randn(2, 3, 4, 2, generator=generator)
    temperature = torch.tensor(1.7)
    energy = read_energy(SimpleNamespace(per_expert=2), reads.flatten(-2))
    expected = temperature * (energy / (energy.sum(-1, keepdim=True) + 1e-12))
    result = interpret(
        default_program("score", "read_share"),
        {"reads": reads, "temperature": temperature},
        default_bindings("score", "read_share"),
    )
    assert torch.equal(result["logits"], expected)
    check_invariants("score", result, sizes={"batch": 2, "tokens": 3, "experts": 4})


def test_unknown_default_refused():
    with pytest.raises(ProgramError, match="unknown default program"):
        default_program("score", "unknown")


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_router_score_matches_native(seed):
    from kilora.compute import norm_energy, orthonormal_keys
    from kilora.ir.driver import _score

    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 4, 7, generator=generator)
    raw = torch.randn(3, 2, 7, generator=generator)
    rows = torch.tensor([1, 2, 1])
    keys = orthonormal_keys(raw, rows)
    assert torch.equal(_score(query, keys, rows, 1.7), 1.7 * norm_energy(query, keys, rows))


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_router_projection_matches_native(seed):
    from kilora.compute import out_project
    from kilora.ir.driver import _project_route

    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 4, 7, generator=generator)
    keys = torch.randn(3, 2, 7, generator=generator)
    weights = torch.rand(2, 4, 3, generator=generator)
    assert torch.equal(_project_route(query, keys, weights), out_project(query, keys, weights))


@pytest.mark.parametrize("rows", [None, torch.tensor([1, 2, 1])])
def test_keys_program_matches_router_logits(rows):
    from kilora.compute import orthonormal_keys, router_logits
    from kilora.ir.driver import _score

    generator = torch.Generator().manual_seed(512)
    query = torch.randn(2, 4, 7, generator=generator)
    raw = torch.randn(3, 2, 7, generator=generator)
    actual = _score(query, orthonormal_keys(raw, rows), rows, 1.7)
    assert torch.equal(actual, router_logits(query, raw, 1.7, rows))


@pytest.mark.parametrize("per", [1, 2, 3, 4, 8, 10])
@pytest.mark.parametrize("partial", [False, True])
def test_preparation_matches_native(per, partial):
    from kilora.compute import orthonormal_keys
    from kilora.ir.defaults import interpret_key_preparation

    generator = torch.Generator().manual_seed(610 + per)
    keys = torch.randn(4, per, 7, generator=generator)
    rows = torch.tensor([1, per, 1, per]) if partial else torch.full((4,), per)
    actual = interpret_key_preparation(keys, rows if partial else None)
    assert torch.equal(actual, orthonormal_keys(keys, rows if partial else None))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.float64])
def test_preparation_row_count_dtypes(dtype):
    from kilora.compute import orthonormal_keys
    from kilora.ir.defaults import interpret_key_preparation

    keys = torch.randn(3, 4, 7, generator=torch.Generator().manual_seed(811))
    rows = torch.tensor([1.5, 2, 4], dtype=dtype)
    assert torch.equal(interpret_key_preparation(keys, rows), orthonormal_keys(keys, rows))


def test_driver_prepares_without_native_key_function(monkeypatch):
    from kilora import compute
    from kilora.ir.driver import _prepare_keys

    generator = torch.Generator().manual_seed(812)
    keys = torch.randn(3, 5, 7, generator=generator)
    rows = torch.tensor([2, 4, 5])
    expected = compute.orthonormal_keys(keys, rows)
    layer = compute.LayerSet(
        router=torch.randn(7, 6, generator=generator), keys_in=keys, rows_in=rows
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native key preparation was called")

    monkeypatch.setattr(compute, "orthonormal_keys", forbidden)
    assert torch.equal(_prepare_keys(layer)[0], expected)


def test_explicit_layer_preparation_reuses_and_invalidates_cache(monkeypatch):
    from kilora import compute
    from kilora.ir import driver

    generator = torch.Generator().manual_seed(933)
    keys = torch.randn(2, 2, 5, generator=generator)
    layer = compute.LayerSet(router=torch.randn(5, 6, generator=generator), keys_in=keys)
    original = driver.interpret_key_preparation
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "interpret_key_preparation", counted)
    before = driver.prepare_layer_ir([layer])[0][0]
    cached = driver.prepare_layer_ir([layer])[0][0]
    assert torch.equal(before, cached)
    assert len(calls) == 1
    keys.add_(0.2)
    after = driver.prepare_layer_ir([layer])[0][0]
    assert len(calls) == 2
    assert torch.equal(after, compute.orthonormal_keys(keys))
