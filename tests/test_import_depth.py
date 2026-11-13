# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch

from kilora import compute


def _set(form="input", rectangular=False):
    width = 3 if rectangular else 4
    return compute.LayerSet(
        router=torch.eye(width),
        keys_in=torch.ones(1, 1, width),
        down=torch.ones(1, 4) * 0.1,
        up=torch.ones(1, width) * 0.1,
        per_expert=1,
        form=form,
        carry="drop",
        gate_mode="none",
        import_score="read_share",
    )


def _run(st, depth, *, dispatch=False, dtype=torch.float32):
    x = torch.ones(1, 2, 4, dtype=dtype)
    y = torch.zeros(1, 2, st.up.shape[-1], dtype=dtype)
    with torch.no_grad():
        if dispatch:
            return compute.run_layer(x, y, [st], depth, target="import-depth")
        return compute._run_input(x, y, [st], [0], depth, [None], "import-depth")


@pytest.fixture(autouse=True)
def _comparison_environment(monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_FIXED_READ", raising=False)
    monkeypatch.setenv("KILORA_MASKS", "expect")


@pytest.mark.parametrize("depth", [3, 4, 16])
def test_fixed_read_has_the_previous_linear_growth(monkeypatch, depth):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    st = _set()
    torch.testing.assert_close(_run(st, depth), depth * _run(st, 1))


@pytest.mark.parametrize("depth", [3, 4, 16])
@pytest.mark.parametrize("rectangular", [False, True])
def test_total_change_is_bounded(depth, rectangular):
    st = _set(rectangular=rectangular)
    first = _run(st, 1).norm()
    assert _run(st, depth).norm() <= 2 * first + 1e-6


def test_last_addition_converges():
    st = _set()
    assert (_run(st, 16) - _run(st, 15)).norm() < 0.1 * _run(st, 1).norm()


def test_step_two_reads_the_carried_state(monkeypatch):
    reads = []
    original = compute._import_depth_coefs

    def record(*args):
        result = original(*args)
        reads.append((args[1].clone(), result.clone()))
        return result

    monkeypatch.setattr(compute, "_import_depth_coefs", record)
    _run(_set(), 2)
    assert len(reads) == 1
    assert not torch.equal(*reads[0])


def test_read_keep_scales_the_output_addition(monkeypatch):
    st = _set(rectangular=True)
    first = _run(st, 1)
    monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **kw: torch.ones_like(z))
    full = _run(st, 2) - first
    monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **kw: torch.full_like(z, 0.5))
    kept = _run(st, 2) - first
    torch.testing.assert_close(kept, full * 0.5)


def test_depth_three_fold_option_matches_loop(monkeypatch):
    st = _set()
    monkeypatch.setattr(compute, "_import_fold_enabled", lambda: True)
    selected = _run(st, 3, dispatch=True)
    monkeypatch.setattr(compute, "_import_fold_enabled", lambda: False)
    with compute.eager():
        loop = _run(st, 3, dispatch=True)
    torch.testing.assert_close(selected, loop, rtol=0, atol=1e-5)


def test_depth_chain_offers_fused_dispatch(monkeypatch):
    calls = []

    def unavailable():
        calls.append(1)
        return None

    monkeypatch.setattr(compute, "_fused_module", unavailable)
    expected = _run(_set(), 3)
    assert torch.equal(_run(_set(), 3, dispatch=True), expected)
    assert calls


@pytest.mark.parametrize("form", ["input", "state", "direction", "exact"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate", ["none", "adaptive"])
def test_depth_one_is_bit_identical_to_fixed_read(monkeypatch, form, dtype, gate):
    st = _set("state" if form == "state" else "input")
    if form == "direction":
        st = dataclasses.replace(st, direction_core=torch.ones(1, 4) * 0.1)
    if form == "exact":
        st = dataclasses.replace(st, exact={0: (torch.eye(2) * 0.1, torch.eye(2) * 0.1)})
    st = dataclasses.replace(st, gate_mode=gate)
    before = _run(st, 1, dispatch=True, dtype=dtype)
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    assert torch.equal(_run(st, 1, dispatch=True, dtype=dtype), before)


def test_path_iteration_is_bit_identical(monkeypatch):
    st = dataclasses.replace(_set(), iteration="path")
    before = _run(st, 3, dispatch=True)
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    assert torch.equal(_run(st, 3, dispatch=True), before)


def test_bound_also_holds_when_read_keep_is_one(monkeypatch):
    monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **kw: torch.ones_like(z))
    st = dataclasses.replace(_set(), up=torch.ones(1, 4) * 10)
    assert _run(st, 16).norm() <= 2 * _run(st, 1).norm() + 1e-5


def test_import_interpreter_matches_depth_chain():
    from kilora import ir

    st = _set()
    ordinary = _run(st, 3, dispatch=True)
    with ir.interpret_all():
        interpreted = _run(st, 3, dispatch=True)
    assert torch.equal(interpreted, ordinary)


@pytest.mark.parametrize("depth", [1.5, 2.5, 3.25])
def test_fractional_depth_blends_whole_chain_outputs(depth):
    st = _set()
    low = int(depth)
    expected = _run(st, low) + (depth - low) * (_run(st, low + 1) - _run(st, low))
    torch.testing.assert_close(_run(st, depth), expected)


def test_exact_import_change_is_bounded():
    st = dataclasses.replace(_set(), exact={0: (torch.eye(2) * 0.1, torch.eye(2) * 0.1)})
    assert _run(st, 16).norm() <= 2 * _run(st, 1).norm() + 1e-6


def test_import_depth_freezes_chain_routes_and_keeps():
    from kilora import route_freeze

    st = dataclasses.replace(_set(), route_freeze=0.0)
    expected = _run(st, 3, dispatch=True)
    cache = {}
    with route_freeze.freezing(cache), compute.sampling_progress(0.9):
        first = _run(st, 3, dispatch=True)
        again = _run(st, 3, dispatch=True)
    assert torch.equal(first, expected) and torch.equal(again, expected)
    assert cache["import-depth"]["chain_sets"]
    assert len(cache["import-depth"]["chain_keeps"]) == 2
