# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute
from kilora.comfyui.patching import StepCarry


@pytest.fixture(autouse=True)
def derived(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "0")
    monkeypatch.setenv("KILORA_FUSED", "0")


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_grouped_coordinates_match_individual(scope, iteration):
    s = dataclasses.replace(planted(), routing_scope=scope, iteration=iteration)
    transition = compute.share_transition(s)
    cache = {}
    for n in range(3):
        c = torch.randn(2, 4, 2)
        logits = torch.randn(2, 4, 2)
        entry = torch.nn.functional.one_hot(torch.full((2, 4), n % 2), 2).float()
        cache[n] = dict(
            c=c,
            logits=logits,
            entry=entry.unsqueeze(0),
            depth=4,
            per_expert=1,
            select="sparsemax",
            scope=scope,
            iteration=iteration,
            gamma=0.8,
            transition=transition,
            signature=n,
            adaptive=False,
            scale=1.0,
        )
    originals = {key: value.copy() for key, value in cache.items()}
    compute._speculate_prepare(cache)
    for key, old in originals.items():
        prev = old["entry"][0]
        z = old["c"] * prev
        for _ in range(2, 4):
            route = old["logits"] * compute.derived_lift_after()
            route = route + 0.8 * (prev @ transition + 1e-6).log()
            prev = compute._gate(route, s, 0)
            part = old["c"] * prev
            z = part if iteration == "path" else z + part
        torch.testing.assert_close(cache[key]["z"][0], z)
        torch.testing.assert_close(cache[key]["prev"][0], prev)


@pytest.mark.parametrize("fixed_read", [False, True])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_correction_from_exact_previous_projection_matches_loop(
    monkeypatch, width, gate_mode, scope, iteration, fixed_read
):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1" if fixed_read else "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = dataclasses.replace(
        planted(),
        path_width=width,
        gate_mode=gate_mode,
        routing_scope=scope,
        iteration=iteration,
        scale=0.7,
    )
    x = torch.tensor([[[0.7, 0.3], [0.4, 0.8]]])
    y = torch.zeros_like(x)
    cache = {}
    calls = []
    original = compute._derived_batched_paths

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_derived_batched_paths", count)
    with torch.no_grad(), compute.dropout_key(3, 1):
        with compute.derived_call():
            expected = compute.run_layer(x, y, [s], 3, target="layer")
        with compute.speculate(cache), compute.derived_call():
            first = compute.run_layer(x, y, [s], 3, target="layer")
        with compute.speculate(cache), compute.derived_call():
            corrected = compute.run_layer(x, y, [s], 3, target="layer")
    assert torch.equal(first, expected)
    torch.testing.assert_close(corrected, expected)
    assert len(calls) == 2


def test_off_is_identical_and_generation_cache_resets(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, x, t):
            with compute.derived_call():
                return compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")

    s = dataclasses.replace(planted(), path_width=1)
    model = Model()
    x = torch.tensor([[[0.7, 0.3]]])
    monkeypatch.delenv("KILORA_SPECULATE", raising=False)
    with torch.no_grad(), compute.generation(model):
        off = model(x, 1)
    with torch.no_grad():
        direct = model(x, 1)
    assert torch.equal(off, direct)
    carrier = StepCarry()
    args = lambda t: {"input": x, "timestep": t, "c": {}}
    first = carrier.cache_for(args(2))
    first["layer"] = torch.ones(1)
    assert carrier.cache_for(args(1)) is first
    assert carrier.cache_for(args(2)) == {}


def test_model_wrapper_restarts_prediction_for_each_generation(monkeypatch):
    from kilora.comfyui.patching import _key_mask_by_step

    monkeypatch.setenv("KILORA_SPECULATE", "1")
    s = dataclasses.replace(planted(), path_width=1)

    class Model:
        model_options = {}

    model = Model()
    _key_mask_by_step(model)
    wrapper = model.model_options["model_function_wrapper"]
    calls = []
    original = compute._derived_batched_paths

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_derived_batched_paths", count)
    x = torch.tensor([[[0.7, 0.3]]])

    def apply(x, t, **kwargs):
        return compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")

    with torch.no_grad():
        for t in (2.0, 1.0, 2.0, 1.0):
            args = {"input": x, "timestep": torch.tensor([t]), "c": {}}
            wrapper(apply, args)
    assert len(calls) == 2


def test_cache_bytes_counts_shared_storage_once():
    values = torch.zeros(3, 4)
    cache = {"one": values[0], "two": {"view": values[1]}, "other": torch.ones(5)}
    assert compute.speculative_cache_bytes(cache) == (12 + 5) * 4


def test_multiple_sets_have_separate_prediction_state(monkeypatch):
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    sets = [dataclasses.replace(planted(), key=str(i), scale=0.5 + i) for i in range(2)]
    x = torch.tensor([[[0.7, 0.3], [0.4, 0.8]]])
    y = torch.zeros_like(x)
    cache = {}
    calls = []
    original = compute._derived_batched_paths

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_derived_batched_paths", count)
    with torch.no_grad(), compute.dropout_key(3, 1):
        with compute.speculate(cache), compute.derived_call():
            expected = compute.run_layer(x, y, sets, 3, target="layer")
        with compute.speculate(cache), compute.derived_call():
            actual = compute.run_layer(x, y, sets, 3, target="layer")
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 2
    assert len(cache) == 2
