# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import ablation_settings, compute
from kilora.adaptive_gate import adaptive_keep
from kilora.comfyui.layer_batch import _separate_layers
from kilora.comfyui.patching import _key_mask_by_step

VARIABLES = (
    "KILORA_READ_DROPOUT",
    "KILORA_CARRY_STEPS",
    "KILORA_GATE_COHERENCE",
    "KILORA_STEP_STATS",
)


@pytest.fixture(autouse=True)
def clear_settings(monkeypatch):
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _case(form="state", carry="drop"):
    sets = [
        dataclasses.replace(s, form=form, carry=carry, gate_mode="none")
        for s in _sets(1, 16, 4, 2, 8, seed=3)
    ]
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(1, 5, 16, generator=gen)
    base = torch.randn(1, 5, 16, generator=gen)
    return sets, x, base


def _run(sets, x, base, depth=3):
    with torch.no_grad(), compute.dropout_key(0, torch.tensor([2.0])):
        return compute.run_layer(x, base, sets, depth, target="layer")


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("fast", [False, True])
def test_unset_is_equal_to_reference_from_same_run(form, fast, monkeypatch):
    sets, x, base = _case(form)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad(), compute.dropout_key(0, torch.tensor([2.0])):
        reference = compute._timed_run_layer(x, base, sets, 3, target="layer")
    actual = _run(sets, x, base)
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("carry", ["drop", "dropped"])
@pytest.mark.parametrize("fast", [False, True])
def test_dropout_off_reads_accumulated_change(form, carry, fast, monkeypatch):
    sets, x, base = _case(form, carry)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    monkeypatch.setenv("KILORA_READ_DROPOUT", "off")
    assert not compute._fast_path_ok(base, sets, 0, None, None)
    seen, accumulated = [], []
    original_dropout = compute.read_dropout
    original_route = compute._route

    def dropout(z, read, per_expert, target, k, columns=None, set_key=""):
        accumulated.append(z.detach().clone())
        mask = original_dropout(z, read, per_expert, target, k, columns, set_key)
        assert torch.equal(mask, torch.ones_like(z))
        return mask

    def route(queries, *args, **kwargs):
        seen.append([q.detach().clone() for q in queries])
        return original_route(queries, *args, **kwargs)

    monkeypatch.setattr(compute, "read_dropout", dropout)
    if form == "input":
        monkeypatch.setattr(compute, "_route", route)
    else:
        original_mixed = compute._mixed_route

        def mixed(state, *args, **kwargs):
            seen.append(state.detach().clone())
            return original_mixed(state, *args, **kwargs)

        monkeypatch.setattr(compute, "_mixed_route", mixed)
    _run(sets, x, base)
    assert len(accumulated) == 2 and len(seen) == 3
    st = sets[0]
    for k, z in enumerate(accumulated, 1):
        state = base + z @ st.up
        if form == "state":
            assert torch.equal(seen[k], state)
        else:
            assert torch.equal(seen[k][0], torch.nn.functional.linear(state, st.router))
    z = accumulated[-1]
    assert torch.equal(compute.carry_transfer(z, z * 0.2, st.per_expert), z)
    assert torch.equal(compute.keep_probability(z, torch.ones_like(z)), torch.ones_like(z))


def _wrapped(sets, x, base, default_carry=False):
    class Model:
        def __init__(self):
            self.model_options = {}

    model = Model()
    _key_mask_by_step(model, carry=default_carry)
    wrapper = model.model_options["model_function_wrapper"]

    def apply(value, timestep, **kwargs):
        return compute.run_layer(value, base, sets, 3, target="layer")

    def call(step):
        with torch.no_grad():
            return wrapper(apply, {"input": x, "timestep": torch.tensor([step]), "c": {}})

    return call


@pytest.mark.parametrize("score", ["keys", "read_share", "derived"])
def test_carry_on_for_routed_import_and_off_for_enabled_mod(score, monkeypatch):
    monkeypatch.setenv("KILORA_READ_DROPOUT", "off")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    sets, x, base = _case("input")
    sets = [dataclasses.replace(st, import_score=score) for st in sets]
    call = _wrapped(sets, x, base)
    monkeypatch.setenv("KILORA_CARRY_STEPS", "on")
    first, second = call(3.0), call(2.0)
    assert not torch.equal(first, second)
    monkeypatch.setenv("KILORA_CARRY_STEPS", "off")
    assert torch.equal(call(1.0), call(0.5))
    sets, x, base = _case()
    enabled = _wrapped(sets, x, base, default_carry=True)
    assert torch.equal(enabled(3.0), enabled(2.0))


def test_gate_coherence_off_keeps_size_term(monkeypatch):
    z = torch.tensor([[[1.0, -0.5], [0.5, 1.0]]])
    before = z * 2
    up = torch.eye(2)
    history = z.clone()
    norm, rms = torch.ones(1, 2), torch.ones(1, 2)
    monkeypatch.setenv("KILORA_GATE_COHERENCE", "off")
    _, channels, _ = adaptive_keep(z, before, history, up, norm, rms)
    size = torch.pow(0.5, z.square().mean(-2).sqrt() / (rms + 1e-6))
    assert torch.equal(channels, size[:, None, :])
    assert ablation_settings.gate_coherence(True) is False
    assert ablation_settings.gate_coherence(False) is False


@pytest.mark.parametrize("form", ["state", "input"])
def test_step_stats_two_product_calls(form, tmp_path, monkeypatch):
    sets, x, base = _case(form)
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("KILORA_STEP_STATS", str(path))
    monkeypatch.setenv("KILORA_CARRY_STEPS", "on")
    call = _wrapped(sets, x, base)
    call(3.0)
    assert len(path.read_text().splitlines()) == 1
    call(2.0)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    for line, step in zip(lines, (3.0, 2.0), strict=True):
        assert line["target"] == "layer"
        assert line["sampler_timestep"] == step
        assert [d["step"] for d in line["depth_steps"]] == [1, 2, 3]
        assert all(d["relative_change"] >= 0 for d in line["depth_steps"])
        assert len(line["keep_probability_histogram"]) == 10
        assert sum(line["keep_probability_histogram"]) == 2 * 5 * 8
        assert line["max_transferred_over_sigma"] >= 0
        assert line["accumulated_change_norm"] >= 0


@pytest.mark.parametrize("name", VARIABLES[:3])
@pytest.mark.parametrize("value", ["invalid", "", "ON"])
def test_invalid_value_raises(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    sets, x, base = _case()
    with pytest.raises(ValueError, match=name):
        _run(sets, x, base)


def test_read_dropout_off_preserves_output_gate_rates(monkeypatch):
    z = torch.tensor([[[1.0, -0.5], [0.5, 1.0]]])
    args = (z, z * 0.5, -z, torch.eye(2), torch.ones(1, 2), torch.ones(1, 2))
    reference = adaptive_keep(*args)
    monkeypatch.setenv("KILORA_READ_DROPOUT", "off")
    actual = adaptive_keep(*args)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))


def test_stats_with_dropout_off_has_only_keep_one(tmp_path, monkeypatch):
    sets, x, base = _case()
    path = tmp_path / "off.jsonl"
    monkeypatch.setenv("KILORA_READ_DROPOUT", "off")
    monkeypatch.setenv("KILORA_STEP_STATS", str(path))
    _run(sets, x, base)
    line = json.loads(path.read_text())
    assert line["keep_probability_histogram"] == [0] * 9 + [80]
    assert line["max_transferred_over_sigma"] is None
    assert line["accumulated_change_norm"] is None


@pytest.mark.parametrize(
    "batch,separate,options,source,zbuffer,expected",
    [
        (True, False, {}, "mask", "", False),
        (False, False, {}, "mask", "", True),
        (True, True, {}, "mask", "", True),
        (True, False, {"kilora_layer_separate": True}, "mask", "", True),
        (True, True, {"kilora_layer_separate": False}, "mask", "", False),
        (True, False, {"kilora_layer_filter": "ip"}, "mask", "", True),
        (True, False, {}, "attention", "", True),
        (True, False, {}, "mask", "layer", True),
    ],
)
def test_carry_override_preserves_layered_fallback(
    batch, separate, options, source, zbuffer, expected, monkeypatch
):
    entries = [
        {
            "kilora_layer": "ip",
            "kilora_layer_batch": batch,
            "kilora_alpha_source": source,
        }
    ]
    monkeypatch.setenv("KILORA_AXIS_ZBUFFER", zbuffer)
    reference = bool(_separate_layers(entries, options, separate))
    assert reference is expected
    for value in ("on", "off"):
        monkeypatch.setenv("KILORA_CARRY_STEPS", value)
        assert bool(_separate_layers(entries, options, separate)) is reference


@pytest.mark.parametrize("score", ["keys", "read_share", "derived"])
@pytest.mark.parametrize("fast", [False, True])
def test_stats_does_not_enable_input_carry(score, fast, tmp_path, monkeypatch):
    sets, x, base = _case("input")
    sets = [dataclasses.replace(st, import_score=score) for st in sets]
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("KILORA_STEP_STATS", str(path))
    cache = {}
    with compute.step_carry(cache):
        first = _run(sets, x, base)
        second = _run(sets, x, base)
    assert torch.equal(second, first)
    assert cache == {}
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert all(line["accumulated_change_norm"] is None for line in lines)
