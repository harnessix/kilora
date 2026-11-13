# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from test_attention_route import routed_set
from test_import_attention_plain import MLP

from kilora import compute

PROJECTIONS = (
    [
        f"blocks.0.{attention}.{projection}_proj"
        for attention in ("self_attn", "cross_attn")
        for projection in ("q", "k", "v", "output")
    ]
    + [
        f"blocks.0.{attention}.{projection}"
        for attention in ("attn1", "attn2", "attn")
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    + [
        f"blocks.0.attn.{projection}"
        for projection in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
    ]
)


@pytest.mark.parametrize("target", PROJECTIONS)
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("layout", ["single", "multiple", "mixed", "empty_column"])
def test_projection_layout_matches_feed_forward(target, depth, layout, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x = torch.eye(4)[None]
    first = routed_set()
    second = replace(first, key="second", down=first.down * 0.7)
    if layout == "mixed":
        second = replace(second, form="state")
    if layout == "empty_column":
        down = first.down.clone()
        down[0] = 0
        first = replace(first, down=down)
    sets = [first] if layout in ("single", "empty_column") else [first, second]
    with torch.no_grad(), compute.eager():
        actual = compute.run_layer(x, x, sets, depth, target=target)
        expected = compute.run_layer(x, x, sets, depth, target=MLP)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("target", [PROJECTIONS[0], MLP])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("depth", [1, 2.5, 3])
@pytest.mark.parametrize("imported_first", [False, True])
def test_derived_mixed_sets_share_one_chain(target, iteration, depth, imported_first, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    x = torch.eye(4)[None]
    imported = replace(routed_set(), iteration=iteration)
    state = replace(routed_set("state"), iteration=iteration, down=torch.eye(4) * 0.3)
    sets = [imported, state] if imported_first else [state, imported]
    seen = []
    with (
        torch.no_grad(),
        compute.eager(),
        compute.recording_shares(lambda _, gates: seen.append(torch.cat(gates, -1))),
    ):
        actual = compute.run_layer(x, x, sets, depth, target=target)
        expected = compute._run_input(x, x, sets, [0, 1], depth, [None, None], target)
    assert torch.equal(actual, expected)
    for shares in seen:
        torch.testing.assert_close(shares.sum(-1), torch.ones_like(shares[..., 0]))


@pytest.mark.parametrize("imported_first", [False, True])
def test_derived_mixed_probes_preserve_original_positions(imported_first, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    x = torch.eye(4)[None].requires_grad_()
    imported, state = routed_set(), routed_set("state")
    sets = [imported, state] if imported_first else [state, imported]
    with compute.eager(), compute.key_probes() as record:
        output = compute.run_layer(x, x, sets, 3, target=MLP)
        output.sum().backward()
    entries = record[MLP]
    steps = [e for e in entries if e["k"] != "output"]
    assert len(steps) == 3
    assert all(all(q is not None for q in e["queries"]) for e in steps)
    assert any(all(g is not None for g in e["grads"]) for e in entries if e["k"] == "output")


def test_derived_mixed_route_freeze_records_the_joint_chain(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    x = torch.eye(4)[None]
    sets = [replace(routed_set(form), route_freeze=0.4) for form in ("input", "state")]
    layouts = []
    original = compute._route_freeze.run

    def record(call, inputs, base, keyed, *args):
        layouts.append([s.form for s in keyed])
        return original(call, inputs, base, keyed, *args)

    monkeypatch.setattr(compute._route_freeze, "run", record)
    with (
        torch.no_grad(),
        compute.eager(),
        compute._route_freeze.freezing({}),
        compute.sampling_progress(0.8),
    ):
        first = compute.run_layer(x, x, sets, 3, target=MLP)
        replayed = compute.run_layer(x, x, sets, 3, target=MLP)
    assert layouts == [["input", "state"], ["input", "state"]]
    assert torch.isfinite(first).all() and torch.isfinite(replayed).all()
