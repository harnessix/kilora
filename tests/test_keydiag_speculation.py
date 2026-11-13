# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute
from kilora.keydiag_speculation import speculation


@pytest.fixture(autouse=True)
def derived(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.delenv("KILORA_SHARE_TRANSITION", raising=False)
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "0")


def record(s, x, depth=3):
    return {
        "sets": [s],
        "x": x,
        "y0": torch.zeros_like(x),
        "depth": depth,
        "target": "layer",
        "step": 0,
    }


def test_identity_transition_matches_and_is_json_safe():
    import json

    s = dataclasses.replace(planted(), up=torch.eye(2), path_width=1)
    result = speculation(record(s, torch.tensor([[1.0, 0.1], [0.1, 1.0]])), 0)
    assert result["gamma"] == 1.0
    assert result["argmax_all"] == 1.0
    assert result["weights_exact_all"] == 1.0
    assert result["weights_within_tolerance_all"] == 1.0
    assert [s["depth"] for s in result["depth_steps"]] == [2, 3]
    json.dumps(result, allow_nan=False)


def test_weights_and_path_use_independent_recursive_predictions(monkeypatch):
    s = dataclasses.replace(planted(), path_width=1)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(compute, "share_transition", lambda s: torch.full((2, 2), 0.5))
    monkeypatch.setattr(
        compute, "derived_entries", lambda *args: [torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])]
    )
    monkeypatch.setattr(
        compute, "derived_logits", lambda s, x: torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    )
    result = speculation(record(s, x), 0, tolerance=1e-5)
    assert result["argmax_all"] == 0.5
    assert result["weights_within_tolerance_all"] == 0.0
    assert result["weights_exact_all"] == 0.0
    assert result["depth_steps"][0]["argmax"] == 0.5


def test_measurement_matches_actual_input_walk():
    generator = torch.Generator().manual_seed(51)
    down = torch.randn(3, 5, generator=generator)
    up = torch.randn(3, 5, generator=generator)
    s = dataclasses.replace(
        planted(),
        router=torch.eye(5),
        keys_in=torch.eye(5)[:3, None],
        down=down,
        up=up,
        path_width=1,
    )
    x = torch.randn(13, 5, generator=generator)
    r = record(s, x, depth=4)
    result = speculation(r, 0)
    with torch.no_grad():
        batched = x.unsqueeze(0)
        entry = compute.derived_entries(s, batched, "layer#input0", 1)[0]
        stats = {"record_macro": True}
        compute._run_input(
            batched,
            torch.zeros_like(batched),
            [s],
            [0],
            4,
            [None],
            "layer",
            entry=entry,
            stats=stats,
        )
        transition = compute.share_transition(s)
        predicted = entry
        matches = []
        for k, actual in enumerate(stats["routes"][1:]):
            predicted = compute._gate((predicted @ transition + 1e-6).log(), s, 0.0)
            match = (actual == predicted).all(-1)
            matches.append(match)
            assert result["depth_steps"][k]["weights_exact"] == pytest.approx(
                match.float().mean().item()
            )
        assert result["weights_exact_all"] == pytest.approx(
            torch.stack(matches).all(0).float().mean().item()
        )


def test_requires_replayable_inputs(monkeypatch):
    s = planted()
    r = record(s, torch.ones(2, 2))
    assert "regional" in speculation(dict(r, has_regions=True), 0)["skipped"]
    assert "scheduled" in speculation(dict(r, has_schedule=True), 0)["skipped"]
    assert (
        "tensor"
        in speculation(record(dataclasses.replace(s, routing_scope="tensor"), r["x"]), 0)["skipped"]
    )
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "2")
    assert "default" in speculation(r, 0)["skipped"]
