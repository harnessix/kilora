# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from live_helpers import build_live_mod

from kilora import compose
from kilora.live import open_session
from kilora.sequence_session import load_session, save_session


def test_session_round_trip_and_allowed_fields(tmp_path):
    steps = [
        {"type": "direction", "basis_id": "basis-1", "coefficients": [0.25, -2], "outcome": "keep"},
        {"type": "prompt_edit", "edit": "+ blue", "outcome": "discard"},
    ]
    path = tmp_path / "session.json"
    save_session(path, steps)
    assert load_session(path) == steps
    assert json.loads(path.read_text()) == steps
    assert "image" not in path.read_text() and "full prompt" not in path.read_text()
    with pytest.raises(ValueError):
        save_session(path, [{**steps[0], "image": "pixels"}])
    assert load_session(path) == steps


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_coefficients_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        save_session(
            tmp_path / "bad.json",
            [{"type": "direction", "basis_id": "x", "coefficients": [value], "outcome": "keep"}],
        )


def test_recording_is_off_by_default(model, x, tmp_path):
    path, _, _ = build_live_mod(model, tmp_path / "ink.kmod")
    session = open_session(path)

    def run():
        adapter = compose(model, session.sets(), depth=2, softmax_scope="per_set")
        with torch.no_grad():
            result = model(x).clone()
        adapter.remove()
        return result

    before = run()
    assert torch.equal(before, run())
    assert not list(tmp_path.rglob("*.json"))
