# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import torch

from kilora import compute
from kilora.keydiag_state import query_motion, state_keys


def layer():
    eye = torch.eye(3)
    return compute.LayerSet(
        router=eye,
        keys_in=eye[:, None],
        keys_out=eye[:, None],
        down=eye,
        up=eye * 0.1,
        routing_select="top1",
        gate_mode="none",
    )


def test_planted_keys_and_row_counts():
    s = layer()
    result = state_keys(s)
    torch.testing.assert_close(torch.tensor(result["own"]), torch.ones(3))
    assert result["best_other"] == [0.0, 0.0, 0.0]
    s = dataclasses.replace(s, up=s.up.clone())
    s.up[0].zero_()
    assert state_keys(s)["own"][0] == 0.0


def test_query_motion_fixed_routing_and_depth_one():
    s = layer()
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    single = query_motion(s, x, x, 1)
    assert single["query_motion"] == [0.0]
    assert single["argmax_changes"] == [0.0]
    result = query_motion(s, x, x, 3)
    assert result["argmax_changes"] == [0.0, 0.0, 0.0]
    assert result["routing_entropy"] == [0.0, 0.0, 0.0]
    assert len(result["query_motion"]) == 3
    assert result["query_motion"][1] > 0


def test_probes_do_not_change_eager_output_bits(monkeypatch):
    s = layer()
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    with torch.no_grad(), compute.eager():
        expected = compute.run_layer(x, x, [s], 3, target="test")
        with compute.key_probes() as records:
            actual = compute.run_layer(x, x, [s], 3, target="test")
    assert len([e for e in records["test"] if isinstance(e["k"], int)]) == 3
    assert torch.equal(expected, actual)


def test_derived_path_exposes_query_motion():
    s = dataclasses.replace(layer(), iteration="path")
    x = torch.ones(1, 2, 3)
    result = query_motion(s, x, x, 2)
    assert len(result["query_motion"]) == 2
    assert result["query_motion"][0] == 0
    assert result["query_motion"][1] > 0
    assert len(result["routing_entropy"]) == 2
