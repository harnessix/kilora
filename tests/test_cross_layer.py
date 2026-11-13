# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from kilora.cross_layer import RLSLink, jvp_responses, pass_back, variance_weight


def test_jvp_matches_finite_difference():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    weights = torch.randn(5, 6, dtype=x.dtype)
    out_basis = torch.randn(6, 3, dtype=x.dtype)
    directions = torch.randn(4, *x.shape, dtype=x.dtype)
    block = lambda value: torch.sin(value @ weights) @ out_basis
    got = jvp_responses(block, x, directions)
    eps = 1e-5
    expected = torch.stack(
        [(block(x + eps * d) - block(x - eps * d)) / (2 * eps) for d in directions], dim=-1
    )
    torch.testing.assert_close(got, expected, atol=1e-8, rtol=1e-7)


def test_pass_back_known_link_and_variance_weight():
    response = torch.tensor([[2.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    target = torch.tensor([[6.0, 8.0, 5.0], [2.0, 4.0, 0.0]])
    torch.testing.assert_close(pass_back(response, target), torch.tensor([[3.0, 2.0], [1.0, 1.0]]))
    assert variance_weight(0) == 1
    assert variance_weight(1) == 0.5
    assert variance_weight(float("nan")) == 0
    assert variance_weight(float("inf")) == 0
    with pytest.raises(ValueError, match="ranks"):
        pass_back(response, torch.ones(2))


def test_rls_learns_stream_and_variance_falls():
    torch.manual_seed(7)
    matrix = torch.randn(3, 2, dtype=torch.float64)
    inputs = torch.randn(1000, 3, dtype=torch.float64)
    outputs = inputs @ matrix
    link = RLSLink(3, 2, forget=0.99)
    link.update(inputs[:10], outputs[:10])
    early = link.residual_variance
    link.update(inputs[10:], outputs[10:])
    torch.testing.assert_close(link.predict(inputs[-10:]), outputs[-10:], atol=1e-6, rtol=1e-6)
    assert link.residual_variance < early * 0.01
    assert link.precision > 1 / early


def test_rls_recovers_slope_and_state_with_large_offsets():
    generator = torch.Generator().manual_seed(123)
    matrix = torch.tensor([[1.2, -0.3], [0.4, 1.7], [-0.8, 0.2]], dtype=torch.float64)
    inputs = torch.randn(2000, 3, generator=generator, dtype=torch.float64) + 200
    noise = torch.randn(2000, 2, generator=generator, dtype=torch.float64) * 0.001
    outputs = inputs @ matrix + 200 + noise
    link = RLSLink(3, 2, forget=0.995)
    link.update(inputs, outputs)
    torch.testing.assert_close(link.weights, matrix, atol=0.001, rtol=0)
    delta = torch.tensor([0.7, -1.1, 0.2], dtype=torch.float64)
    torch.testing.assert_close(link.predict(delta), delta @ matrix, atol=0.001, rtol=0)
    torch.testing.assert_close(
        link.predict_state(inputs[-1]), inputs[-1] @ matrix + 200, atol=0.001, rtol=0
    )


def test_rls_loads_legacy_snapshot_with_zero_means():
    original = RLSLink(2, 3)
    original.weights.fill_(2)
    state = original.state_tensors()
    for key in ("mean_in", "mean_out", "mean_weight"):
        del state[key]
    restored = RLSLink.from_tensors(state)
    assert torch.count_nonzero(restored.mean_in) == 0
    assert torch.count_nonzero(restored.mean_out) == 0
    torch.testing.assert_close(
        restored.predict_state(torch.ones(2)), restored.predict(torch.ones(2))
    )


def test_rls_snapshot_restores_online_updates(tmp_path):
    torch.manual_seed(8)
    inputs = torch.randn(30, 3, dtype=torch.float64)
    outputs = inputs @ torch.randn(3, 2, dtype=torch.float64)
    link = RLSLink(3, 2, forget=0.97)
    link.update(inputs[:15], outputs[:15])
    saved = link.state_tensors()
    path = tmp_path / "link.safetensors"
    save_file(saved, str(path))
    restored = RLSLink.from_tensors(load_file(str(path)))
    link.update(inputs[15:], outputs[15:])
    restored.update(inputs[15:], outputs[15:])
    assert torch.equal(link.weights, restored.weights)
    assert torch.equal(link.covariance, restored.covariance)
    assert link.residual_variance == restored.residual_variance
    assert not torch.equal(saved["weights"], link.weights)


def test_rls_snapshot_rejects_invalid_state():
    state = RLSLink(2, 3).state_tensors()
    state["covariance"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shapes"):
        RLSLink.from_tensors(state)


def test_runtime_does_not_import_cross_layer():
    code = """
import sys
import torch
import kilora.compute
assert 'kilora.cross_layer' not in sys.modules
x = torch.tensor([1., 2.])
layer = torch.nn.Linear(2, 3)
before = layer(x)
import kilora.cross_layer
after = layer(x)
assert torch.equal(before, after)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
