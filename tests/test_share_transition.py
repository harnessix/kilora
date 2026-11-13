# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch

from kilora import compute


def planted():
    return compute.LayerSet(
        router=torch.eye(2),
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(2),
        up=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        per_expert=1,
        form="input",
        import_score="read_share",
        gate_mode="none",
        tau=1.0,
    )


def test_transition_orientation_and_non_square():
    s = planted()
    torch.testing.assert_close(compute.share_transition(s), s.up)
    assert compute.share_transition(dataclasses.replace(s, up=torch.ones(2, 3))) is None
    zero = dataclasses.replace(s, up=torch.zeros_like(s.up))
    assert torch.equal(compute.share_transition(zero), torch.zeros(2, 2))


def test_zero_is_bit_identical_and_keeps_fold(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    y = torch.zeros_like(x)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.delenv("KILORA_SHARE_TRANSITION", raising=False)
    calls = []
    original = compute._read_share_gates

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_read_share_gates", spy)
    with torch.no_grad():
        baseline = compute.run_layer(x, y, [s], 3)
        assert len(calls) == 1
        calls.clear()
        monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
        got = compute.run_layer(x, y, [s], 3)
    assert torch.equal(baseline, got)
    assert len(calls) == 1
    gate = original([s], [x], [None])[0]
    assert torch.equal(got, ((x * gate) * 3) @ s.up)


def test_prior_changes_step_two_and_disables_fold(monkeypatch):
    s = planted()
    x = torch.tensor([[[1.0, 0.5]]])
    y = torch.zeros_like(x)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    routes, reads = [], []
    original = compute._import_depth_coefs

    def record_read(*args, **kwargs):
        value = original(*args, **kwargs)
        reads.append(value)
        return value

    monkeypatch.setattr(compute, "_import_depth_coefs", record_read)
    with torch.no_grad(), compute.recording_shares(lambda _, gates: routes.append(gates[0])):
        compute.run_layer(x, y, [s], 2, target="layer")
    first, second = routes
    expected = compute._gate(
        compute.read_energy(s, reads[0]) / compute.read_energy(s, reads[0]).sum(-1, keepdim=True)
        + (first @ compute.share_transition(s) + 1e-6).log(),
        s,
        0,
    )
    torch.testing.assert_close(second, expected)
    assert second[..., 1].item() > first[..., 1].item()


def test_fold_really_disabled_without_recording(monkeypatch):
    s = planted()
    x = torch.tensor([[[1.0, 0.5]]])
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    calls = []
    original = compute._read_share_gates

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_read_share_gates", spy)
    with torch.no_grad():
        compute.run_layer(x, torch.zeros_like(x), [s], 3)
    assert len(calls) == 3


@pytest.mark.parametrize("kind", ["cosine", "angles", "energy", "weighted"])
@pytest.mark.parametrize("routing", ["token", "tensor", "token_top1", "tensor_top1"])
@pytest.mark.parametrize("depth", [2, 2.5])
def test_fused_matches_eager(monkeypatch, routing, depth, kind):
    pytest.importorskip("triton")
    from test_fused_input import DEVICE, _imported, _inputs, _run

    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.4")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", kind)
    s = _imported(3, n_in=24, routing=routing, gate_mode="none")
    if kind == "weighted":
        s.share_transition_weighted = compute.share_transition(s, states=torch.randn(30, 24))
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    if DEVICE == "cpu":
        import os

        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires the Triton interpreter or a GPU")
    expected, used = _run(monkeypatch, False, x, y, [s], depth)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], depth)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_non_square_keeps_original_result(monkeypatch):
    s = dataclasses.replace(planted(), up=torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]))
    x = torch.tensor([[[1.0, 0.5]]])
    y = torch.zeros(1, 1, 3)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    with torch.no_grad():
        baseline = compute.run_layer(x, y, [s], 3)
        monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
        got = compute.run_layer(x, y, [s], 3)
    assert torch.equal(baseline, got)


def test_plain_and_lowrank_loops_agree(monkeypatch):
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.7, 1.0]]])
    y = torch.zeros_like(x)
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.3")
    with torch.no_grad(), compute.dropout_key(7, 3):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        plain = compute.run_layer(x, y, [s], 3)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, y, [s], 3)
    torch.testing.assert_close(fast, plain)


def test_fused_zero_matches_default(monkeypatch):
    pytest.importorskip("triton")
    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu":
        import os

        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires the Triton interpreter or a GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.delenv("KILORA_SHARE_TRANSITION", raising=False)
    s = _imported(13, n_in=24, gate_mode="none")
    x, y = _inputs(14, batch=1, tokens=5, n_in=24)
    baseline, used = _run(monkeypatch, True, x, y, [s], 3)
    assert used
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    actual, used = _run(monkeypatch, True, x, y, [s], 3)
    assert used
    assert torch.equal(actual, baseline)


def test_transition_kinds_scale_and_repeated_rank(monkeypatch):
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        up=torch.tensor([[1.0, 0.5], [0.2, 1.0]]),
    )
    scaled = dataclasses.replace(s, up=s.up * torch.tensor([[10.0], [1.0]]))
    repeated = dataclasses.replace(
        s,
        down=s.down.repeat_interleave(2, 0),
        up=s.up.repeat_interleave(2, 0),
        per_expert=2,
    )
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "cosine")
    baseline = compute.share_transition(s)
    torch.testing.assert_close(compute.share_transition(scaled), baseline)
    torch.testing.assert_close(compute.share_transition(repeated), baseline)
    uneven = dataclasses.replace(
        s,
        down=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
        up=s.up.repeat_interleave(2, 0),
        per_expert=2,
    )
    torch.testing.assert_close(compute.share_transition(uneven), baseline)
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "energy")
    assert not torch.allclose(compute.share_transition(uneven), compute.share_transition(s))
    scaled_read = dataclasses.replace(s, down=s.down * torch.tensor([[10.0], [1.0]]))
    assert not torch.allclose(compute.share_transition(scaled_read), compute.share_transition(s))


def test_angles_and_weighted_matches(monkeypatch):
    s = planted()
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "cosine")
    cosine = compute.share_transition(s)
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "angles")
    torch.testing.assert_close(compute.share_transition(s), cosine)
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "weighted")
    states = torch.cat([torch.eye(2), -torch.eye(2)])
    weighted = compute.share_transition(s, states=states)
    torch.testing.assert_close(weighted, cosine)
    s.share_transition_weighted = weighted
    torch.testing.assert_close(compute.share_transition(s), cosine)


def test_angles_uses_mean_squared_principal_cosines(monkeypatch):
    s = dataclasses.replace(planted(), up=torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "angles")
    torch.testing.assert_close(compute.share_transition(s)[0], torch.tensor([0.5, 0.5]))
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "cosine")
    assert not torch.allclose(compute.share_transition(s)[1], torch.tensor([0.5, 0.5]))


def test_weighted_uses_centered_state_covariance(monkeypatch):
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 2.0], [1.0, -1.0]]),
        up=torch.tensor([[2.0, 1.0], [0.5, -1.0]]),
    )
    states = torch.tensor([[3.0, 1.0], [1.0, -2.0], [-1.0, 0.0], [2.0, 4.0]])
    x = states - states.mean(0)
    covariance = x.T @ x
    expected = torch.empty(2, 2)
    for e in range(2):
        for f in range(2):
            a, b = s.down[f], s.up[e]
            expected[e, f] = (a @ covariance @ b).abs() / torch.sqrt(
                (a @ covariance @ a) * (b @ covariance @ b)
            )
    expected /= expected.sum(-1, keepdim=True)
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "weighted")
    torch.testing.assert_close(compute.share_transition(s, states=states), expected)


def test_base_transition_matches_explicit_metric(monkeypatch):
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "base")
    s = planted()
    weight = torch.tensor([[2.0, 1.0], [0.0, 1.0], [1.0, -1.0]])
    actual = compute.base_weight_transition(s, weight)
    metric = weight.T @ weight / weight.square().sum()
    graph = torch.empty(2, 2)
    for e in range(2):
        for f in range(2):
            a, b = s.down[f], s.up[e]
            graph[e, f] = (a @ metric @ b).abs() / (
                (a @ metric @ a).sqrt() * (b @ metric @ b).sqrt()
            )
    graph /= graph.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, graph)
    torch.testing.assert_close(
        compute.share_transition(dataclasses.replace(s, share_transition_base=actual)), graph
    )
    assert compute.share_transition(s) is None
