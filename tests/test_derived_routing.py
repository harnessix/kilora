# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute


@pytest.fixture(autouse=True)
def full_derived_mode(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")


def test_rank_scale_and_content(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "4")
    down = torch.zeros(64, 40)
    down[:4, :4] = torch.eye(4)
    down[32:, :32] = torch.eye(32) * 10
    down[32:36, :4] *= 2
    s = dataclasses.replace(planted(), down=down, up=down.clone(), per_expert=32)
    x = torch.zeros(1, 40)
    x[0, :4] = 1
    scores = compute.derived_logits(s, x)
    torch.testing.assert_close(scores[0, 0], scores[0, 1])
    other = dataclasses.replace(s, down=down.clone())
    other.down[32:] = 0
    other.down[32:36, 4:8] = torch.eye(4) * 10
    x = torch.zeros(1, 40)
    x[0, 4:8] = 1
    assert compute.derived_logits(other, x).argmax(-1).item() == 1


def test_cache_tracks_changed_weights(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = planted()
    first = compute.derived_keys(s)
    assert compute.derived_keys(s) is first
    s.down[0] = torch.tensor([0.0, 2.0])
    second = compute.derived_keys(s)
    assert second is not first
    torch.testing.assert_close(second[0][0, 0].abs(), torch.tensor([0.0, 1.0]))


def test_default_unchanged_and_prior(monkeypatch):
    s = planted()
    s.import_score = "keys"
    x = torch.tensor([[[1.0, 0.5]]])
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    with torch.no_grad():
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 2)
        monkeypatch.setenv("KILORA_IMPORT_SCORE", "keys")
        assert torch.equal(baseline, compute.run_layer(x, torch.zeros_like(x), [s], 2))
        monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
        monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
        monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0.00001")
        routes = []
        with compute.recording_shares(lambda _, gates: routes.append(gates[0])):
            compute.run_layer(x, torch.zeros_like(x), [s], 2, target="layer")
    assert routes[1][..., 1].item() > routes[0][..., 1].item()


@pytest.mark.parametrize("kind", ["trace", "transmission", "embed"])
def test_operator_matches_dense(monkeypatch, kind):
    torch.manual_seed(7)
    s = dataclasses.replace(planted(), down=torch.randn(6, 5), up=torch.randn(6, 5), per_expert=3)
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", kind)
    w = [b.T @ a for a, b in zip(s.down.reshape(2, 3, 5), s.up.reshape(2, 3, 5), strict=True)]
    match = torch.empty(2, 2)
    for e in range(2):
        for f in range(2):
            prod = w[f] @ w[e]
            value = torch.trace(prod) if kind == "trace" else torch.linalg.norm(prod)
            match[e, f] = value / (torch.linalg.norm(w[f]) * torch.linalg.norm(w[e]))
    match = match.clamp_min(0)
    match /= match.sum(-1, keepdim=True).clamp_min(1e-30)
    torch.testing.assert_close(compute.share_transition(s), match, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("gamma", ["0", "0.5"])
@pytest.mark.parametrize("routing", ["token", "tensor", "token_top1", "tensor_top1"])
def test_fused_matches_eager(monkeypatch, gamma, routing):
    pytest.importorskip("triton")
    import os

    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", gamma)
    s = _imported(3, n_in=24, routing=routing, gate_mode="none")
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    expected, used = _run(monkeypatch, False, x, y, [s], 2)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], 2)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_plain_and_lowrank_agree(monkeypatch):
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.4")
    with torch.no_grad(), compute.dropout_key(7, 3):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        plain = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, torch.zeros_like(x), [s], 3)
    torch.testing.assert_close(fast, plain)


def test_mixed_sets_are_explicit(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = planted()
    x = torch.ones(1, 1, 2)
    state = dataclasses.replace(s, form="state", import_score="keys")
    assert compute.import_score(state) == "derived"
    routes = []
    with torch.no_grad(), compute.recording_shares(lambda _, gates: routes.append(gates)):
        actual = compute.run_layer(x, x, [s, state], 2)
    assert torch.isfinite(actual).all()
    assert len(routes) == 2
    for gates in routes:
        torch.testing.assert_close(torch.cat(gates, -1).sum(-1), torch.ones_like(x[..., 0]))


def test_sampled_entries_are_seeded_distinct_and_order_independent(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = dataclasses.replace(
        planted(),
        down=torch.eye(4),
        up=torch.eye(4),
        keys_in=torch.eye(4).unsqueeze(1),
        expert_ids=torch.tensor([17, 29, 31, 43]),
    )
    x = torch.ones(1, 64, 4)

    def draw(seed, obj=s):
        with compute.dropout_key(seed, 12):
            return torch.stack(compute.derived_entries(obj, x, "layer", 3), dim=-2)

    first = draw(4)
    assert torch.equal(first, draw(4))
    assert not torch.equal(first, draw(5))
    assert torch.equal(first.sum(-2).sum(-1), torch.full((1, 64), 3.0))
    assert first.sum(-2).amax().item() == 1
    perm = torch.tensor([2, 0, 3, 1])
    reordered = dataclasses.replace(
        s,
        down=s.down[perm],
        up=s.up[perm],
        keys_in=s.keys_in[perm],
        expert_ids=s.expert_ids[perm],
    )
    assert torch.equal(first, draw(4, reordered)[..., torch.argsort(perm)])


def test_entry_temperature_approaches_argmax(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0.00001")
    s = planted()
    x = torch.tensor([[[1.0, 0.1]]]).expand(1, 32, 2)
    with compute.dropout_key(27, 1):
        entry = compute.derived_entries(s, x, "layer", 1)[0]
    assert torch.equal(entry.argmax(-1), compute.derived_logits(s, x).argmax(-1))


def test_derived_import_defaults_are_isolated(monkeypatch):
    from kilora.adapter import imported_schedule, imported_settings

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    assert imported_settings().depth == 3
    assert imported_settings().path_width == 1
    assert imported_schedule() is not None
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    assert imported_settings().depth == 4
    assert imported_settings().path_width == 3
    assert imported_settings(depth=2, path_width=1).depth == 2
    assert imported_settings(depth=2, path_width=1).path_width == 1
    assert imported_schedule() is None


def test_wide_accumulate_routes_each_entry(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = dataclasses.replace(planted(), path_width=2, iteration="accumulate")
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    routes = []
    with (
        torch.no_grad(),
        compute.dropout_key(8, 1),
        compute.recording_shares(lambda _, gates: routes.append(gates[0])),
    ):
        output = compute.run_layer(x, torch.zeros_like(x), [s], 2, target="layer")
    assert torch.isfinite(output).all()
    assert len(routes) == 4
    assert torch.all(routes[0].argmax(-1) != routes[2].argmax(-1))
    assert not torch.equal(routes[0], routes[1])


def test_fused_wide_accumulate_matches_eager(monkeypatch):
    pytest.importorskip("triton")
    import os

    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = _imported(3, n_in=24, gate_mode="none")
    s.path_width = 3
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    expected, used = _run(monkeypatch, False, x, y, [s], 3)
    assert not used
    actual, used = _run(monkeypatch, True, x, y, [s], 3)
    assert used
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_cross_layer_carry_and_boundaries(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "4")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0.00001")
    s = dataclasses.replace(planted(), path_width=1, key="joint")
    a = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    routes = []

    def layer(x, target, index):
        with compute.recording_shares(lambda _, gates: routes.append(gates[0].clone())):
            compute.run_layer(x, torch.zeros_like(x), [s], 1, target=target, index=index)

    with torch.no_grad(), compute.dropout_key(1, 1), compute.derived_call():
        layer(a, "blocks.0.attn.q", 0)
        layer(b[:, :1], "blocks.1.attn.k", 1)
        layer(b, "blocks.1.attn.q", 1)
        layer(b, "blocks.3.attn.q", 3)
        layer(a, "blocks.3.attn.v", 3)
    assert routes[0].argmax(-1).eq(0).all()
    assert routes[1].argmax(-1).eq(1).all()
    assert routes[2].argmax(-1).eq(0).all()
    assert routes[3].argmax(-1).eq(0).all()
    assert routes[4].argmax(-1).eq(0).all()
    routes.clear()
    with torch.no_grad(), compute.dropout_key(1, 1), compute.derived_call():
        layer(b, "blocks.0.attn.q", 0)
    assert routes[0].argmax(-1).eq(1).all()


def test_carry_zero_matches_without_call_context(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "0")
    s = dataclasses.replace(planted(), path_width=2)
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    with torch.no_grad(), compute.dropout_key(2, 1):
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 2, target="layer")
        with compute.derived_call():
            compute.run_layer(x, torch.zeros_like(x), [s], 2, target="previous")
            result = compute.run_layer(x, torch.zeros_like(x), [s], 2, target="layer")
    assert torch.equal(result, baseline)


def test_cross_layer_fused_matches_eager(monkeypatch):
    pytest.importorskip("triton")
    import os

    from test_fused_input import DEVICE, _imported, _inputs, _run

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "0")
    s = _imported(3, n_in=24, gate_mode="none")
    s.path_width = 3
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)

    def pair(on):
        with compute.derived_call():
            first, used = _run(monkeypatch, on, x, y, [s], 2)
            assert used == on
            second, used = _run(monkeypatch, on, x.flip(-1), y, [s], 2)
            assert used == on
        return first, second

    eager = pair(False)
    fused = pair(True)
    for actual, expected in zip(fused, eager, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_generation_separates_model_calls(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "0")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0.00001")
    s = dataclasses.replace(planted(), key="joint")
    routes = []

    class Model(torch.nn.Module):
        def forward(self, x, t):
            with compute.recording_shares(lambda _, gates: routes.append(gates[0].clone())):
                return compute.run_layer(x, torch.zeros_like(x), [s], 1, target="layer")

    model = Model()
    a = torch.tensor([[[1.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0]]])
    with torch.no_grad(), compute.generation(model, carry=False):
        model(a, 1)
        model(b, 1)
        model(a, 0)
    assert [route.argmax(-1).item() for route in routes] == [0, 1, 0]


def test_carry_uses_final_depth_route(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "0")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0.00001")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "10")
    s = dataclasses.replace(planted(), key="joint")
    x = torch.tensor([[[1.0, 0.0]]])
    routes = []
    with (
        torch.no_grad(),
        compute.dropout_key(1, 1),
        compute.derived_call(),
        compute.recording_shares(lambda _, gates: routes.append(gates[0].clone())),
    ):
        compute.run_layer(x, torch.zeros_like(x), [s], 2, target="first", index=0)
        compute.run_layer(x, torch.zeros_like(x), [s], 1, target="second", index=1)
    assert routes[0].argmax(-1).item() == 0
    assert routes[1].argmax(-1).item() == 1
    assert routes[2].argmax(-1).item() == 1


def test_carry_context_does_not_change_legacy_routing(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = planted()
    x = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    with torch.no_grad(), compute.dropout_key(4, 1):
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")
        with compute.derived_call():
            result = compute.run_layer(x, torch.zeros_like(x), [s], 3, target="layer")
    assert torch.equal(result, baseline)


def test_comparison_arm_and_transition_defaults(monkeypatch):
    from kilora.adapter import imported_schedule, imported_settings

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "5")
    assert imported_settings().path_width == 1
    assert imported_settings().depth == 3
    assert imported_schedule() is not None
    assert compute.share_transition_gamma(planted()) == 0
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.delenv("KILORA_SHARE_TRANSITION")
    assert compute.share_transition_gamma(planted()) == 1
    assert imported_settings(depth=2, path_width=1).depth == 2
    assert imported_settings(depth=2, path_width=1).path_width == 1
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    assert compute.share_transition_gamma(planted()) == 0


def test_carry_similarity_is_cached_and_respects_subspaces(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = planted()
    s.up = s.down.clone()
    orthogonal = compute.derived_keys(s)[4]
    torch.testing.assert_close(orthogonal, torch.zeros_like(orthogonal))
    s.down[1] = s.down[0]
    s.up[1] = s.up[0]
    keys = compute.derived_keys(s)
    torch.testing.assert_close(keys[4], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert compute.derived_keys(s) is keys


def test_later_entries_and_periodic_anchor(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    monkeypatch.setenv("KILORA_DERIVED_CARRY_SPREAD", "0")
    monkeypatch.setenv("KILORA_DERIVED_ANCHOR", "3")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    s = dataclasses.replace(planted(), key="joint")
    calls, depths = [], []
    original_entries = compute.derived_entries
    original_run = compute._derived_batched_paths

    def entries(*args, **kwargs):
        calls.append(args[5])
        return original_entries(*args, **kwargs)

    def run(*args, **kwargs):
        depths.append(args[6])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(compute, "derived_entries", entries)
    monkeypatch.setattr(compute, "_derived_batched_paths", run)
    x = torch.tensor([[[1.0, 0.0]]])
    with torch.no_grad(), compute.dropout_key(1, 1), compute.derived_call():
        for index in range(3):
            compute.run_layer(x, torch.zeros_like(x), [s], 4, target=str(index), index=index)
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None
    assert depths == [4, 2, 2]


def test_weaker_lift_keeps_distinct_transition_paths(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    s = planted()
    s.tau = 1.0
    x = torch.tensor([[[1.0, 0.6]]])
    transition = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    coefs = [torch.zeros_like(x)]
    prev_a = [torch.tensor([[[1.0, 0.0]]])]
    prev_b = [torch.tensor([[[0.0, 1.0]]])]
    a = compute._import_gates([s], coefs, [None], x, prev_a, [transition], 1.0)[0]
    b = compute._import_gates([s], coefs, [None], x, prev_b, [transition], 1.0)[0]
    assert a.argmax(-1).item() == 0
    assert b.argmax(-1).item() == 1
    monkeypatch.setenv("KILORA_DERIVED_LIFT_AFTER", "1")
    actual = compute._import_gates([s], coefs, [None], x, prev_a, [transition], 1.0)[0]
    logits = compute.derived_logits(s, x) + (prev_a[0] @ transition + 1e-6).log()
    assert torch.equal(actual, compute._gate(logits, s, 0.0, None))


def test_keys_arm_matches_read_share_for_same_top1(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    s = dataclasses.replace(planted(), routing_select="top1", path_width=3)
    x = torch.tensor([[[1.0, 0.2], [0.2, 1.0]]])
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    with torch.no_grad():
        baseline = compute.run_layer(x, torch.zeros_like(x), [s], 3)
        monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
        result = compute.run_layer(x, torch.zeros_like(x), [s], 3)
    torch.testing.assert_close(result, baseline, rtol=1e-6, atol=1e-7)


def test_zero_entry_temperature_is_deterministic(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    s = planted()
    x = torch.tensor([[[1.0, 0.1]]]).expand(1, 32, 2)
    with compute.dropout_key(3, 1):
        first = compute.derived_entries(s, x, "layer", 1)[0]
    with compute.dropout_key(4, 1):
        second = compute.derived_entries(s, x, "layer", 1)[0]
    assert torch.equal(first, second)
    assert torch.equal(first.argmax(-1), compute.derived_logits(s, x).argmax(-1))


def test_keys_arm_matches_read_share_for_stable_top1_chain(monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = dataclasses.replace(planted(), up=torch.eye(2) * 0.1, routing_select="top1")
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    outputs = []
    for score in ("read_share", "derived"):
        monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
        with torch.no_grad(), compute.dropout_key(3, 1):
            outputs.append(compute.run_layer(x, torch.zeros_like(x), [s], 3))
    torch.testing.assert_close(*outputs, rtol=1e-6, atol=1e-7)
