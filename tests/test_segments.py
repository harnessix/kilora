# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_share_transition import planted

from kilora import compute, segments


def test_discovery_and_budget(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    down = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    up = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    s = dataclasses.replace(planted(), down=down, up=up, per_expert=1)
    assert (0, 1) in segments.discover(s, limit=9, min_similarity=1.0)
    assert (0, 1) not in segments.discover(s, limit=9, min_similarity=1.01)
    assert segments.discover(s, limit=9, max_columns=4) == []
    assert len(segments.discover(s, limit=9, max_columns=5)) == 1


def test_fold_adds_fixed_input_changes():
    s = planted()
    a, b = segments.fold(s, 0, 1)
    x = torch.tensor([[0.3, -0.7]])
    r = s.per_expert
    expected = (x @ s.down[:r].T) @ s.up[:r] + (x @ s.down[r : 2 * r].T) @ s.up[r : 2 * r]
    torch.testing.assert_close((x @ a.T) @ b, expected)


def test_zero_temperature_is_seed_independent(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    s = planted()
    x = torch.tensor([[[1.0, 0.1]]])
    with compute.dropout_key(1, 0):
        first = compute.derived_entries(s, x, "layer", 1)[0]
    with compute.dropout_key(2, 0):
        second = compute.derived_entries(s, x, "layer", 1)[0]
    assert torch.equal(first, second)
    assert torch.equal(first.argmax(-1), compute.derived_logits(s, x).argmax(-1))


def test_apply_variation_reaches_baked_set(denoiser, folders):
    from safetensors.torch import save_file
    from test_import_defaults import _sd

    from kilora.comfyui import nodes, patching

    save_file(_sd(), str(folders["loras"] / "style.safetensors"))
    (chain,) = nodes.KiLoRAImportLoRA().add("style.safetensors", 1.0)
    patched, _, _ = nodes.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, -1, variation=0.0
    )
    assert patching.applied(patched)["variation"] == 0.0
    reapplied, _ = patching.reapply(patched)
    assert patching.applied(reapplied)["variation"] == 0.0


def test_segment_is_an_entry_candidate(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = planted()
    x = torch.tensor([[1.0, 0.2]])
    scores = segments.entry_logits(s, x, [(0, 1)])
    assert scores.shape == (1, 3)
    torch.testing.assert_close(scores[:, :2], compute.derived_logits(s, x))
    assert torch.isfinite(scores).all()


def test_segment_draw_and_first_step(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "1")
    s = planted()
    x = torch.ones(1, 256, 2)
    pairs = [(0, 1)]

    def draw(seed):
        with compute.dropout_key(seed, 0):
            return compute.derived_entries(s, x, "layer", 1, pairs=pairs)[0]

    first = draw(4)
    assert torch.equal(first, draw(4))
    assert not torch.equal(first, draw(5))
    assert (first.sum(-1) == 2).any()
    monkeypatch.setenv("KILORA_FUSED", "0")
    entry = torch.ones(1, 1, 2)
    unit = x[:, :1]
    with torch.no_grad():
        output = compute._run_input(
            unit, torch.zeros_like(unit), [s], [0], 1, [None], "layer", entry=entry
        )
    down, up = segments.fold(s, 0, 1)
    expected = ((unit @ down.T) @ up) * s.scale
    torch.testing.assert_close(output, expected)


def test_fused_segment_entry_matches_eager(monkeypatch):
    import os

    import pytest

    pytest.importorskip("triton")
    from test_fused_input import DEVICE, _imported, _inputs

    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter or GPU")
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    s = _imported(3, experts=3, n_in=24, gate_mode="none")
    x, y = _inputs(4, batch=1, tokens=5, n_in=24)
    entry = torch.zeros((*x.shape[:-1], 3), device=x.device)
    entry[..., :2] = 1
    with torch.no_grad():
        expected = compute._run_input(x, y, [s], [0], 2, [None], "layer", entry=entry)
        from kilora import fused

        actual = fused.run_fused(
            y, [s], 2, False, "layer", [None], x=x, name="layer#input0", entry=entry
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_first_entry_runs_segment_and_carries_its_mass(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    monkeypatch.setenv("KILORA_DERIVED_CARRY", "1")
    s = dataclasses.replace(planted(), path_width=1, key="joint")
    calls = []

    def prepare(_):
        calls.append(1)
        return [(0, 1)]

    def logits(_, x, pairs):
        assert pairs == [(0, 1)]
        return x.new_tensor([0.0, 0.0, 10.0]).expand(*x.shape[:-1], 3)

    monkeypatch.setattr(segments, "prepare", prepare)
    monkeypatch.setattr(segments, "entry_logits", logits)
    x = torch.tensor([[[1.0, 0.5]]])
    with torch.no_grad(), compute.derived_call():
        result = compute.run_layer(x, torch.zeros_like(x), [s], 1, index=0)
        record = next(iter(compute._DERIVED_CARRY[-1].values()))
        torch.testing.assert_close(record["gates"], torch.full_like(record["gates"], 0.5))
        compute.run_layer(x, torch.zeros_like(x), [s], 1, index=1)
    down, up = segments.fold(s, 0, 1)
    torch.testing.assert_close(result, ((x @ down.T) @ up) * s.scale)
    assert len(calls) == 1


def test_operator_metrics_coherence_gain_and_residual():
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        up=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-0.001, 0.0]]),
    )
    norms, gain, match, triple = segments.operator_metrics(s, 1.0)
    torch.testing.assert_close(norms, torch.tensor([1.0, 1.0, 1.0, 0.001]))
    assert gain[3] < 0.01
    assert match[0, 3] > 0 and match[3, 2] > 0
    assert triple[0, 1, 2] > 0.05
    assert triple[0, 3, 2] < 0
    assert segments.shortcut_residual(match, gain, 0, 1, 2) == 0


def test_operator_metrics_match_dense_operators():
    torch.manual_seed(12)
    s = dataclasses.replace(planted(), down=torch.randn(6, 4), up=torch.randn(6, 4), per_expert=2)
    norms, gains, match, triple = segments.operator_metrics(s, 3.0)
    operators = [s.up[i : i + 2].T @ s.down[i : i + 2] for i in range(0, 6, 2)]
    for a in range(3):
        torch.testing.assert_close(norms[a], operators[a].norm())
        torch.testing.assert_close(gains[a], norms[a] / 3.0)
        for b in range(3):
            expected = (operators[b] @ operators[a]).norm() / (norms[a] * 3.0)
            torch.testing.assert_close(match[a, b], expected)
            for c in range(3):
                expected = torch.trace(operators[c] @ operators[b] @ operators[a]) / (
                    norms[a] * norms[b] * norms[c]
                )
                torch.testing.assert_close(triple[a, b, c], expected, atol=1e-6, rtol=1e-5)


def test_operator_decisions_identify_relay_and_coherence():
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        up=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-0.001, 0.0]]),
    )
    accepted, skipped = segments.operator_decisions(s, 1.0)
    assert accepted[0, 1, 2]
    assert not accepted[0, 3, 2]
    assert skipped[0, 3, 2]
    assert not skipped[0, 1, 2]


def test_discovery_requires_matching_coherent_shortcut(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        up=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
    )
    candidates = segments.discover(s, limit=9, min_similarity=1.0, host_norm=1.0)
    assert (0, 1) in candidates
    accepted, _ = segments.operator_decisions(s, 1.0)
    assert not accepted[0, 2, 1]


def test_operator_metrics_do_not_materialize_full_operators():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoFullProduct(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                assert result.shape[-2:] != (7, 7)
            return result

    s = dataclasses.replace(planted(), down=torch.randn(6, 7), up=torch.randn(6, 7), per_expert=2)
    with NoFullProduct():
        segments.operator_metrics(s, 1.0)


def test_large_operator_graph_prunes_pairs():
    torch.manual_seed(17)
    s = dataclasses.replace(planted(), down=torch.randn(11, 3), up=torch.randn(11, 3))
    _, _, _, triples = segments.operator_metrics(s, 1.0)
    assert isinstance(triples, dict)
    assert len(triples) == 10
    assert all(values.shape == (11,) for values in triples.values())
    accepted, _ = segments.operator_decisions(s, 1.0)
    assert set(accepted) == set(triples)


def test_host_norm_does_not_change_routing_scores(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "transmission")
    s = planted()
    x = torch.tensor([[[1.0, 0.3]]])
    with_host = dataclasses.replace(s, host_weight_norm=42.0)
    torch.testing.assert_close(compute.derived_logits(s, x), compute.derived_logits(with_host, x))
    torch.testing.assert_close(compute.share_transition(s), compute.share_transition(with_host))


def test_prepare_requires_base_norm_and_filters_triples(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_ROWS", "1")
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        up=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    assert segments.prepare(s) == []
    assert (0, 1) in segments.prepare(dataclasses.replace(s, host_weight_norm=1.0))
    monkeypatch.setenv("KILORA_SEGMENT_MIN_TRIPLE", "1.1")
    assert segments.prepare(dataclasses.replace(s, host_weight_norm=1.0)) == []


@pytest.mark.parametrize("fixed_read", [False, True])
def test_transparent_relay_is_skipped_on_a_discrete_path(monkeypatch, fixed_read):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1" if fixed_read else "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    monkeypatch.setenv("KILORA_DERIVED_LIFT_AFTER", "0")
    monkeypatch.setenv("KILORA_FUSED", "0")
    s = dataclasses.replace(
        planted(),
        down=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        up=torch.tensor([[1.0, 0.0], [0.001, 0.0], [1.0, 0.0]]),
        keys_in=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]),
        host_weight_norm=1.0,
    )
    transition = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(compute, "share_transition", lambda _: transition)
    x = torch.tensor([[[1.0, 0.0]]])
    y0 = torch.zeros_like(x)
    entry = torch.tensor([[[1.0, 0.0, 0.0]]])

    def run():
        with torch.no_grad():
            return compute._run_input(x, y0, [s], [0], 3, [None], "relay", entry=entry)

    monkeypatch.setenv("KILORA_SEGMENT_SKIP_GAIN", "0")
    baseline = run()
    if fixed_read:
        torch.testing.assert_close(baseline, torch.tensor([[[2.001, 0.0]]]))
    else:
        assert baseline.norm() <= 2 * (x @ s.down[:1].t() @ s.up[:1]).norm() + 1e-6
    monkeypatch.setenv("KILORA_SEGMENT_SKIP_GAIN", "0.01")
    fast = run()
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
    plain = run()
    if fixed_read:
        torch.testing.assert_close(fast, torch.tensor([[[2.0, 0.0]]]))
    else:
        assert fast.norm() <= 2 * (x @ s.down[:1].t() @ s.up[:1]).norm() + 1e-6
    assert torch.equal(fast, plain)
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
    with torch.no_grad():
        dispatched = compute._run_input_fused(x, y0, [s], [0], 3, [None], "relay", entry=entry)
    assert torch.equal(dispatched, fast)


def test_segment_entry_falls_back_from_batched_and_replays_fold(monkeypatch):
    from kilora import route_freeze

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    monkeypatch.setattr(segments, "prepare", lambda _: [(0, 1)])
    monkeypatch.setattr(
        segments,
        "entry_logits",
        lambda s, x, pairs: x.new_tensor([0.0, 0.0, 10.0]).expand(*x.shape[:-1], 3),
    )
    s = dataclasses.replace(planted(), path_width=1, route_freeze=0.4)
    x = torch.tensor([[[1.0, 0.5], [0.2, 1.0]]])
    with torch.no_grad(), compute.dropout_key(1, 1):
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
        expected = compute.run_layer(x, x, [s], 2, target="layer")
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", "1")
        actual = compute.run_layer(x, x, [s], 2, target="layer")
    torch.testing.assert_close(actual, expected)
    with torch.no_grad(), route_freeze.freezing({}) as cache, compute.sampling_progress(0.5):
        with compute.dropout_key(1, 1), compute.derived_call():
            recorded = compute.run_layer(x, x, [s], 2, target="layer")
        assert ("derived_macro", "layer", 0) in cache
        with compute.dropout_key(2, 2), compute.derived_call():
            replayed = compute.run_layer(x, x, [s], 2, target="layer")
    torch.testing.assert_close(recorded, expected)
    torch.testing.assert_close(replayed, recorded)


def test_segment_candidates_support_conflict_prior(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_EXCLUSION", "1")
    monkeypatch.setenv("KILORA_DERIVED_ENTRY_TEMP", "0")
    s = planted()
    x = torch.tensor([[[1.0, 0.5]]])
    entries = compute.derived_entries(s, x, "layer", 2, pairs=[(0, 1)])
    assert len(entries) == 2
    assert all(entry.shape[-1] == 2 for entry in entries)


def test_segments_off_disables_relay_skips(monkeypatch):
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    s = dataclasses.replace(planted(), host_weight_norm=1.0, iteration="accumulate")
    assert segments.prepare(s) == []
    assert segments.skip_decisions(s) is None


def test_frozen_segments_are_only_offered_in_first_block(monkeypatch):
    from kilora import route_freeze

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    calls = []

    def prepare(s):
        calls.append(s)
        return [(0, 1)]

    monkeypatch.setattr(segments, "prepare", prepare)
    s = dataclasses.replace(planted(), key="joint", route_freeze=0.4)
    x = torch.ones(1, 2, 2)
    with (
        torch.no_grad(),
        route_freeze.freezing({}),
        compute.sampling_progress(0.5),
        compute.derived_call(),
    ):
        compute.run_layer(x, x, [s], 2, target="first", index=0)
        compute.run_layer(x, x, [s], 2, target="second", index=1)
    assert len(calls) == 1
