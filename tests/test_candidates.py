# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.candidates import expert_summaries, top_candidates
from kilora.compute import norm_energy, token_gate
from kilora.multi_import import _content_keys


def test_content_keys_and_planted_expert():
    generator = torch.Generator().manual_seed(21)
    up = torch.randn(12, 1, 3, 8, generator=generator)
    router = torch.eye(8)
    keys = _content_keys(up, router, [{"name": str(i)} for i in range(12)], "layer")
    summary = expert_summaries(keys)
    chosen = top_candidates(summary, summary, 3)
    assert (chosen == torch.arange(12)[:, None]).any(dim=-1).all()
    assert summary.shape == (12, 8)


def test_candidate_shapes_and_all_experts():
    summary = torch.eye(4)
    query = torch.tensor([[[1.0, 3.0, 2.0, 0.0], [0.0, 0.0, 0.0, 5.0]]])
    assert top_candidates(summary, query, 2).tolist() == [[[1, 2], [3, 2]]]
    for count in (4, 8):
        assert torch.equal(
            top_candidates(summary, query, count).sort(-1).values,
            torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]]),
        )
    assert top_candidates(summary, query, 0).shape == (1, 2, 0)
    assert top_candidates(summary, query[0, 0], 1).shape == (1,)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        expert_summaries(torch.zeros(2, 2, 3))
    with pytest.raises(ValueError):
        top_candidates(torch.zeros(2, 3), torch.zeros(4), 1)
    with pytest.raises(ValueError):
        top_candidates(torch.zeros(2, 3), torch.zeros(3), -1)


def test_hit_rate_against_exact_key_scores():
    generator = torch.Generator().manual_seed(313)
    keys = torch.randn(128, 1, 32, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(256, 32, generator=generator)
    candidates = top_candidates(expert_summaries(keys), query, 8)
    exact = norm_energy(query, keys, None).topk(8, dim=-1).indices
    hit_rate = (candidates[..., None] == exact[:, None, :]).any(dim=-1).float().mean().item()
    print(f"stage-1 hit rate against exact key top-8 (128 experts): {hit_rate:.4f}", flush=True)
    assert hit_rate >= 0.95


def test_routing_without_candidate_step_matches_base():
    logits = torch.tensor([[1.0, 0.5, -1.0], [0.0, 0.0, 1.0]])
    expected = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.0, 1.0]])
    assert torch.equal(token_gate(logits), expected)


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("select", ["sparsemax", "top1"])
def test_candidate_gate_matches_restricted_selection(scope, select):
    import dataclasses

    from kilora import compute

    keys = torch.eye(4).unsqueeze(1)
    query = torch.tensor([[[2.0, 1.0, -2.0, 0.3], [0.2, 3.0, 0.9, -1.0]]])
    region = torch.tensor([[[1.0], [0.25]]])
    s = compute.LayerSet(
        router=torch.eye(4), keys_in=keys, candidates=2, routing_scope=scope, routing_select=select
    )
    logits = compute.norm_energy(query, keys, None)
    query_at_scope = compute.region_mean(query, region) if scope == "tensor" else query
    indices = top_candidates(keys[:, 0], query_at_scope, 2)
    expected_logits = compute.region_mean(logits, region) if scope == "tensor" else logits
    chosen_logits = expected_logits.gather(-1, indices)
    expected = torch.zeros_like(expected_logits).scatter(
        -1, indices, compute._select(chosen_logits, select)
    )
    expected = expected.expand_as(logits)
    actual = compute._gates(query, [s], False, 0.0, regions=[region])[0]
    torch.testing.assert_close(actual, expected)
    baseline = compute._gates(
        query, [dataclasses.replace(s, candidates=0)], False, 0.0, regions=[region]
    )[0]
    for count in (4, 9):
        assert torch.equal(
            baseline,
            compute._gates(
                query, [dataclasses.replace(s, candidates=count)], False, 0.0, regions=[region]
            )[0],
        )


@pytest.mark.parametrize("select", ["sparsemax", "top1"])
@pytest.mark.parametrize("with_region", [False, True])
def test_tensor_candidate_compact_matches_masked(select, with_region):
    from kilora import compute

    s = compute.LayerSet(
        router=torch.eye(4),
        keys_in=torch.eye(4).unsqueeze(1),
        candidates=2,
        routing_scope="tensor",
        routing_select=select,
    )
    x = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 1.0]]] * 2)
    region = torch.tensor([[[1.0], [0.5]]] * 2) if with_region else None
    keys = [compute._set_keys(s)]
    queries = [torch.nn.functional.linear(x.float(), s.router.float())]
    compact = compute._route(queries, [s], keys, None, False, 0.0, [region])[0]
    mask = compute._candidate_mask(s, queries[0], keys[0][0], region)
    masked = compute._gate(
        s.tau * compute.norm_energy(queries[0], keys[0][0], s.rows_in), s, 0.0, region, mask
    )
    assert torch.equal(compact, masked)


@pytest.mark.parametrize("select", ["sparsemax", "top1"])
def test_tensor_candidate_compact_random_matches_masked(select):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute

    s = dataclasses.replace(
        _sets(1, 16, 4, 3, 8, seed=22)[0],
        candidates=2,
        routing_scope="tensor",
        routing_select=select,
    )
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(42))
    keys = compute.derived_keys(s)[0]
    compact = compute._mixed_route(x, [s], None, None, False, 0.0, [None], True)[0]
    mask = compute._candidate_mask(s, x, keys)
    masked = compute._gate(compute.derived_logits(s, x), s, 0.0, candidates=mask)
    torch.testing.assert_close(compact, masked, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("select", ["sparsemax", "top1"])
@pytest.mark.parametrize("score", ["derived", "read_share"])
def test_import_scores_restricted_to_candidates(scope, select, score, monkeypatch):
    from kilora import compute

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    down = torch.eye(4)
    s = compute.LayerSet(
        router=torch.eye(4),
        keys_in=torch.eye(4).unsqueeze(1),
        down=down,
        up=down,
        per_expert=1,
        form="input",
        candidates=2,
        import_score=score,
        routing_scope=scope,
        routing_select=select,
    )
    x = torch.tensor([[[1.0, 0.2, -0.5, 0.8], [0.3, 1.0, 0.4, -0.6]]])
    region = torch.tensor([[[1.0], [0.5]]])
    coef = x @ down.T
    actual = compute._import_gates([s], [coef], [region], x)[0]
    if score == "derived":
        logits = compute.derived_logits(s, x)
    else:
        energy = compute.read_energy(s, coef)
        logits = energy / (energy.sum(-1, keepdim=True) + 1e-12)
    summary = compute.derived_keys(s)[0][:, 0, :]
    scoped_x = compute.region_mean(x, region) if scope == "tensor" else x
    indices = top_candidates(summary, scoped_x, 2)
    scoped_logits = compute.region_mean(logits, region) if scope == "tensor" else logits
    expected = torch.zeros_like(scoped_logits).scatter(
        -1, indices, compute._select(scoped_logits.gather(-1, indices), select)
    )
    torch.testing.assert_close(actual, expected.expand_as(logits))


def test_candidate_loop_and_lowrank_match(monkeypatch):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute

    s = dataclasses.replace(_sets(1, 16, 4, 3, 8, seed=22)[0], candidates=2)
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        loop = compute.run_layer(x, x, [s], 2, target="layer")
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, x, [s], 2, target="layer")
    torch.testing.assert_close(loop, fast)


@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"path_beam": True},
        {"path_merge": True},
        {"path_starts": "diverse"},
        {"path_prior_min": 0.1},
    ],
)
def test_path_walks_stay_in_candidates(scope, options, monkeypatch):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute

    s = dataclasses.replace(
        _sets(1, 16, 4, 3, 8, seed=19)[0],
        candidates=2,
        routing_scope=scope,
        iteration="path",
        path_width=3,
        **options,
    )
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(12))
    original = compute._run_input
    visited = []

    def check(x, y0, lora, once, depth, regions, target, **kwargs):
        stats = dict(kwargs.pop("stats"))
        capture = stats["capture_step"]

        def step(state, gates, zs, output, k):
            for gate, item, region in zip(gates, lora, regions, strict=True):
                mask = compute._candidate_mask(item, state, compute.derived_keys(item)[0], region)
                assert not (gate.bool() & ~mask).any()
            visited.append(True)
            return capture(state, gates, zs, output, k)

        stats["capture_step"] = step
        return original(x, y0, lora, once, depth, regions, target, stats=stats, **kwargs)

    monkeypatch.setattr(compute, "_run_input", check)
    with torch.no_grad():
        result = compute.run_layer(x, x, [s], 3, target="layer")
    assert visited and torch.isfinite(result).all()


def test_disabled_matches_base_tip_and_replays():
    import dataclasses

    from kilora import compute

    x = torch.tensor([[[4.0, 8.0, 12.0, 16.0], [8.0, 4.0, -4.0, -8.0]]])
    s = compute.LayerSet(
        router=torch.eye(4),
        keys_in=torch.eye(4).unsqueeze(1),
        down=torch.eye(4),
        up=torch.eye(4),
        per_expert=1,
        tau=0.0,
    )
    expected = torch.tensor([[[5.0, 10.0, 15.0, 20.0], [10.0, 5.0, -5.0, -10.0]]])
    with torch.no_grad():
        for summary in ("keys", "read_map"):
            for count in (0, 4, 8):
                actual = compute.run_layer(
                    x, x, [dataclasses.replace(s, candidates=count, candidate_summary=summary)], 1
                )
                assert torch.equal(actual, expected)
        routed = dataclasses.replace(s, candidates=2)
        assert torch.equal(
            compute.run_layer(x, x, [routed], 2), compute.run_layer(x, x, [routed], 2)
        )


def test_candidates_setting_round_trip(model, tmp_path):
    from test_modfile import trained

    from kilora import load_mod

    adapter = trained(model)
    adapter.settings.candidates = 2
    adapter.settings.candidate_summary = "read_map"
    path = tmp_path / "candidates.kmod"
    adapter.save(path, id="candidates")
    loaded = load_mod(path).settings
    assert loaded.candidates == 2
    assert loaded.candidate_summary == "read_map"


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_derived_walk_entries_and_routes_respect_candidates(scope, monkeypatch):
    from kilora import compute

    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_DERIVED_BATCHED", "0")
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    s = compute.LayerSet(
        router=torch.eye(4),
        keys_in=torch.eye(4).unsqueeze(1),
        down=torch.eye(4),
        up=torch.eye(4),
        per_expert=1,
        form="input",
        import_score="derived",
        candidates=2,
        path_width=3,
        routing_scope=scope,
    )
    x = torch.tensor([[[1.0, 0.2, -0.5, 0.8], [0.3, 1.0, 0.4, -0.6]]])
    region = torch.tensor([[[1.0], [0.5]]])
    mask = compute._candidate_mask(s, x, compute.derived_keys(s)[0], region)
    entries = compute.derived_entries(s, x, "test", 3, region)
    assert len(entries) == 2
    assert all(not (entry.bool() & ~mask).any() for entry in entries)
    routes = []
    with torch.no_grad(), compute.recording_shares(lambda _, gates: routes.append(gates[0])):
        y = compute.run_layer(x, torch.zeros_like(x), [s], 2, regions=[region], target="test")
    assert routes and torch.isfinite(y).all()
    assert all(not (gate.bool() & ~mask).any() for gate in routes)


def test_active_candidates_do_not_replay_frozen_token_averages(monkeypatch):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute, route_freeze

    s = dataclasses.replace(_sets(1, 16, 4, 3, 8, seed=2)[0], candidates=2, route_freeze=0.1)
    x = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(3))
    cache = {}
    with torch.no_grad(), route_freeze.freezing(cache):
        first = compute.run_layer(x, x, [s], 2, target="layer", t=0.9)
        second = compute.run_layer(x, x, [s], 2, target="layer", t=0.9)
    assert not cache
    assert torch.equal(first, second)


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_shared_beam_respects_each_sets_candidates(scope, monkeypatch):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute

    sets = _sets(2, 16, 4, 3, 8, seed=17)
    sets = [
        dataclasses.replace(
            s,
            router=sets[0].router,
            candidates=2,
            routing_scope=scope,
            iteration="path",
            path_width=2,
            path_beam=True,
        )
        for s in sets
    ]
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(13))
    original = compute._run_input
    visited = []

    def check(x, y0, lora, once, depth, regions, target, **kwargs):
        stats = dict(kwargs.pop("stats"))
        capture = stats["capture_step"]

        def step(state, gates, zs, output, k):
            for gate, item, region in zip(gates, lora, regions, strict=True):
                mask = compute._candidate_mask(item, state, compute.derived_keys(item)[0], region)
                assert not (gate.bool() & ~mask).any()
            visited.append(True)
            return capture(state, gates, zs, output, k)

        stats["capture_step"] = step
        return original(x, y0, lora, once, depth, regions, target, stats=stats, **kwargs)

    monkeypatch.setattr(compute, "_run_input", check)
    with torch.no_grad():
        result = compute.run_layer(x, x, sets, 3, target="layer", shared=True)
    assert visited and torch.isfinite(result).all()


@pytest.mark.parametrize("form", ["input", "state"])
def test_read_map_selects_planted_expert(form):
    from kilora import compute

    keys = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    down = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    keys[1] = torch.tensor([[0.0, 0.0, 1.0]])
    s = compute.LayerSet(
        router=torch.eye(3),
        keys_in=keys,
        down=down,
        up=torch.ones(3, 3),
        form=form,
        candidates=1,
        candidate_summary="read_map",
    )
    mask = compute._candidate_mask(s, torch.tensor([[[2.0, 0.1, 0.0]]]), keys)
    assert mask.tolist() == [[[False, True, False]]]


def test_multirow_uses_best_valid_row():
    from kilora import compute

    keys = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [9.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    s = compute.LayerSet(
        router=torch.eye(3), keys_in=keys, candidates=1, rows_in=torch.tensor([2, 1, 1])
    )
    assert compute._candidate_mask(s, torch.tensor([[[-2.0, 0.0, 0.0]]]), keys).tolist() == [
        [[True, False, False]]
    ]


@pytest.mark.parametrize("summary", ["keys", "read_map"])
def test_summary_off_matches_base_and_fast_path(summary, monkeypatch):
    import dataclasses

    from test_lowrank_depth import _sets

    from kilora import compute

    base = _sets(1, 16, 4, 3, 8, seed=22)[0]
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        baseline = compute.run_layer(x, x, [base], 2, target="layer")
        off = compute.run_layer(
            x, x, [dataclasses.replace(base, candidate_summary=summary)], 2, target="layer"
        )
        assert torch.equal(baseline, off)
        active = dataclasses.replace(base, candidates=2, candidate_summary=summary)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        loop = compute.run_layer(x, x, [active], 2, target="layer")
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = compute.run_layer(x, x, [active], 2, target="layer")
        torch.testing.assert_close(loop, fast)
