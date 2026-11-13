# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.compute import LayerSet
from kilora.path_graph import PathGraphCache, PathNode, build_path_graph, expected_contribution


def _layer(matrix):
    return LayerSet(
        router=torch.eye(2), keys_in=torch.eye(2)[:, None, :], graph_transition=torch.tensor(matrix)
    )


def test_graph_depth_cross_and_zero_weights():
    a = _layer([[0.25, 0.75], [0.0, 1.0]])
    b = _layer([[1.0, 0.0], [0.5, 0.5]])
    hooks = [("a", a, 2), ("b", b, 1)]
    graph = build_path_graph(hooks, {("a", "b"): torch.tensor([0.2, 0.8])})
    assert len(graph) == 6
    assert graph[1].predecessor == PathNode("a", 0, 0)
    assert graph[1].successor == PathNode("a", 1, 1)
    assert graph[2].weight == 0.0
    assert graph[-1].successor == PathNode("b", 0, 1)
    assert len(build_path_graph(hooks)) == 4


def test_graph_cache_and_dropout_expectation():
    layer = _layer([[1.0, 0.0], [0.0, 1.0]])
    cache = PathGraphCache()
    hooks = [("a", layer, 2)]
    first = cache.get(1, hooks)
    layer.graph_transition = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert cache.get(1, hooks) is first
    second = cache.get(2, hooks)
    assert second[0].weight == 0.0
    p = expected_contribution(first[0], torch.tensor([0.0, 2.0]), torch.tensor(1.0))
    assert torch.allclose(p, torch.tensor(0.625))
    assert expected_contribution(first[1], torch.tensor([0.0]), torch.tensor(1.0)) == 0
    assert expected_contribution(first[0], torch.tensor([100.0]), torch.tensor(1.0)) < 1e-20


def test_graph_builds_transition_from_keys_when_uncached():
    layer = LayerSet(router=torch.eye(2), keys_in=torch.eye(2)[:, None, :])
    graph = build_path_graph([("a", layer, 2)])
    assert [edge.weight for edge in graph] == [1.0, 0.0, 0.0, 1.0]


def test_incremental_graph_tracks_node_changes_and_checkout():
    generator = torch.Generator().manual_seed(19)
    kin = torch.randn(3, 2, 5, generator=generator)
    kout = torch.randn(3, 2, 5, generator=generator)

    def layer(keys):
        return LayerSet(router=torch.eye(5), keys_in=kin, keys_out=keys)

    cache = PathGraphCache()
    versions = {("a", e): 1 for e in range(3)}
    hooks = [("a", layer(kout), 2)]
    graph = cache.get("config", hooks, node_versions=versions)
    assert graph == build_path_graph(hooks)
    assert cache.stats == (3, 3, 9, 9)
    branch = kout.clone()
    branch[0] = torch.randn(2, 5, generator=generator)
    versions[("a", 0)] += 1
    hooks = [("a", layer(branch), 2)]
    graph = cache.get("config", hooks, node_versions=versions)
    assert graph == build_path_graph(hooks)
    assert cache.stats == (1, 3, 5, 9)
    assert graph[4] is cache.get("config", hooks, node_versions=versions)[4]
    assert cache.stats == (0, 3, 0, 9)
    checkout = branch.clone()
    checkout[0] = kout[0]
    checkout[1] = torch.randn(2, 5, generator=generator)
    versions[("a", 0)] += 1
    versions[("a", 1)] += 1
    hooks = [("a", layer(checkout), 2)]
    assert cache.get("config", hooks, node_versions=versions) == build_path_graph(hooks)
    assert cache.stats == (2, 3, 8, 9)


def test_incremental_input_change_renormalizes_other_rows():
    generator = torch.Generator().manual_seed(71)
    kin = torch.randn(3, 2, 5, generator=generator)
    kout = torch.randn(3, 2, 5, generator=generator)
    versions = {("a", e): 1 for e in range(3)}
    cache = PathGraphCache()
    hooks = [("a", LayerSet(router=torch.eye(5), keys_in=kin, keys_out=kout), 2)]
    before = cache.get("config", hooks, node_versions=versions)
    changed = kin.clone()
    changed[0] = torch.randn(2, 5, generator=generator)
    versions[("a", 0)] += 1
    hooks = [("a", LayerSet(router=torch.eye(5), keys_in=changed, keys_out=kout), 2)]
    after = cache.get("config", hooks, node_versions=versions)
    assert after == build_path_graph(hooks)
    assert after[4].weight != before[4].weight
    assert cache.stats == (1, 3, 9, 9)


def test_cross_hook_cache_counts_only_edges_touching_changed_node():
    keys = torch.eye(3)[:, None, :]
    hooks = [(name, LayerSet(router=torch.eye(3), keys_in=keys), 1) for name in ("a", "b")]
    cross = {("a", "b"): torch.ones(3)}
    versions = {(name, e): 1 for name in ("a", "b") for e in range(3)}
    cache = PathGraphCache()
    cache.get("config", hooks, cross, versions)
    versions[("a", 1)] += 1
    assert cache.get("config", hooks, cross, versions) == build_path_graph(hooks, cross)
    assert cache.stats == (1, 6, 1, 3)
