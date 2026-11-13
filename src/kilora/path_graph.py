# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .compute import _set_keys, keep_probability


@dataclass(frozen=True)
class PathNode:
    hook: str
    depth: int
    expert: int


@dataclass(frozen=True)
class PathStep:
    predecessor: PathNode
    successor: PathNode
    weight: float


def _pair_prior(kin, kout, i, j):
    cross = torch.einsum("jar,ibr->ijab", kin[j : j + 1], kout[i : i + 1])
    rows = (kout[i : i + 1].norm(dim=-1) > 0).sum(-1).clamp(min=1).float()
    return (cross.pow(2).sum((-1, -2)) / rows[:, None])[0, 0]


def _transition(layer):
    kin, kout = _set_keys(layer)
    kin, kout = kin.detach().float(), kout.detach().float()
    n = kin.shape[0]
    prior = torch.empty((n, n), device=kin.device, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            prior[i, j] = _pair_prior(kin, kout, i, j)
    prior = prior.cpu()
    sums = prior.sum(-1, keepdim=True)
    return torch.where(sums > 0, prior / sums.clamp_min(1e-30), torch.eye(n, dtype=prior.dtype))


def build_path_graph(
    hooks: list[tuple[str, Any, int]],
    cross_weights: dict[tuple[str, str], torch.Tensor] | None = None,
) -> tuple[PathStep, ...]:

    cross_weights = cross_weights or {}
    steps = []
    for name, layer, depth in hooks:
        if depth < 1:
            raise ValueError("path depth must be positive")
        matrix = layer.graph_transition
        if matrix is None:
            matrix = _transition(layer)
        matrix = matrix.detach().float().cpu()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition matrix must be square")
        for k in range(depth - 1):
            for e in range(matrix.shape[0]):
                for f in range(matrix.shape[1]):
                    steps.append(
                        PathStep(
                            PathNode(name, k, e), PathNode(name, k + 1, f), float(matrix[e, f])
                        )
                    )
    for (name, layer, depth), (next_name, next_layer, _) in zip(hooks, hooks[1:], strict=False):
        weights = cross_weights.get((name, next_name))
        if weights is None:
            continue
        n = layer.keys_in.shape[0]
        if next_layer.keys_in.shape[0] != n or weights.shape != (n,):
            raise ValueError("cross-hook weights must have one entry per shared expert")
        for e, weight in enumerate(weights):
            steps.append(
                PathStep(PathNode(name, depth - 1, e), PathNode(next_name, 0, e), float(weight))
            )
    return tuple(steps)


class PathGraphCache:
    def __init__(self) -> None:
        self._key = object()
        self._graph: tuple[PathStep, ...] = ()
        self._layers = {}
        self.stats = (0, 0, 0, 0)

    def get(self, version, hooks, cross_weights=None, node_versions=None):
        if node_versions is None:
            if self._key != version:
                self._graph = build_path_graph(hooks, cross_weights)
                self._key = version
                self._layers = {}
            return self._graph

        if any(layer.graph_transition is not None for _, layer, _ in hooks):
            self._graph = build_path_graph(hooks, cross_weights)
            self._key, self._layers = version, {}
            count = sum(layer.keys_in.shape[0] for _, layer, _ in hooks)
            self.stats = (count, count, len(self._graph), len(self._graph))
            return self._graph
        if self._key != version:
            self._layers = {}
            self._key = version
        matrices = {}
        changed_by_hook = {}
        changed_nodes = total_nodes = 0
        changed_pairs = total_pairs = 0
        for name, layer, depth in hooks:
            if depth < 1:
                raise ValueError("path depth must be positive")
            kin, kout = _set_keys(layer)
            kin, kout = kin.detach().float(), kout.detach().float()
            n = kin.shape[0]
            total_nodes += n
            previous = self._layers.get(name)
            versions = tuple(node_versions[(name, e)] for e in range(n))
            compatible = (
                previous is not None
                and previous[0].shape == kin.shape
                and previous[1].shape == kout.shape
                and previous[4] == depth
                and previous[0].device == kin.device
                and previous[1].device == kout.device
            )
            if compatible:
                old_in, old_out, old_prior, old_versions, _, old_matrix = previous
                prior = old_prior
                input_changed = [not torch.equal(kin[e], old_in[e]) for e in range(n)]
                output_changed = [not torch.equal(kout[e], old_out[e]) for e in range(n)]
                node_changed = [
                    versions[e] != old_versions[e] or input_changed[e] or output_changed[e]
                    for e in range(n)
                ]
                prior = prior.clone()
            else:
                input_changed = output_changed = node_changed = [True] * n
                prior = torch.empty((n, n), device=kin.device, dtype=torch.float32)
            changed_nodes += sum(node_changed)
            changed_by_hook[name] = node_changed
            touched = set()
            for i in range(n):
                for j in range(n):
                    total_pairs += max(depth - 1, 0)
                    if node_changed[i] or node_changed[j]:
                        prior[i, j] = _pair_prior(kin, kout, i, j)
                        touched.add((i, j))
            prior_cpu = prior.detach().float().cpu()
            identity = torch.eye(n, dtype=prior_cpu.dtype)
            if compatible:
                matrix = old_matrix.clone()
                rows = [i for i in range(n) if not torch.equal(prior[i], old_prior[i])]
            else:
                matrix = torch.empty_like(prior_cpu)
                rows = list(range(n))
            touched.update((i, j) for i in rows for j in range(n))
            changed_pairs += len(touched) * max(depth - 1, 0)
            if rows:
                selected = prior_cpu[rows]
                sums = selected.sum(-1, keepdim=True)
                matrix[rows] = torch.where(
                    sums > 0, selected / sums.clamp_min(1e-30), identity[rows]
                )
            matrices[name] = matrix
            self._layers[name] = (kin.clone(), kout.clone(), prior, versions, depth, matrix)

        old_edges = {(x.predecessor, x.successor): x for x in self._graph}
        steps = []
        for name, _layer, depth in hooks:
            matrix = matrices[name]
            for k in range(depth - 1):
                for e in range(matrix.shape[0]):
                    for f in range(matrix.shape[1]):
                        pred, succ = PathNode(name, k, e), PathNode(name, k + 1, f)
                        weight = float(matrix[e, f])
                        old = old_edges.get((pred, succ))
                        steps.append(
                            old
                            if old is not None and old.weight == weight
                            else PathStep(pred, succ, weight)
                        )
        cross_weights = cross_weights or {}
        for (name, layer, depth), (next_name, next_layer, _) in zip(hooks, hooks[1:], strict=False):
            weights = cross_weights.get((name, next_name))
            if weights is None:
                continue
            n = layer.keys_in.shape[0]
            if next_layer.keys_in.shape[0] != n or weights.shape != (n,):
                raise ValueError("cross-hook weights must have one entry per shared expert")
            for e, weight in enumerate(weights):
                pred, succ = PathNode(name, depth - 1, e), PathNode(next_name, 0, e)
                weight = float(weight)
                old = old_edges.get((pred, succ))
                steps.append(
                    old
                    if old is not None and old.weight == weight
                    else PathStep(pred, succ, weight)
                )
                total_pairs += 1
                if (
                    old is None
                    or old.weight != weight
                    or changed_by_hook[name][e]
                    or changed_by_hook[next_name][e]
                ):
                    changed_pairs += 1
        self.stats = (changed_nodes, total_nodes, changed_pairs, total_pairs)
        self._graph = tuple(steps)
        return self._graph


def expected_contribution(
    step: PathStep, carried: torch.Tensor, sigma0: torch.Tensor
) -> torch.Tensor:
    if carried.numel() == 0:
        raise ValueError("carried components must not be empty")
    return step.weight * keep_probability(carried.float(), sigma0.float()).mean()
