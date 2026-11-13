# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class _Cluster:
    centroid: Tensor
    members: set[str] = field(default_factory=set)


def _vector(value: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
    if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return value


def _cosine(a: Tensor, b: Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), eps=1e-12)[0])


class StreamingCodebook:
    def __init__(self, *, seed: int = 0, balance: float = 0.1):
        if not math.isfinite(balance) or balance < 0:
            raise ValueError("balance must be nonnegative and finite")
        self.seed = int(seed)
        self.balance = float(balance)
        self._vectors: dict[str, Tensor] = {}
        self._clusters: dict[int, _Cluster] = {}
        self._assignment: dict[str, int] = {}
        self._next_id = 0
        self._removed = None
        self.alignment: Tensor | None = None

    @property
    def assignments(self) -> dict[str, int]:
        return self._assignment.copy()

    @property
    def centroids(self) -> dict[int, Tensor]:
        return {key: value.centroid.clone() for key, value in self._clusters.items()}

    @property
    def sizes(self) -> dict[int, int]:
        return {key: len(value.members) for key, value in self._clusters.items()}

    def add_many(self, entries: Iterable[tuple[str, Tensor]]) -> None:
        entries = sorted(entries, key=lambda item: item[0])
        keys = [key for key, _ in entries]
        if len(set(keys)) != len(keys) or any(key in self._vectors for key in keys):
            raise ValueError("duplicate digest")
        prepared = [(key, _vector(value, "summary")) for key, value in entries]
        width = next(iter(self._vectors.values())).numel() if self._vectors else None
        for _, value in prepared:
            if width is None:
                width = value.numel()
            if value.numel() != width:
                raise ValueError("summary widths must match within a family")
        for key, value in prepared:
            self.add(key, value)

    def add(self, key: str, summary: Tensor) -> int:
        if key in self._vectors:
            raise ValueError("duplicate digest")
        value = _vector(summary, "summary")
        if self._vectors and value.shape != next(iter(self._vectors.values())).shape:
            raise ValueError("summary widths must match within a family")
        if self._removed is not None:
            removed_key, removed_value, clusters, assignments, next_id = self._removed
            self._removed = None
            if key == removed_key and torch.equal(value, removed_value):
                self._vectors[key] = value
                self._clusters, self._assignment, self._next_id = clusters, assignments, next_id
                self.alignment = None
                return self._assignment[key]
        self._vectors[key] = value
        self.alignment = None
        if not self._clusters:
            self._clusters[0] = _Cluster(value.clone(), {key})
            self._assignment[key] = 0
            self._next_id = 1
            return 0
        mean = len(self._vectors) / len(self._clusters)
        ids = sorted(self._clusters)
        centers = torch.stack([self._clusters[i].centroid for i in ids])
        scores = F.normalize(centers, dim=1, eps=1e-12) @ F.normalize(value, dim=0, eps=1e-12)
        scores -= torch.tensor([self.balance * len(self._clusters[i].members) / mean for i in ids])
        cluster_id = ids[int(scores.argmax())]
        cluster = self._clusters[cluster_id]
        count = len(cluster.members)
        cluster.centroid += (value - cluster.centroid) / (count + 1)
        cluster.members.add(key)
        self._assignment[key] = cluster_id
        self._rebalance()
        return self._assignment[key]

    def remove(self, key: str) -> None:
        if key not in self._vectors:
            raise KeyError(key)
        clusters = {
            i: _Cluster(c.centroid.clone(), c.members.copy()) for i, c in self._clusters.items()
        }
        value = self._vectors.pop(key)
        self._removed = (key, value, clusters, self._assignment.copy(), self._next_id)
        self.alignment = None
        cluster_id = self._assignment.pop(key)
        cluster = self._clusters[cluster_id]
        count = len(cluster.members)
        cluster.members.remove(key)
        if count == 1:
            del self._clusters[cluster_id]
        else:
            cluster.centroid = (cluster.centroid * count - value) / (count - 1)
        self._merge_small()
        self._rebalance()

    def _merge_small(self) -> None:
        target = math.ceil(math.sqrt(len(self._vectors)))
        while len(self._clusters) > max(1, target - 1):
            small = [i for i, c in self._clusters.items() if len(c.members) < 2]
            if len(self._clusters) < 2 or (not small and len(self._clusters) <= target + 1):
                return
            source = (
                min(small)
                if small
                else min(self._clusters, key=lambda i: (len(self._clusters[i].members), i))
            )
            dest = max(
                sorted(i for i in self._clusters if i != source),
                key=lambda i: _cosine(self._clusters[source].centroid, self._clusters[i].centroid),
            )
            self._clusters[dest].members.update(self._clusters.pop(source).members)
            self._refresh(dest)

    def _refresh(self, cluster_id: int) -> None:
        cluster = self._clusters[cluster_id]
        keys = sorted(cluster.members)
        cluster.centroid = torch.stack([self._vectors[k] for k in keys]).mean(dim=0)
        for key in keys:
            self._assignment[key] = cluster_id

    def _split(self, cluster_id: int) -> None:
        cluster = self._clusters[cluster_id]
        keys = sorted(cluster.members)
        if len(keys) < 2:
            return
        first = keys[self.seed % len(keys)]
        matrix = torch.stack([self._vectors[k] for k in keys])
        normalized = F.normalize(matrix, dim=1, eps=1e-12)
        scores = normalized @ F.normalize(self._vectors[first], dim=0, eps=1e-12)
        second = keys[int(scores.argmin())]
        if second == first:
            second = keys[0] if first != keys[0] else keys[1]
        centers = [self._vectors[first].clone(), self._vectors[second].clone()]
        groups = None
        for _ in range(12):
            center_matrix = F.normalize(torch.stack(centers), dim=1, eps=1e-12)
            sides = (normalized @ center_matrix.T).argmax(dim=1).tolist()
            next_groups = [set(), set()]
            for key, side in zip(keys, sides, strict=True):
                next_groups[side].add(key)
            if not all(next_groups):
                next_groups = [set(keys[: len(keys) // 2]), set(keys[len(keys) // 2 :])]
            if next_groups == groups:
                break
            groups = next_groups
            centers = [torch.stack([self._vectors[k] for k in sorted(g)]).mean(0) for g in groups]
        new_id = self._next_id
        self._next_id += 1
        cluster.members = next_groups[0]
        self._clusters[new_id] = _Cluster(centers[1], next_groups[1])
        self._refresh(cluster_id)
        self._refresh(new_id)

    def _rebalance(self) -> None:
        n = len(self._vectors)
        if not n:
            return
        target = math.ceil(math.sqrt(n))
        while len(self._clusters) < target:
            largest = min(self._clusters, key=lambda i: (-len(self._clusters[i].members), i))
            if len(self._clusters[largest].members) < 2:
                break
            self._split(largest)
        if len(self._clusters) < target + 1:
            largest = min(self._clusters, key=lambda i: (-len(self._clusters[i].members), i))
            if len(self._clusters[largest].members) > 2 * n / len(self._clusters):
                self._split(largest)
        limit = max(1, (2 * n - 1) // len(self._clusters))
        while max(map(lambda c: len(c.members), self._clusters.values())) > limit:
            source = min(self._clusters, key=lambda i: (-len(self._clusters[i].members), i))
            destinations = [i for i, c in self._clusters.items() if len(c.members) < limit]
            keys = sorted(self._clusters[source].members)
            destinations.sort()
            values = F.normalize(torch.stack([self._vectors[k] for k in keys]), dim=1, eps=1e-12)
            centers = F.normalize(
                torch.stack([self._clusters[i].centroid for i in destinations]), dim=1, eps=1e-12
            )
            best = int((values @ centers.T).reshape(-1).argmax())
            key, dest = keys[best // len(destinations)], destinations[best % len(destinations)]
            self._clusters[source].members.remove(key)
            self._clusters[dest].members.add(key)
            self._refresh(source)
            self._refresh(dest)

    def fit_alignment(self, text_features: dict[str, Tensor]) -> Tensor | None:
        keys = sorted(self._vectors.keys() & text_features.keys())
        if not keys:
            self.alignment = None
            return None
        vectors = [_vector(text_features[k], "text feature") for k in keys]
        if any(v.shape != vectors[0].shape for v in vectors):
            raise ValueError("text feature widths must match")
        x = torch.stack(vectors)
        y = torch.stack([self._clusters[self._assignment[k]].centroid for k in keys])
        ridge = 1e-3
        if x.shape[0] < x.shape[1]:
            self.alignment = x.T @ torch.linalg.solve(x @ x.T + ridge * torch.eye(x.shape[0]), y)
        else:
            self.alignment = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1]), x.T @ y)
        return self.alignment.clone()


class FamilyCodebooks:
    def __init__(self, *, seed: int = 0, balance: float = 0.1):
        self.seed = seed
        self.balance = balance
        self.families: dict[str, StreamingCodebook] = {}

    def for_family(self, family: str) -> StreamingCodebook:
        if family not in self.families:
            self.families[family] = StreamingCodebook(seed=self.seed, balance=self.balance)
        return self.families[family]

    def dump(self) -> tuple[dict, dict[str, Tensor]]:
        metadata, tensors = {}, {}
        for family, book in sorted(self.families.items()):
            metadata[family] = {"assignments": book.assignments, "next_id": book._next_id}
            for cluster_id, cluster in book._clusters.items():
                tensors[f"codebook/{family}/{cluster_id}"] = cluster.centroid.clone()
            if book.alignment is not None:
                tensors[f"codebook/{family}/alignment"] = book.alignment.clone()
        return metadata, tensors

    @classmethod
    def load(
        cls, metadata: dict, features: dict[str, Tensor], *, seed: int = 0, balance: float = 0.1
    ) -> FamilyCodebooks:
        books = cls(seed=seed, balance=balance)
        for family, state in metadata.items():
            book = books.for_family(family)
            book._next_id = state["next_id"]
            for digest, cluster_id in state["assignments"].items():
                book._vectors[digest] = _vector(features[f"{digest}/summary"], "summary")
                book._assignment[digest] = cluster_id
                if cluster_id not in book._clusters:
                    book._clusters[cluster_id] = _Cluster(
                        features[f"codebook/{family}/{cluster_id}"].float().clone()
                    )
                book._clusters[cluster_id].members.add(digest)
            alignment = features.get(f"codebook/{family}/alignment")
            if alignment is not None:
                book.alignment = alignment.clone()
        return books
