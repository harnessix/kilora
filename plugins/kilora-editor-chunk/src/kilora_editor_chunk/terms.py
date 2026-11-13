# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DirectionPool:
    threshold: float = 0.99
    directions: dict[str, torch.Tensor] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, value: torch.Tensor, source: str, *, merge: bool = False):
        value = value.detach().flatten().clone()
        size = float(value.norm())
        if not size:
            raise ValueError("a direction must be nonzero")
        unit = value / size
        if merge:
            for key, old in self.directions.items():
                if old.shape == unit.shape:
                    cosine = float(torch.dot(old.to(unit), unit))
                    if abs(cosine) > self.threshold:
                        return key, size * (1 if cosine >= 0 else -1)
        if name in self.directions:
            if not torch.equal(self.directions[name], unit):
                raise ValueError(f"direction id already exists: {name}")
        else:
            self.directions[name] = unit
            self.sources[name] = source
        return name, size


@dataclass
class Term:
    output: str | torch.Tensor
    input: str | torch.Tensor
    strength: float


@dataclass
class TermState:
    pool: DirectionPool
    terms: dict[tuple[str, int], list[Term]] = field(default_factory=dict)

    def add(self, hook: str, expert: int, output, input, strength: float):
        terms = self.terms.setdefault((hook, expert), [])
        if isinstance(output, str) and isinstance(input, str):
            for term in terms:
                if (
                    isinstance(term.output, str)
                    and isinstance(term.input, str)
                    and term.output == output
                    and term.input == input
                ):
                    term.strength += float(strength)
                    return
        terms.append(Term(output, input, float(strength)))

    def _direction(self, value):
        return self.pool.directions[value] if isinstance(value, str) else value

    def stacked(self, hook: str, expert: int = 0):
        terms = self.terms.get((hook, expert), [])
        if not terms:
            return torch.empty(0, 0), torch.empty(0, 0)
        up = torch.stack([self._direction(t.output) * t.strength for t in terms], dim=1)
        down = torch.stack([self._direction(t.input) for t in terms])
        return up, down

    def change_norm(self, hook: str | None = None, expert: int | None = None) -> float:
        total = 0.0
        for h, e in self.terms:
            if (hook is not None and hook != h) or (expert is not None and expert != e):
                continue
            up, down = self.stacked(h, e)
            if up.numel():
                total += float(((up.T @ up) * (down @ down.T)).sum())
        return max(0.0, total) ** 0.5

    def compact(self, budget: int):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        for key, terms in list(self.terms.items()):
            if len(terms) <= budget:
                continue
            up, down = self.stacked(*key)
            qu, ru = torch.linalg.qr(up, mode="reduced")
            qv, rv = torch.linalg.qr(down.T, mode="reduced")
            u, s, vh = torch.linalg.svd(ru @ rv.T, full_matrices=False)
            left, right = qu @ u, vh @ qv.T
            self.terms[key] = [Term(left[:, i], right[i], float(s[i])) for i in range(len(s))]

    def snapshot(self):
        return {
            key: [
                Term(
                    t.output.clone() if isinstance(t.output, torch.Tensor) else t.output,
                    t.input.clone() if isinstance(t.input, torch.Tensor) else t.input,
                    t.strength,
                )
                for t in terms
            ]
            for key, terms in self.terms.items()
        }

    def restore(self, snapshot):
        self.terms = TermState(self.pool, snapshot).snapshot()

    def delta(self, previous):
        def equal(a, b):
            if isinstance(a, str) and isinstance(b, str):
                return a == b
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.equal(a, b)
            return False

        result = {}
        for key in self.terms.keys() | previous.keys():
            unmatched = list(previous.get(key, ()))
            changes = []
            for term in self.terms.get(key, ()):
                index = next(
                    (
                        i
                        for i, old in enumerate(unmatched)
                        if equal(term.output, old.output) and equal(term.input, old.input)
                    ),
                    None,
                )
                strength = term.strength
                if index is not None:
                    strength -= unmatched.pop(index).strength
                if strength:
                    changes.append(Term(term.output, term.input, strength))
            changes.extend(Term(t.output, t.input, -t.strength) for t in unmatched if t.strength)
            if changes:
                result[key] = changes
        return TermState(self.pool, result).snapshot()

    def apply_delta(self, delta):
        for (hook, expert), terms in delta.items():
            for term in terms:
                if isinstance(term.output, torch.Tensor) and isinstance(term.input, torch.Tensor):
                    match = next(
                        (
                            old
                            for old in self.terms.get((hook, expert), [])
                            if isinstance(old.output, torch.Tensor)
                            and isinstance(old.input, torch.Tensor)
                            and torch.equal(old.output, term.output)
                            and torch.equal(old.input, term.input)
                        ),
                        None,
                    )
                    if match is not None:
                        match.strength += term.strength
                        continue
                self.add(hook, expert, term.output, term.input, term.strength)
        for key in list(self.terms):
            self.terms[key] = [term for term in self.terms[key] if term.strength != 0]
            if not self.terms[key]:
                del self.terms[key]

    def save(self, path):
        torch.save(
            {
                "directions": self.pool.directions,
                "sources": self.pool.sources,
                "threshold": self.pool.threshold,
                "terms": self.snapshot(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path):
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            DirectionPool(data["threshold"], data["directions"], data["sources"]), data["terms"]
        )
