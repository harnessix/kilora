# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Direction = tuple[str, tuple[float, ...]]
Session = Sequence[Mapping[str, Any]]


def _kept_directions(session: Session) -> Iterable[Direction]:
    for step in session:
        if step["type"] == "direction" and step["outcome"] == "keep":
            yield step["basis_id"], tuple(step["coefficients"])


@dataclass(frozen=True)
class DirectionEvaluation:
    targets: int
    repeat_hits: int
    frequency_hits: int

    @property
    def repeat_hit_rate_at_5(self) -> float | None:
        return self.repeat_hits / self.targets if self.targets else None

    @property
    def frequency_hit_rate_at_5(self) -> float | None:
        return self.frequency_hits / self.targets if self.targets else None


def evaluate_directions(
    training_sessions: Iterable[Session], evaluation_sessions: Iterable[Session]
) -> DirectionEvaluation:

    counts: Counter[Direction] = Counter()
    for session in training_sessions:
        counts.update(_kept_directions(session))
    frequent = {
        direction for direction, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    }

    targets = repeat_hits = frequency_hits = 0
    for session in evaluation_sessions:
        previous = None
        for direction in _kept_directions(session):
            if previous is not None:
                targets += 1
                repeat_hits += direction == previous
                frequency_hits += direction in frequent
            previous = direction
    return DirectionEvaluation(targets, repeat_hits, frequency_hits)
