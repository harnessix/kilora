# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EditBranch:
    id: str
    parent: str | None
    state: Any
    seed: int | None = None
    temperature: float | None = None
    depth: int | None = None
    paths: list = field(default_factory=list)
    action: str | None = None
    writes: list[int] = field(default_factory=list)
    merge_pairs: list = field(default_factory=list)


class BranchStore:
    def __init__(self, state: Any):
        self.branches = {"main": EditBranch("main", None, state)}
        self.head = self.current = "main"
        self.serial = 0

    @property
    def state(self) -> Any:
        return self.branches[self.current].state

    def fork(self, parent: str | None = None, *, seed=None, temperature=None, depth=None) -> str:
        import copy

        parent = self.head if parent is None else parent
        source = self.branches[parent].state
        self.serial += 1
        branch_id = f"branch-{self.serial}"
        clone = getattr(source, "fork", None)
        state = clone() if clone is not None else copy.deepcopy(source)
        self.branches[branch_id] = EditBranch(branch_id, parent, state, seed, temperature, depth)
        return branch_id

    def checkout(self, branch_id: str) -> None:
        if branch_id not in self.branches:
            raise KeyError(branch_id)
        self.current = branch_id

    def promote(self, branch_id: str) -> None:
        self.checkout(branch_id)
        self.head = branch_id

    def delete(self, branch_id: str) -> None:
        if branch_id == self.head or branch_id == self.current:
            raise ValueError("cannot delete the head or current branch")
        del self.branches[branch_id]

    def list(self) -> list[dict]:
        import copy

        return [
            dict(
                id=b.id,
                parent=b.parent,
                seed=b.seed,
                temperature=b.temperature,
                depth=b.depth,
                paths=copy.deepcopy(b.paths),
                action=b.action,
                writes=list(b.writes),
                head=b.id == self.head,
                current=b.id == self.current,
            )
            for b in self.branches.values()
        ]
