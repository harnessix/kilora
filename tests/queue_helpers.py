# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib

import torch

from kilora.live_queue import yield_point


class FakeSession:
    def __init__(self, width=8):
        self.state = {"layer": torch.zeros(width)}
        self.commits: list[int] = []
        self._undo: list[tuple[torch.Tensor, list[int]]] = []

    def write(self, value: int, log=None, batch_seconds=0.0) -> str:
        import time

        for i in range(3):
            yield_point()
            if log is not None:
                log.append(("batch", value, i))
            time.sleep(batch_seconds)
        self._undo.append((self.state["layer"].clone(), list(self.commits)))
        self.state = {"layer": self.state["layer"] * 0.999 + value * 1e-3}
        self.commits.append(value)
        return f"{len(self.commits)} commits"

    def undo(self) -> None:
        if not self._undo:
            raise RuntimeError("nothing to undo")
        layer, self.commits = self._undo.pop()
        self.state = {"layer": layer}

    def digest(self) -> str:
        return hashlib.sha256(self.state["layer"].numpy().tobytes()).hexdigest()
