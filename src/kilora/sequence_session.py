# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _validate(steps: Any) -> None:
    if not isinstance(steps, list):
        raise ValueError("session must be a list of steps")
    for step in steps:
        if not isinstance(step, dict) or step.get("outcome") not in ("keep", "discard"):
            raise ValueError("each step must have a keep or discard outcome")
        if step.get("type") == "direction":
            if set(step) != {"type", "basis_id", "coefficients", "outcome"}:
                raise ValueError("invalid direction fields")
            coefficients = step["coefficients"]
            if not isinstance(step["basis_id"], str) or not step["basis_id"]:
                raise ValueError("basis_id must be a nonempty string")
            if not isinstance(coefficients, list) or not coefficients:
                raise ValueError("coefficients must be a nonempty list")
            if any(
                isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c)
                for c in coefficients
            ):
                raise ValueError("coefficients must be finite numbers")
        elif step.get("type") == "prompt_edit":
            if set(step) != {"type", "edit", "outcome"} or not isinstance(step["edit"], str):
                raise ValueError("invalid prompt edit fields")
        else:
            raise ValueError("unknown step type")


def save_session(path: str | os.PathLike[str], steps: list[dict[str, Any]]) -> None:
    _validate(steps)
    data = json.dumps(steps, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            name = stream.name
            stream.write(data)
        os.replace(name, target)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def load_session(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        steps = json.load(stream)
    _validate(steps)
    return steps
