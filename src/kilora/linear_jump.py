# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

import torch

from .recording import ContextStack

log = logging.getLogger("kilora")
_RECORDS = ContextStack("kilora_linear_jump_records")


def fit(first: torch.Tensor, last: torch.Tensor, ridge: float = 1e-6):
    x = first.detach().reshape(-1, first.shape[-1]).float()
    y = last.detach().reshape_as(x).float()
    delta = y - x
    gram = x.T @ x
    regularizer = ridge * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    w = torch.linalg.solve(gram + regularizer, x.T @ delta)
    residual = torch.linalg.vector_norm(delta - x @ w) / torch.linalg.vector_norm(y).clamp_min(
        1e-12
    )
    return w, float(residual)


@contextlib.contextmanager
def recording(band: str) -> Iterator[dict]:
    records = {"band": band, "layers": {}}
    with _RECORDS.pushed(records):
        yield records


def note(target, depth, first, last):
    if not _RECORDS:
        return
    try:
        w, residual = fit(first, last)
    except RuntimeError as error:
        log.warning("Linear jump fit failed for %s: %s", target, error)
        return
    _RECORDS[-1]["layers"][(target, depth)] = (w, residual)
    log.info(
        "Linear jump %s depth %s band %s residual %.6g",
        target,
        depth,
        _RECORDS[-1]["band"],
        residual,
    )
