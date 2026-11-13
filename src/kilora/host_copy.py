# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch


def host_copies(
    values: Iterable[torch.Tensor], fallback: Callable[[torch.Tensor], torch.Tensor]
) -> list[torch.Tensor]:

    out: list[Any] = []
    streams: dict[torch.device, Any] = {}
    for v in values:
        v = v.detach()
        if v.is_cuda and v.is_contiguous() and v.layout == torch.strided:
            buf = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
            buf.copy_(v, non_blocking=True)
            if v.device not in streams:
                streams[v.device] = torch.cuda.current_stream(v.device)
            out.append(buf)
        else:
            out.append(fallback(v))
    for stream in streams.values():
        stream.synchronize()
    return out


def host_copy_map(
    values: Mapping[Any, torch.Tensor], fallback: Callable[[torch.Tensor], torch.Tensor]
) -> dict[Any, torch.Tensor]:
    keys = list(values)
    return dict(zip(keys, host_copies((values[k] for k in keys), fallback), strict=True))


def all_finite(values: Iterable[torch.Tensor]) -> bool:
    flags: dict[torch.device, list[torch.Tensor]] = {}
    for v in values:
        flags.setdefault(v.device, []).append(torch.isfinite(v).all())
    return all(bool(torch.stack(items).all()) for items in flags.values())


def nonzero_flags(values: Iterable[torch.Tensor]) -> list[bool]:
    values = list(values)
    groups: dict[torch.device, list[int]] = {}
    for i, v in enumerate(values):
        groups.setdefault(v.device, []).append(i)
    out = [False] * len(values)
    for indices in groups.values():
        counts = torch.stack([values[i].count_nonzero() for i in indices]).tolist()
        for i, n in zip(indices, counts, strict=True):
            out[i] = bool(n)
    return out
