# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

PACK = True
ALIGN = 16


def aligned(nbytes: int) -> int:
    return -(-int(nbytes) // ALIGN) * ALIGN


def nbytes_of(shape, dtype: torch.dtype) -> int:
    count = 1
    for n in shape:
        count *= int(n)
    return aligned(count * torch.empty((), dtype=dtype).element_size())


class DevicePack:
    def __init__(self, hint: int = 0):
        self.hint = max(int(hint), 0)
        self._buffers: dict[str, list] = {}

    def buffers(self, device=None) -> list[torch.Tensor]:
        if device is not None:
            entry = self._buffers.get(str(torch.device(device)))
            return list(entry[0]) if entry else []
        return [b for entry in self._buffers.values() for b in entry[0]]

    def empty(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        shape = tuple(int(n) for n in shape)
        device = torch.device(device)
        if not PACK:
            return torch.empty(shape, dtype=dtype, device=device)
        size = nbytes_of(shape, dtype)
        count = 1
        for n in shape:
            count *= n
        raw_bytes = count * torch.empty((), dtype=dtype).element_size()
        key = str(device)
        entry = self._buffers.get(key)
        buf = entry[0][-1] if entry else None
        if buf is None or entry[1] + size > buf.numel():
            grow = self.hint if buf is None else 2 * buf.numel()
            buf = torch.empty(max(size, grow, ALIGN), dtype=torch.uint8, device=device)
            if entry is None:
                entry = self._buffers[key] = [[], 0]
            entry[0].append(buf)
            entry[1] = 0
        start = entry[1]
        entry[1] = start + size
        return buf[start : start + raw_bytes].view(dtype).view(shape)

    def zeros(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        return self.empty(shape, dtype, device).zero_()

    def put(
        self, value: torch.Tensor, into: torch.Tensor | None = None, device=None
    ) -> torch.Tensor:
        value = value.detach()
        device = value.device if device is None else torch.device(device)
        if (
            into is not None
            and into.shape == value.shape
            and into.dtype == value.dtype
            and into.device == device
        ):
            return into.copy_(value)
        return self.empty(value.shape, value.dtype, device).copy_(value)

    def clear(self) -> None:
        self._buffers = {}


def shares_buffer(t: torch.Tensor, pack: DevicePack) -> bool:
    ptr = t.untyped_storage().data_ptr()
    return any(b.untyped_storage().data_ptr() == ptr for b in pack.buffers())
