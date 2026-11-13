# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping

import torch

from .errors import KiLoRAError


def is_quantized_weight(weight) -> bool:

    if not isinstance(weight, torch.Tensor):
        return callable(getattr(weight, "dequantize", None))
    if type(weight) in (torch.Tensor, torch.nn.Parameter):
        return bool(weight.is_quantized)
    return callable(getattr(weight, "dequantize", None))


def read_base_weight(module, path="base weight", *, device=None):

    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    if getattr(weight, "device", None) is not None and weight.device.type == "meta":
        return weight
    if not is_quantized_weight(weight):
        view = weight.detach()
        return view if device is None else view.to(device=device)
    dense = weight.dequantize()
    if not isinstance(dense, torch.Tensor):
        raise KiLoRAError(f"{path}: base weight reader did not return a tensor")
    return dense.detach().to(device=device, dtype=torch.float32)


class BaseWeights(Mapping):
    def __init__(self, entries, *, device="cpu"):
        self.entries = dict(entries)
        self.device = device

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, target):
        module, part = self.entries[target]
        weight = read_base_weight(
            module, target, device=self.device if is_quantized_weight(module.weight) else None
        )
        if part is not None:
            index, count = part
            rows = weight.shape[0] // count
            weight = weight[index * rows : (index + 1) * rows]
        return weight

    def weight_identity(self, target):
        from .size_norm import weight_identity

        module, part = self.entries[target]
        weight = module.weight
        try:
            identity = weight_identity(weight)
        except RuntimeError:
            try:
                version = weight._version
            except RuntimeError:
                version = None
            identity = (
                id(weight),
                version,
                tuple(weight.shape),
                str(weight.dtype),
                str(weight.device),
            )
        return (*identity, part)

    @property
    def shapes(self):
        out = {}
        for target, (module, part) in self.entries.items():
            shape = tuple(module.weight.shape)
            out[target] = (shape[0] // part[1], *shape[1:]) if part else shape
        return out
