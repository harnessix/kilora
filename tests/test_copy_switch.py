# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.fixed_change import fixed_copy


class Patcher:
    def __init__(self, model, replacement):
        self.model = model
        self.replacement = replacement
        self.backup = None

    def detach_without_unpatching_weights(self):
        self.model.layer = self.backup
        self.backup = None

    def partially_load(self, device):
        self.backup = self.model.layer
        self.model.layer = self.replacement
        if not self.model.lowvram and self.model.loaded_memory > 0:
            return
        for module in self.model.modules():
            if not list(module.children()) and list(module.parameters(recurse=False)):
                if not getattr(module, "comfy_patched_weights", False):
                    module.comfy_patched_weights = True
                module.to(device)
        self.model.lowvram = False
        self.model.loaded_memory = 1


def test_switch_prepared_copies_after_full_load():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(3, 2)
    model.lowvram = False
    model.loaded_memory = 0
    original = model.layer
    first = Patcher(model, fixed_copy(original, []))
    second = Patcher(model, fixed_copy(original, []))
    first.partially_load("meta")
    assert model.layer.weight.device.type == "meta"
    first.detach_without_unpatching_weights()
    second.partially_load("meta")
    assert model.layer.weight.device.type == "cpu"
    result = model.layer(torch.empty(1, 3, device="meta"))
    assert result.device.type == "meta"
    assert model.layer.weight.device.type == "meta"
    assert original.weight.device.type == "cpu"
