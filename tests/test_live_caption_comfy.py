# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest

from kilora.comfyui.live_write import comfy_caption


class _Cond:
    def __init__(self, value):
        self.value = value

    def process_cond(self, batch_size, device):
        return types.SimpleNamespace(cond=self.value)


class _Clip:
    def tokenize(self, caption):
        return [caption]

    def encode_from_tokens_scheduled(self, tokens):
        return [("encoded", tokens)]


def _comfy(monkeypatch, helpers: bool):
    import torch

    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    calls = []

    def convert_cond(conditioning):
        calls.append(conditioning)
        return [{"conditioning": conditioning}]

    def encode_model_conds(fn, conds, x, device, kind):
        return [{"model_conds": {"c": _Cond(torch.ones(1))}}]

    samplers.encode_model_conds = encode_model_conds
    if helpers:
        sampler_helpers = types.ModuleType("comfy.sampler_helpers")
        sampler_helpers.convert_cond = convert_cond
        monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", sampler_helpers)
        comfy.sampler_helpers = sampler_helpers
    else:
        samplers.convert_cond = convert_cond
        monkeypatch.delitem(sys.modules, "comfy.sampler_helpers", raising=False)
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    return calls


@pytest.mark.parametrize("helpers", [True, False])
def test_caption_encoder_finds_convert_cond(monkeypatch, helpers):
    import torch

    calls = _comfy(monkeypatch, helpers)
    encode = comfy_caption(_Clip(), types.SimpleNamespace(extra_conds=None), "a cat")
    out = encode({"input": torch.zeros(2, 4)}, 1)
    assert calls == [[("encoded", ["a cat"])]]
    assert torch.equal(out["c"], torch.ones(1))
