# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.call_values import current_axis_alphas
from kilora.comfyui.auto_alpha import AutoAlpha
from kilora.comfyui.patching import _key_mask_by_step


def test_auto_alpha_uses_previous_step_and_resets():
    auto = AutoAlpha("anima", {"subject": [7]}, 3, 1)
    seen = []

    def invoke(args):
        seen.append(dict(current_axis_alphas()))
        auto.collector._sums["subject"] = torch.ones(1, 4)
        auto.collector._counts["subject"] = 1
        return args["input"]

    def call(step):
        args = {
            "input": torch.ones(1, 1, 2, 2),
            "timestep": torch.tensor([step]),
            "c": {"t5xxl_ids": torch.tensor([1, 7, 2]), "transformer_options": {}},
        }
        return auto.call(args, invoke)

    call(3)
    call(3)
    call(2)
    assert seen[:2] == [{}, {}]
    assert torch.equal(seen[2]["ip"], torch.ones(1, 4))
    assert torch.equal(seen[2]["creative"], torch.ones(1, 4))
    call(4)
    assert seen[3] == {}
    assert current_axis_alphas() == {}


def test_off_wrapper_is_bit_identical():
    class Model:
        model_options = {}

    model = Model()
    _key_mask_by_step(model)
    x = torch.randn(1, 2, 2, 2)
    args = {"input": x, "timestep": torch.tensor([1.0]), "c": {}}
    expected = x * 0.7 + 0.2
    actual = model.model_options["model_function_wrapper"](lambda x, t: x * 0.7 + 0.2, args)
    assert torch.equal(actual, expected)


def test_sampler_conditioning_supplies_ids_and_chains():
    from kilora.comfyui.auto_alpha import install_sampler_hook

    auto = AutoAlpha("anima", {"subject": [7]}, 3, 1)
    seen = []

    def previous(args):
        def invoke(call):
            seen.append(dict(current_axis_alphas()))
            auto.collector._sums["subject"] = torch.ones(1, 4)
            auto.collector._counts["subject"] = 1
            return call["input"]

        return auto.call({"input": args["input"], "timestep": args["sigma"], "c": {}}, invoke)

    options = {"sampler_calc_cond_batch_function": previous}
    install_sampler_hook(options)
    for step in (3, 2):
        options["sampler_calc_cond_batch_function"](
            {
                "conds": [[{"t5xxl_ids": torch.tensor([1, 7, 2])}], None],
                "input": torch.ones(1, 1, 2, 2),
                "sigma": torch.tensor([step]),
            }
        )
    assert torch.equal(seen[1]["ip"], torch.ones(1, 4))


def test_identity_from_caption_precedes_trigger(monkeypatch):
    import json
    from types import SimpleNamespace

    from kilora.comfyui import auto_alpha

    meta = {
        "ss_dataset_dirs": json.dumps({"train": {"img_count": 10}}),
        "ss_tag_frequency": json.dumps({"train": {"bangboo is a mascot": 10}}),
    }
    monkeypatch.setattr(auto_alpha, "_metadata", lambda path: meta)
    monkeypatch.setattr(auto_alpha, "trigger_words_of", lambda path: ["bangboo is a mascot"])
    texts = []

    class Clip:
        def tokenize(self, text):
            texts.append(text)
            return {"t5xxl": [([7, 1] if text else [1])]}

    entry = SimpleNamespace(enabled=True, kind="lora", axis="ip", path="x")
    assert auto_alpha.trigger_ids([entry], Clip()) == {"x": [7]}
    assert "bangboo" in texts
