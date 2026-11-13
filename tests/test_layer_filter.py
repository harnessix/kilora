# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.comfyui.layer_filter import KiLoRALayerFilter


class Patcher:
    def __init__(self, options=None, patches=None, objects=None):
        self.model_options = options or {}
        self.patches = patches or {}
        self.object_patches = objects or {}

    def clone(self):
        return Patcher(self.model_options, self.patches, self.object_patches)


def test_attention_patch_only_runs_for_selected_layer_and_reports_weights():
    calls = []

    def patch(q, k, v, options):
        calls.append(options["kilora_layer"])
        return q + 1, k, v

    before = Patcher()
    model = Patcher(
        {"transformer_options": {"patches": {"attn1_patch": [patch]}}},
        {"diffusion.weight": [object()]},
    )
    out, report = KiLoRALayerFilter().filter(model, before, "ip")["result"]
    gated = out.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    x = torch.randn(2, 3)
    q, k, v = gated(x, x, x, {"kilora_layer": "creative"})
    assert torch.equal(q, x) and torch.equal(k, x) and torch.equal(v, x)
    assert torch.equal(gated(x, x, x, {"kilora_layer": "ip"})[0], x + 1)
    assert calls == ["ip"]
    assert "Global patches: weight: diffusion.weight" in report
    assert model.model_options["transformer_options"]["patches"]["attn1_patch"][0] is patch


def test_existing_replacement_and_wrappers_are_preserved():
    def old(q, k, v, options):
        return q * 2

    def added(q, k, v, options):
        return q * 3

    def wrapper(executor, x, transformer_options):
        return executor(x, transformer_options) + 10

    def model_wrapper(apply, args):
        return apply(args["input"], args["timestep"], **args["c"]) + 20

    key = ("input", 0)
    before = Patcher({"transformer_options": {"patches_replace": {"attn1": {key: old}}}})
    model = Patcher(
        {
            "model_function_wrapper": model_wrapper,
            "transformer_options": {
                "patches_replace": {"attn1": {key: added}},
                "wrappers": {"diffusion_model": {"plugin": [wrapper]}},
            },
        }
    )
    out, _ = KiLoRALayerFilter().filter(model, before, "ip")["result"]
    opts = out.model_options["transformer_options"]
    replace = opts["patches_replace"]["attn1"][key]
    x = torch.tensor([1.0])
    assert torch.equal(replace(x, x, x, {"kilora_layer": "creative"}), x * 2)
    assert torch.equal(replace(x, x, x, {"kilora_layer": "ip"}), x * 3)
    gated = opts["wrappers"]["diffusion_model"]["plugin"][0]
    assert torch.equal(gated(lambda y, options: y, x, {"kilora_layer": "creative"}), x)
    assert torch.equal(gated(lambda y, options: y, x, {"kilora_layer": "ip"}), x + 10)
    gated_model = out.model_options["model_function_wrapper"]
    apply = lambda x, timestep, **c: x
    args = {"input": x, "timestep": x, "c": {"transformer_options": {"kilora_layer": "creative"}}}
    assert torch.equal(gated_model(apply, args), x)
    args["c"]["transformer_options"]["kilora_layer"] = "ip"
    assert torch.equal(gated_model(apply, args), x + 20)


def test_anima_attention_patch_passes_mapping_and_keeps_model_helpers():
    class Attention:
        def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
            return {"q": q + 1, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

        def models(self):
            return ["control"]

        def to(self, device):
            return self

    patch = Attention()
    model = Patcher({"transformer_options": {"patches": {"attn1_patch": [patch]}}})
    out, _ = KiLoRALayerFilter().filter(model, Patcher(), "ip")["result"]
    gated = out.model_options["transformer_options"]["patches"]["attn1_patch"][0]
    x = torch.tensor([1.0])
    assert gated.models() == ["control"]
    assert gated.to("cpu") is gated
    result = gated(x, x, x, extra_options={"kilora_layer": "creative"})
    assert torch.equal(result["q"], x)
    assert torch.equal(gated(x, x, x, extra_options={"kilora_layer": "ip"})["q"], x + 1)
