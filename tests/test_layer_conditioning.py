# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest
import torch

from kilora.call_values import active_axes
from kilora.comfyui.nodes import KiLoRALayerConditioning
from kilora.comfyui.patching import _key_mask_by_step
from kilora.compute import LayerSet, run_layer


def make_set(axis, seed):
    g = torch.Generator().manual_seed(seed)
    return LayerSet(
        router=torch.eye(4),
        keys_in=torch.ones(1, 1, 4),
        down=torch.randn(2, 4, generator=g),
        up=torch.randn(2, 4, generator=g),
        form="input",
        axis=axis,
        gate_mode="none",
        per_expert=2,
    )


def test_layer_wrapper_selects_axes_and_restores():
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher)
    wrapper = patcher.model_options["model_function_wrapper"]
    sets = [make_set("creative", 1), make_set("functional", 2), make_set("ip", 3)]
    x = torch.randn(1, 4, 4)

    def model(x, timestep, **c):
        return run_layer(x, x, sets, 1, target="layer")

    def call(layer=None):
        options = {} if layer is None else {"kilora_layer": layer}
        return wrapper(
            model,
            {"input": x, "timestep": torch.tensor([1.0]), "c": {"transformer_options": options}},
        )

    expected = model(x, None)
    assert torch.equal(call(), expected)
    with active_axes(["ip"]):
        assert torch.equal(call("ip"), model(x, None))
    with active_axes(["creative", "functional"]):
        assert torch.equal(call("creative"), model(x, None))
    assert torch.equal(call(), expected)


def test_layer_hook_preserves_control_and_mask(monkeypatch):
    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

    class Hook:
        def __init__(self, options, hook_scope):
            self.options = options

    def set_mask(conds, mask, area, strength):
        return [[c, {**meta, "mask": mask, "mask_strength": strength}] for c, meta in conds]

    def set_hooks(conds, group):
        return [[c, {**meta, "hooks": group}] for c, meta in conds]

    hooks = types.ModuleType("comfy.hooks")
    hooks.HookGroup, hooks.TransformerOptionsHook = Group, Hook
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    hooks.set_mask_for_conditioning, hooks.set_hooks_for_conditioning = set_mask, set_hooks
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    control = object()
    mask = torch.tensor([[[1.0, 0.0]]])
    node = KiLoRALayerConditioning()
    ip = node.layer([[1, {"control": control}]], "ip", mask, feather=0)[0][0][1]
    bg = node.layer([[1, {}]], "creative", 1 - mask, feather=0)[0][0][1]
    calls = []

    def calc(cond):
        calls.append(cond.get("control"))
        axis = cond["kilora_layer"]
        return cond["mask"] * (2 if axis == "ip" else 5)

    assert torch.equal(calc(ip) + calc(bg), torch.tensor([[[2.0, 5.0]]]))
    assert calls == [control, None]
    attention = node.layer([[1, {}]], "ip", alpha_source="attention")[0][0][1]
    assert attention["kilora_alpha_source"] == "attention"


def test_fixed_changes_follow_axis_selection():
    from kilora.fixed_change import fixed_copy

    class Change:
        def __init__(self, axis, value):
            self.axis, self.value = axis, value

        def add(self, x, y):
            return y + self.value

    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.data.zero_()
    layer = fixed_copy(module, [Change("ip", 2), Change("creative", 3)])
    x = torch.ones(1, 2)
    assert torch.equal(layer(x), torch.full_like(x, 5))
    with active_axes(["ip"]):
        assert torch.equal(layer(x), torch.full_like(x, 2))


@pytest.mark.parametrize("frames", [None, 1, 3])
def test_attention_layer_masks_follow_previous_step_and_chain(frames):
    from kilora.comfyui.layer_alpha import install_layer_alpha

    seen = []
    previous_calls = []
    patcher = types.SimpleNamespace(model_options={})
    mask = torch.full((1, 4, 4), 0.25)
    ip = {
        "kilora_layer": "ip",
        "kilora_alpha_source": "attention",
        "t5xxl_ids": [1, 7, 2],
        "mask": mask,
        "control": "pose",
    }
    creative = {"kilora_layer": "creative", "kilora_alpha_source": "attention", "mask": 1 - mask}
    fixed = {"kilora_layer": "ip", "kilora_alpha_source": "mask", "mask": mask}
    conds = [[ip, creative, fixed], None]

    def previous(args):
        previous_calls.append(True)
        return calc(
            args["model"], args["conds"], args["input"], args["sigma"], args["model_options"]
        )

    def calc(model, groups, x, sigma, options):
        seen.append(groups[0])
        wrapper = options["model_function_wrapper"]

        def forward(x, timestep, **c):
            opts = c["transformer_options"]
            if opts["kilora_layer"] == "ip":
                patch = opts["patches"]["attn2_patch"][-1]
                patch(None, None, None, extra_options={"block_index": 1})
                override = opts["optimized_attention_override"]
                q = torch.ones(1, 4, 1)
                k = torch.tensor([[[0.0], [2.0], [0.0]]])
                override(lambda *a, **kw: None, q, k, k, 1)
            return x

        wrapper(
            forward,
            {"input": x, "timestep": sigma, "c": {"transformer_options": {"kilora_layer": "ip"}}},
        )
        return [x]

    patcher.model_options["sampler_calc_cond_batch_function"] = previous
    install_layer_alpha(patcher, "anima", {"subject": [7]}, 3)
    hook = patcher.model_options["sampler_calc_cond_batch_function"]
    for sigma in (3.0, 2.0, 1.0):
        hook(
            {
                "model": None,
                "conds": conds,
                "input": torch.ones((1, 1, 4, 4) if frames is None else (1, 1, frames, 4, 4)),
                "sigma": torch.tensor([sigma]),
                "model_options": patcher.model_options,
            }
        )
    assert len(previous_calls) == 3
    assert torch.equal(seen[0][0]["mask"], mask)
    assert torch.equal(seen[0][1]["mask"], 1 - mask)
    for group in seen[1:]:
        assert torch.allclose(
            group[0]["mask"],
            torch.ones((1, 4, 4) if frames is None else (1, frames, 4, 4)),
        )
        assert torch.equal(group[1]["mask"], torch.ones_like(group[0]["mask"]))
        assert torch.equal(group[2]["mask"], mask)
        assert group[0]["control"] == "pose"
    assert ip["mask"] is mask


def test_mask_source_and_missing_maps_leave_conditioning_unchanged():
    from kilora.comfyui.layer_alpha import install_layer_alpha

    patcher = types.SimpleNamespace(model_options={})
    mask = torch.tensor([[[0.2, 0.8]]])
    ip = {"kilora_layer": "ip", "kilora_alpha_source": "mask", "mask": mask}
    creative = {"kilora_layer": "creative", "kilora_alpha_source": "mask"}
    seen = []

    def calc(model, conds, x, sigma, options):
        seen.append(conds)
        return [x]

    install_layer_alpha(patcher, "anima", {"subject": [7]}, 3, calc)
    hook = patcher.model_options["sampler_calc_cond_batch_function"]
    for sigma in (2.0, 1.0):
        conds = [[ip, creative]]
        hook(
            {
                "model": None,
                "conds": conds,
                "input": torch.ones(1, 1, 1, 2),
                "sigma": torch.tensor([sigma]),
                "model_options": patcher.model_options,
            }
        )
        assert seen[-1] is conds
    ip["kilora_alpha_source"] = "attention"
    hook(
        {
            "model": None,
            "conds": [[ip, creative]],
            "input": torch.ones(1, 1, 1, 2),
            "sigma": torch.tensor([3.0]),
            "model_options": patcher.model_options,
        }
    )
    assert seen[-1][0][0] is ip
    assert seen[-1][0][1] is creative


def test_attention_mask_enters_five_dimensional_sampler_chain(monkeypatch):
    import kilora.comfyui.layer_alpha as alpha_module
    from kilora.comfyui.layer_batch import install_layer_batch
    from kilora.comfyui.layer_crop import install_layer_crop

    class Group:
        def add(self, hook):
            pass

        @staticmethod
        def combine_all_hooks(groups):
            return groups[-1]

    class Hook:
        def __init__(self, options, hook_scope):
            pass

    class Collector:
        def __init__(self, positions, blocks):
            pass

        def finish_step(self):
            return {"subject": torch.ones(1, 12 * 16)}

    hooks = types.ModuleType("comfy.hooks")
    hooks.HookGroup, hooks.TransformerOptionsHook = Group, Hook
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    monkeypatch.setattr(alpha_module, "AttentionMaps", Collector)
    monkeypatch.setattr(alpha_module, "trigger_positions", lambda *args: ([0], None))
    monkeypatch.setattr(
        alpha_module,
        "alpha_from_maps",
        lambda *args, **kwargs: {
            "ip": torch.full((1, 12 * 16), 0.25),
            "creative": torch.full((1, 12 * 16), 0.75),
        },
    )

    def calc(model, conds, x, sigma, options):
        out = torch.zeros_like(x)
        for cond in conds[0]:
            mask = cond.get("mask")
            mult = (
                torch.ones_like(x)
                if mask is None
                else mask.unsqueeze(1).repeat(
                    (x.shape[0] // mask.shape[0], x.shape[1]) + (1,) * (mask.ndim - 1)
                )
            )
            out += x * mult
        return [out]

    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=calc)
    alpha_module.install_layer_alpha(patcher, "anima", {"subject": [7]}, 1)
    install_layer_crop(patcher)
    x = torch.ones(1, 16, 1, 12, 16)
    conds = [
        [
            {
                "kilora_layer": layer,
                "kilora_alpha_source": "attention",
                "kilora_crop": "auto",
                "kilora_layer_batch": False,
            }
            for layer in ("ip", "creative")
        ]
    ]
    sampler = patcher.model_options["sampler_calc_cond_batch_function"]
    args = {
        "input": x,
        "sigma": torch.tensor([2.0]),
        "conds": conds,
        "model": None,
        "model_options": patcher.model_options,
    }
    assert torch.equal(sampler(args)[0], 2 * x)
    args["sigma"] = torch.tensor([1.0])
    assert torch.equal(sampler(args)[0], x)


def test_layer_compose_input_is_stored_and_defaults_to_separate(monkeypatch):
    hooks = types.ModuleType("comfy.hooks")
    hooks.set_mask_for_conditioning = lambda conds, mask, area, strength: [
        [c, dict(meta)] for c, meta in conds
    ]
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    node = KiLoRALayerConditioning()
    plain = node.layer([[1, {}]], "ip")[0][0][1]
    assert plain["kilora_compose"] == "separate"
    single = node.layer([[1, {}]], "ip", compose="single")[0][0][1]
    assert single["kilora_compose"] == "single"
    assert {k: v for k, v in single.items() if k != "kilora_compose"} == {
        k: v for k, v in plain.items() if k != "kilora_compose"
    }


def test_layer_region_penalty_and_mount_scope_are_stored(monkeypatch):
    hooks = types.ModuleType("comfy.hooks")
    hooks.set_mask_for_conditioning = lambda conds, mask, area, strength: [
        [c, dict(meta)] for c, meta in conds
    ]
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    node = KiLoRALayerConditioning()
    plain = node.layer([[1, {}]], "ip")[0][0][1]
    assert plain["kilora_region_penalty"] is None
    assert plain["kilora_mount_scope"] is None
    set_ = node.layer([[1, {}]], "ip", region_penalty=0, mount_scope="all")[0][0][1]
    assert set_["kilora_region_penalty"] == 0.0
    assert isinstance(set_["kilora_region_penalty"], float)
    assert set_["kilora_mount_scope"] == "all"
