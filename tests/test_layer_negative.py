# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import types

import torch

from kilora.comfyui.layer_batch import install_layer_batch, pair_negatives

VALUES = {"ip": 1.0, "creative": 10.0, "functional": 100.0}
CFG = 4.0


def toy(x, text, layer):
    if layer is None:
        axes = tuple(VALUES)
    elif layer == "creative":
        axes = ("creative", "functional")
    else:
        axes = (layer,)
    return x * text + sum(VALUES[a] for a in axes)


def stub_hooks(monkeypatch):
    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    hooks = types.ModuleType("comfy.hooks")
    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = lambda options, hook_scope: types.SimpleNamespace(
        options=options
    )
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)


def stand_in(calls):
    def calc(model, conds, x, sigma, options):
        wrapper = options.get("model_function_wrapper")
        buckets = {}
        for kind, group in enumerate(conds):
            for cond in group or ():
                key = (id(cond.get("hooks")), cond.get("area"))
                buckets.setdefault(key, []).append((kind, cond))
        totals = [torch.zeros_like(x) for _ in conds]
        weights = [torch.zeros_like(x) for _ in conds]
        for items in buckets.values():
            hooks = items[0][1].get("hooks")
            layer = hooks.hooks[0].options["kilora_layer"] if hooks is not None else None
            kinds = [kind for kind, _ in items]

            def apply_model(inp, timestep, layer=layer, **c):
                return torch.cat([toy(inp[i : i + 1], t, layer) for i, t in enumerate(c["text"])])

            args = {
                "input": x.repeat(len(items), 1, 1, 1),
                "timestep": sigma,
                "c": {
                    "text": [cond["text"] for _, cond in items],
                    "transformer_options": {"kilora_layer": layer, "cond_or_uncond": kinds},
                },
                "cond_or_uncond": kinds,
            }
            if wrapper is not None:
                y = wrapper(apply_model, args)
            else:
                y = apply_model(args["input"], args["timestep"], **args["c"])
            for row, (kind, cond) in enumerate(items):
                calls.append(
                    {
                        "kind": kind,
                        "layer": layer,
                        "hooks": cond.get("hooks"),
                        "mask": cond.get("mask"),
                        "area": cond.get("area"),
                        "window": (cond.get("model_conds") or {}).get("kilora_crop_window"),
                        "text": cond["text"],
                        "batch": id(items),
                    }
                )
                mask = cond.get("mask")
                mult = (torch.ones_like(x) if mask is None else mask) * cond.get("strength", 1.0)
                totals[kind] += y[row : row + 1] * mult
                weights[kind] += mult
        return [t / w for t, w in zip(totals, weights, strict=True)]

    return calc


def layered(masks):
    ip, creative = masks
    window_ip = torch.tensor([[0, 0, 4, 4]])
    return [
        [
            {
                "kilora_layer": "ip",
                "kilora_alpha_source": "mask",
                "kilora_crop": "auto",
                "kilora_layer_batch": False,
                "mask": ip,
                "mask_strength": 1.0,
                "set_area_to_bounds": False,
                "area": (4, 4, 0, 0),
                "model_conds": {"kilora_crop_window": window_ip},
                "text": 2.0,
            },
            {
                "kilora_layer": "creative",
                "kilora_alpha_source": "mask",
                "kilora_crop": "off",
                "kilora_layer_batch": False,
                "mask": creative,
                "mask_strength": 1.0,
                "set_area_to_bounds": False,
                "model_conds": {},
                "text": 3.0,
            },
        ],
        [{"text": 0.5, "model_conds": {}}],
    ]


def masks():
    ip = torch.zeros(1, 4, 4)
    ip[..., :2] = 1.0
    ip[..., 2] = 0.5
    return ip, 1.0 - ip


def run(patcher, conds, sigma=1.0):
    x = torch.arange(16.0).view(1, 1, 4, 4) / 16
    out = patcher.model_options["sampler_calc_cond_batch_function"](
        {
            "conds": conds,
            "input": x,
            "sigma": torch.tensor([sigma]),
            "model": None,
            "model_options": patcher.model_options,
        }
    )
    return x, out


def test_layer_negatives_share_the_state_of_their_layer(monkeypatch):
    stub_hooks(monkeypatch)
    calls = []
    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=stand_in(calls))
    run(patcher, layered(masks()))
    assert len(calls) == 4
    for layer in ("ip", "creative"):
        cond = [c for c in calls if c["kind"] == 0 and c["layer"] == layer]
        negative = [c for c in calls if c["kind"] == 1 and c["layer"] == layer]
        assert len(cond) == 1 and len(negative) == 1
        cond, negative = cond[0], negative[0]
        assert negative["hooks"] is cond["hooks"]
        assert negative["mask"] is cond["mask"]
        assert negative["area"] == cond["area"]
        assert negative["window"] is cond["window"]
        assert negative["text"] == 0.5
        assert negative["batch"] == cond["batch"]


def test_combined_result_is_cfg_per_layer(monkeypatch):
    stub_hooks(monkeypatch)
    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=stand_in([]))
    m = masks()
    x, (cond, uncond) = run(patcher, layered(m))
    guided = uncond + CFG * (cond - uncond)
    total = torch.zeros_like(x)
    for mask, layer, text in zip(m, ("ip", "creative"), (2.0, 3.0), strict=True):
        c, u = toy(x, text, layer), toy(x, 0.5, layer)
        total += mask * (u + CFG * (c - u))
    expected = total / sum(m)
    assert torch.allclose(guided, expected, atol=1e-5)
    plain = toy(x, 0.5, None)
    assert not torch.allclose(plain + CFG * (cond - plain), expected, atol=1e-3)


def test_without_layers_the_call_is_unchanged(monkeypatch):
    stub_hooks(monkeypatch)
    seen = []

    def calc(model, conds, x, sigma, options):
        seen.append((conds, options))
        return [x * 2, x]

    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=calc)
    conds = [[{"text": 2.0, "model_conds": {}}], [{"text": 0.5, "model_conds": {}}]]
    x, out = run(patcher, conds)
    assert seen[0][0] is conds
    assert seen[0][1] is patcher.model_options
    assert torch.equal(out[0], x * 2) and torch.equal(out[1], x)
    assert pair_negatives(conds) == (conds, 0)


def test_routed_negative_is_not_evaluated_twice(monkeypatch):
    stub_hooks(monkeypatch)
    calls = []
    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=stand_in(calls))
    conds = layered(masks())
    conds[1] = [{**c, "text": 0.5} for c in conds[0]]
    run(patcher, conds)
    negatives = [c["layer"] for c in calls if c["kind"] == 1]
    assert sorted(negatives) == ["creative", "ip"]
    conds = layered(masks())
    routed = {**conds[0][0], "text": 0.25}
    paired, added = pair_negatives([conds[0], [routed, *conds[1]]])
    assert added == 1
    assert [c.get("kilora_layer") for c in paired[1]] == ["ip", "creative"]
    assert paired[1][0] is routed
    assert paired[1][1]["text"] == 0.5


def test_unlayered_positive_keeps_the_plain_negative():
    conds = layered(masks())
    plain = {"text": 1.5, "model_conds": {}}
    paired, added = pair_negatives([[*conds[0], plain], conds[1]])
    assert added == 2
    assert [c.get("kilora_layer") for c in paired[1]] == ["ip", "creative", None]
    assert paired[1][2] is conds[1][0]


def test_calls_per_step_are_logged_once_per_run(monkeypatch, caplog):
    stub_hooks(monkeypatch)
    calls = []
    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=stand_in(calls))
    with caplog.at_level(logging.INFO, logger="kilora.comfyui.layer_batch"):
        for sigma in (3.0, 2.0, 1.0):
            run(patcher, layered(masks()), sigma)
    lines = [r.getMessage() for r in caplog.records if "batch rows per step" in r.getMessage()]
    assert lines == [
        "KiLoRA: layered sample, 2 model calls and 4 batch rows per step for 4 conds "
        "(2 negatives paired with a layer, compose separate)."
    ]
    assert len({c["batch"] for c in calls[:4]}) == 2
