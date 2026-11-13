# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import torch

from kilora.call_values import active_axes, axis_alphas, row_axes
from kilora.comfyui.layer_batch import install_layer_batch
from kilora.comfyui.patching import _key_mask_by_step
from kilora.compute import AXES, LayerSet, run_layer


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


def test_batched_rows_match_hooked_calls_and_plain_with_alphas():
    sets = [make_set("creative", 1), make_set("functional", 2), make_set("ip", 3)]
    x = torch.randn(3, 2, 3, 2, 4)
    alphas = {"creative": 0.7, "functional": 0.8, "ip": 0.4}
    with axis_alphas(alphas):
        expected = []
        for row, names in zip(x, [("ip",), ("creative", "functional"), None], strict=True):
            if names is None:
                expected.append(run_layer(row[None], row[None], sets, 1, target="block"))
            else:
                with active_axes(names):
                    expected.append(run_layer(row[None], row[None], sets, 1, target="block"))
        with row_axes(torch.tensor([AXES.index("ip"), AXES.index("creative"), -1])):
            actual = run_layer(x, x, sets, 1, target="block")
        assert torch.equal(actual, torch.cat(expected))
        with row_axes(torch.tensor([AXES.index("ip")])):
            alone = run_layer(x[:1], x[:1], sets, 1, target="block")
        assert torch.equal(alone, expected[0])


def test_sampler_batches_conditioning_and_strips_ids(monkeypatch):
    class Cond:
        def __init__(self, cond):
            self.cond = cond

        def process_cond(self, batch_size, **kwargs):
            return Cond(self.cond.repeat(batch_size))

        def concat(self, others):
            return torch.cat([self.cond, *(v.cond for v in others)])

    comfy = types.ModuleType("comfy")
    conds_module = types.ModuleType("comfy.conds")
    conds_module.CONDRegular = Cond
    comfy.conds = conds_module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", conds_module)
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher)
    seen = []
    sets = [make_set("creative", 1), make_set("functional", 2), make_set("ip", 3)]

    def model(x, timestep, **kwargs):
        assert "kilora_layer_ids" not in kwargs
        seen.append(kwargs["context"].clone())
        return run_layer(x, x + kwargs["context"], sets, 1, target="block")

    def previous(args):
        entries = args["conds"][0] + args["conds"][1]
        ids = [c["model_conds"]["kilora_layer_ids"].process_cond(1) for c in entries]
        assert [c.get("kilora_layer") for c in entries] == ["ip", "creative"] * 2
        context = torch.tensor([1.0, 5.0, 7.0, 9.0]).view(4, 1, 1)
        return args["model_options"]["model_function_wrapper"](
            model,
            {
                "input": torch.ones(4, 2, 4),
                "timestep": torch.ones(4),
                "c": {
                    "kilora_layer_ids": ids[0].concat(ids[1:]),
                    "context": context,
                    "transformer_options": {},
                },
            },
        )

    patcher.model_options.pop("kilora_layer_batch_installed")
    patcher.model_options["sampler_calc_cond_batch_function"] = previous
    install_layer_batch(patcher)
    result = patcher.model_options["sampler_calc_cond_batch_function"](
        {
            "conds": [
                [
                    {"kilora_layer": "ip", "kilora_layer_batch": True, "model_conds": {}},
                    {"kilora_layer": "creative", "kilora_layer_batch": True, "model_conds": {}},
                ],
                [{"model_conds": {}}],
            ],
            "model_options": patcher.model_options,
        }
    )
    assert len(seen) == 1
    assert seen[0].flatten().tolist() == [1.0, 5.0, 7.0, 9.0]
    expected = []
    for context, layer in zip((1.0, 5.0, 7.0, 9.0), ("ip", "creative") * 2, strict=True):
        options = {} if layer is None else {"kilora_layer": layer}
        expected.append(
            patcher.model_options["model_function_wrapper"](
                model,
                {
                    "input": torch.ones(1, 2, 4),
                    "timestep": torch.ones(1),
                    "c": {"context": torch.tensor(context), "transformer_options": options},
                },
            )
        )
    assert torch.equal(result, torch.cat(expected))


def test_fixed_changes_follow_rows():
    from kilora.fixed_change import fixed_copy

    class Change:
        def __init__(self, axis, amount):
            self.axis, self.amount = axis, amount

        def add(self, x, y):
            return y + self.amount

    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.data.zero_()
    layer = fixed_copy(module, [Change("ip", 2), Change("creative", 3), Change("functional", 7)])
    with row_axes(torch.tensor([2, 0, -1])):
        result = layer(torch.ones(3, 2))
    assert torch.equal(result[:, 0], torch.tensor([2.0, 10.0, 12.0]))


def test_filter_and_attention_use_separate_hook_groups(monkeypatch):
    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    class Hook:
        def __init__(self, options, hook_scope):
            self.options = options

    comfy = types.ModuleType("comfy")
    hooks = types.ModuleType("comfy.hooks")
    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = Hook
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    for option, source in [(True, "mask"), (False, "attention")]:
        seen = []
        patcher = types.SimpleNamespace(model_options={"kilora_layer_filter": option})
        install_layer_batch(
            patcher, calc=lambda model, conds, x, sigma, opts, seen=seen: seen.extend(conds[0])
        )
        entries = [
            {
                "kilora_layer": axis,
                "kilora_alpha_source": source,
                "kilora_layer_batch": True,
                "model_conds": {},
            }
            for axis in ("ip", "creative")
        ]
        patcher.model_options["sampler_calc_cond_batch_function"](
            {
                "conds": [entries],
                "model": None,
                "input": None,
                "sigma": None,
                "model_options": patcher.model_options,
            }
        )
        assert [c["hooks"].hooks[0].options["kilora_layer"] for c in seen] == ["ip", "creative"]
        assert all("kilora_layer_ids" not in c["model_conds"] for c in seen)


def test_row_specific_alpha_maps_and_regions():
    sets = [make_set("creative", 11), make_set("functional", 12), make_set("ip", 13)]
    x = torch.randn(3, 4, 4)
    maps = {"ip": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.8, 0.7, 0.6]])}
    regions = [torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9], [0.9, 0.8, 0.7, 0.6]])] * 3
    choices = (2, 0, -1)
    expected = []
    for i, choice in enumerate(choices):
        names = None if choice == -1 else (("creative", "functional") if choice == 0 else ("ip",))
        from contextlib import nullcontext

        selected = nullcontext() if names is None else active_axes(names)
        with selected, axis_alphas({"ip": maps["ip"][i : i + 1]}):
            expected.append(
                run_layer(
                    x[i : i + 1], x[i : i + 1], sets, 1, regions=[r[i : i + 1] for r in regions]
                )
            )
    with axis_alphas(maps), row_axes(torch.tensor(choices)):
        actual = run_layer(x, x, sets, 1, regions=regions)
    assert torch.equal(actual, torch.cat(expected))


def test_route_freeze_keeps_each_selection_in_its_own_cache():
    from kilora.route_freeze import freezing

    sets = [make_set("creative", 21), make_set("functional", 22), make_set("ip", 23)]
    for item in sets:
        item.route_freeze = 0.0
    x = torch.randn(3, 3, 4)
    choices = (("ip",), ("creative", "functional"), None)
    caches = [{}, {}, {}]
    batched_cache = {}
    with torch.no_grad():
        for _ in range(2):
            expected = []
            for i, names in enumerate(choices):
                from contextlib import nullcontext

                selected = nullcontext() if names is None else active_axes(names)
                with freezing(caches[i]), selected:
                    expected.append(run_layer(x[i : i + 1], x[i : i + 1], sets, 2, target="block"))
            with freezing(batched_cache), row_axes(torch.tensor([2, 0, -1])):
                actual = run_layer(x, x, sets, 2, target="block")
            assert torch.equal(actual, torch.cat(expected))


def test_a_later_apply_with_carry_turns_batching_off(monkeypatch):
    hooks = types.ModuleType("comfy.hooks")

    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = lambda options, hook_scope: types.SimpleNamespace(
        options=options
    )
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    seen = []
    patcher = types.SimpleNamespace(model_options={})
    install_layer_batch(patcher, calc=lambda model, conds, x, sigma, opts: seen.extend(conds[0]))
    install_layer_batch(patcher, separate=True)
    entry = {
        "kilora_layer": "ip",
        "kilora_alpha_source": "mask",
        "kilora_layer_batch": True,
        "model_conds": {},
    }
    patcher.model_options["sampler_calc_cond_batch_function"](
        {
            "conds": [[entry]],
            "model": None,
            "input": None,
            "sigma": None,
            "model_options": patcher.model_options,
        }
    )
    assert "hooks" in seen[0] and "kilora_layer_ids" not in seen[0]["model_conds"]


def _stub_hooks(monkeypatch):
    hooks = types.ModuleType("comfy.hooks")

    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    hooks.HookGroup = Group
    hooks.TransformerOptionsHook = lambda options, hook_scope: types.SimpleNamespace(
        options=options
    )
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    comfy = types.ModuleType("comfy")
    comfy.hooks = hooks
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)


def test_layers_run_as_separate_calls_unless_every_layer_asks_for_batching(monkeypatch):
    _stub_hooks(monkeypatch)
    for flags in ((False, False), (True, False)):
        seen = []
        patcher = types.SimpleNamespace(model_options={})
        install_layer_batch(
            patcher, calc=lambda m, conds, x, s, o, seen=seen: seen.extend(conds[0])
        )
        entries = [
            {
                "kilora_layer": axis,
                "kilora_alpha_source": "mask",
                "kilora_layer_batch": flag,
                "model_conds": {},
            }
            for axis, flag in zip(("ip", "creative"), flags, strict=True)
        ]
        patcher.model_options["sampler_calc_cond_batch_function"](
            {
                "conds": [entries],
                "model": None,
                "input": None,
                "sigma": None,
                "model_options": patcher.model_options,
            }
        )
        assert [c["hooks"].hooks[0].options["kilora_layer"] for c in seen] == ["ip", "creative"]
        assert all("kilora_layer_ids" not in c["model_conds"] for c in seen)


def test_layer_node_batch_input_is_off_by_default():
    from kilora.comfyui import nodes as N

    types_ = N.KiLoRALayerConditioning.INPUT_TYPES()
    assert list(types_["optional"])[-2:] == ["batch", "compose"]
    assert types_["optional"]["batch"][1]["default"] is False


def test_five_dimensional_layer_chain_and_batch_masks(monkeypatch):
    from kilora.comfyui.layer_alpha import install_layer_alpha
    from kilora.comfyui.layer_crop import install_layer_crop

    class Group:
        def __init__(self):
            self.hooks = []

        def add(self, hook):
            self.hooks.append(hook)

        @staticmethod
        def combine_all_hooks(groups):
            return next(group for group in groups if group is not None)

    class Hook:
        def __init__(self, options, hook_scope):
            self.options = options

    class Cond:
        def __init__(self, cond):
            self.cond = cond

    comfy = types.ModuleType("comfy")
    hooks = types.ModuleType("comfy.hooks")
    hooks.HookGroup, hooks.TransformerOptionsHook = Group, Hook
    hooks.EnumHookScope = types.SimpleNamespace(HookedOnly=1)
    conds_module = types.ModuleType("comfy.conds")
    conds_module.CONDRegular = Cond
    comfy.hooks, comfy.conds = hooks, conds_module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.hooks", hooks)
    monkeypatch.setitem(sys.modules, "comfy.conds", conds_module)

    for source, batched, frames in [("attention", False, 1), ("mask", True, 3)]:
        patcher = types.SimpleNamespace(model_options={})
        seen = []

        def calc(model, groups, x, sigma, options, seen=seen):
            out = torch.zeros_like(x)
            for cond in groups[0]:
                mask = cond["mask"]
                mult = mask.unsqueeze(1).repeat(
                    (x.shape[0] // mask.shape[0], x.shape[1]) + (1,) * (mask.ndim - 1)
                )
                seen.append((mult.shape, cond))
                out += x * mult
            return [out]

        install_layer_batch(patcher, calc=calc)
        install_layer_alpha(patcher, "anima", {}, 0)
        install_layer_crop(patcher)
        x = torch.ones(1, 16, frames, 12, 16)
        masks = [torch.full((1, 12, 16), 0.25), torch.full((1, 12, 16), 0.75)]
        entries = [
            {
                "kilora_layer": layer,
                "kilora_alpha_source": source,
                "kilora_crop": "auto",
                "kilora_layer_batch": batched,
                "model_conds": {},
                "mask": mask,
            }
            for layer, mask in zip(("ip", "creative"), masks, strict=True)
        ]
        result = patcher.model_options["sampler_calc_cond_batch_function"](
            {
                "input": x,
                "sigma": torch.tensor([1.0]),
                "conds": [entries],
                "model": None,
                "model_options": patcher.model_options,
            }
        )[0]
        assert result.shape == x.shape
        assert torch.equal(result, x)
        assert all(shape == x.shape for shape, _ in seen)
        if batched:
            assert all("kilora_layer_ids" in cond["model_conds"] for _, cond in seen)
        else:
            assert all("hooks" in cond for _, cond in seen)
        assert all(mask.ndim == 3 for mask in masks)


def _two_step_wrapper_outputs(*, batched, carry=False, freeze=False, depth=3):
    from test_depth_forward import _sets

    layer = _sets(n_sets=1, width=4, experts=3, per_expert=2, router_dim=4, seed=41)[0]
    layer.axis = "creative"
    layer.route_freeze = 0.0 if freeze else None
    patcher = types.SimpleNamespace(model_options={})
    _key_mask_by_step(patcher, carry=carry)
    wrapper = patcher.model_options["model_function_wrapper"]
    gen = torch.Generator().manual_seed(42)
    inputs = [torch.randn(2, 8, 4, generator=gen) for _ in range(2)]

    def model(x, timestep, **kwargs):
        return run_layer(x, x, [layer], depth, target="block")

    outputs = []
    with torch.no_grad():
        for step, x in zip((0.9, 0.4), inputs, strict=True):
            if batched:
                args = {
                    "input": x,
                    "timestep": torch.tensor([step, step]),
                    "c": {
                        "kilora_layer_ids": torch.tensor([0, -1]),
                        "transformer_options": {"cond_or_uncond": [0, 1]},
                    },
                }
                outputs.append(wrapper(model, args))
            else:
                rows = []
                for i in range(2):
                    options = {"cond_or_uncond": [i]}
                    if i == 0:
                        options["kilora_layer"] = "creative"
                    args = {
                        "input": x[i : i + 1],
                        "timestep": torch.tensor([step]),
                        "c": {"transformer_options": options},
                    }
                    rows.append(wrapper(model, args))
                outputs.append(torch.cat(rows))
    return outputs


def test_wrapper_row_states_match_separate_calls():
    for carry, freeze, depth in (
        (False, False, 1),
        (False, False, 3),
        (False, True, 3),
        (True, False, 3),
        (True, True, 3),
    ):
        batched = _two_step_wrapper_outputs(batched=True, carry=carry, freeze=freeze, depth=depth)
        separate = _two_step_wrapper_outputs(batched=False, carry=carry, freeze=freeze, depth=depth)
        for step, (actual, expected) in enumerate(zip(batched, separate, strict=True)):
            assert torch.equal(actual, expected), (
                carry,
                freeze,
                depth,
                step,
                (actual - expected).abs().max().item(),
            )


def _state_key_layout_results(model, entries):
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _sets

    from kilora.comfyui import patching

    x = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(91))
    condition = torch.tensor([[1.0, 2.0], [-2.0, 3.0]])
    results = []
    keys = []
    for batched in (False, True):
        patched, _ = patching.apply(
            FakePatcher(FakeBaseModel(model)), entries, 3, patching.LiveEnv(), joint=True
        )
        fn = patched.model_options["model_function_wrapper"]
        c = {
            "c_crossattn": condition if batched else condition[:1],
            "transformer_options": {"cond_or_uncond": [0, 1] if batched else [0]},
        }
        if batched:
            c["kilora_layer_ids"] = torch.tensor([0, -1])
        else:
            c["transformer_options"]["kilora_layer"] = "creative"
        args = {"input": x if batched else x[:1], "timestep": torch.tensor([0.9]), "c": c}
        results.append(fn(lambda inp, t, _patched=patched, **kw: _patched.run(inp), args)[0])
        keys.append([(s.keys_in.clone(), s.keys_out.clone(), s.tau) for s in _sets(patched)])
    return results, keys


from test_comfyui import denoiser, folders
from test_comfyui_state_keys import chain


def test_state_key_pass_batch_and_separate_calls_match(denoiser, chain, monkeypatch):
    from kilora.comfyui import key_cache

    monkeypatch.setattr(key_cache, "load", lambda key: None)
    results, keys = _state_key_layout_results(denoiser, chain)
    for left, right in zip(*keys, strict=True):
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])
        assert left[2] == right[2]
    torch.testing.assert_close(*results, rtol=4e-6, atol=2e-6)


def test_linear_jump_fit_batch_and_separate_calls_match():
    from kilora import linear_jump
    from kilora.comfyui.state_keys import KeyPass

    x = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(81))
    fits = []
    for batched in (False, True):
        layer = types.SimpleNamespace(route_freeze=None, jump_fits={})
        target = types.SimpleNamespace(name="block", depth=3, schedule=None, sets=[layer])
        plan = types.SimpleNamespace(parts=[(0, -1, target)], _cache={})
        probe = KeyPass([plan], [], "tiny", linear_jump=True)
        patcher = types.SimpleNamespace(model_options={"model_function_wrapper": probe.wrap(None)})
        _key_mask_by_step(patcher)

        def model(inp, timestep, **kwargs):
            linear_jump.note("block", 3, inp, inp.square())
            return inp

        c = {"transformer_options": {"cond_or_uncond": [0, 1] if batched else [0]}}
        if batched:
            c["kilora_layer_ids"] = torch.tensor([0, -1])
        else:
            c["transformer_options"]["kilora_layer"] = "creative"
        args = {"input": x if batched else x[:1], "timestep": torch.tensor([0.9]), "c": c}
        patcher.model_options["model_function_wrapper"](model, args)
        fits.append(layer.jump_fits)
    for band in ("before", "after"):
        left, right = (fit[(band, 3)] for fit in fits)
        assert torch.equal(left[0], right[0])
        assert left[1] == right[1]


def test_first_call_passes_with_carry_freeze_and_dropout(denoiser, chain, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import key_cache, patching

    monkeypatch.setattr(key_cache, "load", lambda key: None)
    inputs = torch.randn(2, 2, 8, 4, generator=torch.Generator().manual_seed(117))
    outputs = []
    for batched in (False, True):
        patched, _ = patching.apply(
            FakePatcher(FakeBaseModel(denoiser)),
            chain,
            3,
            patching.LiveEnv(),
            joint=True,
            freeze=0.0,
            linear_jump=True,
        )
        key_pass = patching.applied(patched)["key_pass"]
        patched.model_options["model_function_wrapper"] = key_pass.wrap(None)
        _key_mask_by_step(patched, carry=True)
        wrapper = patched.model_options["model_function_wrapper"]
        result = []
        with torch.no_grad():
            for step, x in zip((0.9, 0.4), inputs, strict=True):
                rows = []
                for indices in ([0, 1],) if batched else ([0], [1]):
                    options = {"cond_or_uncond": indices}
                    c = {"transformer_options": options, "c_crossattn": x[indices, :1, :]}
                    if batched:
                        c["kilora_layer_ids"] = torch.tensor([0, -1])
                    elif indices == [0]:
                        options["kilora_layer"] = "creative"
                    args = {"input": x[indices], "timestep": torch.tensor([step]), "c": c}
                    rows.append(
                        wrapper(lambda inp, t, _patched=patched, **kw: _patched.run(inp), args)
                    )
                result.append(torch.cat(rows))
        outputs.append(result)
    for a, b in zip(*outputs, strict=True):
        torch.testing.assert_close(a, b, rtol=4e-6, atol=2e-6)


def test_probe_selects_first_positive_layer_and_preserves_separate_calls():
    from kilora.call_values import row_axes
    from kilora.comfyui.state_keys import _probe_rows

    x = torch.arange(24).reshape(6, 4)
    args = {
        "input": x,
        "timestep": torch.arange(6),
        "c": {
            "c_crossattn": x + 100,
            "transformer_options": {"cond_or_uncond": [0, 0, 1]},
        },
    }
    with row_axes(None):
        assert _probe_rows(args)[0] is args
    with row_axes(torch.tensor([2, 2, 0, 0, -1, -1])):
        selected, layer = _probe_rows(args)
    assert layer == 2
    assert torch.equal(selected["input"], x[:2])
    assert torch.equal(selected["timestep"], torch.arange(2))
    assert torch.equal(selected["c"]["c_crossattn"], x[:2] + 100)
    assert selected["c"]["transformer_options"]["cond_or_uncond"] == [0]


def test_row_slices_match_indexed_path_with_maps(monkeypatch):
    import kilora.call_values as calls

    torch.manual_seed(42)
    sets = [make_set("creative", 31), make_set("functional", 32), make_set("ip", 33)]
    x = torch.randn(6, 2, 3, 2, 4)
    regions = [torch.rand(6, 12) for _ in sets]
    alphas = {axis: torch.rand(6, 12) for axis in ("creative", "functional", "ip")}
    helper = calls.contiguous_row_slices
    for choices in ((2, 2, 0, 0, -1, -1), (2, 0, 2, 0, -1, -1)):
        assert (helper(choices) is not None) == (choices[0] == choices[1])
        with row_axes(torch.tensor(choices)), axis_alphas(alphas):
            actual = run_layer(x, x, sets, 1, regions=regions, target="block")
        with monkeypatch.context() as patch:
            patch.setattr(calls, "contiguous_row_slices", lambda values: None)
            with row_axes(torch.tensor(choices)), axis_alphas(alphas):
                expected = run_layer(x, x, sets, 1, regions=regions, target="block")
        assert torch.equal(actual, expected)


def test_fixed_row_slices_match_indexed_path(monkeypatch):
    import kilora.call_values as calls
    from kilora.fixed_change import fixed_copy

    class Change:
        def __init__(self, axis, amount):
            self.axis, self.amount = axis, amount

        def add(self, x, y):
            return y + x * self.amount

    layer = fixed_copy(torch.nn.Linear(4, 4), [Change("ip", 0.3), Change("creative", 0.7)])
    x = torch.randn(6, 3, 4)
    for choices in ((2, 2, 0, 0, -1, -1), (2, 0, 2, 0, -1, -1)):
        with row_axes(torch.tensor(choices)):
            actual = layer(x)
        with monkeypatch.context() as patch:
            patch.setattr(calls, "contiguous_row_slices", lambda values: None)
            with row_axes(torch.tensor(choices)):
                expected = layer(x)
        assert torch.equal(actual, expected)
