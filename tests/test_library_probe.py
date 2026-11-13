# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from kilora.library.probe import StepZeroProbe


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_attn = nn.Module()
        self.img_attn.qkv = nn.Linear(4, 4)
        self.img_mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.calls = 0

    def forward(self, x, context):
        self.calls += 1
        x = x + self.img_attn.qkv(x) + context
        return x + self.img_mlp(x)


class Flux(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(3, 4)
        self.double_blocks = nn.ModuleList([Block() for _ in range(3)])
        self.final_layer = nn.Linear(4, 3)

    def forward(self, x, timestep, context):
        x = self.img_in(x) + timestep.reshape(-1, 1, 1)
        for block in self.double_blocks:
            x = block(x, context)
        return self.final_layer(x)


GROUPS = {
    "attention": ["double_blocks.0.img_attn.qkv"],
    "mlp": ["double_blocks.1.img_mlp.0"],
    "late": ["double_blocks.2.img_mlp.0"],
}


def setup(on_probe):
    torch.manual_seed(123)
    model = Flux()
    probe = StepZeroProbe(
        model,
        model,
        model.double_blocks,
        GROUPS,
        on_probe,
        fallbacks={"late": "mlp"},
    )
    args = {
        "input": torch.randn(2, 5, 3),
        "timestep": torch.tensor([3.0, 3.0]),
        "c": {"context": torch.randn(2, 5, 4)},
    }
    return model, probe, args


def test_probe_inputs_match_full_base_pass_and_stop_early():
    selections = []
    model, probe, args = setup(lambda tokens: selections.append(tokens))
    expected = {}
    handles = []
    for group in ("attention", "mlp"):

        def record(_module, inputs, *, key=group):
            expected[key] = inputs[0][:1].detach().clone()

        handles.append(model.get_submodule(GROUPS[group][0]).register_forward_pre_hook(record))
    with torch.no_grad():
        model(args["input"], args["timestep"], **args["c"])
    for handle in handles:
        handle.remove()
    for block in model.double_blocks:
        block.calls = 0
    output = probe.wrap()(model, args)
    assert len(selections) == 1
    assert [block.calls for block in model.double_blocks] == [2, 2, 1]
    for group in expected:
        torch.testing.assert_close(selections[0][group], expected[group], rtol=0, atol=0)
        assert not selections[0][group].requires_grad
    assert torch.equal(selections[0]["late"], expected["mlp"])
    assert output.requires_grad


def test_selection_once_per_run_and_empty_choice_is_base():
    seen = []
    model, probe, args = setup(lambda tokens: seen.append(tokens))
    wrapper = probe.wrap()

    def run(steps):
        for step in steps:
            args["timestep"] = torch.full((2,), step)
            expected = model(args["input"], args["timestep"], **args["c"])
            assert torch.equal(wrapper(model, args), expected)

    probe.outer_sample(run, (3.0, 3.0, 2.0, 2.5, 1.0))
    probe.outer_sample(run, (3.0, 2.0))
    assert len(seen) == 2
    probe.outer_sample(run, (2.0,))
    assert len(seen) == 3


def test_choice_is_used_from_step_zero_and_reused():
    seen = []
    calls = []

    def choose(tokens):
        seen.append(tokens)

        def choice(model, args):
            calls.append(float(args["timestep"][0]))
            return model(args["input"], args["timestep"], **args["c"]) + 1

        return choice

    model, probe, args = setup(choose)
    wrapper = probe.wrap()
    for step in (3.0, 2.0):
        args["timestep"] = torch.full((2,), step)
        expected = model(args["input"], args["timestep"], **args["c"]) + 1
        assert torch.equal(wrapper(model, args), expected)
    assert len(seen) == 1
    assert calls == [3.0, 2.0]


def test_probe_removes_hooks_after_failure():
    model, probe, args = setup(lambda _: None)
    probe.base_function = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("failure"))
    with pytest.raises(RuntimeError, match="failure"):
        probe.wrap()(model, args)
    assert all(
        not module._forward_pre_hooks and not module._forward_hooks for module in model.modules()
    )


def test_install_preserves_previous_wrapper_and_resets_each_run():
    from kilora.library.probe import install

    seen = []
    calls = []
    model, probe, args = setup(lambda tokens: seen.append(tokens))

    class Patcher:
        def __init__(self):
            self.model_options = {"model_function_wrapper": self.previous}
            self.wrappers = {}

        @staticmethod
        def previous(fn, call_args):
            calls.append("previous")
            return fn(call_args["input"], call_args["timestep"], **call_args["c"])

        def clone(self):
            return Patcher()

        def add_wrapper_with_key(self, kind, key, wrapper):
            self.wrappers[kind, key] = wrapper

    patcher = Patcher()
    installed = install(patcher, probe)
    wrapper = installed.model_options["model_function_wrapper"]
    outer = installed.wrappers["outer_sample", "kilora_library_probe"]
    for _ in range(2):
        outer(lambda: wrapper(model, args))
    assert len(seen) == len(calls) == 2
    assert patcher.model_options["model_function_wrapper"] != wrapper


def test_unreached_group_without_fallback_is_left_out():
    seen = []
    model, probe, args = setup(lambda tokens: seen.append(tokens))
    probe.fallbacks = {}
    probe.wrap()(model, args)
    assert set(seen[0]) == {"attention", "mlp"}
