# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


class _ProbeComplete(Exception):
    pass


class StepZeroProbe:
    def __init__(
        self,
        base_model: nn.Module,
        base_function: Callable,
        blocks: Sequence[nn.Module],
        layer_groups: Mapping[str, Sequence[str]],
        on_probe: Callable,
        *,
        probe_blocks: int = 2,
        fallbacks: Mapping[str, str] | None = None,
        pass_args: bool = False,
    ) -> None:
        if probe_blocks < 1 or not blocks:
            raise ValueError("probe_blocks and the block sequence must be nonempty")
        self.base_model = base_model
        self.base_function = base_function
        self.blocks = tuple(blocks)
        self.layer_groups = {group: tuple(paths) for group, paths in layer_groups.items()}
        self.fallbacks = dict(fallbacks or {})
        self.on_probe = on_probe
        self.probe_blocks = probe_blocks
        self.pass_args = pass_args
        self.reset()

    def reset(self) -> None:
        choice = getattr(self, "_choice", None)
        if choice is not None and callable(getattr(choice, "close", None)):
            choice.close()
        self._selected = False
        self._choice = None

    def outer_sample(self, executor, *args, **kwargs):
        self.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            self.reset()

    def _capture(self, args: dict) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        stopped = False
        handles = []
        try:
            for group, paths in self.layer_groups.items():
                for path in paths:
                    layer = self.base_model.get_submodule(path)

                    def record(_module, inputs, *, key=group):
                        if key not in captured and inputs and isinstance(inputs[0], torch.Tensor):
                            captured[key] = inputs[0][:1].detach().clone()

                    handles.append(layer.register_forward_pre_hook(record))

            def stop(_module, _inputs, _output):
                nonlocal stopped
                stopped = True
                raise _ProbeComplete

            last = self.blocks[min(self.probe_blocks, len(self.blocks)) - 1]
            handles.append(last.register_forward_hook(stop))
            try:
                from ..bake import base_only

                with torch.no_grad(), base_only():
                    self.base_function(args["input"], args["timestep"], **(args.get("c") or {}))
            except _ProbeComplete:
                pass
        finally:
            for handle in handles:
                handle.remove()
        if not stopped:
            raise RuntimeError("base pass did not reach the final probe block")
        for group in self.layer_groups:
            source = self.fallbacks.get(group)
            if group not in captured and source in captured:
                captured[group] = captured[source]
        return captured

    def wrap(self, previous=None):
        def wrapper(apply_model, args):
            if not self._selected:
                captured = self._capture(args)
                self._choice = (
                    self.on_probe(captured, args) if self.pass_args else self.on_probe(captured)
                )
                if self._choice is not None and not callable(self._choice):
                    raise TypeError("on_probe must return a model-function wrapper or None")
                self._selected = True

            def invoke(model, call_args):
                if previous is not None:
                    return previous(model, call_args)
                return model(
                    call_args["input"], call_args["timestep"], **(call_args.get("c") or {})
                )

            if self._choice is not None:
                return self._choice(
                    lambda x, t, **c: invoke(apply_model, {"input": x, "timestep": t, "c": c}), args
                )
            return invoke(apply_model, args)

        return wrapper


def install(patcher: Any, probe: StepZeroProbe) -> Any:
    out = patcher.clone()
    if not hasattr(out, "add_wrapper_with_key"):
        raise TypeError("the patcher must support outer sampling wrappers")
    options = out.model_options
    previous = options.get("model_function_wrapper")
    wrapper = probe.wrap(previous)
    if hasattr(out, "set_model_unet_function_wrapper"):
        out.set_model_unet_function_wrapper(wrapper)
    else:
        options["model_function_wrapper"] = wrapper
    if hasattr(out, "remove_wrappers_with_key"):
        out.remove_wrappers_with_key("outer_sample", "kilora_library_probe")
    out.add_wrapper_with_key("outer_sample", "kilora_library_probe", probe.outer_sample)
    return out
