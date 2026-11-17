# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .bake import (
    _copy_layer_state,
    _ensure_input_device,
    _separate_shared_parameters,
    base_only_active,
)
from .sections import LayerChange

__all__ = ["FixedLoRA", "fixed_copy", "is_fixed"]

_FIXED_CLASSES: dict[type, type] = {}


class FixedLoRA:
    _kilora_fixed = True

    @property
    def base(self) -> nn.Module:
        return self.__dict__["_kilora_base"]

    @property
    def changes(self) -> list[LayerChange]:
        return self.__dict__["_kilora_changes"]

    def change(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if base_only_active():
            return y
        from .call_values import (
            active_axes,
            contiguous_row_slices,
            current_active_axes,
            current_mount_share,
            current_row_axes,
            mount_shares,
            row_axes,
            selected_mount_shares,
        )
        from .compute import AXES
        from .timeline import current_scale

        rows = current_row_axes()
        if rows is not None:
            values = rows
            if len(values) != y.shape[0]:
                raise ValueError("layer row count does not match model batch")
            slices = contiguous_row_slices(values)
            parts = []
            for value in dict.fromkeys(values):
                if value < -1 or value >= len(AXES):
                    raise ValueError("invalid layer row selection")
                indices = torch.tensor(
                    [i for i, v in enumerate(values) if v == value], device=y.device
                )
                names = ("creative", "functional") if value == 0 else (AXES[value],)
                if value == -1:
                    from contextlib import nullcontext

                    selected = nullcontext()
                else:
                    selected = active_axes(names)
                with row_axes(None), selected, mount_shares(selected_mount_shares(indices)):
                    if slices is None:
                        out = self.change(x.index_select(0, indices), y.index_select(0, indices))
                    else:
                        out = self.change(x[slices[value]], y[slices[value]])
                if slices is None:
                    y = y.index_copy(0, indices, out)
                else:
                    parts.append(out)
            return y if slices is None or not parts else torch.cat(parts, dim=0)
        active = current_active_axes()
        from .comfyui.expert_text import selected_text

        selection = selected_text()
        for c in self.__dict__["_kilora_changes"]:
            window = getattr(c, "entry_window", None)
            if window is not None:
                from .compute import current_progress

                if not window[0] <= current_progress() <= window[1]:
                    continue
            cast_mult = False
            expert = getattr(c, "text_expert", None)
            if selection is not None and expert is not None and expert != selection[0]:
                continue
            if active is not None and getattr(c, "axis", "creative") not in active:
                continue
            mult = current_scale(getattr(c, "tag", ""), getattr(c, "axis", "creative"))
            expert = getattr(c, "expert_index", None)
            if (
                selection is None
                and expert is not None
                and getattr(c, "tag", "") in ("routed-loras", "routed-ip")
            ):
                share = current_mount_share(c.tag, expert, c.expert_count)
                if share is not None:
                    if isinstance(share, torch.Tensor):
                        if mult is None:
                            cast_mult = True
                        else:
                            share = share.to(device=y.device, dtype=y.dtype)
                        share = share.reshape(*share.shape, *([1] * (y.ndim - share.ndim)))
                    mult = share if mult is None else mult * share
            from .recording import active_perturbation

            trial = active_perturbation()
            changes = getattr(trial, "fixed_changes", {})
            amount = changes.get(getattr(c, "_kilora_import_key", None), 0.0)
            if amount:
                from .fused import scaled_difference

                clean = c._kilora_import_clean
                extra = amount if mult is None else amount * mult
                y = scaled_difference(y, clean.add(x, y), extra, cast_mult=cast_mult)
            if isinstance(mult, torch.Tensor):
                from .fused import scaled_difference

                y = scaled_difference(y, c.add(x, y), mult, cast_mult=cast_mult)
            elif mult is None or mult == 1.0:
                y = c.add(x, y)
            elif mult != 0.0:
                from .fused import scaled_difference

                y = scaled_difference(y, c.add(x, y), mult)
        return y


def _fixed_class(base: type) -> type:
    if base not in _FIXED_CLASSES:

        def forward(self, x, *args, **kwargs):
            _ensure_input_device(self, x)
            return FixedLoRA.change(self, x, base.forward(self, x, *args, **kwargs))

        def change(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return FixedLoRA.change(self, x, y)

        def _apply(self, fn, recurse=True):
            _separate_shared_parameters(self)
            result = base._apply(self, fn, recurse=recurse)
            if self.weight.device.type != "cuda":
                for item in self.__dict__["_kilora_changes"]:
                    cache = getattr(item, "_cast", None)
                    if cache is not None:
                        cache.clear()
            return result

        _FIXED_CLASSES[base] = type(
            f"Fixed{base.__name__}",
            (base, FixedLoRA),
            {"forward": forward, "_kilora_change": change, "_apply": _apply},
        )
    return _FIXED_CLASSES[base]


def fixed_copy(
    module: nn.Module, changes: Sequence[LayerChange], *, source: nn.Module | None = None
) -> nn.Module:

    if is_fixed(module):
        if source is None:
            source = module
        module = module.__dict__["_kilora_base"]
    for change in changes:
        validate_for = getattr(change, "validate_for", None)
        if (
            validate_for is not None
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            validate_for(module.in_features, module.out_features)
    cls = _fixed_class(type(module))
    new = cls.__new__(cls)
    new.__dict__.update(_copy_layer_state(module, source))
    new.__dict__["_kilora_base"] = module
    new.__dict__["_kilora_changes"] = list(changes)
    new.__dict__["comfy_force_forward"] = True
    return new


def is_fixed(module: object) -> bool:
    return getattr(module, "_kilora_fixed", False) is True
