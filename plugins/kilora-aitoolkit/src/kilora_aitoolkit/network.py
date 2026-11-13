# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from kilora.errors import KiLoRAError
from kilora.plugin_support import TrainerNetwork

from . import ARCH_FAMILIES

MANIFEST_KEYS = {
    "mod_id": "id",
    "mod_name": "name",
    "mod_version": "version",
    "author": "author",
    "license": "license",
    "content_source": "content_source",
    "description": "description",
    "trigger_words": "trigger_words",
    "example_prompts": "example_prompts",
    "recommended_strength": "recommended_strength",
}


class KiLoRANetwork(TrainerNetwork):
    def __init__(
        self,
        *,
        unet: torch.nn.Module | None = None,
        lora_dim: int = 4,
        alpha: float | None = None,
        multiplier: float = 1.0,
        base_model: Any = None,
        tool_version: str = "unknown",
        **kwargs: Any,
    ) -> None:
        arch = getattr(base_model, "arch", None)
        family = kwargs.pop("family", None) or ARCH_FAMILIES.get(arch)
        if family is None:
            raise KiLoRAError(
                f"KiLoRA does not support the AI Toolkit model arch {arch!r} yet. "
                f"Supported: {', '.join(sorted(ARCH_FAMILIES))}"
            )
        manifest = {MANIFEST_KEYS[k]: kwargs.pop(k) for k in list(kwargs) if k in MANIFEST_KEYS}
        targets = kwargs.pop("target_modules", None)
        if isinstance(targets, str):
            targets = [targets]
        super().__init__(
            family=family,
            rank=lora_dim,
            alpha=alpha,
            target_modules=targets,
            routes=int(kwargs.pop("routes", 1)),
            num_experts=kwargs.pop("num_experts", None),
            seed=int(kwargs.pop("seed", 0)),
            trained_with={"tool": "ai-toolkit", "version": tool_version},
            manifest=manifest,
        )
        self._unet = [unet]
        self._multiplier = float(multiplier)
        self.is_active = False
        self.is_merged_in = False
        self.can_merge_in = False
        self.did_change_weights = False
        self.is_lorm = False

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: float | list[float]) -> None:
        values = [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)]
        if any(v != values[0] for v in values):
            raise KiLoRAError(
                f"Different strengths per batch item {values} are not supported. "
                "Use one network weight for the whole batch."
            )
        self._multiplier = values[0]
        if getattr(self, "adapter", None) is not None:
            self._sync()

    def _sync(self) -> None:
        if self.adapter is None:
            return
        self.adapter.set_strengths([self._multiplier])
        self.adapter.set_depth(1 if (self.enabled and self.is_active) else 0)

    def __enter__(self):
        self.is_active = True
        self._sync()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.is_active = False
        self._sync()

    def force_to(self, device: Any, dtype: Any = None) -> None:
        self._device = device

    def _update_torch_multiplier(self) -> None:
        self._sync()

    def apply_to(
        self,
        text_encoder: Any,
        unet: Any,
        train_text_encoder: bool = False,
        train_unet: bool = True,
    ) -> None:
        if train_text_encoder:
            raise KiLoRAError(
                "KiLoRA mods change the diffusion model only; set train_text_encoder to false"
            )
        self.attach(self._unet[0] if self._unet[0] is not None else unet)
        self._sync()

    def prepare_grad_etc(self, text_encoder: Any = None, unet: Any = None) -> None:
        self.trainable()

    def prepare_optimizer_params(
        self,
        text_encoder_lr: float | None = None,
        unet_lr: float | None = None,
        default_lr: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        lr = unet_lr if unet_lr is not None else default_lr
        if lr is None:
            raise KiLoRAError("AI Toolkit gave no learning rate")
        return self.parameter_groups(lr)

    def enable_gradient_checkpointing(self) -> None:
        pass

    def get_all_modules(self) -> list[Any]:
        return []

    def save_weights(
        self, file: str, dtype: Any = None, metadata: Any = None, extra_state_dict: Any = None
    ) -> None:
        if extra_state_dict:
            raise KiLoRAError("KiLoRA mods cannot store embeddings; turn off embedding training")
        self.save(file, dtype)

    def load_weights(self, file: str) -> None:
        unused = self.load(file)
        if unused:
            raise KiLoRAError(f"{file} is not a KiLoRA checkpoint ({len(unused)} unknown tensors)")
        return None

    def merge_in(self, *args: Any, **kwargs: Any) -> None:
        raise KiLoRAError("KiLoRA mods cannot merge into the base weights")

    def merge_out(self, *args: Any, **kwargs: Any) -> None:
        raise KiLoRAError("KiLoRA mods cannot merge into the base weights")

    def reset_weights(self) -> None:
        raise KiLoRAError("KiLoRA mods cannot merge into the base weights")
