# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.metadata
import os
from typing import Any

import torch
from safetensors.torch import load_file

from kilora.errors import KiLoRAError
from kilora.plugin_support import STATE_PREFIX, TrainerNetwork, check_trainer_version

from . import SUPPORTED_VERSIONS

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
IGNORED_KEYS = {"exclude_patterns", "include_patterns", "loraplus_lr_ratio", "verbose"}


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("musubi-tuner")
    except importlib.metadata.PackageNotFoundError:
        return None


class KiLoRAMusubiNetwork(TrainerNetwork):
    def is_mergeable(self) -> bool:
        return False

    def prepare_network(self, args: Any) -> None:
        pass

    def apply_to(
        self,
        text_encoders: Any,
        unet: Any,
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ) -> None:
        if not apply_unet:
            raise KiLoRAError("KiLoRA mods change the diffusion model only")
        self.attach(unet)
        self._sync()

    def enable_gradient_checkpointing(self) -> None:
        pass

    def prepare_grad_etc(self, unet: Any) -> None:
        self.trainable()

    def on_epoch_start(self, unet: Any) -> None:
        self.train()

    def on_step_start(self) -> None:
        pass

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs: Any):
        groups = self.parameter_groups(unet_lr)
        return groups, ["" for _ in groups]

    def apply_max_norm_regularization(self, max_norm_value: float, device: Any):
        return 0, 0.0, 0.0

    def save_weights(self, file: str, dtype: Any, metadata: Any) -> None:
        self.save(file, dtype)

    def load_weights(self, file: str):
        unused = self.load(file)
        if unused:
            raise KiLoRAError(f"{file} is not a KiLoRA checkpoint ({len(unused)} unknown tensors)")
        return "loaded KiLoRA weights"


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = {MANIFEST_KEYS[k]: str(kwargs.pop(k)) for k in list(kwargs) if k in MANIFEST_KEYS}
    options = {}
    if "target_modules" in kwargs:
        value = kwargs.pop("target_modules")
        options["target_modules"] = (
            [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else value
        )
    for key in ("routes", "seed", "num_experts", "router_dim"):
        if key in kwargs:
            options[key] = int(kwargs.pop(key))
    for key in IGNORED_KEYS:
        kwargs.pop(key, None)
    if kwargs:
        raise KiLoRAError(f"unknown KiLoRA network arguments: {sorted(kwargs)}")
    return manifest, options


def create_network(
    family: str,
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    unet: Any,
    **kwargs: Any,
) -> KiLoRAMusubiNetwork:
    version = check_trainer_version("musubi-tuner", installed_version(), SUPPORTED_VERSIONS)
    manifest, options = _split_kwargs(dict(kwargs))
    rank = int(network_dim or 4)
    net = KiLoRAMusubiNetwork(
        family=family,
        rank=rank,
        alpha=float(network_alpha) if network_alpha is not None else rank,
        trained_with={"tool": "musubi-tuner", "version": version},
        manifest=manifest,
        **options,
    )
    net.set_multiplier(multiplier)
    return net


def create_network_from_weights(
    family: str,
    multiplier: float,
    weights_sd: dict[str, torch.Tensor] | str | os.PathLike[str],
    unet: Any,
    **kwargs: Any,
) -> KiLoRAMusubiNetwork:
    if not isinstance(weights_sd, dict):
        weights_sd = load_file(os.fspath(weights_sd))
    downs = [v for k, v in weights_sd.items() if k.startswith(STATE_PREFIX) and k.endswith("_down")]
    if not downs:
        raise KiLoRAError("these weights are not a KiLoRA checkpoint")
    experts, routes, rank, _ = downs[0].shape
    kwargs.setdefault("routes", routes)
    kwargs.setdefault("num_experts", experts)
    keys = [v for k, v in weights_sd.items() if k.startswith(STATE_PREFIX) and k.endswith("_keys")]
    if keys:
        kwargs.setdefault("router_dim", keys[0].shape[-1])
    net = create_network(family, multiplier, rank, None, unet, **kwargs)
    if unet is not None:
        net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
        net.load_resume_state(weights_sd)
    return net
