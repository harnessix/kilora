# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .adapter import KiLoRA, Settings, find_target_modules
from .errors import KiLoRAError
from .modfile import save_mod
from .modlist import satisfies
from .naming import MOD_SUFFIX
from .presets import PRESETS
from .sets import new_keyed_set

STATE_PREFIX = "kilora/resume/"


def hint_fields(values: Mapping[str, Any]) -> dict[str, Any]:

    out = dict(values)
    for key in ("trigger_words", "example_prompts"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = [v.strip() for v in value.split("|") if v.strip()]
    if "recommended_strength" in out:
        value = float(out.pop("recommended_strength"))
        out["strength"] = {"recommended": value, "min": -2.0, "max": 2.0}
    return out


class UnsupportedTrainerVersion(KiLoRAError):
    pass


def check_trainer_version(tool: str, version: str | None, supported: str) -> str:
    if version is None:
        raise UnsupportedTrainerVersion(
            f"cannot find the installed {tool} version. This plugin supports {tool} {supported}."
        )
    try:
        ok = satisfies(version, supported)
    except ValueError:
        ok = False
    if not ok:
        raise UnsupportedTrainerVersion(
            f"{tool} {version} is not supported. This plugin supports {tool} {supported}. "
            "Install a supported version, or a newer plugin release."
        )
    return version


DEFAULT_EXPERTS = 16
DEFAULT_ROUTER_DIM = 32
MIN_STATIC_EXPERTS = 2


def check_static_experts(num_experts: int) -> int:
    n = int(num_experts)
    if n < MIN_STATIC_EXPERTS:
        raise KiLoRAError(
            f"A static KiLoRA mod needs at least {MIN_STATIC_EXPERTS} experts per layer, "
            f"got {n}. Each layer is two or more LoRAs that tokens choose between."
        )
    return n


class TrainerNetwork(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        rank: int,
        alpha: float | None = None,
        target_modules: Sequence[str] | None = None,
        routes: int = 1,
        num_experts: int | None = None,
        router_dim: int = DEFAULT_ROUTER_DIM,
        seed: int = 0,
        trained_with: Mapping[str, str] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if family not in PRESETS:
            raise KiLoRAError(f"no preset for base model family {family!r}")
        self.family = family
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.patterns = list(target_modules or PRESETS[family].target_modules)
        self.routes = int(routes)
        self.num_experts = check_static_experts(
            DEFAULT_EXPERTS if num_experts is None else num_experts
        )
        self.router_dim = int(router_dim)
        self.seed = int(seed)
        self.trained_with = dict(trained_with or {})
        self.manifest_fields = dict(manifest or {})
        self.multiplier = 1.0
        self.enabled = True
        self.adapter: KiLoRA | None = None
        self._model: nn.Module | None = None

    def attach(self, model: nn.Module) -> None:
        if self.adapter is not None:
            return
        targets = find_target_modules(model, self.patterns)
        modules = dict(model.named_modules())
        shapes = {t: (modules[t].in_features, modules[t].out_features) for t in targets}
        preset = PRESETS[self.family]
        expert_set = new_keyed_set(
            "main",
            shapes,
            num_experts=self.num_experts,
            routes=self.routes,
            rank=self.rank,
            router_dim=self.router_dim,
            router_seed=preset.router_seed,
            seed=self.seed,
        )
        self.adapter = KiLoRA(model, [expert_set], Settings(depth=1))
        self._model = model
        device = next(model.parameters()).device
        self.adapter.to(device)

    def trainable(self) -> list[nn.Parameter]:
        self._require()
        out = []
        for site in self.adapter.sites:
            for key, p in site.tensors.items():
                active = key.endswith(("_down", "_up"))
                p.requires_grad_(active)
                if active:
                    out.append(p)
        return out

    def parameter_groups(self, lr: float) -> list[dict[str, Any]]:
        self._require()
        down, up = [], []
        for site in self.adapter.sites:
            for key, p in site.tensors.items():
                if key.endswith("_up"):
                    up.append(p)
                elif key.endswith("_down"):
                    down.append(p)
                elif "_keys_" in key:
                    p.requires_grad_(False)
        for p in down + up:
            p.requires_grad_(True)
        return [
            {"params": down, "lr": lr},
            {"params": up, "lr": lr * self.alpha / self.rank},
        ]

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)
        self._sync()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._sync()

    def _sync(self) -> None:
        if self.adapter is None:
            return
        self.adapter.set_strengths([self.multiplier])
        self.adapter.set_depth(1 if self.enabled else 0)

    def state_for_resume(self) -> dict[str, torch.Tensor]:
        self._require()
        return {
            f"{STATE_PREFIX}{target}/{key}": p.detach().float().cpu().contiguous()
            for target, site in zip(self.adapter.targets, self.adapter.sites, strict=True)
            for key, p in site.tensors.items()
        }

    def load_resume_state(self, state: Mapping[str, torch.Tensor]) -> list[str]:
        self._require()
        unused = []
        lookup = {}
        for target, site in zip(self.adapter.targets, self.adapter.sites, strict=True):
            for key, p in site.tensors.items():
                lookup[f"{STATE_PREFIX}{target}/{key}"] = p
        with torch.no_grad():
            for name, value in state.items():
                if name in lookup:
                    lookup[name].copy_(value.to(lookup[name]))
                else:
                    unused.append(name)
        return unused

    def save(self, path: str | os.PathLike[str], dtype: torch.dtype | None = None) -> str:

        self._require()
        path = os.fspath(path)
        state = self.state_for_resume()
        if dtype is not None:
            state = {k: v.to(dtype) for k, v in state.items()}
        save_file(state, path, {"kilora.resume": "1"})
        kmod = os.path.splitext(path)[0] + MOD_SUFFIX
        stem = os.path.splitext(os.path.basename(path))[0]
        fields = {
            "id": re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip(".-")[:64] or "mod",
            "name": stem,
            "base_model": {"family": self.family},
        }
        fields.update(hint_fields(self.manifest_fields))
        fields["trained_with"] = dict(self.trained_with)
        sets = self.adapter.to_sets()
        for s in sets:
            s.strength = 1.0
        save_mod(kmod, sets, self.adapter.settings, **fields)
        return kmod

    def load(self, path: str | os.PathLike[str]) -> list[str]:
        return self.load_resume_state(load_file(os.fspath(path)))

    def remove(self) -> None:
        if self.adapter is not None:
            self.adapter.remove()
            self.adapter = None

    def _require(self) -> None:
        if self.adapter is None:
            raise KiLoRAError("the network is not applied to a model yet")
