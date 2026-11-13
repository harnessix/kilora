# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

from ..compute import IMPORT_SCORES
from ..config import ROUTING_SCOPES, ROUTING_SELECTS
from . import depth as depth_schedules

SETTINGS = "KILORA_SETTINGS"
STACK = "stack"

DEFAULTS: dict[str, Any] = {
    "mix": "balanced",
    "depth": -1,
    "iteration": None,
    "routing_scope": None,
    "routing_select": None,
    "depth_schedule": None,
    "import_score": None,
    "content_keys": "per_expert",
    "gate_stats": "fused",
    "lycoris_apply": "truncated",
    "candidates": 0,
    "candidate_summary": "keys",
    "routing_features": False,
    "cross_token_region": 0.0,
    "cross_token_prompt": 0.0,
    "cross_token_timestep": 0.0,
}

CROSS_SETTINGS = ("cross_token_region", "cross_token_prompt", "cross_token_timestep")

MIXES: dict[str, dict[str, Any]] = {
    "balanced": {},
    "keep_each_strong": {"routing_scope": "token", "routing_select": "top1"},
    "subtle": {"depth": 1, "routing_scope": "tensor", "routing_select": "sparsemax"},
    STACK: {"depth": 1},
}

FROM_MIX = "from mix"
ITERATIONS = ("accumulate", "path")


def mix_defaults(mix: str) -> dict[str, Any]:
    if mix not in MIXES:
        raise ValueError(f"unknown mix {mix!r}; use one of {', '.join(MIXES)}")
    return {**DEFAULTS, **MIXES[mix], "mix": mix}


def resolve_settings(settings: dict | None) -> dict:

    given = dict(settings or {})
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown KiLoRA settings: {', '.join(unknown)}")
    out = mix_defaults(given.pop("mix", None) or DEFAULTS["mix"])
    if given.get("depth", 0) is None:
        del given["depth"]
    out.update(given)
    out["depth"] = int(out["depth"])
    _check_all(out)
    return out


def _check_all(values: dict) -> None:
    if not isinstance(values["routing_features"], bool):
        raise ValueError("routing_features must be a boolean")
    for key in CROSS_SETTINGS:
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -16 <= value <= 16
        ):
            raise ValueError(f"{key} must be finite and between -16 and 16")
    _check(values, "iteration", ITERATIONS)
    _check(values, "routing_scope", ROUTING_SCOPES)
    _check(values, "routing_select", ROUTING_SELECTS)
    _check(values, "import_score", IMPORT_SCORES)
    if values["gate_stats"] not in ("fused", "torch"):
        raise ValueError("gate_stats must be fused or torch")
    if values["content_keys"] not in ("per_expert", "mix"):
        raise ValueError("content_keys must be per_expert or mix")
    if values["lycoris_apply"] not in ("truncated", "exact"):
        raise ValueError("lycoris_apply must be truncated or exact")
    if values["candidate_summary"] not in ("keys", "read_map"):
        raise ValueError("candidate_summary must be keys or read_map")
    if (
        not isinstance(values["candidates"], int)
        or isinstance(values["candidates"], bool)
        or values["candidates"] < 0
    ):
        raise ValueError("candidates must be a nonnegative integer")


def _check(values: dict, key: str, allowed: tuple[str, ...]) -> None:
    if values[key] is not None and values[key] not in allowed:
        raise ValueError(f"unknown {key} {values[key]!r}; use one of {', '.join(allowed)}")


class KiLoRASimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mix": (list(MIXES), {"default": DEFAULTS["mix"]})}}

    RETURN_TYPES = (SETTINGS,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"

    def make(self, mix):
        return (resolve_settings({"mix": mix}),)


class KiLoRASettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mix": (list(MIXES), {"default": DEFAULTS["mix"]}),
                "depth": ("INT", {"default": -2, "min": -2, "max": 16}),
                "iteration": ([FROM_MIX, "mod", *ITERATIONS], {"default": FROM_MIX}),
                "routing_scope": ([FROM_MIX, "mod", *ROUTING_SCOPES], {"default": FROM_MIX}),
                "routing_select": ([FROM_MIX, "mod", *ROUTING_SELECTS], {"default": FROM_MIX}),
            },
            "optional": {
                "depth_schedule": (depth_schedules.DEPTH, {}),
                "lycoris_apply": (["truncated", "exact"], {"default": "truncated"}),
                "candidates": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "candidate_summary": (["keys", "read_map"], {"default": "keys"}),
                "content_keys": (["per_expert", "mix"], {"default": "per_expert"}),
                "gate_stats": (["fused", "torch"], {"default": "fused"}),
                "routing_features": ("BOOLEAN", {"default": False}),
                **{
                    key: ("FLOAT", {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.01})
                    for key in CROSS_SETTINGS
                },
            },
        }

    RETURN_TYPES = (SETTINGS,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"

    def make(
        self,
        mix,
        depth=-2,
        iteration=FROM_MIX,
        routing_scope=FROM_MIX,
        routing_select=FROM_MIX,
        depth_schedule=None,
        candidates=0,
        candidate_summary="keys",
        routing_features=False,
        content_keys="per_expert",
        gate_stats="fused",
        cross_token_region=0.0,
        cross_token_prompt=0.0,
        cross_token_timestep=0.0,
        lycoris_apply="truncated",
    ):
        given: dict[str, Any] = {
            "mix": mix,
            "lycoris_apply": lycoris_apply,
            "depth_schedule": depth_schedule,
            "candidates": candidates,
            "candidate_summary": candidate_summary,
            "routing_features": routing_features,
            "content_keys": content_keys,
            "gate_stats": gate_stats,
            "cross_token_region": cross_token_region,
            "cross_token_prompt": cross_token_prompt,
            "cross_token_timestep": cross_token_timestep,
        }
        if depth != -2:
            given["depth"] = depth
        for key, value in (
            ("iteration", iteration),
            ("routing_scope", routing_scope),
            ("routing_select", routing_select),
        ):
            if value != FROM_MIX:
                given[key] = None if value == "mod" else value
        return (resolve_settings(given),)


def apply_inputs(
    settings: dict | None,
    depth: int,
    iteration: str | None,
    depth_schedule: Any,
    routing_scope: str | None,
    routing_select: str | None,
) -> dict:
    s = resolve_settings(settings)
    given = {
        "depth": depth if depth != DEFAULTS["depth"] else None,
        "iteration": iteration,
        "depth_schedule": depth_schedule,
        "routing_scope": routing_scope,
        "routing_select": routing_select,
    }
    s.update({k: v for k, v in given.items() if v is not None})
    return s
