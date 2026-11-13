# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import os
import weakref
from dataclasses import dataclass, field
from typing import Any

from . import patching
from .patching import ModChain, ModEntry

STATE_KEY = "kilora_dropin"
OUTER_SAMPLE = "outer_sample"
LORA_STACK = "LORA_STACK"
SETTINGS = "KILORA_SETTINGS"
DEPTH_MAP = "KILORA_DEPTH_MAP"
STACK = "stack"


class Comfy:
    def load_lora(self, path: str) -> Any:
        import comfy.utils

        return comfy.utils.load_torch_file(path, safe_load=True)

    def load_lora_for_models(self, model, clip, lora, strength_model, strength_clip):
        import comfy.sd

        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)


COMFY = Comfy()


def resolve(settings: Any) -> dict:
    try:
        from .settings import resolve_settings
    except ImportError:
        return dict(settings or {})
    return dict(resolve_settings(settings))


@dataclass
class DropInState:
    entries: tuple[ModEntry, ...] = ()
    settings: dict = field(default_factory=dict)
    cache: tuple | None = field(default=None, compare=False)
    builds: int = field(default=0, compare=False)

    def __deepcopy__(self, memo):
        return DropInState(self.entries, dict(self.settings))

    def routed(self, patcher: Any) -> tuple[Any, str]:
        mode = os.environ.get("KILORA_LYCORIS_APPLY") or self.settings.get(
            "lycoris_apply", "truncated"
        )
        if self.cache is not None and self.cache[0]() is patcher and self.cache[3] == mode:
            return self.cache[1], self.cache[2]
        from .nodes import COMFY_DEPTH_STEPS, live_env

        s = self.settings
        out, report = patching.apply(
            patcher,
            ModChain(self.entries),
            int(s.get("depth", -1)),
            live_env(),
            s.get("iteration"),
            s.get("depth_schedule"),
            s.get("routing_scope"),
            s.get("routing_select"),
            joint=True,
            timeline=s.get("timeline"),
            token_depth=s.get("token_depth"),
            axis_alpha=s.get("axis_alpha", "off"),
            alpha_clip=s.get("alpha_clip"),
            alpha_mask=s.get("alpha_mask"),
            expert_text=s.get("expert_text", "auto"),
            import_score=s.get("import_score"),
            lycoris_apply=s.get("lycoris_apply", "truncated"),
            gate_stats=s.get("gate_stats", "fused"),
            candidates=s.get("candidates", 0),
            candidate_summary=s.get("candidate_summary", "keys"),
            cross_weights=tuple(
                s.get(k, 0.0)
                for k in ("cross_token_region", "cross_token_prompt", "cross_token_timestep")
            ),
            depth_steps=COMFY_DEPTH_STEPS,
        )
        options = getattr(out, "model_options", None)
        if isinstance(options, dict):
            options.pop(STATE_KEY, None)
        self.cache = (weakref.ref(patcher), out, report, mode)
        self.builds += 1
        return out, report


def state_of(patcher: Any) -> DropInState | None:
    options = getattr(patcher, "model_options", None)
    if isinstance(options, dict):
        state = options.get(STATE_KEY)
        if isinstance(state, DropInState):
            return state
    return None


def outer_sample(executor, *args, **kwargs):

    guider = executor.class_obj
    source = guider.model_patcher
    state = state_of(source)
    if state is None or not state.entries:
        return executor(*args, **kwargs)
    routed, _ = state.routed(source)
    guider.model_patcher = routed
    try:
        return executor(*args, **kwargs)
    finally:
        guider.model_patcher = source


def _register(patcher: Any) -> None:
    if not hasattr(patcher, "add_wrapper_with_key"):
        return
    if hasattr(patcher, "remove_wrappers_with_key"):
        patcher.remove_wrappers_with_key(OUTER_SAMPLE, STATE_KEY)
    patcher.add_wrapper_with_key(OUTER_SAMPLE, STATE_KEY, outer_sample)


def _merged(entries) -> tuple[ModEntry, ...]:
    out: dict[str, ModEntry] = {}
    for e in entries:
        if e.path in out:
            out[e.path] = dataclasses.replace(
                out[e.path], strength=out[e.path].strength + e.strength
            )
        else:
            out[e.path] = e
    return tuple(out.values())


def _record(model: Any, entries: list[ModEntry], settings: dict) -> Any:
    out = model.clone()
    old = state_of(out)
    previous = old.entries if old is not None else ()
    merged = {**(old.settings if old is not None else {}), **settings}
    out.model_options[STATE_KEY] = DropInState(_merged((*previous, *entries)), merged)
    _register(out)
    return out


class KiLoRALoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import FOLDERS

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (FOLDERS.list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": {
                "lora_stack": (LORA_STACK,),
                "timeline": ("TIMELINE",),
                "settings": (SETTINGS,),
                "depth_map": (DEPTH_MAP,),
                "axis_alpha": (["off", "auto"], {"default": "off"}),
                "alpha_mask": ("MASK",),
                "expert_text": (["auto", "off"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "KiLoRA"

    def load_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        lora_stack=None,
        settings=None,
        timeline=None,
        depth_map=None,
        axis_alpha="off",
        alpha_mask=None,
        expert_text="auto",
    ):
        from .nodes import _file

        s = resolve(settings)
        if timeline:
            s["timeline"] = timeline
        items = [
            (name, float(sm), float(sc))
            for name, sm, sc in (lora_stack or [])
            if name and name != "None"
        ]
        items.append((lora_name, float(strength_model), float(strength_clip)))
        entries = []
        for name, sm, sc in items:
            if sm == 0 and sc == 0:
                continue
            path = _file("loras", name, "lora")
            lora = COMFY.load_lora(path)
            if s.get("mix") == STACK:
                model, clip = COMFY.load_lora_for_models(model, clip, lora, sm, sc)
                continue
            if clip is not None and sc != 0:
                clip, _ = patching.apply_clip(model, clip, [ModEntry("lora", path, sc, 1.0, True)])
            if sm != 0:
                entries.append(ModEntry("lora", path, sm, 1.0, True))
        if entries:
            kept = {k: v for k, v in s.items() if k != "mix"}
            kept["expert_text"] = expert_text
            kept["axis_alpha"] = axis_alpha
            if axis_alpha == "auto":
                kept["alpha_clip"] = clip
            if depth_map is not None:
                kept["token_depth"] = depth_map
            if alpha_mask is not None:
                kept["alpha_mask"] = alpha_mask
            model = _record(model, entries, kept)
        return (model, clip)
