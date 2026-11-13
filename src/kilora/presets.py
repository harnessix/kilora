# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import KiLoRAConfig


@dataclass(frozen=True)
class Alias:
    targets: tuple[str, ...]


@dataclass(frozen=True)
class Preset:
    family: str
    denoiser: str
    target_modules: tuple[str, ...]
    router_seed: int
    full_target_modules: tuple[str, ...] = ()
    alias_rules: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)
    rename: Callable[[str], str | None] | None = None
    prediction: str = ""
    latent_format: str = ""
    flow_shift: float | None = None
    resolution: int | None = None
    steps: int | None = None
    guidance: float | None = None
    notice: str = ""
    content_blocks: int = 64
    content_block_offsets: tuple[tuple[str, int], ...] = ()
    functional_words: tuple[str, ...] = (
        "detail",
        "enhancer",
        "enhance",
        "fix",
        "correction",
        "lighting",
        "sharpness",
    )

    def content_group(self, layer: str, width: int, bands: int = 4) -> str | None:
        if width < 1 or bands < 1:
            raise ValueError("width and bands must be positive")
        matches = [p for p in self.full_target_modules if fnmatch.fnmatchcase(layer, p)]
        if not matches:
            return None
        numbers = re.findall(r"(?:^|\.)(\d+)(?=\.)", layer)
        index = int(numbers[0]) if numbers else 0
        for prefix, offset in self.content_block_offsets:
            if layer.startswith(prefix + "."):
                index += offset
                break
        band = min(bands - 1, index * bands // self.content_blocks)
        name = matches[0].lower()
        if any(p in name for p in ("ff.", "ffn", "mlp", "feed_forward")):
            output = any(
                p in name
                for p in (
                    "net.2",
                    "mlp.2",
                    "ffn.2",
                    "ffn_down",
                    "layer2",
                    "down_proj",
                    "mlp_out",
                    ".w2",
                )
            )
            kind = "mlp-out" if output else "mlp-in"
        elif any(
            p in name
            for p in (
                "to_out",
                "attn.proj",
                "attn_out",
                "output_proj",
                ".proj.o",
                "_attn.o",
                ".o_proj",
            )
        ):
            kind = "attention-out"
        else:
            kind = "attention-in"
        return f"{band}-{kind}-{width}"

    def config(self, **overrides) -> KiLoRAConfig:
        values = {"target_modules": list(self.target_modules), "router_seed": self.router_seed}
        values.update(overrides)
        return KiLoRAConfig(**values)

    def aliases(self, module_names: list[str]) -> dict[str, Alias]:
        present = set(module_names)
        out: dict[str, Alias] = {}
        for pattern, replacements in self.alias_rules:
            regex = re.compile(pattern)
            for name in module_names:
                match = regex.fullmatch(name)
                if match is None:
                    continue
                prefix = match.group("prefix")
                alias = prefix + replacements[0]
                targets = tuple(prefix + r for r in replacements[1:])
                if all(t in present for t in targets):
                    out[alias] = Alias(targets)
        if self.rename is not None:
            for name in module_names:
                other = self.rename(name)
                if other is not None and other != name and other not in present:
                    out[other] = Alias((name,))
        return out


_UNET_BLOCK = re.compile(
    r"(?:down_blocks\.(?P<d>\d+)|up_blocks\.(?P<u>\d+)|mid_block)\.attentions\.(?P<j>\d+)\.(?P<rest>.+)"
)


def ldm_unet_name(name: str) -> str | None:
    m = _UNET_BLOCK.fullmatch(name)
    if m is None:
        return None
    j, rest = int(m["j"]), m["rest"]
    if m["d"] is not None:
        return f"input_blocks.{3 * int(m['d']) + j + 1}.1.{rest}"
    if m["u"] is not None:
        return f"output_blocks.{3 * int(m['u']) + j}.1.{rest}"
    return f"middle_block.1.{rest}" if j == 0 else None


_UNET_ATTENTION = tuple(
    f"*.transformer_blocks.*.{attn}.{proj}"
    for attn in ("attn1", "attn2")
    for proj in ("to_q", "to_k", "to_v", "to_out.0")
)
_UNET_FULL = _UNET_ATTENTION + (
    "*.transformer_blocks.*.ff.net.0.proj",
    "*.transformer_blocks.*.ff.net.2",
)


_ANIMA_LAYERS = (
    ("attn1.to_q", "self_attn.q_proj"),
    ("attn1.to_k", "self_attn.k_proj"),
    ("attn1.to_v", "self_attn.v_proj"),
    ("attn1.to_out.0", "self_attn.output_proj"),
    ("attn2.to_q", "cross_attn.q_proj"),
    ("attn2.to_k", "cross_attn.k_proj"),
    ("attn2.to_v", "cross_attn.v_proj"),
    ("attn2.to_out.0", "cross_attn.output_proj"),
    ("ff.net.0.proj", "mlp.layer1"),
    ("ff.net.2", "mlp.layer2"),
)
_ANIMA_BLOCK = re.compile(r"transformer_blocks\.(?P<i>\d+)\.(?P<rest>.+)")


def anima_name(name: str) -> str | None:
    m = _ANIMA_BLOCK.fullmatch(name)
    if m is None:
        return None
    for diffusers, original in _ANIMA_LAYERS:
        if m["rest"] == diffusers:
            return f"blocks.{m['i']}.{original}"
    return None


ANIMA_NOTICE = (
    "This model is a Derivative of Anima by CircleStone Labs and Comfy Org, licensed "
    "under the CircleStone Labs Non-Commercial License v1.2 "
    "(https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md), and of "
    "NVIDIA Cosmos-Predict2-2B-Text2Image under the NVIDIA Open Model License. It was "
    "modified by training with KiLoRA. Non-commercial use only; images it makes may be "
    "used commercially."
)

PRESETS: dict[str, Preset] = {
    "z-image": Preset(
        family="z-image",
        content_blocks=30,
        denoiser="transformer",
        target_modules=(
            "layers.*.attention.to_q",
            "layers.*.attention.to_k",
            "layers.*.attention.to_v",
            "layers.*.attention.to_out.0",
        ),
        full_target_modules=(
            "layers.*.attention.to_q",
            "layers.*.attention.to_k",
            "layers.*.attention.to_v",
            "layers.*.attention.to_out.0",
            "layers.*.feed_forward.w1",
            "layers.*.feed_forward.w2",
            "layers.*.feed_forward.w3",
        ),
        router_seed=0,
        alias_rules=(
            (
                r"(?P<prefix>.*\.attention\.)to_q",
                ("qkv", "to_q", "to_k", "to_v"),
            ),
            (r"(?P<prefix>.*\.attention\.)to_out\.0", ("out", "to_out.0")),
        ),
    ),
    "sd15": Preset(
        family="sd15",
        content_blocks=12,
        denoiser="unet",
        target_modules=_UNET_ATTENTION,
        full_target_modules=_UNET_FULL,
        router_seed=15,
        rename=ldm_unet_name,
    ),
    "anima": Preset(
        family="anima",
        content_blocks=28,
        denoiser="transformer",
        target_modules=tuple(f"transformer_blocks.*.{d}" for d, _ in _ANIMA_LAYERS),
        full_target_modules=tuple(f"transformer_blocks.*.{d}" for d, _ in _ANIMA_LAYERS),
        router_seed=40,
        rename=anima_name,
        prediction="flow",
        latent_format="wan21",
        flow_shift=3.0,
        resolution=1024,
        steps=30,
        guidance=4.0,
        notice=ANIMA_NOTICE,
    ),
    "sdxl": Preset(
        family="sdxl",
        content_blocks=12,
        denoiser="unet",
        target_modules=_UNET_ATTENTION,
        full_target_modules=_UNET_FULL,
        router_seed=20,
        rename=ldm_unet_name,
    ),
}


def get_preset(family: str) -> Preset:
    try:
        return PRESETS[family]
    except KeyError:
        raise KeyError(
            f"no preset for base model family {family!r}; known: {sorted(PRESETS)}"
        ) from None


def pack_license(base_family: str, given: str | None = None) -> tuple[str, str | None]:

    preset = PRESETS.get(base_family)
    notice = preset.notice if preset is not None and preset.notice else None
    licenses = {"anima": "LicenseRef-CircleStone-Labs-NC-1.2"}
    return given or licenses.get(base_family, "Apache-2.0"), notice
