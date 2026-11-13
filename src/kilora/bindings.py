# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

COMFY_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class Rule:
    diffusers: str
    point: str
    comfy: str
    part: tuple[int, int] | None = None


@dataclass(frozen=True)
class Place:
    path: str
    part: tuple[int, int] | None


def _regex(template: str) -> re.Pattern[str]:
    out = ""
    for piece in re.split(r"(\{\w+\})", template):
        if piece == "{i}":
            out += r"(?P<i>\d+)"
        elif piece == "{path}":
            out += r"(?P<path>.+)"
        else:
            out += re.escape(piece)
    return re.compile(out)


@dataclass(frozen=True)
class Binding:
    family: str
    rules: tuple[Rule, ...]
    comfy_markers: tuple[str, ...] = ()

    def _match(self, name: str, side: str) -> tuple[Rule, dict[str, str]] | None:
        for rule in self.rules:
            m = _regex(getattr(rule, side)).fullmatch(name)
            if m is not None:
                return rule, m.groupdict()
        return None

    def hook_point(self, target: str) -> str:
        found = self._match(target, "diffusers")
        if found is None:
            return f"base::{target}"
        rule, groups = found
        return rule.point.format(**groups)

    def to_comfy(self, target: str) -> Place:
        found = self._match(target, "diffusers")
        if found is None:
            raise KeyError(f"{target!r} has no ComfyUI name in the {self.family} binding")
        rule, groups = found
        return Place(COMFY_PREFIX + rule.comfy.format(**groups), rule.part)

    def from_comfy(self, comfy_path: str) -> list[tuple[str, tuple[int, int] | None]]:
        out = []
        for rule in self.rules:
            m = _regex(rule.comfy).fullmatch(comfy_path)
            if m is not None:
                out.append((rule.diffusers.format(**m.groupdict()), rule.part))
        return out

    def matches_comfy(self, module_names: Iterable[str]) -> bool:
        names = set(module_names)
        return bool(self.comfy_markers) and all(
            any(_regex(marker).fullmatch(n) for n in names) for marker in self.comfy_markers
        )


def _z_image_rules() -> tuple[Rule, ...]:
    rules = []
    for prefix, block in (
        ("layers", "block"),
        ("noise_refiner", "noise_block"),
        ("context_refiner", "context_block"),
    ):
        for index, proj in enumerate(("q", "k", "v")):
            rules.append(
                Rule(
                    f"{prefix}.{{i}}.attention.to_{proj}",
                    f"base::{block}.{{i}}.attn.{proj}",
                    f"{prefix}.{{i}}.attention.qkv",
                    (index, 3),
                )
            )
        rules.append(
            Rule(
                f"{prefix}.{{i}}.attention.to_out.0",
                f"base::{block}.{{i}}.attn.out",
                f"{prefix}.{{i}}.attention.out",
            )
        )
        for w in ("w1", "w2", "w3"):
            rules.append(
                Rule(
                    f"{prefix}.{{i}}.feed_forward.{w}",
                    f"base::{block}.{{i}}.ff.{w}",
                    f"{prefix}.{{i}}.feed_forward.{w}",
                )
            )
    return tuple(rules)


def _unet_rules() -> tuple[Rule, ...]:
    layers = [
        (f"{attn}.{proj}", f"{attn}.{short}")
        for attn in ("attn1", "attn2")
        for proj, short in (("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "out"))
    ] + [("ff.net.0.proj", "ff.in"), ("ff.net.2", "ff.out")]
    blocks = [("mid_block.attentions.0", "middle.1", "middle_block.1")]
    for level in range(4):
        for j in range(3):
            n = 3 * level + j + 1
            blocks.append(
                (f"down_blocks.{level}.attentions.{j}", f"input.{n}", f"input_blocks.{n}.1")
            )
            n = 3 * level + j
            blocks.append(
                (f"up_blocks.{level}.attentions.{j}", f"output.{n}", f"output_blocks.{n}.1")
            )
    return tuple(
        Rule(
            f"{diffusers}.transformer_blocks.{{i}}.{layer}",
            f"base::{point}.block.{{i}}.{short}",
            f"{comfy}.transformer_blocks.{{i}}.{layer}",
        )
        for diffusers, point, comfy in blocks
        for layer, short in layers
    )


def _anima_rules() -> tuple[Rule, ...]:
    from .presets import _ANIMA_LAYERS

    shorts = {"attn1": "self", "attn2": "cross", "ff.net.0.proj": "mlp.in", "ff.net.2": "mlp.out"}
    rules = []
    for diffusers, original in _ANIMA_LAYERS:
        if diffusers.startswith("ff."):
            point = shorts[diffusers]
        else:
            attn, proj = diffusers.split(".", 1)
            point = f"{shorts[attn]}.{proj.replace('to_', '').replace('out.0', 'out')}"
        rules.append(
            Rule(
                f"transformer_blocks.{{i}}.{diffusers}",
                f"base::block.{{i}}.{point}",
                f"blocks.{{i}}.{original}",
            )
        )
    return tuple(rules)


BINDINGS: dict[str, Binding] = {
    "z-image": Binding(
        family="z-image",
        rules=_z_image_rules(),
        comfy_markers=("layers.{i}.attention.qkv", "layers.{i}.feed_forward.w3"),
    ),
    "sd15": Binding(
        family="sd15",
        rules=_unet_rules(),
        comfy_markers=("input_blocks.1.1.transformer_blocks.{i}.attn1.to_q",),
    ),
    "sdxl": Binding(
        family="sdxl",
        rules=_unet_rules(),
        comfy_markers=("label_emb.0.0", "input_blocks.4.1.transformer_blocks.{i}.attn1.to_q"),
    ),
    "anima": Binding(
        family="anima",
        rules=_anima_rules(),
        comfy_markers=("blocks.{i}.cross_attn.output_proj", "llm_adapter.out_proj"),
    ),
    "kilora-tiny-1": Binding(
        family="kilora-tiny-1",
        rules=(Rule("{path}", "base::{path}", "{path}"),),
    ),
}


def get_binding(family: str) -> Binding:
    try:
        return BINDINGS[family]
    except KeyError:
        raise KeyError(f"no binding for base model family {family!r}") from None
