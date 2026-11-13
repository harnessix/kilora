# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import torch

from .layout import Layout
from .sets import AdapterSet, new_keyed_set


@dataclass(frozen=True)
class LivePreset:
    name: str
    live_experts: int
    target_stride: int
    target_suffixes: tuple[str, ...]


_UNET_ATTENTION = tuple(
    f"{attn}.{proj}" for attn in ("attn1", "attn2") for proj in ("to_q", "to_k", "to_v", "to_out.0")
)

LIVE_PRESETS: dict[str, LivePreset] = {
    "light": LivePreset(
        "light",
        4,
        2,
        ("attention.to_v", "attention.to_out.0", "attn1.to_v", "attn1.to_out.0"),
    ),
    "full": LivePreset(
        "full",
        4,
        1,
        (
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            *_UNET_ATTENTION,
            "ff.net.0.proj",
            "ff.net.2",
        ),
    ),
    "standard": LivePreset(
        "standard",
        8,
        1,
        ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0")
        + _UNET_ATTENTION,
    ),
}


def default_preset_name() -> str:
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        if total >= 16 * 1024**3:
            return "standard"
    return "light"


def live_targets(host: AdapterSet, preset: LivePreset) -> list[str]:
    chosen: list[str] = []
    for suffix in preset.target_suffixes:
        named = [t for t in host.target_modules if t.endswith(suffix)]
        chosen.extend(named[:: preset.target_stride])
    if not chosen:
        chosen = host.target_modules[:: preset.target_stride]
    return [t for t in host.target_modules if t in chosen]


def make_live_set(
    host: AdapterSet,
    preset: LivePreset,
    family: str,
    name: str = "live",
    seed: int = 1,
    write_scale: float | None = None,
) -> tuple[AdapterSet, Layout]:
    targets = live_targets(host, preset)
    from .presets import PRESETS

    if family in PRESETS:
        patterns = PRESETS[family].full_target_modules or PRESETS[family].target_modules
        targets = [t for t in targets if any(fnmatch.fnmatchcase(t, p) for p in patterns)]
        if not targets:
            raise ValueError(f"the host set has no {family} layers for live experts")
    shapes = {t: (host.experts[t].in_features, host.experts[t].out_features) for t in targets}
    live = new_keyed_set(
        name,
        shapes,
        num_experts=preset.live_experts,
        routes=host.routes,
        rank=host.rank,
        router_dim=host.router_dim,
        router_seed=0,
        seed=seed,
        routers={w: r for w, r in host.routers.items() if w in {o for _, o in shapes.values()}},
        aggregation=host.aggregation,
    )
    live.extra["live"] = True
    if write_scale is None:
        ups = torch.cat([ex.up.reshape(-1).float() for ex in host.experts.values()])
        write_scale = float(ups.pow(2).mean().sqrt()) or 0.01
    gen = torch.Generator().manual_seed(int(seed) + 7_919)
    for ex in live.experts.values():
        ex.up = torch.randn(ex.up.shape, generator=gen) * write_scale
    layout = Layout(
        id=f"{family}-{preset.name}-1",
        targets=tuple(targets),
        widths=tuple(shapes[t][1] for t in targets),
        in_widths=tuple(shapes[t][1] for t in targets),
        experts=preset.live_experts,
        per_expert=host.routes * host.rank,
        key_dim=host.router_dim,
        write_scale=write_scale,
    )
    return live, layout
