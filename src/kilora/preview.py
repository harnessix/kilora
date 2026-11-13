# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

from .adapter import compose
from .compute import generation
from .modfile import load_mod

DEFAULT_STRENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0)
DEFAULT_DEPTHS = (1, 2, 3)


def preview_grid(
    pipe: Any,
    mod_path: str | os.PathLike[str],
    prompt: str,
    *,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    seed: int = 1234,
    steps: int = 8,
    guidance: float = 0.0,
    size: int = 1024,
    label: bool = True,
):
    from PIL import Image, ImageDraw, ImageFont

    mod = load_mod(mod_path)
    base_strengths = [s.strength for s in mod.sets]
    adapter = compose(pipe.transformer, [mod_path])
    carry = adapter.settings.carry_across_steps
    cells: list[list[Any]] = []
    try:
        for depth in depths:
            row = []
            for strength in strengths:
                adapter.set_strengths([b * strength for b in base_strengths])
                with (
                    adapter.depth(depth if strength != 0 else 0),
                    generation(pipe.transformer, carry=carry),
                ):
                    image = _generate(pipe, prompt, steps, guidance, size, seed)
                row.append(image.convert("RGB"))
            cells.append(row)
    finally:
        adapter.remove()
    w, h = cells[0][0].size
    pad = max(16, h // 12) if label else 0
    grid = Image.new("RGB", (pad * 3 + w * len(strengths), pad + h * len(depths)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=max(12, int(pad * 0.6))) if label else None
    for r, row in enumerate(cells):
        for c, img in enumerate(row):
            grid.paste(img, (pad * 3 + c * w, pad + r * h))
        if label:
            draw.text((4, pad + r * h + h // 2), f"depth {depths[r]}", fill="black", font=font)
    if label:
        for c, strength in enumerate(strengths):
            draw.text((pad * 3 + c * w + 4, 2), f"strength {strength:+g}", fill="black", font=font)
    return grid


def _generate(pipe: Any, prompt: str, steps: int, guidance: float, size: int, seed: int):
    from .anima import generate, is_anima

    if is_anima(pipe):
        return generate(pipe, prompt, steps=steps, guidance=guidance, size=size, seed=seed)
    out = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=size,
        width=size,
        generator=torch.Generator().manual_seed(seed),
    )
    return out.images[0]


def save_png(image: Any, path: str | os.PathLike[str]) -> None:
    image.save(os.fspath(path), format="PNG", optimize=False)


SHOWCASE_PROMPTS = (
    "a lighthouse on a rocky coast at dusk",
    "a portrait of an old fisherman with a knitted cap",
    "a bowl of oranges on a wooden table by a window",
    "a quiet city street after rain, with a red umbrella",
)
SHOWCASE_SEEDS = (11, 22, 33, 44)
SHOWCASE_STRENGTHS = (0.0, 0.5, 1.0, 1.5)
SHOWCASE_FORMAT = "kilora-showcase"


def showcase_prompt(prompt: str, trigger: list[str]) -> str:
    return f"{prompt}, {', '.join(trigger)}" if trigger else prompt


def render_showcase(
    pipe: Any,
    mod_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    prompts: Sequence[str] = SHOWCASE_PROMPTS,
    seeds: Sequence[int] = SHOWCASE_SEEDS,
    strengths: Sequence[float] = SHOWCASE_STRENGTHS,
    steps: int = 8,
    guidance: float = 0.0,
    size: int = 640,
) -> dict[str, Any]:

    import json

    from PIL import Image

    from ._version import __version__
    from .card import card_fields
    from .modfile import read_manifest

    if len(prompts) != len(seeds):
        raise ValueError("give one seed per prompt")
    manifest = read_manifest(mod_path)
    card = card_fields(manifest)
    mod = load_mod(mod_path)
    base_strengths = [s.strength for s in mod.sets]
    depth = int(card["depth"] or 1)
    full = [showcase_prompt(p, card["trigger"]) for p in prompts]
    adapter = compose(pipe.transformer, [mod_path])
    carry = adapter.settings.carry_across_steps
    grid = Image.new("RGB", (size * len(strengths), size * len(full)), "white")
    try:
        for r, (prompt, seed) in enumerate(zip(full, seeds, strict=True)):
            for c, strength in enumerate(strengths):
                adapter.set_strengths([b * strength for b in base_strengths])
                with (
                    adapter.depth(depth if strength != 0 else 0),
                    generation(pipe.transformer, carry=carry),
                ):
                    image = _generate(pipe, prompt, steps, guidance, size, seed)
                tile = image.convert("RGB")
                if tile.size != (size, size):
                    tile = tile.resize((size, size))
                grid.paste(tile, (c * size, r * size))
    finally:
        adapter.remove()
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    grid_path = os.path.join(out, "grid.png")
    save_png(grid, grid_path)
    record = {
        "format": SHOWCASE_FORMAT,
        "format_version": "1.0",
        "mod": {
            "id": card["id"],
            "name": card["name"],
            "version": card["version"],
            "file": os.path.basename(os.fspath(mod_path)),
            "digest": manifest.get("digest"),
        },
        "base_model": card["base_model"]["family"],
        "license": card["license"],
        "rows": [{"prompt": p, "seed": s} for p, s in zip(full, seeds, strict=True)],
        "columns": [{"strength": float(s)} for s in strengths],
        "tile": size,
        "depth": depth,
        "steps": steps,
        "guidance": guidance,
        "created_with": f"kilora {__version__}",
    }
    with open(os.path.join(out, "render.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return record
