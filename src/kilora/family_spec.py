# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .bindings import COMFY_PREFIX, Binding, Place

__all__ = [
    "FamilySpec",
    "PatternBinding",
    "TokenLayout",
    "discover_linear",
    "linear_shape",
    "map_region",
    "normalize_lora",
    "prediction_of",
]

LAYOUT_KINDS = ("image2d", "video3d", "audio1d", "text1d", "set")

GENERIC_EXCLUDE = (
    "*adaLN*",
    "*adaln*",
    "*_mod.*",
    "*_mod",
    "*modulation*",
    "*embed*",
    "*adapter*",
    "*time_in*",
    "*vector_in*",
    "*guidance_in*",
    "final_layer*",
    "*lm_head*",
)


@dataclass(frozen=True)
class TokenLayout:
    kind: str
    patch: tuple[int, ...] | None = None
    text: str = ""
    pad: int = 1
    role: str = ""

    def __post_init__(self) -> None:
        if self.kind not in LAYOUT_KINDS:
            raise ValueError(f"unknown token layout {self.kind!r}; known: {LAYOUT_KINDS}")
        if self.text not in ("", "before", "after", "end", "prefix"):
            raise ValueError(
                f"text must be '', 'before', 'after', 'end' or 'prefix', not {self.text!r}"
            )
        if self.role not in ("", "reference"):
            raise ValueError(f"role must be '' or 'reference', not {self.role!r}")

    def grid(self, latent: Sequence[int]) -> tuple[int, ...]:
        if self.patch is None:
            raise ValueError("this layout finds its grid from the token count")
        latent = tuple(int(s) for s in latent)
        if self.kind == "image2d":
            dims = latent[-2:]
        elif self.kind == "video3d":
            dims = (1, *latent[-2:]) if len(latent) < 3 else latent[-3:]
        elif self.kind == "audio1d":
            dims = latent[-1:]
        else:
            raise ValueError("text tokens have no grid")
        if len(dims) != len(self.patch):
            raise ValueError(f"patch {self.patch} does not fit the latent {latent}")
        if self.text == "prefix":
            return tuple(d // p for d, p in zip(dims, self.patch, strict=True))
        return tuple(math.ceil(d / p) for d, p in zip(dims, self.patch, strict=True))

    def image_rows(
        self, tokens: int, latent: Sequence[int], *, text_tokens: int | None = None
    ) -> slice:
        grid = self.grid(latent)
        n = math.prod(grid)
        n_pad = math.ceil(n / self.pad) * self.pad
        if tokens < n_pad:
            shape = "x".join(map(str, grid))
            raise ValueError(f"a {shape} token grid does not fit {tokens} tokens")
        text = tokens - n_pad
        if (
            self.text != "prefix"
            and self.pad == 1
            and text_tokens is not None
            and 0 < text_tokens <= tokens - n
        ):
            text = int(text_tokens)
        start = text if self.text in ("before", "prefix") else 0
        return slice(start, start + n)


def _pool(mask: torch.Tensor, layout: TokenLayout, grid: tuple[int, ...]) -> torch.Tensor:
    m = mask.float()
    if layout.kind == "audio1d":
        if m.dim() == 1:
            m = m[None]
        cells = m if m.shape[-1] == grid[0] else F.adaptive_avg_pool1d(m[:, None], grid)[:, 0]
        return cells.reshape(m.shape[0], -1).clamp(0, 1)
    if layout.kind == "video3d" and grid[0] > 1:
        if m.dim() == 2:
            m = m[None, None]
        elif m.dim() == 3:
            m = m[None]
        cells = F.adaptive_avg_pool3d(m[:, None], grid)[:, 0]
        return cells.reshape(m.shape[0], -1).clamp(0, 1)
    if m.dim() == 4:
        m = m.mean(1)
    if m.dim() == 2:
        m = m[None]
    h, w = grid[-2:]
    cells = m if tuple(m.shape[-2:]) == (h, w) else F.adaptive_avg_pool2d(m[:, None], (h, w))[:, 0]
    return cells.reshape(m.shape[0], h * w).clamp(0, 1)


def map_region(
    mask: torch.Tensor,
    tokens: int,
    layout: TokenLayout,
    latent: Sequence[int],
    *,
    text_tokens: int | None = None,
    text_value: float = 0.0,
) -> torch.Tensor | None:

    if layout.kind == "set":
        return None
    if layout.kind == "text1d" or layout.role == "reference":
        return torch.zeros(1, tokens)
    grid = layout.grid(latent)
    n = math.prod(grid)
    image = _pool(mask, layout, grid)
    n_pad = math.ceil(n / layout.pad) * layout.pad
    selected = layout.image_rows(tokens, latent, text_tokens=text_tokens)
    rows = image.shape[0]
    if not layout.text:
        return torch.cat([image, image.new_zeros(rows, tokens - n)], -1)
    text = selected.start if layout.text in ("before", "prefix") else tokens - n_pad
    if (
        layout.text in ("after", "end")
        and layout.pad == 1
        and text_tokens is not None
        and 0 < text_tokens <= tokens - n
    ):
        text = int(text_tokens)
    span = image.new_full((rows, text), float(text_value))
    if layout.text in ("before", "prefix"):
        parts = [span, image, image.new_zeros(rows, tokens - text - n)]
    elif layout.text == "end":
        parts = [image, image.new_zeros(rows, tokens - n - text), span]
    else:
        pad = image.new_zeros(rows, n_pad - n)
        parts = [image, pad, span, image.new_zeros(rows, tokens - n_pad - text)]
    return torch.cat(parts, -1)


_PREDICTIONS = (
    ("V_PREDICTION", "v"),
    ("V_PREDICTION_EDM", "v"),
    ("V_PREDICTION_CONTINUOUS", "v"),
    ("V_PREDICTION_DDPM", "v"),
    ("CONST", "flow"),
    ("COSMOS_RFLOW", "flow"),
    ("IMG_TO_IMG_FLOW", "flow"),
    ("X0", "x0"),
    ("EPS", "eps"),
)


def prediction_of(model: Any) -> str:
    sampling = getattr(model, "model_sampling", None)
    if sampling is None:
        return ""
    names = [c.__name__ for c in type(sampling).__mro__]
    for name, prediction in _PREDICTIONS:
        if name in names:
            return prediction
    return ""


def linear_shape(module: nn.Module) -> tuple[int, int] | None:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if not type(module).__name__.endswith("Linear"):
        return None
    n_in, n_out = getattr(module, "in_features", None), getattr(module, "out_features", None)
    if isinstance(n_in, int) and isinstance(n_out, int):
        return n_in, n_out
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "tensor_shape", None) or getattr(weight, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1]), int(shape[0])
    return None


def _is_linear(module: nn.Module) -> bool:
    return linear_shape(module) is not None


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def discover_linear(
    module: nn.Module,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[str]:

    out = []
    for name, m in module.named_modules():
        if not name or not _is_linear(m):
            continue
        if include:
            if not _matches(name, include):
                continue
        elif not any(part.isdigit() for part in name.split(".")):
            continue
        if _matches(name, exclude) or (not include and _matches(name, GENERIC_EXCLUDE)):
            continue
        out.append(name)
    return out


LayerRows = tuple[str, tuple[int, int] | None]
NameMap = Callable[[Sequence[str], Mapping[str, tuple[int, int]]], dict[str, LayerRows]]

DIFFUSION_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "base_model.model.",
    "unet.",
)
DIFFUSION_KOHYA = ("lora_unet_", "lycoris_", "lora_transformer_")
TEXT_PREFIXES = (
    r"text_encoders\.[^.]+\.transformer\.model\.",
    r"text_encoders\.[^.]+\.transformer\.",
    r"text_encoder\.model\.",
    r"text_encoder\.",
    r"base_model\.model\.model\.",
    r"base_model\.model\.",
    r"model\.",
)
TEXT_KOHYA = ("lora_te_", "lora_te1_", "lora_te2_")


@dataclass(frozen=True)
class FamilySpec:
    family: str
    naming: str = "comfy"
    root: str = "diffusion_model"
    classes: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    layouts: tuple[tuple[str, TokenLayout], ...] = ()
    image_token_target: str = ""
    names: NameMap | None = None
    router_seed: int = 0
    content_blocks: int = 64
    content_block_offsets: tuple[tuple[str, int], ...] = ()
    markers: tuple[str, ...] = ()
    description: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    comfy_path: Callable[[str], str | None] | None = None
    diffusers_path: Callable[[str], str | None] | None = None

    solve_sampling: Mapping[str, int | float | str] | None = None

    def __post_init__(self):
        if self.solve_sampling is not None:
            object.__setattr__(self, "solve_sampling", MappingProxyType(dict(self.solve_sampling)))

    def targets(self, module: nn.Module) -> list[str]:
        return discover_linear(module, self.include, self.exclude)

    def is_target(self, path: str) -> bool:
        if _matches(path, self.exclude):
            return False
        if self.include:
            return _matches(path, self.include)
        return any(p.isdigit() for p in path.split(".")) and not _matches(path, GENERIC_EXCLUDE)

    def is_excluded(self, path: str) -> bool:
        return _matches(path, self.exclude) or _matches(path, GENERIC_EXCLUDE)

    def is_excluded_kohya(self, body: str) -> bool:
        patterns = [p.replace(".", "_") for p in (*self.exclude, *GENERIC_EXCLUDE)]
        return _matches(body, patterns)

    def layout(self, target: str) -> TokenLayout | None:
        for pattern, layout in self.layouts:
            if fnmatch.fnmatchcase(target, pattern):
                return layout
        return None

    def image_token_rows(self, tokens: int, latent_hw: tuple[int, ...]) -> slice | None:

        layout = self.layout(self.image_token_target) if self.image_token_target else None
        if layout is None or layout.kind not in ("image2d", "video3d") or layout.patch is None:
            return None
        return layout.image_rows(tokens, latent_hw)

    def matches_names(self, paths: Sequence[str]) -> bool:
        excluded = self.extra.get("exclude_names", ())
        if any(fnmatch.fnmatchcase(p, pattern) for p in paths for pattern in excluded):
            return False
        return bool(self.markers) and all(
            any(fnmatch.fnmatchcase(p, m) for p in paths) for m in self.markers
        )

    def identifies(self, module: nn.Module) -> bool:
        names = {c.__name__ for c in type(module).__mro__}
        return bool(self.classes) and any(c in names for c in self.classes)

    def prefixes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.root == "text_encoder":
            return TEXT_PREFIXES, TEXT_KOHYA
        return tuple(re.escape(p) for p in DIFFUSION_PREFIXES), DIFFUSION_KOHYA

    def lora_names(self, shapes: Mapping[str, tuple[int, int]]) -> dict[str, LayerRows]:
        paths = list(shapes)
        out: dict[str, LayerRows] = {p: (p, None) for p in paths}
        if self.names is not None:
            for name, where in self.names(paths, shapes).items():
                out.setdefault(name, where)
        return out


class PatternBinding(Binding):
    def __init__(self, spec: FamilySpec) -> None:
        super().__init__(family=spec.family, rules=())
        object.__setattr__(self, "spec", spec)

    def hook_point(self, target: str) -> str:
        return f"base::{target}"

    def to_comfy(self, target: str) -> Place:
        if not self.spec.is_target(target):
            raise KeyError(f"{target!r} is not a target of the {self.family} family")
        return Place(COMFY_PREFIX + target, None)

    def from_comfy(self, comfy_path: str) -> list[tuple[str, tuple[int, int] | None]]:
        return [(comfy_path, None)] if self.spec.is_target(comfy_path) else []

    def matches_comfy(self, module_names) -> bool:
        return False


_PARTS = (
    (".lora_A.default.weight", "down"),
    (".lora_B.default.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora.down.weight", "down"),
    (".lora.up.weight", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".alpha", "alpha"),
)
_OTHER = (
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".dora_scale",
    ".diff",
    ".diff_b",
)


def _split(key: str) -> tuple[str, str, str] | None:
    for suffix, part in _PARTS:
        if key.endswith(suffix):
            return key[: -len(suffix)], part, suffix
    for suffix in _OTHER:
        if key.endswith(suffix):
            return key[: -len(suffix)], "other", suffix
    return None


class Normalized(NamedTuple):
    state: dict[str, torch.Tensor]
    skipped: list[str]
    fixed: dict[str, tuple[torch.Tensor, torch.Tensor]]
    fixed_keys: int = 0
    fixed_dense: dict[str, torch.Tensor] = {}
    origins: dict[str, list[str]] = {}
    fixed_originals: list[str] = []
    fixed_original_paths: dict[str, str] = {}


def normalize_lora(
    spec: FamilySpec,
    state: Mapping[str, torch.Tensor],
    shapes: Mapping[str, tuple[int, int]],
    other_paths: Sequence[str] = (),
) -> Normalized:

    dotted, kohya = spec.prefixes()
    strip = re.compile("^(?:" + "|".join(dotted) + ")") if dotted else None
    names = spec.lora_names(shapes)
    underscored: dict[str, LayerRows | None] = {}
    for name, where in names.items():
        u = name.replace(".", "_")
        underscored[u] = None if u in underscored and underscored[u] != where else where

    for p in other_paths:
        names.setdefault(p, (p, None))
        u = p.replace(".", "_")
        underscored[u] = None if u in underscored and underscored[u] != (p, None) else (p, None)

    def lookup(stem: str) -> tuple[LayerRows | None, str | None]:
        for k in kohya:
            if stem.startswith(k):
                body = stem[len(k) :]
                return underscored.get(body), body
        body = strip.sub("", stem, count=1) if strip is not None else stem
        return names.get(body), body

    out: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    sliced: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    fixed_parts: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    origins: dict[str, list[str]] = {}
    fixed_stems: dict[str, list[str]] = {}
    slice_origins: dict[str, dict[str, list[str]]] = {}
    for key, tensor in state.items():
        split = _split(key)
        if split is None:
            out[key] = tensor
            origins.setdefault(key, []).append(key)
            continue
        stem, part, suffix = split
        where, body = lookup(stem)
        if where is None:
            excluded = body is not None and (spec.is_excluded(body) or spec.is_excluded_kohya(body))
            if excluded or spec.root == "text_encoder":
                skipped.append(key)
            else:
                out[key] = tensor
                origins.setdefault(key, []).append(key)
            continue
        path, rows = where
        if spec.is_excluded(path):
            if rows is None:
                name = suffix[1:] if part == "other" else part
                fixed_parts.setdefault(path, {}).setdefault(stem, {})[name] = tensor
                fixed_stems.setdefault(stem, []).append(key)
            else:
                skipped.append(key)
        elif rows is None:
            canonical = f"{COMFY_PREFIX}{path}{suffix}"
            out[canonical] = tensor
            origins.setdefault(canonical, []).append(key)
        elif part == "other":
            skipped.append(key)
        else:
            entry = f"{rows[0]}:{rows[1]}:{stem}"
            sliced.setdefault(path, {}).setdefault(entry, {})[part] = tensor
            slice_origins.setdefault(path, {}).setdefault(entry, []).append(key)
    for path, entries in sliced.items():
        if f"{COMFY_PREFIX}{path}.lora_down.weight" in out:
            raise ValueError(f"{path}: the LoRA changes the layer whole and in slices")
        n_out = shapes[path][1]
        downs, ups, accepted = [], [], []
        for entry_key, parts in entries.items():
            offset, size = (int(v) for v in entry_key.split(":", 2)[:2])
            if "down" not in parts or "up" not in parts:
                skipped.extend(slice_origins[path][entry_key])
                continue
            accepted.extend(slice_origins[path][entry_key])
            down, up = parts["down"].float(), parts["up"].float()
            rank = down.shape[0]
            scale = float(parts["alpha"].float().reshape(-1)[0]) / rank if "alpha" in parts else 1.0
            if up.shape[0] != size:
                raise ValueError(f"{path}: a slice of {size} rows got {up.shape[0]} up rows")
            full = up.new_zeros(n_out, rank)
            full[offset : offset + size] = up * scale
            downs.append(down)
            ups.append(full)
        if not downs:
            continue
        down = torch.cat(downs, 0)
        out[f"{COMFY_PREFIX}{path}.lora_down.weight"] = down
        out[f"{COMFY_PREFIX}{path}.lora_up.weight"] = torch.cat(ups, 1)
        origins[f"{COMFY_PREFIX}{path}.lora_down.weight"] = accepted
        if any("alpha" in p for p in entries.values()):
            out[f"{COMFY_PREFIX}{path}.alpha"] = torch.tensor(float(down.shape[0]))
    fixed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    dense: dict[str, torch.Tensor] = {}
    fixed_originals = []
    fixed_original_paths = {}
    for path, stems in fixed_parts.items():
        downs, ups = [], []
        for stem, parts in stems.items():
            if "down" not in parts or "up" not in parts:
                from .multi_import import _lycoris_used_parts, _split_any, lycoris_delta

                found = lycoris_delta(parts)
                if found is not None:
                    dense[path] = dense[path] + found[0] if path in dense else found[0]
                    used_parts = _lycoris_used_parts(parts, found[1])
                    for original in fixed_stems[stem]:
                        part = _split_any(original)[1]
                        used = part in used_parts
                        (fixed_originals if used else skipped).append(original)
                        if used:
                            fixed_original_paths[original] = path
                else:
                    skipped.extend(fixed_stems[stem])
                continue
            down, up = parts["down"].float(), parts["up"].float()
            if down.dim() != 2 or up.dim() != 2:
                skipped.extend(fixed_stems[stem])
                continue
            for original in fixed_stems[stem]:
                part = _split(original)[1]
                used = part in ("down", "up", "alpha")
                (fixed_originals if used else skipped).append(original)
                if used:
                    fixed_original_paths[original] = path
            rank = down.shape[0]
            scale = float(parts["alpha"].float().reshape(-1)[0]) / rank if "alpha" in parts else 1.0
            downs.append(down)
            ups.append(up * scale)
        if downs:
            fixed[path] = (torch.cat(downs, 0), torch.cat(ups, 1))
    n_fixed = sum(len(parts) for stems in fixed_parts.values() for parts in stems.values())
    return Normalized(
        out, skipped, fixed, n_fixed, dense, origins, fixed_originals, fixed_original_paths
    )
