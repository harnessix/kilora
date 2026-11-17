# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ..compute import AXES
from ..region import grid_for, tokens_from_image

BETA = 0.1
TEXT_END = 0.35
GLOBAL_COVER = 0.99
ATTENTION = "split"
MOUNT_WEIGHTS = "text"
SELF_PENALTY = 1.0
SELF_END = None
MOUNT_SCOPE = "all"
MOUNT_SCOPES = ("cross_kv", "all")
SINGLE_READY = True

FAMILIES = ("anima", "cosmos")

_PLACED = (
    "area",
    "strength",
    "mask",
    "mask_strength",
    "set_area_to_bounds",
    "timestep_start",
    "timestep_end",
)
_MAX_REPEAT = 8


@dataclass(frozen=True)
class Segment:
    role: str
    axis: str | None
    length: int
    mask: torch.Tensor | None
    strength: float
    text: int | None = None


@dataclass(frozen=True)
class ComposedLayout:
    segments: tuple[Segment, ...]
    self_penalty: float | None = None
    mount_scope: str | None = None

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(s.length for s in self.segments)

    @property
    def total(self) -> int:
        return sum(self.lengths)

    @property
    def globals(self) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.role == "global")

    @property
    def layers(self) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.role == "layer")

    def caption_layout(self, cond_rows: torch.Tensor):
        from ..region_attention import CaptionLayout

        layers = self.layers
        return CaptionLayout(
            prompt=sum(s.length for s in self.globals),
            lengths=tuple(s.length for s in layers),
            masks=tuple(s.mask for s in layers),
            cond_rows=cond_rows,
        )


def token_weights(
    layout: ComposedLayout,
    tokens: int,
    latent: tuple[int, int],
    progress: float,
    *,
    for_text: bool,
) -> torch.Tensor:

    segments = layout.segments
    globals_ = [i for i, s in enumerate(segments) if s.role == "global"]
    layers = [i for i, s in enumerate(segments) if s.role == "layer"]
    if not globals_:
        raise ValueError("a composed layout needs a global segment")
    grid = grid_for(tokens, latent)
    if grid is None:
        raise ValueError(f"no token grid for {tokens} tokens over a latent of {latent}")
    m = [tokens_from_image(segments[i].mask, grid) for i in layers]
    batch = max([x.shape[0] for x in m], default=1)
    for x in m:
        if batch % x.shape[0]:
            raise ValueError(f"a mask batch of {x.shape[0]} for a batch of {batch}")
    m = [x.repeat(batch // x.shape[0], 1) for x in m]
    dtype = m[0].dtype if m else torch.float32
    device = m[0].device if m else None
    out = torch.zeros(len(segments), batch, tokens, dtype=dtype, device=device)
    text_over = for_text and progress >= TEXT_END
    if m and not text_over:
        stack = torch.stack(m)
        total = stack.sum(0)
        stack = stack / total.clamp_min(1.0)
        for k, i in enumerate(layers):
            share = 1.0 - BETA
            if not for_text and MOUNT_WEIGHTS == "shared" and segments[i].axis == "ip":
                share = 0.5
            out[i] = share * stack[k]
    rest = 1.0 - out[layers].sum(0) if layers else torch.ones_like(out[0])
    strengths = [max(float(segments[i].strength), 0.0) for i in globals_]
    whole = sum(strengths)
    for i, s in zip(globals_, strengths, strict=True):
        share = s / whole if whole > 0 else 1.0 / len(globals_)
        out[i] = share * rest
    return out


def self_end() -> float:
    return float(TEXT_END if SELF_END is None else SELF_END)


def penalty_of(layout: ComposedLayout | None = None) -> float:
    value = getattr(layout, "self_penalty", None)
    return float(SELF_PENALTY if value is None else value)


def scope_of(layout: ComposedLayout | None = None) -> str:
    value = getattr(layout, "mount_scope", None)
    return str(MOUNT_SCOPE if value is None else value)


def self_penalty(progress: float, layout: ComposedLayout | None = None) -> float:
    strength = max(penalty_of(layout), 0.0)
    return strength if strength > 0 and progress < self_end() else 0.0


def settings_note(layout: ComposedLayout | None = None) -> str:
    return (
        f"compose single: self-attention penalty {max(penalty_of(layout), 0.0):g} until "
        f"progress {self_end():g}, mount scope {scope_of(layout)}."
    )


def segment_axes(segment: Segment) -> tuple[str, ...]:
    if segment.axis is None:
        return AXES
    if segment.axis == "creative":
        return ("creative", "functional")
    return (segment.axis,)


def axis_maps(layouts, rows: torch.Tensor, latent: tuple[int, int], progress: float) -> dict:

    from ..call_values import ComposedAlpha

    height, width = latent
    tokens = height * width
    positive = layouts[0]
    cache: dict[int, tuple[torch.Tensor, ComposedLayout]] = {}
    rows = torch.as_tensor(rows).reshape(-1).tolist()
    kinds = sorted(set(rows))
    per = len(rows) // max(len(kinds), 1)
    image = {a: [] for a in AXES}
    text = {a: [] for a in AXES}
    lengths = set()
    image_only_globals = scope_of(positive) == "cross_kv"
    for b, kind in enumerate(rows):
        layout = layouts[kind] if 0 <= kind < len(layouts) and layouts[kind] else positive
        if kind not in cache:
            cache[kind] = (token_weights(layout, tokens, latent, progress, for_text=False), layout)
        weights, layout = cache[kind]
        inner = (b % per if per else b) % weights.shape[1]
        lengths.add(layout.total)
        for a in AXES:
            run = [
                i
                for i, seg in enumerate(layout.segments)
                if a in segment_axes(seg) and not (image_only_globals and seg.role == "layer")
            ]
            row = weights[run, inner].sum(0) if run else weights.new_zeros(tokens)
            image[a].append(row.reshape(height, width))
            text[a].append(
                torch.cat(
                    [
                        torch.full((seg.length,), float(a in segment_axes(seg)))
                        for seg in layout.segments
                    ]
                )
            )
    out = {}
    for a in AXES:
        maps = torch.stack(image[a])
        words = torch.stack(text[a]) if len(lengths) == 1 else None
        out[a] = ComposedAlpha(maps, None if words is None else words.to(maps.device))
    return out


_REGISTRY: OrderedDict[int, tuple] = OrderedDict()
_IDS = itertools.count(1)
_KEEP = 16


def register(layouts: tuple) -> int:
    key = next(_IDS)
    _REGISTRY[key] = tuple(layouts)
    while len(_REGISTRY) > _KEEP:
        _REGISTRY.popitem(last=False)
    return key


def lookup(key: int) -> tuple | None:
    return _REGISTRY.get(int(key))


def _is_layer(c) -> bool:
    return c.get("kilora_layer") in AXES


def _mask(c) -> torch.Tensor | None:
    mask = c.get("mask")
    if not isinstance(mask, torch.Tensor):
        return None
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask[None]
    elif mask.ndim == 4:
        mask = mask.mean(1)
    return mask * float(c.get("mask_strength", 1.0))


def _context(c) -> torch.Tensor | None:
    value = (c.get("model_conds") or {}).get("c_crossattn")
    tensor = getattr(value, "cond", None)
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
        return tensor
    return None


def _repeat(context: torch.Tensor, length: int) -> torch.Tensor:
    return context.repeat(1, length // context.shape[1], 1)


def _batch(contexts: list[torch.Tensor]) -> list[torch.Tensor] | None:
    batch = max(c.shape[0] for c in contexts)
    if any(batch % c.shape[0] for c in contexts):
        return None
    return [c.repeat(batch // c.shape[0], 1, 1) for c in contexts]


def _check(c, base) -> str | None:
    if c.get("control") is not None and c.get("control") is not base.get("control"):
        return "a layer has its own control"
    if c.get("hooks") is not None and c.get("hooks") is not base.get("hooks"):
        return "a layer has its own hooks"
    if c.get("timestep_start") is not None or c.get("timestep_end") is not None:
        return "a layer has its own timestep range"
    if c.get("area") is not None:
        return "a layer has an area instead of a mask"
    if c.get("kilora_alpha_source") == "attention":
        return "a layer uses alpha_source attention"
    from .expert_text import _PREFIX

    if any(k.startswith(_PREFIX) for k in c.get("model_conds") or {}):
        return "expert text gives each expert its own text"
    if "t5xxl_ids" in (c.get("model_conds") or {}):
        return "a prompt passes its text ids to the model"
    return None


DIFFERENT_SETTINGS = "the layers set different region penalties or mount scopes"


def _sample_settings(layered) -> tuple[float | None, str | None, bool]:

    raw_penalties = [c.get("kilora_region_penalty") for c in layered]
    raw_scopes = [c.get("kilora_mount_scope") for c in layered]
    penalties = {float(SELF_PENALTY if v is None else v) for v in raw_penalties}
    scopes = {str(MOUNT_SCOPE if v is None else v) for v in raw_scopes}
    if len(penalties) > 1 or len(scopes) > 1:
        return None, None, False
    penalty = None if all(v is None for v in raw_penalties) else penalties.pop()
    scope = None if all(v is None for v in raw_scopes) else scopes.pop()
    return penalty, scope, True


def compose(conds, *, options=None, family=None, carry=False):

    options = options or {}
    if not conds or conds[0] is None:
        return None, "no positive prompt"
    positive = list(conds[0])
    negative = None if len(conds) < 2 or conds[1] is None else list(conds[1])
    if len(conds) > 2 and any(group for group in conds[2:]):
        return None, "more than two cond groups"
    layered = [c for c in positive if _is_layer(c)]
    if not layered:
        return None, "no layer"
    if any(c.get("kilora_compose", "separate") != "single" for c in layered):
        return None, "a layer uses compose separate"
    penalty, scope, same = _sample_settings(layered)
    if not same:
        return None, DIFFERENT_SETTINGS
    if options.get("kilora_layer_filter"):
        return None, "a KiLoRA Layer Filter is connected"
    if options.get("kilora_alpha_mask"):
        return None, "KiLoRA Apply has an alpha mask"
    if carry or options.get("kilora_layer_separate"):
        return None, "the mods carry state across steps"
    name = str(family or "").lower()
    if not any(f in name for f in FAMILIES):
        return None, "the model family is not Anima or Cosmos"

    globals_, layers = [], []
    for c in positive:
        mask = _mask(c)
        if mask is None:
            globals_.append((c, None))
        elif _is_layer(c) and float(mask.mean()) >= GLOBAL_COVER:
            globals_.append((c, mask))
        elif _is_layer(c):
            layers.append((c, mask))
        else:
            return None, "a prompt without a layer has a mask"
    if not globals_:
        return None, "no global prompt: single needs a prompt that covers the whole image"
    base = globals_[0][0]
    routed = [c for c in negative or () if _is_layer(c)]
    for c in [*positive, *routed]:
        cause = _check(c, base)
        if cause is not None:
            return None, cause
    plain = [c for c in negative or () if not _is_layer(c)]
    if len(plain) > 1:
        return None, "several plain negatives"
    for c in plain:
        if "t5xxl_ids" in (c.get("model_conds") or {}):
            return None, "a prompt passes its text ids to the model"
        if c.get("timestep_start") is not None or c.get("timestep_end") is not None:
            return None, "the negative has its own timestep range"
        if c.get("area") is not None:
            return None, "the negative has an area"

    order = [(c, "global", None) for c, _ in globals_] + [(c, "layer", m) for c, m in layers]
    contexts = [_context(c) for c, _, _ in order]
    if any(x is None for x in contexts):
        return None, "a prompt has no text context"

    neg_contexts = None
    if negative is not None:
        replace = {}
        for c in routed:
            matches = [
                i
                for i, (p, role, _) in enumerate(order)
                if role == "layer" and p["kilora_layer"] == c["kilora_layer"]
            ]
            if len(matches) != 1:
                return None, "a routed negative does not match exactly one layer"
            if matches[0] in replace:
                return None, "several negatives routed through one layer"
            replace[matches[0]] = c
        if not plain and len(replace) < len(order):
            return None, "no plain negative"
        neg_conds = [replace.get(i, plain[0] if plain else None) for i in range(len(order))]
        neg_contexts = [_context(c) for c in neg_conds]
        if any(x is None for x in neg_contexts):
            return None, "a negative has no text context"

    lengths = []
    for i, context in enumerate(contexts):
        length = context.shape[1]
        if neg_contexts is not None:
            other = neg_contexts[i].shape[1]
            length = math.lcm(length, other)
            if length > _MAX_REPEAT * max(context.shape[1], other):
                return None, "the text lengths do not fit in one segment length"
        lengths.append(length)
    contexts = [_repeat(x, n) for x, n in zip(contexts, lengths, strict=True)]
    contexts = _batch(contexts)
    if contexts is None or len({x.shape[-1] for x in contexts}) != 1:
        return None, "the prompts do not fit in one context"

    def layout_of(cs, texts):
        return ComposedLayout(
            tuple(
                Segment(
                    role=role,
                    axis=c.get("kilora_layer") if _is_layer(c) else None,
                    length=n,
                    mask=mask,
                    strength=float(c.get("strength", 1.0)),
                    text=t,
                )
                for (c, role, mask), n, t in zip(cs, lengths, texts, strict=True)
            ),
            self_penalty=penalty,
            mount_scope=scope,
        )

    def composed(template, context):
        out = {
            k: v
            for k, v in template.items()
            if k not in _PLACED and not k.startswith("kilora_") and k != "model_conds"
        }
        out["strength"] = 1.0
        model_conds = {
            k: v
            for k, v in (template.get("model_conds") or {}).items()
            if not k.startswith("kilora_")
        }
        model_conds["c_crossattn"] = _cross_attn(template, context)
        out["model_conds"] = model_conds
        return out

    pos_texts = _text_ids([_context(c) for c, _, _ in order])
    pos_layout = layout_of(order, pos_texts)
    groups = [[composed(base, torch.cat(contexts, 1))]]
    neg_layout = None
    if negative is not None:
        neg = [_repeat(x, n) for x, n in zip(neg_contexts, lengths, strict=True)]
        neg = _batch(neg)
        if neg is None or len({x.shape[-1] for x in neg}) != 1:
            return None, "the negatives do not fit in one context"
        neg_layout = layout_of(order, _text_ids(neg_contexts))
        groups.append([composed(plain[0] if plain else routed[0], torch.cat(neg, 1))])
    else:
        groups.append(None)
    return groups, (pos_layout, neg_layout)


def _text_ids(contexts: list[torch.Tensor]) -> tuple[int, ...]:
    ids: list[int] = []
    for i, context in enumerate(contexts):
        same = next(
            (
                ids[j]
                for j in range(i)
                if contexts[j] is context
                or (
                    contexts[j].shape == context.shape
                    and contexts[j].dtype == context.dtype
                    and torch.equal(contexts[j], context)
                )
            ),
            None,
        )
        ids.append(i if same is None else same)
    return tuple(ids)


def _cross_attn(template, context):
    old = (template.get("model_conds") or {}).get("c_crossattn")
    kind = type(old)
    try:
        return kind(context)
    except TypeError:
        from comfy.conds import CONDCrossAttn

        return CONDCrossAttn(context)
