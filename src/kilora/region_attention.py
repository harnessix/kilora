# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .region import grid_for, tokens_from_image

MODES = ("off", "soft", "hard")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionAttention:
    mode: str = "off"
    strength: float = 0.5
    until: float = 0.6
    outside_region: bool = False
    start: float = 0.3
    captions: bool = False
    timeline: tuple = ()
    attend_now: bool | None = None
    captions_now: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown region attention mode {self.mode!r}")

    def active(self, progress: float) -> bool:
        if self.attend_now is not None:
            return self.mode != "off" and self.attend_now
        return self.mode != "off" and self.in_window(progress)

    def in_window(self, progress: float) -> bool:
        return self.start <= progress < self.until

    def captions_active(self, progress: float) -> bool:
        if self.captions_now is not None:
            return self.captions and self.captions_now
        return self.captions and self.in_window(progress)

    def at(self, clock) -> RegionAttention:
        if not self.timeline:
            return self
        from dataclasses import replace

        from .timeline import window_value

        changes: dict = {"timeline": ()}
        v = window_value(self.timeline, "region_attention", clock)
        if v is not None:
            changes.update(attend_now=v > 0, strength=v if v > 0 else self.strength)
        c = window_value(self.timeline, "caption_window", clock)
        if c is not None:
            changes["captions_now"] = c > 0
        return replace(self, **changes)


def memberships(
    masks: Sequence[torch.Tensor], tokens: int, latent: tuple[int, int], batch: int
) -> torch.Tensor | None:
    grid = grid_for(tokens, latent)
    if grid is None or not masks:
        return None
    out = []
    for mask in masks:
        m = tokens_from_image(mask, grid)
        if m.shape[0] != batch:
            if batch % m.shape[0]:
                raise ValueError(f"a region batch of {m.shape[0]} for a batch of {batch}")
            m = m.repeat(batch // m.shape[0], 1)
        out.append(m)
    return torch.stack(out)


def cross_weight(m: torch.Tensor) -> torch.Tensor:
    inside = m.amax(0)
    shared = torch.zeros(m.shape[1], m.shape[2], m.shape[2], dtype=m.dtype, device=m.device)
    for r in m:
        shared = torch.maximum(shared, torch.minimum(r[:, :, None], r[:, None, :]))
    return (inside[:, :, None] - shared).clamp_min(0)


def region_bias(m: torch.Tensor, config: RegionAttention, dtype: torch.dtype) -> torch.Tensor:
    m = m.float()
    if config.outside_region:
        m = torch.cat([m, (1 - m.amax(0, keepdim=True)).clamp_min(0)])
    w = cross_weight(m)
    if config.mode == "hard":
        bias = torch.zeros_like(w).masked_fill(w >= 0.5, float("-inf"))
    else:
        bias = -float(config.strength) * w
    return bias[:, None].to(dtype)


class _BiasCache:
    def __init__(self, masks: Sequence[torch.Tensor], config: RegionAttention) -> None:
        self.masks = list(masks)
        self.config = config
        self.cache: dict = {}

    def get(
        self, tokens: int, batch: int, latent: tuple[int, int] | None, dtype, device
    ) -> torch.Tensor | None:
        if latent is None:
            return None
        key = (tokens, batch, tuple(latent), dtype, device)
        if key not in self.cache:
            masks = [mk.to(device) for mk in self.masks]
            m = memberships(masks, tokens, latent, batch)
            self.cache[key] = None if m is None else region_bias(m, self.config, dtype)
        return self.cache[key]


def _masked_backend(func: Callable) -> Callable:
    try:
        from comfy.ldm.modules.attention import attention_pytorch
    except ImportError:
        return func
    return attention_pytorch


@dataclass
class CallState:
    progress: float = 0.0
    latent: tuple[int, int] | None = None
    kind: str | None = None
    captions: CaptionLayout | None = None
    owners: dict | None = None
    cond_rows: torch.Tensor | None = None
    owner_mode: str = "single"
    owner_tracker: object | None = None
    compose: bool = False
    real: torch.Tensor | None = None


def owner_bias(owners, tokens, latent, batch, cond_rows, dtype, device):
    names = [name for name in owners if name != "functional"]
    if not names or latent is None:
        return None
    m = memberships([owners[name].to(device) for name in names], tokens, latent, batch)
    if m is None:
        return None
    owned = m > 0.5
    far = owned[names.index("creative")] if "creative" in names else torch.zeros_like(owned[0])
    blocked = torch.zeros((batch, tokens, tokens), dtype=torch.bool, device=device)
    for i in range(len(names)):
        blocked |= owned[i][:, :, None] & ~(owned[i] | far)[:, None, :]
    if cond_rows is not None:
        if cond_rows.numel() != batch:
            return None
        blocked &= cond_rows.to(device)[:, None, None]
    log.info(
        "axis owner: shares=%s masked pairs=%.6f",
        {name: float(m[i].mean()) for i, name in enumerate(names)},
        float(blocked.float().mean()),
    )
    return torch.zeros(blocked.shape, dtype=dtype, device=device).masked_fill(
        blocked, float("-inf")
    )[:, None]


def soft_owner_bias(owners, tokens, latent, batch, cond_rows, dtype, device, strength):
    names = [name for name in owners if name != "functional"]
    if not names or latent is None:
        return None
    m = memberships([owners[name].to(device) for name in names], tokens, latent, batch)
    if m is None:
        return None
    m = m.float()
    bias = torch.zeros((batch, tokens, tokens), dtype=m.dtype, device=device)
    for i in range(len(names)):
        query = m[i] * (m[i] > 0.5)
        for j, name in enumerate(names):
            if i != j and name != "creative":
                bias -= strength * query[:, :, None] * (m[j] * (m[j] > 0.5))[:, None, :]
    if cond_rows is not None:
        if cond_rows.numel() != batch:
            return None
        bias *= cond_rows.to(device)[:, None, None]
    limit = torch.finfo(dtype).max
    return bias.clamp(min=-limit, max=limit)[:, None].to(dtype)


@dataclass(frozen=True)
class CaptionLayout:
    prompt: int
    lengths: tuple[int, ...]
    masks: tuple
    cond_rows: torch.Tensor

    @property
    def total(self) -> int:
        return self.prompt + sum(self.lengths)

    @property
    def segments(self) -> tuple[int, ...]:
        return (self.prompt, *self.lengths)


def caption_weights(
    layout: CaptionLayout, tokens: int, latent: tuple[int, int] | None, device
) -> torch.Tensor | None:
    if latent is None:
        return None
    batch = layout.cond_rows.shape[0]
    m = memberships([mk.to(device) for mk in layout.masks], tokens, latent, batch)
    if m is None:
        return None
    inside = ((m >= 0.5) & layout.cond_rows.to(device)[None, :, None]).float()
    count = inside.sum(0)
    prompt = (count == 0).float()
    return torch.cat([prompt[None], inside / count.clamp_min(1.0)])


def split_attention(
    attend: Callable,
    args: tuple,
    kwargs: dict,
    lengths: Sequence[int],
    weights: torch.Tensor,
) -> torch.Tensor:
    q, k, v, *rest = args
    kdim = 2 if kwargs.get("skip_reshape", False) else 1
    odim = 2 if kwargs.get("skip_output_reshape", False) else 1
    mask = kwargs.get("mask")
    sk = k.shape[kdim]
    out = None
    start = 0
    for s, length in enumerate(lengths):
        w = weights[s]
        if length and bool((w != 0).any()):
            kw = dict(kwargs)
            if isinstance(mask, torch.Tensor) and mask.shape[-1] == sk:
                kw["mask"] = mask[..., start : start + length]
            ks = k.narrow(kdim, start, length).contiguous()
            vs = v.narrow(kdim, start, length).contiguous()
            a = attend(q, ks, vs, *rest, **kw)
            w = w.to(a.device, a.dtype)
            w = w[:, :, None] if odim == 1 else w[:, None, :, None]
            out = a * w if out is None else out + a * w
        start += length
    return attend(q, k, v, *rest, **kwargs) if out is None else out


def merge_equal_segments(
    layouts: Sequence, args: tuple, kwargs: dict, weights: torch.Tensor
) -> torch.Tensor:

    first = layouts[0]
    lengths = first.lengths
    batch = weights.shape[1]
    if len(layouts) != batch or len(lengths) < 2:
        return weights
    pairs: dict = {}
    for b, layout in enumerate(layouts):
        seen: dict = {}
        for s, segment in enumerate(layout.segments):
            text = getattr(segment, "text", None)
            if text is None:
                continue
            if text in seen:
                pairs.setdefault((seen[text], s), []).append(b)
            else:
                seen[text] = s
    if not pairs:
        return weights
    k, v = args[1], args[2]
    kdim = 2 if kwargs.get("skip_reshape", False) else 1
    mask = kwargs.get("mask")
    if (
        k.shape[kdim] != sum(lengths)
        or k.shape[0] != batch
        or v.shape[kdim] != k.shape[kdim]
        or (isinstance(mask, torch.Tensor) and mask.shape[-1] == k.shape[kdim])
    ):
        return weights
    starts = [sum(lengths[:s]) for s in range(len(lengths))]

    def piece(t, s):
        return t.narrow(kdim, starts[s], lengths[s])

    def same(a, s):
        short, long_ = (a, s) if lengths[a] <= lengths[s] else (s, a)
        if lengths[short] == 0 or lengths[long_] % lengths[short]:
            return None
        times = lengths[long_] // lengths[short]
        reps = [1] * k.ndim
        reps[kdim] = times
        out = None
        for t in (k, v):
            eq = (piece(t, short).repeat(*reps) == piece(t, long_)).flatten(1).all(1)
            out = eq if out is None else out & eq
        return out

    weights = weights.clone()
    for (a, s), rows in sorted(pairs.items()):
        eq = same(a, s)
        if eq is None:
            continue
        mask = torch.zeros(batch, dtype=torch.bool, device=eq.device)
        mask[rows] = True
        move = (eq & mask).to(weights.device)[:, None]
        weights[a] = weights[a] + torch.where(move, weights[s], torch.zeros_like(weights[s]))
        weights[s] = torch.where(move, torch.zeros_like(weights[s]), weights[s])
    return weights


def _row_layouts(layouts: tuple, rows: torch.Tensor | None, batch: int) -> list:
    values = [0] * batch
    if rows is not None and rows.numel() == batch:
        values = [int(r) for r in rows.reshape(-1).tolist()]
    out = []
    for r in values:
        layout = layouts[r] if 0 <= r < len(layouts) else None
        out.append(layout if layout is not None else layouts[0])
    return out


def composed_weights(
    layouts: Sequence, tokens: int, latent: tuple[int, int] | None, progress: float, device
) -> torch.Tensor:
    from .comfyui import layer_compose

    first = layouts[0]
    out = torch.zeros(len(first.segments), len(layouts), tokens, device=device)
    groups: dict = {}
    for b, layout in enumerate(layouts):
        groups.setdefault(id(layout), (layout, []))[1].append(b)
    for layout, positions in groups.values():
        try:
            if latent is None:
                raise ValueError("no latent size")
            w = layer_compose.token_weights(layout, tokens, latent, progress, for_text=True)
        except ValueError:
            w = _global_weights(layout, tokens)
        w = w.to(device, torch.float32)
        for j, b in enumerate(positions):
            out[:, b] = w[:, j % w.shape[1]]
    return out


def _global_weights(layout, tokens: int) -> torch.Tensor:
    out = torch.zeros(len(layout.segments), 1, tokens)
    globals_ = [i for i, s in enumerate(layout.segments) if s.role == "global"]
    strengths = [max(float(layout.segments[i].strength), 0.0) for i in globals_]
    whole = sum(strengths)
    for i, s in zip(globals_, strengths, strict=True):
        out[i] = s / whole if whole > 0 else 1.0 / len(globals_)
    return out


def composed_self_bias(
    layouts: Sequence,
    tokens: int,
    latent: tuple[int, int] | None,
    strength: float,
    dtype: torch.dtype,
    device,
) -> torch.Tensor | None:
    if latent is None or strength <= 0:
        return None
    config = RegionAttention(mode="soft", strength=float(strength), outside_region=True)
    groups: dict = {}
    for b, layout in enumerate(layouts):
        layers = [s.mask for s in layout.segments if s.role == "layer" and s.mask is not None]
        groups.setdefault(tuple(id(mk) for mk in layers), (layers, []))[1].append(b)
    out = None
    for layers, positions in groups.values():
        if not layers:
            continue
        masks = [mk.to(device) for mk in layers]
        sizes = [mk.shape[0] if mk.ndim == 3 else 1 for mk in masks]
        inner = max(sizes)
        m = memberships(masks, tokens, latent, inner)
        if m is None:
            return None
        bias = region_bias(m, config, dtype)
        if out is None:
            out = torch.zeros(len(layouts), 1, tokens, tokens, dtype=dtype, device=device)
        for j, b in enumerate(positions):
            out[b] = bias[j % bias.shape[0]]
    return out


def softmax_bias(
    lengths: Sequence[int],
    weights: torch.Tensor,
    real: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    cols = [weights[s][:, :, None].expand(-1, -1, n) for s, n in enumerate(lengths)]
    w = torch.cat(cols, -1).float()
    if real is not None:
        w = w * real.to(w.device)[:, None, :].float()
    bias = torch.full_like(w, float("-inf"))
    bias = torch.where(w > 0, w.clamp_min(torch.finfo(torch.float32).tiny).log(), bias)
    return bias[:, None].to(dtype)


def kind_patches(state: CallState) -> dict[str, Callable]:
    def mark(kind: str) -> Callable:
        def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
            state.kind = kind
            return {}

        return patch

    return {"attn1_patch": mark("self"), "attn2_patch": mark("cross")}


def comfy_override(
    masks: Sequence[torch.Tensor],
    config: RegionAttention,
    state: CallState,
    previous: Callable | None = None,
) -> Callable:
    cache = _BiasCache(masks, config)
    captions: dict = {}
    owners: dict = {}
    composed: dict = {}
    selves: dict = {}

    def compose_now():
        if not state.compose:
            return None
        from .call_values import current_compose_layout

        current = current_compose_layout()
        if current is None or not current[0] or current[0][0] is None:
            return None
        return current

    def call(func, *args, **kwargs):
        return previous(func, *args, **kwargs) if previous is not None else func(*args, **kwargs)

    def with_bias(func, bias, args, kwargs):
        mask = kwargs.get("mask")
        kwargs["mask"] = bias if mask is None else mask + bias
        return call(_masked_backend(func), *args, **kwargs)

    def override(func, *args, **kwargs):
        kind, state.kind = state.kind, None
        marked = kind
        q, k = args[0], args[1]
        dim = 2 if kwargs.get("skip_reshape", False) else 1
        sq, sk = q.shape[dim], k.shape[dim]
        layout = state.captions
        current = compose_now()
        if kind is None:
            totals = [layout.total if layout is not None else None]
            if current is not None:
                totals.append(current[0][0].total)
            if sq != sk and sk in totals:
                kind = "cross"
            elif sq == sk:
                kind = "self"
        if kind == "cross" and current is not None:
            out = composed_cross(func, current, args, kwargs, sq, sk)
            if out is not None:
                return out
        if kind == "cross" and layout is not None and sk == layout.total:
            key = (sq, q.device)
            if key not in captions:
                captions[key] = caption_weights(layout, sq, state.latent, q.device)
            w = captions[key]
            if w is not None and w.shape[1] == q.shape[0]:
                return split_attention(
                    lambda *a, **kw: call(func, *a, **kw), args, kwargs, layout.segments, w
                )
        own = None
        if kind == "self" and sq == sk and current is not None:
            own = composed_self(current, q, sq)
        if kind == "self" and config.active(state.progress):
            bias = cache.get(sq, q.shape[0], state.latent, q.dtype, q.device)
            if bias is not None:
                return with_bias(func, add(bias, own), args, kwargs)
        if marked == "self" and sq == sk and state.owners:
            key = (sq, q.shape[0], q.dtype, q.device)
            if key not in owners:
                if state.owner_mode == "soft":
                    from .axis_owner import nonnegative_setting

                    strength = nonnegative_setting("KILORA_AXIS_BIAS", 2.0)
                    owners[key] = (
                        soft_owner_bias(
                            state.owners,
                            sq,
                            state.latent,
                            q.shape[0],
                            state.cond_rows,
                            q.dtype,
                            q.device,
                            strength,
                        )
                        if strength
                        else None
                    )
                    if owners[key] is not None and state.owner_tracker is not None:
                        state.owner_tracker.add_bias(owners[key])
                else:
                    owners[key] = owner_bias(
                        state.owners,
                        sq,
                        state.latent,
                        q.shape[0],
                        state.cond_rows,
                        q.dtype,
                        q.device,
                    )
            if owners[key] is not None:
                return with_bias(func, add(owners[key], own), args, kwargs)
        if own is not None:
            return with_bias(func, own, args, kwargs)
        return call(func, *args, **kwargs)

    def add(bias, other):
        return bias if other is None else bias + other

    def composed_self(current, q, sq):
        from .comfyui import layer_compose

        strength = layer_compose.self_penalty(state.progress)
        if strength <= 0:
            return None
        layouts, rows = current
        batch = q.shape[0]
        picked = _row_layouts(layouts, rows, batch)
        key = (sq, batch, q.dtype, q.device, strength, tuple(id(x) for x in picked))
        if key not in selves:
            selves[key] = composed_self_bias(picked, sq, state.latent, strength, q.dtype, q.device)
        return selves[key]

    def composed_cross(func, current, args, kwargs, sq, sk):
        from .comfyui import layer_compose

        layouts, rows = current
        q = args[0]
        batch = q.shape[0]
        picked = _row_layouts(layouts, rows, batch)
        lengths = picked[0].lengths
        if sk != picked[0].total or any(x.lengths != lengths for x in picked):
            return None
        key = (sq, batch, q.device, tuple(id(x) for x in picked))
        if key not in composed:
            composed[key] = composed_weights(picked, sq, state.latent, state.progress, q.device)
        w = composed[key]
        if layer_compose.ATTENTION == "softmax":
            real = state.real
            if real is not None and tuple(real.shape) != (batch, sk):
                real = None
            bias = softmax_bias(lengths, w, real, q.dtype)
            return with_bias(func, bias, args, kwargs)
        w = merge_equal_segments(picked, args, kwargs, w)
        return split_attention(lambda *a, **kw: call(func, *a, **kw), args, kwargs, lengths, w)

    return override


class _Processor:
    def __init__(self, inner: Any, bias: Callable[[torch.Tensor], torch.Tensor | None]):
        self.inner = inner
        self.bias = bias

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            bias = self.bias(hidden_states)
            if bias is not None:
                attention_mask = bias if attention_mask is None else attention_mask + bias
        if image_rotary_emb is not None:
            kwargs["image_rotary_emb"] = image_rotary_emb
        return self.inner(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


@contextlib.contextmanager
def diffusers_region_attention(
    transformer: torch.nn.Module,
    masks: Sequence[torch.Tensor],
    config: RegionAttention,
    progress: Callable[[], float] | None = None,
) -> Iterator[None]:
    if config.mode == "off" or not masks:
        yield
        return
    from .compute import current_progress

    progress = progress or current_progress
    state: dict = {"latent": None, "cache": None}

    def before(mod, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        state["latent"] = tuple(x.shape[-2:])
        state["cache"] = _BiasCache(masks, config)

    def bias(h: torch.Tensor) -> torch.Tensor | None:
        if state["cache"] is None or not config.active(progress()):
            return None
        return state["cache"].get(h.shape[1], h.shape[0], state["latent"], h.dtype, h.device)

    wrapped = []
    for module in transformer.modules():
        inner = getattr(module, "processor", None)
        if inner is not None and hasattr(module, "set_processor"):
            wrapped.append((module, inner))
            module.processor = _Processor(inner, bias)
    hook = transformer.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
        for module, inner in wrapped:
            module.processor = inner
