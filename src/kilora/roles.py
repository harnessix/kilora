# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence

import torch

ROLES = ("target", "reference", "text")
POLICIES = (
    "auto",
    "all",
    "target",
    "target+text",
    "reference",
    "reference+text",
    "target+reference",
)
METADATA_KEY = "token_roles"

_LOG = logging.getLogger("kilora.roles")
_ROLE = re.compile(r"^(target|text|reference)(?::(\d+))?$")


def parse_policy(policy: str | None) -> str:

    value = (policy or "auto").strip().lower().replace(",", "+").replace(" ", "")
    if value in ("auto", "all"):
        return value
    parts = sorted({p for p in value.split("+") if p})
    if not parts:
        return "auto"
    for p in parts:
        m = _ROLE.match(p)
        if m is None or (m.group(2) is not None and m.group(1) != "reference"):
            raise ValueError(f"unknown token role {p!r}; roles are {ROLES} and reference:N")
    return "+".join(parts)


def first_policy(*policies: str | None) -> str:
    for p in policies:
        p = parse_policy(p)
        if p != "auto":
            return p
    return "auto"


def file_policy(path: str | os.PathLike | None) -> str:
    if path is None or not os.fspath(path).endswith(".safetensors"):
        return "auto"
    try:
        with open(path, "rb") as f:
            size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(size)) if 0 < size < 100_000_000 else {}
        value = (header.get("__metadata__") or {}).get(METADATA_KEY)
        return parse_policy(value) if isinstance(value, str) else "auto"
    except (OSError, ValueError):
        return "auto"


_GENERIC_TAGS = {
    "a",
    "an",
    "the",
    "1girl",
    "1boy",
    "solo",
    "highres",
    "absurdres",
    "simple background",
    "white background",
    "monochrome",
    "greyscale",
    "grayscale",
    "masterpiece",
    "best quality",
    "high quality",
    "low quality",
    "worst quality",
    "score 9",
    "looking at viewer",
    "upper body",
    "full body",
    "close up",
    "portrait",
    "cowboy shot",
    "wide shot",
    "from above",
    "from below",
    "depth of field",
    "long hair",
    "short hair",
    "blush",
    "smile",
    "open mouth",
    "jpeg artifacts",
    "normal quality",
}
_STYLE_WORDS = ("style", "art", "artist", "lineart", "sketch", "aesthetic")


def identity_token(meta: dict) -> str | None:
    from .comfyui.trigger_words import _load_json

    frequencies = _load_json(meta.get("ss_tag_frequency", ""))
    directories = _load_json(meta.get("ss_dataset_dirs", ""))
    if not isinstance(frequencies, dict) or not isinstance(directories, dict):
        return None
    total = 0.0
    for info in directories.values():
        if isinstance(info, dict):
            try:
                total += max(0.0, float(info.get("img_count", info.get("num_images", 0))))
            except (TypeError, ValueError):
                continue
    if not total:
        return None
    tags: dict[str, float] = {}
    captions: dict[str, float] = {}
    caption_total = 0.0
    for folder in frequencies.values():
        if not isinstance(folder, dict):
            continue
        for raw, count in folder.items():
            try:
                count = float(count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            tag = str(raw).strip().lower()
            if not tag:
                continue
            match = re.match(r"^([\w@-]+)\s+(?:is|has|wears|with)\b", tag)
            if match:
                word = match.group(1)
                captions[word] = captions.get(word, 0) + count
                caption_total += count
            else:
                tags[tag] = tags.get(tag, 0) + count
    candidates = [
        tag
        for tag, count in tags.items()
        if count >= total * 0.9 and tag.replace("_", " ") not in _GENERIC_TAGS
    ]
    if candidates:
        return next((tag for tag in candidates if _style_identity(tag)), candidates[0])
    return next(
        (
            word
            for word, count in captions.items()
            if count >= caption_total * 0.5 and word not in _GENERIC_TAGS
        ),
        None,
    )


def _style_identity(token: str) -> bool:
    return token.startswith("@") or any(word in token for word in _STYLE_WORDS)


def suggest_axis(path: str | os.PathLike) -> tuple[str, str]:
    from .comfyui.trigger_words import _load_json, _metadata, trigger_words_of

    path = os.fspath(path)
    try:
        meta = _metadata(path)
        words = trigger_words_of(path)
    except Exception:
        meta, words = {}, []
    value = str(meta.get("kilora_axis", "")).strip().lower()
    if value in ("creative", "functional", "ip"):
        return value, "metadata kilora_axis"

    token = identity_token(meta)
    if token:
        return ("creative" if _style_identity(token) else "ip"), f"identity token: {token}"
    for word in words:
        if re.fullmatch(r"[^(),]+\s+\([^()]+\)", word.strip()):
            return "ip", f"trigger: {word}"
    for key in ("tags", "modelspec.tags"):
        raw = meta.get(key, "")
        parsed = _load_json(raw)
        tags = parsed if isinstance(parsed, list) else str(raw).split(",")
        if any(isinstance(tag, str) and tag.strip().lower() == "character" for tag in tags):
            return "ip", f"metadata {key}"
    for key in ("civitai_type", "type", "civitai"):
        value = meta.get(key, "")
        parsed = _load_json(value)
        if isinstance(parsed, dict):
            value = parsed.get("type", "")
        if str(value).strip().lower() == "character":
            return "ip", f"metadata {key}"
    name = os.path.basename(path)
    if re.search(r"(?:^|[^a-z0-9])(character|characters|prop|props)(?:$|[^a-z0-9])", name.lower()):
        return "ip", f"file name: {name}"
    return "creative", "no clear axis signal"


_EDIT_WORDS = ("edit", "kontext", "ic-lora", "iclora", "ic_lora", "consistency", "consistence")


_FUNCTIONAL_WORDS = (
    "edit",
    "kontext",
    "extend",
    "outpaint",
    "inpaint",
    "consistency",
    "lightning",
    "turbo",
    "hyper",
    "lcm",
    "control",
)


def looks_functional(name: str) -> bool:
    low = os.path.basename(os.fspath(name)).lower()
    return any(w in low for w in _FUNCTIONAL_WORDS)


def looks_like_edit(name: str) -> bool:
    low = os.path.basename(os.fspath(name)).lower()
    return any(w in low for w in _EDIT_WORDS)


_REFS: list[tuple[tuple[int, ...], ...]] = []


@contextlib.contextmanager
def reference_shapes(shapes: Sequence[Sequence[int]] | None) -> Iterator[None]:
    _REFS.append(tuple(tuple(int(d) for d in s) for s in (shapes or ())))
    try:
        yield
    finally:
        _REFS.pop()


def current_references() -> tuple[tuple[int, ...], ...]:
    return _REFS[-1] if _REFS else ()


def shapes_of(conds: dict) -> tuple[tuple[int, ...], ...]:
    refs = conds.get("ref_latents") if isinstance(conds, dict) else None
    if isinstance(refs, (list, tuple)) and refs:
        return tuple(tuple(int(d) for d in r.shape[2:]) for r in refs if hasattr(r, "shape"))
    shapes = conds.get("ref_shapes") if isinstance(conds, dict) else None
    if isinstance(shapes, (list, tuple)) and shapes:
        return tuple(tuple(int(d) for d in s) for s in shapes)
    return ()


def needs_mask(policy: str, has_refs: bool) -> bool:
    policy = parse_policy(policy)
    if policy == "all":
        return False
    if not has_refs:
        return policy != "auto" and "target" not in policy.split("+")
    return True


def role_spans(
    layout, tokens: int, latent: Sequence[int], refs: Sequence[Sequence[int]], text_tokens=None
) -> dict[str, list[tuple[int, int]]] | None:
    if layout is None or layout.kind == "set":
        return None
    role = getattr(layout, "role", "")
    if layout.kind == "text1d" or role == "text":
        return {"text": [(0, tokens)]}
    if role == "reference":
        return {"reference": [(0, tokens)]}
    if layout.patch is None or layout.pad != 1:
        return None
    try:
        n = math.prod(layout.grid(latent))
        sizes = [math.prod(layout.grid(r)) for r in refs]
    except ValueError:
        return None
    if layout.text == "prefix":
        if tokens < n:
            return None
        return {"text": [(0, tokens - n)], "target": [(tokens - n, tokens)]}
    image = n + sum(sizes)
    text = tokens - image
    if text < 0 or (not layout.text and text != 0):
        return None
    references = [(f"reference:{i}", s) for i, s in enumerate(sizes)]
    if layout.text == "before":
        order = [("text", text), ("target", n), *references]
    elif layout.text == "after":
        order = [("target", n), ("text", text), *references]
    else:
        order = [("target", n), *references, ("text", text)]
    spans: dict[str, list[tuple[int, int]]] = {}
    start = 0
    for name, count in order:
        if count:
            spans.setdefault(name, []).append((start, start + count))
        start += count
    return spans


def _allowed(policy: str, role: str) -> bool:
    if policy == "all":
        return True
    parts = policy.split("+")
    if role in parts:
        return True
    return role.startswith("reference:") and "reference" in parts


PANEL_SIDES = ("left", "right", "top", "bottom")


def check_panels(side: str, split: float = 0.5) -> tuple[str, float]:
    if side not in PANEL_SIDES:
        raise ValueError(f"panel side must be one of {PANEL_SIDES}, not {side!r}")
    split = float(split)
    if not 0.0 < split < 1.0:
        raise ValueError(f"the reference panel must cover between 0 and 1, got {split!r}")
    return side, split


def panel_tokens(layout, latent: Sequence[int], panels: tuple[str, float]) -> torch.Tensor:
    from .call_values import current_crop_window

    side, split = panels
    boxes = current_crop_window()
    grid = layout.grid(latent)
    h, w = grid[-2:]
    if boxes:
        rows = []
        for top, left, full_h, full_w in boxes:
            ys = (torch.arange(h).add(0.5) * latent[-2] / h + top) / full_h
            xs = (torch.arange(w).add(0.5) * latent[-1] / w + left) / full_w
            if side in ("left", "right"):
                plane = (xs < split if side == "left" else xs > 1 - split)[None, :].expand(h, w)
            else:
                plane = (ys < split if side == "top" else ys > 1 - split)[:, None].expand(h, w)
            frames = math.prod(grid[:-2]) if len(grid) > 2 else 1
            rows.append(plane.reshape(1, -1).expand(frames, -1).reshape(-1))
        return torch.stack(rows)
    if side in ("left", "right"):
        cut = torch.arange(w).add(0.5).div(w)
        cols = cut < split if side == "left" else cut > 1 - split
        plane = cols[None, :].expand(h, w)
    else:
        cut = torch.arange(h).add(0.5).div(h)
        rows = cut < split if side == "top" else cut > 1 - split
        plane = rows[:, None].expand(h, w)
    frames = math.prod(grid[:-2]) if len(grid) > 2 else 1
    return plane.reshape(1, -1).expand(frames, -1).reshape(-1)


def role_mask(
    policy: str,
    layout,
    tokens: int,
    latent: Sequence[int],
    refs: Sequence[Sequence[int]],
    text_tokens: int | None = None,
    name: str = "",
    panels: tuple[str, float] | None = None,
) -> torch.Tensor | None:

    policy = parse_policy(policy)
    has_refs = bool(refs) or getattr(layout, "role", "") == "reference" or panels is not None
    if not needs_mask(policy, has_refs):
        return None
    if policy == "auto":
        policy = "target"
    spans = role_spans(layout, tokens, latent, refs, text_tokens)
    if spans is None:
        if name not in _WARNED:
            _WARNED.add(name)
            _LOG.warning("%s: the token roles of this layer are unknown; every token used", name)
        return None
    mask = torch.zeros(1, tokens)
    for role, parts in spans.items():
        if _allowed(policy, role):
            for a, b in parts:
                mask[:, a:b] = 1.0
    if panels is not None and "target" in spans and getattr(layout, "role", "") == "":
        ((a, b),) = spans["target"]
        try:
            ref = panel_tokens(layout, latent, panels)
        except ValueError:
            ref = None
        if ref is None or ref.shape[-1] != b - a:
            if name not in _WARNED:
                _WARNED.add(name)
                _LOG.warning("%s: the two panels do not fit this layer; every token used", name)
            return None
        panel_role = "reference" if refs else "reference:0"
        keep_ref = _allowed(policy, panel_role)
        keep_target = _allowed(policy, "target")
        if ref.ndim == 2:
            mask = mask.expand(ref.shape[0], -1).clone()
            mask[:, a:b] = torch.where(ref, float(keep_ref), float(keep_target))
        else:
            mask[0, a:b] = torch.where(ref, float(keep_ref), float(keep_target))
    return None if bool((mask == 1).all()) else mask


_WARNED: set[str] = set()
