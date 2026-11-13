# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import inspect
import logging

import torch

log = logging.getLogger("kilora")

_EMBEDDERS = ("pos_embedder", "extra_pos_embedder")


def supported(model) -> bool:
    names = {cls.__name__ for cls in type(model).__mro__}
    if "MiniTrainDIT" not in names:
        return False
    embedder = getattr(model, "pos_embedder", None)
    return callable(getattr(embedder, "generate_embeddings", None))


def shared_window(boxes, batch: int):
    rows = {tuple(int(v) for v in row) for row in boxes}
    if len(rows) != 1 or len(boxes) not in (1, batch):
        return None
    return next(iter(rows))


def _slice(full, t, fh, fw, top, left, h, w):
    if full.ndim == 5 and tuple(full.shape[1:4]) == (t, fh, fw):
        return full[:, :, top : top + h, left : left + w].contiguous()
    if full.shape[0] == t * fh * fw:
        rest = tuple(full.shape[1:])
        grid = full.reshape(t, fh, fw, *rest)[:, top : top + h, left : left + w]
        return grid.reshape(t * h * w, *rest).contiguous()
    return None


def _wrap(embedder, window, patch):
    original = embedder.forward
    signature = inspect.signature(original)
    first = next(iter(signature.parameters))
    accepted = inspect.signature(embedder.generate_embeddings).parameters
    takes_any = any(p.kind is p.VAR_KEYWORD for p in accepted.values())
    top, left, full_h, full_w = window

    def forward(x, *args, **kwargs):
        b, t, h, w = x.shape[:4]
        fh, fw = full_h // patch, full_w // patch
        y0, x0 = top // patch, left // patch
        if (h, w) == (fh, fw) or y0 + h > fh or x0 + w > fw:
            return original(x, *args, **kwargs)
        bound = signature.bind(x, *args, **kwargs).arguments
        options = {k: v for k, v in bound.items() if k != first and (takes_any or k in accepted)}
        shape = torch.Size((b, t, fh, fw, *x.shape[4:]))
        result = _slice(embedder.generate_embeddings(shape, **options), t, fh, fw, y0, x0, h, w)
        return original(x, *args, **kwargs) if result is None else result

    return forward


@contextlib.contextmanager
def global_positions(model, boxes, batch: int, on_mixed=None):

    if boxes is None or model is None or not supported(model):
        yield False
        return
    window = shared_window(boxes, batch)
    if window is None:
        if on_mixed is not None:
            on_mixed()
        yield False
        return
    top, left, full_h, full_w = window
    patch = int(getattr(model, "patch_spatial", 1) or 1)
    if any(v % patch for v in window):
        yield False
        return
    patched = []
    try:
        for name in _EMBEDDERS:
            embedder = getattr(model, name, None)
            if embedder is None or not callable(getattr(embedder, "generate_embeddings", None)):
                continue
            had = "forward" in embedder.__dict__
            previous = embedder.__dict__.get("forward")
            embedder.forward = _wrap(embedder, window, patch)
            patched.append((embedder, had, previous))
        yield bool(patched)
    finally:
        for embedder, had, previous in reversed(patched):
            if had:
                embedder.forward = previous
            else:
                del embedder.forward


def diffusion_model(patcher):
    return getattr(getattr(patcher, "model", None), "diffusion_model", None)


def mixed_note():
    state = {"done": False}

    def note():
        if not state["done"]:
            log.warning(
                "KiLoRA: a model call batched layer crops with different windows. "
                "That call keeps the local token positions of each crop."
            )
            state["done"] = True

    note.state = state
    return note
