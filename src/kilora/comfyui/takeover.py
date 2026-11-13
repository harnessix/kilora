# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import save_file

from ..bindings import COMFY_PREFIX, Binding
from ..multi_import import low_rank, lycoris_delta
from . import patching
from .patching import ModEntry

__all__ = ["KiLoRARouteLoaded", "Grouping", "group_patches", "take_over"]

SETTINGS = "KILORA_SETTINGS"


@dataclass
class Load:
    signature: tuple
    entries: dict[str, int] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        return float(self.signature[0])


@dataclass
class Grouping:
    loads: list[Load]
    ambiguous: list[str] = field(default_factory=list)
    left: set[tuple[str, int]] = field(default_factory=set)
    unsupported: dict[tuple[str, int], str] = field(default_factory=dict)


def _adapter(v: Any) -> tuple[str, tuple]:
    if hasattr(v, "weights"):
        return str(getattr(v, "name", type(v).__name__)).lower(), tuple(v.weights)
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
        return v[0], tuple(v[1])
    return "unknown", ()


def _rank_alpha(kind: str, w: tuple) -> tuple[int | None, float | None, Any]:
    def alpha(a):
        return None if a is None else float(torch.as_tensor(a).reshape(-1)[0])

    if kind == "lora":
        return int(w[1].shape[0]), alpha(w[2]), w[1].dtype
    if kind == "lokr":
        side = w[4] if w[4] is not None else w[6]
        rank = None if side is None else int(side.shape[0])
        first = next(t for t in (w[0], w[3], w[1], w[5]) if t is not None)
        return rank, alpha(w[2]), first.dtype
    if kind == "loha":
        return int(w[1].shape[0]), alpha(w[2]), w[0].dtype
    return None, None, None


def _unsupported(key: str, entry: tuple) -> str | None:
    strength_model = entry[2] if len(entry) > 2 else 1.0
    offset = entry[3] if len(entry) > 3 else None
    function = entry[4] if len(entry) > 4 else None
    kind, w = _adapter(entry[1])
    if not key.endswith(".weight") or not key.startswith(COMFY_PREFIX):
        return "not a weight of the diffusion model"
    if offset is not None or function is not None:
        return "patch with an offset or a function"
    if strength_model != 1.0:
        return "patch that also scales the base weight"
    if kind == "lora":
        up, down, _alpha, mid, dora = w[0], w[1], w[2], w[3], w[4] if len(w) > 4 else None
        if mid is not None or dora is not None or up.dim() != 2 or down.dim() != 2:
            return "LoRA with a mid weight, DoRA or a conv layer"
        return None
    if kind == "lokr":
        if (len(w) > 7 and w[7] is not None) or (len(w) > 8 and w[8] is not None):
            return "LoKr with a Tucker weight or DoRA"
        return None
    if kind == "loha":
        if any(len(w) > i and w[i] is not None for i in (5, 6, 7)):
            return "LoHa with a Tucker weight or DoRA"
        return None
    return f"adapter kind {kind!r}"


def signature(entry: tuple) -> tuple:
    kind, w = _adapter(entry[1])
    rank, alpha, dtype = _rank_alpha(kind, w)
    return (float(entry[0]), kind, rank, alpha, str(dtype))


def group_patches(patches: dict[str, list]) -> Grouping:
    unsupported: dict[tuple[str, int], str] = {}
    by_sig: dict[tuple, dict[str, list[int]]] = {}
    for key, entries in patches.items():
        for i, entry in enumerate(entries):
            reason = _unsupported(key, entry)
            if reason is not None:
                unsupported[(key, i)] = reason
            by_sig.setdefault(signature(entry), {}).setdefault(key, []).append(i)

    loads: list[Load] = []
    ambiguous: list[str] = []
    left: set[tuple[str, int]] = set()
    for sig, keys in by_sig.items():
        counts = {len(v) for v in keys.values()}
        if len(counts) != 1:
            ambiguous.append(
                f"{len(keys)} layers have {min(counts)} to {max(counts)} patches with "
                f"{_describe(sig)}: several LoRAs with this signature change different "
                "layers, so it is not certain which patch belongs to which LoRA"
            )
            left.update((k, i) for k, idx in keys.items() for i in idx)
            continue
        (c,) = counts
        for j in range(c):
            loads.append(Load(sig, {k: idx[j] for k, idx in keys.items()}))

    order = _load_order(loads, patches)
    if order is None:
        ambiguous.append(
            "the patch order on the layers contradicts itself, so the LoRAs cannot be told apart"
        )
        left.update((k, i) for load in loads for k, i in load.entries.items())
        loads = []
    else:
        loads = [loads[i] for i in order]
    return Grouping(loads, ambiguous, left, unsupported)


def _load_order(loads: list[Load], patches: dict[str, list]) -> list[int] | None:
    n = len(loads)
    after: list[set[int]] = [set() for _ in range(n)]
    on_key: dict[str, list[tuple[int, int]]] = {}
    for li, load in enumerate(loads):
        for k, i in load.entries.items():
            on_key.setdefault(k, []).append((i, li))
    key_pos = {k: p for p, k in enumerate(patches)}
    first_key: dict[int, list[int]] = {li: [] for li in range(n)}
    for k, pairs in on_key.items():
        pairs.sort()
        first_key[pairs[0][1]].append(key_pos[k])
        for (_, a), (_, b) in zip(pairs, pairs[1:], strict=False):
            after[b].add(a)
    owner = sorted((p, li) for li, ps in first_key.items() for p in ps)
    runs = [li for idx, (_, li) in enumerate(owner) if idx == 0 or owner[idx - 1][1] != li]
    if len(runs) != len(set(runs)):
        return None
    start = {li: (min(ps) if ps else len(key_pos)) for li, ps in first_key.items()}
    done: list[int] = []
    pending = set(range(n))
    while pending:
        ready = [li for li in pending if after[li] <= set(done)]
        if not ready:
            return None
        nxt = min(ready, key=lambda li: start[li])
        done.append(nxt)
        pending.remove(nxt)
    return done


def _describe(sig: tuple) -> str:
    strength, kind, rank, alpha, _ = sig
    alpha_text = "no alpha" if alpha is None else f"alpha {alpha:g}"
    return f"strength {strength:g}, {kind} rank {rank}, {alpha_text}"


def _delta_parts(kind: str, w: tuple) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if kind == "lora":
        up, down, alpha = w[0].float(), w[1].float(), w[2]
        scale = (
            1.0 if alpha is None else float(torch.as_tensor(alpha).reshape(-1)[0]) / down.shape[0]
        )
        return down, up * scale
    names = (
        ("lokr_w1", "lokr_w2", "alpha", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b")
        if kind == "lokr"
        else ("hada_w1_a", "hada_w1_b", "alpha", "hada_w2_a", "hada_w2_b")
    )
    tensors = {n: t for n, t in zip(names, w, strict=False) if t is not None}
    if "alpha" in tensors:
        tensors["alpha"] = torch.as_tensor(tensors["alpha"])
    found = lycoris_delta(tensors)
    assert found is not None
    return found[0]


def group_state(
    load: Load, patches: dict[str, list], binding: Binding, cut: int
) -> tuple[dict[str, torch.Tensor], list[str], list[float]]:
    state: dict[str, torch.Tensor] = {}
    not_targets: list[str] = []
    errors: list[float] = []
    for key, i in load.entries.items():
        path = key[len(COMFY_PREFIX) : -len(".weight")]
        targets = binding.from_comfy(path)
        if not targets:
            not_targets.append(key)
            continue
        kind, w = _adapter(patches[key][i][1])
        parts = _delta_parts(kind, w)
        if isinstance(parts, torch.Tensor):
            down, up, error = low_rank(parts, cut)
            errors.append(error)
        else:
            down, up = parts
        for target, part in targets:
            up_part = up.chunk(part[1], dim=0)[part[0]] if part else up
            state[f"transformer.{target}.lora_A.weight"] = down.clone().contiguous()
            state[f"transformer.{target}.lora_B.weight"] = up_part.clone().contiguous()
    return state, not_targets, errors


def _write(state: dict[str, torch.Tensor], index: int, *, path=None, metadata=None) -> str:
    if path is not None:
        save_file(state, str(path), metadata=metadata)
        return str(path)
    digest = hashlib.sha256()
    for k in sorted(state):
        digest.update(k.encode())
        digest.update(state[k].numpy().tobytes())
    folder = os.path.join(tempfile.gettempdir(), "kilora-takeover")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"loaded-{index + 1}-{digest.hexdigest()[:12]}.safetensors")
    if not os.path.exists(path):
        save_file(state, path)
    return path


MERGE_NOTE = (
    "LoRAs with the same strength, kind, rank, alpha and data type on disjoint layers "
    "look like one LoRA here and route as one expert. To route them separately, give "
    "them different strengths."
)


def _blocks(keys) -> list[int]:
    found = set()
    for key in keys:
        for part in key.split("."):
            if part.isdigit():
                found.add(int(part))
                break
    return sorted(found)


def _ranges(blocks: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for b in blocks:
        if out and b == out[-1][1] + 1:
            out[-1] = (out[-1][0], b)
        else:
            out.append((b, b))
    return out


def take_over(model: Any, settings: dict | None = None) -> tuple[Any, str]:
    from .nodes import COMFY_DEPTH_STEPS
    from .settings import resolve_settings

    given = dict(settings or {})
    lycoris_rank = given.pop("lycoris_rank", None)
    settings = resolve_settings(given)
    patches = dict(getattr(model, "patches", {}) or {})
    if not patches:
        return model.clone(), "No LoRA patches on the model: nothing to take over."
    if settings.get("mix") == "stack":
        return model.clone(), "Mix is stack: the LoRA patches stay as they are."
    grouping = group_patches(patches)
    binding = patching.detect_family(model)
    plain = [load.signature[2] for load in grouping.loads if load.signature[1] == "lora"]
    cut = int(lycoris_rank or (max(plain) if plain else 32))

    chain: list[ModEntry] = []
    taken: set[tuple[str, int]] = set()
    lines: list[str] = []
    for n, load in enumerate(grouping.loads):
        bad = {
            (k, i): grouping.unsupported[(k, i)]
            for k, i in load.entries.items()
            if (k, i) in grouping.unsupported
        }
        if bad:
            reasons = sorted(set(bad.values()))
            lines.append(
                f"LoRA {n + 1} ({_describe(load.signature)}, {len(load.entries)} layers): "
                f"left as a plain LoRA, it has a {'; '.join(reasons)}."
            )
            continue
        state, not_targets, errors = group_state(load, patches, binding, cut)
        if not state:
            lines.append(
                f"LoRA {n + 1} ({_describe(load.signature)}): no layer is a routing "
                "target, left as a plain LoRA."
            )
            continue
        chain.append(ModEntry("lora", _write(state, n), load.strength))
        taken.update((k, i) for k, i in load.entries.items() if k not in not_targets)
        line = (
            f"LoRA {n + 1} ({_describe(load.signature)}): {len(load.entries)} layers, "
            f"routed at strength {load.strength:g}."
        )
        if not_targets:
            line += f" {len(not_targets)} layers that are not routing targets stay plain."
        ranges = _ranges(_blocks(load.entries))
        if len(ranges) > 1:
            spans = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)
            line += (
                f" Warning: its layers sit in separate block ranges ({spans}). This may be "
                "several LoRA files with the same signature, routed as one expert."
            )
        if errors:
            line += (
                f" Cut to rank {cut} by SVD, mean relative error {sum(errors) / len(errors):.4f}."
            )
        lines.append(line)
    for reason in grouping.ambiguous:
        lines.append(f"Ambiguous, left as plain LoRA patches: {reason}.")

    out = model.clone()
    out.patches = {
        k: [e for i, e in enumerate(v) if (k, i) not in taken] for k, v in patches.items()
    }
    out.patches = {k: v for k, v in out.patches.items() if v}
    if hasattr(out, "patches_uuid"):
        out.patches_uuid = uuid.uuid4()
    head = (
        f"Found {len(grouping.loads)} LoRA loads on the model, "
        f"{len(chain)} taken over as routed sets."
    )
    if grouping.loads:
        lines.append(MERGE_NOTE)
    if not chain:
        return out, "\n".join([head, *lines])
    out, applied = patching.apply(
        out,
        chain,
        int(settings["depth"]),
        patching.LiveEnv(),
        settings.get("iteration"),
        settings.get("depth_schedule"),
        settings.get("routing_scope"),
        settings.get("routing_select"),
        lycoris_apply=settings["lycoris_apply"],
        gate_stats=settings["gate_stats"],
        routing_features_enabled=settings["routing_features"],
        cross_weights=tuple(
            settings.get(k, 0.0)
            for k in ("cross_token_region", "cross_token_prompt", "cross_token_timestep")
        ),
        joint=True,
        depth_steps=COMFY_DEPTH_STEPS,
    )
    return out, "\n".join([head, *lines, applied])


class KiLoRARouteLoaded:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {"settings": (SETTINGS,)},
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "route"
    CATEGORY = "KiLoRA"

    def route(self, model, settings=None):
        return take_over(model, settings)
