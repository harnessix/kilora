# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .base_weights import BaseWeights, is_quantized_weight, read_base_weight
from .compute import orthonormal_keys
from .errors import LoRAImportError
from .families import normalize_for
from .lora_import import (
    _TEXT_ENCODER_PREFIXES,
    _license,
    _never_target,
    _never_target_changes,
    _read_config,
    _read_file,
    _resolver,
    _scale,
    _split_key,
    _targets_text_encoder,
)
from .presets import Alias, get_preset
from .sections import Section, fixed_sections, mount_kind
from .sets import AdapterSet, KeyedExperts, make_router
from .text_encoder_lora import is_text_encoder_key, text_encoder_sections

__all__ = [
    "LoRASource",
    "MultiImport",
    "import_loras",
    "lora_file_size",
    "lycoris_delta",
    "low_rank",
]

log = logging.getLogger("kilora.import")

_LYCORIS_SUFFIXES = {
    ".lokr_w1": "lokr_w1",
    ".lokr_w2": "lokr_w2",
    ".lokr_w1_a": "lokr_w1_a",
    ".lokr_w1_b": "lokr_w1_b",
    ".lokr_w2_a": "lokr_w2_a",
    ".lokr_w2_b": "lokr_w2_b",
    ".hada_w1_a": "hada_w1_a",
    ".hada_w1_b": "hada_w1_b",
    ".hada_w2_a": "hada_w2_a",
    ".hada_w2_b": "hada_w2_b",
}


def lycoris_delta(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str] | None:

    t = {k: v.float() for k, v in tensors.items()}
    alpha = float(t["alpha"].reshape(-1)[0]) if "alpha" in t else None
    if any(k.startswith("lokr_") for k in t):
        rank = None
        if "lokr_w1" in t:
            w1 = t["lokr_w1"]
        else:
            w1, rank = t["lokr_w1_a"] @ t["lokr_w1_b"], t["lokr_w1_b"].shape[0]
        if "lokr_w2" in t:
            w2 = t["lokr_w2"]
        else:
            w2, rank = t["lokr_w2_a"] @ t["lokr_w2_b"], t["lokr_w2_b"].shape[0]
        scale = 1.0 if alpha is None or rank is None else alpha / rank
        return torch.kron(w1, w2) * scale, "lokr"
    if any(k.startswith("hada_") for k in t):
        rank = t["hada_w1_b"].shape[0]
        scale = 1.0 if alpha is None else alpha / rank
        delta = (t["hada_w1_a"] @ t["hada_w1_b"]) * (t["hada_w2_a"] @ t["hada_w2_b"])
        return delta * scale, "loha"
    return None


def _lycoris_used_parts(tensors: dict, kind: str) -> set[str]:
    parts = {"alpha"} if "alpha" in tensors else set()
    if kind == "lokr":
        for side in ("w1", "w2"):
            whole = f"lokr_{side}"
            parts.update((whole,) if whole in tensors else (f"{whole}_a", f"{whole}_b"))
    else:
        parts.update(("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
    return parts


def _lycoris_shape(tensors: dict[str, torch.Tensor]) -> tuple[tuple[int, int], str] | None:
    if not any(k.startswith("lokr_t") for k in tensors):
        pair = lokr_parts(tensors)
        if pair is not None and pair[0].ndim == 2 and pair[1].ndim == 2:
            (a, b), (c, d) = pair[0].shape, pair[1].shape
            return (a * c, b * d), "lokr"
    found = lycoris_delta(tensors)
    return None if found is None else (tuple(found[0].shape), found[1])


def lokr_parts(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
    t = {k: v.float() for k, v in tensors.items()}
    if not any(k.startswith("lokr_") for k in t):
        return None
    alpha = float(t["alpha"].reshape(-1)[0]) if "alpha" in t else None
    rank = None
    if "lokr_w1" in t:
        w1 = t["lokr_w1"]
    else:
        w1, rank = t["lokr_w1_a"] @ t["lokr_w1_b"], t["lokr_w1_b"].shape[0]
    if "lokr_w2" in t:
        w2 = t["lokr_w2"]
    else:
        w2, rank = t["lokr_w2_a"] @ t["lokr_w2_b"], t["lokr_w2_b"].shape[0]
    scale = 1.0 if alpha is None or rank is None else alpha / rank
    return w1 * scale, w2


def _exact_lycoris(pending, family, records, sections, source_keys) -> list:
    from .bindings import BINDINGS, COMFY_PREFIX
    from .sections import Section

    binding = BINDINGS.get(family or "")
    kron: dict[int, dict] = {}
    dense: dict[int, dict] = {}
    rest = []
    for e, target, tensors, weight, kind in pending:
        place = binding.to_comfy(target) if binding is not None else None
        if place is not None and place.part is not None:
            rest.append((e, target, tensors, weight, kind))
            continue
        path = place.path.removeprefix(COMFY_PREFIX) if place is not None else target
        keys = source_keys[e, target]
        coverage = records[e]["coverage"]
        coverage["routed"] = sorted(set(coverage["routed"]) - keys)
        coverage["scaled-global"] = sorted(set(coverage["scaled-global"]) | keys)
        records[e]["coverage_counts"] = {kind: len(keys) for kind, keys in coverage.items()}
        records[e]["global_mount_counts"][mount_kind(path)] += len(keys)
        pair = lokr_parts(tensors) if kind == "lokr" else None
        if pair is not None:
            kron.setdefault(e, {})[path] = (pair[0] * weight, pair[1])
        else:
            dense.setdefault(e, {})[path] = lycoris_delta(tensors)[0] * weight
    for e in sorted(set(kron) | set(dense)):
        name = records[e]["name"]
        if kron.get(e):
            sections.append(
                Section("fixed-lokr", name, kron[e], {"layers": len(kron[e]), "lycoris": True})
            )
        if dense.get(e):
            sections.append(
                Section("fixed-delta", name, dense[e], {"layers": len(dense[e]), "lycoris": True})
            )
        records[e]["lycoris"] = "exact"
    return rest


def low_rank(
    delta: torch.Tensor, rank: int, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor, float]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    d = delta.to(device, torch.float32)
    q = min(rank + 8, *d.shape)
    devices = [d.device] if d.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        u, sv, v = torch.svd_lowrank(d, q=q, niter=2)
    r = min(rank, sv.shape[0])
    down = v[:, :r].t()
    up = u[:, :r] * sv[:r]
    norm = float(d.norm())
    error = float((d - up @ down).norm()) / norm if norm else 0.0
    return down.cpu().contiguous(), up.cpu().contiguous(), error


LOKR_ENERGY = 0.95
LOKR_BUDGET_MIB = 1024


@dataclass
class _KronSpectrum:
    values: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    u1: torch.Tensor
    v1: torch.Tensor
    u2: torch.Tensor
    v2: torch.Tensor

    @property
    def shares(self) -> torch.Tensor:
        e = self.values.double().pow(2)
        total = e.sum()
        return e / total if total > 0 else e

    def cut(
        self, rank: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        r = min(rank, self.values.shape[0])
        i, j, sv = self.i[:r], self.j[:r], self.values[:r]
        v1, v2, u1, u2 = (t.to(device) for t in (self.v1, self.v2, self.u1, self.u2))
        i, j, sv = i.to(device), j.to(device), sv.to(device)
        down = (v1[i][:, :, None] * v2[j][:, None, :]).reshape(r, -1).cpu()
        up = (u1[:, i].t()[:, :, None] * u2[:, j].t()[:, None, :]).reshape(r, -1)
        up = (up * sv[:, None]).cpu()
        kept = float(self.shares[:r].sum())
        error = math.sqrt(max(0.0, 1.0 - kept))
        return down.contiguous(), up.t().contiguous(), error


def _kron_parts_svd(
    tensors: dict[str, torch.Tensor], device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor] | None:
    if any(k.startswith("lokr_t") for k in tensors):
        return None
    pair = lokr_parts(tensors)
    if pair is None or pair[0].ndim != 2 or pair[1].ndim != 2:
        return None
    out = {}
    for n, w in zip(("1", "2"), pair, strict=True):
        u, sv, v = torch.linalg.svd(w.to(device, torch.float32), full_matrices=False)
        out |= {f"u{n}": u.cpu(), f"s{n}": sv.cpu(), f"v{n}": v.cpu()}
    return out


def _spectrum_from_parts(parts: dict[str, torch.Tensor], weight: float) -> _KronSpectrum:
    s1, s2 = parts["s1"] * abs(float(weight)), parts["s2"]
    u1 = parts["u1"] * (-1.0 if weight < 0 else 1.0)
    values, order = torch.sort((s1[:, None] * s2[None, :]).flatten(), descending=True, stable=True)
    n2 = s2.shape[0]
    return _KronSpectrum(values, order // n2, order % n2, u1, parts["v1"], parts["u2"], parts["v2"])


def _kron_spectrum(tensors: dict[str, torch.Tensor], weight: float) -> _KronSpectrum | None:
    parts = _kron_parts_svd({k: v.float() for k, v in tensors.items()})
    return None if parts is None else _spectrum_from_parts(parts, weight)


_LOKR_CACHE_FORMAT = 1
_LOKR_CACHE_LOCK = threading.RLock()


def _lokr_parts_key(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(k for k in tensors if k == "alpha" or k.startswith("lokr_")):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(repr((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "@" + digest.hexdigest()


def _lokr_cache_path(sha256: str, device="cpu") -> str:
    from .derived_cache import backend
    from .key_cache import cache_dir

    signature = json.dumps(backend(device), sort_keys=True)
    backend_id = hashlib.sha256(signature.encode()).hexdigest()[:20]
    return os.path.join(
        cache_dir(), "lokr", f"{sha256}-v{_LOKR_CACHE_FORMAT}-{backend_id}.safetensors"
    )


def _load_lokr_cache(sha256: str | None, device="cpu") -> dict[str, dict[str, torch.Tensor]]:
    if not sha256 or os.environ.get("KILORA_LOKR_CACHE", "1") == "0":
        return {}
    path = _lokr_cache_path(sha256, device)
    if not os.path.isfile(path):
        return {}
    try:
        from safetensors.torch import load_file

        flat = load_file(path)
    except Exception:
        return {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in flat.items():
        target, _, name = key.rpartition("/")
        out.setdefault(target, {})[name] = value
    return out


def _save_lokr_cache(
    sha256: str | None, by_target: dict[str, dict[str, torch.Tensor]], device="cpu"
) -> None:
    if not sha256 or not by_target or os.environ.get("KILORA_LOKR_CACHE", "1") == "0":
        return
    with _LOKR_CACHE_LOCK:
        _save_lokr_cache_locked(sha256, by_target, device)


def _save_lokr_cache_locked(
    sha256: str, by_target: dict[str, dict[str, torch.Tensor]], device="cpu"
) -> None:
    try:
        import tempfile

        from safetensors.torch import save_file

        path = _lokr_cache_path(sha256, device)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        by_target = _load_lokr_cache(sha256, device) | by_target
        flat = {
            f"{t}/{n}": v.contiguous() for t, parts in by_target.items() for n, v in parts.items()
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        os.close(fd)
        try:
            save_file(flat, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError as exc:
        log.info("LoKr cache not written (%s)", exc)


def _lokr_ranks(
    spectra: dict[str, _KronSpectrum],
    base: int,
    energy: float,
    budget_bytes: float,
    cap: int | None = None,
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    extra: list[tuple[float, int, str]] = []
    for target, spec in spectra.items():
        shares = spec.shares
        n = max(1, int((shares > 1e-12).sum()))
        need = int((shares.cumsum(0) < energy).sum()) + 1
        ranks[target] = min(base, n)
        cost = 4 * (spec.v1.shape[1] * spec.v2.shape[1] + spec.u1.shape[0] * spec.u2.shape[0])
        top = min(need, n) if cap is None else min(need, n, max(cap, ranks[target]))
        for k in range(ranks[target], top):
            extra.append((float(shares[k]) / cost, cost, target))
    extra.sort(key=lambda item: (-item[0], item[2]))
    used = 0.0
    for _, cost, target in extra:
        if used + cost > budget_bytes:
            continue
        used += cost
        ranks[target] += 1
    return ranks


ENERGY_WARNING = 0.9


@dataclass
class LoRASource:
    path: str
    weight: float = 1.0
    name: str | None = None
    repo: str | None = None
    revision: str | None = None
    license: str | None = None


@dataclass
class MultiImport:
    adapter_set: AdapterSet | None
    sources: list[dict[str, Any]]
    unmatched: dict[str, list[str]] = field(default_factory=dict)
    matched_keys: dict[str, int] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    content_key_counts: tuple[int, int] | None = None

    def cut_notes(self) -> list[str]:
        notes = []
        for src in self.sources:
            energy = src.get("svd_energy")
            if not energy:
                continue
            exact = src.get("exact_scoring_ranks", {})
            if exact:
                ranks = sorted(exact.values())
                notes.append(
                    f"{src['name']}: lycoris_apply=exact; energy kept 100% (exact change) "
                    f"on {len(exact)} "
                    f"LyCORIS layers; routing scores use the SVD cut at rank "
                    f"{ranks[0]} to {ranks[-1]}."
                )
                energy = {target: value for target, value in energy.items() if target not in exact}
                if not energy:
                    continue
            fallback = src.get("exact_fallback", [])
            if fallback:
                notes.append(
                    f"{src['name']}: exact components exceeded the scoring matrix storage "
                    f"for: {', '.join(sorted(fallback))}; these layers use the SVD change."
                )
            kept = sorted(energy.values())
            low = sorted((v, t) for t, v in energy.items() if v < ENERGY_WARNING)
            ranks = sorted(
                rank for target, rank in src.get("svd_ranks", {}).items() if target in energy
            )
            capped = (
                f", capped at {src['rank_cap']} per expert so the fused kernels fit"
                if src.get("rank_cap")
                else ""
            )
            cut = (
                f"the SVD cut (LoKr ranks {ranks[0]} to {ranks[-1]}, median "
                f"{ranks[len(ranks) // 2]}{capped}; other layers rank {src['svd_rank']})"
                if ranks
                else f"the rank-{src['svd_rank']} cut"
            )
            notes.append(
                f"{src['name']}: lycoris_apply=truncated; energy kept by {cut} per layer: "
                f"min {kept[0]:.1%}, median {kept[len(kept) // 2]:.1%}, max {kept[-1]:.1%} "
                f"({len(kept)} layers). lycoris_apply=exact applies the whole change."
            )
            if low:
                worst = ", ".join(f"{t} {v:.1%}" for v, t in low[:5])
                notes.append(
                    f"Warning: {src['name']}: {len(low)} of {len(kept)} layers keep less "
                    f"than {ENERGY_WARNING:.0%} of the change in routed mode (lowest: "
                    f"{worst}). KILORA_LOKR_RANKS=energy lifts the fused-kernel cap (slower), "
                    f"KILORA_LOKR_BUDGET_MIB raises the memory budget, mix=stack gives the "
                    f"exact change."
                )
        return notes

    def content_key_summary(self) -> str:
        return _content_key_summary(*self.content_key_counts)

    def report(self) -> str:
        lines = [
            f"Imported {len(self.sources)} LoRAs as {len(self.sources)} experts at "
            f"{len(self.adapter_set.experts) if self.adapter_set else 0} target layers."
        ]
        if self.content_key_counts is not None:
            lines.append(self.content_key_summary())
        for i, src in enumerate(self.sources):
            scale = src["scale"]
            scale_text = f"{scale:g}" if isinstance(scale, float) else str(scale)
            skipped = len(self.unmatched.get(src["name"], []))
            line = (
                f"Expert {i}: {src['name']} ({src['format']} layout), rank {src['rank']}, "
                f"scale {scale_text}, weight {src['weight']:g}, "
                f"{self.matched_keys.get(src['name'], 0)} keys matched, {skipped} skipped."
            )
            if (
                "svd_error" in src
                and "exact_scoring_ranks" not in src
                and "exact_fallback" not in src
            ):
                line += (
                    f" LyCORIS layers cut to rank {src['svd_rank']} by SVD, mean relative "
                    f"error {src['svd_error']:.4f}."
                )
            if "size_scale" in src:
                from .size_norm import size_note

                line += " " + size_note(
                    "Size",
                    src["size"],
                    src["size_scale"],
                    src.get("applied_scale"),
                    src.get("size_rule"),
                )
            lines.append(line)
        lines += self.cut_notes()
        for section in self.sections:
            what = (
                "text encoder layers are kept for the CLIP input of KiLoRA Apply"
                if section.kind == "text-encoder"
                else f"layers that are not targets (modulation, embedders, head) are kept "
                f"as a fixed {section.kind} section"
            )
            lines.append(f"{section.name}: {section.meta.get('layers', 0)} {what}.")
        for name, keys in self.unmatched.items():
            if keys:
                lines.append(
                    f"{name}: {len(keys)} keys of layers the model does not have were "
                    f"skipped ({_key_families(keys)})."
                )
        return "\n".join(lines)


def import_loras(
    sources: Sequence[LoRASource | str],
    model: nn.Module,
    *,
    family: str | None = None,
    router_dim: int = 32,
    seed: int = 0,
    rank: int | None = None,
    full_model: nn.Module | None = None,
    lycoris_mode: str = "svd",
    lycoris_apply: str = "truncated",
    read_file: Callable | None = None,
    key_time: Callable[[float], None] | None = None,
    device: torch.device | str | None = None,
    checkpoint: Callable[[], None] | None = None,
    prepare_expert_content: bool = False,
    content_keys: str = "per_expert",
    size_norm: str = "off",
    base_weight: Callable[[str], torch.Tensor | None] | None = None,
) -> MultiImport:

    if size_norm not in ("off", "weight"):
        raise ValueError("size_norm must be off or weight")
    lycoris_apply = os.environ.get("KILORA_LYCORIS_APPLY") or lycoris_apply
    if lycoris_apply not in ("truncated", "exact"):
        raise ValueError("lycoris_apply must be truncated or exact")
    srcs = [LoRASource(path=s) if isinstance(s, (str, os.PathLike)) else s for s in sources]
    if not srcs:
        raise ValueError("import_loras needs at least one LoRA file")
    for s in srcs:
        s.path = os.fspath(s.path)
    sizes: list[tuple[float | None, float]] = []
    applied: list[tuple[float, str]] = []
    if size_norm == "weight":
        from dataclasses import replace

        from .size_norm import applied_scale, size_scale

        for s in srcs:
            size = lora_file_size(
                s.path,
                model,
                family=family,
                full_model=full_model,
                base_weight=base_weight,
                read_file=read_file,
            )
            sizes.append((size, size_scale(size)))
        applied = [applied_scale(c) for _, c in sizes]
        srcs = [
            replace(s, weight=float(s.weight) * a) for s, (a, _) in zip(srcs, applied, strict=True)
        ]

    modules = {n: m for n, m in model.named_modules() if n}
    linear_names = [n for n, m in modules.items() if isinstance(m, nn.Linear)]
    aliases: dict[str, Alias] = get_preset(family).aliases(linear_names) if family else {}
    order = {n: i for i, n in enumerate(modules)}

    n_experts = len(srcs)
    parts: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    unit_ups: dict[str, dict[int, torch.Tensor]] = {}
    records: list[dict[str, Any]] = []
    file_dtypes: set[torch.dtype] = set()
    unmatched: dict[str, list[str]] = {}
    matched_keys: dict[str, int] = {}
    sections: list[Section] = []
    pending: list[tuple[int, str, torch.Tensor, str]] = []
    plain_ranks: list[int] = []
    source_keys: dict[tuple[int, str], set[str]] = {}
    for e, src in enumerate(srcs):
        if checkpoint is not None:
            checkpoint()
        name = src.name or os.path.splitext(os.path.basename(src.path))[0]
        if name in unmatched:
            raise LoRAImportError(f"two LoRAs have the name {name!r}; give each a unique name")
        if read_file is None and checkpoint is not None:
            state, metadata, file_record = _read_file(src.path, checkpoint=checkpoint)
        else:
            state, metadata, file_record = (read_file or _read_file)(src.path)
        original_keys = set(state)
        clip_keys = (
            {k for k in state if is_text_encoder_key(k)}
            if not _targets_text_encoder(family)
            else set()
        )
        te_skipped = []
        if not _targets_text_encoder(family):
            te_sections, te_skipped = text_encoder_sections(name, state, src.weight)
            sections.extend(te_sections)
            state = {k: v for k, v in state.items() if not is_text_encoder_key(k)}
        norm = normalize_for(family, state, model, full_model)
        state, excluded = norm.state, norm.skipped
        own = fixed_sections(
            name, norm.fixed, norm.fixed_dense, src.weight, expert_index=e, expert_count=n_experts
        )
        if own:
            sections.extend(own)
            log.info(
                "%s: %d keys on %d layers that are not targets apply as a fixed change",
                src.path,
                norm.fixed_keys,
                len(norm.fixed) + len(norm.fixed_dense),
            )
        file = file_record["original_name"]
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(src.path)), "adapter_config.json"
        )
        config = _read_config(config_path) if os.path.isfile(config_path) else None

        groups, skipped, layout = _group(state, file)
        resolve = _resolver(layout, list(modules), aliases, strict=False)
        skipped_keys = [*excluded, *te_skipped, *skipped]
        ranks: set[int] = set()
        alphas: set[float] = set()
        scales: set[float] = set()
        targets: set[str] = set()
        layout_kinds: set[str] = set()
        n_matched = 0
        routed_keys: set[str] = set()
        global_keys: set[str] = set(norm.fixed_originals)
        global_paths = dict(norm.fixed_original_paths)
        never: dict = {}
        missing_keys: set[str] = set()
        for stem, (tensors, keys) in groups.items():
            if checkpoint is not None:
                checkpoint()
            found = resolve(stem)
            if found:
                file_dtypes.update(
                    v.dtype for k, v in tensors.items() if k != "alpha" and v.is_floating_point()
                )
            if not found:
                body = stem.removeprefix("lora_unet_").removeprefix("diffusion_model.")
                if _never_target(body) and "down" in tensors and "up" in tensors:
                    r = tensors["down"].shape[0]
                    scale = float(src.weight) * _scale(
                        f"{file}: {stem}", tensors.get("alpha"), r, config
                    )
                    never[stem] = (tensors["down"].float(), tensors["up"].float() * scale)
                    continue
                if _never_target(body) and (lyco := lycoris_delta(tensors)) is not None:
                    never[stem] = lyco[0] * float(src.weight)
                    continue
                skipped_keys.extend(keys.values())
                missing_keys.update(keys.values())
                continue
            lycoris = _lycoris_shape(tensors) if "down" not in tensors else None
            if lycoris is not None:
                delta, kind = lycoris
                if len(found) != 1:
                    raise LoRAImportError(f"{file}: {stem}: a {kind} layer on a fused module")
                target = found[0]
                module = modules[target]
                if tuple(delta) != (module.out_features, module.in_features):
                    raise LoRAImportError(
                        f"{file}: {stem}: the {kind} change is {tuple(delta)}, "
                        f"{target} is {(module.out_features, module.in_features)}"
                    )
                if e in parts.get(target, {}) or any(p[:2] == (e, target) for p in pending):
                    raise LoRAImportError(f"{file}: two entries change {target}")
                pending.append((e, target, tensors, float(src.weight), kind))
                del delta
                targets.add(target)
                used_parts = _lycoris_used_parts(tensors, kind)
                routed_keys.update(key for part, key in keys.items() if part in used_parts)
                source_keys[e, target] = {
                    original
                    for part, key in keys.items()
                    if part in used_parts
                    for original in norm.origins.get(key, [key])
                    if original in original_keys
                }
                skipped_keys.extend(key for part, key in keys.items() if part not in used_parts)
                n_matched += sum(part in used_parts for part in keys)
                layout_kinds.add(kind)
                continue
            if "down" not in tensors or "up" not in tensors:
                found_parts = sorted(keys)
                raise LoRAImportError(
                    f"{file}: {stem}: unknown format, the layer has the weights "
                    f"{found_parts} (plain LoRA needs a down and an up weight; LoKr "
                    "lokr_w1/lokr_w2; LoHa hada_w1_a/hada_w1_b/hada_w2_a/hada_w2_b)"
                )
            skipped_keys.extend(
                k for part, k in keys.items() if part not in ("down", "up", "alpha")
            )
            down, up = tensors["down"].float(), tensors["up"].float()
            if down.dim() != 2 or up.dim() != 2:
                raise LoRAImportError(
                    f"{file}: key {keys['down']}: only LoRA on linear layers is supported"
                )
            rank = down.shape[0]
            if up.shape[1] != rank:
                raise LoRAImportError(
                    f"{file}: key {keys['up']}: rank {up.shape[1]} does not match the "
                    f"rank {rank} of {keys['down']}"
                )
            if len(found) > 1 and up.shape[0] % len(found):
                raise LoRAImportError(
                    f"{file}: key {keys['up']}: fused output of {up.shape[0]} rows does "
                    "not split evenly"
                )
            base = _scale(f"{file}: {stem}", tensors.get("alpha"), rank, config)
            scale = float(src.weight) * base
            chunks = up.chunk(len(found), dim=0) if len(found) > 1 else (up,)
            for target, up_part in zip(found, chunks, strict=True):
                module = modules[target]
                if not isinstance(module, nn.Linear):
                    raise LoRAImportError(f"{file}: key {keys['down']}: {target} is not linear")
                if down.shape[1] != module.in_features:
                    raise LoRAImportError(
                        f"{file}: key {keys['down']}: input size {down.shape[1]} does not "
                        f"fit {target} (in_features {module.in_features})"
                    )
                if up_part.shape[0] != module.out_features:
                    raise LoRAImportError(
                        f"{file}: key {keys['up']}: output size {up_part.shape[0]} does "
                        f"not fit {target} (out_features {module.out_features})"
                    )
                by_expert = parts.setdefault(target, {})
                if e in by_expert:
                    raise LoRAImportError(f"{file}: two LoRA entries change {target}")
                by_expert[e] = (down.clone(), up_part * scale)
                if abs(float(src.weight)) not in (0.0, 1.0):
                    unit = up_part * (_strength_sign(src.weight) * base)
                    unit_ups.setdefault(target, {})[e] = unit
                targets.add(target)
            routed_keys.update(k for part, k in keys.items() if part in ("down", "up", "alpha"))
            n_matched += sum(part in ("down", "up", "alpha") for part in keys)
            ranks.add(rank)
            plain_ranks.append(rank)
            scales.add(base)
            if "alpha" in tensors:
                alphas.add(float(tensors["alpha"].float().reshape(-1)[0]))
            elif config is not None and "lora_alpha" in config:
                alphas.add(float(config["lora_alpha"]))
        if never:
            tree = full_model if full_model is not None else model
            fixed, dense, lost = _never_target_changes(never, tree, family)
            sections.extend(
                fixed_sections(name, fixed, dense, 1.0, expert_index=e, expert_count=n_experts)
            )
            for stem in never:
                if stem not in lost:
                    tensors, keys = groups[stem]
                    if "down" in tensors and "up" in tensors:
                        used_parts = {"down", "up", "alpha"}
                    else:
                        kind = "lokr" if any(p.startswith("lokr_") for p in tensors) else "loha"
                        used_parts = _lycoris_used_parts(tensors, kind)
                    used_keys = {key for part, key in keys.items() if part in used_parts}
                    global_keys.update(used_keys)
                    for key in used_keys:
                        for original in norm.origins.get(key, [key]):
                            global_paths[original] = stem
            for stem in lost:
                skipped_keys.extend(groups[stem][1].values())
                missing_keys.update(groups[stem][1].values())
            if lost:
                log.info(
                    "%s: %d keys match no module of the model and are skipped", src.path, len(lost)
                )
        if not targets:
            raise LoRAImportError(f"{file}: no key matches a layer of the model")

        def originals(keys, origins=norm.origins):
            return {original for key in keys for original in origins.get(key, [key])}

        routed_keys = originals(routed_keys) & original_keys
        global_keys = originals(global_keys) & original_keys
        skipped_originals = original_keys - routed_keys - global_keys
        missing_originals = originals(missing_keys)
        coverage = {
            "routed": sorted(routed_keys),
            "scaled-global": sorted(global_keys),
            "skipped": sorted(skipped_originals),
        }
        reasons = {
            k: (
                "text encoder kept separately"
                if k in clip_keys and k not in te_skipped
                else "no matching model layer or shape"
                if k in missing_originals
                else "unsupported or excluded weight"
            )
            for k in skipped_originals
        }
        license_tag = src.license or _license(metadata)
        records.append(
            {
                "name": name,
                "file": file,
                "sha256": file_record["digest"].removeprefix("sha256-"),
                "repo": src.repo,
                "revision": src.revision,
                "license": "NOASSERTION" if license_tag == "unknown" else license_tag,
                "format": "+".join([layout, *sorted(layout_kinds)]),
                "rank": max(ranks) if ranks else None,
                "alpha": alphas.pop() if len(alphas) == 1 else None,
                "scale": scales.pop() if len(scales) == 1 else ("per-module" if scales else 1.0),
                "weight": float(src.weight),
                "targets": len(targets),
                "coverage": coverage,
                "coverage_counts": {kind: len(keys) for kind, keys in coverage.items()},
                "global_mount_counts": {
                    kind: sum(mount_kind(global_paths.get(key, key)) == kind for key in global_keys)
                    for kind in ("adaln", "text-adapter", "embedder", "final", "other")
                },
                "skipped_reasons": reasons,
            }
        )
        if sizes:
            records[-1]["size"], records[-1]["size_scale"] = sizes[e]
            records[-1]["applied_scale"], records[-1]["size_rule"] = applied[e]
        unmatched[name] = skipped_keys
        matched_keys[name] = n_matched

    if lycoris_mode == "exact":
        pending = _exact_lycoris(pending, family, records, sections, source_keys)
    elif lycoris_mode != "svd":
        raise ValueError(f"lycoris_mode must be 'svd' or 'exact', not {lycoris_mode!r}")
    exact_pending = (
        list(pending)
        if n_experts > 1 and lycoris_mode == "svd" and lycoris_apply == "exact"
        else []
    )
    cut = rank or (max(plain_ranks) if plain_ranks else 32)
    errors: dict[int, list[float]] = {}
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    budget = float(os.environ.get("KILORA_LOKR_BUDGET_MIB", LOKR_BUDGET_MIB)) * 2**20
    spectra: dict[int, dict[str, _KronSpectrum]] = {}
    rest = []
    cached = {
        e: _load_lokr_cache(records[e].get("sha256"), device) if read_file is None else {}
        for e in {p[0] for p in pending}
    }
    computed: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for item in pending:
        if checkpoint is not None:
            checkpoint()
        e, target, tensors, weight, kind = item
        part_key = _lokr_parts_key(tensors) if kind == "lokr" else None
        svd_parts = (cached[e].get(target) or cached[e].get(part_key)) if part_key else None
        if kind == "lokr" and svd_parts is None:
            svd_parts = _kron_parts_svd({k: v.float() for k, v in tensors.items()}, device)
            if svd_parts is not None:
                computed.setdefault(e, {})[part_key] = svd_parts
        if svd_parts is not None:
            spectra.setdefault(e, {})[target] = _spectrum_from_parts(
                svd_parts, _strength_sign(weight)
            )
        else:
            rest.append(item)
    if read_file is None:
        for e, by_target in computed.items():
            if checkpoint is not None:
                checkpoint()
            _save_lokr_cache(records[e].get("sha256"), cached[e] | by_target, device)
    cap = None
    if os.environ.get("KILORA_LOKR_RANKS", "fused").strip() != "energy":
        from .adapter import IMPORTED_SCORE
        from .fused import MAX_EXPERTS, max_columns

        tiled = (os.environ.get("KILORA_IMPORT_SCORE") or IMPORTED_SCORE) == "read_share"
        columns = max_columns(router_dim, tiled=tiled) if n_experts <= MAX_EXPERTS else None
        cap = None if columns is None else max(1, columns // n_experts)
    for e, by_target in spectra.items():
        if checkpoint is not None:
            checkpoint()
        ranks = _lokr_ranks(by_target, cut, LOKR_ENERGY, budget, cap)
        records[e]["rank_cap"] = cap
        for target, spec in by_target.items():
            if checkpoint is not None:
                checkpoint()
            down, up, error = spec.cut(ranks[target], device)
            weight = abs(float(records[e]["weight"]))
            if weight not in (0.0, 1.0):
                unit_ups.setdefault(target, {})[e] = up
                up = up * weight
            parts.setdefault(target, {})[e] = (down, up)
            errors.setdefault(e, []).append(error)
            records[e].setdefault("svd_energy", {})[target] = 1.0 - error * error
            records[e].setdefault("svd_ranks", {})[target] = int(down.shape[0])
            log.info(
                "%s: lokr %s at rank %d, relative error %.4f",
                records[e]["name"],
                target,
                down.shape[0],
                error,
            )
    pending = rest
    for e, target, tensors, weight, kind in pending:
        if checkpoint is not None:
            checkpoint()
        from . import derived_cache

        sign = _strength_sign(weight)
        source = (
            derived_cache.identity(
                [{**records[e], "weight": sign}], target, {"cut": cut, "kind": kind}
            )
            if read_file is None
            else None
        )
        settings = {"cut": cut, "kind": kind, "backend": derived_cache.backend(device)}
        saved = derived_cache.load(source, "conversion", settings, ("down", "up", "error"))
        if saved is None:
            delta = lycoris_delta({k: v.to(device) for k, v in tensors.items()})[0] * sign
            down, up, error = low_rank(delta, cut, device)
            del delta
            derived_cache.store(
                source,
                "conversion",
                settings,
                ("down", "up", "error"),
                (down, up, torch.tensor(error)),
            )
        else:
            down, up, error_tensor = saved
            error = float(error_tensor.item())
        if abs(float(weight)) not in (0.0, 1.0):
            unit_ups.setdefault(target, {})[e] = up
            up = up * abs(float(weight))
        parts.setdefault(target, {})[e] = (down, up)
        errors.setdefault(e, []).append(error)
        records[e].setdefault("svd_energy", {})[target] = 1.0 - error * error
        log.info(
            "%s: %s %s at rank %d, relative error %.4f",
            records[e]["name"],
            kind,
            target,
            cut,
            error,
        )
    for e, errs in errors.items():
        ranks = records[e].get("svd_ranks", {}).values()
        records[e]["rank"] = max(records[e]["rank"] or 0, cut, *ranks)
        records[e]["svd_rank"] = cut
        records[e]["svd_error"] = sum(errs) / len(errs)
        log.info(
            "%s: mean relative error of the rank-%d cut: %.4f",
            records[e]["name"],
            cut,
            records[e]["svd_error"],
        )

    from .exact_lycoris import make_exact

    exact_parts = {}
    for e, target, tensors, weight, kind in exact_pending:
        down, up = parts[target][e]
        exact_budget = (down.numel() + up.numel()) * down.element_size()
        components = make_exact(tensors, kind, weight, exact_budget)
        if components is not None:
            exact_parts.setdefault(target, {})[e] = components
            records[e].setdefault("exact_scoring_ranks", {})[target] = int(down.shape[0])
        else:
            records[e].setdefault("exact_fallback", []).append(target)

    router_seed = get_preset(family).router_seed if family else 0
    from . import derived_cache
    from .key_cache import file_sha256

    config_hashes = []
    for src in srcs:
        path = os.path.join(os.path.dirname(os.path.realpath(src.path)), "adapter_config.json")
        config_hashes.append(file_sha256(path) if os.path.isfile(path) else None)

    def layout_signature(tree):
        if tree is None:
            return None
        layout = [
            (
                name,
                type(module).__module__,
                type(module).__qualname__,
                getattr(module, "in_features", None),
                getattr(module, "out_features", None),
            )
            for name, module in tree.named_modules()
        ]
        return hashlib.sha256(json.dumps(layout, default=str).encode()).hexdigest()

    cache_settings = {
        "configs": config_hashes,
        "family": family,
        "model_layout": layout_signature(model),
        "full_layout": layout_signature(full_model),
        "cut": cut,
        "mode": lycoris_mode,
        "lycoris_apply": lycoris_apply,
        "router_dim": router_dim,
        "router_seed": router_seed,
        "cap": cap,
        "budget": budget,
        "backend": derived_cache.backend(device),
        "ranks": os.environ.get("KILORA_LOKR_RANKS", "fused"),
    }
    keyed: dict[str, KeyedExperts] = {}
    content_fallbacks = 0
    content_single = 0
    per_expert = os.environ.get("KILORA_CONTENT_KEYS", content_keys) == "per_expert"
    routers: dict[int, torch.Tensor] = {}
    for target in sorted(parts, key=order.__getitem__):
        if checkpoint is not None:
            checkpoint()
        by_expert = parts[target]
        module = modules[target]
        n_in, n_out = module.in_features, module.out_features
        rank = max(d.shape[0] for d, _ in by_expert.values())
        with torch.inference_mode(False):
            down = torch.zeros(n_experts, 1, rank, n_in)
            up = torch.zeros(n_experts, 1, rank, n_out)
        for e, (d, u) in by_expert.items():
            down[e, 0, : d.shape[0]] = d
            up[e, 0, : d.shape[0]] = u.t()
        units = unit_ups.get(target, {})
        unit_by_expert = {e: (d, units.get(e, u)) for e, (d, u) in by_expert.items()}
        if units:
            with torch.inference_mode(False):
                unit_up = up.clone()
            for e, u in units.items():
                unit_up[e, 0, : u.shape[1]] = u.t()
        else:
            unit_up = up
        router = routers.setdefault(n_out, make_router(router_dim, n_out, router_seed))
        started = time.perf_counter()
        source_id = (
            derived_cache.identity(records, target, cache_settings) if read_file is None else None
        )
        free_ids = (
            derived_cache.strength_free_identities(records, target, by_expert)
            if source_id is not None
            else None
        )
        cache_options = {
            "shape": tuple(up.shape),
            "dtype": str(up.dtype),
            "backend": derived_cache.backend(device),
            "keys_from": _KEYS_FROM,
        }
        canonical = None
        tied = False
        if per_expert or prepare_expert_content:
            canonical, tied = _per_expert_content_keys(
                unit_by_expert,
                records,
                target,
                router,
                router_seed,
                device,
                cache=read_file is None,
                identities=free_ids,
                checkpoint=checkpoint,
                return_tied=True,
            )
        if per_expert:
            content_fallbacks += int(tied)
            content_single += int(not tied)
        cached_keys = (
            None
            if per_expert and not tied
            else derived_cache.load(source_id, "content", cache_options, ("keys",))
        )
        if per_expert and not tied:
            keys = canonical
        elif cached_keys is None:
            expert_ids = (
                derived_cache.expert_identities(records, target, down, unit_up)
                if source_id is not None and n_experts == 1 and torch.device(device).type == "cpu"
                else None
            )
            if expert_ids is not None:
                options = {
                    "backend": derived_cache.backend(device),
                    "router_dim": router_dim,
                    "router_seed": router_seed,
                    "shape": tuple(up.shape[1:]),
                    "dtype": str(up.dtype),
                    "keys_from": _KEYS_FROM,
                }
                entries = [
                    derived_cache.load(identity, "expert-content", options, ("key",))
                    for identity in expert_ids
                ]
                missing = [i for i, entry in enumerate(entries) if entry is None]
                reusable = []
                if missing:
                    computed, reusable = _content_keys(
                        unit_up[missing],
                        router,
                        [records[i] for i in missing],
                        target,
                        device=device,
                        return_reusable=True,
                    )
                    for j, i in enumerate(missing):
                        entries[i] = (computed[j],)
                        if reusable[j]:
                            derived_cache.store(
                                expert_ids[i], "expert-content", options, ("key",), entries[i]
                            )
                if missing and not all(reusable):
                    keys = _content_keys(unit_up, router, records, target, device=device)
                else:
                    keys = torch.stack([entry[0] for entry in entries])
            else:
                keys = _content_keys(unit_up, router, records, target, device=device)
            derived_cache.store(source_id, "content", cache_options, ("keys",), (keys,))
        else:
            (keys,) = cached_keys
        if key_time is not None:
            key_time(time.perf_counter() - started)
        stored = _host_dtype(file_dtypes)
        if stored is not None:
            with torch.inference_mode(False):
                down, up = down.to(stored), up.to(stored)
                unit_up = up if not units else unit_up.to(stored)
        keyed[target] = KeyedExperts(
            down=down,
            up=up,
            keys_in=keys,
            exact=exact_parts.get(target, {}),
            exact_unavailable=tuple(
                e for e, record in enumerate(records) if target in record.get("exact_fallback", ())
            ),
            ranks=tuple(
                by_expert[e][0].shape[0] if e in by_expert else 0 for e in range(n_experts)
            ),
        )
        derived_cache.stamp(
            keyed[target],
            derived_cache.Source(source_id, free_ids, None if unit_up is up else unit_up)
            if source_id
            else None,
        )

    content_key_counts = (content_fallbacks, content_single) if per_expert else None
    if content_key_counts is not None:
        log.info(_content_key_summary(*content_key_counts))

    if not keyed:
        return MultiImport(
            adapter_set=None,
            sources=records,
            unmatched=unmatched,
            matched_keys=matched_keys,
            sections=sections,
            content_key_counts=content_key_counts,
        )
    adapter_set = AdapterSet(
        name=import_set_name([r["name"] for r in records]),
        part_type=("lora",),
        experts=keyed,
        routers=routers,
        aggregation="accumulate",
        form="input",
        strength=1.0,
        imported_from={"sources": records},
        extra={"trained_with_kilora": False},
        key_source="random",
        log_tau=math.log(IMPORT_TAU),
    )
    return MultiImport(
        adapter_set=adapter_set,
        sources=records,
        unmatched=unmatched,
        matched_keys=matched_keys,
        sections=sections,
        content_key_counts=content_key_counts,
    )


_KEYS_FROM = "unscaled-1"


def _host_dtype(dtypes: set[torch.dtype]) -> torch.dtype | None:
    if os.environ.get("KILORA_IMPORT_HOST_DTYPE", "file").strip() == "float32":
        return None
    if len(dtypes) == 1 and next(iter(dtypes)) in (torch.bfloat16, torch.float16):
        return next(iter(dtypes))
    return None


def _strength_sign(weight) -> float:
    weight = float(weight)
    return 1.0 if weight > 0 else -1.0 if weight < 0 else 0.0


def _content_key_summary(fallback, single):
    return f"Content keys: {fallback} layers fell back to mix; {single} used single-expert keys."


def _per_expert_content_keys(
    by_expert,
    records,
    target,
    router,
    router_seed,
    device,
    *,
    cache,
    checkpoint,
    identities=None,
    return_tied=False,
):

    from . import derived_cache

    if cache and identities is None:
        identities = derived_cache.strength_free_identities(records, target, by_expert)

    keys = []
    tied = 0
    for i, record in enumerate(records):
        if checkpoint is not None:
            checkpoint()
        if i in by_expert:
            down, up = by_expert[i]
            b = up.t()[None, None].to(torch.get_default_dtype())
        else:
            source = record.get("sha256") or record.get("name") or str(i)
            digest = hashlib.sha256(f"{source}|{target}".encode()).digest()
            gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)
            key = torch.randn(1, 1, router.shape[0], generator=gen)
            keys.append(orthonormal_keys(key)[0])
            continue
        identity = identities[i] if cache and identities else None
        options = {
            "algorithm": "per-expert-v3",
            "keys_from": _KEYS_FROM,
            "backend": derived_cache.backend(device),
            "router_dim": router.shape[0],
            "router_seed": router_seed,
            "shape": tuple(b.shape[1:]),
            "dtype": str(b.dtype),
        }
        entry = derived_cache.load(identity, "expert-content", options, ("key", "tied"))
        if entry is None:
            result, reusable = _content_keys(
                b, router, [record], target, device=device, return_reusable=True
            )
            entry = (result[0], torch.tensor(not reusable[0]))
            derived_cache.store(identity, "expert-content", options, ("key", "tied"), entry)
        keys.append(entry[0])
        tied += int(entry[1].item())
    if tied:
        log.info("%s: %d experts have nearly tied content directions", target, tied)
    result = torch.stack(keys)
    return (result, bool(tied)) if return_tied else result


def _content_keys(
    up: torch.Tensor,
    router: torch.Tensor,
    records: list,
    target: str,
    *,
    device=None,
    return_reusable=False,
):
    import hashlib

    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    r = router.to(device, torch.float32)
    v = up[:, 0].to(device, torch.float32) @ r.t()
    gram = v.double().transpose(1, 2) @ v.double()
    eigen = torch.linalg.eigh(gram)
    top = eigen.eigenvectors[..., -1].float().cpu()
    nonzero = v.abs().flatten(1).sum(1).cpu() > 0
    keys = []
    for e in range(up.shape[0]):
        if bool(nonzero[e]):
            k = top[e]
            k = k * (1.0 if float(k[k.abs().argmax()]) >= 0 else -1.0)
        else:
            source = records[e].get("sha256") or records[e].get("name") or str(e)
            digest = hashlib.sha256(f"{source}|{target}".encode()).digest()
            gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)
            k = torch.randn(router.shape[0], generator=gen)
        keys.append(k[None])
    result = orthonormal_keys(torch.stack(keys))
    if return_reusable:
        values = eigen.eigenvalues.cpu()
        if values.shape[-1] > 1:
            gap = values[:, -1] - values[:, -2]
            reusable = (~nonzero) | (gap > values[:, -1].abs() * 1e-5)
        else:
            reusable = torch.ones(len(records), dtype=torch.bool)
        return result, reusable.tolist()
    return result


def import_set_name(names: Sequence[str]) -> str:
    seen: dict[str, int] = {}
    parts = []
    for raw in names:
        part = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw)).strip("-.") or "lora"
        seen[part] = seen.get(part, 0) + 1
        parts.append(part if seen[part] == 1 else f"{part}-{seen[part]}")
    return "_".join(parts)[:64].rstrip("_-.") or "imported"


IMPORT_TAU = 1.5


def _key_families(keys: Sequence[str]) -> str:
    import re

    counts: dict[str, int] = {}
    for key in keys:
        stem = key.split(".", 1)[0]
        family = re.sub(r"_\d+(_.*)?$", "", stem) or stem
        counts[family] = counts.get(family, 0) + 1
    return ", ".join(f"{f} ({n})" for f, n in sorted(counts.items()))


def _split_any(key: str) -> tuple[str, str] | None:
    split = _split_key(key)
    if split is not None:
        return split
    for suffix, part in _LYCORIS_SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    if "." in key:
        stem, part = key.rsplit(".", 1)
        if stem.startswith(("lora_unet_", "diffusion_model.")):
            return stem, part
    return None


_SIZES: dict[tuple, float] = {}


_SIZE_CACHE_FORMAT = 2
_STORED_SIZES: dict[tuple[str, str, str], float] = {}
_SIZE_CACHE_LOCK = threading.Lock()

__all__ += ["cached_lora_size", "lora_size_variant", "store_lora_size"]


def lora_size_variant(path: str | os.PathLike[str], family: str | None) -> str:

    from .key_cache import file_sha256

    del family
    folder = os.path.dirname(os.path.realpath(os.fspath(path)))
    config_path = os.path.join(folder, "adapter_config.json")
    config_sha = file_sha256(config_path) if os.path.isfile(config_path) else None
    text = json.dumps([config_sha])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _size_cache_path(sha256: str, base_id: str, variant: str) -> str:
    from .key_cache import cache_dir

    name = f"{sha256}-{base_id}-{variant}-v{_SIZE_CACHE_FORMAT}.json"
    return os.path.join(cache_dir(), "size", name)


def cached_lora_size(sha256: str | None, base_id: str | None, variant: str) -> float | None:
    if not sha256 or not base_id:
        return None
    key = (sha256, base_id, variant)
    if key in _STORED_SIZES:
        return _STORED_SIZES[key]
    if os.environ.get("KILORA_SIZE_CACHE", "1") == "0":
        return None
    path = _size_cache_path(sha256, base_id, variant)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        size = float(data["size"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not math.isfinite(size) or size <= 0.0:
        return None
    _STORED_SIZES[key] = size
    return size


def store_lora_size(sha256: str | None, base_id: str | None, variant: str, size: float) -> None:
    if not sha256 or not base_id or not math.isfinite(size) or size <= 0.0:
        return
    _STORED_SIZES[(sha256, base_id, variant)] = size
    if os.environ.get("KILORA_SIZE_CACHE", "1") == "0":
        return
    import tempfile

    path = _size_cache_path(sha256, base_id, variant)
    with _SIZE_CACHE_LOCK:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"size": size, "sha256": sha256, "base": base_id}, f)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError, UnboundLocalError):
                os.unlink(tmp)


def lora_file_size(
    source: str | os.PathLike[str] | dict[str, torch.Tensor],
    model: nn.Module,
    *,
    family: str | None = None,
    full_model: nn.Module | None = None,
    base_weight: Callable[[str], torch.Tensor | None] | None = None,
    read_file: Callable | None = None,
    config: dict | None = None,
    base_norms: Mapping[str, float] | None = None,
    base_id: str | None = None,
) -> float | None:

    if (
        base_id is None
        and base_norms is None
        and base_weight is None
        and read_file is None
        and not isinstance(source, dict)
    ):
        weights = _model_base_weights(model)
        if weights:
            from .size_norm import base_key

            base_norms, base_id = base_key(weights)
    if base_id is not None and not isinstance(source, dict) and read_file is None:
        from .key_cache import file_sha256

        path = os.fspath(source)
        sha = file_sha256(path)
        variant = lora_size_variant(path, family)
        found = cached_lora_size(sha, base_id, variant)
        if found is not None:
            return found
        size = _lora_file_size(
            source,
            model,
            family=family,
            full_model=full_model,
            base_weight=base_weight,
            config=config,
            base_norms=base_norms,
        )
        if size is not None:
            store_lora_size(sha, base_id, variant, size)
        return size
    return _lora_file_size(
        source,
        model,
        family=family,
        full_model=full_model,
        base_weight=base_weight,
        read_file=read_file,
        config=config,
        base_norms=base_norms,
    )


def _model_base_weights(model: nn.Module) -> dict[str, torch.Tensor] | BaseWeights:
    entries = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if (
            name
            and isinstance(module, nn.Linear)
            and weight is not None
            and weight.ndim == 2
            and weight.device.type != "meta"
            and (weight.dtype.is_floating_point or is_quantized_weight(weight))
        ):
            entries[name] = (module, None)
    if any(is_quantized_weight(module.weight) for module, _ in entries.values()):
        return BaseWeights(entries)
    return {name: module.weight.detach() for name, (module, _) in entries.items()}


def _lora_file_size(
    source,
    model: nn.Module,
    *,
    family: str | None = None,
    full_model: nn.Module | None = None,
    base_weight: Callable[[str], torch.Tensor | None] | None = None,
    read_file: Callable | None = None,
    config: dict | None = None,
    base_norms: Mapping[str, float] | None = None,
) -> float | None:
    from .size_norm import layer_delta_sq

    modules = {n: m for n, m in model.named_modules() if n}
    shapes = tuple(
        (n, m.in_features, m.out_features) for n, m in modules.items() if isinstance(m, nn.Linear)
    )
    if isinstance(source, dict):
        state, file, key = dict(source), "state dict", None
    else:
        path = os.fspath(source)
        state, _, record = (read_file or _read_file)(path)
        file = record.get("original_name", os.path.basename(path))
        key = (record.get("digest"), family, shapes) if read_file is None else None
        if config is None:
            folder = os.path.dirname(os.path.realpath(path))
            config_path = os.path.join(folder, "adapter_config.json")
            config = _read_config(config_path) if os.path.isfile(config_path) else None
    if not _targets_text_encoder(family):
        state = {k: v for k, v in state.items() if not is_text_encoder_key(k)}
    state = normalize_for(family, state, model, full_model).state
    groups, _, layout = _group(state, file)
    linear_names = [n for n, m in modules.items() if isinstance(m, nn.Linear)]
    aliases: dict[str, Alias] = get_preset(family).aliases(linear_names) if family else {}
    resolve = _resolver(layout, list(modules), aliases, strict=False)

    def weight_of(target):
        if base_weight is not None:
            return base_weight(target)
        w = read_base_weight(modules.get(target), target)
        if not isinstance(w, torch.Tensor) or w.device.type == "meta":
            return None
        return w if w.dtype.is_floating_point else None

    delta_sq = base_sq = 0.0
    complete = True
    for stem, (tensors, _) in groups.items():
        found = resolve(stem)
        if not found:
            continue
        scale = 1.0
        if "down" in tensors and "up" in tensors:
            rank = tensors["down"].shape[0]
            scale = _scale(f"{file}: {stem}", tensors.get("alpha"), rank, config)
        for i, target in enumerate(found):
            rows = (i, len(found)) if len(found) > 1 else None
            if base_norms is not None and target in base_norms:
                sq = layer_delta_sq(tensors, scale, None, rows)
                if sq is None:
                    sq = layer_delta_sq(tensors, scale, weight_of(target), rows)
                if sq is None:
                    complete = False
                    continue
                delta_sq += sq
                base_sq += float(base_norms[target])
                continue
            w0 = weight_of(target)
            sq = layer_delta_sq(tensors, scale, w0, rows)
            if w0 is None or sq is None:
                complete = False
                continue
            delta_sq += sq
            base_sq += float(w0.detach().to(torch.float64).pow(2).sum())
    if complete and base_sq > 0.0:
        size = math.sqrt(delta_sq / base_sq)
        if key is not None:
            _SIZES[key] = size
        return size
    return _SIZES.get(key) if key is not None else None


def _group(state: dict[str, torch.Tensor], file: str) -> tuple[dict, list[str], str]:
    groups: dict[str, tuple[dict[str, torch.Tensor], dict[str, str]]] = {}
    skipped: list[str] = []
    layouts: set[str] = set()
    for key, tensor in state.items():
        split = None if key.startswith(_TEXT_ENCODER_PREFIXES) else _split_any(key)
        if split is None:
            skipped.append(key)
            continue
        stem, part = split
        if stem.startswith("lora_unet_"):
            layouts.add("kohya")
        elif stem.startswith("diffusion_model."):
            layouts.add("comfyui")
        else:
            layouts.add("peft")
        tensors, keys = groups.setdefault(stem, ({}, {}))
        tensors[part] = tensor
        keys[part] = key
    if len(layouts) > 1:
        raise LoRAImportError(f"{file}: the file mixes key layouts {sorted(layouts)}")
    return groups, skipped, layouts.pop() if layouts else "peft"


__all__ += ["stack_set"]


def stack_set(result: MultiImport, name: str | None = None) -> AdapterSet:

    routed = result.adapter_set
    gen = torch.Generator().manual_seed(0)
    experts: dict[str, KeyedExperts] = {}
    for target, ex in routed.experts.items():
        down = ex.down.reshape(-1, ex.down.shape[-1])
        up = ex.up.reshape(-1, ex.up.shape[-1])
        used = (down != 0).any(-1) | (up != 0).any(-1)
        if not bool(used.any()):
            used[0] = True
        down, up = down[used], up[used]
        experts[target] = KeyedExperts(
            down=down[None, None].clone(),
            up=up[None, None].clone(),
            keys_in=orthonormal_keys(torch.randn(1, 1, routed.router_dim, generator=gen)),
        )
    record = dict(routed.imported_from or {})
    record["composition"] = "stack"
    return AdapterSet(
        name=name or routed.name,
        part_type=routed.part_type,
        experts=experts,
        routers={w: r.clone() for w, r in routed.routers.items()},
        aggregation="accumulate",
        form="input",
        strength=1.0,
        imported_from=record,
        extra={**routed.extra, "composition": "stack"},
        key_source="random",
    )
