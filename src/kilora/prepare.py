# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import os
import re
from collections.abc import Callable

import torch

from .key_cache import file_sha256
from .lora_import import _read_file
from .multi_import import (
    _group,
    _kron_parts_svd,
    _load_lokr_cache,
    _lokr_parts_key,
    _lycoris_shape,
    _save_lokr_cache,
    import_loras,
)


def prepare_lokr(
    path: str | os.PathLike[str],
    *,
    checkpoint: Callable[[], None] | None = None,
    detailed: bool = False,
    device="cpu",
):

    if checkpoint is not None:
        checkpoint()
    sha256 = file_sha256(path)
    if checkpoint is not None:
        checkpoint()
    cached = _load_lokr_cache(sha256, device)
    if checkpoint is not None:
        checkpoint()
    marker = cached.get("@complete", {}).get("count")
    required = {"u1", "s1", "v1", "u2", "s2", "v2"}
    if marker is not None and marker.numel() == 1 and marker.dtype == torch.int64:
        count = int(marker.item())
        entries = sum(
            name.startswith("@") and name != "@complete" and set(parts) == required
            for name, parts in cached.items()
        )
        if 0 <= count <= entries:
            return (0, count, True) if detailed else (0, count)
    if checkpoint is not None:
        checkpoint()
    state, _, record = _read_file(path, checkpoint=checkpoint) if checkpoint else _read_file(path)
    sha256 = record["digest"].removeprefix("sha256-")
    groups, _, _ = _group(state, os.fspath(path))
    if checkpoint is not None:
        checkpoint()
    cached = _load_lokr_cache(sha256, device)
    computed = {}
    current = 0
    available = set()
    for tensors, _ in groups.values():
        if checkpoint is not None:
            checkpoint()
        if not any(name.startswith("lokr_") for name in tensors):
            continue
        key = _lokr_parts_key(tensors)
        if key in cached and set(cached[key]) == required:
            current += 1
            available.add(key)
            continue
        parts = _kron_parts_svd(tensors, device)
        if parts is not None:
            computed[key] = parts
            available.add(key)
    if checkpoint is not None:
        checkpoint()
    if os.environ.get("KILORA_LOKR_CACHE", "1") != "0":
        marker = {"@complete": {"count": torch.tensor(len(available))}}
        _save_lokr_cache(sha256, computed | marker, device)
        if checkpoint is not None:
            checkpoint()
        saved = _load_lokr_cache(sha256, device)
        if "@complete" not in saved or any(
            key not in saved or set(saved[key]) != required for key in computed
        ):
            raise OSError("LoKr cache could not be written")
    elif computed:
        raise OSError("LoKr cache is disabled")
    result = (len(computed), current)
    return (*result, False) if detailed else result


PLAN_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def plan_dtype(model):

    if model is None:
        return None
    dtype = getattr(model, "manual_cast_dtype", None)
    if dtype is None and hasattr(model, "get_dtype"):
        try:
            dtype = model.get_dtype()
        except Exception:
            dtype = None
    if not isinstance(dtype, torch.dtype) and hasattr(model, "parameters"):
        try:
            dtype = next(model.parameters()).dtype
        except (StopIteration, TypeError, AttributeError):
            dtype = None
    return dtype if dtype in PLAN_DTYPES else None


def padding_dtypes(dtype):
    if isinstance(dtype, torch.dtype) or dtype is None:
        return (dtype,) if dtype in PLAN_DTYPES else PLAN_DTYPES
    try:
        wanted = set(dtype)
    except TypeError:
        return PLAN_DTYPES
    found = tuple(d for d in PLAN_DTYPES if d in wanted)
    return found or PLAN_DTYPES


def layer_ranks(result):

    adapter_set = getattr(result, "adapter_set", None)
    if adapter_set is None:
        return {}
    ranks = {}
    for target, ex in adapter_set.experts.items():
        if not getattr(ex, "has_lora", False) or ex.down is None:
            continue
        found = [int(r) for r in getattr(ex, "ranks", ()) if r]
        if len(found) == 1:
            ranks[(target, int(ex.down.shape[-1]), int(ex.out_features))] = found[0]
    return ranks


def library_paddings(ranks_by_file):

    by_layer = {}
    for ranks in ranks_by_file.values():
        for layer, rank in ranks.items():
            by_layer.setdefault(layer, set()).add(rank)
    result = {}
    for path, ranks in ranks_by_file.items():
        wanted = {}
        for layer, rank in ranks.items():
            larger = sorted(r for r in by_layer[layer] if r > rank)
            if larger:
                wanted[layer] = larger
        result[path] = wanted
    return result


def _store_paddings(plans, paddings, report, checkpoint, device, dtypes=PLAN_DTYPES):
    from . import compute, derived_cache

    individual = torch.device(device).type == "cpu" or (
        torch.device(device).type == "cuda"
        and os.environ.get("KILORA_CONTENT_KEYS", "per_expert") == "per_expert"
    )
    if not individual:
        return
    m = compute.derived_rows()
    wanted = {target: p for (target, _, _), p in paddings.items()}
    for target, plan in plans.items():
        extra = wanted.get(target)
        if not extra:
            continue
        for dtype in dtypes:
            if checkpoint is not None:
                checkpoint()
            moved = plan.to(torch.device(device), dtype)
            for layer in moved.sets:
                weights, down, _ = compute.derived_weights(layer)
                source = derived_cache.layer_source(weights)
                ids = source[1] if isinstance(source, tuple) and len(source) == 2 else None
                if not ids or len(ids) != 1 or down.shape[0] != weights.per_expert:
                    continue
                for padding in extra:
                    if padding <= weights.per_expert:
                        continue
                    with torch.inference_mode(False), torch.no_grad():
                        settings = compute._expert_key_settings(down, weights.up, padding, m)
                        names = compute.EXPERT_KEY_NAMES
                        if derived_cache.load(ids[0], "expert-keys", settings, names) is not None:
                            report["current"] += 1
                            continue
                        entry, settings = compute.padded_expert_bases(layer, padding, m)
                        derived_cache.store(ids[0], "expert-keys", settings, names, entry)
                    report["entries"] += 1
                    report["bytes"] += sum(t.numel() * t.element_size() for t in entry)
                    report["paddings"].add(padding)
            del moved


def prepare_paddings(
    path,
    paddings,
    *,
    model=None,
    family=None,
    checkpoint=None,
    device="cpu",
    dtype=None,
    full_model=None,
):

    if dtype is None:
        dtype = plan_dtype(full_model)
    dtypes = padding_dtypes(dtype)
    report = {"paddings": set(), "entries": 0, "bytes": 0, "current": 0}
    if paddings:
        if model is None:
            prepare_file(
                path,
                family=family,
                checkpoint=checkpoint,
                device=device,
                _paddings=(paddings, report, dtypes),
            )
        else:
            prepare_mix(
                [path],
                model,
                family=family,
                checkpoint=checkpoint,
                device=device,
                _paddings=(paddings, report, dtypes),
            )
    report["paddings"] = sorted(report["paddings"])
    return report


def prepare_mix(
    sources,
    model,
    *,
    family=None,
    full_model=None,
    rank=None,
    checkpoint=None,
    device="cpu",
    _paddings=None,
):

    from . import compute
    from .bake import ScopedSet, plan_targets

    if checkpoint is not None:
        checkpoint()
    result = import_loras(
        sources,
        model,
        family=family,
        full_model=full_model,
        rank=rank,
        device=device,
        checkpoint=checkpoint,
        prepare_expert_content=True,
    )
    if checkpoint is not None:
        checkpoint()
    if result.adapter_set is None:
        return result
    plans = plan_targets([ScopedSet(result.adapter_set, import_score="derived")], 1)
    if _paddings is not None:
        dtypes = _paddings[2] if len(_paddings) > 2 else PLAN_DTYPES
        _store_paddings(plans, _paddings[0], _paddings[1], checkpoint, device, dtypes)
        return result
    for plan in plans.values():
        for dtype in PLAN_DTYPES:
            if checkpoint is not None:
                checkpoint()
            moved = plan.to(torch.device(device), dtype)
            for layer in moved.sets:
                compute.derived_keys(layer)
                if torch.device(device).type == "cuda":
                    compute._derived_keys_current(layer, prepare_experts=True)
                if dtype == torch.float32:
                    compute.share_transition(layer)
                    compute.share_transition(layer, _kind="cosine", _raw=True)
            del layer, moved
    return result


def _pattern_matches(preset, stem, body, layout):
    from .bindings import BINDINGS

    matches = []
    names = []
    numbers = set(re.findall(r"\d+", body)) or {"0"}
    for pattern in preset.full_target_modules:
        for indices in itertools.product(numbers, repeat=pattern.count("*")):
            pieces = iter(indices)
            name = re.sub(r"\*", lambda _, parts=pieces: next(parts), pattern)
            names.append(name)
            options = {name}
            if preset.rename:
                renamed = preset.rename(name)
                if renamed:
                    options.add(renamed)
            binding = BINDINGS.get(preset.family)
            if binding is not None:
                try:
                    place = binding.to_comfy(name)
                except KeyError:
                    pass
                else:
                    options.add(place.path.removeprefix("diffusion_model."))
            if any(
                ("lora_unet_" + x.replace(".", "_") == stem if layout == "kohya" else x == body)
                for x in options
            ):
                matches.append(name)
    for alias, targets in preset.aliases(names).items():
        if ("lora_unet_" + alias.replace(".", "_") if layout == "kohya" else alias) == (
            stem if layout == "kohya" else body
        ):
            matches.extend(targets.targets)
    return matches


def _group_families(groups, layout):
    from .families import LORA_SIBLINGS, detect_names
    from .presets import PRESETS

    names = [stem.removeprefix("diffusion_model.") for stem in groups]
    detected = detect_names(names)
    if detected is not None and detected.family:
        siblings = LORA_SIBLINGS.get(detected.family, ())
        return tuple(sorted({detected.family, *siblings})) if siblings else (detected.family,)
    return tuple(
        sorted(
            name
            for name, preset in PRESETS.items()
            if any(
                _pattern_matches(preset, stem, stem.removeprefix("diffusion_model."), layout)
                for stem in groups
            )
        )
    )


def lora_families(keys, file=""):
    groups, _, layout = _group(dict.fromkeys(keys), os.fspath(file))
    return _group_families(groups, layout)


def prepare_file(path, *, family=None, checkpoint=None, device="cpu", _paddings=None):
    from .bindings import BINDINGS
    from .families import FAMILIES, shape_tree
    from .lora_import import _MODEL_PREFIXES
    from .presets import PRESETS

    if _paddings is None:
        prepare_lokr(path, checkpoint=checkpoint, device=device)
    if checkpoint is not None:
        checkpoint()
    state, _, _ = _read_file(path, checkpoint=checkpoint) if checkpoint else _read_file(path)
    groups, _, layout = _group(state, os.fspath(path))
    if family and family not in PRESETS:
        raise ValueError(f"unknown family: {family}")

    def pattern_matches(preset, stem, body):
        return _pattern_matches(preset, stem, body, layout)

    if not family:
        matched = _group_families(groups, layout)
        if len(matched) == 1:
            family = matched[0]
        elif len(matched) > 1:
            result = None
            for candidate in matched:
                try:
                    result = prepare_file(
                        path,
                        family=candidate,
                        checkpoint=checkpoint,
                        device=device,
                        _paddings=_paddings,
                    )
                except ValueError as exc:
                    if "no unambiguous layer names" not in str(exc):
                        raise
            if result is not None:
                return result
    shapes = {}
    for stem, (tensors, _) in groups.items():
        if checkpoint:
            checkpoint()
        if "down" in tensors and "up" in tensors:
            a, b = tensors["down"], tensors["up"]
            if a.ndim != 2 or b.ndim != 2:
                continue
            fan_in, fan_out = a.shape[1], b.shape[0]
        else:
            result = _lycoris_shape(tensors)
            if result is None:
                continue
            (fan_out, fan_in), _ = result
        body = stem.removeprefix("diffusion_model.")
        for prefix in _MODEL_PREFIXES:
            body = body.removeprefix(prefix)
        candidates = []
        if family:
            preset = PRESETS[family]
            candidates.extend(pattern_matches(preset, stem, body))
            binding = BINDINGS.get(family)
            if binding and layout != "kohya":
                candidates.extend(name for name, part in binding.from_comfy(body) if part is None)
        if not candidates and layout != "kohya":
            candidates = [body]
        candidates = list(dict.fromkeys(candidates))
        if candidates and fan_out % len(candidates) == 0:
            if family and FAMILIES[family].naming == "comfy":
                candidates = [body]
            for target in candidates:
                shapes[target] = (int(fan_in), int(fan_out) // len(candidates))
    if not shapes:
        raise ValueError("no unambiguous layer names found; provide a model or family")
    return prepare_mix(
        [path],
        shape_tree(shapes),
        family=family,
        checkpoint=checkpoint,
        device=device,
        _paddings=_paddings,
    )


_BASE_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_BASE_FILE_NORMS: dict[tuple, tuple[str | None, dict, dict, dict]] = {}


def base_file_norms(base_path, *, family=None, checkpoint=None):

    from safetensors import safe_open

    from .bindings import BINDINGS
    from .families import detect_names
    from .size_norm import weight_sq

    path = os.path.realpath(os.fspath(base_path))
    stat = os.stat(path)
    memo = (path, stat.st_size, stat.st_mtime_ns, family)
    if memo in _BASE_FILE_NORMS:
        return _BASE_FILE_NORMS[memo]
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        bodies = {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            body = key
            for prefix in _BASE_PREFIXES:
                if body.startswith(prefix):
                    body = body[len(prefix) :]
                    break
            bodies[body[: -len(".weight")]] = key
        if family is None:
            spec = detect_names(["diffusion_model." + b for b in bodies]) or detect_names(
                list(bodies)
            )
            family = spec.family if spec is not None else None
        binding = BINDINGS.get(family) if family else None
        norms: dict[str, float] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        places: dict[str, tuple[str, tuple[int, int] | None]] = {}
        if binding is not None:
            for body, key in bodies.items():
                hits = binding.from_comfy(body)
                if not hits:
                    continue
                if checkpoint is not None:
                    checkpoint()
                w = f.get_tensor(key)
                if w.dim() != 2 or not w.dtype.is_floating_point:
                    continue
                for target, part in hits:
                    rows = w
                    if part:
                        index, count = part
                        n = w.shape[0] // count
                        rows = w[index * n : (index + 1) * n]
                    norms[target] = weight_sq(rows)
                    shapes[target] = tuple(rows.shape)
                    places[target] = (key, part)
                del w
    found = (family, norms, shapes, places)
    _BASE_FILE_NORMS[memo] = found
    return found


def prepare_sizes(lora_paths, base_path, *, family=None, checkpoint=None):

    from safetensors import safe_open

    from .families import shape_tree
    from .multi_import import lora_file_size
    from .presets import PRESETS
    from .size_norm import base_identity

    family, norms, shapes, places = base_file_norms(base_path, family=family, checkpoint=checkpoint)
    out = {}
    if not norms:
        return {os.fspath(p): None for p in lora_paths}
    base_id = base_identity(norms, shapes)
    tree = shape_tree({t: (s[1], s[0]) for t, s in shapes.items()})
    with safe_open(os.path.realpath(os.fspath(base_path)), framework="pt", device="cpu") as f:

        def base_weight(target):
            place = places.get(target)
            if place is None:
                return None
            key, part = place
            w = f.get_tensor(key)
            if part:
                index, count = part
                n = w.shape[0] // count
                w = w[index * n : (index + 1) * n]
            return w

        for path in lora_paths:
            if checkpoint is not None:
                checkpoint()
            try:
                out[os.fspath(path)] = lora_file_size(
                    path,
                    tree,
                    family=family if family in PRESETS else None,
                    base_weight=base_weight,
                    base_norms=norms,
                    base_id=base_id,
                )
            except Exception as exc:
                if checkpoint is not None and type(exc).__name__ == "_Stopped":
                    raise
                out[os.fspath(path)] = None
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LoRA files for later use")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--family", default=None)
    args = parser.parse_args()
    for path in args.files:
        prepare_file(path, family=args.family)
        print(f"Prepared: {path}")


if __name__ == "__main__":
    main()
