# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from ..comfyui.trigger_words import trigger_words_of
from . import contrast
from .basis import _license_status
from .index import LibraryIndex, _atomic, _default_path

DEFAULT_PROMPTS = (
    "a person standing on a city street",
    "a cat sitting beside a window",
    "a house surrounded by trees",
    "a vase of flowers on a wooden table",
)
DEFAULT_STRENGTH = 0.7
_CACHE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_pairs(
    source_path: str | Path,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
) -> list[contrast.PromptPair]:
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("bake prompts must be nonempty strings")
    prefix = ", ".join(trigger_words_of(str(source_path)))
    return [
        contrast.PromptPair(", ".join(p for p in (prefix, plain) if p), plain) for plain in prompts
    ]


def cache_key(
    source_sha: str,
    model_sha: str,
    settings: contrast.Settings,
    *,
    pairs,
    trigger_words: Sequence[str] = (),
) -> str:
    record = {
        "version": _CACHE_VERSION,
        "source_sha": source_sha,
        "model_sha": model_sha,
        "settings": asdict(settings),
        "strength": DEFAULT_STRENGTH,
        "pairs": [asdict(pair) for pair in pairs],
        "trigger_words": list(trigger_words),
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _request(source_path, model_sha, settings, prompts):
    source_path = Path(source_path).expanduser().resolve()
    settings.validate()
    if not model_sha:
        raise ValueError("base model sha must be nonempty")
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        reason, _ = _license_status(handle.metadata() or {})
    pairs = prompt_pairs(source_path, prompts)
    source_sha = _sha256(source_path)
    key = cache_key(
        source_sha,
        model_sha,
        settings,
        pairs=pairs,
        trigger_words=trigger_words_of(str(source_path)),
    )
    return source_path, source_sha, pairs, key, reason


def _cached(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return (handle.metadata() or {}).get("kilora_bake_key") == key
    except (OSError, ValueError, SafetensorError):
        return False


def baked_path(
    source_path: str | Path,
    model_sha: str,
    *,
    settings: contrast.Settings | None = None,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    index: LibraryIndex | None = None,
) -> Path | None:

    try:
        _, _, _, key, reason = _request(
            source_path,
            model_sha,
            settings or contrast.Settings(),
            prompts,
        )
    except (OSError, ValueError, SafetensorError):
        return None
    if reason is not None:
        return None
    directory = index.path if index is not None else _default_path()
    path = directory / "bakes" / f"{key}.safetensors"
    return path if _cached(path, key) else None


def find_baked(
    source_path: str | Path,
    *,
    index: LibraryIndex | None = None,
) -> tuple[Path | None, int]:

    directory = (index.path if index is not None else _default_path()) / "bakes"
    try:
        source_sha = _sha256(Path(source_path).expanduser().resolve())
        candidates = sorted(directory.glob("*.safetensors")) if directory.is_dir() else []
    except OSError:
        return None, 0
    models = set()
    for path in candidates:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
        except (OSError, ValueError, SafetensorError):
            continue
        if metadata.get("kilora_bake_source_sha") == source_sha:
            models.add(metadata.get("kilora_bake_model_sha") or "")
    found = sorted(
        {
            str(path)
            for sha in models
            if sha
            for path in [baked_path(source_path, sha, index=index)]
            if path is not None
        }
    )
    return (Path(found[0]) if len(found) == 1 else None), len(found)


def bake_file(
    source_path: str | Path,
    model_path: str | Path,
    *,
    settings: contrast.Settings | None = None,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    model_sha: str | None = None,
    index: LibraryIndex | None = None,
    device: str = "cuda",
    pipe=None,
    progress: Callable[[str], None] = print,
) -> Path | None:

    settings = settings or contrast.Settings()
    sha = model_sha if model_sha is not None else contrast.model_sha256(model_path)
    source_path, source_sha, pairs, key, reason = _request(source_path, sha, settings, prompts)
    if reason is not None:
        progress(f"Skipped {source_path}: derivatives are not allowed ({reason})")
        return None
    directory = index.path if index is not None else _default_path()
    path = directory / "bakes" / f"{key}.safetensors"
    if _cached(path, key):
        progress(f"Cached {source_path}: {path}")
        return path
    if pipe is None:
        from ..anima import load_anima

        if not Path(model_path).is_dir():
            raise ValueError("model must name a local Diffusers Anima folder")
        pipe = load_anima(str(model_path), torch.bfloat16, device)
    from ..diffusers import denoiser_of
    from ..presets import pack_license

    model = denoiser_of(pipe)
    source = contrast.EffectSource.load(model, source_path, DEFAULT_STRENGTH)
    estimate = contrast.memory_estimate(model, settings, len(pairs)) / 1024**3
    progress(
        f"{source_path}: Estimated GPU memory: {estimate:.2f} GiB (weights and solve workspace)"
    )
    samples = contrast.trajectory_samples(pipe, pairs, settings, device, target_source=source)
    license, _ = pack_license("anima")
    result = contrast.solve(
        model,
        samples,
        pairs,
        settings=settings,
        model_sha=sha,
        license=license,
        target_source=source,
        progress=progress,
    )
    _, _, _, current_key, current_reason = _request(source_path, sha, settings, prompts)
    if current_key != key or current_reason is not None:
        raise ValueError("source file or trigger words changed during the bake; no file written")
    triggers = trigger_words_of(str(source_path))
    metadata = {
        "kilora_directions": json.dumps(result.metadata),
        "kilora_bake_key": key,
        "kilora_bake_source_sha": source_sha,
        "kilora_bake_model_sha": sha,
        "modelspec.license": source.license,
        "ss_tag_frequency": json.dumps({str(i): {word: 1} for i, word in enumerate(triggers)}),
    }
    _atomic(path, lambda name: save_file(contrast._weights(result), name, metadata=metadata))
    progress(f"Wrote {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kilora library")
    commands = parser.add_subparsers(dest="command", required=True)
    bake = commands.add_parser("bake", help="prepare missing trigger-baked library files")
    selection = bake.add_mutually_exclusive_group(required=True)
    selection.add_argument("--files", nargs="+", metavar="FILE")
    selection.add_argument("--all", action="store_true", help="bake every indexed local file")
    bake.add_argument("--model", required=True, help="local Diffusers Anima folder")
    bake.add_argument("--index", default=None, help="library index folder")
    bake.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        index = LibraryIndex.open(args.index)
        files = args.files if args.files is not None else sorted(index.paths)
        if not files:
            print("No library files to bake.")
            return 0
        sha = contrast.model_sha256(args.model)
        failed = False
        seen = set()
        for filename in files:
            try:
                identity = str(Path(filename).expanduser().resolve())
                if identity in seen:
                    continue
                seen.add(identity)
                bake_file(filename, args.model, model_sha=sha, index=index, device=args.device)
            except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
                print(f"error: {filename}: {exc}", file=sys.stderr)
                failed = True
        return int(failed)
    except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
