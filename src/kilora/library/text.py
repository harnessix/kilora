# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import weakref
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class TextFeatures:
    tags: tuple[str, ...]
    trigger_words: tuple[str, ...]
    filename_words: tuple[str, ...] = ()

    def text(self) -> str:
        return ", ".join((*self.trigger_words, *self.tags, *self.filename_words))


def _words(value: Any) -> list[str]:
    if isinstance(value, str):
        return [word.strip() for word in re.split(r"[,\n]", value) if word.strip()]
    if isinstance(value, list):
        return [word.strip() for word in value if isinstance(word, str) and word.strip()]
    return []


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def _counts(value: Any) -> Counter[str]:
    result: Counter[str] = Counter()
    value = _json(value)
    if isinstance(value, dict):
        for name, count in value.items():
            if isinstance(count, dict):
                result.update(_counts(count))
            elif (
                isinstance(count, (int, float))
                and not isinstance(count, bool)
                and math.isfinite(count)
                and count > 0
            ):
                result[str(name).strip()] += count
    return result


def _unique(values: list[str]) -> tuple[str, ...]:
    found: dict[str, str] = {}
    for value in values:
        value = value.strip()
        if value:
            found.setdefault(value.casefold(), value)
    return tuple(found.values())


def read_text_features(path: str | os.PathLike[str]) -> TextFeatures:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    triggers = _words(metadata.get("modelspec.trigger_phrase"))
    counts = _counts(metadata.get("ss_tag_frequency"))
    tags = sorted(counts, key=lambda tag: (-counts[tag], tag.casefold(), tag))
    tags.extend(_words(metadata.get("ss_output_name")))
    for suffix in (".txt", ".json", ".civitai.info"):
        sidecar = path.with_suffix(suffix)
        try:
            data = sidecar.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if suffix == ".txt":
            lines = data.splitlines()
            if lines and lines[0].lower().startswith("trigger:"):
                triggers.extend(_words(lines[0][len("trigger:") :]))
                lines = lines[1:]
            tags.extend(_words("\n".join(lines)))
            continue
        payload = _json(data)
        if not isinstance(payload, dict):
            continue
        triggers.extend(_words(payload.get("trainedWords")))
        triggers.extend(_words(payload.get("trigger_words")))
        side_tags = payload.get("tags", [])
        if isinstance(side_tags, dict):
            side_counts = _counts(side_tags)
            tags.extend(sorted(side_counts, key=lambda tag: (-side_counts[tag], tag.casefold())))
        else:
            tags.extend(_words(side_tags))
    words = re.findall(r"[^\W_]+", path.stem, flags=re.UNICODE)
    return TextFeatures(
        tags=_unique(tags)[:20], trigger_words=_unique(triggers), filename_words=_unique(words)
    )


_encoder_cache = weakref.WeakKeyDictionary()


def _output(clip, text):
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True)
    if isinstance(output, dict):
        return (
            tokens,
            output.get("cond", output.get("last_hidden_state")),
            output.get("pooled_output"),
            output.get("attention_mask"),
        )
    if isinstance(output, (tuple, list)):
        return tokens, output[0], output[1] if len(output) > 1 else None, None
    return tokens, output, None, None


def encoder_identity(clip: Any | None) -> str:
    if clip is None:
        return "trigram-512-v1"
    cached = _encoder_cache.get(clip)
    if cached is not None:
        return cached
    with torch.no_grad():
        _, _, pooled, _ = _output(clip, "")
    cls = type(clip)
    model = getattr(clip, "cond_stage_model", clip)
    parameters = getattr(model, "parameters", None)
    count = sum(parameter.numel() for parameter in parameters()) if callable(parameters) else 0
    name = f"{cls.__module__}.{cls.__qualname__}:{count}"
    identity = hashlib.sha256(name.encode("utf-8")).hexdigest()
    if pooled is None:
        identity += "-mean"
    _encoder_cache[clip] = identity
    return identity


def _token_ids(tokens):
    if isinstance(tokens, dict):
        streams = [value for key, value in tokens.items() if key != "attention_mask"]
        if len(streams) != 1:
            raise ValueError("token mean needs one tokenizer stream")
        tokens = streams[0]
    if isinstance(tokens, torch.Tensor):
        return tokens.reshape(-1).tolist()
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("unrecognized tokenizer output")
    ids = []
    for chunk in tokens:
        if isinstance(chunk, (list, tuple)):
            for token in chunk:
                ids.append(token[0] if isinstance(token, (list, tuple)) else token)
        else:
            ids.append(chunk)
    return ids


def _prompt_mask(clip, tokens, length, device):
    ids = _token_ids(tokens)
    empty = _token_ids(clip.tokenize(""))
    if len(ids) != length or not empty:
        raise ValueError("tokenizer output does not match encoder sequence")
    start = 0
    while start < min(len(ids), len(empty)) and ids[start] == empty[start]:
        start += 1
    end = len(ids)
    end_tokens = set(empty[start:])
    while end > start and ids[end - 1] in end_tokens:
        end -= 1
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    mask[start:end] = True
    return mask


def embed_text(text: str, clip: Any | None = None) -> torch.Tensor:
    if clip is not None:
        with torch.no_grad():
            tokens, hidden, pooled, mask = _output(clip, text)
        if pooled is None:
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("encoder must return one sequence of token vectors")
            if mask is None and isinstance(tokens, dict):
                mask = tokens.get("attention_mask")
            if mask is None:
                mask = _prompt_mask(clip, tokens, hidden.shape[1], hidden.device)
            mask = torch.as_tensor(mask, device=hidden.device).reshape(-1).bool()
            if mask.numel() != hidden.shape[1] or not mask.any():
                raise ValueError("invalid attention mask")
            pooled = hidden[0, mask].mean(dim=0)
        if not isinstance(pooled, torch.Tensor) or pooled.ndim not in (1, 2):
            raise ValueError("encoder must return a pooled vector")
        if pooled.ndim == 2:
            if pooled.shape[0] != 1:
                raise ValueError("encoder must return one pooled vector")
            pooled = pooled[0]
        return pooled.detach().to(device="cpu", dtype=torch.float32).contiguous()
    vector = torch.zeros(512, dtype=torch.float32)
    normalized = " ".join(text.casefold().split())
    for index in range(max(0, len(normalized) - 2)):
        gram = normalized[index : index + 3].encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(gram).digest()[:8], "big") % 512
        vector[bucket] += 1
    norm = torch.linalg.vector_norm(vector)
    return vector / norm if norm > 0 else vector


class TextFeatureStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.vectors = load_file(str(self.path), device="cpu") if self.path.exists() else {}

    def get(self, digest: str, clip: Any | None = None) -> torch.Tensor | None:
        return self.vectors.get(f"{digest}/text/{encoder_identity(clip)}")

    def put(self, digest: str, vector: torch.Tensor, clip: Any | None = None) -> None:
        self.vectors[f"{digest}/text/{encoder_identity(clip)}"] = (
            vector.detach().to(device="cpu", dtype=torch.float16).contiguous()
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".text-features-", dir=self.path.parent)
        os.close(fd)
        try:
            save_file(self.vectors, name)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
