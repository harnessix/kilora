# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..families import FAMILIES
from ..family_spec import _split
from ..presets import PRESETS
from ..references import HashIndex
from .codebook import FamilyCodebooks
from .content import extract_content
from .text import TextFeatures, embed_text, encoder_identity, read_text_features


@dataclass
class SyncReport:
    added: int = 0
    removed: int = 0
    skipped: int = 0
    reasons: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True)
class SimilarResult:
    path: str
    sha256: str
    score: float


def _default_path() -> Path:
    if os.environ.get("KILORA_LIBRARY_DIR"):
        return Path(os.environ["KILORA_LIBRARY_DIR"]).expanduser()
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "kilora-library"
    except (ImportError, AttributeError):
        return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kilora-library"


def _family_patterns() -> dict[str, re.Pattern]:
    compiled = {}
    marker = "987654321"
    for name, spec in FAMILIES.items():
        if spec.root == "text_encoder":
            continue
        preset = PRESETS.get(name)
        patterns = spec.include or (preset.full_target_modules if preset else ())
        concrete = [pattern.replace("*", marker) for pattern in patterns]
        shapes = {path: (4, 4) for path in concrete}
        aliases = {
            alias for alias, (target, _) in spec.lora_names(shapes).items() if target in shapes
        }
        if preset:
            aliases.update(preset.aliases(concrete))
            if name in ("sd15", "sdxl"):
                for block in ("down_blocks.0", "up_blocks.0", "mid_block"):
                    paths = [
                        p.replace("*.", f"{block}.attentions.0.", 1).replace("*", marker)
                        for p in patterns
                    ]
                    for alias in preset.aliases(paths):
                        aliases.add(
                            re.sub(r"^(input_blocks|output_blocks)\.\d+\.", r"\1.*.", alias)
                        )
        all_patterns = set(patterns)
        all_patterns.update(alias.replace(marker, "*") for alias in aliases)
        all_patterns.update(pattern.replace(".", "_") for pattern in tuple(all_patterns))
        compiled[name] = re.compile("|".join(fnmatch.translate(p) for p in sorted(all_patterns)))
    return compiled


_CONTENT_VERSION = 2

_FAMILY_PATTERNS = _family_patterns()
_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "base_model.model.",
    "unet.",
    "lora_unet_",
    "lycoris_",
    "lora_transformer_",
)


def _family(keys: list[str]) -> str:
    stems: dict[str, int] = {}
    for key in keys:
        part = _split(key)
        if not part:
            continue
        stem = part[0]
        for prefix in _PREFIXES:
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
        stems[stem] = stems.get(stem, 0) + 1
    scores = {name: [0, 0] for name in _FAMILY_PATTERNS}
    for stem, weight in stems.items():
        for name, pattern in _FAMILY_PATTERNS.items():
            if pattern.fullmatch(stem):
                scores[name][0] += 1
                scores[name][1] += weight
    best = max((tuple(score) for score in scores.values()), default=(0, 0))
    winners = [name for name, score in scores.items() if tuple(score) == best and best[0]]
    return winners[0] if len(winners) == 1 else "unknown"


def _metadata(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as file:
        keys = list(file.keys())
        ranks = []
        for key in keys:
            part = _split(key)
            if part and (
                part[1] == "down" or key.endswith((".lokr_w1_b", ".lokr_w2_b", ".hada_w1_b"))
            ):
                shape = file.get_slice(key).get_shape()
                if shape:
                    ranks.append(shape[0])
        kind = (
            "lokr"
            if any(".lokr_" in k for k in keys)
            else ("loha" if any(".hada_" in k for k in keys) else "lora")
        )
    family = _family(keys)
    triggers = read_text_features(path).trigger_words
    words = set(re.findall(r"[a-z0-9]+", " ".join((path.stem, *triggers)).lower()))
    preset = PRESETS.get(family)
    functional = set(preset.functional_words) if preset else set()
    return {
        "filename": path.name,
        "folder": str(path.parent),
        "size": path.stat().st_size,
        "family": family,
        "kind": kind,
        "rank": max(ranks, default=0),
        "axis": "functional" if words & functional else "creative",
        "tags": [],
        "trigger_words": [],
        "cluster_id": None,
    }


def _content_names(family: str):
    preset = PRESETS[family]
    targets = [
        pattern.replace("*", str(i))
        for pattern in preset.full_target_modules
        for i in range(max(64, preset.content_blocks))
    ]
    aliases = {alias: value.targets[0] for alias, value in preset.aliases(targets).items()}
    aliases.update(
        {
            alias: target
            for alias, (target, _) in FAMILIES[family]
            .lora_names(dict.fromkeys(targets, (4, 4)))
            .items()
        }
    )
    aliases.update({alias.replace(".", "_"): target for alias, target in list(aliases.items())})
    patterns = [
        (pattern, re.compile(re.escape(pattern).replace(r"\*", "(.*?)").replace(r"\.", "[._]")))
        for pattern in preset.full_target_modules
    ]
    return aliases, patterns, {}


def _content(path: Path, family: str, names=None):
    preset = PRESETS.get(family)
    if preset is None:
        return None
    aliases, patterns, resolved = names if names is not None else _content_names(family)
    state = load_file(str(path), device="cpu")
    layers = {}
    for key, tensor in state.items():
        part = _split(key)
        if part is None:
            continue
        stem, name, suffix = part
        if name == "other":
            name = suffix[1:]
        if name not in {
            "down",
            "up",
            "alpha",
            "lokr_w1",
            "lokr_w2",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2_a",
            "lokr_w2_b",
            "hada_w1_a",
            "hada_w1_b",
            "hada_w2_a",
            "hada_w2_b",
        }:
            continue
        layers.setdefault(stem, {})[name] = tensor
    groups, widths = {}, {}
    for stem, parts in layers.items():
        name = stem
        for prefix in (
            "model.diffusion_model.",
            "diffusion_model.",
            "lora_unet_",
            "lycoris_",
            "lora_transformer_",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name not in resolved:
            canonical = None
            for pattern, pattern_re in patterns:
                match = pattern_re.fullmatch(name)
                if match:
                    canonical = pattern
                    for value in match.groups():
                        canonical = canonical.replace("*", value.replace("_", "."), 1)
                    break
            resolved[name] = canonical if canonical is not None else aliases.get(name)
        canonical = resolved[name]
        if canonical is None:
            continue
        if "down" in parts:
            width = parts["down"].shape[-1]
        elif "lokr_w1" in parts or "lokr_w1_b" in parts:
            from ..multi_import import lokr_parts

            a, b = lokr_parts(parts)
            width = a.shape[1] * b.shape[1]
        elif "hada_w1_b" in parts:
            width = parts["hada_w1_b"].shape[-1]
        else:
            continue
        group = preset.content_group(canonical, width)
        if group:
            groups[stem] = group
            widths[group] = width
    if not groups:
        return None
    return extract_content(layers, groups, dict(sorted(widths.items())))


def _text_signature(path: Path):
    signature = [str(path)]
    for suffix in (".txt", ".json", ".civitai.info"):
        try:
            stat = path.with_suffix(suffix).stat()
        except FileNotFoundError:
            signature.append(None)
        else:
            signature.append([stat.st_size, stat.st_mtime_ns])
    return signature


def _atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    os.close(fd)
    try:
        writer(name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class LibraryIndex:
    def __init__(self, path: Path):
        self.path = path
        self.hash_index = HashIndex(path / "hashes.json")
        self.records = {}
        self.paths = {}
        self.codebooks = FamilyCodebooks()
        self._features = None
        self._feature_keys = None
        self._retrieval_cache = {}
        file = path / "library.json"
        if file.exists():
            data = json.loads(file.read_text(encoding="utf-8"))
            if data.get("format") != "kilora-library" or data.get("version") != "1.0":
                raise ValueError("Unsupported library index format")
            self.records = data.get("records", {})
            self.paths = data.get("paths", {})
            features_file = path / "features.safetensors"
            if data.get("codebooks") and features_file.exists():
                self._features = load_file(str(features_file))
                self.codebooks = FamilyCodebooks.load(data["codebooks"], self._features)

    @classmethod
    def open(cls, path: str | os.PathLike | None = None) -> LibraryIndex:
        return cls(Path(path).expanduser() if path is not None else _default_path())

    def retrieval_inputs(self, family, encoder=None):
        identity = encoder_identity(encoder)
        cache_key = (family, identity)
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        if self._features is None:
            path = self.path / "features.safetensors"
            self._features = load_file(str(path)) if path.exists() else {}
        if self._feature_keys is None:
            self._feature_keys = {}
            for key, value in self._features.items():
                digest, separator, rest = key.partition("/")
                if separator:
                    self._feature_keys.setdefault(digest, {})[rest] = value
        records = {key: value for key, value in self.records.items() if value["family"] == family}
        summaries, texts, directions, widths, fallback = {}, {}, {}, {}, set()
        fallback_identity = encoder_identity(None)
        for digest in records:
            values = self._feature_keys.get(digest, {})
            if "summary" in values:
                summaries[digest] = values["summary"]
            vector = values.get(f"text/{identity}")
            if vector is None and identity != fallback_identity:
                vector = values.get(f"text/{fallback_identity}")
                if vector is not None:
                    fallback.add(digest)
            if vector is not None:
                texts[digest] = vector
            for key, value in values.items():
                if key.endswith("/directions"):
                    group = key[: -len("/directions")]
                    widths[group] = value.shape[1]
                    directions.setdefault(digest, {})[group] = value
        result = {
            "records": records,
            "summaries": summaries,
            "text_features": texts,
            "codebook": self.codebooks.for_family(family),
            "groups": dict(sorted(widths.items())),
            "directions": directions,
            "fallback": fallback,
        }
        self._retrieval_cache[cache_key] = result
        return result

    def embedding_inputs(self, family, model_path):

        from .basis import load_basis
        from .coeffs import lora_coefficients
        from .embed import LayerChain

        model_path = Path(model_path).expanduser()
        if model_path.is_file():
            model_path = model_path.parent
        model = LayerChain.load(model_path)
        basis = load_basis(self)
        if not set(model.mounts).issubset(basis.mounts):
            raise ValueError("model mounts do not match the library basis")
        for name in model.mounts:
            mount = basis.mounts[name]
            if mount.u.shape[1] != model.k or mount.v.shape[1] != model.k:
                raise ValueError("model coefficient dimensions do not match the library basis")

        def digest_files(paths):
            digest = hashlib.sha256()
            for path in paths:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            return digest.hexdigest()

        basis_id = digest_files([self.path / "basis.json", self.path / "basis.safetensors"])
        model_id = digest_files([model_path / "config.json", model_path / "model.safetensors"])
        prefix = f"embedding/{family}/{basis_id}/{model_id}"
        coeff_prefix = f"coefficients/{family}/{basis_id}"
        features_path = self.path / "features.safetensors"
        features = (
            dict(self._features)
            if self._features is not None
            else (load_file(str(features_path)) if features_path.exists() else {})
        )
        changed = False
        vectors = {}
        for digest, record in sorted(self.records.items()):
            if record["family"] != family:
                continue
            filenames = [
                p for p, value in self.paths.items() if value == digest and Path(p).is_file()
            ]
            if not filenames:
                continue
            key = f"{digest}/{prefix}"
            if key not in features:
                row_prefix = f"{digest}/{coeff_prefix}/"
                mounts = {}
                for i, name in enumerate(sorted(basis.mounts)):
                    keys = [
                        f"{row_prefix}{i}/{part}"
                        for part in ("core", "output_energy", "input_energy")
                    ]
                    if all(k in features for k in keys):
                        mounts[name] = dict(
                            zip(
                                ("core", "output_energy", "input_energy"),
                                (features[k] for k in keys),
                                strict=True,
                            )
                        )
                if f"{row_prefix}complete" not in features:
                    try:
                        mounts = lora_coefficients(sorted(filenames)[0], basis)
                    except (OSError, ValueError, KeyError, RuntimeError):
                        continue
                    for i, name in enumerate(sorted(basis.mounts)):
                        if name in mounts:
                            for part in ("core", "output_energy", "input_energy"):
                                features[f"{row_prefix}{i}/{part}"] = mounts[name][
                                    part
                                ].contiguous()
                    features[f"{row_prefix}complete"] = torch.tensor([1], dtype=torch.uint8)
                    changed = True
                if not set(mounts).intersection(model.mounts):
                    continue
                features[key] = model.encode([{"mounts": mounts}])[0].cpu().contiguous()
                changed = True
            vectors[digest] = features[key]
        if changed:
            _atomic(features_path, lambda name: save_file(features, name))
            self._features = features
            self._feature_keys = None
            self._retrieval_cache.clear()
        return vectors

    def similar(self, lora: str | os.PathLike, count: int, *, vectors=None) -> list[SimilarResult]:

        if count < 0:
            raise ValueError("count must be nonnegative")
        if count == 0:
            return []
        path = str(Path(lora).expanduser().resolve())
        digest = self.paths.get(path)
        if digest is None or self.hash_index.hash(path) != digest:
            raise ValueError("LoRA is not in the current library index; sync its folder first")
        record = self.records[digest]
        family = record["family"]
        if family == "unknown":
            return []
        data = self.retrieval_inputs(family)
        summaries = data["summaries"] if vectors is None else vectors
        query = summaries.get(digest)
        if query is None or not query.numel():
            return []
        query = query.float().flatten()
        norm = torch.linalg.vector_norm(query)
        if not torch.isfinite(norm) or norm == 0:
            return []
        available = {}
        for filename, item in sorted(self.paths.items()):
            if item != digest and Path(filename).is_file():
                available.setdefault(item, filename)
        results = []
        for item, candidate in summaries.items():
            if item not in available or candidate.numel() != query.numel():
                continue
            candidate = candidate.float().flatten()
            candidate_norm = torch.linalg.vector_norm(candidate)
            if not torch.isfinite(candidate_norm) or candidate_norm == 0:
                continue
            score = float(torch.dot(query, candidate) / (norm * candidate_norm))
            results.append(SimilarResult(available[item], item, max(-1.0, min(1.0, score))))
        return sorted(results, key=lambda result: (-result.score, result.path))[:count]

    def sync(self, folders, clip=None, *, reuse_text=False) -> SyncReport:
        self._features = None
        self._feature_keys = None
        self._retrieval_cache.clear()
        started = time.monotonic()
        report = SyncReport()
        paths = {}
        files = set()
        for folder in folders:
            root = Path(folder).expanduser().resolve()
            if root.is_dir():
                for file in root.rglob("*.safetensors"):
                    resolved = file.resolve()
                    if resolved.is_relative_to(root) and resolved.is_file():
                        files.add(resolved)
        new = {}
        for file in sorted(files):
            try:
                digest = self.hash_index.hash(file)
                record = self.records.get(digest)
                if record is None:
                    record = _metadata(file)
                paths[str(file)] = digest
                new.setdefault(digest, record)
            except Exception as exc:
                report.skipped += 1
                report.reasons.append(f"{file}: {exc}")
        for file, digest in paths.items():
            record = new[digest]
            current = str(Path(record["folder"]) / record["filename"])
            if current not in paths or paths[current] != digest:
                record = dict(record, filename=Path(file).name, folder=str(Path(file).parent))
                new[digest] = record
        report.added = len(new.keys() - self.records.keys())
        report.removed = len(self.records.keys() - new.keys())
        features_path = self.path / "features.safetensors"
        features = load_file(str(features_path)) if features_path.exists() else {}
        removed = self.records.keys() - new.keys()
        for digest in sorted(removed):
            family = self.records[digest]["family"]
            book = self.codebooks.families.get(family)
            if book is not None and digest in book.assignments:
                book.remove(digest)
        features = {k: v for k, v in features.items() if k.partition("/")[0] not in removed}
        content_names = {}
        for digest, record in sorted(new.items()):
            if record.get("content_version", 0) >= _CONTENT_VERSION:
                continue
            file = Path(record["folder"]) / record["filename"]
            try:
                family = record["family"]
                if family in PRESETS and family not in content_names:
                    content_names[family] = _content_names(family)
                content = _content(file, family, content_names.get(family))
                updated = {}
                if content is None:
                    updated[f"{digest}/summary"] = torch.empty(0, dtype=torch.float16)
                    report.reasons.append(f"{file}: no layers matched a content group")
                else:
                    for group, directions in content.directions.items():
                        updated[f"{digest}/{group}/directions"] = directions.to(
                            dtype=torch.float16
                        ).contiguous()
                        updated[f"{digest}/{group}/values"] = (
                            content.singular_values[group].to(dtype=torch.float16).contiguous()
                        )
                    updated[f"{digest}/summary"] = content.summary.to(
                        dtype=torch.float16
                    ).contiguous()
                prefix = f"{digest}/"
                features = {
                    key: value
                    for key, value in features.items()
                    if not key.startswith(prefix)
                    or key[len(prefix) :].startswith(("text/", "coefficients/", "embedding/"))
                }
                features.update(updated)
                record["content_version"] = _CONTENT_VERSION
            except Exception as exc:
                features.setdefault(f"{digest}/summary", torch.empty(0, dtype=torch.float16))
                report.reasons.append(f"{file}: content features: {exc}")
        for digest, record in new.items():
            file = Path(record["folder"]) / record["filename"]
            try:
                signature = _text_signature(file)
                if (
                    reuse_text
                    and record.get("text_signature") == signature
                    and all(k in record for k in ("tags", "trigger_words", "filename_words"))
                ):
                    text = TextFeatures(
                        tuple(record["tags"]),
                        tuple(record["trigger_words"]),
                        tuple(record["filename_words"]),
                    )
                else:
                    text = read_text_features(file)
                record["text_signature"] = signature
                record["tags"] = list(text.tags)
                record["trigger_words"] = list(text.trigger_words)
                record["filename_words"] = list(text.filename_words)
                for encoder in (None, clip) if clip is not None else (None,):
                    identity = encoder_identity(encoder)
                    key = f"{digest}/text/{identity}"
                    if key not in features:
                        features[key] = embed_text(text.text(), encoder).to(dtype=torch.float16)
            except Exception as exc:
                report.reasons.append(f"{file}: text features: {exc}")
        family_groups = {}
        for key, value in features.items():
            digest, separator, rest = key.partition("/")
            if separator and digest in new and rest.endswith("/directions"):
                group = rest[: -len("/directions")]
                family_groups.setdefault(new[digest]["family"], {})[group] = value.shape[1]
        for digest, record in new.items():
            key = f"{digest}/summary"
            if key not in features or not features[key].numel():
                continue
            pieces = []
            for group, width in sorted(family_groups[record["family"]].items()):
                direction = features.get(f"{digest}/{group}/directions")
                values = features.get(f"{digest}/{group}/values")
                pieces.append(
                    direction[0] * values[0]
                    if direction is not None and len(direction)
                    else torch.zeros(width, dtype=torch.float16)
                )
            features[key] = torch.cat(pieces)
        for book in self.codebooks.families.values():
            for digest in list(book.assignments):
                if not features.get(f"{digest}/summary", torch.empty(0)).numel():
                    book.remove(digest)
                    new[digest]["cluster_id"] = None
        for book in self.codebooks.families.values():
            changed = set()
            for digest, cluster_id in book.assignments.items():
                value = features[f"{digest}/summary"].float()
                if not torch.equal(value, book._vectors[digest]):
                    book._vectors[digest] = value
                    changed.add(cluster_id)
            for cluster_id in changed:
                book._refresh(cluster_id)
        for digest in sorted(new):
            summary = features.get(f"{digest}/summary")
            if summary is not None and summary.numel():
                book = self.codebooks.for_family(new[digest]["family"])
                if digest not in book.assignments:
                    book.add(digest, summary)
        for book in self.codebooks.families.values():
            text = {
                digest: features[f"{digest}/text/trigram-512-v1"]
                for digest in book.assignments
                if f"{digest}/text/trigram-512-v1" in features
            }
            book.fit_alignment(text)
            for digest, cluster_id in book.assignments.items():
                new[digest]["cluster_id"] = cluster_id
        features = {k: v for k, v in features.items() if not k.startswith("codebook/")}
        metadata, book_features = self.codebooks.dump()
        features.update(book_features)
        _atomic(features_path, lambda name: save_file(features, name))
        data = {
            "format": "kilora-library",
            "version": "1.0",
            "records": new,
            "paths": paths,
            "codebooks": metadata,
        }
        _atomic(
            self.path / "library.json",
            lambda name: Path(name).write_text(json.dumps(data, sort_keys=True), encoding="utf-8"),
        )
        self.records, self.paths = new, paths
        self._features = features
        self._feature_keys = {}
        for key, value in features.items():
            digest, separator, rest = key.partition("/")
            if separator:
                self._feature_keys.setdefault(digest, {})[rest] = value
        report.seconds = time.monotonic() - started
        return report
