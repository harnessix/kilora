# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._version import __version__
from .compute import RECORD_AXES
from .errors import ModFormatError, UnsupportedFeatureError

EMBEDDING_DRAW_SCALE = 0.2

PALETTE_FORMAT = "kilora-palette"
PALETTE_VERSION = "1.0"
PALETTE_SUFFIX = ".kpalette.json"
REGION_LABELS = ("a", "b")
_MAX_BYTES = 256 * 1024
_MAX_ENTRIES = 512

PALETTE_DIR: str | None = None


@dataclass
class PaletteEntry:
    kind: str
    name: str | None
    sha256: str | None = None
    strength: float = 1.0
    enabled: bool = True
    roles: str = "auto"
    region: str | None = None
    axis: str = "creative"
    triggers: list[str] = field(default_factory=list)
    mod_id: str | None = None
    version: str | None = None
    digest: str | None = None

    @property
    def label(self) -> str:
        return self.name or f"{self.mod_id} {self.version or ''}".strip()


@dataclass
class Palette:
    name: str
    entries: list[PaletteEntry]
    settings: dict[str, Any] = field(default_factory=dict)
    depth: int | None = None


def palette_dir() -> str:
    if PALETTE_DIR is not None:
        return PALETTE_DIR
    try:
        from .host import services

        folder_paths = getattr(services().comfy, "folder_paths", None)
        if folder_paths is None:
            import folder_paths
        user = folder_paths.get_user_directory()
    except Exception:
        user = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
    return os.path.join(user, "kilora_palettes")


def list_palettes() -> list[str]:
    folder = palette_dir()
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.endswith(PALETTE_SUFFIX))


def save_palette(path: str | os.PathLike[str], palette: Palette) -> dict[str, Any]:
    data = {
        "format": PALETTE_FORMAT,
        "format_version": PALETTE_VERSION,
        "name": palette.name,
        "created_with": f"kilora {__version__}",
        "settings": _plain(palette.settings),
        "depth": palette.depth,
        "entries": [
            {
                "kind": e.kind,
                "name": e.name,
                "sha256": e.sha256,
                "strength": e.strength,
                "enabled": e.enabled,
                "roles": e.roles,
                "region": e.region,
                "axis": e.axis,
                "triggers": list(e.triggers),
            }
            for e in palette.entries
        ],
    }
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return data


def _plain(settings: dict) -> dict:
    return {k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool, type(None)))}


def load_palette(path: str | os.PathLike[str]) -> Palette:
    label = os.path.basename(os.fspath(path))
    if os.path.getsize(path) > _MAX_BYTES:
        raise ModFormatError(f"{label} is larger than {_MAX_BYTES} bytes")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModFormatError(f"{label} is not a JSON palette ({exc})") from None
    if isinstance(data, dict) and data.get("format") == "kilora-pack":
        return _from_pack(path)
    if not isinstance(data, dict) or data.get("format") != PALETTE_FORMAT:
        raise ModFormatError(f"{label}: 'format' must be {PALETTE_FORMAT!r}")
    major = str(data.get("format_version", "")).split(".")[0]
    if not major.isdigit():
        raise ModFormatError(f"{label}: 'format_version' must be a 'major.minor' string")
    if int(major) > int(PALETTE_VERSION.split(".")[0]):
        raise UnsupportedFeatureError(
            f"{label} uses palette format {data['format_version']}; update KiLoRA to load it"
        )
    items = data.get("entries")
    if not isinstance(items, list) or len(items) > _MAX_ENTRIES:
        raise ModFormatError(f"{label}: 'entries' must be a list of at most {_MAX_ENTRIES}")
    entries = []
    for item in items:
        if not isinstance(item, dict) or item.get("kind") not in ("lora", "mod"):
            raise ModFormatError(f"{label}: each entry needs a 'kind' of lora or mod")
        if not isinstance(item.get("name"), str):
            raise ModFormatError(f"{label}: each entry needs a file 'name'")
        region = item.get("region")
        if region is not None and region not in REGION_LABELS:
            raise ModFormatError(f"{label}: region must be one of {REGION_LABELS} or null")
        axis = item.get("axis") or "creative"
        if axis not in RECORD_AXES:
            raise ModFormatError(f"{label}: axis must be one of {RECORD_AXES}")
        entries.append(
            PaletteEntry(
                kind=item["kind"],
                name=item["name"],
                sha256=item.get("sha256"),
                strength=float(item.get("strength", 1.0)),
                enabled=bool(item.get("enabled", True)),
                roles=str(item.get("roles") or "auto"),
                region=region,
                axis=axis,
                triggers=[str(w) for w in item.get("triggers") or []],
            )
        )
    depth = data.get("depth")
    if depth is not None and (not isinstance(depth, int) or depth < 0):
        raise ModFormatError(f"{label}: 'depth' must be an integer >= 0 or null")
    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise ModFormatError(f"{label}: 'settings' must be an object")
    return Palette(str(data.get("name", label)), entries, dict(settings), depth)


def _from_pack(path) -> Palette:
    from .packs import load_pack

    pack = load_pack(path)
    entries = [
        PaletteEntry(
            kind="mod",
            name=None,
            sha256=e.file_sha256,
            strength=e.strength,
            enabled=e.enabled,
            mod_id=e.mod_id,
            version=e.version,
            digest=e.digest,
        )
        for e in pack.entries
    ]
    return Palette(pack.name, entries, {}, pack.depth)


@dataclass
class Resolved:
    entry: PaletteEntry
    path: str


def resolve(
    palette: Palette,
    find: Callable[[str, str], str | None],
    sha256: Callable[[str], str],
    library: Callable[[], Sequence[Any]] | None = None,
) -> tuple[list[Resolved], list[str]]:
    found, problems = [], []
    lib = None
    for e in palette.entries:
        if e.name is None:
            from .packs import Pack, PackEntry, pack_choices

            lib = lib if lib is not None else list(library() if library else [])
            pe = PackEntry(
                e.mod_id or "", e.version, e.digest, e.strength, 1.0, e.enabled, e.sha256
            )
            try:
                (choice,) = pack_choices(Pack(palette.name, [pe]), lib)
            except ModFormatError:
                problems.append(f"{e.label}: not found in the mod folder, skipped")
                continue
            found.append(Resolved(e, os.fspath(choice.path)))
            continue
        path = find(e.kind, e.name)
        if path is None or not os.path.isfile(path):
            problems.append(f"{e.name}: not found, skipped")
            continue
        if e.sha256 and sha256(path) != e.sha256:
            problems.append(f"{e.name}: the file changed (another sha256), skipped")
            continue
        found.append(Resolved(e, path))
    return found, problems


@dataclass
class ShuffleState:
    counts: dict[str, list[list[int]]] = field(default_factory=dict)
    embedding: dict[str, dict[str, Any]] = field(default_factory=dict)


def shuffle_entry_key(entry: PaletteEntry) -> str:
    identity = [entry.kind, entry.name, entry.sha256, entry.mod_id, entry.version, entry.digest]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()


def shuffle_state_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    if not path.name.endswith(PALETTE_SUFFIX):
        raise ValueError("Shuffle state requires a .kpalette.json palette")
    return path.with_name(path.name[: -len(".json")] + ".state.json")


def load_shuffle_state(path: str | os.PathLike[str]) -> ShuffleState:
    state_path = shuffle_state_path(path)
    if not state_path.exists():
        return ShuffleState()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Shuffle state must contain a counts object")
    embedding = data.pop("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Invalid Shuffle embedding state")
    for key, model in embedding.items():
        if not _valid_posterior(key, model):
            raise ValueError("Invalid Shuffle embedding state")
    for key, counts in data.items():
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
            or not isinstance(counts, list)
            or len(counts) != 6
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(n) is not int or n < 0 for n in pair)
                for pair in counts
            )
        ):
            raise ValueError("Invalid Shuffle counts")
    return ShuffleState(data, embedding)


def save_shuffle_state(path: str | os.PathLike[str], state: ShuffleState) -> None:
    state_path = shuffle_state_path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=state_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            data = dict(state.counts)
            if state.embedding:
                data["embedding"] = state.embedding
            json.dump(data, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, state_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def reset_shuffle_state(path: str | os.PathLike[str]) -> None:
    shuffle_state_path(path).unlink(missing_ok=True)


def strength_bin(value: float, strength: tuple[float, float]) -> int:
    low, high = sorted(map(float, strength))
    if low == high:
        return 0
    return min(4, max(0, int((value - low) / (high - low) * 5)))


def reward_shuffle(
    state: ShuffleState,
    pick: Sequence[Resolved],
    strength: tuple[float, float],
    reward: str,
    embeddings: dict | None = None,
) -> None:
    if reward not in ("keep", "discard"):
        raise ValueError(f"Unknown Shuffle reward: {reward}")
    column = 0 if reward == "keep" else 1
    features = {}
    for item in pick:
        if not item.entry.enabled or item.entry.axis == "functional":
            continue
        embedding = (embeddings or {}).get(item.path)
        if embedding is not None:
            key, vector = embedding
            features.setdefault(key, []).append(
                _feature(vector, strength_bin(item.entry.strength, strength))
            )
        counts = state.counts.setdefault(shuffle_entry_key(item.entry), [[0, 0] for _ in range(6)])
        counts[0][column] += 1
        counts[1 + strength_bin(item.entry.strength, strength)][column] += 1
    if features:
        import torch

        for key, rows in features.items():
            x = torch.tensor(rows, dtype=torch.float64).sum(dim=0)
            B, f = _posterior(state, key, len(x))
            B += torch.outer(x, x)
            f += x * (1 if reward == "keep" else 0)
            state.embedding[key] = {"B": B.tolist(), "f": f.tolist()}


def _learn_shuffle(pool, keep, entries, rng, n, s_low, s_high, state):
    scores = []
    for i, item in enumerate(pool):
        counts = state.counts.get(shuffle_entry_key(item.entry), [[0, 0] for _ in range(6)])
        score = rng.betavariate(counts[0][0] + 1, counts[0][1] + 1)
        bins = [rng.betavariate(a + 1, b + 1) for a, b in counts[1:]]
        scores.append((score, i, max(range(5), key=bins.__getitem__)))
    chosen = {}
    for _, i, bin_index in sorted(scores, reverse=True)[:n]:
        item = pool[i]
        value = s_low + (s_high - s_low) * (bin_index + 0.5) / 5
        chosen[id(item)] = Resolved(_with_strength(item.entry, value), item.path)
    kept = {id(item) for item in keep}
    return [
        chosen.get(id(item), item) for item in entries if id(item) in chosen or id(item) in kept
    ]


def _valid_posterior(key, model):
    if (
        not isinstance(key, str)
        or len(key) != 64
        or any(c not in "0123456789abcdef" for c in key)
        or not isinstance(model, dict)
        or set(model) != {"B", "f"}
    ):
        return False
    matrix, vector = model["B"], model["f"]
    if not isinstance(vector, list) or not 6 <= len(vector) <= 4096:
        return False
    if not isinstance(matrix, list) or len(matrix) != len(vector):
        return False

    def numbers(row):
        return (
            isinstance(row, list)
            and len(row) == len(vector)
            and all(type(v) in (int, float) and math.isfinite(v) for v in row)
        )

    return numbers(vector) and all(numbers(row) for row in matrix)


def _feature(vector, bin_index):
    return list(map(float, vector)) + [float(i == bin_index) for i in range(5)]


def _posterior(state, key, size):
    import torch

    data = state.embedding.get(key)
    if data is None:
        return torch.eye(size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64)
    if not _valid_posterior(key, data) or len(data["f"]) != size:
        raise ValueError("Invalid Shuffle embedding dimensions")
    return torch.tensor(data["B"], dtype=torch.float64), torch.tensor(
        data["f"], dtype=torch.float64
    )


def _embedding_shuffle(pool, keep, entries, rng, n, s_low, s_high, state, embeddings):
    import torch

    samples = {}
    scores = []
    for i, item in enumerate(pool):
        embedding = embeddings.get(item.path)
        if embedding is None:
            counts = state.counts.get(shuffle_entry_key(item.entry), [[0, 0] for _ in range(6)])
            score = rng.betavariate(counts[0][0] + 1, counts[0][1] + 1)
            bins = [rng.betavariate(a + 1, b + 1) for a, b in counts[1:]]
            bin_index = max(range(5), key=bins.__getitem__)
        else:
            key, vector = embedding
            if key not in samples:
                B, f = _posterior(state, key, len(vector) + 5)
                chol = torch.linalg.cholesky(B)
                mean = torch.cholesky_solve(f[:, None], chol)[:, 0]
                noise = torch.tensor([rng.gauss(0, 1) for _ in f], dtype=torch.float64)
                samples[key] = (
                    mean
                    + EMBEDDING_DRAW_SCALE
                    * torch.linalg.solve_triangular(chol.T, noise[:, None], upper=True)[:, 0]
                )
            weights = samples[key]
            values = [
                float(torch.dot(weights, torch.tensor(_feature(vector, b), dtype=torch.float64)))
                for b in range(5)
            ]
            bin_index = max(range(5), key=values.__getitem__)
            score = values[bin_index]
        scores.append((score, i, bin_index))
    chosen = {}
    for _, i, bin_index in sorted(scores, reverse=True)[:n]:
        item = pool[i]
        value = s_low + (s_high - s_low) * (bin_index + 0.5) / 5
        chosen[id(item)] = Resolved(_with_strength(item.entry, value), item.path)
    kept = {id(item) for item in keep}
    return [
        chosen.get(id(item), item) for item in entries if id(item) in chosen or id(item) in kept
    ]


def shuffle(
    entries: Sequence[Resolved],
    seed: int,
    count: tuple[int, int],
    strength: tuple[float, float],
    mode: str = "random",
    state: ShuffleState | None = None,
    embeddings: dict | None = None,
) -> list[Resolved]:
    if mode not in ("random", "learn", "learn_embedding"):
        raise ValueError(f"Unknown Shuffle mode: {mode}")
    pool = [r for r in entries if r.entry.enabled and r.entry.axis != "functional"]
    keep = [r for r in entries if r.entry.enabled and r.entry.axis == "functional"]
    if not pool:
        return keep
    low, high = sorted((max(1, int(count[0])), max(1, int(count[1]))))
    low, high = min(low, len(pool)), min(high, len(pool))
    s_low, s_high = sorted((float(strength[0]), float(strength[1])))
    rng = random.Random(int(seed))
    n = rng.randint(low, high)
    if mode == "learn":
        return _learn_shuffle(pool, keep, entries, rng, n, s_low, s_high, state or ShuffleState())
    if mode == "learn_embedding":
        return _embedding_shuffle(
            pool, keep, entries, rng, n, s_low, s_high, state or ShuffleState(), embeddings or {}
        )
    picked = sorted(rng.sample(range(len(pool)), n))
    chosen = {}
    for i in picked:
        r = pool[i]
        value = round(rng.uniform(s_low, s_high), 2)
        chosen[id(r)] = Resolved(_with_strength(r.entry, value), r.path)
    kept = {id(r) for r in keep}
    return [chosen.get(id(r), r) for r in entries if id(r) in chosen or id(r) in kept]


def _with_strength(e: PaletteEntry, strength: float) -> PaletteEntry:
    import dataclasses

    return dataclasses.replace(e, strength=strength)


__all__ = [
    "PALETTE_SUFFIX",
    "Palette",
    "PaletteEntry",
    "ShuffleState",
    "load_shuffle_state",
    "reset_shuffle_state",
    "reward_shuffle",
    "save_shuffle_state",
    "shuffle_entry_key",
    "shuffle_state_path",
    "strength_bin",
    "list_palettes",
    "load_palette",
    "palette_dir",
    "resolve",
    "save_palette",
    "shuffle",
]
