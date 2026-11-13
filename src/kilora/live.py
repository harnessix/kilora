# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ._version import __version__
from .errors import KiLoRAError, ModFormatError, UnsupportedFeatureError
from .host_copy import host_copy_map
from .layout import Layout, State
from .modfile import (
    MANIFEST_KEY,
    Mod,
    _check_safetensors,
    _check_sizes,
    compute_digest,
    load_mod,
    save_mod,
)
from .sets import AdapterSet, KeyedExperts

log = logging.getLogger("kilora.live")

STATE_PREFIX = "kilora/state/"
EDITOR_STATE_PREFIX = "kilora/editor/"
PREVIOUS_PREFIX = "kilora/previous/"


def _composition_key(items: Sequence[Mapping[str, Any]]) -> list[tuple]:
    return sorted(
        (
            str(m.get("id")),
            str(m.get("digest")),
            round(float(m.get("strength", 1.0)), 6),
            int(m.get("depth", 0)),
        )
        for m in items
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def check_measure_samples(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise KiLoRAError("measure_samples must be an integer of at least 1")
    return value


SAVE_FORMAT = "kilora-save"
FILE_VERSION = "1.0"
AUTOSAVES = 10
DEFAULT_UNDO_LOG_MB = 256.0
DEFAULT_DRIFT_THRESHOLD = 0.15
LEARNED_STATE_WARNING = (
    "This file holds a live state that was learned from images the user picked. "
    "People who load it get what it learned. Share it only if you are fine with that."
)
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass
class LiveSpec:
    layout: Layout
    updater: str | dict[str, Any]
    live_set: str
    speed: float = 0.5
    speed_max: float = 1.0
    updater_config: dict[str, Any] | None = None
    report: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "layout": self.layout.id,
            "layout_spec": self.layout.to_dict(),
            "editor": self.updater,
            "live_set": self.live_set,
            "speed": self.speed,
            "speed_max": self.speed_max,
        }
        if self.updater_config is not None:
            out["editor_config"] = self.updater_config
        if self.report is not None:
            out["report"] = self.report
        return out


def live_spec(manifest: Mapping[str, Any]) -> LiveSpec | None:
    live = manifest.get("live")
    if not live:
        return None
    try:
        return LiveSpec(
            layout=Layout.from_dict(live["layout_spec"]),
            updater=live.get("editor", "plugin"),
            live_set=str(live["live_set"]),
            speed=float(live.get("speed", 0.5)),
            speed_max=float(live.get("speed_max", 1.0)),
            updater_config=live.get("editor_config"),
            report=live.get("report"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModFormatError(f"the 'live' field is invalid ({exc})") from None


def _check_file_version(manifest: Any, kind: str, label: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != kind:
        raise ModFormatError(f"{label}: 'format' must be {kind!r}")
    version = str(manifest.get("format_version", ""))
    major = version.split(".")[0]
    if not major.isdigit():
        raise ModFormatError(f"{label}: 'format_version' must be a 'major.minor' string")
    if int(major) > int(FILE_VERSION.split(".")[0]):
        raise UnsupportedFeatureError(
            f"{label} uses {kind} format {version}. KiLoRA {__version__} reads format "
            f"{FILE_VERSION.split('.')[0]}.x. Update KiLoRA to load it."
        )
    for feature in manifest.get("requires_features", []):
        raise UnsupportedFeatureError(
            f"{label} needs the feature {feature}, which KiLoRA {__version__} does not "
            "support. Update KiLoRA to load it."
        )


@dataclass
class Pairing:
    learning_default: bool
    message: str | None = None
    source: str = "plugin"
    updater_ref: dict[str, Any] = field(default_factory=dict)
    ready: bool = True


def default_pairing(mod: Mod) -> Pairing:
    return Pairing(True, None, "plugin", {}, True)


@dataclass
class Commit:
    time: str
    images: int
    speed: float
    scaled: bool = False
    source: str = "live write"


@dataclass
class SaveFile:
    manifest: dict[str, Any]
    state: State
    editor_state: dict[str, torch.Tensor] | None = None
    previous: dict[str, torch.Tensor] | None = None


def write_save(
    path: str | os.PathLike[str],
    state: State,
    *,
    mod: Mod,
    layout: Layout,
    updater_ref: Mapping[str, Any],
    commits: Sequence[Commit],
    kind: str,
    name: str,
    sequence: int,
    editor_state: Mapping[str, torch.Tensor] | None = None,
    editor_kind: str | None = None,
    previous: Mapping[str, torch.Tensor] | None = None,
    learned_with: Sequence[Mapping[str, Any]] | None = None,
    editor_packs: Sequence[Mapping[str, str]] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    staged = {
        f"{STATE_PREFIX}{t}/{k}": v.detach().float()
        for t, parts in state.items()
        for k, v in parts.items()
    }
    scalars = []
    for k, v in (editor_state or {}).items():
        if v.dim() == 0:
            scalars.append(k)
            v = v.reshape(1)
        staged[f"{EDITOR_STATE_PREFIX}{k}"] = v
    for k, v in (previous or {}).items():
        staged[f"{PREVIOUS_PREFIX}{k}"] = v
    tensors = host_copy_map(staged, lambda v: v.cpu().contiguous())
    manifest = {
        "format": SAVE_FORMAT,
        "format_version": FILE_VERSION,
        "requires_features": [],
        "mod": {
            "id": mod.manifest["id"],
            "version": mod.manifest["version"],
            "digest": mod.manifest.get("digest"),
        },
        "layout": layout.id,
        "state_targets": list(state),
        "editor": dict(updater_ref),
        "kind": kind,
        "name": name,
        "sequence": sequence,
        "commits": [dataclasses.asdict(c) for c in commits],
        "created_with": f"kilora {__version__}",
    }
    manifest["editor_state"] = (
        {"kind": editor_kind or "unknown", "feature": "editor.state", "scalars": sorted(scalars)}
        if editor_state is not None
        else None
    )
    if learned_with is not None:
        manifest["learned_with"] = [dict(m) for m in learned_with]
    if editor_packs:
        manifest["editor_packs"] = [dict(p) for p in editor_packs]
    if options:
        manifest["options"] = dict(options)
    manifest["previous"] = {"feature": "live.previous", "keys": sorted(previous or {})}
    _check_sizes(manifest, 0)
    manifest["digest"] = compute_digest(manifest, tensors)
    save_file(tensors, os.fspath(path), {MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
    return manifest


def read_save(path: str | os.PathLike[str]) -> SaveFile:
    label = os.path.basename(os.fspath(path))
    _check_safetensors(path)
    with safe_open(os.fspath(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(MANIFEST_KEY)
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if raw is None:
        raise ModFormatError(f"{label} has no KiLoRA manifest")
    manifest = json.loads(raw)
    _check_file_version(manifest, SAVE_FORMAT, label)
    if compute_digest(manifest, tensors) != manifest.get("digest"):
        raise ModFormatError(f"{label}: digest does not match; the file is damaged")
    state: dict[str, dict[str, torch.Tensor]] = {}
    state.update({t: {} for t in manifest.get("state_targets", [])})
    editor: dict[str, torch.Tensor] = {}
    previous = {k: tensors[f"{PREVIOUS_PREFIX}{k}"] for k in manifest["previous"]["keys"]}
    for k, v in tensors.items():
        if k.startswith(STATE_PREFIX):
            target, _, part = k[len(STATE_PREFIX) :].rpartition("/")
            state.setdefault(target, {})[part] = v.float()
        elif k.startswith(EDITOR_STATE_PREFIX):
            editor[k[len(EDITOR_STATE_PREFIX) :]] = v
    for k in (manifest["editor_state"] or {}).get("scalars", []):
        if k in editor:
            editor[k] = editor[k].reshape(())
    return SaveFile(
        manifest=manifest, state=state, editor_state=editor or None, previous=previous or None
    )


@dataclass
class UndoEntry:
    parts: State
    commits: list[Commit]
    nbytes: int
    editor: dict[str, torch.Tensor] | None = None
    previous: dict[str, torch.Tensor] | None = None
    branch: str | None = None
    whole_editor: bool = False
    branch_after: str | None = None
    added: dict[str, list[str]] = field(default_factory=dict)


class UndoLog:
    def __init__(self, max_mb: float = DEFAULT_UNDO_LOG_MB) -> None:
        self.max_mb = float(max_mb)
        self._entries: list[UndoEntry] = []
        self.dropped = 0
        self.last_editor: dict[str, torch.Tensor] | None = None
        self.last_previous: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self._entries)

    def set_size(self, max_mb: float) -> None:
        if not max_mb >= 0:
            raise KiLoRAError("the undo log size must be 0 MB or more")
        self.max_mb = float(max_mb)
        self._trim()

    def push(
        self,
        old: State,
        new: State,
        commits: Sequence[Commit],
        editor: Mapping[str, torch.Tensor] | None = None,
        previous: Mapping[str, torch.Tensor] | None = None,
        branch: str | None = None,
        whole_editor: bool = False,
        branch_after: str | None = None,
    ) -> None:
        changed: dict[tuple[str, str], torch.Tensor] = {}
        for t, old_parts in old.items():
            for k, v in old_parts.items():
                n = new.get(t, {}).get(k)
                if n is None or n.shape != v.shape or not torch.equal(n, v):
                    changed[(t, k)] = v
        parts: dict[str, dict[str, torch.Tensor]] = {}
        for (t, k), v in host_copy_map(changed, lambda v: v.cpu().clone()).items():
            parts.setdefault(t, {})[k] = v
        added = {t: [k for k in parts if k not in old.get(t, {})] for t, parts in new.items()}
        size = sum(v.numel() * v.element_size() for p in parts.values() for v in p.values())
        snapshot = None
        if editor is not None:
            snapshot = host_copy_map(editor, lambda v: v.cpu().clone())
            size += sum(v.numel() * v.element_size() for v in snapshot.values())
        prev = None
        if previous is not None:
            prev = host_copy_map(previous, lambda v: v.cpu().clone())
            size += sum(v.numel() * v.element_size() for v in prev.values())
        self._entries.append(
            UndoEntry(
                parts,
                list(commits),
                size,
                snapshot,
                prev,
                branch,
                whole_editor,
                branch_after,
                added,
            )
        )
        self._trim()

    def pop(self, state: State, index: int = -1) -> tuple[State, list[Commit]]:
        entry = self._entries.pop(index)
        restored = _copy(state)
        for t, keys in entry.added.items():
            for key in keys:
                restored.get(t, {}).pop(key, None)
        for t, parts in entry.parts.items():
            restored.setdefault(t, {}).update({k: v.clone() for k, v in parts.items()})
        self.last_editor = entry.editor
        self.last_previous = entry.previous
        return restored, list(entry.commits)

    def _trim(self) -> None:
        limit = self.max_mb * 1024 * 1024
        while len(self._entries) > 1 and self.nbytes > limit:
            self._entries.pop(0)
            self.dropped += 1


def _undo_log_mb() -> float:
    try:
        return max(float(os.environ.get("KILORA_UNDO_LOG_MB", DEFAULT_UNDO_LOG_MB)), 0.0)
    except ValueError:
        return DEFAULT_UNDO_LOG_MB


@dataclass
class LearnResult:
    images: int
    speed: float
    change: float
    scaled: bool
    discarded: bool
    message: str


class LiveSession:
    MEASURE_SAMPLES = 1

    def __init__(
        self,
        mod: Mod,
        pairing: Pairing,
        saves_dir: str | os.PathLike[str] | None = None,
        undo_log_mb: float | None = None,
    ) -> None:
        spec = live_spec(mod.manifest)
        if spec is None:
            raise KiLoRAError("this mod is not a live mod")
        self.mod = mod
        self.spec = spec
        self.layout = spec.layout
        self.pairing = pairing
        self.learning = pairing.learning_default
        self.record_prediction = True
        self.speed = spec.speed
        self.saves_dir = Path(saves_dir) if saves_dir is not None else None
        live = [s for s in mod.sets if s.name == spec.live_set]
        if len(live) != 1 or not live[0].has_lora:
            raise ModFormatError(f"the live set {spec.live_set!r} is missing from the mod")
        self._live_index = mod.sets.index(live[0])
        self.clean = {
            t: {
                "down": live[0].experts[t].down.float().clone(),
                "up": live[0].experts[t].up.float().clone(),
            }
            for t in self.layout.targets
        }
        for t, parts in self.clean.items():
            if parts["up"].shape[-1] != self.layout.width_of(t):
                raise ModFormatError(f"live state of {t} does not match layout {self.layout.id}")
        self.state = _copy(self.clean)
        self.pinned: dict[str, Any] | None = None
        self._pinned_state: State | None = None
        self.pinned_write_branch: str | None = None
        self.pending: State | None = None
        self._pending_info: LearnResult | None = None
        self.commits: list[Commit] = []
        self.undo_log = UndoLog(_undo_log_mb() if undo_log_mb is None else undo_log_mb)
        self._sequence = 0
        self._memory_saves: dict[str, tuple[State, list[Commit]]] = {}
        self.reference_probe: torch.Tensor | None = None
        self.revision = 0
        self.source_path: str | None = None
        self.editor_state: Any = None
        self.editor_state_kind: str | None = None
        self._pending_editor: tuple[dict[str, torch.Tensor] | None, str | None] | None = None
        self.keep_previous = _env_flag("KILORA_KEEP_PREVIOUS")
        self.editor_packs: list[Any] = []
        self.loaded_packs: list[dict[str, str]] = []
        self.previous: dict[str, torch.Tensor] | None = None
        from .measure_size import DEFAULT_MEASURE_SIZE, DEFAULT_REDUCER

        self.measure_size = DEFAULT_MEASURE_SIZE
        self.measure_reducer = DEFAULT_REDUCER
        self.measure_samples = type(self).MEASURE_SAMPLES
        self.editor_options: dict[str, Any] = {}
        self.composition: list[dict[str, Any]] = []
        from .prediction_write import DEFAULT_LEARN, DEFAULT_PRIOR_SCALE, DEFAULT_WEIGHTS

        self.learn = DEFAULT_LEARN
        self.prediction_weights = dict(DEFAULT_WEIGHTS)
        self.prediction_prior = DEFAULT_PRIOR_SCALE
        self.deferred_guard: Any = None

    def set_options(
        self,
        learning: bool | None = None,
        speed: float | None = None,
        undo_log_mb: float | None = None,
        keep_previous: bool | None = None,
        measure_size: int | None = None,
        measure_reducer: str | None = None,
        editor_options: Mapping[str, Any] | None = None,
        measure_samples: int | None = None,
        learn: str | None = None,
        prediction_weights: Mapping[str, float] | None = None,
        prediction_prior: float | None = None,
    ) -> str:
        from . import prediction_write as pw

        try:
            learn = None if learn is None else pw.check_learn(learn)
            weights = None if prediction_weights is None else pw.check_weights(prediction_weights)
            prior = None if prediction_prior is None else pw.check_prior_scale(prediction_prior)
        except ValueError as exc:
            raise KiLoRAError(str(exc)) from None
        if learn is not None:
            self.learn = learn
        if weights is not None:
            self.prediction_weights = weights
        if prior is not None:
            self.prediction_prior = prior
        if measure_samples is not None:
            self.measure_samples = check_measure_samples(measure_samples)
        if undo_log_mb is not None:
            self.undo_log.set_size(undo_log_mb)
        if measure_size is not None:
            from .measure_size import check_measure_size

            try:
                self.measure_size = check_measure_size(measure_size)
            except ValueError as exc:
                raise KiLoRAError(str(exc)) from None
        if measure_reducer is not None:
            from .measure_size import REDUCERS

            if measure_reducer not in REDUCERS:
                raise KiLoRAError(
                    f"unknown reduction {measure_reducer!r}; known: {', '.join(sorted(REDUCERS))}"
                )
            self.measure_reducer = measure_reducer
        if editor_options:
            from .editors import LIVE_OPTIONS

            unknown = sorted(set(editor_options) - set(LIVE_OPTIONS))
            if unknown:
                raise KiLoRAError(f"unknown editor setting {unknown[0]!r}")
            if editor_options.get("diff") not in (None, "central", "one-sided"):
                raise KiLoRAError("the measurement is central or one-sided")
            if editor_options.get("time_bins") not in (None, 1, 2):
                raise KiLoRAError("the number of time bins is 1 or 2")
            if "path_merge" in editor_options and not isinstance(
                editor_options["path_merge"], bool
            ):
                raise KiLoRAError("the path merge setting is on or off")
            if "keep_merge_pairs" in editor_options and not isinstance(
                editor_options["keep_merge_pairs"], bool
            ):
                raise KiLoRAError("the merge pair setting is on or off")
            if editor_options.get("cross_layer", "off") not in ("off", "rls", "jvp"):
                raise KiLoRAError("the cross-layer setting is off, rls or jvp")
            if "path_angle" in editor_options:
                angle = editor_options["path_angle"]
                if (
                    isinstance(angle, bool)
                    or not isinstance(angle, (int, float))
                    or not math.isfinite(angle)
                    or angle < 0
                ):
                    raise KiLoRAError("the path angle must be a finite nonnegative number")
            if editor_options.get("merge_residual", "keep") not in ("keep", "drop"):
                raise KiLoRAError("merge residual must be keep or drop")
            if "explore" in editor_options:
                explore = editor_options["explore"]
                if (
                    isinstance(explore, bool)
                    or not isinstance(explore, (int, float))
                    or not math.isfinite(explore)
                    or explore < 0
                ):
                    raise KiLoRAError("explore must be a finite nonnegative number")
            if "term_budget" in editor_options and (
                isinstance(editor_options["term_budget"], bool)
                or not isinstance(editor_options["term_budget"], int)
                or editor_options["term_budget"] < 1
            ):
                raise KiLoRAError("term budget must be a positive integer")
            self.editor_options.update(editor_options)
        if keep_previous is not None:
            self.keep_previous = bool(keep_previous)
            if not self.keep_previous:
                self.previous = None
        if learning is not None:
            if learning and not self.pairing.ready:
                raise KiLoRAError(self.pairing.message or "this live mod has no editor")
            self.learning = bool(learning)
        if speed is not None:
            self.speed = min(max(float(speed), 0.0), self.spec.speed_max)
        self.revision += 1
        note = " Outside the editor's trained range." if self.outside_trained_range else ""
        return f"Learning {'on' if self.learning else 'off'}, speed {self.speed:g}.{note}"

    @property
    def mod_id(self) -> str:
        return self.mod.manifest["id"]

    @property
    def outside_trained_range(self) -> bool:
        return self.learning and not self.pairing.learning_default

    def sets(self, use_pending: bool = True) -> list[AdapterSet]:
        state = (
            self._pinned_state
            if self.pinned is not None
            else self.pending
            if (use_pending and self.pending is not None)
            else self.state
        )
        out = list(self.mod.sets)
        live = out[self._live_index]
        experts = dict(live.experts)
        for t, parts in state.items():
            ex = experts[t]
            experts[t] = KeyedExperts(
                keys_in=ex.keys_in,
                keys_out=ex.keys_out,
                rows_in=ex.rows_in,
                rows_out=ex.rows_out,
                down=parts["down"].detach().cpu().clone(),
                up=parts["up"].detach().cpu().clone(),
            )
        extra = {**live.extra, "live_id": self.mod_id}
        out[self._live_index] = dataclasses.replace(live, experts=experts, extra=extra)
        return out

    def pin(self) -> str:
        if self.pinned is not None:
            return "Already pinned."
        forest = getattr(self.editor_state, "forest", None)
        if forest is None:
            raise KiLoRAError("pin requires an editor with branches")
        if self.pending is not None:
            raise KiLoRAError("finish the pending update before pinning")
        self.pinned = {"branch": forest.current, "revision": self.revision}
        self._pinned_state = _copy(self.state)
        self.pinned_write_branch = None
        return f"Pinned {forest.current} at revision {self.revision}."

    def unpin(self) -> str:
        if self.pinned is None:
            return "Not pinned."
        self.pinned = None
        self._pinned_state = None
        self.pinned_write_branch = None
        self.revision += 1
        return "Unpinned."

    def prepare_pinned_write(self) -> None:
        if self.pinned is None or self.pinned_write_branch is not None:
            return
        st = self.editor_state
        branch = st.fork(self.pinned["branch"])
        st.checkout(branch)
        self.pinned_write_branch = branch

    def _saved_editor_snapshot(self):
        tensors = self.editor_snapshot()
        if self.pinned is None or tensors is None:
            return tensors
        metadata = {"pin": self.pinned, "write_branch": self.pinned_write_branch}
        tensors["pin/metadata"] = torch.tensor(
            list(json.dumps(metadata).encode()), dtype=torch.uint8
        )
        for target, parts in self._pinned_state.items():
            for part, value in parts.items():
                tensors[f"pin/state/{target}/{part}"] = value.clone()
        return tensors

    def _restore_pin(self, tensors):
        self.pinned = self._pinned_state = self.pinned_write_branch = None
        if tensors is None or "pin/metadata" not in tensors:
            return
        metadata = json.loads(bytes(tensors["pin/metadata"].tolist()))
        state = {}
        for key, value in tensors.items():
            if key.startswith("pin/state/"):
                target, _, part = key[len("pin/state/") :].rpartition("/")
                state.setdefault(target, {})[part] = value.clone()
        if set(state) != set(self.layout.targets):
            raise ModFormatError("saved pin does not match the live state")
        self.pinned = metadata["pin"]
        self.pinned_write_branch = metadata.get("write_branch")
        self._pinned_state = state

    def discard(self) -> None:
        self.pending = None
        self.revision += 1
        self._pending_info = None

    def attach_editor_state(self, state: Any, kind: str) -> str:
        self.editor_state, self.editor_state_kind = state, kind
        if self._pending_editor is None:
            return ""
        tensors, saved_kind = self._pending_editor
        self._pending_editor = None
        return self._load_editor_state(tensors, saved_kind).strip()

    def editor_snapshot(self) -> dict[str, torch.Tensor] | None:
        if self.editor_state is None:
            return None
        return host_copy_map(self.editor_state.state_tensors(), lambda v: v.cpu().clone())

    def set_previous(
        self, latent: torch.Tensor | None, conditioning: Mapping[str, Any] | None
    ) -> None:
        if not self.keep_previous or latent is None:
            return
        prev = {"latent": latent.detach().cpu().clone()}
        for k, v in (conditioning or {}).items():
            if isinstance(v, torch.Tensor):
                prev[f"cond/{k}"] = v.detach().cpu().clone()
        self.previous = prev

    def previous_snapshot(self) -> dict[str, torch.Tensor] | None:
        if self.previous is None:
            return None
        return {k: v.clone() for k, v in self.previous.items()}

    def commit(
        self,
        editor_before: Mapping[str, torch.Tensor] | None = None,
        previous_before: Any = "current",
    ) -> str:

        if self.pending is None:
            raise KiLoRAError("there is no pending update to save")
        info = self._pending_info
        before = editor_before if editor_before is not None else self.editor_snapshot()
        prev = self.previous_snapshot() if previous_before == "current" else previous_before
        forest = getattr(self.editor_state, "forest", None)
        self.undo_log.push(
            self.state,
            self.pending,
            self.commits,
            before,
            prev,
            branch=None if forest is None else forest.current,
        )
        self.state = self.pending
        self.revision += 1
        self.pending, self._pending_info = None, None
        time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self.commits.append(Commit(time, info.images, info.speed, info.scaled, info.message))
        name = self._autosave()
        return f"Saved. {len(self.commits)} commits. Autosave {name}."

    def undo(self) -> str:
        if self.pending is not None:
            self.discard()
            return "Discarded the pending update."
        if not len(self.undo_log):
            raise KiLoRAError("nothing to undo")
        self.deferred_guard = None
        forest = getattr(self.editor_state, "forest", None)
        index = -1
        if forest is not None:
            index = next(
                (
                    i
                    for i in range(len(self.undo_log._entries) - 1, -1, -1)
                    if self.undo_log._entries[i].branch == forest.current
                    or (
                        self.undo_log._entries[i].whole_editor
                        and self.undo_log._entries[i].branch_after == forest.current
                    )
                    or (
                        self.undo_log._entries[i].branch is None
                        and self.undo_log._entries[i].editor is None
                        and len(forest.branches) == 1
                    )
                ),
                None,
            )
            if index is None:
                raise KiLoRAError("nothing to undo on this branch")
        whole_editor = self.undo_log._entries[index].whole_editor
        self.state, self.commits = self.undo_log.pop(self.state, index)
        if self.editor_state is not None:
            if self.undo_log.last_editor is not None:
                if forest is not None and not whole_editor:
                    self.editor_state.restore_branch(self.undo_log.last_editor)
                    self.state = self.editor_state.expand_live(self.state)
                else:
                    self.editor_state.load_tensors(self.undo_log.last_editor)
            else:
                self._restart_editor(self.state)
        self._pending_editor = None
        self.previous = self.undo_log.last_previous
        self.revision += 1
        return f"Undone. {len(self.commits)} commits. {len(self.undo_log)} more undo steps."

    def reset(self) -> str:
        self.deferred_guard = None
        forest = getattr(self.editor_state, "forest", None)
        if forest is not None:
            before = self.editor_snapshot()
            old = _copy(self.state)
            self.editor_state.reset()
            clean = self.editor_state.expand_live(self.state)
            self.undo_log.push(
                old, clean, self.commits, before, self.previous_snapshot(), branch=forest.current
            )
            self.state = clean
            self.pending = None
            self.revision += 1
            return "Reset the current branch."
        clean = self._reset_state()
        self.undo_log.push(
            self.state, clean, self.commits, self.editor_snapshot(), self.previous_snapshot()
        )
        if self.editor_state is not None:
            self._restart_editor(clean)
        self._pending_editor = None
        self.previous = None
        self.state = clean
        self.commits = []
        self.pending = None
        self.revision += 1
        return "Reset to the state that shipped with the mod."

    def _reset_state(self) -> State:
        return _copy(self.clean)

    def _write(self, kind: str, name: str) -> str:
        self._sequence += 1
        if self.saves_dir is None:
            self._memory_saves[name] = (
                _copy(self.state),
                list(self.commits),
                self._saved_editor_snapshot(),
                self.previous_snapshot(),
            )
            return name
        folder = self.saves_dir / self.mod_id
        folder.mkdir(parents=True, exist_ok=True)
        write_save(
            folder / f"{name}.ksave",
            self.state,
            mod=self.mod,
            layout=self.layout,
            updater_ref=self.pairing.updater_ref,
            commits=self.commits,
            kind=kind,
            name=name,
            sequence=self._sequence,
            editor_state=self._saved_editor_snapshot(),
            editor_kind=self.editor_state_kind,
            previous=self.previous_snapshot(),
            learned_with=self.composition,
            editor_packs=[p.ref.to_manifest() for p in self.editor_packs],
            options=self._saved_options(),
        )
        return name

    def _saved_options(self) -> dict[str, Any] | None:
        from .prediction_write import DEFAULT_LEARN, DEFAULT_PRIOR_SCALE, DEFAULT_WEIGHTS

        options: dict[str, Any] = {}
        if self.measure_samples != type(self).MEASURE_SAMPLES:
            options["measure_samples"] = self.measure_samples
        if getattr(self, "learn", DEFAULT_LEARN) != DEFAULT_LEARN:
            options["learn"] = self.learn
        weights = getattr(self, "prediction_weights", DEFAULT_WEIGHTS)
        if dict(weights) != DEFAULT_WEIGHTS:
            options["prediction_weights"] = dict(weights)
        prior = getattr(self, "prediction_prior", DEFAULT_PRIOR_SCALE)
        if prior != DEFAULT_PRIOR_SCALE:
            options["prediction_prior"] = prior
        return options or None

    def _saved_settings(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        from . import prediction_write as pw

        options = manifest.get("options") or {}
        try:
            return {
                "measure_samples": check_measure_samples(
                    options.get("measure_samples", type(self).MEASURE_SAMPLES)
                ),
                "learn": pw.check_learn(options.get("learn", pw.DEFAULT_LEARN)),
                "prediction_weights": pw.check_weights(options.get("prediction_weights") or {}),
                "prediction_prior": pw.check_prior_scale(
                    options.get("prediction_prior", pw.DEFAULT_PRIOR_SCALE)
                ),
            }
        except (KiLoRAError, ValueError) as exc:
            raise ModFormatError(f"the save has an invalid setting ({exc})") from None

    def _autosave(self) -> str:
        return self._write("autosave", f"autosave-{len(self.commits) % AUTOSAVES}")

    def save_point(self, name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-")[:64] or "save"
        self._write("save point", f"save-{safe}")
        return f"save-{safe}"

    def list_saves(self) -> list[str]:
        if self.saves_dir is None:
            return sorted(self._memory_saves)
        folder = self.saves_dir / self.mod_id
        return sorted(p.stem for p in folder.glob("*.ksave")) if folder.exists() else []

    def load(self, name: str) -> str:
        if self.saves_dir is None:
            if name not in self._memory_saves:
                raise KiLoRAError(f"no save named {name!r}")
            state, commits, editor, previous = self._memory_saves[name]
            editor_kind = self.editor_state_kind
            message = ""
            settings = None
        else:
            save = read_save(self.saves_dir / self.mod_id / f"{name}.ksave")
            message = self._check_save(save)
            settings = self._saved_settings(save.manifest)
            self.loaded_packs = list(save.manifest.get("editor_packs") or [])
            if self.loaded_packs or self.editor_packs:
                from .editor_packs import check_save_record

                for warning in check_save_record(self.loaded_packs, self.editor_packs):
                    message += " " + warning
            state = save.state
            commits = [Commit(**c) for c in save.manifest.get("commits", [])]
            editor = save.editor_state
            editor_kind = (save.manifest["editor_state"] or {}).get("kind")
            previous = save.previous
        if set(state) != set(self.layout.targets):
            raise ModFormatError(f"save {name!r} does not match layout {self.layout.id}")
        loaded = _copy(state)
        forest = getattr(self.editor_state, "forest", None)
        before = self.editor_snapshot()
        load_branch = None if forest is None else forest.current
        preserved = False
        if self.editor_state is not None:
            merge = getattr(self.editor_state, "load_forest", None)
            if (
                forest is not None
                and merge is not None
                and editor is not None
                and editor_kind == self.editor_state_kind
            ):
                has_history = any(entry.branch == load_branch for entry in self.undo_log._entries)
                preserved = merge(editor, allow_replace=not has_history)
                if preserved:
                    loaded = self.editor_state.expand_live(self.state)
                    commits = self.commits
                    previous = self.previous
                    message += " Saved branches were added; the current branch was kept."
            else:
                if forest is not None and (
                    len(forest.branches) > 1
                    or forest.state.events
                    or any(entry.branch == load_branch for entry in self.undo_log._entries)
                ):
                    raise KiLoRAError("the save has no fitting editor forest")
                message += self._load_editor_state(editor, editor_kind, loaded)
        else:
            self._pending_editor = (
                None if editor is None else {k: v.clone() for k, v in editor.items()},
                editor_kind,
            )
        self.undo_log.push(
            self.state,
            loaded,
            self.commits,
            before,
            self.previous_snapshot(),
            branch=load_branch,
            whole_editor=forest is not None,
            branch_after=(self.editor_state.forest.current if forest is not None else None),
        )
        self.previous = (
            {k: v.clone() for k, v in previous.items()} if self.keep_previous and previous else None
        )
        self.state = loaded
        if not preserved:
            self._restore_pin(editor)
        self.commits = list(commits)
        self.pending = None
        if settings is not None:
            self.measure_samples = settings["measure_samples"]
            self.learn = settings["learn"]
            self.prediction_weights = settings["prediction_weights"]
            self.prediction_prior = settings["prediction_prior"]
        self.deferred_guard = None
        self.revision += 1
        return f"Loaded {name}.{message}"

    def _restart_editor(self, state: Mapping[str, Any]) -> None:
        start = getattr(self.editor_state, "start_from", None)
        if start is not None:
            start(state)
        else:
            self.editor_state.reset()

    def _load_editor_state(
        self,
        tensors: Mapping[str, torch.Tensor] | None,
        kind: str | None,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        current = _fixed_part(self.editor_state.state_tensors())
        fixed = _fixed_part(tensors) if tensors is not None else None
        fits = (
            fixed is not None
            and kind == self.editor_state_kind
            and set(fixed) == set(current)
            and all(tuple(fixed[k].shape) == tuple(v.shape) for k, v in current.items())
        )
        if fits:
            self.editor_state.load_tensors({k: v.clone() for k, v in tensors.items()})
            return ""
        self._restart_editor(self.state if state is None else state)
        return " The save has no fitting editor state, so the editor starts from zero."

    def _check_save(self, save: SaveFile) -> str:
        m = save.manifest
        if m.get("layout") != self.layout.id:
            raise ModFormatError(
                f"this save uses layout {m.get('layout')}, the mod uses {self.layout.id}"
            )
        note = ""
        if m.get("mod", {}).get("digest") != self.mod.manifest.get("digest"):
            note += " This save was made with another version of the mod."
        if "learned_with" in m and _composition_key(m["learned_with"]) != _composition_key(
            self.composition
        ):
            then = ", ".join(x["id"] for x in m["learned_with"]) or "no other mods"
            note += f" This state was learned together with {then}; the mods loaded now differ."
            log.info(
                "save %s: learned with %s, loaded with %s",
                m.get("name"),
                m["learned_with"],
                self.composition,
            )
        return note

    def freeze(
        self, path: str | os.PathLike[str], new_id: str, *, name: str | None = None, **fields: Any
    ) -> dict[str, Any]:
        if new_id == self.mod_id:
            raise KiLoRAError("a frozen mod needs a new ID")
        manifest = self.mod.manifest
        keep = (
            "author",
            "license",
            "content_source",
            "base_model",
            "mod_family",
            "strength",
            "depth",
        )
        values = {k: manifest[k] for k in keep if k in manifest}
        values.update(fields)
        if self.composition:
            values.setdefault("learned_with", [dict(m) for m in self.composition])
        sets = []
        for i, s in enumerate(self.sets(use_pending=False)):
            extra = {k: v for k, v in s.extra.items() if k not in ("live", "live_id")}
            if i == self._live_index:
                extra["learned"] = True
            set_name = "learned" if i == self._live_index else s.name
            sets.append(dataclasses.replace(s, extra=extra, name=set_name))
        return save_mod(
            path,
            sets,
            self.mod.settings,
            id=new_id,
            name=name or new_id,
            version="1.0.0",
            based_on=f"{manifest['id']} {manifest['version']}",
            **values,
        )

    def export(
        self,
        path: str | os.PathLike[str],
        *,
        include_learned_state: bool = False,
        new_id: str | None = None,
    ) -> dict[str, Any]:

        manifest = dict(self.mod.manifest)
        keep = {
            k: v
            for k, v in manifest.items()
            if k
            not in (
                "format",
                "format_version",
                "requires_features",
                "adapter",
                "digest",
                "created_with",
            )
        }
        if not include_learned_state:
            return save_mod(path, self.mod.sets, self.mod.settings, **keep)
        if not new_id or new_id == self.mod_id:
            raise KiLoRAError("a mod with a learned state needs a new ID")
        keep["id"] = new_id
        keep["name"] = f"{manifest.get('name', self.mod_id)} (learned)"
        keep["based_on"] = f"{manifest['id']} {manifest['version']}"
        keep["learned_state"] = {"commits": len(self.commits), "note": LEARNED_STATE_WARNING}
        return save_mod(path, self.sets(use_pending=False), self.mod.settings, **keep)

    def probe_drift(
        self,
        probe: Callable[[list[AdapterSet]], torch.Tensor],
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ) -> tuple[float, bool]:
        current = probe(self.sets(use_pending=True)).float()
        if self.reference_probe is None:
            saved_state = self.state
            self.state = self.clean
            try:
                self.reference_probe = probe(self.sets(use_pending=False)).float()
            finally:
                self.state = saved_state
        drift = float(
            (current - self.reference_probe).norm() / self.reference_probe.norm().clamp(min=1e-12)
        )
        return drift, drift > threshold

    def set_probe_reference(self, probe: Callable[[list[AdapterSet]], torch.Tensor]) -> None:
        self.reference_probe = probe(self.sets(use_pending=False)).float()


def _copy(state: Mapping[str, Mapping[str, torch.Tensor]]) -> State:
    return {t: {k: v.detach().clone() for k, v in parts.items()} for t, parts in state.items()}


def open_session(
    path: str | os.PathLike[str],
    *,
    saves_dir: str | os.PathLike[str] | None = None,
) -> LiveSession:
    mod = load_mod(path)
    session = LiveSession(mod, default_pairing(mod), saves_dir)
    session.source_path = os.fspath(path)
    return session


SESSIONS: dict[str, LiveSession] = {}


def session_for(
    path: str | os.PathLike[str],
    *,
    saves_dir: str | os.PathLike[str] | None = None,
) -> LiveSession:
    from .modfile import read_manifest

    digest = read_manifest(path)["digest"]
    if digest not in SESSIONS:
        SESSIONS[digest] = open_session(path, saves_dir=saves_dir)
    return SESSIONS[digest]


def _fixed_part(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in tensors.items()
        if not k.startswith(
            ("region/", "prior/", "branch/", "node_coef/", "residual/", "pin/", "links/")
        )
    }
