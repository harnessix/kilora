# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from safetensors import safe_open

from .errors import KiLoRAError, ModFormatError
from .i18n import text
from .recording import GenerationRecord, PathRecorder

API_VERSION = "1.0"
ENTRY_POINT_GROUP = "kilora.editors"
PLUGINS_ENV = "KILORA_PLUGINS"
MANIFEST_KEY = "kilora.manifest"

State = Mapping[str, Mapping[str, torch.Tensor]]


@dataclass
class EditorSupport:
    rank_min: int = 1
    rank_max: int = 64
    columns_min: int = 1
    columns_max: int = 1024
    hooks: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "live_rank": [self.rank_min, self.rank_max],
            "columns": [self.columns_min, self.columns_max],
            "hooks": list(self.hooks),
        }

    @staticmethod
    def from_manifest(d: Mapping) -> EditorSupport:
        return EditorSupport(*d["live_rank"], *d["columns"], list(d["hooks"]))


def check_mod(
    support: EditorSupport, ranks: Mapping[str, int], columns: Mapping[str, int]
) -> list[str]:
    problems = []
    missing = [h for h in ranks if support.hooks and h not in support.hooks]
    if missing:
        problems.append(
            f"The editor does not support {len(missing)} of the mod's hook points, "
            f"for example {missing[0]}."
        )
    for h, r in ranks.items():
        if not support.rank_min <= r <= support.rank_max:
            problems.append(
                f"Hook point {h} has live rank {r}. The editor supports ranks "
                f"{support.rank_min} to {support.rank_max}."
            )
            break
    for h, c in columns.items():
        if not support.columns_min <= c <= support.columns_max:
            problems.append(
                f"Hook point {h} has {c} columns in A. The editor supports "
                f"{support.columns_min} to {support.columns_max}."
            )
            break
    return problems


@runtime_checkable
class EditorState(Protocol):
    def state_tensors(self) -> dict[str, torch.Tensor]: ...

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None: ...

    def reset(self) -> None: ...


class Editor:
    api_version: str = API_VERSION
    needs_path: bool = False
    supports: EditorSupport | None = None
    options: dict[str, Any] = {}

    def configure(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)

    def check(self, session: Any) -> None:
        pass

    def recorder(self, session: Any) -> PathRecorder | None:
        return None

    def next_state(self, state: State, record: GenerationRecord) -> dict[str, dict] | None:
        raise NotImplementedError

    def guard_trial(self, state: State, new: State) -> Any:
        return None


LIVE_OPTIONS = (
    "diff",
    "time_bins",
    "measure",
    "combos",
    "path_merge",
    "path_angle",
    "cross_layer",
    "keep_merge_pairs",
    "merge_residual",
    "term_budget",
    "explore",
)


@dataclass
class EditorFileInfo:
    label: str
    targets: list[str]
    facts: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)


GUARD_GAP_DROP = 0.2
GUARD_GAP_NOISE_K = 1.0
GUARD_NOISE_WINDOW = 8
GUARD_SPREAD = (0.95, 1.05)
GUARD_ENV = "KILORA_LIVE_GUARD"


def guard_enabled(session: Any) -> bool:
    on = getattr(session, "guard", None)
    if on is None:
        return os.environ.get(GUARD_ENV, "1").strip().lower() not in ("0", "off", "false", "no")
    return bool(on)


def gap_noise(session: Any, diff: Any) -> float:
    diffs = getattr(session, "guard_gap_diffs", None)
    if not isinstance(diffs, list):
        diffs = []
        try:
            session.guard_gap_diffs = diffs
        except AttributeError:
            return 0.0
    if diff is not None and math.isfinite(float(diff)):
        diffs.append(float(diff))
        del diffs[:-GUARD_NOISE_WINDOW]
    if not diffs:
        return 0.0
    return math.sqrt(sum(d * d for d in diffs) / len(diffs) / 2)


def editor_ir_parts(session: Any) -> dict:
    mod = getattr(session, "mod", None)
    if mod is None:
        return {}
    live_name = ((getattr(mod, "manifest", {}) or {}).get("live") or {}).get("live_set")
    live_set = next((item for item in getattr(mod, "sets", ()) if item.name == live_name), None)
    return getattr(live_set, "ir_parts", {}) if live_set is not None else {}


def guard_check(rule: Any, session: Any, record: GenerationRecord, new: State) -> tuple[bool, str]:
    override = editor_ir_parts(session).get("guard")
    if not guard_enabled(session):
        if override is not None:
            raise NotImplementedError("guard override cannot run while guard is disabled")
        return False, text("msg.guard_off_user")
    if record.guard_score is None:
        if override is not None:
            raise NotImplementedError("guard override needs guard measurements")
        return False, text("msg.guard_off_measure")
    trial = getattr(rule, "guard_trial", lambda s, n: None)(session.state, new)
    if trial is None:
        if override is not None:
            raise NotImplementedError("guard override needs an editor guard trial")
        return False, text("msg.guard_off_editor")
    before, after = record.guard_score(None), record.guard_score(trial)
    return guard_verdict(session, before, after, override)


def guard_verdict(session: Any, before: dict, after: dict, override: Any = None):
    gap0, gap1 = float(before["gap"]), float(after["gap"])
    diff = None
    if before.get("gap_alt") is not None and after.get("gap_alt") is not None:
        change_alt = float(after["gap_alt"]) - float(before["gap_alt"])
        diff = change_alt - (gap1 - gap0)
        gap0 = (gap0 + float(before["gap_alt"])) / 2
        gap1 = (gap1 + float(after["gap_alt"])) / 2
    noise = gap_noise(session, diff) / math.sqrt(2)
    s0, s1 = float(before["spread"]), float(after["spread"])
    ratio = s1 / s0 if s0 > 0 else 1.0
    last = {"gap0": gap0, "gap1": gap1, "gap_noise": noise, "spread_ratio": ratio}
    with contextlib.suppress(AttributeError):
        session.guard_last = last
    from .compute import _IR_INTERPRET_ALL

    if _IR_INTERPRET_ALL.get() or override is not None:
        from .ir.dispatch import active_overrides, interpreted_part
        from .ir.driver import _run

        values = {
            "gap_before": gap0,
            "gap_after": gap1,
            "spread_before": s0,
            "spread_after": s1,
            "noise": noise,
            "gap_drop": GUARD_GAP_DROP,
            "gap_noise_k": GUARD_GAP_NOISE_K,
            "spread_low": GUARD_SPREAD[0],
            "spread_high": GUARD_SPREAD[1],
        }
        inputs = {name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()}
        if override is not None:
            from types import SimpleNamespace

            interpreted_part([SimpleNamespace(ir_parts={"guard": override})])
        with active_overrides({"guard": override} if override is not None else {}):
            refused = bool(_run("guard", "default", inputs)["refused"])
        if not refused:
            last["refused"] = None
            return False, text("msg.guard_passed")
        if override is not None:
            last["refused"] = "guard"
            return True, text("msg.guard_follow", before=f"{gap0:.4g}", after=f"{gap1:.4g}")
    if not gap1 >= gap0 - max(GUARD_GAP_DROP * abs(gap0), GUARD_GAP_NOISE_K * noise):
        last["refused"] = "gap"
        return True, text("msg.guard_follow", before=f"{gap0:.4g}", after=f"{gap1:.4g}")
    if not GUARD_SPREAD[0] <= ratio <= GUARD_SPREAD[1]:
        last["refused"] = "spread"
        return True, text("msg.guard_spread", ratio=f"{ratio:.3g}")
    last["refused"] = None
    return False, text("msg.guard_passed")


def guard_calibration(
    rule: Any,
    session: Any,
    record: GenerationRecord,
    ks: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0),
    part: str = "up",
) -> dict[str, Any]:
    if record.guard_score is None:
        raise KiLoRAError("the guard calibration needs a measured record (the CLIP on Record)")
    base = record.guard_score(None)
    gap0, s0 = float(base["gap"]), float(base["spread"])
    rows = []
    for k in ks:
        new = {t: {**p, part: p[part] * float(k)} for t, p in session.state.items()}
        trial = getattr(rule, "guard_trial", lambda s, n: None)(session.state, new)
        if trial is None:
            raise KiLoRAError("the editor gives the guard no trial")
        sc = record.guard_score(trial)
        gap, spread = float(sc["gap"]), float(sc["spread"])
        rows.append(
            {
                "k": float(k),
                "gap": gap,
                "spread": spread,
                "gap_ratio": gap / gap0 if gap0 else math.nan,
                "spread_ratio": spread / s0 if s0 else math.nan,
            }
        )
    return {"base": {k: float(v) for k, v in base.items()}, "rows": rows}


@dataclass(frozen=True)
class EditorPlugin:
    kind: str
    load: Callable[[str], Editor]
    inspect: Callable[[str], EditorFileInfo] | None = None
    title: str = ""
    api_version: str = API_VERSION


_PLUGINS: dict[str, EditorPlugin] = {}
_SOURCES: dict[str, str] = {}
_DISCOVERED = False
DEFAULT_PLUGINS: tuple[str, ...] = ()


def _major(version: str) -> str:
    return str(version).split(".")[0]


def register_editor(plugin: EditorPlugin, source: str = "") -> None:
    if _major(plugin.api_version) != _major(API_VERSION):
        raise KiLoRAError(
            f"The editor plugin for '{plugin.kind}' needs editor API {plugin.api_version}, "
            f"but this KiLoRA has API {API_VERSION}."
        )
    _PLUGINS[plugin.kind] = plugin
    _SOURCES[plugin.kind] = source


def unregister_editor(kind: str) -> None:
    _PLUGINS.pop(kind, None)
    _SOURCES.pop(kind, None)


def register_rule(kind: str, factory: Callable[[str], Editor]) -> None:
    register_editor(EditorPlugin(kind, factory))


def _register_object(obj: Any, source: str) -> None:
    if callable(obj) and not isinstance(obj, EditorPlugin):
        obj = obj()
    items = obj if isinstance(obj, (list, tuple)) else [obj]
    for item in items:
        if not isinstance(item, EditorPlugin):
            raise KiLoRAError(f"{source} is not an EditorPlugin")
        register_editor(item, source)


def _entry_points(group: str) -> list[Any]:
    from importlib.metadata import entry_points

    try:
        return list(entry_points(group=group))
    except TypeError:
        return list(entry_points().get(group, []))


def import_plugin_modules(
    names: list[str] | tuple[str, ...], quiet: bool
) -> list[tuple[Any, str | None]]:
    loaded = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        module_name, _, attr = name.partition(":")
        try:
            loaded.append((importlib.import_module(module_name.strip()), attr.strip() or None))
        except ImportError as exc:
            if not quiet:
                warnings.warn(f"KiLoRA plugin module {name} did not load: {exc}", stacklevel=2)
    return loaded


def module_plugins(
    loaded: list[tuple[Any, str | None]], names: tuple[str, ...], cls: type
) -> list[tuple[Any, str]]:
    found = []
    for module, attr in loaded:
        label = module.__name__ + (f":{attr}" if attr else "")
        if attr is not None and not hasattr(module, attr):
            warnings.warn(f"KiLoRA plugin module {label}: no attribute {attr}", stacklevel=2)
        for a in [attr] if attr else names:
            obj = getattr(module, a, None)
            if attr and callable(obj) and not isinstance(obj, (cls, type)):
                obj = obj()
            items = obj if isinstance(obj, (list, tuple)) else [obj]
            found += [(item, label) for item in items if isinstance(item, cls)]
    return found


def discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import_plugin_modules(DEFAULT_PLUGINS, quiet=True)
    for name in DEFAULT_PLUGINS:
        plugin = getattr(sys.modules.get(name), "PLUGIN", None)
        if isinstance(plugin, EditorPlugin) and plugin.kind not in _PLUGINS:
            register_editor(plugin, name)
    import_plugin_modules(("kilora._site",), quiet=True)
    for ep in _entry_points(ENTRY_POINT_GROUP):
        try:
            _register_object(ep.load(), f"entry point {ep.name}")
        except Exception as exc:
            warnings.warn(f"KiLoRA editor plugin {ep.name} did not load: {exc}", stacklevel=2)
    loaded = import_plugin_modules(tuple(os.environ.get(PLUGINS_ENV, "").split(",")), quiet=False)
    for plugin, label in module_plugins(loaded, ("PLUGIN", "PLUGINS"), EditorPlugin):
        if _PLUGINS.get(plugin.kind) is not plugin:
            register_editor(plugin, f"{PLUGINS_ENV} {label}")


def available_editors() -> dict[str, EditorPlugin]:
    discover()
    return dict(_PLUGINS)


def plugin_for_kind(kind: Any) -> EditorPlugin | None:
    discover()
    return _PLUGINS.get(kind) if isinstance(kind, str) else None


def read_editor_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    label = os.path.basename(os.fspath(path))
    try:
        with safe_open(os.fspath(path), framework="pt") as f:
            return json.loads((f.metadata() or {}).get(MANIFEST_KEY) or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ModFormatError(f"{label}: cannot read the editor ({exc})") from None


def editor_for(path: str | os.PathLike[str]) -> Editor:
    label = os.path.basename(os.fspath(path))
    manifest = read_editor_manifest(path)
    plugin = plugin_for_kind(manifest.get("kind"))
    if plugin is None:
        raise KiLoRAError(text("err.not_editor", name=label))
    return plugin.load(os.fspath(path))


rule_for = editor_for
WriteRule = Editor


def deferred_guard_check(rule: Any, session: Any, record: GenerationRecord) -> str:
    pending = getattr(session, "deferred_guard", None)
    if pending is None:
        return ""
    if not guard_enabled(session):
        session.deferred_guard = None
        return text("msg.prediction_guard_skipped", why=text("msg.guard_off_user"))
    if record.guard_score is None:
        return ""
    session.deferred_guard = None
    if len(session.commits) != pending.commits or session.pending is not None:
        return text("msg.prediction_guard_skipped", why=text("msg.prediction_guard_stale"))
    trial = getattr(rule, "guard_trial", lambda s, n: None)(session.state, pending.before)
    if trial is None:
        return text("msg.prediction_guard_skipped", why=text("msg.guard_off_editor"))
    without, with_write = record.guard_score(trial), record.guard_score(None)
    refused, note = guard_verdict(session, without, with_write)
    if not refused:
        return text("msg.prediction_guard_passed")
    session.undo()
    on_refusal = getattr(rule, "guard_refused", None)
    if on_refusal is not None:
        on_refusal()
    return text("msg.prediction_rolled_back", why=note)


def _write_memory_device(record: GenerationRecord) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    device = torch.device(getattr(record.measure, "device", "cuda"))
    return device if device.type == "cuda" else None


def _finish_write_memory(device: torch.device) -> str:
    note = ""
    try:
        with contextlib.suppress(Exception):
            allocated = torch.cuda.memory_allocated(device) / 2**30
            reserved = torch.cuda.memory_reserved(device) / 2**30
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            note = (
                f"\nGPU memory: allocated {allocated:.1f} GiB, reserved {reserved:.1f} GiB, "
                f"peak {peak:.1f} GiB (freed after the write)"
            )
    finally:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            management = importlib.import_module("comfy.model_management")
            management.soft_empty_cache()
    return note


def commit_write(session: Any, rule: Editor, record: GenerationRecord, *, records=None) -> str:

    device = _write_memory_device(record)
    if device is None:
        return _commit_write(session, rule, record, records=records)
    try:
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device)
        result = _commit_write(session, rule, record, records=records)
    finally:
        memory_note = _finish_write_memory(device)
    return result + memory_note


def _commit_write(session: Any, rule: Editor, record: GenerationRecord, *, records=None) -> str:
    from .live import LearnResult

    if not session.learning:
        return text("msg.learning_off")
    if rule.needs_path and (record.stats is None or not record.stats.c_kk):
        return text("msg.nothing_recorded")
    before = session.editor_snapshot() if hasattr(session, "editor_snapshot") else None
    old_write_branch = getattr(session, "pinned_write_branch", None)
    if hasattr(session, "prepare_pinned_write"):
        session.prepare_pinned_write()

    def restore() -> None:
        if before is not None:
            session.editor_state.load_tensors(before)
        if hasattr(session, "pinned_write_branch"):
            session.pinned_write_branch = old_write_branch

    deferred = deferred_guard_check(rule, session, record)
    if deferred:
        before = session.editor_snapshot() if hasattr(session, "editor_snapshot") else None
    begin_write = getattr(rule, "begin_write", None)
    if begin_write is not None:
        begin_write()
    with torch.no_grad():
        new = (
            rule.next_combined_state(session.state, records)
            if records is not None
            else rule.next_state(session.state, record)
        )
    deferred = f"{deferred} " if deferred else ""
    provisional = bool(getattr(rule, "provisional", False))
    if new is None:
        restore()
        return deferred + text("msg.no_write")
    if set(new) != set(session.state):
        restore()
        raise KiLoRAError("the editor returned a state for other hook points")
    from .host_copy import all_finite, host_copy_map

    if not all_finite(v for parts in new.values() for v in parts.values()):
        restore()
        return deferred + text("msg.invalid_write")
    guard_note = text("msg.prediction_guard_deferred") if provisional else ""
    for item in () if provisional else records if records is not None else [record]:
        refused, guard_note = guard_check(rule, session, item, new)
        if refused:
            restore()
            on_refusal = getattr(rule, "guard_refused", None)
            if on_refusal is not None:
                on_refusal()
            return deferred + text("msg.guard_refused", why=guard_note)
    from .live_queue import yield_point

    yield_point()
    flat = host_copy_map(
        {(t, p): v for t, parts in new.items() for p, v in parts.items()},
        lambda v: v.cpu().clone(),
    )
    session.pending = {t: {p: flat[(t, p)] for p in parts} for t, parts in new.items()}
    images = sum(r.images for r in records) if records is not None else record.images
    session._pending_info = LearnResult(images, 1.0, 0.0, False, False, "live write")
    session.revision += 1
    previous_before = "current"
    if getattr(session, "keep_previous", False):
        previous_before = session.previous_snapshot()
        session.set_previous(record.latent, record.conditioning)
    state_before = session.state
    result = f"{session.commit(editor_before=before, previous_before=previous_before)} {guard_note}"
    if provisional and hasattr(session, "deferred_guard"):
        from .prediction_write import DeferredGuard

        session.deferred_guard = DeferredGuard(
            {t: dict(parts) for t, parts in state_before.items()},
            len(session.commits),
            session.revision,
        )
    result = deferred + result
    finish_write = getattr(rule, "finish_write", None)
    if finish_write is not None:
        with torch.no_grad():
            result += finish_write(session, records if records is not None else [record])
    return result
