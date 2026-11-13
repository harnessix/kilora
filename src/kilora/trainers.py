# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .editors import PLUGINS_ENV, _entry_points, _major, import_plugin_modules, module_plugins
from .errors import KiLoRAError

API_VERSION = "1.0"
ENTRY_POINT_GROUP = "kilora.trainers"
KINDS = ("mod", "editor")


@dataclass(frozen=True)
class TrainerPlugin:
    name: str
    kind: str
    run: Callable[[list[str]], int | None] | None = None
    start: Callable[..., Any] | None = None
    extend: Callable[..., Any] | None = None
    add_arguments: Callable[[Any], None] | None = None
    prepare: Callable[..., Any] | None = None
    commands: Mapping[str, Callable[[list[str]], int | None]] = field(default_factory=dict)
    title: str = ""
    api_version: str = API_VERSION


_PLUGINS: dict[tuple[str, str], TrainerPlugin] = {}
_DISCOVERED = False


def register_trainer(plugin: TrainerPlugin, source: str = "") -> None:
    if plugin.kind not in KINDS:
        raise KiLoRAError(f"Trainer kind must be one of {', '.join(KINDS)}, not {plugin.kind}.")
    if _major(plugin.api_version) != _major(API_VERSION):
        raise KiLoRAError(
            f"The trainer plugin '{plugin.name}' needs trainer API {plugin.api_version}, "
            f"but this KiLoRA has API {API_VERSION}."
        )
    _PLUGINS[(plugin.kind, plugin.name)] = plugin


def _static_run(argv: list[str]) -> int:
    from .train import main

    return main(["--static", *argv])


def discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    register_trainer(TrainerPlugin("static", "mod", run=_static_run, title="Static mods"))
    import_plugin_modules(("kilora._site",), quiet=True)
    for ep in _entry_points(ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            obj = obj() if callable(obj) and not isinstance(obj, TrainerPlugin) else obj
            for item in obj if isinstance(obj, (list, tuple)) else [obj]:
                register_trainer(item, f"entry point {ep.name}")
        except Exception as exc:
            warnings.warn(f"KiLoRA trainer plugin {ep.name} did not load: {exc}", stacklevel=2)
    loaded = import_plugin_modules(tuple(os.environ.get(PLUGINS_ENV, "").split(",")), quiet=False)
    for plugin, label in module_plugins(loaded, ("TRAINER", "TRAINERS"), TrainerPlugin):
        if _PLUGINS.get((plugin.kind, plugin.name)) is not plugin:
            register_trainer(plugin, f"{PLUGINS_ENV} {label}")


def available_trainers(kind: str | None = None) -> dict[str, TrainerPlugin]:
    discover()
    return {n: p for (k, n), p in _PLUGINS.items() if kind is None or k == kind}


def get_trainer(name: str, kind: str = "mod") -> TrainerPlugin | None:
    discover()
    return _PLUGINS.get((kind, name))


def live_trainer() -> TrainerPlugin | None:
    plugin = get_trainer("live", "mod")
    return plugin if plugin is not None and plugin.start is not None else None


def editor_main(argv: Sequence[str] | None = None) -> int:

    argv = list(sys.argv[1:] if argv is None else argv)
    found = available_trainers("editor")
    if argv[:1] == ["--list"] or not found:
        if not found:
            print(
                "No editor trainer is installed. Editor trainers ship as plugins.",
                file=sys.stderr,
            )
            return 1
        for name, p in sorted(found.items()):
            print(f"{name}\t{p.title}")
        return 0
    if argv and argv[0] in found:
        plugin, rest = found[argv[0]], argv[1:]
    elif len(found) == 1:
        plugin, rest = next(iter(found.values())), argv
    else:
        print(f"Choose an editor trainer: {', '.join(sorted(found))}", file=sys.stderr)
        return 2
    return int(plugin.run(rest) or 0)
