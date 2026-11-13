# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import dataclasses
import os
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..comfyui.patching import ModEntry, joint_choice
from ..lora_import import _read_file


class ByteLRU:
    def __init__(self, budget: int):
        if budget < 0:
            raise ValueError("cache budget must be non-negative")
        self.budget = budget
        self.bytes = 0
        self._entries: OrderedDict[Any, tuple[Any, int]] = OrderedDict()

    def get(self, key: Any) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: Any, value: Any, size: int | None = None) -> None:
        size = tensor_bytes(value) if size is None else size
        if size < 0:
            raise ValueError("entry size must be non-negative")
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        if size > self.budget or self.budget == 0:
            return
        while self.bytes + size > self.budget:
            _, (_, removed) = self._entries.popitem(last=False)
            self.bytes -= removed
        self._entries[key] = (value, size)
        self.bytes += size


def tensor_bytes(value: Any) -> int:
    seen_objects: set[int] = set()
    seen_storage: set[tuple] = set()

    def visit(item: Any) -> int:
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return storage.nbytes()
        ident = id(item)
        if ident in seen_objects:
            return 0
        seen_objects.add(ident)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, f.name)) for f in dataclasses.fields(item))
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(v) for v in item)
        return 0

    return visit(value)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            object.__setattr__(result, field.name, _to_device(getattr(value, field.name), device))
        return result
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _pin_memory(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.pin_memory() if value.device.type == "cpu" and not value.is_pinned() else value
    if isinstance(value, dict):
        return {k: _pin_memory(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_pin_memory(v) for v in value)
    return value


@dataclass(frozen=True)
class SelectedFile:
    path: str | os.PathLike[str]
    sha256: str
    strength: float = 0.8


@dataclass
class LoadReport:
    read_seconds: float = 0.0
    import_seconds: float = 0.0
    key_setup_seconds: float = 0.0
    ram_hits: int = 0
    routed_hit: bool = False
    prefetch_mode: str = "off"
    prefetch_window: int = 1
    prefetch_wait_seconds: float = 0.0
    _prefetch_collect: Any = dataclasses.field(default=None, repr=False, compare=False)

    def __str__(self) -> str:
        if self._prefetch_collect is not None:
            self._prefetch_collect()
        return (
            f"Read {self.read_seconds:.3f}s; import {self.import_seconds:.3f}s; "
            f"key setup {self.key_setup_seconds:.3f}s; RAM hits {self.ram_hits}; "
            f"routed-set cache {'hit' if self.routed_hit else 'miss'}. "
            f"Prefetch {self.prefetch_mode}; window {self.prefetch_window}; "
            f"wait {self.prefetch_wait_seconds:.6f}s."
        )


class LibraryLoader:
    def __init__(
        self,
        folders: Iterable[str | os.PathLike[str]],
        *,
        library_ram_gib: float = 8,
        library_vram_gib: float = 2,
        device: str | torch.device = "cpu",
        library_prefetch: str = "auto",
        prefetch_window: int = 1,
    ):
        if library_prefetch not in ("auto", "off") or prefetch_window not in (1, 2):
            raise ValueError("invalid library prefetch mode or window")
        self.library_prefetch = library_prefetch
        self.prefetch_window = prefetch_window
        self.folders = tuple(Path(p).resolve() for p in folders)
        self.device = torch.device(device)
        self.ram = ByteLRU(int(library_ram_gib * 1024**3))
        self.routed = ByteLRU(int(library_vram_gib * 1024**3))
        self._file_stamps: dict[tuple[str, str], tuple[int, int, int, int]] = {}

    def _path(self, selected: SelectedFile) -> Path:
        path = Path(selected.path).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.folders):
            raise ValueError(f"Selected file is outside the library folders: {path}")
        return path

    @staticmethod
    def _file_stamp(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def load(
        self, selected: Iterable[SelectedFile], patcher: Any, binding: Any, shapes_model: Any
    ) -> tuple[Any, LoadReport]:
        items = sorted(selected, key=lambda s: (s.sha256, float(s.strength)))
        if not items:
            raise ValueError("load needs at least one selected file")
        paths = [self._path(item) for item in items]
        if len({(i.sha256, float(i.strength)) for i in items}) != len(items):
            raise ValueError("duplicate selected file and strength")
        report = LoadReport(
            prefetch_mode=self.library_prefetch, prefetch_window=self.prefetch_window
        )
        file_stamps = [self._file_stamp(p) for p in paths]
        known = all(
            self._file_stamps.get((str(p), i.sha256)) == f
            for i, p, f in zip(items, paths, file_stamps, strict=True)
        )
        shapes = tuple(
            (n, m.in_features, m.out_features)
            for n, m in shapes_model.named_modules()
            if isinstance(m, torch.nn.Linear)
        )
        model = patcher.model.diffusion_model
        configs = []
        for path in paths:
            config = path.parent / "adapter_config.json"
            if config.is_file() and not any(
                config.resolve().is_relative_to(root) for root in self.folders
            ):
                raise ValueError(f"Adapter config is outside the library folders: {config}")
            configs.append((str(path), self._file_stamp(config) if config.is_file() else None))
        key = (
            id(model),
            binding.family,
            shapes,
            str(self.device),
            tuple((i.sha256, float(i.strength)) for i in items),
            tuple(configs),
        )
        cached = self.routed.get(key) if known else None
        if cached is not None and cached[0]() is model:
            report.routed_hit = True
            for adapter_set in cached[1].adapter_sets:
                config = adapter_set.extra.get("_library_transfer")
                if config:
                    config["report"] = report
            return cached[1], report

        by_path = {
            str(p): (item, fp) for item, p, fp in zip(items, paths, file_stamps, strict=True)
        }

        def read_file(path: str):
            item, file_stamp = by_path[path]
            ram_key = (path, item.sha256, file_stamp)
            cached = self.ram.get(ram_key)
            if cached is not None:
                report.ram_hits += 1
                return cached
            start = time.perf_counter()
            result = _read_file(path)
            report.read_seconds += time.perf_counter() - start
            if result[2]["digest"] != "sha256-" + item.sha256:
                raise ValueError(f"Selected file changed since sync: {path}")
            if self._file_stamp(Path(path)) != file_stamp:
                raise ValueError(f"Selected file changed while reading: {path}")
            if torch.cuda.is_available() and tensor_bytes(result) <= self.ram.budget:
                result = _pin_memory(result)
            self.ram.put(ram_key, result)
            self._file_stamps[(path, item.sha256)] = file_stamp
            return result

        def key_time(seconds: float) -> None:
            report.key_setup_seconds += seconds

        entries = [
            ModEntry("lora", str(p), strength=i.strength, axis="creative")
            for i, p in zip(items, paths, strict=True)
        ]
        start = time.perf_counter()
        choice, _ = joint_choice(
            patcher, entries, binding, shapes_model, read_file=read_file, key_time=key_time
        )
        choice.sections = [section for section in choice.sections if section.kind != "text-encoder"]
        elapsed = time.perf_counter() - start
        report.import_seconds = max(0.0, elapsed - report.read_seconds - report.key_setup_seconds)
        if self.device.type != "cpu" and tensor_bytes(choice) <= self.routed.budget:
            choice = _to_device(choice, self.device)
        if torch.cuda.is_available() and (
            self.device.type == "cpu" or tensor_bytes(choice) > self.routed.budget
        ):
            for adapter_set in choice.adapter_sets:
                adapter_set.extra = dict(
                    adapter_set.extra,
                    _library_transfer={
                        "mode": self.library_prefetch,
                        "window": self.prefetch_window,
                        "report": report,
                    },
                )
        self.routed.put(key, (weakref.ref(model), choice), tensor_bytes(choice))
        return choice, report
