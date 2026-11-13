# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import asdict

from .adapter import Settings
from .comfyui.settings import resolve_settings
from .modfile import save_mod
from .references import HashIndex
from .sections import Section


def save_recipe(
    entries,
    settings,
    path,
    name,
    license="NOASSERTION",
    *,
    route_freeze=0.0,
    axis_alpha="off",
    library=False,
    hash_index=None,
):
    sections, lines = [], []
    index = hash_index if hash_index is not None else HashIndex()
    for i, entry in enumerate(entries):
        if entry.kind not in ("lora", "mod"):
            raise ValueError(f"unknown chain entry kind {entry.kind!r}")
        digest = index.hash(entry.path, force=hash_index is None, write=False)
        meta = {
            "sha256": digest,
            "size": os.path.getsize(entry.path),
            "filename": os.path.basename(entry.path),
            "form": entry.kind,
            "transforms": {
                "strength": entry.strength,
                "axis": entry.axis,
                "roles": entry.roles,
                "region": None,
                "timeline": [asdict(row) for row in entry.timeline],
                "enabled": entry.enabled,
                "start": entry.start,
                "end": entry.end,
            },
        }
        sections.append(Section("weight-ref", f"entry-{i + 1}", {}, meta))
        lines.append(f"{meta['filename']}: sha256 {digest}")
        if entry.region is not None:
            lines.append(f"{meta['filename']}: region was not saved.")
        if entry.prior != 1.0 or entry.panels is not None or entry.region_caption is not None:
            raise ValueError(f"{meta['filename']}: this entry has settings recipes cannot save")
    index._write()
    values = resolve_settings(settings)
    schedule = values["depth_schedule"]
    if schedule is not None and not isinstance(schedule, str):
        if getattr(schedule, "mode", None) != "expr":
            raise ValueError("depth_schedule must be an expression to save a recipe")
        values["depth_schedule"] = schedule.source
    values.update(
        output_gate="mod",
        gate_kappa=0.0,
        path_width=0,
        path_select="mod",
        route_freeze=route_freeze,
        axis_alpha=axis_alpha,
        library=bool(library),
    )
    sections.append(Section("logic", "settings", {}, {"version": 1, "settings": values}))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_mod(path, [], Settings(), sections=sections, name=name, license=license)
    return lines
