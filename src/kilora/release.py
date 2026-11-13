# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .card import card_fields, card_markdown, embed_card, model_card
from .check import check_file
from .errors import KiLoRAError
from .modfile import read_manifest
from .naming import FAMILY_SHORT, MOD_SUFFIX


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_mod(
    mod_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    sample_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:

    report = check_file(mod_path)
    if not report.ok:
        raise KiLoRAError("the mod does not pass kilora check: " + " ".join(report.problems))
    manifest = read_manifest(mod_path)
    mod_id = manifest["id"]
    family = (manifest.get("base_model") or {}).get("family", "unknown")
    folder = Path(out_dir) / mod_id
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{mod_id}.{FAMILY_SHORT.get(family, family)}{MOD_SUFFIX}"
    target = folder / name
    shutil.copyfile(mod_path, target)
    embed_card(target)
    fields = card_fields(read_manifest(target))
    (folder / "CARD.md").write_text(card_markdown(fields), encoding="utf-8")
    (folder / "README.md").write_text(model_card(target, name), encoding="utf-8")
    open_points = list(report.warnings)
    if sample_dir is not None:
        sample = Path(sample_dir)
        for file in ("grid.png", "render.json"):
            if not (sample / file).exists():
                raise KiLoRAError(f"{sample} has no {file}; run kilora render first")
            shutil.copyfile(sample / file, folder / file)
        rendered = json.loads((sample / "render.json").read_text(encoding="utf-8"))
        if rendered.get("mod", {}).get("digest") != manifest.get("digest"):
            open_points.append("The sample grid was rendered from another version of the mod.")
    else:
        open_points.append("No sample grid: run kilora render and package again.")
    if fields["license"] in ("", "NOASSERTION"):
        open_points.append("The mod has no license. Set one before a release.")
    if not fields["trigger"]:
        open_points.append("The mod has no trigger words.")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "package.json")
    package = {
        "format": "kilora-release-folder",
        "format_version": "1.0",
        "status": "prepared, not published",
        "mod": {
            "id": mod_id,
            "version": manifest.get("version"),
            "digest": read_manifest(target).get("digest"),
            "file": name,
        },
        "card": fields,
        "check": {"ok": report.ok, "warnings": report.warnings},
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files],
        "open_points": open_points,
    }
    (folder / "package.json").write_text(
        json.dumps(package, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package
