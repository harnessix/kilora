# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MIN_SIDE = 512
MAX_ASPECT = 2.0
NEAR_DUPLICATE_BITS = 6


@dataclass
class ImageInfo:
    file: str
    width: int = 0
    height: int = 0
    format: str = ""
    caption: str = ""
    sha256: str = ""
    phash: str = ""


@dataclass
class DataReport:
    folder: str
    trigger: str | None = None
    images: list[ImageInfo] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def markdown(self) -> str:
        name = Path(self.folder).name
        lines = [
            f"# Data check: {name}",
            "",
            f"Result: {'usable' if self.ok else 'not usable'}. "
            f"{len(self.images)} images, {len(self.problems)} problems, "
            f"{len(self.warnings)} warnings.",
            "",
            f"Trigger: {self.trigger or 'not given'}",
        ]
        if self.images:
            sizes = sorted({f"{i.width}x{i.height}" for i in self.images})
            formats = sorted({i.format for i in self.images})
            lines.append(f"Sizes: {', '.join(sizes)}. Formats: {', '.join(formats)}.")
        for title, items in (("Problems", self.problems), ("Warnings", self.warnings)):
            lines += ["", f"## {title}", ""]
            lines += [f"- {p}" for p in items] or ["None."]
        return "\n".join(lines) + "\n"


def image_folder(folder: str | os.PathLike[str]) -> Path:
    base = Path(folder)
    return base / "images" if (base / "images").is_dir() else base


def read_manifest(folder: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(folder) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def phash(img: Any) -> int:
    from PIL import Image

    gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    n = np.arange(32)
    basis = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 64)
    low = (basis @ pixels @ basis.T)[:8, :8].reshape(-1)
    bits = low[1:] > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def check_folder(
    folder: str | os.PathLike[str],
    *,
    trigger: str | None = None,
    min_side: int = MIN_SIDE,
    near_bits: int = NEAR_DUPLICATE_BITS,
) -> DataReport:
    from PIL import Image, UnidentifiedImageError

    base = Path(folder)
    manifest = read_manifest(base)
    report = DataReport(str(base), trigger or manifest.get("trigger"))
    images_dir = image_folder(base)
    if not images_dir.is_dir():
        report.problems.append("The folder does not exist.")
        return report
    files = sorted(p for p in images_dir.iterdir() if p.is_file())
    pictures = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    others = [
        p
        for p in files
        if p.suffix.lower() not in (*IMAGE_EXTENSIONS, ".txt") and not p.name.startswith(".")
    ]
    for p in others:
        report.warnings.append(f"{p.name}: not an image or caption file. Training skips it.")
    if not pictures:
        report.problems.append("The folder has no images (.png, .jpg, .jpeg, .webp).")
        return report
    stems = {p.stem for p in pictures}
    for p in files:
        if p.suffix == ".txt" and p.stem not in stems:
            report.warnings.append(f"{p.name}: a caption without an image.")
    by_hash: dict[str, str] = {}
    for path in pictures:
        info = ImageInfo(path.name)
        data = path.read_bytes()
        info.sha256 = hashlib.sha256(data).hexdigest()
        try:
            with Image.open(path) as img:
                img.load()
                info.width, info.height, info.format = img.width, img.height, img.format or ""
                info.phash = f"{phash(img):016x}"
        except (UnidentifiedImageError, OSError) as exc:
            report.problems.append(f"{path.name}: cannot read the image ({exc}).")
            report.images.append(info)
            continue
        if min(info.width, info.height) < min_side:
            report.problems.append(
                f"{path.name}: {info.width}x{info.height} is smaller than {min_side} pixels "
                "on the short side."
            )
        if max(info.width, info.height) > MAX_ASPECT * min(info.width, info.height):
            report.warnings.append(
                f"{path.name}: {info.width}x{info.height}. The square crop loses much of it."
            )
        caption_file = path.with_suffix(".txt")
        if not caption_file.exists():
            report.problems.append(f"{path.name}: no caption file ({caption_file.name}).")
        else:
            info.caption = caption_file.read_text(encoding="utf-8").strip()
            if not info.caption:
                report.problems.append(f"{path.name}: the caption is empty.")
            elif report.trigger and report.trigger.lower() not in info.caption.lower():
                report.problems.append(f"{path.name}: the caption does not contain the trigger.")
        if info.sha256 in by_hash:
            report.problems.append(f"{path.name}: the same file as {by_hash[info.sha256]}.")
        else:
            by_hash[info.sha256] = path.name
        report.images.append(info)
    hashed = [i for i in report.images if i.phash]
    for a in range(len(hashed)):
        for b in range(a + 1, len(hashed)):
            x, y = hashed[a], hashed[b]
            if x.sha256 == y.sha256:
                continue
            bits = bin(int(x.phash, 16) ^ int(y.phash, 16)).count("1")
            if bits <= near_bits:
                report.near_duplicates.append((x.file, y.file, bits))
                report.warnings.append(
                    f"{x.file} and {y.file} look almost the same ({bits} of 64 hash bits differ)."
                )
    entries = {Path(e.get("file", "")).name: e for e in manifest.get("entries", [])}
    if entries:
        names = {i.file for i in report.images}
        for missing in sorted(set(entries) - names):
            report.warnings.append(f"{missing}: in manifest.json, but the file is missing.")
        for name in sorted(names - set(entries)):
            report.warnings.append(f"{name}: not in manifest.json, so its source is unknown.")
        for info in report.images:
            cap = entries.get(info.file, {}).get("caption")
            if cap and info.caption and cap.strip() != info.caption:
                report.warnings.append(f"{info.file}: the caption differs from manifest.json.")
    if not report.trigger:
        report.warnings.append("No trigger is given, so the captions were not checked for it.")
    return report


def write_report(report: DataReport, out_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(report.folder).name
    md, js = out / f"{name}.data-check.md", out / f"{name}.data-check.json"
    md.write_text(report.markdown(), encoding="utf-8")
    js.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return md, js


def dataset_folders(root: str | os.PathLike[str]) -> list[Path]:
    out = []
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir() or p.name.startswith(("_", ".")):
            continue
        images = image_folder(p)
        if images.is_dir() and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in images.iterdir()):
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="kilora data")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check image and caption pairs, and write a report")
    check.add_argument("folders", nargs="+")
    check.add_argument("--trigger", default=None, help="default: the manifest's trigger")
    check.add_argument("--min-side", type=int, default=MIN_SIDE)
    check.add_argument("--near-bits", type=int, default=NEAR_DUPLICATE_BITS)
    check.add_argument("--report-dir", default=None, help="default: next to each folder")
    args = parser.parse_args(argv)
    failed = 0
    for folder in args.folders:
        report = check_folder(
            folder, trigger=args.trigger, min_side=args.min_side, near_bits=args.near_bits
        )
        md, _ = write_report(report, args.report_dir or Path(folder).parent / "_reports")
        state = "usable" if report.ok else "NOT USABLE"
        print(
            f"{Path(folder).name}: {state}, {len(report.images)} images, "
            f"{len(report.problems)} problems, {len(report.warnings)} warnings. Report: {md}"
        )
        failed += not report.ok
    return 1 if failed else 0


def data_source(folder: str | os.PathLike[str], license: str | None = None) -> dict[str, Any]:
    base = Path(folder)
    if base.name == "images" and not (base / "manifest.json").exists():
        base = base.parent
    manifest = read_manifest(base)
    images = [p for p in image_folder(base).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    generators = sorted(
        {
            " ".join(x for x in (e.get("provider"), e.get("model")) if x)
            for e in manifest.get("entries", [])
            if e.get("provider") or e.get("model")
        }
    )
    out: dict[str, Any] = {"name": base.name, "images": len(images)}
    if generators:
        out["generators"] = generators
    if license or manifest.get("license"):
        out["license"] = license or manifest.get("license")
    if manifest.get("note"):
        out["note"] = str(manifest["note"])[:300]
    return out
