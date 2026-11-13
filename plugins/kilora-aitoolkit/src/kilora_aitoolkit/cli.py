# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHIM = """# Added by kilora-aitoolkit. Remove this folder to remove the KiLoRA extension.
from kilora_aitoolkit.extension import AI_TOOLKIT_EXTENSIONS  # noqa: F401
"""


def install(root: Path, force: bool = False) -> Path:
    if not (root / "version.py").exists() or not (root / "extensions").is_dir():
        raise SystemExit(f"{root} does not look like an AI Toolkit folder")
    folder = root / "extensions" / "kilora"
    target = folder / "__init__.py"
    if target.exists() and not force:
        raise SystemExit(f"{target} exists; use --force to replace it")
    folder.mkdir(exist_ok=True)
    target.write_text(SHIM, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kilora-aitoolkit")
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("install", help="add the KiLoRA extension to an AI Toolkit folder")
    i.add_argument("folder", type=Path)
    i.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    target = install(args.folder.resolve(), args.force)
    print(f"Installed the KiLoRA extension: {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
