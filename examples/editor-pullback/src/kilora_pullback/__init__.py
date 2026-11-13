# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kilora.editors import Editor, EditorFileInfo, EditorPlugin, EditorSupport
from kilora.errors import KiLoRAError, ModFormatError

KIND = "pullback"
FORMAT = "kilora-editor"
FILE_VERSION = "1.0"
MANIFEST_KEY = "kilora.manifest"


def save_pullback_editor(path: str | os.PathLike[str], rate: float, base_family: str) -> dict:
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    manifest = {
        "format": FORMAT,
        "format_version": FILE_VERSION,
        "kind": KIND,
        "id": "pullback",
        "version": "0.1.0",
        "license": "CC0-1.0",
        "base_models": [{"family": base_family}],
        "training": {"state": "trained"},
        "supports": EditorSupport(hooks=[]).to_manifest(),
        "rate": float(rate),
    }
    tensors = {"kilora/pullback/rate": torch.tensor([float(rate)])}
    save_file(tensors, os.fspath(path), metadata={MANIFEST_KEY: json.dumps(manifest)})
    return manifest


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    label = os.path.basename(os.fspath(path))
    try:
        with safe_open(os.fspath(path), framework="pt") as f:
            manifest = json.loads((f.metadata() or {})[MANIFEST_KEY])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ModFormatError(f"{label}: cannot read the editor ({exc})") from None
    if manifest.get("format") != FORMAT or manifest.get("kind") != KIND:
        raise ModFormatError(f"{label}: not a pullback editor")
    if manifest.get("format_version") != FILE_VERSION:
        raise ModFormatError(f"{label}: unknown format version {manifest.get('format_version')}")
    if (manifest.get("training") or {}).get("state") != "trained":
        raise KiLoRAError(f"{label}: the editor is not trained")
    return manifest


class WriteCount:
    def __init__(self) -> None:
        self.count = torch.zeros(1)

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {"count": self.count}

    def load_tensors(self, tensors) -> None:
        self.count = tensors["count"].clone()

    def reset(self) -> None:
        self.count = torch.zeros(1)


class Pullback(Editor):
    def __init__(self, path: str) -> None:
        self.manifest = read_manifest(path)
        self.rate = float(self.manifest["rate"])
        self.supports = EditorSupport.from_manifest(self.manifest["supports"])
        self.session: Any = None

    def check(self, session: Any) -> None:
        family = (session.mod.manifest.get("base_model") or {}).get("family")
        families = [b.get("family") for b in self.manifest["base_models"]]
        if family not in families:
            raise KiLoRAError(f"This editor is for {', '.join(families)}, not for {family}.")
        self.session = session
        if getattr(session, "editor_state_kind", None) != KIND:
            session.attach_editor_state(WriteCount(), KIND)

    def next_state(self, state, record):
        if self.session is None:
            raise KiLoRAError("check() was not called")
        clean = self.session.clean
        new = {}
        for t, parts in state.items():
            up = parts["up"].float()
            target = clean[t]["up"].to(up.device).float()
            new[t] = {
                "down": parts["down"],
                "up": (up + self.rate * (target - up)).to(parts["up"].dtype),
            }
        self.session.editor_state.count += 1
        return new


def inspect(path: str) -> EditorFileInfo:
    manifest = read_manifest(path)
    return EditorFileInfo(label="editor (pullback)", targets=[], facts=[f"rate {manifest['rate']}"])


PLUGIN = EditorPlugin(KIND, Pullback, inspect=inspect, title="Pullback (example)")
