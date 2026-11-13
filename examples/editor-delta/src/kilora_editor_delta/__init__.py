# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kilora.editors import (
    Editor,
    EditorFileInfo,
    EditorPlugin,
    commit_write,
    register_editor,
)
from kilora.errors import KiLoRAError, ModFormatError
from kilora.i18n import text
from kilora.live_basis import (
    BASE_RANK,
    NESTED_MAX_RANK,
    V0_LAYOUT,
    default_v0,
    make_v0,
)
from kilora.live_basis import columns_of as _columns
from kilora.recording import GenerationRecord, PathRecorder, PathStatistics

PATH_EDITOR_FORMAT = "kilora-editor"
PATH_EDITOR_KIND = "path-write"
PATH_EDITOR_FEATURE = "editor.path-write"
MANIFEST_KEY = "kilora.manifest"
FILE_VERSION = "1.0"
PREFIX = "kilora/path/"


@dataclass
class PathEditor:
    manifest: dict[str, Any]
    beta: dict[str, float]
    lam: dict[str, float]
    h_bar: dict[str, torch.Tensor]
    v0: dict[str, torch.Tensor]

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self.v0)


def save_path_editor(
    path: str | os.PathLike[str],
    *,
    id: str,
    version: str,
    base_family: str,
    v0: Mapping[str, torch.Tensor],
    beta: Mapping[str, float],
    lam: Mapping[str, float],
    h_bar: Mapping[str, torch.Tensor],
    **fields: Any,
) -> dict[str, Any]:
    targets = list(v0)
    tensors = {}
    for t in targets:
        tensors[f"{PREFIX}{t}/v0"] = v0[t].float().contiguous()
        tensors[f"{PREFIX}{t}/h_bar"] = h_bar[t].float().contiguous()
    manifest = {
        "format": PATH_EDITOR_FORMAT,
        "format_version": FILE_VERSION,
        "requires_features": [PATH_EDITOR_FEATURE],
        "kind": PATH_EDITOR_KIND,
        "id": id,
        "version": version,
        "base_models": [{"family": base_family}],
        "targets": targets,
        "beta": {t: float(beta[t]) for t in targets},
        "lam": {t: float(lam[t]) for t in targets},
        "training": {"state": "trained"},
        **fields,
    }
    save_file(tensors, os.fspath(path), metadata={MANIFEST_KEY: json.dumps(manifest)})
    return manifest


def load_path_editor(path: str | os.PathLike[str]) -> PathEditor:
    label = os.path.basename(os.fspath(path))
    try:
        with safe_open(os.fspath(path), framework="pt") as f:
            meta = (f.metadata() or {}).get(MANIFEST_KEY)
            if meta is None:
                raise ModFormatError(f"{label}: not a KiLoRA editor file")
            manifest = json.loads(meta)
            if manifest.get("format") != PATH_EDITOR_FORMAT or manifest.get("kind") != (
                PATH_EDITOR_KIND
            ):
                raise ModFormatError(f"{label}: not a path editor")
            if PATH_EDITOR_FEATURE not in manifest.get("requires_features", []):
                raise ModFormatError(f"{label}: missing the feature {PATH_EDITOR_FEATURE}")
            if (manifest.get("training") or {}).get("state") != "trained":
                raise KiLoRAError(f"{label}: the editor is not trained")
            targets = list(manifest["targets"])
            v0 = {t: f.get_tensor(f"{PREFIX}{t}/v0") for t in targets}
            h_bar = {t: f.get_tensor(f"{PREFIX}{t}/h_bar") for t in targets}
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ModFormatError(f"{label}: cannot read the editor ({exc})") from None
    beta = {t: float(manifest["beta"][t]) for t in targets}
    lam = {t: float(manifest["lam"][t]) for t in targets}
    return PathEditor(manifest, beta, lam, h_bar, v0)


def check_editor(editor: PathEditor, session: Any) -> None:
    family = (session.mod.manifest.get("base_model") or {}).get("family")
    families = [b.get("family") for b in editor.manifest.get("base_models", [])]
    if family not in families:
        raise KiLoRAError(
            text("err.wrong_base", families=", ".join(map(str, families)), family=family)
        )
    targets = tuple(session.state)
    if set(editor.targets) != set(targets):
        missing = sorted(set(targets) ^ set(editor.targets))
        raise KiLoRAError(text("err.hook_mismatch", layer=missing[0]))
    for t, parts in session.state.items():
        cols, out = _columns(parts["down"]), parts["up"].shape[-1]
        if editor.v0[t].shape[1] != cols or editor.h_bar[t].shape[0] != out:
            raise KiLoRAError(text("err.size_mismatch", layer=t))
        if editor.h_bar[t].shape[1] != editor.v0[t].shape[0]:
            raise KiLoRAError(f"the editor's H_bar and V0 ranks differ at {t}")


def recorder_for(session: Any, editor: PathEditor) -> PathRecorder:
    a = {
        t: parts["down"].float().reshape(_columns(parts["down"]), parts["down"].shape[-1])
        for t, parts in session.state.items()
    }
    return PathRecorder(
        a=a, v0={t: editor.v0[t].float() for t in a}, enabled=lambda: bool(session.learning)
    )


class DeltaRule(Editor):
    needs_path = True

    def __init__(self, editor: PathEditor):
        self.editor = editor

    def check(self, session: Any) -> None:
        check_editor(self.editor, session)

    def recorder(self, session: Any) -> PathRecorder:
        return recorder_for(session, self.editor)

    def next_state(self, state, record):
        stats = record.stats
        new = {t: {p: v.clone() for p, v in parts.items()} for t, parts in state.items()}
        for t, c_kk in stats.c_kk.items():
            up = state[t]["up"].float()
            shape = up.shape
            b = up.reshape(-1, shape[-1]).t()
            v0 = self.editor.v0[t].float()
            u = b @ v0.t()
            beta, lam = self.editor.beta[t], self.editor.lam[t]
            eye = torch.eye(c_kk.shape[0])
            h_bar = self.editor.h_bar[t].float()
            u_new = u @ (eye - beta * c_kk) + beta * (stats.c_vk[t] - lam * h_bar)
            b_new = b + (u_new - u) @ v0
            new[t]["up"] = b_new.t().reshape(shape).to(state[t]["up"].dtype)
        return new


def delta_write(session: Any, editor: PathEditor, stats: PathStatistics) -> str:
    return commit_write(session, DeltaRule(editor), GenerationRecord(stats=stats))


def inspect(path: str) -> EditorFileInfo:
    editor = load_path_editor(path)
    ranks = {t: int(v.shape[0]) for t, v in editor.v0.items()}
    return EditorFileInfo(label="editor (path-write)", targets=list(editor.targets), ranks=ranks)


def load(path: str) -> DeltaRule:
    return DeltaRule(load_path_editor(path))


PLUGIN = EditorPlugin(PATH_EDITOR_KIND, load, inspect=inspect, title="Delta editor (example)")
register_editor(PLUGIN, "kilora_editor_delta")


def register(services):
    from kilora.host import PackageExports

    register_editor(PLUGIN, "kilora_editor_delta")
    return PackageExports(name="kilora-editor-delta", version="0.1.0", editors=[PLUGIN])
