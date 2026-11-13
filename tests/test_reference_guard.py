# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from kilora import modlist, references
from kilora.errors import ModFormatError


def test_reference_cycle_and_depth_include_chain(tmp_path, monkeypatch):
    paths = [tmp_path / f"{i}.kmod" for i in range(6)]
    edges = {str(paths[i]): paths[i + 1] for i in range(5)}
    monkeypatch.setattr(references, "resolve", lambda meta, dirs: edges[meta["path"]])

    def load(path):
        meta = {"path": str(path), "form": "mod", "transforms": {"enabled": True}}
        return SimpleNamespace(sections=[SimpleNamespace(kind="weight-ref", meta=meta)])

    monkeypatch.setattr(modlist, "_cached_mod", load)
    edges[str(paths[2])] = paths[0]
    with pytest.raises(ModFormatError, match="cycle") as exc:
        list(modlist._reference_paths(paths[0], []))
    assert all(str(p) in str(exc.value) for p in paths[:3])
    edges[str(paths[2])] = paths[3]
    with pytest.raises(ModFormatError, match="depth") as exc:
        list(modlist._reference_paths(paths[0], []))
    assert all(str(p) in str(exc.value) for p in paths)


def test_load_choice_checks_reference_chain(tmp_path, monkeypatch):
    paths = [tmp_path / f"{i}.kmod" for i in range(5)]
    for path in paths:
        path.touch()
    monkeypatch.setattr(references, "resolve", lambda meta, dirs: paths[meta["next"]])

    def load(path):
        i = paths.index(path)
        sections = []
        if i < 4:
            sections = [
                SimpleNamespace(
                    kind="weight-ref",
                    meta={
                        "next": i + 1,
                        "form": "mod",
                        "transforms": {"enabled": True, "strength": 1.0},
                    },
                )
            ]
        return SimpleNamespace(
            manifest={"id": f"mod-{i}"},
            sections=sections,
            sets=[],
            settings=modlist.imported_settings(),
        )

    monkeypatch.setattr(modlist, "_cached_mod", load)
    choice = modlist.ModChoice(path=paths[0])
    assert modlist.load_choice(choice, 0, [tmp_path]).mod_id == "mod-0"
