# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from conftest import randomize_up
from test_comfyui import denoiser, folders
from test_path_write import _setup, fresh_sessions

from kilora import KiLoRAConfig, attach
from kilora.card import read_card
from kilora.check import check_file
from kilora.cli import main
from kilora.errors import KiLoRAError
from kilora.modfile import read_manifest
from kilora.release import package_mod

TARGET = "layers.*.attention.to_out.0"


def _mod(model, path, **fields):
    adapter = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=2))
    randomize_up(adapter)
    adapter.save(
        path,
        **{
            "id": "ink",
            "license": "CC-BY-4.0",
            "trigger_words": ["in ink"],
            "base_model": {"family": "z-image"},
            **fields,
        },
    )
    adapter.remove()
    return path


def _samples(tmp_path, mod):
    d = tmp_path / "renders" / "ink"
    d.mkdir(parents=True)
    (d / "grid.png").write_bytes(b"\x89PNG fake")
    digest = read_manifest(mod)["digest"]
    (d / "render.json").write_text(json.dumps({"mod": {"id": "ink", "digest": digest}}))
    return d


def test_release_folder(model, tmp_path):
    mod = _mod(model, tmp_path / "ink.kmod")
    p = package_mod(mod, tmp_path / "out", sample_dir=_samples(tmp_path, mod))
    folder = tmp_path / "out" / "ink"
    names = {f["name"] for f in p["files"]}
    assert names == {
        "ink.zimage.kmod.safetensors",
        "CARD.md",
        "README.md",
        "grid.png",
        "render.json",
    }
    for f in p["files"]:
        data = (folder / f["name"]).read_bytes()
        assert f["bytes"] == len(data) and f["sha256"] == hashlib.sha256(data).hexdigest()
    assert read_card(folder / "ink.zimage.kmod.safetensors")["trigger"] == ["in ink"]
    assert p["status"] == "prepared, not published" and p["open_points"] == []
    assert json.loads((folder / "package.json").read_text()) == p
    assert "| ID | `ink` 0.1.0 |" in (folder / "CARD.md").read_text()


def test_open_points_are_listed(model, tmp_path):
    mod = _mod(model, tmp_path / "ink.kmod", license="NOASSERTION", trigger_words=[])
    p = package_mod(mod, tmp_path / "out")
    text = " ".join(p["open_points"])
    assert "No sample grid" in text and "no license" in text and "no trigger" in text
    stale = _samples(tmp_path, mod)
    (stale / "render.json").write_text(json.dumps({"mod": {"digest": "sha256-old"}}))
    p = package_mod(mod, tmp_path / "out2", sample_dir=stale)
    assert any("another version" in x for x in p["open_points"])


def test_unusable_mod_is_refused(tmp_path):
    (tmp_path / "bad.kmod").write_text("not a mod")
    with pytest.raises(KiLoRAError, match="kilora check"):
        package_mod(tmp_path / "bad.kmod", tmp_path / "out")


def test_live_mod_and_cli(denoiser, folders, tmp_path, capsys):
    _setup(denoiser, folders, tmp_path)
    mod = folders["kilora"] / "ink.kmod"
    p = package_mod(mod, tmp_path / "out")
    folder = tmp_path / "out" / p["mod"]["id"]
    assert check_file(folder / p["mod"]["file"]).kind == "live mod"
    code = main(
        ["package", str(mod), str(tmp_path / "missing.kmod"), "--output-dir", str(tmp_path / "cli")]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "files in" in captured.out and "missing.kmod: error" in captured.err
