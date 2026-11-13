# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil

import pytest
import torch

pytest.importorskip("diffusers")
from test_comfyui import denoiser, folders
from test_path_write import _setup, fresh_sessions
from test_train import pipeline

from kilora.card import read_card
from kilora.check import check_file
from kilora.cli import main
from kilora.plugin_support import TrainerNetwork
from kilora.preview import render_showcase


def _static_mod(path):
    pipe = pipeline()
    net = TrainerNetwork(
        family="z-image",
        rank=2,
        manifest={
            "id": "film",
            "license": "CC-BY-4.0",
            "trigger_words": ["on film"],
            "data_source": {
                "name": "film",
                "images": 24,
                "note": "Images generated with a paid commercial image API.",
            },
        },
    )
    net.attach(pipe.transformer)
    with torch.no_grad():
        for site in net.adapter.sites:
            site.tensors["s0_up"].normal_(0, 2.0)
    saved = net.save(path)
    net.remove()
    return saved


def _checksums_match(folder, package):
    on_disk = sorted(p.name for p in folder.iterdir() if p.name != "package.json")
    assert on_disk == sorted(f["name"] for f in package["files"])
    for f in package["files"]:
        data = (folder / f["name"]).read_bytes()
        assert f["sha256"] == hashlib.sha256(data).hexdigest() and f["bytes"] == len(data)


def test_batch_folder_to_release_folders(denoiser, folders, tmp_path, capsys):
    batch = tmp_path / "batch"
    (batch / "film").mkdir(parents=True)
    film = _static_mod(batch / "film" / "film.kmod.safetensors")
    (batch / "film" / "card.md").write_text("old card")
    render_showcase(pipeline(), film, tmp_path / "renders" / "film", size=32, steps=2)
    _, _, _, _, _, _ = _setup(denoiser, folders, tmp_path)
    (batch / "ink").mkdir()
    ink = batch / "ink" / "ink.kmod.safetensors"
    shutil.copyfile(folders["kilora"] / "ink.kmod", ink)
    release = tmp_path / "release"

    assert (
        main(
            [
                "package",
                str(film),
                "--output-dir",
                str(release),
                "--samples",
                str(tmp_path / "renders"),
            ]
        )
        == 0
    )
    assert main(["package", str(ink), "--output-dir", str(release)]) == 0
    out = capsys.readouterr().out
    assert "film: 5 files" in out and "ink: 3 files" in out

    film_dir = release / "film"
    p = json.loads((film_dir / "package.json").read_text())
    _checksums_match(film_dir, p)
    assert {f["name"] for f in p["files"]} >= {"grid.png", "render.json"}
    assert p["open_points"] == [] and p["status"] == "prepared, not published"
    card = read_card(film_dir / p["mod"]["file"])
    assert card["trigger"] == ["on film"] and "paid commercial image API" in card["source_line"]
    assert check_file(film_dir / p["mod"]["file"]).ok
    rendered = json.loads((film_dir / "render.json").read_text())
    assert (
        rendered["mod"]["digest"]
        == json.loads((tmp_path / "renders" / "film" / "render.json").read_text())["mod"]["digest"]
    )

    ink_dir = release / "ink"
    q = json.loads((ink_dir / "package.json").read_text())
    _checksums_match(ink_dir, q)
    assert any("sample grid" in x for x in q["open_points"])
    assert check_file(ink_dir / q["mod"]["file"]).kind == "live mod"

    main(
        ["package", str(film), "--output-dir", str(release), "--samples", str(tmp_path / "renders")]
    )
    again = json.loads((film_dir / "package.json").read_text())
    assert again["files"] == p["files"]
