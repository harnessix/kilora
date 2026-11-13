# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from kilora.cli import main
from kilora.data import check_folder, dataset_folders, phash


def _img(path, seed, size=(600, 600)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(small).resize(size, Image.Resampling.BICUBIC).save(path)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "ink"
    (d / "images").mkdir(parents=True)
    for i in range(4):
        _img(d / "images" / f"{i}.png", i)
        (d / "images" / f"{i}.txt").write_text(f"a boat {i}, in ink style")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "trigger": "in ink style",
                "entries": [
                    {"file": f"images/{i}.png", "caption": f"a boat {i}, in ink style"}
                    for i in range(4)
                ],
            }
        )
    )
    return d


def test_clean_folder_is_usable(folder):
    r = check_folder(folder)
    assert r.ok and r.trigger == "in ink style" and len(r.images) == 4
    assert not r.warnings and not r.near_duplicates


def test_each_problem_is_found(folder):
    imgs = folder / "images"
    (imgs / "1.txt").write_text("a boat without the phrase")
    (imgs / "2.txt").unlink()
    (imgs / "dup.png").write_bytes((imgs / "0.png").read_bytes())
    (imgs / "dup.txt").write_text("copy, in ink style")
    _img(imgs / "small.png", 9, size=(300, 400))
    (imgs / "small.txt").write_text("small, in ink style")
    (imgs / "broken.png").write_bytes(b"not an image")
    (imgs / "broken.txt").write_text("broken, in ink style")
    r = check_folder(folder)
    text = "\n".join(r.problems)
    assert "1.png: the caption does not contain the trigger" in text
    assert "2.png: no caption file" in text
    assert "dup.png: the same file as 0.png" in text
    assert "small.png: 300x400 is smaller than 512" in text
    assert "broken.png: cannot read the image" in text
    assert any("not in manifest.json" in w for w in r.warnings)
    assert not r.ok


def test_near_duplicates_by_perceptual_hash(folder):
    imgs = folder / "images"
    with Image.open(imgs / "0.png") as img:
        img.convert("RGB").resize((610, 600)).save(imgs / "near.jpg", quality=80)
    (imgs / "near.txt").write_text("near, in ink style")
    r = check_folder(folder)
    assert r.ok
    assert [(a, b) for a, b, _ in r.near_duplicates] == [("0.png", "near.jpg")]
    with Image.open(imgs / "0.png") as a, Image.open(imgs / "1.png") as b:
        assert bin(phash(a) ^ phash(b)).count("1") > 6


def test_cli_writes_reports_and_lists_folders(folder, tmp_path, capsys):
    out = tmp_path / "reports"
    assert main(["data", "check", str(folder), "--report-dir", str(out)]) == 0
    assert "ink: usable, 4 images" in capsys.readouterr().out
    data = json.loads((out / "ink.data-check.json").read_text())
    assert data["ok"] and len(data["images"]) == 4
    assert "# Data check: ink" in (out / "ink.data-check.md").read_text()
    (tmp_path / "_skip").mkdir()
    assert dataset_folders(tmp_path) == [folder]
    assert main(["data", "check", str(tmp_path / "none"), "--report-dir", str(out)]) == 1
