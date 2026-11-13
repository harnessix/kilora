# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")
from PIL import Image
from test_train import pipeline

from kilora.card import read_card
from kilora.train import run_batch

COMMON = [
    "--resolution",
    "32",
    "--lr",
    "1e-2",
    "--precision",
    "fp32",
    "--device",
    "cpu",
    "--router-dim",
    "8",
    "--static",
    "--license",
    "CC0-1.0",
    "--preset",
    "quick",
    "--steps",
    "2",
    "--num-experts",
    "2",
]


def _dataset(root, name, n=3, trigger="in ink style", captions=True, seed=0):
    d = root / name / "images"
    d.mkdir(parents=True)
    gen = torch.Generator().manual_seed(seed)
    for i in range(n):
        pixels = (torch.rand(520, 520, 3, generator=gen) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(d / f"{i}.png")
        if captions:
            (d / f"{i}.txt").write_text(f"thing {i}, {trigger}")
    (root / name / "manifest.json").write_text(json.dumps({"trigger": trigger}))


def _batch(root, out, *extra):
    argv = [str(root), "--model", "tiny", "--output-dir", str(out), *COMMON, *extra]
    return run_batch(argv, pipe_factory=pipeline)


def test_batch_trains_each_folder_and_writes_a_summary(tmp_path):
    root, out = tmp_path / "data", tmp_path / "out"
    _dataset(root, "ink", seed=1)
    _dataset(root, "film", trigger="on film", seed=2)
    _dataset(root, "nocap", captions=False, seed=3)
    (root / "_abandoned").mkdir()
    rows = {r["mod"]: r for r in _batch(root, out)}
    assert set(rows) == {"film", "ink", "nocap"}
    assert rows["ink"]["status"] == "done" and rows["ink"]["steps"] == 2
    assert rows["nocap"]["status"] == "data problems"
    card = read_card(out / "ink.zimage.kmod.safetensors")
    assert card["trigger"] == ["in ink style"] and card["id"] == "ink"
    assert read_card(out / "film.zimage.kmod.safetensors")["trigger"] == ["on film"]
    assert "step 2/2" in (out / "logs" / "ink.log").read_text()
    assert (out / "logs" / "nocap.data-check.md").exists()
    summary = (out / "summary.md").read_text()
    assert "| ink | done | 2 |" in summary and "| nocap | data problems |" in summary
    assert len(json.loads((out / "summary.json").read_text())) == 3


def test_resume_skips_finished_mods_and_retrain_redoes_them(tmp_path):
    root, out = tmp_path / "data", tmp_path / "out"
    _dataset(root, "ink", seed=1)
    _batch(root, out)
    first = (out / "ink.zimage.kmod.safetensors").read_bytes()
    _dataset(root, "film", trigger="on film", seed=2)
    rows = {r["mod"]: r for r in _batch(root, out)}
    assert rows["ink"]["status"] == "exists" and rows["film"]["status"] == "done"
    assert (out / "ink.zimage.kmod.safetensors").read_bytes() == first
    rows = {r["mod"]: r for r in _batch(root, out, "--retrain", "--only", "ink")}
    assert list(rows) == ["ink"] and rows["ink"]["status"] == "done"


def test_a_failed_mod_leaves_no_file_and_the_batch_goes_on(tmp_path, monkeypatch):
    root, out = tmp_path / "data", tmp_path / "out"
    _dataset(root, "a", seed=1)
    _dataset(root, "b", seed=2)
    import kilora.train as T

    real = T.run

    def flaky(args, pipe=None):
        if args.id == "a":
            raise RuntimeError("out of memory")
        return real(args, pipe)

    monkeypatch.setattr(T, "run", flaky)
    rows = {r["mod"]: r for r in _batch(root, out)}
    assert rows["a"]["status"] == "failed" and "out of memory" in rows["a"]["note"]
    assert rows["b"]["status"] == "done"
    assert not (out / "a.zimage.kmod.safetensors").exists()


def test_batch_passes_the_recommended_values(tmp_path):
    root, out = tmp_path / "data", tmp_path / "out"
    _dataset(root, "ink", seed=1)
    _batch(root, out, "--default-depth", "2", "--default-strength", "0.5")
    card = read_card(out / "ink.zimage.kmod.safetensors")
    assert card["depth"] == 2 and card["strength"]["recommended"] == 0.5
