# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import json

import pytest
import torch

pytest.importorskip("diffusers")
from PIL import Image
from test_train import pipeline

from kilora.cli import _render
from kilora.plugin_support import TrainerNetwork
from kilora.preview import SHOWCASE_PROMPTS, render_showcase


@pytest.fixture
def mod(tmp_path):
    pipe = pipeline()
    net = TrainerNetwork(
        family="z-image", rank=2, manifest={"id": "ink", "trigger_words": ["in ink style"]}
    )
    net.attach(pipe.transformer)
    with torch.no_grad():
        for site in net.adapter.sites:
            site.tensors["s0_up"].normal_(0, 2.0)
    path = net.save(tmp_path / "ink.kmod.safetensors")
    net.remove()
    return path


def test_grid_and_record(mod, tmp_path):
    out = tmp_path / "results" / "showcase" / "ink"
    rec = render_showcase(pipeline(), mod, out, size=32, steps=2)
    with Image.open(out / "grid.png") as grid:
        assert grid.size == (4 * 32, 4 * 32) and not grid.text
        tiles = [
            [grid.crop((c * 32, r * 32, c * 32 + 32, r * 32 + 32)).tobytes() for c in range(4)]
            for r in range(4)
        ]
    assert tiles[0][0] != tiles[0][2]
    assert tiles[0][0] != tiles[1][0]
    assert [c["strength"] for c in rec["columns"]] == [0.0, 0.5, 1.0, 1.5]
    assert rec["rows"][0]["prompt"] == f"{SHOWCASE_PROMPTS[0]}, in ink style"
    assert "ingest_rule" not in rec and rec["tile"] == 32
    assert json.loads((out / "render.json").read_text()) == rec
    again = render_showcase(pipeline(), mod, tmp_path / "again", size=32, steps=2)
    assert (tmp_path / "again" / "grid.png").read_bytes() == (out / "grid.png").read_bytes()
    assert again["rows"] == rec["rows"]


def test_cli(mod, tmp_path, capsys):
    args = argparse.Namespace(
        mod=str(mod),
        model="tiny",
        output_dir=str(tmp_path / "out"),
        strengths=[0.0, 1.0],
        size=32,
        steps=2,
        guidance=0.0,
        device="cpu",
    )
    assert _render(args, pipe=pipeline()) == 0
    assert "4 prompts by 2 strengths" in capsys.readouterr().out
    assert (tmp_path / "out" / "ink" / "render.json").exists()
