# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import torch
from safetensors.torch import load_file, save_file

from kilora.library import LibraryIndex


def make_lora(path, stem, value):
    save_file(
        {
            f"{stem}.lora_down.weight": torch.full((2, 3), value),
            f"{stem}.lora_up.weight": torch.ones(4, 2),
        },
        str(path),
    )


def test_incremental_sync(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    for i in range(6):
        make_lora(folder / f"style{i}.safetensors", "lora_unet_layers_0_proj_ffn_up_gate", float(i))
    index = LibraryIndex.open(tmp_path / "index")
    first = index.sync([folder])
    assert (first.added, first.removed, first.skipped) == (6, 0, 0)
    assert {v["family"] for v in index.records.values()} == {"moss-image"}
    original = set(index.records)
    import kilora.references as references

    def fail_hash(_path):
        raise AssertionError("unchanged file was hashed")

    monkeypatch.setattr(references, "_sha256", fail_hash)
    again = LibraryIndex.open(tmp_path / "index").sync([folder])
    assert (again.added, again.removed, again.skipped) == (0, 0, 0)
    monkeypatch.undo()
    (folder / "style0.safetensors").unlink()
    make_lora(folder / "new.safetensors", "lora_unet_layers_0_proj_ffn_up_gate", 9.0)
    report = index.sync([folder])
    assert (report.added, report.removed) == (1, 1)
    assert len(original & index.records.keys()) == 5
    assert (
        len(
            [
                k
                for k in load_file(str(index.path / "features.safetensors"))
                if k.endswith("/summary")
            ]
        )
        == 6
    )
    assert json.loads((index.path / "library.json").read_text())["version"] == "1.0"


def test_bad_file_and_families(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    make_lora(folder / "one.safetensors", "lora_unet_layers_0_proj_ffn_up_gate", 1.0)
    make_lora(folder / "other.safetensors", "layers.0.attention.to_q", 2.0)
    make_lora(folder / "unknown.safetensors", "unrecognized.0.weight", 3.0)
    (folder / "bad.safetensors").write_bytes(b"bad")
    index = LibraryIndex.open(tmp_path / "index")
    report = index.sync([folder])
    assert (report.added, report.skipped) == (3, 1)
    assert "bad.safetensors" in report.reasons[0]
    families = {record["filename"]: record["family"] for record in index.records.values()}
    assert families == {
        "one.safetensors": "moss-image",
        "other.safetensors": "z-image",
        "unknown.safetensors": "unknown",
    }


def test_codebook_continues_identically_after_restart(tmp_path):
    import shutil

    folder = tmp_path / "loras"
    folder.mkdir()
    for i in range(4):
        make_lora(folder / f"style{i}.safetensors", "layers.0.attention.to_q", float(i + 1))
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    shutil.copytree(index.path, tmp_path / "other")
    restarted = LibraryIndex.open(tmp_path / "other")
    make_lora(folder / "new.safetensors", "layers.0.attention.to_q", 9.0)
    index.sync([folder])
    restarted.sync([folder])
    a = index.codebooks.families["z-image"]
    b = restarted.codebooks.families["z-image"]
    assert a.assignments == b.assignments
    assert all(torch.equal(value, b.centroids[key]) for key, value in a.centroids.items())
    assert all(index.records[k]["cluster_id"] == v for k, v in a.assignments.items())


def test_large_sync_speed_and_unchanged_features(tmp_path, monkeypatch):
    import os
    import time

    folder = tmp_path / "loras"
    folder.mkdir()
    state = {}
    for i in range(30):
        for layer in ("to_q", "to_k", "to_v", "to_out.0"):
            stem = f"layers.{i}.attention.{layer}"
            state[f"{stem}.lora_down.weight"] = torch.ones(1, 2)
            state[f"{stem}.lora_up.weight"] = torch.ones(2, 1)
        for layer in ("w1", "w2", "w3"):
            stem = f"layers.{i}.feed_forward.{layer}"
            state[f"{stem}.lora_down.weight"] = torch.ones(1, 2)
            state[f"{stem}.lora_up.weight"] = torch.ones(2, 1)
    for i in range(90):
        state[f"unmatched.{i}.lora_down.weight"] = torch.ones(1, 2)
        state[f"unmatched.{i}.lora_up.weight"] = torch.ones(2, 1)
    first = folder / "000.safetensors"
    save_file(state, str(first))
    for i in range(1, 500):
        os.link(first, folder / f"{i:03}.safetensors")
    index = LibraryIndex.open(tmp_path / "index")
    monkeypatch.setattr(index.hash_index, "hash", lambda path: path.stem.zfill(64))
    started = time.monotonic()
    report = index.sync([folder])
    assert report.added == 500 and not report.skipped
    assert time.monotonic() - started < 30
    before = load_file(str(index.path / "features.safetensors"))
    inputs = index.retrieval_inputs("z-image")
    assert len(inputs["records"]) == 500
    started = time.monotonic()
    assert index.retrieval_inputs("z-image") is inputs
    assert time.monotonic() - started < 0.05
    index.sync([folder])
    assert index.retrieval_inputs("z-image") is not inputs
    after = load_file(str(index.path / "features.safetensors"))
    assert before.keys() == after.keys()
    assert all(
        torch.equal(before[key], after[key]) for key in before if not key.startswith("codebook/")
    )
