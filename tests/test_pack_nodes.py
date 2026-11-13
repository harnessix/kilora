# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from test_prompt_solve_nodes import run, world

from kilora.comfyui import nodes as N
from kilora.comfyui import pack_nodes as K
from kilora.comfyui.pack_nodes import KiLoRAPack, solved_mod
from kilora.comfyui.prompt_solve import SolveAdapter
from kilora.library.contrast import PromptPair
from kilora.lora_import import import_lora
from kilora.modfile import load_mod, read_manifest
from kilora.packs import load_pack, pack_choices, scan_library
from kilora.palette import load_palette
from kilora.presets import pack_license


def test_solve_mod_metadata_preview_and_import(world):
    paths, chain, _ = run(world, output="both", preview=torch.ones(1, 3, 4, 3))
    mods = [Path(p) for p in paths.splitlines() if p.endswith(".kmod.safetensors")]
    assert len(mods) == len(chain) == 2
    for path, label in zip(mods, ("ink", "wash"), strict=True):
        manifest = read_manifest(path)
        assert manifest["id"] == f"solved-{label}"
        assert manifest["version"] == "1.0.0"
        assert manifest["trigger_words"] == [label]
        assert manifest["license"] == pack_license("anima")[0]
        assert manifest["content_source"] == "original"
        assert manifest["solve_source"] == "prompt-set"
        assert manifest["prompt_solve"]["pairs"]
        mod = load_mod(path)
        assert Image.open(io.BytesIO(mod.previews[0])).size == (4, 3)
        from kilora.comfyui import patching

        adapter = SolveAdapter(world.model)
        plain = world.root / "kilora-solved" / f"solved-{label}.safetensors"
        expected = import_lora(
            plain, patching.shape_model(adapter.shapes), family="anima", experts=1
        ).adapter_set
        from kilora.adapter import imported_settings
        from kilora.modfile import save_mod

        check = world.root / f"{label}-check.kmod.safetensors"
        save_mod(check, [expected], imported_settings())
        actual, reference = load_file(path), load_file(check)
        for key, tensor in reference.items():
            torch.testing.assert_close(actual[key], tensor)


def test_pack_copy_palette_and_resolution(world, monkeypatch):
    paths, _, _ = run(world, output="kmod")
    path, report = KiLoRAPack().save(paths, "styles", copy_mods=True)["result"]
    pack = load_pack(path)
    assert pack.depth is None
    assert all(entry.strength == 0.7 for entry in pack.entries)
    copied = world.root / "styles"
    mods = list(copied.glob("*.kmod.safetensors"))
    assert len(mods) == len(pack.entries) == 2
    assert (copied / "styles.kpack.json").is_file()
    assert len(pack_choices(pack, scan_library(mods))) == 2
    assert len(load_palette(path).entries) == 2
    monkeypatch.setattr(N.FOLDERS, "list", lambda *a: [str(p) for p in mods])
    chain, _, _, _ = N.KiLoRAPalette().load(path)
    assert len(chain) == 2
    assert "preview=no" in report


def test_missing_mod_names_file(world):
    with pytest.raises(FileNotFoundError, match="missing.kmod"):
        KiLoRAPack().save("missing.kmod.safetensors", "missing")


def test_default_lora_bytes_unchanged(world):
    paths, _, _ = run(world)
    before = {p: Path(p).read_bytes() for p in paths.splitlines()}
    explicit, _, _ = run(world, output="lora", overwrite=True)
    assert explicit == paths
    assert before == {p: Path(p).read_bytes() for p in explicit.splitlines()}


@pytest.mark.parametrize("mode", ["single", "joint"])
def test_solved_mod_conflict_uses_hash_and_reuses_it(world, monkeypatch, mode):
    paths, _, _ = run(world, output="both", mode=mode)
    mod, source = map(Path, paths.splitlines()[:2])
    before = mod.read_bytes()
    record = copy.deepcopy(read_manifest(mod)["prompt_solve"])
    record["settings"]["rank"] = 3
    pairs = [PromptPair(**pair) for pair in record["pairs"]]
    adapter = SolveAdapter(world.model)
    label = "joint" if mode == "joint" else "ink"
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    mod_id = f"{source.stem}-{label}" if mode == "joint" else source.stem
    result = solved_mod(source, adapter, record, label, pairs, None, overwrite=False)
    assert result == mod.with_name(f"{mod_id}-{digest}.kmod.safetensors")
    assert read_manifest(result)["id"] == f"{mod_id}-{digest}"
    assert read_manifest(result)["prompt_solve"] == record
    assert mod.read_bytes() == before
    monkeypatch.setattr(K, "import_lora", lambda *a, **kw: pytest.fail("cached mod imported"))
    assert solved_mod(source, adapter, record, label, pairs, None, overwrite=False) == result
    assert mod.read_bytes() == before


@pytest.mark.parametrize("output", ["kmod", "both"])
def test_solved_mod_new_inputs_and_overwrite(world, output):
    initial, _, _ = run(world, output=output)
    originals = {
        Path(path): Path(path).read_bytes()
        for path in initial.splitlines()
        if path.endswith(".kmod.safetensors")
    }
    paths, _, _ = run(world, output=output, rank=3)
    mods = [Path(path) for path in paths.splitlines() if path.endswith(".kmod.safetensors")]
    for old, new in zip(originals, mods, strict=True):
        record = read_manifest(new)["prompt_solve"]
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        assert new == old.with_name(f"{read_manifest(old)['id']}-{digest}.kmod.safetensors")
        assert read_manifest(new)["id"] == f"{read_manifest(old)['id']}-{digest}"
        assert old.read_bytes() == originals[old]
    replaced, _, _ = run(world, output=output, rank=3, overwrite=True)
    assert replaced == initial
    assert all(read_manifest(path)["prompt_solve"]["settings"]["rank"] == 3 for path in originals)


def test_solved_mod_hash_clash_still_raises(world, monkeypatch):
    initial, _, _ = run(world, output="kmod")
    monkeypatch.setattr(
        K, "hashlib", SimpleNamespace(sha256=lambda *a: SimpleNamespace(hexdigest=lambda: "0" * 64))
    )
    for name in initial.splitlines():
        path = Path(name)
        mod_id = read_manifest(path)["id"]
        path.with_name(f"{mod_id}-00000000.kmod.safetensors").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="different solve inputs or preview"):
        run(world, output="kmod", rank=3)
