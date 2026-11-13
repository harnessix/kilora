# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import re
from datetime import date

from test_comfyui import denoiser, folders

from kilora import palette, usage_history
from kilora.adapter import Settings
from kilora.modfile import save_mod
from kilora.sections import Section


def test_history_privacy_disable_and_clear(tmp_path, monkeypatch, denoiser, folders):
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_modlist import write_mod

    from kilora.comfyui import nodes
    from kilora.key_cache import file_sha256

    path = write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = nodes.KiLoRASelect().select(path.name, 1.0, 1.0)
    nodes.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    rows = json.loads(usage_history.history_path().read_text())
    assert rows == [{"date": date.today().isoformat(), "sha256": [file_sha256(str(path))]}]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for row in rows for h in row["sha256"])
    a = "a" * 64
    before = usage_history.history_path().read_bytes()
    usage_history.record([a], enabled=False)
    assert usage_history.history_path().read_bytes() == before
    usage_history.clear()
    assert not usage_history.history_path().exists()


def test_counts_from_all_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    a, b, c = (x * 64 for x in "abc")
    usage_history.record([a, b, a])
    path = tmp_path / "test.kpalette.json"
    palette.save_palette(
        path,
        palette.Palette(
            "test",
            [
                palette.PaletteEntry("lora", "a", a),
                palette.PaletteEntry("lora", "c", c),
            ],
        ),
    )
    recipe = tmp_path / "test.kmod"
    transforms = {
        "strength": 1.0,
        "axis": "creative",
        "roles": "auto",
        "region": None,
        "timeline": [],
        "enabled": True,
    }
    sections = [
        Section("weight-ref", str(i), {}, {"sha256": h, "form": "lora", "transforms": transforms})
        for i, h in enumerate((b, c))
    ]
    save_mod(recipe, [], Settings(), sections=sections, name="test", license="NOASSERTION")
    singles, pairs = usage_history.counts([path], [recipe])
    assert singles == {a: 2, b: 2, c: 2}
    assert pairs == {(a, b): 1, (a, c): 1, (b, c): 1}


def test_apply_history_off_preserves_output(tmp_path, monkeypatch):
    from kilora.comfyui import nodes, patching

    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    file = tmp_path / "one.safetensors"
    file.write_bytes(b"weights")
    chain = patching.ModChain((patching.ModEntry("lora", str(file)),))
    result = object()
    monkeypatch.setattr(patching, "expand_chain", lambda chain, env: (chain, None))
    monkeypatch.setattr(patching, "apply", lambda *args, **kwargs: (result, "ok"))
    node = nodes.KiLoRAApply()
    off = node.apply(object(), chain, -1, record_history=False)
    assert not usage_history.history_path().exists()
    on = node.apply(object(), chain, -1, record_history=True)
    assert off == on == (result, "ok", None)
    assert len(usage_history.read_history()) == 1


def test_history_retains_newest_chains(tmp_path, monkeypatch):
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    path = usage_history.history_path()
    rows = [{"date": "2000-01-01", "sha256": [f"{i:064x}"]} for i in range(10_001)]
    path.write_text(json.dumps(rows))
    usage_history.record(["f" * 64])
    result = json.loads(path.read_text())
    assert len(result) == usage_history.MAX_HISTORY_CHAINS == 10_000
    assert result[:-1] == rows[-9_999:]
    assert result[-1]["sha256"] == ["f" * 64]


def test_counts_can_exclude_history(tmp_path, monkeypatch):
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path))
    usage_history.record(["a" * 64])
    monkeypatch.setattr(usage_history, "read_history", lambda: 1 / 0)
    assert usage_history.counts(history=False) == ({}, {})
