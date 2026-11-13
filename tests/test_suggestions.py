# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types
from collections import Counter
from itertools import combinations

from test_comfyui import denoiser, folders

from kilora.comfyui import nodes, patching
from kilora.key_cache import file_sha256
from kilora.suggestions import suggest


def test_planted_partners_and_scores():
    current, *partners = [str(i) * 64 for i in range(8)]
    rows = [{current, partner} for partner in partners[:5]]
    rows += [{current}, {partners[5]}]
    singles, pairs = Counter(), Counter()
    for row in rows:
        singles.update(row)
        pairs.update(combinations(sorted(row), 2))
    installed = {f"{i}.safetensors": h for i, h in enumerate(partners[:4])}
    installed["four.safetensors"] = partners[4]
    installed["unrelated.safetensors"] = partners[5]
    result = suggest([current, current], installed, singles, pairs)
    assert {name for name, _ in result} == set(installed) - {"unrelated.safetensors"}
    assert all(abs(score - 1 / (6**0.5)) < 1e-12 for _, score in result)
    assert all(partners[6] not in name for name, _ in result)
    assert suggest([current], {}, singles, pairs) == []


def test_node_lists_only_installed_and_does_not_change_chain(tmp_path, folders, monkeypatch):
    current_file = folders["loras"] / "current.safetensors"
    partner_file = folders["loras"] / "partner.safetensors"
    current_file.write_bytes(b"current")
    partner_file.write_bytes(b"partner")
    current = file_sha256(str(current_file))
    partner = file_sha256(str(partner_file))
    missing = file_sha256(str(current_file))[:-1] + "0"
    singles = Counter({current: 2, partner: 1, missing: 1})
    pairs = Counter({tuple(sorted((current, partner))): 1, tuple(sorted((current, missing))): 1})
    history = types.ModuleType("kilora.usage_history")
    history.counts = lambda **kwargs: (singles.copy(), pairs.copy())
    monkeypatch.setitem(sys.modules, "kilora.usage_history", history)
    import kilora

    monkeypatch.setattr(kilora, "usage_history", history, raising=False)
    monkeypatch.setattr("kilora.palette.list_palettes", lambda: [])
    chain = patching.ModChain((patching.ModEntry("lora", str(current_file)),))
    before = tuple(chain)
    report = nodes.KiLoRASuggestions().show(chain)["result"][0]
    assert "partner.safetensors" in report
    assert "current.safetensors" not in report
    assert missing not in report
    assert tuple(chain) == before


def test_real_history_planted_partners(folders, tmp_path, monkeypatch):
    import pytest

    from kilora import palette

    history = pytest.importorskip("kilora.usage_history")
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    paths = []
    for i in range(7):
        path = folders["loras"] / f"{i}.safetensors"
        path.write_bytes(f"file {i}".encode())
        paths.append(path)
    hashes = [file_sha256(str(path)) for path in paths]
    for partner in hashes[1:]:
        history.record([hashes[0], partner])
    paths[-1].unlink()
    chain = patching.ModChain((patching.ModEntry("lora", str(paths[0])),))
    report = nodes.KiLoRASuggestions().show(chain)["result"][0]
    assert all(path.name in report for path in paths[1:6])
    assert paths[-1].name not in report


def test_without_suggestion_node_preserves_model_output(denoiser, folders, tmp_path, monkeypatch):
    import torch
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_modlist import write_mod

    from kilora import palette

    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = nodes.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    model = FakePatcher(FakeBaseModel(denoiser))
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(3))
    reference, _, _ = nodes.KiLoRAApply().apply(model, chain, 1)
    expected = reference.run(x)
    nodes.KiLoRASuggestions().show(chain)
    actual, _, _ = nodes.KiLoRAApply().apply(model, chain, 1)
    assert torch.equal(actual.run(x), expected)


def test_suggestions_cache_recipes_and_refresh_changed_file(tmp_path, folders, monkeypatch):
    from kilora import palette, usage_history
    from kilora.adapter import Settings
    from kilora.modfile import save_mod
    from kilora.sections import Section

    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    current = folders["loras"] / "current.safetensors"
    partner = folders["loras"] / "partner.safetensors"
    current.write_bytes(b"current")
    partner.write_bytes(b"partner")
    hashes = [file_sha256(str(path)) for path in (current, partner)]
    recipe = folders["kilora"] / "pair.kmod"
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
        for i, h in enumerate(hashes)
    ]
    save_mod(recipe, [], Settings(), sections=sections, name="pair", license="NOASSERTION")
    chain = patching.ModChain((patching.ModEntry("lora", str(current)),))
    node = nodes.KiLoRASuggestions()
    reads = []
    read_history = usage_history.read_history

    def count_history_reads():
        reads.append(1)
        return read_history()

    monkeypatch.setattr(usage_history, "read_history", count_history_reads)
    assert "partner.safetensors" in node.show(chain)["result"][0]
    with monkeypatch.context() as patch:
        patch.setattr("kilora.modfile.load_mod", lambda *args: 1 / 0)
        assert "partner.safetensors" in node.show(chain)["result"][0]
    assert len(reads) == 2
    save_mod(recipe, [], Settings(), sections=sections[:1], name="pair", license="NOASSERTION")
    assert "partner.safetensors" not in node.show(chain)["result"][0]
