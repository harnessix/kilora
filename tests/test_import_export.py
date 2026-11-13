# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from test_import_modes import DIM, LAYERS, Toy, _lora

from kilora.adapter import KiLoRA
from kilora.errors import KiLoRAError
from kilora.live_imports import imported_session_for
from kilora.modfile import load_mod
from kilora.multi_import import import_loras


def _session(tmp_path, count=1):
    model = Toy().eval()
    paths = [tmp_path / f"source-{i}.safetensors" for i in range(count)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    sources = [
        dict(sha256=h, path=str(p), strength=1.0, axis="both")
        for p, h in zip(paths, hashes, strict=True)
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "toy", shapes)
    return model, paths, hashes, session


def test_embedded_export_preserves_routing_and_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model, paths, hashes, session = _session(tmp_path, 2)
    x = torch.randn(1, 3, DIM)
    baseline = KiLoRA(model, [session._source], session.mod.settings)
    with torch.no_grad():
        unchanged = model(x)
    baseline.remove()
    for option in ("on", "off"):
        session.set_options(edits=option)
        neutral = KiLoRA(model, session.sets(), session.mod.settings)
        with torch.no_grad():
            assert torch.equal(model(x), unchanged)
        neutral.remove()
    session.set_options(edits="on")
    target = next(iter(session.state))
    session.state[target]["core_0"][0, 1] = 0.7
    session.state[target]["strength_1"] *= 0.4
    original = KiLoRA(model, session.sets(), session.mod.settings)
    with torch.no_grad():
        expected = model(x)
    original.remove()
    output = tmp_path / "frozen.kmod"
    session.export(output, include_sources=True)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == hashes
    for path in paths:
        path.unlink()
    loaded = load_mod(output)
    restored = KiLoRA(model, loaded.sets, loaded.settings)
    with torch.no_grad():
        actual = model(x)
    restored.remove()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.manifest["edits"]["sources"] == session.sources
    assert all(s.frozen for s in loaded.sets)
    session.state[target]["core_0"].zero_()
    session.load_edits(output)
    assert session.state[target]["core_0"][0, 1] == 0.7


def test_plain_export_and_source_protection(tmp_path):
    _, paths, hashes, session = _session(tmp_path)
    target = next(iter(session._source.experts))
    ex = session._source.experts[target]
    output = tmp_path / "merged.safetensors"
    note = session.export(output, format="lora")
    state = load_file(str(output))
    actual = (
        state[f"transformer.{target}.lora_B.weight"] @ state[f"transformer.{target}.lora_A.weight"]
    )
    expected = ex.up[0, 0].T @ ex.down[0, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert note["relative_svd_residual"][target] < 1e-5
    with safe_open(str(output), framework="pt") as f:
        metadata = json.loads(f.metadata()["kilora.export"])
    assert metadata == note
    assert "Routing is not kept" in note["note"]
    with pytest.raises(KiLoRAError):
        session.export(paths[0], format="lora")
    with pytest.raises(KiLoRAError):
        session.export(paths[0], include_sources=True)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == hashes
