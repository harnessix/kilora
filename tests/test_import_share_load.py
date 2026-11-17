# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher
from test_import_modes import DIM, Toy, _lora

from kilora.adapter import KiLoRA
from kilora.comfyui.nodes import COMFY_DEPTH_STEPS, live_env
from kilora.comfyui.patching import ModChain, ModEntry, apply
from kilora.errors import KiLoRAError
from kilora.live import SESSIONS, LearnResult
from kilora.live_imports import imported_session_for
from kilora.modfile import load_mod, save_mod
from kilora.multi_import import import_loras

LAYER = "blocks.0.self_attn.q_proj"
PATCH = "diffusion_model." + LAYER
DEPTH = 1


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _session(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    model.kilora_family = "kilora-tiny-1"
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    sources = [
        dict(sha256=h, path=str(p), strength=1.0, axis="both")
        for p, h in zip(paths, hashes, strict=True)
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "kilora-tiny-1", shapes)
    return model, paths, session


def _copy(model):
    fresh = Toy().eval()
    fresh.load_state_dict(model.state_dict())
    fresh.kilora_family = "kilora-tiny-1"
    return fresh


def _load(model, path, depth=DEPTH):
    patcher = FakePatcher(FakeBaseModel(_copy(model)))
    entry = ModEntry("mod", str(path), 1.0, 1.0, True)
    loaded, report = apply(
        patcher,
        ModChain().add(entry),
        depth,
        live_env(),
        None,
        None,
        None,
        None,
        depth_steps=COMFY_DEPTH_STEPS,
    )
    return loaded, report


def _hook(module, seen):
    def keep(_module, _inputs, output):
        seen["y"] = output.detach().clone()

    return module.register_forward_hook(keep)


def _patched(patcher, x):
    seen = {}
    module = patcher.object_patches.get(PATCH)
    if module is None:
        module = patcher.model.diffusion_model.get_submodule(LAYER)
    handle = _hook(module, seen)
    try:
        full = patcher.run(x)
    finally:
        handle.remove()
    return full, seen["y"]


def _live(model, session, x, depth=DEPTH):
    fresh = _copy(model)
    adapter = KiLoRA(fresh, session.sets(), session.mod.settings)
    seen = {}
    handle = _hook(fresh.get_submodule(LAYER), seen)
    try:
        with adapter.depth(depth), torch.no_grad():
            full = fresh(x)
    finally:
        handle.remove()
        adapter.remove()
    return full, seen["y"]


def _commit_core(session, target, value, experts=(0,)):
    session.pending = {
        t: {part: tensor.clone() for part, tensor in parts.items()}
        for t, parts in session.state.items()
    }
    for expert in experts:
        core = session.pending[target][f"core_{expert}"].clone()
        core[0, 1] = value
        session.pending[target][f"core_{expert}"] = core
    session._pending_info = LearnResult(1, 0.1, 0.0, False, False, "write")
    session.commit()


def _make_exact(session, target, experts=(0,)):
    expert = session._source.experts[target]
    rank = int(expert.down.shape[2])
    for index in experts:
        session.state[target][f"core_{index}"] = torch.eye(rank)
        session.state[target][f"strength_{index}"] = torch.ones(1)
    exact = {index: (torch.eye(3), torch.eye(4)) for index in experts}
    experts = dict(session._source.experts)
    experts[target] = dataclasses.replace(expert, exact=exact)
    session._source = dataclasses.replace(session._source, experts=experts)


def test_load_after_writes_matches_live(tmp_path, monkeypatch):
    model, _paths, session = _session(tmp_path, monkeypatch)
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(4))
    assert "core_0" in session.state[LAYER]
    zero = {}
    for include in (False, True):
        path = tmp_path / f"zero-{int(include)}.kmod"
        session.export(path, include_sources=include)
        zero[include] = path
    _commit_core(session, LAYER, 0.7)
    live_full, live_layer = _live(model, session, x)
    reports = []
    for include in (False, True):
        path = tmp_path / f"edit-{int(include)}.kmod"
        session.export(path, include_sources=include)
        if not include:
            assert load_mod(path).sets == []
        loaded, report = _load(model, path)
        full, layer = _patched(loaded, x)
        torch.testing.assert_close(full, live_full, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(layer, live_layer, rtol=1e-5, atol=1e-6)
        _zero_loaded, _zero_report = _load(model, zero[include])
        _zero_full, zero_layer = _patched(_zero_loaded, x)
        assert not torch.allclose(layer, zero_layer)
        assert "edits:" in report and "mounts" in report
        reports.append(report)
    assert reports[0] == reports[1]


def test_exact_core_survives_export_and_load(tmp_path, monkeypatch):
    model, _paths, session = _session(tmp_path, monkeypatch)
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(5))
    experts = range(session._source.experts[LAYER].down.shape[0])
    _make_exact(session, LAYER, experts)
    zero = tmp_path / "exact-zero.kmod"
    session.export(zero, include_sources=True)
    _commit_core(session, LAYER, 0.4, experts)
    live_full, live_layer = _live(model, session, x)
    for include in (False, True):
        path = tmp_path / f"exact-{int(include)}.kmod"
        session.export(path, include_sources=include)
        if include:
            saved = load_mod(path).sets[0].experts[LAYER]
            assert not saved.exact
        loaded, report = _load(model, path)
        full, layer = _patched(loaded, x)
        torch.testing.assert_close(full, live_full, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(layer, live_layer, rtol=1e-5, atol=1e-6)
        assert "edits:" in report
    untouched, _report = _load(model, zero)
    _full, zero_layer = _patched(untouched, x)
    assert not torch.allclose(live_layer, zero_layer)


def test_missing_source_files_are_named(tmp_path, monkeypatch):
    model, paths, session = _session(tmp_path, monkeypatch)
    _commit_core(session, LAYER, 0.3)
    shared = tmp_path / "shared.kmod"
    session.export(shared, include_sources=False)
    for path in paths:
        path.unlink()
    with pytest.raises(KiLoRAError) as caught:
        _load(model, shared)
    message = str(caught.value)
    for path in paths:
        assert str(path) in message


def test_reports_differ_when_the_edits_differ(tmp_path, monkeypatch):
    model, _paths, session = _session(tmp_path, monkeypatch)
    first = tmp_path / "one.kmod"
    _commit_core(session, LAYER, 0.2)
    session.export(first, include_sources=True)
    second = tmp_path / "two.kmod"
    _commit_core(session, LAYER, -0.5)
    session.export(second, include_sources=False)
    _loaded, left = _load(model, first)
    _loaded, right = _load(model, second)
    assert left != right
    assert "edits:" in left and "edits:" in right
    assert "1 mounts" in left and "1 mounts" in right


def test_edits_off_and_no_writes_stay(tmp_path, monkeypatch):
    model, _paths, session = _session(tmp_path, monkeypatch)
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(6))
    bare = FakePatcher(FakeBaseModel(_copy(model)))
    base_full, base_layer = _patched(bare, x)

    zero = tmp_path / "no-writes.kmod"
    session.export(zero, include_sources=False)
    loaded, report = _load(model, zero)
    full, layer = _patched(loaded, x)
    live_full, live_layer = _live(model, session, x)
    assert torch.equal(full, live_full)
    assert torch.equal(layer, live_layer)
    assert "edits:" not in report

    legacy = tmp_path / "no-forest.kmod"
    save_mod(
        legacy,
        [],
        session.mod.settings,
        id=session.mod_id,
        name=session.mod_id,
        version="1.0.0",
        edits={"sources": session.sources, "shapes": session.shapes, "mounts": 0},
    )
    legacy_full, legacy_layer = _patched(_load(model, legacy)[0], x)
    assert torch.equal(legacy_full, base_full)
    assert torch.equal(legacy_layer, base_layer)

    kept = tmp_path / "sources.kmod"
    session.export(kept, include_sources=True)
    control = tmp_path / "control.kmod"
    save_mod(
        control,
        [dataclasses.replace(item, frozen=True) for item in session.sets()],
        session.mod.settings,
        id="control",
        name="control",
        version="1.0.0",
        base_model={"family": "kilora-tiny-1"},
    )
    kept_full, kept_layer = _patched(_load(model, kept)[0], x)
    control_full, control_layer = _patched(_load(model, control)[0], x)
    assert torch.equal(kept_full, control_full)
    assert torch.equal(kept_layer, control_layer)

    session.set_options(edits="off")
    session.state[LAYER]["core_0"][0, 1] = 0.9
    off = tmp_path / "off.kmod"
    session.export(off, include_sources=False)
    full, layer = _patched(_load(model, off)[0], x)
    off_full, off_layer = _live(model, session, x)
    assert torch.equal(full, off_full)
    assert torch.equal(layer, off_layer)
    on = tmp_path / "off-sources.kmod"
    session.export(on, include_sources=True)
    direct = tmp_path / "off-control.kmod"
    save_mod(
        direct,
        [dataclasses.replace(item, frozen=True) for item in session.sets()],
        session.mod.settings,
        id="off-control",
        name="off-control",
        version="1.0.0",
        base_model={"family": "kilora-tiny-1"},
    )
    on_full, on_layer = _patched(_load(model, on)[0], x)
    direct_full, direct_layer = _patched(_load(model, direct)[0], x)
    assert torch.equal(on_full, direct_full)
    assert torch.equal(on_layer, direct_layer)


@pytest.mark.parametrize("include_sources", [False, True])
def test_load_mod_preserves_import_forest_and_pinned_render(tmp_path, monkeypatch, include_sources):
    from kilora.live_imports import ImportEditor

    model, _paths, session = _session(tmp_path, monkeypatch)
    editor = ImportEditor(session)
    session.pin()
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(151))
    pinned, _layer = _live(model, session, x)
    session.prepare_pinned_write()
    _commit_core(session, LAYER, 0.35)
    branch = session.pinned_write_branch
    exported = tmp_path / "forest.kmod"
    session.export(exported, include_sources=include_sources)
    SESSIONS.clear()
    loaded, _report = _load(model, exported)
    output, _layer = _patched(loaded, x)
    assert torch.equal(output, pinned)
    from kilora.live_imports import session_for_shared_edits

    restored = session_for_shared_edits(exported, _copy(model))
    assert restored.editor_state.forest.list() == session.editor_state.forest.list()
    assert restored.pinned_write_branch == branch
    editor.checkout_branch(branch, promote=True)
    ImportEditor(restored).checkout_branch(branch, promote=True)
    expected, _layer = _live(model, session, x)
    actual, _layer = _live(model, restored, x)
    assert torch.equal(actual, expected)
