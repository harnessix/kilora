# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher
from test_import_modes import DIM, LAYERS, Toy, _lora

from kilora.adapter import IMPORTED_DEPTH_SCHEDULE, IMPORTED_SCORE, Settings
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.comfyui.patching import ModChain
from kilora.compute import sampling_progress
from kilora.live import SESSIONS
from kilora.live_imports import imported_session_for
from kilora.modfile import read_manifest, save_mod
from kilora.modlist import ModChoice, load_choice
from kilora.multi_import import import_loras

_SCHEDULE_LINE = f"Depth schedule: DepthSchedule('expr', {IMPORTED_DEPTH_SCHEDULE!r})"


@pytest.fixture(autouse=True)
def _fresh_sessions(monkeypatch):
    SESSIONS.clear()
    for name in (
        "KILORA_IMPORT_SCORE",
        "KILORA_CONTENT_KEYS",
        "KILORA_LYCORIS_APPLY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    yield
    SESSIONS.clear()


def _session(tmp_path):
    model = Toy().eval()
    model.kilora_family = "kilora-tiny-1"
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    sources = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(dict(sha256=digest, path=str(path), strength=1.0, axis="both"))
    result = import_loras([str(p) for p in paths], model)
    routed = result.adapter_set
    shapes = {name: list(expert.down.shape) for name, expert in routed.experts.items()}
    session = imported_session_for(
        routed, sources, "kilora-tiny-1", shapes, sections=result.sections
    )
    return model, session


def _bind_folder(monkeypatch, directory):
    directory.mkdir()
    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(directory / name))
    return directory


def _copy(model):
    copied = Toy().eval()
    copied.load_state_dict(model.state_dict())
    copied.kilora_family = model.kilora_family
    return copied


def _forward(patcher, x, t):
    with sampling_progress(t):
        return patcher.run(x)


def _live(model, session, depth):
    choice = ModChoice(
        adapter_sets=session.sets(),
        session=session,
        sections=session.sections(),
        strength=1.0,
        label="routed LoRAs",
        mod_id="lora-routed",
    )
    return patching.apply(
        FakePatcher(FakeBaseModel(model)),
        ModChain(),
        depth,
        joint_choice_override=choice,
        depth_steps=N.COMFY_DEPTH_STEPS,
    )


def _outputs_match(model, session, name):
    live, live_report = _live(model, session, -1)
    loaded, load_report = N.KiLoRALoadMod().load(
        FakePatcher(FakeBaseModel(_copy(model))), name, 1.0
    )
    flat, _ = _live(_copy(model), session, 3)
    assert _SCHEDULE_LINE in load_report
    assert _SCHEDULE_LINE in live_report
    used = patching.applied(loaded)["depth_schedule"]
    assert used.depth(0, 1, "mlp", 0.1, "") == 1
    assert used.depth(0, 1, "mlp", 0.5, "") == 3
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(2))
    for t in (0.1, 0.5):
        assert torch.allclose(_forward(live, x, t), _forward(loaded, x, t), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(_forward(live, x, 0.1), _forward(flat, x, 0.1), rtol=1e-5, atol=1e-6)
    assert torch.allclose(_forward(live, x, 0.5), _forward(flat, x, 0.5), rtol=1e-5, atol=1e-6)


def _save_without_plan(path, session):
    save_mod(
        path,
        [dataclasses.replace(item, frozen=True) for item in session.sets(use_pending=False)],
        session.mod.settings,
        sections=session.sections(use_pending=False) or None,
        id=session.mod_id,
        name=session.mod_id,
        version="1.0.0",
        base_model={"family": session.family},
        edits={
            "sources": session.sources,
            "shapes": session.shapes,
            "options": {},
            "dimensions": {},
        },
    )


def test_load_mod_matches_with_zero_writes(tmp_path, monkeypatch):
    folder = _bind_folder(monkeypatch, tmp_path / "kilora")
    model, session = _session(tmp_path)
    path = folder / "export.kmod"
    session.export(path, include_sources=True)
    manifest = read_manifest(path)
    assert manifest["depth_schedule"] == IMPORTED_DEPTH_SCHEDULE
    assert manifest["schedule_yields_to_depth"] is True
    assert manifest["content_keys"] == "per_expert"
    assert manifest["lycoris_apply"] == "truncated"
    assert manifest["import_score"] == IMPORTED_SCORE
    _outputs_match(model, session, "export.kmod")


def test_load_mod_matches_after_one_write(tmp_path, monkeypatch):
    folder = _bind_folder(monkeypatch, tmp_path / "kilora")
    model, session = _session(tmp_path)
    target = next(iter(session.state))
    session.state[target]["core_0"][0, 1] = 0.7
    session.export(folder / "export.kmod", include_sources=True)
    _outputs_match(model, session, "export.kmod")


def test_old_export_loads_routed_import_defaults(tmp_path, monkeypatch):
    folder = _bind_folder(monkeypatch, tmp_path / "kilora")
    model, session = _session(tmp_path)
    path = folder / "old.kmod"
    _save_without_plan(path, session)
    assert "depth_schedule" not in read_manifest(path)
    loaded = load_choice(ModChoice(path), 0)
    assert loaded.manifest["depth_schedule"] == IMPORTED_DEPTH_SCHEDULE
    assert loaded.manifest["schedule_yields_to_depth"] is True
    assert loaded.manifest["content_keys"] == "per_expert"
    assert loaded.manifest["lycoris_apply"] == "truncated"
    assert loaded.manifest["import_score"] == IMPORTED_SCORE
    _outputs_match(model, session, "old.kmod")
    assert "depth_schedule" not in read_manifest(path)
    assert "content_keys" not in read_manifest(path)


def test_trained_mod_keeps_its_schedule(tmp_path, monkeypatch):
    folder = _bind_folder(monkeypatch, tmp_path / "kilora")
    model, session = _session(tmp_path)
    trained = folder / "trained.kmod"
    save_mod(
        trained,
        [session._source],
        Settings(depth=2),
        id="trained",
        name="trained",
        version="1.0.0",
        base_model={"family": "kilora-tiny-1"},
        depth_schedule="2, 2",
    )
    loaded, report = N.KiLoRALoadMod().load(FakePatcher(FakeBaseModel(model)), "trained.kmod", 1.0)
    used = patching.applied(loaded)["depth_schedule"]
    assert used.depth(0, 1, "mlp", 0.1, "") == 2
    assert used.depth(0, 1, "mlp", 0.5, "") == 2
    assert IMPORTED_DEPTH_SCHEDULE not in report
    stored = read_manifest(trained)
    assert stored["depth_schedule"] == "2, 2"
    assert "schedule_yields_to_depth" not in stored
    assert "content_keys" not in stored

    plain = folder / "plain.kmod"
    save_mod(
        plain,
        [session._source],
        Settings(depth=3),
        id="plain",
        name="plain",
        version="1.0.0",
        base_model={"family": "kilora-tiny-1"},
    )
    loaded, report = N.KiLoRALoadMod().load(
        FakePatcher(FakeBaseModel(_copy(model))), "plain.kmod", 1.0
    )
    assert patching.applied(loaded)["depth_schedule"] is None
    assert "Depth schedule:" not in report
    stored = read_manifest(plain)
    assert "depth_schedule" not in stored
    assert "schedule_yields_to_depth" not in stored
    assert "content_keys" not in stored


def test_load_mod_names_the_joint_experts(tmp_path, monkeypatch):
    folder = _bind_folder(monkeypatch, tmp_path / "kilora")
    model, session = _session(tmp_path)
    joint = dataclasses.replace(session._source, name=patching.JOINT_SET)
    sources = [dict(item, axis="creative") for item in session.sources]
    shared = imported_session_for(
        joint, sources, "kilora-tiny-1", session.shapes, sections=session.sections()
    )
    path = folder / "joint.kmod"
    shared.export(path, include_sources=True)
    chain = ModChain().add(patching.ModEntry("mod", str(path), 1.0, 1.0, True))
    patcher = FakePatcher(FakeBaseModel(_copy(model)))
    plan, _ = patching.plan_chain(patcher, chain)
    (mod,) = plan.mods
    assert patching._share_axis(mod) == "creative"
    routed = [s for s in patching._with_regions(plan, chain) if s.tag == patching.JOINT_SET]
    assert routed
    for item in routed:
        assert item.roles == "all"
        assert item.expert_names == ["0", "1"]
        assert len(item.expert_roles) == 2
