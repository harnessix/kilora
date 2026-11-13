# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from live_helpers import build_live_mod
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_path_write import fresh_sessions

from kilora import editor_packs as ep
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.errors import KiLoRAError
from kilora.testing import TINY_FAMILY


def _pack(folder, pid, version, role="common", families=(), kind="example-editor"):
    m = {
        "id": pid,
        "version": version,
        "role": role,
        "editor": {"kind": kind, "version": ">=1.0"},
        "base_models": [{"family": TINY_FAMILY}],
        "mod_families": list(families),
        "requires": {"common": ">=1.0"} if role == "family" else {},
        "storage": "int8-block",
        "license": "CC0-1.0",
        "training": {"state": "trained"},
    }
    path = folder / f"{pid}-{version}{ep.PACK_SUFFIX}"
    ep.write_pack(path, m, {"directions": torch.randn(4, 32, generator=torch.Generator())})
    return path


@pytest.fixture
def live(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setattr(N, "live_env", lambda: patching.LiveEnv(str(tmp_path / "saves")))
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod", family="style")
    _pack(folders["kilora"], "common", "1.0.0")
    _pack(folders["kilora"], "style", "1.0.0", "family", ["style"])
    _pack(folders["kilora"], "texture", "1.0.0", "family", ["texture"])
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    return patched, patching.sessions_of(patched)[0]


def test_lists_packs_and_chooses_common_and_family(live):
    patched, session = live
    model, report = N.KiLoRALivePacks().choose(patched, "", "auto", "")
    assert model is patched
    assert [(p.id, p.version) for p in session.editor_packs] == [
        ("common", "1.0.0"),
        ("style", "1.0.0"),
    ]
    assert "[on] common 1.0.0, common" in report and "[on] style 1.0.0, family (style)" in report
    assert "[not used] texture 1.0.0" in report


def test_turn_a_pack_off_and_common_only(live):
    patched, session = live
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "auto", "style")
    assert [p.id for p in session.editor_packs] == ["common"] and "[off] style" in report
    N.KiLoRALivePacks().choose(patched, "ink", "common only", "")
    assert [p.id for p in session.editor_packs] == ["common"]
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "auto", "common")
    assert "No common pack is in use" in report


def test_save_records_packs_and_load_warns_when_they_differ(live, folders):
    patched, session = live
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "")
    name = session.save_point("with-packs")
    _pack(folders["kilora"], "common", "1.1.0")
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "")
    assert session.editor_packs[0].version == "1.1.0"
    message = session.load(name)
    assert "Now 1.1.0 is in use" in message
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "auto", "")
    assert "The loaded save used: common 1.0.0, style 1.0.0." in report
    assert "Warning:" in report
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "pinned", "")
    assert session.editor_packs[0].version == "1.0.0" and "Warning:" not in report


def test_no_packs_and_bad_files(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setattr(N, "live_env", lambda: patching.LiveEnv(str(tmp_path / "saves")))
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    (folders["kilora"] / f"broken{ep.PACK_SUFFIX}").write_bytes(b"not a pack")
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    _, report = N.KiLoRALivePacks().choose(patched, "", "auto", "")
    assert "none installed" in report and "not a safetensors file" in report
    with pytest.raises(KiLoRAError, match="no live mod"):
        N.KiLoRALivePacks().choose(patched, "other", "auto", "")


def test_several_editors_need_editor_kind(live, folders):
    patched, session = live
    _pack(folders["kilora"], "other-common", "1.0.0", kind="other-editor")
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "auto", "")
    assert "Packs of several editors" in report and session.editor_packs == []
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "", editor_kind="example-editor")
    assert [p.id for p in session.editor_packs] == ["common", "style"]


def test_the_families_input_chooses_family_packs(live):
    patched, session = live
    _, report = N.KiLoRALivePacks().choose(patched, "ink", "auto", "", families="texture")
    assert [p.id for p in session.editor_packs] == ["common", "texture"]
    assert "[on] texture" in report
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "", families="texture, style")
    assert [p.id for p in session.editor_packs] == ["common", "texture", "style"]


def test_auto_takes_every_family_pack_for_a_mod_without_one(
    denoiser,
    folders,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(N, "live_env", lambda: patching.LiveEnv(str(tmp_path / "saves")))
    build_live_mod(denoiser, folders["kilora"] / "vroid.kmod", family="vroid")
    _pack(folders["kilora"], "common", "1.0.0")
    _pack(folders["kilora"], "style", "1.0.0", "family", ["style"])
    _pack(folders["kilora"], "texture", "1.0.0", "family", ["texture"])
    (chain,) = N.KiLoRASelect().select("vroid.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    session = patching.sessions_of(patched)[0]
    N.KiLoRALivePacks().choose(patched, "", "auto", "")
    assert [p.id for p in session.editor_packs] == ["common", "style", "texture"]
