# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from live_helpers import build_live_mod, write
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.live import SESSIONS


@pytest.fixture(autouse=True)
def fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def latent():
    return {"samples": torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(5))}


def _session(model, mod_id="ink"):
    return [s for s in patching.sessions_of(model) if s.mod_id == mod_id][0]


def test_write_undo_and_apply_again(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    patcher = FakePatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    applied, report, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    assert "live ink: learning on, speed 0.5, 0 commits" in report
    x = latent()["samples"]
    before = applied.run(x)
    rev = N.KiLoRAApply.IS_CHANGED(sets=chain)
    write(_session(applied), size=0.2)
    assert N.KiLoRAApply.IS_CHANGED(sets=chain) != rev
    applied, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    after = applied.run(x)
    assert not torch.allclose(after, before)
    assert (folders["kilora"] / "saves" / "ink" / "autosave-1.ksave").exists()
    undone, _ = N.KiLoRALiveSave().act(applied, "undo", "", "all")["result"]
    torch.testing.assert_close(undone.run(x), before)
    again, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    torch.testing.assert_close(again.run(x), before)


def test_save_point_load_options_and_freeze(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    patcher = FakePatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    model, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    write(_session(model), size=0.2)
    model, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    model, text = N.KiLoRALiveSave().act(model, "save point", "warm", "ink")["result"]
    assert "Saved save-warm." in text
    x = latent()["samples"]
    want = model.run(x)
    model, _ = N.KiLoRALiveSave().act(model, "reset", "", "all")["result"]
    model, text = N.KiLoRALiveSave().act(model, "load", "save-warm", "all")["result"]
    torch.testing.assert_close(model.run(x), want)
    model, text = N.KiLoRALiveOptions().set(model, "ink", False, -1)
    assert "Learning off" in text and "learning off" in text
    result = N.KiLoRAFreeze().freeze(model, "ink", "Ink Warm")
    assert "Saved ink-warm.tiny.kmod.safetensors" in result["result"][0]
    (frozen_chain,) = N.KiLoRASelect().select("ink-warm.tiny.kmod.safetensors", 1.0, 1.0)
    frozen, report, _ = N.KiLoRAApply().apply(patcher, frozen_chain, 2)
    assert "live ink-warm" not in report and "learned +1" in report
    torch.testing.assert_close(frozen.run(x), want)


def test_two_live_mods_report_shared_hook_points(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod", mod_id="ink", seed=1)
    build_live_mod(denoiser, folders["kilora"] / "parchment.kmod", mod_id="parchment", seed=2)
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    (chain,) = N.KiLoRASelect().select("parchment.kmod", 1.0, 1.0, sets=chain)
    model, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 2)
    assert "ink and parchment both learn at 1 hook points" in report
    assert "each keeps its own live state" in report
    ink, parchment = _session(model, "ink"), _session(model, "parchment")
    write(ink)
    assert len(ink.commits) == 1 and not parchment.commits


def test_live_export_node(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    patcher = FakePatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    model, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    write(_session(model), size=0.2)
    model, _, _ = N.KiLoRAApply().apply(patcher, chain, 2)
    report = N.KiLoRALiveExport().export(model, "ink", False, "")["result"][0]
    assert "ink-share.tiny.kmod.safetensors" in report and "not what it learned" in report
    shared = folders["kilora"] / "exports" / "ink-share.tiny.kmod.safetensors"
    assert shared.read_bytes() == (folders["kilora"] / "ink.kmod").read_bytes()
    with pytest.raises(Exception, match="set new_id"):
        N.KiLoRALiveExport().export(model, "ink", True, "")
    report = N.KiLoRALiveExport().export(model, "ink", True, "Ink Mine")["result"][0]
    assert "Warning:" in report and "ink-mine.tiny.kmod.safetensors" in report
