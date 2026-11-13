# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from live_helpers import build_live_mod
from test_comfyui import denoiser, folders

from kilora.editors import WriteRule, commit_write
from kilora.live import open_session, read_save
from kilora.recording import GenerationRecord


class _Nudge(WriteRule):
    needs_path = False

    def next_state(self, state, record):
        return {t: {**p, "up": p["up"] + 0.01} for t, p in state.items()}


def _record(i):
    gen = torch.Generator().manual_seed(i)
    cond = {"c_crossattn": torch.randn(1, 3, 4, generator=gen), "note": "text is not kept"}
    return GenerationRecord(latent=torch.randn(1, 4, 8, 8, generator=gen), conditioning=cond)


def _session(denoiser, folders, tmp_path, keep=True, saves=True):
    path = folders["kilora"] / "ink.kmod"
    build_live_mod(denoiser, path)
    s = open_session(path, saves_dir=(tmp_path / "saves") if saves else None)
    s.learning = True
    s.set_options(keep_previous=keep)
    return path, s


def _same(a, b):
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_off_by_default(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.delenv("KILORA_KEEP_PREVIOUS", raising=False)
    path = folders["kilora"] / "ink.kmod"
    build_live_mod(denoiser, path)
    s = open_session(path)
    s.learning = True
    assert s.keep_previous is False
    commit_write(s, _Nudge(), _record(1))
    assert s.previous is None
    monkeypatch.setenv("KILORA_KEEP_PREVIOUS", "1")
    assert open_session(path).keep_previous is True


def test_commit_keeps_the_generation_and_undo_restores(denoiser, folders, tmp_path):
    _, s = _session(denoiser, folders, tmp_path)
    commit_write(s, _Nudge(), _record(1))
    first = s.previous_snapshot()
    assert set(first) == {"latent", "cond/c_crossattn"}
    assert torch.equal(first["latent"], _record(1).latent)
    commit_write(s, _Nudge(), _record(2))
    assert torch.equal(s.previous["latent"], _record(2).latent)
    s.undo()
    assert _same(first, s.previous)
    s.undo()
    assert s.previous is None


def test_reset_clears_and_undo_brings_back(denoiser, folders, tmp_path):
    _, s = _session(denoiser, folders, tmp_path)
    commit_write(s, _Nudge(), _record(1))
    kept = s.previous_snapshot()
    s.reset()
    assert s.previous is None
    s.undo()
    assert _same(kept, s.previous)


def test_save_and_reload_in_a_fresh_session(denoiser, folders, tmp_path):
    path, s = _session(denoiser, folders, tmp_path)
    commit_write(s, _Nudge(), _record(3))
    name = s.save_point("p")
    save = read_save(tmp_path / "saves" / s.mod_id / f"{name}.ksave")
    assert save.manifest["previous"]["feature"] == "live.previous"
    fresh = open_session(path, saves_dir=tmp_path / "saves")
    fresh.set_options(keep_previous=True)
    fresh.load(name)
    assert _same(s.previous, fresh.previous)
    fresh.undo()
    assert fresh.previous is None
    off = open_session(path, saves_dir=tmp_path / "saves")
    off.load(name)
    assert off.previous is None


def test_memory_saves_and_turning_the_flag_off(denoiser, folders, tmp_path):
    _, s = _session(denoiser, folders, tmp_path, saves=False)
    commit_write(s, _Nudge(), _record(4))
    name = s.save_point("m")
    kept = s.previous_snapshot()
    commit_write(s, _Nudge(), _record(5))
    s.load(name)
    assert _same(kept, s.previous)
    s.set_options(keep_previous=False)
    assert s.previous is None
