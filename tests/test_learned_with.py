# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import torch
from live_helpers import build_live_mod
from safetensors import safe_open
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders, write_mod
from test_path_write import fresh_sessions

from kilora.card import card_fields, card_markdown
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.live import SESSIONS, read_save
from kilora.modfile import read_manifest


def _apply(denoiser, chain_files, strengths=None):
    chain = None
    for i, name in enumerate(chain_files):
        s = (strengths or [1.0] * len(chain_files))[i]
        (chain,) = N.KiLoRASelect().select(name, s, 1.0, chain)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1)
    return patched, patching.sessions_of(patched)[0]


def _setup(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    write_mod(denoiser, folders["kilora"], "parchment", seed=3)
    return "ink.kmod", "parchment-1.0.0.kmod"


def test_save_records_the_composition(denoiser, folders):
    ink, parchment = _setup(denoiser, folders)
    patched, s = _apply(denoiser, [ink, parchment], [1.0, 0.7])
    digest = read_manifest(folders["kilora"] / parchment)["digest"]
    assert s.composition == [{"id": "parchment", "digest": digest, "strength": 0.7, "depth": 1}]
    name = s.save_point("mix")
    save = read_save(folders["kilora"] / "saves" / s.mod_id / f"{name}.ksave")
    assert save.manifest["learned_with"] == s.composition
    assert s.load(name) == f"Loaded {name}."


def test_load_in_another_composition_gives_a_plain_note(denoiser, folders, caplog):
    ink, parchment = _setup(denoiser, folders)
    patched, s = _apply(denoiser, [ink, parchment])
    name = s.save_point("mix")
    SESSIONS.clear()
    patched, alone = _apply(denoiser, [ink])
    assert alone.composition == []
    with caplog.at_level("INFO", logger="kilora.live"):
        message = alone.load(name)
    assert "learned together with parchment; the mods loaded now differ" in message
    assert "learned with" in caplog.text


def test_old_saves_without_the_field_load_without_a_note(denoiser, folders):
    ink, parchment = _setup(denoiser, folders)
    patched, s = _apply(denoiser, [ink, parchment])
    name = s.save_point("old")
    path = folders["kilora"] / "saves" / s.mod_id / f"{name}.ksave"
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata())
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    m = json.loads(meta["kilora.manifest"])
    m.pop("learned_with")
    from kilora.modfile import compute_digest

    m["digest"] = compute_digest(m, tensors)
    meta["kilora.manifest"] = json.dumps(m)
    save_file(tensors, str(path), metadata=meta)
    assert s.load(name) == f"Loaded {name}."


def test_freeze_copies_the_list_into_the_card(denoiser, folders, tmp_path):
    ink, parchment = _setup(denoiser, folders)
    patched, s = _apply(denoiser, [ink, parchment], [1.0, 0.5])
    with torch.no_grad():
        for parts in s.state.values():
            parts["up"].add_(0.01)
    s.freeze(tmp_path / "frozen.kmod", "ink-frozen")
    m = read_manifest(tmp_path / "frozen.kmod")
    assert m["learned_with"] == s.composition
    fields = card_fields(m)
    assert fields["learned_with"][0]["id"] == "parchment"
    assert "| Learned together with | parchment (strength 0.5, depth 1) |" in card_markdown(fields)
