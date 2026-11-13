# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from live_helpers import build_live_mod
from test_chunk_live import _packs
from test_comfyui import FakeBaseModel, denoiser, folders
from test_path_write import WrapPatcher, fresh_sessions

from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.live import SESSIONS, read_save

WORKFLOW = {
    "1": {
        "class_type": "KiLoRASelect",
        "inputs": {"file": "ink.kmod", "strength": 1.0, "prior": 1.0},
    },
    "2": {
        "class_type": "KiLoRAApply",
        "inputs": {"model": ["base", 0], "sets": ["1", 0], "depth": 1},
    },
    "3": {
        "class_type": "KiLoRALiveOptions",
        "inputs": {"model": ["2", 0], "mod": "ink", "learning": True, "speed": -1.0},
    },
    "4": {
        "class_type": "KiLoRALivePacks",
        "inputs": {
            "model": ["3", 0],
            "mod": "ink",
            "packs": "auto",
            "turned_off": "",
            "editor_kind": "",
            "measurement": "one-sided",
            "time_bins": "1",
        },
    },
    "5": {
        "class_type": "KiLoRALiveRecord",
        "inputs": {"model": ["4", 0], "editor": "chunk.kedit.safetensors"},
    },
    "6": {"class_type": "KiLoRALiveWrite", "inputs": {"path": ["5", 1], "samples": ["sampler", 0]}},
}
EVENTS = 5


def _run(denoiser, folders, x):
    base = WrapPatcher(FakeBaseModel(denoiser))
    out = {"base": (base,)}
    order = ["1", "2", "3", "4", "5"]
    for node_id in order:
        node = WORKFLOW[node_id]
        cls = N.NODE_CLASS_MAPPINGS[node["class_type"]]
        args = {
            k: (out[v[0]][v[1]] if isinstance(v, list) else v) for k, v in node["inputs"].items()
        }
        result = getattr(cls(), cls.FUNCTION)(**args)
        result = result["result"] if isinstance(result, dict) else result
        out[node_id] = result
    recorded, handle = out["5"]
    samples = recorded.sample(x, steps=2)[-1][: x.shape[0]]
    N.KiLoRALiveWrite().write(handle, {"samples": samples})
    return patching.sessions_of(out["2"][0])[0]


def _setup(denoiser, folders):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod", family="retro_game")
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(WrapPatcher(FakeBaseModel(denoiser)), chain, 1)
    session = patching.sessions_of(patched)[0]
    _packs(session, folders["kilora"])
    SESSIONS.clear()


def _state(session):
    up = {t: p["up"].clone() for t, p in session.state.items()}
    return up, session.editor_snapshot()


def _bits(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _same(a, b):
    (ua, ea), (ub, eb) = a, b
    return (
        all(_bits(ua[t]) == _bits(ub[t]) for t in ua)
        and ea.keys() == eb.keys()
        and all(_bits(ea[k]) == _bits(eb[k]) for k in ea)
    )


def _generations(denoiser, folders, monkeypatch, asynchronous):
    monkeypatch.setenv("KILORA_LIVE_ASYNC", "1" if asynchronous else "0")
    SESSIONS.clear()
    states = []
    for i in range(EVENTS):
        x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(100 + i))
        session = _run(denoiser, folders, x)
        q = getattr(session, "write_queue", None)
        if q is not None:
            assert q.wait_idle(60)
        states.append(_state(session))
    return session, states


@pytest.mark.async_live
def test_saved_workflow_with_packs_async(denoiser, folders, monkeypatch):
    _setup(denoiser, folders)
    ref, want = _generations(denoiser, folders, monkeypatch, asynchronous=False)
    session, got = _generations(denoiser, folders, monkeypatch, asynchronous=True)
    assert len(session.commits) == EVENTS
    assert [p.id for p in session.editor_packs] == ["chunk-common", "chunk-retro-game"]
    assert session.editor_options == {"diff": "one-sided", "time_bins": 1}
    assert all(_same(a, b) for a, b in zip(got, want, strict=True))

    with session.write_queue.exclusive():
        session.undo()
    ref.undo()
    assert _same(_state(session), _state(ref))
    name = session.save_point("five")
    save = read_save(folders["kilora"] / "saves" / session.mod_id / f"{name}.ksave")
    recorded = [r["id"] for r in save.manifest["editor_packs"]]
    assert recorded == ["chunk-common", "chunk-retro-game"]

    SESSIONS.clear()
    monkeypatch.setenv("KILORA_LIVE_ASYNC", "0")
    base = WrapPatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(base, chain, 1)
    fresh = patching.sessions_of(patched)[0]
    N.KiLoRALiveRecord().record(patched, "chunk.kedit.safetensors")
    assert "Loaded" in fresh.load(name)
    assert _same(_state(fresh), _state(session))
    assert json.loads(json.dumps(WORKFLOW)) == WORKFLOW
