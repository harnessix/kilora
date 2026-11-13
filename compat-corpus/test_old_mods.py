# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kilora import compose, load_mod
from kilora.testing import TinyDenoiser

ROOT = Path(__file__).parent
ENTRIES = sorted(p.parent for p in ROOT.glob("*/*/entry.json"))


def _kind(folder):
    return json.loads((folder / "entry.json").read_text()).get("kind", "mod")


MOD_ENTRIES = [p for p in ENTRIES if _kind(p) == "mod"]
PACK_ENTRIES = [p for p in ENTRIES if _kind(p) == "pack"]
LIVE_ENTRIES = [p for p in ENTRIES if _kind(p) == "live"]
MODELS = {"TinyDenoiser": TinyDenoiser}


@pytest.mark.old_mod_corpus
@pytest.mark.parametrize("folder", MOD_ENTRIES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_corpus_entry(folder):
    entry = json.loads((folder / "entry.json").read_text())
    model = MODELS[entry["model"]["class"]](**entry["model"]["args"]).eval()
    mod_path = folder / entry["mod"]
    load_mod(mod_path)
    reference = load_file(str(folder / "reference.safetensors"))
    adapter = compose(model, [mod_path])
    by_depth = entry["tolerance"].get("by_depth", {})
    with torch.no_grad():
        for depth in entry["depths"]:
            tol = by_depth.get(str(depth), entry["tolerance"]["relative"])
            with adapter.depth(depth):
                got = model(reference["input"])
            want = reference[f"output.depth{depth}"]
            err = ((got - want).norm() / want.norm()).item()
            assert err <= tol, f"depth {depth}: relative error {err:.2e} > {tol:.0e}"


@pytest.mark.old_mod_corpus
@pytest.mark.parametrize("folder", PACK_ENTRIES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_corpus_pack(folder):
    from kilora.modlist import load_mods
    from kilora.packs import load_pack, pack_choices, scan_library

    entry = json.loads((folder / "entry.json").read_text())
    model = MODELS[entry["model"]["class"]](**entry["model"]["args"]).eval()
    pack = load_pack(folder / entry["pack"])
    library = scan_library([(folder / p).resolve() for p in entry["library"]])
    reference = load_file(str(folder / "reference.safetensors"))
    baked, _, _ = load_mods(model, pack_choices(pack, library), depth=pack.depth)
    with torch.no_grad():
        got = model(reference["input"])
    baked.remove()
    want = reference["output"]
    err = ((got - want).norm() / want.norm()).item()
    assert err <= entry["tolerance"]["relative"]


def _rel(got, want):
    return ((got - want).norm() / want.norm()).item()


@pytest.mark.old_mod_corpus
@pytest.mark.parametrize("folder", LIVE_ENTRIES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_corpus_live_mod(folder, tmp_path):
    import shutil

    from kilora.live import open_session
    from kilora.testing import write_steps

    entry = json.loads((folder / "entry.json").read_text())
    model = MODELS[entry["model"]["class"]](**entry["model"]["args"]).eval()
    reference = load_file(str(folder / "reference.safetensors"))
    tol = entry["tolerance"]["relative"]

    session = open_session(folder / entry["mod"], saves_dir=tmp_path / "new")
    write_steps(session, entry["writes"]["seeds"], entry["writes"]["size"])
    for t, parts in session.state.items():
        for part, v in parts.items():
            assert _rel(v, reference[f"state/{t}/{part}"]) <= tol
    st = session.mod.settings
    routing = dict(routing_scope=st.routing_scope, routing_select=st.routing_select)
    adapter = compose(model, session.sets(), depth=entry["depth"], **routing)
    with torch.no_grad():
        assert _rel(model(reference["input"]), reference["output"]) <= tol
    adapter.remove()

    saves = tmp_path / "copy"
    shutil.copytree(folder / "saves", saves)
    loaded = open_session(folder / entry["mod"], saves_dir=saves)
    loaded.load(Path(entry["save"]).stem)
    adapter = compose(model, loaded.sets(), depth=entry["depth"], **routing)
    with torch.no_grad():
        assert _rel(model(reference["input"]), reference["output"]) <= tol
    adapter.remove()


def test_corpus_is_not_empty():
    assert MOD_ENTRIES and PACK_ENTRIES and LIVE_ENTRIES
