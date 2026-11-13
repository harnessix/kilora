# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
from test_comfyui import denoiser, folders
from test_editor_pack_nodes import live
from test_path_write import WrapPatcher, _setup, fresh_sessions

from kilora.comfyui import nodes as N
from kilora.editors import Editor, EditorPlugin, register_editor, unregister_editor
from kilora.errors import KiLoRAError


class _Spy(Editor):
    seen = []

    def __init__(self, path):
        pass

    def configure(self, options):
        super().configure(options)
        _Spy.seen.append(dict(options))

    def next_state(self, state, record):
        return None


@pytest.fixture
def spy_kind(folders):
    import json

    import torch
    from safetensors.torch import save_file

    register_editor(EditorPlugin("spy-chunk", _Spy))
    path = folders["kilora"] / "spy.kedit.safetensors"
    manifest = json.dumps({"format": "kilora-editor", "kind": "spy-chunk"})
    save_file({"x": torch.zeros(1)}, str(path), metadata={"kilora.manifest": manifest})
    _Spy.seen.clear()
    yield path.name
    unregister_editor("spy-chunk")


def test_settings_reach_the_editor(denoiser, folders, tmp_path, spy_kind):
    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    assert "measurement" not in N.KiLoRALiveOptions.INPUT_TYPES()["optional"]
    opts = N.KiLoRALivePacks.INPUT_TYPES()["optional"]
    assert list(opts) == [
        "editor_kind",
        "measurement",
        "time_bins",
        "families",
        "path_merge",
        "path_angle",
        "cross_layer",
        "keep_merge_pairs",
        "merge_residual",
        "term_budget",
        "explore",
    ]
    N.KiLoRALivePacks().choose(patched, session.mod_id, "auto", "")
    assert session.editor_options == {}
    N.KiLoRALivePacks().choose(patched, session.mod_id, "auto", "", "", "one-sided", "2")
    assert session.editor_options == {"diff": "one-sided", "time_bins": 2}
    N.KiLoRALivePacks().choose(patched, session.mod_id, "auto", "", "", "central")
    assert session.editor_options == {"diff": "central", "time_bins": 2}
    N.KiLoRALivePacks().choose(patched, session.mod_id, "auto", "", path_merge="on")
    assert session.editor_options["path_merge"] is True
    N.KiLoRALivePacks().choose(patched, session.mod_id, "auto", "", path_angle=0.1)
    assert session.editor_options["path_angle"] == 0.1
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    N.KiLoRALiveRecord().record(wp, spy_kind)
    assert _Spy.seen[-1] == session.editor_options


def test_bad_settings_are_refused(denoiser, folders, tmp_path):
    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    with pytest.raises(KiLoRAError, match="central or one-sided"):
        session.set_options(editor_options={"diff": "forward"})
    with pytest.raises(KiLoRAError, match="1 or 2"):
        session.set_options(editor_options={"time_bins": 3})
    with pytest.raises(KiLoRAError, match="path merge"):
        session.set_options(editor_options={"path_merge": "on"})
    with pytest.raises(KiLoRAError, match="path angle"):
        session.set_options(editor_options={"path_angle": float("nan")})
    with pytest.raises(KiLoRAError, match="unknown editor setting"):
        session.set_options(editor_options={"packs": ["common"]})


def test_the_base_editor_keeps_the_settings():
    e = Editor()
    e.configure({"diff": "one-sided"})
    assert e.options == {"diff": "one-sided"} and Editor.options == {}


def test_the_chosen_packs_reach_the_editor(live, spy_kind):
    patched, session = live
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    N.KiLoRALiveRecord().record(wp, spy_kind)
    assert "packs" not in _Spy.seen[-1]
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "", "", "one-sided", "1")
    N.KiLoRALiveRecord().record(wp, spy_kind)
    assert _Spy.seen[-1] == {"diff": "one-sided", "time_bins": 1, "packs": ["common", "style"]}
    N.KiLoRALivePacks().choose(patched, "ink", "auto", "style")
    N.KiLoRALiveRecord().record(wp, spy_kind)
    assert _Spy.seen[-1]["packs"] == ["common"]


def test_import_choices_leave_existing_live_mod_behavior_unchanged(denoiser, folders, tmp_path):
    import torch

    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    before = [patched.run(x) for x in xs]
    out, _ = N.KiLoRALiveOptions().set(patched, session.mod_id, True, -1.0)
    assert all(torch.equal(a, out.run(x)) for a, x in zip(before, xs, strict=True))
    for option in ({"edits": "off"}, {"controller": "rprop"}):
        with pytest.raises(KiLoRAError, match="imported LoRA session"):
            N.KiLoRALiveOptions().set(out, session.mod_id, True, -1.0, **option)
