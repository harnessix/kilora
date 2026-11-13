# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import torch
from safetensors.torch import save_file
from test_comfyui import denoiser, folders
from test_path_write import _setup, fresh_sessions

from kilora.check import check_file
from kilora.cli import main
from kilora.live import open_session
from kilora.modfile import load_mod


def _files(denoiser, folders, tmp_path):
    _, _, _, session, _, _ = _setup(denoiser, folders, tmp_path)
    mod = folders["kilora"] / "ink.kmod"
    path_editor = tmp_path / "path.kedit.safetensors"
    student = None
    live = open_session(mod, saves_dir=tmp_path / "saves")
    save = tmp_path / "saves" / live.mod_id / f"{live.save_point('a')}.ksave"
    return mod, path_editor, student, save


def test_each_kind_passes(denoiser, folders, tmp_path):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    kinds = {
        mod: "live mod",
        path_editor: "editor (path-write)",
        save: "live state (save)",
    }
    for path, kind in kinds.items():
        r = check_file(path, mod=None if path == mod else mod)
        assert r.ok, r.lines()
        assert r.kind == kind
        assert any(f.startswith("size:") for f in r.facts)
        assert any(f.startswith("hook layers:") for f in r.facts)
        if path != mod:
            assert "fits ink.kmod" in r.facts


def test_problems_have_plain_messages(denoiser, folders, tmp_path):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    r = check_file(tmp_path / "missing.kmod")
    assert r.problems == ["The file does not exist."]
    (tmp_path / "empty.kmod").write_bytes(b"")
    assert check_file(tmp_path / "empty.kmod").problems == ["The file is empty."]
    (tmp_path / "text.kmod").write_text("hello world, not a model")
    assert "not a safetensors file" in check_file(tmp_path / "text.kmod").problems[0]
    save_file({"x": torch.zeros(1)}, str(tmp_path / "plain.safetensors"))
    assert "no KiLoRA manifest" in check_file(tmp_path / "plain.safetensors").problems[0]
    meta = {"kilora.manifest": json.dumps({"format": "other"})}
    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"), metadata=meta)
    assert "Unknown file format" in check_file(tmp_path / "other.safetensors").problems[0]
    data = bytearray(mod.read_bytes())
    data[-5] ^= 0xFF
    (tmp_path / "damaged.kmod").write_bytes(bytes(data))
    assert "damaged" in check_file(tmp_path / "damaged.kmod").problems[0]


def test_fit_against_another_mod(denoiser, folders, tmp_path):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    m = load_mod(mod)
    from kilora.modfile import save_mod

    other = tmp_path / "other.kmod"
    fields = {
        k: v
        for k, v in m.manifest.items()
        if k not in ("format", "format_version", "requires_features", "adapter", "digest")
    }
    fields["base_model"] = {"family": "z-image"}
    save_mod(other, m.sets, m.settings, **fields)
    r = check_file(path_editor, mod=other)
    assert not r.ok and "is for" in r.problems[0] and "z-image" in r.problems[0]
    static = tmp_path / "static.kmod"
    fields.pop("live")
    save_mod(static, [m.sets[0]], m.settings, **fields)
    assert "is a static mod" in check_file(save, mod=static).problems[0]


def test_license_and_hash_warnings(denoiser, folders, tmp_path):
    mod, *_ = _files(denoiser, folders, tmp_path)
    m = load_mod(mod)
    from kilora.modfile import save_mod

    fields = {
        k: v
        for k, v in m.manifest.items()
        if k not in ("format", "format_version", "requires_features", "adapter", "digest")
    }
    fields["license"] = "NOASSERTION"
    fields["base_model"] = {"family": m.manifest["base_model"]["family"], "sha256": "abc"}
    save_mod(tmp_path / "w.kmod", m.sets, m.settings, **fields)
    r = check_file(tmp_path / "w.kmod")
    assert any("license field is empty" in w for w in r.warnings)
    assert any("not a SHA-256 hash" in p for p in r.problems)


def test_cli(denoiser, folders, tmp_path, capsys):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    assert main(["check", str(path_editor), "--mod", str(mod)]) == 0
    out = capsys.readouterr().out
    assert "editor (path-write), OK" in out and "fits ink.kmod" in out
    assert main(["check", str(tmp_path / "nothing.kmod")]) == 1
    assert "NOT USABLE" in capsys.readouterr().out


def _with_supports(src, dst, supports):
    from safetensors import safe_open

    with safe_open(str(src), framework="pt") as f:
        meta = dict(f.metadata())
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    manifest = json.loads(meta["kilora.manifest"])
    manifest["supports"] = supports
    meta["kilora.manifest"] = json.dumps(manifest)
    save_file(tensors, str(dst), metadata=meta)
    return dst


def _live_targets(mod):
    return load_mod(mod).manifest["live"]["layout_spec"]["targets"]


def test_old_editor_without_supports_is_rank_four(denoiser, folders, tmp_path):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    r = check_file(path_editor, mod=mod)
    assert r.ok, r.lines()
    fact = next(f for f in r.facts if f.startswith("supports"))
    hooks = len(_live_targets(mod))
    assert "live rank 4 to 4" in fact and f"{hooks} hook points" in fact
    assert "treated as live rank 4" in fact


def test_rank_general_supports(denoiser, folders, tmp_path):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    hooks = _live_targets(mod)
    wide = {"live_rank": [1, 64], "columns": [1, 1024], "hooks": hooks}
    r = check_file(_with_supports(path_editor, tmp_path / "g.kedit", wide), mod=mod)
    assert r.ok, r.lines()
    assert any("live rank 1 to 64" in f and f"{len(hooks)} hook points" in f for f in r.facts)
    high = {**wide, "live_rank": [8, 16]}
    r = check_file(_with_supports(path_editor, tmp_path / "h.kedit", high), mod=mod)
    assert any("has live rank 4. The editor supports ranks 8 to 16." in p for p in r.problems)
    few = {**wide, "hooks": ["layers.9.attention.to_out.0"]}
    r = check_file(_with_supports(path_editor, tmp_path / "f.kedit", few), mod=mod)
    assert any("does not support" in p and "hook points" in p for p in r.problems)
    narrow = {**wide, "columns": [1, 2]}
    r = check_file(_with_supports(path_editor, tmp_path / "n.kedit", narrow), mod=mod)
    assert any("columns in A" in p for p in r.problems)


def test_declared_mod_rank_is_used(denoiser, folders, tmp_path):
    from kilora.modfile import save_mod

    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    m = load_mod(mod)
    fields = {
        k: v
        for k, v in m.manifest.items()
        if k not in ("format", "format_version", "requires_features", "adapter", "digest")
    }
    fields["live"] = {**fields["live"], "rank": 32}
    save_mod(tmp_path / "r32.kmod", m.sets, m.settings, **fields)
    supports = {"live_rank": [1, 16], "columns": [1, 1024], "hooks": _live_targets(mod)}
    editor = _with_supports(path_editor, tmp_path / "g.kedit", supports)
    r = check_file(editor, mod=tmp_path / "r32.kmod")
    assert any("has live rank 32. The editor supports ranks 1 to 16." in p for p in r.problems)


def test_cli_prints_rank_problems(denoiser, folders, tmp_path, capsys):
    mod, path_editor, student, save = _files(denoiser, folders, tmp_path)
    supports = {"live_rank": [8, 16], "columns": [1, 1024], "hooks": _live_targets(mod)}
    editor = _with_supports(path_editor, tmp_path / "h.kedit", supports)
    assert main(["check", str(editor), "--mod", str(mod)]) == 1
    out = capsys.readouterr().out
    assert "problem: Hook point" in out and "supports ranks 8 to 16" in out


def test_warns_about_one_expert_with_derived_state_keys(tmp_path):
    import dataclasses

    from conftest import TARGET

    from kilora import KiLoRAConfig, attach
    from kilora.adapter import Settings
    from kilora.modfile import save_mod
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser(seed=0).eval()
    for experts, same_keys, name in ((1, False, "one"), (3, True, "same"), (3, False, "ok")):
        adapter = attach(
            model, KiLoRAConfig(target_modules=[TARGET], num_experts=experts, router_dim=8)
        )
        (s,) = adapter.to_sets()
        adapter.remove()
        if same_keys:
            s = dataclasses.replace(
                s,
                experts={
                    t: dataclasses.replace(ex, keys_in=ex.keys_in[:1].expand_as(ex.keys_in).clone())
                    for t, ex in s.experts.items()
                },
            )
        save_mod(tmp_path / f"{name}.kmod", [s], Settings(depth=1), id=name, license="MIT")
        r = check_file(tmp_path / f"{name}.kmod")
        text = " ".join(r.warnings)
        assert r.ok
        assert ("1 expert per layer" in text) == (name == "one")
        assert "keys of the set" not in text
        assert "not trained" not in text
