# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from live_helpers import build_live_mod, write

from kilora import KiLoRAError, compose, load_mod, read_manifest
from kilora.live import open_session, read_save, write_save


def output(model, x, sets):
    adapter = compose(model, sets, depth=2, softmax_scope="per_set")
    with torch.no_grad():
        out = model(x)
    adapter.remove()
    return out


def test_live_mod_manifest(model, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    m = read_manifest(path)
    assert m["requires_features"] == ["experts.lora", "live.state"]
    assert m["live"]["layout"] == layout.id == "kilora-tiny-1-light-1"
    assert m["mod_family"] == "style"
    mod = load_mod(path)
    assert [s.extra.get("live", False) for s in mod.sets] == [False, True]
    assert [s.part_type for s in mod.sets] == [("lora",), ("lora",)]


def test_write_commit_undo_reset(model, x, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path, saves_dir=tmp_path / "saves")
    s.learning = True
    clean_out = output(model, x, s.sets())
    assert "1 commits" in write(s)
    written = output(model, x, s.sets())
    assert not torch.allclose(written, clean_out)
    assert (tmp_path / "saves" / "ink" / "autosave-1.ksave").exists()
    save = read_save(tmp_path / "saves" / "ink" / "autosave-1.ksave")
    assert save.manifest["format"] == "kilora-save" and len(save.manifest["commits"]) == 1
    assert save.manifest["editor_state"] is None
    assert save.manifest["previous"]["keys"] == []
    s.undo()
    torch.testing.assert_close(output(model, x, s.sets()), clean_out)
    write(s, seed=1)
    s.reset()
    torch.testing.assert_close(output(model, x, s.sets()), clean_out)
    s.undo()
    assert len(s.commits) == 1


def test_learning_off_and_invalid_writes_change_nothing(model, x, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path)
    s.learning = False
    before = output(model, x, s.sets())
    write(s)
    s.learning = True
    write(s, value=float("nan"))
    assert s.pending is None and not s.commits
    torch.testing.assert_close(output(model, x, s.sets()), before)
    with pytest.raises(KiLoRAError, match="no pending update"):
        s.commit()


def test_save_points_and_loading(model, x, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path, saves_dir=tmp_path / "saves")
    s.learning = True
    write(s)
    name = s.save_point("my look")
    want = output(model, x, s.sets())
    s.reset()
    assert name in s.list_saves()
    s.load(name)
    torch.testing.assert_close(output(model, x, s.sets()), want)
    again = open_session(path, saves_dir=tmp_path / "saves")
    again.load(name)
    torch.testing.assert_close(output(model, x, again.sets()), want)
    assert len(again.commits) == 1


def test_save_from_other_mod_version_warns(model, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path, saves_dir=tmp_path / "saves")
    s.learning = True
    write(s)
    s.save_point("a")
    build_live_mod(model, tmp_path / "ink.kmod", seed=3)
    newer = open_session(tmp_path / "ink.kmod", saves_dir=tmp_path / "saves")
    assert "another version of the mod" in newer.load("save-a")


def test_freeze_writes_a_static_mod(model, x, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path)
    s.learning = True
    write(s)
    want = output(model, x, s.sets())
    with pytest.raises(KiLoRAError, match="new ID"):
        s.freeze(tmp_path / "f.kmod", "ink")
    s.freeze(tmp_path / "frozen.kmod", "ink-frozen")
    frozen = load_mod(tmp_path / "frozen.kmod")
    assert "live" not in frozen.manifest and all(x.has_lora for x in frozen.sets)
    assert frozen.manifest["based_on"] == "ink 0.1.0"
    assert frozen.manifest["requires_features"] == ["experts.lora"]
    torch.testing.assert_close(output(model, x, frozen.sets), want)


def test_hooked_and_baked_models_agree_on_the_live_state(model, x, tmp_path):
    from kilora.bake import bake

    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path)
    s.learning = True
    write(s, size=0.2)
    hooked = output(model, x, s.sets())
    handle = bake(model, s.sets(), depth=2)
    with torch.no_grad():
        baked = model(x)
    handle.remove()
    torch.testing.assert_close(hooked, baked, rtol=1e-4, atol=1e-5)


def test_probe_drift_offers_rollback(model, x, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    s = open_session(path)
    s.learning = True

    def probe(sets):
        return output(model, x, sets)

    drift, offer = s.probe_drift(probe, threshold=1e-6)
    assert drift == 0 and not offer
    write(s, size=0.2)
    drift, offer = s.probe_drift(probe, threshold=1e-6)
    assert drift > 0 and offer
    s.set_probe_reference(probe)
    assert s.probe_drift(probe)[0] == 0


def test_export_keeps_learned_state_out_unless_asked(model, x, tmp_path):
    from kilora.cli import main as cli_main
    from kilora.live import LEARNED_STATE_WARNING

    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod.safetensors")
    s = open_session(path, saves_dir=tmp_path / "saves")
    clean_out = output(model, x, s.sets())
    write(s, size=0.2)
    s.save_point("mine")
    learned_out = output(model, x, s.sets())

    shared = tmp_path / "share.kmod.safetensors"
    s.export(shared)
    assert shared.read_bytes() == path.read_bytes()
    torch.testing.assert_close(output(model, x, open_session(shared).sets()), clean_out)

    with pytest.raises(KiLoRAError, match="new ID"):
        s.export(tmp_path / "x.kmod.safetensors", include_learned_state=True)
    mine = tmp_path / "mine.kmod.safetensors"
    s.export(mine, include_learned_state=True, new_id="ink-mine")
    m = load_mod(mine).manifest
    assert m["id"] == "ink-mine" and m["based_on"] == "ink 0.1.0"
    assert (
        m["learned_state"]["commits"] == 1 and m["learned_state"]["note"] == LEARNED_STATE_WARNING
    )
    torch.testing.assert_close(output(model, x, open_session(mine).sets()), learned_out)

    out = tmp_path / "cli.kmod.safetensors"
    assert cli_main(["export-live", str(path), "--output", str(out)]) == 0
    assert out.read_bytes() == path.read_bytes()
    assert (
        cli_main(
            [
                "export-live",
                str(path),
                "--output",
                str(out),
                "--include-learned-state",
                "--saves-dir",
                str(tmp_path / "saves"),
                "--save",
                "save-mine",
                "--new-id",
                "ink-cli",
            ]
        )
        == 0
    )
    assert load_mod(out).manifest["id"] == "ink-cli"


def test_save_round_trips_editor_and_previous(model, tmp_path):
    path, _, layout = build_live_mod(model, tmp_path / "ink.kmod")
    session = open_session(path)
    save_path = tmp_path / "state.ksave"
    write_save(
        save_path,
        session.state,
        mod=session.mod,
        layout=layout,
        updater_ref={},
        commits=[],
        kind="manual",
        name="test",
        sequence=1,
        editor_state={"count": torch.tensor(3)},
        editor_kind="counter",
        previous={"latent": torch.ones(2)},
    )
    saved = read_save(save_path)
    assert saved.manifest["editor_state"]["kind"] == "counter"
    assert saved.editor_state["count"].shape == ()
    assert saved.editor_state["count"].item() == 3
    assert saved.manifest["previous"]["keys"] == ["latent"]
    torch.testing.assert_close(saved.previous["latent"], torch.ones(2))
