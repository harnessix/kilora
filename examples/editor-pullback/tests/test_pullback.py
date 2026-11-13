# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import kilora_pullback as pb
import pytest
import torch

from kilora.check import check_file
from kilora.editors import commit_write, editor_for, register_editor, unregister_editor
from kilora.errors import KiLoRAError
from kilora.live import open_session
from kilora.recording import GenerationRecord
from kilora.testing import TINY_FAMILY, make_live_mod


@pytest.fixture
def setup(tmp_path):
    register_editor(pb.PLUGIN, "test")
    make_live_mod(tmp_path / "live.kmod")
    editor_file = tmp_path / "pullback.kedit.safetensors"
    pb.save_pullback_editor(editor_file, rate=0.5, base_family=TINY_FAMILY)
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    session.state = {t: {**p, "up": p["up"] + 1.0} for t, p in session.state.items()}
    yield session, editor_file
    unregister_editor(pb.KIND)


def write(session, editor_file):
    editor = editor_for(editor_file)
    editor.check(session)
    return commit_write(session, editor, GenerationRecord(images=1))


def gap(session):
    return max(
        float((session.state[t]["up"] - session.clean[t]["up"]).abs().max()) for t in session.state
    )


def test_write_moves_halfway_back(setup):
    session, editor_file = setup
    assert gap(session) == pytest.approx(1.0, abs=1e-5)
    assert "1 commits" in write(session, editor_file)
    assert gap(session) == pytest.approx(0.5, abs=1e-5)
    assert float(session.editor_state.count) == 1.0


def test_undo_restores_live_state_and_editor_state(setup):
    session, editor_file = setup
    write(session, editor_file)
    write(session, editor_file)
    assert float(session.editor_state.count) == 2.0
    session.undo()
    assert gap(session) == pytest.approx(0.5, abs=1e-5)
    assert float(session.editor_state.count) == 1.0


def test_save_point_keeps_the_editor_state(setup, tmp_path):
    session, editor_file = setup
    write(session, editor_file)
    name = session.save_point("one")
    write(session, editor_file)
    session.load(name)
    assert gap(session) == pytest.approx(0.5, abs=1e-5)
    assert float(session.editor_state.count) == 1.0


def test_deterministic(setup, tmp_path):
    session, editor_file = setup
    write(session, editor_file)
    first = {t: p["up"].clone() for t, p in session.state.items()}
    session.undo()
    write(session, editor_file)
    assert all(torch.equal(session.state[t]["up"], first[t]) for t in first)


def test_learning_off_means_no_call(setup):
    session, editor_file = setup
    session.learning = False
    before = {t: p["up"].clone() for t, p in session.state.items()}
    write(session, editor_file)
    assert all(torch.equal(session.state[t]["up"], before[t]) for t in before)


def test_wrong_base_model_and_bad_files(setup, tmp_path):
    session, _ = setup
    other = tmp_path / "other.kedit.safetensors"
    pb.save_pullback_editor(other, rate=0.5, base_family="z-image")
    with pytest.raises(KiLoRAError, match="This editor is for z-image"):
        editor_for(other).check(session)
    with pytest.raises(ValueError):
        pb.save_pullback_editor(tmp_path / "x.kedit", rate=2.0, base_family="z-image")


def test_kilora_check_reports_the_editor(setup):
    _, editor_file = setup
    r = check_file(editor_file)
    assert r.kind == "editor (pullback)" and "rate 0.5" in r.facts


def test_runs_without_gradients(setup):
    session, editor_file = setup
    for parts in session.state.values():
        parts["up"].requires_grad_(True)
    write(session, editor_file)
    assert all(not p["up"].requires_grad for p in session.state.values())
