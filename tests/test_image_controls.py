# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from test_image_terms import MOUNTS
from test_learn_from_images import Editor, Recorder, Session, _items

from kilora import image_terms as it


def _run(session, recorder, items=None, editor=None, **kwargs):
    return it.learn_from_images(
        session,
        _items() if items is None else items,
        "style",
        None,
        samples=recorder.samples,
        losses=recorder.losses,
        drift=recorder.drift,
        editor=editor or Editor(),
        model=session.model,
        **kwargs,
    )


def _changed(session):
    return {t for t, parts in session.state.items() if float(parts["c"].abs().sum()) > 0}


def _same_state(a, b):
    assert set(a.state) == set(b.state)
    for target in a.state:
        assert torch.equal(a.state[target]["c"], b.state[target]["c"]), target


def test_every_control_at_its_default_gives_the_t621_write():
    plain, plain_recorder = Session(), Recorder()
    reference = _run(plain, plain_recorder)
    for controls in (
        {"mounts": None, "axis": None, "mask": None, "single_image": False},
        {"mounts": "", "axis": "all"},
        {"mounts": []},
    ):
        session, recorder = Session(), Recorder()
        report = _run(session, recorder, **controls)
        assert report.committed and report.passed
        assert report.norm == reference.norm and report.terms == reference.terms
        _same_state(session, plain)
        assert recorder.captured == plain_recorder.captured
        assert recorder.measured == plain_recorder.measured
        note = session.commits[0]
        assert not any(k in note for k in ("mode=single", "mounts=", "axis=", "mask=on"))


def test_default_time_range_keeps_the_t621_sampled_steps():
    for steps in range(1, 41):
        assert it.sampled_steps(steps) == tuple(sorted({0, (steps - 1) // 2, steps - 1}))
        assert it.sampled_steps(steps, 0.0, 1.0) == it.sampled_steps(steps)


def test_full_mask_gives_the_same_write_as_no_mask():
    plain = Session()
    _run(plain, Recorder())
    session = Session()
    report = _run(session, Recorder(), mask=torch.ones(1, 1, 3, 1))
    assert report.committed
    _same_state(session, plain)


def test_mount_filter_leaves_the_other_mounts_unchanged():
    session = Session()
    report = _run(session, Recorder(), mounts=f"{MOUNTS[0]}")
    assert report.committed
    assert _changed(session) == {MOUNTS[0]}
    untouched = session.state[MOUNTS[1]]["c"]
    assert torch.equal(untouched, torch.zeros_like(untouched))
    assert f"mounts={MOUNTS[0]}" in session.commits[0]
    session = Session()
    _run(session, Recorder(), mounts="transformer_blocks.1.*, nothing_like_this")
    assert _changed(session) == {MOUNTS[1]}


def test_mount_filter_without_a_match_commits_nothing():
    session = Session()
    report = _run(session, Recorder(), mounts="no_such_mount*")
    assert not report.committed and report.terms == 0 and session.commits == []
    assert _changed(session) == set()


def test_mount_patterns_parse_lines_and_commas():
    assert it.parse_mounts(" a*\nb , c\n\n") == ["a*", "b", "c"]
    assert it.parse_mounts(None) == [] and it.parse_mounts("") == []
    assert it.mount_matches("transformer_blocks.3.attn", ["transformer_blocks.3.*"])
    assert not it.mount_matches("transformer_blocks.13.attn", ["transformer_blocks.3.*"])


def test_time_range_builds_only_samples_inside_the_range():
    steps = 20
    high = it.sampled_steps(steps, 0.6, 1.0)
    low = it.sampled_steps(steps, 0.0, 0.3)
    for chosen, (t_min, t_max) in ((high, (0.6, 1.0)), (low, (0.0, 0.3))):
        assert chosen
        assert all(t_min <= it.step_time(k, steps) <= t_max for k in chosen)
    assert high[0] == 0 and low[-1] == steps - 1
    assert set(high).isdisjoint(low)
    with pytest.raises(ValueError, match="t_min"):
        it.sampled_steps(steps, 0.8, 0.2)
    with pytest.raises(ValueError, match="No sampler step"):
        it.sampled_steps(4, 0.4, 0.5)


def test_settings_take_the_time_range(monkeypatch):
    from kilora.comfyui import image_learn

    monkeypatch.setattr(
        image_learn.patching, "detect_family", lambda patcher: SimpleNamespace(family="x")
    )
    monkeypatch.setattr(image_learn.families, "get_family", lambda name: None)
    monkeypatch.setattr(
        image_learn.P,
        "_solve_sampling",
        lambda spec: ({"steps": 20, "size": 512, "cfg": 4.0}, ""),
    )
    assert image_learn.settings_for(None).sample_steps == (0, 9, 19)
    settings = image_learn.settings_for(None, 0.5, 1.0)
    assert settings.sample_steps == it.sampled_steps(20, 0.5, 1.0)
    assert all(it.step_time(k, 20) >= 0.5 for k in settings.sample_steps)


def test_all_zero_mask_gives_no_change():
    session = Session()
    report = _run(session, Recorder(), mask=torch.zeros(1, 1, 3, 1))
    assert not report.committed and session.commits == []
    assert _changed(session) == set()


def test_mask_without_an_image_grid_leaves_the_reads_out():
    session = Session()
    report = _run(session, Recorder(), mask=torch.ones(1, 1, 1, 1))
    assert report.terms == 0 and not report.committed


def test_axis_keeps_only_frames_of_that_axis():
    class AxisEditor(Editor):
        def frame_axes(self, state):
            return {(MOUNTS[0], 0): "ip", (MOUNTS[1], 0): "creative"}

    session = Session()
    report = _run(session, Recorder(), editor=AxisEditor(), axis="ip")
    assert report.committed and _changed(session) == {MOUNTS[0]}
    assert "axis=ip" in session.commits[0]
    with pytest.raises(ValueError, match="no axes"):
        _run(Session(), Recorder(), axis="ip")


def test_single_image_skips_the_guard_and_warns():
    session, recorder = Session(), Recorder(gain=1.5)
    report = _run(session, recorder, items=_items()[:1], single_image=True)
    assert recorder.measured == [] and recorder.captured == [0]
    assert report.held_out == [] and report.fitted == [0]
    assert report.passed and report.committed
    assert it.SINGLE_IMAGE_WARNING in report.text()
    assert "single-image mode" in report.text()
    note = session.commits[0]
    assert "mode=single" in note and "guard=passed" in note and "held_out=0" in note
    assert _changed(session)


def test_single_image_keeps_the_drift_check():
    session, recorder = Session(), Recorder(drift=0.5)
    report = _run(session, recorder, items=_items()[:1], single_image=True, drift_bound=0.15)
    assert not report.passed and not report.committed and session.commits == []
    assert "drift" in report.message and recorder.measured == []
    assert report.drift == 0.5
    assert it.SINGLE_IMAGE_WARNING in report.text()


def test_single_image_needs_exactly_one_image():
    with pytest.raises(ValueError, match="exactly 1 image"):
        _run(Session(), Recorder(), single_image=True)


def test_node_has_the_optional_controls():
    from kilora.comfyui.nodes import KiLoRALiveLearnFromImages

    optional = KiLoRALiveLearnFromImages.INPUT_TYPES()["optional"]
    assert list(optional) == [
        "images",
        "captions",
        "folder",
        "clip",
        "mounts",
        "t_min",
        "t_max",
        "mask",
        "axis",
        "single_image",
    ]
    assert optional["t_min"][1]["default"] == 0.0 and optional["t_max"][1]["default"] == 1.0
    assert optional["axis"][1]["default"] == "all" and optional["mounts"][1]["default"] == ""
    assert optional["single_image"][1]["default"] is False
    assert optional["mask"][0] == "MASK"
