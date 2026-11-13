# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType

import pytest
import torch

from kilora import editors
from kilora.editors import Editor, commit_write
from kilora.live import open_session
from kilora.recording import GenerationRecord
from kilora.testing import make_live_mod

MEMORY_LINE = (
    "GPU memory: allocated 0.4 GiB, reserved 9.1 GiB, peak 7.9 GiB (freed after the write)"
)
DEVICE = torch.device("cuda:3")


class Step(Editor):
    def __init__(self, events):
        self.events = events

    def next_state(self, state, record):
        self.events.append("write")
        return {t: {**parts, "up": parts["up"] + 0.1} for t, parts in state.items()}

    def guard_trial(self, state, new):
        return "trial"


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.delenv(editors.GUARD_ENV, raising=False)
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    return session


@pytest.fixture
def gpu(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def counter(name, value=None):
        def call(device):
            assert device == DEVICE
            events.append(name)
            return value

        return call

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", counter("reset"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", counter("allocated", 0.4 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", counter("reserved", 9.1 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", counter("peak", 7.9 * 2**30))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    management.soft_empty_cache = lambda: events.append("soft")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return events


def _record(*, refused=False, device=DEVICE):
    def measure(trial):
        return 1.0

    measure.device = device
    return GenerationRecord(
        measure=measure,
        guard_score=lambda trial: {
            "gap": 0.0 if refused and trial is not None else 1.0,
            "spread": 1.0,
        },
    )


def _state(session):
    return {t: {k: v.clone() for k, v in parts.items()} for t, parts in session.state.items()}


def _equal_state(session, before):
    assert set(session.state) == set(before)
    for target, parts in before.items():
        assert set(session.state[target]) == set(parts)
        assert all(torch.equal(value, session.state[target][key]) for key, value in parts.items())


def test_cleanup_follows_commit_and_reports_counters(session, gpu, monkeypatch):
    commit = session.commit

    def tracked_commit(**kwargs):
        result = commit(**kwargs)
        gpu.append("commit")
        return result

    monkeypatch.setattr(session, "commit", tracked_commit)
    report = commit_write(session, Step(gpu), _record())
    assert len(session.commits) == 1
    assert "Guard: passed." in report
    assert report.splitlines()[-1] == MEMORY_LINE
    assert gpu == ["reset", "write", "commit", "allocated", "reserved", "peak", "empty", "soft"]


def test_cleanup_after_guard_refusal(session, gpu):
    before = _state(session)
    report = commit_write(session, Step(gpu), _record(refused=True))
    assert report.startswith("Refused by the guard")
    assert report.splitlines()[-1] == MEMORY_LINE
    assert not session.commits
    _equal_state(session, before)
    assert gpu == ["reset", "write", "allocated", "reserved", "peak", "empty", "soft"]


@pytest.mark.parametrize("stage", ["write", "guard", "commit"])
def test_cleanup_after_error(session, gpu, monkeypatch, stage):
    error = RuntimeError("write failed")

    def fail(*args, **kwargs):
        gpu.append(stage)
        raise error

    rule, record = Step(gpu), _record()
    if stage == "write":
        monkeypatch.setattr(rule, "next_state", fail)
    elif stage == "guard":
        record.guard_score = fail
    else:
        monkeypatch.setattr(session, "commit", fail)
    with pytest.raises(RuntimeError) as caught:
        commit_write(session, rule, record)
    assert caught.value is error
    assert gpu[0] == "reset"
    assert gpu[-5:] == ["allocated", "reserved", "peak", "empty", "soft"]
    assert gpu.count("empty") == gpu.count("soft") == 1


@pytest.mark.parametrize("cuda_available, device", [(False, DEVICE), (False, "cpu"), (True, "cpu")])
def test_cpu_has_no_memory_report_or_cleanup(session, gpu, monkeypatch, cuda_available, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    report = commit_write(session, Step(gpu), _record(device=torch.device(device)))
    assert len(session.commits) == 1
    assert "GPU memory:" not in report
    assert gpu == ["write"]


def test_written_state_and_other_report_lines_are_identical(session, gpu, monkeypatch):
    record = _record()
    report = commit_write(session, Step(gpu), record)
    written = _state(session)
    session.undo()
    with monkeypatch.context() as bypass:
        bypass.setattr(editors, "_write_memory_device", lambda _: None)
        original_report = commit_write(session, Step([]), record)
    _equal_state(session, written)
    assert report == original_report + "\n" + MEMORY_LINE


def test_cleanup_without_comfyui(session, gpu, monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    report = commit_write(session, Step(gpu), _record())
    assert report.splitlines()[-1] == MEMORY_LINE
    assert gpu[-1] == "empty" and gpu.count("empty") == 1
    assert "soft" not in gpu


def test_combined_write_releases_caches_once(session, gpu):
    class Combined(Step):
        def next_combined_state(self, state, records):
            assert len(records) == 2
            return self.next_state(state, records[-1])

    records = [_record(), _record()]
    report = commit_write(session, Combined(gpu), records[-1], records=records)
    assert len(session.commits) == 1
    assert report.count(MEMORY_LINE) == 1
    assert gpu == ["reset", "write", "allocated", "reserved", "peak", "empty", "soft"]


def test_memory_counter_error_still_releases_both_caches(session, gpu, monkeypatch):
    def fail(device):
        raise RuntimeError("counter unavailable")

    monkeypatch.setattr(torch.cuda, "memory_allocated", fail)
    report = commit_write(session, Step(gpu), _record())
    assert "Guard: passed." in report and len(session.commits) == 1
    assert "GPU memory:" not in report
    assert gpu[-2:] == ["empty", "soft"]
