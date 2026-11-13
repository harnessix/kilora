# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from kilora import editors
from kilora.editors import Editor, commit_write
from kilora.live import open_session
from kilora.recording import GenerationRecord
from kilora.testing import make_live_mod


class Step(Editor):
    def next_state(self, state, record):
        return {t: {**p, "up": p["up"] + 0.1} for t, p in state.items()}

    def guard_trial(self, state, new):
        return "trial"


class NoTrial(Step):
    def guard_trial(self, state, new):
        return None


def _scorer(gap=1.0, spread=1.0):
    def score(trial):
        return {"gap": 1.0, "spread": 1.0} if trial is None else {"gap": gap, "spread": spread}

    return score


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.delenv(editors.GUARD_ENV, raising=False)
    make_live_mod(tmp_path / "live.kmod")
    s = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    s.learning = True
    return s


def _ups(session):
    return {t: p["up"].clone() for t, p in session.state.items()}


def _unchanged(session, before):
    return all(torch.equal(session.state[t]["up"], before[t]) for t in before)


def test_a_safe_write_passes(session):
    before = _ups(session)
    report = commit_write(session, Step(), GenerationRecord(guard_score=_scorer(0.9, 1.04)))
    assert "Guard: passed." in report and not _unchanged(session, before)
    assert len(session.commits) == 1


@pytest.mark.parametrize(
    "gap, spread, why",
    [
        (0.7, 1.0, "prompt following"),
        (1.0, 0.94, "spread"),
        (1.0, 1.06, "spread"),
        (math.nan, 1.0, "prompt following"),
        (1.0, math.nan, "spread"),
    ],
)
def test_a_harmful_write_is_refused_and_restored(session, gap, spread, why):
    before = _ups(session)
    report = commit_write(session, Step(), GenerationRecord(guard_score=_scorer(gap, spread)))
    assert report.startswith("Refused by the guard") and why in report
    assert _unchanged(session, before) and len(session.commits) == 0
    assert session.pending is None


def test_the_thresholds():
    assert editors.GUARD_GAP_DROP == 0.2 and editors.GUARD_GAP_NOISE_K == 1.0
    assert editors.GUARD_SPREAD == (0.95, 1.05) and editors.GUARD_NOISE_WINDOW == 8


def _gaps(gap0, gap1, alt0=None, alt1=None):
    def score(trial):
        gap, alt = (gap0, alt0) if trial is None else (gap1, alt1)
        out = {"gap": gap, "spread": 1.0}
        if alt is not None:
            out["gap_alt"] = alt
        return out

    return score


def test_a_write_that_turns_a_tiny_gap_negative_is_refused(session):
    report = commit_write(
        session, Step(), GenerationRecord(guard_score=_gaps(1e-3, -3.1e-4, 1.4e-3, 0.9e-4))
    )
    assert report.startswith("Refused by the guard") and "prompt following" in report
    assert session.guard_last["refused"] == "gap"


def test_the_paired_noise_ignores_the_noise_shared_by_both_arms(session):
    for _ in range(3):
        commit_write(session, Step(), GenerationRecord(guard_score=_gaps(1e-3, 1e-3, 2e-3, 2e-3)))
    assert session.guard_last["gap_noise"] == 0.0
    report = commit_write(
        session, Step(), GenerationRecord(guard_score=_gaps(1e-3, 0.7e-3, 2e-3, 1.7e-3))
    )
    assert report.startswith("Refused by the guard")


@pytest.mark.parametrize(
    "gap1, refused",
    [(0.85e-3, False), (0.75e-3, True)],
)
def test_the_share_counts_when_the_noise_is_small(session, gap1, refused):
    report = commit_write(session, Step(), GenerationRecord(guard_score=_gaps(1e-3, gap1)))
    assert report.startswith("Refused by the guard") is refused


def test_a_drop_within_the_paired_noise_passes(session):
    for _ in range(3):
        commit_write(
            session, Step(), GenerationRecord(guard_score=_gaps(1e-3, 1.4e-3, 1e-3, 0.6e-3))
        )
    noise = session.guard_last["gap_noise"]
    assert noise == pytest.approx(8e-4 / 2**0.5 / 2**0.5, rel=1e-3)
    ok = _gaps(1e-3, 0.71e-3, 1e-3, 0.71e-3)
    assert "Guard: passed." in commit_write(session, Step(), GenerationRecord(guard_score=ok))
    bad = _gaps(1e-3, 0.5e-3, 1e-3, 0.5e-3)
    assert commit_write(session, Step(), GenerationRecord(guard_score=bad)).startswith("Refused")


def test_the_noise_window_keeps_the_last_writes(session):
    for d in [1.0] * 10 + [0.0] * 8:
        commit_write(session, Step(), GenerationRecord(guard_score=_gaps(1.0, 1.0, 1.0, 1.0 + d)))
    assert len(session.guard_gap_diffs) == 8 and session.guard_last["gap_noise"] == 0.0


@pytest.mark.parametrize(
    "rule, record, note",
    [
        (Step(), GenerationRecord(), "no image measurement"),
        (NoTrial(), GenerationRecord(guard_score=_scorer(0.0)), "gives the guard no trial"),
    ],
)
def test_the_report_says_when_the_guard_is_off(session, rule, record, note):
    report = commit_write(session, rule, record)
    assert "Guard off" in report and note in report
    assert len(session.commits) == 1


def test_the_user_can_turn_the_guard_off(session, monkeypatch):
    monkeypatch.setenv(editors.GUARD_ENV, "0")
    report = commit_write(session, Step(), GenerationRecord(guard_score=_scorer(0.0, 0.0)))
    assert "Guard off (KILORA_LIVE_GUARD=0)." in report and len(session.commits) == 1
    monkeypatch.delenv(editors.GUARD_ENV)
    session.guard = False
    report = commit_write(session, Step(), GenerationRecord(guard_score=_scorer(0.0, 0.0)))
    assert "Guard off" in report and len(session.commits) == 2


def test_a_refused_write_restores_the_editor_state(session):
    class Stateful(Step):
        def next_state(self, state, record):
            session.editor_state.n += 1
            return super().next_state(state, record)

    class Counter:
        n = 0

        def state_tensors(self):
            return {"n": torch.tensor([float(self.n)])}

        def load_tensors(self, tensors):
            self.n = int(tensors["n"])

        def reset(self):
            self.n = 0

    session.attach_editor_state(Counter(), "counter")
    commit_write(session, Stateful(), GenerationRecord(guard_score=_scorer(0.0)))
    assert session.editor_state.n == 0
