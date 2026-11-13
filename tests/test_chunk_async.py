# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch
from test_chunk_live import _packs, _record

from kilora.editors import commit_write, editor_for
from kilora.live import open_session
from kilora.live_queue import WriteQueue
from kilora.recording import GenerationRecord
from kilora.testing import make_live_mod

pytestmark = pytest.mark.async_live
EVENTS = 10


def _slow(record: GenerationRecord, seconds: float) -> GenerationRecord:
    measure = record.measure

    def slow(trial):
        time.sleep(seconds)
        return measure(trial)

    return GenerationRecord(images=1, measure=slow)


def _session(folder):
    make_live_mod(folder / "live.kmod")
    s = open_session(folder / "live.kmod", saves_dir=folder / "saves")
    s.learning = True
    return s


def _targets(session, n):
    g = torch.Generator().manual_seed(9)
    return [
        {t: torch.randn(int(p["up"].shape[-1]), 4, generator=g) for t, p in session.state.items()}
        for _ in range(n)
    ]


def _snapshot(session):
    state = {t: p["up"].clone() for t, p in session.state.items()}
    return state, session.editor_snapshot()


def _bits(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _same(a, b):
    (ua, ea), (ub, eb) = a, b
    if ea.keys() != eb.keys():
        return False
    return all(_bits(ua[t]) == _bits(ub[t]) for t in ua) and all(
        _bits(ea[k]) == _bits(eb[k]) for k in ea
    )


def test_ten_events_async_match_the_node(tmp_path):
    (tmp_path / "sync").mkdir()
    (tmp_path / "async").mkdir()
    ref = _session(tmp_path / "sync")
    path_ref = _packs(ref, tmp_path / "sync")
    targets = _targets(ref, EVENTS)
    editor = editor_for(path_ref)
    editor.check(ref)
    want = []
    for target in targets:
        commit_write(ref, editor, _record(ref, target))
        want.append(_snapshot(ref))

    s = _session(tmp_path / "async")
    path = _packs(s, tmp_path / "async")
    ed = editor_for(path)
    ed.check(s)
    q = WriteQueue()
    s.write_queue = q
    got, waits = [], []
    for target in targets:
        q.begin_generation()
        start = time.perf_counter()
        rec = _slow(_record(s, target), 0.002)
        q.submit(lambda rec=rec: commit_write(s, ed, rec))
        waits.append(time.perf_counter() - start)
        q.end_generation()
        assert q.wait_idle(30)
        got.append(_snapshot(s))
    q.close()
    assert max(waits) < 0.05
    assert len(s.commits) == EVENTS == len(ref.commits)
    assert all(_same(a, b) for a, b in zip(got, want, strict=True))

    with q.exclusive():
        s.undo()
    ref.undo()
    assert _same(_snapshot(s), _snapshot(ref))
    name = s.save_point("ten")
    fresh = open_session(tmp_path / "async" / "live.kmod", saves_dir=tmp_path / "async" / "saves")
    fresh_editor = editor_for(path)
    fresh_editor.check(fresh)
    fresh.load(name)
    assert _same(_snapshot(fresh), _snapshot(s))
