# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import logging

import pytest
import torch
from test_chunk_live import _packs, _record

from kilora.comfyui import live_write, patching
from kilora.comfyui.live_write import PathHandle, QueuedWrite
from kilora.editors import commit_write, editor_for
from kilora.live import open_session
from kilora.live_queue import WriteQueue
from kilora.testing import make_live_mod

pytestmark = pytest.mark.async_live


def setup_session(tmp_path):
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod")
    session.learning = True
    rule = editor_for(_packs(session, tmp_path))
    rule.check(session)
    rule.configure({"path_merge": False})
    generator = torch.Generator().manual_seed(31)
    targets = [
        {
            t: torch.randn(int(p["up"].shape[-1]), 4, generator=generator)
            for t, p in session.state.items()
        }
        for _ in range(4)
    ]
    return session, rule, targets


def recorded(session, target):
    record = _record(session, target)
    record.latent = torch.ones(1, 4, 2, 2)
    record.conditioning = {"embedding": torch.ones(1, 4)}
    return record


def test_backlog_matches_separate_commits_and_resets_each_mean(tmp_path, monkeypatch):
    import kilora.write_estimates as estimates

    session, rule, targets = setup_session(tmp_path)
    for target in targets:
        commit_write(session, rule, recorded(session, target))
    expected = copy.deepcopy(session.editor_state.chunks.snapshot())
    expected_state = copy.deepcopy(session.state)
    for _ in targets:
        session.undo()
    counts = []
    average = estimates.mean_write_values

    def observe(*args, **kwargs):
        result = average(*args, **kwargs)
        counts.append([row[1] for row in session._write_estimates.values()])
        return result

    monkeypatch.setattr(estimates, "mean_write_values", observe)
    queue = WriteQueue()
    queue.begin_generation()
    records = [recorded(session, target) for target in targets]
    try:
        for rec in records:
            queue.submit(QueuedWrite(session, rule, rec, None))
        assert all(
            rec.latent.device.type == "cpu" and rec.conditioning["embedding"].device.type == "cpu"
            for rec in records
        )
        queue.end_generation()
        assert queue.wait_idle(30)
        assert len(session.commits) == 4 and not queue.dropped, queue.reports
        assert "combined=1" in queue.reports[0][1]
        assert len(counts) == 4 and all(row and set(row) == {1} for row in counts)
        assert session._write_estimates == {}
        for key, value in expected["coef"].items():
            assert torch.equal(session.editor_state.chunks.coef[key], value)
        for target, parts in expected_state.items():
            for key, value in parts.items():
                assert torch.equal(session.state[target][key], value)
    finally:
        queue.close()


def test_backlog_retains_refused_estimate_only_until_next_commit(tmp_path, monkeypatch):
    import kilora.editors as editors
    import kilora.write_estimates as estimates

    session, rule, targets = setup_session(tmp_path)
    counts = []
    average = estimates.mean_write_values

    def observe(*args, **kwargs):
        result = average(*args, **kwargs)
        counts.append([row[1] for row in session._write_estimates.values()])
        return result

    monkeypatch.setattr(estimates, "mean_write_values", observe)
    monkeypatch.setattr(
        editors,
        "guard_check",
        lambda _rule, _session, record, _new: (record.extra.get("refuse", False), "test guard"),
    )
    records = [recorded(session, target) for target in targets[:3]]
    records[0].extra["refuse"] = True
    queue = WriteQueue()
    queue.begin_generation()
    try:
        for record in records:
            queue.submit(QueuedWrite(session, rule, record, None))
        queue.end_generation()
        assert queue.wait_idle(30), queue.reports
        assert len(session.commits) == 2 and not queue.dropped, queue.reports
        assert len(counts) == 3
        assert set(counts[0]) == {1}
        assert 2 in counts[1] and set(counts[1]) <= {1, 2}
        assert set(counts[2]) == {1}
        assert session._write_estimates == {}
    finally:
        queue.close()


def test_async_disabled_keeps_direct_write_bit_identical(tmp_path, monkeypatch):
    session, rule, targets = setup_session(tmp_path)
    monkeypatch.setenv("KILORA_LIVE_ASYNC", "0")
    target = targets[0]
    commit_write(session, rule, recorded(session, target))
    expected = {t: p["up"].clone() for t, p in session.state.items()}
    session.undo()
    monkeypatch.setattr(live_write, "image_loss", lambda *a, **kw: _record(session, target).measure)
    handle = PathHandle(
        session, rule, None, timesteps=[0.5], calls=1, images=1, apply_model=lambda *a: None
    )
    live_write.write(handle, {"samples": torch.zeros(1, 4, 2, 2)})
    assert all(torch.equal(v, session.state[t]["up"]) for t, v in expected.items())


def test_measurement_is_created_only_when_queued_record_runs(tmp_path, monkeypatch):
    session, rule, targets = setup_session(tmp_path)
    queue = session.write_queue = WriteQueue()
    calls = []

    def make_measure(handle, latent, *args, **kwargs):
        calls.append((latent.device.type, handle.conditioning["embedding"].device.type))
        measure = _record(session, targets[0]).measure
        measure.guard_score = None
        return measure

    monkeypatch.setattr(live_write, "image_loss", make_measure)
    handle = PathHandle(
        session,
        rule,
        None,
        timesteps=[0.5],
        calls=1,
        images=1,
        conditioning={"embedding": torch.ones(1, 4)},
        apply_model=lambda *a: None,
        queue=queue,
    )
    queue.begin_generation()
    try:
        live_write.write(handle, {"samples": torch.zeros(1, 4, 2, 2)})
        assert not calls
        job = queue._pending[0].run
        assert job.record.latent.device.type == "cpu"
        assert job.record.conditioning["embedding"].device.type == "cpu"
        queue.end_generation()
        assert queue.wait_idle(30)
        assert calls == [("cpu", "cpu")]
        assert len(session.commits) == 1, queue.reports
    finally:
        queue.close()


@pytest.mark.parametrize("action,count", [("undo", 2), ("reset", 1), ("load", 1)])
def test_pending_writes_follow_pre_action_state(tmp_path, monkeypatch, caplog, action, count):
    caplog.set_level(logging.INFO)
    session, rule, targets = setup_session(tmp_path)
    commit_write(session, rule, recorded(session, targets[0]))
    save = session.save_point("before")
    pre_action = copy.deepcopy(session.editor_state.chunks.snapshot())
    parent = session.editor_state.forest.current
    queue = session.write_queue = WriteQueue()
    monkeypatch.setattr(patching, "sessions_of", lambda _: [session])
    monkeypatch.setattr(patching, "reapply", lambda _: (None, "applied"))
    queue.begin_generation()
    try:
        for target in targets[1 : 1 + count]:
            queue.submit(QueuedWrite(session, rule, recorded(session, target), None))
        _, message = patching.live_action(object(), action, save, session.mod_id)
        branch = next(
            b for b in session.editor_state.forest.branches.values() if b.action == action
        )
        assert branch.parent == parent and branch.writes == list(range(1, count + 1))
        assert f"kept on side branch {branch.id} after {action}" in message
        assert f"kept on side branch {branch.id} after {action}" in caplog.text
        current = session.editor_state.forest.current
        visible = {t: p["up"].clone() for t, p in session.state.items()}
        queue.end_generation()
        assert queue.wait_idle(30), queue.reports
        assert session.editor_state.forest.current == current
        assert all(torch.equal(v, session.state[t]["up"]) for t, v in visible.items())
        branch = session.editor_state.forest.branches[branch.id]
        assert branch.state.events == pre_action["events"] + count, queue.reports
        assert not queue.dropped
    finally:
        queue.close()


def test_pending_groups_captured_revisions_and_reuses_fork(tmp_path):
    session, rule, targets = setup_session(tmp_path)
    queue = WriteQueue(max_pending=1)
    queue.begin_generation()
    try:
        commit_write(session, rule, recorded(session, targets[0]))
        revision = session.revision
        first = copy.deepcopy(session.editor_state.chunks.snapshot())
        for target in targets[1:3]:
            queue.submit(QueuedWrite(session, rule, recorded(session, target), None))
        commit_write(session, rule, recorded(session, targets[3]))
        second = copy.deepcopy(session.editor_state.chunks.snapshot())
        queue.submit(QueuedWrite(session, rule, recorded(session, targets[3]), None))
        kept = queue.keep_pending(session.editor_state, "undo")
        assert len(kept) == 2
        a, b = (entry[0] for entry in kept)
        forest = session.editor_state.forest
        assert forest.branches[a].state.events == first["events"]
        assert forest.branches[b].state.events == second["events"]
        session.undo()
        visible = {t: p["up"].clone() for t, p in session.state.items()}
        session.revision = revision
        queue.submit(QueuedWrite(session, rule, recorded(session, targets[1]), None))
        assert queue.keep_pending(session.editor_state, "undo")[0][0] == a
        current = forest.current
        queue.end_generation()
        assert queue.wait_idle(30), queue.reports
        assert forest.current == current
        assert all(torch.equal(v, session.state[t]["up"]) for t, v in visible.items())
        assert forest.branches[a].state.events == first["events"] + 3
        assert forest.branches[b].state.events == second["events"] + 1
        assert len(queue.reports) == 3
        assert "combined=1" in queue.reports[0][1]
    finally:
        queue.close()
