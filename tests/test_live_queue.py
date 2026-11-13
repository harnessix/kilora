# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import threading
import time

import pytest
from queue_helpers import FakeSession

from kilora.live_queue import WriteQueue

pytestmark = pytest.mark.async_live


def _submit(q, s, value, log=None, batch_seconds=0.0):
    return q.submit(lambda: s.write(value, log, batch_seconds))


def test_writes_run_in_order_and_match_the_synchronous_state():
    a = FakeSession()
    q = WriteQueue(max_pending=10)
    for v in (1, 2, 3):
        _submit(q, a, v)
    assert q.wait_idle(10)
    assert [n for n, _ in q.reports] == [1, 2, 3] and a.commits == [1, 2, 3]
    b = FakeSession()
    for v in (1, 2, 3):
        b.write(v)
    assert a.digest() == b.digest()
    q.close()


def test_a_write_yields_to_a_generation():
    s = FakeSession()
    q = WriteQueue()
    log = []
    q.begin_generation()
    _submit(q, s, 1, log, batch_seconds=0.15)
    time.sleep(0.2)
    assert log == [] and s.commits == []
    q.end_generation()
    deadline = time.time() + 5
    while not log and time.time() < deadline:
        time.sleep(0.01)
    q.begin_generation()
    n = len(log)
    time.sleep(0.3)
    assert len(log) <= n + 1
    assert s.commits == []
    q.end_generation()
    assert q.wait_idle(5) and s.commits == [1] and len(log) == 3
    q.close()


def test_backlog_keeps_every_write(caplog):
    s = FakeSession()
    q = WriteQueue()
    q.begin_generation()
    for v in (1, 2, 3, 4):
        _submit(q, s, v)
    q.end_generation()
    assert q.wait_idle(5)
    assert [n for n, _ in q.reports] == [1, 2, 3, 4] and q.dropped == []
    assert s.commits == [1, 2, 3, 4]
    assert "dropped" not in caplog.text
    q.close()


def test_undo_while_a_write_waits_drops_it():
    s = FakeSession()
    s.write(1)
    q = WriteQueue()
    q.begin_generation()
    _submit(q, s, 2)
    with q.exclusive():
        assert q.drop_pending("undo before the write ran") == 1
        s.undo()
    q.end_generation()
    assert q.wait_idle(5)
    assert s.commits == [] and q.reports == []
    q.close()


def test_undo_waits_for_a_running_write():
    s = FakeSession()
    q = WriteQueue()
    log = []
    _submit(q, s, 1, log, batch_seconds=0.1)
    deadline = time.time() + 5
    while not log and time.time() < deadline:
        time.sleep(0.005)
    done = threading.Event()

    def undo_now():
        with q.exclusive():
            s.undo()
        done.set()

    t = threading.Thread(target=undo_now)
    t.start()
    assert done.wait(5)
    t.join()
    assert s.commits == [] and len(log) == 3
    q.close()
