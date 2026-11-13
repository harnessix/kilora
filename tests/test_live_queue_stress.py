# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import random
import tracemalloc

import pytest
from queue_helpers import FakeSession

from kilora.live_queue import WriteQueue

pytestmark = pytest.mark.async_live

GENERATIONS = 200


def _schedule(seed=7):
    rng = random.Random(seed)
    return [rng.random() < 0.3 for _ in range(GENERATIONS)]


def _run(schedule):
    s = FakeSession()
    q = WriteQueue()
    digests, peak_pending = [], 0
    q.begin_generation()
    for i, pause in enumerate(schedule, 1):
        q.submit(lambda i=i: s.write(i))
        peak_pending = max(peak_pending, q.pending())
        if pause:
            q.end_generation()
            assert q.wait_idle(10)
            digests.append(s.digest())
            q.begin_generation()
        else:
            q.begin_generation()
            q.end_generation()
        if i == GENERATIONS // 2:
            tracemalloc.start()
            middle = tracemalloc.take_snapshot()
    q.end_generation()
    assert q.wait_idle(10)
    digests.append(s.digest())
    end = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(stat.size_diff for stat in end.compare_to(middle, "filename"))
    ran = [n for n, _ in q.reports]
    q.close()
    return s, digests, ran, list(q.dropped), peak_pending, grown


def test_two_hundred_generations(caplog):
    schedule = _schedule()
    with caplog.at_level(logging.WARNING, logger="kilora.live"):
        s, digests, ran, dropped, peak, grown = _run(schedule)
    want_ran = list(range(1, GENERATIONS + 1))
    assert ran == want_ran and dropped == []
    assert s.commits == want_ran
    assert peak > 2
    drops_logged = [r for r in caplog.records if "dropped" in r.getMessage()]
    assert not drops_logged
    assert grown < 1_000_000

    caplog.clear()
    s2, digests2, *_ = _run(schedule)
    assert digests == digests2

    sync = FakeSession()
    for i in want_ran:
        sync.write(i)
    assert sync.digest() == digests[-1]
