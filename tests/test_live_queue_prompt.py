# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import threading
import time
import types

import pytest

from kilora import live_queue
from kilora.comfyui import live_write
from kilora.live_queue import STARVE_PROMPTS, WriteQueue, note_timing, yield_point

pytestmark = pytest.mark.async_live


def _executor():
    class Executor:
        def execute(self, steps):
            for step in steps:
                step()
            return "done"

    return Executor


def _until(predicate, timeout=5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def queue():
    q = WriteQueue()
    yield q
    q.close()


def test_write_waits_for_decode(queue):
    executor = _executor()
    assert live_write.install_prompt_hook(executor)
    log, lock = [], threading.Lock()

    def add(item):
        with lock:
            log.append(item)

    def encode():
        queue.submit(lambda: add("write") or "ok")
        time.sleep(0.05)
        add("encode")

    def sampling():
        queue.begin_generation()
        time.sleep(0.05)
        add("sampling")
        queue.end_generation()

    def decode():
        time.sleep(0.2)
        add("decode")

    executor().execute([encode, sampling, decode])
    add("prompt end")
    assert queue.wait_idle(5)
    assert log == ["encode", "sampling", "decode", "prompt end", "write"]
    assert queue._prompts == 0


def test_starved_write_runs_outside_sampling(queue):
    phase = ["before"]
    phases, lock = [], threading.Lock()

    def run():
        for _ in range(40):
            yield_point()
            with lock:
                phases.append(phase[0])
            time.sleep(0.01)
        return "ok"

    queue.begin_prompt()
    queue.submit(run)
    for _ in range(STARVE_PROMPTS):
        queue.begin_prompt()
        queue.end_prompt()
    assert queue._starved == STARVE_PROMPTS
    assert queue._prompts == 1

    phase[0] = "encode"
    assert _until(lambda: "encode" in phases)
    queue.begin_generation()
    with queue.copy_guard():
        phase[0] = "sampling"
    time.sleep(0.1)
    with queue.copy_guard():
        phase[0] = "decode"
    queue.end_generation()
    assert queue.wait_idle(5)
    queue.end_prompt()

    assert "encode" in phases and "decode" in phases
    assert "sampling" not in phases
    assert queue.reports[-1][1] == "ok"
    assert queue._starved == 0

    done = threading.Event()
    queue.begin_prompt()
    queue.submit(lambda: done.set() or "second")
    assert not done.wait(0.2)
    queue.end_prompt()
    assert done.wait(5)


def test_failed_prompt_ends_the_window(queue):
    executor = _executor()
    live_write.install_prompt_hook(executor)

    def fail():
        raise RuntimeError("the prompt failed")

    with pytest.raises(RuntimeError):
        executor().execute([fail])
    assert queue._prompts == 0
    done = threading.Event()
    queue.submit(lambda: done.set() or "ok")
    assert done.wait(5)


def test_marks_work_without_the_prompt_hook(queue):
    assert queue._prompts == 0
    done = threading.Event()
    queue.begin_generation()
    queue.submit(lambda: done.set() or "ok")
    assert not done.wait(0.2)
    queue.end_generation()
    assert done.wait(5)


def test_hook_wraps_once():
    executor = _executor()
    original = executor.execute
    assert live_write.install_prompt_hook(executor)
    wrapped = executor.execute
    assert live_write.install_prompt_hook(executor)
    assert executor.execute is wrapped
    assert wrapped.__wrapped__ is original

    q = WriteQueue()
    try:
        seen = []
        executor().execute([lambda: seen.append(q._prompts)])
        assert seen == [1]
        assert q._prompts == 0
    finally:
        q.close()


def test_missing_executor_logs_and_keeps_behavior(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="kilora.live")
    monkeypatch.setitem(sys.modules, "execution", types.ModuleType("execution"))
    assert live_write.install_prompt_hook() is False
    assert any("no prompt hook" in r.getMessage() for r in caplog.records)

    class NoExecute:
        pass

    caplog.clear()
    assert live_write.install_prompt_hook(NoExecute) is False
    assert not hasattr(NoExecute, "execute")
    assert any("no prompt hook" in r.getMessage() for r in caplog.records)


def test_async_write_logs_one_line(queue, caplog):
    caplog.set_level(logging.INFO, logger="kilora.live")

    def run():
        note_timing({"preempts": 2, "preempt_wait_s": 0.5, "preempt_dropped_s": 0.25})
        return "ok"

    number = queue.submit(run)
    assert queue.wait_idle(5)

    def line():
        return [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith(f"live write {number}:")
        ]

    assert _until(lambda: line())
    assert line() == [
        f"live write {number}: 2 stops for a render, 0.50 s of overlap, "
        "0.25 s dropped, 0 prompts waited"
    ]
    assert queue._write_timing == {}


def test_note_timing_outside_worker_returns_false():
    assert live_queue.note_timing({"preempts": 1}) is False
