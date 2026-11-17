# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from test_multi_import import _lokr, write

from kilora import key_cache, prepare
from kilora.comfyui import live_write, prepare_node, startup_prepare
from kilora.live_queue import WriteQueue, write_activity


@pytest.fixture(autouse=True)
def idle_worker(monkeypatch):
    startup_prepare._stop.clear()
    monkeypatch.setattr(startup_prepare.os, "nice", lambda value: None)
    monkeypatch.setattr(startup_prepare, "base_model_files", lambda: [])
    yield
    startup_prepare._stop.clear()


def _observe_pause(monkeypatch):
    paused = threading.Event()
    original = startup_prepare.busy

    def busy():
        result = original()
        if result:
            paused.set()
        return result

    monkeypatch.setattr(startup_prepare, "busy", busy)
    return paused


def test_busy_write_blocks_the_next_file_and_keeps_order(monkeypatch, tmp_path, caplog):
    paths = [str(tmp_path / name) for name in ("first", "second")]
    seen = []
    first = threading.Event()
    proceed = threading.Event()
    paused = _observe_pause(monkeypatch)
    monkeypatch.setattr(prepare_node, "apply_device", lambda: "cpu")
    monkeypatch.setattr(prepare_node, "selected_files", lambda *args: (paths, []))
    monkeypatch.setattr(prepare_node, "display_path", lambda path, roots: path)
    monkeypatch.setattr(startup_prepare, "_plan_dtype", lambda *args: torch.float32)
    monkeypatch.setattr(prepare, "layer_ranks", lambda result: {})

    def convert(path, **kwargs):
        seen.append(path)
        return 0, 0, False

    def derived(path, **kwargs):
        if path == paths[0]:
            first.set()
            assert proceed.wait(10)
        return object()

    monkeypatch.setattr(prepare, "prepare_lokr", convert)
    monkeypatch.setattr(prepare, "prepare_file", derived)
    caplog.set_level(logging.INFO, logger="kilora.prepare")
    worker = threading.Thread(target=startup_prepare._run, args=([str(tmp_path)],))
    worker.start()
    try:
        assert first.wait(10)
        with write_activity():
            proceed.set()
            assert paused.wait(10)
            assert seen == paths[:1]
            assert worker.is_alive()
    finally:
        proceed.set()
        worker.join(10)
    assert not worker.is_alive()
    assert seen == paths
    pauses = [r.getMessage() for r in caplog.records if "preparation paused:" in r.getMessage()]
    resumes = [r.getMessage() for r in caplog.records if "preparation resumed:" in r.getMessage()]
    assert len(pauses) == len(resumes) == 1
    assert paths[1] in pauses[0] and paths[1] in resumes[0]


def test_busy_write_blocks_the_next_lokr_layer_and_keeps_cache(monkeypatch, tmp_path, caplog):
    weights, _ = _lokr(1)
    other, _ = _lokr(2)
    weights.update(
        {
            key.replace(".lokr_", "_second.lokr_").replace(".alpha", "_second.alpha"): value
            for key, value in other.items()
        }
    )
    path = write(tmp_path, "two-layers", weights)
    monkeypatch.setenv("KILORA_LOKR_CACHE", "1")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "expected"))
    assert prepare.prepare_lokr(path) == (2, 0)
    expected = load_file(str(next((tmp_path / "expected" / "lokr").glob("*.safetensors"))))
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "actual"))
    first = threading.Event()
    proceed = threading.Event()
    paused = _observe_pause(monkeypatch)
    original = prepare._kron_parts_svd
    calls = []
    results, errors = [], []

    def convert(*args):
        result = original(*args)
        calls.append(result)
        if len(calls) == 1:
            first.set()
            assert proceed.wait(10)
        return result

    def run():
        try:
            results.append(
                prepare.prepare_lokr(
                    path, checkpoint=lambda: startup_prepare._checkpoint("cpu", "two-layers")
                )
            )
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(prepare, "_kron_parts_svd", convert)
    caplog.set_level(logging.INFO, logger="kilora.prepare")
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert first.wait(10)
        with write_activity():
            proceed.set()
            assert paused.wait(10)
            assert len(calls) == 1 and worker.is_alive()
    finally:
        proceed.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    assert results == [(2, 0)] and len(calls) == 2
    actual = load_file(str(next((tmp_path / "actual" / "lokr").glob("*.safetensors"))))
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in actual)
    assert caplog.text.count("Startup preparation paused: two-layers") == 1
    assert caplog.text.count("Startup preparation resumed: two-layers") == 1


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_live_write_is_busy_before_measurement_through_commit(monkeypatch, background, fail):
    session = SimpleNamespace(
        commits=[], _probe_writes=[], editor_state=SimpleNamespace(forest=None)
    )
    queue = WriteQueue() if background else None
    handle = live_write.PathHandle(session, object(), None, calls=1, queue=queue)
    observations = []

    def end_generation(handle):
        observations.append(("start", startup_prepare.busy()))

    def commit(session, rule, record):
        observations.append(("commit", startup_prepare.busy()))
        if fail:
            raise ValueError("write failed")
        session.commits.append(record)
        return "written"

    monkeypatch.setattr(live_write, "_end_generation", end_generation)
    monkeypatch.setattr(live_write, "commit_write", commit)
    monkeypatch.setattr(live_write, "editor_note", lambda rule: "")
    try:
        assert not startup_prepare.busy()
        if fail and not background:
            with pytest.raises(ValueError, match="write failed"):
                live_write.write(handle)
        else:
            live_write.write(handle)
        if queue is not None:
            assert queue.wait_idle(10)
            assert len(queue.reports) == 1
            assert ("failed" in queue.reports[0][1]) == fail
        assert observations == [("start", True), ("commit", True)]
        assert len(session.commits) == (0 if fail else 1)
        assert not startup_prepare.busy()
    finally:
        if queue is not None:
            queue.close()
