# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextvars
import threading

import pytest
import torch

from kilora import compute, live_queue, recording
from kilora.compute import LayerSet, run_layer
from kilora.recording import ContextStack, observing_blocks

CALLS = 3
TIMEOUT = 30


def _set(seed: int, key: str) -> LayerSet:
    gen = torch.Generator().manual_seed(seed)
    return LayerSet(
        router=torch.randn(8, 32, generator=gen),
        keys_in=torch.randn(4, 2, 8, generator=gen),
        per_expert=2,
        down=torch.randn(8, 32, generator=gen) / 6,
        up=torch.randn(8, 32, generator=gen) * 0.1,
        key=key,
    )


SETS = [_set(n, f"s{n}") for n in range(2)]


def _y(seed: int) -> torch.Tensor:
    return torch.randn(1, 20, 32, generator=torch.Generator().manual_seed(seed))


class _Blocks:
    def __init__(self):
        self.seen: list[tuple[int, int]] = []

    def observe_block(self, index, value):
        self.seen.append((index, threading.get_ident()))


class _Recorders:
    def __init__(self, block):
        self.block = block
        self.blocks = _Blocks()
        self.shares: list[str] = []
        self.routes: dict = {}
        self.probes: dict = {}
        self.timer: dict = {}
        self.active: list[bool] = []

    def __enter__(self):
        self._stack = [
            observing_blocks([(0, self.block)], self.blocks),
            compute.key_probes(),
            compute.recording_shares(lambda target, gates: self.shares.append(target)),
            compute.recording_routes(),
            compute.timing_layers(),
        ]
        values = [cm.__enter__() for cm in self._stack]
        self.probes, self.routes, self.timer = values[1], values[3], values[4]
        return self

    def __exit__(self, *exc):
        for cm in reversed(self._stack):
            cm.__exit__(*exc)
        return False


def _forwards(block, target: str, seed: int) -> list[torch.Tensor]:
    outs = []
    with torch.no_grad():
        for n in range(CALLS):
            y = block(_y(seed + n))
            outs.append(run_layer(y, y, SETS, 3, target=target))
    return outs


def _summary(rec: _Recorders) -> dict:
    return {
        "blocks": len(rec.blocks.seen),
        "shares": list(rec.shares),
        "routes": {k: len(v) for k, v in rec.routes.items()},
        "probes": {k: len(v) if hasattr(v, "__len__") else 1 for k, v in rec.probes.items()},
        "layer_calls": rec.timer["layer_calls"],
    }


def _reference(block, target: str, seed: int) -> dict:
    with _Recorders(block) as rec:
        _forwards(block, target, seed)
    return _summary(rec)


def _run(*workers):
    errors: list[BaseException] = []

    def wrap(fn):
        def run():
            try:
                fn()
            except BaseException as error:
                errors.append(error)

        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert not any(thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]


@pytest.mark.parametrize("other", ["plain", "own"])
def test_each_thread_records_only_its_own_forwards(other):
    torch.manual_seed(0)
    block = torch.nn.Linear(32, 32)
    expected_a = _reference(block, "a", 10)
    expected_b = _reference(block, "b", 50)
    assert expected_a["blocks"] == CALLS and expected_a["layer_calls"] == CALLS
    assert not compute._PROBES and not live_queue.recorders_active()

    start, end = threading.Barrier(2), threading.Barrier(2)
    found: dict = {}

    def recorder_thread():
        with _Recorders(block) as rec:
            start.wait(TIMEOUT)
            rec.active.append(live_queue.recorders_active())
            _forwards(block, "a", 10)
            end.wait(TIMEOUT)
        found["a"] = rec

    def other_thread():
        if other == "own":
            with _Recorders(block) as rec:
                start.wait(TIMEOUT)
                _forwards(block, "b", 50)
                end.wait(TIMEOUT)
            found["b"] = rec
        else:
            start.wait(TIMEOUT)
            found["active_b"] = live_queue.recorders_active()
            found["probes_b"] = bool(compute._PROBES)
            _forwards(block, "b", 50)
            end.wait(TIMEOUT)

    _run(recorder_thread, other_thread)
    rec_a = found["a"]
    assert rec_a.active == [True]
    assert _summary(rec_a) == expected_a
    assert {ident for _, ident in rec_a.blocks.seen} == {rec_a.blocks.seen[0][1]}
    assert "b" not in rec_a.shares and "b" not in rec_a.routes and "b" not in rec_a.probes
    if other == "own":
        assert _summary(found["b"]) == expected_b
        assert "a" not in found["b"].shares and "a" not in found["b"].probes
    else:
        assert found["active_b"] is False
        assert found["probes_b"] is False
    assert not compute._PROBES and not live_queue.recorders_active()


def test_a_thread_without_recorders_takes_the_fast_path_during_a_probes_pass(monkeypatch):
    block = torch.nn.Identity()
    checked = compute._fast_path_ok
    results: dict[int, list[bool]] = {}
    lock = threading.Lock()

    def spy(*args, **kwargs):
        ok = checked(*args, **kwargs)
        with lock:
            results.setdefault(threading.get_ident(), []).append(ok)
        return ok

    monkeypatch.setattr(compute, "_fast_path_ok", spy)
    alone = _forwards(block, "b", 50)
    reference = results.pop(threading.get_ident())
    assert any(reference)

    start, end = threading.Barrier(2), threading.Barrier(2)
    idents: dict[str, int] = {}
    outs: dict[str, list] = {}

    def probes_thread():
        idents["probes"] = threading.get_ident()
        with compute.key_probes():
            start.wait(TIMEOUT)
            _forwards(block, "a", 10)
            end.wait(TIMEOUT)

    def plain_thread():
        idents["plain"] = threading.get_ident()
        start.wait(TIMEOUT)
        outs["plain"] = _forwards(block, "b", 50)
        end.wait(TIMEOUT)

    _run(probes_thread, plain_thread)
    assert results[idents["plain"]] == reference
    assert not any(results.get(idents["probes"], []))
    assert all(torch.equal(x, y) for x, y in zip(outs["plain"], alone, strict=True))


def test_recorders_active_is_false_in_a_thread_without_recorders():
    start, end = threading.Barrier(2), threading.Barrier(2)
    seen: dict[str, bool] = {}

    def recorder_thread():
        with compute.key_probes(), compute.timing_layers(), observing_blocks([], _Blocks()):
            seen["recording"] = live_queue.recorders_active()
            start.wait(TIMEOUT)
            end.wait(TIMEOUT)

    def plain():
        start.wait(TIMEOUT)
        seen["plain"] = live_queue.recorders_active()
        seen["blocks"] = bool(recording._BLOCK_OBSERVERS)
        end.wait(TIMEOUT)

    _run(recorder_thread, plain)
    assert seen == {"recording": True, "plain": False, "blocks": False}


def test_a_context_copy_sees_the_recorders_and_a_new_thread_does_not():
    with compute.key_probes() as record, compute.recording_shares(lambda *a: None):
        copied = contextvars.copy_context()
        assert copied.run(lambda: list(compute._PROBES)) == [record]
        assert copied.run(live_queue.recorders_active) is True
        seen = {}

        def thread():
            seen["probes"] = list(compute._PROBES)
            seen["active"] = live_queue.recorders_active()

        _run(thread)
        assert seen == {"probes": [], "active": False}
    assert copied.run(lambda: list(compute._PROBES)) == [record]
    assert not compute._PROBES


def test_a_copied_context_does_not_change_the_parent():
    def inner():
        with compute.recording_routes():
            return bool(compute._ROUTES)

    assert contextvars.copy_context().run(inner) is True
    assert not compute._ROUTES


def test_context_stack_order_and_out_of_order_exit():
    stack = ContextStack("kilora_test_stack")
    assert not stack and len(stack) == 0 and list(stack) == []
    a, b = object(), object()
    outer = stack.pushed(a)
    inner = stack.pushed(b)
    outer.__enter__()
    inner.__enter__()
    assert list(stack) == [a, b] and stack[-1] is b and a in stack and stack.holds(b)
    outer.__exit__(None, None, None)
    assert list(stack) == [b]
    inner.__exit__(None, None, None)
    assert not stack and stack.get() == ()


def test_the_same_entry_twice_is_removed_once():
    stack = ContextStack("kilora_test_stack_twice")
    fn = object()
    with stack.pushed(fn):
        with stack.pushed(fn):
            assert len(stack) == 2
        assert list(stack) == [fn]
    assert not stack


def test_hidden_entries_return_after_the_block():
    stack = ContextStack("kilora_test_stack_hidden")
    a, b = object(), object()
    with stack.pushed(a):
        with stack.hidden() as entries:
            assert entries == (a,) and not stack
            with stack.pushed(b):
                assert list(stack) == [b]
            assert not stack
        assert list(stack) == [a]
    assert not stack


def test_layer_depth_is_per_context():
    assert compute._LAYER_DEPTH.get() == 0
    block = torch.nn.Identity()
    with compute.timing_layers() as timer:
        _forwards(block, "a", 10)
    assert timer["layer_calls"] == CALLS
    assert compute._LAYER_DEPTH.get() == 0
