# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import threading
import time

import pytest
import torch

from kilora import call_values, compute, linear_jump, live_queue, route_freeze
from kilora.comfyui import live_write
from kilora.comfyui.live_write import PathHandle, image_loss
from kilora.compute import LayerSet, run_layer
from kilora.live_queue import WriteQueue
from kilora.recording import Perturbation

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


def _y(seed: int) -> torch.Tensor:
    return torch.randn(1, 20, 32, generator=torch.Generator().manual_seed(seed))


@contextlib.contextmanager
def _call(tag: int, cache: dict):
    with contextlib.ExitStack() as stack:
        stack.enter_context(call_values.token_depth_values(True, float(tag)))
        stack.enter_context(call_values.axis_alphas({"axis": float(tag)}))
        stack.enter_context(call_values.depth_pass(tag))
        stack.enter_context(compute.sampling_progress(tag / 10))
        stack.enter_context(compute.step_carry(cache))
        stack.enter_context(compute.dropout_key(tag, tag))
        stack.enter_context(compute.derived_call())
        stack.enter_context(route_freeze.freezing({"tag": tag}))
        stack.enter_context(linear_jump.recording(f"band{tag}"))
        yield


def _values() -> dict:
    return {
        "depth": call_values.current_token_depth(),
        "alphas": dict(call_values.current_axis_alphas()),
        "pass": call_values.current_depth_pass(),
        "progress": compute.current_progress(),
        "carry": compute._STEP_CARRY[-1] if compute._STEP_CARRY else None,
        "carries": len(compute._STEP_CARRY),
        "key": compute._DROPOUT_KEY[-1],
        "derived": len(compute._DERIVED_CARRY),
        "freeze": route_freeze._CACHES[-1] if route_freeze._CACHES else None,
        "jump": linear_jump._RECORDS[-1]["band"] if linear_jump._RECORDS else None,
    }


def _expected(tag: int, cache: dict) -> dict:
    return {
        "depth": (True, float(tag)),
        "alphas": {"axis": float(tag)},
        "pass": (tag, None),
        "progress": tag / 10,
        "carry": cache,
        "carries": 1,
        "key": (tag, tag),
        "derived": 1,
        "freeze": {"tag": tag},
        "jump": f"band{tag}",
    }


_EMPTY = {
    "depth": (None, None),
    "alphas": {},
    "pass": None,
    "progress": 0.0,
    "carry": None,
    "carries": 0,
    "key": (0, 0),
    "derived": 0,
    "freeze": None,
    "jump": None,
}


def _join(threads, errors):
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
        assert not t.is_alive()
    if errors:
        raise errors[0]


def test_each_thread_reads_only_its_own_call_values_and_no_cross_pop():
    inside = threading.Barrier(2, timeout=TIMEOUT)
    inner_left = threading.Event()
    seen: dict[str, list] = {"outer": [], "inner": []}
    errors: list[BaseException] = []
    sets = [_set(1, "s")]

    def worker(name: str, tag: int, inner: bool):
        cache: dict = {}
        try:
            assert _values() == _EMPTY
            with _call(tag, cache):
                run_layer(_y(tag), _y(tag), sets, 2, target=f"layer_{name}")
                inside.wait()
                seen[name].append(_values() == _expected(tag, cache))
                run_layer(_y(tag), _y(tag), sets, 2, target=f"layer_{name}")
                if not inner:
                    assert inner_left.wait(TIMEOUT)
                    seen[name].append(_values() == _expected(tag, cache))
            if inner:
                inner_left.set()
            seen[name].append(_values() == _EMPTY)
        except BaseException as error:
            errors.append(error)
            inside.abort()
            inner_left.set()

    _join(
        [
            threading.Thread(target=worker, args=("outer", 3, False)),
            threading.Thread(target=worker, args=("inner", 7, True)),
        ],
        errors,
    )
    assert seen == {"outer": [True, True, True], "inner": [True, True]}
    assert _values() == _EMPTY


def test_the_hard_gates_flag_is_per_context():
    entered, release = threading.Event(), threading.Event()
    seen = []

    def other():
        with compute.hard_gates():
            entered.set()
            release.wait(TIMEOUT)

    t = threading.Thread(target=other)
    t.start()
    try:
        assert entered.wait(TIMEOUT)
        seen.append(bool(compute._HARD_GATES))
    finally:
        release.set()
        t.join(TIMEOUT)
    assert seen == [False]
    with compute.hard_gates():
        assert compute._HARD_GATES
    assert not compute._HARD_GATES


class _Queue:
    def __init__(self):
        self._generating = 0
        self._prompts = 0
        self._stop = False
        self._generation_began = 0.0
        self._prompt_began = 0.0

    def _busy(self):
        return bool(self._generating or self._prompts)


def _handle(apply_model) -> PathHandle:
    return PathHandle(
        session=None,
        rule=None,
        recorder=None,
        timesteps=[0.5],
        conditioning={"c_crossattn": torch.zeros(1, 2)},
        empty_conditioning={"c_crossattn": torch.ones(1, 2)},
        apply_model=apply_model,
    )


_WRITE_SETS = [_set(n, f"w{n}") for n in range(3)]
_RENDER_SETS = [_set(10 + n, f"r{n}") for n in range(3)]


def _write(fire_count: int, on_fire=None, on_wait=None):
    queue, fire = _Queue(), [fire_count]

    def apply_model(x_t, t, c_crossattn):
        with _call(5, {}):
            y = _y(1)
            for n, s in enumerate(_WRITE_SETS):
                if n == 1 and fire[0]:
                    fire[0] -= 1
                    if on_fire is not None:
                        on_fire()
                    queue._generating = 1
                    queue._generation_began = time.perf_counter()
                y = run_layer(y, y, [s], 3, target=f"layer{n}")
        return torch.zeros_like(x_t) + y.mean() + c_crossattn.mean()

    def wait():
        if queue._generating:
            if on_wait is not None:
                on_wait()
            queue._generating = 0
            return 0.01
        return 0.0

    latent = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(3))
    tokens = (live_queue._QUEUE.set(queue), live_queue._YIELD.set(wait))
    try:
        measure = image_loss(_handle(apply_model), latent, event=0)
        value = measure(Perturbation({}, {}))
    finally:
        live_queue._YIELD.reset(tokens[1])
        live_queue._QUEUE.reset(tokens[0])
    return value, measure.timing


def _render(middle=None):
    cache: dict = {}
    with _call(2, cache):
        y = _y(2)
        y = run_layer(y, y, [_RENDER_SETS[0]], 3, target="render0")
        before = _values()
        if middle is not None:
            middle()
        after = _values()
        for n, s in enumerate(_RENDER_SETS[1:], 1):
            y = run_layer(y, y, [s], 3, target=f"render{n}")
    expected = _expected(2, cache)
    return y, before == expected, after == expected


@pytest.mark.async_live
def test_a_render_beside_a_running_write_gives_the_render_alone(monkeypatch):
    for name in (live_write.TIMING_ENV, live_write.FLIPS_ENV):
        monkeypatch.delenv(name, raising=False)
    alone, before, after = _render()
    assert before and after
    reference, _ = _write(0)

    inside = threading.Barrier(2, timeout=TIMEOUT)
    results: dict = {}
    errors: list[BaseException] = []

    def render():
        try:
            results["render"] = _render(inside.wait)
        except BaseException as error:
            errors.append(error)
            inside.abort()

    def write():
        try:
            results["write"] = _write(1, on_fire=inside.wait)
        except BaseException as error:
            errors.append(error)
            inside.abort()

    _join([threading.Thread(target=render), threading.Thread(target=write)], errors)
    y, before, after = results["render"]
    assert before and after
    assert torch.equal(y, alone)
    value, timing = results["write"]
    assert timing["preempts"] == 0
    assert value == reference


@pytest.mark.async_live
def test_a_write_runs_with_the_settings_of_the_submitting_thread():
    q = WriteQueue(max_pending=10)
    seen = {}

    def run():
        seen["mode"] = compute._GATE_STATS_MODE.get()
        seen["progress"] = compute.current_progress()
        seen["key"] = compute._DROPOUT_KEY[-1]
        seen["depth"] = call_values.current_token_depth()
        seen["queue"] = live_queue._QUEUE.get() is q
        seen["yield"] = live_queue._YIELD.get() is not None
        return "ok"

    def submit():
        with (
            compute.gate_stats_mode("torch"),
            compute.sampling_progress(0.5),
            compute.dropout_key(4, 4),
            call_values.token_depth_values(True, 2.0),
        ):
            q.submit(run)

    try:
        t = threading.Thread(target=submit)
        t.start()
        t.join(TIMEOUT)
        assert q.wait_idle(TIMEOUT)
    finally:
        q.close()
    assert seen == {
        "mode": "torch",
        "progress": 0.0,
        "key": (0, 0),
        "depth": (None, None),
        "queue": True,
        "yield": True,
    }
    assert [report for _, report in q.reports] == ["ok"]


def test_a_write_context_drops_every_call_stack():
    with (
        compute.sampling_progress(0.3),
        compute.step_carry({}),
        call_values.axis_alphas({"a": 1.0}),
        route_freeze.freezing({}),
    ):
        context = live_queue.write_context()
        assert compute._STEP_CARRY and compute.current_progress() == 0.3
    assert context.run(compute.current_progress) == 0.0
    assert not context.run(bool, compute._STEP_CARRY)
    assert context.run(call_values.current_axis_alphas) == {}
    assert not context.run(bool, route_freeze._CACHES)
