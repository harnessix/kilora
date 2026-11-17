# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import time

import pytest
import torch

from kilora import compute, live_queue
from kilora.comfyui import live_write
from kilora.comfyui.live_write import PathHandle, image_loss
from kilora.compute import LayerSet, run_layer
from kilora.live_queue import Preempted, preempt_check, preemptible
from kilora.recording import Perturbation, observing_blocks

pytestmark = pytest.mark.async_live


class _Queue:
    def __init__(self):
        self._generating = 0
        self._prompts = 0
        self._stop = False
        self._generation_began = 0.0
        self._prompt_began = 0.0

    def _busy(self):
        return bool(self._generating or self._prompts)


class _Observer:
    def observe_block(self, index, value):
        raise AssertionError("no block to observe")


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


def _model(queue, fire, render):
    sets = [_set(n, f"s{n}") for n in range(3)]

    def apply_model(x_t, t, c_crossattn):
        y = torch.randn(1, 20, 32, generator=torch.Generator().manual_seed(1))
        for n, s in enumerate(sets):
            if n == 1 and fire[0]:
                fire[0] -= 1
                if render is not None:
                    render.enter_context(observing_blocks((), _Observer()))
                queue._generating = 1
                queue._generation_began = time.perf_counter()
            y = run_layer(y, y, [s], 3, target=f"layer{n}")
        return torch.zeros_like(x_t) + y.mean() + c_crossattn.mean()

    return apply_model


def _measured(monkeypatch, fire_count, env=(), observes=False):
    for name in (live_write.TIMING_ENV, live_write.FLIPS_ENV):
        monkeypatch.delenv(name, raising=False)
    for name in env:
        monkeypatch.setenv(name, "1")
    queue, fire, waits = _Queue(), [0], []
    render = contextlib.ExitStack() if observes else None

    def wait():
        if queue._generating:
            waits.append(time.perf_counter())
            if render is not None:
                render.close()
            queue._generating = 0
            return 0.01
        return 0.0

    latent = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(3))
    tokens = (live_queue._QUEUE.set(queue), live_queue._YIELD.set(wait))
    try:
        measure = image_loss(_handle(_model(queue, fire, render)), latent, event=0)
        fire[0] = fire_count
        value = measure(Perturbation({}, {}))
    finally:
        live_queue._YIELD.reset(tokens[1])
        live_queue._QUEUE.reset(tokens[0])
        if render is not None:
            render.close()
    return value, measure.timing, waits


def test_a_render_does_not_stop_a_write_call(monkeypatch):
    reference, timing0, _ = _measured(monkeypatch, 0)
    assert timing0["preempts"] == 0
    value, timing, _waits = _measured(monkeypatch, 2)
    assert value == reference
    assert timing["preempts"] == 0 and timing["preempt_dropped_s"] == 0.0
    assert len(timing["trial_s"]) == 1


def test_a_learning_render_with_block_observers_does_not_stop_a_write_call(monkeypatch):
    reference, _, _ = _measured(monkeypatch, 0)
    value, timing, _waits = _measured(monkeypatch, 2, observes=True)
    assert timing["preempts"] == 0
    assert value == reference


@pytest.mark.parametrize(
    "env", [(live_write.FLIPS_ENV,), (live_write.TIMING_ENV,)], ids=["routes", "timing"]
)
def test_no_stop_while_routes_or_layer_times_are_recorded(monkeypatch, env):
    reference, _, _ = _measured(monkeypatch, 0, env)
    value, timing, _ = _measured(monkeypatch, 1, env)
    assert timing["preempts"] == 0
    assert value == reference


def test_the_check_does_nothing_outside_the_writer():
    preempt_check()
    with preemptible():
        preempt_check()


def test_no_stop_while_the_writer_records():
    queue = _Queue()
    token = live_queue._QUEUE.set(queue)
    try:
        for recorder in (compute.key_probes, compute.timing_layers):
            queue._generating = 0
            with recorder(), preemptible():
                queue._generating = 1
                preempt_check()
        queue._generating = 0
        with preemptible(False):
            queue._generating = 1
            preempt_check()
    finally:
        live_queue._QUEUE.reset(token)


def test_a_recorder_opened_after_the_start_does_not_prevent_the_stop():
    queue = _Queue()
    token = live_queue._QUEUE.set(queue)
    try:
        with preemptible(), observing_blocks((), _Observer()):
            queue._generating = 1
            with pytest.raises(Preempted):
                preempt_check()
        queue._generating = 0
        with preemptible(), compute.key_probes():
            queue._generating = 1
            with pytest.raises(Preempted):
                preempt_check()
    finally:
        live_queue._QUEUE.reset(token)


def test_no_stop_when_a_render_runs_at_the_start():
    queue = _Queue()
    queue._generating = 1
    token = live_queue._QUEUE.set(queue)
    try:
        with preemptible():
            preempt_check()
        queue._generating = 0
        with preemptible():
            queue._generating = 1
            with pytest.raises(Preempted):
                preempt_check()
    finally:
        live_queue._QUEUE.reset(token)
