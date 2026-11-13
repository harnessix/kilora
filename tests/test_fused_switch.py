# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import threading

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute, fused
from kilora.compute import dropout_key, run_layer


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "device_ok", lambda t: True)
    calls = []
    me = threading.get_ident()

    def stand_in(y0, *a, **k):
        if threading.get_ident() == me:
            calls.append(k.get("name", "state"))
        return y0

    monkeypatch.setattr(fused, "run_fused", stand_in)
    return calls


def _layer(form):
    s = dataclasses.replace(_sets(1, 16, 4, 2, 8, seed=1)[0], gate_mode="adaptive")
    if form == "input":
        s = dataclasses.replace(s, form="input")
    return s


@pytest.mark.parametrize("form", ["state", "input"])
def test_switch_is_read_at_every_call(form, available, monkeypatch):
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(2))
    s = _layer(form)
    for setting, want in (("1", 1), ("0", 0), ("1", 1), ("0", 0)):
        monkeypatch.setenv("KILORA_FUSED", setting)
        before = len(available)
        with torch.no_grad(), dropout_key(0, 0.5):
            run_layer(y0, y0, [s], 3, target="layer")
        assert len(available) - before == want, (form, setting)
    assert all(c in ("state", "layer#input0") for c in available)


def test_switch_off_is_the_eager_result(available, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "0")
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(3))
    s = _layer("input")
    with torch.no_grad(), dropout_key(0, 0.5):
        got = run_layer(y0, y0, [s], 3, target="layer")
        want = compute._run_input(y0, y0, [s], [0], 3, [None], "layer")
    assert torch.equal(got, want) and not available


@pytest.mark.parametrize("form", ["state", "input"])
def test_eager_block_turns_the_fused_path_off(form, available, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "1")
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(4))
    s = _layer(form)
    with torch.no_grad(), dropout_key(0, 0.5), compute.eager():
        run_layer(y0, y0, [s], 3, target="layer")
    assert not available
    with torch.no_grad(), dropout_key(0, 0.5):
        run_layer(y0, y0, [s], 3, target="layer")
    assert len(available) == 1


def test_state_key_pass_runs_eager(monkeypatch):
    from kilora import fast_keys

    seen = []
    monkeypatch.setattr(
        fast_keys,
        "record_queries",
        lambda *a, **k: (
            seen.append(bool(compute._EAGER)) or (_ for _ in ()).throw(RuntimeError("stop"))
        ),
    )

    class Adapter:
        current_depth, depth_schedule, eval_mode = 1, None, False
        log_taus = {"s0": torch.zeros(())}
        settings = type("Settings", (), {"route_freeze": None})()

        def set_depth(self, d):
            self.current_depth = d

    with pytest.raises(RuntimeError):
        fast_keys.state_keys_from_call(Adapter(), lambda: None)
    assert seen == [True] and not compute._EAGER
