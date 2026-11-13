# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora import compute, fused
from kilora.comfyui import live_write
from kilora.comfyui.live_write import PathHandle, image_loss, timing_note, timing_split
from kilora.compute import LayerSet, observing_set_share_layers, recording_set_shares, run_layer
from kilora.recording import Perturbation


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


def test_the_time_split_adds_up_to_the_measured_total(monkeypatch):
    monkeypatch.setenv(live_write.TIMING_ENV, "1")
    monkeypatch.setenv(live_write.FLIPS_ENV, "1")
    sets = [_set(n, f"s{n}") for n in range(3)]
    gen = torch.Generator().manual_seed(5)
    weight = torch.randn(256, 256, generator=gen)

    def apply_model(x_t, t, c_crossattn):
        y = torch.randn(1, 20, 32, generator=torch.Generator().manual_seed(1))
        for n, s in enumerate(sets):
            y = run_layer(y, y, [s], 3, target=f"layer{n}")
        h = torch.randn(64, 256, generator=torch.Generator().manual_seed(2))
        for _ in range(20):
            h = torch.tanh(h @ weight)
        return torch.zeros_like(x_t) + float(y.mean()) * 0 + float(h.mean()) * 0

    measure = image_loss(_handle(apply_model), torch.randn(1, 5, 4), event=0)
    measure(None)
    for _ in range(3):
        measure(Perturbation({}, {}))
    timing = measure.timing
    split = timing_split(timing)
    assert split["observers"] > 0 and split["routed"] > 0 and split["rest"] > 0
    parts = split["observers"] + split["routed"] + split["rest"]
    assert abs(parts - split["total"]) <= 0.05 * split["total"]
    assert timing["layers_s"] <= split["total"] * 1.05
    note = timing_note(type("R", (), {"measure": measure})())
    assert "split: observers" in note and "routed layers" in note


def test_without_the_switch_the_measurement_does_not_time_layers(monkeypatch):
    monkeypatch.delenv(live_write.TIMING_ENV, raising=False)
    seen = []

    def apply_model(x_t, t, c_crossattn):
        seen.append(bool(compute._LAYER_TIMERS))
        return torch.zeros_like(x_t)

    measure = image_loss(_handle(apply_model), torch.randn(1, 5, 4), event=0)
    measure(Perturbation({}, {}))
    assert seen == [False] and "layers_s" not in measure.timing


def test_eager_layer_calls_are_counted_per_reason():
    y0 = torch.randn(1, 20, 32)
    before_eager, before = fused.COUNTS["eager"], sum(fused.REASONS.values())
    with recording_set_shares(lambda *a: None):
        for _ in range(3):
            fused.fused_ok(y0, [_set(0, "a")])
    assert fused.COUNTS["eager"] - before_eager == 3
    assert sum(fused.REASONS.values()) - before == 3


def _shares_and_outputs(scoped: bool, monkeypatch):
    shares, calls, recording = [], [], {}
    layers = {"a": [_set(0, "imp")], "b": [_set(1, "other")]}

    def observe(target, gates, sets):
        calls.append(target)
        for gate, layer in zip(gates, sets or (), strict=False):
            if layer.key == "imp":
                shares.append(gate.float().reshape(-1, gate.shape[-1]).mean(0))

    original = compute._run_layer

    def tracked(*args, **kwargs):
        recording[kwargs.get("target", "")] = bool(compute._SET_SHARES)
        return original(*args, **kwargs)

    monkeypatch.setattr(compute, "_run_layer", tracked)
    y = torch.randn(1, 20, 32, generator=torch.Generator().manual_seed(9))
    outs = []

    def wants(_target, sets):
        return any(s.key == "imp" for s in sets)

    block = observing_set_share_layers(wants, observe) if scoped else recording_set_shares(observe)
    with torch.no_grad(), block:
        for target in ("a", "b", "a"):
            outs.append(run_layer(y, y, layers[target], 3, target=target))
    monkeypatch.setattr(compute, "_run_layer", original)
    return shares, calls, recording, outs


def test_the_scoped_share_observer_reads_the_same_shares(monkeypatch):
    shares_g, calls_g, rec_g, outs_g = _shares_and_outputs(False, monkeypatch)
    shares_s, calls_s, rec_s, outs_s = _shares_and_outputs(True, monkeypatch)
    assert shares_g and len(shares_g) == len(shares_s)
    for a, b in zip(shares_g, shares_s, strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    assert torch.stack(shares_g).mean(0).allclose(torch.stack(shares_s).mean(0), atol=1e-6)
    for a, b in zip(outs_g, outs_s, strict=True):
        assert (a - b).abs().max() <= 1e-5
    assert "b" in calls_g and "b" not in calls_s
    assert rec_g["b"] and not rec_s["b"] and rec_s["a"]


def test_the_import_editor_uses_the_scoped_observer():
    import inspect

    from kilora import live_imports

    source = inspect.getsource(live_imports.ImportEditor._r1)
    assert "observing_set_share_layers" in source
