# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_lowrank_depth import _sets

from kilora.comfyui.patching import StepCarry
from kilora.compute import dropout_key, run_layer, step_carry

SETS = _sets(1, 16, 4, 2, 8, seed=3)


def _generation(x, carrier, steps=4, depth=1):
    outs = []
    for i in range(steps):
        t = torch.tensor([float(steps - i)])
        y0 = x * (1.0 + 0.1 * i)
        args = {"input": y0, "timestep": t, "c": {}}
        with dropout_key(0, t), torch.no_grad():
            if carrier is None:
                outs.append(run_layer(y0, y0, SETS, depth, target="layer"))
            else:
                with step_carry(carrier.cache_for(args)):
                    outs.append(run_layer(y0, y0, SETS, depth, target="layer"))
    return outs


def _x(batch=2, seed=1):
    return torch.randn(batch, 7, 16, generator=torch.Generator().manual_seed(seed))


def test_carry_off_is_bit_identical_and_the_first_step_matches():
    off = _generation(_x(), None)
    on = _generation(_x(), StepCarry())
    assert torch.equal(on[0], off[0])
    assert not torch.equal(on[2], off[2])


def test_two_generations_in_a_row_are_identical():
    carrier = StepCarry()
    first = _generation(_x(), carrier)
    second = _generation(_x(), carrier)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_reset_on_a_rising_timestep_and_on_a_new_shape():
    carrier = StepCarry()
    a = carrier.cache_for({"input": _x(), "timestep": torch.tensor([3.0]), "c": {}})
    a["layer"] = ["state"]
    same = carrier.cache_for({"input": _x(), "timestep": torch.tensor([2.0]), "c": {}})
    assert same is a
    rising = carrier.cache_for({"input": _x(), "timestep": torch.tensor([4.0]), "c": {}})
    assert rising == {}
    other = carrier.cache_for({"input": _x(batch=3), "timestep": torch.tensor([1.0]), "c": {}})
    assert other == {} and other is not rising


def test_batch_items_are_independent():
    both = _generation(_x(batch=2), StepCarry())
    alone = _generation(_x(batch=2)[:1], StepCarry())
    for a, b in zip(both, alone, strict=True):
        torch.testing.assert_close(a[:1], b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("calls", ["batched", "separate"])
def test_preview_loop_builds_up_depth_with_guidance(calls, monkeypatch):
    from kilora import compute

    class Toy(torch.nn.Module):
        def forward(self, x, t):
            return run_layer(x, x, SETS, 1, target="layer")

    toy, log = Toy(), []
    real = compute._carry_start

    def spy(target, shape, n):
        out = real(target, shape, n)
        log.append(None if out is None else out["depth"])
        return out

    monkeypatch.setattr(compute, "_carry_start", spy)
    x = _x(batch=1)
    ratios = []
    with compute.generation(toy, carry=True), torch.no_grad():
        for i in range(8):
            t = torch.tensor([1.0 - i / 8])
            y0 = x * (1 + 0.05 * i)
            if calls == "batched":
                out = toy(torch.cat([y0, -y0]), t)[:1]
            else:
                out = toy(y0, t)
                toy(-y0, t)
            ratios.append(float((out - y0).norm() / y0.norm()))
    per_call = 1 if calls == "batched" else 2
    want = [None] * per_call + [d for d in range(1, 8) for _ in range(per_call)]
    assert log == want
    assert all(b > a for a, b in zip(ratios, ratios[1:], strict=False))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("depth", [1, 5])
def test_carried_state_stays_bounded_over_a_generation(depth, carry, fast, monkeypatch):
    import dataclasses
    import math

    from kilora import compute

    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    sets = [dataclasses.replace(s, carry=carry) for s in SETS]
    carrier = StepCarry()
    x = _x()
    norms = []
    for i in range(30):
        t = torch.tensor([float(30 - i)])
        y0 = x * (1.0 + 0.01 * i)
        args = {"input": y0, "timestep": t, "c": {}}
        cache = carrier.cache_for(args)
        with dropout_key(0, t), torch.no_grad(), step_carry(cache):
            run_layer(y0, y0, sets, depth, target="layer")
        for s, c in zip(sets, cache["layer"]["carried"], strict=True):
            r = s.per_expert
            base = (y0 @ s.down.t()).reshape(*y0.shape[:-1], -1, r)
            sigma0 = base.pow(2).mean(-1, keepdim=True).sqrt()
            peak = sigma0 / (math.e * math.log(2))
            assert bool((c.reshape(base.shape).abs() <= peak * (1 + 1e-5) + 1e-6).all())
        norms.append(float(torch.cat([c.float() for c in cache["layer"]["carried"]], -1).norm()))
    assert max(norms) <= 2 * norms[0], norms


def test_carry_is_an_opt_in_and_old_files_load_with_it_off(model, tmp_path):
    from safetensors.torch import load_file
    from test_adapter import make

    from kilora import load_mod
    from kilora.adapter import Settings
    from kilora.modfile import _unpack_sets, save_mod

    assert Settings().carry_across_steps is False
    adapter = make(model)
    sets = adapter.to_sets()
    adapter.remove()
    path = tmp_path / "c.kmod.safetensors"
    save_mod(path, sets, Settings(carry_across_steps=True), id="c")
    mod = load_mod(path)
    assert mod.settings.carry_across_steps is True
    record = dict(mod.manifest["adapter"])
    del record["carry_across_steps_experimental"]
    record["carry_across_steps"] = True
    _, settings = _unpack_sets(record, load_file(str(path)))
    assert settings.carry_across_steps is False
