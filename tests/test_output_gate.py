# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import KiLoRAConfig, attach
from kilora import compute as C
from kilora.compute import dropout_key, gate_summary, new_gate, recording_gates, run_layer


def _gated(kind, sets, seed=0, **kw):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in sets:
        gate = new_gate(kind, s.keys_in.shape[0], s.up.shape[1], gen, s.up.shape[0])
        out.append(dataclasses.replace(s, output_gate=kind, gate=gate, **kw))
    return out


def _run(y0, sets, depth, fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, target="layer")


def _y0(seed=1):
    return torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(seed))


def _double(sets):
    out = []
    for s in sets:
        parts = {n: getattr(s, n).double() for n in ("router", "keys_in", "down", "up")}
        gate = None if s.gate is None else {k: v.double() for k, v in s.gate.items()}
        out.append(dataclasses.replace(s, gate=gate, **parts))
    return out


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_gate_of_one_equals_no_gate(kind, iteration, monkeypatch):
    sets = _sets(2, 16, 4, 2, 8, seed=2)
    plain = _double([dataclasses.replace(s, iteration=iteration) for s in sets])
    gated = _gated(kind, plain)
    gated = _double(gated)
    for s in gated:
        for name in {"neuron": ["b2"], "lowrank": ["b"], "separable": ["cb2", "tb2"]}[kind]:
            s.gate[name].fill_(100.0)
    for depth in (1, 3):
        with dropout_key(1, 0.5):
            torch.testing.assert_close(
                _run(_y0().double(), gated, depth, False, monkeypatch),
                _run(_y0().double(), plain, depth, False, monkeypatch),
                rtol=0,
                atol=1e-12,
            )


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
def test_replay_is_bit_identical_and_keyed(kind, monkeypatch):
    sets = _gated(kind, _sets(2, 16, 4, 2, 8, seed=3))
    with dropout_key(2, 0.3):
        a = _run(_y0(), sets, 3, False, monkeypatch)
        b = _run(_y0(), sets, 3, False, monkeypatch)
    with dropout_key(2, 0.7):
        c = _run(_y0(), sets, 3, False, monkeypatch)
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_separable_gate_fast_path_equals_the_loop(carry, monkeypatch):
    sets = _gated("separable", _sets(2, 16, 4, 2, 8, seed=8, scales=[1.0, -0.7]), carry=carry)
    assert C._fast_path_ok(_y0(), sets, 0.0, None, None)
    for depth in (1, 3, 5):
        with dropout_key(1, 0.4):
            fast = _run(_y0(), sets, depth, True, monkeypatch)
            loop = _run(_y0(), sets, depth, False, monkeypatch)
        torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-5)


def test_fast_path_falls_back_to_the_loop(monkeypatch):
    sets = _gated("neuron", _sets(1, 16, 4, 2, 8, seed=4))
    assert not C._fast_path_ok(_y0(), sets, 0.0, None, None)
    with dropout_key(0, 0.1):
        assert torch.equal(
            _run(_y0(), sets, 4, True, monkeypatch), _run(_y0(), sets, 4, False, monkeypatch)
        )


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_mask_has_the_shape_of_the_change_and_is_hard(depth, monkeypatch):
    sets = _gated("neuron", _sets(1, 16, 4, 2, 8, seed=5))
    seen = []
    real = C.output_gate

    def spy(s, delta, *args):
        g = real(s, delta, *args)
        seen.append((delta.shape, g))
        return g

    monkeypatch.setattr(C, "output_gate", spy)
    out = _run(_y0(), sets, depth, False, monkeypatch)
    assert len(seen) == depth and bool(torch.isfinite(out).all())
    for shape, g in seen:
        assert g.shape == shape and set(g.unique().tolist()) <= {0.0, 1.0}
    rate = float(torch.stack([g for _, g in seen]).mean())
    assert 0.9 < rate < 0.99


def test_gated_positions_keep_the_base_output(monkeypatch):
    s = _gated("neuron", _sets(1, 16, 4, 2, 8, seed=6))[0]
    s.gate["b2"].fill_(-100.0)
    assert torch.equal(_run(_y0(), [s], 3, False, monkeypatch), _y0())


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
def test_straight_through_gradient_reaches_the_scorer(kind):
    sets = _gated(kind, _sets(1, 16, 4, 2, 8, seed=7))
    for t in sets[0].gate.values():
        t.requires_grad_(True)
    with dropout_key(0, 0.2):
        run_layer(_y0(), _y0(), sets, 3, target="layer").pow(2).mean().backward()
    for name, t in sets[0].gate.items():
        assert t.grad is not None and float(t.grad.abs().sum()) > 0, name
    if kind == "separable":
        return
    with C.hard_gates():
        g = C.output_gate(
            sets[0],
            _y0(),
            _y0(),
            _y0() * 0,
            torch.full((2, 9, 4), 0.25),
            torch.ones(2, 1, 16),
            "layer",
            0,
            1,
        )
    assert not g.requires_grad


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
def test_adapter_trains_saves_and_loads_the_gate(model, x, tmp_path, kind):
    from kilora import compose, load_mod
    from kilora.modfile import save_mod

    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.feed_forward.w2"],
            num_experts=4,
            routes=2,
            router_dim=8,
            depth=2,
            train_depth=None,
            output_gate=kind,
        ),
    )
    opt = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=0.1)
    model.train()
    for site in adapter.sites:
        with torch.no_grad():
            site.tensors["s0_up"].add_(0.1)
    with recording_gates() as qs:
        model(x).pow(2).mean().backward()
    summary = gate_summary(qs)
    assert 0.9 < summary["mean"] < 1.0 and summary["high"] > 0.5
    opt.step()
    model.eval()
    with torch.no_grad():
        want = model(x)
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    path = tmp_path / "gated.kmod.safetensors"
    save_mod(path, sets, settings, id="g")
    mod = load_mod(path)
    assert mod.settings.output_gate == kind
    assert next(iter(mod.sets[0].experts.values())).gate is not None
    loaded = compose(model, [path])
    with torch.no_grad():
        torch.testing.assert_close(model(x), want, rtol=1e-5, atol=1e-6)
    loaded.remove()


def test_neuron_gate_chunks_give_the_same_scores(monkeypatch):
    gate = new_gate("neuron", 2, 600, torch.Generator().manual_seed(0))
    feats = torch.randn(2, 5, 600, 3, generator=torch.Generator().manual_seed(1))
    chunked = C.gate_score("neuron", gate, feats, 1)
    monkeypatch.setattr(C, "GATE_CHUNK", 10**9)
    assert torch.equal(chunked, C.gate_score("neuron", gate, feats, 1))


def test_compiled_gate_scores_match_eager(monkeypatch):
    pytest.importorskip("torch._inductor")
    sets = _gated("separable", _sets(1, 16, 4, 2, 8, seed=9))
    with dropout_key(0, 0.3):
        eager = _run(_y0(), sets, 2, True, monkeypatch)
        monkeypatch.setenv("KILORA_COMPILE", "1")
        try:
            compiled = _run(_y0(), sets, 2, True, monkeypatch)
        except Exception as exc:
            pytest.skip(f"torch.compile is not available here: {exc}")
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["separable", "neuron", "lowrank"])
def test_attention_records_and_trains_one_output_gate(model, x, kind):
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=4,
            routes=2,
            router_dim=8,
            depth=3,
            train_depth=None,
            output_gate=kind,
        ),
    )
    with torch.no_grad():
        adapter.sites[0].tensors["s0_up"].add_(0.1)
    model.train()
    with recording_gates() as stats:
        model(x).pow(2).mean().backward()
    assert gate_summary(stats)["mean"] > 0
    assert any(p.grad is not None for n, p in adapter.named_parameters() if "gate_" in n)
