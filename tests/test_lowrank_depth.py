# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch

from kilora import compute
from kilora.compute import LayerSet, run_layer


def _sets(n_sets, width, experts, per_expert, router_dim, seed, modes=None, scales=None):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_sets):
        cols = experts * per_expert
        out.append(
            LayerSet(
                import_score="keys",
                router=torch.randn(router_dim, width, generator=gen) / width**0.5,
                keys_in=torch.randn(experts, min(4, router_dim), router_dim, generator=gen),
                per_expert=per_expert,
                down=torch.randn(cols, width, generator=gen) / width**0.5,
                up=torch.randn(cols, width, generator=gen) * 0.3 / cols**0.5,
                scale=(scales or [1.0] * n_sets)[i],
                log_prior=0.1 * i,
                aggregation=(modes or ["accumulate"] * n_sets)[i],
                key=f"s{i}",
            )
        )
    return out


def _both(y0, sets, depth, shared, monkeypatch):
    choices = {}
    real = compute._gates_from_logits

    def run(fast):
        seen = []

        def spy(logits, s, sh, noise, regions=None):
            gates = real(logits, s, sh, noise, regions)
            seen.append([g.argmax(-1) for g in gates])
            return gates

        monkeypatch.setattr(compute, "_gates_from_logits", spy)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
        with torch.no_grad():
            out = run_layer(y0.clone(), y0.clone(), sets, depth, shared=shared)
        choices[fast] = seen
        return out

    return run(False), run(True), choices


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("n_sets", [1, 2, 3])
@pytest.mark.parametrize("shared", [False, True])
def test_fast_path_matches_the_loop(depth, n_sets, shared, monkeypatch):
    scales = [1.0, -0.7, 1.6][:n_sets]
    modes = ["accumulate", "replace", "accumulate"][:n_sets]
    sets = _sets(
        n_sets,
        48,
        experts=5,
        per_expert=2,
        router_dim=8,
        seed=depth * 10 + n_sets,
        modes=modes,
        scales=scales,
    )
    y0 = torch.randn(2, 11, 48, generator=torch.Generator().manual_seed(depth))
    loop, fast, choices = _both(y0, sets, depth, shared, monkeypatch)
    assert torch.allclose(loop, fast, rtol=1e-4, atol=1e-5), (loop - fast).abs().max()
    assert len(choices[False]) == len(choices[True]) == depth
    for a, b in zip(choices[False], choices[True], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_fast_path_replays_bit_identically():
    sets = _sets(2, 64, experts=4, per_expert=3, router_dim=8, seed=1)
    y0 = torch.randn(3, 7, 64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a = run_layer(y0, y0, sets, 5, shared=True)
        b = run_layer(y0, y0, sets, 5, shared=True)
    assert torch.equal(a, b)


def test_cache_follows_changed_tensors():
    sets = _sets(1, 32, experts=3, per_expert=2, router_dim=8, seed=3)
    y0 = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        before = run_layer(y0, y0, sets, 3)
        sets[0].up.mul_(2.0)
        after = run_layer(y0, y0, sets, 3)
        compute.LOWRANK_ENABLED = False
        try:
            want = run_layer(y0, y0, sets, 3)
        finally:
            compute.LOWRANK_ENABLED = True
    assert not torch.allclose(before, after)
    assert torch.allclose(after, want, rtol=1e-4, atol=1e-5)


def test_fast_path_is_not_used_where_it_would_differ(monkeypatch):
    calls = []
    real = compute._run_lowrank
    monkeypatch.setattr(compute, "_run_lowrank", lambda *a, **k: calls.append(1) or real(*a, **k))
    sets = _sets(1, 32, experts=3, per_expert=2, router_dim=8, seed=5)
    y0 = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        run_layer(y0, y0, sets, 2, noise_scale=0.5)
        run_layer(y0, y0, sets, 2, query_policy=lambda q: q * 0.5)
        sets[0].aggregation = "average"
        run_layer(y0, y0, sets, 2)
        sets[0].aggregation = "accumulate"
    y = y0.clone().requires_grad_(True)
    run_layer(y, y, sets, 2)
    assert calls == []
    with torch.no_grad():
        run_layer(y0, y0, sets, 2)
    assert calls == [1]


def test_benchmark_realistic_widths(monkeypatch, capsys):
    sets = _sets(1, 3840, experts=16, per_expert=4, router_dim=32, seed=7)
    y0 = torch.randn(1, 512, 3840, generator=torch.Generator().manual_seed(8))
    rows = []
    for depth in (1, 3, 7):
        times = {}
        for fast in (False, True):
            monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
            with torch.no_grad():
                run_layer(y0, y0, sets, depth)
                start = time.perf_counter()
                for _ in range(3):
                    run_layer(y0, y0, sets, depth)
            times[fast] = (time.perf_counter() - start) / 3 * 1000
        rows.append((depth, times[False], times[True]))
    with capsys.disabled():
        for depth, loop, fast in rows:
            print(f"depth {depth}: loop {loop:.1f} ms, fast {fast:.1f} ms")


def test_loop_setting(monkeypatch):
    import subprocess
    import sys

    calls = []
    real = compute._run_lowrank
    monkeypatch.setattr(compute, "_run_lowrank", lambda *a, **k: calls.append(1) or real(*a, **k))
    sets = _sets(1, 32, experts=3, per_expert=2, router_dim=8, seed=9)
    y0 = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(9))
    try:
        compute.use_depth_loop(True)
        with torch.no_grad():
            run_layer(y0, y0, sets, 2)
        assert calls == []
    finally:
        compute.use_depth_loop(False)
    with torch.no_grad():
        run_layer(y0, y0, sets, 2)
    assert calls == [1]
    code = "import kilora.compute as c; print(c.LOWRANK_ENABLED)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "KILORA_DEPTH_PATH": "loop"},
    )
    assert out.stdout.strip() == "False"


def test_set_share_observer_preserves_plain_and_fast_routes(monkeypatch):
    sets = _sets(1, 32, experts=3, per_expert=2, router_dim=8, seed=27)
    y = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(28))
    outputs, routes = [], []
    for fast in (False, True):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
        observed = []
        with (
            torch.no_grad(),
            compute.recording_set_shares(
                lambda target, gates, layer_sets, observed=observed: observed.append(
                    (gates[0], layer_sets[0].key)
                )
            ),
        ):
            result = run_layer(y, y, sets, 3, target="test")
        outputs.append(result)
        routes.append(observed)
        with torch.no_grad():
            replay = run_layer(y, y, sets, 3, target="test")
        assert torch.equal(result, replay)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
    assert len(routes[0]) == len(routes[1]) == 3
    for a, b in zip(*routes, strict=True):
        assert a[1] == b[1] == "s0"
        torch.testing.assert_close(a[0], b[0], rtol=1e-4, atol=1e-5)
