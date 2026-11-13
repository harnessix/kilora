# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import KiLoRAConfig, attach
from kilora import compute as C
from kilora.compute import (
    dropout_key,
    norm_energy,
    orthonormal_keys,
    out_project,
    recording_routes,
    run_layer,
)
from kilora.subspace_keys import (
    build_subspace_keys,
    load_subspace_keys,
    save_subspace_keys,
    subspace_distance,
)


def _run(y0, sets, depth, fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, target="layer")


def test_energy_is_scale_invariant():
    keys = orthonormal_keys(torch.randn(5, 3, 8, generator=torch.Generator().manual_seed(0)))
    q = torch.randn(4, 7, 8, generator=torch.Generator().manual_seed(1))
    s = norm_energy(q, keys, None)
    for scale in (0.05, 0.5, 40.0):
        torch.testing.assert_close(norm_energy(q * scale, keys, None), s, rtol=1e-3, atol=1e-4)
    inside = keys[2, 0] * 3.0
    torch.testing.assert_close(
        norm_energy(inside, keys, None)[2], torch.tensor(8 / 3), atol=1e-4, rtol=0
    )


def test_normalization_removes_the_row_count_advantage():
    d = 16
    q = torch.randn(20000, d, generator=torch.Generator().manual_seed(2))
    for m in (1, 4, 12):
        keys = orthonormal_keys(torch.randn(1, m, d, generator=torch.Generator().manual_seed(m)))
        assert abs(float(norm_energy(q, keys, None).mean()) - 1.0) < 0.03


def test_rows_per_expert_use_a_prefix_and_pad_with_zeros():
    raw = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(3))
    rows = torch.tensor([1, 4, 2])
    k = orthonormal_keys(raw, rows)
    for e, r in enumerate(rows.tolist()):
        torch.testing.assert_close(k[e, :r] @ k[e, :r].t(), torch.eye(r), atol=1e-5, rtol=0)
        assert bool((k[e, r:] == 0).all())
        full = orthonormal_keys(raw[e : e + 1, :r])[0]
        assert subspace_distance(k[e, :r], full) < 1e-5
    q = torch.randn(10, 8, generator=torch.Generator().manual_seed(4))
    alone = norm_energy(q, orthonormal_keys(raw[2:3, :2]), None)[:, 0]
    torch.testing.assert_close(norm_energy(q, k, rows)[:, 2], alone, atol=1e-5, rtol=1e-5)


def test_transition_projects_onto_the_acting_experts_output_keys():
    kout = orthonormal_keys(torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(5)))
    q = torch.randn(6, 8, generator=torch.Generator().manual_seed(6))
    one = torch.nn.functional.one_hot(torch.full((6,), 1), 3).float()
    torch.testing.assert_close(out_project(q, kout, one), (q @ kout[1].t()) @ kout[1])
    mix = torch.tensor([0.25, 0.75, 0.0]).expand(6, 3)
    want = 0.25 * (q @ kout[0].t()) @ kout[0] + 0.75 * (q @ kout[1].t()) @ kout[1]
    torch.testing.assert_close(out_project(q, kout, mix), want)


def _dual(sets, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return [
        dataclasses.replace(s, keys_out=torch.randn(s.keys_in.shape, generator=gen)) for s in sets
    ]


@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_fast_path_equals_the_loop_and_replays(iteration, monkeypatch):
    sets = _dual(_sets(2, 16, 4, 2, 8, seed=8, modes=["accumulate", "replace"], scales=[1.0, -0.7]))
    sets = [dataclasses.replace(s, iteration=iteration) for s in sets]
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(9))
    for depth in (1, 3, 6):
        with dropout_key(4, 0.2):
            with recording_routes() as a:
                loop = _run(y0, sets, depth, False, monkeypatch)
            again = _run(y0, sets, depth, False, monkeypatch)
        with dropout_key(4, 0.2), recording_routes() as b:
            fast = _run(y0, sets, depth, True, monkeypatch)
        assert C.routing_flips(a, b) == 0
        torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-6)
        assert torch.equal(loop, again)


def test_path_output_replaces_the_previous_change(monkeypatch):
    s = dataclasses.replace(_sets(1, 16, 1, 4, 8, seed=10)[0], iteration="path", carry="drop")
    y0 = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(11))
    monkeypatch.setattr(C, "read_dropout", lambda z, *a, **k: torch.ones_like(z))
    state = y0
    for depth in (1, 2, 3, 4):
        z = state @ s.down.t()
        state = y0 + z @ s.up
        actual = _run(y0, [s], depth, True, monkeypatch)
        torch.testing.assert_close(actual, state)


def test_path_with_one_expert_rereads_the_carried_state(monkeypatch):
    s = dataclasses.replace(_sets(1, 16, 1, 4, 8, seed=12)[0], iteration="path")
    changed = dataclasses.replace(
        s, keys_in=torch.zeros_like(s.keys_in), keys_out=torch.ones_like(s.keys_in)
    )
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(13))
    for depth in (1, 2, 5):
        with dropout_key(3, 0.5):
            expected = _run(y0, [s], depth, False, monkeypatch)
            actual = _run(y0, [changed], depth, True, monkeypatch)
        assert torch.equal(actual, expected)


def test_derived_state_path_ignores_learned_output_keys(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = _sets(1, 16, 4, 2, 8, seed=14)[0]
    s = dataclasses.replace(s, iteration="path", import_score="derived")
    changed = dataclasses.replace(
        s,
        router=torch.zeros_like(s.router),
        keys_in=torch.zeros_like(s.keys_in),
        keys_out=torch.ones_like(s.keys_in),
    )
    y0 = torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(16))
    for depth in (1, 3):
        with recording_routes() as a:
            expected = _run(y0, [s], depth, True, monkeypatch)
        with recording_routes() as b:
            actual = _run(y0, [changed], depth, True, monkeypatch)
        assert torch.equal(actual, expected)
        assert C.routing_flips(a, b) == 0


@pytest.mark.parametrize("loss", ["accumulate", "path", "mix"])
def test_training_keeps_keys_orthonormal_in_every_mode(model, x, loss):
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=4,
            routes=2,
            router_dim=8,
            m_in=3,
            m_out=2,
            depth=3,
            train_depth=None,
            train_iteration=loss,
        ),
    )
    params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.5)
    model.train()
    for step in range(3):
        adapter.start_step(step)
        for site in adapter.sites:
            with torch.no_grad():
                site.tensors["s0_up"].add_(0.1)
        model(x).pow(2).mean().backward()
        assert adapter.log_taus["s0"].grad is not None
        opt.step()
        opt.zero_grad()
    ex = next(iter(adapter.to_sets()[0].experts.values()))
    for keys, rows in ((ex.keys_in, ex.rows_in), (ex.keys_out, ex.rows_out)):
        assert keys.shape[0] == 4 and keys.shape[-1] == 16
        for expert, count in zip(keys, rows.tolist(), strict=True):
            torch.testing.assert_close(
                expert[:count] @ expert[:count].T, torch.eye(count), atol=1e-5, rtol=0
            )
            assert not expert[count:].count_nonzero()
    assert all(
        not p.requires_grad
        for site in adapter.sites
        for key, p in site.tensors.items()
        if "_keys_" in key
    )


class _One(torch.nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)


def _planted(depth):
    torch.manual_seed(17)
    net = _One()
    with torch.no_grad():
        net.proj.weight.copy_(torch.eye(16))
    adapter = attach(
        net,
        KiLoRAConfig(
            target_modules=["proj"],
            num_experts=3,
            routes=1,
            rank=2,
            router_dim=8,
            m_in=1,
            m_out=1,
            depth=depth,
            train_depth=None,
            carry="drop",
            output_gate="none",
        ),
    )
    t = adapter.sites[0].tensors
    with torch.no_grad():
        adapter.routers["s0_16"].copy_(torch.eye(16)[:8])
        adapter.log_taus["s0"].fill_(8.0)
        down = torch.zeros_like(t["s0_down"])
        down[..., 0, 0] = 1.0
        down[..., 1, 1] = 1.0
        t["s0_down"].copy_(down)
        up = torch.zeros_like(t["s0_up"])
        up[..., 0, 2] = 0.5
        up[..., 1, 3] = 0.5
        t["s0_up"].copy_(up)
    gen = torch.Generator().manual_seed(18)
    batches = []
    for _ in range(3):
        b = torch.zeros(2, 32, 16)
        b[..., :2] = torch.randn(2, 32, 1, generator=gen)
        batches.append(b)
    real = C.read_dropout
    C.read_dropout = lambda z, *a: torch.ones_like(z)
    try:
        keys = build_subspace_keys(
            adapter, batches, lambda b: net.proj(b).sum(), source="states", m_in=1, m_out=1
        )["proj"]
    finally:
        C.read_dropout = real
    return keys


def test_state_keys_recover_the_read_and_write_subspaces():
    read = torch.eye(16)[:2]
    write = torch.eye(16)[2:4]
    for depth in (1, 2):
        keys = _planted(depth)
        for expert in range(3):
            assert subspace_distance(keys["keys_in"][expert], read) < 1e-4
            assert subspace_distance(keys["keys_out"][expert], write) < 1e-4


def test_state_key_export_does_not_fit_loss_directions():
    torch.manual_seed(19)
    net = _One()
    adapter = attach(
        net,
        KiLoRAConfig(
            target_modules=["proj"],
            num_experts=3,
            routes=1,
            router_dim=8,
            m_in=2,
            m_out=2,
            depth=1,
            train_depth=None,
        ),
    )

    def unused_loss(_batch):
        raise AssertionError("state key export must not fit a second routing mechanism")

    keys = build_subspace_keys(adapter, [None], unused_loss)["proj"]
    exported = adapter.to_sets()[0].experts["proj"]
    for name in ("keys_in", "keys_out", "rows_in", "rows_out"):
        assert torch.equal(keys[name], getattr(exported, name))
    adapter.remove()


def test_key_files_round_trip_and_start_training(tmp_path):
    keys = {
        "proj": {
            "keys_in": orthonormal_keys(
                torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(22))
            ),
            "keys_out": orthonormal_keys(
                torch.randn(3, 3, 8, generator=torch.Generator().manual_seed(23))
            ),
            "rows_in": torch.tensor([2, 2, 2]),
            "rows_out": torch.tensor([3, 3, 3]),
        }
    }
    path = tmp_path / "keys.safetensors"
    save_subspace_keys(path, keys, "states")
    loaded, source = load_subspace_keys(path)
    assert source == "states"
    fresh = attach(
        _One(),
        KiLoRAConfig(
            target_modules=["proj"], num_experts=3, routes=1, router_dim=8, m_in=2, m_out=3
        ),
        keys=loaded,
        key_source=source,
    )
    s = fresh.to_sets()[0]
    assert s.key_source == "states"
    ex = s.experts["proj"]
    assert ex.keys_in.shape[-1] == ex.down.shape[-1] == 16
    assert ex.keys_out.shape[-1] == ex.up.shape[-1] == 16
    assert not ex.keys_out.count_nonzero()
    assert torch.equal(loaded["proj"]["keys_in"], keys["proj"]["keys_in"])
    fresh.remove()


@pytest.mark.parametrize("layers", [1, 4])
def test_key_probes_keep_only_small_detached_tensors(layers):
    from kilora.compute import key_probes
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser(channels=4, width=16, num_layers=layers, seed=0)
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.feed_forward.w2"],
            num_experts=3,
            routes=1,
            router_dim=8,
            depth=3,
            train_depth=None,
        ),
    )
    for p in adapter.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(0))
    with key_probes() as record:
        model(x).pow(2).sum().backward()
    assert len(record) == layers
    for entries in record.values():
        assert [e["k"] for e in entries] == [1, 2, 3, "output"]
        kept = [
            t
            for e in entries
            for v in e.values()
            if isinstance(v, list)
            for t in v
            if t is not None
        ]
        assert kept and all(not t.requires_grad and t.shape[-1] <= 16 for t in kept)


@pytest.mark.parametrize("m_out", [1, 3])
def test_imported_probe_completes_output_keys_to_requested_size(m_out):
    from kilora.compute import key_probes
    from kilora.subspace_keys import keys_from_records
    from kilora.testing import TinyDenoiser

    model = TinyDenoiser(channels=4, width=16, num_layers=1, seed=0)
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=3,
            routes=1,
            router_dim=8,
            m_in=2,
            m_out=m_out,
            depth=3,
        ),
    )
    adapter._set_info[0]["form"] = "input"
    target = adapter.targets[0]
    site = adapter.sites[0]
    t = site.tensors
    current = {target: (t["s0_keys_in"], t["s0_keys_out"], *site.rows[0])}
    with key_probes() as record, torch.no_grad():
        model(torch.randn(2, 6, 4))
    keys = keys_from_records(record, {target: 0}, current, m_in=2, m_out=m_out)[target]
    assert keys["keys_out"].shape == (3, m_out, 8)
    assert torch.all(keys["rows_out"] == m_out)
    torch.testing.assert_close(
        keys["keys_out"] @ keys["keys_out"].transpose(-1, -2),
        torch.eye(m_out).expand(3, m_out, m_out),
        atol=1e-5,
        rtol=0,
    )
