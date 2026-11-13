# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora import route_freeze as R
from kilora.compute import dropout_key, run_layer, sampling_progress


def _y0(seed=1, width=16):
    return torch.randn(2, 9, width, generator=torch.Generator().manual_seed(seed))


def _call(sets, depth, progress, cache, x=None, y0=None, seed=0):
    y0 = _y0() if y0 is None else y0
    x = y0 if x is None else x
    contexts = (dropout_key(seed, progress), sampling_progress(progress), R.freezing(cache))
    with torch.no_grad(), contexts[0], contexts[1], contexts[2]:
        return run_layer(x.clone(), y0.clone(), sets, depth, target="layer")


def _freeze(sets, value=0.4, **kw):
    return [dataclasses.replace(s, route_freeze=value, **kw) for s in sets]


@pytest.mark.parametrize("gate", ["none", "adaptive"])
def test_before_the_freeze_point_nothing_changes(gate):
    plain = [dataclasses.replace(s, gate_mode=gate) for s in _sets(2, 16, 4, 2, 8, seed=2)]
    frozen = _freeze(plain)
    for p in (0.0, 0.2, 0.39):
        cache: dict = {}
        assert torch.equal(_call(frozen, 3, p, cache), _call(plain, 3, p, {}))
        assert not cache


@pytest.mark.parametrize("gate", ["none", "adaptive"])
@pytest.mark.parametrize("depth", [1, 3, 2.5])
def test_after_the_freeze_point_the_layer_is_the_frozen_product(gate, depth):
    sets = _freeze([dataclasses.replace(s, gate_mode=gate) for s in _sets(2, 16, 4, 2, 8, seed=3)])
    cache: dict = {}
    first = _call(sets, depth, 0.4, cache)
    assert torch.equal(
        first, _call([dataclasses.replace(s, route_freeze=None) for s in sets], depth, 0.4, {})
    )
    entry = cache["layer"]
    y0 = _y0(5)
    want = R.output(entry, y0, y0)
    got = _call(sets, depth, 0.7, cache, y0=y0, seed=9)
    torch.testing.assert_close(got, want)
    c = torch.cat([y0 @ s.down.t() for s in sets], -1)
    by_hand = y0 + torch.matmul(c * entry["scale"].unsqueeze(-2), entry["up"])
    torch.testing.assert_close(got, by_hand)


def test_one_expert_at_depth_one_without_a_gate_is_the_plain_lora():
    s = _sets(1, 16, 1, 3, 8, seed=4)[0]
    s = dataclasses.replace(s, route_freeze=0.0, gate_mode="none")
    cache: dict = {}
    _call([s], 1, 0.5, cache)
    y0 = _y0(6)
    got = _call([s], 1, 0.9, cache, y0=y0)
    torch.testing.assert_close(got, y0 + (y0 @ s.down.t()) @ s.up * s.scale)


def test_imported_set_freezes_and_reads_x(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    gen = torch.Generator().manual_seed(7)
    s = dataclasses.replace(
        _sets(1, 16, 4, 2, 8, seed=7)[0],
        form="input",
        down=torch.randn(8, 12, generator=gen),
        route_freeze=0.3,
        gate_mode="adaptive",
    )
    x = torch.randn(2, 9, 12, generator=gen)
    cache: dict = {}
    _call([s], 3, 0.3, cache, x=x)
    y0 = _y0(8)
    got = _call([s], 3, 0.8, cache, x=x, y0=y0)
    entry = cache["layer"]
    want = y0 + torch.matmul((x @ s.down.t()) * entry["scale"].unsqueeze(-2), entry["up"])
    torch.testing.assert_close(got, want)


def test_a_new_generation_or_shape_freezes_again():
    sets = _freeze(_sets(1, 16, 4, 2, 8, seed=9))
    cache: dict = {}
    _call(sets, 2, 0.5, cache)
    assert R.frozen("layer", (2, 9)) is None
    y = torch.randn(3, 9, 16)
    _call(sets, 2, 0.6, cache, y0=y)
    assert cache["layer"]["lead"] == (3, 9)


def test_no_cache_means_no_freezing():
    sets = _freeze(_sets(1, 16, 4, 2, 8, seed=10))
    with torch.no_grad(), dropout_key(0, 0.9), sampling_progress(0.9):
        a = run_layer(_y0(), _y0(), sets, 3, target="layer")
    plain = [dataclasses.replace(s, route_freeze=None) for s in sets]
    with torch.no_grad(), dropout_key(0, 0.9), sampling_progress(0.9):
        b = run_layer(_y0(), _y0(), plain, 3, target="layer")
    assert torch.equal(a, b)


def test_generation_helper_freezes_and_settings_round_trip(model, x, tmp_path):
    from kilora import KiLoRAConfig, attach, load_mod
    from kilora.modfile import save_mod

    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=2,
            train_depth=None,
        ),
    )
    adapter.settings.route_freeze = 0.25
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    path = tmp_path / "f.kmod.safetensors"
    save_mod(path, sets, settings, id="f")
    assert load_mod(path).settings.route_freeze == 0.25
    with pytest.raises(ValueError):
        R.check_route_freeze(1.5)
    assert C.LayerSet.__dataclass_fields__["route_freeze"].default is None


def test_frozen_layer_keeps_the_token_role_mask():
    gen = torch.Generator().manual_seed(11)
    s = dataclasses.replace(
        _sets(1, 16, 4, 2, 8, seed=11)[0],
        form="input",
        down=torch.randn(8, 16, generator=gen),
        route_freeze=0.3,
        gate_mode="adaptive",
    )
    mask = (torch.arange(9) < 5).float()[None].expand(2, 9)
    cache: dict = {}
    y0 = _y0(12)

    def call(progress, y):
        ctx = (dropout_key(0, progress), sampling_progress(progress), R.freezing(cache))
        with torch.no_grad(), ctx[0], ctx[1], ctx[2]:
            return run_layer(y.clone(), y.clone(), [s], 3, target="layer", regions=[mask])

    call(0.3, y0)
    y1 = _y0(13)
    got = call(0.9, y1)
    assert torch.equal(got[:, 5:], y1[:, 5:])
    assert (got[:, :5] - y1[:, :5]).abs().sum(-1).min() > 0


@pytest.mark.parametrize("width", [12, 16])
def test_import_chain_freezes_routes_and_replays_current_input(monkeypatch, width):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_FUSED", "0")
    gen = torch.Generator().manual_seed(71)
    s = dataclasses.replace(
        _sets(1, 16, 4, 2, 8, seed=71)[0],
        form="input",
        down=torch.randn(8, width, generator=gen),
        route_freeze=0.3,
        gate_mode="none",
    )
    x = torch.randn(2, 9, width, generator=gen)
    y0 = _y0(72)
    cache = {}
    first = _call([s], 3, 0.3, cache, x=x, y0=y0)
    entry = cache["layer"]
    assert len(entry["chain_routes"]) == 3 and len(entry["chain_keeps"]) == 2
    assert all(
        isinstance(m, tuple) and m[0].dtype == torch.uint8
        for step in entry["chain_keeps"]
        for m in step
    )

    def no_routing(*args, **kwargs):
        raise AssertionError("frozen chain routed again")

    monkeypatch.setattr(C, "_route", no_routing)
    again = _call([s], 3, 0.8, cache, x=x, y0=y0, seed=9)
    assert torch.equal(again, first)
    new_x = x * 0.7
    actual = _call([s], 3, 0.8, cache, x=new_x, y0=y0, seed=9)
    with torch.no_grad():
        expected = C._run_input(
            new_x,
            y0,
            [s],
            [0],
            3,
            [None],
            "layer",
            stats={
                "forced_routes_all": entry["chain_routes"],
                "forced_read_keeps_all": [
                    [R._unpack_keep(m) for m in step] for step in entry["chain_keeps"]
                ],
            },
        )
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, first)


@pytest.mark.parametrize("cols", [1, 7, 8, 13, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_packed_read_masks_restore_the_same_values(cols, dtype, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "sample")
    gen = torch.Generator().manual_seed(cols)
    m = (torch.rand(2, 5, cols, generator=gen) < 0.5).to(dtype)
    packed = R._pack_keep(m)
    assert isinstance(packed, tuple) and packed[0].dtype == torch.uint8
    assert packed[0].numel() == 2 * 5 * ((cols + 7) // 8)
    restored = R._unpack_keep(packed)
    assert restored.dtype == dtype and restored.shape == m.shape
    assert torch.equal(restored, m)
