# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_input_depth import _input_sets
from test_lowrank_depth import _sets

from kilora import compute
from kilora.adapter import _layer_token_depth
from kilora.compute import dropout_key, run_layer

TOKENS = 8


def _xy(seed=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, TOKENS, 16, generator=gen), torch.randn(2, TOKENS, 16, generator=gen)


def _kinds():
    return {"state": _sets(2, 16, 4, 2, 8, seed=5), "input": _input_sets(n=2, seed=6)}


def _run(sets, depth, token_depth=None, fast=False, monkeypatch=None, seed=3):
    x, y0 = _xy(seed)
    if monkeypatch is not None:
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    with torch.no_grad(), dropout_key(0, torch.tensor([0.5])):
        return run_layer(x, y0, sets, depth, target="layer", token_depth=token_depth)


def _map(values):
    return torch.tensor(values, dtype=torch.float32)[None].expand(2, TOKENS).clone()


@pytest.mark.parametrize("kind", ["state", "input"])
@pytest.mark.parametrize("depth", [1, 3, 2.5])
def test_a_constant_map_is_the_plain_depth_bit_for_bit(kind, depth):
    sets = _kinds()[kind]
    plain = _run(sets, depth)
    mapped = _run(sets, 1, token_depth=torch.full((2, TOKENS), float(depth)))
    assert torch.equal(mapped, plain)


@pytest.mark.parametrize("kind", ["state", "input"])
def test_tokens_follow_the_run_of_their_own_depth(kind):
    sets = _kinds()[kind]
    mixed = _run(sets, 1, token_depth=_map([1, 1, 1, 3, 3, 3, 2, 2]))
    torch.testing.assert_close(mixed[:, :3], _run(sets, 1)[:, :3], rtol=1e-5, atol=1e-6)
    split_map = _map([1.5, 1.5, 2, 2, 1.5, 2, 1.5, 2])
    got = _run(sets, 1, token_depth=split_map)
    for value in (1.5, 2):
        sel = split_map[0] == value
        want = _run(sets, value)
        torch.testing.assert_close(got[:, sel], want[:, sel], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["state", "input"])
def test_tokens_at_depth_zero_keep_the_base_output(kind):
    sets = _kinds()[kind]
    x, y0 = _xy()
    d = _map([0, 0, 2, 2, 3, 3, 1, 1])
    with torch.no_grad(), dropout_key(0, torch.tensor([0.5])):
        out = run_layer(x, y0, sets, 1, target="layer", token_depth=d)
    assert torch.equal(out[:, :2], y0[:, :2])
    assert not torch.allclose(out[:, 2:], y0[:, 2:])


@pytest.mark.parametrize("kind", ["state", "input"])
def test_fast_path_and_loop_agree(kind, monkeypatch):
    sets = _kinds()[kind]
    d = _map([1, 2, 3, 2.5, 1.5, 3, 1, 2])
    a = _run(sets, 1, token_depth=d, fast=False, monkeypatch=monkeypatch)
    b = _run(sets, 1, token_depth=d, fast=True, monkeypatch=monkeypatch)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_mixed_layer_of_both_forms_runs_per_token_depth():
    sets = [_kinds()["state"][0], _kinds()["input"][0]]
    d = _map([1, 1, 1, 1, 3, 3, 3, 3])
    out = _run(sets, 1, token_depth=d)
    assert torch.isfinite(out).all()
    assert torch.equal(_run(sets, 3), _run(sets, 1, token_depth=torch.full((2, TOKENS), 3.0)))


def test_path_iteration_with_a_map_is_refused():
    sets = [dataclasses.replace(s, iteration="path") for s in _kinds()["state"]]
    with pytest.raises(ValueError, match="accumulate"):
        _run(sets, 1, token_depth=_map([1, 2, 3, 1, 2, 3, 1, 2]))


def test_compose_maps_an_image_map_to_each_layer():
    image = torch.zeros(1, 32, 32)
    image[:, :, 16:] = 3.0
    image[:, :, :16] = 1.0
    h = torch.zeros(2, 16, 4)
    tok = _layer_token_depth(image, h)
    assert tok.shape == (1, 16)
    grid = tok.reshape(4, 4)
    assert torch.equal(grid[:, :2], torch.ones(4, 2)) and torch.equal(
        grid[:, 2:], torch.full((4, 2), 3.0)
    )
    frames = torch.zeros(2, 3, 4, 4, 8)
    assert _layer_token_depth(image, frames).shape == (1, 48)
    assert _layer_token_depth(image, torch.zeros(2, 7, 4)) is None


def test_compose_with_a_token_depth(monkeypatch):
    from test_import_modes import DIM, Toy, three

    from kilora import KiLoRAConfig, attach

    m = Toy().eval()
    x = torch.randn(1, 4, DIM, generator=torch.Generator().manual_seed(1))
    adapter = attach(
        m,
        KiLoRAConfig(
            target_modules=["blocks.0.mlp.layer1"],
            num_experts=3,
            router_dim=8,
            depth=3,
            train_depth=None,
        ),
    )
    with torch.no_grad():
        for site in adapter.sites:
            up = site.tensors["s0_up"]
            up.copy_(torch.randn(up.shape, generator=torch.Generator().manual_seed(2)))
    with torch.no_grad():
        plain = m(x)
        adapter.set_token_depth(torch.full((1, 4), 3.0))
        assert torch.equal(m(x), plain)
        adapter.set_token_depth(torch.tensor([[1.0, 3.0, 3.0, 1.0]]))
        mixed = m(x)
    assert not torch.equal(mixed, plain)


def test_token_depth_start_runs_the_base_depth_before_it():
    from test_import_modes import DIM, Toy

    from kilora import KiLoRAConfig, attach
    from kilora.compute import sampling_progress

    m = Toy().eval()
    x = torch.randn(1, 4, DIM, generator=torch.Generator().manual_seed(1))
    adapter = attach(
        m,
        KiLoRAConfig(
            target_modules=["blocks.0.mlp.layer1"],
            num_experts=3,
            router_dim=8,
            depth=3,
            train_depth=None,
        ),
    )
    with torch.no_grad():
        for site in adapter.sites:
            up = site.tensors["s0_up"]
            up.copy_(torch.randn(up.shape, generator=torch.Generator().manual_seed(2)))
    split_map = torch.tensor([[1.0, 3.0, 3.0, 1.0]])
    with torch.no_grad():
        adapter.set_token_depth(torch.full((1, 4), 2.0))
        uniform2 = m(x)
        adapter.set_token_depth(split_map)
        mapped = m(x)
        adapter.set_token_depth(split_map, start=0.3)
        assert adapter.token_depth_base == 2.0
        with sampling_progress(0.1):
            assert torch.equal(m(x), uniform2)
        with sampling_progress(0.3):
            assert torch.equal(m(x), mapped)
        adapter.set_token_depth(split_map, start=0.5, base=1)
        adapter_one = m(x)
        adapter.set_token_depth(torch.full((1, 4), 1.0))
        assert torch.equal(adapter_one, m(x))
    with pytest.raises(ValueError, match="sampling progress"):
        adapter.set_token_depth(split_map, start=1.5)
