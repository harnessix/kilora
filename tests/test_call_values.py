# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_depth_map import _apply, _build, _run, chain
from test_input_depth import _input_sets
from test_lowrank_depth import _sets

from kilora.call_values import expert_scale_vector, expert_scales, token_depth_values
from kilora.compute import _gates, dropout_key, recording_routes, run_layer, sampling_progress
from kilora.route_freeze import freezing


def _xy(seed=4):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 7, 16, generator=gen), torch.randn(2, 7, 16, generator=gen)


def _layer(sets, depth, scales=None, x=None, y0=None):
    if x is None:
        x, y0 = _xy()
    with torch.no_grad(), dropout_key(0, torch.tensor([0.4])):
        return run_layer(x, y0, sets, depth, target="layer", expert_scales=scales)


def test_ones_change_nothing_and_scales_multiply_the_expert_weights():
    sets = _input_sets(n=1)
    x, y0 = _xy()
    assert torch.equal(_layer(sets, 3, [[1, 1, 1, 1]]), _layer(sets, 3))
    v = torch.tensor([1.0, 0.0, 2.0, 0.5])
    (s,) = sets
    with torch.no_grad():
        (gate,) = _gates(y0, sets, False, 0.0)
        weight = (gate * v).repeat_interleave(s.per_expert, dim=-1)
        want = y0 + ((torch.nn.functional.linear(x, s.down) * weight) @ s.up) * s.scale
    torch.testing.assert_close(_layer(sets, 1, [v]), want, rtol=1e-5, atol=1e-6)
    assert torch.equal(_layer(sets, 2, [[0, 0, 0, 0]]), y0)


def test_the_routing_stays_as_it_is():
    sets = _input_sets(n=1)
    with recording_routes() as plain:
        _layer(sets, 1)
    with recording_routes() as scaled:
        _layer(sets, 1, [[3.0, 0.0, 1.0, 0.5]])
    assert plain and plain.keys() == scaled.keys()
    for key in plain:
        for a, b in zip(plain[key], scaled[key], strict=True):
            assert torch.equal(a, b)


def test_bad_scales_are_refused():
    with pytest.raises(ValueError, match="imported"):
        _layer(_sets(1, 16, 4, 2, 8, seed=5), 1, [[1, 1, 1, 0]])
    with pytest.raises(ValueError, match="3 expert scales for 4 experts"):
        _layer(_input_sets(n=1), 1, [[1, 1, 0]])


def test_frozen_layers_apply_the_scales_of_each_call():
    sets = [dataclasses.replace(s, route_freeze=0.0) for s in _input_sets(n=1)]
    x, y0 = _xy()
    cache: dict = {}
    with freezing(cache), sampling_progress(0.5):
        built = _layer(sets, 3, x=x, y0=y0)
        assert cache
        frozen = _layer(sets, 3, x=x, y0=y0)
        assert torch.equal(_layer(sets, 3, [[1, 1, 1, 1]], x, y0), frozen)
        assert torch.equal(_layer(sets, 3, [[0, 0, 0, 0]], x, y0), y0)
        one_off = _layer(sets, 3, [[1, 0, 1, 1]], x, y0)
    assert not torch.equal(one_off, frozen) and not torch.equal(built, y0)


def test_scale_vector_by_entry_name():
    assert expert_scale_vector(["a", "b"]) is None
    with expert_scales({"b": 0.5}):
        assert expert_scale_vector(["a", "b"]) == [1.0, 0.5]
        assert expert_scale_vector(["c"]) is None
        with expert_scales({"a": 2}):
            assert expert_scale_vector(["a", "b"]) == [2.0, 1.0]


def test_apply_takes_the_expert_scales_of_the_call(denoiser, chain):
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    patched = _apply(denoiser, chain, 3)
    plain = _run(patched, 0.0, x)
    with expert_scales({"s0": 1.0, "s1": 1.0}):
        assert torch.equal(_run(patched, 0.0, x), plain)
    with expert_scales({"s0": 0.0, "s1": 0.0}):
        assert torch.equal(_run(patched, 0.0, x), FakePatcher(FakeBaseModel(denoiser)).run(x))
    with expert_scales({"s1": 0.0}):
        s1_off = _run(patched, 0.0, x)
    assert not torch.equal(s1_off, plain)


def test_the_call_sets_when_the_map_applies_and_its_base(denoiser, chain):
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    uniform = {d: _run(_apply(denoiser, chain, d), 0.0, x) for d in (2, 3)}
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    split, _ = _build(mask=mask, min_depth=1.0, max_depth=3.0)
    patched = _apply(denoiser, chain, 1, split)
    late = _run(patched, 0.5, x)
    with token_depth_values(on=False):
        assert torch.equal(_run(patched, 0.5, x), uniform[2])
    with token_depth_values(on=False, base=3):
        assert torch.equal(_run(patched, 0.5, x), uniform[3])
    with token_depth_values(on=True):
        assert torch.equal(_run(patched, 0.1, x), late)


def test_the_adapter_reads_the_call_values_too():
    from test_import_modes import DIM, Toy

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
        adapter.set_token_depth(torch.full((1, 4), 1.0))
        one = m(x)
        split_map = torch.tensor([[1.0, 3.0, 3.0, 1.0]])
        adapter.set_token_depth(split_map, start=0.0)
        mapped = m(x)
        with token_depth_values(on=False, base=1):
            assert torch.equal(m(x), one)
        adapter.set_token_depth(split_map, start=0.9)
        with token_depth_values(on=True):
            assert torch.equal(m(x), mapped)
