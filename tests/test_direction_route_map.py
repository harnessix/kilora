# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.comfyui.route_map import RouteShares, render
from kilora.compute import run_layer
from kilora.library.basis import LibraryBasis, MountBasis
from kilora.library.nodes import direction_set


def test_direction_routes_list_sources_per_region_and_step():
    eye = torch.eye(2)
    sources = [[{"file": "/library/ink.safetensors"}], [{"file": "/library/parchment.safetensors"}]]
    basis = LibraryBasis(
        {"layer": MountBasis(eye, eye, torch.ones(2), torch.ones(2), sources, sources)},
        {},
        [],
        [],
        2,
        2,
        _parts={"fixture": {"layer": (eye, eye)}},
    )
    layer = direction_set(basis, "layer")
    routes = RouteShares()
    routes.observe("layer", [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])])
    routes.observe_directions("layer", 1, 0, layer, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    routes.observe_directions("layer", 2, 0, layer, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    _, report = render(routes, torch.zeros(1, 8, 16, 3))
    assert "region 1,1/1x2, step 1: direction 1 (ink.safetensors" in report
    assert "region 1,1/1x2, step 2: direction 2 (parchment.safetensors" in report


def test_direction_set_is_observed_during_layer_execution():
    eye = torch.eye(2)
    sources = [[{"file": "a"}], [{"file": "b"}]]
    basis = LibraryBasis(
        {"layer": MountBasis(eye, eye, torch.ones(2), torch.ones(2), sources, sources)},
        {},
        [],
        [],
        2,
        2,
        _parts={"fixture": {"layer": (eye, eye)}},
    )
    layer = direction_set(basis, "layer")
    routes = RouteShares()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    with routes.recording():
        run_layer(x, torch.zeros_like(x), [layer], 2, target="layer")
    assert set(routes.directions) == {("layer", 0, 1), ("layer", 0, 2)}
    routes.clear()
    assert not routes.directions


def test_baked_direction_routes_keep_sources():
    from kilora.bake import ScopedSet, plan_targets
    from kilora.library.nodes import direction_adapters

    eye = torch.eye(2)
    sources = [[{"file": "a"}], [{"file": "b"}]]
    basis = LibraryBasis(
        {"layer": MountBasis(eye, eye, torch.ones(2), torch.ones(2), sources, sources)},
        {},
        [],
        [],
        2,
        2,
        _parts={"fixture": {"layer": (eye, eye)}},
    )
    adapter = direction_adapters(basis, {"layer": (2, 2)})[0]
    layer = (
        plan_targets([ScopedSet(adapter, scope="independent")], 1)["layer"]
        .to(torch.device("cpu"), torch.float32)
        .sets[0]
    )
    assert layer.direction_sources == sources
    routes = RouteShares()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    with routes.recording():
        run_layer(x, torch.zeros_like(x), [layer], 2, target="layer")
    assert set(routes.directions) == {("layer", 0, 1), ("layer", 0, 2)}
