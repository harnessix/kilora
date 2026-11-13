# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import torch
from test_axis_alphas import _set

from kilora import route_freeze
from kilora.call_values import axis_alphas
from kilora.compute import axis_budget, run_layer, sampling_progress


def _run(y, sets):
    return run_layer(y, y, sets, 1, target="layer")


def _y():
    return torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(7))


def test_one_axis_is_unchanged():
    y = _y()
    a = _set("creative")
    alone = _run(y, [a])
    with axis_alphas({"creative": 1.0}):
        assert torch.equal(_run(y, [a]), alone)
    with axis_alphas({"creative": 0.4}):
        torch.testing.assert_close(_run(y, [a]), y + 0.4 * (alone - y))


def test_two_axes_without_alphas_keep_full_changes():
    y = _y()
    a, b = _set("creative"), _set("ip")
    one, two = _run(y, [a]), _run(y, [b])
    torch.testing.assert_close(_run(y, [a, b]), y + (one - y) + (two - y))


def test_functional_axis_stays_outside_the_budget():
    y = _y()
    a, f = _set("creative"), dataclasses.replace(_set("ip"), axis="functional")
    one, two = _run(y, [a]), _run(y, [f])
    torch.testing.assert_close(_run(y, [a, f]), y + (one - y) + (two - y))


def test_hard_mask_gives_each_axis_its_region():
    y = _y()
    a, b = _set("creative"), _set("ip")
    one, two = _run(y, [a]), _run(y, [b])
    m = (torch.arange(6) < 3).float()[None].expand(2, 6)
    with axis_alphas({"ip": m, "creative": 1 - m}):
        got = _run(y, [a, b])
    inside, outside = m.bool()[..., None].expand_as(y), ~m.bool()[..., None].expand_as(y)
    assert torch.allclose(got[inside], two[inside]) and torch.allclose(got[outside], one[outside])


def test_axis_alphas_above_one_keep_their_values():
    soft = torch.tensor([[0.2, 0.9, 1.0]])
    w = axis_budget({"ip": soft, "creative": torch.full_like(soft, 0.8), "functional": 1.0})
    total = w["ip"] + w["creative"]
    torch.testing.assert_close(total, soft + 0.8)
    assert total.max() > 1
    assert torch.equal(w["ip"], soft)
    assert torch.equal(w["creative"], torch.full_like(soft, 0.8))
    torch.testing.assert_close(total[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(w["ip"] / w["creative"], soft / 0.8)
    assert w["functional"] == 1.0
    assert axis_budget({"creative": 1.0}) == {"creative": 1.0}


def test_route_freezing_applies_the_alpha_once():
    y = _y()
    a = dataclasses.replace(_set("creative"), route_freeze=0.0)
    cache: dict = {}
    with torch.no_grad(), axis_alphas({"creative": 0.5}):
        plain = _run(y, [a])
        with route_freeze.freezing(cache), sampling_progress(0.9):
            first = _run(y, [a])
            again = _run(y, [a])
    assert cache
    torch.testing.assert_close(first, plain)
    torch.testing.assert_close(again, plain, rtol=1e-4, atol=1e-5)


def test_ip_map_leaves_creative_strength_unchanged():
    y = _y()
    a, b = _set("creative"), _set("ip")
    creative, ip = _run(y, [a]) - y, _run(y, [b]) - y
    mask = torch.linspace(0, 1, 6).expand(2, 6)
    with axis_alphas({"creative": 1.0, "ip": mask}):
        actual = _run(y, [a, b])
    torch.testing.assert_close(actual, y + creative + mask[..., None] * ip)
