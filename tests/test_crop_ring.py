# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import types

import pytest
import torch

from kilora.comfyui import layer_crop
from kilora.comfyui.layer_crop import (
    RING_MAX,
    RING_MIN,
    base_extent,
    blend_weight,
    crop_box,
    install_layer_crop,
    ring_cost,
    ring_fraction,
    ring_shift,
    ring_width,
    sampling_progress,
)
from kilora.comfyui.nodes import KiLoRALayerConditioning

SIZE = (128, 128)


@pytest.fixture(autouse=True)
def comfy_conds(monkeypatch):
    import sys

    class Cond:
        def __init__(self, cond):
            self.cond = cond

    comfy = types.ModuleType("comfy")
    module = types.ModuleType("comfy.conds")
    module.CONDRegular = Cond
    comfy.conds = module
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", module)
    monkeypatch.delenv("KILORA_CROP_RING_SHIFT", raising=False)


def covers(area, bounds, extra, size=SIZE, patch=2):
    extra_y, extra_x = extra if isinstance(extra, tuple) else (extra, extra)
    h, w, top, left = area
    b_top, b_left, b_bottom, b_right = bounds
    assert top >= 0 and left >= 0 and top + h <= size[0] and left + w <= size[1]
    assert (h % 16 == 0 or h == size[0]) and (w % 16 == 0 or w == size[1])
    assert top % patch == 0 and left % patch == 0
    assert top <= max(0, b_top - extra_y) and left <= max(0, b_left - extra_x)
    assert top + h >= min(size[0], b_bottom + extra_y)
    assert left + w >= min(size[1], b_right + extra_x)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((56, 56, 72, 72), (48, 48, 40, 40)),
        ((0, 56, 16, 72), (32, 48, 0, 40)),
        ((112, 56, 128, 72), (32, 48, 96, 40)),
        ((56, 0, 72, 16), (48, 32, 40, 0)),
        ((56, 112, 72, 128), (48, 32, 40, 96)),
        ((0, 0, 16, 16), (32, 32, 0, 0)),
        ((112, 112, 128, 128), (32, 32, 96, 96)),
    ],
)
def test_window_is_box_plus_ring_clamped_on_grid(bounds, expected):
    area = crop_box(bounds, SIZE, ring=8)
    assert area == expected
    covers(area, bounds, 16)


@pytest.mark.parametrize("patch", [1, 2, 4])
@pytest.mark.parametrize("ring", [0, 3, 8, 21])
def test_window_covers_ring_for_other_boxes(patch, ring):
    for bounds in ((20, 50, 100, 100), (0, 90, 30, 128), (100, 0, 128, 40), (5, 7, 9, 11)):
        area = crop_box(bounds, SIZE, patch=patch, ring=ring)
        h, w = bounds[2] - bounds[0], bounds[3] - bounds[1]
        extra = (max(8, math.ceil(h * 0.125)) + ring, max(8, math.ceil(w * 0.125)) + ring)
        covers(area, bounds, extra, patch=patch)


def test_ring_zero_is_the_t018d_window_and_weight():
    mask = torch.zeros(1, 128, 128)
    mask[..., 40:60, 70:90] = torch.linspace(0.1, 1, 20)
    for bounds in ((40, 70, 60, 90), (0, 0, 8, 8), (120, 100, 128, 128), (0, 0, 120, 120)):
        for patch in (1, 2, 4):
            t018d = crop_box(bounds, SIZE, patch=patch)
            assert crop_box(bounds, SIZE, patch=patch, ring=0, shift=(0, 0)) == t018d
            assert torch.equal(blend_weight(mask, t018d), mask[(..., *_slices(t018d))])
    with pytest.raises(ValueError):
        crop_box((40, 40, 48, 48), SIZE, ring=2, shift=(3, 0))
    with pytest.raises(ValueError):
        crop_box((40, 40, 48, 48), SIZE, ring=-1)


def _slices(area):
    h, w, top, left = area
    return slice(top, top + h), slice(left, left + w)


def test_blend_weight_is_zero_in_the_ring_and_keeps_the_feather_ramp():
    mask = torch.zeros(1, 128, 128)
    ramp = torch.linspace(0.0625, 1, 8)
    box = torch.ones(16, 16)
    box[:8] *= ramp[:, None]
    mask[0, 56:72, 56:72] = box
    area = crop_box((56, 56, 72, 72), SIZE, ring=8)
    weight = blend_weight(mask, area, strength=0.8)
    assert weight.shape[-2:] == area[:2]
    inner = torch.zeros_like(weight, dtype=torch.bool)
    inner[..., 16:32, 16:32] = True
    assert torch.equal(weight[~inner], torch.zeros_like(weight[~inner]))
    assert torch.allclose(weight[..., 16:32, 16:32], 0.8 * box)
    column = weight[0, 16:24, 24]
    assert torch.all(column[1:] > column[:-1])
    assert torch.all((weight > 0) == (mask[..., 40:88, 40:88] > 0))


def test_schedule_endpoints_and_monotone():
    assert ring_fraction(0.0) == RING_MAX == 0.5
    assert ring_fraction(1.0) == RING_MIN == 0.0
    values = [ring_fraction(i / 100) for i in range(101)]
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))
    bounds = (56, 56, 72, 72)
    assert ring_width(bounds, 0.0) == 8
    assert ring_width(bounds, 1.0) == 0
    widths = [ring_width(bounds, i / 50) for i in range(51)]
    assert all(b <= a for a, b in zip(widths, widths[1:], strict=False))


def test_sampling_progress_follows_sigma():
    sigmas = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    assert sampling_progress(4.0, sigmas) == 0.0
    assert sampling_progress(1.0, sigmas) == 1.0
    assert sampling_progress(2.5, sigmas) == 0.5
    assert sampling_progress(8.0, None, first=8.0) == 0.0
    assert sampling_progress(2.0, None, first=8.0) == 0.75
    assert sampling_progress(1.0, None, first=None) == 0.0


def test_cost_ratio_expression():
    assert ring_cost(32, 32, 8) == (48 * 48) / (32 * 32) == 2.25
    assert ring_cost(20, 40, 5) == (30 * 50) / (20 * 40)
    assert ring_cost(16, 24, 0) == 1.0
    assert base_extent((56, 56, 72, 72)) == (32, 32)
    assert base_extent((0, 0, 100, 40)) == (126, 56)


def test_shift_is_seeded_and_inside_the_ring():
    assert ring_shift(0, 3, 0, (0, 0)) == (0, 0)
    first = [ring_shift(8, 3, step, (0, 0)) for step in range(20)]
    assert first == [ring_shift(8, 3, step, (0, 0)) for step in range(20)]
    assert all(-8 <= dy <= 8 and -8 <= dx <= 8 for dy, dx in first)
    assert len(set(first)) > 1


def _run(crop, sigmas, shift_seed=None, bounds=(56, 56, 72, 72)):
    seen = []

    def calc(model, groups, x, sigma, options):
        cond = groups[0][0]
        window = cond.get("model_conds", {}).get("kilora_crop_window")
        seen.append((cond.get("area"), None if window is None else window.cond))
        return [torch.zeros_like(x)]

    patcher = types.SimpleNamespace(model_options={})
    install_layer_crop(patcher, 2, calc=calc, shift_seed=shift_seed)
    sampler = patcher.model_options["sampler_calc_cond_batch_function"]
    mask = torch.zeros(1, 128, 128)
    top, left, bottom, right = bounds
    mask[..., top:bottom, left:right] = 1
    cond = {"kilora_crop": crop, "kilora_alpha_source": "mask", "mask": mask}
    schedule = torch.tensor(sigmas)
    options = {"transformer_options": {"sample_sigmas": schedule}}
    for sigma in sigmas:
        if sigma <= 0:
            continue
        sampler(
            {
                "input": torch.zeros(1, 1, 128, 128),
                "sigma": torch.tensor([sigma]),
                "conds": [[cond]],
                "model": None,
                "model_options": options,
            }
        )
    return seen, sampler._kilora_ring_stats


SIGMAS = [4.0, 3.0, 2.0, 1.0, 0.0]


def test_sampler_ring_schedule_cost_and_t018d_identity():
    ring, stats = _run("auto", SIGMAS)
    box = (56, 56, 72, 72)
    assert ring[0][0] == (48, 48, 40, 40)
    sizes = [area[0] * area[1] for area, _ in ring]
    assert all(b <= a for a, b in zip(sizes, sizes[1:], strict=False))
    assert ring[-1][0] == crop_box(box, SIZE, patch=2) == (32, 32, 48, 48)
    assert torch.equal(ring[-1][1], torch.tensor([[48, 48, 128, 128]]))
    mask = torch.zeros(1, 128, 128)
    mask[..., 56:72, 56:72] = 1
    assert torch.equal(
        blend_weight(mask, ring[-1][0]), blend_weight(mask, crop_box(box, SIZE, patch=2))
    )
    assert stats["costs"][0] == ring_cost(32, 32, 8) == 2.25
    assert stats["costs"][-1] == 1.0
    assert len(stats["costs"]) == 4
    assert stats["mean"] == pytest.approx(sum(stats["costs"]) / 4)
    assert stats["runs"] == [stats["mean"]]


def test_sampler_off_never_crops():
    off, stats = _run("off", SIGMAS)
    assert off == [(None, None)] * 4
    assert stats["costs"] == []


def test_sampler_keeps_the_seventy_percent_rule():
    ring, stats = _run("auto", SIGMAS, bounds=(40, 40, 88, 88))
    assert ring[0] == (None, None)
    assert ring[-1][0] == (64, 64, 32, 32)
    cropped = sum(area is not None for area, _ in ring)
    assert 0 < cropped < 4
    assert len(stats["costs"]) == cropped
    assert stats["costs"][-1] == 1.0


def test_sampler_shift_off_by_default_and_seeded(monkeypatch):
    off, _ = _run("auto", SIGMAS)
    box = (56, 56, 72, 72)
    plain = [crop_box(box, SIZE, ring=ring_width(box, p)) for p in (0.0, 1 / 3, 2 / 3, 1.0)]
    assert [area for area, _ in off] == plain
    first, _ = _run("auto", SIGMAS, shift_seed=11)
    second, _ = _run("auto", SIGMAS, shift_seed=11)
    assert [a for a, _ in first] == [a for a, _ in second]
    for area, _ in first:
        covers(area, (56, 56, 72, 72), 8)
    monkeypatch.setenv("KILORA_CROP_RING_SHIFT", "11")
    from_env, _ = _run("auto", SIGMAS)
    assert [a for a, _ in from_env] == [a for a, _ in first]
    monkeypatch.setenv("KILORA_CROP_RING_SHIFT", "x")
    assert layer_crop._shift_seed() is None


def test_node_defaults_to_off_and_auto_uses_the_ring():
    crop = KiLoRALayerConditioning.INPUT_TYPES()["optional"]["crop"]
    assert crop[0] == ["off", "auto"]
    assert crop[1]["default"] == "off"
    assert "context" in crop[1].get("tooltip", "context")
