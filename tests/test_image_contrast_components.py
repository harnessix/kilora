# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file
from test_directions_contrast import Model, _samples, _set_change, _settings, _solve

from kilora.library import contrast


def _totals(per_image):
    records = list(per_image.values())
    return [
        sum(record[0] for record in records),
        *(torch.stack([record[index] for record in records]).sum(0) for index in (1, 2, 3)),
    ]


def _rank(matrix):
    values = torch.linalg.svdvals(matrix.double())
    return int((values > 1e-3 * values[0]).sum())


@pytest.mark.parametrize("images", [2, 3, 6])
@pytest.mark.parametrize("components", [0, 1, 2, 4])
def test_independent_image_contrasts_give_mean_plus_components(images, components):
    eye = torch.eye(images, dtype=torch.float64)
    per_image = {i: [1, eye[i], torch.zeros(images).double(), eye[i]] for i in range(images)}
    cross = contrast._mean_cross(_totals(per_image), per_image, components)
    assert _rank(cross) == min(components, images - 1) + 1
    if components >= images - 1:
        torch.testing.assert_close(cross, eye)


@pytest.mark.parametrize("components", [0, 1, 4])
def test_identical_image_contrasts_keep_rank_one(components):
    direction = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    per_image = {
        i: [1, direction, torch.zeros_like(direction), torch.eye(3)[i].double()] for i in range(3)
    }
    cross = contrast._mean_cross(_totals(per_image), per_image, components)
    assert _rank(cross) == 1
    torch.testing.assert_close(cross, torch.outer(direction, torch.ones(3).double()))


def test_zero_components_is_bit_identical_to_previous_cross():
    generator = torch.Generator().manual_seed(42)
    per_image = {
        i: [17, *(torch.randn(7, generator=generator) for _ in range(3))] for i in range(5)
    }
    count, target, student, inputs = totals = _totals(per_image)
    previous = torch.outer(contrast.mean_contrast(target, student, count), inputs)
    assert torch.equal(contrast._mean_cross(totals, per_image, 0), previous)


def test_sums_pool_times_and_seeds_within_each_image(monkeypatch):
    generator = torch.Generator().manual_seed(13)
    captured, identities, expected = [], [], {}
    for image in (0, 1, 0, 1):
        rows = 3 + len(captured)
        x, student, target = [torch.randn(1, rows, 4, generator=generator) for _ in range(3)]
        captured.append((target, (x, student)))
        identities.append(image)
        record = [rows, target.sum((0, 1)), student.sum((0, 1)), x.sum((0, 1))]
        if image not in expected:
            expected[image] = record
        else:
            expected[image][0] += rows
            for total, value in zip(expected[image][1:], record[1:], strict=True):
                total.add_(value)
    seen = []
    real = contrast._mean_cross

    def record(totals, per_image, components):
        seen.append((totals, per_image))
        return real(totals, per_image, components)

    monkeypatch.setattr(contrast, "_mean_cross", record)

    def observe(target, student):
        return {"mount": (None, target)}, {"mount": student}

    stats = contrast._group_statistics(
        ["mount"], captured, observe, lambda x: x, "mean", 1, identities
    )
    totals, per_image = seen[0]
    for image, expected_record in expected.items():
        assert per_image[image][0] == expected_record[0]
        for actual, value in zip(per_image[image][1:], expected_record[1:], strict=True):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
    for actual, value in zip(totals, _totals(per_image), strict=True):
        if isinstance(actual, int):
            assert actual == value
        else:
            torch.testing.assert_close(actual, value)
    inputs = torch.cat([student[0].reshape(-1, 4) for _, student in captured])
    torch.testing.assert_close(stats["mount"][0], inputs.T @ inputs)


def test_zero_components_writes_identical_weights_to_previous_statistics(tmp_path, monkeypatch):
    model = Model()
    _set_change(model)
    samples = _samples()
    settings = replace(_settings(), contrast_mode="mean", contrast_components=0)
    current = _solve(model, samples, settings=settings)
    real = contrast._mean_cross
    monkeypatch.setattr(
        contrast,
        "_mean_cross",
        lambda totals, *_: torch.outer(
            contrast.mean_contrast(totals[1], totals[2], totals[0]), totals[3]
        ),
    )
    previous = _solve(model, samples, settings=settings)
    monkeypatch.setattr(contrast, "_mean_cross", real)
    paths = [tmp_path / "current.safetensors", tmp_path / "previous.safetensors"]
    for path, result in zip(paths, [current, previous], strict=True):
        contrast.write_lora(path, result)
    current_weights, previous_weights = map(load_file, paths)
    assert current_weights.keys() == previous_weights.keys()
    assert all(torch.equal(value, previous_weights[key]) for key, value in current_weights.items())
    assert current.metadata == previous.metadata


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_components_group_by_image_across_replays_and_blocks(batch, monkeypatch):
    model = Model()
    _set_change(model)
    samples = _samples(count=6)
    for index, sample in enumerate(samples):
        sample.image_index = index % 3
    settings = replace(_settings(), contrast_mode="mean", contrast_components=2, replay_batch=batch)
    seen = []
    real = contrast._group_statistics

    def record(names, captured, observe, rows, mode, components=0, image_indices=None):
        assert set(image_indices) == {0, 1, 2}
        seen.append(tuple(image_indices))
        return real(names, captured, observe, rows, mode, components, image_indices)

    monkeypatch.setattr(contrast, "_group_statistics", record)
    result = _solve(model, samples, settings=settings)
    assert seen and all(ids == seen[0] for ids in seen)
    assert len(seen[0]) == (6 if batch == 1 else 3)
    for name, solution in result.mounts.items():
        assert result.metadata["fit"][name]["measured_rank"] == contrast.measured_rank(solution)
        assert solution.down.shape[0] == settings.rank


@pytest.mark.parametrize("value", [-1, 5, 1.5, True, None])
def test_components_setting_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="contrast_components"):
        replace(_settings(), contrast_components=value).validate()


def test_components_require_image_identities():
    with pytest.raises(ValueError, match="per-image identities"):
        _solve(Model(), settings=replace(_settings(), contrast_mode="mean", contrast_components=1))


def test_measured_rank_uses_relative_singular_threshold():
    solution = contrast.MountSolution(
        torch.diag(torch.tensor([1.0, 0.002, 0.0005, 0.0])), torch.eye(4), 0.1, 1.0
    )
    assert contrast.measured_rank(solution) == 2
    solution.down.zero_()
    assert contrast.measured_rank(solution) == 0
