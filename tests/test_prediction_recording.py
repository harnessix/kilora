# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_lowrank_depth import _sets

from kilora.compute import run_layer, sampling_progress
from kilora.recording import PathRecorder


@pytest.mark.parametrize("depth", [1, 3])
def test_single_expert_prediction_record(depth):
    layer = _sets(1, 8, 1, 3, 4, seed=1)[0]
    base = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(2))
    v0 = torch.eye(3)
    recorder = PathRecorder({"mount": layer.down}, {"mount": v0})
    with torch.no_grad(), sampling_progress(0.75):
        expected = run_layer(base, base, [layer], depth, target="mount")
        second = run_layer(base, base, [layer], 2, target="mount")
        with recorder.recording():
            actual = run_layer(base, base, [layer], depth, target="mount")
    assert torch.equal(actual, expected)
    errors = recorder.finish().prediction_errors.get("mount", [])
    if depth == 1:
        assert not errors
        return
    assert len(errors) == 2
    projection = (v0.double() @ layer.down.double()).t()
    expected_read = base.double().reshape(-1, 8) @ projection
    expected_second = second.double().reshape(-1, 8) @ projection
    expected_deep = expected.double().reshape(-1, 8) @ projection
    assert [entry.node for entry in errors] == [0, 1]
    assert not torch.equal(errors[0].short, errors[1].short)
    assert torch.equal(errors[0].deep, expected_second)
    assert torch.equal(errors[1].deep, expected_deep)
    for entry in errors:
        assert torch.equal(entry.read, expected_read)
        assert torch.equal(entry.error, entry.deep - entry.short)
        assert 0 < entry.precision <= 1


def test_unsupported_multi_expert_is_not_recorded():
    layer = _sets(1, 8, 2, 3, 4, seed=1)[0]
    base = torch.randn(1, 5, 8)
    recorder = PathRecorder({"mount": layer.down}, {"mount": torch.eye(6)})
    with torch.no_grad(), sampling_progress(0.75), recorder.recording():
        run_layer(base, base, [layer], 3, target="mount")
    assert not recorder.finish().prediction_errors


def test_precision_uses_fraction_of_observed_steps_run_deep():
    layer = _sets(1, 8, 1, 3, 4, seed=1)[0]
    base = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(2))
    recorder = PathRecorder({"mount": layer.down}, {"mount": torch.eye(3)})
    with torch.no_grad(), recorder.recording():
        with sampling_progress(0.0):
            run_layer(base, base, [layer], 1, target="mount")
        with sampling_progress(1.0):
            run_layer(base, base, [layer], 3, target="mount")
    entries = recorder.finish().prediction_errors["mount"]
    assert all(0 < entry.precision <= 0.5 for entry in entries)


def test_adjacent_errors_for_each_mount_and_sampler_step():
    layer = _sets(1, 8, 1, 3, 4, seed=1)[0]
    base = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(2))
    mounts = ("first", "second")
    recorder = PathRecorder(
        {name: layer.down for name in mounts}, {name: torch.eye(3) for name in mounts}
    )
    with torch.no_grad(), recorder.recording():
        for progress in (0.25, 0.75):
            with sampling_progress(progress):
                for name in mounts:
                    run_layer(base, base, [layer], 3, target=name)
    errors = recorder.finish().prediction_errors
    for name in mounts:
        for progress in (0.25, 0.75):
            entries = [entry for entry in errors[name] if entry.progress == progress]
            assert [entry.node for entry in entries] == [0, 1]
            assert not torch.equal(entries[0].short, entries[1].short)
    assert len({(name, item.node) for name, items in errors.items() for item in items}) == 4
