# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch
from torch.nn import functional as F

from kilora.library.contrast import MountSolution, Settings, fit_mount


def _reference(c, s, rank, ridge):
    c = c.double()
    c = (c + c.T) * 0.5
    s = s.to(device=c.device, dtype=torch.float64)
    ridge = max(abs(float(c.diagonal().mean())) * ridge, torch.finfo(c.dtype).eps)
    eye = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
    for attempt in range(7):
        lower, info = torch.linalg.cholesky_ex(c + ridge * eye)
        if int(info) == 0:
            break
        if attempt == 6:
            raise ValueError("input Gram matrix is not positive definite after ridge increases")
        ridge *= 10
    metric = torch.linalg.solve_triangular(lower, s.T, upper=False).T
    u, values, _ = torch.linalg.svd(metric, full_matrices=False)
    keep = min(rank, len(values))
    up = u[:, :keep]
    down = torch.linalg.solve_triangular(lower.T, (up.T @ metric).T, upper=True).T
    energy = float(values.square().sum())
    share = float(values[:keep].square().sum()) / energy if energy else 1.0
    down = F.pad(down, (0, 0, 0, rank - keep))
    up = F.pad(up.T, (0, 0, 0, rank - keep))
    if not torch.count_nonzero(s):
        down.zero_()
        up.zero_()
    return MountSolution(down.float().cpu(), up.float().cpu(), ridge, share)


def _compare(c, s, rank, ridge=0.1):
    actual = fit_mount(c, s, rank, ridge=ridge)
    expected = _reference(c, s, rank, ridge)
    error = torch.linalg.vector_norm(actual.delta() - expected.delta())
    scale = torch.linalg.vector_norm(expected.delta())
    assert float(error) <= 1e-6 * float(scale)
    assert actual.ridge == pytest.approx(expected.ridge, rel=1e-9, abs=1e-9)
    assert actual.fit_share == pytest.approx(expected.fit_share, rel=1e-9, abs=1e-9)
    assert actual.down.device.type == actual.up.device.type == "cpu"
    assert actual.down.dtype == actual.up.dtype == torch.float32
    return actual


@pytest.mark.parametrize("out_width,in_width", [(9, 17), (17, 9), (12, 12)])
def test_fit_mount_matches_full_svd(out_width, in_width):
    generator = torch.Generator().manual_seed(722)
    x = torch.randn(3 * in_width, in_width, generator=generator)
    c = x.T @ x + torch.eye(in_width)
    s = torch.randn(out_width, in_width, generator=generator)
    actual = _compare(c, s, 4)
    assert not actual.svd_fallback


@pytest.mark.parametrize("out_width,in_width", [(9, 17), (17, 9), (12, 12)])
@pytest.mark.parametrize("padding", [0, 2])
def test_fit_mount_full_rank_share_is_exactly_one(out_width, in_width, padding):
    generator = torch.Generator().manual_seed(722)
    x = torch.randn(3 * in_width, in_width, generator=generator)
    c = x.T @ x + torch.eye(in_width)
    s = torch.randn(out_width, in_width, generator=generator)
    actual = fit_mount(c, s, min(out_width, in_width) + padding)
    assert actual.fit_share == 1.0


@pytest.mark.parametrize("out_width,in_width", [(5, 8), (8, 5), (5, 5)])
def test_fit_mount_rank_deficient_and_padding(out_width, in_width):
    c = torch.eye(in_width)
    s = torch.zeros(out_width, in_width)
    s[0, 0], s[1, 1] = 3, 1
    actual = _compare(c, s, max(out_width, in_width) + 2)
    assert torch.isfinite(actual.down).all()
    assert torch.isfinite(actual.up).all()
    assert actual.down.shape == (max(out_width, in_width) + 2, in_width)
    assert actual.up.shape == (max(out_width, in_width) + 2, out_width)
    assert not actual.svd_fallback


@pytest.mark.parametrize("out_width,in_width", [(5, 8), (8, 5), (5, 5)])
def test_fit_mount_rank_deficient_cut_skips_svd(out_width, in_width):
    s = torch.zeros(out_width, in_width)
    s[0, 0], s[1, 1] = 3, 1
    actual = _compare(torch.eye(in_width), s, 3)
    assert not actual.svd_fallback


def test_fit_mount_zero():
    actual = _compare(torch.zeros(5, 5), torch.zeros(8, 5), 9)
    assert torch.count_nonzero(actual.down) == torch.count_nonzero(actual.up) == 0
    assert actual.fit_share == 1.0
    assert not actual.svd_fallback


@pytest.mark.parametrize("out_width,in_width", [(4, 6), (6, 4), (4, 4)])
def test_fit_mount_equal_values_at_cut_use_svd(out_width, in_width):
    s = torch.zeros(out_width, in_width)
    s[:4, :4] = torch.diag(torch.tensor([4.0, 2.0, 2.0, 1.0]))
    actual = _compare(torch.eye(in_width), s, 2)
    assert actual.svd_fallback
    print("SVD fallbacks: 1")


def test_fit_mount_large_cpu_timing():
    generator = torch.Generator().manual_seed(722)
    c = torch.diag(1 + torch.rand(8192, generator=generator))
    s = torch.randn(2048, 8192, generator=generator)
    started = time.perf_counter()
    actual = fit_mount(c, s, 16)
    elapsed = time.perf_counter() - started
    print(
        f"fit_mount 2048 x 8192 rank 16: {elapsed:.3f}s; SVD fallbacks: {int(actual.svd_fallback)}"
    )
    assert elapsed < 30
    assert actual.down.shape == (16, 8192)
    assert actual.up.shape == (16, 2048)
    assert torch.isfinite(actual.down).all()
    assert torch.isfinite(actual.up).all()


@pytest.mark.parametrize("fallback", [False, True])
def test_fit_mount_metadata_only_records_fallback(monkeypatch, fallback):
    from test_directions_contrast import Model, _samples, _settings

    from kilora.library import contrast

    original = contrast.fit_mount

    def fit(c, s, rank, **kwargs):
        solution = original(c, s, rank, **kwargs)
        assert not solution.svd_fallback
        solution.svd_fallback = fallback
        return solution

    monkeypatch.setattr(contrast, "fit_mount", fit)
    result = contrast.solve(
        Model(),
        _samples(),
        [("same", "same")],
        model_sha="toy",
        license="Apache-2.0",
        settings=_settings(),
    )
    records = result.metadata["fit"]
    assert records
    if fallback:
        assert result.metadata["svd_fallbacks"] == len(records)
        assert all(record["svd_fallback"] is True for record in records.values())
    else:
        assert "svd_fallbacks" not in result.metadata
        assert all(set(record) == {"ridge", "fit_share"} for record in records.values())


@pytest.mark.parametrize("out_width,in_width", [(5, 8), (8, 5)])
def test_fit_mount_linear_algebra_devices(monkeypatch, out_width, in_width):
    calls = []

    def record(name):
        original = getattr(torch.linalg, name)

        def wrapped(*args, **kwargs):
            calls.append((name, [(arg.device, arg.dtype) for arg in args if torch.is_tensor(arg)]))
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, wrapped)

    for name in ("cholesky_ex", "solve_triangular", "qr", "svd", "eigh"):
        record(name)
    c = torch.eye(in_width)
    s = torch.randn(out_width, in_width, generator=torch.Generator().manual_seed(723))
    fit_mount(c, s, 2)
    names = [name for name, _ in calls]
    assert names[0:2] == ["cholesky_ex", "solve_triangular"]
    assert names[-1] == "solve_triangular"
    assert names.count("qr") == 13
    assert names.count("svd") == 2
    assert "eigh" not in names
    for _name, inputs in calls:
        assert all(device == c.device and dtype == torch.float64 for device, dtype in inputs)


@pytest.mark.parametrize("copy_gram", [False, True])
def test_group_reuses_cholesky_and_matches_separate_fits(monkeypatch, copy_gram):
    generator = torch.Generator().manual_seed(723)
    x = torch.randn(20, 8, generator=generator)
    c = x.T @ x
    statistics = [torch.randn(5, 8, generator=generator) for _ in range(3)]
    expected = [fit_mount(c, s, 2) for s in statistics]
    original = torch.linalg.cholesky_ex
    calls = []

    def record(matrix, **kwargs):
        calls.append(matrix.device)
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    cache = []
    for s, reference in zip(statistics, expected, strict=True):
        actual = fit_mount(c.clone() if copy_gram else c, s, 2, cholesky_cache=cache)
        assert torch.equal(actual.down, reference.down)
        assert torch.equal(actual.up, reference.up)
        assert actual.ridge == reference.ridge
        assert actual.fit_share == reference.fit_share
        assert actual.svd_fallback == reference.svd_fallback
    assert len(calls) == 1


def test_group_different_grams_get_separate_cholesky(monkeypatch):
    original = torch.linalg.cholesky_ex
    calls = []

    def record(matrix, **kwargs):
        calls.append(True)
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    cache = []
    for scale in (1, 2, 1):
        fit_mount(torch.eye(4) * scale, torch.eye(4), 4, cholesky_cache=cache)
    assert len(calls) == 2


def test_solve_shares_cholesky_only_within_group(monkeypatch):
    from test_directions_contrast import Model, _samples, _settings

    from kilora.library import contrast

    original = torch.linalg.cholesky_ex
    calls = []

    def record(matrix, **kwargs):
        calls.append(True)
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    mounts = [
        f"transformer_blocks.{index}.attn1.to_{projection}"
        for index in range(2)
        for projection in ("q", "k", "v")
    ]
    result = contrast.solve(
        Model(),
        _samples(),
        [("same", "same")],
        mounts=mounts,
        model_sha="toy",
        license="Apache-2.0",
        settings=_settings(),
    )
    assert len(result.mounts) == 6
    assert len(calls) == 2


@pytest.mark.parametrize("comfy", [False, True])
def test_memory_estimate_counts_weights_captures_and_live_fit_once(comfy):
    from test_directions_contrast import Model
    from torch import nn

    from kilora.library import contrast

    model = Model().to(dtype=torch.bfloat16)
    model.transformer_blocks[0].ff.net[0].proj = nn.Linear(4, 16, bias=False, dtype=torch.bfloat16)
    model.transformer_blocks[0].ff.net[2] = nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
    settings = contrast.Settings(size=32, steps=3, sample_steps=(0, 1, 2))
    shapes = {
        name: (module.in_features, module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    weights = 416 * 2
    captures = 2 * 12 * 4 * 4 * (4 if comfy else 2)
    statistics = 4 * (16 * 16 + 4 * 16)
    cholesky = (4 if settings.fit_dtype == "float32" else 8) * 3 * 16 * 16
    actual = contrast.memory_estimate(model, settings, 2, shapes=shapes if comfy else None)
    replay = settings.replay_batch * 4 * (2 * 4 + 16) * (4 if comfy else 2)
    assert actual == weights + captures + statistics + cholesky + replay
    model.patch_spatial = 1
    assert contrast.memory_estimate(
        model, settings, 2, shapes=shapes if comfy else None
    ) == actual + 3 * (captures + replay)


def test_memory_estimate_does_not_combine_unrelated_mount_dimensions():
    from torch import nn

    from kilora.library import contrast

    model = nn.Linear(1, 1, bias=False)
    shapes = {"a": (16, 4), "b": (4, 32)}
    settings = contrast.Settings(size=16, seeds=(1,), steps=1, sample_steps=(0,))
    assert contrast.memory_estimate(model, settings, 1, shapes=shapes) == (
        4
        + 2 * 1 * 1 * 4 * 4
        + 4 * (16 * 16 + 16 * 4)
        + (4 if settings.fit_dtype == "float32" else 8) * 3 * 16 * 16
        + 1 * 1 * (2 * 4 + 32) * 4
    )


def test_fit_mount_fallback_decomposes_on_statistics_device(monkeypatch):
    original = torch.linalg.svd
    calls = []

    def record(matrix, **kwargs):
        calls.append((matrix.device.type, matrix.dtype))
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", record)
    c = torch.eye(4)
    s = torch.diag(torch.tensor([4.0, 2.0, 2.0, 1.0]))
    solution = fit_mount(c, s, 2)
    assert solution.svd_fallback
    assert 3 <= len(calls) <= 7
    assert all(call == (c.device.type, torch.float64) for call in calls)
    assert solution.exact_fit


def test_group_reuses_ridge_search_without_mutating_statistics(monkeypatch):
    c = torch.diag(torch.tensor([-0.005, 1.0, 1.0, 1.0]))
    before = c.clone()
    s = torch.eye(4)
    expected = fit_mount(c, s, 2, ridge=1e-4)
    original = torch.linalg.cholesky_ex
    calls = []

    def record(matrix, **kwargs):
        calls.append(True)
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    cache = []
    for _ in range(3):
        actual = fit_mount(c.clone(), s, 2, ridge=1e-4, cholesky_cache=cache)
        assert torch.equal(actual.delta(), expected.delta())
        assert actual.ridge == expected.ridge
    assert len(calls) == 3
    assert torch.equal(c, before)


def test_fit_mount_default_ridge_starts_at_tenth_mean_diagonal(monkeypatch):
    c = torch.diag(torch.tensor([2.0, 4.0, 6.0]))
    original = torch.linalg.cholesky_ex
    calls = []

    def record(matrix, **kwargs):
        calls.append(matrix.clone())
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    solution = fit_mount(c, torch.eye(3), 2)
    expected_ridge = 0.1 * float(c.diagonal().mean())
    assert Settings().ridge == 0.1
    assert len(calls) == 1
    assert torch.equal(calls[0], c.double() + expected_ridge * torch.eye(3).double())
    assert solution.ridge == pytest.approx(expected_ridge)


@pytest.mark.parametrize("rows,columns", [(64, 128), (128, 64), (128, 512), (512, 128)])
def test_randomized_fit_matches_exact_with_clear_gap(rows, columns):
    generator = torch.Generator().manual_seed(727)
    width, rank = min(rows, columns), 8
    left, _ = torch.linalg.qr(torch.randn(rows, width, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(
        torch.randn(columns, width, generator=generator, dtype=torch.float64)
    )
    values = torch.cat((torch.linspace(10, 6, rank), torch.linspace(0.3, 0.01, width - rank)))
    s = (left * values) @ right.T
    actual = _compare(torch.eye(columns, dtype=torch.float64), s, rank)
    assert not actual.exact_fit
    assert not actual.svd_fallback


def test_equal_values_at_cut_accept_change_before_svd_fallback(monkeypatch):
    width, rank = 64, 4
    values = torch.cat(
        (torch.tensor([10.0, 6.0, 3.0, 1e-4, 1e-4]), torch.linspace(1e-6, 1e-8, width - 5))
    ).double()
    c = torch.eye(width, dtype=torch.float64)
    s = torch.diag(values)
    qr = torch.linalg.qr
    svd = torch.linalg.svd
    qr_calls, svd_calls = [], []

    def record_qr(*args, **kwargs):
        qr_calls.append(True)
        return qr(*args, **kwargs)

    def record_svd(matrix, **kwargs):
        svd_calls.append((matrix.shape, matrix.device, matrix.dtype))
        return svd(matrix, **kwargs)

    def unexpected_eigh(*args, **kwargs):
        pytest.fail("the change converges before the exact eigenvalue path")

    monkeypatch.setattr(torch.linalg, "qr", record_qr)
    monkeypatch.setattr(torch.linalg, "svd", record_svd)
    monkeypatch.setattr(torch.linalg, "eigh", unexpected_eigh)
    actual = fit_mount(c, s, rank)
    assert len(qr_calls) == 13
    assert [shape for shape, _, _ in svd_calls] == [(20, width), (20, width), (width, width)]
    assert all(device == c.device and dtype == torch.float64 for _, device, dtype in svd_calls)
    assert actual.svd_fallback
    assert actual.exact_fit
    expected = _reference(c, s, rank, 0.1)
    error = torch.linalg.vector_norm(actual.delta() - expected.delta())
    assert error <= 1e-3 * torch.linalg.vector_norm(expected.delta())


def test_slowly_converging_spectrum_uses_exact_fit(monkeypatch):
    width, rank = 64, 4
    s = torch.diag(torch.linspace(1, 0.8, width, dtype=torch.float64))
    c = torch.eye(width)
    qr = torch.linalg.qr
    eigh = torch.linalg.eigh
    calls = []
    exact_devices = []

    def record(*args, **kwargs):
        calls.append(True)
        return qr(*args, **kwargs)

    def record_eigh(matrix, **kwargs):
        exact_devices.append((matrix.device, matrix.dtype))
        return eigh(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", record)
    monkeypatch.setattr(torch.linalg, "eigh", record_eigh)
    actual = _compare(c, s, rank)
    assert actual.exact_fit
    assert not actual.svd_fallback
    assert len(calls) == 25
    assert exact_devices == [(c.device, torch.float64)]


def test_randomized_fit_repeats_without_changing_global_rng():
    generator = torch.Generator().manual_seed(728)
    c = torch.eye(64)
    s = torch.randn(96, 64, generator=generator)
    s[:, 8:] *= 0.001
    state = torch.random.get_rng_state().clone()
    first = fit_mount(c, s, 8)
    second = fit_mount(c, s, 8)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first.up, second.up)
    assert torch.equal(first.down, second.down)
    assert first.fit_share == second.fit_share
    assert not first.exact_fit
    assert first.exact_fit == second.exact_fit


@pytest.mark.parametrize("exact", [False, True])
def test_solve_counts_exact_fits_and_measures_only_fit_calls(monkeypatch, tmp_path, exact):
    from test_directions_contrast import Model, _samples, _settings

    from kilora.library import contrast

    original = contrast.fit_mount
    clock = [0.0]

    def fit(*args, **kwargs):
        solution = original(*args, **kwargs)
        solution.exact_fit = exact
        clock[0] += 2.0
        return solution

    monkeypatch.setattr(contrast, "fit_mount", fit)
    monkeypatch.setattr(contrast.time, "perf_counter", lambda: clock[0])
    model = Model()
    partial = tmp_path / "fits.partial.safetensors"
    result = contrast.solve(
        model,
        _samples(),
        [("same", "same")],
        model_sha="toy",
        license="Apache-2.0",
        settings=_settings(),
        partial=partial,
    )
    assert result.metadata["exact_fits"] == len(result.mounts) * int(exact)
    assert result.fit_seconds == 2.0 * len(result.mounts)
    assert "fit_seconds" not in result.metadata
    assert all(
        record.get("exact_fit", False) == exact for record in result.metadata["fit"].values()
    )
    resumed = contrast.solve(
        model,
        _samples(),
        [("same", "same")],
        model_sha="toy",
        license="Apache-2.0",
        settings=_settings(),
        partial=partial,
    )
    assert resumed.metadata["exact_fits"] == result.metadata["exact_fits"]
    assert resumed.fit_seconds == 0.0
    assert all(solution.exact_fit == exact for solution in resumed.mounts.values())


def test_float32_fit_matches_float64(monkeypatch):
    generator = torch.Generator().manual_seed(733)
    width = 32
    x = torch.randn(128, width, generator=generator)
    c = x.T @ x
    s = torch.randn(24, width, generator=generator)
    reference = fit_mount(c, s, 8, ridge=0.1)
    seen = []

    def record(name):
        original = getattr(torch.linalg, name)

        def wrapped(*args, **kwargs):
            seen.append(tuple(arg.dtype for arg in args if torch.is_tensor(arg)))
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, wrapped)

    for name in ("cholesky_ex", "solve_triangular", "eigh"):
        record(name)
    actual = fit_mount(c, s, 8, ridge=0.1, fit_dtype="float32")
    assert seen
    assert all(dtype == torch.float32 for call in seen for dtype in call)
    assert torch.allclose(actual.delta(), reference.delta(), rtol=1e-3, atol=1e-5)
    assert actual.ridge == pytest.approx(reference.ridge, rel=1e-3, abs=1e-6)
    assert actual.down.dtype == actual.up.dtype == torch.float32
    assert actual.down.device.type == actual.up.device.type == "cpu"
