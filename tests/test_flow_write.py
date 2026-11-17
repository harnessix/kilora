# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import sys
import types

import pytest
import torch
from test_import_edit_dimensions import _session as _imported

from kilora import live_queue
from kilora.comfyui import live_write
from kilora.comfyui.live_write import (
    PathHandle,
    flow_pairs,
    flow_seed,
    flow_sigmas,
    image_loss,
    min_sigma,
)
from kilora.forward_estimate import ForwardEstimate
from kilora.live import SESSIONS
from kilora.live_imports import accept_measure
from kilora.recording import Perturbation


@pytest.fixture(autouse=True)
def _clear_sessions(monkeypatch):
    monkeypatch.delenv(live_write.PROFILE_ENV, raising=False)
    SESSIONS.clear()
    yield
    SESSIONS.clear()


class FlowSampling:
    def __init__(self, shift=3.0, multiplier=1.0):
        self.shift, self.multiplier = shift, multiplier
        steps = torch.arange(1, 1001, dtype=torch.float64) / 1000 * multiplier
        self.sigmas = self.sigma(steps)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def sigma(self, timestep):
        t = timestep / self.multiplier
        return self.shift * t / (1 + (self.shift - 1) * t)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class NoTimesteps(list):
    def _fail(self, *_args, **_kwargs):
        raise AssertionError("the write read the render's timesteps")

    __iter__ = __len__ = __getitem__ = __bool__ = _fail


def _model(seen):
    def apply_model(x_t, t, c_crossattn):
        seen.append((x_t.clone(), t.clone()))
        rows = c_crossattn.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))
        return 0.5 * x_t + 0.1 * t.reshape(-1, *([1] * (x_t.dim() - 1))) + rows

    return apply_model


def _handle(session, seen, sampling=None):
    return PathHandle(
        session=session,
        rule=None,
        recorder=None,
        timesteps=NoTimesteps(),
        conditioning={"c_crossattn": torch.full((1, 2), 0.3)},
        empty_conditioning={"c_crossattn": torch.full((1, 2), -0.2)},
        apply_model=_model(seen),
        patcher=object(),
        sampling=sampling,
    )


def _x0():
    return torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(11))


def _estimate(session, draws=16):
    return ForwardEstimate(session, session.state, draws=draws, sigma=0.05, chunk=8, seed=3)


def _write(measure, estimate):
    estimate.run_clean(measure)
    estimate.set_target(None)
    assert estimate.run_draws(measure)
    measure.held(Perturbation({}, {}))


def test_the_write_reads_no_render_timesteps_and_repeats_its_pairs(tmp_path):
    session = _imported(tmp_path)
    assert not hasattr(live_write, "sample_plan")
    first = image_loss(_handle(session, [], FlowSampling()), _x0(), 0)
    with first.hold():
        _write(first, _estimate(session))
    second = image_loss(_handle(session, [], FlowSampling()), _x0(), 5)
    assert first.sigmas == second.sigmas
    for (s_a, n_a), (s_b, n_b) in zip(first.pairs, second.pairs, strict=True):
        assert s_a == s_b and torch.equal(n_a, n_b)
    session.commits.append(None)
    try:
        third = image_loss(_handle(session, [], FlowSampling()), _x0(), 0)
    finally:
        session.commits.pop()
    assert third.sigmas != first.sigmas
    assert flow_seed(session) != flow_seed(session, retry=1)


def test_sigma_follows_the_sampling_shift_and_the_minimum(tmp_path):
    sampling = FlowSampling()
    low = 3 * 0.001 / (1 + 2 * 0.001)
    assert min_sigma(sampling) == pytest.approx(low, rel=1e-12)
    assert min_sigma(types.SimpleNamespace(shift=3.0)) == pytest.approx(low, rel=1e-12)
    u = torch.linspace(0, 1, 101, dtype=torch.float64)
    expected = (3 * u / (1 + 2 * u)).clamp(min=low, max=1.0)
    assert torch.allclose(flow_sigmas(sampling, u), expected)
    assert torch.allclose(flow_sigmas(types.SimpleNamespace(shift=3.0), u), expected)
    assert float(flow_sigmas(sampling, u).min()) >= low
    session = _imported(tmp_path)
    session.set_options(measure_samples=64)
    measure = image_loss(_handle(session, [], sampling), _x0(), 0)
    assert len(measure.sigmas) == 128
    assert min(measure.sigmas) >= low
    gen = torch.Generator().manual_seed(measure.seed)
    for sigma in measure.sigmas:
        u = torch.rand((1,), generator=gen, dtype=torch.float64).clamp(min=1e-12)
        torch.randn((1, 5, 4), generator=gen)
        assert sigma == pytest.approx(float(flow_sigmas(sampling, u)[0]), rel=1e-12)


def test_the_loss_is_the_denoised_error_over_sigma_squared(tmp_path):
    session = _imported(tmp_path)
    sampling = FlowSampling()
    measure = image_loss(_handle(session, [], sampling), _x0(), 0)
    x0 = _x0()
    expected = []
    for sigma, noise in measure.pairs[: measure.sample_count]:
        x_t = sigma * noise + (1 - sigma) * x0
        t = torch.full((1,), sigma)
        losses = [
            float((_model([])(x_t, t, c_crossattn=torch.full((1, 2), c)) - x0).pow(2).mean())
            for c in (0.3, -0.2)
        ]
        expected.append(sum(losses) / 2 / sigma**2)
    assert measure.batched(None) == pytest.approx(sum(expected) / len(expected), rel=1e-5)
    assert measure.sample_losses == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(("count", "draws", "rows"), [(4, 16, 8), (4, 16, 4), (3, 5, 4)])
def test_the_model_calls_of_one_write(tmp_path, count, draws, rows):
    session = _imported(tmp_path)
    session.set_options(measure_samples=count, forward_chunk=rows)
    measure = image_loss(_handle(session, []), _x0(), 0)
    estimate = _estimate(session, draws)
    with measure.hold():
        _write(measure, estimate)
    calls = math.ceil(2 * count / rows) + math.ceil(count * draws / rows) + math.ceil(count / rows)
    assert measure.calls[0] == calls
    if (count, draws, rows) == (4, 16, 8):
        assert calls == math.ceil(count * (1 + draws) / rows) + math.ceil(count / rows) == 10
    measure.held(Perturbation({}, {}))
    assert measure.calls[0] == calls + math.ceil(count / rows)


def test_the_accept_pairs_differ_from_the_estimate_pairs(tmp_path):
    session = _imported(tmp_path)
    seen = []
    measure = image_loss(_handle(session, seen, FlowSampling()), _x0(), 0)
    count = measure.sample_count
    assert measure.accept_count == count
    estimate_pairs, held_pairs = measure.pairs[:count], measure.pairs[count:]
    for _s, noise in held_pairs:
        assert all(not torch.equal(noise, n) for _t, n in estimate_pairs)
    assert set(measure.sigmas[count:]).isdisjoint(measure.sigmas[:count])
    alone, after = flow_pairs(FlowSampling(), (1, 5, 4), count, count, measure.seed)
    assert [s for s, _n in alone] == measure.sigmas[:count]
    assert [s for s, _n in after] == measure.sigmas[count:]
    del seen[:]
    measure.held(Perturbation({}, {}))
    held_x = {float(v) for _x, t in seen for v in t}
    assert held_x == {float(torch.tensor(s)) for s in measure.sigmas[count:]}
    run = accept_measure(measure)
    assert run.held_out


class FakeManagement(types.ModuleType):
    def __init__(self, state):
        super().__init__("comfy.model_management")
        self.state, self.loads = state, 0

    def load_models_gpu(self, models, *args, **kwargs):
        assert not self.state["render"], "the write loaded the model inside a render"
        self.loads += 1


def test_no_load_and_no_stop_inside_a_render(tmp_path, monkeypatch):
    state = {"render": True, "waits": 0}
    fake = FakeManagement(state)
    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_management = fake
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)

    def wait():
        if state["render"]:
            state["render"] = False
            state["waits"] += 1
            return 0.5
        return 0.0

    def no_stop(*_args, **_kwargs):
        raise AssertionError("the write made a model call preemptible")

    monkeypatch.setattr(live_write, "yield_point", wait)
    monkeypatch.setattr(live_queue, "rendering", lambda: state["render"])
    monkeypatch.setattr(live_queue, "preemptible", no_stop)
    assert not hasattr(live_write, "preemptible")
    session = _imported(tmp_path)
    handle = _handle(session, [])
    inner = handle.apply_model
    calls = [0]

    def apply_model(x_t, t, **cond):
        out = inner(x_t, t, **cond)
        calls[0] += 1
        if calls[0] % 3 == 0:
            state["render"] = True
        return out

    handle.apply_model = apply_model
    measure = image_loss(handle, _x0(), 0)
    with measure.hold():
        _write(measure, _estimate(session))
    assert state["waits"] >= 2
    assert fake.loads == state["waits"]
    assert measure.timing["preempts"] == 0
    count = measure.sample_count
    assert (
        measure.calls[0]
        == calls[0]
        == (math.ceil(2 * count / 8) + math.ceil(count * 16 / 8) + math.ceil(count / 8))
    )


def _render(h=128, w=96, channels=4, seed=5):
    return torch.randn(1, channels, h, w, generator=torch.Generator().manual_seed(seed))


def test_the_target_latent_has_the_write_resolution(tmp_path):
    session = _imported(tmp_path)
    assert session.write_resolution == live_write.write_size(session) == 512
    small, note = live_write.write_latent(_render(), session)
    assert note == ""
    assert tuple(small.shape) == (1, 4, 64, 48)
    assert torch.allclose(small, torch.nn.functional.avg_pool2d(_render(), 2), atol=1e-6)
    odd, _ = live_write.write_latent(_render(128, 75), session)
    h, w = odd.shape[-2:]
    assert max(h, w) * 8 == 512 and h % 2 == 0 and w % 2 == 0
    assert abs(w / h - 75 / 128) <= 2 / h
    frames, _ = live_write.write_latent(_render()[:, :, None], session)
    assert tuple(frames.shape) == (1, 4, 1, 64, 48)
    session.set_options(write_resolution=0)
    latent = _render()
    assert live_write.write_latent(latent, session)[0] is latent
    session.set_options(write_resolution=2048)
    assert live_write.write_latent(latent, session)[0] is latent
    plain = types.SimpleNamespace(measure_size=256, measure_reducer="latent")
    assert tuple(live_write.write_latent(latent, plain)[0].shape[-2:]) == (32, 24)


class ModelSamplingFlux:
    def __init__(self, shift=1.15):
        self.shift = shift

    def sigma(self, timestep):
        return math.exp(self.shift) / (math.exp(self.shift) + (1 / timestep - 1))

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


def test_sigma_follows_the_shift_rule_for_the_write_tokens(tmp_path):
    assert live_write.write_tokens((1, 4, 64, 48)) == 32 * 24
    assert live_write.write_tokens((1, 16, 1, 128, 128)) == 4096
    anima = FlowSampling()
    assert live_write.write_sampling(anima, 4096, 1024) is anima
    session = _imported(tmp_path)
    small, _ = live_write.write_latent(_render(), session)
    at_render = image_loss(_handle(session, [], anima), small, 0)
    at_write = image_loss(_handle(session, [], anima), small, 0, render_shape=(1, 4, 128, 96))
    assert at_write.sigmas == at_render.sigmas
    assert (at_write.tokens, at_write.render_tokens) == (768, 3072)
    flux = ModelSamplingFlux(1.15)
    shifted = live_write.write_sampling(flux, 4096, 1024)
    mu = 1.15 + (1.15 - 0.5) / (4096 - 256) * (1024 - 4096)
    assert shifted.shift == pytest.approx(mu)
    assert float(shifted.sigma(torch.tensor(0.5))) == pytest.approx(
        math.exp(mu) / (math.exp(mu) + 1)
    )
    assert live_write.write_sampling(flux, 4096, 4096).shift == pytest.approx(1.15)
    u = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    full = flow_sigmas(live_write.write_sampling(flux, 4096, 4096), u)
    assert bool((flow_sigmas(shifted, u) < full).all())
    measure = image_loss(_handle(session, [], flux), small, 0, render_shape=(1, 4, 128, 96))
    assert measure.sampling.shift == pytest.approx(1.15 + (1.15 - 0.5) / 3840 * (768 - 3072))
    gen = torch.Generator().manual_seed(measure.seed)
    for sigma in measure.sigmas:
        u = torch.rand((1,), generator=gen, dtype=torch.float64).clamp(min=1e-12)
        torch.randn(tuple(small.shape), generator=gen)
        assert sigma == pytest.approx(float(flow_sigmas(measure.sampling, u)[0]), rel=1e-12)


def test_write_resolution_zero_gives_the_loss_of_before(tmp_path):
    session = _imported(tmp_path)
    latent = _render(32, 24)
    before = image_loss(_handle(session, [], FlowSampling()), latent, 0)
    session.set_options(write_resolution=0)
    target, _ = live_write.write_latent(latent, session)
    after = image_loss(
        _handle(session, [], FlowSampling()), target, 0, render_shape=tuple(latent.shape)
    )
    assert after.sigmas == before.sigmas
    assert after.batched(None) == before.batched(None)
    assert after.held(None) == before.held(None)
    assert after.resolution == (256, 192)


def test_the_accept_checks_run_at_the_resolution_of_the_estimate(tmp_path):
    from kilora.measure_size import resolution_clause

    session = _imported(tmp_path)
    session.set_options(measure_samples=2, forward_chunk=4)
    small, _ = live_write.write_latent(_render(), session)
    seen = []
    measure = image_loss(
        _handle(session, seen, FlowSampling()), small, 0, render_shape=(1, 4, 128, 96)
    )
    with measure.hold():
        _write(measure, _estimate(session, draws=2))
    estimate_shapes = {tuple(x.shape[-2:]) for x, _t in seen}
    del seen[:]
    measure.held(Perturbation({}, {}))
    accept_measure(measure)(None)
    accept_shapes = {tuple(x.shape[-2:]) for x, _t in seen}
    assert estimate_shapes == accept_shapes == {(64, 48)}
    assert measure.resolution == (512, 384)
    assert resolution_clause(measure) == ", samples at 384x512 px, 768 tokens per row"


def test_single_pair_changes_per_write_and_retry_and_is_shared(tmp_path):
    session = _imported(tmp_path)
    session.set_options(measure_samples=1)
    assert session.measure_samples == 1
    seen = []
    first = image_loss(_handle(session, seen, FlowSampling()), _x0(), 0)
    with first.hold():
        _write(first, _estimate(session, draws=4))
    times = torch.cat([t for _x, t in seen])
    sigma = float(torch.tensor(first.sigmas[0]))
    assert int((times == sigma).sum()) == 10
    assert first.calls[0] == 3 and first.rows_count[0] == 7
    assert first.accept_count == 1
    retry = image_loss(_handle(session, [], FlowSampling()), _x0(), 0, retry=1)
    session.commits.append(None)
    try:
        second = image_loss(_handle(session, [], FlowSampling()), _x0(), 0)
    finally:
        session.commits.pop()
    assert len({first.sigmas[0], retry.sigmas[0], second.sigmas[0]}) == 3
    for measure in (first, retry, second):
        gen = torch.Generator().manual_seed(measure.seed)
        u = torch.rand((1,), generator=gen, dtype=torch.float64).clamp(min=1e-12)
        assert measure.sigmas[0] == float(flow_sigmas(FlowSampling(), u)[0])
    from kilora.comfyui.live_write import timing_note

    note = timing_note(types.SimpleNamespace(measure=first))
    assert "T=1" in note and "sigma=[" in note and "held-out sigma=[" in note
    assert "3 model calls" in note and "7 sample rows" in note


def test_four_pairs_retain_the_base_generator_loss_and_calls(tmp_path):
    session = _imported(tmp_path)
    session.set_options(measure_samples=4)
    seen = []
    measure = image_loss(_handle(session, seen, FlowSampling()), _x0(), 0)
    gen = torch.Generator().manual_seed(flow_seed(session))
    expected = []
    for _ in range(8):
        u = torch.rand((1,), generator=gen, dtype=torch.float64).clamp(min=1e-12)
        sigma = float(flow_sigmas(FlowSampling(), u)[0])
        noise = torch.randn(tuple(_x0().shape), generator=gen)
        expected.append((sigma, noise))
    for actual, before in zip(measure.pairs, expected, strict=True):
        assert actual[0] == before[0] and torch.equal(actual[1], before[1])
    losses = []
    for sigma, noise in expected[:4]:
        x = sigma * noise + (1 - sigma) * _x0()
        t = torch.full((1,), sigma)
        losses.append(
            sum(
                float((_model([])(x, t, torch.full((1, 2), c)) - _x0()).pow(2).mean())
                for c in (0.3, -0.2)
            )
            / 2
            / sigma**2
        )
    assert measure.batched(None) == pytest.approx(sum(losses) / 4, rel=1e-6)
    measure = image_loss(_handle(session, [], FlowSampling()), _x0(), 0)
    with measure.hold():
        _write(measure, _estimate(session, draws=4))
    assert measure.calls[0] == 4 and measure.rows_count[0] == 28


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_probe_step_uses_next_write_pair(shift):
    from kilora.comfyui.probe_rows import probe_step

    session = types.SimpleNamespace(mod_id="probe", commits=[])
    sampling = FlowSampling(shift)
    schedule = torch.tensor([1.0, 0.8, 0.4, 0.1, 0.0])
    pairs, _ = flow_pairs(sampling, (1,), 1, 0, flow_seed(session))
    expected = min(range(4), key=lambda i: abs(float(schedule[i]) - pairs[0][0]))
    assert probe_step(schedule, sampling, session) == expected
    assert probe_step([0.0], sampling, session) is None


def test_retained_probe_reflow_loss():
    from kilora.comfyui.probe_rows import ProbeOutputs

    gen = torch.Generator().manual_seed(93)
    x0 = torch.randn((1, 2, 3, 3), generator=gen)
    noise = torch.randn(x0.shape, generator=gen)
    sigma = 0.43
    xt = sigma * noise + (1 - sigma) * x0
    clean = torch.randn((2, *x0.shape), generator=gen)
    draws = torch.randn((4, 2, *x0.shape), generator=gen)
    record = ProbeOutputs.capture(seed=1, sigma=sigma, x_t=xt, clean=clean, draws=draws)
    assert torch.allclose(record.reflow_noise(x0), noise, atol=1e-6)
    baseline, losses = record.losses(x0)
    target = record.reflow_noise(x0) - x0
    predicted = (xt - draws) / sigma
    expected = (predicted - target).square().flatten(1).mean(1)
    assert torch.allclose(losses, expected, rtol=1e-6)
    assert baseline == pytest.approx(float((clean - x0).square().mean() / sigma**2))
    draws.zero_()
    assert record.draws.count_nonzero() > 0


def test_probe_budget_preserves_pairs_and_image_rows():
    from kilora.comfyui.probe_rows import probe_capacity

    assert probe_capacity(8, 100, None) == 8
    assert probe_capacity(8, 100, 1200) == 6
    assert probe_capacity(8, 100, 399) == 0
    assert probe_capacity(7, 100, None) == 6


def test_probe_noise_pairs_do_not_consume_global_rng():
    from kilora.comfyui.probe_rows import paired_noise

    state = torch.random.get_rng_state().clone()
    for draw in (0, 2, 4):
        plus = paired_noise((2, 3), seed=37, draw=draw)
        minus = paired_noise((2, 3), seed=37, draw=draw + 1)
        assert torch.equal(plus, -minus)
    assert torch.equal(state, torch.random.get_rng_state())


def test_probe_revision_and_shape_guard():
    from kilora.comfyui.probe_rows import ProbeOutputs

    session = types.SimpleNamespace(mod_id="probe", commits=[])
    x = torch.zeros(1, 2, 3, 3)
    record = ProbeOutputs.capture(
        seed=flow_seed(session),
        sigma=0.5,
        x_t=x,
        clean=torch.stack([x, x]),
        draws=torch.stack([torch.stack([x, x])] * 2),
    )
    assert record.matches(session, x)
    assert not record.matches(session, x[..., :1])
    session.commits.append(object())
    assert not record.matches(session, x)


def test_deferred_probe_acceptance_halves_until_next_render():
    from kilora.comfyui.probe_rows import DeferredProbeStep

    commits = []
    step = DeferredProbeStep({"candidate": 1})
    note = step.decide(1, 2.0, 3.0, lambda *args: commits.append(args))
    assert step.scale == 0.5
    assert "next render" in note
    assert not commits
    with pytest.raises(ValueError, match="new render"):
        step.decide(1, 2.0, 1.0, lambda *args: commits.append(args))
    note = step.decide(2, 2.0, 1.0, lambda *args: commits.append(args))
    assert step.accepted
    assert commits == [({"candidate": 1}, 0.5)]
    assert "committed" in note
    step.decide(2, 2.0, 1.0, lambda *args: commits.append(args))
    assert len(commits) == 1


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), 2.0])
def test_deferred_probe_acceptance_requires_finite_improvement(loss):
    from kilora.comfyui.probe_rows import DeferredProbeStep

    step = DeferredProbeStep(None)

    def unexpected(*_args):
        raise AssertionError("candidate must not commit")

    step.decide(0, 2.0, loss, unexpected)
    assert not step.accepted
    assert step.scale == 0.5
