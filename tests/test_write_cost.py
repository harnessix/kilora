# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import sys
import types

import pytest
import torch
from test_import_edit_dimensions import _session as _imported

from kilora.comfyui import live_write
from kilora.comfyui.live_write import PathHandle, WriteProfile, image_loss
from kilora.editors import commit_write
from kilora.forward_estimate import ForwardEstimate
from kilora.live import SESSIONS
from kilora.recording import GenerationRecord, Perturbation, prediction_note

TIMESTEPS = [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]
EVENT = 1


@pytest.fixture(autouse=True)
def _clear_sessions(monkeypatch):
    monkeypatch.delenv(live_write.PROFILE_ENV, raising=False)
    monkeypatch.delenv("KILORA_PREDICTION_DEBUG", raising=False)
    SESSIONS.clear()
    yield
    SESSIONS.clear()


class FakeManagement(types.ModuleType):
    def __init__(self):
        super().__init__("comfy.model_management")
        self.loads = 0

    def load_models_gpu(self, models, *args, **kwargs):
        self.loads += 1


@pytest.fixture
def mm(monkeypatch):
    fake = FakeManagement()
    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_management = fake
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
    return fake


def _handle(session, seen):
    def apply_model(x_t, t, c_crossattn):
        seen.append((x_t.clone(), t.clone()))
        rows = c_crossattn.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))
        return 0.5 * x_t + 0.1 * t.reshape(-1, *([1] * (x_t.dim() - 1))) + rows

    return PathHandle(
        session=session,
        rule=None,
        recorder=None,
        timesteps=list(TIMESTEPS),
        conditioning={"c_crossattn": torch.full((1, 2), 0.3)},
        empty_conditioning={"c_crossattn": torch.full((1, 2), -0.2)},
        apply_model=apply_model,
        patcher=object(),
    )


def _x0():
    return torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(11))


def _estimate(session, draws=16, chunk=4):
    return ForwardEstimate(session, session.state, draws=draws, sigma=0.05, chunk=chunk, seed=3)


def _write(session, measure, estimate):
    estimate.run_clean(measure)
    estimate.set_target(None)
    assert estimate.run_draws(measure)
    measure.held(Perturbation({}, {}))


def _calls(count, draws, rows):
    return math.ceil(2 * count / rows) + math.ceil(count * draws / rows) + math.ceil(count / rows)


class WritingRule:
    needs_path = False

    def __init__(self, session, estimate):
        self.session, self.estimate = session, estimate

    def next_state(self, state, record):
        _write(self.session, record.measure, self.estimate)
        return None


def _record(measure):
    return GenerationRecord(
        latent=_x0(),
        conditioning=None,
        timesteps=list(TIMESTEPS),
        images=1,
        measure=measure,
        guard_score=measure.guard_score,
    )


def test_one_write_loads_the_model_once(tmp_path, mm):
    session = _imported(tmp_path)
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    estimate = _estimate(session)
    session.learning = True
    commit_write(session, WritingRule(session, estimate), _record(measure))
    assert estimate.forwards == 2
    assert mm.loads == 1
    measure(Perturbation({}, {}))
    measure(Perturbation({}, {}))
    assert mm.loads == 3


def test_a_pause_for_a_render_loads_the_model_again(tmp_path, mm, monkeypatch):
    session = _imported(tmp_path)
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    with measure.hold():
        measure(Perturbation({}, {}))
        measure(Perturbation({}, {}))
        assert mm.loads == 1
        pauses = iter([0.5])
        monkeypatch.setattr(live_write, "yield_point", lambda: next(pauses, 0.0))
        measure(Perturbation({}, {}))
        assert mm.loads == 2


def _sample_rows(seen):
    return [
        part
        for x, t in seen
        for part in zip(x.chunk(x.shape[0] // 2), t.chunk(t.shape[0] // 2), strict=True)
    ]


def test_every_draw_uses_the_pair_of_its_clean_row(tmp_path):
    session = _imported(tmp_path)
    count = session.measure_samples
    assert count == 4
    seen = []
    measure = image_loss(_handle(session, seen), _x0(), EVENT)
    estimate = _estimate(session, draws=10)
    with measure.hold():
        _write(session, measure, estimate)
    rows = _sample_rows(seen)
    clean, held = rows[:count], rows[count : 2 * count]
    draws, accept = rows[2 * count : 2 * count + 10 * count], rows[2 * count + 10 * count :]
    assert len({float(t[0]) for _x, t in clean}) == count
    for r, (x, t) in enumerate(draws):
        x_c, t_c = clean[r % count]
        assert torch.equal(x, x_c) and torch.equal(t, t_c)
    assert len(accept) == count
    for (x, t), (x_h, t_h) in zip(accept, held, strict=True):
        assert torch.equal(x, x_h) and torch.equal(t, t_h)
    assert measure.rows_count[0] == 2 * count + 10 * count + count


@pytest.mark.parametrize(("draws", "chunk"), [(16, 4), (10, 4), (5, 8), (7, 3)])
def test_the_model_calls_of_one_write(tmp_path, draws, chunk):
    session = _imported(tmp_path)
    session.set_options(forward_chunk=chunk)
    count = session.measure_samples
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    estimate = _estimate(session, draws=draws, chunk=chunk)
    with measure.hold():
        _write(session, measure, estimate)
    assert measure.calls[0] == _calls(count, draws, chunk)
    measure.held(Perturbation({}, {}))
    assert measure.calls[0] == _calls(count, draws, chunk) + math.ceil(count / chunk)


def test_the_default_model_calls(tmp_path):
    from kilora.live_experts import FORWARD_CHUNK, FORWARD_DRAWS

    session = _imported(tmp_path)
    assert FORWARD_CHUNK == 8 and FORWARD_DRAWS == 4
    assert session.measure_samples == 4 and live_write.sample_chunk(session) == FORWARD_CHUNK
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    estimate = _estimate(session, draws=FORWARD_DRAWS, chunk=FORWARD_CHUNK)
    with measure.hold():
        _write(session, measure, estimate)
    assert measure.calls[0] == _calls(4, 4, 8)
    assert measure.rows_count[0] == 4 + 4 + 16 + 4
    assert estimate.clean_calls == 1 and estimate.draw_calls == 2


def test_the_write_profile(tmp_path, mm, monkeypatch):
    session = _imported(tmp_path)
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    assert measure.profile is None
    monkeypatch.setenv(live_write.PROFILE_ENV, "1")
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    assert isinstance(measure.profile, WriteProfile)
    with measure.hold():
        _write(session, measure, _estimate(session))
    profile = measure.profile
    assert len(profile.loads) == 1
    assert len(profile.calls) == measure.calls[0] == 10
    assert sorted({rows for rows, _s in profile.calls}) == [8, 16]
    assert [name for name, _s in profile.phases] == ["clean", "draw"]
    line = profile.line()
    assert line.startswith(" Write profile: loads 1")
    assert "model calls 10" in line and "9 x 16 rows" in line and "1 x 8 rows" in line
    assert "batch rows 152" in line


def test_the_write_note_has_no_eps_fields_without_debug(monkeypatch):
    from types import SimpleNamespace

    cross = {"a": [SimpleNamespace(step=s, node=0) for s in (2, 3)]}
    note = prediction_note({}, cross, None, None, 1.5)
    assert "eps_" not in note and "target_forward_s=1.50" in note
    assert prediction_note({}, cross) == ""
    monkeypatch.setenv("KILORA_PREDICTION_DEBUG", "1")
    note = prediction_note({}, cross, None, None, 1.5)
    assert "eps_steps=2 eps_nodes=1" in note and "eps_cross_steps=2" in note


def test_measure_samples_defaults_to_4(tmp_path):
    from kilora.live_imports import ImportedLiveSession

    assert ImportedLiveSession.MEASURE_SAMPLES == 4
    assert _imported(tmp_path).measure_samples == 4


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 8])
def test_clean_pass_and_accept_use_ceil_rows_over_chunk_calls(tmp_path, chunk):
    session = _imported(tmp_path)
    session.set_options(forward_chunk=chunk)
    count = session.measure_samples
    seen = []
    measure = image_loss(_handle(session, seen), _x0(), EVENT)
    estimate = _estimate(session, draws=8)
    estimate.run_clean(measure)
    assert len(seen) == math.ceil(2 * count / chunk)
    assert len(_sample_rows(seen)) == 2 * count
    assert measure.chunking["used"] == min(chunk, 2 * count)
    held = _sample_rows(seen)[count:]
    del seen[:]
    measure.held(Perturbation({}, {}))
    assert len(seen) == math.ceil(count / chunk)
    for (x, t), (x_h, t_h) in zip(_sample_rows(seen), held, strict=True):
        assert torch.equal(x, x_h) and torch.equal(t, t_h)


def test_the_batched_loss_equals_the_per_sample_loss(tmp_path):
    session = _imported(tmp_path)
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    per_sample = measure(Perturbation({}, {}))
    losses = list(measure.sample_losses)
    for chunk in (1, 2, 4):
        measure.chunking["chunk"] = chunk
        assert measure.batched(Perturbation({}, {})) == pytest.approx(per_sample, rel=1e-6)
        assert measure.sample_losses == pytest.approx(losses, rel=1e-6)


def _bounded_handle(session, seen, rows):
    handle = _handle(session, seen)
    inner = handle.apply_model

    def apply_model(x_t, t, **cond):
        if x_t.shape[0] > rows:
            raise live_write._out_of_memory()[0]("out of memory (test bound)")
        return inner(x_t, t, **cond)

    handle.apply_model = apply_model
    return handle


def test_a_batch_over_the_memory_bound_splits_into_chunks(tmp_path):
    session = _imported(tmp_path)
    count = session.measure_samples
    whole = image_loss(_handle(session, []), _x0(), EVENT)
    expected = whole.batched(None)
    assert whole.chunking == {"chunk": 8, "used": 4, "split": 0, "separate": False, "budget": None}
    seen = []
    measure = image_loss(_bounded_handle(session, seen, rows=4), _x0(), EVENT)
    loss = measure.batched(None)
    assert loss == pytest.approx(expected, rel=1e-6)
    assert measure.chunking == {
        "chunk": 2,
        "used": 2,
        "split": 1,
        "separate": False,
        "budget": None,
    }
    assert [int(x.shape[0]) for x, _t in seen] == [4] * (count // 2)
    est_whole, est_split = _estimate(session), _estimate(session)
    clean_whole = est_whole.run_clean(image_loss(_handle(session, []), _x0(), EVENT))
    clean_split = est_split.run_clean(
        image_loss(_bounded_handle(session, [], rows=4), _x0(), EVENT)
    )
    assert clean_split == pytest.approx(clean_whole, rel=1e-6)
    assert est_split.reads.keys() == est_whole.reads.keys()


def test_one_sample_over_the_bound_runs_one_condition_per_call(tmp_path):
    session = _imported(tmp_path)
    count = session.measure_samples
    expected = image_loss(_handle(session, []), _x0(), EVENT).batched(None)
    seen = []
    measure = image_loss(_bounded_handle(session, seen, rows=1), _x0(), EVENT)
    assert measure.batched(None) == pytest.approx(expected, rel=1e-6)
    assert measure.chunking["chunk"] == 1 and measure.chunking["separate"]
    assert [int(x.shape[0]) for x, _t in seen] == [1] * (2 * count)
    measure = image_loss(_bounded_handle(session, [], rows=0), _x0(), EVENT)
    with pytest.raises(live_write._out_of_memory()[0]):
        measure.batched(None)


def test_a_draw_pass_over_the_bound_halves_its_rows(tmp_path):
    session = _imported(tmp_path)
    whole = _estimate(session, draws=8)
    measure = image_loss(_handle(session, []), _x0(), EVENT)
    whole.run_clean(measure)
    whole.set_target(None)
    assert whole.run_draws(measure)
    split = _estimate(session, draws=8)
    seen = []
    bounded = image_loss(_bounded_handle(session, seen, rows=4), _x0(), EVENT)
    split.run_clean(bounded)
    split.set_target(None)
    del seen[:]
    assert split.run_draws(bounded)
    assert bounded.chunking["chunk"] == 2 and bounded.chunking["split"] >= 1
    assert {int(x.shape[0]) for x, _t in seen} == {4}
    assert split.losses == pytest.approx(whole.losses, rel=1e-6)
    assert split.clean_loss == pytest.approx(whole.clean_loss, rel=1e-6)
    assert split.accept_before == pytest.approx(whole.accept_before, rel=1e-6)


def test_a_write_load_reserves_its_working_memory(monkeypatch):
    from types import SimpleNamespace

    from kilora.comfyui import patching

    seen = []

    def reserve(device, reserve_bytes, *, phase, mm=None, keep=()):
        seen.append((device, reserve_bytes, phase, list(keep)))
        return 0

    monkeypatch.setattr(patching, "reserve_working_memory", reserve)
    patcher = SimpleNamespace(load_device="cuda:0")
    handle = SimpleNamespace(patcher=patcher)
    live_write._reserve_write_memory(handle, 8)
    assert seen == [("cuda:0", 8 * live_write.WRITE_ROW_BYTES, "live_write", [patcher])]
    seen.clear()
    monkeypatch.setenv(live_write.RESIDENT_ENV, "0")
    live_write._reserve_write_memory(handle, 8)
    assert seen == [("cuda:0", 8 * live_write.WRITE_ROW_BYTES, "live_write", [])]


def test_the_call_split_sums_to_the_call_time(tmp_path, mm, monkeypatch):
    monkeypatch.setenv(live_write.PROFILE_ENV, "1")
    import time

    session = _imported(tmp_path)
    handle = _handle(session, [])
    model = handle.apply_model

    def working_model(*args, **kwargs):
        time.sleep(0.02)
        return model(*args, **kwargs)

    handle.apply_model = working_model
    measure = image_loss(handle, _x0(), EVENT)
    with measure.hold():
        _write(session, measure, _estimate(session))
    profile = measure.profile
    split = profile.split
    calls = sum(seconds for _rows, seconds in profile.calls)
    assert split.calls == len(profile.calls) == 10
    assert split.total == pytest.approx(calls)
    assert abs(split.sum() - calls) <= 0.1 * calls
    assert "call split (mean of 10 calls, s)" in profile.line()


def test_the_call_split_marks_the_layers_and_the_depth_steps():
    import time

    from kilora import call_profile

    def chain():
        time.sleep(0.01)
        call_profile.tick("prep")
        for k in (1, 2):
            time.sleep(0.01)
            call_profile.tick("route", k)
            time.sleep(0.02)
            call_profile.tick("update", k)
            time.sleep(0.005)
            call_profile.tick("stats", k)
            time.sleep(0.005)
            call_profile.tick("gate", k)
        time.sleep(0.01)

    def layer():
        time.sleep(0.01)
        call_profile.note_walks(4)
        return call_profile.run_chain(chain)

    profile = call_profile.CallProfile()
    assert not call_profile.active()
    start = time.perf_counter()
    with call_profile.profiling(profile):
        assert call_profile.active()
        time.sleep(0.02)
        call_profile.run_layer(layer)
        time.sleep(0.02)
    total = time.perf_counter() - start
    assert not call_profile.active()
    profile.total += total
    assert profile.calls == 1
    assert abs(profile.sum() - total) <= 0.1 * total
    assert profile.part("base") >= 0.04
    assert profile.part("layer") >= 0.01
    assert profile.part("update") >= 0.04
    assert profile.part("output") >= 0.01
    assert set(profile.steps()) == {1, 2}
    assert profile.steps()[1] >= 0.04
    assert profile.walks == {4: 1}
    line = profile.line()
    assert line.startswith("call split (mean of 1 calls, s): total ")
    assert "depth steps 1: " in line and "fused chains 1 x 4 stacked walks" in line


def test_a_mark_without_a_profile_does_nothing():
    from kilora import call_profile

    call_profile.tick("route", 1)
    call_profile.note_walks(2)
    assert not call_profile.active()
    assert not call_profile.outside_chain()


class ResidentManagement(types.ModuleType):
    def __init__(self, free):
        super().__init__("comfy.model_management")
        self.free, self.loads, self.frees = free, [], []
        self.current_loaded_models = []

    def load_models_gpu(self, models, *args, **kwargs):
        self.loads.append(dict(kwargs))
        for patcher in models:
            patcher.loaded = patcher.size if kwargs.get("force_full_load") else patcher.size // 2
            if not any(item.model is patcher for item in self.current_loaded_models):
                self.current_loaded_models.append(types.SimpleNamespace(model=patcher))

    def free_memory(self, memory_required, device, keep_loaded=()):
        self.frees.append((memory_required, [item.model for item in keep_loaded]))

    def get_free_memory(self, device=None):
        return self.free


class SizedPatcher:
    def __init__(self, size, device="cpu"):
        self.size, self.loaded, self.load_device = size, 0, device

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.loaded


@pytest.fixture
def sized_mm(monkeypatch):
    def make(free):
        fake = ResidentManagement(free)
        package = types.ModuleType("comfy")
        package.__path__ = []
        package.model_management = fake
        monkeypatch.setitem(sys.modules, "comfy", package)
        monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
        return fake

    return make


def _sized_handle(session, patcher):
    handle = _handle(session, [])
    handle.patcher = patcher
    return handle


def test_a_write_keeps_its_weights_resident_when_they_fit(tmp_path, sized_mm):
    mm = sized_mm(free=64 << 30)
    session = _imported(tmp_path)
    patcher = SizedPatcher(8 << 30)
    measure = image_loss(_sized_handle(session, patcher), _x0(), EVENT)
    estimate = _estimate(session)
    session.learning = True
    commit_write(session, WritingRule(session, estimate), _record(measure))
    assert mm.loads == [{"force_full_load": True}]
    assert patcher.loaded == patcher.size


def test_a_write_loads_as_before_when_its_weights_do_not_fit(tmp_path, sized_mm):
    mm = sized_mm(free=1 << 30)
    session = _imported(tmp_path)
    patcher = SizedPatcher(8 << 30)
    measure = image_loss(_sized_handle(session, patcher), _x0(), EVENT)
    estimate = _estimate(session)
    session.learning = True
    commit_write(session, WritingRule(session, estimate), _record(measure))
    assert mm.loads == [{}]


def test_resident_weights_can_be_turned_off(tmp_path, sized_mm, monkeypatch):
    monkeypatch.setenv(live_write.RESIDENT_ENV, "0")
    mm = sized_mm(free=64 << 30)
    session = _imported(tmp_path)
    measure = image_loss(_sized_handle(session, SizedPatcher(8 << 30)), _x0(), EVENT)
    estimate = _estimate(session)
    session.learning = True
    commit_write(session, WritingRule(session, estimate), _record(measure))
    assert mm.loads == [{}]


def test_resident_fits_counts_the_memory_of_the_other_models(sized_mm):
    mm = sized_mm(free=2 << 30)
    patcher = SizedPatcher(4 << 30)
    reserve = 1 << 30
    assert not live_write.resident_fits(mm, patcher, reserve)
    other = SizedPatcher(4 << 30)
    other.loaded = 4 << 30
    mm.current_loaded_models.append(types.SimpleNamespace(model=other))
    assert live_write.resident_fits(mm, patcher, reserve)
    patcher.loaded = 3 << 30
    mm.current_loaded_models.clear()
    assert live_write.resident_fits(mm, patcher, reserve)
    assert not live_write.resident_fits(mm, object(), reserve)


def test_the_reserve_keeps_the_given_model_loaded(sized_mm):
    from kilora.comfyui.patching import reserve_working_memory

    mm = sized_mm(free=3 << 30)
    ours, other = SizedPatcher(1), SizedPatcher(1)
    mm.current_loaded_models += [types.SimpleNamespace(model=m) for m in (other, ours)]
    reserve_working_memory("cuda:0", 1 << 30, phase="live_write", mm=mm, keep=[ours])
    assert mm.frees == [(1 << 30, [ours])]

    class Old(ResidentManagement):
        def free_memory(self, memory_required, device):
            self.frees.append((memory_required, None))

    old = Old(3 << 30)
    old.current_loaded_models = list(mm.current_loaded_models)
    reserve_working_memory("cuda:0", 1 << 30, phase="live_write", mm=old, keep=[ours])
    assert old.frees == [(1 << 30, None)]


def test_the_dense_skip_table_gives_the_mask_of_the_loop(monkeypatch):
    from kilora import segments

    experts = 5
    gen = torch.Generator().manual_seed(4)
    skips = {
        (a, b): torch.rand(experts, generator=gen) > 0.5
        for a in range(experts)
        for b in range(experts)
        if (a + b) % 2
    }

    def routes(n):
        picks = torch.randint(0, experts, (2, n), generator=gen)
        return torch.nn.functional.one_hot(picks, experts).float()

    previous, current, following = routes(64), routes(64), routes(64)
    following[0, :3] = 0.5
    fast = segments.skip_mask(skips, previous, current, following)
    monkeypatch.setenv("KILORA_SKIP_TABLE", "0")
    loop = segments.skip_mask(skips, previous, current, following)
    assert fast.dtype == loop.dtype == torch.bool
    assert torch.equal(fast, loop) and bool(loop.any())
    first = segments._dense_table(skips, experts, previous.device)
    assert segments._dense_table(skips, experts, previous.device) is first
    assert segments.has_skips(skips) is True
    assert segments._SKIP_WORK[id(skips)][1] is True
    assert segments.has_skips({(0, 1): torch.zeros(experts, dtype=torch.bool)}) is False


def test_the_gram_is_made_once_per_weight_version():
    from kilora import compute

    up = torch.randn(6, 9, generator=torch.Generator().manual_seed(2)).to(torch.float16)
    with torch.no_grad():
        first = compute.up_gram(up)
        assert compute.up_gram(up) is first
        u = up.detach().float()
        assert torch.equal(first, u @ u.t())
        up.mul_(2)
        again = compute.up_gram(up)
        assert again is not first
        u = up.detach().float()
        assert torch.equal(again, u @ u.t())
    with torch.enable_grad():
        assert compute.up_gram(up) is not compute.up_gram(up)


def _scoped_handle(session, seen):
    from kilora import call_values, compute

    handle = _handle(session, [])
    plain = handle.apply_model

    def apply_model(x_t, t, c_crossattn):
        seen.append(
            (
                int(x_t.shape[0]),
                compute._DERIVED_CARRY[-1] if compute._DERIVED_CARRY else None,
                len(call_values._HOST_SCALARS),
            )
        )
        return plain(x_t, t, c_crossattn)

    handle.apply_model = apply_model
    return handle


def test_the_write_calls_run_in_the_call_scope_once_per_call(tmp_path, monkeypatch):
    entered = []
    real = live_write.write_call_scope

    def counted():
        entered.append(1)
        return real()

    monkeypatch.setattr(live_write, "write_call_scope", counted)
    session = _imported(tmp_path)
    seen = []
    measure = image_loss(_scoped_handle(session, seen), _x0(), EVENT)
    estimate = _estimate(session)
    with measure.hold():
        _write(session, measure, estimate)
    calls = measure.calls[0]
    assert len(entered) == len(seen) == calls
    assert sum(rows for rows, _carry, _host in seen) > calls
    carries = [carry for _rows, carry, _host in seen]
    assert all(carry is not None for carry in carries)
    assert len({id(carry) for carry in carries}) == calls
    assert all(host == 1 for _rows, _carry, host in seen)
    monkeypatch.setattr(live_write, "write_call_scope", real)
    monkeypatch.setenv(live_write.CALL_SCOPE_ENV, "0")
    plain_seen = []
    plain = image_loss(_scoped_handle(session, plain_seen), _x0(), EVENT)
    plain_estimate = _estimate(session)
    with plain.hold():
        _write(session, plain, plain_estimate)
    assert all(carry is None and host == 0 for _rows, carry, host in plain_seen)
    assert plain_estimate.clean_loss == estimate.clean_loss
    assert plain_estimate.losses == estimate.losses
