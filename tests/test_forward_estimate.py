# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_edit_dimensions import _session

from kilora import live_experts
from kilora.forward_estimate import ForwardEstimate, supports
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor, ImportTrial
from kilora.path_merge import KeepMeans, mean_keeps


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _outputs(session, state, x):
    out = {}
    for target, ex in session._source.experts.items():
        parts = state.get(target) or {}
        y = None
        for e in range(int(ex.up.shape[0])):
            if f"core_{e}" not in parts:
                continue
            core = parts[f"core_{e}"].float()
            rank = int(core.shape[0])
            c = x[target] @ ex.down[e, 0, :rank].float().T
            part = float(parts[f"strength_{e}"]) * (c @ core.T) @ ex.up[e, 0, :rank].float()
            y = part if y is None else y + part
        if y is not None:
            out[target] = y
    return out


class LinearMeasure:
    def __init__(self, session, tokens=4, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.session = session
        self.state = session.state
        self.x, self.w = {}, {}
        for target, ex in session._source.experts.items():
            width_in, width_out = int(ex.down.shape[-1]), int(ex.up.shape[-1])
            self.x[target] = torch.randn(tokens, width_in, generator=gen)
            self.w[target] = torch.randn(tokens, width_out, generator=gen)
        self.base = _outputs(session, self.state, self.x)
        self.calls = 0
        self.batched = 0

    def _loss(self, outputs):
        return sum(float((self.w[t] * y).sum()) for t, y in outputs.items())

    def __call__(self, trial):
        self.calls += 1
        outputs = dict(self.base)
        if isinstance(trial, ImportTrial):
            for target, du in trial.du.items():
                ex = self.session._source.experts[target]
                down = ex.down.reshape(-1, ex.down.shape[-1]).float()
                outputs[target] = outputs[target] + (self.x[target] @ down.T) @ du.float()
        return self._loss(outputs)

    def copies(self, trial, copies):
        self.batched += 1
        losses = [0.0] * copies
        trial.begin_sample(0)
        for target, y in self.base.items():
            x = self.x[target].repeat(copies, 1)
            noise = trial.input_delta(target, x)
            for j in range(copies):
                part = y if noise is None else y + noise[j * y.shape[0] : (j + 1) * y.shape[0]]
                losses[j] += float((self.w[target] * part).sum())
        return losses


def _true_gradients(session, measure):
    out = {}
    for target, ex in session._source.experts.items():
        parts = session.state.get(target) or {}
        for e in range(int(ex.up.shape[0])):
            if f"core_{e}" not in parts:
                continue
            core = parts[f"core_{e}"].float()
            rank = int(core.shape[0])
            strength = float(parts[f"strength_{e}"])
            c = measure.x[target] @ ex.down[e, 0, :rank].float().T
            u = measure.w[target] @ ex.up[e, 0, :rank].float().T
            out[(target, e)] = (strength * u.T @ c, float((u * (c @ core.T)).sum()))
    return out


def _estimate(session, measure, draws, chunk, seed=7, sigma=0.05):
    estimate = ForwardEstimate(
        session, session.state, draws=draws, sigma=sigma, chunk=chunk, seed=seed
    )
    estimate.run_clean(measure)
    estimate.set_target(None)
    assert estimate.run_draws(measure)
    return estimate


def _cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def test_settings_default_to_the_forward_estimate(tmp_path):
    session = _session(tmp_path)
    assert session.write_measure == "forward"
    assert (session.forward_draws, session.forward_sigma, session.forward_chunk) == (4, 0.05, 8)


def test_estimate_follows_the_true_gradient(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    measure = LinearMeasure(session)
    estimate = _estimate(session, measure, draws=8192, chunk=1024)
    grads = estimate.gradients()
    truth = _true_gradients(session, measure)
    assert grads and set(grads) == set(truth)
    for key, (core, _strength) in truth.items():
        assert _cosine(grads[key][0], core) > 0.9
    keys = sorted(truth)
    estimated = torch.tensor([grads[k][1] for k in keys])
    exact = torch.tensor([truth[k][1] for k in keys])
    assert _cosine(estimated, exact) > 0.9
    assert measure.batched == 1 + 8192 // 1024
    assert estimate.forwards == measure.batched


def test_chunks_do_not_change_the_estimate(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    measure = LinearMeasure(session)
    one = _estimate(session, measure, draws=12, chunk=1).gradients()
    many = _estimate(session, measure, draws=12, chunk=5).gradients()
    again = _estimate(session, measure, draws=12, chunk=5).gradients()
    for key in one:
        torch.testing.assert_close(one[key][0], many[key][0], rtol=1e-5, atol=1e-7)
        assert one[key][1] == pytest.approx(many[key][1], rel=1e-5, abs=1e-9)
        assert torch.equal(many[key][0], again[key][0])
    other = _estimate(session, measure, draws=12, chunk=5, seed=8).gradients()
    assert any(not torch.equal(other[k][0], many[k][0]) for k in many)


def test_noise_stays_on_the_write_target(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    measure = LinearMeasure(session)
    first = sorted(measure.base)[0]
    estimate = ForwardEstimate(session, session.state, draws=4, sigma=0.05, chunk=2, seed=1)
    estimate.run_clean(measure)
    estimate.set_target({first: [0]})
    assert estimate.run_draws(measure)
    assert set(estimate.gradients()) == {(first, 0)}


def test_slopes_are_directional_derivatives(tmp_path):
    session = _session(tmp_path)
    session.align_for_moves()
    measure = LinearMeasure(session)
    estimate = _estimate(session, measure, draws=8, chunk=4)
    grads = estimate.gradients()
    directions = session.moves()
    slopes = estimate.slopes(directions)
    for move in directions:
        grad = grads.get((move.target, move.expert))
        if grad is None:
            assert slopes[move.id] == 0.0
        elif move.part == "core":
            expected = float((grad[0] * move.direction.double()).sum())
            assert slopes[move.id] == pytest.approx(expected)
        else:
            assert slopes[move.id] == pytest.approx(grad[1] * float(move.direction.sum()))


class _Record:
    def __init__(self, measure):
        self.measure = measure
        self.stats = None
        self.guard_score = lambda _trial: {"gap": 1.0, "spread": 1.0}


def test_write_runs_one_clean_pass_and_the_batched_draws(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=1, forward_draws=512, forward_chunk=256)
    editor = ImportEditor(session)
    measure = LinearMeasure(session)
    assert supports(measure)

    def no_routes(_session, record):
        import contextlib

        return contextlib.nullcontext(record)

    monkeypatch.setattr(live_experts, "recording_routes", no_routes)
    new = editor._next_unmerged(session.state, _Record(measure))
    assert measure.batched == 3
    assert "Forward estimate: 512 draws, 2 batched passes" in editor.note
    forwards = measure.batched + measure.calls
    assert f"model calls: {forwards}," in editor.note
    assert "write time" in editor.note
    if new is not None:
        assert measure._loss(_outputs(session, new, measure.x)) < measure._loss(measure.base)


def test_candidates_mode_keeps_the_paired_forwards(tmp_path):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=1, write_measure="candidates", measure="full")
    editor = ImportEditor(session)
    measure = LinearMeasure(session)
    editor._next_unmerged(session.state, _Record(measure))
    assert measure.batched == 0 and measure.calls > 0
    assert "Forward estimate" not in editor.note


def test_keep_means_do_not_grow_with_the_step_count():
    experts, rows = 2, 3
    gen = torch.Generator().manual_seed(0)
    recorder = KeepMeans(experts)
    raw = {}
    sizes = []
    for step in range(64):
        values = torch.rand(5, experts, rows, generator=gen)
        key = ("layer", step % 4)
        recorder.add(key, values)
        raw.setdefault(key, []).append(values)
        sizes.append(recorder.nbytes())
    assert len(set(sizes[4:])) == 1
    reduced = mean_keeps(recorder, experts)
    full = mean_keeps(raw, experts)
    assert set(reduced) == set(full)
    for key, values in full.items():
        torch.testing.assert_close(reduced[key][0], torch.stack(values).mean(0))


def test_candidates_record_the_paths_on_the_first_measured_trial(tmp_path, monkeypatch):
    import contextlib

    from test_live_experts import _counting

    session = _session(tmp_path)
    session.set_options(measure="full", core_dims=1, write_measure="candidates")
    editor = ImportEditor(session)
    recorded = []

    @contextlib.contextmanager
    def first_trial(_session, record):
        for target in session.state:
            record.mass[target] = torch.tensor([1.0, 0.0])
            record.ran.add(target)
        recorded.append(1)
        yield record

    monkeypatch.setattr(live_experts, "recording_routes", first_trial)
    seen = _counting(monkeypatch)
    calls = []

    def loss(_trial):
        calls.append(1)
        return float(len(calls)) * 1e-3

    editor._next_unmerged(session.state, _Record(loss))
    assert recorded == [1]
    measured = seen[: len(calls)]
    assert calls and all(measured)
    first = measured[0]
    later = [pairs for pairs in measured[2:] if pairs != first]
    assert later and all(expert == 0 for pairs in later for _target, expert in pairs)
    assert editor._target is not None
    assert all(experts == [0] for experts in editor._target.values())


class FusedStatsMeasure(LinearMeasure):
    def __init__(self, session, shares):
        super().__init__(session)
        from types import SimpleNamespace

        source = session._source
        self.layer = SimpleNamespace(key=source.extra.get("live_id", source.name))
        self.shares = shares
        self.noted = {}

    def copies(self, trial, copies):
        from kilora import compute

        assert not (compute._ROUTES or compute._SHARES or compute._SET_SHARES)
        assert not (compute._SET_SHARE_SCOPES or compute._READ_KEEP_RAW_OBSERVERS)
        assert not compute._READ_KEEP_DETAIL_OBSERVERS
        for target, calls in self.shares.items():
            for gates in calls:
                compute._note_fused_mass(target, [gates], [self.layer])
                self.noted.setdefault(target, []).append(gates.mean(0))
        return super().copies(trial, copies)


def test_a_write_records_no_eager_routes_and_takes_the_fused_mass(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=1, forward_draws=8, forward_chunk=4)
    experts = session._source.num_experts
    gen = torch.Generator().manual_seed(5)
    shares = {}
    for target in session.state:
        calls = []
        for _ in range(3):
            gates = torch.rand(6, experts, generator=gen)
            gates[:, 0] += 4.0
            calls.append(gates / gates.sum(-1, keepdim=True))
        shares[target] = calls

    def eager(*_args, **_kwargs):
        raise AssertionError("a write must not record routes on the eager path")

    monkeypatch.setattr(live_experts, "recording_routes", eager)
    monkeypatch.setattr(live_experts, "measure_routes", eager)
    measure = FusedStatsMeasure(session, shares)
    measure.chunking = {"chunk": 2, "used": 2, "split": 1}
    editor = ImportEditor(session)
    editor._next_unmerged(session.state, _Record(measure))
    assert "2 samples per model call (halved 1 times for memory)" in editor.note
    routes = editor._routes
    assert routes is not None and set(routes.mass) == set(session.state)
    for target in session.state:
        expected = torch.stack(measure.noted[target][:3]).mean(0)
        assert torch.allclose(routes.mass[target], expected, atol=1e-6)
        assert 0 in editor._target.get(target, ())
        assert routes.keeps[target].shape[0] == experts


def test_the_fused_mass_observer_keeps_walk_steps_out():
    from types import SimpleNamespace

    from kilora import compute
    from kilora.live_experts import RouteRecord, recording_fused_routes

    source = SimpleNamespace(extra={}, name="set")
    session = SimpleNamespace(_source=source, state={"layer": {}})
    layer = SimpleNamespace(key="set")
    step = torch.tensor([[0.75, 0.25]])
    final = torch.tensor([[0.25, 0.75]])
    record = RouteRecord()
    with recording_fused_routes(session, record):
        compute._note_fused_mass("layer", [step], [layer])
        with compute._SET_SHARE_HOLD.pushed(True):
            compute._note_fused_mass("layer", [step], [layer])
            compute._note_fused_mass("layer", [final], [layer], walks=True)
        compute._note_fused_mass("other", [step], [layer])
    assert set(record.mass) == {"layer"}
    assert torch.allclose(record.mass["layer"], (step[0] + final[0]) / 2)
    assert record.ran == {"layer"}


def test_the_read_keeps_are_observed_on_the_clean_pass_only(tmp_path, monkeypatch):
    from kilora import compute
    from kilora.forward_estimate import ForwardEstimate

    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=1, forward_draws=8, forward_chunk=8)
    editor = ImportEditor(session)
    editor.configure({"path_merge": True})
    measure = LinearMeasure(session)
    seen = {}
    run_clean, run_draws = ForwardEstimate.run_clean, ForwardEstimate.run_draws

    def clean(self, m):
        seen["clean"] = bool(compute._READ_KEEP_RAW_OBSERVERS)
        return run_clean(self, m)

    def draws(self, m):
        seen["draws"] = bool(compute._READ_KEEP_RAW_OBSERVERS)
        return run_draws(self, m)

    def no_routes(_session, record):
        import contextlib

        return contextlib.nullcontext(record)

    monkeypatch.setattr(live_experts, "recording_routes", no_routes)
    monkeypatch.setattr(ForwardEstimate, "run_clean", clean)
    monkeypatch.setattr(ForwardEstimate, "run_draws", draws)
    editor._stage_writes = True
    editor.next_state(session.state, _Record(measure))
    assert seen == {"clean": True, "draws": False}
    assert editor._keep_observer is None


def _in_pack(t, estimate, kind):
    from kilora.device_pack import shares_buffer

    return shares_buffer(t, estimate.packs[kind])


def test_the_kept_small_tensors_live_in_one_buffer_per_kind(tmp_path):
    session = _session(tmp_path)
    measure = LinearMeasure(session)
    estimate = ForwardEstimate(session, session.state, draws=8, sigma=0.05, chunk=4, seed=7)
    estimate.run_clean(measure)
    assert estimate._device_parts and estimate._squares and estimate._live_rows
    for core, _strength, down, up in estimate._device_parts.values():
        for t in (core, down, up):
            assert _in_pack(t, estimate, "parts")
    for entries in estimate._squares.values():
        for cell, _count in entries.values():
            assert _in_pack(cell, estimate, "squares")
    for live in estimate._live_rows.values():
        assert _in_pack(live, estimate, "live")
    for kind in ("parts", "squares", "live"):
        assert len(estimate.packs[kind].buffers()) == 1
    estimate.set_target(None)
    target, expert = next((t, e) for t, es in estimate.target.items() for e in es)
    core_sums, strength_sums = estimate._sums_on(target, expert, torch.device("cpu"))
    assert _in_pack(core_sums, estimate, "sums") and _in_pack(strength_sums, estimate, "sums")
    assert len(estimate.packs["sums"].buffers()) == 1


def test_a_clean_call_that_runs_again_writes_its_cell_in_place(tmp_path):
    session = _session(tmp_path)
    measure = LinearMeasure(session)
    estimate = ForwardEstimate(session, session.state, draws=4, sigma=0.05, chunk=4, seed=7)
    estimate.run_clean(measure)
    before = {k: dict(v) for k, v in estimate._squares.items()}
    used = {kind: [b.data_ptr() for b in p.buffers()] for kind, p in estimate.packs.items()}
    estimate.run_clean(measure)
    for key, entries in estimate._squares.items():
        for call, (cell, count) in entries.items():
            old_cell, old_count = before[key][call]
            assert cell.data_ptr() == old_cell.data_ptr() and count == old_count
    for kind in ("squares", "live"):
        assert [b.data_ptr() for b in estimate.packs[kind].buffers()] == used[kind]


def test_the_packed_estimate_equals_the_unpacked_one(tmp_path, monkeypatch):
    from kilora import device_pack

    session = _session(tmp_path)
    measure = LinearMeasure(session)
    packed = _estimate(session, measure, draws=16, chunk=4)
    monkeypatch.setattr(device_pack, "PACK", False)
    plain = _estimate(session, measure, draws=16, chunk=4)
    assert packed.clean_loss == plain.clean_loss
    assert packed.losses == plain.losses
    assert packed.noise == plain.noise
    a, b = packed.gradients(), plain.gradients()
    assert a.keys() == b.keys() and a
    for key in a:
        assert torch.equal(a[key][0], b[key][0])
        assert a[key][1] == b[key][1]


def test_the_route_mass_sums_add_in_place_in_one_buffer(monkeypatch):
    from types import SimpleNamespace

    from kilora import compute, device_pack
    from kilora.live_experts import RouteRecord, recording_fused_routes

    packs = []

    class Spy(device_pack.DevicePack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            packs.append(self)

    monkeypatch.setattr(device_pack, "DevicePack", Spy)
    source = SimpleNamespace(extra={}, name="set")
    session = SimpleNamespace(_source=source, state={"a": {}, "b": {}})
    layer = SimpleNamespace(key="set")
    gen = torch.Generator().manual_seed(3)
    noted = {"a": [], "b": []}
    record = RouteRecord()
    with recording_fused_routes(session, record):
        for _call in range(4):
            for target in ("a", "b"):
                gates = torch.rand(5, 3, generator=gen)
                noted[target].append(gates.mean(0))
                compute._note_fused_mass(target, [gates], [layer])
    assert len(packs) == 1 and len(packs[0].buffers()) == 1
    for target in ("a", "b"):
        expected = torch.stack(noted[target]).mean(0)
        assert torch.allclose(record.mass[target], expected, atol=1e-6)


class _PassMeasure(FusedStatsMeasure):
    def __init__(self, session, shares):
        super().__init__(session, shares)
        self.observed = []

    def copies(self, trial, copies):
        from kilora import compute

        on = bool(compute._FUSED_MASS)
        self.observed.append((getattr(trial, "mode", None), on))
        if not on:
            return LinearMeasure.copies(self, trial, copies)
        return super().copies(trial, copies)


def test_the_route_mass_is_recorded_in_the_clean_pass_only(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.learning = True
    session.set_options(core_dims=1, forward_draws=8, forward_chunk=4)
    experts = session._source.num_experts
    gen = torch.Generator().manual_seed(9)
    shares = {}
    for target in session.state:
        gates = torch.rand(6, experts, generator=gen)
        gates[:, 0] += 4.0
        shares[target] = [gates / gates.sum(-1, keepdim=True)]

    def eager(*_args, **_kwargs):
        raise AssertionError("a write must not record routes on the eager path")

    monkeypatch.setattr(live_experts, "recording_routes", eager)
    monkeypatch.setattr(live_experts, "measure_routes", eager)
    measure = _PassMeasure(session, shares)
    editor = ImportEditor(session)
    editor._next_unmerged(session.state, _Record(measure))
    observed = [on for _mode, on in measure.observed]
    assert observed[0] is True
    assert observed.count(True) == 1
    assert len(observed) > 1
    routes = editor._routes
    assert routes is not None
    for target in session.state:
        expected = torch.stack(measure.noted[target]).mean(0)
        assert len(measure.noted[target]) == 1
        assert torch.allclose(routes.mass[target], expected, atol=1e-6)
