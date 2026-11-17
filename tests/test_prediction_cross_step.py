# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest
import torch
from test_lowrank_depth import _sets

from kilora.bake import ModulePlan, TargetPlan
from kilora.compute import sampling_progress
from kilora.recording import (
    CrossFrame,
    CrossStepRecorder,
    PathRecorder,
    noise_weight,
    orthonormal_rows,
    prediction_counts,
    prediction_note,
    prediction_report,
)

SIGMAS = (3.0, 2.0, 0.5)


def _hand(frame, xs, outs, sigmas, weight):
    p_in, p_out = frame.p_in.double(), frame.p_out.double()
    jhat = p_out @ weight.double() @ p_in.t()
    if frame.up is not None:
        jhat = jhat + p_out @ frame.up.double() @ frame.down.double() @ p_in.t()
    z = [p_in @ x.double().reshape(-1, x.shape[-1]).mean(0) for x in xs]
    y = [p_out @ o.double().reshape(-1, o.shape[-1]).mean(0) for o in outs]
    out = []
    for t in range(1, len(xs)):
        predicted = y[t - 1] + jhat @ (z[t] - z[t - 1])
        error = (y[t] - predicted) / (sigmas[t - 1] - sigmas[t])
        cos = torch.dot(z[t], z[t - 1]) / (z[t].norm() * z[t - 1].norm())
        precision = (1 / (1 + sigmas[t] ** 2)) * (1 - cos.abs())
        out.append((z[t], y[t], predicted, error, precision))
    return out


def test_linear_toy_matches_the_definition():
    gen = torch.Generator().manual_seed(0)
    width_in, width_out, rank = 6, 5, 3
    weight = torch.randn(width_out, width_in, generator=gen)
    up = torch.randn(width_out, 2, generator=gen) * 0.1
    down = torch.randn(2, width_in, generator=gen) * 0.1
    p_in = torch.randn(rank, width_in, generator=gen)
    p_in = p_in / p_in.norm(dim=1, keepdim=True)
    p_out = orthonormal_rows(torch.randn(width_out, rank, generator=gen))
    frame = CrossFrame(p_in, p_out, "input", weight, up, down)
    recorder = CrossStepRecorder({"mount": [frame]})
    xs = [torch.randn(2, 4, width_in, generator=gen) for _ in SIGMAS]
    outs = [x @ (weight + up @ down).t() + 0.05 * torch.randn(2, 4, width_out) for x in xs]
    with recorder.recording():
        for i, (sigma, x, o) in enumerate(zip(SIGMAS, xs, outs, strict=True)):
            recorder.set_sigma(sigma)
            with sampling_progress(i / 2):
                recorder.observe_output("mount", x, None, o)
    errors = recorder.finish().cross_errors["mount"]
    assert [entry.step for entry in errors] == [2, 3]
    for entry, (z, y, predicted, error, precision) in zip(
        errors, _hand(frame, xs, outs, SIGMAS, weight), strict=True
    ):
        assert torch.allclose(entry.z, z, atol=1e-10)
        assert torch.allclose(entry.y, y, atol=1e-10)
        assert torch.allclose(entry.predicted, predicted, atol=1e-10)
        assert torch.allclose(entry.error, error, atol=1e-10)
        assert torch.allclose(entry.precision.double(), precision, atol=1e-6)
    assert noise_weight(2.0) == pytest.approx(0.2)


def test_without_a_sigma_nothing_is_recorded():
    frame = CrossFrame(torch.eye(3), torch.eye(3), "input")
    recorder = CrossStepRecorder({"mount": [frame]})
    for _ in range(3):
        recorder.observe_output("mount", torch.randn(1, 2, 3), None, torch.randn(1, 2, 3))
    assert not recorder.finish().cross_errors


def _plan(layer):
    return ModulePlan([(0, -1, TargetPlan(sets=[layer], depth=1, name="mount"))])


def _run(plan, xs, weight, recorder=None):
    outs = []
    with torch.no_grad():
        for i, (sigma, x) in enumerate(zip(SIGMAS, xs, strict=True)):
            h = x @ weight.t()
            with sampling_progress(i / 2):
                if recorder is None:
                    outs.append(plan.run(x, h))
                    continue
                recorder.set_sigma(sigma)
                with recorder.recording():
                    outs.append(plan.run(x, h))
    return outs


def test_depth_one_records_through_the_host_hook_and_outputs_are_equal():
    layer = _sets(1, 8, 1, 3, 4, seed=1)[0]
    gen = torch.Generator().manual_seed(3)
    weight = torch.randn(8, 8, generator=gen) / 8**0.5
    xs = [torch.randn(2, 5, 8, generator=gen) for _ in SIGMAS]
    p_in = torch.randn(3, 8, generator=gen)
    p_in = p_in / p_in.norm(dim=1, keepdim=True)
    frame = CrossFrame(p_in, orthonormal_rows(torch.randn(8, 3, generator=gen)), "input", weight)
    expected = _run(_plan(layer), xs, weight)
    v0 = torch.eye(3)
    recorder = PathRecorder(
        {"mount": layer.down}, {"mount": v0}, cross=CrossStepRecorder({"mount": [frame]})
    )
    actual = _run(_plan(layer), xs, weight, recorder)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    stats = recorder.finish()
    assert not stats.prediction_errors
    errors = stats.cross_errors["mount"]
    assert [entry.step for entry in errors] == [2, 3]
    for entry, (_, _, _, error, precision) in zip(
        errors, _hand(frame, xs, expected, SIGMAS, weight), strict=True
    ):
        assert torch.allclose(entry.error, error, atol=1e-8)
        assert torch.allclose(entry.precision.double(), precision, atol=1e-6)


def test_counts_keep_the_old_fields_as_sums(monkeypatch):
    depth = {
        "a": [SimpleNamespace(progress=p, node=n) for p in (0.5, 0.75) for n in (0, 1)],
    }
    cross = {
        "a": [SimpleNamespace(step=s, node=0) for s in (2, 3, 4)],
        "b": [SimpleNamespace(step=2, node=0)],
    }
    monkeypatch.setenv("KILORA_PREDICTION_DEBUG", "1")
    counts = prediction_counts(depth, cross)
    assert counts["eps_depth_steps"] == 2 and counts["eps_depth_nodes"] == 2
    assert counts["eps_cross_steps"] == 3 and counts["eps_cross_nodes"] == 2
    assert counts["eps_steps"] == counts["eps_depth_steps"] + counts["eps_cross_steps"]
    assert counts["eps_nodes"] == counts["eps_depth_nodes"] + counts["eps_cross_nodes"]
    assert "eps_steps=5 eps_nodes=4" in prediction_note(depth, cross)
    assert prediction_note({}, {}) == ""


def test_debug_report_names_sources_and_top_mounts():
    entry = lambda value, pi: SimpleNamespace(
        error=torch.full((3,), value), precision=torch.tensor(pi)
    )
    cross = {"a": [entry(1.0, 0.5)], "b": [entry(2.0, 0.5)], "c": [entry(0.1, 1.0)]}
    line = prediction_report({}, cross)
    assert "depth: count=0" in line and "cross: count=3" in line
    assert line.index("b:") < line.index("a:") < line.index("c:")


def test_chunk_style_recording_off_adds_no_hook():
    recorder = PathRecorder({"mount": torch.eye(3)}, {"mount": torch.eye(3)})
    assert not recorder.wants_output("mount")


@pytest.fixture
def imported(tmp_path, monkeypatch):
    from test_import_modes import LAYERS, Toy, _lora

    from kilora.live import SESSIONS
    from kilora.live_imports import imported_session_for
    from kilora.multi_import import import_loras

    SESSIONS.clear()
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    sources = [
        dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            path=str(p),
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")
    yield session, model
    SESSIONS.clear()


def _weights(session, model):
    out = {}
    for target in session.state:
        try:
            out[target] = model.get_submodule(target).weight
        except AttributeError:
            continue
    return out


def test_imported_frames_come_from_the_source_directions(imported):
    from kilora.live_imports import ImportEditor, import_source_frames

    session, model = imported
    session.align_for_moves()
    frames = import_source_frames(session, _weights(session, model))
    assert frames
    for target, items in frames.items():
        ex = session._source.experts[target]
        weight = model.get_submodule(target).weight.double()
        for expert, p_in, p_out, base in items:
            rank = p_in.shape[0]
            a = ex.down[expert, 0, :rank].double()
            b = ex.up[expert, 0, :rank].double().T
            assert torch.allclose(p_in, a / a.norm(dim=1, keepdim=True))
            assert torch.allclose(p_out @ p_out.t(), torch.eye(rank, dtype=torch.float64))
            assert torch.allclose(p_out.t() @ (p_out @ b), b, atol=1e-8)
            assert torch.allclose(base, p_out @ weight @ p_in.t(), atol=1e-8)
    assert import_source_frames(session) is frames
    recorder = ImportEditor(session).recorder(session)
    assert isinstance(recorder, CrossStepRecorder)
    assert set(recorder.frames) == set(frames)
    session.record_prediction = False
    assert ImportEditor(session).recorder(session) is None


class _OutputProbe:
    def __init__(self):
        self.calls = []

    def wants(self, target):
        return False

    def wants_output(self, target):
        return True

    def observe_output(self, target, x, h, o):
        self.calls.append((target, x, h, o))


def test_the_adapter_hook_hands_full_outputs_to_the_recorder(imported):
    from test_import_modes import DIM, Toy

    from kilora import compose
    from kilora.recording import _ACTIVE

    session, _ = imported
    base = Toy().eval()
    adapter = compose(base, session.sets())
    target = next(t for t in session.state if t in adapter.targets)
    seen = []
    handle = base.get_submodule(target).register_forward_hook(
        lambda module, args, output: seen.append(output)
    )
    probe = _OutputProbe()
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(7))
    token = _ACTIVE.set(probe)
    try:
        with torch.no_grad():
            base(x)
    finally:
        _ACTIVE.reset(token)
        handle.remove()
    adapter.remove()
    names = [call[0] for call in probe.calls]
    assert target in names
    _, _, h, o = probe.calls[names.index(target)]
    assert o is seen[0]
    assert h.shape == o.shape


def test_imported_recording_leaves_outputs_and_state_equal(imported, monkeypatch):
    from test_import_modes import DIM, Toy

    from kilora import compose
    from kilora.live_imports import ImportEditor, import_source_frames

    session, model = imported
    session.align_for_moves()
    import_source_frames(session, _weights(session, model))
    before = {t: {k: v.clone() for k, v in parts.items()} for t, parts in session.state.items()}
    base = Toy().eval()
    adapter = compose(base, session.sets())
    gen = torch.Generator().manual_seed(5)
    xs = [torch.randn(1, 3, DIM, generator=gen) for _ in SIGMAS]
    with torch.no_grad():
        expected = [base(x) for x in xs]
        recorder = ImportEditor(session).recorder(session)
        actual = []
        for sigma, x in zip(SIGMAS, xs, strict=True):
            recorder.set_sigma(sigma)
            with recorder.recording():
                actual.append(base(x))
    adapter.remove()
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    for target, parts in before.items():
        for key, value in parts.items():
            assert torch.equal(session.state[target][key], value)
    stats = recorder.finish()
    assert stats.cross_errors
    assert all(entry.step >= 2 for items in stats.cross_errors.values() for entry in items)
    monkeypatch.delenv("KILORA_PREDICTION_DEBUG", raising=False)
    assert "eps_" not in ImportEditor._prediction_line(SimpleNamespace(stats=stats))
    monkeypatch.setenv("KILORA_PREDICTION_DEBUG", "1")
    line = ImportEditor._prediction_line(SimpleNamespace(stats=stats))
    counts = prediction_counts({}, stats.cross_errors)
    assert f"eps_steps={counts['eps_steps']} eps_nodes={counts['eps_nodes']}" in line
    assert f"eps_cross_nodes={counts['eps_cross_nodes']}" in line
    assert counts["eps_depth_steps"] == 0
