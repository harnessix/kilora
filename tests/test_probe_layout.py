# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.comfyui.probe_rows import ProbeLayout


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("draws", [2, 4, 8])
def test_layout_round_trip(batch, draws):
    layout = ProbeLayout(batch, draws)
    image = torch.randn(batch, 3, 5)
    expanded = layout.expand(image)
    kept, probes = layout.split(expanded)
    assert torch.equal(kept, image)
    assert probes.shape == (draws, *image.shape)
    for probe in probes:
        assert torch.equal(probe, image)
    assert torch.equal(layout.image_values(expanded).mean(0), image.mean(0))


def test_nested_conditions_do_not_mutate_source():
    layout = ProbeLayout(2, 2)
    tensor = torch.randn(2, 3)
    shared = torch.randn(5, 3)
    source = {"cross": [tensor], "extra": (shared, torch.tensor(1)), "flag": True}
    expanded = layout.expand(source)
    assert source["cross"][0] is tensor
    assert expanded["cross"][0].shape == (6, 3)
    assert expanded["extra"][0] is shared
    assert expanded["flag"] is True


def test_routes_use_image_rows_only():
    layout = ProbeLayout(2, 4)
    routes = torch.arange(30).reshape(10, 3)
    copied = layout.copy_routes(routes)
    for chunk in copied.chunk(5):
        assert torch.equal(chunk, routes[:2])
    assert torch.equal(routes, torch.arange(30).reshape(10, 3))


@pytest.mark.parametrize("batch,draws", [(0, 2), (2, 0), (2, 3), (2, -2)])
def test_invalid_layout(batch, draws):
    with pytest.raises(ValueError):
        ProbeLayout(batch, draws)


def test_wrong_output_batch_is_rejected():
    layout = ProbeLayout(2, 2)
    with pytest.raises(ValueError):
        layout.split(torch.zeros(5, 3))
    with pytest.raises(ValueError):
        layout.copy_routes(torch.zeros(5, 3))


@pytest.mark.parametrize("fails", [False, True])
def test_isolated_call_restores_records_and_random_stream(fails):
    from kilora import compute, recording
    from kilora.comfyui.probe_rows import isolated_probe_call

    original = torch.random.get_rng_state().clone()
    sentinel = object()

    def invoke():
        assert not compute._ROUTES
        assert recording.active_perturbation() is None
        torch.randn(20)
        if fails:
            raise RuntimeError("probe failed")
        return 17

    with compute.recording_routes() as routes, recording.perturbing(sentinel):
        if fails:
            with pytest.raises(RuntimeError, match="probe failed"):
                isolated_probe_call(invoke)
        else:
            assert isolated_probe_call(invoke) == 17
        assert compute._ROUTES[-1] is routes
        assert routes == {}
        assert recording.active_perturbation() is sentinel
    assert torch.equal(torch.random.get_rng_state(), original)


def test_isolated_call_keeps_progress_and_dropout_but_hides_axis_owner():
    from kilora import axis_owner, compute
    from kilora.comfyui.probe_rows import isolated_probe_call

    def invoke():
        assert compute.current_progress() == 0.75
        assert compute._DROPOUT_KEY[-1] == (17, 3)
        assert axis_owner._current.get() is None

    owner = object()
    with (
        compute.sampling_progress(0.75),
        compute.dropout_key(17, 3),
        axis_owner.recording(owner),
    ):
        isolated_probe_call(invoke)
        assert axis_owner._current.get()[0] is owner


def test_separate_probes_use_image_random_state_and_keep_sampler_output():
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.recording import active_perturbation

    class Trial:
        value = 0.25

    def invoke(args):
        trial = active_perturbation()
        noise = torch.randn_like(args["input"])
        return args["input"] + noise + (0 if trial is None else trial.value)

    args = {"input": torch.ones(2, 3)}
    initial = torch.random.get_rng_state()
    expected = invoke(args)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(initial)
    request = SeparateProbeCall([Trial()], lambda _: True)
    actual = request.call(invoke, args)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    reference, draws = request.outputs
    assert torch.equal(reference, expected)
    assert torch.equal(draws[0], expected + 0.25)
    assert request.reference_difference == 0


def test_separate_request_runs_inside_model_wrapper_without_changing_image():
    from kilora.comfyui.patching import _key_mask_by_step
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.recording import active_perturbation

    class Model:
        def __init__(self):
            self.model_options = {}

    class Trial:
        value = 0.125

    calls = []

    def apply(x, timestep):
        trial = active_perturbation()
        calls.append(trial)
        return x * 0.7 + torch.randn_like(x) + (0 if trial is None else trial.value)

    model = Model()
    _key_mask_by_step(model)
    wrapped = model.model_options["model_function_wrapper"]
    args = {"input": torch.ones(1, 2, 2, 2), "timestep": torch.tensor([1.0]), "c": {}}
    state = torch.random.get_rng_state()
    expected = wrapped(apply, args)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    request = SeparateProbeCall([Trial()], lambda _: True)
    model.model_options["kilora_render_probe"] = request
    actual = wrapped(apply, args)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert request.reference_difference == 0
    assert len(calls) == 4


@pytest.mark.parametrize("mode", ["keys", "full"])
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("fused", [False, True])
def test_separate_render_latent_identity_with_routed_layers(mode, depth, fused, monkeypatch):
    import dataclasses

    from test_share_transition import planted

    from kilora.bake import ModulePlan, TargetPlan
    from kilora.comfyui.patching import _key_mask_by_step
    from kilora.comfyui.probe_rows import SeparateProbeCall

    if fused:
        monkeypatch.setenv("TRITON_INTERPRET", "1")
        pytest.importorskip("triton")
        monkeypatch.delenv("KILORA_FUSED", raising=False)
        monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    else:
        monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_SEGMENTS", "0")
    layer = dataclasses.replace(planted(), import_score="derived")
    target = TargetPlan(sets=[layer], depth=depth, name="layer", roles=["all"])
    plan = ModulePlan(parts=[(0, -1, target)])

    class Model:
        def __init__(self):
            self.model_options = {}

    class Trial:
        set_key = None
        replay_all = False
        du = {}

        def __init__(self, sign):
            self.sign = sign

        def delta(self, target, part):
            return torch.full_like(part, self.sign * 0.01)

    def apply(x, timestep):
        tokens = x.flatten(2).transpose(1, 2)
        result = plan.run(tokens, tokens * 0.7)
        return result.transpose(1, 2).reshape_as(x)

    def render(probes):
        model = Model()
        _key_mask_by_step(model)
        request = SeparateProbeCall([Trial(1), Trial(-1)], lambda a: float(a["timestep"][0]) == 0.5)
        if probes:
            model.model_options["kilora_render_probe"] = request
        wrapper = model.model_options["model_function_wrapper"]
        latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
        with torch.no_grad():
            for sigma in (1.0, 0.5, 0.25):
                args = {"input": latent, "timestep": torch.tensor([sigma]), "c": {}}
                latent = latent - wrapper(apply, args) * 0.1
        return latent, request

    initial = torch.random.get_rng_state()
    expected, _ = render(False)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(initial)
    actual, request = render(True)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert request.outputs is not None
    assert request.reference_difference == 0
    assert not torch.equal(request.outputs[1][0], request.outputs[1][1])


def test_probe_memory_failure_keeps_image_and_random_state():
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.recording import active_perturbation

    trial = object()

    def invoke(args):
        value = args["input"] + torch.randn_like(args["input"])
        if active_perturbation() is trial:
            raise torch.cuda.OutOfMemoryError("probe allocation")
        return value

    args = {"input": torch.ones(2, 3)}
    state = torch.random.get_rng_state()
    expected = invoke(args)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    request = SeparateProbeCall([trial], lambda _: True)
    actual = request.call(invoke, args)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert request.outputs is None
    assert request.failure is not None


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("budget", [2, 4, 8])
def test_render_forward_estimate_uses_retained_reflow_losses_without_more_calls(
    guided, invalid, budget
):
    from types import SimpleNamespace

    from kilora import compute
    from kilora.comfyui.probe_rows import ForwardProbeDraw, ForwardRenderProbe
    from kilora.recording import active_perturbation

    expert = SimpleNamespace(
        down=torch.tensor([[[[1.0, 0.5]]]]),
        up=torch.tensor([[[[0.5, 1.0]]]]),
        exact={},
    )
    source = SimpleNamespace(experts={"layer": expert}, extra={}, name="probe")
    session = SimpleNamespace(
        mod_id="probe",
        commits=[],
        _source=source,
        _section_pairs=set(),
        probe_budget=budget,
        state={"layer": {"core_0": torch.ones(1, 1), "strength_0": torch.tensor(1.0)}},
    )
    calls = []

    def invoke(args):
        calls.append(1)
        x = args["input"]
        trial = active_perturbation()
        delta = None if trial is None else trial.input_delta("layer", x)
        compute._note_fused_mass(
            "layer", [torch.ones(*x.shape[:-1], 1)], [SimpleNamespace(key="probe")]
        )
        clean = (x @ expert.down[0, 0].T) @ expert.up[0, 0]
        if invalid and isinstance(trial, ForwardProbeDraw):
            return torch.full_like(clean, float("nan"))
        return clean if delta is None else clean + delta

    x = torch.tensor([[[1.0, 0.5]], [[1.0, 0.5]]])
    args = {
        "input": x if guided else x[:1],
        "timestep": torch.tensor([0.5, 0.5]) if guided else torch.tensor([0.5]),
        "cond_or_uncond": [0, 1] if guided else [0],
        "c": {},
    }
    probe = ForwardRenderProbe(
        session, lambda _: True, empty_conditioning=None if guided else lambda _: {}
    )
    expected = invoke(args)
    actual = probe.call(invoke, args)
    assert torch.equal(actual, expected)
    if budget == 2 and guided:
        assert len(calls) == 4
        assert probe.request.row_count == 2
    if budget == 8 and not invalid:
        assert probe.request.outputs is None
        next_args = {**args, "timestep": args["timestep"] * 0.5}
        assert torch.equal(probe.call(invoke, next_args), expected)
    before_finish = len(calls)
    final = torch.zeros_like(x[:1])
    retained = probe.finish(final)
    assert len(calls) == before_finish
    if invalid:
        assert retained is None
        assert probe.request.failure is not None
        assert not probe.estimate.reads and not probe.estimate._device_parts
        return
    estimate, routes, outputs = retained
    assert estimate.render_ready
    if budget <= 4:
        assert estimate.clean_loss == pytest.approx(outputs.losses(final)[0])
        assert estimate.losses == pytest.approx(outputs.losses(final)[1].tolist())
    else:
        assert len(probe.frames) == 2
        assert len(estimate.losses) == 4
        assert all(torch.isfinite(torch.tensor(estimate.losses)))
    assert routes.mass["layer"].tolist() == [1.0]
    assert set(estimate.gradients()) == {("layer", 0)}
    assert all(torch.isfinite(g).all() for g, _ in estimate.gradients().values())
    assert probe.finish(final) is None


def test_isolated_probe_does_not_update_depth_or_fused_reports():
    from kilora import depth_map, fused
    from kilora.comfyui.probe_rows import isolated_probe_call

    marker = object()
    counts, reasons = dict(fused.COUNTS), dict(fused.REASONS)
    imports = dict(fused.IMPORT_REASONS)
    last = list(fused.LAST_REASON)

    def invoke():
        assert depth_map.current_measurement() is None
        fused.count_import(True)
        fused.count_folded(False, "probe test")
        fused.count_walks(False, "probe test")
        fused.import_eager("probe test")
        fused.refuse_import("probe test")

    with depth_map.measuring(marker):
        isolated_probe_call(invoke)
        assert depth_map.current_measurement() is marker
    assert counts == fused.COUNTS
    assert reasons == fused.REASONS
    assert imports == fused.IMPORT_REASONS
    assert last == fused.LAST_REASON


def test_isolation_hides_share_scope_factories_and_call_caches():
    from kilora import call_values, compute
    from kilora.comfyui.probe_rows import isolated_probe_call

    seen = []

    def wants(target, sets):
        seen.append(target)
        return True

    def invoke():
        with compute.layer_set_share_observation("probe", []):
            assert not compute._SET_SHARES
        call_values.current_alpha_cache()["probe"] = 1
        call_values.host_scalar(torch.tensor([0.25]))
        assert not compute._DERIVED_FIRST_BLOCK

    with (
        compute.observing_set_share_layers(wants, lambda *args: None),
        compute._DERIVED_FIRST_BLOCK.pushed({"image": 0}),
        call_values.axis_alphas({}),
        call_values.host_values(),
    ):
        alpha_cache = call_values.current_alpha_cache()
        host_cache = call_values._HOST_SCALARS[-1]
        isolated_probe_call(invoke)
        assert not alpha_cache and not host_cache
        assert compute._DERIVED_FIRST_BLOCK[-1] == {"image": 0}
        with compute.layer_set_share_observation("image", []):
            pass
    assert seen == ["image"]


def test_nested_wrappers_collect_one_probe_request():
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.recording import active_perturbation

    calls = []

    class Trial:
        value = 0.125

    def inner(args):
        calls.append(1)
        trial = active_perturbation()
        return args["input"] + (0 if trial is None else trial.value)

    def outer(args):
        return request.call(inner, args)

    request = SeparateProbeCall([Trial()], lambda _: True)
    args = {"input": torch.ones(1, 2)}
    assert torch.equal(request.call(outer, args), args["input"])
    assert len(calls) == 3
    assert request.outputs is not None
    assert not request._running


@pytest.mark.parametrize("budget,expected", [(2, 2), (3, 2), (4, 2), (8, 4), (12, 6)])
def test_render_probe_budget_keeps_complete_caption_pairs(budget, expected):
    import types

    from kilora.comfyui.probe_rows import ForwardRenderProbe

    session = types.SimpleNamespace(
        state={},
        commits=[],
        mod_id="budget",
        probe_budget=budget,
        _source=types.SimpleNamespace(experts={}),
    )
    probe = ForwardRenderProbe(session, lambda _: True, draws=8)
    assert probe.estimate.draws == expected
    assert len(probe.all_trials) == expected
    assert len(probe.request.trials) == 2


def test_image_reference_reuse_avoids_an_extra_clean_call():
    from kilora.comfyui.probe_rows import SeparateProbeCall

    calls = []
    args = {"input": torch.ones(2, 3), "timestep": torch.tensor([0.5])}

    def invoke(arguments):
        calls.append(1)
        return arguments["input"] * 2

    request = SeparateProbeCall([None, None], lambda _: True, reuse_image_reference=True)
    image = request.call(invoke, args)
    assert len(calls) == 3
    assert torch.equal(request.outputs[0], image)
    assert request.reference_difference == 0


def test_probe_budget_does_not_shrink_image_batch():
    from types import SimpleNamespace

    from kilora.comfyui.probe_rows import ForwardRenderProbe

    session = SimpleNamespace(
        state={},
        commits=[],
        mod_id="budget",
        probe_budget=2,
        _source=SimpleNamespace(experts={}),
    )
    probe = ForwardRenderProbe(session, lambda _: True)
    args = {
        "input": torch.ones(4, 3),
        "cond_or_uncond": [0, 1],
        "timestep": torch.full((4,), 0.5),
    }
    calls = []

    def invoke(arguments):
        calls.append(arguments["input"].shape[0])
        return arguments["input"] * 2

    assert torch.equal(probe.call(invoke, args), args["input"] * 2)
    assert calls == [4]
    assert not probe.frames
