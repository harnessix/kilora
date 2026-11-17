# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch

from kilora.comfyui.probe_rows import PendingProbeWrite, copy_probe_state


@pytest.mark.parametrize("pinned", [False, True])
def test_accepted_import_probe_keeps_empty_source_targets(tmp_path, pinned):
    from test_import_forest import _assert_complete_sets, _session_with_empty_target

    from kilora.comfyui.live_write import QueuedWrite
    from kilora.live import SESSIONS
    from kilora.live_queue import Event
    from kilora.recording import GenerationRecord

    SESSIONS.clear()
    try:
        session, editor, directions, empty = _session_with_empty_target(tmp_path)
        if pinned:
            session.pin()
        before = copy_probe_state(session.state)
        proposal = copy_probe_state(before)
        target = next(t for t, parts in proposal.items() if "strength_0" in parts)
        proposal[target]["strength_0"] -= 0.01
        snapshot = session.editor_snapshot()
        pending = PendingProbeWrite(before, proposal, GenerationRecord(), snapshot, snapshot)
        pending.tested_base = copy_probe_state(before)
        pending.tested_state = proposal
        clean = torch.ones(2, 1, 1, 2)
        assert pending.evaluate(session, torch.zeros(1, 1, 2), 0.5, clean, clean * 0.5)
        record = GenerationRecord(images=1)
        record.extra["probe_acceptances"] = [pending]
        job = QueuedWrite(session, editor, record, None)
        job.run_many([Event(1, job)])
        assert len(session.commits) == 1
        _assert_complete_sets(session)
        assert session.state[empty] == {}
        assert torch.equal(session.state[target]["strength_0"], proposal[target]["strength_0"])
        session.undo()
        _assert_complete_sets(session)
        assert torch.equal(session.state[target]["strength_0"], before[target]["strength_0"])
    finally:
        SESSIONS.clear()


def test_rejected_probe_is_retried_at_a_smaller_scale():
    state = {"layer": {"core_0": torch.ones(1, 1)}}
    proposal = {"layer": {"core_0": torch.full((1, 1), 2.0)}}
    pending = PendingProbeWrite(state, proposal, object(), {}, {})
    session = types.SimpleNamespace(guard=False)
    final = torch.zeros(1, 1, 2)
    clean = torch.ones(2, 1, 1, 2)
    assert not pending.evaluate(session, final, 0.5, clean, clean * 2)
    assert pending.scale == 0.5 and not pending.ready
    assert pending.state_at(state)["layer"]["core_0"].item() == 1.5
    assert pending.evaluate(session, final, 0.5, clean, clean * 0.5)
    assert pending.ready and pending.scale == 0.5
    assert pending.attempts == 2


def test_history_patch_keeps_forest_values_and_accumulates_events():
    state = {"layer": {"core_0": torch.ones(1, 1)}}
    before = {"prior/event": torch.tensor(0), "forest/serial": torch.tensor(2)}
    after = {"prior/event": torch.tensor(1), "forest/serial": torch.tensor(9)}
    pending = PendingProbeWrite(state, state, object(), before, after)
    current = {"prior/event": torch.tensor(4), "forest/serial": torch.tensor(7)}
    patched = pending.history_at(current)
    assert patched["prior/event"].item() == 5
    assert patched["forest/serial"].item() == 7


@pytest.mark.parametrize("side_branch", [False, True])
def test_pending_backlog_uses_queued_commit_path_without_model_calls(tmp_path, side_branch):
    from test_import_edit_dimensions import _session

    from kilora.comfyui.live_write import QueuedWrite
    from kilora.live import SESSIONS
    from kilora.live_imports import ImportEditor
    from kilora.live_queue import Event
    from kilora.recording import GenerationRecord

    SESSIONS.clear()
    session = _session(tmp_path)
    session.learning = True
    session.guard = False
    if side_branch:
        branch = session.editor_state.fork(seed=17, temperature=0.2, depth=2)
        session.editor_state.checkout(branch)
    branch = session.editor_state.forest.current
    serial = session.editor_state.forest.serial
    target = next(iter(session.state))
    initial = float(session.state[target]["strength_0"])

    class Rule(ImportEditor):
        def next_combined_state(self, *args, **kwargs):
            raise AssertionError("probe records must be staged individually")

        def next_state(self, state, record):
            if record.extra.get("accepted_probe") is not None:
                return super().next_state(state, record)
            increment = record.extra.get("increment")
            if increment is None:
                return None
            self.session.editor_state.event += 1
            result = copy_probe_state(state)
            result[target]["strength_0"] += increment
            return result

    rule = Rule(session)
    jobs = []
    for number, increment in enumerate((0.1, 0.2, 0.3), 1):
        record = GenerationRecord(images=1)
        record.extra.update(
            render_probe=object(),
            increment=increment,
            draw={"seed": number, "temperature": 0.3, "depth": 2},
        )
        jobs.append(Event(number, QueuedWrite(session, rule, record, None)))
    jobs[0].run.run_many(jobs)
    assert len(session._probe_writes) == 3
    assert len(session.commits) == 0
    assert float(session.state[target]["strength_0"]) == initial
    assert session.editor_state.event == 0
    records = [p.record for p in session._probe_writes]

    for pending in list(session._probe_writes):
        pending.tested_base = copy_probe_state(session.state)
        pending.tested_state = pending.state_at(session.state)
        clean = torch.ones(2, 1, 1, 2)
        assert pending.evaluate(session, torch.zeros(1, 1, 2), 0.5, clean, clean * 0.5)
        record = GenerationRecord(images=1)
        record.extra["probe_acceptances"] = [pending]
        job = QueuedWrite(session, rule, record, None)
        job.run_many([Event(0, job)])
    assert not session._probe_writes
    assert len(session.commits) == 3
    assert float(session.state[target]["strength_0"]) == pytest.approx(initial + 0.6)
    assert session.editor_state.event == 3
    assert session.editor_state.commit_count == 3
    assert session.editor_state.forest.current == branch
    assert session.editor_state.forest.serial == serial
    assert [entry.branch for entry in session.undo_log._entries] == [branch] * 3
    assert all(record.extra["increment"] > 0 for record in records)
    record = GenerationRecord(images=1)
    record.extra["increment"] = 0.4
    job = QueuedWrite(session, rule, record, None)
    job.run_many([Event(0, job)])
    assert session.editor_state.event == 4
    assert session.editor_state.commit_count == 4
    for count, total in ((3, 0.6), (2, 0.3), (1, 0.1), (0, 0.0)):
        session.undo()
        assert session.editor_state.event == count
        assert session.editor_state.commit_count == count
        assert len(session.commits) == count
        assert float(session.state[target]["strength_0"]) == pytest.approx(initial + total)
    if side_branch:
        untouched = session.editor_state.forest.branches["main"].state
        assert untouched.event == untouched.commit_count == 0
    SESSIONS.clear()


def test_loss_improvement_still_obeys_collapse_guard():
    state = {"layer": {"core_0": torch.ones(1, 1)}}
    pending = PendingProbeWrite(state, state, object(), {}, {})
    session = types.SimpleNamespace(guard=True)
    clean = torch.tensor([[[[1.0, 2.0]]], [[[1.0, 2.0]]]])
    assert not pending.evaluate(
        session, torch.zeros(1, 1, 2), 0.5, clean, torch.full_like(clean, 0.01)
    )
    assert pending.after_loss < pending.before_loss
    assert pending.scale == 0.5 and not pending.ready


def test_next_render_tests_candidate_only_on_probe_rows():
    from kilora.bake import ModulePlan, TargetPlan
    from kilora.comfyui.probe_rows import ForwardRenderProbe
    from kilora.compute import LayerSet
    from kilora.recording import GenerationRecord

    expert = types.SimpleNamespace(
        down=torch.tensor([[[[1.0, 0.5]]]]),
        up=torch.tensor([[[[0.5, 1.0]]]]),
        exact={},
    )
    source = types.SimpleNamespace(experts={"layer": expert}, extra={}, name="probe")
    state = {"layer": {"core_0": torch.ones(1, 1), "strength_0": torch.tensor(1.0)}}
    session = types.SimpleNamespace(
        mod_id="probe",
        commits=[],
        _source=source,
        _section_pairs=set(),
        state=state,
        guard=False,
        probe_budget=4,
    )
    proposed = copy_probe_state(state)
    proposed["layer"]["core_0"] *= 0.5
    pending = PendingProbeWrite(copy_probe_state(state), proposed, GenerationRecord(), {}, {})
    session._probe_writes = [pending]
    layer = LayerSet(
        router=torch.eye(2),
        keys_in=torch.ones(1, 1, 2),
        down=expert.down.reshape(1, 2),
        up=expert.up.reshape(1, 2),
        per_expert=1,
        form="input",
        key="probe",
        import_score="derived",
        gate_mode="none",
    )
    plan = ModulePlan(
        parts=[
            (
                0,
                -1,
                TargetPlan(
                    sets=[layer],
                    depth=1,
                    name="layer",
                    roles=["all"],
                ),
            )
        ]
    )
    x = torch.ones(2, 2, 1, 1)
    args = {"input": x, "timestep": torch.tensor([0.5, 0.5]), "cond_or_uncond": [0, 1]}
    calls = []

    def invoke(args):
        calls.append(1)
        tokens = args["input"].flatten(2).transpose(1, 2)
        return plan.run(tokens, torch.zeros_like(tokens)).transpose(1, 2).reshape_as(x)

    with torch.no_grad():
        expected = invoke(args)
        probe = ForwardRenderProbe(session, lambda _: False)
        actual = probe.call(invoke, args)
        count = len(calls)
        assert probe.finish(torch.zeros_like(x[:1])) is not None
    assert len(calls) == count
    assert torch.equal(actual, expected)
    assert torch.equal(session.state["layer"]["core_0"], torch.ones(1, 1))
    assert pending.ready
    assert pending.after_loss < pending.before_loss
    assert pending.tested_state["layer"]["core_0"].item() == 0.5


def test_pending_record_does_not_keep_prior_acceptance_or_model_closures():
    from kilora.comfyui.probe_rows import stage_probe_write
    from kilora.recording import GenerationRecord

    state = {"layer": {"core_0": torch.ones(1, 1)}}
    session = types.SimpleNamespace(state=state, editor_snapshot=lambda: {})
    record = GenerationRecord(measure=lambda trial: 0.0, guard_score=lambda trial: {})
    previous = object()
    record.extra.update(render_probe=object(), probe_acceptances=[previous], generation=7)
    pending = stage_probe_write(session, record, state, {})
    assert pending.record.extra == {"generation": 7}
    assert pending.record.measure is None and pending.record.guard_score is None
    assert record.extra["probe_acceptances"] == [previous]
    assert record.measure is not None


def test_pinned_render_with_separate_requests_while_forest_commits(tmp_path, monkeypatch):
    from test_live_backlog_branches import recorded, setup_session

    from kilora.bake import ModulePlan, ScopedSet, plan_targets
    from kilora.comfyui.live_write import QueuedWrite
    from kilora.comfyui.patching import _key_mask_by_step
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.live_queue import Event

    monkeypatch.setenv("KILORA_FUSED", "0")
    session, rule, targets = setup_session(tmp_path)
    session.pin()
    pinned_branch = session.pinned["branch"]
    serial = session.editor_state.forest.serial

    class Model:
        def __init__(self):
            self.model_options = {}

    class Trial:
        def __init__(self, sign):
            self.sign = sign

        def delta(self, target, part):
            return torch.full_like(part, self.sign * 0.01)

    def render(probes):
        plans = plan_targets(
            [ScopedSet(adapter, gate_mode="none", roles="all") for adapter in session.sets()],
            depth=1,
        )
        target = next(iter(plans.values()))
        plan = ModulePlan(parts=[(0, -1, target)])
        width = target.sets[0].router.shape[-1]
        latent = torch.arange(width * 4, dtype=torch.float32).reshape(1, width, 2, 2) / 16
        model = Model()
        _key_mask_by_step(model)
        request = SeparateProbeCall([Trial(1), Trial(-1)], lambda a: float(a["timestep"][0]) == 0.5)
        if probes:
            model.model_options["kilora_render_probe"] = request

        def apply(x, timestep):
            tokens = x.flatten(2).transpose(1, 2)
            out = plan.run(tokens, tokens * 0.7)
            return out.transpose(1, 2).reshape_as(x)

        with torch.no_grad():
            for sigma in (1.0, 0.5, 0.25):
                args = {"input": latent, "timestep": torch.tensor([sigma]), "c": {}}
                latent = latent - model.model_options["model_function_wrapper"](apply, args) * 0.1
        return latent, request

    baseline, _ = render(False)
    for index, target in enumerate(targets[:2]):
        with_probes, request = render(True)
        assert torch.equal(with_probes, baseline)
        assert request.outputs is not None and request.reference_difference == 0
        job = QueuedWrite(session, rule, recorded(session, target), None)
        job.run_many([Event(index, job)])
    assert len(session.commits) == 2
    assert session.editor_state.forest.serial > serial
    assert session.editor_state.forest.current != pinned_branch
    assert torch.equal(render(False)[0], baseline)
    assert torch.equal(render(True)[0], baseline)


def test_probe_condition_allocation_failure_preserves_completed_image():
    from kilora.comfyui.probe_rows import SeparateProbeCall

    def invoke(args):
        return args["input"] + torch.randn_like(args["input"])

    def allocate(args):
        raise torch.cuda.OutOfMemoryError("probe conditions")

    args = {"input": torch.ones(1, 2)}
    state = torch.random.get_rng_state()
    expected = invoke(args)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    request = SeparateProbeCall([], lambda _: True, probe_args=allocate)
    assert torch.equal(request.call(invoke, args), expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert request.outputs is None and request.failure is not None


def test_imported_pinned_probe_render_keeps_latent_while_committing(tmp_path, monkeypatch):
    from test_import_edit_dimensions import _session

    from kilora.bake import ModulePlan, ScopedSet, plan_targets
    from kilora.comfyui.live_write import QueuedWrite
    from kilora.comfyui.patching import _key_mask_by_step
    from kilora.comfyui.probe_rows import SeparateProbeCall
    from kilora.errors import KiLoRAError
    from kilora.live import SESSIONS
    from kilora.live_imports import ImportEditor
    from kilora.live_queue import Event
    from kilora.recording import GenerationRecord

    SESSIONS.clear()
    monkeypatch.setenv("KILORA_FUSED", "0")
    session = _session(tmp_path)
    session.learning = True
    session.guard = False
    rule = ImportEditor(session)
    if getattr(session.editor_state, "forest", None) is None:
        with pytest.raises(KiLoRAError, match="pin requires an editor with branches"):
            session.pin()
        SESSIONS.clear()
        pytest.xfail("Imported ImportHistory has no forest")
    session.pin()
    pinned_branch = session.pinned["branch"]

    class Model:
        def __init__(self):
            self.model_options = {}

    class Trial:
        def __init__(self, sign):
            self.sign = sign

        def delta(self, target, part):
            return torch.full_like(part, self.sign * 0.01)

    def render(probes):
        plans = plan_targets(
            [
                ScopedSet(adapter, gate_mode="none", roles="all", import_score="derived")
                for adapter in session.sets()
            ],
            depth=1,
        )
        target = plans[next(name for name in plans if name.endswith("q_proj"))]
        plan = ModulePlan(parts=[(0, -1, target)])
        width = target.sets[0].down.shape[-1]
        latent = torch.arange(width * 4, dtype=torch.float32).reshape(1, width, 2, 2) / 16
        model = Model()
        _key_mask_by_step(model)
        if probes:
            model.model_options["kilora_render_probe"] = SeparateProbeCall(
                [Trial(1), Trial(-1)], lambda a: float(a["timestep"][0]) == 0.5
            )

        def apply(x, timestep):
            tokens = x.flatten(2).transpose(1, 2)
            out = plan.run(tokens, tokens * 0.7)
            return out.transpose(1, 2).reshape_as(x)

        with torch.no_grad():
            for sigma in (1.0, 0.5, 0.25):
                args = {"input": latent, "timestep": torch.tensor([sigma]), "c": {}}
                latent = latent - model.model_options["model_function_wrapper"](apply, args) * 0.1
        return latent

    baseline = render(False)
    target = next(iter(session.state))
    for index in range(2):
        assert torch.equal(render(True), baseline)
        proposed = copy_probe_state(session.state)
        proposed[target]["strength_0"] -= 0.01
        pending = PendingProbeWrite(
            copy_probe_state(session.state),
            proposed,
            GenerationRecord(),
            session.editor_snapshot(),
            session.editor_snapshot(),
        )
        pending.tested_base = copy_probe_state(session.state)
        pending.tested_state = proposed
        clean = torch.ones(2, 1, 1, 2)
        assert pending.evaluate(session, torch.zeros(1, 1, 2), 0.5, clean, clean * 0.5)
        record = GenerationRecord()
        record.extra["probe_acceptances"] = [pending]
        job = QueuedWrite(session, rule, record, None)
        job.run_many([Event(index, job)])
    assert len(session.commits) == 2
    assert session.editor_state.forest.current != pinned_branch
    assert torch.equal(render(False), baseline)
    assert torch.equal(render(True), baseline)
    SESSIONS.clear()


def test_probe_is_dropped_after_the_smaller_retry_fails():
    state = {"layer": {"core_0": torch.ones(1, 1)}}
    pending = PendingProbeWrite(state, state, object(), {}, {})
    session = types.SimpleNamespace(guard=False)
    clean = torch.ones(2, 1, 1, 2)
    for _ in range(2):
        assert not pending.evaluate(session, torch.zeros(1, 1, 2), 0.5, clean, clean * 2)
    assert pending.dropped and pending.decision == "dropped"
    assert pending.scale == 0.5


def test_imported_probe_write_cycle_has_no_calls_between_renders(tmp_path):
    from test_import_edit_dimensions import _session

    from kilora import compute
    from kilora.comfyui.live_write import QueuedWrite
    from kilora.comfyui.probe_rows import ForwardRenderProbe, RetainedProbeMeasure, same_probe_state
    from kilora.live import SESSIONS
    from kilora.live_imports import ImportEditor
    from kilora.live_queue import Event
    from kilora.recording import GenerationRecord, active_perturbation

    SESSIONS.clear()
    session = _session(tmp_path)
    session.set_options(probe_rows=True, probe_budget=4)
    session.learning, session.guard = True, False
    target = next(iter(session.state))
    expert = session._source.experts[target]
    width = expert.down.shape[-1]
    x = torch.linspace(0.1, 1, width).reshape(1, 1, width).repeat(2, 2, 1)
    calls = []

    def invoke(args):
        calls.append(1)
        value = args["input"]
        output = torch.zeros_like(value)
        for e in range(expert.up.shape[0]):
            core = session.state[target][f"core_{e}"]
            output += session.state[target][f"strength_{e}"] * (
                ((value @ expert.down[e, 0].T) @ core.T) @ expert.up[e, 0]
            )
        compute._note_fused_mass(
            target,
            [torch.ones(*value.shape[:-1], expert.up.shape[0])],
            [types.SimpleNamespace(key=session.mod_id)],
        )
        trial = active_perturbation()
        delta = None if trial is None else trial.input_delta(target, value)
        return output if delta is None else output + delta

    class Rule(ImportEditor):
        def next_state(self, state, record):
            if record.extra.get("accepted_probe") is not None:
                return super().next_state(state, record)
            retained = record.extra.get("render_probe")
            if retained is None:
                return None
            if getattr(retained[0], "acceptance_only", False):
                return None
            proposal = copy_probe_state(state)
            for (name, e), (core, strength) in retained[0].gradients().items():
                proposal[name][f"core_{e}"] -= 0.001 * core.float()
                proposal[name][f"strength_{e}"] -= 0.001 * strength
            return proposal

    rule = Rule(session)
    args = {"input": x, "timestep": torch.tensor([0.5, 0.5]), "cond_or_uncond": [0, 1]}
    final = torch.zeros_like(x[:1])
    with torch.no_grad():
        first = ForwardRenderProbe(session, lambda _: True)
        first.call(invoke, args)
        retained = first.finish(final)
        assert retained is not None
        record = GenerationRecord(images=1, measure=RetainedProbeMeasure(retained))
        record.extra["render_probe"] = retained
        count = len(calls)
        job = QueuedWrite(session, rule, record, None)
        job.run_many([Event(1, job)])
        assert len(calls) == count
        assert len(session._probe_writes) == 1
        pending = session._probe_writes[0]
        expected = invoke(args)
        second = ForwardRenderProbe(session, lambda _: True)
        assert torch.equal(second.call(invoke, args), expected)
        count = len(calls)
        assert second.finish(final) is not None
        assert len(calls) == count
        if not pending.ready:
            assert pending.scale == 0.5
            third = ForwardRenderProbe(session, lambda _: True)
            assert torch.equal(third.call(invoke, args), expected)
            count = len(calls)
            assert third.finish(final) is not None
            assert len(calls) == count
        if pending.ready:
            candidate = copy_probe_state(pending.tested_state)
            accepted = GenerationRecord(images=1)
            accepted.extra["probe_acceptances"] = [pending]
            job = QueuedWrite(session, rule, accepted, None)
            job.run_many([Event(2, job)])
            assert len(calls) == count
            assert same_probe_state(session.state, candidate)
        else:
            assert pending.dropped
    SESSIONS.clear()


def test_write_estimates_average_per_target_and_reset_on_changed_state():
    from kilora.live_imports import RESPONSE_FLOOR
    from kilora.write_estimates import accumulate_write_estimate

    state = {"layer": {"core_0": torch.ones(1, 1)}}
    session = types.SimpleNamespace(state=state)

    def estimate(value):
        return types.SimpleNamespace(
            state=copy_probe_state(state),
            target={"layer": [0]},
            gradients=lambda: {("layer", 0): (torch.tensor([[value]], dtype=torch.float64), 0.0)},
        )

    first = estimate(0.0)
    accumulate_write_estimate(session, first)
    assert first.mean_gradients[("layer", 0)][0].item() < RESPONSE_FLOOR
    second = estimate(4 * RESPONSE_FLOOR)
    note = accumulate_write_estimate(session, second)
    assert second.mean_gradients[("layer", 0)][0].item() == pytest.approx(2 * RESPONSE_FLOOR)
    assert "over 2 estimates" in note and "floor" in note
    accumulate_write_estimate(session, second)
    assert session._write_estimates["layer"][1] == 2
    state["layer"]["core_0"] += 1
    changed = estimate(RESPONSE_FLOOR)
    accumulate_write_estimate(session, changed)
    assert session._write_estimates["layer"][1] == 1


def test_six_probe_renders_keep_learning_and_report_every_decision(tmp_path, caplog):
    import logging

    from test_import_edit_dimensions import _session

    from kilora.bake import ModulePlan, ScopedSet, plan_targets
    from kilora.comfyui.live_write import QueuedWrite
    from kilora.comfyui.probe_rows import ForwardRenderProbe, RetainedProbeMeasure
    from kilora.live import SESSIONS
    from kilora.live_imports import ImportEditor
    from kilora.live_queue import Event
    from kilora.recording import GenerationRecord

    SESSIONS.clear()
    session = _session(tmp_path)
    session.set_options(probe_rows=True, probe_budget=2)
    session.learning, session.guard = True, False
    for expert in session._source.experts.values():
        expert.down.fill_(1)
        expert.up[..., 0] = 1
        expert.up[..., 1:] = 0

    class Rule(ImportEditor):
        def next_state(self, state, record):
            if record.extra.get("accepted_probe") is not None:
                return super().next_state(state, record)
            retained = record.extra.get("render_probe")
            if retained is None or retained[0].acceptance_only:
                return None
            from kilora.write_estimates import accumulate_write_estimate

            self.note = accumulate_write_estimate(session, retained[0])
            proposal = copy_probe_state(state)
            for (target, expert), (core, strength) in retained[0].gradients().items():
                proposal[target][f"core_{expert}"] -= 1e-4 * core / (1 + core.norm())
                proposal[target][f"strength_{expert}"] -= 1e-4 * strength / (1 + abs(strength))
            return proposal

    rule = Rule(session)
    seeds, captions, decisions = [], [], []
    caplog.set_level(logging.INFO)
    with torch.no_grad():
        for render in range(6):
            plans = plan_targets(
                [
                    ScopedSet(adapter, gate_mode="none", roles="all", import_score="derived")
                    for adapter in session.sets()
                ],
                depth=1,
            )
            target = next(name for name in plans if name.endswith("q_proj"))
            plan = ModulePlan(parts=[(0, -1, plans[target])])
            width = plans[target].sets[0].down.shape[-1]
            x = torch.linspace(0.1, 1.0, width).reshape(1, 1, width).repeat(2, 2, 1)
            args = {
                "input": x,
                "timestep": torch.full((2,), 0.8 if render % 2 else 0.3),
                "cond_or_uncond": [0, 1],
                "c": {},
            }
            calls = []

            def invoke(args, calls=calls, plan=plan):
                calls.append(1)
                value = args["input"]
                return plan.run(value, torch.zeros_like(value))

            expected = invoke(args)
            pending = bool(getattr(session, "_probe_writes", []))
            probe = ForwardRenderProbe(session, lambda _, pending=pending: not pending)
            seeds.append(probe.estimate.seed)
            captions.append(probe.caption_kind)
            assert torch.equal(probe.call(invoke, args), expected)
            count = len(calls)
            retained = probe.finish(torch.zeros_like(x[:1]))
            assert retained is not None
            assert len(calls) == count
            record = GenerationRecord(images=1, measure=RetainedProbeMeasure(retained))
            record.extra["render_probe"] = retained
            record.extra["probe_acceptances"] = [
                p for p in getattr(session, "_probe_writes", []) if p.ready
            ]
            job = QueuedWrite(session, rule, record, None)
            report = job.run_many([Event(render, job)])
            assert len(calls) == count
            assert report
            decision = probe.acceptance_note if pending else "estimate staged"
            decisions.append(f"render {render}: {decision}; commits {len(session.commits)}")
            if pending:
                assert "loss" in probe.acceptance_note
                assert any(word in decision for word in ("accepted", "retried", "dropped"))
                if probe.pending.ready:
                    assert session._write_estimates == {}
                    assert session._write_estimate_state is None
                else:
                    assert session._write_estimates
                    assert session._probe_writes
            else:
                assert "Write target" in caplog.text
                assert session._probe_writes
            print(decisions[-1])
    assert len(session.commits) >= 3, "\n".join(decisions)
    assert len(set(seeds)) == 6
    assert captions[0] != captions[1] and captions[0] == captions[2]
    assert caplog.text.count("Probe candidate accepted") >= 3
    SESSIONS.clear()


def test_pending_probe_reports_why_render_could_not_test_it(caplog):
    import logging

    from kilora.comfyui.probe_rows import ForwardRenderProbe

    state = {"layer": {"core_0": torch.ones(1, 1)}}
    pending = PendingProbeWrite(state, state, object(), {}, {})
    session = types.SimpleNamespace(
        state=state,
        commits=[],
        mod_id="no-room",
        probe_budget=2,
        _source=types.SimpleNamespace(experts={}, extra={}, name="no-room"),
        _section_pairs=set(),
        _probe_writes=[pending],
    )
    args = {
        "input": torch.ones(4, 3),
        "cond_or_uncond": [1],
        "timestep": torch.ones(4),
    }
    caplog.set_level(logging.INFO)
    probe = ForwardRenderProbe(session, lambda _: False)
    assert torch.equal(probe.call(lambda a: a["input"], args), args["input"])
    assert probe.finish(torch.ones(2, 3)) is None
    assert "probe not tested" in probe.request.failure
    assert "probe not tested" in caplog.text
    assert pending.attempts == 0 and not pending.dropped


def test_failed_candidate_is_updated_from_running_mean(tmp_path):
    from test_import_edit_dimensions import _session

    from kilora.comfyui.probe_rows import stage_probe_write
    from kilora.live import SESSIONS
    from kilora.live_imports import ImportEditor
    from kilora.recording import GenerationRecord
    from kilora.write_estimates import accumulate_write_estimate

    SESSIONS.clear()
    session = _session(tmp_path)
    ImportEditor(session)
    target = next(iter(session.state))
    key = (target, 0)
    core = session.state[target]["core_0"]

    def measure(value):
        estimate = types.SimpleNamespace(
            state=copy_probe_state(session.state),
            target={target: [0]},
            gradients=lambda: {key: (torch.full_like(core, value).double(), 0.0)},
        )
        accumulate_write_estimate(session, estimate)
        proposal = copy_probe_state(session.state)
        proposal[target]["core_0"] -= 0.001 * estimate.mean_gradients[key][0]
        record = GenerationRecord()
        record.extra["render_probe"] = object()
        return stage_probe_write(session, record, proposal, session.editor_snapshot())

    first = measure(1.0)
    first.refresh = True
    first.scale = 0.5
    first.attempts = 1
    updated = measure(3.0)
    assert len(session._probe_writes) == 1
    assert updated.scale == 0.5
    expected = session.state[target]["core_0"] - 0.002
    torch.testing.assert_close(updated.proposed[target]["core_0"], expected)
    assert session._write_estimates[target][1] == 2
    updated.refresh, updated.dropped = True, True
    replacement = measure(5.0)
    assert replacement.scale == 1 and not replacement.dropped
    assert len(session._probe_writes) == 1
    assert session._write_estimates[target][1] == 3
    SESSIONS.clear()
