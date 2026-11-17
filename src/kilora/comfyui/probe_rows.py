# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..write_estimates import copy_write_state as copy_probe_state
from ..write_estimates import same_write_state as same_probe_state


def probe_step(schedule: Sequence[float] | torch.Tensor, sampling: Any, session: Any) -> int | None:

    from .live_write import flow_pairs, flow_seed

    candidates = [
        (i, float(s)) for i, s in enumerate(schedule) if math.isfinite(float(s)) and float(s) > 0
    ]
    if not candidates:
        return None
    pairs, _ = flow_pairs(sampling, (1,), 1, 0, flow_seed(session))
    sigma = pairs[0][0]
    return min(candidates, key=lambda item: abs(item[1] - sigma))[0]


def probe_capacity(draws: int, row_bytes: int, budget_bytes: int | None) -> int:

    if draws < 0 or row_bytes < 1:
        raise ValueError("draws must be nonnegative and row_bytes positive")
    available = (
        draws if budget_bytes is None else min(draws, max(budget_bytes, 0) // (2 * row_bytes))
    )
    return int(available // 2 * 2)


def paired_noise(shape, *, seed: int, draw: int, device="cpu", dtype=torch.float32):
    if draw < 0:
        raise ValueError("draw must be nonnegative")
    generator = torch.Generator(device=device).manual_seed((int(seed) + draw // 2) % (2**63 - 1))
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return noise if draw % 2 == 0 else -noise


@dataclass(frozen=True)
class ProbeOutputs:
    seed: int
    sigma: float
    x_t: torch.Tensor
    clean: torch.Tensor
    draws: torch.Tensor
    peak_bytes: int = 0

    @classmethod
    def capture(cls, *, seed, sigma, x_t, clean, draws, peak_bytes=0):
        if not math.isfinite(float(sigma)) or sigma <= 0:
            raise ValueError("probe sigma must be finite and positive")
        expected = (2, *x_t.shape)
        if tuple(clean.shape) != expected:
            raise ValueError("clean outputs must contain both captions")
        if draws.ndim != clean.ndim + 1 or tuple(draws.shape[1:]) != expected:
            raise ValueError("draw outputs must contain both captions and match the input")
        if draws.shape[0] < 2 or draws.shape[0] % 2:
            raise ValueError("probe draws must contain complete pairs")

        def keep(value):
            return value.detach().to(device="cpu", copy=True)

        return cls(int(seed), float(sigma), keep(x_t), keep(clean), keep(draws), int(peak_bytes))

    def matches(self, session, latent) -> bool:
        from .live_write import flow_seed

        return self.seed == flow_seed(session) and tuple(latent.shape) == tuple(self.x_t.shape)

    def reflow_noise(self, final_latent):
        x0 = final_latent.detach().to(device="cpu", dtype=torch.float32)
        if x0.shape != self.x_t.shape:
            raise ValueError("final latent must match the probe input")
        return (self.x_t.float() - (1 - self.sigma) * x0) / self.sigma

    def losses(self, final_latent):

        x0 = final_latent.detach().to(device="cpu", dtype=torch.float32)
        if x0.shape != self.x_t.shape:
            raise ValueError("final latent must match the probe input")
        clean = (self.clean.float() - x0).square().mean() / self.sigma**2
        axes = tuple(range(1, self.draws.ndim))
        losses = (self.draws.float() - x0).square().mean(dim=axes) / self.sigma**2
        return float(clean), losses


@dataclass
class DeferredProbeStep:
    candidate: Any
    scale: float = 1.0
    last_render: int = -1
    accepted: bool = False
    attempts: int = 0
    note: str = "probe acceptance pending"

    def decide(self, render: int, clean_loss: float, candidate_loss: float, commit) -> str:
        if self.accepted:
            return self.note
        if render <= self.last_render:
            raise ValueError("probe acceptance needs a new render")
        self.last_render = int(render)
        self.attempts += 1
        finite = math.isfinite(clean_loss) and math.isfinite(candidate_loss)
        if finite and candidate_loss < clean_loss:
            commit(self.candidate, self.scale)
            self.accepted = True
            self.note = (
                f"probe acceptance committed at render {render}: "
                f"{clean_loss:.6g} -> {candidate_loss:.6g}, scale {self.scale:g}"
            )
        else:
            self.scale *= 0.5
            self.note = (
                f"probe acceptance deferred at render {render}: "
                f"{clean_loss:.6g} -> {candidate_loss:.6g}; "
                f"retry scale {self.scale:g} on the next render"
            )
        return self.note


@dataclass(frozen=True)
class ProbeLayout:
    image_rows: int
    draws: int

    def __post_init__(self):
        if self.image_rows < 1 or self.draws < 2 or self.draws % 2:
            raise ValueError("probe layout needs image rows and complete draw pairs")

    @property
    def total_rows(self):
        return self.image_rows * (self.draws + 1)

    def expand(self, value):
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == self.image_rows:
                return torch.cat([value] * (self.draws + 1), dim=0)
            return value
        if isinstance(value, dict):
            return {key: self.expand(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.expand(item) for item in value)
        if isinstance(value, list):
            return [self.expand(item) for item in value]
        return value

    def split(self, output):
        if output.shape[0] != self.total_rows:
            raise ValueError("model output does not match the probe row layout")
        image = output[: self.image_rows]
        draws = output[self.image_rows :].reshape(self.draws, self.image_rows, *output.shape[1:])
        return image, draws

    def image_values(self, value):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.total_rows:
            return value[: self.image_rows]
        return value

    def copy_routes(self, routes):
        if routes.shape[0] != self.total_rows:
            raise ValueError("routes do not match the probe row layout")
        return torch.cat([routes[: self.image_rows]] * (self.draws + 1), dim=0)


def isolated_probe_call(invoke, *args, device=None, **kwargs):

    from .. import (
        ablation_settings,
        axis_owner,
        call_values,
        compute,
        depth_map,
        fused,
        linear_jump,
        recording,
        route_freeze,
    )

    context = contextvars.copy_context()

    def clear_records():
        names = (
            "_READ_KEEP_OBSERVERS",
            "_READ_KEEP_DETAIL_OBSERVERS",
            "_READ_KEEP_RAW_OBSERVERS",
            "_BUDGET_RECORDS",
            "_LAYER_TIMERS",
            "_GATE_STATS",
            "_STEP_CARRY",
            "_DERIVED_CARRY",
            "_DERIVED_FIRST_BLOCK",
            "_SPECULATE",
            "_PROBES",
            "_ROUTES",
            "_DIRECTION_SHARES",
            "_SHARES",
            "_SET_SHARES",
            "_SET_SHARE_SCOPES",
            "_FUSED_MASS",
        )
        for name in names:
            getattr(compute, name)._var.set(None)
        for name in ("_CACHES", "RECORDS", "_FOLD_COUNTS", "_RECORD_CALLS"):
            getattr(route_freeze, name)._var.set(None)
        for name in ("_ALPHA_CACHES", "_HOST_SCALARS"):
            stack = getattr(call_values, name)
            entries = stack.get()
            stack._var.set(tuple({} for _ in entries) if entries else None)
        recording._ACTIVE.set(None)
        recording._PERTURB.set(None)
        recording._BLOCK_OBSERVERS._var.set(None)
        axis_owner._current.set(None)
        depth_map._MEASURING._var.set(None)
        linear_jump._RECORDS._var.set(None)
        ablation_settings._ACTIVE.set(None)
        fused._DIAGNOSTICS.set(False)
        _RENDER_REQUEST.set(None)

    context.run(clear_records)
    devices = []
    if device is not None and torch.device(device).type == "cuda":
        selected = torch.device(device).index
        devices = [torch.cuda.current_device() if selected is None else selected]
    with torch.random.fork_rng(devices=devices):
        return context.run(invoke, *args, **kwargs)


@dataclass
class SeparateProbeCall:
    trials: Sequence[Any]
    at_step: Any
    outputs: Any = None
    reference_difference: float | None = None
    failure: str | None = None
    image_scope: Any = None
    after_image: Any = None
    peak_bytes: int = 0
    probe_args: Any = None
    reference_trial: Any = None
    reuse_image_reference: bool = False
    reference_image: Any = None
    row_count: int = 0
    _running: bool = False

    def call(self, invoke, args):
        if (
            self._running
            or self.outputs is not None
            or self.failure is not None
            or not self.at_step(args)
        ):
            return invoke(args)
        self._running = True
        try:
            return self._collect(invoke, args)
        finally:
            self._running = False

    def _collect(self, invoke, args):
        from ..recording import perturbing

        device = args["input"].device
        before_cpu = torch.random.get_rng_state()
        before_device = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        import contextlib

        scope = self.image_scope(args) if self.image_scope is not None else contextlib.nullcontext()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with scope:
            image = invoke(args)
        try:
            if self.after_image is not None:
                self.after_image(args, image)
            probe_arguments = args if self.probe_args is None else self.probe_args(args)
        except torch.cuda.OutOfMemoryError:
            self.failure = "probe rows did not fit; use ordinary write samples"
            if device.type == "cuda":
                self.peak_bytes = int(torch.cuda.max_memory_allocated(device))
            return image
        if probe_arguments is None:
            self.failure = "probe caption rows could not be joined; use ordinary write samples"
            if device.type == "cuda":
                self.peak_bytes = int(torch.cuda.max_memory_allocated(device))
            return image

        def probe(trial):
            torch.random.set_rng_state(before_cpu)
            if before_device is not None:
                torch.cuda.set_rng_state(before_device, device)
            begin = getattr(trial, "begin_probe", None)
            if begin is not None:
                begin()
            try:
                with perturbing(trial), torch.no_grad():
                    result = invoke(probe_arguments).detach().to(device="cpu", copy=True)
            except BaseException:
                abort = getattr(trial, "abort_probe", None)
                if abort is not None:
                    abort()
                raise
            end = getattr(trial, "end_probe", None)
            if end is not None:
                end()
            return result

        self.row_count = probe_arguments["input"].shape[0] * len(self.trials)

        def report(reference):
            comparable = (
                image if self.reference_image is None else self.reference_image(args, image)
            )
            if comparable is None:
                difference = float("nan")
            else:
                difference = float(
                    (reference[: comparable.shape[0]].float() - comparable.detach().cpu().float())
                    .abs()
                    .max()
                )
            self.reference_difference = difference
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            logging.getLogger(__name__).info(
                "Render probe: sigma=%g rows=%d peak_bytes=%d reference_difference=%g",
                float(args["timestep"].reshape(-1)[0]) if "timestep" in args else 0.0,
                probe_arguments["input"].shape[0],
                peak,
                difference,
            )

        try:
            image_clean = (
                image if self.reference_image is None else self.reference_image(args, image)
            )
            reused = (
                self.reuse_image_reference
                and self.reference_trial is None
                and image_clean is not None
            )
            reference = (
                image_clean.detach().to(device="cpu", copy=True)
                if reused
                else isolated_probe_call(probe, self.reference_trial, device=device)
            )
            if not reused:
                report(reference)
            draws = []
            for trial in self.trials:
                draws.append(isolated_probe_call(probe, trial, device=device))
                report(reference)
        except torch.cuda.OutOfMemoryError:
            self.failure = "probe rows did not fit; use ordinary write samples"
            return image
        finally:
            if device.type == "cuda":
                self.peak_bytes = int(torch.cuda.max_memory_allocated(device))
        comparable = image if self.reference_image is None else self.reference_image(args, image)
        if comparable is None:
            self.reference_difference = float("nan")
        else:
            image_reference = reference[: comparable.shape[0]]
            difference = image_reference.float() - comparable.detach().cpu().float()
            self.reference_difference = float(difference.abs().max())
        self.outputs = (reference, draws)
        return image


class ForwardProbeDraw:
    def __init__(self, estimate, index):
        self.estimate, self.index = estimate, int(index)

    def begin_probe(self):
        self.estimate.begin_rows([(0, self.index)])

    def end_probe(self):
        self.estimate.end_rows()
        self.estimate._rows = None

    def abort_probe(self):
        self.estimate.drop_rows()
        self.estimate._rows = None

    def delta(self, target, part):
        return None

    def input_delta(self, target, x):
        return self.estimate.input_delta(target, x)


class ForwardRenderProbe:
    def __init__(self, session, at_step, *, draws=None, empty_conditioning=None):
        import contextlib

        from ..forward_estimate import ForwardEstimate, write_seed
        from ..live_experts import RouteRecord, recording_fused_routes, setting, write_target
        from ..recording import perturbing
        from .live_write import flow_seed

        self.session = session
        self.seed = flow_seed(session)
        self.routes = RouteRecord()
        count = setting(session, "forward_draws") if draws is None else int(draws)
        limit = setting(session, "probe_draws")
        if limit:
            count = min(count, limit)
        budget = setting(session, "probe_budget")
        self.single_caption = 2 <= budget < 4
        caption_serial = int(getattr(session, "_probe_caption_serial", write_seed(session)))
        self.caption_kind = caption_serial % 2
        count = min(count, 2 if self.single_caption else budget // 2)
        count = count // 2 * 2
        if count < 2:
            raise ValueError("render probes need at least two draws")
        self.estimate = ForwardEstimate(
            session,
            copy_probe_state(session.state),
            draws=count,
            sigma=setting(session, "forward_sigma"),
            chunk=1,
            seed=write_seed(session) + int(getattr(session, "_probe_serial", 0)),
        )
        session._probe_serial = int(getattr(session, "_probe_serial", 0)) + 1
        self.estimate.antithetic = True
        self.x_t = None
        self.sigma = None
        self.kinds = None
        self.used = False
        self.frames = []
        self.next_draw = 0
        self.started = False
        self.peak_bytes = 0
        self.row_count = 0

        def eligible(args):
            sigma = float(args["timestep"].reshape(-1)[0])
            if not math.isfinite(sigma) or sigma <= 0:
                return False
            kinds = args.get("cond_or_uncond")
            allowed = ([0, 1], [1, 0]) if empty_conditioning is None else ([0, 1], [1, 0], [0])
            if kinds is None or list(kinds) not in allowed:
                return False
            if not self.started:
                batch = args["input"].shape[0] // len(kinds)
                captions = 1 if budget < 4 else 2
                capacity = min(self.estimate.draws, budget // (batch * captions)) // 2 * 2
                if capacity < 2:
                    return False
                if capacity < self.estimate.draws:
                    self.estimate.draws = capacity
                    self.all_trials = self.all_trials[:capacity]
                    self.request.trials = self.all_trials if capacity == 2 else self.all_trials[:2]
            return self.pending is not None or self.started or at_step(args)

        @contextlib.contextmanager
        def image_scope(args):
            self.estimate.begin_rows([(0, "clean")])
            with perturbing(self.estimate), recording_fused_routes(session, self.routes):
                yield
            self.estimate._rows = None

        def after_image(args, image):
            if not self.started:
                self.estimate.add_keeps(self.routes)
                self.estimate.set_target(write_target(session, self.routes), self.routes)
            self.x_t = args["input"].detach().cpu().clone()
            self.sigma = float(args["timestep"].reshape(-1)[0])
            self.kinds = list(args["cond_or_uncond"])

        def probe_args(args):
            if list(args["cond_or_uncond"]) != [0]:
                return args
            from ..call_values import current_axis_alphas, current_crop_window, current_row_axes
            from .live_write import batch_conditions

            if current_row_axes() is not None or current_crop_window() is not None:
                return None
            if any(
                isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[0] > 1
                for v in current_axis_alphas().values()
            ):
                return None
            empty = empty_conditioning(args)
            if empty is None:
                return None
            x = args["input"]
            caption = args["c"]
            empty = {
                **{k: v for k, v in caption.items() if not isinstance(v, torch.Tensor)},
                **empty,
            }
            empty = {
                k: v.to(caption[k])
                if isinstance(v, torch.Tensor) and isinstance(caption.get(k), torch.Tensor)
                else v
                for k, v in empty.items()
            }
            joined = batch_conditions(caption, empty, x.shape[0])
            if joined is None:
                return None
            if "transformer_options" in joined:
                joined["transformer_options"] = {
                    **joined["transformer_options"],
                    "cond_or_uncond": [0, 1],
                }
            timestep = args["timestep"].reshape(-1)
            if timestep.numel() == 1:
                timestep = timestep.expand(x.shape[0])
            expanded = {
                **args,
                "input": torch.cat([x, x]),
                "timestep": torch.cat([timestep, timestep]),
                "c": joined,
                "cond_or_uncond": [0, 1],
            }
            self.x_t = expanded["input"].detach().cpu().clone()
            self.kinds = [0, 1]
            self.request.reference_trial = ForwardProbeReference(self)
            return expanded

        paired_probe_args = probe_args

        def selected_probe_args(args):
            expanded = paired_probe_args(args)
            if expanded is None or not self.single_caption:
                return expanded
            rows = expanded["input"].shape[0]
            kinds = list(expanded["cond_or_uncond"])
            index = kinds.index(self.caption_kind)
            per_kind = rows // 2

            def select(value):
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == rows:
                    return value[index * per_kind : (index + 1) * per_kind]
                if isinstance(value, dict):
                    return {key: select(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [select(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(select(item) for item in value)
                return value

            selected = select(expanded)
            selected["cond_or_uncond"] = [self.caption_kind]
            if "transformer_options" in selected.get("c", {}):
                selected["c"]["transformer_options"]["cond_or_uncond"] = [self.caption_kind]
            if self.caption_kind in list(args["cond_or_uncond"]):
                original_kinds = list(args["cond_or_uncond"])
                if len(original_kinds) == 2:
                    selected_index = original_kinds.index(self.caption_kind)
                    self.estimate.reads = {
                        key: {
                            expert: value.chunk(2)[selected_index]
                            for expert, value in reads.items()
                        }
                        for key, reads in self.estimate.reads.items()
                    }
                self.request.reference_trial = None
            else:
                self.request.reference_trial = ForwardProbeReference(self)
            return selected

        trials = [ForwardProbeDraw(self.estimate, k) for k in range(self.estimate.draws)]
        self.pending = next(
            (p for p in getattr(session, "_probe_writes", ()) if not p.ready and not p.dropped),
            None,
        )
        if self.pending is not None:
            from ..live_imports import ImportTrial

            try:
                self.pending.tested_base = copy_probe_state(session.state)
                self.pending.tested_state = self.pending.state_at(session.state)
                acceptance_trial = ImportTrial(session, session.state, self.pending.tested_state)
            except (KeyError, RuntimeError):
                self.pending.dropped = True
                self.acceptance_note = (
                    "Probe candidate dropped: incompatible state; no trial losses."
                )
                logging.getLogger(__name__).info(self.acceptance_note)
                session._probe_writes.remove(self.pending)
                self.pending = None
        session._probe_caption_serial = caption_serial + 1

        def reference_image(args, image):
            if not self.single_caption:
                return image
            kinds = list(args["cond_or_uncond"])
            if self.caption_kind not in kinds:
                return None
            return image.chunk(len(kinds))[kinds.index(self.caption_kind)]

        self.estimate.acceptance_only = False
        self.all_trials = trials
        first_trials = trials if count == 2 else trials[:2]
        self.request = SeparateProbeCall(
            first_trials,
            eligible,
            image_scope=image_scope,
            after_image=after_image,
            probe_args=selected_probe_args,
            reuse_image_reference=True,
            reference_image=reference_image,
        )
        self.accept_request = None
        if self.pending is not None:

            def acceptance_args(args):
                saved = self.x_t, self.kinds, self.request.reference_trial
                try:
                    return paired_probe_args(args)
                finally:
                    self.x_t, self.kinds, self.request.reference_trial = saved

            def acceptance_image(args, image):
                return image if len(args["cond_or_uncond"]) == 2 else None

            def acceptance_after(args, image):
                self.accept_sigma = float(args["timestep"].reshape(-1)[0])
                self.accept_kinds = list(args["cond_or_uncond"])
                if self.accept_kinds == [0]:
                    self.accept_kinds = [0, 1]

            def acceptance_eligible(args):
                kinds = list(args.get("cond_or_uncond", ()))
                sigma = float(args["timestep"].reshape(-1)[0])
                return (
                    math.isfinite(sigma)
                    and sigma > 0
                    and (
                        kinds in ([0, 1], [1, 0])
                        or (kinds == [0] and empty_conditioning is not None)
                    )
                )

            self.accept_request = SeparateProbeCall(
                [acceptance_trial],
                acceptance_eligible,
                after_image=acceptance_after,
                probe_args=acceptance_args,
                reuse_image_reference=True,
                reference_image=acceptance_image,
            )

    def call(self, invoke, args):
        if self.accept_request is not None and self.accept_request.outputs is None:
            first = True

            def image_and_estimates(arguments):
                nonlocal first
                if first:
                    first = False
                    return self._estimate_call(invoke, arguments)
                return invoke(arguments)

            return self.accept_request.call(image_and_estimates, args)
        return self._estimate_call(invoke, args)

    def _estimate_call(self, invoke, args):
        if self.next_draw >= self.estimate.draws:
            return invoke(args)
        result = self.request.call(invoke, args)
        if self.request.failure is not None:
            self.release()
        elif self.request.outputs is not None:
            if self.single_caption:
                reference, draws = self.request.outputs
                self.request.outputs = (
                    torch.cat([reference, reference]),
                    [torch.cat([draw, draw]) for draw in draws],
                )
            self.started = True
            self.row_count += self.request.row_count
            self.peak_bytes = max(self.peak_bytes, self.request.peak_bytes)
            self.frames.append((self.x_t, self.sigma, self.kinds, self.request.outputs))
            self.next_draw += 2
            if self.next_draw < self.estimate.draws:
                stop = self.next_draw + 2
                self.request.trials = self.all_trials[self.next_draw : stop]
                self.request.outputs = None
        return result

    def release(self):
        for pack in self.estimate.packs.values():
            pack.clear()
        self.estimate._device_parts = {}
        self.estimate._device_sums = {}
        self.estimate._squares = {}
        self.estimate._pending = []
        self.estimate._live_rows = {}
        self.estimate.reads = {}

    def finish(self, latent):
        if self.used:
            return None
        if self.pending is not None:
            self.peak_bytes = max(self.peak_bytes, self.accept_request.peak_bytes)
            if self.accept_request.outputs is not None:
                reference, trials = self.accept_request.outputs

                def acceptance_captions(value):
                    parts = value.chunk(2)
                    return torch.stack([parts[self.accept_kinds.index(k)] for k in (0, 1)])

                self.pending.evaluate(
                    self.session,
                    latent,
                    self.accept_sigma,
                    acceptance_captions(reference),
                    acceptance_captions(trials[0]),
                )
                self.pending.refresh = not self.pending.ready
                self.acceptance_note = self.pending.note
            else:
                reason = self.accept_request.failure or "no eligible caption call"
                self.acceptance_note = f"Probe candidate not tested: {reason}."
                logging.getLogger(__name__).info(self.acceptance_note)
        if self.x_t is None or self.routes.empty:
            self.request.failure = self.request.failure or (
                "probe not tested: no eligible caption rows, row budget, or traversed target"
            )
            logging.getLogger(__name__).info(self.request.failure)
            self.release()
            return None
        if self.next_draw < self.estimate.draws:
            self.request.failure = (
                "schedule ended before complete probe pairs; use ordinary write samples"
            )
            self.release()
            return None
        from .live_write import flow_seed

        if self.seed != flow_seed(self.session):
            self.request.failure = "probe not tested: session changed during render"
            self.release()
            return None
        reference, draws = self.request.outputs
        order = self.kinds
        chunks = self.x_t.chunk(2)
        x_t = chunks[order.index(0)]
        if tuple(latent.shape) != tuple(x_t.shape):
            self.request.failure = "probe not tested: final latent shape changed"
            self.release()
            return None

        def captions(value):
            parts = value.chunk(2)
            return torch.stack([parts[order.index(0)], parts[order.index(1)]])

        outputs = ProbeOutputs.capture(
            seed=self.seed,
            sigma=self.sigma,
            x_t=x_t,
            clean=captions(reference),
            draws=torch.stack([captions(draw) for draw in draws[:2]]),
            peak_bytes=self.peak_bytes,
        )
        clean_losses, draw_losses = [], []
        for frame_x, frame_sigma, frame_kinds, (clean, frame_draws) in self.frames:

            def ordered(value, kinds=frame_kinds):
                parts = value.chunk(2)
                return torch.stack([parts[kinds.index(0)], parts[kinds.index(1)]])

            frame = ProbeOutputs.capture(
                seed=self.seed,
                sigma=frame_sigma,
                x_t=frame_x.chunk(2)[frame_kinds.index(0)],
                clean=ordered(clean),
                draws=torch.stack([ordered(d) for d in frame_draws[:2]]),
                peak_bytes=self.peak_bytes,
            )
            clean_loss, losses = frame.losses(latent)
            clean_losses.append(clean_loss)
            draw_losses.extend(float(loss) for loss in losses)
        self.estimate.clean_loss = sum(clean_losses) / len(clean_losses)
        self.estimate.losses = draw_losses
        if not all(math.isfinite(v) for v in [self.estimate.clean_loss, *self.estimate.losses]):
            self.request.failure = "nonfinite probe loss; use ordinary write samples"
            self.release()
            return None
        self.estimate._collect_sums()
        self.estimate.reads = {}
        self.estimate.render_rows = self.row_count
        self.estimate.render_ready = True
        self.estimate.render_peak_bytes = outputs.peak_bytes
        self.estimate.render_reference_difference = self.request.reference_difference
        self.release()
        self.used = True
        return self.estimate, self.routes, outputs


_RENDER_REQUEST = contextvars.ContextVar("kilora_render_probe", default=None)


@contextlib.contextmanager
def render_probe(request):
    token = _RENDER_REQUEST.set(request)
    try:
        yield
    finally:
        _RENDER_REQUEST.reset(token)


def current_render_probe():
    return _RENDER_REQUEST.get()


@dataclass
class PendingProbeWrite:
    base: Any
    proposed: Any
    record: Any
    editor_before: Any
    editor_after: Any
    scale: float = 1.0
    ready: bool = False
    tested_base: Any = None
    tested_state: Any = None
    guard_before: Any = None
    guard_after: Any = None
    before_loss: float | None = None
    after_loss: float | None = None
    attempts: int = 0
    dropped: bool = False
    decision: str = "pending"
    note: str = "Probe candidate pending on the next render."
    refresh: bool = False
    estimate_note: str = ""

    def state_at(self, current):
        if same_probe_state(current, self.base) and self.scale == 1:
            return copy_probe_state(self.proposed)
        return {
            t: {
                k: value.detach().cpu() + self.scale * (self.proposed[t][k] - self.base[t][k])
                for k, value in parts.items()
            }
            for t, parts in current.items()
        }

    def history_at(self, current):
        result = {k: v.clone() for k, v in current.items()}
        before, after = self.editor_before or {}, self.editor_after or {}
        for key, value in after.items():
            old = before.get(key)
            if (
                old is not None
                and value.shape == old.shape
                and torch.allclose(value, old, equal_nan=True)
            ):
                continue
            if not key.startswith("prior/"):
                continue
            if key == "prior/event" and old is not None:
                result[key] = result.get(key, old) + (value - old)
            elif key.startswith("prior/rule/") and value.numel() == 5:
                row = value.clone()
                baseline = torch.zeros_like(row) if old is None else old
                existing = result.get(key, baseline)
                row[4] = existing[4] + self.scale * (value[4] - baseline[4])
                if "prior/event" in result:
                    row[3] = result["prior/event"]
                result[key] = row
            else:
                result[key] = value.clone()
        return result

    def evaluate(self, session, final_latent, sigma, before, after):
        from ..editors import guard_enabled, guard_verdict

        x0 = final_latent.detach().cpu().float()

        def scores(outputs):
            losses = (outputs.float() - x0).square().flatten(1).mean(1) / sigma**2
            return float(losses.mean()), {
                "gap": float(losses[1] - losses[0]),
                "spread": float(outputs[0].float().std()),
            }

        self.before_loss, self.guard_before = scores(before)
        self.after_loss, self.guard_after = scores(after)
        self.attempts += 1
        improves = (
            math.isfinite(self.before_loss)
            and math.isfinite(self.after_loss)
            and self.after_loss < self.before_loss
        )
        refused = False
        if improves and guard_enabled(session):
            refused, _ = guard_verdict(session, self.guard_before, self.guard_after)
        self.ready = improves and not refused
        if self.ready:
            self.decision = "accepted"
        elif self.attempts == 1:
            self.scale = 0.5
            self.decision = "retried at 0.5"
        else:
            self.dropped = True
            self.decision = "dropped"
        self.note = (
            f"Probe candidate {self.decision}: loss {self.before_loss:.9g} -> "
            f"{self.after_loss:.9g}, tested scale {1 if self.attempts == 1 else 0.5:g}"
            + (", collapse guard refused" if refused else "")
            + "."
        )
        if self.estimate_note:
            self.note += " " + self.estimate_note
        logging.getLogger(__name__).info(self.note)
        return self.ready


def stage_probe_write(session, record, proposed, editor_before):
    retained = copy.copy(record)
    retained.extra = {
        key: value
        for key, value in record.extra.items()
        if key not in ("render_probe", "probe_acceptances")
    }
    retained.measure = None
    retained.guard_score = None
    pending = PendingProbeWrite(
        copy_probe_state(session.state),
        copy_probe_state(proposed),
        retained,
        editor_before,
        session.editor_snapshot(),
    )
    estimate = record.extra.get("render_probe")
    if isinstance(estimate, tuple):
        pending.estimate_note = getattr(estimate[0], "mean_response_note", "")
    writes = getattr(session, "_probe_writes", None)
    if writes is None:
        writes = session._probe_writes = []
    for index, previous in enumerate(writes):
        if previous.refresh and same_probe_state(previous.base, session.state):
            pending.scale = 1.0 if previous.dropped else previous.scale
            pending.attempts = 0 if previous.dropped else previous.attempts
            writes[index] = pending
            logging.getLogger(__name__).info(
                "Probe candidate updated from accumulated render estimates; scale %g", pending.scale
            )
            return pending
    writes.append(pending)
    return pending


class RetainedProbeMeasure:
    def __init__(self, retained):
        _estimate, _routes, outputs = retained
        self.calls, self.rows_count = [0], [0]
        self.sample_count = 1
        self.sigmas = [outputs.sigma]
        self.samples = [(outputs.sigma, outputs.seed)]
        self.held_samples = []
        self.memory = {"peak": outputs.peak_bytes}
        self.timing = {"base_s": [], "trial_s": [], "fused": 0, "eager": 0}
        self.profile = None
        self.device = torch.device("cpu")
        self.guard_score = None

    def __call__(self, trial):
        raise RuntimeError("render probe measurements cannot call the model between renders")

    def copies(self, trial, count):
        return self(trial)


class ForwardProbeReference:
    def __init__(self, owner):
        self.owner = owner
        self.estimate = owner.estimate
        self.scope = None

    def begin_probe(self):
        from ..live_experts import RouteRecord, recording_fused_routes

        self.estimate.reads = {}
        self.estimate._squares = {}
        self.estimate._live_rows = {}
        self.estimate.packs["squares"].clear()
        self.estimate.packs["live"].clear()
        self.estimate.begin_rows([(0, "clean")])
        self.owner.routes = RouteRecord()
        self.scope = recording_fused_routes(self.owner.session, self.owner.routes)
        self.scope.__enter__()

    def end_probe(self):
        from ..live_experts import write_target

        self.scope.__exit__(None, None, None)
        self.scope = None
        self.estimate._rows = None
        self.estimate.add_keeps(self.owner.routes)
        if not self.owner.started:
            self.estimate.set_target(
                write_target(self.owner.session, self.owner.routes), self.owner.routes
            )

    def abort_probe(self):
        if self.scope is not None:
            self.scope.__exit__(None, None, None)
            self.scope = None
        self.estimate._rows = None

    def delta(self, target, part):
        return None

    def input_delta(self, target, x):
        return self.estimate.input_delta(target, x)
