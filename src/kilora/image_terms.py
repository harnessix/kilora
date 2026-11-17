# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import fnmatch
import math
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .prediction_write import Term

SOURCE = "images"


class _ReadRecorder:
    prediction_set_key = None

    def __init__(self, frames: Mapping[str, Sequence[Any]], mask: torch.Tensor | None = None):
        self.frames = frames
        self.mask = mask
        self.sigma: float | None = None
        self.sums: dict[tuple[str, int], list] = {}

    def wants(self, target: str) -> bool:
        return False

    def observe(self, target: str, y: torch.Tensor) -> None:
        return None

    def wants_output(self, target: str) -> bool:
        return target in self.frames

    def set_sigma(self, sigma: float | None) -> None:
        self.sigma = None if sigma is None else float(sigma)

    @staticmethod
    def _rows(value: torch.Tensor) -> torch.Tensor:
        return value.detach().reshape(-1, value.shape[-1]).double()

    def _weights(self, o: torch.Tensor) -> torch.Tensor | None:
        from .regions import token_weights

        if o.dim() < 3 or o.shape[0] == 0:
            return None
        batch = o.shape[0]
        tokens = o[..., 0].reshape(batch, -1).shape[1]
        mask = self.mask.float()
        if mask.shape[0] != batch:
            mask = mask[:1].expand(batch, *mask.shape[1:])
        w = token_weights(mask, tokens)
        return None if w is None else w.reshape(-1, 1).double()

    def observe_output(self, target: str, x, h, o) -> None:
        if target not in self.frames:
            return
        with torch.no_grad():
            out = self._rows(o)
            if out.shape[0] == 0:
                return
            weights = None
            if self.mask is not None:
                weights = self._weights(o)
                if weights is None:
                    return
            for node, frame in enumerate(self.frames[target]):
                source = self._rows(x if getattr(frame, "reads", "input") == "input" else h)
                if source.shape[0] != out.shape[0]:
                    continue
                p_in = frame.p_in.to(source.device).double()
                p_out = frame.p_out.to(out.device).double()
                z_rows, y_rows = source @ p_in.t(), out @ p_out.t()
                if weights is not None:
                    w = weights.to(out.device)
                    z_rows, y_rows = z_rows * w, y_rows * w
                z = z_rows.sum(0).cpu()
                y = y_rows.sum(0).cpu()
                entry = self.sums.get((target, node))
                if entry is None:
                    self.sums[(target, node)] = [z, y, out.shape[0]]
                else:
                    entry[0] = entry[0] + z
                    entry[1] = entry[1] + y
                    entry[2] += out.shape[0]

    def reads(self) -> dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, int]]:
        return {key: (z / n, y, n) for key, (z, y, n) in self.sums.items()}


def _blocks(model) -> list:
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        blocks = getattr(model, "blocks", None)
    return list(blocks) if blocks is not None else []


def _device(model):
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def replay(model, call: Mapping[str, Any]) -> None:
    from .library import contrast

    device = _device(model)
    call = contrast._map_tensors(dict(call), lambda t: t.to(device))
    blocks = _blocks(model)
    if not blocks:
        model(**call)
        return
    inputs = contrast._capture(model, blocks[0], call)
    for block in blocks:
        output = contrast._run_block(block, inputs)
        inputs = contrast._advance(inputs, contrast._tensor_output(output))


def _recorded_reads(model, frames, call, sigma, mask=None):
    from .recording import _ACTIVE

    recorder = _ReadRecorder(frames, mask)
    recorder.set_sigma(sigma)
    token = _ACTIVE.set(recorder)
    try:
        replay(model, call)
    finally:
        _ACTIVE.reset(token)
    return recorder.reads()


def _sigma(call: Mapping[str, Any]) -> float | None:
    for name in ("sigma", "sigmas", "timestep", "timesteps", "t"):
        value = call.get(name)
        if isinstance(value, torch.Tensor) and value.numel():
            return float(value.reshape(-1)[0])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _session_parts(session, model, editor):
    model = model if model is not None else getattr(session, "model", None)
    if model is None:
        raise ValueError("image terms need the session's model")
    if editor is None:
        editor = getattr(session, "last_editor", None)
    if editor is None or not hasattr(editor, "prediction_setup"):
        raise ValueError("image terms need an editor with a prediction setup")
    recorder = editor.recorder(session)
    frames = getattr(recorder, "frames", None) if recorder is not None else None
    setup = editor.prediction_setup(session.state)
    return model, frames or {}, setup


@torch.no_grad()
def image_terms(
    session,
    samples: Iterable[Any],
    *,
    progress: Callable[[str], None] | None = None,
    model=None,
    editor=None,
    frames: Mapping[str, Sequence[Any]] | None = None,
    responses: Mapping[tuple[str, int], tuple[Hashable, Callable]] | None = None,
    p_out: Mapping[tuple[str, int], torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
) -> list[Term]:

    from .library import contrast

    if frames is None or responses is None or p_out is None:
        found_model, found_frames, setup = _session_parts(session, model, editor)
        model = found_model
        frames = found_frames if frames is None else frames
        if setup is not None:
            responses = setup[1] if responses is None else responses
            p_out = setup[2] if p_out is None else p_out
    elif model is None:
        model = getattr(session, "model", None)
        if model is None:
            raise ValueError("image terms need the session's model")
    responses = responses or {}
    p_out = p_out or {}
    if not frames or not responses:
        return []
    training = {module: module.training for module in model.modules()}
    model.eval()
    terms: list[Term] = []
    samples = list(samples)
    try:
        for index, sample in enumerate(samples):
            image = getattr(sample, "image", index)
            if progress is not None:
                progress(f"image terms: sample {index + 1} of {len(samples)}")
            sigma = _sigma(sample.target)
            target = _recorded_reads(model, frames, sample.target, sigma, mask)
            student = _recorded_reads(model, frames, sample.student, sigma, mask)
            for key in sorted(student, key=lambda k: (str(k[0]), k[1])):
                if key not in target or key not in responses or key not in p_out:
                    continue
                z_gen, y_gen, count = student[key]
                _, y_img, count_img = target[key]
                if y_img.shape != y_gen.shape or count_img != count:
                    continue
                node, response = responses[key]
                frame = p_out[key].double()
                a = frame @ response(z_gen.double())
                if a.shape[0] != y_gen.shape[0]:
                    continue
                error = contrast.mean_contrast(y_img.double(), y_gen.double(), count)
                terms.append(Term(SOURCE, node, error, -a, 1.0, (image, index)))
    finally:
        for module, mode in training.items():
            module.train(mode)
    return terms


def image_terms_note(terms: Iterable[Term]) -> str:
    terms = list(terms)
    images, samples, nodes = set(), set(), set()
    sums: dict[str, list[float]] = {}
    for term in terms:
        step = term.step
        if isinstance(step, tuple) and len(step) == 2:
            images.add(step[0])
            samples.add(step[1])
        else:
            samples.add(step)
        nodes.add(term.node)
        error = torch.as_tensor(term.error).double()
        entry = sums.setdefault(term.source, [0.0, 0])
        entry[0] += float(error.square().sum())
        entry[1] += error.numel()
    parts = [
        f"images={len(images) or len(samples)}",
        f"samples={len(samples)}",
        f"mounts={len(nodes)}",
    ]
    for name in sorted(sums):
        total, count = sums[name]
        parts.append(f"mse_{name}={(total / count if count else 0.0):.4g}")
    return "image terms: " + " ".join(parts) + "."


HOLDOUT_EVERY = 5
RIGHTS_NOTE = "You are responsible for the rights to the images you use."
SINGLE_IMAGE_WARNING = (
    "Warning: single-image mode has no held-out images; the result can copy the "
    "content of the image."
)


def parse_mounts(text: str | Sequence[str] | None) -> list[str]:
    if text is None:
        return []
    parts = text.replace(",", "\n").splitlines() if isinstance(text, str) else list(text)
    return [part.strip() for part in parts if part and part.strip()]


def mount_matches(name: Hashable, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(str(name), pattern) for pattern in patterns)


def step_time(step: int, steps: int) -> float:
    return 1.0 if steps <= 1 else 1.0 - step / (steps - 1)


def sampled_steps(steps: int, t_min: float = 0.0, t_max: float = 1.0) -> tuple[int, ...]:
    t_min, t_max = float(t_min), float(t_max)
    if not (0.0 <= t_min <= t_max <= 1.0):
        raise ValueError("t_min and t_max must be numbers from 0 to 1, with t_min <= t_max")
    inside = [k for k in range(steps) if t_min <= step_time(k, steps) <= t_max]
    if not inside:
        raise ValueError(
            f"No sampler step of {steps} has a time from {t_min:g} to {t_max:g}; widen the range"
        )
    return tuple(sorted({inside[0], inside[(len(inside) - 1) // 2], inside[-1]}))


def holdout_indices(count: int) -> list[int]:
    if count < 3:
        raise ValueError("Supply at least 3 images so at least 2 remain after holding out images")
    wanted = max(1, round(count / HOLDOUT_EVERY))
    held = list(range(HOLDOUT_EVERY - 1, count, HOLDOUT_EVERY))[:wanted]
    if len(held) < wanted and count - 1 not in held:
        held.append(count - 1)
    if count - len(held) < 2:
        raise ValueError("At least 2 images must remain after holding out images")
    return held


class _Tagged:
    def __init__(self, sample: Any, image: int):
        self.target, self.student, self.image = sample.target, sample.student, image


@dataclass
class LearnReport:
    images: int
    held_out: list[int]
    fitted: list[int]
    terms: int
    without: float = float("nan")
    with_change: float = float("nan")
    drift: float | None = None
    drift_bound: float | None = None
    passed: bool = False
    committed: bool = False
    warning: str = ""
    single_image: bool = False
    branch: str | None = None
    norm: float = 0.0
    equivalent_writes: float = 0.0
    note: str = ""
    seconds: float = 0.0
    message: str = ""

    @property
    def relative_gain(self) -> float:
        if not math.isfinite(self.without) or not math.isfinite(self.with_change):
            return float("nan")
        return (self.without - self.with_change) / max(abs(self.without), 1e-30)

    def text(self) -> str:
        if self.single_image:
            lines = [
                f"Images: {self.images} (single-image mode, none held out)",
                f"Terms: {self.terms}",
                "Held-out loss: not measured (single-image mode)",
            ]
        else:
            lines = [
                f"Images: {self.images} ({len(self.fitted)} learned from, "
                f"{len(self.held_out)} held out: "
                f"{', '.join(str(i + 1) for i in self.held_out)})",
                f"Terms: {self.terms}",
                f"Held-out loss: without change {self.without:.8g}, with change "
                f"{self.with_change:.8g}, relative gain {self.relative_gain:.2%}",
            ]
        if self.drift is None:
            lines.append("Drift on unrelated prompts: not measured")
        else:
            lines.append(
                f"Drift on unrelated prompts: {self.drift:.4g} (bound {self.drift_bound:.4g})"
            )
        if self.message:
            lines.append(self.message)
        if self.warning:
            lines.append(self.warning)
        if self.committed:
            where = f" on branch {self.branch}" if self.branch else ""
            lines.append(
                f"Committed{where}: write norm {self.norm:.4g}, "
                f"{self.equivalent_writes:.3g} equivalent writes."
            )
        else:
            lines.append("Not committed.")
        lines.append(f"Time: {self.seconds:.3f} s")
        lines.append(RIGHTS_NOTE)
        return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def equivalent_writes(session, norm: float) -> float:
    norms = [n for n in getattr(session, "write_norms", None) or [] if n > 0]
    if not norms or not norm > 0:
        return 1.0 if norm > 0 else 0.0
    norms = sorted(norms)
    middle = len(norms) // 2
    median = norms[middle] if len(norms) % 2 else (norms[middle - 1] + norms[middle]) / 2
    return float(norm) / median


def _commit(session, editor, new, note, images):
    from .live import LearnResult

    before = session.editor_snapshot() if hasattr(session, "editor_snapshot") else None
    state = getattr(session, "editor_state", None)
    branch = None
    if state is not None and hasattr(state, "fork") and hasattr(state, "checkout"):
        branch = state.fork()
        state.checkout(branch)
    session.pending = {
        t: {p: v.detach().cpu().clone() for p, v in parts.items()} for t, parts in new.items()
    }
    session._pending_info = LearnResult(images, 1.0, 0.0, False, False, note.strip())
    session.revision += 1
    session.commit(editor_before=before)
    if editor is not None and hasattr(editor, "note"):
        editor.note = note
    return branch


@torch.no_grad()
def learn_from_images(
    session,
    items: Sequence[Any],
    cue: str,
    settings: Any,
    *,
    ignore_guard: bool = False,
    progress: Callable[[str], None] | None = None,
    samples: Callable[[Sequence[Any], str, Any], Sequence[Any]] | None = None,
    losses: Callable[[Sequence[Any], str, Any, Any], Sequence[float]] | None = None,
    drift: Callable[[Any], float] | None = None,
    drift_bound: float | None = None,
    editor=None,
    model=None,
    mounts: str | Sequence[str] | None = None,
    axis: str | None = None,
    mask: torch.Tensor | None = None,
    single_image: bool = False,
) -> LearnReport:

    from . import prediction_write as pw
    from .live import DEFAULT_DRIFT_THRESHOLD

    started = time.perf_counter()
    items = list(items)
    if single_image:
        if len(items) != 1:
            raise ValueError("Single-image mode needs exactly 1 image")
        held = []
    else:
        held = holdout_indices(len(items))
    fitted = [i for i in range(len(items)) if i not in held]
    report = LearnReport(len(items), held, fitted, 0, single_image=bool(single_image))
    patterns = parse_mounts(mounts)
    if axis == "all":
        axis = None
    if samples is None or (losses is None and not single_image):
        raise ValueError("learning from images needs a sample builder and a loss function")
    if editor is None:
        editor = getattr(session, "last_editor", None)
    if editor is None or not hasattr(editor, "prediction_setup"):
        raise ValueError("learning from images needs an editor with a prediction setup")

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    state = session.state
    setup = editor.prediction_setup(state)
    if setup is None:
        raise ValueError("the editor has no prediction frames for this live state")
    by_target, responses, p_out, priors = setup
    if axis is not None:
        axes = editor.frame_axes(state) if hasattr(editor, "frame_axes") else None
        if axes is None:
            raise ValueError("the editor has no axes for its frames")
        responses = {key: value for key, value in responses.items() if axes.get(key) == axis}
    if patterns:
        responses = {
            key: value for key, value in responses.items() if mount_matches(key[0], patterns)
        }
        priors = {node: prior for node, prior in priors.items() if mount_matches(node, patterns)}
    tagged: list[_Tagged] = []
    for number, index in enumerate(fitted):
        say(f"learn from images: capture image {number + 1} of {len(fitted)}")
        tagged += [_Tagged(sample, index) for sample in samples([items[index]], cue, settings)]
    routes = _route_record(session)
    with _recording_routes(session, routes):
        terms = image_terms(
            session,
            tagged,
            progress=progress,
            model=model,
            editor=editor,
            frames=_frames(session, editor),
            responses=responses,
            p_out=p_out,
            mask=mask,
        )
    if routes is not None and not routes.empty:
        from .live_experts import mask_terms, record_history, write_target

        target = write_target(session, routes, state)
        record_history(session, routes)
        terms = mask_terms(terms, by_target, target) if target else []
    if patterns:
        terms = [term for term in terms if mount_matches(term.node, patterns)]
    report.terms = len(terms)
    result = pw.solve_map(terms, priors, pw.session_weights(session))
    applied = editor.apply_prediction(state, by_target, result) if terms else None
    if applied is None:
        report.message = "No change: the images gave no terms for the live state."
        report.seconds = time.perf_counter() - started
        return report
    new, _written, norm = applied
    report.norm = float(norm)
    trial = editor.guard_trial(state, new)
    if not single_image:
        held_items = [items[i] for i in held]
        say("learn from images: held-out loss")
        report.without = _mean(losses(held_items, cue, settings, None))
        report.with_change = _mean(losses(held_items, cue, settings, trial))
    bound = DEFAULT_DRIFT_THRESHOLD if drift_bound is None else float(drift_bound)
    if drift is not None:
        say("learn from images: drift on unrelated prompts")
        report.drift, report.drift_bound = float(drift(trial)), bound
    loss_ok = single_image or (
        math.isfinite(report.without)
        and math.isfinite(report.with_change)
        and report.with_change < report.without
    )
    drift_ok = report.drift is None or (math.isfinite(report.drift) and report.drift <= bound)
    report.passed = loss_ok and drift_ok
    ignored = False
    if single_image:
        report.warning = SINGLE_IMAGE_WARNING
    if not report.passed:
        why = []
        if not loss_ok:
            why.append("the held-out loss did not go down")
        if not drift_ok:
            why.append("the drift on unrelated prompts is over the bound")
        report.message = "Guard failed: " + " and ".join(why) + "."
        if not ignore_guard:
            report.seconds = time.perf_counter() - started
            return report
        ignored = True
        ignored_line = "Warning: the guard failed; ignore_guard commits the write anyway."
        report.warning = "\n".join(filter(None, [report.warning, ignored_line]))
    report.equivalent_writes = equivalent_writes(session, report.norm)
    controls = "".join(
        [
            " mode=single" if single_image else "",
            f" mounts={'|'.join(patterns)}" if patterns else "",
            f" axis={axis}" if axis is not None else "",
            " mask=on" if mask is not None else "",
        ]
    )
    report.note = (
        f"source={SOURCE} images={len(fitted)} held_out={len(held)} terms={len(terms)}"
        f" loss_without={report.without:.6g} loss_with={report.with_change:.6g}"
        f" relative_gain={report.relative_gain:.4g}"
        + ("" if report.drift is None else f" drift={report.drift:.4g}")
        + f" equivalent_writes={report.equivalent_writes:.4g}"
        + (" guard=ignored" if ignored else " guard=passed")
        + controls
        + result.note(report.norm).replace(" source=prediction", "")
    )
    report.branch = _commit(session, editor, new, report.note, len(fitted))
    report.committed = True
    norms = getattr(session, "write_norms", None)
    if isinstance(norms, list):
        norms.append(report.norm)
    else:
        with contextlib.suppress(AttributeError):
            session.write_norms = [report.norm]
    report.seconds = time.perf_counter() - started
    return report


def _frames(session, editor):
    recorder = editor.recorder(session) if hasattr(editor, "recorder") else None
    return (getattr(recorder, "frames", None) if recorder is not None else None) or {}


def _route_record(session):
    if not hasattr(session, "_source") or not hasattr(session, "_section_pairs"):
        return None
    from .live_experts import RouteRecord

    return RouteRecord()


def _recording_routes(session, routes):
    if routes is None:
        return contextlib.nullcontext()
    from .live_experts import recording_routes

    return recording_routes(session, routes)
