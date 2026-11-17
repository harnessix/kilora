# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ..modfile import write_safetensors
from ..presets import anima_name, get_preset
from ..regions import token_weights

_GROUPS = (
    ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    ("attn1.to_out.0",),
    ("attn2.to_q",),
    ("attn2.to_k", "attn2.to_v"),
    ("attn2.to_out.0",),
    ("ff.net.0.proj",),
    ("ff.net.2",),
)
_CAPTION_MOUNTS = ("attn2.to_k", "attn2.to_v")
_EXCLUSION = "Target and student captions differ; caption tokens have no paired target."


@dataclass(frozen=True)
class PromptPair:
    target: str
    student: str
    concept: str = ""
    cue: str = ""


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...] = (811, 1811)
    size: int = 768
    steps: int = 15
    cfg: float = 1.0
    negative_prompt: str = ""
    sample_steps: tuple[int, ...] = (0, 7, 14)
    rank: int = 16
    layout_source: str = "target"
    refit_rounds: int = 0
    ridge: float = 0.1
    replay_batch: int = 8
    contrast_mode: str = "tokens"
    contrast_components: int = 0
    stats_precision: str = "tf32"
    fit_dtype: str = "float32"

    def validate(self) -> None:
        if self.contrast_mode not in ("tokens", "mean"):
            raise ValueError("contrast_mode must be tokens or mean")
        if type(self.contrast_components) is not int or not 0 <= self.contrast_components <= 4:
            raise ValueError("contrast_components must be an integer from 0 to 4")
        if self.stats_precision not in ("tf32", "float32"):
            raise ValueError("stats_precision must be tf32 or float32")
        if self.fit_dtype not in ("float32", "float64"):
            raise ValueError("fit_dtype must be float32 or float64")
        if type(self.replay_batch) is not int or self.replay_batch < 1:
            raise ValueError("replay_batch must be an integer of at least 1")
        if not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if self.layout_source not in ("target", "plain"):
            raise ValueError("layout_source must be target or plain")
        if not isinstance(self.refit_rounds, int) or self.refit_rounds < 0:
            raise ValueError("refit_rounds must be a nonnegative integer")
        if self.size < 16 or self.size % 16:
            raise ValueError("size must be a positive multiple of 16")
        if self.steps < 1 or self.rank < 1 or not self.seeds or not self.sample_steps:
            raise ValueError("steps, rank, seeds and sampled steps must be nonempty and positive")
        if len(set(self.sample_steps)) != len(self.sample_steps) or any(
            i < 0 or i >= self.steps for i in self.sample_steps
        ):
            raise ValueError("sampled steps must be unique indices within the trajectory")
        if not math.isfinite(self.cfg) or self.cfg < 0:
            raise ValueError("cfg must be finite and nonnegative")


@dataclass
class Sample:
    target: dict[str, Any]
    student: dict[str, Any]
    image_index: int | None = None
    weights: torch.Tensor | None = None


@dataclass
class MountSolution:
    down: torch.Tensor
    up: torch.Tensor
    ridge: float
    fit_share: float
    svd_fallback: bool = False
    exact_fit: bool = False

    def delta(self) -> torch.Tensor:
        return self.up.T @ self.down


@dataclass
class ContrastResult:
    mounts: dict[str, MountSolution]
    metadata: dict[str, Any]
    fit_seconds: float = 0.0


def input_basis(solution: MountSolution) -> torch.Tensor:
    down = solution.down.to(torch.float64)
    _, values, vh = torch.linalg.svd(down, full_matrices=False)
    if not values.numel() or not float(values[0]):
        return vh[:0].T
    keep = values > torch.finfo(values.dtype).eps * max(down.shape) * values[0]
    return vh[keep].T


@dataclass
class ConceptCoupling:
    overlap_penalty: float = 0.0
    deflate: bool = False
    others: Sequence[Mapping[str, MountSolution]] = ()
    earlier: Sequence[Mapping[str, MountSolution]] = ()
    order: Sequence[str] = ()
    overlap_round: int = 0

    def earlier_changes(self):
        if not self.deflate:
            return []
        return [(name, solution) for mounts in self.earlier for name, solution in mounts.items()]

    def adjust(self, name: str, c: torch.Tensor, s: torch.Tensor):
        others = [m[name] for m in self.others if name in m] if self.overlap_penalty else []
        penalty = None
        for solution in others:
            basis = input_basis(solution).to(c.device)
            term = basis @ basis.T
            penalty = term if penalty is None else penalty + term
        if penalty is None or not float(penalty.trace()) > 0:
            return c, s
        c = c.to(torch.float64)
        scale = self.overlap_penalty * float(c.trace()) / float(penalty.trace())
        return c + scale * penalty, s

    def metadata(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        if self.overlap_penalty:
            record["overlap_penalty"] = self.overlap_penalty
            record["overlap_round"] = self.overlap_round
        if self.deflate:
            record["deflate"] = True
        if record:
            record["concept_order"] = list(self.order)
        return record


def check_coupling(overlap_penalty: float, deflate: bool) -> None:
    if not isinstance(overlap_penalty, (int, float)) or isinstance(overlap_penalty, bool):
        raise ValueError("overlap_penalty must be a number")
    if not math.isfinite(overlap_penalty) or overlap_penalty < 0:
        raise ValueError("overlap_penalty must be finite and nonnegative")
    if not isinstance(deflate, bool):
        raise ValueError("deflate must be true or false")


def solve_concepts(
    labels: Sequence[str],
    solve_one: Callable[[str, ConceptCoupling | None], ContrastResult],
    *,
    overlap_penalty: float = 0.0,
    deflate: bool = False,
) -> dict[str, ContrastResult]:

    check_coupling(overlap_penalty, deflate)
    labels = list(labels)
    if len(set(labels)) != len(labels):
        raise ValueError("concept labels must be unique")
    penalty = float(overlap_penalty) if len(labels) > 1 else 0.0

    def run(round_index, previous):
        results: dict[str, ContrastResult] = {}
        for position, label in enumerate(labels):
            coupling = None
            if deflate or round_index:
                coupling = ConceptCoupling(
                    overlap_penalty=penalty if round_index else 0.0,
                    deflate=deflate,
                    others=[previous[o].mounts for o in labels if o != label]
                    if round_index
                    else (),
                    earlier=[results[o].mounts for o in labels[:position]] if deflate else (),
                    order=labels,
                    overlap_round=round_index,
                )
            results[label] = solve_one(label, coupling)
        return results

    results = run(0, {})
    if penalty:
        results = run(1, results)
    return results


@dataclass
class EffectSource:
    mounts: dict[str, MountSolution]
    metadata: dict[str, Any]
    license: str

    @classmethod
    def load(cls, model: nn.Module, path: str | Path, strength: float = 0.7):
        from ..lora_import import _read_file
        from ..multi_import import LoRASource, import_loras
        from .basis import _license_status

        if not math.isfinite(strength):
            raise ValueError("LoRA strength must be finite")

        def read_file(filename):
            state, metadata, record = _read_file(filename)
            reason, _ = _license_status(metadata)
            if reason is not None:
                raise ValueError(f"Cannot extract a derivative from {Path(path).name}: {reason}")
            return state, metadata, record

        imported = import_loras(
            [LoRASource(str(path), weight=strength)],
            model,
            family="anima",
            read_file=read_file,
            device="cpu",
        )
        if imported.adapter_set is None or not imported.adapter_set.experts:
            raise ValueError("LoRA file has no supported routed mounts")
        selected = resolve_mounts(model, list(imported.adapter_set.experts))
        weights = {
            name: MountSolution(
                imported.adapter_set.experts[name].down[0, 0].detach().float().cpu(),
                imported.adapter_set.experts[name].up[0, 0].detach().float().cpu(),
                0.0,
                1.0,
            )
            for name in selected
        }
        record = imported.sources[0]
        return cls(
            weights,
            {
                "file": record["file"],
                "sha256": record["sha256"],
                "strength": strength,
            },
            record["license"],
        )

    @contextmanager
    def applied(self, model):
        changes = _Changes(model)
        try:
            for name, solution in self.mounts.items():
                changes.add(name, solution)
            with changes.active(True):
                yield
        finally:
            changes.close()


def mean_contrast(target_sum, student_sum, count):
    if isinstance(count, torch.Tensor):
        count = count.clamp_min(torch.finfo(count.dtype).tiny)
    else:
        count = max(count, 1)
    return (target_sum - student_sum) / count


@contextmanager
def _matmul_allow_tf32(enabled: bool):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _input_gram(x: torch.Tensor) -> torch.Tensor:
    return x.T @ x


class _WeightedPair(tuple):
    def __new__(cls, target, student, weights):
        pair = super().__new__(cls, (target, student))
        pair.weights = weights
        return pair


def _carry_weights(source, target, student):
    weights = getattr(source, "weights", None)
    if weights is None:
        return (target, student)
    return _WeightedPair(target, student, weights)


def _cat_weights(weights):
    if any(item is None for item in weights):
        raise ValueError("a replay batch cannot mix weighted and unweighted samples")
    return torch.cat([item for item in weights], dim=0)


def _token_row_weights(weights, reference, *, require_image=False):
    if not isinstance(weights, torch.Tensor) or weights.ndim not in (2, 4):
        raise ValueError("token weights must have shape [batch, tokens] or [batch, 1, h, w]")
    rows = reference.reshape(-1, reference.shape[-1]).shape[0]
    if weights.ndim == 4:
        batch = weights.shape[0]
        if not batch or weights.shape[1] != 1 or rows % batch:
            raise ValueError("spatial mask batch does not match image rows")
        if reference.ndim >= 3 and reference.shape[0] != batch:
            raise ValueError("spatial mask batch does not match image rows")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("token weights must be finite")
        reduced = token_weights(weights, rows // batch)
        if reduced is None:
            if require_image:
                raise ValueError("spatial mask cannot reduce to the block image token grid")
            return None
        weights = reduced
    if weights.numel() != rows:
        raise ValueError(f"token weights {tuple(weights.shape)} do not match {rows} image rows")
    flat = weights.reshape(-1).to(dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("token weights must be finite")
    if float(flat.sum()) <= 0:
        raise ValueError("token weights cover no rows")
    return flat


def _mean_cross(totals, per_image, components):
    count, target_sum, student_sum, input_sum = totals
    mean = mean_contrast(target_sum, student_sum, count)
    cross = torch.outer(mean, input_sum)
    keep = min(components, len(per_image) - 1)
    if keep <= 0:
        return cross
    centered = torch.stack(
        [
            mean_contrast(target, student, count) - mean
            for count, target, student, _ in per_image.values()
        ]
    )
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:keep].T
    projected = (centered @ basis) @ basis.T
    for residual, (_, _, _, inputs) in zip(projected, per_image.values(), strict=True):
        cross.add_(torch.outer(residual, inputs))
    return cross


def measured_rank(solution):

    _, left = torch.linalg.qr(solution.up.T.double(), mode="reduced")
    _, right = torch.linalg.qr(solution.down.T.double(), mode="reduced")
    values = torch.linalg.svdvals(left @ right.T)
    return int((values > 1e-3 * values[0]).sum()) if values.numel() else 0


def _group_statistics(
    names, captured, observe, image_rows, contrast_mode, contrast_components=0, image_indices=None
):

    statistics = {}
    by_image = {name: {} for name in names}
    if contrast_mode == "mean" and contrast_components and image_indices is None:
        raise ValueError("per-image identities are required for contrast components")
    for sample_index, item in enumerate(captured):
        target, student = item
        sample_weights = getattr(item, "weights", None)
        desired, actual = observe(target, student)
        grams = {}
        updated = set()
        for name in names:
            x, y = actual[name]
            target_y = desired[name][1]
            x, y, target_y = image_rows(x), image_rows(y), image_rows(target_y)
            if x.shape[-2] != y.shape[-2] or y.shape[-2] != target_y.shape[-2]:
                raise ValueError(
                    f"{name}: target and student image row counts differ "
                    f"(input={x.shape[-2]}, student={y.shape[-2]}, "
                    f"target={target_y.shape[-2]})"
                )
            if y.shape != target_y.shape:
                raise ValueError(f"{name}: target and student token shapes differ")
            flat = x.reshape(-1, x.shape[-1])
            row_weights = None
            if sample_weights is None:
                key = (x.data_ptr(), x.shape, x.dtype)
            else:
                row_weights = _token_row_weights(sample_weights, x)
                key = (
                    x.data_ptr(),
                    x.shape,
                    x.dtype,
                    sample_weights.data_ptr(),
                    tuple(sample_weights.shape),
                )
            gram = grams.get(key)
            if gram is None:
                if row_weights is None:
                    gram = _input_gram(flat)
                else:
                    scaled = flat * row_weights[:, None]
                    gram = scaled.T @ flat
                grams[key] = gram
            if contrast_mode == "mean":
                if row_weights is None:
                    sums = (
                        target_y.reshape(-1, y.shape[-1]).sum(0),
                        y.reshape(-1, y.shape[-1]).sum(0),
                        flat.sum(0),
                    )
                    count = flat.shape[0]
                else:
                    scale = row_weights[:, None]
                    sums = (
                        (target_y.reshape(-1, y.shape[-1]) * scale).sum(0),
                        (y.reshape(-1, y.shape[-1]) * scale).sum(0),
                        (flat * scale).sum(0),
                    )
                    count = row_weights.sum()
                if contrast_components:
                    image_index = image_indices[sample_index]
                    per_image = by_image[name]
                    if image_index not in per_image:
                        per_image[image_index] = [count, *(value.clone() for value in sums)]
                    else:
                        image_totals = per_image[image_index]
                        image_totals[0] = image_totals[0] + count
                        for total, value in zip(image_totals[1:], sums, strict=True):
                            total.add_(value)
                if name in statistics:
                    stored, totals = statistics[name]
                    if stored is not gram and id(stored) not in updated:
                        stored.add_(gram)
                        updated.add(id(stored))
                    totals[0] = totals[0] + count
                    for total, value in zip(totals[1:], sums, strict=True):
                        total.add_(value)
                else:
                    statistics[name] = (gram, [count, *sums])
                continue
            difference = (target_y - y).reshape(-1, y.shape[-1])
            if row_weights is None:
                cross = difference.T @ flat
            else:
                cross = (difference * row_weights[:, None]).T @ flat
            if name in statistics:
                stored, total = statistics[name]
                if stored is not gram and id(stored) not in updated:
                    stored.add_(gram)
                    updated.add(id(stored))
                total.add_(cross)
            else:
                statistics[name] = (gram, cross)
    if contrast_mode == "mean":
        for name in names:
            gram, totals = statistics[name]
            statistics[name] = (gram, _mean_cross(totals, by_image[name], contrast_components))
    return statistics


@torch.no_grad()
def _ridge_cholesky(c: torch.Tensor, ridge: float) -> tuple[torch.Tensor, float]:
    c = (c + c.T) * 0.5
    ridge = max(abs(float(c.diagonal().mean())) * ridge, torch.finfo(c.dtype).eps)
    for attempt in range(7):
        regularized = c.clone()
        regularized.diagonal().add_(ridge)
        lower, info = torch.linalg.cholesky_ex(regularized)
        del regularized
        if int(info) == 0:
            return lower, ridge
        if attempt == 6:
            raise ValueError("input Gram matrix is not positive definite after ridge increases")
        del lower
        ridge *= 10


@torch.no_grad()
def fit_mount(
    c: torch.Tensor,
    s: torch.Tensor,
    rank: int,
    *,
    ridge: float = 0.1,
    cholesky_cache: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
    fit_dtype: str = "float64",
) -> MountSolution:

    if c.ndim != 2 or c.shape[0] != c.shape[1] or s.ndim != 2 or s.shape[1] != c.shape[0]:
        raise ValueError("C must be square and S must have the same input width")
    if rank < 1 or not torch.isfinite(c).all() or not torch.isfinite(s).all():
        raise ValueError("rank must be positive and statistics must be finite")
    if fit_dtype not in ("float32", "float64"):
        raise ValueError("fit_dtype must be float32 or float64")
    dtype = torch.float32 if fit_dtype == "float32" else torch.float64
    lower, applied_ridge = None, None
    for previous, cached, value in cholesky_cache or ():
        if (
            c.device == previous.device
            and c.dtype == previous.dtype
            and cached.dtype == dtype
            and (c is previous or torch.equal(c, previous))
        ):
            lower, applied_ridge = cached, value
            break
    if lower is None:
        work = c if c.dtype == dtype else c.to(dtype=dtype)
        lower, applied_ridge = _ridge_cholesky(work, ridge)
        if cholesky_cache is not None:
            cholesky_cache.append((c, lower, applied_ridge))
    s = s.to(device=c.device, dtype=dtype)
    metric = torch.linalg.solve_triangular(lower, s.T, upper=False).T
    zero = not torch.count_nonzero(s)
    del s
    rows, columns = metric.shape
    keep = min(rank, rows, columns)
    k = min(rank + 16, rows, columns)
    generator = torch.Generator(device=metric.device).manual_seed(0)
    start = torch.randn(columns, k, device=metric.device, dtype=metric.dtype, generator=generator)
    q, _ = torch.linalg.qr(metric @ start, mode="reduced")
    exact = True
    iterations = 0
    previous = None
    while iterations < 12:
        for _ in range(4 if iterations == 0 else 2):
            right, _ = torch.linalg.qr(metric.T @ q, mode="reduced")
            q, _ = torch.linalg.qr(metric @ right, mode="reduced")
            iterations += 1
        u_small, values, _ = torch.linalg.svd(q.T @ metric, full_matrices=False)
        eigenvalues = values.square()
        up = q @ u_small[:, :keep]
        change = up @ (up.T @ metric)
        if previous is not None and bool(
            torch.linalg.vector_norm(change - previous) <= 1e-4 * torch.linalg.vector_norm(change)
        ):
            exact = False
            break
        previous = change
    del previous, change
    if exact:
        gram = metric @ metric.T if rows <= columns else metric.T @ metric
        eigenvalues, vectors = torch.linalg.eigh(gram)
        del gram
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        vectors = vectors.flip(1)
    floor = torch.finfo(metric.dtype).eps * max(rows, columns) * eigenvalues[0]
    fallback = keep < len(eigenvalues) and bool(
        (eigenvalues[keep - 1] - eigenvalues[keep] < 1e-10 * eigenvalues[0])
        & (eigenvalues[keep - 1] > floor)
    )
    if fallback:
        u, values, _ = torch.linalg.svd(metric, full_matrices=False)
        eigenvalues = values.square()
        up = u[:, :keep].to(metric.device)
    elif exact and rows <= columns:
        up = vectors[:, :keep].to(metric.device)
    elif exact:
        values = eigenvalues[:keep].sqrt().to(metric.device)
        nonzero = eigenvalues[:keep] > (
            torch.finfo(metric.dtype).eps * max(rows, columns) * eigenvalues[0]
        )
        nonzero = nonzero.to(metric.device)
        up = metric @ vectors[:, :keep].to(metric.device)
        up /= torch.where(nonzero, values, torch.ones_like(values))
        up[:, ~nonzero] = 0
    down = torch.linalg.solve_triangular(lower.T, (up.T @ metric).T, upper=True).T
    energy = float(metric.square().sum())
    if keep == min(rows, columns) or not energy:
        share = 1.0
    else:
        share = min(1.0, float(eigenvalues[:keep].sum()) / energy)
    down = F.pad(down, (0, 0, 0, rank - keep))
    up = F.pad(up.T, (0, 0, 0, rank - keep))
    if zero:
        down.zero_()
        up.zero_()
    solution = MountSolution(down.float().cpu(), up.float().cpu(), applied_ridge, share)
    solution.svd_fallback = fallback
    solution.exact_fit = exact or fallback
    return solution


def resolve_mounts(model: nn.Module, mounts: Sequence[str] | None = None) -> list[str]:
    from ..lora_import import _resolver

    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    aliases = get_preset("anima").aliases(names)
    resolve = _resolver("kohya", names, aliases)
    selected = names if mounts is None else mounts
    result = []
    for name in selected:
        if mounts is None and anima_name(name) is None:
            continue
        resolved = (name,) if name in names else resolve(name)
        for target in resolved:
            if anima_name(target) is None or target not in names:
                raise ValueError(f"{name}: expected an Anima attention or MLP mount")
            if target not in result:
                result.append(target)
    if not result:
        raise ValueError("no supported image model mounts were selected")
    return result


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, function) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, function) for v in value]
    if isinstance(value, dict):
        return {k: _map_tensors(v, function) for k, v in value.items()}
    return value


class _BlockReached(Exception):
    pass


@contextmanager
def _without_malloc_graph():
    try:
        prefetch = import_module("comfy.model_prefetch")
    except ImportError:
        yield
        return
    enabled = getattr(prefetch, "malloc_graph_enabled", None)
    if not callable(enabled):
        yield
        return

    prefetch.malloc_graph_enabled = lambda *args, **kwargs: False
    try:
        yield
    except _BlockReached:
        recording = getattr(prefetch, "malloc_graph_recording", None)
        end = getattr(prefetch, "malloc_graph_end", None)
        if callable(recording) and callable(end) and recording():
            end()
        raise
    finally:
        prefetch.malloc_graph_enabled = enabled


def _capture(model, block, call):
    captured = None

    def stop(module, args, kwargs):
        nonlocal captured
        captured = _map_tensors((args, kwargs), lambda t: t.detach().cpu().clone())
        raise _BlockReached

    handle = block.register_forward_pre_hook(stop, with_kwargs=True)
    try:
        with suppress(_BlockReached), _without_malloc_graph():
            model(**call)
    finally:
        handle.remove()
    if captured is None:
        raise ValueError("model call did not reach the requested block")
    return captured


def _run_block(block, inputs):
    tensors = []
    _map_tensors(inputs, lambda t: tensors.append(t) or t)
    device = tensors[0].device if tensors else next(block.parameters()).device
    args, kwargs = _map_tensors(inputs, lambda t: t.to(device))
    return block(*args, **kwargs)


def _advance(inputs, output):
    args, kwargs = inputs
    if args:
        return (output, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": output}


def _same_structure(first, other):
    if type(first) is not type(other):
        return False
    if isinstance(first, torch.Tensor):
        return (first.shape, first.dtype, first.device) == (other.shape, other.dtype, other.device)
    if isinstance(first, dict):
        return first.keys() == other.keys() and all(
            _same_structure(value, other[key]) for key, value in first.items()
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(other) and all(
            _same_structure(a, b) for a, b in zip(first, other, strict=True)
        )
    try:
        return bool(first == other)
    except (TypeError, ValueError, RuntimeError):
        return first is other


def _batch_call(calls):

    args, kwargs = calls[0]
    hidden = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim < 3:
        return None
    dimension = hidden.shape[0]

    def combine(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if first.ndim and first.shape[0] == dimension:
                return torch.cat(values, dim=0)
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError("shared replay tensor differs")
            return first
        if isinstance(first, dict):
            return {key: combine([value[key] for value in values]) for key in first}
        if isinstance(first, (tuple, list)):
            return type(first)(combine([value[i] for value in values]) for i in range(len(first)))
        return first

    if not all(_same_structure(calls[0], call) for call in calls[1:]):
        return None
    try:
        return combine(calls)
    except ValueError:
        return None


def _replay_batches(captured, limit, weights=None):

    if weights is not None and len(weights) != len(captured):
        raise ValueError("token weights must match the replay samples")
    if limit == 1:
        if weights is None:
            return captured
        return [
            _WeightedPair(sample[0], sample[1], weight)
            for sample, weight in zip(captured, weights, strict=True)
        ]
    partitions = []
    weight_parts = []
    for index, sample in enumerate(captured):
        placed = False
        for part_index, partition in enumerate(partitions):
            if len(partition) < limit and _same_structure(partition[0], sample):
                partition.append(sample)
                if weights is not None:
                    weight_parts[part_index].append(weights[index])
                placed = True
                break
        if not placed:
            partitions.append([sample])
            if weights is not None:
                weight_parts.append([weights[index]])
    batches = []
    for part_index, partition in enumerate(partitions):
        if len(partition) == 1:
            if weights is None:
                batches.extend(partition)
            else:
                batches.append(
                    _WeightedPair(partition[0][0], partition[0][1], weight_parts[part_index][0])
                )
            continue
        target = _batch_call([sample[0] for sample in partition])
        student = _batch_call([sample[1] for sample in partition])
        if target is None or student is None:
            if weights is None:
                batches.extend(partition)
            else:
                batches.extend(
                    _WeightedPair(sample[0], sample[1], weight)
                    for sample, weight in zip(partition, weight_parts[part_index], strict=True)
                )
            continue
        if weights is None:
            batches.append((target, student))
        else:
            batches.append(_WeightedPair(target, student, _cat_weights(weight_parts[part_index])))
    return batches


def _direct_mounts(block, inputs, names, modules):
    arguments = []
    _map_tensors(inputs, lambda t: arguments.append(t) or t)

    def identity(tensor):
        version = None if torch.is_inference(tensor) else tensor._version
        return tensor.data_ptr(), tensor.shape, tensor.dtype, tensor.device, version

    identities = [identity(tensor) for tensor in arguments]
    direct, called, handles = set(), set(), []
    try:
        for name in names:

            def record(module, args, name=name):
                if name in called:
                    direct.discard(name)
                    return
                called.add(name)
                if (
                    args
                    and isinstance(args[0], torch.Tensor)
                    and args[0].numel()
                    and identity(args[0]) in identities
                ):
                    direct.add(name)

            register = getattr(modules[name], "register_forward_pre_hook", None)
            if register is not None:
                handles.append(register(record))
        with _without_malloc_graph():
            _run_block(block, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return direct


def _merge_groups(groups, direct):
    merged = []
    for group in groups:
        if merged and all(name in direct for name in group):
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return merged


class _Changes:
    def __init__(self, model):
        self.modules = dict(model.named_modules())
        self.enabled = False
        self.handles = []

    def add(self, name, solution):
        module = self.modules[name]
        down = solution.down
        up = solution.up

        def apply(module, args, output):
            if not self.enabled:
                return output
            x = args[0].float()
            change = F.linear(F.linear(x, down.to(x.device)), up.T.to(x.device))
            return output + change.to(output.dtype)

        self.handles.append(module.register_forward_hook(apply))

    @contextmanager
    def active(self, enabled):
        previous = self.enabled
        self.enabled = enabled
        try:
            yield
        finally:
            self.enabled = previous

    def close(self):
        for handle in self.handles:
            handle.remove()


def _observe(block, inputs, names, modules):
    observed, handles, shared = {}, [], {}

    def hook(name, module, args, output):
        raw = args[0]
        found = shared.get(id(raw))
        if found is not None and found[0] is raw:
            cast = found[1]
        else:
            cast = raw.detach().float()
            shared[id(raw)] = (raw, cast)
        observed[name] = (cast, output.detach().float())
        if len(observed) == len(names):
            raise _BlockReached

    try:
        for name in names:

            def record(module, args, output, name=name):
                hook(name, module, args, output)

            handles.append(modules[name].register_forward_hook(record))
        with suppress(_BlockReached), _without_malloc_graph():
            _run_block(block, inputs)
    finally:
        for handle in handles:
            handle.remove()
    if set(observed) != set(names):
        raise ValueError("a selected mount was not called by its block")
    return observed


def _tensor_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    return output.sample


def _weights(result):
    tensors = {}
    for name, solution in result.mounts.items():
        original = result.metadata.get("export_names", {}).get(name, anima_name(name))
        if original is None:
            raise ValueError(f"{name}: has no Anima export name")
        stem = (
            original
            if result.metadata.get("canonical_export")
            else "lora_unet_" + original.replace(".", "_")
        )
        tensors[stem + ".lora_down.weight"] = solution.down.contiguous()
        tensors[stem + ".lora_up.weight"] = solution.up.T.contiguous()
        tensors[stem + ".alpha"] = torch.tensor(float(solution.down.shape[0]))
    return tensors


def _atomic_save(path, tensors, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_safetensors(temporary, tensors, metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_lora(path: str | Path, result: ContrastResult) -> None:
    if Path(path).suffix != ".safetensors":
        raise ValueError("output must end in .safetensors")
    header = {"kilora_directions": json.dumps(result.metadata)}
    if "axis" in result.metadata:
        header["kilora_axis"] = result.metadata["axis"]
    cues = list(dict.fromkeys(p.get("cue", "") for p in result.metadata.get("pairs", [])))
    cues = [cue for cue in cues if cue]
    if cues and "prompt_solve" in result.metadata:
        header["kilora_cue"] = ", ".join(cues)
        header["ss_tag_frequency"] = json.dumps({str(i): {cue: 1} for i, cue in enumerate(cues)})
    _atomic_save(path, _weights(result), header)


def _save_partial(path, result, next_block):
    _atomic_save(
        path,
        _weights(result),
        {
            "kilora_directions": json.dumps(result.metadata),
            "next_block": str(next_block),
        },
    )


def _load_partial(path, expected):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        header = file.metadata()
    metadata = json.loads(header["kilora_directions"])
    if "ridge" in expected and "ridge" not in metadata:
        return ContrastResult({}, {**expected, "fit": {}, "block_gaps": {}}), 0
    metadata.setdefault("contrast_mode", "tokens")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"partial file settings differ: {key}")
    weights = load_file(str(path))
    solved = {}
    for name, record in metadata["fit"].items():
        stem = "lora_unet_" + anima_name(name).replace(".", "_")
        solved[name] = MountSolution(
            weights[stem + ".lora_down.weight"],
            weights[stem + ".lora_up.weight"].T,
            record["ridge"],
            record["fit_share"],
            record.get("svd_fallback", False),
            record.get("exact_fit", False),
        )
    return ContrastResult(solved, metadata), int(header["next_block"])


@torch.no_grad()
def solve(
    model: nn.Module,
    samples: Sequence[Sample],
    pairs: Sequence[PromptPair | tuple[str, str]],
    *,
    settings: Settings | None = None,
    mounts: Sequence[str] | None = None,
    model_sha: str,
    license: Mapping[str, Any] | str,
    partial: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    target_source: EffectSource | None = None,
    refit_round: int = 0,
    adapter=None,
    interrupt: Callable[[], None] | None = None,
    coupling: ConceptCoupling | None = None,
) -> ContrastResult:

    settings = settings or Settings()
    settings.validate()
    pairs = [p if isinstance(p, PromptPair) else PromptPair(*p) for p in pairs]
    if not pairs or not samples:
        raise ValueError("at least one pair and trajectory sample are required")
    requested = list(target_source.mounts) if mounts is None and target_source else mounts
    selected = resolve_mounts(model, requested) if adapter is None else list(adapter.mounts)
    if target_source is not None and set(selected) - target_source.mounts.keys():
        raise ValueError("effect mounts must be routed mounts of the source file")
    row_selector = getattr(adapter, "rows", None)

    def image_rows(tensor):
        if row_selector is None:
            return tensor
        rows = row_selector(tensor.shape[-2])
        return tensor if rows is None else tensor[..., rows, :]

    excluded = {}
    if not getattr(adapter, "select_image_rows", False) and any(
        p.target != p.student for p in pairs
    ):
        excluded = {
            n: _EXCLUSION
            for n in selected
            if n.endswith(_CAPTION_MOUNTS) or (adapter is not None and adapter.is_caption(n))
        }
    fitted = [n for n in selected if n not in excluded]
    if not fitted:
        raise ValueError("no mounts remain after caption exclusions")
    metadata = json.loads(
        json.dumps(
            {
                **asdict(settings),
                "pairs": [asdict(p) for p in pairs],
                "mounts": selected,
                "excluded_mounts": excluded,
                "model_sha": model_sha,
                "license": license,
                "family": "anima",
                "mode": "joint",
                "sample_count": len(samples),
                "license_notice": get_preset("anima").notice,
            }
        )
    )
    if settings.refit_rounds:
        metadata["refit_round"] = refit_round
    if target_source is not None:
        metadata.update(
            {
                "target_source": "lora",
                "source": target_source.metadata,
                "license": target_source.license,
                "prompts": [{"target": p.target, "student": p.student} for p in pairs],
            }
        )
    if adapter is not None:
        metadata.update(adapter.metadata())
    if coupling is not None:
        metadata.update(coupling.metadata())
    result, start = ContrastResult({}, {**metadata, "fit": {}, "block_gaps": {}}), 0
    if partial is not None and Path(partial).exists():
        result, start = _load_partial(partial, metadata)
    result.metadata["exact_fits"] = sum(m.exact_fit for m in result.mounts.values())
    modules = dict(model.named_modules()) if adapter is None else adapter.modules
    owner = (lambda n: int(n.split(".")[1])) if adapter is None else adapter.owner
    indices = sorted({owner(n) for n in fitted})

    def block_at(index):
        return modules[f"transformer_blocks.{index}"] if adapter is None else adapter.blocks[index]

    if start < 0 or start > len(indices):
        raise ValueError("partial file has an invalid next block")
    completed = {n for n in fitted if owner(n) in indices[:start]}
    if set(result.mounts) != completed:
        raise ValueError("partial file does not contain exactly the completed blocks")
    layout = None
    if settings.layout_source == "plain":
        from .layout import PlainLayout

        layout = PlainLayout(model)
    changes = _Changes(model) if adapter is None else adapter.changes()
    target_changes = _Changes(model) if adapter is None else adapter.changes()

    @contextmanager
    def student_chain(enabled):
        with changes.active(enabled), target_changes.active(not enabled):
            yield

    training = {module: module.training for module in model.modules()}
    model.eval()
    try:
        if adapter is not None:
            adapter.install()
        if layout is not None:
            layout.install()
        if target_source is not None:
            for name, solution in target_source.mounts.items():
                target_changes.add(name, solution)
        for name, solution in result.mounts.items():
            changes.add(name, solution)
        if coupling is not None:
            for name, solution in coupling.earlier_changes():
                if name in selected:
                    changes.add(name, solution)
        model_passes = 0
        captured = []
        image_indices = None
        weight_rows = []
        if start < len(indices):
            block = block_at(indices[start])
            device = next(model.parameters()).device if adapter is None else adapter.device
            for sample in samples:
                target_call = _map_tensors(sample.target, lambda t, d=device: t.to(d))
                student_call = _map_tensors(sample.student, lambda t, d=device: t.to(d))
                if layout is None:
                    with student_chain(False):
                        target = _capture(model, block, target_call)
                    with student_chain(True):
                        student = _capture(model, block, student_call)
                else:
                    with student_chain(True), layout.active("record"):
                        student = _capture(model, block, student_call)
                    with student_chain(False), layout.active("replay"):
                        target = _capture(model, block, target_call)
                captured.append(_map_tensors((target, student), lambda t: t.to(device)))
                weight = sample.weights
                weight_rows.append(None if weight is None else weight.detach().to(device))
            model_passes += 2 * len(samples)
            if settings.contrast_mode == "mean" and settings.contrast_components:
                per_image = {}
                for sample, calls, weight in zip(samples, captured, weight_rows, strict=True):
                    if sample.image_index is None:
                        raise ValueError(
                            "per-image identities are required for contrast components"
                        )
                    per_image.setdefault(sample.image_index, []).append((calls, weight))
                captured, image_indices = [], []
                for image_index, calls in per_image.items():
                    pairs, weights = zip(*calls, strict=True)
                    batches = _replay_batches(
                        list(pairs),
                        settings.replay_batch,
                        list(weights) if any(w is not None for w in weights) else None,
                    )
                    captured.extend(batches)
                    image_indices.extend([image_index] * len(batches))
            elif any(weight is not None for weight in weight_rows):
                captured = _replay_batches(captured, settings.replay_batch, weight_rows)
            else:
                captured = _replay_batches(captured, settings.replay_batch)
        for position in range(start, len(indices)):
            if interrupt is not None:
                interrupt()
            index = indices[position]
            prefix = f"transformer_blocks.{index}"
            block = block_at(index)
            groups = (
                [[f"{prefix}.{suffix}" for suffix in group] for group in _GROUPS]
                if adapter is None
                else adapter.groups(index, captured[0][0])
            )
            groups = [[name for name in group if name in fitted] for group in groups]
            groups = [group for group in groups if group]
            names = [name for group in groups for name in group]
            direct = set(names)
            if len(groups) > 1:
                for target, student in captured:
                    if layout is None:
                        with student_chain(False):
                            direct &= _direct_mounts(block, target, names, modules)
                        with student_chain(True):
                            direct &= _direct_mounts(block, student, names, modules)
                    else:
                        with student_chain(True), layout.active("record"):
                            direct &= _direct_mounts(block, student, names, modules)
                        with student_chain(False), layout.active("replay"):
                            direct &= _direct_mounts(block, target, names, modules)
                    if not direct:
                        break
            for names in _merge_groups(groups, direct):

                def observe(target, student, names=names, block=block):
                    if layout is None:
                        with student_chain(False):
                            desired = _observe(block, target, names, modules)
                        with student_chain(True):
                            actual = _observe(block, student, names, modules)
                    else:
                        with student_chain(True), layout.active("record"):
                            actual = _observe(block, student, names, modules)
                        with student_chain(False), layout.active("replay"):
                            desired = _observe(block, target, names, modules)
                    return desired, actual

                with _matmul_allow_tf32(settings.stats_precision == "tf32"):
                    statistics = _group_statistics(
                        names,
                        captured,
                        observe,
                        image_rows,
                        settings.contrast_mode,
                        settings.contrast_components,
                        image_indices,
                    )
                cholesky_cache = []
                for name in names:
                    fit_started = time.perf_counter()
                    c, s = statistics.pop(name)
                    if coupling is not None:
                        c, s = coupling.adjust(name, c, s)
                    solution = fit_mount(
                        c,
                        s,
                        settings.rank,
                        ridge=settings.ridge,
                        cholesky_cache=cholesky_cache,
                        fit_dtype=settings.fit_dtype,
                    )
                    result.fit_seconds += time.perf_counter() - fit_started
                    result.metadata["exact_fits"] += int(solution.exact_fit)
                    result.mounts[name] = solution
                    result.metadata["fit"][name] = {
                        "ridge": solution.ridge,
                        "fit_share": solution.fit_share,
                    }
                    if settings.contrast_mode == "mean":
                        result.metadata["fit"][name]["measured_rank"] = measured_rank(solution)
                    if solution.exact_fit:
                        result.metadata["fit"][name]["exact_fit"] = True
                    if solution.svd_fallback:
                        result.metadata["fit"][name]["svd_fallback"] = True
                        result.metadata["svd_fallbacks"] = sum(
                            int(record.get("svd_fallback", False))
                            for record in result.metadata["fit"].values()
                        )
                    changes.add(name, solution)
                cholesky_cache.clear()
            numerator = denominator = 0.0
            advanced = []
            for pair in captured:
                target, student = pair
                if layout is None:
                    with student_chain(False):
                        wanted = _tensor_output(_run_block(block, target))
                    with student_chain(True):
                        got = _tensor_output(_run_block(block, student))
                else:
                    with student_chain(True), layout.active("record"):
                        got = _tensor_output(_run_block(block, student))
                    with student_chain(False), layout.active("replay"):
                        wanted = _tensor_output(_run_block(block, target))
                advanced.append(
                    _carry_weights(pair, _advance(target, wanted), _advance(student, got))
                )
                wanted, got = image_rows(wanted).float(), image_rows(got).float()
                if wanted.shape != got.shape:
                    raise ValueError(
                        f"Block {index}: target and student image row shapes differ "
                        f"({tuple(wanted.shape)} versus {tuple(got.shape)})"
                    )
                numerator += float((wanted - got).square().sum())
                denominator += float(wanted.square().sum())
            gap = math.sqrt(numerator / max(denominator, 1e-30))
            result.metadata["block_gaps"][str(index)] = gap
            captured = advanced
            next_index = indices[position + 1] if position + 1 < len(indices) else index + 1
            for between in range(index + 1, next_index):
                intermediate = block_at(between)
                advanced = []
                for pair in captured:
                    target, student = pair
                    if layout is None:
                        with student_chain(False):
                            wanted = _tensor_output(_run_block(intermediate, target))
                        with student_chain(True):
                            got = _tensor_output(_run_block(intermediate, student))
                    else:
                        with student_chain(True), layout.active("record"):
                            got = _tensor_output(_run_block(intermediate, student))
                        with student_chain(False), layout.active("replay"):
                            wanted = _tensor_output(_run_block(intermediate, target))
                    advanced.append(
                        _carry_weights(pair, _advance(target, wanted), _advance(student, got))
                    )
                captured = advanced
            if partial is not None:
                _save_partial(partial, result, position + 1)
            if progress is not None:
                progress(f"Block {index}: relative gap {gap:.6g}")
    finally:
        changes.close()
        target_changes.close()
        if adapter is not None:
            adapter.close()
        if layout is not None:
            layout.close()
        for module, was_training in training.items():
            module.training = was_training
    print(f"Model passes: {model_passes}")
    return result


def model_sha256(folder: str | Path) -> str:
    folder = Path(folder)
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix
        in (
            ".safetensors",
            ".bin",
            ".json",
        )
    )
    if not files:
        raise ValueError("model folder contains no weights or configuration")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def memory_estimate(model: nn.Module, settings: Settings, pair_count: int, *, shapes=None) -> int:
    comfy_shapes = shapes is not None
    if shapes is None:
        shapes = {
            n: (m.in_features, m.out_features)
            for n, m in model.named_modules()
            if anima_name(n) and isinstance(m, nn.Linear)
        }
    if not shapes:
        raise ValueError("no mounts found for memory estimate")
    widths = [i for n, (i, _) in shapes.items() if n.endswith("attn1.to_q")]
    width = max(widths or [min(shape) for shape in shapes.values()])
    config = getattr(model, "config", None)
    patch = getattr(model, "patch_spatial", None)
    if patch is None:
        patch = getattr(model, "patch_size", getattr(config, "patch_size", (1, 2, 2)))
    patch = patch[-1] if isinstance(patch, (tuple, list)) else patch
    tokens = (settings.size // (8 * int(patch or 1))) ** 2
    points = pair_count * len(settings.seeds) * len(settings.sample_steps)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    capture_bytes = max(
        4 if comfy_shapes else 2,
        max((p.element_size() for p in model.parameters() if p.is_floating_point()), default=2),
    )
    captures = 2 * points * tokens * width * capture_bytes
    fit_bytes = 4 if settings.fit_dtype == "float32" else 8
    cholesky_work = max(
        fit_bytes * max(3 * i * i, i * i + 2 * i * o + min(i, o) ** 2) for i, o in shapes.values()
    )
    grouped = {}
    for name, (i, o) in shapes.items():
        block, _, suffix = name.partition(".attn")
        if not suffix:
            block, _, suffix = name.partition(".ff")
        group = next(
            (index for index, members in enumerate(_GROUPS) if name.endswith(members)),
            name,
        )
        grouped.setdefault((block, group), []).append((i, o))
    statistics = max(4 * sum(i * i + i * o for i, o in group) for group in grouped.values())
    batch = min(settings.replay_batch, points)
    projection_width = max(max(shape) for shape in shapes.values())
    replay_work = batch * tokens * (2 * width + projection_width) * capture_bytes
    working = statistics + cholesky_work + replay_work
    if settings.layout_source == "plain":
        heads = max(
            getattr(m, "heads", 1) for n, m in model.named_modules() if n.endswith(".attn1")
        )
        working += batch * 3 * heads * tokens * tokens * 4
    return weights + captures + working


@torch.no_grad()
def trajectory_samples(
    pipe,
    pairs: Sequence[PromptPair],
    settings: Settings,
    device: str,
    target_source: EffectSource | None = None,
    student_change: ContrastResult | None = None,
):
    from ..anima import encode_caption
    from ..diffusers import denoiser_of
    from ..train import swap_to_denoiser

    if student_change is not None and target_source is not None:
        raise ValueError("student trajectory cannot also apply the target source")
    dev = torch.device(device)
    denoiser_of(pipe).to("cpu")
    for name in ("text_encoder", "text_conditioner"):
        encoder = getattr(pipe, name, None)
        if isinstance(encoder, nn.Module):
            encoder.to(dev)
    captions = dict.fromkeys(
        [
            settings.negative_prompt,
            *[p.target for p in pairs],
            *[p.student for p in pairs],
        ]
    )
    captions = {text: encode_caption(pipe, text, dev) for text in captions}
    swap_to_denoiser(pipe, device)
    model = denoiser_of(pipe)
    model.eval()
    dtype = torch.bfloat16
    model.to(device=dev, dtype=dtype)
    source = target_source
    if student_change is not None:
        source = EffectSource(student_change.mounts, {}, "")
    with source.applied(model) if source is not None else nullcontext():
        return _sample_trajectories(
            pipe,
            model,
            pairs,
            settings,
            dev,
            dtype,
            captions,
            student_trajectory=student_change is not None,
        )


def _sample_trajectories(
    pipe,
    model,
    pairs,
    settings,
    dev,
    dtype,
    captions,
    *,
    student_trajectory=False,
):
    samples = []
    for pair in pairs:
        for seed in settings.seeds:
            scheduler = copy.deepcopy(pipe.scheduler)
            scheduler.set_timesteps(settings.steps, device=dev)
            latent = torch.randn(
                (1, 16, 1, settings.size // 8, settings.size // 8),
                generator=torch.Generator().manual_seed(seed),
                dtype=torch.float32,
            ).to(dev) * getattr(scheduler, "init_noise_sigma", 1.0)
            for step, timestep in enumerate(scheduler.timesteps):
                sigma = scheduler.sigmas[step].reshape(1).to(dev, dtype)

                def call(caption, latent=latent, sigma=sigma):
                    return {
                        "hidden_states": latent.to(dtype),
                        "timestep": sigma,
                        "encoder_hidden_states": captions[caption].unsqueeze(0).to(dev, dtype),
                        "padding_mask": torch.zeros(
                            1,
                            1,
                            settings.size,
                            settings.size,
                            device=dev,
                            dtype=dtype,
                        ),
                        "return_dict": False,
                    }

                target = call(pair.target)
                if step in settings.sample_steps:
                    samples.append(
                        _map_tensors(
                            Sample(target, call(pair.student)).__dict__,
                            lambda t: t.detach().cpu().clone(),
                        )
                    )
                conditional_call = call(pair.student) if student_trajectory else target
                conditional = _tensor_output(model(**conditional_call)).float()
                if settings.cfg == 1.0:
                    prediction = conditional
                else:
                    unconditional = _tensor_output(model(**call(settings.negative_prompt))).float()
                    prediction = unconditional + settings.cfg * (conditional - unconditional)
                latent = scheduler.step(
                    prediction,
                    timestep,
                    latent.float(),
                    return_dict=False,
                )[0]
    return [Sample(**sample) for sample in samples]


def read_prompt_set(path: str | Path) -> dict[str, list[PromptPair]]:
    return prompt_set_from_data(json.loads(Path(path).read_text(encoding="utf-8")))


def prompt_set_from_data(data) -> dict[str, list[PromptPair]]:
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("prompt set pairs must be a nonempty list")
        result = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("each pair must be an object")
            label = pair.get("concept")
            if not isinstance(label, str) or not re.fullmatch(r"[\w-]+", label):
                raise ValueError(
                    "concept labels must contain only letters, numbers, underscores or dashes"
                )
            if any(not isinstance(pair.get(k), str) for k in ("target", "student", "cue")):
                raise ValueError("pair target, student and cue must be strings")
            if not pair["target"].strip() or not pair["student"].strip():
                raise ValueError("pair target and student must be nonempty prompt strings")
            result.setdefault(label, []).append(
                PromptPair(pair["target"], pair["student"], label, pair["cue"])
            )
        return result
    concepts = data.get("concepts") if isinstance(data, dict) else None
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("prompt set must contain a nonempty concepts list")
    result = {}
    for concept in concepts:
        if not isinstance(concept, dict):
            raise ValueError("each concept must be an object")
        label, style = concept.get("label"), concept.get("style")
        cue = concept.get("cue", "")
        subjects = concept.get("subjects", data.get("subjects"))
        if not isinstance(label, str) or not re.fullmatch(r"[\w-]+", label):
            raise ValueError(
                "concept labels must contain only letters, numbers, underscores or dashes"
            )
        if label in result:
            raise ValueError(f"duplicate concept label: {label}")
        if not isinstance(style, str) or not isinstance(cue, str):
            raise ValueError("style and cue must be strings")
        if (
            not isinstance(subjects, list)
            or not subjects
            or any(not isinstance(subject, str) or not subject.strip() for subject in subjects)
        ):
            raise ValueError("concept subjects must be a nonempty list of prompt strings")
        result[label] = [
            PromptPair(
                ", ".join(p for p in (subject, style) if p),
                ", ".join(p for p in (subject, cue) if p),
                label,
                cue,
            )
            for subject in subjects
        ]
    return result


def refit_results(result, settings, sample, solve_round):

    for round_index in range(1, settings.refit_rounds + 1):
        started = time.perf_counter()
        samples = sample(student_change=result)
        result = solve_round(samples, round_index)
        yield round_index, result, time.perf_counter() - started


def extract(
    model_path: str | Path,
    out: str | Path,
    pairs: Sequence[PromptPair | tuple[str, str]],
    *,
    settings: Settings | None = None,
    mounts: Sequence[str] | None = None,
    device: str = "cuda",
    pipe=None,
    model_sha: str | None = None,
    progress: Callable[[str], None] = print,
    lora: str | Path | None = None,
    strength: float = 0.7,
    coupling: ConceptCoupling | None = None,
) -> ContrastResult:

    from ..anima import load_anima
    from ..diffusers import denoiser_of
    from ..presets import pack_license

    settings = settings or Settings()
    settings.validate()
    out = Path(out)
    if out.suffix != ".safetensors":
        raise ValueError("output must end in .safetensors")
    pairs = [p if isinstance(p, PromptPair) else PromptPair(*p) for p in pairs]
    if not pairs:
        raise ValueError("at least one prompt pair is required")
    if pipe is None:
        if not Path(model_path).is_dir():
            raise ValueError("model must name a local Diffusers Anima folder")
        pipe = load_anima(str(model_path), torch.bfloat16, device)
    model = denoiser_of(pipe)
    source = EffectSource.load(model, lora, strength) if lora is not None else None
    requested = list(source.mounts) if mounts is None and source else mounts
    selected = resolve_mounts(model, requested)
    if source is not None and set(selected) - source.mounts.keys():
        raise ValueError("effect mounts must be routed mounts of the source file")
    if settings.layout_source == "plain":
        from .layout import PlainLayout

        PlainLayout(model)
    estimate = memory_estimate(model, settings, len(pairs)) / 1024**3
    progress(f"Estimated GPU memory: {estimate:.2f} GiB (weights and solve workspace)")
    sha = model_sha if model_sha is not None else model_sha256(model_path)
    samples = (
        trajectory_samples(pipe, pairs, settings, device)
        if source is None
        else trajectory_samples(pipe, pairs, settings, device, target_source=source)
    )
    partial = out.with_name(out.name + ".partial.safetensors")
    license, _ = pack_license("anima")
    result = solve(
        model,
        samples,
        pairs,
        settings=settings,
        mounts=selected,
        model_sha=sha,
        license=license,
        partial=partial,
        progress=progress,
        target_source=source,
        **({} if coupling is None else {"coupling": coupling}),
    )
    write_lora(out, result)
    partial.unlink(missing_ok=True)
    progress(f"Wrote {out}")

    def resample(*, student_change):
        return trajectory_samples(pipe, pairs, settings, device, student_change=student_change)

    def solve_round(samples, round_index):
        round_out = out.with_name(f"{out.stem}.round-{round_index}.safetensors")
        round_partial = round_out.with_name(round_out.name + ".partial.safetensors")
        return solve(
            model,
            samples,
            pairs,
            settings=settings,
            mounts=selected,
            model_sha=sha,
            license=license,
            partial=round_partial,
            progress=progress,
            target_source=source,
            refit_round=round_index,
            **({} if coupling is None else {"coupling": coupling}),
        )

    for round_index, round_result, _seconds in refit_results(
        result, settings, resample, solve_round
    ):
        result = round_result
        round_out = out.with_name(f"{out.stem}.round-{round_index}.safetensors")
        round_partial = round_out.with_name(round_out.name + ".partial.safetensors")
        write_lora(round_out, result)
        round_partial.unlink(missing_ok=True)
        progress(f"Wrote {round_out}")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    class StudentPrompt(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prompts = getattr(namespace, "prompt", None) or []
            students = dict(getattr(namespace, self.dest, None) or {})
            index = len(prompts) - 1
            if index < 0 or index in students:
                raise argparse.ArgumentError(
                    self,
                    "place one --student-prompt after its --prompt",
                )
            students[index] = values
            setattr(namespace, self.dest, students)

    parser = argparse.ArgumentParser(prog="kilora directions")
    commands = parser.add_subparsers(dest="command", required=True)
    contrast = commands.add_parser("contrast", help="extract prompt changes as plain LoRA files")
    contrast.add_argument("--model", required=True, help="local Diffusers Anima folder")
    contrast.add_argument("--out", required=True, help="LoRA file, or a folder with --set")
    contrast.add_argument("--pair", nargs=2, action="append", metavar=("TARGET", "STUDENT"))
    contrast.add_argument(
        "--set",
        dest="prompt_set",
        help="JSON concepts with style, cue and subjects",
    )
    contrast.add_argument(
        "--overlap-penalty",
        type=float,
        default=0.0,
        help="with --set: λ ≥ 0 that pushes each concept off the input bases of the others",
    )
    contrast.add_argument(
        "--deflate",
        action="store_true",
        help="with --set: each later concept fits its target minus the earlier changes",
    )
    effect = commands.add_parser("effect", help="extract one LoRA file's sampled effect")
    effect.add_argument("--model", required=True, help="local Diffusers Anima folder")
    effect.add_argument("--out", required=True, help="output LoRA safetensors file")
    effect.add_argument("--lora", required=True, help="source LoRA file")
    effect.add_argument("--strength", type=float, default=0.7)
    effect.add_argument(
        "--prompt", action="append", required=True, help="target caption; repeatable"
    )
    effect.add_argument(
        "--student-prompt",
        action=StudentPrompt,
        help="student caption for the preceding --prompt; defaults to that target caption",
    )
    for command in (contrast, effect):
        command.add_argument("--seeds", nargs="+", type=int, default=[811, 1811])
        command.add_argument("--size", type=int, default=768)
        command.add_argument("--steps", type=int, default=30)
        command.add_argument("--cfg", type=float, default=1.0)
        command.add_argument("--negative-prompt", default="")
        command.add_argument("--sample-steps", nargs="+", type=int, default=None)
        command.add_argument("--rank", type=int, default=16)
        command.add_argument("--mounts", nargs="+", default=None)
        command.add_argument("--device", default="cuda")
        command.add_argument(
            "--layout-source",
            choices=("target", "plain"),
            default="target",
            help="use target self-attention maps or replay the plain student's maps",
        )
        command.add_argument(
            "--refit-rounds",
            type=int,
            default=0,
            help="additional student trajectory solves; retain each as .round-N.safetensors",
        )
    args = parser.parse_args(argv)
    if args.command == "contrast" and bool(args.pair) == bool(args.prompt_set):
        parser.error("give repeated --pair arguments or one --set file")
    coupled = args.command == "contrast" and (args.overlap_penalty != 0 or args.deflate)
    if coupled and not args.prompt_set:
        parser.error("--overlap-penalty and --deflate need a --set file")
    sampled = args.sample_steps
    if sampled is None:
        sampled = sorted({0, (args.steps - 1) // 2, args.steps - 1})
    settings = Settings(
        seeds=tuple(args.seeds),
        size=args.size,
        steps=args.steps,
        cfg=args.cfg,
        negative_prompt=args.negative_prompt,
        sample_steps=tuple(sampled),
        rank=args.rank,
        layout_source=args.layout_source,
        refit_rounds=args.refit_rounds,
    )
    try:
        settings.validate()
        if args.command == "effect":
            students = args.student_prompt or {}
            pairs = [PromptPair(p, students.get(i, p)) for i, p in enumerate(args.prompt)]
            extract(
                args.model,
                args.out,
                pairs,
                settings=settings,
                mounts=args.mounts,
                device=args.device,
                lora=args.lora,
                strength=args.strength,
            )
        elif args.prompt_set:
            groups = read_prompt_set(args.prompt_set)
            directory = Path(args.out)
            if directory.suffix == ".safetensors":
                raise ValueError("--set requires --out to name a folder")
            from ..anima import load_anima

            if not Path(args.model).is_dir():
                raise ValueError("model must name a local Diffusers Anima folder")
            pipe = load_anima(args.model, torch.bfloat16, args.device)
            sha = model_sha256(args.model)
            if coupled:
                check_coupling(args.overlap_penalty, args.deflate)

                def solve_one(label, coupling):
                    return extract(
                        args.model,
                        directory / f"{label}.safetensors",
                        groups[label],
                        settings=settings,
                        mounts=args.mounts,
                        device=args.device,
                        pipe=pipe,
                        model_sha=sha,
                        coupling=coupling,
                    )

                solve_concepts(
                    list(groups),
                    solve_one,
                    overlap_penalty=args.overlap_penalty,
                    deflate=args.deflate,
                )
            else:
                for label, pairs in groups.items():
                    extract(
                        args.model,
                        directory / f"{label}.safetensors",
                        pairs,
                        settings=settings,
                        mounts=args.mounts,
                        device=args.device,
                        pipe=pipe,
                        model_sha=sha,
                    )
        else:
            extract(
                args.model,
                args.out,
                args.pair,
                settings=settings,
                mounts=args.mounts,
                device=args.device,
            )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
