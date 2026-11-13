# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
import functools
import logging
import math
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

import torch

from .. import fused
from ..compute import recording_routes, routing_flips, timing_layers
from ..editors import Editor as WriteRule
from ..editors import commit_write
from ..errors import KiLoRAError
from ..i18n import text
from ..live_queue import (
    Preempted,
    async_enabled,
    begin_prompts,
    end_prompts,
    note_timing,
    on_pause,
    preemptible,
    queue_for,
    yield_point,
)
from ..recording import GenerationRecord, PathRecorder, observing_blocks
from . import patching

PATH_STATE = "KILORA_PATH"
log = logging.getLogger("kilora.live")


def _same_patch_values(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_patch_values(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_patch_values(x, y) for x, y in zip(a, b, strict=True))
    return isinstance(a, (str, int, float, bool, type(None))) and a == b


class _CopyExtraConds:
    def __init__(self, target: Any):
        self.target = target
        self._kilora_original_extra_conds = target._kilora_original_extra_conds
        self._kilora_previous_extra_patch = target._kilora_previous_extra_patch

    def __call__(self, **kwargs):
        return self.target(**kwargs)


def _same_callable(a: Any, b: Any) -> bool:
    return a is b or (
        getattr(a, "__self__", None) is not None
        and getattr(a, "__self__", None) is getattr(b, "__self__", None)
        and getattr(a, "__func__", None) is getattr(b, "__func__", None)
    )


def _reuse_copy_record(session: Any, fresh: Any) -> Any:
    from ..bake import is_baked
    from ..fixed_change import is_fixed

    info = patching.applied(fresh) or {}
    mode = info.get("bypass") or os.environ.get("KILORA_BYPASS", "auto")
    if mode != "off" or getattr(fresh, "injections", {}).get(patching.INJECTION_KEY):
        return fresh
    copies = fresh.object_patches
    cache = session.__dict__.setdefault("_kilora_copy_records", {})

    def remember() -> Any:
        for name, value in copies.items():
            if hasattr(value, "_kilora_original_extra_conds") and not isinstance(
                value, _CopyExtraConds
            ):
                copies[name] = _CopyExtraConds(value)
        cache[key] = fresh
        return fresh

    key = id(fresh.model)
    old = cache.get(key)
    if old is None or old.model is not fresh.model:
        if any(is_baked(v) or is_fixed(v) for v in copies.values()):
            return remember()
        return fresh
    compatible = copies.keys() == old.object_patches.keys() and _same_patch_values(
        getattr(old, "patches", {}), getattr(fresh, "patches", {})
    )
    if compatible:
        for name, new in copies.items():
            previous = old.object_patches[name]
            if is_baked(new) or is_fixed(new):
                if type(previous) is not type(new):
                    compatible = False
                    break
            elif isinstance(previous, _CopyExtraConds) and hasattr(
                new, "_kilora_original_extra_conds"
            ):
                if not (
                    _same_callable(
                        previous._kilora_original_extra_conds, new._kilora_original_extra_conds
                    )
                    and previous._kilora_previous_extra_patch is new._kilora_previous_extra_patch
                ):
                    compatible = False
                    break
            elif previous is not new:
                compatible = False
                break
    if not compatible:
        try:
            import comfy.model_management as mm
        except ImportError:
            mm = None
        unload = getattr(mm, "unload_model_clones", None)
        if unload is not None:
            unload(fresh)
        else:
            if callable(getattr(old, "detach", None)):
                old.detach()
            session.__dict__.setdefault("_kilora_retired_copy_records", []).append(old)
        return remember()
    for name, new in copies.items():
        if isinstance(old.object_patches[name], _CopyExtraConds):
            if new is not old.object_patches[name]:
                old.object_patches[name].target = new
            continue
        if not (is_baked(new) or is_fixed(new)):
            continue
        previous = old.object_patches[name]
        for attr in ("_kilora_plan", "_kilora_changes"):
            if attr in new.__dict__:
                previous.__dict__[attr] = new.__dict__[attr]
    old.model_options = fresh.model_options
    return old


@dataclass
class PathHandle:
    session: Any
    rule: WriteRule
    recorder: PathRecorder | None
    timesteps: list[float] = field(default_factory=list)
    conditioning: dict[str, Any] | None = None
    empty_conditioning: dict[str, Any] | None = None
    encode_empty: Any = None
    encode_caption: Any = None
    first_args: dict[str, Any] | None = None
    calls: int = 0
    images: int = 0
    apply_model: Any = None
    patcher: Any = None
    sampling: Any = None
    process_in: Any = None
    queue: Any = None
    generating: bool = False
    failed: bool = False
    out_of_memory: bool = False
    seed: int | None = None
    depth: float | None = None
    entry_temperature: float = 0.0
    block_modules: Any = None


def uncond_rows(batch: int, cond_or_uncond: list[int] | None) -> list[int]:
    if not cond_or_uncond:
        return []
    size = batch // len(cond_or_uncond)
    return [i * size + j for i, cu in enumerate(cond_or_uncond) if cu == 1 for j in range(size)]


def comfy_empty_caption(clip: Any, model: Any):
    return comfy_caption(clip, model, "")


def comfy_caption_encoder(clip: Any, model: Any):
    def encode(caption: str, args: dict, n: int) -> dict:
        return comfy_caption(clip, model, caption)(args, n)

    return encode


def comfy_caption(clip: Any, model: Any, caption: str):
    tokens = clip.tokenize(caption)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    def encode(args: dict, n: int) -> dict:
        import comfy.samplers as samplers

        try:
            from comfy.sampler_helpers import convert_cond
        except ImportError:
            convert_cond = samplers.convert_cond
        x = args["input"][:n]
        conds = convert_cond(conditioning)
        conds = samplers.encode_model_conds(model.extra_conds, conds, x, x.device, "positive")
        model_conds = conds[0]["model_conds"]
        return {
            k: v.process_cond(batch_size=n, device=x.device).cond.detach().cpu()
            for k, v in model_conds.items()
        }

    return encode


def cond_rows(batch: int, cond_or_uncond: list[int] | None) -> list[int] | None:
    if not cond_or_uncond:
        return None
    size = batch // len(cond_or_uncond)
    return [i * size + j for i, cu in enumerate(cond_or_uncond) if cu == 0 for j in range(size)]


def _rows_of(c: dict[str, Any], rows: list[int] | None, batch: int) -> dict[str, Any]:
    out = {}
    for k, v in c.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()
            if rows is not None and v.dim() > 0 and v.shape[0] == batch:
                v = v[rows]
            out[k] = v.cpu().clone()
        else:
            out[k] = v
    return out


def pack_families(packs: Any) -> list[str]:
    out: list[str] = []
    for p in packs:
        for family in ["common"] if p.role == "common" else list(p.families):
            if family not in out:
                out.append(family)
    return out


def editor_options(session: Any) -> dict[str, Any]:
    options = dict(getattr(session, "editor_options", {}))
    chosen = getattr(session, "editor_packs", None)
    if chosen:
        options["packs"] = pack_families(chosen)
    return options


def record(patcher: Any, rule: WriteRule, mod_id: str | None = None) -> tuple[Any, PathHandle]:
    from ..live_imports import ImportEditor, ImportedLiveSession

    if rule is None and patching.applied(patcher) is None:
        raise KiLoRAError(text("err.no_editor"))
    sessions = patching.sessions_of(patcher)
    live = [s for s in sessions if not isinstance(s, ImportedLiveSession)]
    if live:
        sessions = live
    elif not sessions:
        info = patching._require_applied(patcher)
        plan, _ = patching.plan_chain(
            patcher,
            info["chain"],
            info["env"],
            bool(info.get("joint")),
            lycoris_apply=info.get("lycoris_apply", "truncated"),
            open_imports=True,
        )
        sessions = patching.live_sessions(plan)
    if mod_id:
        sessions = [s for s in sessions if s.mod_id == mod_id]
    if not sessions:
        raise KiLoRAError(text("err.no_live_mod"))
    session = sessions[0]
    if isinstance(session, ImportedLiveSession):
        rule = ImportEditor(session)
        patcher, _ = patching.reapply(patcher)
    elif rule is None:
        raise KiLoRAError(text("err.no_editor"))
    if getattr(session, "host_norms", None) is None:
        try:
            session.host_norms = patching.host_weight_norms(patcher, list(session.state))
        except Exception:
            session.host_norms = {}
    rule.check(session)
    configure = getattr(rule, "configure", None)
    if configure is not None:
        configure(editor_options(session))
    if editor_options(session).get("cross_layer") == "jvp":
        from ..cross_layer import (
            inner_attention_candidates,
            inner_attention_links,
            sequential_links,
        )

        model = getattr(getattr(patcher, "model", None), "diffusion_model", None)
        session.cross_layer_blocks = sequential_links(model, getattr(rule, "targets", ()))
        if not hasattr(session, "cross_layer_inner"):
            targets = getattr(rule, "targets", ())
            try:
                binding = patching.detect_family(patcher)
                candidates = inner_attention_candidates(targets)
                weights = patching.host_weights(patcher, [b for _, b in candidates])
                shapes = patching.diffusers_shapes(patcher, binding)
                session.cross_layer_inner = inner_attention_links(targets, weights, shapes)
            except Exception:
                session.cross_layer_inner = {}
    if (
        isinstance(session, ImportedLiveSession)
        and getattr(session, "record_prediction", True)
        and "_cross_frames" not in session.__dict__
    ):
        from ..live_imports import import_source_frames

        try:
            weights = patching.host_weights(patcher, list(session.state))
        except Exception:
            weights = {}
        import_source_frames(session, weights)
        del weights
    handle = PathHandle(session, rule, rule.recorder(session))
    _attach_blocks(handle, patcher)
    applied = patching.applied(patcher) or {}
    handle.depth = applied.get("effective_depth", applied.get("depth"))
    import_score = (
        applied.get("import_score")
        or os.environ.get("KILORA_IMPORT_SCORE")
        or (session.mod.settings.import_score if isinstance(session, ImportedLiveSession) else None)
    )
    default_temp = "1.0" if import_score == "derived" else "0"
    handle.entry_temperature = float(os.environ.get("KILORA_DERIVED_ENTRY_TEMP", default_temp))
    if async_enabled():
        handle.queue = queue_for(session)
    model = getattr(patcher, "model", None)
    handle.sampling = getattr(model, "model_sampling", None)
    handle.process_in = getattr(model, "process_latent_in", None)
    out = patcher.clone()
    handle.patcher = out
    previous = (getattr(out, "model_options", None) or {}).get("model_function_wrapper")

    def call(apply_model, args):
        if previous is not None:
            return previous(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    def wrapper(apply_model, args):
        if not handle.generating and handle.queue is not None:
            handle.queue.begin_generation()
            handle.generating = True
        x = args["input"]
        options = (args.get("c") or {}).get("transformer_options") or {}
        seed = args.get("seed", options.get("seed"))
        if seed is not None and handle.seed is None:
            handle.seed = int(seed)
        rows = cond_rows(x.shape[0], args.get("cond_or_uncond"))
        learning = bool(session.learning) and rows != []
        try:
            if handle.recorder is not None and learning:
                handle.recorder.rows = rows
                set_sigma = getattr(handle.recorder, "set_sigma", None)
                if set_sigma is not None:
                    sigma = args["timestep"]
                    set_sigma(
                        float(sigma.reshape(-1)[0])
                        if isinstance(sigma, torch.Tensor)
                        else float(sigma)
                    )
                blocks = _block_recorder(handle.recorder)
                with (
                    handle.recorder.recording(),
                    observing_blocks(handle.block_modules or (), blocks),
                ):
                    result = call(apply_model, args)
            else:
                result = call(apply_model, args)
        except BaseException as exc:
            handle.failed = True
            _end_generation(handle)
            handle.out_of_memory = isinstance(exc, torch.cuda.OutOfMemoryError)
            raise
        if learning:
            t = args["timestep"]
            t = float(t.reshape(-1)[0]) if isinstance(t, torch.Tensor) else float(t)
            if not handle.timesteps or handle.timesteps[-1] != t:
                handle.timesteps.append(t)
            if handle.conditioning is None:
                handle.conditioning = _rows_of(args["c"], rows, x.shape[0])
                handle.apply_model = apply_model
                if handle.encode_caption is not None:
                    n = x.shape[0] if rows is None else len(rows)
                    handle.first_args = {"input": x[:n].detach()}
            if handle.empty_conditioning is None and handle.encode_empty is not None:
                n = x.shape[0] if rows is None else len(rows)
                handle.empty_conditioning = handle.encode_empty(args, n)
            handle.images = max(handle.images, x.shape[0] if rows is None else len(rows))
            handle.calls += 1
        if session.learning and handle.empty_conditioning is None and handle.encode_empty is None:
            uncond = uncond_rows(x.shape[0], args.get("cond_or_uncond"))
            if uncond:
                handle.empty_conditioning = _rows_of(args["c"], uncond, x.shape[0])
        return result

    out.set_model_unet_function_wrapper(wrapper)
    _begin_at_record(handle, out)
    queue = getattr(session, "write_queue", None)
    guard = queue.copy_guard() if queue is not None else contextlib.nullcontext()
    with guard:
        out = _reuse_copy_record(session, out)
        handle.patcher = out
    _end_with_sampling(handle, out)
    return out, handle


def _begin_at_record(handle: PathHandle, patcher: Any) -> None:

    if handle.queue is None or not hasattr(patcher, "add_wrapper_with_key"):
        return
    session = handle.session
    marked = session.__dict__.get("_kilora_marked_record")
    older = marked() if marked is not None else None
    if older is not None and older is not handle:
        _end_generation(older)
    handle.queue.begin_generation()
    handle.generating = True
    session.__dict__["_kilora_marked_record"] = weakref.ref(handle)


OUTER_SAMPLE = "outer_sample"


def _end_with_sampling(handle: PathHandle, patcher: Any) -> None:

    if handle.queue is None or not hasattr(patcher, "add_wrapper_with_key"):
        return
    key = f"kilora_live_record_{id(handle.session)}"

    def outer_sample(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            _end_generation(handle)

    if hasattr(patcher, "remove_wrappers_with_key"):
        patcher.remove_wrappers_with_key(OUTER_SAMPLE, key)
    patcher.add_wrapper_with_key(OUTER_SAMPLE, key, outer_sample)


def _block_recorder(recorder: Any):
    blocks = getattr(recorder, "blocks", None)
    return blocks if blocks is not None and hasattr(blocks, "observe_block") else None


def _attach_blocks(handle: PathHandle, patcher: Any) -> None:
    blocks = _block_recorder(handle.recorder)
    if blocks is None:
        return
    from ..families import block_list

    model = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    try:
        family = patching.detect_family(patcher).family
        attr, modules = block_list(family, model)
    except Exception as exc:
        handle.recorder.blocks = None
        handle.session.prediction_blocks_note = str(exc)
        return
    blocks.attr = attr
    handle.block_modules = modules


TARGET_CACHE_SIZE = 512


def _latent_digest(latent: torch.Tensor) -> str:
    import hashlib

    value = latent.detach().to("cpu", torch.float32).contiguous()
    digest = hashlib.sha256(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def target_reads(handle: PathHandle, blocks: Any, latent: torch.Tensor) -> tuple[dict, float]:

    from ..recording import TOP_BLOCKS, BlockStreamRecorder

    if (
        blocks is None
        or not handle.block_modules
        or handle.apply_model is None
        or handle.conditioning is None
        or handle.seed is None
        or not handle.timesteps
    ):
        return {}, 0.0
    start = time.perf_counter()
    session = handle.session
    cache = session.__dict__.setdefault("_target_reads", {})
    digest, seed = _latent_digest(latent), int(handle.seed)
    top = list(handle.block_modules)[-TOP_BLOCKS:]
    out: dict = {}
    missing = []
    for sigma in dict.fromkeys(float(t) for t in handle.timesteps):
        key = (digest, sigma, seed)
        if key in cache:
            out[sigma] = cache[key]
        else:
            missing.append(sigma)
    if missing:
        with torch.no_grad():
            ensure_loaded(handle)
            x0 = latent.detach()
            if handle.process_in is not None:
                x0 = handle.process_in(x0)
            x0 = x0.to(_model_device(handle, x0.device))
            gen = torch.Generator().manual_seed(seed)
            noise = torch.randn(x0.shape, generator=gen).to(x0.device, x0.dtype)
            cond = {
                k: (v.to(x0.device) if isinstance(v, torch.Tensor) else v)
                for k, v in (handle.conditioning or {}).items()
            }
            sampling = handle.sampling
            reader = BlockStreamRecorder(blocks.frames, blocks.attr, levels={i for i, _ in top})
            for sigma in missing:
                t = torch.full((x0.shape[0],), sigma, device=x0.device, dtype=x0.dtype)
                shape = (-1, *([1] * (x0.dim() - 1)))
                if sampling is not None and hasattr(sampling, "noise_scaling"):
                    x_t = sampling.noise_scaling(t.reshape(shape), noise, x0)
                else:
                    value = min(max(sigma, 0.0), 1.0)
                    x_t = value * noise + (1 - value) * x0
                reader.set_sigma(sigma)
                with observing_blocks(top, reader):
                    handle.apply_model(x_t, t, **cond)
                reads = reader.finish_reads()
                value = reads[0].reads if reads else {}
                if len(cache) >= TARGET_CACHE_SIZE:
                    cache.pop(next(iter(cache)))
                cache[(digest, sigma, seed)] = value
                out[sigma] = value
    return out, time.perf_counter() - start


def _attach_targets(handle: PathHandle, blocks: Any, latent: Any, record: Any) -> None:
    if blocks is None or latent is None:
        return
    if not getattr(handle.session, "record_prediction", True):
        return
    try:
        reads, seconds = target_reads(handle, blocks, latent)
    except Exception as exc:
        handle.session.prediction_blocks_note = f"target forward failed: {exc}"
        return
    if reads:
        record.extra["target_reads"] = reads
        record.extra["target_forward_s"] = seconds


def ensure_loaded(handle: PathHandle) -> None:
    if handle.patcher is None:
        return
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    load = getattr(mm, "load_models_gpu", None)
    if load is not None:
        from .bypass import same_patcher_reload

        loaded = getattr(mm, "current_loaded_models", ())
        same = any(getattr(item, "model", None) is handle.patcher for item in loaded)
        context = same_patcher_reload(handle.patcher) if same else contextlib.nullcontext()
        try:
            with context:
                load([handle.patcher])
        except BaseException:
            eject_model = getattr(handle.patcher, "eject_model", None)
            if callable(eject_model):
                eject_model()
            for item in getattr(handle.patcher, "injections", {}).get("kilora", ()):
                item.eject(handle.patcher)
            raise
    elif hasattr(mm, "load_model_gpu"):
        mm.load_model_gpu(handle.patcher)


def _model_device(handle: PathHandle, fallback: torch.device) -> torch.device:
    device = getattr(handle.patcher, "load_device", None)
    return torch.device(device) if device is not None else fallback


def _lcm_repeat(a: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    la, le = a.shape[1], e.shape[1]
    n = math.lcm(la, le)
    if n > 4 * max(la, le):
        return None
    return a.repeat(1, n // la, *([1] * (a.dim() - 2))), e.repeat(
        1, n // le, *([1] * (e.dim() - 2))
    )


def batch_conditions(caption: dict, empty: dict, rows: int) -> dict | None:
    if set(caption) != set(empty):
        return None
    out = {}
    for k, a in caption.items():
        e = empty[k]
        if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor):
            if a.dim() == 0 or a.shape[0] != rows or e.shape[0] != rows:
                return None
            if a.shape != e.shape and k == "c_crossattn" and a.dim() >= 2:
                pair = _lcm_repeat(a, e)
                if pair is None or pair[0].shape[2:] != pair[1].shape[2:]:
                    return None
                a, e = pair
            if a.shape != e.shape:
                return None
            out[k] = torch.cat([a, e])
        elif a is e:
            out[k] = a
        else:
            try:
                same = (
                    not isinstance(a, torch.Tensor) and not isinstance(e, torch.Tensor) and a == e
                )
            except Exception:
                same = False
            if not same:
                return None
            out[k] = a
    return out


def _retry(session: Any) -> int:
    tries = getattr(session, "_measure_tries", None)
    if not isinstance(tries, dict):
        tries = {}
        try:
            session._measure_tries = tries
        except AttributeError:
            return 0
    key = len(getattr(session, "commits", ()))
    n = tries.get(key, 0)
    tries.clear()
    tries[key] = n + 1
    return n


FLIPS_ENV = "KILORA_LIVE_FLIPS"
TIMING_ENV = "KILORA_IMPORT_TIMING"


def wants_timing() -> bool:
    return os.environ.get(TIMING_ENV, "").strip() in ("1", "on", "true")


def wants_flips(handle: PathHandle) -> bool:
    if os.environ.get(FLIPS_ENV, "").strip() in ("1", "on", "true"):
        return True
    return bool(getattr(handle.rule, "wants_flips", False))


def editor_note(rule: Any) -> str:
    note = getattr(rule, "write_note", None)
    try:
        text_ = note() if callable(note) else ""
    except Exception:
        return ""
    return f" {text_}" if text_ else ""


def timing_note(record: Any) -> str:
    timing = getattr(getattr(record, "measure", None), "timing", None)
    calls = getattr(getattr(record, "measure", None), "calls", None)
    if timing and not note_timing(timing):
        log.info(
            "live write: %d stops for a render, %.2f s of overlap, %.2f s dropped",
            timing.get("preempts", 0),
            timing.get("preempt_wait_s", 0.0),
            timing.get("preempt_dropped_s", 0.0),
        )
    if not timing or not (timing["base_s"] or timing["trial_s"]):
        return ""
    base, trial = timing["base_s"], timing["trial_s"]
    total = sum(base) + sum(trial)
    layers = timing["fused"] + timing["eager"]
    parts = [f"Measurement: {calls[0] if calls else len(base) + len(trial)} model calls"]
    parts.append(f"{total:.1f} s")
    count = getattr(record.measure, "sample_count", 1)
    if isinstance(count, int) and count > 1:
        parts.append(f"measure_samples={count} ({sum(trial):.2f} s on the samples)")
    if base:
        parts.append(f"base {base[0]:.2f} s")
    if trial:
        parts.append(f"trial {sum(trial) / len(trial):.2f} s each ({len(trial)})")
    if "layers_s" in timing:
        hooks = timing["hooks_s"]
        routed = max(timing["layers_s"] - hooks, 0.0)
        parts.append(
            f"split: observers {hooks:.2f} s, routed layers {routed:.2f} s, "
            f"rest {max(total - timing['layers_s'], 0.0):.2f} s"
        )
    if layers:
        parts.append(f"fused layers {timing['fused']}/{layers}")
        reasons = timing.get("reasons") or {}
        if reasons:
            parts.append(
                "eager: " + ", ".join(f"{why} ({n})" for why, n in sorted(reasons.items()))
            )
        elif timing["eager"] and fused.LAST_REASON[0]:
            parts.append(f"eager: {fused.LAST_REASON[0]}")
    return " " + ", ".join(parts) + "."


def timing_split(timing: dict[str, Any]) -> dict[str, float]:
    total = sum(timing["base_s"]) + sum(timing["trial_s"])
    hooks = timing["hooks_s"]
    return {
        "observers": hooks,
        "routed": max(timing["layers_s"] - hooks, 0.0),
        "rest": max(total - timing["layers_s"], 0.0),
        "total": total,
    }


SAMPLE_SEED_STEP = 104729


def measure_samples(session: Any) -> int:
    count = getattr(session, "measure_samples", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return 1
    return count


def sample_plan(timesteps: Any, seed: int, count: int) -> list[tuple[float, int]]:
    ordered = sorted(timesteps)
    n = len(ordered)
    return [
        (ordered[int((j + 0.5) / count * n)], seed + SAMPLE_SEED_STEP * j) for j in range(count)
    ]


def image_loss(
    handle: PathHandle,
    latent: torch.Tensor,
    event: int,
    mask: torch.Tensor | None = None,
    region_conditioning: dict[str, Any] | None = None,
    region: Any = None,
    retry: int = 0,
):

    from ..compute import key_probes
    from ..recording import perturbing
    from ..regions import is_full, latent_mask, masked_mean

    x0 = latent.detach()
    if handle.process_in is not None:
        x0 = handle.process_in(x0)
    x0 = x0.to(_model_device(handle, x0.device))
    t_value = sorted(handle.timesteps)[len(handle.timesteps) // 2]
    seed = 1009 * (event + 1) + 7919 * retry
    t = torch.full((x0.shape[0],), t_value, device=x0.device, dtype=x0.dtype)
    sampling = handle.sampling

    def noised(seed_: int, value: float | None = None) -> torch.Tensor:
        value, t_ = (t_value, t) if value is None else (value, torch.full_like(t, value))
        gen = torch.Generator().manual_seed(seed_)
        noise = torch.randn(x0.shape, generator=gen).to(x0.device, x0.dtype)
        if sampling is not None and hasattr(sampling, "noise_scaling"):
            return sampling.noise_scaling(t_.reshape(-1, *([1] * (x0.dim() - 1))), noise, x0)
        sigma = min(max(value, 0.0), 1.0)
        return sigma * noise + (1 - sigma) * x0

    x_t = noised(seed)
    count = measure_samples(handle.session)
    plan = sample_plan(handle.timesteps, seed, count)
    samples = []
    if count > 1:
        for value, seed_j in plan:
            t_j = torch.full_like(t, value)
            samples.append((noised(seed_j, value), t_j, torch.cat([t_j, t_j])))
    if mask is not None:
        mask = latent_mask(mask, x0).to(x0.device)
        if is_full(mask):
            mask = None

    def on_device(c):
        return {
            k: (v.to(x0.device) if isinstance(v, torch.Tensor) else v) for k, v in (c or {}).items()
        }

    if handle.empty_conditioning is None:
        raise KiLoRAError(
            "The live measurement needs the empty caption: connect the CLIP to "
            "KiLoRA Live: 1. Record, or sample with guidance."
        )
    caption = handle.conditioning if region_conditioning is None else region_conditioning
    conds = [on_device(caption), on_device(handle.empty_conditioning)]

    base_routes: dict = {}
    sample_routes: dict = {}
    flips_on = wants_flips(handle)
    timing: dict[str, Any] = {
        "base_s": [],
        "trial_s": [],
        "fused": 0,
        "eager": 0,
        "reasons": {},
        "preempts": 0,
        "preempt_wait_s": 0.0,
        "preempt_dropped_s": 0.0,
    }
    timed = wants_timing()
    if timed:
        timing.update(layers_s=0.0, hooks_s=0.0)
    base_outs: list = []
    joined = batch_conditions(conds[0], conds[1], int(x_t.shape[0]))
    t_2 = torch.cat([t, t])
    calls = [0]
    home = x0.device

    def release() -> None:
        nonlocal x0, x_t, t, t_2, mask, joined
        x_t = None
        samples.clear()
        x0, t, t_2, mask, joined = _moved((x0, t, t_2, mask, joined), "cpu")
        conds[:] = _moved(conds, "cpu")
        base_outs[:] = _moved(base_outs, "cpu")

    def restore() -> None:
        nonlocal x0, x_t, t, t_2, mask, joined
        x0, t, t_2, mask, joined = _moved((x0, t, t_2, mask, joined), home)
        conds[:] = _moved(conds, home)
        base_outs[:] = _moved(base_outs, home)
        x_t = noised(seed)
        if count > 1:
            for value, seed_j in plan:
                t_j = torch.full_like(t, value)
                samples.append((noised(seed_j, value), t_j, torch.cat([t_j, t_j])))

    on_pause(release, restore)

    def outputs(trial, probes_pass=False, x=None, sample=None):
        x = x_t if x is None else x
        tt, tt_2 = t, t_2
        if sample is not None:
            x, tt, tt_2 = sample
        with contextlib.ExitStack() as stack:
            if trial is not None:
                stack.enter_context(perturbing(trial))
            if joined is not None and not probes_pass:
                calls[0] += 1
                out = handle.apply_model(torch.cat([x, x]), tt_2, **joined)
                return list(out.chunk(2))
            outs = []
            for n, cond in enumerate(conds):
                calls[0] += 1
                if probes_pass and n == 0:
                    with key_probes() as probes:
                        outs.append(handle.apply_model(x, tt, **cond))
                    region.probes = probes
                else:
                    outs.append(handle.apply_model(x, tt, **cond))
            return outs

    def loss_of(outs) -> float:
        losses = [float(masked_mean((o.float() - x0.float()).pow(2), mask)) for o in outs]
        return sum(losses) / len(losses)

    def run_preemptible(run, allowed):
        lost = 0.0
        while True:
            start = time.perf_counter()
            try:
                with preemptible(allowed):
                    return run(), lost
            except Preempted as stop:
                dropped = time.perf_counter() - start
                timing["preempts"] += 1
                timing["preempt_wait_s"] += stop.seconds
                timing["preempt_dropped_s"] += dropped
                yield_point()
                ensure_loaded(handle)
                lost += time.perf_counter() - start

    def forward_samples(trial):
        yield_point()
        ensure_loaded(handle)
        record = recording_routes() if flips_on else contextlib.nullcontext({})
        clock = timing_layers() if timed else contextlib.nullcontext(None)
        if timed and x_t.is_cuda:
            torch.cuda.synchronize(x_t.device)
        t0, f0, r0 = time.perf_counter(), dict(fused.COUNTS), dict(fused.REASONS)
        between = not flips_on and not timed
        per_sample, waited = [], 0.0
        with torch.no_grad(), record as routes, clock as layer_time:
            for i in range(len(samples)):
                if i and between:
                    paused = yield_point()
                    if paused:
                        waited += paused
                        ensure_loaded(handle)
                outs, lost = run_preemptible(lambda i=i: outputs(trial, sample=samples[i]), between)
                waited += lost
                per_sample.append(outs)
        last = per_sample[-1]
        if last and last[0].is_cuda:
            torch.cuda.synchronize(last[0].device)
        seconds = time.perf_counter() - t0 - waited
        timing["base_s" if trial is None else "trial_s"].append(seconds)
        for k in ("fused", "eager"):
            timing[k] += fused.COUNTS[k] - f0[k]
        for why, n in fused.REASONS.items():
            if n > r0.get(why, 0):
                timing["reasons"][why] = timing["reasons"].get(why, 0) + n - r0.get(why, 0)
        if layer_time is not None:
            timing["layers_s"] += layer_time["layers_s"]
            timing["hooks_s"] += layer_time["hooks_s"]
        losses = [loss_of(outs) for outs in per_sample]
        measure.sample_losses = losses
        return sum(losses) / len(losses), routes

    def forward(trial):
        yield_point()
        ensure_loaded(handle)
        probes_pass = trial is None and region is not None and region.probes is None
        record = recording_routes() if flips_on else contextlib.nullcontext({})
        clock = timing_layers() if timed else contextlib.nullcontext(None)
        if timed and x_t.is_cuda:
            torch.cuda.synchronize(x_t.device)
        t0, f0, r0 = time.perf_counter(), dict(fused.COUNTS), dict(fused.REASONS)
        allowed = not flips_on and not timed and not probes_pass
        with torch.no_grad(), record as routes, clock as layer_time:
            outs, lost = run_preemptible(lambda: outputs(trial, probes_pass), allowed)
        if outs and outs[0].is_cuda:
            torch.cuda.synchronize(outs[0].device)
        seconds = time.perf_counter() - t0 - lost
        timing["base_s" if trial is None else "trial_s"].append(seconds)
        for k in ("fused", "eager"):
            timing[k] += fused.COUNTS[k] - f0[k]
        for why, n in fused.REASONS.items():
            if n > r0.get(why, 0):
                timing["reasons"][why] = timing["reasons"].get(why, 0) + n - r0.get(why, 0)
        if layer_time is not None:
            timing["layers_s"] += layer_time["layers_s"]
            timing["hooks_s"] += layer_time["hooks_s"]
        if trial is None:
            base_outs[:] = [o.float() for o in outs]
        return loss_of(outs), routes

    def measure(trial) -> float:
        if trial is None:
            value, routes = forward(None)
            base_routes.clear()
            base_routes.update(routes)
            measure.last_flips = 0
            return value
        if samples:
            if flips_on and not sample_routes:
                _, routes = forward_samples(None)
                sample_routes.update(routes)
            value, routes = forward_samples(trial)
            reference = sample_routes
        else:
            if flips_on and not base_routes:
                _, routes = forward(None)
                base_routes.update(routes)
            value, routes = forward(trial)
            reference = base_routes
        if not flips_on:
            return value
        flips = routing_flips(reference, routes)
        measure.last_flips = flips
        measure.flips.append(flips)
        if flips:
            log.info("live measurement: %d routing choices changed by the trial step", flips)
        return value

    def gap_of(out_c, out_e) -> float:
        loss_c = float(masked_mean((out_c.float() - x0.float()).pow(2), mask))
        loss_e = float(masked_mean((out_e.float() - x0.float()).pow(2), mask))
        return loss_e - loss_c

    def guard_score(trial) -> dict[str, float]:
        if trial is None and base_outs:
            out_c, out_e = base_outs
        else:
            yield_point()
            ensure_loaded(handle)
            with torch.no_grad():
                out_c, out_e = (o.float() for o in outputs(trial))
        score = {"gap": gap_of(out_c, out_e), "spread": float(out_c.float().std())}
        yield_point()
        ensure_loaded(handle)
        with torch.no_grad():
            score["gap_alt"] = gap_of(*outputs(trial, x=noised(seed + 104729)))
        return score

    measure.last_flips = 0
    measure.device = x0.device
    measure.flips = []
    measure.guard_score = guard_score
    measure.calls = calls
    measure.timing = timing
    measure.samples = plan
    measure.sample_count = count
    measure.sample_losses = []
    if region is not None:
        region.ensure_probes = lambda: forward(None)
    return measure


def _moved(value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _moved(v, device) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_moved(v, device) for v in value)
    return value


def _end_generation(handle: PathHandle) -> None:
    if handle.generating and handle.queue is not None:
        handle.queue.end_generation()
        handle.generating = False


def measured_record(handle: PathHandle, samples: dict[str, Any], vae: Any = None):
    from ..measure_size import measurement_latent

    _end_generation(handle)
    latent = samples.get("samples")
    if latent is None or handle.apply_model is None or not handle.timesteps:
        raise KiLoRAError("the generation left nothing to measure")
    s = handle.session
    small, _ = measurement_latent(
        latent,
        size=getattr(s, "measure_size", 512),
        reducer=getattr(s, "measure_reducer", "latent"),
        vae=vae,
    )
    measure = image_loss(handle, small, len(s.commits))
    return GenerationRecord(
        latent=latent.detach().cpu().clone(),
        conditioning=handle.conditioning,
        timesteps=list(handle.timesteps),
        images=max(int(latent.shape[0]), 1),
        measure=measure,
        guard_score=measure.guard_score,
    )


class QueuedWrite:
    def __init__(self, session, rule, record, prepare):
        self.session, self.rule, self.record, self.prepare = session, rule, record, prepare

    def release(self):
        self.record = None
        self.prepare = None

    def ready(self):
        if self.prepare is not None:
            self.record.measure = self.prepare()
            self.record.guard_score = self.record.measure.guard_score
            self.prepare = None
        return self.record

    def run_many(self, events):
        jobs = [e.run for e in events]
        session, rule = self.session, self.rule
        original = session
        branch = events[0].branch
        if branch is not None:
            session = copy.copy(original)
            session.editor_state = copy.copy(original.editor_state)
            session.editor_state.load_tensors(original.editor_snapshot())
            session.editor_state.checkout(branch)
            session.commits = list(original.commits)
            session.undo_log = copy.deepcopy(original.undo_log)
            session._memory_saves = {}
            session.saves_dir = None
            session.pending = None
            session.pinned = None
            session._pinned_state = None
            session.pinned_write_branch = None
            rule = copy.copy(rule)
            rule.session = session
            session.editor_state.expand_live = rule._expanded_state
            session.state = rule._expanded_state(original.state)

        records = [job.ready() for job in jobs]
        for r in records:
            note_timing(getattr(getattr(r, "measure", None), "timing", None))
        compatible = (
            len(records) > 1
            and hasattr(rule, "next_combined_state")
            and all(
                type(job.rule) is type(rule) and job.rule.options == rule.options for job in jobs
            )
            and all(
                not r.extra.get("region")
                and not float((r.extra.get("draw") or {}).get("temperature", 0) or 0)
                for r in records
            )
        )
        reports = []
        if compatible:
            before = len(session.commits)
            reports.append(commit_write(session, rule, records[-1], records=records))
            if len(session.commits) == before:
                compatible = False
        if not compatible:
            for job, record in zip(jobs, records, strict=True):
                active_rule = rule if branch is not None else job.rule
                reports.append(commit_write(session, active_rule, record))
        if branch is not None:
            source = session.editor_state
            target = original.editor_state
            for key, item in source.forest.branches.items():
                if key == branch or key not in target.forest.branches:
                    target.forest.branches[key] = item
                    target._branch_regions[key] = (
                        source.regions
                        if key == source.forest.current
                        else source._branch_regions[key]
                    )
            target.forest.serial = max(target.forest.serial, source.forest.serial)
            original._autosave()
        written = (
            session.editor_state.forest.current
            if getattr(session.editor_state, "forest", None)
            else "main"
        )
        count = len(records) if compatible else 1
        return " ".join(reports) + f" combined={count} branch={written}." + editor_note(rule)


def write(
    handle: PathHandle,
    samples: dict[str, Any] | None = None,
    vae: Any = None,
    mask: torch.Tensor | None = None,
    region_caption: str = "",
) -> str:

    _end_generation(handle)
    if handle.failed:
        if handle.recorder is not None:
            handle.recorder.discard()
        if handle.out_of_memory:
            return text("msg.oom")
        return text("msg.not_finished")
    if handle.calls == 0:
        if not handle.session.learning:
            return text("msg.learning_off")
        return text("msg.nothing_recorded")
    latent = None if samples is None else samples.get("samples")
    images = handle.images if latent is None else int(latent.shape[0])
    measure, prepare, note = None, None, ""
    region = None
    region_caption = (region_caption or "").strip()
    if mask is not None and (region_caption or not bool((mask == 1).all())):
        from ..regions import Region

        prompt = str(getattr(handle, "prompt", "") or "")
        region = Region(mask=mask, name=region_caption or prompt, caption=region_caption)
    if latent is not None and handle.apply_model is not None and handle.timesteps:
        from ..measure_size import measurement_latent

        s = handle.session
        small, note = measurement_latent(
            latent,
            size=getattr(s, "measure_size", 512),
            reducer=getattr(s, "measure_reducer", "latent"),
            vae=vae,
        )
        region_conditioning = None
        if region is not None:
            from ..regions import latent_mask

            region.mask = latent_mask(mask, small)
            if region_caption and handle.encode_caption is not None and handle.first_args:
                n = int(handle.first_args["input"].shape[0])
                region_conditioning = handle.encode_caption(region_caption, handle.first_args, n)
            elif region_caption:
                note = (note + " " if note else "") + text("msg.region_caption_needs_clip")
        if handle.queue is not None:
            saved_handle = copy.copy(handle)
            saved_handle.timesteps = list(handle.timesteps)
            saved_handle.first_args = None
            saved_handle.recorder = None
            small = small.detach().cpu().clone()
            if region is not None:
                region.mask = region.mask.detach().cpu().clone()
            event, retry = len(s.commits), _retry(s)

            def prepare():
                return image_loss(
                    saved_handle,
                    small,
                    event,
                    mask=None if region is None else region.mask,
                    region_conditioning=region_conditioning,
                    region=region,
                    retry=retry,
                )
        else:
            measure = image_loss(
                handle,
                small,
                len(s.commits),
                mask=None if region is None else region.mask,
                region_conditioning=region_conditioning,
                region=region,
                retry=_retry(s),
            )
    blocks = _block_recorder(handle.recorder)
    rec = GenerationRecord(
        stats=handle.recorder.finish() if handle.recorder is not None else None,
        latent=None if latent is None else latent.detach().cpu().clone(),
        conditioning=handle.conditioning,
        timesteps=list(handle.timesteps),
        images=max(images, 1),
        measure=measure,
        guard_score=getattr(measure, "guard_score", None),
    )
    if handle.seed is None and samples is not None and samples.get("noise_seed") is not None:
        handle.seed = int(samples["noise_seed"])
    if blocks is not None and latent is not None:
        if handle.queue is not None and prepare is not None:
            saved_handle.seed = handle.seed
            target, measured = latent.detach().cpu().clone(), prepare

            def prepare():
                _attach_targets(saved_handle, blocks, target, rec)
                return measured()
        elif handle.queue is None:
            _attach_targets(handle, blocks, latent, rec)
    rec.extra["draw"] = {
        "seed": handle.seed,
        "temperature": handle.entry_temperature,
        "depth": handle.depth,
    }
    if region is not None:
        rec.extra["region"] = region
    handle.calls, handle.timesteps, handle.conditioning, handle.images = 0, [], None, 0
    handle.empty_conditioning = None
    handle.first_args = None
    handle.seed = None
    session, rule = handle.session, handle.rule
    note = f" {note}" if note else ""
    if handle.queue is not None:
        number = handle.queue.submit(QueuedWrite(session, rule, rec, prepare))
        return text("msg.queued", number=number) + note
    return commit_write(session, rule, rec) + timing_note(rec) + editor_note(rule) + note


PROMPT_HOOK_MARK = "_kilora_prompt_hook"


def install_prompt_hook(executor_cls: Any = None) -> bool:

    if executor_cls is None:
        try:
            import execution

            executor_cls = getattr(execution, "PromptExecutor", None)
        except Exception:
            executor_cls = None
    run = getattr(executor_cls, "execute", None) if executor_cls is not None else None
    if run is None or not callable(run):
        log.info("live writes: no prompt hook (PromptExecutor.execute is missing)")
        return False
    if getattr(run, PROMPT_HOOK_MARK, False):
        return True

    @functools.wraps(run)
    def execute(self, *args, **kwargs):
        begin_prompts()
        try:
            return run(self, *args, **kwargs)
        finally:
            end_prompts()

    setattr(execute, PROMPT_HOOK_MARK, True)
    executor_cls.execute = execute
    return True
