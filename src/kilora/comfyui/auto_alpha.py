# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars
import logging
import math
import os
from types import SimpleNamespace

import torch

from ..attention_maps import AttentionMaps, alpha_from_maps, trigger_positions
from ..call_values import axis_alphas
from ..i18n import text
from ..roles import identity_token
from .trigger_words import _metadata, trigger_words_of

log = logging.getLogger(__name__)
_conditioning = contextvars.ContextVar("kilora_auto_alpha_conditioning", default=None)


def install_sampler_hook(options):
    previous = options.get("sampler_calc_cond_batch_function")

    def hook(args):
        conds = args.get("conds") or []
        positive = conds[0] if conds else None
        conditioning = next(
            (c for c in (positive or []) if isinstance(c, dict) and "t5xxl_ids" in c), {}
        )
        token = _conditioning.set(conditioning)
        try:
            if previous is not None:
                return previous(args)
            from comfy.samplers import calc_cond_batch

            return calc_cond_batch(
                args["model"], conds, args["input"], args["sigma"], args["model_options"]
            )
        finally:
            _conditioning.reset(token)

    options["sampler_calc_cond_batch_function"] = hook


def _ids(clip, text):
    tokens = clip.tokenize(text).get("t5xxl", [])
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token[0] if isinstance(token, (tuple, list)) else token) for token in tokens]


def trigger_ids(chain, clip, reasons=None):
    reasons = [] if reasons is None else reasons
    ip = [e for e in chain if e.enabled and e.kind == "lora" and e.axis == "ip"]
    if not ip:
        reasons.append("no LoRA is on the ip axis")
        return {}
    if clip is None or not hasattr(clip, "tokenize"):
        reasons.append("no CLIP reaches KiLoRA Apply (connect its clip input)")
        return {}
    result = {}
    for entry in ip:
        name = os.path.basename(entry.path)
        try:
            empty = _ids(clip, "")
            words = trigger_words_of(entry.path)
            identity = identity_token(_metadata(entry.path))
            if not words and not identity:
                reasons.append(f"{name} has no trigger word or identity token")
                return {}
            ids = _ids(clip, identity or words[0])
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            reasons.append(f"{name}: the trigger does not tokenize ({exc})")
            return {}
        while ids and empty and ids[-1] == empty[-1]:
            ids.pop()
            empty = empty[:-1]
        if not ids:
            reasons.append(f"{name}: the trigger gives no tokens")
            return {}
        result[entry.path] = ids
    return result


class ProbeAttentionMaps(AttentionMaps):
    def __init__(self, positions, blocks, probes, denoiser):
        super().__init__(positions, blocks)
        self.probes = probes
        self.denoiser = denoiser
        self._contexts = {}
        self.failed = False

    def collect(self, q, k, heads, skip_reshape=False, mask=None, cond_or_uncond=None):
        if self.failed:
            return
        try:
            self._collect(q, k, heads, skip_reshape, mask, cond_or_uncond)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.failed = True
            self._sums.clear()
            self._counts.clear()
            log.warning(text("auto_alpha.probe_failed", reason=str(exc)))

    def _collect(self, q, k, heads, skip_reshape, mask, cond_or_uncond):
        super().collect(q, k, heads, skip_reshape, mask, cond_or_uncond)
        if self.block is None or not (
            self.blocks // 3 <= self.block < max(self.blocks // 3 + 1, 2 * self.blocks // 3)
        ):
            return
        if not skip_reshape:
            q = q.reshape(q.shape[0], q.shape[1], heads, -1).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], heads, -1).transpose(1, 2)
        batch = q.shape[0]
        rows = list(range(batch))
        if cond_or_uncond is not None:
            groups = list(cond_or_uncond)
            if not groups or batch % len(groups):
                return
            count = batch // len(groups)
            rows = [
                i
                for g, kind in enumerate(groups)
                if kind == 0
                for i in range(g * count, (g + 1) * count)
            ]
            if not rows:
                return
        if mask is not None:
            mask = torch.broadcast_to(mask, (batch, heads, q.shape[-2], k.shape[-2]))[rows]
        attn = self.denoiser.blocks[self.block].cross_attn
        with torch.no_grad():
            keys = {}
            for name, (caption, count) in self.probes.items():
                from .patching import encode_region_caption

                cache_key = (name, k.device, k.dtype)
                if cache_key not in self._contexts:
                    self._contexts[cache_key] = encode_region_caption(
                        SimpleNamespace(diffusion_model=self.denoiser), caption, k.device, k.dtype
                    )[:, :count]
                context = self._contexts[cache_key]
                key = attn.k_proj(context).reshape(1, count, heads, -1)
                norm = getattr(attn, "k_norm", None)
                if norm is not None:
                    key = norm(key)
                keys[name] = key.transpose(1, 2).float()
            values = {name: [] for name in keys}
            q, k = q[rows], k[rows].float()
            for start in range(0, q.shape[-2], 1024):
                query = q[..., start : start + 1024, :].float() / math.sqrt(q.shape[-1])
                scores = query @ k.transpose(-1, -2)
                if mask is not None:
                    part = mask[..., start : start + 1024, :]
                    scores = (
                        scores.masked_fill(~part, float("-inf"))
                        if part.dtype == torch.bool
                        else scores + part
                    )
                total = torch.logsumexp(scores, -1, keepdim=True)
                for name, key in keys.items():
                    probe = query @ key.transpose(-1, -2)
                    value = (probe - torch.logaddexp(probe, total)).exp().mean(-1).mean(1)
                    values[name].append(value)
            for name, chunks in values.items():
                value = torch.cat(chunks, -1).detach()
                if name in self._sums and self._sums[name].shape != value.shape:
                    raise ValueError("attention map shapes differ within a step")
                self._sums[name] = self._sums.get(name, 0) + value
                self._counts[name] = self._counts.get(name, 0) + 1


def alpha_stats(step, maps, alphas) -> dict:
    def q(v, p):
        return round(float(torch.quantile(v.flatten().float(), p)), 4)

    ip = alphas.get("ip")
    out = {"sigma": round(float(step), 4)}
    for name, m in maps.items():
        out[f"map_p50:{os.path.basename(str(name))}"] = q(m, 0.5)
        out[f"map_p99:{os.path.basename(str(name))}"] = q(m, 0.99)
    if isinstance(ip, torch.Tensor):
        out.update(ip_mean=round(float(ip.mean()), 4), ip_p10=q(ip, 0.1), ip_p90=q(ip, 0.9))
    return out


class AutoAlpha:
    def __init__(
        self,
        family,
        triggers,
        blocks,
        patch,
        share=1.0,
        *,
        chain=(),
        clip=None,
        denoiser=None,
        reports=None,
    ):
        self.family, self.triggers, self.blocks, self.patch = family, triggers, blocks, patch
        self.share = share
        self.chain, self.clip, self.denoiser = chain, clip, denoiser
        self.reports = reports if reports is not None else []
        self.report = ""
        self._probe_cache = {}
        self.collector = None
        self.step = None
        self.alphas = {}
        self.shape = None
        self.warned = False
        self.stats: list[dict] = []

    def _collector(self, conditioning):
        mentions = conditioning.get("kilora_mentions")
        if not mentions or self.family != "anima" or self.clip is None:
            positions, message = trigger_positions(self.family or "", conditioning, self.triggers)
            return (AttentionMaps(positions, self.blocks) if positions else None), message
        ids = conditioning.get("t5xxl_ids")
        if ids is None:
            return None, text("auto_alpha.probe_ids")
        ids = torch.as_tensor(ids)
        if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
            return None, text("auto_alpha.probe_ids")
        triggers = dict(self.triggers)
        probes, positions = {}, {}
        entries = [e for e in self.chain if e.enabled and e.kind == "lora" and e.axis == "ip"]
        records = {}
        for entry in entries:
            record = next(
                (m for m in mentions if entry.path in (m.get("path"), m.get("baked_path"))),
                None,
            )
            if record is not None:
                records[entry.path] = record
            if entry.path not in triggers:
                source = (
                    entry
                    if record is None
                    else SimpleNamespace(enabled=True, kind="lora", axis="ip", path=record["path"])
                )
                source_ids = trigger_ids([source], self.clip)
                if source.path in source_ids:
                    triggers[entry.path] = source_ids[source.path]
        if any(e.path not in triggers for e in entries):
            return None, text("auto_alpha.probe_missing")
        for path, ids in triggers.items():
            found, message = trigger_positions(self.family, conditioning, {path: ids})
            if found:
                positions.update(found)
                continue
            record = records.get(path)
            if record is None or self.denoiser is None:
                return None, message
            words = record.get("trigger_words") or trigger_words_of(record["path"])
            if not words:
                return None, message
            trigger_text = ", ".join(words)
            cache_key = (record["path"], trigger_text)
            if cache_key not in self._probe_cache:
                tokens = _ids(self.clip, trigger_text)
                empty = _ids(self.clip, "")
                while tokens and empty and tokens[-1] == empty[-1]:
                    tokens.pop()
                    empty.pop()
                if not tokens:
                    return None, message
                with torch.no_grad():
                    caption = self.clip.encode_from_tokens_scheduled(
                        self.clip.tokenize(trigger_text)
                    )
                if len(caption) != 1 or caption[0][0].shape[1] < len(tokens):
                    return None, text("auto_alpha.probe_encoding")
                self._probe_cache[cache_key] = (caption, len(tokens))
            probes[path] = self._probe_cache[cache_key]
        if not probes:
            return (AttentionMaps(positions, self.blocks) if positions else None), None
        for block in range(self.blocks // 3, max(self.blocks // 3 + 1, 2 * self.blocks // 3)):
            attn = getattr(self.denoiser.blocks[block], "cross_attn", None)
            if not callable(getattr(attn, "k_proj", None)):
                return None, text("auto_alpha.probe_projection")
        self.report = text(
            "auto_alpha.probe_report",
            files=", ".join(os.path.basename(records[path]["path"]) for path in probes),
        )
        if self.report not in self.reports:
            self.reports.append(self.report)
            log.info(self.report)
        return ProbeAttentionMaps(positions, self.blocks, probes, self.denoiser), None

    def call(self, args, invoke):
        c = args.get("c") or {}
        timestep = args["timestep"]
        step = (
            float(timestep.flatten()[0]) if isinstance(timestep, torch.Tensor) else float(timestep)
        )
        size = tuple(args["input"].shape[2:])
        if self.step is None or step > self.step or size != self.shape:
            if self.report in self.reports:
                self.reports.remove(self.report)
            self.report = ""
            self.collector = None
            self.alphas = {}
            self.stats = []
        elif step != self.step and self.collector is not None:
            maps = self.collector.finish_step()
            if maps:
                count = next(iter(maps.values())).shape[-1]
                grid = None
                if len(size) >= 2:
                    h, w = size[-2:]
                    if h * w == count:
                        grid = (h, w)
                    else:
                        for stride in range(2, min(h, w) + 1):
                            if (
                                h % stride == 0
                                and w % stride == 0
                                and (h // stride) * (w // stride) == count
                            ):
                                grid = (h // stride, w // stride)
                                break
                if grid is not None:
                    self.alphas = alpha_from_maps(maps, self.alphas, grid=grid, share=self.share)
                    self.stats.append(alpha_stats(step, maps, self.alphas))
                    log.info("axis alpha: %s", self.stats[-1])
                elif not self.warned:
                    log.warning("Trigger attention maps are off: image token grid is unknown.")
                    self.warned = True
        self.step, self.shape = step, size
        if self.collector is None:
            try:
                collector, message = self._collector(_conditioning.get() or c)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                IndexError,
                RuntimeError,
            ) as exc:
                collector, message = None, text("auto_alpha.probe_failed", reason=str(exc))
            if collector is not None and self.blocks:
                self.collector = collector
            elif not self.warned:
                log.warning(message or "Trigger attention maps are off: block count is unknown.")
                self.warned = True
        if self.collector is not None:
            c = dict(c)
            c["transformer_options"] = self.collector.patch_options(
                c.get("transformer_options") or {}
            )
            args = {**args, "c": c}
        with axis_alphas(self.alphas):
            return invoke(args)
