# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .build import DRAW_RULES, ZImageCollector, build, collect, draw_noise, host_hooks

FAMILY_DEFAULTS = {
    "z-image": (512, 12, 5.0),
    "anima": (512, 20, 4.0),
    "sd15": (512, 20, 7.5),
    "sdxl": (768, 20, 5.0),
}
SAMPLER, SCHEDULER = "euler", "simple"
CACHE_VERSION = 1


def _no_interrupt() -> None:
    return None


class ComfyCollector(ZImageCollector):
    def __init__(
        self,
        patcher: Any,
        clip: Any,
        base_family: str | None = None,
        size: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        hooks: str | None = None,
        columns: int = 4,
        interrupt: Callable[[], None] | None = None,
    ):
        from kilora.comfyui import patching

        self.patcher, self.clip = patcher, clip
        self.binding = patching.detect_family(patcher)
        self.family = base_family or self.binding.family
        if self.family not in FAMILY_DEFAULTS:
            raise ValueError(
                f"no pack collection for base family {self.family!r}; "
                f"supported: {', '.join(FAMILY_DEFAULTS)}"
            )
        default_size, default_steps, default_guidance = FAMILY_DEFAULTS[self.family]
        self.size = int(size or default_size)
        self.steps = int(steps or default_steps)
        self.guidance = float(default_guidance if guidance is None else guidance)
        self.rule = DRAW_RULES[self.family]
        self.interrupt = interrupt or _no_interrupt
        self.model = patcher.model.diffusion_model
        self.device = torch.device(getattr(patcher, "load_device", None) or "cpu")
        sessions = patching.sessions_of(patcher) if patching.applied(patcher) else []
        if len(sessions) > 1:
            raise ValueError("several live mods are loaded: connect a model with one live mod")
        self.session = sessions[0] if sessions else None
        if self.session is not None:
            self._init_mod(self.session)
            self.live_key = None
        else:
            shapes = patching.diffusers_shapes(patcher, self.binding)
            names = host_hooks(list(shapes), self.family, hooks)
            self._init_host_state(names, {n: shapes[n] for n in names}, hooks, columns)
        tree = dict(self.model.named_modules())
        places = {}
        for name in self.names:
            place = self.binding.to_comfy(name)
            path = place.path.removeprefix("diffusion_model.")
            if path not in tree:
                raise ValueError(f"the model has no layer {path} for the hook point {name}")
            places[name] = (tree[path], place.part)
        self._register(places)
        self.empty = self._embed("")

    @property
    def path(self) -> str:
        return "mod" if self.session is not None else "host"

    def _embed(self, text: str) -> Any:
        return self.clip.encode_from_tokens_scheduled(self.clip.tokenize(text))

    def generate(self, prompt, seed):
        import comfy.sample

        from kilora.comfyui.prompt_solve import empty_latent

        self.interrupt()
        self.keys, self.counts = {}, {}
        positive = self._embed(prompt)
        latent = empty_latent(self.patcher, self.size)
        noise = comfy.sample.prepare_noise(latent, seed)
        try:
            with torch.no_grad():
                out = comfy.sample.sample(
                    self.patcher,
                    noise,
                    self.steps,
                    self.guidance,
                    SAMPLER,
                    SCHEDULER,
                    positive,
                    self.empty,
                    latent,
                    seed=seed,
                    disable_pbar=True,
                )
            keys = {t: (self.keys[t] / max(self.counts[t], 1)).float() for t in self.keys}
        finally:
            self.keys = None
        return out[0].float().cpu(), keys

    def _predict(self, x: torch.Tensor, sigma: torch.Tensor, conditioning: Any) -> torch.Tensor:
        import comfy.samplers
        from comfy.sampler_helpers import convert_cond

        model = self.patcher.model
        cond = comfy.samplers.encode_model_conds(
            model.extra_conds, convert_cond(conditioning), x, x.device, "positive"
        )
        options = getattr(self.patcher, "model_options", {})
        return comfy.samplers.calc_cond_batch(model, [cond], x, sigma, options)[0]

    def grad(self, latent, prompt, seed):
        import comfy.model_management as mm

        self.interrupt()
        sampling = self.patcher.model.model_sampling
        train_steps = len(getattr(sampling, "sigmas", ())) or 1000
        value, noise = draw_noise(seed, latent.shape, self.rule, train_steps)
        sigma = sampling.sigma(value.float()) if self.rule == "step" else value
        sigma = sigma.reshape(1).float().to(self.device)
        x0 = self.patcher.model.process_latent_in(latent[None]).float().to(self.device)
        xt = sampling.noise_scaling(sigma, noise[None].to(self.device), x0)
        conditioning = self.empty if prompt is None else self._embed(prompt)
        mm.load_models_gpu([self.patcher])
        self.du = {
            t: torch.zeros(self.shapes[t], device=self.device, requires_grad=True)
            for t in self.names
        }
        try:
            with torch.enable_grad():
                denoised = self._predict(xt, sigma, conditioning).float()
                scale = sigma.view(-1, *([1] * (x0.dim() - 1)))
                loss = ((denoised - x0) / scale).pow(2).mean()
                gs = torch.autograd.grad(loss, [self.du[t] for t in self.names])
        finally:
            self.du = None
        return {t: g.detach().float().cpu() for t, g in zip(self.names, gs, strict=True)}


def _digest(value: Any, digest: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            digest.update(str(key).encode() + b"\0")
            _digest(value[key], digest)
    else:
        digest.update(repr(value).encode())


def cache_key(collector: ComfyCollector, prompts, draws, seed, model_sha, extra=None) -> str:
    live = None
    if collector.session is not None:
        digest = hashlib.sha256()
        _digest(collector.session.state, digest)
        live = {"mod": collector.mod_id, "state": digest.hexdigest()}
    record = {
        "version": CACHE_VERSION,
        "prompts": {k: list(v) for k, v in prompts.items()},
        "draws": draws,
        "seed": seed,
        "family": collector.family,
        "binding": collector.binding.family,
        "path": collector.path,
        "collection": collector.collection,
        "names": collector.names,
        "size": collector.size,
        "steps": collector.steps,
        "guidance": collector.guidance,
        "sampler": [SAMPLER, SCHEDULER],
        "model_sha": model_sha,
        "live": live,
        "extra": extra,
    }
    text = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


@dataclass
class PackBuild:
    folder: Path
    packs: dict[str, Path]
    directions: dict[str, int]
    events: int
    reused: int
    seconds: float
    estimate: str = ""
    events_folder: Path | None = None
    lines: list[str] = field(default_factory=list)


def build_pack(
    model: Any,
    clip: Any,
    prompts: Mapping[str, Sequence[str]],
    out: str | os.PathLike[str],
    *,
    cache: str | os.PathLike[str] | None = None,
    base_family: str | None = None,
    draws: int = 2,
    common: int = 48,
    family: int = 16,
    orthogonalize: bool = False,
    size: int | None = None,
    steps: int | None = None,
    seed: int = 0,
    hooks: str | None = None,
    columns: int = 4,
    license: str | None = None,
    extra: Any = None,
    interrupt: Callable[[], None] | None = None,
    progress: Callable[[int], None] | None = None,
    log: Callable[[str], None] = print,
) -> PackBuild:

    from kilora.comfyui.prompt_solve import state_sha

    start = time.monotonic()
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tick = progress or (lambda _count: None)
    lines: list[str] = []
    estimate = ""

    def note(line: str) -> None:
        nonlocal estimate
        lines.append(line)
        log(line)
        if line.startswith("Covariance estimates"):
            estimate = line
        if line.startswith("event ") and line.endswith(" done"):
            tick(1)

    collector = ComfyCollector(
        model, clip, base_family, size, steps, hooks=hooks, columns=columns, interrupt=interrupt
    )
    try:
        key = cache_key(collector, prompts, draws, seed, state_sha(model), extra)
        events = Path(cache) / key if cache is not None else out.with_name(out.name + ".events")
        counts = [(fam, len(items)) for fam, items in prompts.items()]
        reused = sum((events / f"{fam}-{i:03d}.pt").exists() for fam, n in counts for i in range(n))
        if reused:
            note(f"Reused {reused} cached events from {events}")
            tick(reused)
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            files = collect(collector, prompts, events, draws=draws, seed=seed, log=note)
            (interrupt or _no_interrupt)()
            result = build(
                files,
                collector.names,
                collector.shapes,
                tmp,
                collector.family,
                common=common,
                family=family,
                license=license,
                orthogonalize=orthogonalize,
            )
            old = out.with_name(out.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            if out.exists():
                os.replace(out, old)
            os.replace(tmp, out)
            shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    finally:
        collector.close()
    packs = {fam: out / path.name for fam, path in result.packs.items()}
    directions = {c.family: int(c.q.shape[0]) for c in result.editor.chunks}
    return PackBuild(
        out,
        packs,
        directions,
        len(files),
        reused,
        time.monotonic() - start,
        estimate,
        events,
        lines,
    )
