# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .editor import Chunk, ChunkEditor
from .plugin import DEFAULT_CAP_FRAC, DEFAULT_WRITE_FRAC, save_chunk_editor


class Collector:
    names: list[str]
    shapes: dict[str, list[int]]
    collection: dict | None = None

    def generate(self, prompt: str, seed: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def grad(self, latent: torch.Tensor, prompt: str | None, seed: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def collect(
    collector: Collector,
    prompts: Mapping[str, Sequence[str]],
    out: str | os.PathLike[str],
    draws: int = 2,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if getattr(collector, "host", False):
        covariance_file = out / "covariance.pt"
        if not covariance_file.exists():
            collector.covariance_sums, collector.covariance_counts = {}, {}
            collector.covariance_index = {}
            collector._covariance_next_group = 0
            estimate = sum(collector.input_widths[n] ** 2 * 4 for n in collector.names)
            log(
                f"Covariance estimates (per-hook upper bound): GPU running sums "
                f"{estimate} bytes; covariance.pt tensor data {estimate} bytes"
            )
            collector.covariance_active = True
            try:
                index = 0
                for items in prompts.values():
                    for prompt in items:
                        index += 1
                        collector.generate(prompt, seed + 1000 * index)
            finally:
                collector.covariance_active = False
            data = {
                "sums": {n: v.cpu() for n, v in collector.covariance_sums.items()},
                "counts": collector.covariance_counts,
                "index": collector.covariance_index,
            }
            torch.save(data, str(covariance_file) + ".tmp")
            os.replace(str(covariance_file) + ".tmp", covariance_file)
        data = torch.load(covariance_file, map_location="cpu", weights_only=False)
        bases, _ = _input_bases(data, collector.names, collector.shapes)
        collector.proj = {n: bases[n].t().contiguous() for n in collector.names}
    files = []
    for family, items in prompts.items():
        for i, prompt in enumerate(items):
            f = out / f"{family}-{i:03d}.pt"
            files.append(f)
            if f.exists():
                if getattr(collector, "host", False):
                    old = torch.load(f, map_location="cpu", weights_only=False)
                    if "keys" in old or "input_frame" in old:
                        raise ValueError("old host events must be collected again")
                continue
            s = seed + 1000 * len(files)
            latent, keys = collector.generate(prompt, s)
            g_empty = _mean([collector.grad(latent, None, s + d) for d in range(draws)])
            g_full = _mean([collector.grad(latent, prompt, s + d) for d in range(draws)])
            event = {"family": family, "prompt": prompt, "seed": s, "draws": draws}
            if collector.collection is not None:
                event["collection"] = dict(collector.collection)
            event.update(g_empty=g_empty, g_full=g_full)
            if not getattr(collector, "host", False):
                event["keys"] = keys
            torch.save(event, str(f) + ".tmp")
            os.replace(str(f) + ".tmp", f)
            log(f"event {f.name} done")
    return files


def _mean(gs: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {h: sum(g[h].float() for g in gs) / len(gs) for h in gs[0]}


def _top(rows: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(k, rows.shape[0])
    _u, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    q = vh[:k]
    sign = torch.sign(q.gather(1, q.abs().argmax(1, keepdim=True)))
    return (q * sign).float(), (s[:k] ** 2 / rows.shape[0]).float()


def _input_bases(data, names, shapes):
    bases, eigenvalues = {}, {}
    if "index" not in data:
        raise ValueError("old covariance.pt has no hook index; collect again")
    if (
        set(data["index"]) != set(names)
        or set(data["sums"]) != set(data["counts"])
        or set(data["index"].values()) != set(data["sums"])
    ):
        raise ValueError("covariance must include every hook and group")
    for h in names:
        group = data["index"][h]
        cov = data["sums"][group].double() / data["counts"][group]
        values, vectors = torch.linalg.eigh(cov)
        count = min(int(shapes[h][1]), len(values))
        values, vectors = values[-count:].flip(0), vectors[:, -count:].flip(1)
        signs = torch.sign(vectors.gather(0, vectors.abs().argmax(0, keepdim=True)))
        bases[h], eigenvalues[h] = (vectors * signs).float(), values.float()
    return bases, eigenvalues


@dataclass
class BuildResult:
    editor: ChunkEditor
    packs: dict[str, Path]
    editor_file: Path


def build(
    events: Sequence[str | os.PathLike[str]],
    names: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
    out: str | os.PathLike[str],
    base_family: str,
    common: int = 48,
    family: int = 16,
    version: str = "1.0.0",
    license: str | None = None,
    write_norm: float = 0.05,
    total_cap: float | None = None,
    write_frac: float | None = DEFAULT_WRITE_FRAC,
    cap_frac: float | None = DEFAULT_CAP_FRAC,
    orthogonalize: bool = False,
) -> BuildResult:
    names = list(names)
    evs = [torch.load(f, map_location="cpu", weights_only=False) for f in events]
    if not evs:
        raise ValueError("no events")
    records = {json.dumps(e.get("collection"), sort_keys=True) for e in evs}
    if len(records) > 1:
        raise ValueError("the events were collected in different ways; collect them again")
    collection = evs[0].get("collection")

    def vec(g):
        return torch.cat([g[h].float().flatten() for h in names])

    host = bool(collection and collection.get("host"))
    rows = {e["family"]: [] for e in evs}
    keys: dict[str, list] = {f: [] for f in rows}
    for e in evs:
        rows[e["family"]] += [vec(e["g_empty"]), vec(e["g_full"])]
        if not host:
            keys[e["family"]].append(e["keys"])
    all_rows = torch.stack([r for f in rows for r in rows[f]])
    q, ev = _top(all_rows, common)
    if host:
        paths = {Path(f).parent / "covariance.pt" for f in events}
        if len(paths) != 1:
            raise ValueError("host events must share a covariance collection")
        data = torch.load(paths.pop(), map_location="cpu", weights_only=False)
        bases, eigenvalues = _input_bases(data, names, shapes)

    def make_chunk(fam, directions, values, covariances):
        if not host:
            return Chunk(fam, directions, values, _mean_keys(covariances, names))
        return Chunk(fam, directions, values, input_basis=bases, input_eigenvalues=eigenvalues)

    chunks = [make_chunk("common", q, ev, [k for f in keys for k in keys[f]])]
    span = q
    for fam in sorted(f for f in rows if not f.startswith("_")):
        r = torch.stack(rows[fam])
        if orthogonalize:
            r = r - (r @ span.t()) @ span
        qf, ef = _top(r, family)
        if orthogonalize:
            qf = qf - (qf @ span.t()) @ span
            qf, _ = torch.linalg.qr(qf.t().double())
            qf = qf.t().float()
        chunks.append(make_chunk(fam, qf, ef, keys[fam]))
        span = torch.cat([span, qf])
    alpha = {"to_q": 1.0, "to_k": 1.0, "to_v": 1.0, "to_out.0": 1.0}
    ed = ChunkEditor(names, {h: list(shapes[h]) for h in names}, alpha, chunks)
    if orthogonalize:
        ed.check_orthogonal()
    out = Path(out)
    license, notice = pack_license(base_family, license)
    packs = ed.save_packs(
        out, version, base_family, license, license_notice=notice, collection=collection
    )
    editor_file = out / "chunk.kedit.safetensors"
    save_chunk_editor(
        editor_file,
        list(packs.values()),
        base_family,
        write_norm=write_norm,
        total_cap=total_cap,
        write_frac=write_frac,
        cap_frac=cap_frac,
        license=license,
        license_notice=notice,
    )
    return BuildResult(ed, packs, editor_file)


FAMILY_LICENSES = {"anima": "LicenseRef-CircleStone-Labs-NC-1.2"}
DEFAULT_LICENSE = "Apache-2.0"


def pack_license(base_family: str, given: str | None = None) -> tuple[str, str | None]:
    from kilora.presets import pack_license as family_license

    return family_license(base_family, given)


def _mean_keys(items: Sequence[Mapping[str, torch.Tensor]], names) -> dict[str, torch.Tensor]:
    items = [k for k in items if k]
    if not items:
        return {}
    return {h: sum(k[h].float() for k in items) / len(items) for h in names if h in items[0]}


DRAW_RULES = {"z-image": "uniform", "anima": "logit", "sd15": "step", "sdxl": "step"}


def draw_noise(
    seed: int, shape: Sequence[int], rule: str, train_steps: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    if rule == "uniform":
        value = torch.rand((), generator=gen) * 0.9 + 0.05
    elif rule == "logit":
        value = torch.sigmoid(torch.randn((), generator=gen))
    elif rule == "step":
        low, high = int(0.05 * train_steps), int(0.95 * train_steps)
        value = torch.randint(low, high, (1,), generator=gen)
    else:
        raise ValueError(f"unknown noise draw {rule!r}")
    return value, torch.randn(tuple(shape), generator=gen)


def _rows(out: torch.Tensor, part: tuple[int, int] | None) -> torch.Tensor:
    if part is None:
        return out
    index, count = part
    width = out.shape[-1] // count
    return out[..., index * width : (index + 1) * width]


def _pad(change: torch.Tensor, part: tuple[int, int] | None) -> torch.Tensor:
    if part is None:
        return change
    index, count = part
    width = change.shape[-1]
    return torch.nn.functional.pad(change, (index * width, (count - index - 1) * width))


def host_hooks(
    candidates: Sequence[str],
    family: str,
    hooks: str | None,
    resolve: Callable[[str], Sequence[str]] | None = None,
) -> list[str]:
    from kilora.presets import get_preset

    preset = get_preset(family)
    patterns = preset.full_target_modules or preset.target_modules
    return [
        n
        for n in candidates
        if (
            fnmatch.fnmatchcase(n, hooks)
            if hooks is not None
            else any(part in ("attn1", "attn2") for part in n.split("."))
        )
        and any(fnmatch.fnmatchcase(n, p) for p in patterns)
        and (resolve is None or n in resolve(n))
    ]


def _set_key(adapter: Any, name: str | None) -> str | None:
    for info in getattr(adapter, "_set_info", []):
        if info["name"] == name:
            return info["extra"].get("live_id", info["name"])
    return None


@dataclass(frozen=True)
class DepthMix:
    full: float = 0.5
    depth_max: int = 7

    def draw(self, seed: int) -> tuple[int, int | None]:
        gen = torch.Generator().manual_seed((int(seed) * 2654435761 + 97) % 2**63)
        full = float(torch.rand((), generator=gen)) < self.full
        depth = int(torch.randint(1, self.depth_max + 1, (), generator=gen))
        step = int(torch.randint(1, depth + 1, (), generator=gen))
        return depth, (None if full else step)

    def record(self) -> dict:
        return {"full": self.full, "one_step": 1.0 - self.full, "depth": [1, self.depth_max]}


class ZImageCollector(Collector):
    def __init__(
        self,
        pipe: Any,
        mod_path: str | None,
        size: int = 512,
        steps: int = 12,
        model=None,
        host_family: str | None = None,
        hooks: str | None = None,
        columns: int = 4,
    ):
        from kilora import compose
        from kilora.live import open_session

        self.pipe, self.size, self.steps = pipe, size, steps
        self.model = pipe.transformer if model is None else model
        if host_family is not None:
            self._init_host(host_family, hooks, columns)
            return
        self.adapter = compose(self.model, [mod_path])
        session = open_session(mod_path)
        self._init_mod(session)
        live = session.mod.manifest.get("live") or {}
        self.live_key = _set_key(self.adapter, live.get("live_set"))
        self._register({t: (self.model.get_submodule(t), None) for t in self.names})
        self.empty = self._embed("")

    def _init_mod(self, session) -> None:
        from kilora.live_basis import default_v0

        self.host = False
        self.names = list(session.state)
        self.mod_id = str(session.mod.manifest.get("id", ""))
        live = session.mod.manifest.get("live") or {}
        rank = int(live.get("rank", 4))
        downs = {t: p["down"].float() for t, p in session.state.items()}
        downs = {t: d.reshape(-1, d.shape[-1]) for t, d in downs.items()}
        cols = {t: d.shape[0] for t, d in downs.items()}
        v0 = default_v0(self.names, cols, seed=int(live.get("v0_seed", 0)), rank=rank)
        self.v0 = {t: v0[t].float() for t in self.names}
        self.proj = {t: v0[t] @ downs[t] for t in self.names}
        self.shapes = {t: [downs[t].shape[1], rank] for t in self.names}
        self.du: dict[str, torch.Tensor] | None = None
        self.keys: dict[str, torch.Tensor] | None = None
        self.counts: dict[str, int] = {}

    def _register(self, places: Mapping[str, tuple[Any, tuple[int, int] | None]]) -> None:
        handles = self.__dict__.setdefault("handles", [])
        if self.host:
            handles.append(self.model.register_forward_pre_hook(self._start_covariance))
            handles.append(
                self.model.register_forward_hook(self._finish_covariance, always_call=True)
            )
        for name, (module, part) in places.items():
            handles.append(module.register_forward_hook(self._hook(name, part)))

    def close(self) -> None:
        for handle in self.__dict__.pop("handles", []):
            handle.remove()

    def _init_host(self, family, hooks, columns):
        from kilora.lora_import import _resolver
        from kilora.presets import get_preset

        modules = dict(self.model.named_modules())
        linear = {n: m for n, m in modules.items() if isinstance(m, torch.nn.Linear)}
        preset = get_preset(family)
        resolve = _resolver("diffusers", list(modules), preset.aliases(list(linear)), strict=False)
        names = host_hooks(list(linear), family, hooks, resolve)
        widths = {n: (linear[n].in_features, linear[n].out_features) for n in names}
        self._init_host_state(names, widths, hooks, columns)
        self._register({n: (linear[n], None) for n in self.names})
        self.empty = self._embed("")

    def _init_host_state(self, names, widths, hooks, columns) -> None:
        if columns < 1:
            raise ValueError("columns must be positive")
        if not names:
            raise ValueError("no host hooks match the family and --hooks")
        if any(columns > widths[n][0] for n in names):
            raise ValueError("columns exceeds a hook's input width")
        self.names = list(names)
        self.host = True
        self.collection = {"host": True, "hooks": hooks, "columns": columns}
        self.shapes = {n: [widths[n][1], columns] for n in self.names}
        self.input_widths = {n: widths[n][0] for n in self.names}
        self.proj = {}
        self.covariance_active = False
        self.covariance_sums = {}
        self.covariance_counts = {}
        self.covariance_index = {}
        self._covariance_next_group = 0
        self._covariance_seen = {}
        self._covariance_start_index = {}
        self._covariance_remaining = {}
        self.du = None
        self.keys = None
        self.counts = {}

    def _hook(self, name, part=None):
        def hook(_module, _inputs, full):
            out = _rows(full, part)
            source = _inputs[0] if self.host else out
            if self.host and self.covariance_active:
                key = (
                    source.data_ptr(),
                    tuple(source.shape),
                    tuple(source.stride()),
                    source.device,
                )
                old_group = self._covariance_start_index.get(name)
                partition = (old_group, key)
                seen = self._covariance_seen.get(partition)
                if seen is None or seen[1]() is None:
                    group = self._covariance_next_group
                    self._covariance_next_group += 1
                    flat = source.detach().reshape(-1, source.shape[-1]).float()
                    if old_group is None:
                        total = flat.t() @ flat
                    elif len(self._covariance_remaining[old_group]) == 1:
                        total = self.covariance_sums.pop(old_group)
                        total.addmm_(flat.t(), flat)
                    else:
                        total = self.covariance_sums[old_group].clone()
                        total.addmm_(flat.t(), flat)
                    self.covariance_sums[group] = total
                    self.covariance_counts[group] = (
                        self.covariance_counts.get(old_group, 0) + flat.shape[0]
                    )
                    self._covariance_seen[partition] = (group, weakref.ref(source))
                self.covariance_index[name] = self._covariance_seen[partition][0]
                if old_group is not None:
                    remaining = self._covariance_remaining[old_group]
                    remaining.discard(name)
                    if not remaining:
                        self.covariance_sums.pop(old_group, None)
                        self.covariance_counts.pop(old_group, None)
            if self.host and self.du is None:
                return full
            p = self.proj[name].to(out.device, torch.float32)
            c = source.float() @ p.t()
            if not self.host and self.keys is not None:
                flat = c.detach().reshape(-1, c.shape[-1]).double()
                self.keys[name] = self.keys.get(name, 0) + (flat.t() @ flat).cpu()
                self.counts[name] = self.counts.get(name, 0) + flat.shape[0]
            if self.du is None:
                return full
            change = (c @ self.du[name].to(out.device).t()).to(out.dtype)
            return full + _pad(change, part)

        return hook

    def _start_covariance(self, _module, _inputs):
        if self.covariance_active:
            self._covariance_seen.clear()
            self._covariance_start_index = dict(self.covariance_index)
            self._covariance_remaining = {}
            for name, group in self._covariance_start_index.items():
                self._covariance_remaining.setdefault(group, set()).add(name)

    def _finish_covariance(self, _module, _inputs, _output):
        if self.covariance_active:
            used = set(self.covariance_index.values())
            for group in set(self.covariance_sums) - used:
                del self.covariance_sums[group]
                del self.covariance_counts[group]
            self._covariance_seen.clear()
            self._covariance_start_index.clear()
            self._covariance_remaining.clear()

    def _embed(self, text: str) -> torch.Tensor:
        dev = next(self.pipe.text_encoder.parameters()).device
        with torch.no_grad():
            e, _ = self.pipe.encode_prompt(text, device=dev, do_classifier_free_guidance=False)
        return e[0].to(self.model.device)

    def generate(self, prompt, seed):
        self.keys, self.counts = {}, {}
        emb = self._embed(prompt)
        parts = (self.pipe.text_encoder, self.pipe.vae)
        self.pipe.text_encoder = self.pipe.vae = None
        try:
            with torch.no_grad():
                lat = self.pipe(
                    prompt_embeds=[emb],
                    negative_prompt_embeds=[self.empty],
                    height=self.size,
                    width=self.size,
                    num_inference_steps=self.steps,
                    guidance_scale=5.0,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    output_type="latent",
                ).images
            keys = {t: (self.keys[t] / max(self.counts[t], 1)).float() for t in self.keys}
        finally:
            self.keys = None
            self.pipe.text_encoder, self.pipe.vae = parts
        x0 = lat[0].float().cpu()
        return (x0[:, 0] if x0.dim() == 4 else x0), keys

    def grad(self, latent, prompt, seed):
        sigma, noise = draw_noise(seed, latent.shape, "uniform")
        dev, dtype = self.model.device, self.model.dtype
        x = latent.to(dev)
        n = noise.to(dev)
        xt = (1 - sigma) * x + sigma * n
        emb = self.empty if prompt is None else self._embed(prompt)
        self.du = {
            t: torch.zeros(self.shapes[t], device=dev, requires_grad=True) for t in self.names
        }
        self.model.enable_gradient_checkpointing()
        try:
            with torch.enable_grad():
                out = self.model(
                    [xt.unsqueeze(1).to(dtype)],
                    (1 - sigma).view(1).to(dev, dtype),
                    [emb.to(dev, dtype)],
                    return_dict=False,
                )[0]
                pred = torch.stack([o.float() for o in out]).squeeze(2)[0]
                loss = (pred - (x - n)).pow(2).mean()
                gs = torch.autograd.grad(loss, [self.du[t] for t in self.names])
        finally:
            self.du = None
            self.model.disable_gradient_checkpointing()
        return {t: g.detach().float().cpu() for t, g in zip(self.names, gs, strict=True)}


class UNetCollector(ZImageCollector):
    def __init__(self, pipe, mod_path, size=512, steps=20, guidance=7.5, **host_options):
        self.guidance = guidance
        self.xl = hasattr(pipe, "text_encoder_2")
        super().__init__(pipe, mod_path, size, steps, model=pipe.unet, **host_options)

    def _embed(self, text):
        dev = next(self.pipe.text_encoder.parameters()).device
        with torch.no_grad():
            out = self.pipe.encode_prompt(
                text, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
        emb = out[0].to(self.model.device)
        pooled = out[2].to(self.model.device) if self.xl else None
        return emb, pooled

    def generate(self, prompt, seed):
        self.keys, self.counts = {}, {}
        (emb, pooled), (nemb, npooled) = self._embed(prompt), self.empty
        extra = (
            {"pooled_prompt_embeds": pooled, "negative_pooled_prompt_embeds": npooled}
            if self.xl
            else {}
        )
        try:
            with torch.no_grad():
                lat = self.pipe(
                    prompt_embeds=emb,
                    negative_prompt_embeds=nemb,
                    height=self.size,
                    width=self.size,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    output_type="latent",
                    **extra,
                ).images
            keys = {t: (self.keys[t] / max(self.counts[t], 1)).float() for t in self.keys}
        finally:
            self.keys = None
        return lat[0].float().cpu(), keys

    def grad(self, latent, prompt, seed):
        sched = self.pipe.scheduler
        t, noise = draw_noise(seed, latent.shape, "step", int(sched.config.num_train_timesteps))
        dev, dtype = self.model.device, self.model.dtype
        x, n, tt = latent[None].to(dev), noise[None].to(dev), t.to(dev)
        xt = sched.add_noise(x, n, tt)
        if sched.config.prediction_type == "v_prediction":
            target = sched.get_velocity(x, n, tt)
        else:
            target = n
        emb, pooled = self.empty if prompt is None else self._embed(prompt)
        cond = {}
        if self.xl:
            ids = torch.tensor([[self.size, self.size, 0, 0, self.size, self.size]], device=dev)
            cond = {
                "added_cond_kwargs": {"text_embeds": pooled.to(dtype), "time_ids": ids.to(dtype)}
            }
        self.du = {
            h: torch.zeros(self.shapes[h], device=dev, requires_grad=True) for h in self.names
        }
        self.model.enable_gradient_checkpointing()
        try:
            with torch.enable_grad():
                pred = self.model(xt.to(dtype), tt, encoder_hidden_states=emb.to(dtype), **cond)
                loss = (pred.sample.float() - target.float()).pow(2).mean()
                gs = torch.autograd.grad(loss, [self.du[h] for h in self.names])
        finally:
            self.du = None
            self.model.disable_gradient_checkpointing()
        return {h: g.detach().float().cpu() for h, g in zip(self.names, gs, strict=True)}


def check_family_targets(names: Sequence[str], family: str) -> None:
    import fnmatch

    from kilora.presets import get_preset

    preset = get_preset(family)
    patterns = tuple(preset.full_target_modules or preset.target_modules)
    stray = [n for n in names if not any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if stray:
        raise ValueError(
            f"the live mod's hook points {stray[:3]} are not {family} targets; "
            "it is not a mod of this base model"
        )


class AnimaCollector(ZImageCollector):
    family = "anima"

    def __init__(
        self,
        pipe,
        mod_path,
        size=512,
        steps=20,
        guidance=4.0,
        encode=None,
        path="hook",
        mix: DepthMix | None = None,
        **host_options,
    ):
        from kilora.anima import encode_caption, ensure_mask_resize

        if path not in ("hook", "mod"):
            raise ValueError(f"path {path!r}: use 'hook' or 'mod'")
        ensure_mask_resize()
        self.guidance = guidance
        self.path, self.mix = path, (mix or DepthMix()) if path == "mod" else None
        if encode is None:

            def encode(text):
                dev = next(pipe.text_encoder.parameters()).device
                return encode_caption(pipe, text, dev)

        self._encode = encode
        if host_options.get("host_family") and path != "hook":
            raise ValueError("--host requires the hook path")
        super().__init__(pipe, mod_path, size, steps, model=pipe.transformer, **host_options)
        check_family_targets(self.names, self.family)
        if self.path == "mod":
            if self.live_key is None:
                raise ValueError("the mod has no live set to collect for")
            self.collection = {
                "path": "mod",
                "mod": self.mod_id,
                "read_dropout": True,
                "depth_mix": self.mix.record(),
            }

    def _embed(self, text):
        return self._encode(text).float().to(self.model.device)

    def _run(self, x, sigma, emb):
        dev, dtype = self.model.device, self.model.dtype
        h, w = x.shape[-2:]
        return self.model(
            hidden_states=x.to(dev, dtype),
            timestep=sigma.reshape(1).to(dev, dtype),
            encoder_hidden_states=emb[None].to(dev, dtype),
            padding_mask=torch.zeros(1, 1, h * 8, w * 8, device=dev, dtype=dtype),
            return_dict=False,
        )[0].float()

    def generate(self, prompt, seed):
        self.keys, self.counts = {}, {}
        emb = self._embed(prompt)
        sched = self.pipe.scheduler
        side = self.size // 8
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 16, 1, side, side, generator=gen).to(self.model.device)
        sched.set_timesteps(self.steps, device=self.model.device)
        n_train = float(sched.config.num_train_timesteps)
        try:
            with torch.no_grad():
                for t in sched.timesteps:
                    sigma = (t / n_train).float()
                    cond = self._run(x, sigma, emb)
                    if self.guidance and self.guidance != 1.0:
                        uncond = self._run(x, sigma, self.empty)
                        cond = uncond + self.guidance * (cond - uncond)
                    x = sched.step(cond, t, x, return_dict=False)[0]
            keys = {t: (self.keys[t] / max(self.counts[t], 1)).float() for t in self.keys}
        finally:
            self.keys = None
        return x[0, :, 0].float().cpu(), keys

    def grad(self, latent, prompt, seed):
        if self.path == "mod":
            return self.grad_mod(latent, prompt, seed, *self.mix.draw(seed))
        sigma, noise = draw_noise(seed, latent.shape, "logit")
        dev = self.model.device
        x0, n = latent.to(dev), noise.to(dev)
        xt = ((1 - sigma) * x0 + sigma * n)[None, :, None]
        emb = self.empty if prompt is None else self._embed(prompt)
        self.du = {
            t: torch.zeros(self.shapes[t], device=dev, requires_grad=True) for t in self.names
        }
        checkpointing = getattr(self.model, "_supports_gradient_checkpointing", False)
        if checkpointing:
            self.model.enable_gradient_checkpointing()
        try:
            with torch.enable_grad():
                pred = self._run(xt, sigma, emb)[0, :, 0]
                loss = (pred - (n - x0)).pow(2).mean()
                gs = torch.autograd.grad(loss, [self.du[t] for t in self.names])
        finally:
            self.du = None
            if checkpointing:
                self.model.disable_gradient_checkpointing()
        return {t: g.detach().float().cpu() for t, g in zip(self.names, gs, strict=True)}

    def grad_mod(self, latent, prompt, seed, depth, step=None):
        from kilora.call_values import depth_pass, up_deltas
        from kilora.compute import dropout_key, eager

        sigma, noise = draw_noise(seed, latent.shape, "logit")
        dev = self.model.device
        x0, n = latent.to(dev), noise.to(dev)
        xt = ((1 - sigma) * x0 + sigma * n)[None, :, None]
        emb = self.empty if prompt is None else self._embed(prompt)
        du = {t: torch.zeros(self.shapes[t], device=dev, requires_grad=True) for t in self.names}
        checkpointing = getattr(self.model, "_supports_gradient_checkpointing", False)
        if checkpointing:
            self.model.enable_gradient_checkpointing()
        try:
            with (
                torch.enable_grad(),
                eager(),
                dropout_key(seed, sigma),
                depth_pass(depth, step),
            ):
                deltas = {t: (du[t] @ self.v0[t].to(dev)).t() for t in self.names}
                with up_deltas({self.live_key: deltas}):
                    pred = self._run(xt, sigma, emb)[0, :, 0]
                    loss = (pred - (n - x0)).pow(2).mean()
                    gs = torch.autograd.grad(loss, [du[t] for t in self.names])
        finally:
            if checkpointing:
                self.model.disable_gradient_checkpointing()
        return {t: g.detach().float().cpu() for t, g in zip(self.names, gs, strict=True)}


def repack(
    events: Sequence[str | os.PathLike[str]],
    packs: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    **editor: Any,
) -> BuildResult:
    from kilora.editor_packs import read_pack

    read = [read_pack(f) for f in packs]
    common = [(m, t) for m, t in read if m.role == "common"]
    if len(common) != 1:
        raise ValueError("give the old packs with exactly one common pack")
    m, t = common[0]
    chunk = m.raw["chunk"]
    sizes = [int(tt["q"].shape[0]) for mm, tt in read if mm.role == "family"]
    return build(
        events,
        chunk["hooks"],
        chunk["shapes"],
        out,
        m.base_families[0],
        common=int(t["q"].shape[0]),
        family=max(sizes) if sizes else 16,
        version=m.version,
        license=m.license,
        **editor,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kilora-editor-chunk")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-packs", help="collect gradient steps and build the packs")
    b.add_argument(
        "--model", required=True, help="Diffusers model folder (Z-Image, Anima, SD 1.5, SDXL)"
    )
    source = b.add_mutually_exclusive_group(required=True)
    source.add_argument("--mod", help="the live mod (.kmod) that the packs are for")
    source.add_argument("--host", action="store_true", help="collect on the base model")
    b.add_argument("--hooks", default=None, help="host hook glob (default: attention projections)")
    b.add_argument("--columns", type=int, default=4, help="host columns per hook (default: 4)")
    b.add_argument("--prompts", required=True, help='JSON: {"family": ["prompt", ...], ...}')
    b.add_argument("--out", required=True, help="output folder (events/ and the packs)")
    b.add_argument("--base-family", default="z-image", choices=("z-image", "anima", "sd15", "sdxl"))
    b.add_argument("--size", type=int, default=None, help="image size (Anima: explicitly use 512)")
    b.add_argument(
        "--steps", type=int, default=None, help="sampler steps (Anima: explicitly use 20)"
    )
    b.add_argument("--variant", default=None, help='weights variant, for example "fp16"')
    b.add_argument("--draws", type=int, default=2)
    b.add_argument("--common", type=int, default=48)
    b.add_argument("--family", type=int, default=16)
    b.add_argument("--version", default="1.0.0")
    b.add_argument(
        "--license",
        default=None,
        help="licence id of the packs (default: the base model family's; Anima packs are "
        "non-commercial Derivatives, LicenseRef-CircleStone-Labs-NC-1.2, with the Anima "
        "notice)",
    )
    b.add_argument("--orthogonalize", action="store_true", help="orthogonalize family directions")
    b.add_argument("--write-norm", type=float, default=0.05)
    b.add_argument(
        "--total-cap",
        type=float,
        default=None,
        help="cap of the whole live state, as the norm of its change (default: 10 write norms)",
    )
    b.add_argument(
        "--write-frac",
        type=float,
        default=DEFAULT_WRITE_FRAC,
        help="one write as a share of the live state's norm (0: use --write-norm)",
    )
    b.add_argument(
        "--cap-frac",
        type=float,
        default=DEFAULT_CAP_FRAC,
        help="the cap as a share of the live state's norm (0: use --total-cap)",
    )
    b.add_argument(
        "--path",
        default="hook",
        choices=("hook", "mod"),
        help="Anima: 'mod' takes the gradients on the mod's own path (host experts, depth "
        "loop, read dropout) with a mix of full-chain and one-step-replay gradients; "
        "'hook' adds the live change after each layer, at one step",
    )
    b.add_argument("--full-share", type=float, default=0.5, help="mod path: full-chain share")
    b.add_argument("--depth-max", type=int, default=7, help="mod path: depths 1..depth-max")
    b.add_argument("--device", default="cuda:0", help="the device of the denoiser")
    b.add_argument(
        "--aux-device", default=None, help="the text encoder and VAE device (default: --device)"
    )
    r = sub.add_parser(
        "repack", help="build the packs again from their event files (no new collection)"
    )
    r.add_argument("--events", required=True, help="the events/ folder of the old build")
    r.add_argument("--packs", required=True, nargs="+", help="the old pack files")
    r.add_argument("--out", required=True, help="a new output folder")
    a = p.parse_args(argv)
    if a.command == "repack":
        files = sorted(Path(a.events).glob("*.pt"))
        if not files:
            p.error(f"--events {a.events}: no event files")
        result = repack(files, a.packs, a.out)
        print(f"Wrote {len(result.packs)} packs and {result.editor_file}")
        return 0
    if a.mod and not Path(a.mod).is_file():
        p.error(f"--mod {a.mod}: not a mod file (a live mod is one .kmod file, not a folder)")
    if a.host and a.path != "hook":
        p.error("--host requires --path hook")
    if a.host and a.columns < 1:
        p.error("--columns must be positive")
    host_options = (
        {"host_family": a.base_family, "hooks": a.hooks, "columns": a.columns} if a.host else {}
    )
    prompts = json.loads(Path(a.prompts).read_text())
    from diffusers import DiffusionPipeline

    defaults = {"z-image": (512, 12), "anima": (512, 20), "sd15": (512, 20), "sdxl": (768, 20)}[
        a.base_family
    ]
    size, steps = a.size or defaults[0], a.steps or defaults[1]
    if a.base_family == "z-image":
        pipe = DiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16)
        pipe.transformer.to(a.device)
        for part in ("text_encoder", "vae"):
            getattr(pipe, part).to(a.aux_device or a.device)
        collector = ZImageCollector(pipe, a.mod, size=size, steps=steps, **host_options)
    elif a.base_family == "anima":
        from kilora.anima import load_anima

        pipe = load_anima(a.model, torch.bfloat16, a.aux_device or a.device)
        pipe.transformer.to(a.device).requires_grad_(False)
        mix = DepthMix(full=a.full_share, depth_max=a.depth_max)
        collector = AnimaCollector(
            pipe, a.mod, size=size, steps=steps, path=a.path, mix=mix, **host_options
        )
    else:
        extra = (
            {"safety_checker": None, "feature_extractor": None} if a.base_family == "sd15" else {}
        )
        pipe = DiffusionPipeline.from_pretrained(
            a.model, torch_dtype=torch.float16, variant=a.variant, **extra
        )
        pipe.to(a.device)
        guidance = 7.5 if a.base_family == "sd15" else 5.0
        collector = UNetCollector(
            pipe, a.mod, size=size, steps=steps, guidance=guidance, **host_options
        )
    files = collect(collector, prompts, Path(a.out) / "events", draws=a.draws)
    r = build(
        files,
        collector.names,
        collector.shapes,
        a.out,
        a.base_family,
        common=a.common,
        family=a.family,
        version=a.version,
        license=a.license,
        write_norm=a.write_norm,
        total_cap=a.total_cap,
        write_frac=a.write_frac or None,
        cap_frac=a.cap_frac or None,
        orthogonalize=a.orthogonalize,
    )
    for fam, f in r.packs.items():
        print(f"{fam}: {f.name} ({os.path.getsize(f) / 2**20:.1f} MB)")
    print(f"editor: {r.editor_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
