# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .adapter import KiLoRA, attach
from .compute import gate_summary, recording_gates
from .config import ROUTING_SCOPES, ROUTING_SELECTS, KiLoRAConfig, check_depth
from .naming import FAMILY_SHORT, MOD_SUFFIX, MOD_SUFFIXES, strip_suffix
from .presets import get_preset
from .trainers import live_trainer

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_RECOMMENDED = {"default_depth": 1, "default_strength": 1.0}
TRAIN_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {"steps": 500, "rank": 1, "num_experts": 8, "hooks": "attention", **_RECOMMENDED},
    "standard": {
        "steps": 2000,
        "rank": 1,
        "num_experts": 16,
        "hooks": "attention",
        **_RECOMMENDED,
    },
    "strong": {"steps": 4000, "rank": 2, "num_experts": 16, "hooks": "full", **_RECOMMENDED},
}


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    values = TRAIN_PRESETS[getattr(args, "preset", None) or "standard"]
    for key, value in values.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def hook_patterns(args: argparse.Namespace, preset: Any) -> list[str]:
    if args.target_modules:
        return list(args.target_modules)
    if args.hooks == "full" and preset.full_target_modules:
        return list(preset.full_target_modules)
    return list(preset.target_modules)


@dataclass
class Example:
    latent: torch.Tensor
    caption: torch.Tensor


def load_pipeline(model: str, dtype: torch.dtype, device: str) -> Any:

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
    for name in ("text_encoder", "vae"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
    return pipe


def swap_to_denoiser(pipe: Any, device: str) -> None:
    for name in ("text_encoder", "text_conditioner", "vae"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pipe.transformer.to(device)


def find_images(folder: str | os.PathLike[str]) -> list[tuple[Path, str]]:
    pairs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            caption_file = path.with_suffix(".txt")
            caption = (
                caption_file.read_text(encoding="utf-8").strip() if caption_file.exists() else ""
            )
            pairs.append((path, caption))
    if not pairs:
        raise SystemExit(f"no images in {folder}")
    return pairs


def load_image(path: Path, resolution: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = resolution / min(img.size)
        size = (
            max(resolution, round(img.width * scale)),
            max(resolution, round(img.height * scale)),
        )
        img = img.resize(size, Image.Resampling.BICUBIC)
        left = (img.width - resolution) // 2
        top = (img.height - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))
        data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    data = data.reshape(resolution, resolution, 3).permute(2, 0, 1).float()
    return data / 127.5 - 1.0


@torch.no_grad()
def prepare_examples(
    pipe: Any, pairs: Sequence[tuple[Path, str]], resolution: int
) -> list[Example]:
    vae = pipe.vae
    encoder = getattr(pipe, "text_encoder", None)
    if isinstance(encoder, torch.nn.Module):
        device = next(encoder.parameters()).device
    else:
        device = next(pipe.transformer.parameters()).device
    vae_device = next(vae.parameters()).device
    examples = []
    for path, caption in pairs:
        pixels = load_image(path, resolution).unsqueeze(0).to(vae_device, vae.dtype)
        latent = vae.encode(pixels).latent_dist.sample()
        latent = (latent - (vae.config.shift_factor or 0.0)) * vae.config.scaling_factor
        embeds, _ = pipe.encode_prompt(caption, device=device, do_classifier_free_guidance=False)
        examples.append(Example(latent=latent[0].float().cpu(), caption=embeds[0].float().cpu()))
    return examples


@torch.no_grad()
def prepare_anima_examples(
    pipe: Any, pairs: Sequence[tuple[Path, str]], resolution: int
) -> list[Example]:
    from .anima import encode_caption, encode_image

    device = next(pipe.text_conditioner.parameters()).device
    return [
        Example(
            latent=encode_image(pipe, load_image(path, resolution)),
            caption=encode_caption(pipe, caption, device),
        )
        for path, caption in pairs
    ]


def flow_matching_loss(
    denoiser: torch.nn.Module,
    batch: Sequence[Example],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:

    device = next(denoiser.parameters()).device
    x0 = torch.stack([ex.latent for ex in batch]).to(device)
    noise = torch.randn(x0.shape, generator=generator).to(device)
    sigma = torch.sigmoid(torch.randn(len(batch), generator=generator)).to(device)
    s = sigma.view(-1, 1, 1, 1)
    noisy = (1 - s) * x0 + s * noise
    inputs = [x.unsqueeze(1).to(dtype) for x in noisy.unbind(0)]
    captions = [ex.caption.to(device, dtype) for ex in batch]
    out = denoiser(inputs, (1 - sigma).to(dtype), captions, return_dict=False)[0]
    pred = torch.stack([o.float() for o in out]).squeeze(2)
    return torch.nn.functional.mse_loss(pred, x0 - noise)


@dataclass
class TrainResult:
    losses: list[float]
    seconds: float
    live_report: dict | None = None


_STOP = [False]


@contextlib.contextmanager
def stop_on_sigterm():
    import signal

    _STOP[0] = False

    def handle(signum, frame):
        _STOP[0] = True

    try:
        old = signal.signal(signal.SIGTERM, handle)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)
        _STOP[0] = False


def train_loop(
    trainable: Sequence[torch.nn.Parameter],
    examples: Sequence[Example],
    loss_fn: Callable[[Sequence[Example], torch.Generator], torch.Tensor],
    *,
    steps: int,
    lr: float,
    batch_size: int = 1,
    seed: int = 0,
    adapter: KiLoRA | None = None,
    on_step: Callable[[int, float], None] | None = None,
    gate_log_every: int = 0,
    time_limit: float | None = None,
) -> TrainResult:
    optimizer = torch.optim.AdamW(list(trainable), lr=lr, weight_decay=0.0)
    order = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    start = time.time()
    deadline = None if time_limit is None else start + 60.0 * float(time_limit)
    for step in range(steps):
        if _STOP[0] or (deadline is not None and time.time() >= deadline):
            why = "SIGTERM" if _STOP[0] else f"the time limit of {time_limit:g} minutes"
            print(f"Stopping after {step} steps ({why}); saving.", file=sys.stderr)
            break
        batch = order.sample(list(examples), k=min(batch_size, len(examples)))
        if adapter is not None:
            adapter.start_step(step)
        log_gate = (
            adapter is not None
            and gate_log_every > 0
            and adapter.settings.output_gate != "none"
            and step % gate_log_every == 0
        )
        with recording_gates() if log_gate else contextlib.nullcontext() as qs:
            loss = loss_fn(batch, generator)
        if log_gate:
            g = gate_summary(qs)
            if g:
                print(
                    f"step {step + 1} output gate q: mean {g['mean']:.3f}, "
                    f"<0.1 {g['low']:.3f}, 0.1-0.9 {g['mid']:.3f}, >0.9 {g['high']:.3f}",
                    file=sys.stderr,
                )
        if adapter is not None:
            loss = adapter.wrap_loss(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"loss is not finite at step {step}")
        losses.append(value)
        if on_step is not None:
            on_step(step, value)
    return TrainResult(losses=losses, seconds=time.time() - start)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kilora-train", description="Train a static KiLoRA mod.")
    p.add_argument("--model", required=True, help="Diffusers model id or folder")
    p.add_argument(
        "--data", default=None, help="folder with images and .txt captions (not with --fast)"
    )
    p.add_argument(
        "--output",
        default=None,
        help="output file (.kmod.safetensors, or .safetensors for --method lora)",
    )
    p.add_argument("--family", default="z-image", help="base model family preset")
    p.add_argument("--method", choices=("kilora", "lora"), default="kilora")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument(
        "--preset",
        choices=tuple(TRAIN_PRESETS),
        default="standard",
        help="steps, rank, experts and hooks: quick, standard or strong",
    )
    p.add_argument("--steps", type=int, default=None, help="default: from --preset")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="minutes; stop before the next step after this time and save (also on SIGTERM)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--save-every", type=int, default=0, help="also save every N steps")
    p.add_argument(
        "--save-precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="precision of the tensors in the mod file",
    )
    g = p.add_argument_group("KiLoRA settings")
    g.add_argument("--target-modules", nargs="+", help="glob patterns; default: --hooks")
    g.add_argument(
        "--hooks",
        choices=("attention", "full"),
        default=None,
        help="attention layers, or attention and feed-forward layers; default: from --preset",
    )
    g.add_argument("--num-experts", type=int, default=None, help="default: from --preset")
    g.add_argument("--routes", type=int, default=4)
    g.add_argument("--rank", type=int, default=None, help="default: from --preset")
    g.add_argument("--router-dim", type=int, default=32)
    g.add_argument(
        "--learned-cross",
        action="store_true",
        help="train a rank-8 routing cross for a static KiLoRA mod",
    )
    g.add_argument("--depth", type=int, default=3)
    g.add_argument("--train-depth", type=int, nargs=2, default=(1, 4), metavar=("LOW", "HIGH"))
    g.add_argument("--last-step-training", action="store_true")
    g.add_argument("--routing-noise", type=float, default=0.0)
    g.add_argument(
        "--routing-scope",
        choices=ROUTING_SCOPES,
        default="token",
        help="token: each token routes on its own logits (default); tensor: the token mean",
    )
    g.add_argument(
        "--routing-select",
        choices=ROUTING_SELECTS,
        default="sparsemax",
        help="sparsemax of the logits (default) or top1: one-hot on the largest",
    )
    g.add_argument(
        "--output-gate",
        choices=("separable", "neuron", "lowrank", "none"),
        default="separable",
        help="a learnable hard mask on each depth step's change (output gate)",
    )
    g.add_argument(
        "--gate-log-every",
        type=int,
        default=10,
        help="log the mean and spread of the output gate's q every N steps (0: never)",
    )
    g.add_argument("--m-in", type=int, default=4, help="rows of each expert's input keys")
    g.add_argument("--m-out", type=int, default=4, help="rows of each expert's output keys")
    g.add_argument(
        "--loss",
        choices=("accumulate", "path", "mix"),
        default="accumulate",
        help="the iteration mode of the training loss (mix: one mode drawn per sample)",
    )
    g.add_argument(
        "--iteration",
        choices=("accumulate", "path"),
        default=None,
        help="the mod's default inference mode; default: the --loss mode (accumulate for mix)",
    )
    g.add_argument(
        "--fixed-point-stop",
        action="store_true",
        help="path mode: stop a sample when its expert is chosen again",
    )
    g.add_argument(
        "--keys-from",
        default=None,
        help="start the input and output keys from a file of --build-keys (default: random)",
    )
    g.add_argument(
        "--carry",
        choices=("drop", "dropped"),
        default="dropped",
        help="how the read dropout makes the state passed to the next step",
    )
    k = p.add_argument_group("subspace keys from data (--build-keys)")
    k.add_argument(
        "--build-keys",
        metavar="OUT",
        default=None,
        help="build subspace keys for --mod on --data and write them to OUT; no training",
    )
    k.add_argument("--mod", default=None, help="the mod to run for --build-keys")
    k.add_argument(
        "--key-source",
        "--source",
        dest="key_source",
        choices=("gradients", "states"),
        default="gradients",
        help="gradients of the loss at each layer output, or the router queries (control)",
    )
    k.add_argument(
        "--fast",
        action="store_true",
        help="one forward pass over --prompts on noise latents (needs --source states, "
        "no --data or --steps); OUT is the .kmod with the new keys written in",
    )
    k.add_argument("--prompts", nargs="+", default=[], help="the prompts of --fast")
    p.add_argument(
        "--static",
        action="store_true",
        help="train a static mod, also when a live trainer plugin is installed",
    )
    plugin = live_trainer()
    if plugin is not None and plugin.add_arguments is not None:
        plugin.add_arguments(p)
    lo = p.add_argument_group("plain LoRA settings (--method lora)")
    lo.add_argument("--lora-rank", type=int, default=16)
    lo.add_argument("--lora-alpha", type=float, default=16.0)
    lo.add_argument("--lora-targets", nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])
    m = p.add_argument_group("mod manifest")
    m.add_argument("--id", default=None)
    m.add_argument("--name", default=None)
    m.add_argument("--author", default="")
    m.add_argument("--license", default="NOASSERTION")
    m.add_argument("--content-source", default="unknown")
    m.add_argument("--description", default="")
    m.add_argument("--data-license", default=None, help="license of the training images")
    m.add_argument(
        "--storage",
        choices=("int8-block",),
        default=None,
        help="smaller file for a static mod: int8 LoRA parts, bfloat16 keys and router",
    )
    m.add_argument("--trigger-words", nargs="+", default=[])
    m.add_argument("--example-prompts", nargs="+", default=[])
    m.add_argument(
        "--default-strength",
        type=float,
        default=None,
        help="recommended strength written into the mod and its card; default: from --preset",
    )
    m.add_argument(
        "--default-depth",
        type=int,
        default=None,
        help="recommended depth written into the mod and its card; default: from --preset",
    )
    return p


def zimage_backend(denoiser: torch.nn.Module, dtype: torch.dtype, examples: Sequence[Example]):
    from .backends.zimage import ZImageBackend

    return ZImageBackend(denoiser, dtype, channels=int(examples[0].latent.shape[0]))


def _build_keys(args, denoiser, examples, loss_fn) -> TrainResult:
    from .adapter import compose
    from .subspace_keys import build_subspace_keys, save_subspace_keys

    if not args.mod:
        raise SystemExit("--build-keys needs --mod")
    adapter = compose(denoiser, [args.mod])
    order = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    start = time.time()
    batches = [
        order.sample(list(examples), k=min(args.batch_size, len(examples)))
        for _ in range(args.steps)
    ]
    keys = build_subspace_keys(
        adapter,
        batches,
        lambda b: loss_fn(b, gen),
        source=args.key_source,
        m_in=args.m_in,
        m_out=args.m_out,
    )
    adapter.remove()
    save_subspace_keys(args.build_keys, keys, args.key_source)
    print(f"Wrote input and output keys ({args.key_source}) to {args.build_keys}", file=sys.stderr)
    return TrainResult(losses=[], seconds=time.time() - start)


FAST_RESOLUTION = 256


def _load_pipe(args: argparse.Namespace, dtype: torch.dtype) -> Any:
    if getattr(args, "family", None) == "anima":
        from .anima import load_anima

        return load_anima(args.model, dtype, args.device)
    return load_pipeline(args.model, dtype, args.device)


def _fast_build_keys(args: argparse.Namespace, pipe: Any) -> TrainResult:
    from .anima import is_anima
    from .fast_keys import PromptCache, fast_state_keys, write_keys
    from .modfile import load_mod

    if not args.mod:
        raise SystemExit("--build-keys needs --mod")
    if args.key_source != "states":
        raise SystemExit("--fast needs --source states")
    if not args.prompts:
        raise SystemExit("--fast needs --prompts")
    start = time.time()
    phases: dict[str, float] = {}

    def lap(name: str) -> None:
        now = time.time()
        phases[name] = now - lap.last
        lap.last = now

    lap.last = start
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    if pipe is None:
        pipe = _load_pipe(args, dtype)
    lap("load pipeline")
    anima = is_anima(pipe)
    if anima:
        from .anima import encode_caption

        device = next(pipe.text_conditioner.parameters()).device
        cache = PromptCache(lambda p: encode_caption(pipe, p, device))
        latent_shape: tuple[int, ...] = (16, FAST_RESOLUTION // 8, FAST_RESOLUTION // 8)
    else:
        encoder = getattr(pipe, "text_encoder", None)
        if isinstance(encoder, torch.nn.Module):
            device = next(encoder.parameters()).device
        else:
            device = next(pipe.transformer.parameters()).device

        def encode(prompt: str) -> torch.Tensor:
            with torch.no_grad():
                embeds, _ = pipe.encode_prompt(
                    prompt, device=device, do_classifier_free_guidance=False
                )
            return embeds[0].float().cpu()

        cache = PromptCache(encode)
        config = pipe.vae.config
        size = FAST_RESOLUTION // 2 ** (len(config.block_out_channels) - 1)
        latent_shape = (int(config.latent_channels), size, size)
    for prompt in args.prompts:
        cache.get(prompt)
    lap("encode prompts")
    swap_to_denoiser(pipe, args.device)
    lap("move the denoiser to the device")
    denoiser = pipe.transformer
    denoiser.requires_grad_(False)
    denoiser.eval()
    timing = []

    def run_model(latents, sigmas, text):
        dev = next(denoiser.parameters()).device
        t0 = time.time()
        if anima:
            from .anima import ensure_mask_resize

            ensure_mask_resize()
            h, w = latents.shape[-2:]
            out = denoiser(
                hidden_states=latents.unsqueeze(2).to(dev, dtype),
                timestep=sigmas.to(dev, dtype),
                encoder_hidden_states=text.to(dev, dtype),
                padding_mask=latents.new_zeros(1, 1, h * 8, w * 8, dtype=dtype, device=dev),
                return_dict=False,
            )[0]
        else:
            captions = []
            for t in text:
                filled = t.abs().sum(-1).nonzero()
                n = int(filled[-1]) + 1 if len(filled) else t.shape[0]
                captions.append(t[:n].to(dev, dtype))
            inputs = [x.unsqueeze(1).to(dev, dtype) for x in latents.unbind(0)]
            out = denoiser(inputs, (1 - sigmas).to(dev, dtype), captions, return_dict=False)[0]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        timing.append(time.time() - t0)
        return out

    from .adapter import compose

    adapter = compose(denoiser, [args.mod])
    lap("compose the mod")
    try:
        keys = fast_state_keys(
            adapter,
            run_model,
            cache,
            args.prompts,
            latent_shape=latent_shape,
            m_in=args.m_in,
            m_out=args.m_out,
            seed=args.seed,
        )
    finally:
        adapter.remove()
    lap("forward pass and keys")
    mod = load_mod(args.mod)
    sets = [write_keys(mod.sets[0], keys, keys.log_tau), *mod.sets[1:]]
    _save_like(args.mod, args.build_keys, mod, sets)
    lap("write the mod")
    keys_time = time.time() - start - phases["load pipeline"]
    breakdown = ", ".join(f"{k} {v:.2f} s" for k, v in phases.items())
    print(
        f"Forward pass: {timing[0]:.2f} s for {len(args.prompts)} prompts. "
        f"Key time without the load: {keys_time:.2f} s ({breakdown}). "
        f"Wrote the state keys into {args.build_keys}",
        file=sys.stderr,
    )
    return TrainResult(losses=[], seconds=keys_time)


def _save_like(source: Any, out: Any, mod: Any, sets: list) -> None:
    from safetensors import safe_open

    from .card import embed_card
    from .modfile import NAMESPACE, PREVIEW_PREFIX, _pack_sets, _store, save_mod

    storage = "int8-block" if "storage.int8-block" in mod.manifest["requires_features"] else None
    packed, _ = _pack_sets(mod.sets, mod.settings, torch.float32)
    if storage is not None:
        packed = _store(packed, storage)
    with safe_open(os.fspath(source), framework="pt") as f:
        extra = {
            n: f.get_tensor(n)
            for n in f.keys()
            if n.startswith(NAMESPACE) and n not in packed and not n.startswith(PREVIEW_PREFIX)
        }
    fixed = ("format", "format_version", "digest", "adapter", "requires_features", "previews")
    fields = {k: v for k, v in mod.manifest.items() if k not in fixed}
    save_mod(
        out,
        sets,
        mod.settings,
        previews=mod.previews,
        extra_tensors=extra,
        storage=storage,
        **fields,
    )
    embed_card(out)


def _enable_training_cross(adapter: KiLoRA, width: int, seed: int):
    from .sections import Section

    generator = torch.Generator().manual_seed(seed)
    device = next(adapter.routers.parameters()).device
    params = torch.nn.ParameterDict(
        {
            "down": torch.nn.Parameter(
                torch.randn(8, width, generator=generator) / math.sqrt(width)
            ),
            "up": torch.nn.Parameter(torch.zeros(width, 8)),
            "bias": torch.nn.Parameter(torch.full((width,), 0.01)),
        }
    ).to(device)
    adapter.add_module("training_cross", params)
    adapter._set_info[0]["learned_cross"] = params
    adapter._set_info[0]["learned_cross_zero"] = False
    return Section("learned-cross", "cross", params)


def run(args: argparse.Namespace, pipe: Any = None) -> TrainResult:
    if args.learned_cross and (args.method != "kilora" or args.build_keys):
        raise ValueError("--learned-cross requires static KiLoRA training")
    if args.build_keys and getattr(args, "fast", False):
        return _fast_build_keys(args, pipe)
    for flag in ("data", "output"):
        if getattr(args, flag) is None:
            raise SystemExit(f"kilora-train: the argument --{flag} is required")
    apply_preset(args)
    if args.method == "kilora":
        from .plugin_support import check_static_experts

        check_static_experts(args.num_experts)
    live_plugin = None
    if not args.static and args.method == "kilora":
        live_plugin = live_trainer()
        if live_plugin is None:
            print(
                "No live trainer plugin is installed, so this trains a static mod.",
                file=sys.stderr,
            )
            args.static = True
    if args.learned_cross and not args.static:
        raise ValueError("--learned-cross requires static KiLoRA training")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    if pipe is None:
        pipe = _load_pipe(args, dtype)
    denoiser = pipe.transformer
    denoiser.requires_grad_(False)
    print(f"Encoding {args.data} ...", file=sys.stderr)
    from .anima import is_anima
    from .data import data_source, image_folder

    anima = is_anima(pipe)
    pairs = find_images(image_folder(args.data))
    if anima:
        examples = prepare_anima_examples(pipe, pairs, args.resolution)
    else:
        examples = prepare_examples(pipe, pairs, args.resolution)
    probes = []
    if live_plugin is not None and live_plugin.prepare is not None:
        probes = live_plugin.prepare(pipe, args, examples)
    swap_to_denoiser(pipe, args.device)
    if args.gradient_checkpointing and hasattr(denoiser, "enable_gradient_checkpointing"):
        denoiser.enable_gradient_checkpointing()
    denoiser.train()

    def report(step: int, loss: float) -> None:
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step + 1}/{args.steps} loss {loss:.4f}", file=sys.stderr)

    def loss_fn(batch, gen):
        if anima:
            from .anima import flow_loss

            return flow_loss(denoiser, batch, gen, dtype)
        return flow_matching_loss(denoiser, batch, gen, dtype)

    if args.method == "lora":
        return _run_lora(args, denoiser, examples, loss_fn, report)
    if args.build_keys:
        return _build_keys(args, denoiser, examples, loss_fn)

    preset = get_preset(args.family)
    config = preset.config(
        target_modules=hook_patterns(args, preset),
        num_experts=args.num_experts,
        routes=args.routes,
        rank=args.rank,
        router_dim=args.router_dim,
        depth=args.depth,
        train_depth=tuple(args.train_depth),
        last_step_training=args.last_step_training,
        routing_noise=args.routing_noise,
        routing_scope=args.routing_scope,
        routing_select=args.routing_select,
        carry=args.carry,
        m_in=args.m_in,
        m_out=args.m_out,
        iteration=args.iteration or ("path" if args.loss == "path" else "accumulate"),
        train_iteration=args.loss,
        fixed_point_stop=args.fixed_point_stop,
        output_gate=args.output_gate,
        seed=args.seed,
    )
    keys, key_source = None, "random"
    if args.keys_from:
        from .subspace_keys import load_subspace_keys

        keys, key_source = load_subspace_keys(args.keys_from)
    live = None
    if args.static:
        adapter = attach(denoiser, config, keys=keys, key_source=key_source)
    else:
        live = live_plugin.start(
            denoiser, config, preset.family, args, zimage_backend(denoiser, dtype, examples)
        )
        adapter = live.adapter
    cross_section = (
        _enable_training_cross(adapter, args.router_dim, args.seed) if args.learned_cross else None
    )
    mod_id = args.id or Path(args.output).stem.lower()
    save_dtype = _DTYPES[getattr(args, "save_precision", "fp32")]
    manifest = {
        "id": mod_id,
        "name": args.name or mod_id,
        "author": args.author,
        "license": args.license,
        "content_source": args.content_source,
        **mod_hints(args),
        "base_model": {"family": preset.family, "checkpoints": [Path(args.model).name]},
        "data_source": data_source(args.data, getattr(args, "data_license", None)),
        **({"license_notice": preset.notice} if preset.notice else {}),
        **(
            {"license": "LicenseRef-CircleStone-Labs-NC-1.2"}
            if preset.notice and args.license == "NOASSERTION"
            else {}
        ),
        "depth": {
            "recommended": check_depth(int(args.default_depth)),
            "trained": [int(d) for d in args.train_depth],
        },
    }
    if live is not None and getattr(args, "mod_family", None):
        manifest["mod_family"] = args.mod_family

    def on_step(step: int, loss: float) -> None:
        report(step, loss)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            stem = strip_suffix(args.output, MOD_SUFFIXES)
            adapter.save(
                f"{stem}.step{step + 1}{MOD_SUFFIX}",
                dtype=save_dtype,
                sections=[cross_section] if cross_section else None,
                **manifest,
            )

    result = train_loop(
        list(adapter.trainable_parameters()),
        examples,
        loss_fn,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        adapter=adapter,
        on_step=on_step,
        gate_log_every=args.gate_log_every,
        time_limit=args.time_limit,
    )
    if live is None:
        storage = getattr(args, "storage", None)
        extra = {"storage": storage} if storage else {}
        adapter.save(
            args.output,
            dtype=save_dtype,
            sections=[cross_section] if cross_section else None,
            **manifest,
            **extra,
        )
    else:
        live.finish(examples, probes)
        live.save(args.output, dtype=save_dtype, **manifest)
        result.live_report = live.report
    from .card import embed_card

    embed_card(args.output)
    adapter.remove()
    print(f"Saved {args.output} ({result.seconds:.0f} s)", file=sys.stderr)
    return result


def _strength(args: argparse.Namespace) -> float:
    value = getattr(args, "default_strength", None)
    return 1.0 if value is None else float(value)


def mod_hints(args: argparse.Namespace) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "strength": {"recommended": _strength(args), "min": -2.0, "max": 2.0},
    }
    if args.description:
        hints["description"] = args.description
    if args.trigger_words:
        hints["trigger_words"] = list(args.trigger_words)
    if args.example_prompts:
        hints["example_prompts"] = list(args.example_prompts)
    return hints


def _run_lora(args, denoiser, examples, loss_fn, report) -> TrainResult:
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file

    denoiser.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=list(args.lora_targets),
            init_lora_weights="gaussian",
        )
    )
    params = [p for p in denoiser.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    result = train_loop(
        params,
        examples,
        loss_fn,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        on_step=report,
        time_limit=getattr(args, "time_limit", None),
    )
    scale = args.lora_alpha / args.lora_rank
    state = {}
    for key, value in get_peft_model_state_dict(denoiser).items():
        value = value.detach().float().cpu()
        if ".lora_B." in key:
            value = value * scale
        state[f"transformer.{key}"] = value.contiguous()
    save_file(state, args.output)
    print(f"Saved {args.output} ({result.seconds:.0f} s)", file=sys.stderr)
    return result


def build_batch_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kilora-train batch",
        description="Train one mod per dataset folder, with resume, logs and a summary.",
    )
    p.add_argument("root", help="folder with one dataset folder per mod")
    p.add_argument("--model", required=True, help="Diffusers model id or folder")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--only", nargs="+", default=None, help="train only these folders")
    p.add_argument("--retrain", action="store_true", help="train again if the mod file exists")
    p.add_argument(
        "--skip-check",
        action="store_true",
        help="train folders that fail kilora data check (not recommended)",
    )
    return p


def run_batch(
    argv: list[str], pipe_factory: Callable[[], Any] | None = None
) -> list[dict[str, Any]]:
    import contextlib
    import io

    from .data import check_folder, dataset_folders, read_manifest, write_report

    batch, rest = build_batch_parser().parse_known_args(argv)
    out = Path(batch.output_dir)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    folders = dataset_folders(batch.root)
    if batch.only:
        folders = [f for f in folders if f.name in batch.only]
    rows: list[dict[str, Any]] = []
    for folder in folders:
        probe = build_parser().parse_args(
            ["--model", batch.model, "--data", str(folder), "--output", "x", *rest]
        )
        short = FAMILY_SHORT.get(probe.family, probe.family)
        target = out / f"{folder.name}.{short}{MOD_SUFFIX}"
        row: dict[str, Any] = {"mod": folder.name, "file": target.name, "status": "", "note": ""}
        rows.append(row)
        if target.exists() and not batch.retrain:
            row["status"], row["note"] = "exists", "skipped; use --retrain to train again"
            continue
        report = check_folder(folder)
        write_report(report, logs)
        if not report.ok and not batch.skip_check:
            row["status"] = "data problems"
            row["note"] = f"{len(report.problems)} problems; see logs/{folder.name}.data-check.md"
            continue
        trigger = read_manifest(folder).get("trigger")
        args = [
            "--model",
            batch.model,
            "--data",
            str(folder),
            "--output",
            str(target),
            "--id",
            folder.name,
            *rest,
        ]
        if trigger and "--trigger-words" not in rest:
            args += ["--trigger-words", trigger]
        log = logs / f"{folder.name}.log"
        buffer = io.StringIO()
        start = time.time()
        try:
            with contextlib.redirect_stderr(buffer):
                result = run(build_parser().parse_args(args), pipe=pipe_factory and pipe_factory())
            row["status"] = "done"
            row["steps"] = len(result.losses)
            row["final_loss"] = round(float(result.losses[-1]), 6) if result.losses else None
        except Exception as exc:
            row["status"], row["note"] = "failed", f"{type(exc).__name__}: {exc}"[:300]
            if target.exists():
                target.unlink()
        row["seconds"] = round(time.time() - start, 1)
        log.write_text(buffer.getvalue() + (f"\n{row['note']}\n" if row["note"] else ""))
    _write_summary(out, rows)
    return rows


def _write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    import json

    lines = [
        "# Batch training summary",
        "",
        "| Mod | Status | Steps | Final loss | Seconds | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['mod']} | {r['status']} | {r.get('steps', '')} | {r.get('final_loss', '')} "
            f"| {r.get('seconds', '')} | {r['note']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":
        rows = run_batch(argv[1:])
        for r in rows:
            print(f"{r['mod']}: {r['status']} {r['note']}".rstrip())
        return 1 if any(r["status"] in ("failed", "data problems") for r in rows) else 0
    plugin = live_trainer()
    if argv and plugin is not None and argv[0] in plugin.commands:
        return int(plugin.commands[argv[0]](argv[1:]) or 0)
    args = build_parser().parse_args(argv)
    with stop_on_sigterm():
        run(args)
    return 0


__all__ = [
    "Example",
    "KiLoRAConfig",
    "TrainResult",
    "find_images",
    "flow_matching_loss",
    "main",
    "prepare_examples",
    "run",
    "train_loop",
]

if __name__ == "__main__":
    raise SystemExit(main())
