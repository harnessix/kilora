# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def ensure_mask_resize() -> None:
    from diffusers.models.transformers import transformer_cosmos as tc

    if hasattr(tc, "transforms"):
        return
    import types

    def resize(x, size, interpolation=None):
        return torch.nn.functional.interpolate(x, size=tuple(size), mode="nearest")

    tc.transforms = types.SimpleNamespace(
        functional=types.SimpleNamespace(resize=resize),
        InterpolationMode=types.SimpleNamespace(NEAREST="nearest"),
    )


def load_anima(path: str, dtype: torch.dtype, device: str) -> Any:
    from diffusers import ModularPipeline

    ensure_mask_resize()
    pipe = ModularPipeline.from_pretrained(path)
    pipe.load_components(torch_dtype=dtype)
    for name in ("text_encoder", "text_conditioner", "vae"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
    return pipe


def is_anima(pipe: Any) -> bool:
    return type(pipe).__name__ == "AnimaModularPipeline" or hasattr(pipe, "text_conditioner")


@torch.no_grad()
def encode_caption(pipe: Any, caption: str, device: torch.device) -> torch.Tensor:
    from diffusers.modular_pipelines.anima.before_denoise import AnimaTextConditioningStep
    from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep

    enc = AnimaTextEncoderStep.encode_prompt(
        pipe, caption, prepare_unconditional_embeds=False, device=device
    )
    embeds = AnimaTextConditioningStep._condition_prompt_embeds(
        pipe,
        qwen_prompt_embeds=enc["qwen_prompt_embeds"],
        qwen_attention_mask=enc["qwen_attention_mask"],
        t5_input_ids=enc["t5_input_ids"],
        t5_attention_mask=enc["t5_attention_mask"],
        device=device,
        conditioning_dtype=pipe.text_conditioner.dtype,
        output_dtype=torch.float32,
    )
    return embeds[0].float().cpu()


@torch.no_grad()
def encode_image(pipe: Any, pixels: torch.Tensor) -> torch.Tensor:
    from diffusers.modular_pipelines.anima.encoders import encode_vae_image

    vae = pipe.vae
    device = next(vae.parameters()).device
    latents = encode_vae_image(
        pixels.unsqueeze(0), vae, generator=None, device=device, dtype=vae.dtype
    )
    return latents[0, :, 0].float().cpu()


def flow_loss(
    denoiser: torch.nn.Module,
    batch: Sequence[Any],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    ensure_mask_resize()
    device = next(denoiser.parameters()).device
    x0 = torch.stack([ex.latent for ex in batch]).to(device)
    noise = torch.randn(x0.shape, generator=generator).to(device)
    sigma = torch.sigmoid(torch.randn(len(batch), generator=generator)).to(device)
    s = sigma.view(-1, 1, 1, 1)
    noisy = ((1 - s) * x0 + s * noise).unsqueeze(2).to(dtype)
    longest = max(ex.caption.shape[0] for ex in batch)
    context = torch.stack(
        [
            torch.nn.functional.pad(ex.caption, (0, 0, 0, longest - ex.caption.shape[0]))
            for ex in batch
        ]
    ).to(device, dtype)
    h, w = x0.shape[-2:]
    padding = x0.new_zeros(1, 1, h * 8, w * 8, dtype=dtype)
    out = denoiser(
        hidden_states=noisy,
        timestep=sigma.to(dtype),
        encoder_hidden_states=context,
        padding_mask=padding,
        return_dict=False,
    )[0]
    pred = out[:, :, 0].float()
    return torch.nn.functional.mse_loss(pred, noise - x0)


def generate(
    pipe: Any,
    prompt: str,
    *,
    steps: int,
    guidance: float,
    size: int,
    seed: int,
    negative: str | None = None,
):
    ensure_mask_resize()
    if guidance:
        from diffusers.guiders import ClassifierFreeGuidance

        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance))
    extra = {} if negative is None else {"negative_prompt": negative}
    images = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        height=size,
        width=size,
        generator=torch.Generator().manual_seed(seed),
        output="images",
        **extra,
    )
    return images[0]
