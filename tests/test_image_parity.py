# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
)
from test_diffusers import random_lora, tiny_transformer

from kilora import compose, import_lora

pytestmark = pytest.mark.lora_parity


def tiny_pipeline(transformer):
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        norm_num_groups=4,
        scaling_factor=0.5,
        shift_factor=0.1,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    return ZImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer
    )


def generate(pipe):
    embeds = [torch.randn(5, 16, generator=torch.Generator().manual_seed(21))]
    with torch.no_grad():
        out = pipe(
            prompt_embeds=embeds,
            guidance_scale=0.0,
            num_inference_steps=4,
            height=32,
            width=32,
            generator=torch.Generator().manual_seed(1234),
            output_type="np",
        )
    return torch.from_numpy(out.images[0]).float()


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def test_imported_lora_image_matches_plain_lora():
    sd = random_lora(tiny_transformer(), seed=3)

    reference = tiny_transformer()
    reference.load_lora_adapter(dict(sd), prefix="transformer", adapter_name="one")
    reference.set_adapters(["one"], weights=[0.9])
    want = generate(tiny_pipeline(reference))

    plain = generate(tiny_pipeline(tiny_transformer()))

    model = tiny_transformer()
    compose(model, [import_lora(sd, model, weight=0.9).adapter_set], mix="stack")
    got = generate(tiny_pipeline(model))

    assert psnr(want, plain) < 40
    assert psnr(got, want) > 40
