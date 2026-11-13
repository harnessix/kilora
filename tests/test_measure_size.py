# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.measure_size import (
    DEFAULT_MEASURE_SIZE,
    REDUCERS,
    check_measure_size,
    decode_resize_encode,
    latent_area,
    measurement_latent,
    register_reducer,
)


class _VAE:
    def __init__(self):
        self.calls = []

    def decode(self, z):
        self.calls.append("decode")
        img = z[:, :3].repeat_interleave(8, -1).repeat_interleave(8, -2)
        return img.movedim(1, -1)

    def encode(self, img):
        self.calls.append("encode")
        x = img.movedim(-1, 1)
        x = torch.nn.functional.avg_pool2d(x, 8)
        return torch.cat([x, x[:, :1]], 1)


def test_latent_area_resizes_to_the_typed_size():
    z = torch.randn(1, 16, 128, 128)
    small = latent_area(z, 512)
    assert small.shape == (1, 16, 64, 64)
    torch.testing.assert_close(small[0, 0, 0, 0], z[0, 0, :2, :2].mean())
    assert latent_area(z, 384).shape[-2:] == (48, 48)
    wide = latent_area(torch.randn(1, 16, 96, 128), 512)
    assert wide.shape[-2:] == (48, 64)


def test_typed_size_is_checked_and_rounded():
    assert check_measure_size(512) == 512
    assert check_measure_size(500) == 496
    assert check_measure_size(1030) == 1024
    for bad in (100, 255, 5000):
        with pytest.raises(ValueError, match="256 to 4096"):
            check_measure_size(bad)
    assert DEFAULT_MEASURE_SIZE == 512


def test_decode_resize_encode_uses_the_vae():
    vae = _VAE()
    z = torch.randn(1, 4, 128, 96)
    small = decode_resize_encode(z, 512, vae)
    assert vae.calls == ["decode", "encode"] and small.shape[-2:] == (64, 48)
    with pytest.raises(ValueError, match="needs a VAE"):
        decode_resize_encode(z, 512, None)


def test_measurement_latent_rules():
    z = torch.randn(1, 16, 128, 128)
    assert measurement_latent(z, size=1024, reducer="latent")[0] is z
    assert measurement_latent(z, size=2048, reducer="latent")[0] is z
    small, note = measurement_latent(z, size=512, reducer="latent")
    assert small.shape[-1] == 64 and note == ""
    small, note = measurement_latent(z, size=512, reducer="decode-resize-encode")
    assert small.shape[-1] == 64 and "No VAE" in note
    tokens = torch.randn(1, 5, 4)
    assert measurement_latent(tokens, size=512, reducer="latent")[0] is tokens
    register_reducer("every-other", lambda z, size, vae: z[..., ::2, ::2])
    assert measurement_latent(z, size=512, reducer="every-other")[0].shape[-1] == 64
    del REDUCERS["every-other"]


def test_latent_area_resizes_frame_latents():
    z = torch.randn(2, 16, 1, 128, 128, generator=torch.Generator().manual_seed(0))
    small = latent_area(z, 512)
    assert small.shape == (2, 16, 1, 64, 64)
    torch.testing.assert_close(small[:, :, 0], latent_area(z[:, :, 0], 512))
    video = torch.randn(1, 4, 3, 96, 64, generator=torch.Generator().manual_seed(1))
    out = latent_area(video, 384)
    assert (
        out.shape[:3] == (1, 4, 3) and out.shape[-2:] == latent_area(video[:, :, 1], 384).shape[-2:]
    )
    torch.testing.assert_close(out[:, :, 1], latent_area(video[:, :, 1], 384))
    latent, _ = measurement_latent(z, size=512, reducer="latent")
    assert latent.shape == (2, 16, 1, 64, 64)
