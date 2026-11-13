# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora import devices


def test_cpu_info():
    if not torch.cuda.is_available():
        info = devices.device_info()
        assert info.backend in ("cpu", "mps") and info.fp8 == "none"


def test_fp8_format_per_architecture():
    assert devices.compute_fp8("gfx942") == torch.float8_e4m3fnuz
    assert devices.compute_fp8("gfx950") == torch.float8_e4m3fn
    assert devices.compute_fp8("sm_89") == torch.float8_e4m3fn


def test_scaled_conversion_fits_the_fnuz_range():
    x = torch.linspace(-448, 448, 101)
    q, scale = devices.to_compute_fp8(x, torch.float8_e4m3fnuz)
    assert q.dtype == torch.float8_e4m3fnuz and torch.isfinite(q.float()).all()
    back = q.float() * scale
    assert (back - x).abs().max() / 448 < 0.07
