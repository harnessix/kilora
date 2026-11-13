# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.quant import FORMATS, dequantize, load_quantized, quantize, save_quantized, stored_bytes


def _t(shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_error_and_size(fmt):
    t = _t((48, 3840))
    stored = quantize(t, fmt)
    back = dequantize(stored, fmt, tuple(t.shape))
    err = float((t - back).norm() / t.norm())
    limit = {"bf16": 5e-3, "fp8-row": 4e-2, "int8-block": 1e-2, "nf4-block": 0.12, "nf4-dq": 0.12}[
        fmt
    ]
    assert back.shape == t.shape and err < limit
    ratio = stored_bytes(stored) / (t.numel() * 2)
    assert (
        ratio
        <= {"bf16": 1.0, "fp8-row": 0.51, "int8-block": 0.51, "nf4-block": 0.29, "nf4-dq": 0.26}[
            fmt
        ]
    )


def test_odd_shapes_and_small_tensors():
    for shape in ((16, 4, 1, 37), (5,), (16, 32)):
        t = _t(shape, 1)
        for fmt in FORMATS:
            assert dequantize(quantize(t, fmt), fmt, shape).shape == shape


def test_file_is_deterministic_and_loads_bit_identically(tmp_path):
    tensors = {"q": _t((48, 640)), "layer/keys": _t((16, 32), 2), "layer/up": _t((64, 640), 3)}
    fmts = {"q": "nf4-dq", "layer/keys": "int8-block", "layer/up": "nf4-block"}
    save_quantized(tmp_path / "a.safetensors", tensors, fmts)
    save_quantized(tmp_path / "b.safetensors", tensors, fmts)
    assert (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()
    one = load_quantized(tmp_path / "a.safetensors")
    two = load_quantized(tmp_path / "a.safetensors")
    assert all(torch.equal(one[k], two[k]) for k in tensors)
    for k, fmt in fmts.items():
        want = dequantize(quantize(tensors[k], fmt), fmt, tuple(tensors[k].shape))
        assert torch.equal(one[k], want)
    low = load_quantized(tmp_path / "a.safetensors", dtype=torch.bfloat16)
    assert low["q"].dtype == torch.bfloat16


def test_unknown_format_and_file_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        quantize(_t((4, 4)), "nf2")
    from safetensors.torch import save_file

    save_file(
        {"x": torch.zeros(1)}, str(tmp_path / "x.safetensors"), metadata={"kilora.manifest": "{}"}
    )
    with pytest.raises(ValueError, match="not a quantized"):
        load_quantized(tmp_path / "x.safetensors")


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_writes_and_reads_a_file(fmt, tmp_path):
    tensors = {"q": _t((8, 512), 5), "small": _t((3, 7), 6)}
    save_quantized(tmp_path / "f.safetensors", tensors, fmt)
    got = load_quantized(tmp_path / "f.safetensors")
    for name, t in tensors.items():
        want = dequantize(quantize(t, fmt), fmt, tuple(t.shape))
        assert torch.equal(got[name], want)
    save_quantized(tmp_path / "g.safetensors", tensors, fmt)
    assert (tmp_path / "f.safetensors").read_bytes() == (tmp_path / "g.safetensors").read_bytes()


def test_float8_tensors_use_the_standard_safetensors_tag(tmp_path):
    import json
    import struct

    save_quantized(tmp_path / "f.safetensors", {"q": _t((4, 256))}, "fp8-row")
    raw = (tmp_path / "f.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    assert header["q/data"]["dtype"] == "F8_E4M3"
    from safetensors.torch import load_file

    assert load_file(str(tmp_path / "f.safetensors"))["q/data"].dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("fmt", ["int8-block", "nf4-block", "nf4-dq"])
def test_tiny_and_huge_blocks_keep_their_values(fmt):
    g = torch.Generator().manual_seed(4)
    for size in (3e-9, 1e-7, 4e7):
        t = torch.randn(4, 256, generator=g) * size
        back = dequantize(quantize(t, fmt), fmt, tuple(t.shape))
        err = float((t - back).norm() / t.norm())
        assert torch.isfinite(back).all() and err < (1e-2 if fmt == "int8-block" else 0.12), size


def test_files_with_float16_scales_still_read():
    t = _t((8, 512))
    stored = quantize(t, "int8-block")
    old = {"data": stored["data"], "scale": stored["scale"].to(torch.float16)}
    shape = tuple(t.shape)
    torch.testing.assert_close(
        dequantize(old, "int8-block", shape),
        dequantize(stored, "int8-block", shape),
        rtol=2e-3,
        atol=1e-6,
    )
