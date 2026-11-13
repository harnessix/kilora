# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from collections.abc import Mapping

import torch
from safetensors import safe_open

FORMAT = "kilora-quantized"
FORMATS = ("bf16", "fp8-row", "int8-block", "nf4-block", "nf4-dq")
INT8_BLOCK = 256
NF4_BLOCK = 64
DQ_GROUP = 256
FP8_MAX = 448.0

NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1, t.shape[-1]) if t.dim() > 1 else t.float()[None]


def _blocks(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    n = x.numel()
    pad = (-n) % block
    return torch.nn.functional.pad(x.reshape(-1), (0, pad)).reshape(-1, block), n


def quantize(t: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    x = _rows(t)
    if fmt == "bf16":
        return {"data": t.detach().to(torch.bfloat16).contiguous()}
    if fmt == "fp8-row":
        scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / FP8_MAX
        return {"data": (x / scale).to(torch.float8_e4m3fn), "scale": scale}
    if fmt == "int8-block":
        b, _ = _blocks(x, INT8_BLOCK)
        scale = b.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 127.0
        data = torch.round(b / scale).clamp(-127, 127).to(torch.int8)
        return {"data": data, "scale": scale}
    if fmt in ("nf4-block", "nf4-dq"):
        b, _ = _blocks(x, NF4_BLOCK)
        absmax = b.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
        codes = (b / absmax)[..., None].sub(NF4_LEVELS).abs().argmin(-1).to(torch.uint8)
        packed = (codes[:, 0::2] << 4 | codes[:, 1::2]).contiguous()
        if fmt == "nf4-block":
            return {"data": packed, "scale": absmax.float()}
        mean = absmax.mean().reshape(1)
        g, _ = _blocks(absmax - mean, DQ_GROUP)
        gscale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 127.0
        q = torch.round(g / gscale).clamp(-127, 127).to(torch.int8)
        return {"data": packed, "scale_q": q, "scale_scale": gscale, "scale_mean": mean}
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


def dequantize(
    stored: Mapping[str, torch.Tensor], fmt: str, shape: tuple[int, ...], dtype=torch.float32
) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    if fmt == "bf16":
        return stored["data"].to(dtype).reshape(shape)
    if fmt == "fp8-row":
        return (stored["data"].float() * stored["scale"]).reshape(shape).to(dtype)
    if fmt == "int8-block":
        x = stored["data"].float() * stored["scale"].float()
        return x.reshape(-1)[:n].reshape(shape).to(dtype)
    if fmt in ("nf4-block", "nf4-dq"):
        packed = stored["data"]
        codes = torch.stack([packed >> 4, packed & 15], -1).reshape(packed.shape[0], -1)
        if fmt == "nf4-block":
            absmax = stored["scale"].float()
        else:
            g = stored["scale_q"].float() * stored["scale_scale"]
            blocks = codes.shape[0]
            absmax = (g.reshape(-1)[:blocks] + stored["scale_mean"]).reshape(-1, 1)
        x = NF4_LEVELS[codes.long()] * absmax
        return x.reshape(-1)[:n].reshape(shape).to(dtype)
    raise ValueError(f"unknown format {fmt!r}")


def stored_bytes(stored: Mapping[str, torch.Tensor]) -> int:
    return sum(v.numel() * v.element_size() for v in stored.values())


def save_quantized(
    path: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
    fmt: str | Mapping[str, str],
) -> dict:
    from .modfile import MANIFEST_KEY, write_safetensors

    out: dict[str, torch.Tensor] = {}
    entries = {}
    for name in sorted(tensors):
        t = tensors[name]
        f = fmt if isinstance(fmt, str) else fmt[name]
        for part, value in quantize(t, f).items():
            out[f"{name}/{part}"] = value.contiguous()
        entries[name] = {"format": f, "shape": list(t.shape)}
    manifest = {"format": FORMAT, "format_version": "1.0", "tensors": entries}
    write_safetensors(path, out, {MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
    return manifest


def load_quantized(
    path: str | os.PathLike[str], dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    from .modfile import MANIFEST_KEY

    with safe_open(os.fspath(path), framework="pt") as f:
        manifest = json.loads((f.metadata() or {})[MANIFEST_KEY])
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{os.fspath(path)} is not a quantized KiLoRA file")
        stored: dict[str, dict[str, torch.Tensor]] = {}
        for key in f.keys():
            name, _, part = key.rpartition("/")
            stored.setdefault(name, {})[part] = f.get_tensor(key)
    return {
        name: dequantize(stored[name], e["format"], tuple(e["shape"]), dtype)
        for name, e in manifest["tensors"].items()
    }
