# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from dataclasses import dataclass

import torch

STORAGE_FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    name: str
    arch: str
    memory_gib: float
    fp8: str


def backend() -> str:
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _arch(index: int = 0) -> str:
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", "") or ""
    if arch:
        return arch.split(":")[0]
    return f"sm_{props.major}{props.minor}"


def uses_fnuz(arch: str) -> bool:
    return arch.startswith("gfx94")


def compute_fp8(arch: str | None = None) -> torch.dtype:
    if arch is None:
        arch = _arch() if torch.cuda.is_available() else ""
    return torch.float8_e4m3fnuz if uses_fnuz(arch) else torch.float8_e4m3fn


def device_info(index: int = 0) -> DeviceInfo:
    kind = backend()
    if kind not in ("cuda", "rocm"):
        return DeviceInfo(kind, kind, "", 0.0, "none")
    props = torch.cuda.get_device_properties(index)
    arch = _arch(index)
    fp8 = "e4m3fnuz" if uses_fnuz(arch) else "e4m3fn"
    return DeviceInfo(kind, props.name, arch, round(props.total_memory / 2**30, 1), fp8)


OFFLOAD_BELOW_GIB = 40.0


def needs_offload(index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(index).total_memory / 2**30 < OFFLOAD_BELOW_GIB


def smi_command() -> list[str] | None:
    names = ("amd-smi", "rocm-smi") if backend() == "rocm" else ("nvidia-smi",)
    for name in names:
        if shutil.which(name):
            return [name]
    return None


def to_compute_fp8(x: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:

    top = torch.finfo(dtype).max
    amax = x.detach().abs().amax().float().clamp(min=1e-12)
    scale = amax / top
    q = (x.float() / scale).clamp(-top, top).to(dtype)
    return q, scale
