# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from kilora import host
from kilora.comfyui import nodes, startup_prepare
from kilora.prepare import PLAN_DTYPES, padding_dtypes

FLUX = ("double_blocks.0.img_attn.qkv", "single_blocks.0.linear1", "txt_in")
WAN = ("blocks.0.self_attn.q", "blocks.0.cross_attn.o", "blocks.0.ffn.0")
ANIMA = ("blocks.0.self_attn.q_proj", "blocks.0.cross_attn.output_proj", "llm_adapter.out_proj")


def _model_file(path, modules, dtype, prefix=""):
    save_file({f"{prefix}{m}.weight": torch.zeros(4, 4, dtype=dtype) for m in modules}, str(path))
    return str(path)


def _lora_file(path, modules):
    tensors = {}
    for m in modules:
        tensors[f"diffusion_model.{m}.lora_down.weight"] = torch.zeros(2, 4)
        tensors[f"diffusion_model.{m}.lora_up.weight"] = torch.zeros(4, 2)
    save_file(tensors, str(path))
    return str(path)


@pytest.fixture
def comfy(monkeypatch, tmp_path):
    state = {
        "files": {"diffusion_models": {}, "checkpoints": {}},
        "supported": [torch.bfloat16, torch.float32],
        "fail": False,
        "calls": [],
    }

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = str(tmp_path / "models")
    folder_paths.get_filename_list = lambda folder: sorted(state["files"].get(folder, {}))
    folder_paths.get_full_path = lambda folder, name: state["files"][folder][name]
    folder_paths.get_folder_paths = lambda folder: [str(tmp_path / folder)]

    def unet_prefix_from_state_dict(sd):
        return "model.diffusion_model." if any(k.startswith("model.") for k in sd) else ""

    def model_config_from_unet(sd, prefix, metadata=None):
        assert all(t.device.type == "meta" for t in sd.values())
        if state["fail"]:
            return None
        if f"{prefix}double_blocks.0.img_attn.qkv.weight" in sd:
            return types.SimpleNamespace(supported_inference_dtypes=list(state["supported"]))
        if f"{prefix}blocks.0.self_attn.q.weight" in sd:
            return types.SimpleNamespace(supported_inference_dtypes=[torch.float16, torch.float32])
        if f"{prefix}blocks.0.cross_attn.output_proj.weight" in sd:
            return types.SimpleNamespace(
                supported_inference_dtypes=[torch.bfloat16, torch.float16, torch.float32]
            )
        return None

    detection = types.ModuleType("comfy.model_detection")
    detection.unet_prefix_from_state_dict = unet_prefix_from_state_dict
    detection.model_config_from_unet = model_config_from_unet

    def weight_dtype(sd, prefix=""):
        counts = {}
        for key, tensor in sd.items():
            if key.startswith(prefix):
                counts[tensor.dtype] = counts.get(tensor.dtype, 0) + tensor.numel()
        return max(counts, key=counts.get) if counts else None

    utils = types.ModuleType("comfy.utils")
    utils.weight_dtype = weight_dtype
    utils.calculate_parameters = lambda sd, prefix="": sum(
        t.numel() for k, t in sd.items() if k.startswith(prefix)
    )

    def unet_dtype(device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
        state["calls"].append(("unet_dtype", model_params, weight_dtype))
        supported = supported_dtypes or [torch.float32]
        return weight_dtype if weight_dtype in supported else supported[0]

    def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=None):
        state["calls"].append(("unet_manual_cast", weight_dtype))
        return None

    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []
    mm.unet_dtype = unet_dtype
    mm.unet_manual_cast = unet_manual_cast
    mm.get_torch_device = lambda: torch.device("cpu")

    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_detection = detection
    package.model_management = mm
    package.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_detection", detection)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(host, "SERVICES", host.HostServices())
    monkeypatch.setattr(nodes, "FOLDERS", nodes.Folders())
    monkeypatch.delenv("KILORA_PLAN_DTYPE", raising=False)
    return state


def test_header_prediction_follows_the_comfyui_rule(comfy, tmp_path, caplog):
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
        "flux-fp16.safetensors": _model_file(
            tmp_path / "b.safetensors", FLUX, torch.float16, prefix="model.diffusion_model."
        ),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", FLUX)]
    with caplog.at_level(logging.INFO, logger="kilora.prepare"):
        _refuse_reads()
        assert startup_prepare._plan_dtype("cpu", library) == torch.bfloat16
    weights = [c[2] for c in comfy["calls"] if c[0] == "unet_dtype"]
    assert sorted(map(str, weights)) == ["torch.bfloat16", "torch.float16"]
    assert all(c[1] == 48 for c in comfy["calls"] if c[0] == "unet_dtype")
    lines = [r.getMessage() for r in caplog.records if "Plan dtype" in r.getMessage()]
    assert len(lines) == 1
    assert "header prediction" in lines[0]
    assert "flux-bf16.safetensors" in lines[0] and "flux-fp16.safetensors" in lines[0]


def _refuse_reads():
    import safetensors
    import safetensors.torch

    def refuse(*args, **kwargs):
        raise AssertionError("the prediction read tensor data")

    patch = pytest.MonkeyPatch()
    patch.setattr(safetensors, "safe_open", refuse)
    patch.setattr(safetensors.torch, "load_file", refuse)
    patch.setattr(safetensors.torch, "load", refuse)
    patch.setattr(torch, "load", refuse)
    _PATCHES.append(patch)


_PATCHES: list[pytest.MonkeyPatch] = []


@pytest.fixture(autouse=True)
def _undo_patches():
    yield
    while _PATCHES:
        _PATCHES.pop().undo()


def test_files_that_disagree_give_their_dtypes(comfy, tmp_path, caplog):
    comfy["supported"] = [torch.bfloat16, torch.float16, torch.float32]
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
        "flux-fp16.safetensors": _model_file(tmp_path / "b.safetensors", FLUX, torch.float16),
        "wan-fp32.safetensors": _model_file(tmp_path / "c.safetensors", WAN, torch.float32),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", FLUX)]
    with caplog.at_level(logging.INFO, logger="kilora.prepare"):
        _refuse_reads()
        dtype = startup_prepare._plan_dtype("cpu", library)
    assert dtype == (torch.float16, torch.bfloat16)
    assert padding_dtypes(dtype) == (torch.float16, torch.bfloat16)
    line = next(r.getMessage() for r in caplog.records if "Plan dtype" in r.getMessage())
    assert "wan-fp32.safetensors" not in line


def test_only_the_library_family_counts(comfy, tmp_path):
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
        "wan-fp32.safetensors": _model_file(tmp_path / "c.safetensors", WAN, torch.float32),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", WAN)]
    assert startup_prepare._plan_dtype("cpu", library) == torch.float32


def test_a_failed_detection_keeps_the_three_dtypes(comfy, tmp_path, caplog):
    comfy["fail"] = True
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", FLUX)]
    with caplog.at_level(logging.INFO, logger="kilora.prepare"):
        dtype = startup_prepare._plan_dtype("cpu", library)
    assert dtype is None
    assert padding_dtypes(dtype) == PLAN_DTYPES
    lines = [r.getMessage() for r in caplog.records if "No plan dtype prediction" in r.getMessage()]
    assert len(lines) == 1
    assert "the detection failed" in lines[0] and "flux-bf16.safetensors" in lines[0]


def test_a_broken_file_never_stops_the_worker(comfy, tmp_path):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"\xff" * 16)
    comfy["files"]["diffusion_models"] = {
        "broken.safetensors": str(broken),
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", FLUX)]
    assert startup_prepare._plan_dtype("cpu", library) == torch.bfloat16


def test_the_environment_still_wins(comfy, tmp_path, monkeypatch):
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
    }
    library = [_lora_file(tmp_path / "lora.safetensors", FLUX)]
    monkeypatch.setenv("KILORA_PLAN_DTYPE", "float32")
    assert startup_prepare._plan_dtype("cpu", library) == torch.float32
    assert comfy["calls"] == []


def test_padding_dtypes_of_several():
    assert padding_dtypes({torch.bfloat16, torch.float32}) == (torch.float32, torch.bfloat16)
    assert padding_dtypes(()) == PLAN_DTYPES


ANIMA_KOHYA = (
    "lora_unet_blocks_0_self_attn_q_proj",
    "lora_unet_blocks_0_cross_attn_output_proj",
    "lora_unet_blocks_0_mlp_layer1",
)
ANIMA_DIFFUSERS = (
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn2.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
)


def _named_lora_file(path, stems, prefix=""):
    tensors = {}
    for stem in stems:
        tensors[f"{prefix}{stem}.lora_down.weight"] = torch.randn(2, 16)
        tensors[f"{prefix}{stem}.lora_up.weight"] = torch.randn(16, 2)
        tensors[f"{prefix}{stem}.alpha"] = torch.tensor(2.0)
    save_file(tensors, str(path))
    return str(path)


def _prepared_families(monkeypatch, path):
    from kilora import prepare

    found = []
    monkeypatch.setattr(prepare, "prepare_lokr", lambda *a, **k: (0, 0))

    def mix(sources, model, *, family=None, **kwargs):
        found.append(family)
        return types.SimpleNamespace(adapter_set=None)

    monkeypatch.setattr(prepare, "prepare_mix", mix)
    prepare.prepare_file(path, device="cpu")
    return tuple(sorted(set(found)))


@pytest.mark.parametrize(
    ("stems", "prefix"),
    [
        (ANIMA_KOHYA, ""),
        (ANIMA_DIFFUSERS, ""),
        (ANIMA_DIFFUSERS, "diffusion_model."),
    ],
    ids=["kohya", "diffusers", "diffusers-comfyui-prefix"],
)
def test_library_family_is_the_family_of_prepare_file(comfy, tmp_path, monkeypatch, stems, prefix):
    lora = _named_lora_file(tmp_path / "anima-lora.safetensors", stems, prefix)
    prepared = _prepared_families(monkeypatch, lora)
    assert "anima" in prepared
    assert tuple(sorted(startup_prepare._lora_families(lora))) == prepared

    comfy["files"]["diffusion_models"] = {
        "anima-bf16.safetensors": _model_file(
            tmp_path / "anima.safetensors", ANIMA, torch.bfloat16
        ),
        "flux-fp16.safetensors": _model_file(tmp_path / "flux.safetensors", FLUX, torch.float16),
    }
    _refuse_reads()
    dtypes, files, reason = startup_prepare._predict("cpu", [lora])
    assert reason is None
    assert dtypes == {torch.bfloat16}
    assert files == ["anima-bf16.safetensors"]
    assert startup_prepare._plan_dtype("cpu", [lora]) == torch.bfloat16


def test_library_family_reads_the_header_only(tmp_path):
    lora = _named_lora_file(tmp_path / "anima-lora.safetensors", ANIMA_KOHYA)
    _refuse_reads()
    assert "anima" in startup_prepare._lora_families(lora)


def test_an_unknown_library_family_keeps_the_three_dtypes(comfy, tmp_path, caplog):
    comfy["files"]["diffusion_models"] = {
        "flux-bf16.safetensors": _model_file(tmp_path / "a.safetensors", FLUX, torch.bfloat16),
    }
    lora = _named_lora_file(tmp_path / "other.safetensors", ("lora_unet_unknown_layer",))
    assert startup_prepare._lora_families(lora) == ()
    with caplog.at_level(logging.INFO, logger="kilora.prepare"):
        assert startup_prepare._plan_dtype("cpu", [lora]) is None
    lines = [r.getMessage() for r in caplog.records if "No plan dtype prediction" in r.getMessage()]
    assert len(lines) == 1 and "family of the library files is unknown" in lines[0]
