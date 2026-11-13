# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_modlist import write_mod
from test_path_write import WrapPatcher, _setup, fresh_sessions

from kilora import KiLoRAError
from kilora.comfyui import nodes as N
from kilora.modlist import LoadOrderError


def _patcher(denoiser):
    return FakePatcher(FakeBaseModel(denoiser))


def test_no_mod_selected(denoiser, folders):
    with pytest.raises(KiLoRAError, match="No mod selected"):
        N.KiLoRALoadMod().load(_patcher(denoiser), "", 1.0)
    with pytest.raises(KiLoRAError, match="No mod selected"):
        N.KiLoRASelect().select("", 1.0, 1.0)
    with pytest.raises(KiLoRAError, match="No mod selected"):
        N.KiLoRAModInfo().info("")


def test_missing_files(denoiser, folders):
    with pytest.raises(KiLoRAError, match="was not found in ComfyUI/models/kilora"):
        N.KiLoRALoadMod().load(_patcher(denoiser), "gone.kmod", 1.0)
    with pytest.raises(KiLoRAError, match="LoRA gone.safetensors was not found"):
        N.KiLoRAImportLoRA().add("gone.safetensors", 1.0)
    with pytest.raises(KiLoRAError, match="mod pack gone.kpack.json was not found"):
        N.KiLoRAPalette().load("gone.kpack.json")


def test_strength_out_of_range(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    for bad in (float("nan"), 3.0, -2.5):
        with pytest.raises(KiLoRAError, match="Strength must be a number from -2 to 2"):
            N.KiLoRALoadMod().load(_patcher(denoiser), "ink-1.0.0.kmod", bad)


def test_wrong_base_model(denoiser, folders):
    write_mod(denoiser, folders["kilora"], "other", seed=1, base_model={"family": "sdxl"})
    with pytest.raises(LoadOrderError, match="made for sdxl"):
        N.KiLoRALoadMod().load(_patcher(denoiser), "other-1.0.0.kmod", 1.0)


def test_missing_editor(denoiser, folders):
    with pytest.raises(KiLoRAError, match="No live-mode editor found"):
        N.KiLoRALiveRecord().record(_patcher(denoiser), "")
    with pytest.raises(KiLoRAError, match="No live-mode editor found"):
        N.KiLoRALiveRecord().record(_patcher(denoiser), "gone.kedit.safetensors")


def test_editor_of_another_kind(denoiser, folders, tmp_path):
    import json

    from safetensors.torch import save_file

    _, _, patched, _, _, _ = _setup(denoiser, folders, tmp_path)
    save_file(
        {"x": torch.zeros(1)},
        str(folders["kilora"] / "odd.kedit.safetensors"),
        metadata={"kilora.manifest": json.dumps({"kind": "something-else"})},
    )
    with pytest.raises(KiLoRAError, match="is not a live-mode editor"):
        N.KiLoRALiveRecord().record(patched, "odd.kedit.safetensors")


def test_hook_mismatch_through_the_node(denoiser, folders, tmp_path):
    from kilora_editor_delta import load_path_editor, save_path_editor

    _, _, patched, session, _, _ = _setup(denoiser, folders, tmp_path)
    editor = load_path_editor(tmp_path / "path.kedit.safetensors")
    first = next(iter(editor.v0))
    save_path_editor(
        folders["kilora"] / "wrong.kedit.safetensors",
        id="wrong",
        version="1.0.0",
        base_family=session.mod.manifest["base_model"]["family"],
        v0={"layers.9.attention.to_out.0": editor.v0[first]},
        beta={"layers.9.attention.to_out.0": 0.5},
        lam={"layers.9.attention.to_out.0": 0.1},
        h_bar={"layers.9.attention.to_out.0": editor.h_bar[first]},
    )
    with pytest.raises(KiLoRAError, match="does not fit this mod"):
        N.KiLoRALiveRecord().record(patched, "wrong.kedit.safetensors")


def test_no_live_mod_loaded(denoiser, folders, tmp_path):
    import shutil

    _, _, _, _, _, _ = _setup(denoiser, folders, tmp_path)
    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "p.kedit.safetensors")
    write_mod(denoiser, folders["kilora"], "ink", seed=1)
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    static, _, _ = N.KiLoRAApply().apply(_patcher(denoiser), chain, 1)
    with pytest.raises(KiLoRAError, match="No live mod is loaded"):
        N.KiLoRALiveRecord().record(static, "p.kedit.safetensors")


def test_out_of_gpu_memory_during_sampling(denoiser, folders, tmp_path):
    import shutil

    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "p.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, "p.kedit.safetensors")
    fn = model.model_options["model_function_wrapper"]

    def apply_model(inp, t, **c):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        fn(apply_model, {"input": xs[0], "timestep": torch.tensor([0]), "c": {}})
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "Not enough GPU memory" in report and session.commits == []


def test_learning_off_message(denoiser, folders, tmp_path):
    import shutil

    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "p.kedit.safetensors")
    session.learning = False
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, "p.kedit.safetensors")
    model.sample(xs[0], steps=2)
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "Learning is off for this mod" in report and "KiLoRA Live: Settings" in report
