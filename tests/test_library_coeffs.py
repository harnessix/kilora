# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from safetensors.torch import save_file

from kilora.library import basis as builder
from kilora.library import coeffs
from kilora.library.index import LibraryIndex

STEM = "layers.0.attention.to_q"


def write_lora(path, up, down):
    save_file({STEM + ".lora_up.weight": up, STEM + ".lora_down.weight": down}, str(path))


def basis_for(path, *, rank=2, flag="deriv", store=False):
    index = LibraryIndex(path.parent / "index")
    index.paths[str(path)] = "digest"
    index.records["digest"] = {"license_flag": flag}
    return builder.build_basis(index, rank=rank, top=rank, store=store)


def test_projection_and_energy_match_dense(tmp_path):
    torch.manual_seed(17)
    source = tmp_path / "source.safetensors"
    target = tmp_path / "target.safetensors"
    write_lora(source, torch.randn(17, 2), torch.randn(2, 19))
    up, down = torch.randn(17, 4), torch.randn(4, 19)
    write_lora(target, up, down)
    basis = basis_for(source)
    mount = basis.mounts[STEM]
    actual = coeffs.lora_coefficients(target, basis)[STEM]
    dense = up @ down
    core = mount.u.T @ dense @ mount.v
    assert torch.allclose(actual["core"], core, atol=2e-5)
    assert torch.allclose(actual["output_energy"], (mount.u.T @ dense).square().sum(1), atol=2e-4)
    assert torch.allclose(actual["input_energy"], (dense @ mount.v).square().sum(0), atol=2e-4)
    assert torch.allclose(actual["energy"], dense.square().sum(), atol=2e-4)
    assert torch.allclose(actual["captured_share"], core.square().sum() / dense.square().sum())
    assert actual["captured_share"] < 0.1
    assert torch.allclose(
        coeffs.lora_coefficients(source, basis)[STEM]["captured_share"],
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_rank_truncation(tmp_path):
    source = tmp_path / "source.safetensors"
    target = tmp_path / "target.safetensors"
    write_lora(source, torch.eye(3), torch.eye(3))
    up = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    write_lora(target, up, torch.eye(3))
    basis = basis_for(source, rank=3)
    got = coeffs.lora_coefficients(target, basis, rank=1)[STEM]["core"]
    mount = basis.mounts[STEM]
    expected = mount.u.T @ torch.diag(torch.tensor([4.0, 0.0, 0.0])) @ mount.v
    assert torch.allclose(got, expected, atol=1e-6)
