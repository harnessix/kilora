# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_program_validate import lora

from kilora import Settings, load_mod, save_mod
from kilora.comfyui import nodes as N
from kilora.program import program_identity
from kilora.references import MissingReferenceError
from kilora.sections import Section


def make_sections(a, b, weights=None):
    program = lora()
    program["hooks"] = ["layers.0.feed_forward.w1"]
    sections = []
    data = {"A": a, "B": b}
    if weights is not None:
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        sections.append(
            Section(
                "weight-ref",
                "weights",
                {},
                {
                    "sha256": digest,
                    "filename": weights.name,
                    "size": weights.stat().st_size,
                    "form": "lora",
                    "transforms": {
                        "strength": 1.0,
                        "axis": "auto",
                        "roles": "auto",
                        "region": None,
                        "timeline": [],
                        "enabled": False,
                    },
                },
            )
        )
        program["bindings"]["B"].update(kind="ref", sha256=digest, tensor_key="B")
        del data["B"]
    program["identity"] = program_identity(program)
    sections.append(Section("program", "change", data, program))
    return sections


def apply(base, chain):
    return N.KiLoRAApply().apply(base, chain, -1)[0]


def test_program_files_apply_bit_exact(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    gen = torch.Generator().manual_seed(51)
    a, b = torch.randn(8, 16, generator=gen), torch.randn(32, 8, generator=gen)
    weights = folders["loras"] / "weights.safetensors"
    save_file({"B": b}, str(weights))
    base = FakePatcher(FakeBaseModel(denoiser))
    x = torch.randn(1, 5, 4, generator=gen)
    outputs = []
    for name, sections in (
        ("embedded", make_sections(a, b)),
        ("referenced", make_sections(a, b, weights)),
    ):
        path = folders["kilora"] / f"{name}.kmod"
        save_mod(path, [], Settings(), sections=sections, id=name)
        loaded = load_mod(path)
        assert loaded.sections[-1].meta == sections[-1].meta
        (chain,) = N.KiLoRASelect().select(path.name, 1.0, 1.0)
        outputs.append(apply(base, chain).run(x))
    from kilora.fixed_change import fixed_copy
    from kilora.sections import layer_changes

    path = "layers.0.feed_forward.w1"
    target = denoiser.layers[0].feed_forward.w1
    original = base.clone()
    original.add_object_patch(
        f"diffusion_model.{path}", fixed_copy(target, layer_changes(make_sections(a, b), 1)[path])
    )
    fixed = Section("fixed-lora", "fixed", {path: (a, b)}, {})
    native = base.clone()
    native.add_object_patch(
        f"diffusion_model.{path}", fixed_copy(target, layer_changes([fixed], 1)[path])
    )
    assert not torch.equal(outputs[0], base.run(x))
    assert all(torch.equal(out, original.run(x)) for out in outputs)
    assert torch.equal(outputs[0], native.run(x))
    weights.unlink()
    (chain,) = N.KiLoRASelect().select("referenced.kmod", 1.0, 1.0)
    with pytest.raises(MissingReferenceError):
        apply(base, chain)
