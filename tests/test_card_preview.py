# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from conftest import TARGET, randomize_up

from kilora import KiLoRAConfig, attach
from kilora.card import model_card
from kilora.cli import main as cli_main


def card_mod(model, tmp_path):
    a = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=3, router_dim=8))
    randomize_up(a)
    path = tmp_path / "ink.zimage.kmod.safetensors"
    a.save(
        path,
        id="ink",
        name="Ink Wash",
        license="CC-BY-4.0",
        mod_family="style",
        base_model={"family": "z-image"},
        trigger_words=["ink style"],
        example_prompts=["a boat, ink style"],
        description="Soft ink lines.",
        strength={"recommended": 0.8, "min": -1.0, "max": 1.5},
        trained_with={"tool": "musubi-tuner", "version": "0.3.6"},
    )
    a.remove()
    return path


def test_model_card(model, tmp_path):
    path = card_mod(model, tmp_path)
    card = model_card(path)
    head = card.split("---")[1]
    assert "library_name: kilora" in head
    assert 'base_model: "Tongyi-MAI/Z-Image"' in head
    assert 'license: "cc-by-4.0"' in head
    for tag in ("kilora", "lora", "z-image", "style"):
        assert f'  - "{tag}"' in head
    assert 'instance_prompt: "ink style"' in head
    assert "needs a KiLoRA node" in card and "Recommended strength | 0.8" in card
    assert "musubi-tuner 0.3.6" in card and "- a boat, ink style" in card
    assert card == model_card(path)
    out = tmp_path / "README.md"
    assert cli_main(["card", str(path), "--output", str(out)]) == 0
    assert out.read_text() == card


def test_unknown_license_and_live_card(model, tmp_path):
    from live_helpers import build_live_mod

    path, _, _ = build_live_mod(model, tmp_path / "live.kmod.safetensors")
    card = model_card(path)
    assert 'license: "other"' in card and 'license_name: "NOASSERTION"' in card
    assert "## Live mod" in card and '  - "live-mod"' in card


def test_preview_grid(tmp_path):
    pytest.importorskip("diffusers")
    from test_train import pipeline

    from kilora.cli import _preview
    from kilora.plugin_support import TrainerNetwork
    from kilora.preview import preview_grid

    pipe = pipeline()
    net = TrainerNetwork(
        family="z-image", rank=2, manifest={"id": "pv", "example_prompts": ["a boat"]}
    )
    net.attach(pipe.transformer)
    with torch.no_grad():
        for site in net.adapter.sites:
            site.tensors["s0_up"].normal_(0, 2.0)
    mod = net.save(tmp_path / "pv.safetensors")
    net.remove()
    grid = preview_grid(
        pipe, mod, "a boat", strengths=(-1, 0, 1), depths=(1, 2), size=32, steps=2, label=False
    )
    assert grid.size == (3 * 32, 2 * 32)
    cells = [
        [grid.crop((c * 32, r * 32, c * 32 + 32, r * 32 + 32)) for c in range(3)] for r in range(2)
    ]
    assert cells[0][1].tobytes() == cells[1][1].tobytes()
    assert cells[0][0].tobytes() != cells[0][2].tobytes()
    import argparse

    args = argparse.Namespace(
        mod=mod,
        model="tiny",
        prompt=None,
        output=str(tmp_path / "g.png"),
        strengths=[-1.0, 1.0],
        depths=[1],
        seed=1,
        steps=2,
        guidance=0.0,
        size=32,
        device="cpu",
    )
    assert _preview(args, pipe=pipe) == 0
    from PIL import Image

    with Image.open(tmp_path / "g.png") as img:
        assert img.format == "PNG" and not img.text


def test_preview_carries_the_state_across_sampler_steps(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    from test_train import pipeline

    from kilora import compute
    from kilora.plugin_support import TrainerNetwork
    from kilora.preview import preview_grid

    pipe = pipeline()
    net = TrainerNetwork(
        family="z-image", rank=2, manifest={"id": "pc"}, target_modules=["*.feed_forward.w2"]
    )
    net.attach(pipe.transformer)
    with torch.no_grad():
        for site in net.adapter.sites:
            site.tensors["s0_up"].normal_(0, 2.0)
    off = net.save(tmp_path / "off.safetensors")
    net.adapter.settings.carry_across_steps = True
    mod = net.save(tmp_path / "pc.safetensors")
    net.remove()
    starts = []
    real = compute._carry_start

    def spy(*args):
        out = real(*args)
        starts.append(out is not None)
        return out

    monkeypatch.setattr(compute, "_carry_start", spy)
    kw = dict(strengths=(1,), depths=(1,), size=32, steps=3, label=False)
    preview_grid(pipe, off, "a boat", **kw)
    assert starts and not any(starts)
    starts.clear()
    a = preview_grid(pipe, mod, "a boat", **kw)
    b = preview_grid(pipe, mod, "a boat", **kw)
    assert any(starts) and not all(starts)
    assert a.tobytes() == b.tobytes()
