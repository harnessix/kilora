# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_comfyui import denoiser, folders
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora import route_freeze as R
from kilora.compute import dropout_key, run_layer, sampling_progress


def _y0(seed=1):
    return torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(seed))


def _run(sets, depth, y0=None, x=None, target="layer", progress=0.0, cache=None, **kw):
    y0 = _y0() if y0 is None else y0
    x = y0 if x is None else x
    freeze = R.freezing(cache) if cache is not None else None
    with torch.no_grad(), dropout_key(3, progress), sampling_progress(progress):
        if freeze is None:
            return run_layer(x.clone(), y0.clone(), sets, depth, target=target, **kw)
        with freeze:
            return run_layer(x.clone(), y0.clone(), sets, depth, target=target, **kw)


def _creative(seed=2, n=2):
    return [dataclasses.replace(s, gate_mode="adaptive") for s in _sets(n, 16, 4, 2, 8, seed=seed)]


def _functional(seed=5, form="input"):
    s = _sets(1, 16, 1, 3, 8, seed=seed)[0]
    return dataclasses.replace(s, form=form, axis="functional", gate_mode="none", fixed_depth=1)


@pytest.mark.parametrize("fast", [False, True])
def test_one_axis_is_bit_identical(fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    sets = _creative()
    same = [dataclasses.replace(s, axis="creative") for s in sets]
    for depth in (1, 3):
        assert torch.equal(_run(sets, depth), _run(same, depth))
    named = [dataclasses.replace(s, axis="style") for s in sets]
    for depth in (1, 3):
        a = _run(named, depth, target="layer@style")
        assert torch.equal(a, _run(sets, depth, target="layer@style"))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("form", ["input", "state"])
def test_two_axes_add_their_changes_alone(fast, form, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    creative, functional = _creative(), _functional(form=form)
    y0 = _y0(4)
    both = _run([*creative, functional], 3, y0=y0)
    alone_c = _run(creative, 3, y0=y0)
    alone_f = _run([functional], 1, y0=y0, target="layer@functional")
    torch.testing.assert_close(both, y0 + (alone_c - y0) + (alone_f - y0), rtol=0, atol=1e-6)
    order = _run([functional, *creative], 3, y0=y0)
    torch.testing.assert_close(order, both, rtol=0, atol=1e-6)


def test_functional_lora_alone_is_the_plain_lora():
    f = dataclasses.replace(_functional(form="input"), scale=0.7)
    y0 = _y0(6)
    got = _run([*_creative(), f], 3, y0=y0) - _run(_creative(), 3, y0=y0)
    plain = 0.7 * (y0 @ f.down.t()) @ f.up
    torch.testing.assert_close(got, plain, rtol=1e-5, atol=1e-5)


def test_no_crossing_between_axes():
    creative = _creative()
    y0 = _y0(7)

    def creative_routes(functional):
        with C.recording_routes() as routes:
            _run([*creative, functional], 3, y0=y0)
        return routes["layer"]

    a = creative_routes(_functional(seed=8))
    b = creative_routes(dataclasses.replace(_functional(seed=9), scale=-3.0))
    assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


def test_axes_freeze_and_carry_apart():
    creative = [dataclasses.replace(s, route_freeze=0.3) for s in _creative()]
    functional = dataclasses.replace(_functional(), route_freeze=0.3, gate_mode="adaptive")
    cache: dict = {}
    _run([*creative, functional], 3, progress=0.5, cache=cache)
    assert set(cache) == {"layer", "layer@functional"}


def _save(dirs, name, seed, scale=0.3):
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(seed)
    a, b = torch.randn(2, 16, generator=gen), torch.randn(16, 2, generator=gen) * scale
    save_file(
        {
            "lora_unet_layers_0_attention_to_q.lora_down.weight": a,
            "lora_unet_layers_0_attention_to_q.lora_up.weight": b,
        },
        str(dirs["loras"] / f"{name}.safetensors"),
    )


def test_comfyui_functional_lora_is_plain_beside_the_routed_ones(denoiser, folders):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui.patching import JOINT_SET

    for i, name in enumerate(("style", "ink", "helper")):
        _save(folders, name, 20 + i)
    (c,) = N.KiLoRAImportLoRA().add("style.safetensors", 1.0)
    (c,) = N.KiLoRAImportLoRA().add("ink.safetensors", 1.0, sets=c)
    (both,) = N.KiLoRAImportLoRA().add("helper.safetensors", 0.8, sets=c, axis="functional")
    (f_only,) = N.KiLoRAImportLoRA().add("helper.safetensors", 0.8, axis="functional")
    apply = N.KiLoRAApply().apply
    y_both, report, _ = apply(FakePatcher(FakeBaseModel(denoiser)), both, 1)
    y_c, _, _ = apply(FakePatcher(FakeBaseModel(denoiser)), c, 1)
    y_f, _, _ = apply(FakePatcher(FakeBaseModel(denoiser)), f_only, 1)
    assert JOINT_SET in report
    path = "diffusion_model.layers.0.attention.to_q"
    base = denoiser.get_submodule("layers.0.attention.to_q")
    h = torch.randn(1, 5, 16)
    with torch.no_grad():
        y0 = base(h)
        layer = {
            k: p.object_patches[path](h) - y0 for k, p in (("b", y_both), ("c", y_c), ("f", y_f))
        }
    torch.testing.assert_close(layer["b"], layer["c"] + layer["f"], rtol=1e-5, atol=1e-6)


def test_name_hint_suggests_the_functional_axis(denoiser, folders):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N

    _save(folders, "anima-extend-lightning", 30)
    (c,) = N.KiLoRAImportLoRA().add("anima-extend-lightning.safetensors", 1.0)
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), c, 1)
    assert "looks like a functional LoRA" in report
    (f,) = N.KiLoRAImportLoRA().add("anima-extend-lightning.safetensors", 1.0, axis="functional")
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), f, 1)
    assert "looks like a functional LoRA" not in report


def _expert_bytes(plan):
    total = 0
    for m in plan.mods:
        for s in m.sets:
            for ex in s.experts.values():
                for t in (ex.down, ex.up):
                    if t is not None:
                        total += t.numel() * t.element_size()
    return total


def test_two_axes_cost_at_most_the_sum_of_each_alone(denoiser, folders):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    for i, name in enumerate(("style", "ink", "helper")):
        _save(folders, name, 40 + i)
    (c,) = N.KiLoRAImportLoRA().add("style.safetensors", 1.0)
    (c,) = N.KiLoRAImportLoRA().add("ink.safetensors", 1.0, sets=c)
    (both,) = N.KiLoRAImportLoRA().add("helper.safetensors", 0.8, sets=c, axis="functional")
    (f,) = N.KiLoRAImportLoRA().add("helper.safetensors", 0.8, axis="functional")

    def cost(chain):
        plan, _ = patching.plan_chain(FakePatcher(FakeBaseModel(denoiser)), chain, joint=True)
        return _expert_bytes(plan)

    assert cost(both) <= cost(c) + cost(f)
    plan, _ = patching.plan_chain(FakePatcher(FakeBaseModel(denoiser)), f, joint=True)
    (only,) = [s for m in plan.mods for s in m.sets]
    assert all(ex.down.shape[0] == 1 for ex in only.experts.values())


@pytest.mark.parametrize("fast", [False, True])
def test_ip_plain_lora_and_axis_target(fast, monkeypatch):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    ip = dataclasses.replace(_functional(), axis="ip", scale=0.7, route_freeze=0.3)
    y0 = _y0(12)
    plain = 0.7 * (y0 @ ip.down.t()) @ ip.up
    torch.testing.assert_close(_run([ip], 3, y0=y0) - y0, plain, rtol=1e-5, atol=1e-5)
    cache = {}
    creative = [dataclasses.replace(s, route_freeze=0.3) for s in _creative()]
    _run([*creative, ip], 3, progress=0.5, cache=cache)
    assert set(cache) == {"layer", C.axis_target("layer", "ip")}


def test_ip_and_creative_joint_sets_are_separate(denoiser, folders, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    monkeypatch.delenv("KILORA_IP_DEPTH", raising=False)
    chain = None
    for i, (name, axis) in enumerate(
        (("style", "creative"), ("person", "ip"), ("ink", "creative"), ("prop", "ip"))
    ):
        _save(folders, name, 100 + i)
        (chain,) = N.KiLoRAImportLoRA().add(name + ".safetensors", 0.8, sets=chain, axis=axis)
    plan, _ = patching.plan_chain(FakePatcher(FakeBaseModel(denoiser)), chain, joint=True)
    scoped = patching._with_regions(plan, chain)
    by_name = {s.adapter_set.name: s for s in scoped}
    assert set(by_name) == {patching.JOINT_SET, patching.IP_JOINT_SET}
    assert by_name[patching.JOINT_SET].expert_names == ["style", "ink"]
    ip = by_name[patching.IP_JOINT_SET]
    assert ip.expert_names == ["person", "prop"]
    assert ip.axis == "ip" and ip.fixed_depth == 1
    assert ip.gate_mode == by_name[patching.JOINT_SET].gate_mode
    assert ip.route_freeze is None
    one, _ = patching.plan_chain(FakePatcher(FakeBaseModel(denoiser)), [chain[1]], joint=True)
    (single,) = patching._with_regions(one, [chain[1]])
    assert single.adapter_set.name != patching.IP_JOINT_SET
    assert single.axis == "ip" and single.fixed_depth == 1
