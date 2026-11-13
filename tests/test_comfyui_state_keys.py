# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora.comfyui import key_cache, patching
from kilora.comfyui import nodes as N
from kilora.comfyui.patching import JOINT_SET, applied
from kilora.comfyui.state_keys import M_IN

LAYERS = ("0_attention_to_q", "0_attention_to_v", "1_attention_to_q")


@pytest.fixture
def chain(folders, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "key-cache"))
    out = None
    for i in range(3):
        gen = torch.Generator().manual_seed(20 + i)
        sd = {}
        for layer in LAYERS:
            sd[f"lora_unet_layers_{layer}.lora_down.weight"] = torch.randn(2, 16, generator=gen)
            sd[f"lora_unet_layers_{layer}.lora_up.weight"] = torch.randn(16, 2, generator=gen)
        save_file(sd, str(folders["loras"] / f"s{i}.safetensors"))
        (out,) = N.KiLoRAImportLoRA().add(f"s{i}.safetensors", 1.0, sets=out)
    return out


def _call(patched, x, cond):
    fn = patched.model_options["model_function_wrapper"]
    args = {
        "input": x,
        "timestep": torch.tensor([0.9]),
        "c": {"c_crossattn": cond},
        "cond_or_uncond": [0],
    }
    return fn(lambda inp, t, **c: patched.run(inp), args)


def _sets(patched):
    out = []
    for module in patched.object_patches.values():
        plan = module.__dict__.get("_kilora_plan")
        for _, _, target in plan.parts if plan else []:
            out += [s for s in target.sets if s.key == JOINT_SET]
    return out


def test_add_lora_and_apply_route_three_loras_with_state_keys(denoiser, chain):
    patched, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, -1)
    assert JOINT_SET in report and "3 experts per target" in report
    sets = _sets(patched)
    before = [s.keys_in.clone() for s in sets]
    _call(patched, torch.randn(1, 5, 4), torch.ones(1, 2))
    assert applied(patched)["key_pass"].done
    for s, old in zip(sets, before, strict=True):
        assert s.keys_in.shape != old.shape or not torch.equal(s.keys_in, old)


def test_mix_stack_is_lora_loader_exactly(denoiser, chain):
    from kilora.comfyui.settings import KiLoRASimple

    (settings,) = KiLoRASimple().make("stack")
    base = FakePatcher(FakeBaseModel(denoiser))
    patched, report, _ = N.KiLoRAApply().apply(base, chain, -1, settings=settings)
    assert JOINT_SET not in report and "key_pass" not in applied(patched)
    x = torch.randn(1, 5, 4)
    hooks = []
    modules = dict(denoiser.named_modules())
    from safetensors.torch import load_file

    for entry in chain:
        sd = load_file(entry.path)
        for layer in LAYERS:
            a = sd[f"lora_unet_layers_{layer}.lora_down.weight"]
            b = sd[f"lora_unet_layers_{layer}.lora_up.weight"]
            name = "layers." + layer.replace("_attention_", ".attention.")

            def hook(m, i, o, a=a, b=b, s=entry.strength):
                return o + s * (i[0] @ a.t()) @ b.t()

            hooks.append(modules[name].register_forward_hook(hook))
    with torch.no_grad():
        want = denoiser(x)
    for h in hooks:
        h.remove()
    torch.testing.assert_close(patched.run(x), want, rtol=1e-4, atol=1e-5)


def test_joint_import_gets_state_keys_at_the_first_call_and_caches_them(denoiser, chain):
    base = FakePatcher(FakeBaseModel(denoiser))
    patched, report = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    assert JOINT_SET in report and "3 experts per target" in report
    sets = _sets(patched)
    assert sets and all(s.keys_in.shape[1] == 1 for s in sets)
    tau0 = [float(s.tau) for s in sets]
    x, cond = torch.randn(1, 5, 4), torch.ones(1, 2)
    first = _call(patched, x, cond)
    pass_ = applied(patched)["key_pass"]
    assert pass_.done and "key pass" in pass_.report
    for s in sets:
        assert s.keys_in.shape[1:] == (M_IN, s.keys_in.shape[2])
        gram = s.keys_in @ s.keys_in.transpose(-1, -2)
        torch.testing.assert_close(gram, torch.eye(M_IN).expand_as(gram), atol=1e-5, rtol=0)
    assert [float(s.tau) for s in sets] != tau0
    assert len(list(Path(key_cache.CACHE_DIR).glob("*.safetensors"))) == 1
    torch.testing.assert_close(_call(patched, x, cond), first)
    again, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    torch.testing.assert_close(_call(again, x, cond), first)
    assert "cache" in applied(again)["key_pass"].report
    for a, b in zip(_sets(again), sets, strict=True):
        torch.testing.assert_close(a.keys_in, b.keys_in)
    other, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    _call(other, x, -cond)
    assert "key pass" in applied(other)["key_pass"].report


def test_second_seed_reuses_keys_and_skips_warm_up(denoiser, chain, monkeypatch):
    from kilora import fused
    from kilora.comfyui.state_keys import KeyPass

    fused.wait_for_warm_up()
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "enabled", lambda: True)
    monkeypatch.setattr(fused, "_WARMED", set())
    monkeypatch.setattr(fused, "_WARM_PENDING", set())
    warms, passes = [], []

    def warm(device, templates, dtypes):
        warms.append(1)
        with fused._WARMED_LOCK:
            for s, depth in templates:
                for dtype in dtypes:
                    fused._WARMED.add((str(device), dtype, fused.shape_key(s, depth)))
        return True

    monkeypatch.setattr(fused, "warm_up", warm)
    schedule = fused.warm_up_sets
    monkeypatch.setattr(
        fused,
        "warm_up_sets",
        lambda pairs, dtypes=None: schedule(pairs, device="cpu", dtypes=dtypes),
    )
    original = KeyPass._compute

    def compute(self, *args):
        passes.append(1)
        return original(self, *args)

    monkeypatch.setattr(KeyPass, "_compute", compute)
    base = FakePatcher(FakeBaseModel(denoiser))
    first, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    fused.wait_for_warm_up()
    cond = torch.ones(1, 2)
    _call(first, torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(1)), cond)
    x2 = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(2))
    expected = _call(first, x2, cond)
    monkeypatch.setattr(key_cache, "load", lambda key: pytest.fail("disk cache was read"))
    second, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    fused.wait_for_warm_up()
    actual = _call(second, x2, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert passes == [1]
    assert warms == [1]


def test_plain_import_cache_tracks_file_content_and_config(denoiser, chain, tmp_path, monkeypatch):
    import shutil

    base = FakePatcher(FakeBaseModel(denoiser))
    binding = patching.detect_family(base)
    shapes = patching.shape_model(patching.diffusers_shapes(base, binding))
    source = Path(chain[0].path)
    other = tmp_path / "copy"
    other.mkdir()
    copy = other / source.name
    shutil.copyfile(source, copy)
    original = patching.import_lora
    calls = []

    def load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(patching, "import_lora", load)
    patching._LORA_CACHE.clear()
    first = patching._import_result(str(source), binding, shapes, denoiser)
    assert patching._import_result(str(copy), binding, shapes, denoiser) is first
    assert len(calls) == 1
    (other / "adapter_config.json").write_text("{}")
    patching._import_result(str(copy), binding, shapes, denoiser)
    assert len(calls) == 2


def test_a_region_on_one_lora_is_its_experts_mask_in_the_joint_set(denoiser, chain):
    import dataclasses

    mask = torch.zeros(1, 4, 4)
    mask[:, :, :2] = 1.0
    entries = list(chain)
    entries[1] = dataclasses.replace(entries[1], region=mask)
    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), type(chain)(entries), -1
    )
    plans = [
        m.__dict__["_kilora_plan"]
        for m in patched.object_patches.values()
        if "_kilora_plan" in m.__dict__
    ]
    found = [t.expert_regions for p in plans for _, _, t in p.parts]
    assert found and all(r == [[None, mask, None]] for r in found)


def test_key_pass_probes_unfrozen_and_restores_route_freezing(denoiser, chain, monkeypatch):
    from kilora import compute

    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, -1, route_freeze=0.4
    )
    sets = _sets(patched)
    assert sets and all(s.route_freeze == 0.4 for s in sets)
    seen = []
    real = compute.run_layer

    def spy(x, y0, layer_sets, *a, **kw):
        if compute._PROBES:
            seen.extend(s.route_freeze for s in layer_sets)
        return real(x, y0, layer_sets, *a, **kw)

    monkeypatch.setattr(compute, "run_layer", spy)
    import kilora.bake as bake

    monkeypatch.setattr(bake, "run_layer", spy)
    _call(patched, torch.randn(1, 5, 4), torch.ones(1, 2))
    assert seen and all(f is None for f in seen)
    assert all(s.route_freeze == 0.4 for s in _sets(patched))


def test_weighted_transition_survives_key_cache(denoiser, chain, monkeypatch):
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.5")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "weighted")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    base = FakePatcher(FakeBaseModel(denoiser))
    x, cond = torch.randn(1, 5, 4), torch.ones(1, 2)
    first, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    _call(first, x, cond)
    transitions = [s.share_transition_weighted for s in _sets(first)]
    assert transitions and all(t is not None for t in transitions)
    again, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    _call(again, x, cond)
    assert "cache" in applied(again)["key_pass"].report
    for expected, s in zip(transitions, _sets(again), strict=True):
        torch.testing.assert_close(s.share_transition_weighted, expected)
