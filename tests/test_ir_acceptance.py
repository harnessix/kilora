# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_modfile import trained

from kilora import compute, fused, ir, load_mod
from kilora.comfyui import nodes as N
from kilora.comfyui.settings import resolve_settings
from kilora.ir import driver
from kilora.ir.dispatch import active_overrides
from kilora.ir.driver import _run


def _saved(tiny, path):
    adapter = trained(tiny)
    adapter.settings.gate_mode = "none"
    adapter.save(path, id="acceptance")
    adapter.remove()


def test_example_load_node_and_simplex(denoiser, folders):
    source = folders["kilora"] / "source.kmod"
    destination = folders["kilora"] / "softmax.kmod"
    _saved(denoiser, source)
    path = Path(__file__).parents[1] / "examples/softmax-select/make_mod.py"
    spec = importlib.util.spec_from_file_location("softmax_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    generated = folders["kilora"] / "generated.kmod"
    example.write_example(source, generated, 2.0)
    shipped = path.parent / "tiny-softmax.kmod"
    shutil.copyfile(shipped, destination)
    assert load_mod(generated).sections[-1].meta == load_mod(destination).sections[-1].meta
    mod = load_mod(destination)
    assert mod.sections[-1].data["temperature"].item() == 2.0
    logits = torch.tensor([[[3.0, 1.0, -1.0, 0.0]]])
    with active_overrides({"select": mod.sets[0].ir_parts["select"]}):
        weights = _run(
            "select", "token_sparsemax", {"logits": logits, "noise": torch.zeros_like(logits)}
        )["weights"]
    assert torch.equal(weights, (logits / 2).softmax(-1))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))
    assert not torch.equal(weights, compute.sparsemax(logits))
    base = FakePatcher(FakeBaseModel(denoiser))
    patched, report = N.KiLoRALoadMod().load(base, destination.name, 1.0, depth=3)
    native, _ = N.KiLoRALoadMod().load(base, source.name, 1.0, depth=3)
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(17))
    with compute.eager(), compute.dropout_key(17, 4):
        assert not torch.equal(patched.run(x), native.run(x)), (report, patched.object_patches)


@pytest.mark.parametrize("derived", ["full", "keys"])
@pytest.mark.parametrize(
    "iteration,fold", [("accumulate", "0"), ("path", "0"), ("accumulate", "1")]
)
def test_apply_interpreter_parity(
    denoiser,
    folders,
    iteration,
    fold,
    derived,
    monkeypatch,
):

    monkeypatch.setenv("KILORA_MASKS", "sample")
    monkeypatch.setenv("KILORA_DERIVED_MODE", derived)
    if fold == "1":
        monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_IMPORT_FOLD", fold)
    path = folders["kilora"] / "routed.kmod"
    _saved(denoiser, path)
    stem = "lora_unet_layers_0_feed_forward_w1"
    generator = torch.Generator().manual_seed(24)
    lora = folders["loras"] / "imported.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=generator),
            f"{stem}.lora_up.weight": torch.randn(32, 2, generator=generator),
        },
        str(lora),
    )
    (chain,) = N.KiLoRASelect().select(path.name, 1.0, 1.0)
    (chain,) = N.KiLoRAImportLoRA().add(lora.name, 0.5, sets=chain)
    base = FakePatcher(FakeBaseModel(denoiser))
    calls = []
    original = driver._run

    def record(part, *args, **kwargs):
        calls.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    original_layer = driver.run_layer_ir
    details = []
    interpreted = []

    def trace_layer(*args, **kwargs):
        try:
            result = original_layer(*args, **kwargs)
            interpreted.append((kwargs.get("target"), args[2]))
            return result
        except NotImplementedError:
            details.append(
                {
                    "sets": [
                        (
                            s.form,
                            s.axis,
                            s.axis_joint,
                            s.import_score,
                            s.aggregation,
                            s.gate_mode,
                            s.iteration,
                            s.direction_core is not None,
                            s.expert_mask is not None,
                            s.expert_offsets is not None,
                            s.expert_inputs is not None,
                            s.candidates,
                            s.keys_in.shape[0],
                            s.route_freeze,
                            s.cross_weights,
                            s.carry,
                            s.routing_scope,
                            s.down is None,
                            s.up is None,
                            s.gate_head_limit,
                        )
                        for s in args[2]
                    ],
                    "expert_regions": repr(kwargs.get("expert_regions")),
                    "expert_scales": repr(kwargs.get("expert_scales")),
                    "target": kwargs.get("target"),
                    "depth": args[3],
                }
            )
            raise

    monkeypatch.setattr(driver, "run_layer_ir", trace_layer)
    notes = []
    monkeypatch.setattr(fused, "_note", lambda note: notes.append(note))
    patched, _, _ = N.KiLoRAApply().apply(
        base,
        chain,
        3,
        iteration=iteration,
        output_gate="adaptive",
        gate_kappa=0.5,
        path_width=1,
        settings=resolve_settings({"mix": "balanced"}),
        radial_step=1.0,
        orthogonal_step=1.0,
        radial_budget=0.0,
    )
    x = torch.randn(1, 5, 4, generator=generator)
    with compute.eager(), compute.dropout_key(17, 4):
        native = patched.run(x)
        calls.clear()
        with ir.interpret_all():
            actual = patched.run(x)
    assert torch.equal(actual, native)
    if derived == "full":
        assert any(d["target"] == "layers.0.feed_forward.w1" for d in details), details
        return
    assert any(
        target == "layers.0.feed_forward.w1"
        and any(s.form == "input" and compute.import_score(s) == "derived" for s in sets)
        for target, sets in interpreted
    ), interpreted
    parts = {"select", "gate", "iterate"}
    if iteration == "accumulate":
        parts.add("keep")
    assert parts <= set(calls), (calls, notes)
    if any("IR driver refused" in note for note in notes):
        pytest.fail(f"{notes!r} {details!r}", pytrace=False)


def test_comfy_depth_steps_match_eager(denoiser, folders, monkeypatch):

    assert N.COMFY_DEPTH_STEPS == (0.5, 1.0, 0.0)
    monkeypatch.setenv("KILORA_MASKS", "sample")
    path = folders["kilora"] / "routed-default-steps.kmod"
    _saved(denoiser, path)
    stem = "lora_unet_layers_0_feed_forward_w1"
    generator = torch.Generator().manual_seed(24)
    lora = folders["loras"] / "imported-default-steps.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=generator),
            f"{stem}.lora_up.weight": torch.randn(32, 2, generator=generator),
        },
        str(lora),
    )
    (chain,) = N.KiLoRASelect().select(path.name, 1.0, 1.0)
    (chain,) = N.KiLoRAImportLoRA().add(lora.name, 0.5, sets=chain)
    calls = []
    original = driver._run

    def record(part, variant, *args, **kwargs):
        calls.append((part, variant))
        return original(part, variant, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", record)
    notes = []
    monkeypatch.setattr(fused, "_note", notes.append)
    radial_step, orthogonal_step, radial_budget = N.COMFY_DEPTH_STEPS
    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)),
        chain,
        3,
        iteration="accumulate",
        output_gate="adaptive",
        gate_kappa=0.5,
        path_width=1,
        settings=resolve_settings({"mix": "balanced"}),
        radial_step=radial_step,
        orthogonal_step=orthogonal_step,
        radial_budget=radial_budget,
    )
    x = torch.randn(1, 5, 4, generator=generator)
    with compute.eager(), compute.dropout_key(17, 4):
        native = patched.run(x)
        calls.clear()
        notes.clear()
        with ir.interpret_all():
            actual = patched.run(x)
    assert torch.equal(actual, native)
    assert ("iterate", "accumulate") not in calls
    assert any("IR driver refused" in note for note in notes), notes


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_attention_depth_chain_interpreted(depth, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    generator = torch.Generator().manual_seed(243)
    x = torch.randn(2, 4, 6, generator=generator)
    y = torch.randn(2, 4, 5, generator=generator)
    down = torch.randn(3, 6, generator=generator)
    up = torch.randn(3, 5, generator=generator)
    layer = compute.LayerSet(
        form="input",
        import_score="derived",
        route_freeze=0.4,
        router=torch.randn(2, 5, generator=generator),
        keys_in=torch.randn(1, 1, 2, generator=generator),
        down=down,
        up=up,
        per_expert=3,
        scale=0.5,
        gate_mode="none",
    )
    calls = []
    original = driver._run

    def trace(part, variant, *args, **kwargs):
        calls.append((part, variant))
        return original(part, variant, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", trace)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(x, y, [layer], depth, target="layers.0.attention.to_q")
        with ir.interpret_all():
            actual = compute.run_layer(x, y, [layer], depth, target="layers.0.attention.to_q")
            attention_parts = list(calls)
            calls.clear()
            compute.run_layer(x, y, [layer], depth, target="layers.0.feed_forward.w1")
    assert calls == attention_parts
    assert ("entry", "chance_from_input") in calls
    if depth > 1:
        assert ("keep", "default") in calls
    assert torch.equal(actual, native)
    if depth == 1:
        torch.testing.assert_close(
            actual, y + (torch.nn.functional.linear(x, down) @ up) * 0.5, rtol=0, atol=1e-6
        )
    else:
        with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
            first = compute.run_layer(x, y, [layer], 1, target="layers.0.attention.to_q")
        assert not torch.equal(actual, first)


@pytest.mark.parametrize("experts", [1, 2])
def test_attention_adaptive_gate_interpreted(experts, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    generator = torch.Generator().manual_seed(412)
    x = torch.randn(2, 4, 6, generator=generator)
    y = torch.randn(2, 4, 5, generator=generator)
    layer = compute.LayerSet(
        form="input",
        import_score="derived",
        route_freeze=0.4,
        gate_mode="adaptive",
        gate_kappa=0.5,
        router=torch.randn(2, 5, generator=generator),
        keys_in=torch.randn(experts, 1, 2, generator=generator),
        down=torch.randn(experts * 3, 6, generator=generator),
        up=torch.randn(experts * 3, 5, generator=generator),
        per_expert=3,
        scale=0.5,
    )
    calls = []
    original = driver._run

    def trace(part, variant, *args, **kwargs):
        calls.append((part, variant))
        return original(part, variant, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", trace)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        native = compute.run_layer(x, y, [layer], 3, target="layers.0.attention.to_q")
        with ir.interpret_all():
            actual = compute.run_layer(x, y, [layer], 3, target="layers.0.attention.to_q")
    assert ("gate", "adaptive") in calls
    assert ("iterate", "output") in calls
    assert torch.equal(actual, native)


def test_attention_import_with_state_set_interpreted(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    generator = torch.Generator().manual_seed(722)
    x = torch.randn(2, 4, 6, generator=generator)
    y = torch.randn(2, 4, 5, generator=generator)
    plain = compute.LayerSet(
        form="input",
        iteration="path",
        gate_mode="none",
        router=torch.randn(2, 5, generator=generator),
        keys_in=torch.randn(1, 1, 2, generator=generator),
        down=torch.randn(3, 6, generator=generator),
        up=torch.randn(3, 5, generator=generator),
        per_expert=3,
        scale=0.5,
    )
    state = compute.LayerSet(
        form="state",
        iteration="path",
        gate_mode="none",
        router=torch.randn(2, 5, generator=generator),
        keys_in=torch.randn(2, 1, 2, generator=generator),
        down=torch.randn(4, 5, generator=generator) * 0.1,
        up=torch.randn(4, 5, generator=generator) * 0.1,
        per_expert=2,
    )
    calls = []
    original = driver._run

    def trace(part, variant, *args, **kwargs):
        calls.append((part, variant))
        return original(part, variant, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", trace)
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        expected = compute.run_layer(x, y, [plain, state], 3, target="layers.0.attention.to_q")
        with ir.interpret_all():
            actual = compute.run_layer(x, y, [plain, state], 3, target="layers.0.attention.to_q")
    assert any(part == "iterate" and variant.startswith("path") for part, variant in calls)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("iteration", ["accumulate", "path"])
def test_apply_import_with_active_freeze(denoiser, folders, iteration, monkeypatch):
    from kilora import route_freeze

    monkeypatch.setenv("KILORA_MASKS", "sample")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    generator = torch.Generator().manual_seed(723)
    stem = "lora_unet_layers_0_feed_forward_w1"
    path = folders["loras"] / "freeze.safetensors"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=generator),
            f"{stem}.lora_up.weight": torch.randn(32, 2, generator=generator),
        },
        str(path),
    )
    (chain,) = N.KiLoRAImportLoRA().add(path.name, 0.5)
    patched, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)),
        chain,
        3,
        iteration=iteration,
        output_gate="adaptive",
        gate_kappa=0.5,
        path_width=1,
        route_freeze=0.4,
        settings=resolve_settings({"mix": "balanced", "import_score": "derived"}),
        radial_step=1.0,
        orthogonal_step=1.0,
        radial_budget=0.0,
    )
    parts = []
    notes = []
    original = driver._run

    def trace(part, *args, **kwargs):
        parts.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(driver, "_run", trace)
    monkeypatch.setattr(fused, "_note", notes.append)
    x = torch.randn(1, 5, 4, generator=generator)
    with compute.eager(), compute.dropout_key(17, 4), compute.sampling_progress(0.6):
        with route_freeze.freezing({}) as cache:
            expected = [patched.run(x), patched.run(x)]
            assert cache
        with route_freeze.freezing({}) as cache:
            with ir.interpret_all():
                actual = [patched.run(x), patched.run(x)]
            assert cache
    assert {"select", "gate", "freeze"} <= set(parts)
    assert not any("IR driver refused" in note for note in notes), notes
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
