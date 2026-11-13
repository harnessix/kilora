# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher
from test_families import H, _ab, _model
from torch import nn

from kilora.comfyui import nodes as N
from kilora.comfyui.bypass import INJECTION_KEY, is_quantized
from kilora.comfyui.patching import clear


class Int8Weight:
    def __init__(self, w: torch.Tensor) -> None:
        self.scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        self.q = torch.round(w / self.scale).to(torch.int8)
        self.tensor_shape = tuple(w.shape)

    def dequantize(self) -> torch.Tensor:
        return self.q.float() * self.scale


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.weight = Int8Weight(linear.weight.detach())
        self.bias = linear.bias.detach().clone()

    def forward(self, x):
        return nn.functional.linear(x, self.weight.dequantize(), self.bias)


class Patcher(FakePatcher):
    def __init__(self, model, object_patches=None, model_options=None, injections=None):
        super().__init__(model, object_patches, model_options)
        self.injections = {k: list(v) for k, v in (injections or {}).items()}

    def clone(self):
        return Patcher(self.model, self.object_patches, self.model_options, self.injections)

    def set_injections(self, key, injections):
        self.injections[key] = list(injections)

    def inject(self):
        for injections in self.injections.values():
            for i in injections:
                i.inject(self)

    def eject(self):
        for injections in self.injections.values():
            for i in injections:
                i.eject(self)


def _quantize(model, paths):
    for path in paths:
        parent_name, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, QuantLinear(getattr(parent, attr)))


QKV = "double_blocks.1.img_attn.qkv"
MOD = "double_blocks.0.img_mod.lin"


@pytest.fixture
def lora(tmp_path, monkeypatch):
    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    gen = torch.Generator().manual_seed(1)
    sd = {}
    for name, (n_in, n_out) in {
        "transformer.transformer_blocks.1.attn.to_q": (H, H),
        "transformer.transformer_blocks.0.norm1.linear": (H, 6 * H),
        "transformer.transformer_blocks.0.attn.to_out.0": (H, H),
    }.items():
        a, b = _ab(gen, n_in, n_out)
        sd[f"{name}.lora_A.weight"], sd[f"{name}.lora_B.weight"] = a, b
    save_file(sd, str(tmp_path / "flux.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("flux.safetensors", 0.8)
    return chain


def test_quantized_layers_get_a_hook_not_a_baked_copy(lora):
    plain = _model("flux")
    model = _model("flux")
    _quantize(model, [QKV, MOD])
    q_layer = model.get_submodule(QKV)
    weight = q_layer.weight
    assert is_quantized(q_layer) and not is_quantized(
        model.get_submodule("double_blocks.0.img_attn.proj")
    )

    patched, _, _ = N.KiLoRAApply().apply(
        Patcher(FakeBaseModel(model)), lora, 1, output_gate="none"
    )
    assert not patched.object_patches
    assert len(patched.injections[INJECTION_KEY]) == 1

    reference, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(plain)), lora, 1, output_gate="none"
    )
    x, v = torch.randn(2, 5, H), torch.randn(2, H)
    with torch.no_grad():
        before = q_layer(x)
        patched.inject()
        try:
            assert type(q_layer) is QuantLinear and q_layer.weight is weight
            got_qkv, got_mod = q_layer(x), model.get_submodule(MOD)(v)
        finally:
            patched.eject()
        after = q_layer(x)
        want_qkv = reference.object_patches[f"diffusion_model.{QKV}"](x)
        want_mod = reference.object_patches[f"diffusion_model.{MOD}"](v)
        base_q = q_layer(x) - plain.get_submodule(QKV)(x)
        base_m = model.get_submodule(MOD)(v) - plain.get_submodule(MOD)(v)
    torch.testing.assert_close(got_qkv - base_q, want_qkv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_mod - base_m, want_mod, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(got_qkv, before)
    assert torch.equal(after, before)
    assert "forward" not in q_layer.__dict__ and "comfy_force_forward" not in q_layer.__dict__
    assert INJECTION_KEY not in clear(patched).injections


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_bypass_mode_on_and_off(lora, mode):
    model = _model("flux")
    patched, _, _ = N.KiLoRAApply().apply(
        Patcher(FakeBaseModel(model)), lora, 1, output_gate="none"
    )
    from kilora.comfyui import patching

    patcher = Patcher(FakeBaseModel(model))
    out, _ = patching.apply(patcher, lora, None, bypass=mode)
    if mode in ("auto", "on"):
        assert not out.object_patches and len(out.injections[INJECTION_KEY]) == 1
    else:
        assert INJECTION_KEY not in out.injections and len(out.object_patches) == 3
    assert not patched.object_patches


def test_auto_falls_back_once_without_injections(lora, caplog):
    from kilora.comfyui import patching

    with caplog.at_level("WARNING", logger="kilora"):
        out, _ = patching.apply(FakePatcher(FakeBaseModel(_model("flux"))), lora, 1)
    assert len(out.object_patches) == 3
    assert sum("no set_injections" in record.message for record in caplog.records) == 1


def test_auto_matches_copies_for_routed_and_fixed_changes(tmp_path):
    from test_working_copies import routed_chain

    from kilora.comfyui import patching

    chain = routed_chain(tmp_path)
    base = Patcher(FakeBaseModel(_model("flux")))
    auto, _ = patching.apply(base, chain, 1, joint=True, bypass="auto", import_score="keys")
    off, _ = patching.apply(base, chain, 1, joint=True, bypass="off", import_score="keys")
    for out in (auto, off):
        info = patching.applied(out)
        assert info["bypass"] in ("auto", "off")
        key_pass = info["key_pass"]
        assert key_pass.jobs and all(key_pass._targets(key) for key, _ in key_pass.jobs)
    paths = [
        "diffusion_model.double_blocks.0.img_attn.qkv",
        "diffusion_model.double_blocks.0.img_mod.lin",
    ]
    x = torch.randn(2, 5, H)
    with torch.no_grad():
        expected = [off.get_model_object(path)(x) for path in paths]
        auto.inject()
        try:
            actual = [auto.get_model_object(path)(x) for path in paths]
        finally:
            auto.eject()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("kind", ["routed", "text", "fixed", "combined"])
@pytest.mark.parametrize("layered", [False, True])
def test_call_values_match_copies_and_base_only(kind, layered):
    from test_expert_input import _text_module

    from kilora.bake import base_only, run_plan_forward
    from kilora.comfyui.bypass import ForwardHook, injection
    from kilora.comfyui.expert_text import conditioning_values
    from kilora.family_spec import TokenLayout
    from kilora.fixed_change import fixed_copy
    from kilora.sections import DenseChange

    baked, _ = _text_module(two=True)
    base = baked.__dict__["_kilora_base"]
    plan = baked._kilora_plan
    if kind == "routed":
        plan.parts[0][2].layout = TokenLayout("set")
    change = DenseChange(torch.eye(4), 0.25)
    fixed = fixed_copy(base, [change])
    copy = fixed if kind == "fixed" else baked
    if kind == "combined":
        copy = fixed_copy(baked, [change])

    def full_after(x, original, *args, **kwargs):
        value = run_plan_forward(plan, original, x, *args, **kwargs)
        return fixed._kilora_change(x, value) if kind == "combined" else value

    hook = ForwardHook(
        "diffusion_model.layer",
        fixed._kilora_change if kind == "fixed" else plan.run,
        None if kind == "fixed" else full_after,
    )
    tree = nn.Module()
    tree.layer = base
    patcher = Patcher(FakeBaseModel(tree))
    patcher.set_injections(INJECTION_KEY, [injection([hook])])
    x = torch.randn(2, 3, 4)
    values = {"base": x + 1, "expert_first": x + 2, "expert_second": x - 1} if layered else {}
    with torch.no_grad(), conditioning_values(values):
        expected = copy(x)
        with base_only():
            expected_base = copy(x)
        plain = base(x)
        patcher.inject()
        try:
            actual = tree.layer(x)
            with base_only():
                actual_base = tree.layer(x)
        finally:
            patcher.eject()
    assert torch.equal(actual, expected)
    assert torch.equal(expected_base, plain)
    assert torch.equal(actual_base, plain)
    if layered and kind in ("text", "combined"):
        with torch.no_grad():
            assert not torch.equal(expected, copy(x))


@pytest.mark.parametrize("hooked", [False, True])
def test_expert_text_strength_is_isolated(hooked):
    from kilora.comfyui.bypass import ForwardHook, injection
    from kilora.comfyui.expert_text import ExpertTextCache
    from kilora.fixed_change import fixed_copy
    from kilora.sections import DenseChange

    base = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(4))
    first = DenseChange(torch.eye(4), 0.5)
    first.text_expert = "first"
    second = DenseChange(torch.eye(4), 0.25)
    second.text_expert = "second"
    fixed = fixed_copy(base, [first, second])
    tree = nn.Module()
    tree.layer = base
    patcher = Patcher(FakeBaseModel(tree))
    patcher.set_injections(
        INJECTION_KEY, [injection([ForwardHook("diffusion_model.layer", fixed._kilora_change)])]
    )
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    if hooked:
        patcher.inject()
    try:
        layer = tree.layer if hooked else fixed
        values = ExpertTextCache(["first", "second"]).prepare(lambda: layer(x), (x,), 1)
    finally:
        if hooked:
            patcher.eject()
    assert torch.equal(values["base"], x)
    assert torch.equal(values["expert_first"], 1.5 * x)
    assert torch.equal(values["expert_second"], 1.25 * x)


def test_anima_two_expert_text_adapters_match_copies(tmp_path):
    import dataclasses

    from test_families import MLP, _tree

    from kilora.comfyui import patching
    from kilora.comfyui.expert_text import conditioning_values

    layers = {f"blocks.0.self_attn.{p}": (H, H) for p in ("q_proj", "k_proj", "v_proj")}
    layers.update(
        {
            "blocks.0.self_attn.output_proj": (H, H),
            "blocks.0.mlp.layer1": (H, MLP),
            "blocks.0.mlp.layer2": (MLP, H),
            "llm_adapter.out_proj": (H, H),
            "blocks.0.cross_attn.k_proj": (H, H),
            "blocks.0.cross_attn.v_proj": (H, H),
        }
    )
    model = _tree("Anima", layers)
    base = FakeBaseModel(model)
    base.extra_conds = lambda **kw: {"c_crossattn": model.llm_adapter.out_proj(kw["cross_attn"])}
    chain = []
    for i, strength in enumerate((0.5, 1.0)):
        a, b = _ab(torch.Generator().manual_seed(i), H, H)
        tensors = {}
        for stem in ("blocks_0_cross_attn_k_proj", "llm_adapter_out_proj"):
            tensors[f"lora_unet_{stem}.lora_down.weight"] = a.clone()
            tensors[f"lora_unet_{stem}.lora_up.weight"] = b.clone()
        path = tmp_path / f"expert{i}.safetensors"
        save_file(tensors, str(path))
        chain.append(patching.ModEntry("lora", str(path), strength))
    patcher = Patcher(base)
    hook, _ = patching.apply(patcher, chain, 1, bypass="on", joint=True, import_score="keys")
    copy, _ = patching.apply(patcher, chain, 1, bypass="off", joint=True, import_score="keys")
    x = torch.randn(1, 3, H)
    kwargs = {"cross_attn": x, "t5xxl_ids": torch.ones(1, 3), "kilora_layer": "ip"}
    projection_path = "diffusion_model.blocks.0.cross_attn.k_proj"
    adapter = model.llm_adapter
    original = adapter.out_proj
    with torch.no_grad():
        adapter.out_proj = copy.get_model_object("diffusion_model.llm_adapter.out_proj")
        try:
            expected_values = copy.object_patches["extra_conds"](**kwargs)
        finally:
            adapter.out_proj = original
        hook.inject()
        try:
            actual_values = hook.object_patches["extra_conds"](**kwargs)
        finally:
            hook.eject()
        for key in ("base", "expert_expert0", "expert_expert1"):
            assert torch.equal(
                actual_values["kilora_expert_text_" + key],
                expected_values["kilora_expert_text_" + key],
            )
        values = {
            key: actual_values["kilora_expert_text_" + key]
            for key in ("base", "expert_expert0", "expert_expert1")
        }
        with conditioning_values(values):
            expected = copy.get_model_object(projection_path)(x)
            hook.inject()
            try:
                actual = hook.get_model_object(projection_path)(x)
            finally:
                hook.eject()
        assert torch.equal(actual, expected)
        assert not torch.equal(values["base"], values["expert_expert0"])
        assert not torch.equal(values["base"], values["expert_expert1"])
        full_chain = [dataclasses.replace(chain[0], strength=1.0), chain[1]]
        for mode in ("on", "off"):
            full, _ = patching.apply(
                patcher, full_chain, 1, bypass=mode, joint=True, import_score="keys"
            )
            if mode == "on":
                full.inject()
            else:
                adapter.out_proj = full.get_model_object("diffusion_model.llm_adapter.out_proj")
            try:
                full_values = full.object_patches["extra_conds"](**kwargs)
            finally:
                if mode == "on":
                    full.eject()
                else:
                    adapter.out_proj = original
            assert torch.equal(
                full_values["kilora_expert_text_expert_expert1"], values["expert_expert1"]
            )
            torch.testing.assert_close(
                values["expert_expert0"] - values["base"],
                0.5 * (full_values["kilora_expert_text_expert_expert0"] - values["base"]),
            )


def test_off_keeps_hooks_for_quantized_layers(lora):
    from kilora.comfyui import patching

    model = _model("flux")
    _quantize(model, [QKV])
    out, _ = patching.apply(Patcher(FakeBaseModel(model)), lora, 1, bypass="off")
    assert f"diffusion_model.{QKV}" not in out.object_patches
    assert INJECTION_KEY in out.injections
    assert out.object_patches


class Fp8Linear(nn.Linear):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(linear.in_features, linear.out_features)
        with torch.no_grad():
            self.weight = nn.Parameter(
                linear.weight.detach().to(torch.float8_e4m3fn), requires_grad=False
            )
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x.dtype), self.bias)


class GGMLTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, tensor_type, tensor_shape):
        t = torch.Tensor._make_subclass(cls, data, False)
        t.tensor_type, t.tensor_shape = tensor_type, tensor_shape
        return t


class GGUFLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach()
        self.weight = nn.Parameter(
            GGMLTensor(w.to(torch.float16), "Q8_0", tuple(w.shape)), requires_grad=False
        )
        self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        w = torch.Tensor(self.weight.data).as_subclass(torch.Tensor).float()
        return nn.functional.linear(x, w, self.bias)


def _swap(model, path, cls):
    parent_name, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_name)
    setattr(parent, attr, cls(getattr(parent, attr)))
    return getattr(parent, attr)


@pytest.mark.parametrize("cls", [Fp8Linear, GGUFLinear], ids=["fp8", "gguf"])
def test_fp8_and_gguf_layers_get_a_hook(lora, cls):
    plain = _model("flux")
    model = _model("flux")
    layer = _swap(model, QKV, cls)
    weight = layer.weight
    assert is_quantized(layer)
    patched, _, _ = N.KiLoRAApply().apply(
        Patcher(FakeBaseModel(model)), lora, 1, output_gate="none"
    )
    assert f"diffusion_model.{QKV}" not in patched.object_patches
    reference, _, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(plain)), lora, 1, output_gate="none"
    )
    x = torch.randn(2, 5, H)
    with torch.no_grad():
        base = layer(x)
        patched.inject()
        try:
            assert type(layer) is cls and layer.weight is weight
            got = layer(x)
        finally:
            patched.eject()
        want = reference.object_patches[f"diffusion_model.{QKV}"](x)
        rounding = base - plain.get_submodule(QKV)(x)
    torch.testing.assert_close(got - rounding, want, rtol=1e-4, atol=1e-4)
    assert torch.equal(layer(x), base)


def test_flux3_and_chroma_apply_on_the_comfyui_path(tmp_path, monkeypatch):
    from test_families import _chroma, _tree

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    gen = torch.Generator().manual_seed(3)
    for name, model, stem, path, rows in (
        (
            "f3",
            _model("flux3"),
            "diffusion_model.single_blocks.0.k_proj",
            "single_blocks.0.qkv",
            slice(H, 2 * H),
        ),
        (
            "chroma",
            _tree("Chroma", _chroma()),
            "transformer.single_transformer_blocks.1.proj_out",
            "single_blocks.1.linear2",
            slice(None),
        ),
    ):
        target = model.get_submodule(path)
        n_out = H if rows == slice(H, 2 * H) else target.out_features
        a, b = _ab(gen, target.in_features, n_out)
        save_file(
            {f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b},
            str(tmp_path / f"{name}.safetensors"),
        )
        (chain,) = N.KiLoRAImportLoRA().add(f"{name}.safetensors", 1.0)
        patched, _, _ = N.KiLoRAApply().apply(
            FakePatcher(FakeBaseModel(model)), chain, 1, output_gate="none"
        )
        assert list(patched.object_patches) == [f"diffusion_model.{path}"]
        x = torch.randn(2, 3, target.in_features)
        with torch.no_grad():
            got = patched.object_patches[f"diffusion_model.{path}"](x)
            want = target(x).clone()
            want[..., rows] += x @ a.t() @ b.t()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_baked_and_fixed_layers_ask_comfyui_for_their_forward(lora):
    model = _model("flux")
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(model)), lora, -1)
    assert patched.object_patches
    for name, layer in patched.object_patches.items():
        assert getattr(layer, "comfy_force_forward", False) is True, name

    def fused(linear, x):
        if getattr(linear, "comfy_force_forward", False):
            return linear(x)
        return nn.functional.linear(x, linear.weight, linear.bias)

    x = torch.randn(2, 3, H)
    proj = patched.object_patches["diffusion_model.double_blocks.0.img_attn.proj"]
    base = model.get_submodule("double_blocks.0.img_attn.proj")
    with torch.no_grad():
        assert not torch.allclose(fused(proj, x), base(x))
    assert getattr(base, "comfy_force_forward", False) is False


def test_repeated_injection_does_not_resolve_paths_again(monkeypatch):
    from kilora.comfyui import bypass

    model = nn.Sequential(nn.Linear(3, 3))
    patcher = type("Patcher", (), {"model": model})()
    hook = bypass.ForwardHook("0", lambda x, h: h + x)
    injection = bypass.injection([hook])
    original = bypass._module
    paths = []

    def resolve(patcher, path):
        paths.append(path)
        return original(patcher, path)

    monkeypatch.setattr(bypass, "_module", resolve)
    injection.inject(patcher)
    installed = model[0].forward
    for _ in range(5):
        injection.inject(patcher)
    assert paths == ["0"] * 6 and model[0].forward is installed
    injection.eject(patcher)
    replacement = nn.Linear(3, 3)
    model[0] = replacement
    injection.inject(patcher)
    assert hook.module is replacement
    injection.eject(patcher)


def test_live_writer_hook_and_copy_reuse_paths_and_match(monkeypatch):
    chunk = pytest.importorskip("kilora_editor_chunk")
    from kilora.bake import baked_copy
    from kilora.comfyui import bypass

    torch.manual_seed(23)
    rows = torch.linalg.qr(torch.randn(64, 8)).Q.T
    name = "layer.to_q"
    editor = chunk.ChunkEditor(
        [name], {name: [8, 8]}, {"to_q": 1.0}, [chunk.Chunk("one", rows, torch.ones(8))]
    )
    x = torch.randn(2, 16, 8)
    base = nn.Linear(8, 8)
    original = bypass._module
    resolved = []

    def resolve(patcher, path):
        resolved.append(path)
        return original(patcher, path)

    monkeypatch.setattr(bypass, "_module", resolve)

    class Plan:
        parts = []

        def __init__(self):
            self.delta = torch.zeros(8, 8)

        def run(self, x, h):
            return h + x @ self.delta.T

    def write(mode):
        plan = Plan()
        model = nn.Module()
        model.layer = base if mode == "hook" else baked_copy(base, plan)
        patcher = type("Patcher", (), {"model": model})()
        hook = bypass.ForwardHook("layer", plan.run)
        injection = bypass.injection([hook])
        if mode == "hook":
            injection.inject(patcher)
        resolved.clear()
        outputs = []

        def forward(delta):
            plan.delta = delta
            if mode == "hook":
                injection.inject(patcher)
            out = model.layer(x)
            outputs.append(out.clone())
            return float(out.square().mean())

        def loss(pert, t, seed):
            return forward(pert[name])

        state = chunk.ChunkState.new(editor)
        writer = chunk.ChunkWriter(editor, [0.7], [1.0], [0.1], 0.2, per_event=4)
        with torch.no_grad():
            writer.step(state, loss, {}, 123)
            forward(torch.zeros(8, 8))
            forward(state.expanded_u(editor)[name].mean(dim=0))
        lookups = len(resolved)
        if mode == "hook":
            injection.eject(patcher)
        return state, outputs, lookups

    copy_state, copy_outputs, copy_lookups = write("copy")
    hook_state, hook_outputs, hook_lookups = write("hook")
    assert len(copy_outputs) == len(hook_outputs) == 6
    assert all(torch.equal(a, b) for a, b in zip(copy_outputs, hook_outputs, strict=True))
    assert torch.equal(copy_state.coef["one"], hook_state.coef["one"])
    assert copy_lookups == 0 and hook_lookups <= len(hook_outputs)


def test_write_after_sampling_reloads_hooks_and_matches_copy(monkeypatch):
    import sys
    import types

    chunk = pytest.importorskip("kilora_editor_chunk")
    from kilora.bake import baked_copy
    from kilora.comfyui import bypass
    from kilora.comfyui.live_write import PathHandle, image_loss
    from kilora.recording import Perturbation, active_perturbation

    torch.manual_seed(47)
    rows = torch.linalg.qr(torch.randn(64, 8)).Q.T
    name = "layer.to_q"
    editor = chunk.ChunkEditor(
        [name], {name: [8, 8]}, {"to_q": 1.0}, [chunk.Chunk("one", rows, torch.ones(8))]
    )
    latent = torch.randn(1, 4, 8)
    base = nn.Linear(8, 8)

    class Plan:
        parts = []

        def run(self, x, h):
            out = h + 0.25 * x
            trial = active_perturbation()
            delta = trial.delta(name, h) if trial is not None else None
            return out if delta is None else out + delta

    def write(mode):
        counts = {key: 0 for key in ("load", "inject", "eject", "forward")}
        model = nn.Module()
        plan = Plan()
        model.layer = base if mode == "hook" else baked_copy(base, plan)
        patcher = types.SimpleNamespace(model=model, load_device="cpu")
        hook = bypass.ForwardHook("layer", plan.run)
        injection = bypass.injection([hook])

        def load(models):
            assert models == [patcher]
            counts["load"] += 1
            if mode == "hook":
                counts["inject"] += 1
                injection.inject(patcher)

        management = types.ModuleType("comfy.model_management")
        management.load_models_gpu = load
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = management
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.model_management", management)
        outputs = []

        def apply_model(x, t, c_crossattn):
            counts["forward"] += 1
            if mode == "hook":
                assert hook.injected
            out = model.layer(x + c_crossattn.reshape(-1, 1, 1))
            outputs.append(out.clone())
            return out

        with torch.no_grad():
            load([patcher])
            for t in (1.0, 0.5, 0.0):
                apply_model(latent, torch.tensor([t]), torch.ones(1, 1))
            if mode == "hook":
                counts["eject"] += 1
                injection.eject(patcher)
            sample = outputs[-1].clone()
            outputs.clear()
            before = dict(counts)
            handle = PathHandle(
                None,
                None,
                None,
                timesteps=[0.5],
                conditioning={"c_crossattn": torch.ones(1, 1)},
                empty_conditioning={"c_crossattn": torch.zeros(1, 1)},
                apply_model=apply_model,
                patcher=patcher,
            )
            measure = image_loss(handle, latent, event=0)
            state = chunk.ChunkState.new(editor)
            writer = chunk.ChunkWriter(editor, [0.7], [1.0], [0.1], 0.2, per_event=4)

            def trial(values):
                return Perturbation({name: torch.eye(8)}, {name: values[name]})

            writer.step(state, lambda pert, t, seed: measure(trial(pert)), {}, 123)
            guard = measure.guard_score(None)
            guard_trial = measure.guard_score(
                trial({name: state.expanded_u(editor)[name].mean(dim=0)})
            )
            during = {key: counts[key] - before[key] for key in counts}
            if mode == "hook":
                injection.eject(patcher)
        return sample, state, outputs, guard, guard_trial, during

    copy = write("copy")
    hook = write("hook")
    assert torch.equal(copy[0], hook[0])
    assert torch.equal(copy[1].coef["one"], hook[1].coef["one"])
    assert len(copy[2]) == len(hook[2]) == 8
    assert all(torch.equal(a, b) for a, b in zip(copy[2], hook[2], strict=True))
    assert copy[3:5] == hook[3:5]
    assert copy[5] == {"load": 8, "inject": 0, "eject": 0, "forward": 8}
    assert hook[5] == {"load": 8, "inject": 8, "eject": 0, "forward": 8}
