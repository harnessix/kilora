# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from kilora.comfyui.expert_text import ExpertTextCache, install, select_text
from kilora.fixed_change import fixed_copy
from kilora.sections import DenseChange


class TextModel:
    def __init__(self):
        base = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(2))
        first = DenseChange(torch.eye(2), 0.5)
        first.text_expert = "first"
        second = DenseChange(torch.eye(2), 0.25)
        second.text_expert = "second"
        self.layer = fixed_copy(base, [first, second])
        self.calls = 0

    def extra_conds(self, **kwargs):
        self.calls += 1
        return {"c_crossattn": self.layer(kwargs["cross_attn"])}


class Patcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}

    def add_object_patch(self, path, value):
        self.object_patches[path] = value


def test_isolated_conditioning_cache_and_base_shared_output():
    model = TextModel()
    patcher = Patcher(model)
    fixed = {"diffusion_model.llm_adapter.proj": model.layer.changes}
    groups = install(patcher, fixed, True, False)
    assert {name: len(paths) for name, paths in groups.items()} == {"first": 1, "second": 1}
    extra = patcher.object_patches["extra_conds"]
    x = torch.tensor([[2.0, 4.0]])
    ids = torch.tensor([1, 2])
    kwargs = {"cross_attn": x, "t5xxl_ids": ids, "kilora_layer": "ip"}
    values = extra(**kwargs)
    assert torch.equal(values["c_crossattn"], x)
    assert torch.equal(values["kilora_expert_text_base"], x)
    assert torch.equal(values["kilora_expert_text_expert_first"], 1.5 * x)
    assert torch.equal(values["kilora_expert_text_expert_second"], 1.25 * x)
    assert model.calls == 4
    again = extra(**kwargs)
    assert torch.equal(again["c_crossattn"], values["c_crossattn"])
    assert model.calls == 5
    x.add_(1)
    extra(**kwargs)
    assert model.calls == 9
    unlayered = extra(cross_attn=x, t5xxl_ids=ids)
    assert list(unlayered) == ["c_crossattn"]
    assert model.calls == 10


def test_cache_plan_revision_and_inference_tensor_mutation():
    cache = ExpertTextCache(["first"])
    calls = []
    with torch.inference_mode():
        prompt = torch.ones(1)

        def forward():
            calls.append(1)
            return prompt.clone()

        cache.prepare(forward, (prompt,), 1)
        cache.prepare(forward, (prompt,), 1)
        assert len(calls) == 2
        cache.prepare(forward, (prompt,), 2)
        assert len(calls) == 4
        prompt.add_(1)
        cache.prepare(forward, (prompt,), 2)
        assert len(calls) == 6


def test_off_does_not_install_hook():
    model = TextModel()
    patcher = Patcher(model)
    install(patcher, {"diffusion_model.llm_adapter.proj": model.layer.changes}, False, True)
    assert patcher.object_patches == {}


def test_text_selection_leaves_other_fixed_mounts_unchanged():
    model = TextModel()
    base = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2))
    global_mount = fixed_copy(base, [DenseChange(torch.eye(2), 0.125)])
    x = torch.tensor([[2.0, 4.0]])
    shared = model.layer(x)
    global_shared = global_mount(x)
    with select_text(""):
        base_text = model.layer(x)
        global_base = global_mount(x)
    with select_text("first"):
        expert_text = model.layer(x)
        global_expert = global_mount(x)
    assert torch.equal(shared, 1.75 * x)
    assert torch.equal(base_text, x)
    assert torch.equal(expert_text, 1.5 * x)
    assert torch.equal(global_shared, global_base)
    assert torch.equal(global_shared, global_expert)


def test_apply_reports_text_coverage_and_preserves_off_paths(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import MLP, H, _ab, _tree

    from kilora.comfyui import nodes

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
    generator = torch.Generator().manual_seed(2)
    a, b = _ab(generator, H, H)
    save_file(
        {
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": a,
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": b,
            "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": a.clone(),
            "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": b.clone(),
            "lora_unet_llm_adapter_out_proj.lora_down.weight": a.clone(),
            "lora_unet_llm_adapter_out_proj.lora_up.weight": b.clone(),
        },
        str(tmp_path / "style.safetensors"),
    )
    monkeypatch.setattr(nodes.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    (chain,) = nodes.KiLoRAImportLoRA().add("style.safetensors", 0.5)
    base = FakeBaseModel(model)
    base.extra_conds = lambda **kwargs: {
        "c_crossattn": base.diffusion_model.llm_adapter.out_proj(kwargs["cross_attn"])
    }
    patched, report, _ = nodes.KiLoRAApply().apply(FakePatcher(base), chain, 1)
    off, _, _ = nodes.KiLoRAApply().apply(FakePatcher(base), chain, 1, expert_text="off")
    assert "1 text mounts; text change reaches 1 of 2 cross-attention projections" in report
    assert "extra_conds" in patched.object_patches
    assert "extra_conds" not in off.object_patches
    x = torch.randn(1, 3, H)
    kwargs = {"cross_attn": x, "t5xxl_ids": torch.ones(1, 3)}

    def call(patcher, **extra):
        backup = {}
        for path, obj in patcher.object_patches.items():
            parent_name, _, attr = path.rpartition(".")
            parent = patcher.get_model_object(parent_name) if parent_name else base
            backup[path] = (parent, attr, getattr(parent, attr))
            setattr(parent, attr, obj)
        try:
            return base.extra_conds(**kwargs, **extra)
        finally:
            for parent, attr, original in backup.values():
                setattr(parent, attr, original)

    assert torch.equal(call(patched)["c_crossattn"], call(off)["c_crossattn"])
    assert torch.equal(call(off, kilora_layer="ip")["c_crossattn"], call(off)["c_crossattn"])
    layered = call(patched, kilora_layer="ip")
    assert torch.equal(layered["c_crossattn"], layered["kilora_expert_text_base"])
    projection = getattr(model.blocks, "0").cross_attn.v_proj
    assert torch.equal(
        projection(layered["c_crossattn"]), projection(model.llm_adapter.out_proj(x))
    )
    assert not torch.equal(
        projection(layered["c_crossattn"]), projection(layered["kilora_expert_text_expert_style"])
    )
    assert not torch.equal(layered["c_crossattn"], layered["kilora_expert_text_expert_style"])
