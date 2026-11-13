# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from kilora.bake import bake
from kilora.base_weights import read_base_weight
from kilora.families import block_list, detect, detect_names, get_family, normalize_for
from kilora.family_spec import map_region
from kilora.multi_import import import_loras, lora_file_size
from kilora.roles import role_spans


class DiTBlock(nn.Module):
    def forward(self, x):
        q, k, v = self.attn.qkv_proj(x).chunk(3, dim=-1)
        x = x + self.attn.out_proj(q + k + v)
        gate, value = self.mlp.fc1(x).chunk(2, dim=-1)
        return x + self.mlp.fc2(torch.nn.functional.silu(gate) * value)


class MiniMaxH3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = DiTBlock()
            block.attn = nn.Module()
            block.attn.qkv_proj = nn.Linear(4, 12, bias=False)
            block.attn.out_proj = nn.Linear(4, 4)
            block.attn.to_gate_compress = nn.Linear(4, 4)
            block.mlp = nn.Module()
            block.mlp.fc1 = nn.Linear(4, 16)
            block.mlp.fc2 = nn.Linear(8, 4)
            block.adaln_proj = nn.Module()
            block.adaln_proj.linear = nn.Linear(4, 4)
            self.blocks.append(block)
        self.token_projection = nn.Linear(4, 4)
        self.condition_proj = nn.Linear(4, 4)
        self.video_patch_proj = nn.Linear(4, 4)
        self.time_embedder = nn.Sequential(nn.Linear(4, 4))
        self.final_layer = nn.Module()
        self.final_layer.video_out = nn.Linear(4, 4)

    def forward(self, x):
        return self.blocks[0].attn.qkv_proj(x)


class MiniMaxH3ImageModel(MiniMaxH3Model):
    pass


def test_detection_and_targets():
    model = MiniMaxH3ImageModel()
    spec = get_family("minimax-h3-image")
    assert detect(model) is spec
    assert detect(MiniMaxH3Model()) is None
    assert detect_names(list(dict(model.named_modules()))) is spec
    assert spec.content_blocks == 50
    assert spec.targets(model) == [
        f"blocks.{i}.{name}"
        for i in range(2)
        for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    ]


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer.", "base_model.model."])
@pytest.mark.parametrize(
    "name,offset,rows,target",
    [
        ("attn.qkv_proj", 0, 12, "attn.qkv_proj"),
        ("attn.out_proj", 0, 4, "attn.out_proj"),
        ("mlp.fc1", 0, 16, "mlp.fc1"),
        ("mlp.fc2", 0, 4, "mlp.fc2"),
        ("attn.to_q", 0, 4, "attn.qkv_proj"),
        ("attn.to_k", 4, 4, "attn.qkv_proj"),
        ("attn.to_v", 8, 4, "attn.qkv_proj"),
        ("attn.q_proj", 0, 4, "attn.qkv_proj"),
        ("attn.k_proj", 4, 4, "attn.qkv_proj"),
        ("attn.v_proj", 8, 4, "attn.qkv_proj"),
        ("mlp.gate_proj", 0, 8, "mlp.fc1"),
        ("mlp.up_proj", 8, 8, "mlp.fc1"),
    ],
)
def test_names(prefix, name, offset, rows, target):
    model = MiniMaxH3ImageModel()
    width = 8 if target == "mlp.fc2" else 4
    down, up = torch.randn(2, width), torch.randn(rows, 2)
    stem = prefix + "blocks.0." + name
    result = normalize_for(
        "minimax-h3-image",
        {stem + ".lora_down.weight": down, stem + ".lora_up.weight": up},
        model,
    )
    key = "diffusion_model.blocks.0." + target
    actual = result.state[key + ".lora_up.weight"] @ result.state[key + ".lora_down.weight"]
    expected = torch.zeros_like(actual)
    expected[offset : offset + rows] = up @ down
    torch.testing.assert_close(actual, expected)


def test_2d_import_matches_merged_weight(tmp_path):
    model = MiniMaxH3ImageModel().eval()
    down, up = torch.randn(2, 4), torch.randn(12, 2)
    stem = "diffusion_model.blocks.0.attn.qkv_proj"
    from safetensors.torch import save_file

    path = tmp_path / "import.safetensors"
    save_file(
        {stem + ".lora_down.weight": down, stem + ".lora_up.weight": up},
        path,
    )
    imported = import_loras([str(path)], model, family="minimax-h3-image")
    x = torch.randn(7, 4)
    expected = torch.nn.functional.linear(x, model.blocks[0].attn.qkv_proj.weight + up @ down)
    baked = bake(model, [imported.adapter_set], depth=1)
    torch.testing.assert_close(model(x), expected)
    baked.remove()


def test_prefix_region_and_roles():
    layout = get_family("minimax-h3-image").layout("blocks.0.attn.qkv_proj")
    region = map_region(torch.ones(4, 4), 11, layout, (4, 4), text_tokens=2)
    torch.testing.assert_close(region, torch.tensor([[0.0] * 7 + [1.0] * 4]))
    assert role_spans(layout, 11, (4, 4), [(2, 2)]) == {
        "text": [(0, 7)],
        "target": [(7, 11)],
    }


class QuantizedWeight(torch.Tensor):
    calls = 0

    def dequantize(self):
        type(self).calls += 1
        return self.as_subclass(torch.Tensor) / 4


class QuantizedLinear(nn.Linear):
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.dequantize(), self.bias)


def _quantize_layer(layer):
    quantized = QuantizedLinear(layer.in_features, layer.out_features, bias=layer.bias is not None)
    del quantized._parameters["weight"]
    quantized.register_buffer("weight", (layer.weight.detach() * 4).as_subclass(QuantizedWeight))
    if layer.bias is not None:
        quantized.bias.data.copy_(layer.bias)
    return quantized


def test_quantized_base_reader_dequantizes_before_cast():
    layer = nn.Linear(4, 4, dtype=torch.float64)
    quantized = _quantize_layer(layer)
    QuantizedWeight.calls = 0
    weight = read_base_weight(quantized, "test", device="cpu")
    assert QuantizedWeight.calls == 1
    assert weight.dtype == torch.float32 and weight.device.type == "cpu"
    torch.testing.assert_close(weight, layer.weight.float())
    plain = read_base_weight(layer, device="cpu")
    assert plain.dtype == layer.weight.dtype and not plain.requires_grad
    assert plain.data_ptr() == layer.weight.data_ptr()


@pytest.mark.parametrize("size_norm", ["off", "weight"])
def test_quantized_import_size_and_depth_one_match_dense(tmp_path, size_norm):
    import copy

    from safetensors.torch import save_file

    dense = MiniMaxH3ImageModel().eval()
    quantized = copy.deepcopy(dense)
    target = "blocks.0.attn.qkv_proj"
    quantized.blocks[0].attn.qkv_proj = _quantize_layer(dense.get_submodule(target))
    stem = "diffusion_model." + target
    state = {
        stem + ".lora_down.weight": torch.randn(2, 4),
        stem + ".lora_up.weight": torch.randn(12, 2),
    }
    path = tmp_path / "quantized-import.safetensors"
    save_file(state, path)
    dense_size = lora_file_size(state, dense, family="minimax-h3-image")
    quantized_size = lora_file_size(state, quantized, family="minimax-h3-image")
    assert quantized_size == pytest.approx(dense_size)
    assert lora_file_size(path, quantized, family="minimax-h3-image") == pytest.approx(dense_size)
    results = [
        import_loras([str(path)], model, family="minimax-h3-image", size_norm=size_norm)
        for model in (dense, quantized)
    ]
    for field in ("down", "up"):
        torch.testing.assert_close(
            getattr(results[0].adapter_set.experts[target], field),
            getattr(results[1].adapter_set.experts[target], field),
        )
    x = torch.randn(7, 4)
    outputs = []
    for model, result in zip((dense, quantized), results, strict=True):
        baked = bake(model, [result.adapter_set], depth=1)
        try:
            outputs.append(model(x))
        finally:
            baked.remove()
    torch.testing.assert_close(outputs[0], outputs[1])


def test_base_weight_mapping_does_not_retain_dense_weights():
    from kilora.base_weights import BaseWeights
    from kilora.size_norm import base_key

    layer = _quantize_layer(nn.Linear(4, 4))
    weights = BaseWeights({"target": (layer, None)})
    QuantizedWeight.calls = 0
    norms, identity = base_key(weights)
    assert QuantizedWeight.calls == 1
    assert identity is not None and norms["target"] > 0
    assert weights.entries == {"target": (layer, None)}
    base_key(weights)
    assert QuantizedWeight.calls == 1


def test_block_list_mounts_called_once_in_order():
    from types import SimpleNamespace

    from kilora.bindings import BINDINGS
    from kilora.comfyui.patching import diffusers_shapes

    model = MiniMaxH3ImageModel()
    binding = BINDINGS["minimax-h3-image"]
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    shapes = diffusers_shapes(patcher, binding)
    assert list(shapes) == get_family("minimax-h3-image").targets(model)
    calls, handles = [], []
    for name in shapes:
        place = binding.to_comfy(name)
        path = place.path.removeprefix("diffusion_model.")
        module = model.get_submodule(path)
        handles.append(module.register_forward_hook(lambda m, a, o, name=name: calls.append(name)))
    attr, blocks = block_list("minimax-h3-image", model)
    assert attr == "blocks"
    assert [i for i, _ in blocks] == [0, 1]
    x = torch.randn(9, 4)
    try:
        for _, block in blocks:
            x = block(x)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == list(shapes)


def test_h3_image_rows_at_block_input():
    spec = get_family("minimax-h3-image")
    model = MiniMaxH3ImageModel()
    image = torch.arange(16).reshape(4, 4).float()
    selected = []

    def record(module, args):
        x = args[0]
        selected.append(x[spec.image_token_rows(x.shape[0], (4, 4))])

    handle = model.blocks[0].register_forward_pre_hook(record)
    try:
        for text_length in (2, 7):
            packed = torch.cat([torch.full((text_length + 3, 4), -1.0), image])
            model.blocks[0](packed)
    finally:
        handle.remove()
    for rows in selected:
        torch.testing.assert_close(rows, image)
    assert selected[0].shape == selected[1].shape


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer.", "base_model.model."])
@pytest.mark.parametrize("suffix", ["", ".weight", ".lora_down.weight"])
def test_name_detection_for_image_without_audio(prefix, suffix):
    paths = [prefix + "blocks.3.attn.qkv_proj" + suffix]
    assert detect_names(paths).family == "minimax-h3-image"
    for audio in ("audio_patch_proj", "final_layer.audio_out", "blocks.0.audio_attn.qkv_proj"):
        assert detect_names([*paths, prefix + audio + ".weight"]) is None


def test_h3_image_rows_use_patched_latent_grid():
    spec = get_family("minimax-h3-image")
    rows = spec.image_token_rows(17, (5, 7))
    assert rows == slice(11, 17)
    region = map_region(torch.ones(5, 7), 17, spec.layout(spec.image_token_target), (5, 7))
    torch.testing.assert_close(region[0, rows], torch.ones(6))
    assert int(region.sum()) == 6


def test_quantized_dora_size_matches_dense():
    import copy

    dense = MiniMaxH3ImageModel().eval()
    quantized = copy.deepcopy(dense)
    target = "blocks.0.attn.qkv_proj"
    quantized.blocks[0].attn.qkv_proj = _quantize_layer(dense.get_submodule(target))
    stem = "diffusion_model." + target
    state = {
        stem + ".lora_down.weight": torch.randn(2, 4),
        stem + ".lora_up.weight": torch.randn(12, 2),
        stem + ".dora_scale": torch.rand(1, 4) + 0.5,
    }
    expected = lora_file_size(state, dense, family="minimax-h3-image")
    assert expected is not None and expected > 0
    assert lora_file_size(state, quantized, family="minimax-h3-image") == pytest.approx(expected)


def test_quantized_state_conversion_matches_dense(tmp_path):
    import copy

    from safetensors.torch import save_file

    from kilora.lora_import import to_state_form

    dense = MiniMaxH3ImageModel().eval()
    quantized = copy.deepcopy(dense)
    target = "blocks.0.attn.qkv_proj"
    quantized.blocks[0].attn.qkv_proj = _quantize_layer(dense.get_submodule(target))
    stem = "diffusion_model." + target
    path = tmp_path / "state-conversion.safetensors"
    save_file(
        {
            stem + ".lora_down.weight": torch.randn(2, 4),
            stem + ".lora_up.weight": torch.randn(12, 2),
        },
        path,
    )
    imported = import_loras([str(path)], dense, family="minimax-h3-image").adapter_set
    dense_state, dense_lost = to_state_form(imported, dense)
    quantized_state, quantized_lost = to_state_form(imported, quantized)
    assert dense_lost == quantized_lost
    torch.testing.assert_close(
        dense_state.experts[target].down, quantized_state.experts[target].down
    )


def test_family_solve_sampling_defaults_are_optional_and_frozen():
    from kilora.family_spec import FamilySpec

    values = {"steps": 6, "cfg": 1.0, "size": 1024}
    assert get_family("minimax-h3-image").solve_sampling == values
    for family in ("anima", "z-image", "moss-image"):
        assert get_family(family).solve_sampling is None
    spec = FamilySpec(family="test-sampling", solve_sampling=values)
    values["steps"] = 30
    assert spec.solve_sampling["steps"] == 6
    with pytest.raises(TypeError):
        spec.solve_sampling["steps"] = 30
