# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from kilora.base_weights import is_quantized_weight, read_base_weight
from kilora.families import (
    BLOCK_LISTS,
    FAMILIES,
    block_list,
    detect,
    detect_names,
    get_family,
    normalize_for,
    video_latent_size,
)
from kilora.family_spec import TokenLayout, map_region
from kilora.multi_import import import_loras
from kilora.roles import role_spans

H = 4
LAYERS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _block(adaln=True):
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.qkv_proj = nn.Linear(H, 3 * H, bias=False)
    block.attn.out_proj = nn.Linear(H, H, bias=False)
    block.mlp = nn.Module()
    block.mlp.fc1 = nn.Linear(H, 4 * H, bias=False)
    block.mlp.fc2 = nn.Linear(2 * H, H, bias=False)
    if adaln:
        block.adaln_proj = nn.Module()
        block.adaln_proj.linear = nn.Linear(H, H)
    return block


class MiniMaxH3Model(nn.Module):
    def __init__(self, audio=True):
        super().__init__()
        self.video_patch_proj = nn.Linear(H, H)
        if audio:
            self.audio_patch_proj = nn.Linear(H, H)
        self.condition_proj = nn.Linear(H, H)
        self.time_embedder = nn.Module()
        self.time_embedder.proj_in = nn.Linear(H, H)
        self.time_embedder.proj_out = nn.Linear(H, H)
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([_block(adaln=False) for _ in range(2)])
        self.blocks = nn.ModuleList([_block() for _ in range(2)])
        self.final_layer = nn.Module()
        self.final_layer.video_out = nn.Linear(H, H)
        if audio:
            self.final_layer.audio_out = nn.Linear(H, H)

    def forward(self, x):
        return self.blocks[0].attn.qkv_proj(x)


class MiniMaxH3ImageModel(MiniMaxH3Model):
    def __init__(self):
        super().__init__(audio=False)


BLOCK_TARGETS = [f"blocks.{i}.{name}" for i in range(2) for name in LAYERS]


def test_detection_by_class_and_targets():
    spec = get_family("minimax-h3")
    model = MiniMaxH3Model()
    assert detect(model) is spec
    assert detect(MiniMaxH3ImageModel()).family == "minimax-h3-image"
    assert detect(MiniMaxH3Model(audio=False)) is None
    assert spec.content_blocks == 50
    assert spec.targets(model) == BLOCK_TARGETS
    for path in (
        "video_patch_proj",
        "audio_patch_proj",
        "condition_proj",
        "time_embedder.proj_in",
        "final_layer.video_out",
        "final_layer.audio_out",
        "blocks.0.adaln_proj.linear",
        "token_refiner.blocks.0.attn.qkv_proj",
        "token_refiner.blocks.1.mlp.fc2",
    ):
        assert not spec.is_target(path)
        assert spec.is_excluded(path)


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_name_detection_of_checkpoint_keys(prefix):
    tree = dict(MiniMaxH3Model().named_modules())
    keys = [prefix + name + ".weight" for name, m in tree.items() if isinstance(m, nn.Linear)]
    assert detect_names(keys).family == "minimax-h3"
    image = [k for k in keys if "audio" not in k]
    assert detect_names(image).family == "minimax-h3-image"


def test_name_detection_of_gguf_names():
    names = [
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "condition_proj.weight",
        *(f"blocks.{i}.{n}.weight" for i in range(50) for n in LAYERS),
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
    ]
    assert detect_names(names).family == "minimax-h3"
    module_paths = [n.rsplit(".", 1)[0] for n in names]
    assert detect_names(module_paths).family == "minimax-h3"


def _lora_keys(layout):
    stems = [f"blocks.{i}.{n}" for i in range(2) for n in LAYERS]
    stems += [f"token_refiner.blocks.0.{n}" for n in LAYERS]
    if layout == "kohya":
        return [
            f"lora_unet_{s.replace('.', '_')}.lora_{p}.weight"
            for s in stems
            for p in ("down", "up")
        ]
    prefix = {"comfy": "diffusion_model.", "diffusers": "transformer.", "peft": "base_model.model."}
    return [f"{prefix.get(layout, '')}{s}.lora_{p}.weight" for s in stems for p in "AB"]


@pytest.mark.parametrize("layout", ["comfy", "diffusers", "peft", "plain"])
def test_lora_keys_name_both_h3_models(layout):
    from kilora.prepare import lora_families

    keys = _lora_keys(layout)
    assert detect_names(keys).family == "minimax-h3-image"
    assert lora_families(keys) == ("minimax-h3", "minimax-h3-image")
    audio = [*keys, keys[0].replace("blocks.0.attn.qkv_proj", "audio_patch_proj")]
    assert detect_names(audio).family == "minimax-h3"


def _packed(text, cond, audio, video):
    rows = [[0.0, -1, -1, -1]] * text + [[1.0, -1, -1, -1]] * cond + [[2.0, -1, -1, -1]] * audio
    t, h, w = video
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                rows.append([3.0, ti, hi, wi])
    return torch.tensor(rows)


def test_video_rows_frames_and_audio_rows():
    spec = get_family("minimax-h3")
    layout = spec.layout("blocks.7.mlp.fc1")
    assert layout == TokenLayout("video3d", (1, 2, 2), text="prefix")
    latent = (3, 4, 6)
    seq = _packed(text=5, cond=6, audio=8, video=(3, 2, 3))
    tokens = seq.shape[0]
    rows = spec.image_token_rows(tokens, latent)
    assert rows == slice(tokens - 18, tokens)
    video = seq[rows]
    assert bool((video[:, 0] == 3).all())
    grid = video[:, 1:].reshape(3, 2, 3, 3)
    for t in range(3):
        for h in range(2):
            for w in range(3):
                assert grid[t, h, w].tolist() == [t, h, w]
    assert bool((seq[: rows.start, 0] < 3).all())
    assert role_spans(layout, tokens, latent, []) == {
        "text": [(0, rows.start)],
        "target": [(rows.start, tokens)],
    }
    single = _packed(text=5, cond=0, audio=4, video=(1, 2, 3))
    assert spec.image_token_rows(single.shape[0], (4, 6)) == slice(9, 15)


def test_region_covers_only_video_rows_of_its_area_in_every_frame():
    spec = get_family("minimax-h3")
    layout = spec.layout(spec.image_token_target)
    latent = (3, 4, 6)
    seq = _packed(text=5, cond=6, audio=8, video=(3, 2, 3))
    tokens = seq.shape[0]
    mask = torch.zeros(4, 6)
    mask[:, :4] = 1
    region = map_region(mask, tokens, layout, latent, text_tokens=5)
    assert region.shape == (1, tokens)
    start = tokens - 18
    torch.testing.assert_close(region[0, :start], torch.zeros(start))
    video = region[0, start:].reshape(3, 2, 3)
    torch.testing.assert_close(video[:, :, :2], torch.ones(3, 2, 2))
    torch.testing.assert_close(video[:, :, 2], torch.zeros(3, 2))
    frames = torch.zeros(3, 4, 6)
    frames[1] = 1
    region = map_region(frames, tokens, layout, latent)
    video = region[0, start:].reshape(3, 2, 3)
    torch.testing.assert_close(video[1], torch.ones(2, 3))
    assert float(video[0].sum()) == 0 and float(video[2].sum()) == 0
    assert float(region[0, :start].sum()) == 0


class _Constant:
    def __init__(self, value):
        self.cond = value


def test_video_size_of_a_packed_sampler_latent():
    shapes = [(1, 24, 3, 4, 6), (1, 32, 2, 7)]
    packed = (1,)
    assert video_latent_size("minimax-h3", packed, {"latent_shapes": shapes}) == (3, 4, 6)
    constant = {"latent_shapes": _Constant(shapes)}
    assert video_latent_size("minimax-h3", packed, constant) == (3, 4, 6)
    assert video_latent_size("minimax-h3", (3, 4, 6), {}) == (3, 4, 6)
    for family in ("minimax-h3-image", "ltxv", "wan", None):
        assert video_latent_size(family, packed, {"latent_shapes": shapes}) == packed


def _lora_state(layout, gen):
    state = {}
    shapes = {"attn.qkv_proj": (H, 3 * H), "attn.out_proj": (H, H)}
    shapes.update({"mlp.fc1": (H, 4 * H), "mlp.fc2": (2 * H, H)})
    canonical = {}
    for stem in [f"blocks.{i}.{n}" for i in range(2) for n in LAYERS] + [
        f"token_refiner.blocks.0.{n}" for n in LAYERS
    ]:
        fan_in, fan_out = shapes[".".join(stem.split(".")[-2:])]
        canonical[stem] = (
            torch.randn(2, fan_in, generator=gen),
            torch.randn(fan_out, 2, generator=gen),
        )
    for stem, (down, up) in canonical.items():
        if layout == "kohya":
            body = "lora_unet_" + stem.replace(".", "_")
            state[body + ".lora_down.weight"] = down
            state[body + ".lora_up.weight"] = up
        else:
            prefix = {"comfy": "diffusion_model.", "diffusers": "transformer."}
            prefix["peft"] = "base_model.model."
            body = prefix.get(layout, "") + stem
            state[body + ".lora_A.weight"] = down
            state[body + ".lora_B.weight"] = up
    return state, canonical


LAYOUTS = ["comfy", "diffusers", "peft", "plain", "kohya"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_key_layout_normalizes_to_the_same_targets(layout):
    model = MiniMaxH3Model()
    state, canonical = _lora_state(layout, torch.Generator().manual_seed(3))
    result = normalize_for("minimax-h3", state, model)
    assert result.skipped == []
    for target in BLOCK_TARGETS:
        down, up = canonical[target]
        key = "diffusion_model." + target
        found = {
            part: result.state.get(key + f".lora_{part}.weight")
            for part in ("A", "B", "down", "up")
        }
        found_down = found["down"] if found["down"] is not None else found["A"]
        found_up = found["up"] if found["up"] is not None else found["B"]
        torch.testing.assert_close(found_up @ found_down, up @ down)
    assert {k.rsplit(".lora_", 1)[0] for k in result.state} == {
        "diffusion_model." + t for t in BLOCK_TARGETS
    }
    assert sorted(result.fixed) == sorted(f"token_refiner.blocks.0.{n}" for n in LAYERS)
    for name in LAYERS:
        down, up = canonical[f"token_refiner.blocks.0.{name}"]
        fixed_down, fixed_up = result.fixed[f"token_refiner.blocks.0.{name}"]
        torch.testing.assert_close(fixed_up @ fixed_down, up @ down)


def _imported_change(expert):

    n_experts, n_routes = expert.down.shape[:2]
    return torch.stack(
        [
            expert.up[index, route].t() @ expert.down[index, route]
            for index in range(n_experts)
            for route in range(n_routes)
        ]
    )


def test_each_key_layout_imports_the_same_set(tmp_path):
    from safetensors.torch import save_file

    model = MiniMaxH3Model().eval()
    deltas = []
    for layout in LAYOUTS:
        state, _ = _lora_state(layout, torch.Generator().manual_seed(5))
        path = tmp_path / f"{layout}.safetensors"
        save_file(state, path)
        result = import_loras([str(path)], model, family="minimax-h3")
        experts = result.adapter_set.experts
        assert sorted(experts) == sorted(BLOCK_TARGETS)
        deltas.append({t: _imported_change(experts[t]) for t in BLOCK_TARGETS})
    for other in deltas[1:]:
        for target in BLOCK_TARGETS:
            torch.testing.assert_close(other[target], deltas[0][target])


@pytest.mark.parametrize(
    "name,offset,rows,target",
    [
        ("attn.to_q", 0, H, "attn.qkv_proj"),
        ("attn.to_v", 2 * H, H, "attn.qkv_proj"),
        ("attn.k_proj", H, H, "attn.qkv_proj"),
        ("mlp.gate_proj", 0, 2 * H, "mlp.fc1"),
        ("mlp.up_proj", 2 * H, 2 * H, "mlp.fc1"),
    ],
)
def test_split_projection_names_land_in_the_fused_layers(name, offset, rows, target):
    model = MiniMaxH3Model()
    down, up = torch.randn(2, H), torch.randn(rows, 2)
    stem = "transformer.blocks.1." + name
    result = normalize_for(
        "minimax-h3", {stem + ".lora_A.weight": down, stem + ".lora_B.weight": up}, model
    )
    key = "diffusion_model.blocks.1." + target
    actual = result.state[key + ".lora_up.weight"] @ result.state[key + ".lora_down.weight"]
    expected = torch.zeros_like(actual)
    expected[offset : offset + rows] = up @ down
    torch.testing.assert_close(actual, expected)


class QuantizedWeight(torch.Tensor):
    def dequantize(self):
        return self.as_subclass(torch.Tensor) / 4


def test_quantized_weight_reads_through_dequantize():
    layer = nn.Linear(H, H)
    stored = (layer.weight.detach() * 4).as_subclass(QuantizedWeight)
    quantized = nn.Linear(H, H)
    del quantized._parameters["weight"]
    quantized.register_buffer("weight", stored)
    assert is_quantized_weight(quantized.weight)
    weight = read_base_weight(quantized, "blocks.0.attn.qkv_proj", device="cpu")
    assert type(weight) is torch.Tensor and weight.dtype == torch.float32
    torch.testing.assert_close(weight, layer.weight.detach())


class GGMLTensor(torch.Tensor):
    tensor_type = None
    tensor_shape = None


READS = []


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    READS.append(tensor.tensor_type)
    raw = tensor.as_subclass(torch.Tensor)
    return (raw / 2).reshape(tensor.tensor_shape).to(dtype)


def test_gguf_weight_reads_through_the_loader_dequantizer():
    layer = nn.Linear(H, 3 * H, bias=False)
    raw = (layer.weight.detach() * 2).reshape(-1).as_subclass(GGMLTensor)
    raw.tensor_type = "Q8_0"
    raw.tensor_shape = torch.Size((3 * H, H))
    module = nn.Linear(H, 3 * H, bias=False)
    del module._parameters["weight"]
    module.register_buffer("weight", raw)
    assert is_quantized_weight(module.weight)
    READS.clear()
    weight = read_base_weight(module, "blocks.0.attn.qkv_proj", device="cpu")
    assert READS == ["Q8_0"]
    assert weight.shape == (3 * H, H) and weight.dtype == torch.float32
    torch.testing.assert_close(weight, layer.weight.detach())


def test_block_list_and_solve_sampling():
    model = MiniMaxH3Model()
    assert BLOCK_LISTS["minimax-h3"] == "blocks"
    attr, blocks = block_list("minimax-h3", model)
    assert attr == "blocks" and [i for i, _ in blocks] == [0, 1]
    sampling = get_family("minimax-h3").solve_sampling
    assert sampling["size"] < get_family("minimax-h3-image").solve_sampling["size"]
    assert set(sampling) == {"steps", "cfg", "size"}


def test_image_family_is_unchanged():
    spec = get_family("minimax-h3-image")
    assert spec.classes == ("MiniMaxH3ImageModel",)
    assert spec.layouts == (("*", TokenLayout("image2d", (2, 2), text="prefix")),)
    assert spec.markers == ("blocks.*.attn.qkv_proj",)
    assert dict(spec.solve_sampling) == {"steps": 6, "cfg": 1.0, "size": 1024}
    assert spec.router_seed == 130
    assert spec.image_token_rows(17, (5, 7)) == slice(11, 17)
    assert BLOCK_LISTS["minimax-h3-image"] == "blocks"
    seeds = [s.router_seed for s in FAMILIES.values() if s.router_seed]
    assert len(seeds) == len(set(seeds))
