# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .bindings import BINDINGS, Binding
from .family_spec import (
    DIFFUSION_PREFIXES,
    FamilySpec,
    LayerRows,
    Normalized,
    PatternBinding,
    TokenLayout,
    linear_shape,
    normalize_lora,
)
from .presets import PRESETS, Preset

__all__ = ["FAMILIES", "detect", "detect_names", "get_family", "normalize_for", "register"]

TEXT = TokenLayout("text1d")


def _indices(paths: Sequence[str], container: str) -> list[int]:
    found = {int(m[1]) for p in paths if (m := re.match(rf"{container}\.(\d+)\.", p))}
    return sorted(found)


def flux_names(paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]) -> dict[str, LayerRows]:
    out: dict[str, LayerRows] = {}
    width = next((shapes[p][0] for p in paths if p.endswith(".img_attn.proj")), None)
    if width is None:
        width = next((shapes[p][1] for p in paths if p.endswith(".linear2")), 0)
    h = int(width)
    for i in _indices(paths, "double_blocks"):
        src, dst = f"transformer_blocks.{i}", f"double_blocks.{i}"
        for j, proj in enumerate(("q", "k", "v")):
            out[f"{src}.attn.to_{proj}"] = (f"{dst}.img_attn.qkv", (j * h, h))
            out[f"{src}.attn.add_{proj}_proj"] = (f"{dst}.txt_attn.qkv", (j * h, h))
        for name, comfy in (
            ("attn.to_out.0", "img_attn.proj"),
            ("attn.to_add_out", "txt_attn.proj"),
            ("ff.net.0.proj", "img_mlp.0"),
            ("ff.net.2", "img_mlp.2"),
            ("ff.linear_in", "img_mlp.0"),
            ("ff.linear_out", "img_mlp.2"),
            ("ff_context.net.0.proj", "txt_mlp.0"),
            ("ff_context.net.2", "txt_mlp.2"),
            ("ff_context.linear_in", "txt_mlp.0"),
            ("ff_context.linear_out", "txt_mlp.2"),
            ("norm1.linear", "img_mod.lin"),
            ("norm1_context.linear", "txt_mod.lin"),
        ):
            out[f"{src}.{name}"] = (f"{dst}.{comfy}", None)
    for i in _indices(paths, "single_blocks"):
        src, dst = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
        fused = shapes.get(f"{dst}.linear1", (0, 4 * h))[1]
        for j, proj in enumerate(("q", "k", "v")):
            out[f"{src}.attn.to_{proj}"] = (f"{dst}.linear1", (j * h, h))
        out[f"{src}.proj_mlp"] = (f"{dst}.linear1", (3 * h, fused - 3 * h))
        out[f"{dst}.linear1_qkv"] = (f"{dst}.linear1", (0, 3 * h))
        out[f"{src}.attn.to_qkv_mlp_proj"] = (f"{dst}.linear1", None)
        out[f"{src}.proj_out"] = (f"{dst}.linear2", None)
        out[f"{src}.attn.to_out"] = (f"{dst}.linear2", None)
        out[f"{src}.norm.linear"] = (f"{dst}.modulation.lin", None)
    for name, comfy in (
        ("x_embedder", "img_in"),
        ("context_embedder", "txt_in"),
        ("proj_out", "final_layer.linear"),
        ("norm_out.linear", "final_layer.adaLN_modulation.1"),
    ):
        out[name] = (comfy, None)
    return out


def hunyuan_video_names(
    paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]
) -> dict[str, LayerRows]:
    out: dict[str, LayerRows] = {}
    for p in paths:
        name = (
            p.replace("_attn.qkv", "_attn_qkv")
            .replace("_attn.proj", "_attn_proj")
            .replace("mlp.0", "mlp.fc1")
            .replace("mlp.2", "mlp.fc2")
        )
        if name != p:
            out[name] = (p, None)
    return out


_WAN_DIFFUSERS = {
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.k_img": "attn2.add_k_proj",
    "cross_attn.v_img": "attn2.add_v_proj",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
}


def wan_names(paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]) -> dict[str, LayerRows]:
    out: dict[str, LayerRows] = {}
    for p in paths:
        m = re.fullmatch(r"blocks\.(\d+)\.(.+)", p)
        if m and m[2] in _WAN_DIFFUSERS:
            out[f"blocks.{m[1]}.{_WAN_DIFFUSERS[m[2]]}"] = (p, None)
    return out


def ace_step_names(
    paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]
) -> dict[str, LayerRows]:
    return {p.removeprefix("decoder."): (p, None) for p in paths if p.startswith("decoder.")}


def moss_image_names(
    paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]
) -> dict[str, LayerRows]:
    out: dict[str, LayerRows] = {}
    for p in paths:
        if p.endswith(".proj.qkv"):
            third = shapes[p][1] // 3
            for j, proj in enumerate("qkv"):
                out[f"{p[: -len('qkv')]}{proj}"] = (p, (j * third, third))
    return out


def fused_qkv_names(suffixes: tuple[str, str, str]):
    def names(paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]) -> dict[str, LayerRows]:
        out: dict[str, LayerRows] = {}
        for p in paths:
            if p.endswith(".qkv"):
                third = shapes[p][1] // 3
                for j, proj in enumerate(suffixes):
                    out[f"{p[: -len('qkv')]}{proj}"] = (p, (j * third, third))
        return out

    return names


def minimax_h3_image_names(
    paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]
) -> dict[str, LayerRows]:
    renamed = {p.removesuffix("_proj"): shapes[p] for p in paths if p.endswith(".qkv_proj")}
    out: dict[str, LayerRows] = {}
    for suffixes in (("to_q", "to_k", "to_v"), ("q_proj", "k_proj", "v_proj")):
        for name, (target, rows) in fused_qkv_names(suffixes)(list(renamed), renamed).items():
            out[name] = (target + "_proj", rows)
    for path in paths:
        if path.endswith(".mlp.fc1"):
            rows_per_part = shapes[path][1] // 2
            for i, name in enumerate(("gate_proj", "up_proj")):
                out[path.removesuffix("fc1") + name] = (path, (i * rows_per_part, rows_per_part))
    return out


_FLUX3_LAYERS = {
    "attn.to_q": "q_proj",
    "attn.to_k": "k_proj",
    "attn.to_v": "v_proj",
    "attn.to_out.0": "attn_out",
    "ff.linear_in": "mlp_in",
    "ff.linear_out": "mlp_out",
}


def flux3_names(
    paths: Sequence[str], shapes: Mapping[str, tuple[int, int]]
) -> dict[str, LayerRows]:
    out: dict[str, LayerRows] = fused_qkv_names(("q_proj", "k_proj", "v_proj"))(paths, shapes)
    blocks = {re.match(r"(.+?\.\d+)\.", p)[1] for p in paths if re.match(r"(.+?\.\d+)\.", p)}
    for block in blocks:
        diffusers = block.replace("content_mode_blocks.", "mode_blocks.", 1).replace(
            "single_blocks.", "single_transformer_blocks.", 1
        )
        fused = f"{block}.qkv" in shapes
        for name, comfy in _FLUX3_LAYERS.items():
            path = f"{block}.{comfy}"
            if path in shapes:
                out[f"{diffusers}.{name}"] = (path, None)
            elif fused and comfy in ("q_proj", "k_proj", "v_proj"):
                third = shapes[f"{block}.qkv"][1] // 3
                j = ("q_proj", "k_proj", "v_proj").index(comfy)
                out[f"{diffusers}.{name}"] = (f"{block}.qkv", (j * third, third))
    for stream in ("early_stream", "single_stream"):
        for mode in ("image", "image_cond", "txt"):
            out[f"{stream}_modulations.{mode}.linear"] = (f"{stream}_modulations.{mode}.lin", None)
    out["norm_out.linear"] = ("final_layer.image.adaLN_modulation.1", None)
    return out


_ANIMA_ADALN = {"self_attn": "norm1", "cross_attn": "norm2", "mlp": "norm3"}
_ANIMA_OTHER = {
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
    "t_embedder.1.linear_1": "time_embed.t_embedder.linear_1",
    "t_embedder.1.linear_2": "time_embed.t_embedder.linear_2",
    "x_embedder.proj.1": "patch_embed.proj",
}


def anima_never_target_diffusers(path: str) -> str | None:
    m = re.fullmatch(r"blocks\.(\d+)\.adaln_modulation_(self_attn|cross_attn|mlp)\.([12])", path)
    if m is not None:
        return f"transformer_blocks.{m[1]}.{_ANIMA_ADALN[m[2]]}.linear_{m[3]}"
    return _ANIMA_OTHER.get(path)


def anima_never_target_comfy(path: str) -> str | None:
    m = re.fullmatch(r"transformer_blocks\.(\d+)\.(norm[123])\.linear_([12])", path)
    if m is not None:
        attn = {v: k for k, v in _ANIMA_ADALN.items()}[m[2]]
        return f"blocks.{m[1]}.adaln_modulation_{attn}.{m[3]}"
    return {v: k for k, v in _ANIMA_OTHER.items()}.get(path)


def diffusers_tree(model: nn.Module, **parts: nn.Module) -> nn.Module:
    root = nn.Module()
    for name, child in model.named_children():
        root.add_module(name, child)
    for name, part in parts.items():
        root.add_module(name, part)
    return root


def _stream_layouts(image: TokenLayout) -> tuple[tuple[str, TokenLayout], ...]:
    joint = TokenLayout(image.kind, image.patch, text="before")
    return (
        ("double_blocks.*.txt_*", TEXT),
        ("double_blocks.*.img_*", image),
        ("single_blocks.*", joint),
    )


_DOUBLE_STREAM = tuple(
    f"double_blocks.*.{stream}_{layer}"
    for stream in ("img", "txt")
    for layer in ("attn.qkv", "attn.proj", "mlp.0", "mlp.2")
) + ("single_blocks.*.linear1", "single_blocks.*.linear2")

FAMILIES: dict[str, FamilySpec] = {
    "z-image": FamilySpec(
        family="z-image",
        naming="diffusers",
        image_token_target="layers.0.attention.to_q",
        layouts=(
            ("context_refiner.*", TEXT),
            ("noise_refiner.*", TokenLayout("image2d", (2, 2), pad=32)),
            ("layers.*", TokenLayout("image2d", (2, 2), text="before", pad=32)),
        ),
        router_seed=0,
    ),
    "sd15": FamilySpec(
        family="sd15",
        naming="diffusers",
        layouts=(
            ("*.attn2.to_k", TEXT),
            ("*.attn2.to_v", TEXT),
            ("*", TokenLayout("image2d")),
        ),
        router_seed=15,
    ),
    "sdxl": FamilySpec(
        family="sdxl",
        naming="diffusers",
        layouts=(
            ("*.attn2.to_k", TEXT),
            ("*.attn2.to_v", TEXT),
            ("*", TokenLayout("image2d")),
        ),
        router_seed=20,
    ),
    "anima": FamilySpec(
        family="anima",
        naming="diffusers",
        classes=("Anima",),
        layouts=(
            ("*.attn2.to_k", TEXT),
            ("*.attn2.to_v", TEXT),
            ("*", TokenLayout("video3d", (1, 2, 2))),
        ),
        router_seed=40,
        comfy_path=anima_never_target_comfy,
        diffusers_path=anima_never_target_diffusers,
    ),
    "flux": FamilySpec(
        family="flux",
        content_blocks=57,
        content_block_offsets=(("single_blocks", 19),),
        markers=("double_blocks.*.img_attn.qkv", "single_blocks.*.linear1", "txt_in"),
        classes=("Flux", "Chroma"),
        include=_DOUBLE_STREAM,
        exclude=(
            "*_mod.lin",
            "*.modulation.lin",
            "img_in",
            "txt_in",
            "final_layer.*",
            "distilled_guidance_layer.*",
        ),
        layouts=_stream_layouts(TokenLayout("image2d", (2, 2))),
        names=flux_names,
        router_seed=50,
        description="FLUX.1 and FLUX.2 (comfy.ldm.flux.model.Flux), Chroma (comfy.ldm.chroma)",
    ),
    "flux3": FamilySpec(
        family="flux3",
        content_blocks=60,
        markers=("txt_mode_blocks.*.mlp_in", "single_blocks.*.attn_out"),
        classes=("Flux3",),
        include=tuple(
            f"{blocks}.*.{p}"
            for blocks in ("content_mode_blocks", "txt_mode_blocks", "single_blocks")
            for p in ("qkv", "q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
        ),
        exclude=("emb_in.*", "txt_in", "time_in.*", "*modulations.*", "final_layer.*"),
        layouts=(
            ("txt_mode_blocks.*", TEXT),
            ("content_mode_blocks.image.*", TokenLayout("image2d", (1, 1))),
            ("content_mode_blocks.*", TokenLayout("image2d", (1, 1), role="reference")),
            ("single_blocks.*", TokenLayout("image2d", (1, 1), text="end")),
        ),
        names=flux3_names,
        router_seed=55,
        description="Flux.3 (comfy.ldm.flux3.model.Flux3)",
    ),
    "qwen-image": FamilySpec(
        family="qwen-image",
        content_blocks=60,
        markers=("transformer_blocks.*.attn.add_q_proj", "transformer_blocks.*.img_mlp.net.0.proj"),
        classes=("QwenImageTransformer2DModel",),
        include=tuple(
            f"transformer_blocks.*.attn.{p}"
            for p in (
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            )
        )
        + tuple(
            f"transformer_blocks.*.{s}_mlp.net.{p}" for s in ("img", "txt") for p in ("0.proj", "2")
        ),
        exclude=("*_mod.*", "img_in", "txt_in", "proj_out", "norm_out.*", "time_text_embed.*"),
        layouts=(
            ("*.attn.add_*", TEXT),
            ("*.attn.to_add_out", TEXT),
            ("*.txt_mlp.*", TEXT),
            ("*", TokenLayout("image2d", (2, 2))),
        ),
        router_seed=60,
        description="Qwen-Image (comfy.ldm.qwen_image): 60 MMDiT blocks, image and text streams",
    ),
    "moss-image": FamilySpec(
        family="moss-image",
        image_token_target="layers.0.proj.qkv",
        content_blocks=48,
        markers=("layers.*.proj.ffn_up_gate", "shared_block_modulation"),
        classes=("MOSSImage",),
        include=tuple(f"layers.*.proj.{p}" for p in ("q", "k", "v", "qkv", "o"))
        + ("layers.*.proj.ffn_up_gate", "layers.*.proj.ffn_down"),
        exclude=(
            "text_embedder.*",
            "video_embedder",
            "time_embedding.*",
            "shared_block_modulation",
            "final_linear_video",
            "video_head_modulation",
        ),
        layouts=(("*", TokenLayout("image2d", (1, 1), text="after")),),
        names=moss_image_names,
        router_seed=65,
        description="MOSS-Image (comfy.ldm.moss_image.model.MOSSImage), image tokens first",
    ),
    "minimax-h3-image": FamilySpec(
        family="minimax-h3-image",
        solve_sampling={"steps": 6, "cfg": 1.0, "size": 1024},
        image_token_target="blocks.0.attn.qkv_proj",
        content_blocks=50,
        classes=("MiniMaxH3ImageModel",),
        include=tuple(
            f"blocks.*.{layer}"
            for layer in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        ),
        exclude=(
            "blocks.*.adaln_proj.*",
            "blocks.*.attn.to_gate_compress",
            "token_*",
            "condition_proj",
            "video_patch_proj",
            "time_embedder.*",
            "final_layer.*",
        ),
        layouts=(("*", TokenLayout("image2d", (2, 2), text="prefix")),),
        names=minimax_h3_image_names,
        router_seed=130,
        description="MiniMax-H3-Image: packed text, references, then target image",
        markers=("blocks.*.attn.qkv_proj",),
        extra={"exclude_names": ("audio_*", "*.audio_*")},
    ),
    "wan": FamilySpec(
        family="wan",
        content_blocks=40,
        markers=("blocks.*.self_attn.q", "blocks.*.cross_attn.o", "blocks.*.ffn.0"),
        classes=("WanModel",),
        include=tuple(f"blocks.*.self_attn.{p}" for p in "qkvo")
        + tuple(f"blocks.*.cross_attn.{p}" for p in ("q", "k", "v", "o", "k_img", "v_img"))
        + ("blocks.*.ffn.0", "blocks.*.ffn.2"),
        exclude=(
            "text_embedding.*",
            "time_embedding.*",
            "time_projection.*",
            "head.*",
            "img_emb.*",
        ),
        layouts=(
            ("*.cross_attn.k", TEXT),
            ("*.cross_attn.v", TEXT),
            ("*.cross_attn.k_img", TEXT),
            ("*.cross_attn.v_img", TEXT),
            ("*", TokenLayout("video3d", (1, 2, 2))),
        ),
        names=wan_names,
        router_seed=70,
        description="Wan 2.1 / 2.2 (comfy.ldm.wan.model.WanModel): video DiT with cross-attention",
    ),
    "ltxv": FamilySpec(
        family="ltxv",
        content_blocks=48,
        markers=("transformer_blocks.*.attn1.to_q", "adaln_single.*", "patchify_proj"),
        classes=("LTXVModel",),
        include=tuple(
            f"transformer_blocks.*.{a}.{p}"
            for a in ("attn1", "attn2")
            for p in ("to_q", "to_k", "to_v", "to_out.0")
        )
        + ("transformer_blocks.*.ff.net.0.proj", "transformer_blocks.*.ff.net.2"),
        exclude=("patchify_proj", "adaln_single.*", "caption_projection.*", "proj_out"),
        layouts=(
            ("*.attn2.to_k", TEXT),
            ("*.attn2.to_v", TEXT),
            ("*", TokenLayout("video3d", (1, 1, 1))),
        ),
        router_seed=80,
        description="LTX-Video (comfy.ldm.lightricks.model.LTXVModel)",
    ),
    "hunyuan-video": FamilySpec(
        family="hunyuan-video",
        content_blocks=60,
        markers=(
            "double_blocks.*.img_attn.qkv",
            "single_blocks.*.linear1",
            "txt_in.input_embedder",
            "txt_in.individual_token_refiner.*",
        ),
        classes=("HunyuanVideo",),
        include=_DOUBLE_STREAM,
        exclude=("*_mod.lin", "*.modulation.lin", "img_in.*", "txt_in.*", "final_layer.*"),
        layouts=_stream_layouts(TokenLayout("video3d", (1, 2, 2))),
        names=hunyuan_video_names,
        router_seed=90,
        description="HunyuanVideo (comfy.ldm.hunyuan_video.model.HunyuanVideo)",
    ),
    "hunyuan3d": FamilySpec(
        family="hunyuan3d",
        content_blocks=40,
        markers=("double_blocks.*.img_attn.qkv", "single_blocks.*.linear1", "latent_in", "cond_in"),
        classes=("Hunyuan3Dv2",),
        include=_DOUBLE_STREAM,
        exclude=("*_mod.lin", "*.modulation.lin", "latent_in", "cond_in", "final_layer.*"),
        layouts=(("double_blocks.*.txt_*", TEXT), ("*", TokenLayout("set"))),
        names=flux_names,
        router_seed=110,
        description="Hunyuan3D 2.0 (comfy.ldm.hunyuan3d.model.Hunyuan3Dv2)",
    ),
    "ace-step15": FamilySpec(
        family="ace-step15",
        content_blocks=24,
        markers=("decoder.layers.*.self_attn.q_proj", "decoder.layers.*.cross_attn.q_proj"),
        classes=("AceStepConditionGenerationModel",),
        include=tuple(
            f"decoder.layers.*.{a}.{p}_proj" for a in ("self_attn", "cross_attn") for p in "qkvo"
        )
        + tuple(f"decoder.layers.*.mlp.{p}_proj" for p in ("gate", "up", "down")),
        exclude=("decoder.condition_embedder", "decoder.proj_in.*", "decoder.time_*"),
        layouts=(
            ("*.cross_attn.k_proj", TEXT),
            ("*.cross_attn.v_proj", TEXT),
            ("*", TokenLayout("audio1d", (2,))),
        ),
        names=ace_step_names,
        router_seed=120,
        description="ACE-Step 1.5 (comfy.ldm.ace.ace_step15.AceStepConditionGenerationModel)",
    ),
    "qwen-text": FamilySpec(
        family="qwen-text",
        markers=("layers.*.self_attn.q_proj", "layers.*.mlp.gate_proj"),
        root="text_encoder",
        classes=("Llama2_",),
        include=tuple(f"layers.*.self_attn.{p}_proj" for p in "qkvo")
        + tuple(f"layers.*.mlp.{p}_proj" for p in ("gate", "up", "down")),
        exclude=("embed_tokens", "lm_head"),
        layouts=(("*", TEXT),),
        router_seed=100,
        description="Qwen3 / Qwen2.5 text model of a text encoder",
    ),
}


def register(spec: FamilySpec) -> None:
    FAMILIES[spec.family] = spec
    if spec.naming != "comfy":
        return
    PRESETS[spec.family] = Preset(
        family=spec.family,
        denoiser=spec.root,
        target_modules=spec.include,
        full_target_modules=spec.include,
        router_seed=spec.router_seed,
        content_blocks=spec.content_blocks,
        content_block_offsets=spec.content_block_offsets,
    )
    BINDINGS[spec.family] = PatternBinding(spec)


for _spec in list(FAMILIES.values()):
    register(_spec)


def get_family(family: str) -> FamilySpec:
    try:
        return FAMILIES[family]
    except KeyError:
        raise KeyError(f"no base model family {family!r}; known: {sorted(FAMILIES)}") from None


def detect(module: nn.Module, root: str = "diffusion_model") -> FamilySpec | None:
    declared = getattr(module, "kilora_family", None)
    if declared in FAMILIES:
        return FAMILIES[declared]
    for spec in FAMILIES.values():
        if spec.root == root and spec.identifies(module):
            if spec.extra.get("require_class") and not spec.matches_names(
                [name for name, _ in module.named_modules()]
            ):
                continue
            return spec
    return None


def detect_names(paths: Sequence[str], root: str = "diffusion_model") -> FamilySpec | None:
    found = []
    for spec in FAMILIES.values():
        if spec.root != root or spec.extra.get("require_class"):
            continue
        names = paths
        if spec.family == "minimax-h3-image":
            names = []
            for path in paths:
                for prefix in DIFFUSION_PREFIXES:
                    if path.startswith(prefix):
                        path = path[len(prefix) :]
                        break
                names.append(path)
                match = re.match(r"(blocks\.\d+\.attn\.qkv_proj)(?:\.|$)", path)
                if match:
                    names.append(match[1])
        if spec.matches_names(names):
            found.append(spec)
    return max(found, key=lambda s: len(s.markers)) if found else None


def detect_binding(module: nn.Module) -> Binding | None:
    spec = detect(module)
    return BINDINGS.get(spec.family) if spec is not None else None


BLOCK_LISTS: dict[str, str] = {"anima": "blocks", "minimax-h3-image": "blocks"}


def block_list(family: str | None, model: nn.Module) -> tuple[str, list[tuple[int, nn.Module]]]:
    from .errors import KiLoRAError

    attr = BLOCK_LISTS.get(family or "")
    if attr is None:
        raise KiLoRAError(
            f"The block stream reads need the block list of the base model family, and "
            f"{family!r} has none. Supported families: {', '.join(sorted(BLOCK_LISTS))}."
        )
    blocks = getattr(model, attr, None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        raise KiLoRAError(
            f"The base model of family {family!r} has no block list {attr!r}, so the "
            "block stream reads are not recorded."
        )
    return attr, list(enumerate(blocks))


def normalize_for(
    family: str | None,
    state: Mapping[str, torch.Tensor],
    model: nn.Module,
    full_model: nn.Module | None = None,
) -> Normalized:
    spec = FAMILIES.get(family or "")
    if spec is None or spec.naming != "comfy":
        return Normalized(dict(state), [], {}, 0, {})
    shapes = {n: shape for n, m in model.named_modules() if n and (shape := linear_shape(m))}
    tree = full_model if full_model is not None else model
    other = [n for n, m in tree.named_modules() if n and n not in shapes and linear_shape(m)]
    return normalize_lora(spec, state, shapes, other)


VARIABLE_PATCH = frozenset({"flux"})


def layout_resolver(family: str | None, patch: int | None = None):
    spec = FAMILIES.get(family or "")
    if spec is None or not spec.layouts:
        return None
    if family not in VARIABLE_PATCH or not isinstance(patch, int) or patch < 1:
        return spec.layout
    import dataclasses

    def resolve(target: str):
        layout = spec.layout(target)
        if (
            layout is not None
            and layout.kind == "image2d"
            and layout.patch is not None
            and len(set(layout.patch)) == 1
            and layout.patch[0] != patch
        ):
            return dataclasses.replace(layout, patch=(patch, patch))
        return layout

    return resolve


DIFFUSERS_CLASSES = {
    "ZImageTransformer2DModel": "z-image",
    "CosmosTransformer3DModel": "anima",
}

CHECKPOINT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.", "model.", "")


def diffusers_family(model: nn.Module) -> str | None:
    name = type(model).__name__
    if name == "UNet2DConditionModel":
        config = getattr(model, "config", None)
        extra = getattr(config, "addition_embed_type", None) if config is not None else None
        return "sdxl" if extra == "text_time" else "sd15"
    return DIFFUSERS_CLASSES.get(name)


def shape_tree(shapes: Mapping[str, tuple[int, int]]) -> nn.Module:
    root = nn.Module()
    for name, (fan_in, fan_out) in shapes.items():
        parent = root
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        if not hasattr(parent, parts[-1]):
            parent.add_module(parts[-1], nn.Linear(fan_in, fan_out, device="meta"))
    return root


def checkpoint_tree(path: str, family: str | None = None) -> tuple[nn.Module, FamilySpec]:

    from safetensors import safe_open

    shapes_all: dict[str, tuple[int, int]] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".weight"):
                shape = f.get_slice(key).get_shape()
                if len(shape) == 2:
                    shapes_all[key[: -len(".weight")]] = (int(shape[1]), int(shape[0]))
    spec = FAMILIES.get(family) if family else None
    if family and spec is None:
        raise ValueError(f"no base model family {family!r}; known: {sorted(FAMILIES)}")
    if spec is not None and spec.naming != "comfy":
        raise ValueError(f"{family} mods use Diffusers module names: give a Diffusers model folder")
    best: tuple[int, dict, FamilySpec] | None = None
    for prefix in CHECKPOINT_PREFIXES:
        shapes = {k[len(prefix) :]: v for k, v in shapes_all.items() if k.startswith(prefix)}
        if not shapes:
            continue
        found = spec
        if found is None:
            found = detect_names(list(shapes)) or detect_names(list(shapes), "text_encoder")
        if found is None or found.naming != "comfy":
            continue
        count = sum(found.is_target(p) for p in shapes)
        if count and (best is None or count > best[0]):
            best = (count, shapes, found)
    if best is None:
        raise ValueError(
            f"{path}: no supported base model family matches the file"
            + (f" for --family {family}" if family else "; give --family")
        )
    return shape_tree(best[1]), best[2]
