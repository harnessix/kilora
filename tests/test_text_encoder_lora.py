# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from safetensors.torch import save_file

from kilora import Settings, load_mod, save_mod
from kilora.lora_import import import_lora
from kilora.multi_import import import_loras
from kilora.sets import new_keyed_set
from kilora.testing import TinyDenoiser
from kilora.text_encoder_lora import KIND, apply_to_clip, clip_lora_state


def _te_lora(gen, n=3):
    sd = {}
    for enc in ("lora_te1", "lora_te2"):
        for i in range(n):
            stem = f"{enc}_text_model_encoder_layers_{i}_self_attn_q_proj"
            sd[f"{stem}.lora_down.weight"] = torch.randn(4, 8, generator=gen)
            sd[f"{stem}.lora_up.weight"] = torch.randn(8, 4, generator=gen)
            sd[f"{stem}.alpha"] = torch.tensor(2.0)
    stem = "lora_te2_text_model_encoder_layers_0_mlp_fc1"
    sd[f"{stem}.lokr_w1"] = torch.randn(2, 2, generator=gen)
    sd[f"{stem}.lokr_w2"] = torch.randn(4, 4, generator=gen)
    return sd


def _plain_deltas(sd):
    out = {}
    for key in [k for k in sd if k.endswith(".lora_down.weight")]:
        stem = key[: -len(".lora_down.weight")]
        down, up = sd[key], sd[f"{stem}.lora_up.weight"]
        scale = float(sd[f"{stem}.alpha"]) / down.shape[0] if f"{stem}.alpha" in sd else 1.0
        out[stem] = up @ down * scale
    for key in [k for k in sd if k.endswith(".lokr_w1")]:
        stem = key[: -len(".lokr_w1")]
        out[stem] = torch.kron(sd[key], sd[f"{stem}.lokr_w2"])
    return out


def _deltas(state):
    out = {}
    for key in state:
        if key.endswith(".lora_down.weight"):
            stem = key[: -len(".lora_down.weight")]
            scale = float(state[f"{stem}.alpha"]) / state[key].shape[0]
            out[stem] = state[f"{stem}.lora_up.weight"] @ state[key] * scale
        elif key.endswith(".diff"):
            out[key[: -len(".diff")]] = state[key]
    return out


def _unet_keys(model, gen):
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            sd[f"{name}.lora_A.weight"] = torch.randn(2, m.in_features, generator=gen)
            sd[f"{name}.lora_B.weight"] = torch.randn(m.out_features, 2, generator=gen)
            break
    return sd


def test_import_keeps_text_encoder_keys_as_a_section(tmp_path):
    gen = torch.Generator().manual_seed(0)
    model = TinyDenoiser(seed=0)
    te = _te_lora(gen)
    sd = {**_unet_keys(model, gen), **te}
    path = tmp_path / "style.safetensors"
    save_file(sd, str(path))
    result = import_lora(str(path), model, weight=0.5)
    (section,) = [s for s in result.sections if s.kind == KIND]
    assert section.meta["layers"] == 7 and result.skipped_keys == []
    assert "Kept 7 text encoder layers" in result.report()
    got = _deltas(clip_lora_state([section]))
    for stem, delta in _plain_deltas(te).items():
        torch.testing.assert_close(got[stem], 0.5 * delta, rtol=1e-5, atol=1e-5)
    multi = import_loras([str(path)], model)
    (section,) = [s for s in multi.sections if s.kind == KIND]
    assert section.meta["layers"] == 7
    assert multi.unmatched["style"] == []


def test_section_round_trip(tmp_path):
    gen = torch.Generator().manual_seed(1)
    model = TinyDenoiser(seed=0)
    path = tmp_path / "te.safetensors"
    save_file({**_unet_keys(model, gen), **_te_lora(gen)}, str(path))
    (section,) = [s for s in import_lora(str(path), model).sections if s.kind == KIND]
    s = new_keyed_set(
        "s",
        {"blocks.0.attn": (8, 8)},
        num_experts=2,
        routes=1,
        rank=1,
        router_dim=4,
        router_seed=0,
        seed=0,
    )
    out = tmp_path / "te.kmod"
    manifest = save_mod(
        out, [s], Settings(depth=1), id="te", base_model={"family": "sdxl"}, sections=[section]
    )
    assert "section.text-encoder" in manifest["requires_features"]
    (back,) = load_mod(out).sections
    assert back.kind == KIND and sorted(back.data) == sorted(section.data)
    for stem, parts in section.data.items():
        for part, value in parts.items():
            assert torch.equal(back.data[stem][part], value)


class FakeClip:
    def __init__(self, patches=None):
        self.patches = list(patches or [])

    def clone(self):
        return FakeClip(self.patches)

    def add_patches(self, patches, strength):
        self.patches.append((patches, strength))


class Mod:
    def __init__(self, sections, strength, label="m"):
        self.sections, self.strength, self.label = sections, strength, label


def test_apply_to_clip_uses_comfyui_loader_and_strength(tmp_path):
    gen = torch.Generator().manual_seed(2)
    model = TinyDenoiser(seed=0)
    path = tmp_path / "te.safetensors"
    save_file({**_unet_keys(model, gen), **_te_lora(gen)}, str(path))
    sections = import_lora(str(path), model).sections
    calls = []

    def load_lora(state, key_map):
        calls.append((sorted(state), key_map))
        return {f"clip.{k}": v for k, v in state.items() if not k.endswith(".alpha")}

    clip = FakeClip()
    out, lines = apply_to_clip(clip, [Mod(sections, 0.7)], key_map={"k": "v"}, load_lora=load_lora)
    assert out is not clip and clip.patches == []
    ((patches, strength),) = out.patches
    assert strength == 0.7 and calls[0][1] == {"k": "v"}
    assert len(lines) == 1


def test_no_text_encoder_keys_leave_the_clip_as_it_is():
    clip = FakeClip()
    out, lines = apply_to_clip(clip, [Mod([], 1.0)], key_map={}, load_lora=lambda s, k: {})
    assert out is clip and lines == []
    assert apply_to_clip(None, [Mod([], 1.0)])[0] is None


def test_apply_node_passes_the_clip(tmp_path, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N

    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(tmp_path / name))
    model = TinyDenoiser(seed=0).eval()
    model.kilora_family = "kilora-tiny-1"
    gen = torch.Generator().manual_seed(3)
    save_file(_unet_keys(model, gen), str(tmp_path / "plain.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("plain.safetensors", 1.0)
    clip = FakeClip()
    patched, report, clip_out = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(model)), chain, -1, clip=clip
    )
    assert clip_out is clip
    assert N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(model)), chain, -1)[2] is None
