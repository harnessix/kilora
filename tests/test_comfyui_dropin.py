# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora.comfyui import NODE_CLASS_MAPPINGS, dropin, patching
from kilora.comfyui import nodes as N
from kilora.comfyui.patching import applied


class WrappedPatcher(FakePatcher):
    def __init__(self, model, object_patches=None, model_options=None, patches=()):
        super().__init__(model, object_patches, model_options)
        self.patches = list(patches)

    def clone(self):
        n = WrappedPatcher(self.model, self.object_patches, self.model_options, self.patches)
        wrappers = self.model_options.get("wrappers", {})
        n.model_options["wrappers"] = {t: dict(d) for t, d in wrappers.items()}
        return n

    def add_wrapper_with_key(self, kind, key, fn):
        self.model_options.setdefault("wrappers", {}).setdefault(kind, {}).setdefault(
            key, []
        ).append(fn)

    def remove_wrappers_with_key(self, kind, key):
        self.model_options.get("wrappers", {}).get(kind, {}).pop(key, None)

    def wrappers(self, kind):
        return [
            f for fs in self.model_options.get("wrappers", {}).get(kind, {}).values() for f in fs
        ]


class FakeClip:
    def __init__(self, patches=()):
        self.patches = list(patches)

    def clone(self):
        return FakeClip(self.patches)

    def tokenize(self, text):
        return {"t5xxl": [[(i + 1, 1.0) for i, _ in enumerate(text.split())]]}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 4), {}]]


class FakeComfy:
    def load_lora(self, path):
        return load_file(path)

    def load_lora_for_models(self, model, clip, lora, strength_model, strength_clip):
        key = tuple(sorted(lora))
        new_model = None
        if model is not None:
            new_model = model.clone()
            new_model.patches.append((key, strength_model))
        new_clip = None
        if clip is not None:
            new_clip = clip.clone()
            new_clip.patches.append((key, strength_clip))
        return new_model, new_clip


class LoraLoader:
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        lora = dropin.COMFY.load_lora(N.FOLDERS.path("loras", lora_name))
        return dropin.COMFY.load_lora_for_models(model, clip, lora, strength_model, strength_clip)


class Executor:
    def __init__(self, guider, x):
        self.class_obj = guider
        self.x = x

    def __call__(self):
        return self.class_obj.model_patcher.run(self.x)


class Guider:
    def __init__(self, patcher):
        self.model_patcher = patcher


def sample(patcher, x):
    guider = Guider(patcher)
    (wrapper,) = patcher.wrappers(dropin.OUTER_SAMPLE)
    out = wrapper(Executor(guider, x))
    assert guider.model_patcher is patcher
    return out


@pytest.fixture
def comfy(monkeypatch):
    monkeypatch.setattr(dropin, "COMFY", FakeComfy())


@pytest.fixture
def te_path(monkeypatch):
    calls = []

    def apply_clip(model, clip, chain, env=None):
        (entry,) = chain
        calls.append((entry.path.rsplit("/", 1)[-1], entry.strength))
        return FakeClip([*clip.patches, (entry.path.rsplit("/", 1)[-1], entry.strength)]), ""

    monkeypatch.setattr(dropin.patching, "apply_clip", apply_clip)
    return calls


@pytest.fixture
def loras(folders):
    names = []
    for i, layer in enumerate([0, 1, 0]):
        gen = torch.Generator().manual_seed(10 + i)
        base = f"lora_unet_layers_{layer}_attention_to_{'qkv'[i]}"
        sd = {
            base + ".lora_down.weight": torch.randn(2, 16, generator=gen),
            base + ".lora_up.weight": torch.randn(16, 2, generator=gen),
            base + ".alpha": torch.tensor(1.0),
        }
        name = f"l{i}.safetensors"
        save_file(sd, str(folders["loras"] / name))
        names.append(name)
    return names


def _patcher(denoiser):
    return WrappedPatcher(FakeBaseModel(denoiser))


def test_node_is_registered_with_loraloader_sockets():
    cls = NODE_CLASS_MAPPINGS["KiLoRALoRALoader"]
    assert cls is dropin.KiLoRALoRALoader
    assert cls.RETURN_TYPES == ("MODEL", "CLIP")
    assert cls.RETURN_NAMES == ("model", "clip")
    assert cls.FUNCTION == "load_lora"


def test_input_names(folders):
    types = dropin.KiLoRALoRALoader.INPUT_TYPES()
    assert list(types["required"]) == [
        "model",
        "clip",
        "lora_name",
        "strength_model",
        "strength_clip",
    ]
    assert list(types["optional"]) == [
        "lora_stack",
        "timeline",
        "settings",
        "depth_map",
        "axis_alpha",
        "alpha_mask",
        "expert_text",
    ]
    assert types["optional"]["lora_stack"][0] == "LORA_STACK"
    assert types["optional"]["settings"][0] == "KILORA_SETTINGS"


def test_chain_of_three_gives_one_routed_composition(denoiser, folders, loras, comfy):
    node = dropin.KiLoRALoRALoader()
    model, clip = _patcher(denoiser), FakeClip()
    strengths = [0.8, 0.5, -0.3]
    for name, s in zip(loras, strengths, strict=True):
        model, clip = node.load_lora(model, clip, name, s, 1.0)
    assert model.patches == []
    state = dropin.state_of(model)
    assert [e.strength for e in state.entries] == strengths
    assert state.builds == 0
    assert len(model.wrappers(dropin.OUTER_SAMPLE)) == 1

    x = torch.randn(1, 5, 4)
    got = sample(model, x)
    assert state.builds == 1
    routed = state.cache[1]
    assert len(applied(routed)["chain"]) == 3
    assert dropin.STATE_KEY not in routed.model_options

    chain = None
    for name, s in zip(loras, strengths, strict=True):
        (chain,) = N.KiLoRAImportLoRA().add(name, s, sets=chain)
    want, report = patching.apply(
        FakePatcher(FakeBaseModel(denoiser)),
        chain,
        -1,
        patching.LiveEnv(),
        joint=True,
        depth_steps=N.COMFY_DEPTH_STEPS,
    )
    assert report == state.cache[2]
    assert "routed-loras" in report and "3 experts per target" in report
    torch.testing.assert_close(got, want.run(x))

    torch.testing.assert_close(sample(model, x), got)
    assert state.builds == 1


def test_lora_stack_and_chained_loaders_combine(denoiser, folders, loras, comfy, te_path):
    node = dropin.KiLoRALoRALoader()
    stack = [(loras[0], 0.8, 1.0), ("None", 1.0, 1.0), (loras[1], 0.5, 0.2)]
    model, clip = node.load_lora(_patcher(denoiser), FakeClip(), loras[2], -0.3, 1.0, stack)
    model, clip = node.load_lora(model, clip, loras[0], 0.4, 0.0)
    state = dropin.state_of(model)
    names = [e.path.rsplit("/", 1)[-1] for e in state.entries]
    assert names == [loras[0], loras[1], loras[2]]
    assert [e.strength for e in state.entries] == pytest.approx([1.2, 0.5, -0.3])
    assert [s for _, s in clip.patches] == [1.0, 0.2, 1.0]
    sample(model, torch.randn(1, 5, 4))
    assert len(applied(state.cache[1])["chain"]) == 3


def test_mix_stack_equals_loraloader(denoiser, folders, loras, comfy):
    node = dropin.KiLoRALoRALoader()
    base, base_clip = _patcher(denoiser), FakeClip()
    model, clip = base, base_clip
    ref_model, ref_clip = base, base_clip
    for name, s in zip(loras, [0.8, 0.5, -0.3], strict=True):
        model, clip = node.load_lora(model, clip, name, s, 0.7, settings={"mix": "stack"})
        ref_model, ref_clip = LoraLoader().load_lora(ref_model, ref_clip, name, s, 0.7)
    assert model.patches == ref_model.patches and len(model.patches) == 3
    assert clip.patches == ref_clip.patches
    assert dropin.state_of(model) is None
    assert model.wrappers(dropin.OUTER_SAMPLE) == []
    assert model.object_patches == ref_model.object_patches
    assert node.load_lora(base, base_clip, loras[0], 0.0, 0.0) == (base, base_clip)


def test_clip_goes_through_the_text_encoder_sections(denoiser, folders, loras, comfy, te_path):
    node = dropin.KiLoRALoRALoader()
    model, clip = _patcher(denoiser), FakeClip()
    for name, s in zip(loras, [0.6, 0.9, -0.4], strict=True):
        model, clip = node.load_lora(model, clip, name, 1.0, s)
    assert te_path == list(zip(loras, [0.6, 0.9, -0.4], strict=True)) == clip.patches
    m, c = node.load_lora(_patcher(denoiser), None, loras[0], 1.0, 1.0)
    assert c is None and len(dropin.state_of(m).entries) == 1


def test_clone_does_not_share_the_routed_cache(denoiser, folders, loras, comfy):
    import copy

    node = dropin.KiLoRALoRALoader()
    model, _ = node.load_lora(_patcher(denoiser), None, loras[0], 1.0, 1.0)
    sample(model, torch.randn(1, 5, 4))
    copied = copy.deepcopy(dropin.state_of(model))
    assert copied.cache is None and copied.entries == dropin.state_of(model).entries
