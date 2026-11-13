# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from kilora_editor_chunk import ChunkEditor
from kilora_editor_chunk import comfy as C
from kilora_editor_chunk.build import draw_noise
from torch import nn

from kilora.editor_packs import scan_packs


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(4, 4), nn.Linear(4, 4)
        self.v_proj, self.output_proj = nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x, context):
        q, k, v = self.q_proj(x), self.k_proj(context), self.v_proj(context)
        weights = torch.softmax(q @ k.transpose(-1, -2) / 2.0, dim=-1)
        return self.output_proj(weights @ v)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(), Attention()
        self.mlp = nn.Module()
        self.mlp.layer1, self.mlp.layer2 = nn.Linear(4, 8), nn.Linear(8, 4)

    def forward(self, x, context):
        x = x + self.self_attn(x, x)
        x = x + self.cross_attn(x, context)
        return x + self.mlp.layer2(torch.tanh(self.mlp.layer1(x)))


class ToyAnima(nn.Module):
    kilora_family = "anima"

    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x, context):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, context.to(tokens.dtype))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class Sampling:
    sigmas = ()

    def sigma(self, timestep):
        return timestep

    def noise_scaling(self, sigma, noise, latent):
        return sigma.view(-1, 1, 1, 1) * noise + (1 - sigma.view(-1, 1, 1, 1)) * latent


class BaseModel(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.diffusion_model = denoiser
        self.latent_format = SimpleNamespace(
            latent_channels=4, spacial_downscale_ratio=8, latent_dimensions=2
        )
        self.model_sampling = Sampling()
        self.extra_conds = None

    def process_latent_in(self, latent):
        return latent

    def denoised(self, x, sigma, conditioning):
        context = conditioning[0][0]
        velocity = self.diffusion_model(x, context)
        return x - sigma.view(-1, 1, 1, 1) * velocity


class Patcher:
    def __init__(self, denoiser):
        self.model = BaseModel(denoiser)
        self.model_options = {}


class Clip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text):
        gen = torch.Generator().manual_seed(len(text) * 7919 + sum(map(ord, text)))
        return [[torch.randn(1, 3, 4, generator=gen), {}]]


class Interrupted(Exception):
    pass


@pytest.fixture
def comfy(monkeypatch):
    calls = {"sample": 0, "predict": 0}
    sample = ModuleType("comfy.sample")
    samplers = ModuleType("comfy.samplers")
    helpers = ModuleType("comfy.sampler_helpers")
    mm = ModuleType("comfy.model_management")

    def prepare_noise(latent, seed):
        return torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed))

    def run(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kw):
        calls["sample"] += 1
        base = model.model
        x = noise
        sigmas = torch.linspace(1.0, 0.0, 3)
        for s, t in zip(sigmas[:-1], sigmas[1:], strict=True):
            sigma = s.reshape(1)
            cond = base.denoised(x, sigma, positive)
            uncond = base.denoised(x, sigma, negative)
            x0 = uncond + cfg * (cond - uncond)
            x = x0 + t * (x - x0) / s
        return x

    def calc_cond_batch(model, conds, x, sigma, options):
        calls["predict"] += 1
        return [model.denoised(x, sigma, conds[0])]

    sample.prepare_noise, sample.sample = prepare_noise, run
    samplers.encode_model_conds = lambda extra, conds, x, device, name: conds
    samplers.calc_cond_batch = calc_cond_batch
    helpers.convert_cond = lambda conditioning: conditioning
    mm.load_models_gpu = lambda models, **kw: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    comfy = ModuleType("comfy")
    comfy.sample, comfy.samplers, comfy.sampler_helpers = sample, samplers, helpers
    comfy.model_management = mm
    for name, module in (
        ("comfy", comfy),
        ("comfy.sample", sample),
        ("comfy.samplers", samplers),
        ("comfy.sampler_helpers", helpers),
        ("comfy.model_management", mm),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


PROMPTS = {"_common": ["a cat"], "ink": ["an ink cat", "an ink dog"]}


def _build(model, tmp_path, prompts=PROMPTS, **kw):
    kw = {"draws": 1, "common": 2, "family": 1, "size": 16, "log": lambda _line: None, **kw}
    return C.build_pack(model, Clip(), prompts, tmp_path / "packs", cache=tmp_path / "cache", **kw)


def test_build_pack_host_events_match_the_command_line_and_load(comfy, tmp_path):
    ticks = []
    result = _build(Patcher(ToyAnima()), tmp_path, progress=ticks.append)
    assert result.folder == tmp_path / "packs" and result.folder.is_dir()
    assert not (tmp_path / "packs.tmp").exists()
    assert sum(ticks) == result.events == 3 and result.reused == 0
    assert result.estimate.startswith("Covariance estimates")
    events = sorted(result.events_folder.glob("*-*.pt"))
    assert len(events) == 3
    event = torch.load(events[0], weights_only=False)
    assert set(event) == {"family", "prompt", "seed", "draws", "collection", "g_empty", "g_full"}
    assert event["collection"] == {"host": True, "hooks": None, "columns": 4}
    assert all("attn" in name for name in event["g_full"])
    installed, problems = scan_packs(sorted(result.folder.glob("*.kedit-pack.safetensors")))
    assert not problems and {p.role for p in installed} == {"common", "family"}
    assert result.directions["common"] == 2 and result.directions["ink"] == 1
    editor = ChunkEditor.load_packs(list(result.packs.values()))
    assert editor.names == list(event["g_full"])
    assert (result.folder / "chunk.kedit.safetensors").exists()


def test_hooks_are_removed_after_the_build(comfy, tmp_path):
    model = Patcher(ToyAnima())
    _build(model, tmp_path)
    for module in model.model.diffusion_model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_cache_reuses_events_and_changes_with_prompt_or_model(comfy, tmp_path):
    model = Patcher(ToyAnima())
    first = _build(model, tmp_path)
    samples = comfy["sample"]
    second = _build(model, tmp_path)
    assert second.reused == 3 and comfy["sample"] == samples
    assert second.events_folder == first.events_folder
    changed = {**PROMPTS, "ink": ["an ink cat", "an ink bird"]}
    third = _build(model, tmp_path, prompts=changed)
    assert third.events_folder != first.events_folder and third.reused == 0
    assert comfy["sample"] > samples
    other = _build(Patcher(ToyAnima(seed=1)), tmp_path)
    assert other.events_folder not in (first.events_folder, third.events_folder)
    assert other.reused == 0


def test_interrupt_leaves_no_pack_folder(comfy, tmp_path):
    calls = []

    def interrupt():
        calls.append(1)
        if len(calls) > 6:
            raise Interrupted

    model = Patcher(ToyAnima())
    with pytest.raises(Interrupted):
        _build(model, tmp_path, interrupt=interrupt)
    assert not (tmp_path / "packs").exists() and not (tmp_path / "packs.tmp").exists()
    for module in model.model.diffusion_model.modules():
        assert not module._forward_hooks
    result = _build(model, tmp_path)
    assert result.reused >= 1 and (tmp_path / "packs").is_dir()


def test_a_second_build_replaces_the_pack_folder(comfy, tmp_path):
    model = Patcher(ToyAnima())
    _build(model, tmp_path)
    (tmp_path / "packs" / "stale.txt").write_text("old")
    result = _build(model, tmp_path)
    assert not (result.folder / "stale.txt").exists()
    assert not (tmp_path / "packs.tmp").exists() and not (tmp_path / "packs.old").exists()


def test_noise_draws_are_the_collectors_draws():
    gen = torch.Generator().manual_seed(5)
    sigma = torch.sigmoid(torch.randn((), generator=gen))
    noise = torch.randn((4, 2, 2), generator=gen)
    value, drawn = draw_noise(5, (4, 2, 2), "logit")
    assert torch.equal(value, sigma) and torch.equal(drawn, noise)
    gen = torch.Generator().manual_seed(5)
    step = torch.randint(50, 950, (1,), generator=gen)
    assert torch.equal(draw_noise(5, (1,), "step", 1000)[0], step)


def test_unsupported_family_is_named(comfy):
    model = Patcher(ToyAnima())
    with pytest.raises(ValueError, match="supported: z-image, anima, sd15, sdxl"):
        C.ComfyCollector(model, Clip(), base_family="flux")
