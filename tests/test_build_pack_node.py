# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from live_helpers import build_live_mod
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora import editors
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.errors import KiLoRAError
from kilora.live import SESSIONS
from kilora.testing import TINY_FAMILY

comfy_packs = pytest.importorskip("kilora_editor_chunk.comfy")


@pytest.fixture(autouse=True)
def fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


class Sampling:
    sigmas = ()

    def sigma(self, timestep):
        return timestep

    def noise_scaling(self, sigma, noise, latent):
        return sigma.view(-1, 1, 1, 1) * noise + (1 - sigma.view(-1, 1, 1, 1)) * latent


class Base(FakeBaseModel):
    def __init__(self, denoiser):
        super().__init__(denoiser)
        self.latent_format = SimpleNamespace(latent_channels=4, spacial_downscale_ratio=8)
        self.model_sampling = Sampling()
        self.extra_conds = None

    def process_latent_in(self, latent):
        return latent

    def denoised(self, x, sigma, conditioning):
        b, c, h, w = x.shape
        tokens = self.diffusion_model(x.flatten(2).transpose(1, 2))
        velocity = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x - sigma.view(-1, 1, 1, 1) * velocity


class Clip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text):
        return [[torch.zeros(1, 1, 4), {"text": text}]]


@pytest.fixture
def world(denoiser, folders, tmp_path, monkeypatch):
    monkeypatch.setitem(comfy_packs.FAMILY_DEFAULTS, TINY_FAMILY, (16, 2, 1.0))
    monkeypatch.setitem(comfy_packs.DRAW_RULES, TINY_FAMILY, "uniform")
    monkeypatch.setattr(N, "live_env", lambda: patching.LiveEnv(str(tmp_path / "saves")))

    def listing(folder, extension=None):
        root = folders[folder]
        names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        return [n for n in names if extension is None or n.endswith(extension)]

    monkeypatch.setattr(N.FOLDERS, "list", listing)
    state = {"interrupt_after": None, "checks": 0, "progress": []}
    sample, samplers = ModuleType("comfy.sample"), ModuleType("comfy.samplers")
    helpers, mm = ModuleType("comfy.sampler_helpers"), ModuleType("comfy.model_management")
    utils = ModuleType("comfy.utils")

    def run(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kw):
        x, sigmas = noise, torch.linspace(1.0, 0.0, steps + 1)
        for s, t in zip(sigmas[:-1], sigmas[1:], strict=True):
            x0 = model.model.denoised(x, s.reshape(1), positive)
            x = x0 + t * (x - x0) / s
        return x

    def check():
        state["checks"] += 1
        limit = state["interrupt_after"]
        if limit is not None and state["checks"] > limit:
            raise InterruptedError("stopped")

    class ProgressBar:
        def __init__(self, total):
            state["progress"].append([total, 0])

        def update(self, count):
            state["progress"][-1][1] += count

    sample.prepare_noise = lambda latent, seed: torch.randn(
        latent.shape, generator=torch.Generator().manual_seed(seed)
    )
    sample.sample = run
    samplers.encode_model_conds = lambda extra, conds, x, device, name: conds
    samplers.calc_cond_batch = lambda model, conds, x, sigma, options: [
        model.denoised(x, sigma, conds[0])
    ]
    helpers.convert_cond = lambda conditioning: conditioning
    mm.load_models_gpu = lambda models, **kw: None
    mm.throw_exception_if_processing_interrupted = check
    utils.ProgressBar = ProgressBar
    comfy = ModuleType("comfy")
    comfy.sample, comfy.samplers, comfy.sampler_helpers = sample, samplers, helpers
    comfy.model_management, comfy.utils = mm, utils
    for name, module in (
        ("comfy", comfy),
        ("comfy.sample", sample),
        ("comfy.samplers", samplers),
        ("comfy.sampler_helpers", helpers),
        ("comfy.model_management", mm),
        ("comfy.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod", family="style")
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(Base(denoiser)), chain, 1)
    return SimpleNamespace(model=patched, folders=folders, state=state)


PROMPTS = '{"_common": ["a cat", "a dog"], "style": ["an ink cat"]}'


def _run(world, name="my-packs", prompts=PROMPTS, **kw):
    node = N.KiLoRABuildEditorPack()
    kw = {"draws": 1, "common": 2, "family": 1, **kw}
    return node.build(world.model, Clip(), name, prompts, "auto", **kw)["result"]


def test_node_builds_packs_that_live_packs_turns_on(world):
    path, report = _run(world)
    folder = world.folders["kilora"] / "my-packs"
    assert path == str(folder) and folder.is_dir()
    assert not (world.folders["kilora"] / "my-packs.tmp").exists()
    assert world.state["progress"] == [[3, 3]]
    assert "Wrote 2 packs to models/kilora/my-packs" in report
    assert "common 2" in report and "style 1" in report
    assert "Events: 3 (0 from the cache)" in report
    assert "finds 2 packs" in report
    events = sorted((world.folders["kilora"] / ".cache" / "editor-events").glob("*/*-*.pt"))
    event = torch.load(events[0], weights_only=False)
    assert set(event) == {"family", "prompt", "seed", "draws", "g_empty", "g_full", "keys"}
    (session,) = patching.sessions_of(world.model)
    assert list(event["g_full"]) == list(session.state)
    _, choice = N.KiLoRALivePacks().choose(world.model, "ink", "auto", "")
    assert [p.id for p in session.editor_packs][0] == "chunk-common"
    assert len(session.editor_packs) == 2 and "[on]" in choice


def test_second_run_reuses_events(world):
    _run(world)
    _, report = _run(world)
    assert "Events: 3 (3 from the cache)" in report
    assert world.state["progress"][-1] == [3, 3]
    _, report = _run(world, prompts='{"_common": ["a cat", "a bird"], "style": ["an ink cat"]}')
    assert "(0 from the cache)" in report


def test_interrupt_leaves_no_folder(world):
    world.state["interrupt_after"] = 4
    with pytest.raises(InterruptedError):
        _run(world, name="stopped")
    assert not (world.folders["kilora"] / "stopped").exists()
    assert not (world.folders["kilora"] / "stopped.tmp").exists()


def test_bad_inputs_are_plain_errors(world):
    with pytest.raises(KiLoRAError, match="JSON object"):
        _run(world, prompts="{")
    with pytest.raises(KiLoRAError, match="JSON object"):
        _run(world, prompts='{"style": []}')
    with pytest.raises(KiLoRAError, match="Set a name"):
        _run(world, name="///")


def test_missing_plugin_is_named(world, monkeypatch):
    monkeypatch.setattr(editors, "plugin_for_kind", lambda kind: None)
    with pytest.raises(KiLoRAError, match="kilora-editor-chunk"):
        _run(world)


def test_node_is_registered_last():
    assert list(N.NODE_CLASS_MAPPINGS)[-1] == "KiLoRABuildEditorPack"
    assert N.NODE_DISPLAY_NAME_MAPPINGS["KiLoRABuildEditorPack"] == "KiLoRA Live: Build Editor Pack"
