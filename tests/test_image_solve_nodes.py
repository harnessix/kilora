# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from test_directions_contrast import _samples, _settings
from test_prompt_solve_nodes import run as prompt_run
from test_prompt_solve_nodes import world
from torch.nn import functional as F

from kilora.comfyui import image_solve as I
from kilora.comfyui import prompt_solve as P
from kilora.library import contrast


def _guard_payload(by_scale, drifts=None):
    drifts = drifts or {scale: 0.0 for scale in I._GUARD_SCALES if scale > 0}
    return I._guard_result(by_scale, drifts)


_LOSSES = {0.0: 2.0, 0.1: 1.8, 0.2: 1.6, 0.3: 1.5, 0.5: 1.0, 0.75: 1.2, 1.0: 1.4}


def _match_tensors(saved, expected):
    used = [False] * len(expected)
    assert len(saved) == len(expected) and expected
    for tensor in saved:
        matched = False
        for index, candidate in enumerate(expected):
            if not used[index] and torch.allclose(tensor, candidate):
                used[index] = True
                matched = True
                break
        assert matched
    assert all(used)


def _assert_saved_weights(path, solved_up, solved_down, multiplier):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        saved_up = [handle.get_tensor(key) for key in keys if key.endswith("lora_up.weight")]
        saved_down = [handle.get_tensor(key) for key in keys if key.endswith("lora_down.weight")]
    expected_up = [(tensor * multiplier).T.contiguous() for tensor in solved_up.values()]
    expected_down = [tensor.contiguous() for tensor in solved_down.values()]
    _match_tensors(saved_up, expected_up)
    _match_tensors(saved_down, expected_down)


class VAE:
    def __init__(self):
        self.pixels = []

    def encode(self, image):
        self.pixels.append(image.clone())
        image = F.adaptive_avg_pool2d(image.movedim(-1, 1), (2, 2))
        return torch.cat([image, image[:, :1]], dim=1)


@pytest.fixture
def images_world(world, monkeypatch):
    world.vae = VAE()
    world.model.model.process_latent_in = lambda latent: latent * 2 + 1
    world.images = torch.linspace(0, 1, 3 * 8 * 12 * 3).reshape(3, 8, 12, 3)
    world.fitted = []

    def samples(patcher, vae, items, cue, settings, progress, *, clip=None):
        world.fitted.append(items)
        world.counts.trajectories += 1
        progress.update(len(items) * len(settings.seeds))
        return _samples(count=2)

    monkeypatch.setattr(I, "image_samples", samples)
    monkeypatch.setattr(
        I,
        "_guard",
        lambda *args: _guard_payload(_LOSSES),
    )
    return world


def run(world, **kwargs):
    return I.KiLoRASolveFromImages().solve(
        **{
            "model": world.model,
            "clip": world.clip,
            "vae": world.vae,
            "images": world.images,
            "cue": "ink",
            **world.kwargs,
            **kwargs,
        }
    )["result"]


@pytest.mark.parametrize("shape", [(10, 30), (30, 10), (23, 11)])
def test_resize_short_side_then_center_crop(images_world, shape):
    height, width = shape
    image = torch.arange(height * width * 3).reshape(height, width, 3).float() / 100
    latent = I._encode_image(images_world.model, images_world.vae, image, 16)
    resized = F.interpolate(
        image.movedim(-1, 0).unsqueeze(0),
        size=(round(height * 16 / min(shape)), round(width * 16 / min(shape))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top, left = (resized.shape[-2] - 16) // 2, (resized.shape[-1] - 16) // 2
    expected = resized[..., top : top + 16, left : left + 16].movedim(1, -1)
    assert images_world.vae.pixels[0].shape == (1, 16, 16, 3)
    torch.testing.assert_close(images_world.vae.pixels[0], expected)
    torch.testing.assert_close(latent, images_world.vae.encode(expected) * 2 + 1)


@pytest.mark.parametrize(
    "count, expected",
    [(3, [2]), (5, [4]), (10, [4, 9]), (11, [4, 9]), (8, [4, 7]), (13, [4, 9, 12])],
)
def test_holdout_indices(count, expected):
    assert I._holdout_indices(count) == expected


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_images(images_world, count):
    with pytest.raises(ValueError, match="at least 3 images"):
        run(images_world, images=images_world.images[:count])


@pytest.mark.parametrize("both", [False, True])
def test_exactly_one_image_source(images_world, both):
    with pytest.raises(ValueError, match="either images or folder"):
        run(
            images_world,
            images=images_world.images if both else None,
            folder="folder" if both else "",
        )


def test_caption_count_mismatch(images_world):
    with pytest.raises(ValueError, match="Caption count 2 does not match image count 3"):
        run(images_world, captions="one\ntwo")


def test_folder_captions_and_sorted_images(tmp_path):
    Image.new("RGB", (20, 10), "red").save(tmp_path / "b.png")
    Image.new("RGB", (10, 20), "blue").save(tmp_path / "a.jpg")
    (tmp_path / "b.txt").write_text("a red object\n", encoding="utf-8")
    items = I._load_items(None, "", str(tmp_path))
    assert [caption for _, caption in items] == ["", "a red object"]
    assert [tuple(image.shape) for image, _ in items] == [(20, 10, 3), (10, 20, 3)]


def test_failed_guard_blocks_publication_and_override_warns(images_world, monkeypatch):
    solved = []
    real_solve = contrast.solve

    def spy_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        solved.append(
            (
                {name: solution.up.detach().clone() for name, solution in result.mounts.items()},
                {name: solution.down.detach().clone() for name, solution in result.mounts.items()},
            )
        )
        return result

    monkeypatch.setattr(contrast, "solve", spy_solve)
    failed = _guard_payload({scale: 1.0 if scale == 0.0 else 2.0 for scale in I._GUARD_SCALES})
    monkeypatch.setattr(I, "_guard", lambda *args: failed)
    paths, chain, report = run(images_world)
    assert paths == "" and not chain
    assert "Scale 0: held-out loss 1\nScale 0.1: held-out loss 2, drift 0, not allowed" in report
    assert "No scale is within the drift bound, not saved" in report
    assert "saved for strength 0.7" not in report
    assert "ignore_guard saves anyway" in report
    assert not list(images_world.root.rglob("*.safetensors"))
    paths, chain, report = run(images_world, ignore_guard=True)
    assert Path(paths).is_file() and len(chain) == 1
    assert "Warning: guard failed" in report
    record = I._record(paths)["image_guard"]
    assert record["scale"] == 0.0
    assert record["multiplier"] == 1.0
    ups, downs = solved[1]
    _assert_saved_weights(paths, ups, downs, 1.0)
    paths, chain, report = run(images_world)
    assert paths == "" and not chain
    assert "No scale is within the drift bound, not saved" in report


def test_holdout_does_not_contribute_and_metadata_has_hashes(images_world):
    path, chain, report = run(images_world, captions="first\nsecond\nlast")
    assert len(chain) == 1
    assert [item.caption for item in images_world.fitted[0]] == ["first", "second"]
    assert len(images_world.vae.pixels) == 3
    record = I._record(path)["image_solve"]
    expected = I._prepare(
        images_world.model,
        images_world.vae,
        list(zip(images_world.images, ["first", "second", "last"], strict=True)),
        16,
    )
    assert record["image_hashes"] == [
        hashlib.sha256(
            item.latent.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for item in expected
    ]
    assert record["holdout_indices"] == [2]
    assert record["cue"] == "ink"
    for text in (
        "Images: 3",
        "Hold-out images: 1",
        "Times: trajectories",
        "Chose scale",
        f"Wrote {path}",
    ):
        assert text in report
    assert not list(images_world.temp.iterdir())


def test_reuse_and_changed_captions_use_hashed_name(images_world, monkeypatch):
    first, _, _ = run(images_world)
    before = Path(first).read_bytes()
    loads = images_world.counts.loads
    reused, _, report = run(images_world)
    assert reused == first and f"Reused {first}" in report
    assert images_world.counts.loads == loads
    assert "Times:" not in report
    changed, _, _ = run(images_world, captions="one\ntwo\nthree")
    record = I._record(changed)["image_solve"]
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    assert Path(changed) == Path(first).with_stem(f"{Path(first).stem}-{digest}")
    assert Path(first).read_bytes() == before
    monkeypatch.setattr(contrast, "solve", lambda *args, **kwargs: pytest.fail("cached solve ran"))
    assert run(images_world, captions="one\ntwo\nthree")[0] == changed


@pytest.mark.parametrize("fits", [False, True])
def test_memory_load_rule(images_world, monkeypatch, fits):
    monkeypatch.setattr(contrast, "memory_estimate", lambda *args, **kwargs: 1000)
    monkeypatch.setattr(images_world.mm, "get_free_memory", lambda device: 1000 if fits else 999)
    run(images_world)
    assert images_world.counts.load_options == [(1000, fits)]


@pytest.fixture
def sampler_world(world, monkeypatch):
    samplers, helpers, sample = (
        ModuleType(f"comfy.{name}") for name in ("samplers", "sampler_helpers", "sample")
    )
    comfy = sys.modules["comfy"]
    monkeypatch.setattr(comfy, "samplers", samplers, raising=False)
    monkeypatch.setattr(comfy, "sample", sample, raising=False)
    for name, module in (("samplers", samplers), ("sampler_helpers", helpers), ("sample", sample)):
        monkeypatch.setitem(sys.modules, f"comfy.{name}", module)
    helpers.convert_cond = lambda cond: cond
    samplers.encode_model_conds = lambda extra, cond, *args: cond
    sample.prepare_noise = lambda latent, seed: torch.randn(
        latent.shape, generator=torch.Generator().manual_seed(seed)
    )
    samplers.calculate_sigmas = lambda sampling, scheduler, steps: torch.arange(
        steps, -1, -1
    ).float()
    samplers.sampler_object = lambda name: name
    world.model.model.extra_conds = lambda **kwargs: {}
    world.model.model.latent_format = SimpleNamespace(
        latent_channels=4,
        spacial_downscale_ratio=8,
        latent_dimensions=2,
    )
    world.model.model.process_latent_in = lambda x: x * 2
    world.model.model.model_sampling = SimpleNamespace(
        noise_scaling=lambda sigma, noise, z: z + sigma * noise
    )
    world.model.pre_run = lambda: None
    world.model.cleanup = lambda: None
    world.calls, world.sampler_calls, world.denoised = [], [], []
    world.packed, world.chunks = False, (1, 0)

    def calc(model, conds, x, timestep, options):
        text = conds[0][0][1]["text"]
        world.calls.append((x.clone(), timestep.clone(), text))
        hidden = x.reshape(x.shape[0], -1, 4)
        if world.packed:
            hidden = hidden.flatten(0, 1)
        output = model.diffusion_model(
            hidden_states=hidden,
            encoder_hidden_states=torch.ones_like(hidden) * len(text),
        )[0]
        denoised = output.reshape_as(x)
        world.denoised.append(denoised.detach().clone())
        return [denoised]

    samplers.calc_cond_batch = calc

    class Guider:
        def __init__(self, patcher):
            pass

        def set_conds(self, positive, negative):
            self.original_conds = {"positive": positive, "negative": negative}

        def set_cfg(self, cfg):
            pass

        def inner_sample(self, noise, latent, device, sampler, sigmas, *args):
            world.sampler_calls.append(noise.clone())
            wrapped = self.model_options["model_function_wrapper"]
            for sigma in sigmas[:-1]:
                conditional = noise + sigma
                x = torch.cat(
                    [conditional if chunk == 0 else conditional + 1000 for chunk in world.chunks]
                )
                arguments = {
                    "input": x,
                    "timestep": sigma.expand(len(x)),
                    "c": {},
                    "cond_or_uncond": list(world.chunks),
                }
                wrapped(lambda *args, **kwargs: noise, arguments)
                wrapped(lambda *args, **kwargs: noise, arguments)

    samplers.CFGGuider = Guider
    world.vae = VAE()
    world.progress = SimpleNamespace(update=lambda count: None)
    return world


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("chunks", [(0, 1), (1, 0), (0,)])
def test_matched_samples_same_prompt_time_and_seed_noise(sampler_world, packed, chunks):
    world = sampler_world
    world.packed, world.chunks = packed, chunks
    settings = replace(_settings(), seeds=(1, 7), size=16)
    items = [(torch.ones(8, 12, 3), "a tree"), (torch.ones(12, 8, 3) * 0.5, "")]
    samples = I.image_samples(
        world.model, world.vae, items, "ink", settings, world.progress, clip=world.clip
    )
    assert len(world.sampler_calls) == 2
    assert len(samples) == 2 * len(settings.seeds) * len(settings.sample_steps)
    prepared = I._prepare(world.model, world.vae, items, 16)
    index = 0
    for item_index, item in enumerate(prepared):
        expected_prompt = "a tree. ink." if item_index == 0 else "ink"
        for seed_index, seed in enumerate(settings.seeds):
            noise = torch.randn(item.latent.shape, generator=torch.Generator().manual_seed(seed))
            torch.testing.assert_close(
                world.sampler_calls[item_index][seed_index : seed_index + 1], noise
            )
            for step in settings.sample_steps:
                sample = samples[index]
                index += 1
                sigma = settings.steps - step
                shape = sample.target["hidden_states"].shape
                torch.testing.assert_close(
                    sample.target["hidden_states"], (item.latent + sigma * noise).reshape(shape)
                )
                torch.testing.assert_close(
                    sample.student["hidden_states"], (noise + sigma).reshape(shape)
                )
                torch.testing.assert_close(
                    sample.target["encoder_hidden_states"], sample.student["encoder_hidden_states"]
                )
        assert expected_prompt in world.counts.encoded
    for target_call, student_call in zip(world.calls[::2], world.calls[1::2], strict=True):
        torch.testing.assert_close(target_call[1], student_call[1])
        assert target_call[2] == student_call[2]
    assert all(float(t[0]) in (1.0, 2.0) for _, t, _ in world.calls)
    assert all(text in ("a tree. ink.", "ink") for _, _, text in world.calls)
    assert not world.model.model.diffusion_model._forward_pre_hooks


def test_guard_measures_model_denoised_output_and_cleans_hooks(sampler_world, monkeypatch):
    world = sampler_world
    settings = replace(_settings(), seeds=(1, 7), size=16)
    adapter = P.SolveAdapter(world.model)
    result = contrast.solve(
        world.model.model.diffusion_model,
        _samples(2),
        [("same", "same")],
        settings=settings,
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    seen = []
    real_changes = P._AdapterChanges

    class Recording(real_changes):
        def __init__(self, adapter):
            super().__init__(adapter)
            seen.append(self)

    monkeypatch.setattr(P, "_AdapterChanges", Recording)
    import comfy.samplers

    original_calc = comfy.samplers.calc_cond_batch
    world.scales = []
    plain_inputs = []

    def tracing(model, conds, x, timestep, options):
        output = original_calc(model, conds, x, timestep, options)
        scale = seen[-1].scale
        world.scales.append(scale)
        if scale == 0.0:
            plain_inputs.append(
                (conds, x.detach().clone(), timestep.detach().clone(), dict(options))
            )
        return output

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", tracing)
    from kilora import recording
    from kilora.comfyui import image_learn

    drift_calls = []

    def fake_drift(patcher, clip, settings_, trial):
        changes = seen[-1]
        assert not changes.enabled
        with recording.perturbing(trial):
            assert changes.enabled
        assert (patcher, clip, settings_) == (world.model, world.clip, settings)
        drift_calls.append(changes.scale)
        return 0.01 * changes.scale

    monkeypatch.setattr(image_learn, "unrelated_drift", fake_drift)
    items = I._prepare(world.model, world.vae, [(torch.ones(8, 12, 3), "tree")], 16)
    losses = I._guard(
        world.model, world.clip, items, "ink", settings, adapter, result, lambda: None
    )
    assert drift_calls == [scale for scale in I._GUARD_SCALES if scale > 0]
    assert [item["drift"] for item in losses["losses"]] == pytest.approx(
        [0.01 * scale for scale in I._GUARD_SCALES]
    )
    width = len(I._GUARD_SCALES)
    groups = len(settings.seeds) * len(settings.sample_steps)
    assert world.scales == list(I._GUARD_SCALES) * groups
    assert len(world.calls) == width * groups
    assert [item["scale"] for item in losses["losses"]] == list(I._GUARD_SCALES)
    assert all(math.isfinite(item["loss"]) for item in losses["losses"])
    z = items[0].latent
    means = []
    for offset, scale in enumerate(I._GUARD_SCALES):
        expected = [
            float(F.mse_loss(value.float(), z.float())) for value in world.denoised[offset::width]
        ]
        mean = sum(expected) / len(expected)
        means.append(mean)
        reported = next(item["loss"] for item in losses["losses"] if item["scale"] == scale)
        assert reported == pytest.approx(mean)
    scales = list(I._GUARD_SCALES)
    helping = [scale for scale in scales[1:] if means[scales.index(scale)] < means[0]]
    best = min(helping, key=lambda scale: (means[scales.index(scale)], scale), default=0.0)
    assert losses["scale"] == best
    base = means[0]
    chosen = scales.index(best)
    gain = (base - means[chosen]) / max(base, 1e-30) if helping else 0.0
    assert losses["relative_gain"] == pytest.approx(gain)
    for group in range(groups):
        block = world.calls[group * width : (group + 1) * width]
        for call in block:
            torch.testing.assert_close(call[0], block[0][0])
            torch.testing.assert_close(call[1], block[0][1])
            assert call[2] == block[0][2] == "tree. ink."
    scale0 = [tensor.clone() for tensor in world.denoised[::width]]
    denoiser = world.model.model.diffusion_model
    was_training = denoiser.training
    denoiser.eval()
    try:
        with torch.inference_mode():
            for saved, (conds, x, sigma, options) in zip(scale0, plain_inputs, strict=True):
                again = original_calc(world.model.model, conds, x, sigma, options)[0]
                torch.testing.assert_close(again, saved)
    finally:
        denoiser.training = was_training
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in world.model.model.diffusion_model.modules()
    )


def test_contrast_mode_mean_for_images_tokens_for_prompts(world, monkeypatch):
    real = contrast.solve
    modes = []

    def record(*args, **kwargs):
        modes.append(kwargs["settings"].contrast_mode)
        return real(*args, **kwargs)

    monkeypatch.setattr(contrast, "solve", record)
    prompt_run(world, mode="joint")
    assert modes == ["tokens"]
    world.vae = VAE()
    world.model.model.process_latent_in = lambda latent: latent * 2 + 1
    world.images = torch.linspace(0, 1, 3 * 8 * 12 * 3).reshape(3, 8, 12, 3)
    monkeypatch.setattr(I, "image_samples", lambda *args, **kwargs: _samples(count=2))
    monkeypatch.setattr(
        I,
        "_guard",
        lambda *args: _guard_payload(_LOSSES),
    )
    path, _, _ = run(world)
    assert modes == ["tokens", "mean"]
    assert I._record(path)["contrast_mode"] == "mean"


def test_best_scale_rescales_saved_up_weights(images_world, monkeypatch):
    solved = {}
    real_solve = contrast.solve

    def spy_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        solved["up"] = {
            name: solution.up.detach().clone() for name, solution in result.mounts.items()
        }
        solved["down"] = {
            name: solution.down.detach().clone() for name, solution in result.mounts.items()
        }
        return result

    monkeypatch.setattr(contrast, "solve", spy_solve)
    by_scale = {0.0: 0.2, 0.1: 0.15, 0.2: 0.12, 0.3: 0.1, 0.5: 0.05, 0.75: 0.08, 1.0: 0.12}
    monkeypatch.setattr(I, "_guard", lambda *args: _guard_payload(by_scale))
    path, chain, report = run(images_world)
    assert len(chain) == 1
    assert "Scale 0.5: held-out loss 0.05, drift 0, allowed" in report
    assert "Chose scale 0.5, saved for strength 0.7" in report
    record = I._record(path)["image_guard"]
    assert record["scale"] == 0.5
    assert record["multiplier"] == pytest.approx(0.5 / 0.7)
    assert [item["scale"] for item in record["losses"]] == list(I._GUARD_SCALES)
    assert [item["loss"] for item in record["losses"]] == pytest.approx(
        [by_scale[scale] for scale in I._GUARD_SCALES]
    )
    _assert_saved_weights(path, solved["up"], solved["down"], 0.5 / 0.7)


def _spy_solve(monkeypatch):
    solved = {}
    real_solve = contrast.solve

    def spy_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        solved["up"] = {
            name: solution.up.detach().clone() for name, solution in result.mounts.items()
        }
        solved["down"] = {
            name: solution.down.detach().clone() for name, solution in result.mounts.items()
        }
        return result

    monkeypatch.setattr(contrast, "solve", spy_solve)
    return solved


def test_drift_bound_limits_the_chosen_scale(images_world, monkeypatch):
    solved = _spy_solve(monkeypatch)
    losses = {0.0: 0.1, 0.1: 0.095, 0.2: 0.09, 0.3: 0.081, 0.5: 0.07, 0.75: 0.063, 1.0: 0.069}
    drifts = {0.1: 0.04, 0.2: 0.08, 0.3: 0.12, 0.5: 0.2, 0.75: 0.3, 1.0: 0.4}
    monkeypatch.setattr(I, "_guard", lambda *args: _guard_payload(losses, drifts))
    path, chain, report = run(images_world)
    assert len(chain) == 1
    assert "Scale 0.3: held-out loss 0.081, drift 0.12, allowed" in report
    assert "Scale 0.75: held-out loss 0.063, drift 0.3, not allowed" in report
    assert "Chose scale 0.3, saved for strength 0.7" in report
    record = I._record(path)["image_guard"]
    assert record["scale"] == 0.3
    assert record["multiplier"] == pytest.approx(0.3 / 0.7)
    assert record["drift_bound"] == pytest.approx(0.15)
    assert [item["drift"] for item in record["losses"]] == pytest.approx(
        [0.0, *(drifts[scale] for scale in I._GUARD_SCALES[1:])]
    )
    assert [item["allowed"] for item in record["losses"]] == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    _assert_saved_weights(path, solved["up"], solved["down"], 0.3 / 0.7)


def test_tie_goes_to_the_smaller_scale():
    losses = {0.0: 0.1, 0.1: 0.09, 0.2: 0.08, 0.3: 0.08, 0.5: 0.2, 0.75: 0.2, 1.0: 0.2}
    guard = _guard_payload(losses)
    assert guard["scale"] == 0.2
    assert guard["relative_gain"] == pytest.approx(0.2)


def test_every_helping_scale_above_drift_bound_saves_nothing(images_world, monkeypatch):
    drifts = {scale: 0.16 for scale in I._GUARD_SCALES if scale > 0}
    monkeypatch.setattr(I, "_guard", lambda *args: _guard_payload(_LOSSES, drifts))
    paths, chain, report = run(images_world)
    assert paths == "" and not chain
    assert "Scale 0.5: held-out loss 1, drift 0.16, not allowed" in report
    assert "No scale is within the drift bound, not saved" in report
    assert not list(images_world.root.rglob("*.safetensors"))
    assert not I._guard_passed(_guard_payload(_LOSSES, drifts))
