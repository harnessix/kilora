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
from kilora.regions import token_weights


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
        return [replace(sample, image_index=i) for i, sample in enumerate(_samples(count=2))]

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
                assert sample.image_index == item_index
                index += 1
                sigma = settings.steps - step
                shape = sample.target["hidden_states"].shape
                torch.testing.assert_close(
                    sample.target["hidden_states"], (item.latent + sigma * noise).reshape(shape)
                )
                torch.testing.assert_close(
                    sample.student["hidden_states"], (noise + sigma).reshape(shape)
                )
                assert sample.weights is None
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
    monkeypatch.setattr(
        I,
        "image_samples",
        lambda *args, **kwargs: [
            replace(sample, image_index=i) for i, sample in enumerate(_samples(count=2))
        ],
    )
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


@pytest.mark.parametrize("components, expected", [(0, 0), (1, 1), (4, 1)])
def test_contrast_components_saved_reported_and_reused(images_world, components, expected):
    path, _, report = run(images_world, contrast_components=components)
    metadata = I._record(path)
    assert metadata["contrast_components"] == expected
    assert metadata["image_solve"]["settings"]["contrast_components"] == expected
    assert f"Contrast components: {expected}" in report
    assert metadata["fit"]
    ranks = {}
    for fit in metadata["fit"].values():
        rank = fit["measured_rank"]
        ranks[rank] = ranks.get(rank, 0) + 1
    summary = ", ".join(f"{rank} x{count}" for rank, count in sorted(ranks.items()))
    rank_lines = [line for line in report.splitlines() if line.startswith("Fitted rank per mount:")]
    assert rank_lines == [f"Fitted rank per mount: {summary}"]
    assert not any(line.startswith("Rank ") for line in report.splitlines())
    reused, _, reused_report = run(images_world, contrast_components=components)
    assert reused == path
    assert f"Reused {path}" in reused_report
    assert f"Contrast components: {expected}" in reused_report
    reused_rank_lines = [
        line for line in reused_report.splitlines() if line.startswith("Fitted rank per mount:")
    ]
    assert reused_rank_lines == rank_lines


def test_rank_report_groups_many_mounts(images_world, monkeypatch):
    real_solve = contrast.solve
    fits = {f"mount_{index}": {"measured_rank": 3 if index < 268 else 1} for index in range(280)}

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        result.metadata["fit"] = fits
        return result

    monkeypatch.setattr(contrast, "solve", solve)
    path, _, report = run(images_world)
    assert I._record(path)["fit"] == fits
    _, _, reused_report = run(images_world)
    for text in (report, reused_report):
        rank_lines = [
            line for line in text.splitlines() if line.startswith("Fitted rank per mount:")
        ]
        assert rank_lines == ["Fitted rank per mount: 1 x12, 3 x268"]
        assert not any(line.startswith("Rank ") for line in text.splitlines())


def test_contrast_components_changes_cache_identity(images_world):
    previous, _, _ = run(images_world, contrast_components=0)
    current, _, report = run(images_world)
    assert previous != current
    assert f"Kept {previous}" in report


@pytest.mark.parametrize("value", [-1, 5, 0.5, True])
def test_invalid_contrast_components(images_world, value):
    with pytest.raises(ValueError, match="contrast_components"):
        run(images_world, contrast_components=value)


def _install_recording_sampler(world):
    import comfy.samplers

    def inner_sample(self, noise, latent, device, sampler, sigmas, *args, **kwargs):
        scaling = world.model.model.model_sampling.noise_scaling
        world.starts.append(
            (
                noise.detach().cpu().clone(),
                latent.detach().cpu().clone(),
                sigmas.detach().cpu().clone(),
                scaling(sigmas[0], noise, latent).detach().cpu().clone(),
            )
        )
        wrapped = self.model_options["model_function_wrapper"]
        for sigma in sigmas[:-1]:
            x = scaling(sigma, noise, latent)
            arguments = {
                "input": x,
                "timestep": sigma.expand(x.shape[0]),
                "c": {},
                "cond_or_uncond": [0],
            }
            wrapped(lambda *args, **kwargs: noise, arguments)
            wrapped(lambda *args, **kwargs: noise, arguments)

    world.starts = []
    comfy.samplers.CFGGuider.inner_sample = inner_sample


def _partial_mask(height, width):
    mask = torch.zeros(height, width)
    mask[:, width // 2 :] = 1
    return mask


def test_masks_are_cropped_like_images_and_reduced_to_tokens(sampler_world):
    image = torch.arange(30 * 10 * 3).reshape(30, 10, 3).float() / 100
    mask = _partial_mask(30, 10)
    resampled = F.interpolate(
        mask[None, None],
        size=(12, 8),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    prepared = I._prepare(
        sampler_world.model,
        sampler_world.vae,
        [(image, "tree", mask), (image, "tree", resampled)],
        16,
    )
    height, width = image.shape[:2]
    scale = 16 / min(height, width)
    resized_hw = (max(16, round(height * scale)), max(16, round(width * scale)))
    for source, item in ((mask, prepared[0]), (resampled, prepared[1])):
        plane = I._mask_plane(source, height, width)
        cropped = I._crop_spatial(plane[None, None], 16)
        tokens = int(item.latent.shape[-2] * item.latent.shape[-1])
        torch.testing.assert_close(
            item.weights, token_weights(cropped, tokens).reshape(1, 1, *item.latent.shape[-2:])
        )
        resized = F.interpolate(
            plane[None, None],
            size=resized_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized.shape[-2] - 16) // 2
        left = (resized.shape[-1] - 16) // 2
        torch.testing.assert_close(cropped, resized[..., top : top + 16, left : left + 16])
    image_crop = I._crop_spatial(image.unsqueeze(0)[..., :3].movedim(-1, 1), 16)
    resized_image = F.interpolate(
        image.unsqueeze(0)[..., :3].movedim(-1, 1),
        size=resized_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_image.shape[-2] - 16) // 2
    left = (resized_image.shape[-1] - 16) // 2
    torch.testing.assert_close(image_crop, resized_image[..., top : top + 16, left : left + 16])
    mask_crop = I._crop_spatial(mask[None, None], 16)
    assert image_crop.shape[-2:] == mask_crop.shape[-2:]


def test_masked_student_starts_from_the_image_at_partial_noise(sampler_world):
    world = sampler_world
    _install_recording_sampler(world)
    steps = 15
    start, chosen = I._mask_schedule(steps)
    assert (start, chosen) == (6, (8, 11, 14))
    assert I._mask_schedule(20) == (8, (11, 15, 19))
    assert I._mask_schedule(5) == (2, (3, 4))
    assert all(step > start for step in chosen)
    settings = replace(_settings(), seeds=(1, 7), size=16, steps=steps, sample_steps=chosen)
    image = torch.ones(8, 12, 3)
    mask = _partial_mask(8, 12)
    items = [(image, "a tree", mask)]
    samples = I.image_samples(
        world.model, world.vae, items, "ink", settings, world.progress, clip=world.clip
    )
    prepared = I._prepare(world.model, world.vae, items, 16)
    item = prepared[0]
    noise = torch.cat(
        [
            torch.randn(item.latent.shape, generator=torch.Generator().manual_seed(seed))
            for seed in settings.seeds
        ]
    )
    full = torch.arange(steps, -1, -1).float()
    got_noise, got_latent, got_sigmas, got_start = world.starts[0]
    torch.testing.assert_close(got_noise, noise)
    torch.testing.assert_close(got_latent, item.latent.expand_as(noise))
    torch.testing.assert_close(got_sigmas, full[start:])
    torch.testing.assert_close(got_start, item.latent.expand_as(noise) + full[start] * noise)
    assert len(samples) == len(settings.seeds) * len(chosen)
    index = 0
    for seed_index, _seed in enumerate(settings.seeds):
        row_noise = noise[seed_index : seed_index + 1]
        for step in chosen:
            sample = samples[index]
            index += 1
            sigma = float(full[step])
            shape = sample.target["hidden_states"].shape
            expected = (item.latent + sigma * row_noise).reshape(shape)
            torch.testing.assert_close(sample.target["hidden_states"], expected)
            torch.testing.assert_close(sample.student["hidden_states"], expected)
            torch.testing.assert_close(sample.weights, item.weights)
    assert index == len(samples)


def test_mask_contrast_mean_matches_a_masked_mean(sampler_world):
    world = sampler_world
    _install_recording_sampler(world)
    steps = 5
    _start, chosen = I._mask_schedule(steps)
    settings = replace(
        _settings(),
        seeds=(1,),
        size=16,
        steps=steps,
        sample_steps=chosen,
        contrast_mode="mean",
    )
    image = torch.ones(8, 12, 3)
    items = [(image, "a tree", _partial_mask(8, 12))]
    samples = I.image_samples(
        world.model, world.vae, items, "ink", settings, world.progress, clip=world.clip
    )
    weights = samples[0].weights
    assert weights is not None and float(weights.sum()) > 0
    assert not torch.allclose(weights, torch.ones_like(weights))
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(samples[0].target["hidden_states"].shape, generator=generator)
    y = torch.randn(x.shape, generator=generator)
    target_y = y + torch.randn(x.shape, generator=generator)

    def observe(_target, _student):
        return {"mount": (x, target_y)}, {"mount": (x, y)}

    pair = contrast._WeightedPair(samples[0].target, samples[0].student, weights)
    gram, cross = contrast._group_statistics(["mount"], [pair], observe, lambda t: t, "mean")[
        "mount"
    ]
    scale = weights.reshape(-1, 1).to(dtype=x.dtype)
    flat = x.reshape(-1, x.shape[-1])
    rows = x.shape[-1]
    count = weights.sum()
    mean = ((target_y - y).reshape(-1, rows) * scale).sum(0) / count
    torch.testing.assert_close(gram, (flat * scale).T @ flat)
    torch.testing.assert_close(cross, torch.outer(mean, (flat * scale).sum(0)))
    plain = contrast._group_statistics(["mount"], [("t", "s")], observe, lambda t: t, "mean")
    assert not torch.allclose(gram, plain["mount"][0])


def test_masks_select_contrast_and_change_the_cache_record(images_world):
    right = torch.zeros(3, 8, 12)
    right[:, :, 4:] = 1
    top = torch.zeros(3, 8, 12)
    top[:, :4, :] = 1
    first, _, _ = run(images_world, masks=right)
    second, _, _ = run(images_world, masks=top)
    averaged, _, _ = run(images_world, masks=right, mask_contrast="mean")
    plain, _, _ = run(images_world)
    assert len({first, second, averaged, plain}) == 4
    tokens = I._record(first)["image_solve"]
    other = I._record(second)["image_solve"]
    mean_record = I._record(averaged)["image_solve"]
    bare = I._record(plain)["image_solve"]
    assert tokens["mask_hashes"] != other["mask_hashes"]
    assert tokens["mask_contrast"] == "tokens"
    assert tokens["settings"]["contrast_mode"] == "tokens"
    assert tokens["settings"]["sample_steps"] == [1]
    assert mean_record["mask_contrast"] == "mean"
    assert mean_record["settings"]["contrast_mode"] == "mean"
    assert mean_record["mask_hashes"] == tokens["mask_hashes"]
    assert "mask_hashes" not in bare and "mask_contrast" not in bare
    assert bare["settings"]["contrast_mode"] == "mean"
    assert bare["settings"]["sample_steps"] == [0, 1]
    assert bare["settings"]["steps"] == tokens["settings"]["steps"]


def test_mask_count_mismatch_raises(images_world, tmp_path):
    with pytest.raises(ValueError, match="Mask count 2 does not match image count 3"):
        run(images_world, masks=torch.ones(2, 8, 12))
    with pytest.raises(ValueError, match="Mask count 1 does not match image count 3"):
        run(images_world, masks=torch.ones(8, 12))
    Image.new("RGB", (20, 10), "red").save(tmp_path / "b.png")
    Image.new("RGB", (10, 20), "blue").save(tmp_path / "a.jpg")
    (tmp_path / "b.txt").write_text("a red object\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Mask count 1 does not match image count 2"):
        I._load_items(None, "", str(tmp_path), torch.ones(8, 8))
    paired = I._load_items(None, "", str(tmp_path), torch.ones(2, 8, 8))
    assert [item[1] for item in paired] == ["", "a red object"]
    assert all(tuple(item[2].shape) == (8, 8) for item in paired)


def test_mask_contrast_must_be_tokens_or_mean(images_world):
    with pytest.raises(ValueError, match="mask_contrast must be tokens or mean"):
        run(images_world, mask_contrast="other")


def test_held_out_loss_is_masked(sampler_world, monkeypatch):
    denoised = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    target = torch.zeros_like(denoised)
    weights = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss = I._held_out_loss(denoised, target, weights)
    err = (denoised - target).square()
    scale = weights.reshape(1, 1, 2, 2)
    expected = float((err * scale).sum() / (scale.sum() * err.shape[1]))
    assert loss == pytest.approx(expected)
    assert loss != pytest.approx(float(F.mse_loss(denoised, target)))
    ones = torch.ones(1, 4)
    assert I._held_out_loss(denoised, target, ones) == pytest.approx(
        float(F.mse_loss(denoised, target))
    )
    assert I._held_out_loss(denoised, target, None) == pytest.approx(
        float(F.mse_loss(denoised, target))
    )

    world = sampler_world
    from kilora.comfyui import image_learn

    monkeypatch.setattr(image_learn, "unrelated_drift", lambda *args, **kwargs: 0.0)
    image = torch.ones(8, 12, 3)
    items = I._prepare(world.model, world.vae, [(image, "tree", _partial_mask(8, 12))], 16)
    settings = replace(_settings(), seeds=(1,), size=16)
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
    losses = I._guard(
        world.model, world.clip, items, "ink", settings, adapter, result, lambda: None
    )
    z = items[0].latent
    width = len(I._GUARD_SCALES)
    for offset, scale in enumerate(I._GUARD_SCALES):
        group = [
            I._held_out_loss(value, z, items[0].weights) for value in world.denoised[offset::width]
        ]
        reported = next(item["loss"] for item in losses["losses"] if item["scale"] == scale)
        assert reported == pytest.approx(sum(group) / len(group))


def test_two_stage_style_uses_inverse_mask():
    latent = torch.ones(1, 4, 2, 2)
    weights = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
    item = I._Image(latent, "person", weights)
    style = I._style_items([item])[0]
    torch.testing.assert_close(style.weights, 1 - weights)
    torch.testing.assert_close(item.weights, weights)
    assert style.latent is latent
    assert I._style_items([I._Image(latent, "person")])[0].weights is None
    with pytest.raises(ValueError, match="background coverage"):
        I._style_items([I._Image(latent, "person", torch.ones_like(weights))])


@pytest.mark.parametrize("masked", [False, True])
def test_two_stage_order_deflation_axes_and_reuse(images_world, monkeypatch, masked):
    calls = []
    real = contrast.solve

    def solve(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((kwargs.get("coupling"), result))
        return result

    monkeypatch.setattr(contrast, "solve", solve)
    options = {"masks": torch.stack([_partial_mask(8, 12)] * 3)} if masked else {}
    paths, chain, report = run(images_world, two_stage=True, **options)
    style, character = paths.splitlines()
    assert "-style" in style and "-character" in character
    assert "Style stage" in report and "Character stage" in report
    assert len(calls) == 2
    assert calls[0][0] is None
    coupling = calls[1][0]
    assert coupling.deflate and coupling.order == ("style", "character")
    expected = I._saved_mounts(style)
    for mount, solution in coupling.earlier[0].items():
        torch.testing.assert_close(solution.down, expected[mount].down)
        torch.testing.assert_close(solution.up, expected[mount].up)
    for path, axis in ((style, "creative"), (character, "ip")):
        assert I._record(path)["axis"] == axis
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert handle.metadata()["kilora_axis"] == axis
    if masked:
        for background, foreground in zip(
            images_world.fitted[0], images_world.fitted[1], strict=True
        ):
            torch.testing.assert_close(background.weights, 1 - foreground.weights)
    else:
        assert all(item.weights is None for batch in images_world.fitted for item in batch)
    again, _, reused = run(images_world, two_stage=True, **options)
    assert again == paths and len(calls) == 2
    assert reused.count("Reused") == 2


def test_two_stage_failed_style_stops_character(images_world, monkeypatch):
    monkeypatch.setattr(
        I, "_guard", lambda *args: _guard_payload({s: 2.0 for s in I._GUARD_SCALES})
    )
    paths, _, report = run(images_world, two_stage=True)
    assert paths == ""
    assert "character solve skipped" in report
    assert images_world.counts.trajectories == 1


def test_two_stage_default_is_off():
    value = I.KiLoRASolveFromImages.INPUT_TYPES()["optional"]["two_stage"]
    assert value[0] == "BOOLEAN"
    assert value[1]["default"] is False


def test_character_guard_has_style_active_and_closes_hooks(images_world, monkeypatch):
    changes = []
    real_changes = P.SolveAdapter.changes
    guarded = []

    def track(adapter):
        result = real_changes(adapter)
        changes.append(result)
        return result

    def guard(*args):
        active = [change for change in changes if change.enabled and change.handles]
        guarded.append(len(active))
        return _guard_payload(_LOSSES)

    monkeypatch.setattr(P.SolveAdapter, "changes", track)
    monkeypatch.setattr(I, "_guard", guard)
    run(images_world, two_stage=True)
    assert guarded == [0, 1]
    for module in images_world.model.model.diffusion_model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_two_stage_character_guard_failure_keeps_only_style(images_world, monkeypatch):
    count = 0

    def guard(*args):
        nonlocal count
        count += 1
        return _guard_payload(_LOSSES if count == 1 else {s: 2.0 for s in I._GUARD_SCALES})

    monkeypatch.setattr(I, "_guard", guard)
    paths, _, report = run(images_world, two_stage=True)
    assert len(paths.splitlines()) == 1 and "-style" in paths
    assert "Character stage" in report
    assert not list(Path(paths).parent.glob("*-character*.safetensors"))


def test_held_out_loss_reduces_spatial_mask_to_output_grid():
    mask = torch.zeros(2, 1, 8, 12)
    mask[:, :, :4] = 1
    target = torch.zeros(2, 3, 4, 6)
    denoised = torch.ones_like(target)
    denoised[:, :, 2:] = 10
    assert I._held_out_loss(denoised, target, mask) == pytest.approx(1.0)
    assert I._held_out_loss(denoised, target, None) > 1.0


@pytest.mark.parametrize("masked", [False, True])
def test_switch_defaults_preserve_schedule_guard_and_saved_bytes(images_world, masked):
    assert I._mask_schedule(15) == I._mask_schedule(15, 0.4, "") == (6, (8, 11, 14))
    expected = I._guard_result(_LOSSES, {scale: 0 for scale in I._GUARD_SCALES[1:]})
    assert expected["scale"] == 0.5 and "rule" not in expected
    options = {"masks": torch.ones(3, 8, 12)} if masked else {}
    first, _, _ = run(images_world, **options)
    previous = Path(first).read_bytes()
    path, _, report = run(
        images_world,
        **options,
        mask_start=0.4,
        mask_samples="",
        refit_rounds=0,
        guard_rule="loss",
        trajectory_check=False,
        overwrite=True,
    )
    assert path == first
    assert Path(path).read_bytes() == previous
    record = I._record(path)["image_solve"]
    assert not {"mask_start", "mask_samples", "guard_rule", "trajectory_check"} & record.keys()
    assert "trajectory" not in I._record(path)["image_guard"]
    assert record["settings"]["refit_rounds"] == 0
    assert "Guard rule: loss" in report
    assert "Refit round" not in report


@pytest.mark.parametrize(
    "start, positions, expected",
    [
        (0.2, "", (3, (6, 10, 14))),
        (0.0, "0 0.5 1.0", (0, (0, 7, 14))),
        (0.4, "0.5 0.75 1.0", (6, (7, 10, 14))),
        (0.6, "0.75, 1.0", (8, (10, 14))),
        (0.4, "1.0 0.75 0.75", (6, (10, 14))),
    ],
)
def test_mask_schedule_switches(start, positions, expected):
    assert I._mask_schedule(15, start, positions) == expected


@pytest.mark.parametrize(
    "start, positions, message",
    [
        ("bad", "", "mask_start"),
        (-0.1, "", "mask_start"),
        (0.96, "", "mask_start"),
        (math.nan, "", "mask_start"),
        (math.inf, "", "mask_start"),
        (0.4, "bad", "numeric"),
        (0.4, " ", "at least one"),
        (0.4, ", ,", "at least one"),
        (0.4, "0.3 0.75", "at or after"),
        (0.4, "1.01", "at or after"),
        (0.4, "nan", "at or after"),
        (0.4, "inf", "at or after"),
    ],
)
def test_mask_schedule_rejects_invalid_values(start, positions, message):
    with pytest.raises(ValueError, match=message):
        I._mask_schedule(15, start, positions)


def test_explicit_mask_schedule_reaches_aligned_sampler(sampler_world):
    world = sampler_world
    _install_recording_sampler(world)
    start, chosen = I._mask_schedule(15, 0.2, "0.5 0.75 1.0")
    settings = replace(_settings(), size=16, steps=15, sample_steps=chosen)
    samples = I.image_samples(
        world.model,
        world.vae,
        [(torch.ones(8, 12, 3), "tree", _partial_mask(8, 12))],
        "ink",
        settings,
        world.progress,
        clip=world.clip,
        mask_start=0.2,
        mask_samples="0.5 0.75 1.0",
    )
    torch.testing.assert_close(world.starts[0][2], torch.arange(15, -1, -1).float()[start:])
    assert len(samples) == len(chosen) * len(settings.seeds)
    assert [float(call[1][0]) for call in world.calls[::2]] == [15 - step for step in chosen]


@pytest.mark.parametrize("rounds", [1, 3])
def test_refit_resamples_last_solution_and_guards_saves_once(images_world, monkeypatch, rounds):
    world = images_world
    original_samples = I.image_samples
    original_solve, original_write = contrast.solve, contrast.write_lora
    changes, solutions, guarded, written = [], [], [], []

    def samples(*args, student_change=None, **kwargs):
        changes.append(student_change)
        return original_samples(*args, **kwargs)

    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        solutions.append(result)
        assert kwargs["settings"].refit_rounds == rounds
        assert kwargs["refit_round"] == len(solutions) - 1
        return result

    def guard(*args):
        guarded.append(args[6])
        return _guard_payload(_LOSSES)

    def write(path, result):
        written.append(result)
        return original_write(path, result)

    monkeypatch.setattr(I, "image_samples", samples)
    monkeypatch.setattr(contrast, "solve", solve)
    monkeypatch.setattr(I, "_guard", guard)
    monkeypatch.setattr(contrast, "write_lora", write)
    path, _, report = run(world, refit_rounds=rounds)
    assert len(solutions) == rounds + 1
    assert changes == [None, *solutions[:-1]]
    assert guarded == written == solutions[-1:]
    assert I._record(path)["refit_round"] == rounds
    assert len([line for line in report.splitlines() if line.startswith("Refit round ")]) == rounds
    assert len(list(world.root.rglob("*.safetensors"))) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_refit_change_only_runs_on_student_and_hooks_are_removed(sampler_world, monkeypatch, fail):
    import comfy.samplers

    world = sampler_world
    states, captured = [], []
    original_capture = P._capture_call
    adapter = P.SolveAdapter(world.model)
    mount = adapter.mounts[0]
    solution = contrast.MountSolution(torch.eye(4), torch.eye(4), 0.1, 1.0)
    result = contrast.ContrastResult({mount: solution}, {})
    module = adapter.locations[mount][0]
    hook_count = len(module._forward_hooks)
    original_change = P._AdapterChanges.active

    def capture(*args, **kwargs):
        captured.append(states[-1] if states else False)
        return original_capture(*args, **kwargs)

    from contextlib import contextmanager

    @contextmanager
    def active(self, enabled):
        states.append(enabled)
        try:
            with original_change(self, enabled):
                yield
        finally:
            states.pop()

    def inner_sample(self, noise, latent, device, sampler, sigmas, *args):
        wrapped = self.model_options["model_function_wrapper"]
        for sigma in sigmas[:-1]:
            arguments = {
                "input": noise,
                "timestep": sigma.expand(len(noise)),
                "c": {},
                "cond_or_uncond": [0],
            }

            def denoise(x, timestep):
                assert states == [True]
                hidden = x.reshape(x.shape[0], -1, 4)
                with_change = module(hidden)
                base = F.linear(hidden, module.weight, module.bias)
                torch.testing.assert_close(with_change, base + hidden)
                if fail:
                    raise RuntimeError("sample interrupted")
                return x

            wrapped(denoise, arguments)

    monkeypatch.setattr(P, "_capture_call", capture)
    monkeypatch.setattr(P._AdapterChanges, "active", active)
    monkeypatch.setattr(comfy.samplers.CFGGuider, "inner_sample", inner_sample)
    settings = replace(_settings(), size=16)
    arguments = (
        world.model,
        world.vae,
        [(torch.ones(8, 12, 3), "tree")],
        "ink",
        settings,
        world.progress,
    )
    if fail:
        with pytest.raises(RuntimeError, match="sample interrupted"):
            I.image_samples(*arguments, clip=world.clip, student_change=result)
    else:
        I.image_samples(*arguments, clip=world.clip, student_change=result)
    assert captured and not any(captured)
    assert len(module._forward_hooks) == hook_count
    assert "comfy_force_forward" not in module.__dict__
    assert states == []


def test_drift_rule_chooses_largest_allowed_scale_and_stops_collapse():
    losses = {scale: 2.0 for scale in I._GUARD_SCALES}
    losses.update({0.1: 1.0, 0.75: 2.5, 1.0: 2.51})
    drifts = {scale: 0.01 for scale in I._GUARD_SCALES[1:]}
    guard = I._guard_result(losses, drifts, bound=0.1, rule="drift")
    assert guard["scale"] == 0.75 and guard["rule"] == "drift"
    assert I._guard_passed(guard)
    assert not guard["losses"][-1]["allowed"]
    assert "Guard rule: drift" in I._guard_report(guard)
    drifts[0.75] = 0.11
    assert I._guard_result(losses, drifts, bound=0.1, rule="drift")["scale"] == 0.5
    assert I._guard_result(losses, drifts, bound=0.1)["scale"] == 0.1


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_drift_rule_rejects_nonfinite_measurements(bad):
    losses = dict(_LOSSES)
    drifts = {scale: 0.0 for scale in I._GUARD_SCALES[1:]}
    losses[1.0] = bad
    drifts[0.75] = bad
    guard = I._guard_result(losses, drifts, rule="drift")
    assert guard["scale"] == 0.5
    losses[0.0] = bad
    assert I._guard_result(losses, drifts, rule="drift")["scale"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"refit_rounds": -1},
        {"refit_rounds": 9},
        {"refit_rounds": True},
        {"refit_rounds": 0.5},
        {"guard_rule": "unknown"},
    ],
)
def test_invalid_switch_inputs(images_world, options):
    with pytest.raises(ValueError, match="refit_rounds|guard_rule"):
        run(images_world, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"mask_start": 0.2},
        {"mask_samples": "0.5 0.75 1.0"},
        {"refit_rounds": 1},
        {"guard_rule": "drift"},
    ],
)
def test_switches_change_cache_identity(images_world, monkeypatch, options):
    original_samples, original_guard = I.image_samples, I._guard
    monkeypatch.setattr(I, "image_samples", lambda *a, **kw: original_samples(*a, clip=kw["clip"]))
    monkeypatch.setattr(I, "_guard", lambda *a, **kw: original_guard(*a))
    first, _, _ = run(images_world)
    before = Path(first).read_bytes()
    changed, _, _ = run(images_world, **options)
    assert changed != first
    assert Path(first).read_bytes() == before
    record = I._record(changed)["image_solve"]
    for key, value in options.items():
        assert (record["settings"] if key == "refit_rounds" else record)[key] == value
    assert run(images_world, **options)[0] == changed


def test_node_passes_custom_mask_schedule_and_drift_rule(images_world, monkeypatch):
    sampled, checked = [], []
    original_samples = I.image_samples

    def samples(*args, mask_start, mask_samples, **kwargs):
        sampled.append((mask_start, mask_samples, args[4].sample_steps))
        return original_samples(*args, **kwargs)

    def guard(*args, guard_rule):
        checked.append(guard_rule)
        return I._guard_result(
            _LOSSES, {scale: 0.0 for scale in I._GUARD_SCALES[1:]}, rule=guard_rule
        )

    monkeypatch.setattr(I, "image_samples", samples)
    monkeypatch.setattr(I, "_guard", guard)
    path, _, report = run(
        images_world,
        masks=torch.ones(3, 8, 12),
        steps=15,
        mask_start=0.2,
        mask_samples="0.5 0.75 1.0",
        guard_rule="drift",
    )
    assert sampled == [(0.2, "0.5 0.75 1.0", (7, 10, 14))]
    assert checked == ["drift"]
    assert I._record(path)["image_guard"]["scale"] == 1.0
    assert "Guard rule: drift" in report


@pytest.fixture
def trajectory_world(sampler_world, monkeypatch):
    from contextlib import contextmanager

    import comfy.samplers

    tw = sampler_world

    class Changes:
        enabled = False
        scale = 0.0

        @contextmanager
        def active(self, enabled):
            previous = self.enabled
            self.enabled = enabled
            try:
                yield
            finally:
                self.enabled = previous

    tw.changes = Changes()
    tw.runs = []
    tw.schedule_calls = []
    tw.fail = lambda scale, prompt: "flat" if scale >= 0.5 else None
    tw.settings = replace(_settings(), seeds=(17, 99), size=16, cfg=3.5)
    tw.sampling = {"sampler": "family-sampler", "scheduler": "family-schedule"}
    tw.lifecycle = []
    tw.model.pre_run = lambda: tw.lifecycle.append("start")
    tw.model.cleanup = lambda: tw.lifecycle.append("end")
    original_sigmas = comfy.samplers.calculate_sigmas

    def sigmas(model_sampling, scheduler, steps):
        tw.schedule_calls.append((scheduler, steps))
        return original_sigmas(model_sampling, scheduler, steps)

    class Guider:
        def __init__(self, patcher):
            self.model_options = patcher.model_options

        def set_conds(self, positive, negative):
            self.original_conds = {"positive": positive, "negative": negative}
            self.prompt = positive[0][1]["text"]

        def set_cfg(self, cfg):
            assert cfg == tw.settings.cfg

        def inner_sample(
            self,
            noise,
            latent,
            device,
            sampler,
            sigmas,
            mask,
            callback,
            disable,
            seed,
            *,
            latent_shapes,
        ):
            assert sampler == tw.sampling["sampler"]
            assert seed == tw.settings.seeds[0]
            assert latent_shapes == [tuple(latent.shape)]
            assert latent.shape == (1, 4, 2, 2)
            assert len(sigmas) == 9
            tw.runs.append((self.prompt, tw.changes.enabled, tw.changes.scale, noise.clone()))
            output = noise.clone()
            mode = tw.fail(tw.changes.scale, self.prompt) if tw.changes.enabled else None
            for _ in sigmas[:-1]:
                callback()
                if mode == "flat":
                    output *= 0.8
                elif mode == "distance":
                    output += noise * 0.2
                elif mode == "nan":
                    output *= math.nan
            return output

    monkeypatch.setattr(comfy.samplers, "CFGGuider", Guider)
    monkeypatch.setattr(comfy.samplers, "calculate_sigmas", sigmas)
    tw.vae.encode = lambda *a: pytest.fail("trajectory encoded an image")
    tw.vae.decode = lambda *a: pytest.fail("trajectory decoded a latent")
    return tw


def _run_trajectory(tw, rule="loss"):
    guard = I._guard_result(_LOSSES, {scale: 0 for scale in I._GUARD_SCALES[1:]}, rule=rule)
    return I._trajectory_guard(
        tw.model, tw.clip, tw.settings, tw.sampling, tw.changes, guard, lambda: None
    )


@pytest.mark.parametrize("rule, scales", [("loss", [0.5, 0.3]), ("drift", [1.0, 0.75, 0.5, 0.3])])
def test_trajectory_refuses_flat_scale_then_uses_next_allowed(trajectory_world, rule, scales):
    tw = trajectory_world
    guard = _run_trajectory(tw, rule)
    assert guard["scale"] == 0.3 and I._guard_passed(guard)
    assert guard["relative_gain"] == pytest.approx(0.25)
    assert [check["scale"] for check in guard["trajectory"]] == scales
    for check in guard["trajectory"][:-1]:
        assert check["collapse"] and check["std_ratio"] < 0.5
        assert check["distance"] < 1.0
    assert not guard["trajectory"][-1]["collapse"]
    assert all(check["seconds"] >= 0 for check in guard["trajectory"])
    report = I._guard_report(guard)
    lines = [line for line in report.splitlines() if line.startswith("Trajectory check at scale")]
    assert len(lines) == len(scales)
    assert all("std ratio" in line and "distance" in line and line.endswith(" s") for line in lines)
    assert "collapse" in lines[0] and "passed" in lines[-1]
    from kilora.comfyui import image_learn

    prompts = image_learn.UNRELATED_PROMPTS[:2]
    assert len(tw.runs) == 2 + 2 * len(scales)
    assert [run[0] for run in tw.runs] == list(prompts) * (1 + len(scales))
    assert [run[1] for run in tw.runs] == [False, False] + [True, True] * len(scales)
    for index, changed in enumerate(tw.runs[2:]):
        base = tw.runs[index % 2]
        torch.testing.assert_close(base[3], changed[3], rtol=0, atol=0)
    assert tw.schedule_calls == [(tw.sampling["scheduler"], 8)] * len(tw.runs)
    assert tw.lifecycle == ["start", "end"] * len(tw.runs)
    assert not tw.changes.enabled


@pytest.mark.parametrize("mode", ["flat", "distance", "nan"])
def test_trajectory_all_scales_collapsing_fail_guard(trajectory_world, mode):
    tw = trajectory_world
    tw.fail = lambda scale, prompt: mode
    guard = _run_trajectory(tw)
    assert guard["scale"] == guard["relative_gain"] == 0.0
    assert not I._guard_passed(guard)
    assert [check["scale"] for check in guard["trajectory"]] == [0.5, 0.3, 0.2, 0.1]
    assert all(check["collapse"] for check in guard["trajectory"])
    if mode == "distance":
        assert all(check["distance"] > 1.0 for check in guard["trajectory"])
        assert all(check["std_ratio"] > 0.5 for check in guard["trajectory"])


def test_trajectory_second_prompt_alone_can_refuse_scale(trajectory_world):
    tw = trajectory_world
    from kilora.comfyui import image_learn

    tw.fail = lambda scale, prompt: (
        "distance" if scale >= 0.5 and prompt == image_learn.UNRELATED_PROMPTS[1] else None
    )
    guard = _run_trajectory(tw)
    assert guard["scale"] == 0.3
    assert [item["collapse"] for item in guard["trajectory"][0]["prompts"]] == [False, True]


def test_trajectory_does_not_sample_when_loss_and_drift_allow_nothing(trajectory_world):
    tw = trajectory_world
    guard = _guard_payload({scale: 2.0 for scale in I._GUARD_SCALES})
    I._trajectory_guard(
        tw.model, tw.clip, tw.settings, tw.sampling, tw.changes, guard, lambda: None
    )
    assert guard["trajectory"] == [] and not tw.runs
    assert not I._guard_passed(guard)


def test_trajectory_guard_uses_installed_changes_and_restores_on_interrupt(
    sampler_world, monkeypatch
):
    tw = sampler_world
    settings = replace(_settings(), seeds=(1,), size=16)
    adapter = P.SolveAdapter(tw.model)
    result = contrast.solve(
        tw.model.model.diffusion_model,
        _samples(2),
        [("same", "same")],
        settings=settings,
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    items = I._prepare(tw.model, tw.vae, [(torch.ones(8, 12, 3), "tree")], 16)
    monkeypatch.setattr(I.image_learn, "unrelated_drift", lambda *a: 0.0)
    checked = []
    model = tw.model.model.diffusion_model
    training = model.training

    def check(patcher, clip, settings_, sampling, changes, guard, interrupt):
        assert changes.handles and not model.training
        assert settings_ is settings and changes.adapter is adapter
        checked.append(True)
        raise RuntimeError("interrupted trajectory")

    monkeypatch.setattr(I, "_trajectory_guard", check)
    with pytest.raises(RuntimeError, match="interrupted trajectory"):
        I._guard(
            tw.model,
            tw.clip,
            items,
            "ink",
            settings,
            adapter,
            result,
            lambda: None,
            trajectory_check=True,
        )
    assert checked == [True] and model.training == training
    assert not model._forward_pre_hooks
    assert all(not module._forward_hooks for module in model.modules())


def test_trajectory_option_has_distinct_cache_and_metadata(images_world, monkeypatch):
    first, _, _ = run(images_world)
    seen = []

    def guard(*args, trajectory_check=False):
        seen.append(trajectory_check)
        result = _guard_payload(_LOSSES)
        if trajectory_check:
            result["trajectory"] = [
                {
                    "scale": 0.5,
                    "std_ratio": 1.0,
                    "distance": 0.1,
                    "collapse": False,
                    "seconds": 0.25,
                    "prompts": [],
                }
            ]
        return result

    monkeypatch.setattr(I, "_guard", guard)
    path, _, report = run(images_world, trajectory_check=True)
    assert path != first and seen == [True]
    metadata = I._record(path)
    assert metadata["image_solve"]["trajectory_check"] is True
    assert metadata["image_guard"]["trajectory"][0]["seconds"] == 0.25
    assert report.count("Trajectory check at scale") == 1
    reused, _, _ = run(images_world, trajectory_check=True)
    assert reused == path and seen == [True]


@pytest.mark.parametrize("gain", [0.5, 2.0])
def test_trajectory_thresholds_are_strict(trajectory_world, monkeypatch, gain):
    tw = trajectory_world
    base = torch.tensor([-1.0, 1.0, -1.0, 1.0]).reshape(1, 1, 2, 2)
    monkeypatch.setattr(
        I,
        "_trajectory_sample",
        lambda *args: base * gain if tw.changes.enabled else base,
    )
    guard = _run_trajectory(tw)
    assert guard["scale"] == 0.5 and I._guard_passed(guard)
    check = guard["trajectory"][0]
    assert check["std_ratio"] == pytest.approx(gain)
    assert check["distance"] == pytest.approx(abs(gain - 1.0))
    assert not check["collapse"]


def test_trajectory_equal_constant_latents_do_not_collapse(trajectory_world, monkeypatch):
    tw = trajectory_world
    monkeypatch.setattr(I, "_trajectory_sample", lambda *args: torch.zeros(1, 4, 2, 2))
    guard = _run_trajectory(tw)
    assert guard["scale"] == 0.5 and I._guard_passed(guard)
    assert guard["trajectory"][0]["std_ratio"] == 1.0
    assert guard["trajectory"][0]["distance"] == 0.0
