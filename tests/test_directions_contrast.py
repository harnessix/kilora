# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from kilora.cli import main as cli_main
from kilora.library import contrast
from kilora.library.contrast import (
    PromptPair,
    Sample,
    Settings,
    extract,
    fit_mount,
    read_prompt_set,
    solve,
    write_lora,
)
from kilora.lora_import import import_lora
from kilora.presets import anima_name, pack_license


class Linear(nn.Linear):
    def __init__(self, width):
        super().__init__(width, width, bias=False)
        self.register_buffer("target_delta", torch.zeros(width, width))
        self.target = False
        with torch.no_grad():
            self.weight.copy_(torch.eye(width))

    def forward(self, x):
        output = super().forward(x)
        if self.target:
            output = output + F.linear(x, self.target_delta)
        return output


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.to_q = Linear(width)
        self.to_k = Linear(width)
        self.to_v = Linear(width)
        self.to_out = nn.Sequential(Linear(width))

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        if k.shape != q.shape:
            k = k.mean(-2, keepdim=True).expand_as(q)
            v = v.mean(-2, keepdim=True).expand_as(q)
        return self.to_out((q + k + v) / 3)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn1, self.attn2 = Attention(width), Attention(width)
        self.ff = nn.Module()
        first = nn.Module()
        first.proj = Linear(width)
        self.ff.net = nn.ModuleList([first, nn.Identity(), Linear(width)])
        self.inputs = []

    def forward(self, hidden_states, encoder_hidden_states, target=False):
        self.inputs.append((target, hidden_states.detach().clone()))
        for module in self.modules():
            if isinstance(module, Linear):
                module.target = target
        x = self.attn1(hidden_states)
        x = self.attn2(x, encoder_hidden_states)
        return self.ff.net[2](self.ff.net[0].proj(x))


class Model(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(width), Block(width)])

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        target=False,
        timestep=None,
        padding_mask=None,
        return_dict=False,
    ):
        shape = hidden_states.shape
        x = hidden_states
        if x.ndim == 5:
            x = x.reshape(shape[0], shape[1], -1).transpose(1, 2)
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, target=target)
        if len(shape) == 5:
            x = x.transpose(1, 2).reshape(shape)
        return (x,)


def _samples(count=4, width=4):
    generator = torch.Generator().manual_seed(18)
    samples = []
    for _ in range(count):
        x = torch.randn(1, 12, width, generator=generator)
        caption = torch.randn(1, 12, width, generator=generator)
        shared = {"hidden_states": x, "encoder_hidden_states": caption}
        samples.append(Sample({**shared, "target": True}, {**shared, "target": False}))
    return samples


def _settings(rank=2):
    return Settings(seeds=(1,), size=16, steps=2, sample_steps=(0, 1), rank=rank, ridge=1e-4)


def _solve(model, samples=None, **kwargs):
    return solve(
        model,
        _samples() if samples is None else samples,
        kwargs.pop("pairs", [("same", "same")]),
        settings=kwargs.pop("settings", _settings()),
        model_sha="toy-model",
        license=pack_license("anima")[0],
        **kwargs,
    )


def _set_change(model, block=0):
    delta = torch.zeros(4, 4)
    delta[0, 1] = 0.3
    model.transformer_blocks[block].attn1.to_q.target_delta.copy_(delta)
    return delta


def test_zero_contrast_and_unused_path_are_bit_identical():
    model = Model()
    model.train()
    model.transformer_blocks[1].eval()
    samples = _samples()
    before = model(**samples[0].student)[0].detach().clone()
    original = copy.deepcopy(model.state_dict())
    training = [m.training for m in model.modules()]
    result = _solve(model, samples)
    after = model(**samples[0].student)[0]
    assert torch.equal(before, after)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    assert [m.training for m in model.modules()] == training
    assert len(result.mounts) == 20
    assert "svd_fallbacks" not in result.metadata
    for solution in result.mounts.values():
        assert torch.allclose(solution.delta(), torch.zeros(4, 4), atol=1e-8)
    assert all(gap == 0 for gap in result.metadata["block_gaps"].values())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_known_low_rank_change_is_recovered():
    model = Model()
    expected = _set_change(model)
    result = _solve(model)
    recovered = result.mounts["transformer_blocks.0.attn1.to_q"].delta()
    assert torch.allclose(recovered, expected, atol=2e-4, rtol=2e-3)
    assert result.metadata["block_gaps"]["1"] < 2e-4


def test_svd_fallback_count_is_recorded(monkeypatch):
    real = contrast.fit_mount

    def record(c, s, rank, **kwargs):
        solution = real(c, s, rank, **kwargs)
        solution.svd_fallback = True
        return solution

    monkeypatch.setattr(contrast, "fit_mount", record)
    result = _solve(Model())
    assert result.metadata["svd_fallbacks"] == len(result.mounts)
    assert all(record["svd_fallback"] for record in result.metadata["fit"].values())


def test_statistics_sum_every_pair_and_sample(monkeypatch):
    model = Model()
    delta = _set_change(model)
    samples = _samples(count=4)
    seen = []
    real = contrast.fit_mount

    def record(c, s, rank, **kwargs):
        seen.append((c.clone(), s.clone()))
        return real(c, s, rank, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    _solve(model, samples, pairs=[("first", "first"), ("second", "second")])
    expected_c = torch.zeros(4, 4)
    expected_s = torch.zeros(4, 4)
    with contrast._matmul_allow_tf32(_settings().stats_precision == "tf32"):
        for sample in samples:
            x = sample.student["hidden_states"].reshape(-1, 4)
            expected_c += x.T @ x
            expected_s += F.linear(x, delta).T @ x
    assert torch.allclose(seen[0][0], expected_c, atol=1e-6)
    assert torch.allclose(seen[0][1], expected_s, atol=2e-6)


def test_second_block_student_receives_first_block_change():
    model = Model()
    _set_change(model)
    samples = _samples(count=1)
    model(**samples[0].student)
    base = model.transformer_blocks[1].inputs[-1][1]
    model.transformer_blocks[1].inputs.clear()
    result = _solve(model, samples)
    seen = model.transformer_blocks[1].inputs
    target_input = next(x for target, x in seen if target)
    student_input = next(x for target, x in seen if not target)
    assert not torch.allclose(student_input, base)
    assert torch.allclose(student_input, target_input, atol=2e-4, rtol=2e-3)
    assert result.metadata["block_gaps"]["0"] < 2e-4


def test_ridge_increases_and_is_recorded():
    c = torch.diag(torch.tensor([1.0, -0.0002]))
    result = fit_mount(c, torch.ones(2, 2), rank=1, ridge=1e-4)
    initial = abs(float(c.diagonal().mean())) * 1e-4
    assert result.ridge == pytest.approx(initial * 10)
    assert torch.isfinite(result.delta()).all()
    assert 0 <= result.fit_share <= 1


def test_zero_gram_and_unrepairable_gram():
    zero = fit_mount(torch.zeros(2, 2), torch.zeros(3, 2), rank=4)
    assert zero.down.shape == (4, 2)
    assert zero.up.shape == (4, 3)
    assert torch.count_nonzero(zero.delta()) == 0
    with pytest.raises(ValueError, match="positive definite"):
        fit_mount(torch.diag(torch.tensor([1.0, -1.0])), torch.ones(2, 2), rank=1)


def test_rank_cut_uses_gram_metric():
    c = torch.diag(torch.tensor([100.0, 1.0]))
    s = torch.diag(torch.tensor([10.0, 2.0]))
    solution = fit_mount(c, s, rank=1, ridge=1e-4)
    assert abs(float(solution.delta()[0, 0])) < 1e-6
    assert float(solution.delta()[1, 1]) == pytest.approx(2, rel=0.01)
    assert solution.fit_share == pytest.approx(0.8, abs=0.002)


def test_plain_lora_round_trip(tmp_path):
    model = Model()
    _set_change(model)
    result = _solve(model)
    path = tmp_path / "line.safetensors"
    write_lora(path, result)
    weights = load_file(str(path))
    for name, solution in result.mounts.items():
        stem = "lora_unet_" + anima_name(name).replace(".", "_")
        down, up = weights[stem + ".lora_down.weight"], weights[stem + ".lora_up.weight"]
        assert torch.equal(up @ down, solution.delta())
        assert float(weights[stem + ".alpha"]) == 2
    with safe_open(str(path), framework="pt") as file:
        metadata = json.loads(file.metadata()["kilora_directions"])
    assert metadata["pairs"][0]["target"] == "same"
    assert metadata["license"] == "LicenseRef-CircleStone-Labs-NC-1.2"
    assert metadata["model_sha"] == "toy-model"
    assert metadata["fit"] == result.metadata["fit"]
    imported = import_lora(path, model, family="anima", experts=1)
    assert set(imported.adapter_set.experts) == set(result.mounts)


def test_caption_mismatch_excludes_cross_keys_and_values(tmp_path):
    model = Model()
    result = _solve(model, pairs=[PromptPair("full style", "cue"), PromptPair("same", "same")])
    assert len(result.mounts) == 16
    excluded = result.metadata["excluded_mounts"]
    assert len(excluded) == 4
    assert all(n.endswith(("attn2.to_k", "attn2.to_v")) for n in excluded)
    assert all("captions differ" in reason for reason in excluded.values())
    path = tmp_path / "different.safetensors"
    write_lora(path, result)
    with safe_open(str(path), framework="pt") as file:
        assert json.loads(file.metadata()["kilora_directions"])["excluded_mounts"] == excluded


@pytest.mark.parametrize("exact", [True, False])
def test_partial_resume_starts_at_next_block(tmp_path, monkeypatch, exact):
    model = Model()
    _set_change(model)
    samples = _samples(count=2)
    partial = tmp_path / "run.partial.safetensors"
    settings = (
        replace(_settings(), stats_precision="float32", fit_dtype="float64")
        if exact
        else _settings()
    )

    def stop(message):
        raise InterruptedError("stop after checkpoint")

    with pytest.raises(InterruptedError):
        _solve(model, samples, partial=partial, progress=stop, settings=settings)
    assert partial.exists()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    real = contrast.fit_mount
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    resumed = _solve(model, samples, partial=partial, settings=settings)
    assert len(calls) == 10
    uninterrupted = _solve(model, samples, settings=settings)
    if exact:
        assert resumed.metadata == uninterrupted.metadata
        for name in resumed.mounts:
            assert torch.equal(resumed.mounts[name].down, uninterrupted.mounts[name].down)
            assert torch.equal(resumed.mounts[name].up, uninterrupted.mounts[name].up)
        return
    gaps = resumed.metadata.pop("block_gaps"), uninterrupted.metadata.pop("block_gaps")
    fits = resumed.metadata.pop("fit"), uninterrupted.metadata.pop("fit")
    assert resumed.metadata == uninterrupted.metadata
    assert gaps[0].keys() == gaps[1].keys()
    for key in gaps[1]:
        assert gaps[0][key] == pytest.approx(gaps[1][key], rel=1e-3, abs=1e-6)
    assert fits[0].keys() == fits[1].keys()
    for name in resumed.mounts:
        torch.testing.assert_close(
            resumed.mounts[name].delta(), uninterrupted.mounts[name].delta(), atol=1e-5, rtol=1e-3
        )


def test_resume_rejects_different_settings(tmp_path):
    partial = tmp_path / "run.partial.safetensors"
    model = Model()
    _solve(model, partial=partial)
    with pytest.raises(ValueError, match="pairs"):
        _solve(model, partial=partial, pairs=[("other", "same")])


def test_explicit_mounts_resolve_import_aliases():
    model = Model()
    result = _solve(model, mounts=["lora_unet_blocks_0_self_attn_q_proj"])
    assert list(result.mounts) == ["transformer_blocks.0.attn1.to_q"]
    with pytest.raises(ValueError):
        _solve(model, mounts=["missing"])


def test_prompt_set_and_cli_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "label": "line",
                        "style": "thin lines",
                        "cue": "line",
                        "subjects": ["a tree", "a house"],
                    },
                    {
                        "label": "wash",
                        "style": "soft watercolor",
                        "cue": "wash",
                        "subjects": ["a cat"],
                    },
                ]
            }
        )
    )
    groups = read_prompt_set(path)
    assert groups["line"][0] == PromptPair("a tree, thin lines", "a tree, line", "line", "line")
    calls = []
    pipe = SimpleNamespace(transformer=Model())
    monkeypatch.setattr("kilora.anima.load_anima", lambda *args: pipe)
    monkeypatch.setattr(contrast, "model_sha256", lambda path: "toy")
    monkeypatch.setattr(contrast, "extract", lambda *args, **kwargs: calls.append((args, kwargs)))
    out = tmp_path / "results"
    model_folder = tmp_path / "model"
    model_folder.mkdir()
    assert (
        cli_main(
            [
                "directions",
                "contrast",
                "--model",
                str(model_folder),
                "--out",
                str(out),
                "--set",
                str(path),
                "--steps",
                "4",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert [call[0][1].name for call in calls] == ["line.safetensors", "wash.safetensors"]
    assert calls[0][1]["settings"].sample_steps == (0, 1, 3)
    assert calls[0][1]["settings"].cfg == 1.0
    calls.clear()
    assert (
        cli_main(
            [
                "directions",
                "contrast",
                "--model",
                "toy",
                "--out",
                str(out / "joint.safetensors"),
                "--pair",
                "style 1",
                "plain 1",
                "--pair",
                "style 2",
                "plain 2",
            ]
        )
        == 0
    )
    assert calls[0][0][2] == [["style 1", "plain 1"], ["style 2", "plain 2"]]


class Scheduler:
    init_noise_sigma = 1.0

    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, device=device)
        self.sigmas = torch.linspace(1, 0, steps + 1, device=device)

    def step(self, prediction, timestep, latent, return_dict):
        return (latent - prediction * 0.1,)


@pytest.mark.parametrize("cfg", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("student_trajectory", [False, True])
def test_trajectory_guidance_predictions_and_model_calls(cfg, student_trajectory):
    settings = replace(_settings(), cfg=cfg, steps=3, sample_steps=(0, 1, 2))
    captions = {
        "target": torch.tensor([2.0]),
        "student": torch.tensor([1.0]),
        "": torch.tensor([0.5]),
    }
    calls, predictions = [], []

    class RecordingScheduler(Scheduler):
        def step(self, prediction, timestep, latent, return_dict):
            predictions.append(prediction.clone())
            return super().step(prediction, timestep, latent, return_dict)

    def model(hidden_states, encoder_hidden_states, **kwargs):
        calls.append(encoder_hidden_states.item())
        return (hidden_states * 0.25 + encoder_hidden_states.item(),)

    samples = contrast._sample_trajectories(
        SimpleNamespace(scheduler=RecordingScheduler()),
        model,
        [PromptPair("target", "student")],
        settings,
        torch.device("cpu"),
        torch.bfloat16,
        captions,
        student_trajectory=student_trajectory,
    )
    caption = "student" if student_trajectory else "target"
    expected_calls = [captions[caption].item()]
    if cfg != 1.0:
        expected_calls.append(captions[""].item())
    assert calls == expected_calls * settings.steps
    latent = torch.randn(
        (1, 16, 1, 2, 2), generator=torch.Generator().manual_seed(settings.seeds[0])
    )
    for sample, prediction in zip(samples, predictions, strict=True):
        hidden = latent.to(torch.bfloat16)
        assert torch.equal(sample.target["hidden_states"], hidden)
        assert torch.equal(sample.student["hidden_states"], hidden)
        conditional = (hidden * 0.25 + captions[caption].item()).float()
        unconditional = (hidden * 0.25 + captions[""].item()).float()
        expected = unconditional + cfg * (conditional - unconditional)
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
        latent = latent - expected * 0.1


def test_fake_pipeline_extracts_bf16_trajectory_and_exports(tmp_path, monkeypatch):
    pipe = SimpleNamespace(transformer=Model(width=16), scheduler=Scheduler())
    prompts = []

    def encode(pipe, caption, device):
        prompts.append(caption)
        return torch.full((3, 16), 0.1 if caption == "target" else 0.0)

    monkeypatch.setattr("kilora.anima.encode_caption", encode)
    monkeypatch.setattr("kilora.train.swap_to_denoiser", lambda pipe, dev: None)
    settings = _settings()
    pairs = [PromptPair("target", "student")]
    samples = contrast.trajectory_samples(pipe, pairs, settings, "cpu")
    assert len(samples) == 2
    assert samples[0].target["hidden_states"].dtype == torch.bfloat16
    assert torch.equal(samples[0].target["hidden_states"], samples[0].student["hidden_states"])
    assert set(prompts) == {"target", "student", ""}
    messages = []
    path = tmp_path / "direction.safetensors"
    result = extract(
        "unused",
        path,
        pairs,
        settings=settings,
        device="cpu",
        pipe=pipe,
        model_sha="fake",
        progress=messages.append,
    )
    assert path.exists()
    assert not path.with_name(path.name + ".partial.safetensors").exists()
    assert "GPU memory" in messages[0]
    assert result.metadata["sample_count"] == 2
    assert result.metadata["cfg"] == settings.cfg
    assert len(result.metadata["excluded_mounts"]) == 4


@pytest.mark.parametrize("label", ["../escape", "a/b", "", "."])
def test_prompt_set_rejects_unsafe_labels(tmp_path, label):
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps(
            {
                "concepts": [
                    {"label": label, "style": "line", "subjects": ["a tree"]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="labels"):
        read_prompt_set(path)


@pytest.mark.parametrize("ridge", [0, -0.1, float("nan"), float("inf")])
def test_settings_reject_invalid_ridge(ridge):
    with pytest.raises(ValueError, match="ridge must be finite and positive"):
        Settings(ridge=ridge).validate()


def test_partial_rejects_different_ridge(tmp_path):
    partial = tmp_path / "run.partial.safetensors"
    model = Model()
    _solve(model, partial=partial)
    with pytest.raises(ValueError, match="partial file settings differ: ridge"):
        _solve(model, partial=partial, settings=replace(_settings(), ridge=0.1))


def test_partial_without_ridge_starts_fresh(tmp_path, monkeypatch):
    partial = tmp_path / "run.partial.safetensors"
    model = Model()
    _set_change(model)
    samples = _samples(count=1)
    settings = replace(_settings(), ridge=0.1)
    old = _solve(model, samples)
    del old.metadata["ridge"]
    contrast._save_partial(partial, old, 2)
    original = contrast.fit_mount
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["ridge"])
        return original(*args, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    fresh = _solve(model, samples, partial=partial, settings=settings)
    assert calls == [0.1] * 20
    assert fresh.metadata["ridge"] == 0.1
    uninterrupted = _solve(model, samples, settings=settings)
    assert fresh.metadata == uninterrupted.metadata
    for name in fresh.mounts:
        assert torch.equal(fresh.mounts[name].down, uninterrupted.mounts[name].down)
        assert torch.equal(fresh.mounts[name].up, uninterrupted.mounts[name].up)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_replay_batch(value):
    with pytest.raises(ValueError, match="replay_batch"):
        replace(_settings(), replay_batch=value).validate()


def _assert_replay_equal(actual, expected):
    assert actual.mounts.keys() == expected.mounts.keys()
    for name, solution in actual.mounts.items():
        torch.testing.assert_close(
            solution.delta(), expected.mounts[name].delta(), atol=2e-6, rtol=2e-5
        )
    actual_metadata = copy.deepcopy(actual.metadata)
    expected_metadata = copy.deepcopy(expected.metadata)
    for metadata in (actual_metadata, expected_metadata):
        metadata.pop("replay_batch")
    assert actual_metadata.pop("block_gaps") == pytest.approx(
        expected_metadata.pop("block_gaps"), abs=2e-6, rel=2e-5
    )
    actual_fit, expected_fit = actual_metadata.pop("fit"), expected_metadata.pop("fit")
    assert actual_fit.keys() == expected_fit.keys()
    for name, record in actual_fit.items():
        assert record == pytest.approx(expected_fit[name], abs=2e-6, rel=2e-5)
    assert actual_metadata == expected_metadata


@pytest.mark.parametrize("sparse", [False, True])
def test_replay_batches_match_single_samples_and_remain_batched(sparse):
    model = Model()
    _set_change(model)
    samples = _samples(count=10)
    mounts = ["transformer_blocks.1.attn1.to_q"] if sparse else None
    expected = _solve(
        copy.deepcopy(model), samples, settings=replace(_settings(), replay_batch=1), mounts=mounts
    )
    actual = _solve(model, samples, settings=replace(_settings(), replay_batch=8), mounts=mounts)
    _assert_replay_equal(actual, expected)
    index = 1 if sparse else 0
    assert {x.shape[0] for _, x in model.transformer_blocks[index].inputs} == {8, 2}
    if not sparse:
        assert {x.shape[0] for _, x in model.transformer_blocks[1].inputs} == {8, 2}


def test_replay_partitions_different_caption_lengths(monkeypatch):
    samples = _samples(count=4)
    for index, sample in enumerate(samples):
        for call in (sample.target, sample.student):
            call["encoder_hidden_states"] = call["encoder_hidden_states"][:, : 5 + index % 2]
    model = Model()
    _set_change(model)
    seen = []
    original = contrast._replay_batches

    def record(captured, limit):
        batches = original(captured, limit)
        seen.extend(batches)
        return batches

    monkeypatch.setattr(contrast, "_replay_batches", record)
    pairs = [("first", "first"), ("second", "second")]
    expected = _solve(
        copy.deepcopy(model), samples, pairs=pairs, settings=replace(_settings(), replay_batch=1)
    )
    seen.clear()
    actual = _solve(model, samples, pairs=pairs, settings=replace(_settings(), replay_batch=8))
    assert len(seen) == 2
    assert {target[1]["encoder_hidden_states"].shape[1] for target, _ in seen} == {5, 6}
    assert all(target[0][0].shape[0] == 2 for target, _ in seen)
    _assert_replay_equal(actual, expected)


def test_replay_shared_conditioning_requires_equal_values():
    samples = _samples(count=2)
    captured = []
    for index, sample in enumerate(samples):
        sides = []
        for call in (sample.target, sample.student):
            sides.append(
                (
                    (call["hidden_states"],),
                    {
                        "conditioning": {"shared": torch.full((2, 4), float(index))},
                        "context": call["encoder_hidden_states"],
                    },
                )
            )
        captured.append(tuple(sides))
    assert contrast._replay_batches(captured, 8) == captured
    for target, student in captured:
        target[1]["conditioning"]["shared"].zero_()
        student[1]["conditioning"]["shared"].zero_()
    batches = contrast._replay_batches(captured, 8)
    assert len(batches) == 1
    for call in batches[0]:
        assert call[0][0].shape == (2, 12, 4)
        assert call[1]["context"].shape == (2, 12, 4)
        assert call[1]["conditioning"]["shared"].shape == (2, 4)
        assert torch.equal(call[1]["conditioning"]["shared"], torch.zeros(2, 4))


@pytest.mark.parametrize("difference", ["position", "key", "scalar", "dtype", "shape"])
def test_replay_call_structure_must_match_on_both_sides(difference):
    x = torch.ones(1, 3, 4)
    call = ((x,), {"context": x.clone(), "scale": 1})
    other = copy.deepcopy(call)
    if difference == "position":
        other = ((), {"hidden_states": x, **other[1]})
    elif difference == "key":
        other[1]["extra"] = None
    elif difference == "scalar":
        other[1]["scale"] = 2
    elif difference == "dtype":
        other[1]["context"] = x.double()
    else:
        other[1]["context"] = x[:, :2]
    captured = [(call, call), (call, other)]
    assert len(contrast._replay_batches(captured, 8)) == 2


def test_observe_stops_after_last_mount_and_matches_full_forward(monkeypatch):
    model = Model()
    _set_change(model)
    samples = _samples()
    block = model.transformer_blocks[0]
    names = ["attn1.to_q", "attn1.to_k", "attn1.to_v"]
    seen = []
    handle = block.ff.net[2].register_forward_hook(lambda *args: seen.append(True))
    try:
        contrast._observe(
            block,
            (
                (samples[0].target["hidden_states"],),
                {
                    "encoder_hidden_states": samples[0].target["encoder_hidden_states"],
                    "target": True,
                },
            ),
            names,
            dict(block.named_modules()),
        )
        assert seen == []
    finally:
        handle.remove()
    actual = _solve(copy.deepcopy(model), samples)

    def full_observe(block, inputs, names, modules):
        observed, handles = {}, []
        try:
            for name in names:

                def record(module, args, output, name=name):
                    observed[name] = (args[0].detach().float(), output.detach().float())

                handles.append(modules[name].register_forward_hook(record))
            contrast._run_block(block, inputs)
        finally:
            for handle in handles:
                handle.remove()
        return observed

    monkeypatch.setattr(contrast, "_observe", full_observe)
    expected = _solve(copy.deepcopy(model), samples)
    _assert_replay_equal(actual, expected)


class GroupBlock(nn.Module):
    def __init__(self, dependent):
        super().__init__()
        self.first, self.later = Linear(4), Linear(4)
        self.first.target_delta[0, 1] = 0.3
        self.later.target_delta[1, 2] = 0.2
        self.dependent = dependent

    def forward(self, hidden_states, encoder_hidden_states, target=False):
        self.first.target = self.later.target = target
        earlier = self.first(hidden_states)
        later = self.later(earlier if self.dependent else encoder_hidden_states)
        return earlier + later


@pytest.mark.parametrize("dependent", [False, True])
def test_group_merge_requires_direct_block_arguments(monkeypatch, dependent):
    model = Model()
    model.transformer_blocks = nn.ModuleList([GroupBlock(dependent)])
    names = ["transformer_blocks.0.first", "transformer_blocks.0.later"]
    monkeypatch.setattr(contrast, "resolve_mounts", lambda model, mounts: names)
    monkeypatch.setattr(contrast, "_GROUPS", (("first",), ("later",)))
    passes, statistics = [], []
    observe, fit = contrast._observe, contrast.fit_mount

    def record_observe(block, inputs, names, modules):
        passes.append(list(names))
        return observe(block, inputs, names, modules)

    def record_fit(c, s, rank, **kwargs):
        statistics.append((c.clone(), s.clone()))
        return fit(c, s, rank, **kwargs)

    monkeypatch.setattr(contrast, "_observe", record_observe)
    monkeypatch.setattr(contrast, "fit_mount", record_fit)
    actual = _solve(copy.deepcopy(model), settings=replace(_settings(), replay_batch=8))
    if dependent:
        assert passes == [[names[0]], [names[0]], [names[1]], [names[1]]]
    else:
        assert passes == [names, names]
    actual_statistics = statistics.copy()
    statistics.clear()
    monkeypatch.setattr(contrast, "_merge_groups", lambda groups, direct: groups)
    expected = _solve(model, settings=replace(_settings(), replay_batch=8))
    _assert_replay_equal(actual, expected)
    for actual_pair, expected_pair in zip(actual_statistics, statistics, strict=True):
        for actual_tensor, expected_tensor in zip(actual_pair, expected_pair, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)


class SharedConditionBlock(Block):
    def forward(self, hidden_states, encoder_hidden_states, target=False, shared=None):
        x = hidden_states + shared.mean(dim=0)
        return super().forward(x, encoder_hidden_states, target=target)


class SharedConditionModel(Model):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([SharedConditionBlock(4), SharedConditionBlock(4)])

    def forward(self, hidden_states, encoder_hidden_states, target=False, shared=None):
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, target=target, shared=shared)
        return (x,)


def test_differing_shared_conditioning_solve_falls_back_to_single_samples(monkeypatch):
    model = SharedConditionModel()
    _set_change(model)
    samples = _samples(count=4)
    for index, sample in enumerate(samples):
        for call in (sample.target, sample.student):
            call["shared"] = torch.full((2, 4), float(index) / 10)
    batches = []
    original = contrast._replay_batches

    def record(captured, limit):
        result = original(captured, limit)
        batches.append(len(result))
        return result

    monkeypatch.setattr(contrast, "_replay_batches", record)
    expected = _solve(copy.deepcopy(model), samples, settings=replace(_settings(), replay_batch=1))
    actual = _solve(model, samples, settings=replace(_settings(), replay_batch=8))
    assert batches == [4, 4]
    _assert_replay_equal(actual, expected)


def test_replay_group_probe_checks_the_student_chain(monkeypatch):
    class ConditionalGroupBlock(GroupBlock):
        def forward(self, hidden_states, encoder_hidden_states, target=False):
            self.dependent = not target
            return super().forward(hidden_states, encoder_hidden_states, target=target)

    model = Model()
    model.transformer_blocks = nn.ModuleList([ConditionalGroupBlock(False)])
    names = ["transformer_blocks.0.first", "transformer_blocks.0.later"]
    monkeypatch.setattr(contrast, "resolve_mounts", lambda model, mounts: names)
    monkeypatch.setattr(contrast, "_GROUPS", (("first",), ("later",)))
    seen = []
    original = contrast._merge_groups

    def merge(groups, direct):
        seen.append(set(direct))
        return original(groups, direct)

    monkeypatch.setattr(contrast, "_merge_groups", merge)
    _solve(model)
    assert seen == [{names[0]}]


def test_replay_unbatched_hidden_states_remain_separate():
    x = torch.ones(3, 4)
    inputs = ((x,), {})
    assert len(contrast._replay_batches([(inputs, inputs), (inputs, inputs)], 8)) == 2


def test_replay_batches_carry_through_unselected_blocks():
    model = Model()
    model.transformer_blocks.append(Block(4))
    _set_change(model)
    mounts = ["transformer_blocks.0.attn1.to_q", "transformer_blocks.2.attn1.to_q"]
    samples = _samples(count=4)
    expected = _solve(
        copy.deepcopy(model),
        samples,
        mounts=mounts,
        settings=replace(_settings(), replay_batch=1),
    )
    actual = _solve(model, samples, mounts=mounts, settings=replace(_settings(), replay_batch=8))
    _assert_replay_equal(actual, expected)
    assert all(x.shape[0] == 4 for block in model.transformer_blocks for _, x in block.inputs)


_MOUNT = "transformer_blocks.0.attn1.to_q"


def _concept_model(delta):
    model = Model()
    model.transformer_blocks[0].attn1.to_q.target_delta.copy_(delta)
    return model


def _solve_set(deltas, *, overlap_penalty=0.0, deflate=False, rank=1, samples=None):
    samples = _samples() if samples is None else samples
    calls = []

    def solve_one(label, coupling):
        calls.append((label, coupling))
        return _solve(
            _concept_model(deltas[label]),
            samples,
            settings=_settings(rank),
            coupling=coupling,
        )

    results = contrast.solve_concepts(
        list(deltas), solve_one, overlap_penalty=overlap_penalty, deflate=deflate
    )
    return results, calls


def _overlap(results, first, second):
    p1 = contrast.input_basis(results[first].mounts[_MOUNT])
    p2 = contrast.input_basis(results[second].mounts[_MOUNT])
    return float(torch.linalg.matrix_norm(p1.T @ p2))


def _overlapping_deltas():
    a = torch.zeros(4, 4)
    a[0, 1] = 0.3
    b = torch.zeros(4, 4)
    b[2, 1], b[2, 2] = 0.3, 0.3
    return {"line": a, "wash": b}


def test_overlap_penalty_zero_is_bit_identical_to_separate_solves():
    deltas = _overlapping_deltas()
    samples = _samples()
    results, calls = _solve_set(deltas, samples=samples)
    assert [label for label, _ in calls] == ["line", "wash"]
    assert all(coupling is None for _, coupling in calls)
    for label, delta in deltas.items():
        plain = _solve(_concept_model(delta), samples, settings=_settings(1))
        assert results[label].metadata == plain.metadata
        for name, solution in plain.mounts.items():
            assert torch.equal(results[label].mounts[name].down, solution.down)
            assert torch.equal(results[label].mounts[name].up, solution.up)
        assert "overlap_penalty" not in plain.metadata and "deflate" not in plain.metadata


def test_overlap_penalty_reduces_input_basis_overlap():
    deltas = _overlapping_deltas()
    plain, _ = _solve_set(deltas)
    penalized, calls = _solve_set(deltas, overlap_penalty=0.25)
    assert [label for label, _ in calls] == ["line", "wash", "line", "wash"]
    assert calls[0][1] is None and calls[1][1] is None
    assert calls[2][1].overlap_penalty == 0.25 and calls[2][1].overlap_round == 1
    before = _overlap(plain, "line", "wash")
    after = _overlap(penalized, "line", "wash")
    assert before > 0.1
    assert after < 0.75 * before
    for label in deltas:
        record = penalized[label].metadata
        assert record["overlap_penalty"] == 0.25
        assert record["overlap_round"] == 1
        assert record["concept_order"] == ["line", "wash"]
        assert "deflate" not in record


def test_overlap_penalty_scales_to_the_gram_trace():
    other = contrast.MountSolution(torch.tensor([[0.0, 2.0, 0.0]]), torch.ones(1, 3), 0.1, 1.0)
    coupling = contrast.ConceptCoupling(overlap_penalty=0.5, others=[{"m": other}])
    c = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    s = torch.ones(3, 3)
    adjusted, same = coupling.adjust("m", c, s)
    expected = c.double().clone()
    expected[1, 1] += 0.5 * 6.0
    assert torch.allclose(adjusted, expected)
    assert same is s
    unchanged = coupling.adjust("other", c, s)
    assert unchanged[0] is c and unchanged[1] is s


def test_deflation_fits_the_residual_after_earlier_concepts_in_order():
    first = torch.zeros(4, 4)
    first[0, 1] = 0.3
    extra = torch.zeros(4, 4)
    extra[2, 3] = 0.2
    deltas = {"line": first, "wash": first + extra}
    results, calls = _solve_set(deltas, deflate=True, rank=2)
    assert calls[0][1].earlier == [] and len(calls[1][1].earlier) == 1
    line = results["line"].mounts[_MOUNT].delta()
    wash = results["wash"].mounts[_MOUNT].delta()
    assert torch.allclose(line, first, atol=2e-4, rtol=2e-3)
    assert torch.allclose(wash, extra, atol=1e-3, rtol=1e-2)
    assert torch.allclose(line + wash, first + extra, atol=1e-3)
    assert results["wash"].metadata["block_gaps"]["1"] < 1e-3
    assert results["wash"].metadata["deflate"] is True
    assert results["wash"].metadata["concept_order"] == ["line", "wash"]
    reverse, _ = _solve_set({"wash": first + extra, "line": first}, deflate=True, rank=2)
    assert torch.allclose(reverse["wash"].mounts[_MOUNT].delta(), first + extra, atol=1e-3)
    assert torch.allclose(reverse["line"].mounts[_MOUNT].delta(), -extra, atol=1e-3)
    plain, _ = _solve_set(deltas, rank=2)
    assert torch.allclose(plain["wash"].mounts[_MOUNT].delta(), first + extra, atol=1e-3)


def test_coupling_options_round_trip_through_the_file(tmp_path):
    deltas = _overlapping_deltas()
    results, _ = _solve_set(deltas, overlap_penalty=0.5, deflate=True)
    path = tmp_path / "wash.safetensors"
    write_lora(path, results["wash"])
    with safe_open(str(path), framework="pt") as file:
        record = json.loads(file.metadata()["kilora_directions"])
    assert record["mode"] == "joint"
    assert record["overlap_penalty"] == 0.5
    assert record["overlap_round"] == 1
    assert record["deflate"] is True
    assert record["concept_order"] == ["line", "wash"]


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "1"])
def test_coupling_rejects_invalid_overlap_penalty(value):
    with pytest.raises(ValueError, match="overlap_penalty"):
        contrast.solve_concepts(["a", "b"], lambda *a: None, overlap_penalty=value)


def test_cli_passes_the_coupling_options(tmp_path, monkeypatch):
    path = tmp_path / "set.json"
    concepts = [
        {"label": label, "style": label + " style", "cue": label, "subjects": ["a cat"]}
        for label in ("line", "wash")
    ]
    path.write_text(json.dumps({"concepts": concepts}))
    calls = []

    def fake_extract(*args, **kwargs):
        calls.append((args[1].name, kwargs.get("coupling")))
        return contrast.ContrastResult({}, {})

    monkeypatch.setattr("kilora.anima.load_anima", lambda *args: SimpleNamespace())
    monkeypatch.setattr(contrast, "model_sha256", lambda path: "toy")
    monkeypatch.setattr(contrast, "extract", fake_extract)
    model_folder = tmp_path / "model"
    model_folder.mkdir()
    base = ["directions", "contrast", "--model", str(model_folder), "--out", str(tmp_path)]
    base += ["--set", str(path), "--steps", "4", "--device", "cpu"]
    assert cli_main([*base, "--overlap-penalty", "0.5", "--deflate"]) == 0
    assert [name for name, _ in calls] == ["line.safetensors", "wash.safetensors"] * 2
    assert calls[0][1].deflate and not calls[0][1].overlap_penalty
    assert calls[3][1].overlap_penalty == 0.5 and len(calls[3][1].earlier) == 1
    calls.clear()
    assert cli_main(base) == 0
    assert [coupling for _, coupling in calls] == [None, None]
    pair = ["directions", "contrast", "--model", "toy", "--out", str(tmp_path / "x.safetensors")]
    with pytest.raises(SystemExit):
        cli_main([*pair, "--pair", "a", "b", "--deflate"])


def _mean_settings(**kwargs):
    return replace(_settings(), contrast_mode="mean", **kwargs)


def test_settings_reject_unknown_contrast_mode():
    assert Settings().contrast_mode == "tokens"
    Settings(contrast_mode="mean").validate()
    with pytest.raises(ValueError, match="contrast_mode"):
        Settings(contrast_mode="other").validate()


def test_mean_contrast_statistics_match_direct_computation(monkeypatch):
    model = Model()
    delta = _set_change(model)
    samples = _samples(count=4)
    seen = []
    real = contrast.fit_mount

    def record(c, s, rank, **kwargs):
        seen.append((c.clone(), s.clone()))
        return real(c, s, rank, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    result = _solve(model, samples, settings=_mean_settings())
    assert result.metadata["contrast_mode"] == "mean"
    inputs = torch.cat([sample.student["hidden_states"].reshape(-1, 4) for sample in samples])
    target_y = inputs + F.linear(inputs, delta)
    mean = target_y.mean(0) - inputs.mean(0)
    with contrast._matmul_allow_tf32(_settings().stats_precision == "tf32"):
        gram = inputs.T @ inputs
        cross = torch.outer(mean, inputs.sum(0))
    assert torch.allclose(seen[0][0], gram, atol=1e-5)
    assert torch.allclose(seen[0][1], cross, atol=1e-5)


def test_mean_contrast_writes_rank_one_changes():
    model = Model()
    _set_change(model)
    result = _solve(model, settings=_mean_settings())
    nonzero = 0
    for solution in result.mounts.values():
        values = torch.linalg.svdvals(solution.up.double().T @ solution.down.double())
        if values[0] > 0:
            nonzero += 1
            assert bool((values[1:] < 1e-6 * values[0]).all())
    assert nonzero


def test_mean_contrast_with_equal_outputs_writes_zero_changes():
    result = _solve(Model(), settings=_mean_settings())
    assert len(result.mounts) == 20
    for solution in result.mounts.values():
        assert torch.count_nonzero(solution.delta()) == 0


def test_tokens_mode_statistics_are_unchanged(monkeypatch):
    model = Model()
    delta = _set_change(model)
    samples = _samples(count=1)
    seen = []
    real = contrast.fit_mount

    def record(c, s, rank, **kwargs):
        seen.append((c.clone(), s.clone()))
        return real(c, s, rank, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    explicit = _solve(model, samples, settings=replace(_settings(), contrast_mode="tokens"))
    hidden = samples[0].student["hidden_states"].float()
    weight = model.transformer_blocks[0].attn1.to_q.weight
    y = F.linear(hidden, weight)
    target_y = F.linear(hidden, weight) + F.linear(hidden, delta)
    x = hidden.reshape(-1, 4)
    difference = (target_y - y).reshape(-1, 4)
    with contrast._matmul_allow_tf32(_settings().stats_precision == "tf32"):
        gram = x.T @ x
        cross = difference.T @ x
    assert torch.equal(seen[0][0], gram)
    assert torch.equal(seen[0][1], cross)
    default = _solve(model, samples)
    assert explicit.mounts.keys() == default.mounts.keys()
    for name, solution in explicit.mounts.items():
        assert torch.equal(solution.down, default.mounts[name].down)
        assert torch.equal(solution.up, default.mounts[name].up)


def test_partial_rejects_different_contrast_mode(tmp_path):
    partial = tmp_path / "run.partial.safetensors"
    model = Model()
    _solve(model, partial=partial)
    with pytest.raises(ValueError, match="partial file settings differ: contrast_mode"):
        _solve(model, partial=partial, settings=_mean_settings())


def test_settings_reject_unknown_statistics_and_fit_dtype():
    assert Settings().stats_precision == "tf32"
    assert Settings().fit_dtype == "float32"
    Settings().validate()
    with pytest.raises(ValueError, match="stats_precision"):
        Settings(stats_precision="other").validate()
    with pytest.raises(ValueError, match="fit_dtype"):
        Settings(fit_dtype="float16").validate()


def test_default_precision_stays_near_the_exact_fit():
    model = Model()
    _set_change(model)
    samples = _samples()
    default = _solve(model, samples)
    exact = _solve(
        model,
        samples,
        settings=replace(_settings(), stats_precision="float32", fit_dtype="float64"),
    )
    difference = reference = 0.0
    for name, solution in exact.mounts.items():
        change = solution.delta()
        other = default.mounts[name].delta()
        difference += float((change - other).square().sum())
        reference += float(change.square().sum())
    assert difference**0.5 <= 1e-3 * max(reference, 1e-30) ** 0.5


def test_shared_inputs_use_one_gram(monkeypatch):
    grams, crosses, products = [], [], []
    real_fit = contrast.fit_mount
    real_gram = contrast._input_gram

    def record_fit(c, s, rank, **kwargs):
        grams.append(c)
        crosses.append(s)
        return real_fit(c, s, rank, **kwargs)

    def record_gram(x):
        products.append(torch.backends.cuda.matmul.allow_tf32)
        return real_gram(x)

    monkeypatch.setattr(contrast, "fit_mount", record_fit)
    monkeypatch.setattr(contrast, "_input_gram", record_gram)
    mounts = [
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn1.to_k",
        "transformer_blocks.0.attn1.to_v",
        "transformer_blocks.0.ff.net.2",
    ]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _solve(Model(), _samples(count=2), mounts=mounts)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert grams[0] is grams[1] is grams[2]
    assert grams[3] is not grams[0]
    assert crosses[0] is not crosses[1]
    assert crosses[1] is not crosses[2]
    assert len(products) == 2
    assert all(products)
    assert torch.backends.cuda.matmul.allow_tf32 is previous


def test_float32_statistics_leave_tf32_off(monkeypatch):
    products = []
    real_gram = contrast._input_gram

    def record_gram(x):
        products.append(torch.backends.cuda.matmul.allow_tf32)
        return real_gram(x)

    monkeypatch.setattr(contrast, "_input_gram", record_gram)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        _solve(
            Model(),
            _samples(count=1),
            mounts=["transformer_blocks.0.attn1.to_q"],
            settings=replace(_settings(), stats_precision="float32"),
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert len(products) == 1 and products[0] is False
    assert torch.backends.cuda.matmul.allow_tf32 is previous


def test_partial_rejects_other_precision(tmp_path):
    partial = tmp_path / "run.partial.safetensors"
    _solve(Model(), partial=partial)
    with pytest.raises(ValueError, match="stats_precision"):
        _solve(Model(), partial=partial, settings=replace(_settings(), stats_precision="float32"))
    with pytest.raises(ValueError, match="fit_dtype"):
        _solve(Model(), partial=partial, settings=replace(_settings(), fit_dtype="float64"))


def _observed(x, y, target_y, weights, mode):
    names = ["mount"]

    def observe(_target, _student):
        return {names[0]: (x, target_y)}, {names[0]: (x, y)}

    item = ("target", "student")
    if weights is not None:
        item = contrast._WeightedPair(*item, weights)
    statistics = contrast._group_statistics(names, [item], observe, lambda tensor: tensor, mode)
    return statistics["mount"]


def test_unit_weights_match_unweighted_statistics():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(2, 5, 3, generator=generator)
    y = torch.randn(2, 5, 3, generator=generator)
    target_y = y + torch.randn(2, 5, 3, generator=generator)
    weights = torch.ones(2, 5)
    for mode in ("tokens", "mean"):
        plain = _observed(x, y, target_y, None, mode)
        weighted = _observed(x, y, target_y, weights, mode)
        torch.testing.assert_close(weighted[0], plain[0])
        torch.testing.assert_close(weighted[1], plain[1])


def test_zero_weights_leave_those_rows_out_of_the_statistics():
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(2, 4, 3, generator=generator)
    y = torch.randn(2, 4, 3, generator=generator)
    target_y = y + 0.4
    weights = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    kept = weights.reshape(-1) > 0
    flat = x.reshape(-1, 3)
    difference = (target_y - y).reshape(-1, 3)
    gram = flat[kept].T @ flat[kept]
    cross = difference[kept].T @ flat[kept]
    weighted_gram, weighted_cross = _observed(x, y, target_y, weights, "tokens")
    torch.testing.assert_close(weighted_gram, gram)
    torch.testing.assert_close(weighted_cross, cross)
    full_gram, full_cross = _observed(x, y, target_y, None, "tokens")
    assert not torch.allclose(weighted_gram, full_gram)
    assert not torch.allclose(weighted_cross, full_cross)


def test_masked_mean_matches_a_direct_weighted_mean():
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(2, 4, 3, generator=generator)
    y = torch.randn(2, 4, 3, generator=generator)
    target_y = y + torch.randn(2, 4, 3, generator=generator)
    weights = torch.tensor([[0.0, 0.5, 1.0, 0.25], [1.0, 0.0, 0.5, 1.0]])
    scale = weights.reshape(-1, 1)
    flat = x.reshape(-1, 3)
    target_rows = target_y.reshape(-1, 3)
    student_rows = y.reshape(-1, 3)
    count = weights.sum()
    mean = ((target_rows - student_rows) * scale).sum(0) / count
    expected_cross = torch.outer(mean, (flat * scale).sum(0))
    expected_gram = (flat * scale).T @ flat
    gram, cross = _observed(x, y, target_y, weights, "mean")
    torch.testing.assert_close(gram, expected_gram)
    torch.testing.assert_close(cross, expected_cross)


def test_weights_survive_replay_batching():
    left = torch.randn(1, 4, 3)
    right = torch.randn(1, 4, 3)
    first = ((left,), {"context": left})
    second = ((right,), {"context": right})
    weights = [torch.tensor([[1.0, 0.0, 0.5, 1.0]]), torch.tensor([[0.2, 0.4, 0.6, 0.8]])]
    batches = contrast._replay_batches([(first, first), (second, second)], 8, weights)
    assert len(batches) == 1
    assert batches[0][0][0][0].shape[0] == 2
    torch.testing.assert_close(batches[0].weights, torch.cat(weights, dim=0))
    single = contrast._replay_batches([(first, first)], 8, weights[:1])
    assert len(single) == 1
    torch.testing.assert_close(single[0].weights, weights[0])


def test_weighted_replay_matches_single_samples():
    model = Model()
    _set_change(model)
    samples = _samples(count=4)
    generator = torch.Generator().manual_seed(9)
    for sample in samples:
        sample.weights = torch.rand(1, 12, generator=generator).clamp_min(0.05)
    mounts = ["transformer_blocks.0.attn1.to_q"]
    expected = _solve(
        copy.deepcopy(model),
        samples,
        mounts=mounts,
        settings=replace(_settings(), replay_batch=1),
    )
    actual = _solve(model, samples, mounts=mounts, settings=replace(_settings(), replay_batch=8))
    _assert_replay_equal(actual, expected)


@pytest.mark.parametrize("ones", [False, True])
def test_weighted_image_components_use_per_image_means(ones):
    from kilora.library import contrast

    generator = torch.Generator().manual_seed(735)
    captured, expected, identities = [], {}, []
    for image in (0, 1, 0, 1):
        x, student, target = [torch.randn(1, 3, 2, generator=generator).double() for _ in range(3)]
        w = torch.ones(1, 3).double() if ones else torch.tensor([[0.01, 0.1, 0.2]]).double()
        captured.append(contrast._WeightedPair(target, (x, student), w))
        identities.append(image)
        record = [
            w.sum(),
            (target * w[..., None]).sum((0, 1)),
            (student * w[..., None]).sum((0, 1)),
            (x * w[..., None]).sum((0, 1)),
        ]
        if image not in expected:
            expected[image] = record
        else:
            expected[image] = [a + b for a, b in zip(expected[image], record, strict=True)]

    def observe(target, student):
        return {"m": (None, target)}, {"m": student}

    gram, cross = contrast._group_statistics(
        ["m"], captured, observe, lambda x: x, "mean", 1, identities
    )["m"]
    totals = [sum(row[i] for row in expected.values()) for i in range(4)]
    means = torch.stack([(r[1] - r[2]) / r[0] for r in expected.values()])
    mean = (totals[1] - totals[2]) / totals[0]
    centered = means - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:1].T @ vh[:1]
    direct = torch.outer(mean, totals[3])
    for residual, row in zip(projected, expected.values(), strict=True):
        direct += torch.outer(residual, row[3])
    torch.testing.assert_close(cross, direct)
    if ones:
        plain = [(p[0], p[1]) for p in captured]
        c, s = contrast._group_statistics(
            ["m"], plain, observe, lambda x: x, "mean", 1, identities
        )["m"]
        torch.testing.assert_close(c, gram, rtol=0, atol=0)
        torch.testing.assert_close(s, cross, rtol=0, atol=0)


@pytest.mark.parametrize("patch", [1, 2])
@pytest.mark.parametrize("mode", ["tokens", "mean"])
def test_spatial_mask_matches_patchified_model_rows(patch, mode):
    from kilora.library import contrast

    class PatchModel(torch.nn.Module):
        def forward(self, latent):
            pooled = torch.nn.functional.avg_pool2d(latent, patch)
            return pooled.flatten(2).transpose(1, 2)

    latent = torch.arange(2 * 3 * 8 * 12).reshape(2, 3, 8, 12).float() / 100
    x = PatchModel()(latent)
    weights = torch.zeros(2, 1, 8, 12)
    weights[:, :, :4] = 1
    per_sample = (8 // patch) * (12 // patch)
    expected = torch.cat([torch.ones(per_sample // 2), torch.zeros(per_sample // 2)])
    expected = expected.repeat(2)
    rows = contrast._token_row_weights(weights, x)
    torch.testing.assert_close(rows, expected)
    torch.testing.assert_close(contrast._token_row_weights(weights, x.reshape(-1, 3)), expected)
    student = x * 0.2
    target = student + x * 0.3
    gram, cross = _observed(x, student, target, weights, mode)
    direct = _observed(x, student, target, expected.reshape(2, per_sample), mode)
    torch.testing.assert_close(gram, direct[0])
    torch.testing.assert_close(cross, direct[1])


def test_spatial_mask_rejects_incompatible_block_grid():
    from kilora.library import contrast

    with pytest.raises(ValueError, match="batch"):
        contrast._token_row_weights(torch.ones(2, 1, 8, 12), torch.ones(1, 7, 3))
    with pytest.raises(ValueError, match="token grid"):
        contrast._token_row_weights(
            torch.ones(1, 1, 8, 12), torch.ones(1, 999, 3), require_image=True
        )


@pytest.mark.parametrize("mode", ["tokens", "mean"])
def test_spatial_mask_leaves_cross_attention_text_statistics_unweighted(mode):
    from kilora.library import contrast

    class CrossAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_k = torch.nn.Linear(4, 3, bias=False)

        def forward(self, text):
            return self.to_k(text)

    model = CrossAttention()
    text = torch.arange(2 * 512 * 4).reshape(2, 512, 4).float() / 1000
    student = model(text)
    target = student.clone()
    weights = torch.zeros(2, 1, 96, 96)
    weights[:, :, :48] = 1
    assert contrast._token_row_weights(weights, text) is None

    def observe(_target, _student):
        return {"attn2.to_k": (text, target)}, {"attn2.to_k": (text, student)}

    masked = contrast._WeightedPair("target", "student", weights)
    actual = contrast._group_statistics(
        ["attn2.to_k"], [masked], observe, lambda value: value, mode
    )["attn2.to_k"]
    expected = contrast._group_statistics(
        ["attn2.to_k"], [("target", "student")], observe, lambda value: value, mode
    )["attn2.to_k"]
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)
