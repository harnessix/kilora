# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_directions_contrast import Model, Scheduler, _samples, _settings
from torch.nn import functional as F

from kilora.cli import main as cli_main
from kilora.library import contrast
from kilora.library.contrast import EffectSource, PromptPair, extract, solve, trajectory_samples
from kilora.presets import anima_name

Q = "transformer_blocks.0.attn1.to_q"
K = "transformer_blocks.0.attn2.to_k"
V = "transformer_blocks.0.attn2.to_v"
SECOND_Q = "transformer_blocks.1.attn1.to_q"


def _file(tmp_path, names=(Q, K, V, SECOND_Q), width=4, license="Apache-2.0", gain=1.0):
    tensors, expected = {}, {}
    for i, name in enumerate(names):
        down = torch.zeros(2, width)
        down[0, 0], down[1, 1] = 1.0, 1.0
        up = torch.zeros(width, 2)
        up[0, 0], up[1, 1] = (0.2 + i * 0.02) * gain, -0.12 * gain
        alpha = 3.0
        stem = "lora_unet_" + anima_name(name).replace(".", "_")
        tensors[stem + ".lora_down.weight"] = down
        tensors[stem + ".lora_up.weight"] = up
        tensors[stem + ".alpha"] = torch.tensor(alpha)
        expected[name] = (up @ down) * (alpha / 2)
    path = tmp_path / "source.safetensors"
    save_file(tensors, str(path), metadata={"modelspec.license": license})
    return path, expected


def _solve(model, source=None, *, samples=None, pairs=None, **kwargs):
    return solve(
        model,
        _samples() if samples is None else samples,
        [("same", "same")] if pairs is None else pairs,
        settings=_settings(),
        model_sha="toy-model",
        license="base-license",
        target_source=source,
        **kwargs,
    )


def _no_hooks(model):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_source_changes_only_target_and_restores_base(tmp_path):
    model = Model()
    path, expected = _file(tmp_path, names=(Q,))
    original = copy.deepcopy(model.state_dict())
    sample = _samples(count=1)[0]
    before = model(**sample.student)[0].detach().clone()
    source = EffectSource.load(model, path, strength=0.7)
    module = dict(model.named_modules())[Q]
    x = sample.student["hidden_states"]
    plain = module(x)
    with source.applied(model):
        assert torch.allclose(module(x), plain + F.linear(x, expected[Q] * 0.7))
        target = model(**sample.target)[0]
        assert not torch.equal(before, target)
    assert torch.equal(before, model(**sample.student)[0])
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    _no_hooks(model)


@pytest.mark.parametrize("strength", [0.7, -0.4, 0.0])
def test_rank_two_file_is_recovered_on_every_routed_mount(tmp_path, strength):
    model = Model()
    path, expected = _file(tmp_path)
    source = EffectSource.load(model, path, strength)
    sample = _samples(count=1)[0]
    base = model(**sample.student)[0].detach().clone()
    result = _solve(model, source)
    assert set(result.mounts) == set(expected)
    assert result.metadata["excluded_mounts"] == {}
    for name, delta in expected.items():
        assert torch.allclose(result.mounts[name].delta(), delta * strength, atol=4e-4, rtol=0.004)
    assert max(result.metadata["block_gaps"].values()) < 8e-4
    assert torch.equal(base, model(**sample.student)[0])
    _no_hooks(model)


def test_different_captions_exclude_cross_keys_and_values(tmp_path):
    model = Model()
    path, _ = _file(tmp_path)
    source = EffectSource.load(model, path)
    samples = _samples()
    for sample in samples:
        sample.target["encoder_hidden_states"] = sample.target["encoder_hidden_states"][:, :5]
    result = _solve(model, source, samples=samples, pairs=[("trigger subject", "subject")])
    assert set(result.mounts) == {Q, SECOND_Q}
    assert set(result.metadata["excluded_mounts"]) == {K, V}
    assert all("captions differ" in text for text in result.metadata["excluded_mounts"].values())
    _no_hooks(model)


@pytest.mark.parametrize("license", ["CC-BY-ND-4.0", "No derivatives allowed", "No-Merge"])
def test_no_derivatives_license_refused_before_sampling(tmp_path, monkeypatch, license):
    path, _ = _file(tmp_path, license=license)
    out = tmp_path / "effect.safetensors"

    def unexpected(*args, **kwargs):
        pytest.fail("a restricted source must be refused before sampling or fitting")

    monkeypatch.setattr(contrast, "trajectory_samples", unexpected)
    monkeypatch.setattr(contrast, "solve", unexpected)
    with pytest.raises(ValueError, match=license):
        extract(
            "unused",
            out,
            [("same", "same")],
            pipe=SimpleNamespace(transformer=Model()),
            model_sha="toy",
            settings=_settings(),
            lora=path,
            progress=lambda text: None,
        )
    assert not out.exists()
    assert not out.with_name(out.name + ".partial.safetensors").exists()


def test_metadata_tracks_source_strength_prompts_and_license(tmp_path):
    model = Model()
    path, _ = _file(tmp_path)
    source = EffectSource.load(model, path, 0.35)
    pairs = [PromptPair("first", "first"), PromptPair("second", "second")]
    result = _solve(model, source, pairs=pairs)
    out = tmp_path / "effect.safetensors"
    contrast.write_lora(out, result)
    with safe_open(str(out), framework="pt") as file:
        metadata = json.loads(file.metadata()["kilora_directions"])
    assert metadata["source"] == {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "strength": 0.35,
    }
    assert metadata["license"] == "Apache-2.0"
    assert metadata["target_source"] == "lora"
    assert metadata["prompts"] == [
        {"target": "first", "student": "first"},
        {"target": "second", "student": "second"},
    ]


def test_statistics_include_every_prompt_sample(tmp_path, monkeypatch):
    model = Model()
    path, expected = _file(tmp_path, names=(Q,))
    source = EffectSource.load(model, path)
    samples = _samples(count=6)
    seen = []
    real = contrast.fit_mount

    def record(c, s, rank, **kwargs):
        seen.append((c.clone(), s.clone()))
        return real(c, s, rank, **kwargs)

    monkeypatch.setattr(contrast, "fit_mount", record)
    _solve(model, source, samples=samples, pairs=[("first", "first"), ("second", "second")])
    c, s = torch.zeros(4, 4), torch.zeros(4, 4)
    with contrast._matmul_allow_tf32(_settings().stats_precision == "tf32"):
        for sample in samples:
            x = sample.student["hidden_states"].reshape(-1, 4)
            c += x.T @ x
            s += F.linear(x, expected[Q] * 0.7).T @ x
    assert len(seen) == 1
    assert torch.allclose(seen[0][0], c, atol=1e-6)
    assert torch.allclose(seen[0][1], s, atol=3e-6)


def _pipeline(monkeypatch):
    def encode(pipe, caption, device):
        return torch.full((3, 16), 0.12 if caption else -0.08)

    monkeypatch.setattr("kilora.anima.encode_caption", encode)
    monkeypatch.setattr("kilora.train.swap_to_denoiser", lambda pipe, dev: None)
    return SimpleNamespace(transformer=Model(width=16), scheduler=Scheduler())


def test_cfg_trajectory_uses_source_for_both_branches(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    path, _ = _file(tmp_path, names=(Q,), width=16, gain=20.0)
    pairs = [PromptPair("subject", "subject"), PromptPair("another", "another")]
    settings = _settings()
    source = EffectSource.load(pipe.transformer, path)
    base = trajectory_samples(pipe, pairs, settings, "cpu")
    actual = trajectory_samples(pipe, pairs, settings, "cpu", target_source=source)
    with source.applied(pipe.transformer):
        expected = trajectory_samples(pipe, pairs, settings, "cpu")
    assert len(actual) == len(pairs) * len(settings.seeds) * len(settings.sample_steps)
    for got, wanted in zip(actual, expected, strict=True):
        assert torch.equal(got.target["hidden_states"], wanted.target["hidden_states"])
        assert torch.equal(got.target["hidden_states"], got.student["hidden_states"])
    assert not torch.equal(base[1].target["hidden_states"], actual[1].target["hidden_states"])
    _no_hooks(pipe.transformer)


def test_effect_extract_exports_source_record_and_removes_checkpoint(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    path, _ = _file(tmp_path, names=(Q, K, V), width=16)
    out = tmp_path / "effect.safetensors"
    result = extract(
        "unused",
        out,
        [("subject", "subject")],
        lora=path,
        strength=0.4,
        pipe=pipe,
        model_sha="toy",
        settings=_settings(),
        device="cpu",
        progress=lambda text: None,
    )
    assert out.exists()
    assert result.metadata["source"]["strength"] == 0.4
    assert set(result.mounts) == {Q, K, V}
    assert not out.with_name(out.name + ".partial.safetensors").exists()
    _no_hooks(pipe.transformer)


def test_unused_effect_preserves_plain_extraction_bits(tmp_path):
    model = Model()
    samples = _samples()
    before = model(**samples[0].student)[0].detach().clone()
    baseline = _solve(model)
    path, _ = _file(tmp_path)
    source = EffectSource.load(model, path)
    _solve(model, source)
    after = _solve(model)
    assert after.metadata == baseline.metadata
    for name in baseline.mounts:
        assert torch.equal(after.mounts[name].down, baseline.mounts[name].down)
        assert torch.equal(after.mounts[name].up, baseline.mounts[name].up)
    assert torch.equal(before, model(**samples[0].student)[0])
    _no_hooks(model)


def test_interrupt_cleans_hooks_and_resume_checks_source(tmp_path):
    model = Model()
    path, _ = _file(tmp_path)
    partial = tmp_path / "effect.partial.safetensors"
    source = EffectSource.load(model, path)

    def stop(message):
        raise InterruptedError("stop after the first block")

    with pytest.raises(InterruptedError):
        _solve(model, source, partial=partial, progress=stop)
    _no_hooks(model)
    with pytest.raises(ValueError, match="source"):
        _solve(model, EffectSource.load(model, path, 0.3), partial=partial)
    resumed = _solve(model, source, partial=partial)
    fresh = _solve(model, source)
    gaps = resumed.metadata.pop("block_gaps"), fresh.metadata.pop("block_gaps")
    fits = resumed.metadata.pop("fit"), fresh.metadata.pop("fit")
    assert resumed.metadata == fresh.metadata
    assert gaps[0].keys() == gaps[1].keys()
    for key in gaps[1]:
        assert gaps[0][key] == pytest.approx(gaps[1][key], rel=1e-4)
    assert fits[0].keys() == fits[1].keys()
    for name, record in fits[1].items():
        assert fits[0][name].keys() == record.keys()
        for field, value in record.items():
            assert fits[0][name][field] == pytest.approx(value, rel=1e-6)
    for name in fresh.mounts:
        torch.testing.assert_close(
            resumed.mounts[name].delta(),
            fresh.mounts[name].delta(),
            atol=2e-6,
            rtol=2e-5,
        )


def test_cli_pairs_prompts_and_defaults_strength(monkeypatch):
    calls = []
    monkeypatch.setattr(contrast, "extract", lambda *args, **kwargs: calls.append((args, kwargs)))
    command = [
        "directions",
        "effect",
        "--model",
        "local-model",
        "--lora",
        "source.safetensors",
        "--out",
        "effect.safetensors",
        "--prompt",
        "first",
        "--prompt",
        "second",
    ]
    assert cli_main(command) == 0
    args, options = calls[-1]
    assert args[2] == [PromptPair("first", "first"), PromptPair("second", "second")]
    assert options["strength"] == 0.7
    assert options["lora"] == "source.safetensors"
    paired = command[:-4] + [
        "--prompt",
        "first",
        "--student-prompt",
        "plain first",
        "--prompt",
        "second",
        "--student-prompt",
        "plain second",
        "--strength",
        "0.4",
    ]
    assert cli_main(paired) == 0
    args, options = calls[-1]
    assert args[2] == [PromptPair("first", "plain first"), PromptPair("second", "plain second")]
    assert options["strength"] == 0.4
    assert cli_main(command + ["--student-prompt", "plain second"]) == 0
    assert calls[-1][0][2] == [PromptPair("first", "first"), PromptPair("second", "plain second")]
    calls.clear()
    with pytest.raises(SystemExit):
        cli_main(paired + ["--student-prompt", "duplicate"])
    assert not calls


@pytest.mark.parametrize("strength", [float("nan"), float("inf")])
def test_nonfinite_strength_refused(tmp_path, strength):
    path, _ = _file(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        EffectSource.load(Model(), path, strength)


def test_source_cannot_fit_an_unrouted_mount(tmp_path):
    model = Model()
    path, _ = _file(tmp_path, names=(Q,))
    source = EffectSource.load(model, path)
    with pytest.raises(ValueError, match="routed mounts"):
        _solve(model, source, mounts=[K])
    _no_hooks(model)
