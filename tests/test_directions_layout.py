# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from test_directions_contrast import Attention, Model, Scheduler, _samples, _set_change, _settings
from torch import nn

from kilora.library import contrast
from kilora.library.contrast import EffectSource, PromptPair, extract, fit_mount, solve, write_lora
from kilora.library.layout import LayoutProcessor, PlainLayout


class Processor:
    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        self.calls += 1
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        probabilities = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(-1)
        return attn.to_out(probabilities @ v)


class ProcessedAttention(Attention):
    def __init__(self, width):
        super().__init__(width)
        self.heads = 1
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.processor = Processor()

    def set_processor(self, processor):
        self.processor = processor

    def forward(self, x, context=None):
        return self.processor(self, x, encoder_hidden_states=context)


def _model(width=4):
    model = Model(width)
    for block in model.transformer_blocks:
        block.attn1 = ProcessedAttention(width)
        block.attn2 = ProcessedAttention(width)
    return model


def _solve(model, settings=None, **kwargs):
    return solve(
        model,
        _samples(count=2),
        [("target", "student")],
        settings=settings or _settings(),
        model_sha="toy",
        license="Apache-2.0",
        **kwargs,
    )


def test_plain_maps_are_replayed_and_cross_attention_keeps_its_processor():
    model = _model()
    layout = PlainLayout(model)
    originals = [(b.attn1.processor, b.attn2.processor) for b in model.transformer_blocks]
    layout.install()
    x = torch.eye(4).unsqueeze(0)
    try:
        for block, (_, cross) in zip(model.transformer_blocks, originals, strict=True):
            with layout.active("record"):
                plain = block.attn1(x)
            recorded = block.attn1.processor.maps[0].squeeze(1).clone()
            with torch.no_grad():
                block.attn1.to_q.weight.mul_(3)
            ordinary = block.attn1.processor.original(block.attn1, x)
            assert not torch.equal(ordinary, plain)
            with layout.active("replay"):
                target = block.attn1(x)
            assert torch.equal(target, recorded)
            assert torch.equal(target, plain)
            assert block.attn2.processor is cross
            expected = cross(block.attn2, x, encoder_hidden_states=x * 2)
            with layout.active("record"):
                actual = block.attn2(x, x * 2)
            assert torch.equal(actual, expected)
    finally:
        layout.close()
    for block, (self_processor, cross) in zip(model.transformer_blocks, originals, strict=True):
        assert block.attn1.processor is self_processor
        assert block.attn2.processor is cross


def test_replay_uses_target_values_and_rejects_missing_maps():
    attention = ProcessedAttention(4)
    processor = LayoutProcessor(attention.processor)
    x = torch.eye(4).unsqueeze(0)
    processor.mode = "record"
    plain = processor(attention, x)
    with torch.no_grad():
        attention.to_v.weight.mul_(2)
    processor.mode = "replay"
    assert torch.equal(processor(attention, x), plain * 2)
    with pytest.raises(ValueError, match="no matching student map"):
        processor(attention, x)


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("effect", [False, True])
def test_solver_installs_all_blocks_and_always_restores_processors(interrupt, effect):
    model = _model()
    delta = _set_change(model)
    source = None
    if effect:
        source = EffectSource(
            {
                "transformer_blocks.0.attn1.to_q": contrast.MountSolution(
                    torch.eye(4),
                    delta.T,
                    0,
                    1,
                ),
            },
            {"file": "toy"},
            "Apache-2.0",
        )
    originals = [b.attn1.processor for b in model.transformer_blocks]
    settings = replace(_settings(), layout_source="plain")

    def progress(message):
        if interrupt:
            raise RuntimeError("stop")

    if interrupt:
        with pytest.raises(RuntimeError, match="stop"):
            _solve(model, settings, target_source=source, progress=progress)
    else:
        result = _solve(model, settings, target_source=source, progress=progress)
        assert result.metadata["layout_source"] == "plain"
    for block, original in zip(model.transformer_blocks, originals, strict=True):
        assert block.attn1.processor is original
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_default_files_match_the_original_single_mount_solve(tmp_path, monkeypatch):
    model = Model()
    _set_change(model)
    samples = _samples(count=2)
    c = s = None
    mount = model.transformer_blocks[0].attn1.to_q
    for sample in samples:
        x = sample.student["hidden_states"]
        mount.target = True
        target = mount(x).detach().float()
        mount.target = False
        actual = mount(x).detach().float()
        x = x.reshape(-1, 4)
        with contrast._matmul_allow_tf32(_settings().stats_precision == "tf32"):
            gram = x.T @ x
            cross = (target - actual).reshape(-1, 4).T @ x
        c = gram if c is None else c + gram
        s = cross if s is None else s + cross
    name = "transformer_blocks.0.attn1.to_q"
    expected = fit_mount(
        c, s, _settings().rank, ridge=_settings().ridge, fit_dtype=_settings().fit_dtype
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("default solve must not install layout processors")

    monkeypatch.setattr(PlainLayout, "__init__", unexpected)
    result = solve(
        model,
        samples,
        [("same", "same")],
        settings=_settings(),
        mounts=[name],
        model_sha="toy",
        license="Apache-2.0",
    )
    legacy = contrast.ContrastResult({name: expected}, {})
    paths = [tmp_path / "legacy.safetensors", tmp_path / "default.safetensors"]
    write_lora(paths[0], legacy)
    write_lora(paths[1], result)
    before, after = (load_file(str(path)) for path in paths)
    assert before.keys() == after.keys()
    for key in before:
        if key.endswith(".lora_down.weight"):
            up = key.replace(".lora_down.weight", ".lora_up.weight")
            torch.testing.assert_close(
                after[up] @ after[key], before[up] @ before[key], atol=2e-6, rtol=2e-5
            )
    assert result.metadata["layout_source"] == "target"
    assert result.metadata["refit_rounds"] == 0


@pytest.mark.parametrize("effect", [False, True])
@pytest.mark.parametrize("layout_source", ["target", "plain"])
@pytest.mark.parametrize("rounds", [1, 2])
def test_refit_samples_solved_student_and_keeps_each_file(
    tmp_path,
    monkeypatch,
    effect,
    layout_source,
    rounds,
):
    pipe = SimpleNamespace(transformer=_model(width=16), scheduler=Scheduler())
    monkeypatch.setattr(
        "kilora.anima.encode_caption",
        lambda pipe, caption, dev: torch.full((3, 16), 0.1 if caption == "target" else 0.0),
    )
    monkeypatch.setattr("kilora.train.swap_to_denoiser", lambda pipe, dev: None)
    original = contrast.trajectory_samples
    calls = []

    def sample(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(contrast, "trajectory_samples", sample)
    kwargs = {}
    if effect:
        name = "transformer_blocks.0.attn1.to_q"
        source = EffectSource(
            {
                name: contrast.MountSolution(
                    torch.eye(16),
                    torch.eye(16) * 0.1,
                    0,
                    1,
                )
            },
            {"file": "toy"},
            "Apache-2.0",
        )
        monkeypatch.setattr(EffectSource, "load", lambda *args: source)
        kwargs["lora"] = "unused"
    path = tmp_path / "direction.safetensors"
    result = extract(
        "unused",
        path,
        [PromptPair("target", "student")],
        settings=replace(_settings(), refit_rounds=rounds, layout_source=layout_source),
        device="cpu",
        pipe=pipe,
        model_sha="toy",
        progress=lambda message: None,
        **kwargs,
    )
    assert len(calls) == rounds + 1
    for index, call in enumerate(calls[1:]):
        solved = call["student_change"]
        assert solved.mounts and solved.metadata["refit_round"] == index
        assert "target_source" not in call
    paths = sorted(tmp_path.glob("*.safetensors"))
    assert len(paths) == rounds + 1
    assert path in paths
    for index in range(1, rounds + 1):
        assert (tmp_path / f"direction.round-{index}.safetensors") in paths
    assert result.metadata["refit_round"] == rounds
    for output in paths:
        with safe_open(str(output), framework="pt") as file:
            record = json.loads(file.metadata()["kilora_directions"])
        assert record["refit_rounds"] == rounds
        assert record["layout_source"] == layout_source


def test_student_sampler_uses_solved_change_and_plain_caption(monkeypatch):
    pipe = SimpleNamespace(transformer=Model(width=16), scheduler=Scheduler())
    captions = {"target": 0.8, "student": 0.1, "": 0.0}
    monkeypatch.setattr(
        "kilora.anima.encode_caption",
        lambda pipe, caption, dev: torch.full((3, 16), captions[caption]),
    )
    monkeypatch.setattr("kilora.train.swap_to_denoiser", lambda pipe, dev: None)
    name = "transformer_blocks.0.attn1.to_q"
    solved = contrast.ContrastResult(
        {
            name: contrast.MountSolution(
                torch.eye(16),
                torch.eye(16) * 0.2,
                0,
                1,
            )
        },
        {},
    )
    pairs = [PromptPair("target", "student")]
    actual = contrast.trajectory_samples(pipe, pairs, _settings(), "cpu", student_change=solved)
    with EffectSource(solved.mounts, {}, "").applied(pipe.transformer):
        expected = contrast.trajectory_samples(
            pipe,
            [PromptPair("student", "student")],
            _settings(),
            "cpu",
        )
    base = contrast.trajectory_samples(pipe, pairs, _settings(), "cpu")
    assert torch.equal(actual[1].student["hidden_states"], expected[1].student["hidden_states"])
    assert not torch.equal(actual[1].student["hidden_states"], base[1].student["hidden_states"])
    assert torch.equal(
        actual[0].target["encoder_hidden_states"],
        torch.full((1, 3, 16), 0.8, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("command", ["contrast", "effect"])
def test_cli_passes_layout_and_refit_settings(monkeypatch, command):
    calls = []
    monkeypatch.setattr(contrast, "extract", lambda *args, **kwargs: calls.append(kwargs))
    inputs = (
        ["--pair", "target", "student"]
        if command == "contrast"
        else [
            "--lora",
            "source.safetensors",
            "--prompt",
            "target",
        ]
    )
    assert (
        contrast.main(
            [
                command,
                "--model",
                "unused",
                "--out",
                "out.safetensors",
                *inputs,
                "--layout-source",
                "plain",
                "--refit-rounds",
                "1",
            ]
        )
        == 0
    )
    assert calls[0]["settings"].layout_source == "plain"
    assert calls[0]["settings"].refit_rounds == 1


@pytest.mark.parametrize("changes", [{"layout_source": "other"}, {"refit_rounds": -1}])
def test_invalid_settings_are_refused(changes):
    with pytest.raises(ValueError):
        replace(_settings(), **changes).validate()


def test_plain_layout_refuses_unsupported_attention_without_mutation():
    model = _model()
    original = model.transformer_blocks[0].attn1.processor
    model.transformer_blocks[1].attn1 = Attention(4)
    with pytest.raises(ValueError, match="processor support"):
        _solve(model, replace(_settings(), layout_source="plain"))
    assert model.transformer_blocks[0].attn1.processor is original


def test_layout_settings_are_checked_when_resuming(tmp_path):
    model = _model()
    partial = tmp_path / "partial.safetensors"
    _solve(model, partial=partial)
    with pytest.raises(ValueError, match="layout_source"):
        _solve(model, replace(_settings(), layout_source="plain"), partial=partial)


def test_plain_layout_is_validated_before_sampling_or_writing(tmp_path, monkeypatch):
    pipe = SimpleNamespace(transformer=Model(width=16), scheduler=Scheduler())

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported layout must be refused before sampling")

    monkeypatch.setattr(contrast, "trajectory_samples", unexpected)
    with pytest.raises(ValueError, match="processor support"):
        extract(
            "unused",
            tmp_path / "out.safetensors",
            [("target", "student")],
            settings=replace(_settings(), layout_source="plain"),
            pipe=pipe,
            model_sha="toy",
            device="cpu",
            progress=lambda message: None,
        )
    assert not list(tmp_path.iterdir())
