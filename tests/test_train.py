# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")
from PIL import Image
from test_diffusers import tiny_transformer
from test_image_parity import tiny_pipeline

from kilora import compose, import_lora, load_mod
from kilora.train import build_parser, run


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    gen = torch.Generator().manual_seed(0)
    for i in range(5):
        pixels = (torch.rand(40, 48, 3, generator=gen) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(folder / f"img{i}.png")
        (folder / f"img{i}.txt").write_text(f"caption {i}")
    return folder


def pipeline():
    pipe = tiny_pipeline(tiny_transformer())

    def encode_prompt(prompt, device=None, do_classifier_free_guidance=False, **_):
        seed = sum(map(ord, prompt if isinstance(prompt, str) else prompt[0]))
        embeds = torch.randn(5, 16, generator=torch.Generator().manual_seed(seed))
        return [embeds], []

    pipe.encode_prompt = encode_prompt
    return pipe


def test_kilora_training_writes_a_loadable_mod(data, tmp_path):
    out = tmp_path / "test.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "6",
            "--lr",
            "1e-2",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--num-experts",
            "4",
            "--router-dim",
            "8",
            "--train-depth",
            "0",
            "3",
            "--last-step-training",
            "--save-every",
            "3",
            "--license",
            "CC0-1.0",
            "--static",
            "--trigger-words",
            "ink style",
            "--example-prompts",
            "a boat, ink style",
            "--default-strength",
            "0.8",
            "--description",
            "Soft ink lines.",
        ]
    )
    result = run(args, pipe=pipeline())
    assert len(result.losses) == 6
    mod = load_mod(out)
    assert mod.manifest["id"] == "test"
    assert mod.manifest["trigger_words"] == ["ink style"]
    assert mod.manifest["example_prompts"] == ["a boat, ink style"]
    assert mod.manifest["strength"]["recommended"] == 0.8
    assert mod.manifest["description"] == "Soft ink lines."
    assert mod.manifest["depth"]["trained"] == [0, 3]
    assert mod.settings.last_step_training
    assert (tmp_path / "test.step3.kmod.safetensors").exists()
    ups = [ex.up for ex in mod.sets[0].experts.values()]
    assert any(u.abs().sum() > 0 for u in ups)


def test_lora_method_writes_an_importable_lora(data, tmp_path):
    out = tmp_path / "plain.safetensors"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "3",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--method",
            "lora",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
        ]
    )
    run(args, pipe=pipeline())
    model = tiny_transformer()
    result = import_lora(out, model)
    names = list(result.adapter_set.experts)
    assert len(names) == 16
    assert all(n.split(".attention.")[1] in ("to_q", "to_k", "to_v", "to_out.0") for n in names)
    compose(model, [result.adapter_set])


def test_without_a_live_trainer_plugin_a_static_mod_is_trained(data, tmp_path, monkeypatch, capsys):
    from kilora import trainers

    monkeypatch.setattr(trainers, "_PLUGINS", {})
    monkeypatch.setattr(trainers, "_DISCOVERED", True)
    out = tmp_path / "s.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "2",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--num-experts",
            "4",
            "--router-dim",
            "8",
        ]
    )
    run(args, pipe=pipeline())
    assert "No live trainer plugin" in capsys.readouterr().err
    assert not load_mod(out).manifest.get("live")


@pytest.mark.parametrize("carry", ["drop", "dropped"])
def test_carry_flag_trains_and_loads(data, tmp_path, carry):
    from kilora.bake import plan_targets
    from kilora.modlist import ModChoice, resolve

    out = tmp_path / "mode.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "3",
            "--lr",
            "1e-2",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--num-experts",
            "4",
            "--router-dim",
            "8",
            "--train-depth",
            "1",
            "3",
            "--license",
            "CC0-1.0",
            "--static",
            "--carry",
            carry,
        ]
    )
    result = run(args, pipe=pipeline())
    assert len(result.losses) == 3 and all(torch.isfinite(torch.tensor(result.losses)))
    mod = load_mod(out)
    assert mod.settings.carry == carry
    plan = resolve([ModChoice(path=out)])
    layer = next(iter(plan_targets(plan.scoped_sets(), 3).values())).sets[0]
    assert layer.carry == carry


@pytest.mark.parametrize("source", ["gradients", "states"])
def test_build_keys_then_train_from_them(data, tmp_path, source):
    from kilora.subspace_keys import load_subspace_keys

    common = [
        "--model",
        "tiny",
        "--data",
        str(data),
        "--resolution",
        "32",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-experts",
        "4",
        "--router-dim",
        "8",
        "--m-in",
        "2",
        "--m-out",
        "3",
        "--license",
        "CC0-1.0",
        "--static",
        "--target-modules",
        "layers.*.feed_forward.w2",
    ]
    first = tmp_path / "first.kmod"
    run(
        build_parser().parse_args([*common, "--output", str(first), "--steps", "2"]),
        pipe=pipeline(),
    )
    keys_file = tmp_path / "keys.safetensors"
    args = [*common, "--output", str(tmp_path / "unused.kmod"), "--steps", "2"]
    args += ["--build-keys", str(keys_file), "--mod", str(first), "--key-source", source]
    run(build_parser().parse_args(args), pipe=pipeline())
    keys, got = load_subspace_keys(keys_file)
    assert got == source and keys
    second = tmp_path / "second.kmod"
    args = [*common, "--output", str(second), "--steps", "1", "--keys-from", str(keys_file)]
    run(build_parser().parse_args(args), pipe=pipeline())
    mod = load_mod(second)
    ex = next(iter(mod.sets[0].experts.values()))
    assert mod.sets[0].key_source == source
    assert ex.keys_in.shape[-1] == ex.down.shape[-1]
    assert ex.keys_out.shape[-1] == ex.up.shape[-1]
    assert ex.keys_in.shape[1] == min(8, ex.down.shape[1] * ex.down.shape[2])
    assert ex.keys_out.shape[1] == min(8, ex.up.shape[1] * ex.up.shape[2])


def test_output_gate_flag_trains_logs_and_loads(data, tmp_path, capsys):
    out = tmp_path / "gate.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "3",
            "--lr",
            "1e-2",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--num-experts",
            "4",
            "--router-dim",
            "8",
            "--train-depth",
            "1",
            "3",
            "--license",
            "CC0-1.0",
            "--static",
            "--output-gate",
            "neuron",
            "--target-modules",
            "layers.*.feed_forward.w2",
            "--gate-log-every",
            "1",
        ]
    )
    result = run(args, pipe=pipeline())
    assert len(result.losses) == 3
    assert "output gate q: mean" in capsys.readouterr().err
    mod = load_mod(out)
    assert mod.settings.output_gate == "neuron"
    assert next(iter(mod.sets[0].experts.values())).gate is not None


def test_time_limit_and_sigterm_stop_and_save(data, tmp_path, monkeypatch):
    import os
    import signal

    from kilora import train as T

    common = [
        "--model",
        "tiny",
        "--data",
        str(data),
        "--resolution",
        "32",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-experts",
        "4",
        "--router-dim",
        "8",
        "--license",
        "CC0-1.0",
        "--static",
        "--steps",
        "50",
    ]
    out = tmp_path / "limit.kmod"
    args = build_parser().parse_args([*common, "--output", str(out), "--time-limit", "0"])
    result = run(args, pipe=pipeline())
    assert result.losses == [] and load_mod(out).manifest["id"] == "limit"

    out = tmp_path / "term.kmod"
    real = T.flow_matching_loss
    calls = [0]

    def loss_then_term(*a, **k):
        calls[0] += 1
        if calls[0] == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*a, **k)

    monkeypatch.setattr(T, "flow_matching_loss", loss_then_term)
    with T.stop_on_sigterm():
        result = run(build_parser().parse_args([*common, "--output", str(out)]), pipe=pipeline())
    assert len(result.losses) == 2 and load_mod(out).manifest["id"] == "term"


def test_static_training_cross_saved_in_checkpoint_and_final(data, tmp_path):
    out = tmp_path / "cross.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--output",
            str(out),
            "--resolution",
            "32",
            "--steps",
            "4",
            "--lr",
            "0.01",
            "--precision",
            "fp32",
            "--save-precision",
            "fp32",
            "--device",
            "cpu",
            "--num-experts",
            "4",
            "--router-dim",
            "8",
            "--train-depth",
            "1",
            "1",
            "--static",
            "--learned-cross",
            "--save-every",
            "2",
            "--target-modules",
            "layers.0.cross_test_projection",
        ]
    )
    pipe = pipeline()
    block = pipe.transformer.layers[0]
    block.cross_test_projection = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        block.cross_test_projection.weight.copy_(torch.eye(32))
    block.feed_forward.register_forward_hook(
        lambda _m, _args, output: block.cross_test_projection(output)
    )
    result = run(args, pipe=pipe)
    assert len(result.losses) == 4
    final = load_mod(out)
    checkpoint = load_mod(tmp_path / "cross.step2.kmod.safetensors")
    assert [s.kind for s in final.sections] == ["learned-cross"]
    assert [s.kind for s in checkpoint.sections] == ["learned-cross"]
    assert not torch.equal(final.sections[0].data["bias"], torch.full((8,), 0.01))
    assert not torch.equal(final.sections[0].data["bias"], checkpoint.sections[0].data["bias"])
