# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_learned_cross import _model, _set

from kilora import Settings, load_mod
from kilora.adapter import KiLoRA
from kilora.train import _enable_training_cross, build_parser, run, train_loop


def test_cross_trains_on_routing_objective_and_roundtrips(tmp_path):
    model = _model()
    adapter = KiLoRA(model, [_set()], Settings(depth=1))
    for param in adapter.parameters():
        param.requires_grad_(False)
    adapter.log_taus["s0"].requires_grad_(True)
    section = _enable_training_cross(adapter, 4, 5)
    x = torch.tensor([[[1.0, 1.0, 0.2, -0.3], [0.5, 1.0, 0.4, 0.2]]])
    with torch.no_grad():
        section.data["bias"].copy_(torch.tensor([0.9, 0.0, 0.0, 0.0]))
        target = model.layer(x).detach()
        section.data["bias"].fill_(0.01)
    initial = (model.layer(x) - target).square().mean().item()

    def loss(_batch, _generator):
        return (model.layer(x) - target).square().mean()

    baseline_model = _model()
    baseline = KiLoRA(baseline_model, [_set()], Settings(depth=1))
    for param in baseline.parameters():
        param.requires_grad_(False)
    baseline.log_taus["s0"].requires_grad_(True)

    def baseline_loss(_batch, _generator):
        return (baseline_model.layer(x) - target).square().mean()

    train_loop(list(baseline.trainable_parameters()), [x], baseline_loss, steps=30, lr=0.05)
    result = train_loop(list(adapter.trainable_parameters()), [x], loss, steps=30, lr=0.05)
    final = loss(None, None).item()
    assert final < initial * 0.5
    assert final < baseline_loss(None, None).item()
    assert result.losses[-1] < initial
    assert section.data["up"].detach().abs().sum() > 0
    path = tmp_path / "trained.kmod"
    adapter.save(path, sections=[section])
    mod = load_mod(path)
    assert len(mod.sections) == 1
    for key in section.data:
        torch.testing.assert_close(mod.sections[0].data[key], section.data[key].detach())
    adapter.remove()
    baseline.remove()


def test_option_is_off_and_rejects_non_static_modes():
    parser = build_parser()
    args = parser.parse_args(["--model", "tiny"])
    assert not args.learned_cross
    for extra in (["--method", "lora"], ["--build-keys", "out"]):
        args = parser.parse_args(["--model", "tiny", "--learned-cross", *extra])
        with pytest.raises(ValueError, match="requires static KiLoRA training"):
            run(args)


def test_static_trainer_saves_cross_and_checkpoints(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kilora.anima as anima
    import kilora.data as data
    import kilora.train as train

    original_cross = train._enable_training_cross

    def enable_cross(adapter, *args):
        with torch.no_grad():
            adapter.log_taus["s0"].fill_(-3.0)
        return original_cross(adapter, *args)

    monkeypatch.setattr(train, "_enable_training_cross", enable_cross)
    model = _model()
    pipe = SimpleNamespace(transformer=model)
    x = torch.tensor([[[1.0, 0.8, 0.2, -0.3], [0.5, 1.0, 0.4, 0.2]]])
    monkeypatch.setattr(anima, "is_anima", lambda _: False)
    monkeypatch.setattr(data, "image_folder", lambda path: path)
    monkeypatch.setattr(data, "data_source", lambda *args: {})
    monkeypatch.setattr(train, "find_images", lambda _: [None])
    monkeypatch.setattr(train, "prepare_examples", lambda *args: [x])
    monkeypatch.setattr(train, "swap_to_denoiser", lambda *args: None)
    monkeypatch.setattr(
        train,
        "flow_matching_loss",
        lambda denoiser, batch, gen, dtype: (
            (denoiser.layer(batch[0]) - 2 * batch[0]).square().mean()
        ),
    )
    path = tmp_path / "cross.kmod"
    args = build_parser().parse_args(
        [
            "--model",
            "tiny",
            "--data",
            str(tmp_path),
            "--output",
            str(path),
            "--static",
            "--learned-cross",
            "--target-modules",
            "layer",
            "--steps",
            "4",
            "--save-every",
            "2",
            "--save-precision",
            "fp32",
            "--precision",
            "fp32",
            "--num-experts",
            "2",
            "--routes",
            "1",
            "--rank",
            "1",
            "--router-dim",
            "4",
            "--train-depth",
            "1",
            "1",
            "--lr",
            "0.03",
            "--device",
            "cpu",
        ]
    )
    result = run(args, pipe=pipe)
    assert len(result.losses) == 4
    final = load_mod(path)
    checkpoint = load_mod(tmp_path / "cross.step2.kmod.safetensors")
    assert [s.kind for s in final.sections] == ["learned-cross"]
    assert [s.kind for s in checkpoint.sections] == ["learned-cross"]
    assert not torch.equal(final.sections[0].data["bias"], torch.full((4,), 0.01))
    assert not torch.equal(final.sections[0].data["bias"], checkpoint.sections[0].data["bias"])


def test_disabled_matches_base_and_trained_cross_matches_loop(monkeypatch):
    from kilora import compute

    x = torch.tensor([[[1.0, 0.8, 0.2, -0.3], [0.5, 1.0, 0.4, 0.2]]])
    first_model, second_model = _model(), _model()
    first = KiLoRA(first_model, [_set()], Settings(depth=1))
    second = KiLoRA(second_model, [_set()], Settings(depth=1))
    with torch.no_grad(), compute.eager():
        assert torch.equal(first_model.layer(x), second_model.layer(x))
        section = _enable_training_cross(first, 4, 5)
        section.data["bias"].copy_(torch.tensor([0.5, 0.2, -0.1, 0.3]))
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", True)
        fast = first_model.layer(x)
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", False)
        loop = first_model.layer(x)
        torch.testing.assert_close(fast, loop)
    first.remove()
    second.remove()
