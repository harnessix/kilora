# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

pytest.importorskip("diffusers")
pytest.importorskip("peft")
from test_train import data, pipeline

from kilora import load_mod
from kilora.train import TRAIN_PRESETS, apply_preset, build_parser, run


def _args(*extra, data_dir="d", out="o.kmod"):
    return build_parser().parse_args(
        ["--model", "tiny", "--data", str(data_dir), "--output", str(out), *extra]
    )


def test_standard_preset_keeps_the_old_defaults():
    args = apply_preset(_args())
    assert (args.steps, args.rank, args.num_experts, args.hooks) == (2000, 1, 16, "attention")


@pytest.mark.parametrize("name", list(TRAIN_PRESETS))
def test_each_preset_fills_open_settings(name):
    args = apply_preset(_args("--preset", name))
    for key, value in TRAIN_PRESETS[name].items():
        assert getattr(args, key) == value


def test_explicit_flags_win_over_the_preset():
    args = apply_preset(_args("--preset", "strong", "--steps", "7", "--rank", "3"))
    assert (args.steps, args.rank) == (7, 3)
    assert args.num_experts == 16 and args.hooks == "full"


def _train(data, tmp_path, *extra):
    out = tmp_path / "p.kmod"
    common = ["--resolution", "32", "--lr", "1e-2", "--precision", "fp32", "--device", "cpu"]
    common += ["--router-dim", "8", "--static", "--license", "CC0-1.0"]
    run(_args(*common, *extra, data_dir=data, out=out), pipe=pipeline())
    return load_mod(out)


def test_quick_preset_trains_a_static_mod(data, tmp_path):
    mod = _train(data, tmp_path, "--preset", "quick", "--steps", "3")
    (s,) = mod.sets
    assert not mod.manifest.get("live")
    assert all("attention" in t for t in s.experts)
    assert all(ex.up.shape[0] == 8 for ex in s.experts.values())
    from kilora.card import read_card

    card = read_card(tmp_path / "p.kmod")
    assert card["data_source"]["images"] == 5 and card["license"] == "CC0-1.0"


def test_full_hooks_add_the_feed_forward_layers(data, tmp_path):
    mod = _train(data, tmp_path, "--hooks", "full", "--steps", "2", "--num-experts", "4")
    targets = set(mod.sets[0].experts)
    assert any("feed_forward" in t for t in targets)
    assert any("attention" in t for t in targets)


def test_presets_recommend_depth_one_and_strength_one():
    for name in TRAIN_PRESETS:
        args = apply_preset(_args("--preset", name))
        assert (args.default_depth, args.default_strength) == (1, 1.0)
    args = apply_preset(_args("--default-depth", "2", "--default-strength", "0.6"))
    assert (args.default_depth, args.default_strength) == (2, 0.6)


def test_recommended_values_reach_the_mod_and_its_card(data, tmp_path):
    from kilora.card import read_card

    mod = _train(data, tmp_path, "--preset", "quick", "--steps", "2")
    assert mod.manifest["depth"]["recommended"] == 1
    assert mod.manifest["strength"]["recommended"] == 1.0
    mod = _train(
        data, tmp_path, "--steps", "2", "--default-depth", "2", "--default-strength", "0.7"
    )
    assert mod.manifest["depth"]["recommended"] == 2
    card = read_card(tmp_path / "p.kmod")
    assert card["depth"] == 2 and card["strength"]["recommended"] == 0.7


def test_training_refuses_one_expert(data, tmp_path):
    from kilora.errors import KiLoRAError

    with pytest.raises(KiLoRAError, match="at least 2 experts per layer"):
        _train(data, tmp_path, "--num-experts", "1", "--steps", "1")


def test_training_can_store_int8(data, tmp_path):
    mod = _train(data, tmp_path, "--steps", "2", "--num-experts", "2", "--storage", "int8-block")
    assert "storage.int8-block" in mod.manifest["requires_features"]
    from kilora.card import read_card

    assert read_card(tmp_path / "p.kmod")["id"]
