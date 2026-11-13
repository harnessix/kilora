# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_directions_contrast import Model, _samples, _settings
from test_directions_effect import Q, _file

from kilora.cli import main as cli_main
from kilora.comfyui.trigger_words import trigger_words_of
from kilora.library import bake, contrast
from kilora.library.index import LibraryIndex


@pytest.fixture
def setup_bake(tmp_path, monkeypatch):
    model = Model()
    pipe = SimpleNamespace(transformer=model)
    index = LibraryIndex.open(tmp_path / "index")
    source, _ = _file(tmp_path, names=(Q,))
    calls = {"pairs": [], "solve": 0, "sources": []}
    real_solve = contrast.solve

    def trajectories(pipe, pairs, settings, device, target_source):
        calls["pairs"].append(pairs)
        calls["sources"].append(target_source)
        return _samples(count=1)

    def solve(*args, **kwargs):
        calls["solve"] += 1
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(contrast, "trajectory_samples", trajectories)
    monkeypatch.setattr(contrast, "solve", solve)
    options = {
        "settings": _settings(),
        "model_sha": "toy-base",
        "index": index,
        "pipe": pipe,
        "device": "cpu",
    }
    return source, options, calls


def _add_triggers(source):
    from safetensors.torch import load_file

    weights = load_file(str(source))
    save_file(
        weights,
        str(source),
        metadata={
            "modelspec.license": "Apache-2.0",
            "ss_tag_frequency": json.dumps({"a": {"inkcat": 12, "linework": 10, "sketch": 8}}),
        },
    )
    source.with_suffix(".json").write_text(json.dumps({"trainedWords": ["soft lines"]}))


@pytest.mark.parametrize("has_triggers", [False, True])
def test_prompts_metadata_and_unused_model_are_unchanged(setup_bake, has_triggers):
    source, options, calls = setup_bake
    if has_triggers:
        _add_triggers(source)
    words = trigger_words_of(str(source))
    model = options["pipe"].transformer
    sample = _samples(count=1)[0]
    before = model(**sample.student)[0].detach().clone()
    weights = copy.deepcopy(model.state_dict())
    output = bake.bake_file(source, "unused", **options)
    assert output.parent == options["index"].path / "bakes"
    assert trigger_words_of(str(output)) == words
    pairs = calls["pairs"][0]
    assert len(pairs) == 4
    for pair, plain in zip(pairs, bake.DEFAULT_PROMPTS, strict=True):
        assert pair.student == plain
        assert pair.target == (", ".join([*words, plain]) if words else plain)
        assert all(word not in pair.student for word in words)
    assert calls["sources"][0].metadata["strength"] == 0.7
    assert torch.equal(before, model(**sample.student)[0])
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    with safe_open(str(output), framework="pt") as handle:
        metadata = handle.metadata()
    record = json.loads(metadata["kilora_directions"])
    assert record["source"]["strength"] == 0.7
    assert record["model_sha"] == "toy-base"
    assert metadata["modelspec.license"] == "Apache-2.0"
    assert metadata["kilora_bake_key"] == output.stem


def test_default_settings_match_preparation_measurements():
    settings = contrast.Settings()
    assert settings.seeds == (811, 1811)
    assert settings.sample_steps == (0, 7, 14)
    assert settings.steps == 15
    assert settings.cfg == 1.0
    assert settings.size == 768
    assert settings.rank == 16
    assert settings.stats_precision == "tf32"
    assert settings.fit_dtype == "float32"
    assert len(bake.DEFAULT_PROMPTS) == 4


def test_reuse_and_lookup_do_not_call_solver_or_loader(setup_bake, monkeypatch):
    source, options, calls = setup_bake
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(options["index"].path))
    lookup = {"settings": options["settings"]}
    assert bake.baked_path(source, "toy-base", **lookup) is None
    output = bake.bake_file(source, "unused", **options)

    def no_load(*args, **kwargs):
        pytest.fail("a cached bake must not load a model")

    monkeypatch.setattr("kilora.anima.load_anima", no_load)
    assert bake.bake_file(source, "unused", **{**options, "pipe": None}) == output
    assert calls["solve"] == 1
    assert bake.baked_path(source, "toy-base", **lookup) == output
    assert bake.baked_path(source, "different-base", **lookup) is None
    assert bake.baked_path(source, "toy-base") is None
    output.unlink()
    assert bake.baked_path(source, "toy-base", **lookup) is None
    assert bake.baked_path(source.with_name("missing.safetensors"), "toy-base") is None


@pytest.mark.parametrize("change", ["source", "model", "settings", "prompts", "sidecar"])
def test_changed_inputs_rebuild(setup_bake, change):
    source, options, calls = setup_bake
    first = bake.bake_file(source, "unused", **options)
    updated = dict(options)
    if change == "source":
        _file(source.parent, names=(Q,), width=4, license="MIT")
    elif change == "model":
        updated["model_sha"] = "new-base"
    elif change == "settings":
        updated["settings"] = replace(options["settings"], rank=1)
    elif change == "prompts":
        updated["prompts"] = ("a bird beside a tree",)
    else:
        source.with_suffix(".json").write_text(json.dumps({"trainedWords": ["birdstyle"]}))
    second = bake.bake_file(source, "unused", **updated)
    assert first != second
    assert first.exists() and second.exists()
    assert calls["solve"] == 2
    assert (
        bake.baked_path(
            source,
            updated["model_sha"],
            settings=updated["settings"],
            prompts=updated.get("prompts", bake.DEFAULT_PROMPTS),
            index=updated["index"],
        )
        == second
    )


@pytest.mark.parametrize("license", ["CC-BY-ND-4.0", "no derivatives", "no-merge"])
def test_denied_derivatives_skip_without_loading_or_writing(setup_bake, monkeypatch, license):
    source, options, calls = setup_bake
    _file(source.parent, names=(Q,), license=license)
    messages = []

    def no_load(*args, **kwargs):
        pytest.fail("denied files must be skipped before loading")

    monkeypatch.setattr("kilora.anima.load_anima", no_load)
    assert (
        bake.bake_file(
            source,
            "unused",
            **{**options, "pipe": None, "progress": messages.append},
        )
        is None
    )
    assert calls["solve"] == 0
    assert any("Skipped" in message and "derivatives" in message for message in messages)
    assert not options["index"].path.exists()
    assert bake.baked_path(source, "toy-base", index=options["index"]) is None


def test_changed_source_during_solve_is_not_published(setup_bake, monkeypatch):
    source, options, _ = setup_bake
    real = contrast.solve

    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        source.with_suffix(".json").write_text(json.dumps({"trainedWords": ["newword"]}))
        return result

    monkeypatch.setattr(contrast, "solve", changed)
    with pytest.raises(ValueError, match="changed during"):
        bake.bake_file(source, "unused", **options)
    assert not options["index"].path.exists()


def test_failed_solver_leaves_no_cache(setup_bake, monkeypatch):
    source, options, _ = setup_bake

    def interrupted(*args, **kwargs):
        raise RuntimeError("solve interrupted")

    monkeypatch.setattr(contrast, "solve", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        bake.bake_file(source, "unused", **options)
    assert not options["index"].path.exists()


def test_estimate_is_printed_per_file(setup_bake):
    source, options, _ = setup_bake
    messages = []
    bake.bake_file(source, "unused", **options, progress=messages.append)
    assert any(
        str(source) in message and "Estimated GPU memory:" in message for message in messages
    )


def test_cli_files_and_all_are_sequential(tmp_path, monkeypatch):
    index = LibraryIndex.open(tmp_path / "index")
    index.paths = {str(tmp_path / name): name for name in ("first", "second")}
    monkeypatch.setattr(LibraryIndex, "open", lambda path=None: index)
    monkeypatch.setattr(contrast, "model_sha256", lambda path: "base-sha")
    calls = []

    def prepare(source, model, **kwargs):
        calls.append((source, model, kwargs))

    monkeypatch.setattr(bake, "bake_file", prepare)
    assert cli_main(["library", "bake", "--all", "--model", "local-model"]) == 0
    assert [call[0] for call in calls] == sorted(index.paths)
    assert all(call[2]["model_sha"] == "base-sha" for call in calls)
    calls.clear()
    assert (
        cli_main(
            [
                "library",
                "bake",
                "--files",
                "one",
                "two",
                "--model",
                "local-model",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert [call[0] for call in calls] == ["one", "two"]
    assert all(call[2]["device"] == "cpu" for call in calls)


def test_cli_requires_explicit_selection():
    with pytest.raises(SystemExit):
        cli_main(["library", "bake", "--model", "local-model"])


def test_cli_reports_file_errors_and_continues(monkeypatch, capsys):
    monkeypatch.setattr(contrast, "model_sha256", lambda path: "base-sha")
    calls = []

    def prepare(source, model, **kwargs):
        calls.append(source)
        if source == "broken":
            raise ValueError("unsupported file")

    monkeypatch.setattr(bake, "bake_file", prepare)
    assert (
        cli_main(
            [
                "library",
                "bake",
                "--files",
                "broken",
                "other",
                "--model",
                "local-model",
            ]
        )
        == 1
    )
    assert calls == ["broken", "other"]
    assert "broken: unsupported file" in capsys.readouterr().err


def test_default_two_argument_lookup(setup_bake, monkeypatch):
    source, options, calls = setup_bake
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(options["index"].path))
    options.pop("settings")
    output = bake.bake_file(source, "unused", **options)
    assert bake.baked_path(source, "toy-base") == output
    assert calls["solve"] == 1


def test_every_solve_setting_is_part_of_cache_key():
    settings = contrast.Settings()
    pairs = [contrast.PromptPair("trigger, a tree", "a tree")]
    key = bake.cache_key("source", "base", settings, pairs=pairs)
    changes = {
        "seeds": (812,),
        "size": 512,
        "steps": 31,
        "cfg": 3.0,
        "negative_prompt": "blurry",
        "sample_steps": (0, 14),
        "rank": 8,
        "stats_precision": "float32",
        "fit_dtype": "float64",
    }
    assert key == bake.cache_key("source", "base", settings, pairs=pairs)
    assert key != bake.cache_key("new-source", "base", settings, pairs=pairs)
    assert key != bake.cache_key("source", "new-base", settings, pairs=pairs)
    for name, value in changes.items():
        changed = replace(settings, **{name: value})
        changed.validate()
        assert key != bake.cache_key("source", "base", changed, pairs=pairs)


def test_baked_ip_entry_has_trigger_ids(setup_bake):
    from kilora.comfyui.auto_alpha import trigger_ids

    source, options, _ = setup_bake
    _add_triggers(source)
    output = bake.bake_file(source, "unused", **options)
    entry = SimpleNamespace(enabled=True, kind="lora", axis="ip", path=str(output))

    class Clip:
        def tokenize(self, text):
            assert text in ("", "inkcat")
            return {"t5xxl": [[(41, 1.0), (0, 1.0)]] if text else [[(0, 1.0)]]}

    reasons = []
    assert trigger_ids([entry], Clip(), reasons) == {str(output): [41]}
    assert reasons == []


def test_corrupt_cache_is_ignored_and_rebuilt(setup_bake):
    source, options, calls = setup_bake
    output = bake.bake_file(source, "unused", **options)
    output.write_bytes(b"incomplete")
    assert (
        bake.baked_path(
            source,
            "toy-base",
            settings=options["settings"],
            index=options["index"],
        )
        is None
    )
    assert bake.bake_file(source, "unused", **options) == output
    assert calls["solve"] == 2
    assert trigger_words_of(str(output)) == []


def test_failed_atomic_write_does_not_publish_a_bake(setup_bake, monkeypatch):
    source, options, _ = setup_bake

    def broken_save(tensors, filename, **kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"interrupted write")
        raise OSError("disk full")

    monkeypatch.setattr(bake, "save_file", broken_save)
    with pytest.raises(OSError, match="disk full"):
        bake.bake_file(source, "unused", **options)
    assert list((options["index"].path / "bakes").iterdir()) == []


def test_sidecar_trigger_groups_are_part_of_cache_identity(setup_bake):
    source, options, calls = setup_bake
    sidecar = source.with_suffix(".json")
    sidecar.write_text(json.dumps({"trainedWords": ["inkcat", "linework"]}))
    first = bake.bake_file(source, "unused", **options)
    sidecar.write_text(json.dumps({"trainedWords": ["inkcat, linework"]}))
    second = bake.bake_file(source, "unused", **options)
    assert calls["pairs"][0] == calls["pairs"][1]
    assert first != second
    assert trigger_words_of(str(second)) == ["inkcat, linework"]


def test_invalid_source_lookup_returns_none(tmp_path):
    source = tmp_path / "invalid.safetensors"
    source.write_bytes(b"incomplete")
    assert bake.baked_path(source, "base") is None
