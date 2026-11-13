# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from conftest import TARGET, randomize_up
from safetensors import safe_open
from safetensors.torch import save_file

from kilora import (
    KiLoRAConfig,
    ModFormatError,
    UnsupportedFeatureError,
    attach,
    compose,
    load_mod,
    read_manifest,
)
from kilora.cli import main as cli_main
from kilora.modfile import MANIFEST_KEY, compute_digest


def trained(model, seed=0, name="main"):
    adapter = attach(
        model,
        KiLoRAConfig(target_modules=[TARGET], num_experts=4, routes=2, router_dim=8, seed=seed),
        name=name,
    )
    randomize_up(adapter, seed=seed)
    return adapter


def rewrite_manifest(path, out, change):
    with safe_open(str(path), framework="pt") as f:
        manifest = json.loads(f.metadata()[MANIFEST_KEY])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    change(manifest)
    manifest["digest"] = compute_digest(manifest, tensors)
    save_file(tensors, str(out), {MANIFEST_KEY: json.dumps(manifest)})


def test_round_trip_gives_same_output(model, x, tmp_path):
    adapter = trained(model)
    expected = model(x)
    path = tmp_path / "ink.kmod"
    adapter.save(path, id="ink", name="Ink", license="CC-BY-4.0", base_model={"family": "tiny"})
    adapter.remove()
    loaded = compose(model, [path])
    torch.testing.assert_close(model(x), expected)
    manifest = loaded.metadata
    assert manifest["base_model"] == {"family": "tiny"}
    m = read_manifest(path)
    assert m["format"] == "kilora-mod" and m["format_version"] == "1.0"
    assert m["requires_features"] == ["experts.lora"]
    assert m["depth"] == {"recommended": 1, "trained": [1, 4]}
    assert m["digest"].startswith("sha256-")


def test_composed_file_keeps_set_names_and_splits(model, x, tmp_path):
    a = trained(model, seed=1, name="a")
    set_a = a.to_sets()[0]
    a.remove()
    b = trained(model, seed=2, name="b")
    set_b = b.to_sets()[0]
    b.remove()
    combined = compose(model, [set_a, set_b], strengths=[0.5, -1.0])
    expected = model(x)
    combined.save(tmp_path / "pack.kmod", id="pack")
    combined.remove()
    mod = load_mod(tmp_path / "pack.kmod")
    assert [s.name for s in mod.sets] == ["a", "b"]
    assert [s.strength for s in mod.sets] == [0.5, -1.0]
    torch.testing.assert_close(
        mod.sets[1].experts[mod.sets[1].target_modules[0]].up,
        set_b.experts[set_b.target_modules[0]].up,
    )
    compose(model, [tmp_path / "pack.kmod"])
    torch.testing.assert_close(model(x), expected)


def test_pickle_file_is_refused(tmp_path):
    path = tmp_path / "evil.kmod"
    torch.save({"weights": torch.zeros(3)}, path)
    with pytest.raises(ModFormatError, match="not a safetensors file"):
        load_mod(path)


def test_plain_safetensors_without_manifest_is_refused(tmp_path):
    path = tmp_path / "plain.safetensors"
    save_file({"a": torch.zeros(2)}, str(path))
    with pytest.raises(ModFormatError, match="no KiLoRA manifest"):
        load_mod(path)


def test_unknown_required_feature_is_refused(model, tmp_path):
    trained(model).save(tmp_path / "a.kmod", id="a")
    rewrite_manifest(
        tmp_path / "a.kmod",
        tmp_path / "b.kmod",
        lambda m: m["requires_features"].append("hook.route"),
    )
    with pytest.raises(UnsupportedFeatureError, match="needs the feature hook.route"):
        load_mod(tmp_path / "b.kmod")


def test_newer_major_version_is_refused(model, tmp_path):
    trained(model).save(tmp_path / "a.kmod", id="a")
    rewrite_manifest(
        tmp_path / "a.kmod", tmp_path / "b.kmod", lambda m: m.update(format_version="2.0")
    )
    with pytest.raises(UnsupportedFeatureError, match="mod format 2.0"):
        load_mod(tmp_path / "b.kmod")


def test_newer_minor_version_with_unknown_optional_data_loads(model, x, tmp_path):
    adapter = trained(model)
    expected = model(x)
    adapter.save(tmp_path / "a.kmod", id="a")
    adapter.remove()

    def add_optional(m):
        m["format_version"] = "1.7"
        m["extensions"]["example.org/curves"] = {"points": [0, 1]}
        m["some_future_field"] = True
        m["adapter"]["some_future_setting"] = 3

    rewrite_manifest(tmp_path / "a.kmod", tmp_path / "b.kmod", add_optional)
    compose(model, [tmp_path / "b.kmod"])
    torch.testing.assert_close(model(x), expected)


def test_damaged_file_is_detected(model, tmp_path):
    trained(model).save(tmp_path / "a.kmod", id="a")
    with safe_open(str(tmp_path / "a.kmod"), framework="pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    name = next(k for k in tensors if k.startswith("kilora/router/"))
    tensors[name] = tensors[name] + 1
    save_file(tensors, str(tmp_path / "b.kmod"), meta)
    with pytest.raises(ModFormatError, match="digest"):
        load_mod(tmp_path / "b.kmod")


def test_manifest_limits(model, tmp_path):
    adapter = trained(model)
    with pytest.raises(ModFormatError, match="'id'"):
        adapter.save(tmp_path / "a.kmod", id="Bad Id")
    with pytest.raises(ModFormatError, match="longer than"):
        adapter.save(tmp_path / "a.kmod", id="a", author="x" * 5000)
    with pytest.raises(ValueError, match="set by KiLoRA"):
        adapter.save(tmp_path / "a.kmod", id="a", digest="sha256-0")


def test_previews_round_trip(model, tmp_path):
    from kilora import save_mod

    adapter = trained(model)
    png = b"\x89PNG\r\n\x1a\nnot really an image"
    save_mod(tmp_path / "a.kmod", adapter.to_sets(), adapter.settings, id="a", previews=[png])
    assert load_mod(tmp_path / "a.kmod").previews == [png]


def test_bf16_save(model, x, tmp_path):
    import dataclasses

    from kilora import save_mod
    from kilora.sets import KeyedExperts

    adapter = trained(model)
    sets = adapter.to_sets()
    save_mod(tmp_path / "a.kmod", sets, adapter.settings, id="a", dtype=torch.bfloat16)
    adapter.remove()

    def rounded(t):
        return None if t is None else t.to(torch.bfloat16).float()

    s = sets[0]
    want = dataclasses.replace(
        s,
        routers={w: rounded(r) for w, r in s.routers.items()},
        experts={
            t: KeyedExperts(
                keys_in=rounded(ex.keys_in),
                keys_out=rounded(ex.keys_out),
                down=rounded(ex.down),
                up=rounded(ex.up),
                gate=ex.gate,
            )
            for t, ex in s.experts.items()
        },
    )
    loaded = load_mod(tmp_path / "a.kmod").sets[0]
    for t, ex in want.experts.items():
        assert torch.equal(loaded.experts[t].up, ex.up)
    a = compose(model, [want])
    expected = model(x)
    a.remove()
    compose(model, [tmp_path / "a.kmod"])
    torch.testing.assert_close(model(x), expected)


def test_cli_check(model, tmp_path, capsys):
    trained(model).save(tmp_path / "a.kmod", id="a", name="A mod", license="MIT")
    assert cli_main(["check", str(tmp_path / "a.kmod")]) == 0
    out = capsys.readouterr().out
    assert "A mod (a 0.1.0), license MIT" in out
    torch.save({}, tmp_path / "bad.kmod")
    assert cli_main(["check", str(tmp_path / "bad.kmod")]) == 1


def test_static_mod_has_no_live_parts(model, tmp_path):
    trained(model).save(tmp_path / "a.kmod", id="a")
    with safe_open(str(tmp_path / "a.kmod"), framework="pt") as f:
        manifest = json.loads(f.metadata()[MANIFEST_KEY])
        names = list(f.keys())
    assert manifest["requires_features"] == ["experts.lora"]
    assert "live" not in manifest
    assert "learn" not in manifest
    assert not any(s.get("live") for s in manifest["adapter"]["sets"])
    assert not any(n.startswith(("kilora/state/", "kilora/learn/")) for n in names)
    assert all(n.startswith("kilora/") for n in names)


def test_unknown_live_feature_is_refused_by_name(model, x, tmp_path):
    trained(model).save(tmp_path / "a.kmod", id="a")
    rewrite_manifest(
        tmp_path / "a.kmod",
        tmp_path / "b.kmod",
        lambda m: m["requires_features"].append("live.state-2"),
    )
    with pytest.raises(UnsupportedFeatureError, match="live.state-2"):
        load_mod(tmp_path / "b.kmod")
    load_mod(tmp_path / "a.kmod")


def test_set_axis_round_trip_and_default_output(model, x, tmp_path):
    from kilora.modfile import save_mod

    adapter = trained(model)
    sets = adapter.to_sets()
    plain = tmp_path / "plain.kmod"
    save_mod(plain, sets, adapter.settings, id="plain")
    sets[0].extra["axis"] = "shared"
    path = tmp_path / "shared.kmod"
    save_mod(path, sets, adapter.settings, id="shared")
    assert load_mod(path).sets[0].extra["axis"] == "shared"
    adapter.remove()
    normal = compose(model, [plain])
    expected = model(x).detach().clone()
    normal.remove()
    compose(model, [path])
    assert torch.equal(model(x), expected)


@pytest.mark.parametrize("score", ["derived", "keys", "read_share"])
def test_state_score_round_trip_keeps_learned_tensors_loadable(model, tmp_path, score):
    adapter = trained(model)
    adapter.settings.import_score = score
    before = adapter.to_sets()[0]
    path = tmp_path / "scored.kmod"
    adapter.save(path, id="scored")
    loaded = load_mod(path)
    assert loaded.settings.import_score == score
    after = loaded.sets[0]
    for width, router in before.routers.items():
        torch.testing.assert_close(after.routers[width], router)
    for target, expert in before.experts.items():
        torch.testing.assert_close(after.experts[target].keys_in, expert.keys_in)
        torch.testing.assert_close(after.experts[target].keys_out, expert.keys_out)
    adapter.remove()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("storage", [None, "int8-block"])
def test_state_keys_equal_bases_of_stored_weights(model, tmp_path, dtype, storage):
    from kilora import Settings, compute
    from kilora.modfile import save_mod, tensor_name

    adapter = trained(model)
    sets = adapter.to_sets()
    adapter.remove()
    path = tmp_path / "stored.kmod"
    save_mod(path, sets, Settings(), dtype=dtype, storage=storage, id="stored")
    loaded = load_mod(path)
    with safe_open(str(path), framework="pt") as file:
        for target, expert in loaded.sets[0].experts.items():
            n, routes, rank, width = expert.down.shape
            layer = compute.LayerSet(
                router=torch.eye(width),
                keys_in=expert.keys_in,
                down=expert.down.reshape(n * routes * rank, width),
                up=expert.up.reshape(n * routes * rank, expert.up.shape[-1]),
                per_expert=routes * rank,
            )
            kin, rin, kout, rout, _ = compute.derived_keys(layer)
            for part, expected in (
                ("keys_in", kin),
                ("keys_out", kout),
                ("rows_in", rin),
                ("rows_out", rout),
            ):
                assert torch.equal(getattr(expert, part), expected)
                assert torch.equal(file.get_tensor(tensor_name(target, f"s0/{part}")), expected)
            assert file.get_tensor(tensor_name(target, "s0/keys_in")).dtype == torch.float32


def test_state_save_load_save_keeps_identical_bytes(model, tmp_path):
    from kilora.modfile import save_mod

    adapter = trained(model)
    first, second = tmp_path / "first.kmod", tmp_path / "second.kmod"
    save_mod(first, adapter.to_sets(), adapter.settings, id="same")
    adapter.remove()
    loaded = load_mod(first)
    save_mod(second, loaded.sets, loaded.settings, id="same")
    assert first.read_bytes() == second.read_bytes()
