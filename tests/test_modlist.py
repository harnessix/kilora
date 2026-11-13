# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from conftest import TARGET, randomize_up

from kilora import KiLoRAConfig, ModFormatError, UnsupportedFeatureError, attach, compose
from kilora.bake import is_baked
from kilora.modlist import (
    LoadOrderError,
    ModChoice,
    load_mods,
    parse_requirement,
    resolve,
    satisfies,
)
from kilora.packs import load_pack, pack_choices, pack_from_mods, save_pack, scan_library


def write_mod(model, folder, mod_id, seed=0, version="1.0.0", router_seed=0, **fields):
    target_modules = fields.pop("target_modules", [TARGET])
    a = attach(
        model,
        KiLoRAConfig(
            target_modules=target_modules,
            num_experts=4,
            routes=2,
            router_dim=8,
            seed=seed,
            router_seed=router_seed,
        ),
        name=mod_id,
    )
    randomize_up(a, seed=seed)
    path = folder / f"{mod_id}-{version}.kmod"
    fields.setdefault("base_model", {"family": "kilora-tiny-1"})
    a.save(path, id=mod_id, version=version, **fields)
    a.remove()
    return path


def test_versions():
    assert satisfies("1.2.0", ">=1.2, <2")
    assert not satisfies("2.0.0", ">=1.2, <2")
    assert satisfies("1.4", "1.4.0")
    assert parse_requirement("inkwash >=1.2, <2") == ("inkwash", ">=1.2, <2")
    assert parse_requirement("parchment") == ("parchment", "")
    with pytest.raises(ModFormatError):
        parse_requirement("Bad Name")


def test_order_follows_requires_then_priority_then_user_order(model, tmp_path):
    host = write_mod(model, tmp_path, "host", seed=1)
    patch = write_mod(model, tmp_path, "patch", seed=2, requires=["host >=1.0"])
    early = write_mod(model, tmp_path, "early", seed=3, priority=-5)
    plan = resolve([ModChoice(patch), ModChoice(host), ModChoice(early)], family="kilora-tiny-1")
    assert plan.ok, plan.report()
    assert [m.mod_id for m in plan.mods] == ["early", "host", "patch"]
    assert plan.reasons["patch"] == ["after host: requires"]
    report = plan.report(depth=3)
    assert "1. early" in report and "after host: requires" in report


def test_missing_requirement_and_wrong_version_are_errors(model, tmp_path):
    host = write_mod(model, tmp_path, "host", seed=1, version="0.9.0")
    patch = write_mod(model, tmp_path, "patch", seed=2, requires=["host >=1.0", "parchment"])
    plan = resolve([ModChoice(patch), ModChoice(host)])
    messages = [p.message for p in plan.problems]
    assert "patch requires host >=1.0, found 0.9.0" in messages
    assert "patch requires parchment" in messages
    with pytest.raises(LoadOrderError, match="patch requires parchment"):
        load_mods(model, [ModChoice(patch), ModChoice(host)])
    assert not is_baked(model.layers[0].attention.to_out[0])


def test_conflicts_cycles_and_duplicates(model, tmp_path):
    a = write_mod(model, tmp_path, "a", seed=1, conflicts=["b"])
    b = write_mod(model, tmp_path, "b", seed=2)
    assert "a conflicts with b 1.0.0" in resolve([ModChoice(a), ModChoice(b)]).report()
    c = write_mod(model, tmp_path, "c", seed=3, load_after=["d"])
    d = write_mod(model, tmp_path, "d", seed=4, load_after=["c"])
    assert "cycle between: c, d" in resolve([ModChoice(c), ModChoice(d)]).report()
    plan = resolve([ModChoice(b), ModChoice(b)])
    assert "the mod b is in the list twice" in plan.report()


def test_exclusive_transform_conflict(model, tmp_path):
    point = "base::layers.0.attention.to_out.0"
    a = write_mod(
        model,
        tmp_path,
        "a",
        seed=1,
        hooks=[{"kind": "transform", "point": point, "exclusive": True}],
    )
    b = write_mod(model, tmp_path, "b", seed=2)
    plan = resolve([ModChoice(a), ModChoice(b)], family="kilora-tiny-1")
    assert not plan.ok
    assert f"exclusive transform conflict at {point}: a, b" in plan.report()
    assert resolve([ModChoice(a)], family="kilora-tiny-1").ok


def test_pins_and_family(model, tmp_path):
    host = write_mod(model, tmp_path, "host", seed=1)
    other_host = write_mod(model, tmp_path, "host", seed=9, version="1.0.1")
    from kilora import read_manifest

    digest = read_manifest(host)["digest"]
    child = write_mod(
        model, tmp_path, "child", seed=2, requires=["host"], pins=[{"id": "host", "digest": digest}]
    )
    assert resolve([ModChoice(host), ModChoice(child)]).ok
    assert (
        "pins an exact file of host" in resolve([ModChoice(other_host), ModChoice(child)]).report()
    )
    zimg = write_mod(model, tmp_path, "zimg", seed=3, base_model={"family": "z-image"})
    assert (
        "zimg is made for z-image, the model is kilora-tiny-1"
        in resolve([ModChoice(zimg)], family="kilora-tiny-1").report()
    )


def test_load_mods_bakes_with_strength_and_matches_compose(model, x, tmp_path):
    a = write_mod(model, tmp_path, "a", seed=1)
    b = write_mod(model, tmp_path, "b", seed=2)
    hooked = compose(model, [a, b], strengths=[0.5, -1.0], depth=2)
    want = model(x)
    hooked.remove()
    baked, plan, depth = load_mods(
        model, [ModChoice(a, strength=0.5), ModChoice(b, strength=-1.0)], depth=2
    )
    assert depth == 2
    torch.testing.assert_close(model(x), want, rtol=1e-5, atol=1e-6)
    baked.remove()
    _, plan, depth = load_mods(model, [ModChoice(a, enabled=False)])
    assert depth == 0 and plan.disabled


def test_pack_round_trip(model, x, tmp_path):
    a = write_mod(model, tmp_path, "a", seed=1)
    b = write_mod(model, tmp_path, "b", seed=2)
    plan = resolve([ModChoice(a, strength=0.75), ModChoice(b, strength=-0.5)])
    pack_path = tmp_path / "set.kpack.json"
    data = save_pack(
        pack_path,
        pack_from_mods("My set", plan.mods, depth=4, sources={"a": "https://example.org/a"}),
    )
    assert str(tmp_path) not in str(data)
    assert len(data["mods"][0]["file_sha256"]) == 64
    assert data["mods"][0]["source"] == "https://example.org/a" and data["format_version"] == "1.0"
    assert pack_path.read_text() == pack_path.read_text()
    pack = load_pack(pack_path)
    assert pack.depth == 4 and [e.mod_id for e in pack.entries] == ["a", "b"]
    choices = pack_choices(pack, scan_library(sorted(tmp_path.glob("*.kmod"))))
    assert [c.strength for c in choices] == [0.75, -0.5]
    other = tmp_path / "other"
    other.mkdir()
    write_mod(model, other, "a", seed=5)
    lib = scan_library(sorted(other.glob("*.kmod")) + sorted(tmp_path.glob("*.kmod")))
    assert pack_choices(pack, lib)[0].path == a
    for e in pack.entries:
        e.digest = None
    assert pack_choices(pack, lib)[0].path == a


def test_pack_refusals(tmp_path):
    path = tmp_path / "p.kpack"
    path.write_text('{"format": "kilora-pack", "format_version": "2.0", "mods": []}')
    with pytest.raises(UnsupportedFeatureError, match="pack format 2.0"):
        load_pack(path)
    path.write_text("not json")
    with pytest.raises(ModFormatError):
        load_pack(path)
