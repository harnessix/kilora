# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_modlist import write_mod

from kilora import palette
from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.palette import Palette, PaletteEntry, Resolved, load_palette, save_palette, shuffle


@pytest.fixture
def pal_dir(tmp_path, monkeypatch):
    folder = tmp_path / "palettes"
    monkeypatch.setattr(palette, "PALETTE_DIR", str(folder))
    return folder


@pytest.fixture
def loras(folders):
    names = []
    for i in range(4):
        gen = torch.Generator().manual_seed(30 + i)
        stem = "lora_unet_layers_0_attention_to_q"
        sd = {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
        }
        name = f"style{i}.safetensors"
        save_file(
            sd,
            str(folders["loras"] / name),
            metadata={"ss_tag_frequency": json.dumps({"x": {f"word{i}": 3}})},
        )
        names.append(name)
    return names


def _chain(names, strengths):
    chain = None
    for n, s in zip(names, strengths, strict=True):
        (chain,) = N.KiLoRAImportLoRA().add(n, s, sets=chain)
    return chain


def test_save_load_round_trip(denoiser, folders, loras, pal_dir):
    chain = _chain(loras[:3], [0.8, 0.5, -0.3])
    report = N.KiLoRASavePalette().save(chain, "Three styles", "three")["result"][0]
    assert "Saved three.kpalette.json with 3 entries." in report
    pal = load_palette(pal_dir / "three.kpalette.json")
    assert pal.name == "Three styles"
    assert [(e.kind, e.name, e.strength) for e in pal.entries] == [
        ("lora", "style0.safetensors", 0.8),
        ("lora", "style1.safetensors", 0.5),
        ("lora", "style2.safetensors", -0.3),
    ]
    assert all(len(e.sha256) == 64 for e in pal.entries)
    assert pal.entries[0].triggers == ["word0"]
    save_palette(pal_dir / "again.kpalette.json", pal)
    assert load_palette(pal_dir / "again.kpalette.json") == pal


def test_old_mod_packs_load_as_palettes(denoiser, folders, pal_dir):
    from kilora.modlist import ModChoice, resolve
    from kilora.packs import pack_from_mods, save_pack

    a = write_mod(denoiser, folders["kilora"], "ink", seed=1)
    plan = resolve([ModChoice(path=a, strength=0.6)])
    save_pack(folders["kilora"] / "old.kpack.json", pack_from_mods("Old", plan.mods, depth=3))
    assert "old.kpack.json" in N.KiLoRAPalette.INPUT_TYPES()["required"]["palette"][0]
    sets, _, settings, report = N.KiLoRAPalette().load("old.kpack.json")
    assert [(e.kind, e.strength) for e in sets] == [("mod", 0.6)]
    assert settings["depth"] == 3 and "1 of 1 entries" in report


def test_palette_and_the_equal_add_lora_chain_give_the_same_plan(denoiser, folders, loras, pal_dir):
    chain = _chain(loras[:3], [0.8, 0.5, -0.3])
    N.KiLoRASavePalette().save(chain, "p", "p")
    loaded, words, _, _ = N.KiLoRAPalette().load("p.kpalette.json")
    assert [e.axis for e in loaded] == ["creative"] * len(chain)
    patcher = FakePatcher(FakeBaseModel(denoiser))
    a, _ = patching.plan_chain(patcher, chain, joint=True)
    b, _ = patching.plan_chain(patcher, loaded, joint=True)
    assert a.report(1) == b.report(1)
    assert words == "word0, word1, word2"


def test_palette_appends_to_an_incoming_list_and_maps_region_labels(
    denoiser,
    folders,
    loras,
    pal_dir,
):
    pal = Palette("r", [PaletteEntry("lora", loras[0], region="a"), PaletteEntry("lora", loras[1])])
    save_palette(pal_dir / "r.kpalette.json", pal)
    first = _chain(loras[3:], [1.0])
    mask = torch.ones(1, 4, 4)
    sets, _, _, report = N.KiLoRAPalette().load("r.kpalette.json", sets=first, mask_a=mask)
    assert [e.path.rsplit("/", 1)[-1] for e in sets] == [loras[3], loras[0], loras[1]]
    assert sets[1].region is mask and sets[2].region is None
    _, _, _, report = N.KiLoRAPalette().load("r.kpalette.json")
    assert "needs mask_a" in report


def _resolved(n):
    return [Resolved(PaletteEntry("lora", f"l{i}.safetensors"), f"/x/l{i}") for i in range(n)]


def test_shuffle_is_seeded_and_stays_in_bounds():
    entries = _resolved(8)
    state = torch.random.get_rng_state()
    a = shuffle(entries, 7, (2, 4), (0.6, 1.0))
    b = shuffle(entries, 7, (2, 4), (0.6, 1.0))
    assert [(r.path, r.entry.strength) for r in a] == [(r.path, r.entry.strength) for r in b]
    assert torch.equal(torch.random.get_rng_state(), state)
    picks = {tuple(r.path for r in shuffle(entries, s, (2, 4), (0.6, 1.0))) for s in range(20)}
    assert len(picks) > 5
    for s in range(50):
        pick = shuffle(entries, s, (2, 4), (0.6, 1.0))
        assert 2 <= len(pick) <= 4
        assert all(0.6 <= r.entry.strength <= 1.0 for r in pick)
        order = [int(r.path[-1]) for r in pick]
        assert order == sorted(order)
    assert len(shuffle(entries[:2], 1, (3, 5), (0.5, 0.5))) == 2


def test_shuffle_node_reports_the_pick_and_saves_it(denoiser, folders, loras, pal_dir):
    N.KiLoRASavePalette().save(_chain(loras, [1.0] * 4), "four", "four")
    sets, words, _, report = N.KiLoRAPalette().load(
        "four.kpalette.json", shuffle=True, pick_min=2, pick_max=2, seed=3
    )
    assert len(sets) == 2 and "Shuffle (seed 3) picked 2" in report
    again, _, _, _ = N.KiLoRAPalette().load(
        "four.kpalette.json", shuffle=True, pick_min=2, pick_max=2, seed=3
    )
    assert list(again) == list(sets)
    assert words == ", ".join(f"word{e.path[-13]}" for e in sets)
    N.KiLoRASavePalette().save(sets, "pick", "pick")
    kept = load_palette(pal_dir / "pick.kpalette.json")
    assert [e.strength for e in kept.entries] == [e.strength for e in sets]


def test_missing_and_changed_files_are_reported_and_skipped(denoiser, folders, loras, pal_dir):
    N.KiLoRASavePalette().save(_chain(loras[:3], [1.0] * 3), "p", "p")
    (folders["loras"] / loras[0]).unlink()
    data = folders["loras"] / loras[1]
    data.write_bytes(data.read_bytes() + b" ")
    sets, _, _, report = N.KiLoRAPalette().load("p.kpalette.json")
    assert [e.path.rsplit("/", 1)[-1] for e in sets] == [loras[2]]
    assert f"{loras[0]}: not found, skipped" in report
    assert f"{loras[1]}: the file changed" in report and "1 of 3 entries" in report


@pytest.mark.parametrize("axis", ["ip", "shared"])
def test_axis_round_trip_and_old_palettes_load_as_creative(denoiser, folders, loras, pal_dir, axis):
    (chain,) = N.KiLoRAImportLoRA().add(loras[0], 1.0)
    (chain,) = N.KiLoRAImportLoRA().add(loras[1], 0.7, sets=chain, axis=axis)
    N.KiLoRASavePalette().save(chain, "axes", "axes")
    pal = load_palette(pal_dir / "axes.kpalette.json")
    assert [e.axis for e in pal.entries] == ["creative", axis]
    sets, _, _, _ = N.KiLoRAPalette().load("axes.kpalette.json")
    assert [e.axis for e in sets] == ["creative", axis]
    data = json.loads((pal_dir / "axes.kpalette.json").read_text())
    for item in data["entries"]:
        del item["axis"]
    (pal_dir / "old.kpalette.json").write_text(json.dumps(data))
    assert [e.axis for e in load_palette(pal_dir / "old.kpalette.json").entries] == [
        "creative",
        "creative",
    ]
    data["entries"][0]["axis"] = "sideways"
    (pal_dir / "bad.kpalette.json").write_text(json.dumps(data))
    with pytest.raises(palette.ModFormatError, match="axis"):
        load_palette(pal_dir / "bad.kpalette.json")


def test_shuffle_picks_creative_entries_and_keeps_functional_ones():
    entries = _resolved(6)
    for i in (1, 4):
        entries[i] = Resolved(
            PaletteEntry("lora", f"l{i}.safetensors", strength=0.9, axis="functional"), f"/x/l{i}"
        )
    for s in range(30):
        pick = shuffle(entries, s, (1, 2), (0.6, 1.0))
        functional = [r for r in pick if r.entry.axis == "functional"]
        creative = [r for r in pick if r.entry.axis == "creative"]
        assert [r.path for r in functional] == ["/x/l1", "/x/l4"]
        assert all(r.entry.strength == 0.9 for r in functional)
        assert 1 <= len(creative) <= 2
        order = [int(r.path[-1]) for r in pick]
        assert order == sorted(order)
    only = [entries[1], entries[4]]
    assert shuffle(only, 0, (1, 2), (0.6, 1.0)) == only


def test_shuffle_random_matches_base_pick():
    entries = _resolved(8)
    expected = [("/x/l2", 0.62), ("/x/l3", 0.93), ("/x/l5", 0.64)]
    for options in ({}, {"mode": "random", "state": palette.ShuffleState()}):
        result = shuffle(entries, 7, (2, 4), (0.6, 1.0), **options)
        assert [(r.path, r.entry.strength) for r in result] == expected


def test_learn_shuffle_uses_counts_and_keeps_functional():
    entries = _resolved(4)
    entries[1].entry.axis = "functional"
    entries[1].entry.strength = 0.87
    state = palette.ShuffleState()
    for i in (0, 2, 3):
        state.counts[palette.shuffle_entry_key(entries[i].entry)] = [
            [1000, 0] if i == 2 else [0, 1000],
            *([[0, 1000]] * 3),
            [1000, 0],
            [0, 1000],
        ]
    a = shuffle(entries, 42, (1, 1), (0.0, 1.0), mode="learn", state=state)
    b = shuffle(entries, 42, (1, 1), (0.0, 1.0), mode="learn", state=state)
    assert a == b
    assert [r.path for r in a] == ["/x/l1", "/x/l2"]
    assert a[0].entry.strength == 0.87
    assert palette.strength_bin(a[1].entry.strength, (0.0, 1.0)) == 3


def test_shuffle_state_round_trip_reset_and_validation(tmp_path):
    path = tmp_path / "name.kpalette.json"
    key = palette.shuffle_entry_key(_resolved(1)[0].entry)
    state = palette.ShuffleState({key: [[2, 3], *([[0, 0]] * 5)]})
    assert palette.load_shuffle_state(path) == palette.ShuffleState()
    palette.save_shuffle_state(path, state)
    state_path = tmp_path / "name.kpalette.state.json"
    assert json.loads(state_path.read_text()) == state.counts
    assert palette.load_shuffle_state(path) == state
    state_path.write_text('{"bad": []}')
    with pytest.raises(ValueError, match="Invalid Shuffle counts"):
        palette.load_shuffle_state(path)
    palette.reset_shuffle_state(path)
    palette.reset_shuffle_state(path)
    assert palette.load_shuffle_state(path) == palette.ShuffleState()


def test_palette_node_learn_and_reset(denoiser, folders, loras, pal_dir):
    N.KiLoRASavePalette().save(_chain(loras, [1.0] * 4), "four", "four")
    path = pal_dir / "four.kpalette.json"
    entry = load_palette(path).entries[2]
    state = palette.ShuffleState({palette.shuffle_entry_key(entry): [[1000, 0]] * 6})
    palette.save_shuffle_state(path, state)
    sets, _, _, _ = N.KiLoRAPalette().load(
        path.name, shuffle=True, pick_min=1, pick_max=1, seed=3, mode="learn"
    )
    assert len(sets) == 1 and sets[0].path.endswith(loras[2])
    N.KiLoRAPalette().load(path.name, reset_state=True)
    assert not palette.shuffle_state_path(path).exists()


def test_shuffle_reward_updates_only_selected_entries_and_bins():
    entries = _resolved(4)
    entries[1].entry.axis = "functional"
    state = palette.ShuffleState()
    pick = shuffle(entries, 17, (2, 2), (0.0, 1.0), mode="learn", state=state)
    palette.reward_shuffle(state, pick, (0.0, 1.0), "keep")
    palette.reward_shuffle(state, pick, (0.0, 1.0), "discard")
    creative = [item for item in pick if item.entry.axis != "functional"]
    assert len(state.counts) == len(creative) == 2
    for item in creative:
        counts = state.counts[palette.shuffle_entry_key(item.entry)]
        index = palette.strength_bin(item.entry.strength, (0.0, 1.0)) + 1
        assert counts[0] == [1, 1]
        assert counts[index] == [1, 1]
        assert sum(sum(pair) for pair in counts[1:]) == 2
    with pytest.raises(ValueError, match="reward"):
        palette.reward_shuffle(state, pick, (0.0, 1.0), "invalid")


def test_shuffle_reward_node_records_current_pick_only(denoiser, folders, loras, pal_dir):
    N.KiLoRASavePalette().save(_chain(loras, [1.0] * 4), "four", "four")
    path = pal_dir / "four.kpalette.json"
    node = N.KiLoRAPalette()
    kwargs = dict(shuffle=True, pick_min=1, pick_max=1, seed=9, mode="learn")
    original, _, _, _ = node.load(path.name, **kwargs)
    assert not palette.shuffle_state_path(path).exists()
    rated, _, _, report = node.load(path.name, reward="keep", **kwargs)
    assert rated[0].path == original[0].path
    assert "feedback recorded: keep" in report
    state = palette.load_shuffle_state(path)
    assert len(state.counts) == 1
    assert next(iter(state.counts.values()))[0] == [1, 0]
    node.load(path.name, shuffle=True, reward="discard")
    assert palette.load_shuffle_state(path) == state


def test_shuffle_learns_with_noisy_feedback():
    import random

    entries = _resolved(4)
    state = palette.ShuffleState()
    noise = [True] * 6 + [False] * 24
    random.Random(21).shuffle(noise)
    best = {entries[0].path, entries[1].path}
    for seed in range(30):
        pick = shuffle(entries, seed, (2, 2), (0.0, 1.0), mode="learn", state=state)
        keep = ({item.path for item in pick} == best) != noise[seed]
        palette.reward_shuffle(state, pick, (0.0, 1.0), "keep" if keep else "discard")
    a = shuffle(entries, 123, (2, 2), (0.0, 1.0), mode="learn", state=state)
    assert a == shuffle(entries, 123, (2, 2), (0.0, 1.0), mode="learn", state=state)
    learned = sum(
        {
            item.path
            for item in shuffle(entries, seed, (2, 2), (0.0, 1.0), mode="learn", state=state)
        }
        == best
        for seed in range(1000, 2000)
    )
    random_picks = sum(
        {item.path for item in shuffle(entries, seed, (2, 2), (0.0, 1.0))} == best
        for seed in range(1000, 2000)
    )
    print(f"After 30 rewards: learn {learned / 10}%, random {random_picks / 10}%")
    assert learned >= 700
    assert learned > random_picks
