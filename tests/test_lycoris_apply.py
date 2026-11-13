# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher
from test_multi_import import Q, ToyAnima, _lokr, lora_weights, to_keys, write

from kilora import compose, compute
from kilora.comfyui import patching
from kilora.comfyui.settings import KiLoRASettings, resolve_settings
from kilora.multi_import import import_loras


def paths_for(tmp_path):
    model = ToyAnima().eval()
    plain = lora_weights(model, layers=[Q], rank=4, seed=5)
    paths = [write(tmp_path, "lyco", _lokr()[0]), write(tmp_path, "plain", to_keys(plain, "kohya"))]
    return model, paths


def run_layer(model, imported, x, y, without_exact=False):
    adapter = compose(model, [imported.adapter_set], mix="stack")
    layer = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))[0]
    layer = dataclasses.replace(layer, import_score="read_share")
    if without_exact:
        layer = dataclasses.replace(layer, exact={})
    with torch.no_grad():
        result = compute.run_layer(x, y, [layer], 1, target=Q)
    adapter.remove()
    return result


def test_default_matches_low_rank_reference_and_exact_matches_previous(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model, paths = paths_for(tmp_path)
    truncated = import_loras(paths, model, rank=4)
    exact = import_loras(paths, model, rank=4, lycoris_apply="exact")
    assert all(not e.exact for e in truncated.adapter_set.experts.values())
    assert exact.adapter_set.experts[Q].exact
    x, y = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    assert torch.equal(run_layer(model, truncated, x, y), run_layer(model, exact, x, y, True))
    assert not torch.equal(run_layer(model, exact, x, y), run_layer(model, truncated, x, y))
    assert "lycoris_apply=exact" in exact.report()
    assert "lycoris_apply=truncated" in truncated.report()
    assert "lycoris_apply=exact applies the whole change" in truncated.report()


@pytest.mark.parametrize(
    "setting,override,expected", [("exact", "truncated", False), ("truncated", "exact", True)]
)
def test_environment_overrides_setting(tmp_path, monkeypatch, setting, override, expected):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    monkeypatch.setenv("KILORA_LYCORIS_APPLY", override)
    model, paths = paths_for(tmp_path)
    imported = import_loras(paths, model, rank=4, lycoris_apply=setting)
    assert bool(imported.adapter_set.experts[Q].exact) is expected


def test_switching_mode_rebuilds_routed_import(tmp_path, monkeypatch):
    import kilora.multi_import as multi

    model, paths = paths_for(tmp_path)
    patcher = FakePatcher(FakeBaseModel(model))
    binding = SimpleNamespace(family=None)
    entries = [patching.ModEntry("lora", p, axis="creative") for p in paths]
    original = multi.import_loras
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs["lycoris_apply"])
        return original(*args, **kwargs)

    monkeypatch.setattr(multi, "import_loras", tracked)
    patching._ROUTED_CACHE.clear()
    for mode in ("truncated", "exact", "truncated"):
        patching.joint_choice(patcher, entries, binding, model, lycoris_apply=mode)
    assert calls == ["truncated", "exact"]


def test_single_lycoris_is_plain_in_both_modes(tmp_path, monkeypatch):
    model, paths = paths_for(tmp_path)
    binding = SimpleNamespace(family=None)
    entry = patching.ModEntry("lora", paths[0], axis="creative")
    assert patching._routed_groups([entry], True) == {}
    outputs = []
    for mode in ("truncated", "exact"):
        monkeypatch.setenv("KILORA_LYCORIS_APPLY", mode)
        result = patching._import_result(paths[0], binding, model, model)
        outputs.append(result)
    assert len(outputs[0].sections) == len(outputs[1].sections) > 0
    for left, right in zip(outputs[0].sections, outputs[1].sections, strict=True):
        assert left.kind == right.kind
        for key in left.data:
            for a, b in zip(left.data[key], right.data[key], strict=True):
                assert torch.equal(a, b)


def test_setting_node_and_validation():
    assert KiLoRASettings().make("balanced", lycoris_apply="exact")[0]["lycoris_apply"] == "exact"
    with pytest.raises(ValueError, match="lycoris_apply"):
        resolve_settings({"lycoris_apply": "bad"})


def test_apply_switches_and_recipe_preserves_mode(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_families import _model

    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    gen = torch.Generator().manual_seed(17)
    stem = "diffusion_model.double_blocks.0.img_attn.proj"
    files = [tmp_path / "plain.safetensors", tmp_path / "kron.safetensors"]
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(2, 8, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(8, 2, generator=gen),
        },
        str(files[0]),
    )
    save_file(
        {
            f"{stem}.lokr_w1": torch.randn(2, 2, generator=gen),
            f"{stem}.lokr_w2": torch.randn(4, 4, generator=gen),
        },
        str(files[1]),
    )
    chain = patching.ModChain(patching.ModEntry("lora", str(path)) for path in files)
    base = FakePatcher(FakeBaseModel(_model("flux")))
    patching._ROUTED_CACHE.clear()
    truncated, tr_report = patching.apply(base, chain, 1, joint=True, lycoris_apply="truncated")
    exact, ex_report = patching.apply(base, chain, 1, joint=True, lycoris_apply="exact")
    assert len(patching._ROUTED_CACHE) == 2
    assert "lycoris_apply=truncated" in tr_report
    assert "lycoris_apply=exact" in ex_report
    assert patching.applied(truncated)["lycoris_apply"] == "truncated"
    assert patching.applied(exact)["lycoris_apply"] == "exact"

    from test_recipes import make_recipe

    from kilora.adapter import Settings
    from kilora.modfile import save_mod
    from kilora.modlist import ModChoice, load_choice
    from kilora.sections import Section

    recipe = tmp_path / "recipe.kmod"
    make_recipe(recipe, [(p, 1.0) for p in files])
    loaded = load_choice(ModChoice(path=recipe), 0, [tmp_path])
    settings = dict(loaded.logic)
    settings.update(mix="balanced", lycoris_apply="exact")
    sections = [s for s in loaded.sections if s.kind != "logic"]
    sections.append(Section("logic", "settings", {}, {"version": 1, "settings": settings}))
    recipe = tmp_path / "recipe-exact.kmod"
    save_mod(recipe, [], Settings(), sections=sections, id="recipe", name="recipe")
    expanded, saved = patching.expand_chain(
        [patching.ModEntry("mod", str(recipe))], patching.LiveEnv(search_dirs=[str(tmp_path)])
    )
    assert saved["lycoris_apply"] == "exact"
    assert len(expanded) == 2
    from kilora.comfyui import nodes

    monkeypatch.setattr(nodes, "live_env", lambda: patching.LiveEnv(search_dirs=[str(tmp_path)]))
    from kilora.comfyui.settings import DEFAULTS

    reference, _, _ = nodes.KiLoRAApply().apply(
        base,
        chain,
        1,
        settings={k: v for k, v in settings.items() if k in DEFAULTS},
        output_gate="none",
    )
    restored, _, _ = nodes.KiLoRAApply().apply(
        base, patching.ModChain([patching.ModEntry("mod", str(recipe))]), 1
    )
    assert torch.equal(reference.run(2), restored.run(2))
