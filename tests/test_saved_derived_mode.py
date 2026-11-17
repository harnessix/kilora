# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kilora import compose, compute
from kilora.adapter import apply_imported_share_defaults, imported_settings
from kilora.lora_import import import_lora, import_to_mod
from kilora.modfile import load_mod, read_manifest
from kilora.testing import TinyDenoiser

OLD = Path(__file__).parents[1] / "compat-corpus" / "0.1.0" / "imported-and-keyed"
MODE_VARS = ("KILORA_DERIVED_MODE", "KILORA_IMPORT_SCORE", "KILORA_DERIVED_DIFFUSE")


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    for name in MODE_VARS:
        monkeypatch.delenv(name, raising=False)


def _tiny():
    m = TinyDenoiser(channels=4, width=16, num_layers=2, seed=0).eval()
    m.kilora_family = "kilora-tiny-1"
    return m


def _sd(seed=3):
    gen = torch.Generator().manual_seed(seed)
    return {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": torch.randn(2, 16, generator=gen),
        "lora_unet_layers_0_attention_to_q.lora_up.weight": torch.randn(16, 2, generator=gen),
    }


def _layer_modes(adapter):
    out = []
    for index, site in enumerate(adapter.sites):
        if site.keyed:
            out += [
                (layer.key, layer.form, compute.derived_mode(layer))
                for layer in adapter._layer_sets(index, site, torch.device("cpu"))
            ]
    return out


def _render(path, x):
    model = _tiny()
    adapter = compose(model, [path])
    with torch.no_grad():
        out = model(x)
    adapter.remove()
    return out


def _old_copy(tmp_path):
    path = tmp_path / "old.kmod.safetensors"
    shutil.copy(OLD / "mod.kmod.safetensors", path)
    return path


def test_an_old_file_without_the_field_loads_as_keys(tmp_path):
    path = _old_copy(tmp_path)
    assert "derived_mode" not in read_manifest(path)["adapter"]
    mod = load_mod(path)
    assert mod.settings.derived_mode == "keys"
    imports = [s for s in mod.sets if s.form == "input"]
    assert imports and all(s.extra["derived_mode"] == "keys" for s in imports)
    adapter = compose(_tiny(), [path])
    modes = {mode for _, form, mode in _layer_modes(adapter) if form == "input"}
    assert modes == {"keys"}
    adapter.remove()


def test_an_old_file_renders_as_the_keys_mode(tmp_path, monkeypatch):
    path = _old_copy(tmp_path)
    x = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(0))
    plain = _render(path, x)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    forced = _render(path, x)
    assert torch.equal(plain, forced)


def test_a_trained_mod_without_imports_stores_no_mode(tmp_path):
    from kilora import KiLoRAConfig, attach
    from kilora.modfile import save_mod

    model = _tiny()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=2,
            train_depth=None,
        ),
    )
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    path = tmp_path / "trained.kmod.safetensors"
    save_mod(path, sets, settings, id="trained")
    assert "derived_mode" not in read_manifest(path)["adapter"]
    assert load_mod(path).settings.derived_mode is None


def _imported_file(tmp_path, name="style"):
    src = tmp_path / f"{name}.safetensors"
    save_file(_sd(), str(src))
    out = tmp_path / f"{name}.kmod.safetensors"
    import_to_mod(str(src), _tiny(), out)
    return out


def test_a_file_saved_now_stores_full_and_loads_as_full(tmp_path):
    path = _imported_file(tmp_path)
    assert read_manifest(path)["adapter"]["derived_mode"] == "full"
    mod = load_mod(path)
    assert mod.settings.derived_mode == "full"
    assert (mod.settings.depth, mod.settings.path_width) == (4, 3)
    adapter = compose(_tiny(), [path])
    assert {m for _, form, m in _layer_modes(adapter) if form == "input"} == {"full"}
    assert adapter.depth_schedule is None
    adapter.remove()


def test_a_file_saved_with_the_keys_switch_stores_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    path = _imported_file(tmp_path)
    monkeypatch.delenv("KILORA_DERIVED_MODE")
    assert read_manifest(path)["adapter"]["derived_mode"] == "keys"
    mod = load_mod(path)
    assert mod.settings.derived_mode == "keys"
    adapter = compose(_tiny(), [path])
    assert {m for _, form, m in _layer_modes(adapter) if form == "input"} == {"keys"}
    adapter.remove()


def test_a_saved_mode_survives_a_second_save(tmp_path):
    from kilora.modfile import save_mod

    mod = load_mod(_old_copy(tmp_path))
    again = tmp_path / "again.kmod.safetensors"
    save_mod(again, mod.sets, mod.settings, id="again")
    assert read_manifest(again)["adapter"]["derived_mode"] == "keys"
    assert load_mod(again).settings.derived_mode == "keys"


@pytest.mark.parametrize("forced", ["keys", "full"])
def test_the_environment_switch_overrides_a_saved_mode(tmp_path, monkeypatch, forced):
    old, new = _old_copy(tmp_path), _imported_file(tmp_path)
    monkeypatch.setenv("KILORA_DERIVED_MODE", forced)
    for path in (old, new):
        mod = load_mod(path)
        assert compute.derived_mode(mod.settings) == forced
        adapter = compose(_tiny(), [path])
        assert {m for _, form, m in _layer_modes(adapter) if form == "input"} == {forced}
        adapter.remove()


def test_derived_mode_resolution():
    layer = compute.LayerSet(router=torch.eye(4), keys_in=torch.zeros(1, 1, 4))
    assert compute.derived_mode(layer) == compute.DEFAULT_DERIVED_MODE == "full"
    assert compute.derived_mode(dataclasses.replace(layer, derived_mode="keys")) == "keys"
    assert compute.derived_mode("keys") == "keys"
    assert compute.derived_mode(None) == "full"
    assert imported_settings().derived_mode == "full"
    assert imported_settings(derived_mode="keys").derived_mode == "keys"
    assert imported_settings(derived_mode="keys").path_width == 1


def test_an_old_live_export_takes_the_late_schedule():
    old = {"edits": {"sources": [{"path": "a.safetensors"}]}, "adapter": {"sets": []}}
    assert apply_imported_share_defaults(old)["depth_schedule"] == "1 + 2 * (t >= 0.3)"
    new = {**old, "adapter": {"sets": [], "derived_mode": "full"}}
    assert "depth_schedule" not in apply_imported_share_defaults(new)


def _by_key(adapter):
    out = {}
    for index, site in enumerate(adapter.sites):
        if site.keyed:
            for layer in adapter._layer_sets(index, site, torch.device("cpu")):
                if layer.form == "input":
                    out[layer.key] = (
                        compute.derived_mode(layer),
                        compute.share_transition_gamma(layer),
                    )
    return out


def test_two_sets_with_different_modes_at_one_mount(tmp_path, monkeypatch):
    model = _tiny()
    fresh = import_lora(_sd(3), model, name="fresh").adapter_set
    old = load_mod(_old_copy(tmp_path))
    loaded = next(s for s in old.sets if s.form == "input")
    loaded = dataclasses.replace(loaded, name="loaded")
    assert "derived_mode" not in fresh.extra and loaded.extra["derived_mode"] == "keys"
    adapter = compose(model, [fresh, loaded])
    assert _by_key(adapter) == {"fresh": ("full", 1.0), "loaded": ("keys", 0.0)}
    x = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()
    adapter.remove()
    adapter = compose(model, [_old_copy(tmp_path), _imported_file(tmp_path, "fresh")])
    modes = {mode for mode, _ in _by_key(adapter).values()}
    assert modes == {"keys", "full"}
    assert adapter.settings.derived_mode == "keys"
    adapter.remove()
    for forced in ("keys", "full"):
        monkeypatch.setenv("KILORA_DERIVED_MODE", forced)
        adapter = compose(model, [fresh, loaded])
        assert {mode for mode, _ in _by_key(adapter).values()} == {forced}
        adapter.remove()
