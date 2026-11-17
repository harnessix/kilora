# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_comfyui_state_keys import chain
from test_import_modes import three

from kilora import compose
from kilora.adapter import IMPORTED_DEPTH, IMPORTED_ROUTE_FREEZE, imported_settings
from kilora.lora_import import import_lora


def _sd(seed=3):
    gen = torch.Generator().manual_seed(seed)
    return {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": torch.randn(2, 16, generator=gen),
        "lora_unet_layers_0_attention_to_q.lora_up.weight": torch.randn(16, 2, generator=gen),
    }


def test_the_import_defaults(monkeypatch):
    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    s = imported_settings()
    assert (IMPORTED_DEPTH, IMPORTED_ROUTE_FREEZE) == (3, 0.4)
    assert (s.depth, s.iteration, s.route_freeze, s.gate_mode) == (4, "accumulate", 0.4, "auto")
    assert s.path_width == 3


def test_the_keys_arm_keeps_the_old_import_defaults(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    s = imported_settings()
    assert (s.depth, s.path_width, s.route_freeze, s.gate_mode) == (3, 1, 0.4, "auto")


def test_compose_of_imports_is_routed_and_stack_is_plain(denoiser, monkeypatch):
    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)
    imported = import_lora(_sd(), denoiser).adapter_set
    routed = compose(denoiser, [imported])
    st = routed.settings
    assert (st.depth, st.route_freeze, st.gate_mode) == (4, 0.4, "auto")
    routed.remove()
    plain = compose(denoiser, [imported], mix="stack")
    st = plain.settings
    assert (st.depth, st.route_freeze, st.gate_mode) == (1, None, "none")
    plain.remove()


def test_trained_mods_keep_their_settings(model):
    from kilora import KiLoRAConfig, attach

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
    sets = adapter.to_sets()
    adapter.remove()
    again = compose(model, sets)
    assert (again.settings.depth, again.settings.route_freeze) == (1, None)
    again.remove()


def test_kilora_import_files_store_the_defaults(denoiser, tmp_path, monkeypatch):
    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)
    from kilora.lora_import import import_to_mod
    from kilora.modfile import load_mod

    src = tmp_path / "style.safetensors"
    save_file(_sd(), str(src))
    out = tmp_path / "style.kmod.safetensors"
    import_to_mod(str(src), denoiser, out)
    st = load_mod(out).settings
    assert (st.depth, st.route_freeze, st.gate_mode) == (4, 0.4, "auto")


def test_kilora_import_files_store_the_late_schedule(denoiser, folders, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    from kilora.adapter import IMPORTED_DEPTH_SCHEDULE
    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching
    from kilora.lora_import import import_to_mod
    from kilora.modfile import load_mod

    src = folders["loras"] / "style.safetensors"
    save_file(_sd(), str(src))
    out = folders["kilora"] / "style.kmod"
    import_to_mod(str(src), denoiser, out)
    manifest = load_mod(out).manifest
    assert manifest["depth_schedule"] == IMPORTED_DEPTH_SCHEDULE
    assert manifest["schedule_yields_to_depth"] is True
    a = compose(denoiser, [str(out)])
    assert a.depth_schedule is not None
    a.remove()
    b = compose(denoiser, [str(out)], depth=2)
    assert b.depth_schedule is None
    b.remove()
    (selected,) = N.KiLoRASelect().select("style.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    routed, _, _ = N.KiLoRAApply().apply(base, selected, -1)
    assert patching.applied(routed)["depth_schedule"] is not None
    explicit, _, _ = N.KiLoRAApply().apply(base, selected, 2)
    assert patching.applied(explicit)["depth_schedule"] is None


def test_add_lora_in_comfyui_runs_at_depth_four(denoiser, folders, monkeypatch):
    from kilora.comfyui import nodes as N

    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)

    save_file(_sd(), str(folders["loras"] / "style.safetensors"))
    (selected,) = N.KiLoRAImportLoRA().add("style.safetensors", 1.0)
    _, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), selected, -1)
    assert "Depth: 4" in report


def test_simple_mixes():
    from kilora.comfyui.settings import resolve_settings

    assert resolve_settings({"mix": "balanced"})["depth"] == -1
    assert resolve_settings({"mix": "subtle"})["depth"] == 1
    assert resolve_settings({"mix": "stack"})["depth"] == 1


def test_saved_import_uses_derived_defaults(denoiser, tmp_path, monkeypatch):
    from kilora.lora_import import import_to_mod
    from kilora.modlist import ModChoice, load_choice

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    src = tmp_path / "style.safetensors"
    out = tmp_path / "style.kmod.safetensors"
    save_file(_sd(), str(src))
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    import_to_mod(str(src), denoiser, out)
    loaded = load_choice(ModChoice(out), 0)
    assert loaded.recommended_depth == 4
    assert loaded.settings.path_width == 3
    assert "depth_schedule" not in loaded.manifest
    routed = compose(denoiser, [str(out)])
    assert (routed.settings.depth, routed.settings.path_width, routed.depth_schedule) == (
        4,
        3,
        None,
    )
    routed.remove()
    explicit = compose(denoiser, [str(out)], depth=2)
    assert explicit.settings.depth == 2
    explicit.remove()


def test_default_routes_with_derived_keys_without_state_pass(three, monkeypatch, tmp_path):
    from test_import_modes import DIM, Toy

    from kilora import compute, key_cache
    from kilora.multi_import import import_loras

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    model, paths, _, _ = three
    routed = import_loras(paths, model).adapter_set
    m = Toy().eval()
    adapter = compose(m, [routed])
    assert adapter.settings.import_score == "derived"
    assert compute.derived_mode() == "full"
    assert not adapter._keys_todo
    x = torch.randn(1, 3, DIM)
    with torch.no_grad():
        default = m(x)
        monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
        explicit = m(x)
    assert torch.equal(default, explicit)
    assert not list((tmp_path / "cache").glob("*.safetensors"))
    adapter.remove()


def test_default_joint_import_skips_key_cache(denoiser, chain, monkeypatch, tmp_path):
    from test_comfyui_state_keys import _call

    from kilora.comfyui import key_cache, patching

    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_MODE", raising=False)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    patched, _ = patching.apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, -1, patching.LiveEnv(), joint=True
    )
    assert "key_pass" not in patching.applied(patched)
    assert torch.isfinite(_call(patched, torch.randn(1, 5, 4), torch.ones(1, 2))).all()
    assert not list((tmp_path / "cache").glob("*.safetensors"))


def test_derived_mode_defaults_to_full(monkeypatch):
    from kilora import compute
    from kilora.adapter import imported_schedule

    for name in ("KILORA_DERIVED_MODE", "KILORA_DERIVED_DIFFUSE", "KILORA_IMPORT_SCORE"):
        monkeypatch.delenv(name, raising=False)
    assert compute.DEFAULT_DERIVED_MODE == "full"
    assert compute.derived_mode() == "full"
    s = imported_settings()
    assert (s.depth, s.path_width) == (4, 3)
    assert imported_schedule() is None
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    assert compute.derived_mode() == "keys"
    s = imported_settings()
    assert (s.depth, s.path_width) == (3, 1)
    assert imported_schedule() is not None
