# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from test_comfyui import denoiser, folders
from test_comfyui_state_keys import chain
from test_multi_import import ToyAnima, _lokr, write

from kilora import compose, key_cache
from kilora.multi_import import import_loras
from kilora.prepare import prepare_lokr, prepare_mix


def test_prepared_parts_are_reused_and_file_changes_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_LOKR_CACHE", "1")
    weights, _ = _lokr()
    path = write(tmp_path, "lokr", weights)
    assert prepare_lokr(path) == (1, 0)
    from kilora import prepare

    with monkeypatch.context() as context:

        def no_read(*args, **kwargs):
            raise AssertionError("current file should not be loaded")

        context.setattr(prepare, "_read_file", no_read)
        assert prepare_lokr(path) == (0, 1)

    from kilora import multi_import

    def no_part_svd(*args, **kwargs):
        raise AssertionError("prepared parts were not reused")

    monkeypatch.setattr(multi_import, "_kron_parts_svd", no_part_svd)
    assert import_loras([path], ToyAnima().eval(), rank=1).adapter_set is not None

    monkeypatch.undo()
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    changed = dict(weights)
    name = next(name for name in changed if name.endswith("lokr_w1"))
    changed[name] = changed[name] + 1
    path = write(tmp_path, "lokr", changed)
    assert prepare_lokr(path) == (1, 0)


def test_part_key_is_sensitive_to_tensor_content():
    from kilora.multi_import import _lokr_parts_key

    parts = {"lokr_w1": torch.ones(2, 2), "lokr_w2": torch.ones(2, 3)}
    assert _lokr_parts_key(parts) != _lokr_parts_key({**parts, "lokr_w1": torch.zeros(2, 2)})


def test_lokr_cache_isolated_by_backend(monkeypatch, tmp_path):
    from kilora import derived_cache

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    weights, _ = _lokr()
    path = write(tmp_path, "lokr", weights)
    assert prepare_lokr(path) == (1, 0)
    monkeypatch.setattr(derived_cache, "backend", lambda device: {"device": "stand-in-cuda"})
    assert prepare_lokr(path) == (1, 0)
    assert prepare_lokr(path) == (0, 1)


def test_missing_cached_parts_are_rebuilt(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    from kilora.multi_import import _lokr_cache_path

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    weights, _ = _lokr()
    path = write(tmp_path, "lokr", weights)
    assert prepare_lokr(path) == (1, 0)
    cache_path = _lokr_cache_path(key_cache.file_sha256(path))
    save_file({"@complete/count": torch.tensor(1)}, cache_path)
    assert prepare_lokr(path) == (1, 0)


def test_preparation_preserves_import_and_forward(monkeypatch, tmp_path):
    monkeypatch.setenv("KILORA_LOKR_CACHE", "1")
    monkeypatch.setenv("KILORA_FUSED", "0")
    weights, _ = _lokr()
    path = write(tmp_path, "lokr", weights)
    model = ToyAnima().eval()
    x, ctx = torch.randn(1, 3, 16), torch.randn(1, 3, 8)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cold"))
    cold = import_loras([path], model, rank=1).adapter_set
    first = compose(model, [cold])
    with torch.no_grad():
        expected = model(x, ctx)
    first.remove()

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "warm"))
    assert prepare_lokr(path) == (1, 0)
    warm = import_loras([path], model, rank=1).adapter_set
    second = compose(model, [warm])
    with torch.no_grad():
        actual = model(x, ctx)
    second.remove()
    assert torch.equal(expected, actual)


def test_prepare_exact_mix_reuses_cpu_derived_keys(monkeypatch, tmp_path):
    from test_apply_derived_cache import _clear, _files
    from test_import_modes import Toy, _lora

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    model = Toy().eval()
    paths = _files(tmp_path)
    prepare_mix(paths, model)
    _clear()

    def no_svd(*args, **kwargs):
        raise AssertionError("prepared derived keys were not reused")

    with monkeypatch.context() as context:
        context.setattr(torch.linalg, "svd", no_svd)
        adapter = compose(model, [import_loras(paths, model, device="cpu").adapter_set])
        with torch.no_grad():
            model(torch.randn(1, 3, 12))
        adapter.remove()

    _lora(model, paths[0], rank=3, seed=99)
    _clear()
    calls = 0
    svd = torch.linalg.svd

    def count_svd(*args, **kwargs):
        nonlocal calls
        calls += 1
        return svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", count_svd)
    adapter = compose(model, [import_loras(paths, model, device="cpu").adapter_set])
    with torch.no_grad():
        model(torch.randn(1, 3, 12))
    adapter.remove()
    assert calls > 0


def test_file_hash_changes_when_mtime_is_restored(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"first")
    before = path.stat()
    original = key_cache.file_sha256(path)
    path.write_bytes(b"other")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert key_cache.file_sha256(path) != original


def test_prepare_node_reports_prepared_and_current(monkeypatch, tmp_path):
    from kilora.comfyui.nodes import FOLDERS
    from kilora.comfyui.prepare_node import KiLoRAPrepareLoRAs

    folder = tmp_path / "loras"
    folder.mkdir()
    weights, _ = _lokr()
    path = write(folder, "one", weights)
    monkeypatch.setattr(FOLDERS, "dirs", lambda kind: [str(folder)])
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    first = KiLoRAPrepareLoRAs().prepare("one.safetensors")
    assert f"Prepared LoKr parts: {os.path.basename(path)}" in first["result"][0]
    second = KiLoRAPrepareLoRAs().prepare("one.safetensors")
    assert f"Skipped (current): {os.path.basename(path)}" in second["result"][0]
    assert "Prepared derived keys" in second["result"][0]
    (folder / "other.pt").write_bytes(b"unsupported")
    report = KiLoRAPrepareLoRAs().prepare("other.pt\nmissing.safetensors")["result"][0]
    assert "Failed: other.pt (unsupported file type)" in report
    assert "Failed: missing.safetensors" in report


def test_prepare_node_locale_entries_match_catalog(monkeypatch):
    from kilora import i18n
    from kilora.comfyui import help as help_texts
    from kilora.comfyui.nodes import FOLDERS, NODE_CLASS_MAPPINGS

    monkeypatch.setattr(FOLDERS, "list", lambda *args, **kwargs: [])
    root = Path(__file__).resolve().parents[1]
    for lang in i18n.LOCALES:
        shipped = json.loads((root / "locales" / lang / "nodeDefs.json").read_text())
        expected = help_texts.node_defs(lang, NODE_CLASS_MAPPINGS)
        assert shipped["KiLoRAPrepareLoRAs"] == expected["KiLoRAPrepareLoRAs"]


def test_prepare_node_warms_matching_comfy_mix(denoiser, chain, monkeypatch):
    from test_apply_derived_cache import _clear
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _call

    from kilora import derived_cache
    from kilora.comfyui import patching
    from kilora.comfyui.prepare_node import KiLoRAPrepareLoRAs

    monkeypatch.setattr(derived_cache, "backend", lambda device: {"device": "stand-in-cuda"})
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "0")
    base = FakePatcher(FakeBaseModel(denoiser))
    report = KiLoRAPrepareLoRAs().prepare("", base)["result"][0]
    assert "Prepared the selected ordered mix" in report
    _clear()
    patching._ROUTED_CACHE.clear()

    def no_svd(*args, **kwargs):
        raise AssertionError("prepared derived keys were not reused")

    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    patched, _ = patching.apply(base, chain, -1, patching.LiveEnv(), joint=True)
    _call(patched, torch.randn(1, 5, 4), torch.ones(1, 2))


def test_startup_switch_and_busy_flag(monkeypatch):
    from kilora.comfyui import startup_prepare

    monkeypatch.setenv("KILORA_PREPARE_AT_STARTUP", "0")
    assert startup_prepare.start() is None
    assert not startup_prepare.busy()
    with startup_prepare.model_call():
        assert startup_prepare.busy()
        with startup_prepare.model_call():
            assert startup_prepare.busy()
        assert startup_prepare.busy()
    assert not startup_prepare.busy()
    with pytest.raises(ValueError), startup_prepare.model_call():
        raise ValueError("example")
    assert not startup_prepare.busy()


def test_startup_setting_persists_and_environment_overrides(monkeypatch, tmp_path):
    from kilora.comfyui import startup_prepare

    monkeypatch.setattr(startup_prepare, "_settings_path", lambda: str(tmp_path / "setting.json"))
    monkeypatch.setattr(startup_prepare, "start", lambda: None)
    monkeypatch.delenv("KILORA_PREPARE_AT_STARTUP", raising=False)
    assert startup_prepare.enabled()
    startup_prepare.set_enabled(False)
    assert not startup_prepare.enabled()
    startup_prepare.set_enabled(True)
    assert startup_prepare.enabled()
    monkeypatch.setenv("KILORA_PREPARE_AT_STARTUP", "0")
    assert not startup_prepare.enabled()


def test_startup_worker_yields_and_reuses_parts(monkeypatch, tmp_path, caplog):
    from kilora.comfyui import startup_prepare

    weights, _ = _lokr()
    folder = tmp_path / "loras"
    folder.mkdir()
    write(folder, "lokr", weights)
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_LOKR_CACHE", "1")
    startup_prepare._stop.clear()
    cache = tmp_path / "cache" / "lokr"
    with caplog.at_level("INFO", logger="kilora.prepare"):
        with startup_prepare.model_call():
            worker = threading.Thread(target=startup_prepare._run, args=([str(folder)],))
            worker.start()
            assert not threading.Event().wait(1)
            assert worker.is_alive() and not list(cache.glob("*.safetensors"))
        worker.join(30)
        assert not worker.is_alive()
        assert len(list(cache.glob("*.safetensors"))) == 1
        startup_prepare._run([str(folder)])
    assert "'prepared': 1" in caplog.text and "'current': 1" in caplog.text


def test_startup_checkpoint_waits_for_busy_and_releases_cache(monkeypatch):
    from kilora.comfyui import startup_prepare

    running = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    running.set()
    monkeypatch.setattr(startup_prepare, "busy", running.is_set)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", released.set)
    startup_prepare._stop.clear()
    worker = threading.Thread(
        target=lambda: (startup_prepare._checkpoint("cuda:0"), finished.set())
    )
    worker.start()
    assert released.wait(5)
    assert not finished.is_set()
    running.clear()
    worker.join(5)
    assert finished.is_set()


def test_worker_selects_available_device(monkeypatch):
    from kilora.comfyui import prepare_node, startup_prepare

    seen = []
    monkeypatch.setattr(prepare_node, "selected_files", lambda *args: (["one"], []))
    monkeypatch.setattr(startup_prepare, "busy", lambda: False)
    monkeypatch.setattr(os, "nice", lambda value: None)
    monkeypatch.setattr("kilora.prepare.prepare_lokr", lambda *a, **k: (0, 1, True))
    monkeypatch.setattr("kilora.prepare.prepare_file", lambda *a, **k: seen.append(k["device"]))
    startup_prepare._stop.clear()
    for device in (torch.device("cuda:0"), torch.device("cpu")):
        monkeypatch.setattr(prepare_node, "apply_device", lambda model=None, chosen=device: chosen)
        startup_prepare._run(["root"])
    assert seen == [torch.device("cuda:0"), torch.device("cpu")]


def test_family_only_preparation_is_reused(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from test_apply_derived_cache import _clear

    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets
    from kilora.families import shape_tree
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    target = "transformer_blocks.0.attn1.to_q"
    model = shape_tree({target: (12, 12)})
    path = tmp_path / "family.safetensors"
    save_file(
        {
            f"{target}.lora_down.weight": torch.randn(3, 12),
            f"{target}.lora_up.weight": torch.randn(12, 3),
        },
        str(path),
    )
    prepare_file(path, family="anima")
    _clear()
    imported = import_loras([str(path)], model, family="anima", device="cpu").adapter_set
    layer = next(iter(plan_targets([ScopedSet(imported)], 1).values())).sets[0]
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    compute.derived_keys(layer)


def test_family_only_prepares_fused_alias(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from test_apply_derived_cache import _clear

    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets
    from kilora.families import shape_tree
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    prefix = "layers.0.attention."
    names = [prefix + "to_" + suffix for suffix in "qkv"]
    model = shape_tree({name: (12, 12) for name in names})
    path = tmp_path / "fused.safetensors"
    stem = "lora_unet_layers_0_attention_qkv"
    save_file(
        {
            f"{stem}.lora_down.weight": torch.randn(3, 12),
            f"{stem}.lora_up.weight": torch.randn(36, 3),
        },
        str(path),
    )
    prepare_file(path, family="z-image")
    _clear()
    imported = import_loras([str(path)], model, family="z-image", device="cpu").adapter_set
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    for plan in plan_targets([ScopedSet(imported)], 1).values():
        compute.derived_keys(plan.sets[0])


def test_family_prepared_file_is_reused_through_comfy_apply(folders, monkeypatch, tmp_path):
    from test_apply_derived_cache import _clear
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _sets
    from test_multi_import import ToyAnima

    from kilora import compute
    from kilora.comfyui import nodes, patching
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    model = ToyAnima().eval()
    model.kilora_family = "anima"
    outside = tmp_path / "outside"
    outside.mkdir()
    path = outside / "anima.safetensors"
    link = folders["loras"] / "linked.safetensors"
    link.symlink_to(path)
    (outside / "adapter_config.json").write_text(json.dumps({"r": 3, "lora_alpha": 6}))
    (folders["loras"] / "adapter_config.json").write_text(json.dumps({"r": 3, "lora_alpha": 3}))
    from safetensors.torch import save_file

    save_file(
        {
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(3, 16),
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(16, 3),
        },
        str(path),
    )
    prepare_file(path, family="anima")
    _clear()
    patching._ROUTED_CACHE.clear()
    (entries,) = nodes.KiLoRAImportLoRA().add(link.name, 1.0)
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    patched, _ = patching.apply(
        FakePatcher(FakeBaseModel(model)), entries, -1, patching.LiveEnv(), joint=True
    )
    layers = _sets(patched)
    assert layers
    for layer in layers:
        compute.derived_keys(layer)


@pytest.mark.parametrize("linked", [False, True])
def test_model_free_anima_mix_reuses_expert_identities(folders, monkeypatch, tmp_path, linked):
    from safetensors.torch import save_file
    from test_apply_derived_cache import _clear
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora import derived_cache
    from kilora.comfyui import nodes, patching
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    model = ToyAnima().eval()
    model.kilora_family = "anima"
    captured = []
    original_identity = derived_cache.strength_free_identities

    def capture(records, target, by_expert):
        result = original_identity(records, target, by_expert)
        for e, identity in enumerate(result or ()):
            if e in by_expert:
                down, up = by_expert[e]
                captured.append((target, down.clone(), up.clone(), identity))
        return result

    monkeypatch.setattr(derived_cache, "strength_free_identities", capture)
    paths = []
    for i, rank in enumerate((2, 3, 4)):
        path = tmp_path / f"source{i}.safetensors"
        gen = torch.Generator().manual_seed(i)
        stem = "diffusion_model.blocks.0.self_attn.q_proj"
        save_file(
            {
                f"{stem}.lora_down.weight": torch.randn(rank, 16, generator=gen).to(torch.float16),
                f"{stem}.lora_up.weight": torch.randn(16, rank, generator=gen).to(torch.float16),
                f"{stem}.alpha": torch.tensor(rank * 2.0),
            },
            str(path),
        )
        prepare_file(path, device="cpu")
        selected = folders["loras"] / path.name
        if linked:
            selected.symlink_to(path)
        else:
            selected.write_bytes(path.read_bytes())
        paths.append(selected)

    prepared = {item[3]: item for item in captured if item[3] is not None}
    assert prepared
    captured.clear()
    _clear()
    patching._ROUTED_CACHE.clear()
    calls = []
    original_eigh = torch.linalg.eigh

    def count_eigh(*args, **kwargs):
        calls.append(1)
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", count_eigh)
    for order in (paths, list(reversed(paths))):
        entries = ()
        for path in order:
            (entries,) = nodes.KiLoRAImportLoRA().add(path.name, 1.0, entries)
        patching.apply(
            FakePatcher(FakeBaseModel(model)), entries, -1, patching.LiveEnv(), joint=True
        )
        patching._ROUTED_CACHE.clear()
    actual = [item for item in captured if item[3] is not None]
    assert len(actual) == 6
    assert all(item[0] == "transformer_blocks.0.attn1.to_q" for item in actual)
    assert all(item[1].dtype == item[2].dtype == torch.float32 for item in actual)
    assert all(item[3] in prepared for item in actual)
    for target, down, up, identity in actual:
        expected = prepared[identity]
        assert (target, down.shape, up.shape) == (expected[0], expected[1].shape, expected[2].shape)
        assert torch.equal(down, expected[1])
        assert torch.equal(up, expected[2])
    assert calls == []


def test_model_free_preparation_preserves_mix_keys(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    from kilora.families import shape_tree
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    target = "transformer_blocks.0.attn1.to_q"
    model = shape_tree({target: (16, 16)})
    paths = []
    for i in range(2):
        path = tmp_path / f"mix{i}.safetensors"
        generator = torch.Generator().manual_seed(i)
        stem = "diffusion_model.blocks.0.self_attn.q_proj"
        save_file(
            {
                f"{stem}.lora_down.weight": torch.randn(3, 16, generator=generator),
                f"{stem}.lora_up.weight": torch.randn(16, 3, generator=generator),
            },
            str(path),
        )
        paths.append(path)
    before = import_loras(paths, model, family="anima", device="cpu").adapter_set.experts[target]
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "prepared-cache"))
    for path in paths:
        prepare_file(path, device="cpu")
    after = import_loras(paths, model, family="anima", device="cpu").adapter_set.experts[target]
    assert torch.equal(before.down, after.down)
    assert torch.equal(before.up, after.up)
    assert torch.equal(before.keys_in, after.keys_in)


def test_symlink_selection_report_and_import_cache(monkeypatch, tmp_path):
    from kilora import multi_import
    from kilora.comfyui.nodes import FOLDERS
    from kilora.comfyui.prepare_node import KiLoRAPrepareLoRAs, selected_files

    root = tmp_path / "loras"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    weights, _ = _lokr()
    first = write(outside, "one", weights)
    second = write(outside, "two", weights)
    (root / "linked.safetensors").symlink_to(first)
    (root / "duplicate.safetensors").symlink_to(first)
    (root / "linked_folder").symlink_to(outside, target_is_directory=True)
    (outside / "loop").symlink_to(root, target_is_directory=True)
    files, missing = selected_files(
        [str(root)],
        "linked.safetensors\nduplicate.safetensors\nlinked_folder\n../outside/one.safetensors",
    )
    assert missing == ["../outside/one.safetensors"]
    assert len(files) == 2
    assert {os.path.realpath(p) for p in files} == {str(first), str(second)}
    monkeypatch.setattr(FOLDERS, "dirs", lambda kind: [str(root)])
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    report = KiLoRAPrepareLoRAs().prepare("linked.safetensors\nlinked_folder")["result"][0]
    assert "Prepared LoKr parts: linked.safetensors" in report
    assert "linked_folder/two.safetensors" in report
    assert str(outside) not in report
    assert key_cache.file_sha256(root / "linked.safetensors") == key_cache.file_sha256(first)
    monkeypatch.setattr(multi_import, "_kron_parts_svd", lambda *a, **k: pytest.fail("cache miss"))
    assert (
        import_loras([str(root / "linked.safetensors")], ToyAnima().eval(), rank=1).adapter_set
        is not None
    )


def test_worker_exit_joins_slow_layer(tmp_path):
    script = """
import threading, time
from kilora.comfyui import startup_prepare as worker
from kilora.comfyui import prepare_node
import kilora.prepare as prepare
prepare_node.apply_device = lambda: 'cpu'
prepare_node.selected_files = lambda *args: (['one'], [])
worker.busy = lambda: False
ready = threading.Event()
def slow(*args, checkpoint=None, **kwargs):
    ready.set()
    time.sleep(0.3)
    print("layer finished", flush=True)
    checkpoint()
    return (0, 0, False)
prepare.prepare_lokr = slow
worker._worker = threading.Thread(target=worker._run, args=(['.'],), daemon=True)
worker._worker.start()
assert ready.wait(10)
"""
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "layer finished" in result.stdout
    assert time.monotonic() - started < 30


def test_interrupted_cache_write_is_a_miss(monkeypatch, tmp_path):
    from kilora import derived_cache, multi_import

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    real_replace = os.replace

    def interrupt(source, target):
        assert os.path.isfile(source) and os.path.getsize(source) > 0
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", interrupt)
    tensor = torch.ones(1)
    derived_cache.store("identity", "kind", {}, ("value",), (tensor,))
    assert derived_cache.load("identity", "kind", {}, ("value",)) is None
    multi_import._save_lokr_cache("digest", {"part": {"value": tensor}})
    assert multi_import._load_lokr_cache("digest") == {}
    keys = {"layer": {name: tensor.clone() for name in key_cache.PARTS}}
    with pytest.raises(OSError, match="interrupted"):
        key_cache.store("key", keys, 0.0)
    assert key_cache.load("key") is None
    assert not list((tmp_path / "cache").rglob("*.tmp"))
    monkeypatch.setattr(os, "replace", real_replace)


def test_worker_stop_during_synchronize_does_not_empty_cache(monkeypatch):
    from kilora.comfyui import startup_prepare

    startup_prepare._stop.clear()
    monkeypatch.setattr(startup_prepare, "busy", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: startup_prepare._stop.set())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("cache cleared after stop"))
    try:
        with pytest.raises(startup_prepare._Stopped):
            startup_prepare._checkpoint("cuda:0")
    finally:
        startup_prepare._stop.clear()


def test_file_read_checks_for_stop_between_tensors(tmp_path):
    from safetensors.torch import save_file

    from kilora.lora_import import _read_file

    path = tmp_path / "multiple.safetensors"
    save_file({"a": torch.ones(1), "b": torch.zeros(1)}, str(path))
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _read_file(path, checkpoint=checkpoint)
    assert calls == 2


def test_prepare_model_uses_listed_order_and_apply_mix_cache(denoiser, chain, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora import derived_cache
    from kilora.comfyui import patching
    from kilora.comfyui.prepare_node import KiLoRAPrepareLoRAs

    monkeypatch.setenv("KILORA_CONTENT_KEYS", "mix")
    monkeypatch.setattr(derived_cache, "expert_identities", lambda *args: None)
    ordered = list(reversed(chain))
    base = FakePatcher(FakeBaseModel(denoiser))
    selection = "\n".join(os.path.basename(entry.path) for entry in ordered)
    report = KiLoRAPrepareLoRAs().prepare(selection, base)["result"][0]
    assert "Prepared the selected ordered mix" in report
    patching._ROUTED_CACHE.clear()
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    patched, _ = patching.apply(base, ordered, -1, patching.LiveEnv(), joint=True)
    assert patched is not None


def test_family_only_prepares_new_per_expert_mix(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    from kilora.families import shape_tree
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    target = "transformer_blocks.0.attn1.to_q"
    model = shape_tree({target: (12, 12)})
    paths = []
    for i in range(2):
        path = tmp_path / f"family-{i}.safetensors"
        save_file(
            {
                f"{target}.lora_down.weight": torch.randn(3 + i, 12),
                f"{target}.lora_up.weight": torch.randn(12, 3 + i),
            },
            str(path),
        )
        prepare_file(path, family="anima", device="cpu")
        paths.append(str(path))
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    assert import_loras(paths, model, family="anima", device="cpu").adapter_set.experts


@pytest.mark.parametrize("kind", ["lokr", "loha"])
@pytest.mark.parametrize("worker", [False, True])
def test_exact_single_files_prepare_routed_mix(monkeypatch, tmp_path, kind, worker):
    from safetensors.torch import save_file

    from kilora import compute
    from kilora.bake import ScopedSet, plan_targets
    from kilora.comfyui import prepare_node, startup_prepare
    from kilora.families import shape_tree
    from kilora.prepare import prepare_file

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KILORA_CONTENT_KEYS", raising=False)
    target = "transformer_blocks.0.attn1.to_q"
    model = shape_tree({target: (12, 12)})
    paths = []
    for i in range(2):
        generator = torch.Generator().manual_seed(i + 301)
        if kind == "lokr":
            tensors = {
                "lokr_w1": torch.randn(3, 3, generator=generator),
                "lokr_w2": torch.randn(4, 4, generator=generator),
            }
        else:
            tensors = {
                name: torch.randn(*shape, generator=generator)
                for name, shape in (
                    ("hada_w1_a", (12, 2)),
                    ("hada_w1_b", (2, 12)),
                    ("hada_w2_a", (12, 2)),
                    ("hada_w2_b", (2, 12)),
                )
            }
        path = tmp_path / f"{kind}-{i}.safetensors"
        save_file({f"{target}.{name}": value for name, value in tensors.items()}, str(path))
        paths.append(str(path))
        exact = import_loras([path], model, family="anima", lycoris_mode="exact", device="cpu")
        assert exact.adapter_set is None and exact.sections
    if worker:
        monkeypatch.setattr(prepare_node, "selected_files", lambda *a: (paths, []))
        monkeypatch.setattr(prepare_node, "display_path", lambda path, roots: path)
        monkeypatch.setattr(prepare_node, "apply_device", lambda: "cpu")
        monkeypatch.setattr(startup_prepare, "_checkpoint", lambda device: None)
        monkeypatch.setattr(startup_prepare.os, "nice", lambda value: None)
        startup_prepare._stop.clear()
        startup_prepare._run([str(tmp_path)])
    else:
        for path in paths:
            prepare_file(path, family="anima", device="cpu")
    monkeypatch.setenv("KILORA_CONTENT_KEYS", "per_expert")
    compute._DERIVED.clear()
    compute._DERIVED_KEYS_BY_DIGEST.clear()
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("unexpected SVD"))
    mixed = import_loras(
        paths, model, family="anima", device="cpu", lycoris_apply="exact"
    ).adapter_set
    assert mixed.experts[target].exact
    for plan in plan_targets([ScopedSet(mixed, import_score="derived")], 1).values():
        compute.derived_keys(plan.sets[0])
