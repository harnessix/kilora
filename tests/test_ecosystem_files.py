# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import re

import torch
from conftest import TARGET, randomize_up
from safetensors import safe_open
from safetensors.torch import save_file

from kilora import KiLoRAConfig, attach, import_lora, load_mod, save_mod
from kilora.naming import MOD_SUFFIXES, mod_filename, strip_suffix

PLAIN_LORA_KEY = re.compile(
    r"(^lora_(unet|te\d?)_)|"
    r"(\.lora_(A|B|up|down|mid)\.)|(\.lora\.(up|down)\.)|(lora_linear_layer)|"
    r"(\.alpha$)|(\.(hada|lokr)_)|(dora_scale)|(\.diff(_b)?$)|(\.(weight|bias)$)"
)


def mod_file(model, tmp_path, name="ink"):
    a = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=3, router_dim=8))
    randomize_up(a)
    path = tmp_path / mod_filename(name, "z-image")
    a.save(
        path,
        id=name,
        name="Ink Wash",
        author="someone",
        license="CC-BY-4.0",
        base_model={"family": "z-image"},
        trigger_words=["inkwash style"],
        example_prompts=["a harbor at dawn, inkwash style"],
        mod_family="style",
        strength={"recommended": 0.8, "min": -1.0, "max": 1.5},
    )
    a.remove()
    return path


def test_file_names():
    assert mod_filename("Ink Wash", "z-image") == "ink-wash.zimage.kmod.safetensors"
    assert mod_filename("ink", None) == "ink.kmod.safetensors"
    assert strip_suffix("a.zimage.kmod.safetensors", MOD_SUFFIXES) == "a.zimage"
    assert strip_suffix("a.kmod", MOD_SUFFIXES) == "a"


def test_new_and_old_names_load(model, tmp_path):
    path = mod_file(model, tmp_path)
    assert path.name == "ink.zimage.kmod.safetensors"
    old = tmp_path / "ink.kmod"
    old.write_bytes(path.read_bytes())
    assert load_mod(path).manifest == load_mod(old).manifest


def test_modelspec_metadata(model, tmp_path):
    path = mod_file(model, tmp_path)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    assert meta["modelspec.sai_model_spec"] == "1.0.0"
    assert meta["modelspec.architecture"] == "z-image/kilora"
    assert meta["modelspec.title"] == "Ink Wash"
    assert meta["modelspec.author"] == "someone"
    assert meta["modelspec.license"] == "CC-BY-4.0"
    assert meta["modelspec.trigger_phrase"] == "inkwash style"
    assert meta["modelspec.tags"] == "kilora,style"
    assert "KiLoRA node" in meta["kilora.notice"] and "KiLoRA node" in meta["modelspec.usage_hint"]
    assert "Recommended strength: 0.8" in meta["modelspec.usage_hint"]
    assert not any("date" in k for k in meta)
    data = path.read_bytes()
    length = int.from_bytes(data[:8], "little")
    assert meta["modelspec.hash_sha256"] == "0x" + hashlib.sha256(data[8 + length :]).hexdigest()


def test_keys_do_not_match_plain_lora_loaders(model, tmp_path):
    keyed = mod_file(model, tmp_path)
    gen = torch.Generator().manual_seed(0)
    sd = {
        "layers.0.attention.to_q.lora_A.weight": torch.randn(2, 16, generator=gen),
        "layers.0.attention.to_q.lora_B.weight": torch.randn(16, 2, generator=gen),
    }
    imported = import_lora(sd, model).adapter_set
    from kilora.adapter import Settings

    imp_path = tmp_path / "imp.kmod.safetensors"
    save_mod(imp_path, [imported], Settings(depth=1), id="imp")
    for path in (keyed, imp_path):
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
        hits = [k for k in keys if PLAIN_LORA_KEY.search(k)]
        assert keys and hits == [], hits
        assert all(k.startswith("kilora/") for k in keys)
    save_file(sd, str(tmp_path / "plain.safetensors"))
    with safe_open(str(tmp_path / "plain.safetensors"), framework="pt") as f:
        assert all(PLAIN_LORA_KEY.search(k) for k in f.keys())


def test_export_is_byte_identical(model, tmp_path):
    a = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=3, router_dim=8))
    randomize_up(a)
    sets = a.to_sets()
    a.remove()
    from kilora.adapter import Settings

    one, two = tmp_path / "one.kmod.safetensors", tmp_path / "two.kmod.safetensors"
    save_mod(one, sets, Settings(depth=2), id="same", license="MIT")
    save_mod(two, list(reversed(list(reversed(sets)))), Settings(depth=2), id="same", license="MIT")
    assert one.read_bytes() == two.read_bytes()
    mod = load_mod(one)
    three = tmp_path / "three.kmod.safetensors"
    fields = {k: mod.manifest[k] for k in ("id", "license")}
    save_mod(three, mod.sets, mod.settings, **fields)
    assert three.read_bytes() == one.read_bytes()
