# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from conftest import TARGET, randomize_up
from safetensors import safe_open

from kilora import KiLoRAConfig, attach, compose, load_mod
from kilora.check import check_file
from kilora.errors import UnsupportedFeatureError
from kilora.modfile import FEATURES, save_mod


@pytest.fixture
def sets(model):
    adapter = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=4, router_dim=8))
    randomize_up(adapter)
    out = adapter.to_sets(), adapter.settings
    adapter.remove()
    return out


def test_round_trip_and_size(model, sets, x, tmp_path):
    s, settings = sets
    save_mod(tmp_path / "f.kmod", s, settings, id="f")
    save_mod(tmp_path / "q.kmod", s, settings, id="q", storage="int8-block")
    with safe_open(str(tmp_path / "q.kmod"), framework="pt") as f:
        names = list(f.keys())
        kinds = {n: f.get_tensor(n).dtype for n in names}
    assert any(n.endswith("/up.scale") for n in names)
    assert all(kinds[n] == torch.int8 for n in names if n.endswith(("/up", "/down")))
    key_names = [n for n in names if n.endswith(("/keys_in", "/keys_out"))]
    assert key_names and all(kinds[n] == torch.float32 for n in key_names)
    router_names = [n for n in names if n.startswith("kilora/router/")]
    assert router_names and all(kinds[n] == torch.bfloat16 for n in router_names)
    assert "storage.int8-block" in load_mod(tmp_path / "q.kmod").manifest["requires_features"]
    a, b = load_mod(tmp_path / "f.kmod"), load_mod(tmp_path / "q.kmod")
    for sa, sb in zip(a.sets, b.sets, strict=True):
        for t, ea in sa.experts.items():
            eb = sb.experts[t]
            for part in ("down", "up"):
                va, vb = getattr(ea, part), getattr(eb, part)
                assert float((va - vb).norm() / va.norm()) < 0.01
    with torch.no_grad():
        full = compose(model, [tmp_path / "f.kmod"])
        want = model(x)
        full.remove()
        small = compose(model, [tmp_path / "q.kmod"])
        got = model(x)
        small.remove()
    assert float((got - want).norm() / want.norm()) < 0.02
    assert check_file(tmp_path / "q.kmod").ok


def test_bit_identical_save_load_and_resave(sets, tmp_path):
    s, settings = sets
    save_mod(tmp_path / "a.kmod", s, settings, id="q", storage="int8-block")
    save_mod(tmp_path / "b.kmod", s, settings, id="q", storage="int8-block")
    assert (tmp_path / "a.kmod").read_bytes() == (tmp_path / "b.kmod").read_bytes()
    one, two = load_mod(tmp_path / "a.kmod"), load_mod(tmp_path / "a.kmod")
    for sa, sb in zip(one.sets, two.sets, strict=True):
        for t in sa.experts:
            assert torch.equal(sa.experts[t].up, sb.experts[t].up)
    save_mod(tmp_path / "c.kmod", one.sets, one.settings, id="q", storage="int8-block")
    assert (tmp_path / "c.kmod").read_bytes() == (tmp_path / "a.kmod").read_bytes()


def test_an_older_loader_refuses_the_file(sets, tmp_path, monkeypatch):
    s, settings = sets
    save_mod(tmp_path / "q.kmod", s, settings, id="q", storage="int8-block")
    monkeypatch.delitem(FEATURES, "storage.int8-block")
    with pytest.raises(UnsupportedFeatureError, match="storage.int8-block"):
        load_mod(tmp_path / "q.kmod")


def test_unknown_storage_is_refused(sets, tmp_path):
    s, settings = sets
    with pytest.raises(ValueError, match="unknown storage"):
        save_mod(tmp_path / "x.kmod", s, settings, id="x", storage="int4")


def test_tiny_lora_parts_keep_their_values_and_scales_are_float32():
    from kilora.modfile import NAMESPACE, _store, _unstore

    name = f"{NAMESPACE}0/layer/up"
    t = {name: torch.randn(64, 256, generator=torch.Generator().manual_seed(2)) * 1e-8}
    stored = _store(t, "int8-block")
    assert stored[f"{name}.scale"].dtype == torch.float32
    back = _unstore(stored)[name]
    assert float((back - t[name]).norm() / t[name].norm()) < 0.01
    old = dict(stored, **{f"{name}.scale": stored[f"{name}.scale"].to(torch.float16)})
    assert float(_unstore(old)[name].abs().max()) == 0.0
    big = {name: torch.randn(4, 256, generator=torch.Generator().manual_seed(3))}
    s2 = _store(big, "int8-block")
    s2_old = dict(s2, **{f"{name}.scale": s2[f"{name}.scale"].to(torch.float16)})
    torch.testing.assert_close(_unstore(s2_old)[name], _unstore(s2)[name], rtol=2e-3, atol=1e-6)


def test_input_keys_keep_their_stored_values(tmp_path):
    from test_fast_keys import _input_set

    from kilora import Settings
    from kilora.modfile import tensor_name

    setting = _input_set()
    path = tmp_path / "input.kmod"
    save_mod(path, [setting], Settings(), storage="int8-block", id="input")
    loaded = load_mod(path).sets[0]
    with safe_open(str(path), framework="pt") as file:
        for target, expert in setting.experts.items():
            for part in ("keys_in", "keys_out"):
                stored = file.get_tensor(tensor_name(target, f"s0/{part}"))
                assert stored.dtype == torch.bfloat16
                assert torch.equal(stored, getattr(expert, part).to(torch.bfloat16))
                assert torch.equal(getattr(loaded.experts[target], part), stored.float())
