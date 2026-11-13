# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest
import torch
from test_import_modes import DIM, LAYERS, Toy, three

from kilora import KiLoRAConfig, attach, compose, fast_keys
from kilora import adapter as adapter_module
from kilora.lora_import import import_lora
from kilora.multi_import import import_loras
from kilora.sets import needs_state_keys


@pytest.fixture
def passes(monkeypatch):
    calls = []
    real = fast_keys.state_keys_from_call

    def spy(*args, **kwargs):
        calls.append(args[2] if len(args) > 2 else kwargs.get("set_index", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(fast_keys, "state_keys_from_call", spy)
    return calls


def _keys(adapter, i=0):
    return [site.tensors[f"s{i}_keys_in"].detach().clone() for site in adapter.sites]


def test_imported_set_gets_state_keys_at_the_first_forward_and_the_cache_later(
    three,
    passes,
    monkeypatch,
):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    model, paths, _, _ = three
    routed = import_loras(paths, model).adapter_set
    assert needs_state_keys(routed)
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(1))
    m = Toy().eval()
    adapter = compose(m, [routed], depth=2)
    before = _keys(adapter)
    assert all(k.shape[1] == 1 for k in before)
    with torch.no_grad():
        first = m(x)
    assert passes == [0] and adapter._set_info[0]["key_source"] == "states"
    after = _keys(adapter)
    assert all(k.shape[1] == 4 for k in after)
    with torch.no_grad():
        torch.testing.assert_close(m(x), first)
    assert passes == [0]
    adapter.remove()
    m2 = Toy().eval()
    again = compose(m2, [routed], depth=2)
    with torch.no_grad():
        torch.testing.assert_close(m2(x), first)
    assert passes == [0]
    for a, b in zip(_keys(again), after, strict=True):
        torch.testing.assert_close(a, b)


def test_trained_keys_and_plain_copies_are_left_alone(three, passes):
    model, paths, _, _ = three
    trained = attach(
        Toy().eval(), KiLoRAConfig(target_modules=list(LAYERS), num_experts=3, router_dim=8)
    ).to_sets()[0]
    copies = import_lora(paths[0], model).adapter_set
    assert not needs_state_keys(trained) and not needs_state_keys(copies)
    for s in (trained, copies):
        m = Toy().eval()
        adapter = compose(m, [s])
        before = _keys(adapter)
        with torch.no_grad():
            m(torch.randn(1, 3, DIM))
        for a, b in zip(_keys(adapter), before, strict=True):
            assert torch.equal(a, b)
        adapter.remove()
    assert passes == []


def test_pass_off_warns_once(three, passes, monkeypatch, caplog):
    model, paths, _, _ = three
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    monkeypatch.setattr(adapter_module, "AUTO_KEYS", False)
    m = Toy().eval()
    compose(m, [import_loras(paths, model).adapter_set])
    with caplog.at_level(logging.WARNING, logger="kilora.keys"), torch.no_grad():
        m(torch.randn(1, 3, DIM))
        m(torch.randn(1, 3, DIM))
    assert caplog.text.count("runs on random keys") == 1 and passes == []


def test_weighted_transition_is_cached_for_adapter(three, monkeypatch, tmp_path):
    from kilora import key_cache

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0.5")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION_KIND", "weighted")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    model, paths, _, _ = three
    routed = import_loras(paths, model).adapter_set
    x = torch.randn(1, 3, DIM)
    first_model = Toy().eval()
    first = compose(first_model, [routed], depth=2)
    with torch.no_grad():
        first_model(x)
    matrices = {
        index: site.tensors["s0_share_transition_weighted"].detach().clone()
        for index, site in enumerate(first.sites)
        if "s0_share_transition_weighted" in site.tensors
    }
    assert matrices
    first.remove()
    second_model = Toy().eval()
    second = compose(second_model, [routed], depth=2)
    with torch.no_grad():
        second_model(x)
    for index, expected in matrices.items():
        got = second.sites[index].tensors["s0_share_transition_weighted"]
        torch.testing.assert_close(got, expected)
