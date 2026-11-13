# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from test_attention_route import routed_set
from test_import_attention_plain import ATTN, MLP

from kilora import compute, ir
from kilora.ir.driver import run_layer_ir


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("score", ["keys", "read_share", "derived"])
def test_attention_parity(depth, form, adaptive, score, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x = torch.eye(4)[None]
    s = replace(routed_set(form), import_score=score, gate_mode="adaptive" if adaptive else "none")
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        expected = compute.run_layer(x, x, [s], depth, target=ATTN)
        with ir.interpret_all():
            actual = run_layer_ir(x, x, [s], depth, target=ATTN)
            ordinary = run_layer_ir(x, x, [s], depth, target=MLP)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual, ordinary)


@pytest.mark.parametrize("depth", [1, 3])
def test_attention_mixed_sets_use_the_ordinary_path(depth, monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x = torch.eye(4)[None]
    sets = [replace(routed_set(form), iteration="path") for form in ("input", "state")]
    with torch.no_grad(), compute.eager():
        expected = compute.run_layer(x, x, sets, depth, target=ATTN)
        with ir.interpret_all():
            actual = run_layer_ir(x, x, sets, depth, target=ATTN)
            ordinary = run_layer_ir(x, x, sets, depth, target=MLP)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual, ordinary)


def test_folded_import_stays_bit_exact_with_interpret_all():
    g = torch.Generator().manual_seed(0)
    x, y = (torch.randn(2, 4, 6, generator=g) for _ in range(2))
    s = compute.LayerSet(
        form="input",
        import_score="read_share",
        router=torch.randn(3, 6, generator=g),
        keys_in=torch.randn(2, 1, 3, generator=g),
        down=torch.randn(4, 6, generator=g),
        up=torch.randn(4, 6, generator=g),
        per_expert=2,
        gate_mode="adaptive",
        scale=0.731,
    )
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        expected = compute.run_layer(x, y, [s], 4, target="feed")
        with ir.interpret_all():
            actual = run_layer_ir(x, y, [s], 4, target="feed")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("target", [ATTN, MLP])
@pytest.mark.parametrize("iteration", ["accumulate", "path"])
@pytest.mark.parametrize("masks", ["sample", "expect"])
@pytest.mark.parametrize("imported_first", [False, True])
def test_derived_mixed_ir_preserves_dropout_positions(
    target, iteration, masks, imported_first, monkeypatch
):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_DERIVED_DIFFUSE", "0")
    monkeypatch.setenv("KILORA_MASKS", masks)
    x = torch.eye(4)[None]
    imported = replace(routed_set(), iteration=iteration)
    state = replace(routed_set("state"), iteration=iteration)
    sets = [imported, state] if imported_first else [state, imported]
    with torch.no_grad(), compute.eager(), compute.dropout_key(17, 4):
        expected = compute.run_layer(x, x, sets, 3, target=target)
        with ir.interpret_all():
            actual = run_layer_ir(x, x, sets, 3, target=target)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_multiple_derived_imports_keep_grouped_reads_in_ir(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x = torch.eye(4)[None]
    sets = [
        routed_set(),
        routed_set("state"),
        replace(routed_set(), down=torch.eye(4) * 0.4),
    ]
    with torch.no_grad(), compute.eager():
        expected = compute.run_layer(x, x, sets, 3, target=ATTN)
        with ir.interpret_all():
            actual = run_layer_ir(x, x, sets, 3, target=ATTN)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
