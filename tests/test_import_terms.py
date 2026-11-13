# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.import_terms import PackTerm, needs_pack_moves, pack_directions, term_matrices


def test_host_pack_directions_use_input_basis():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    basis = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    outputs, inputs = pack_directions(
        ["first", "target"],
        {"first": [1, 2], "target": [2, 2]},
        q,
        {"target": basis},
        "target",
        "pack",
    )
    up, down = term_matrices([PackTerm("pack:out:0:1", "pack:in:0", 0.5)], outputs, inputs)
    assert torch.equal(up @ down, 0.5 * torch.outer(torch.tensor([4.0, 6.0]), basis[:, 0]))


def test_old_pack_has_no_directions():
    assert pack_directions(["target"], {"target": [1, 1]}, torch.ones(1, 1), {}, "target", "p") == (
        {},
        {},
    )


def test_rejects_mismatched_input_basis():
    with pytest.raises(ValueError, match="input basis"):
        pack_directions(
            ["target"],
            {"target": [1, 2]},
            torch.ones(1, 2),
            {"target": torch.ones(3, 1)},
            "target",
            "p",
        )


def test_single_term_is_outer_product():
    output = torch.tensor([1.0, -2.0, 4.0])
    input_ = torch.tensor([2.0, 3.0])
    up, down = term_matrices([PackTerm("o", "i", 0.5)], {"o": output}, {"i": input_})
    assert torch.equal(up @ down, 0.5 * torch.outer(output, input_))


def test_nonorthogonal_directions_are_not_projected():
    outputs = {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 2.0])}
    inputs = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([1.0, 1.0])}
    terms = [PackTerm("a", "x", 2.0), PackTerm("b", "y", -1.0)]
    up, down = term_matrices(terms, outputs, inputs)
    expected = 2 * torch.outer(outputs["a"], inputs["x"]) - torch.outer(outputs["b"], inputs["y"])
    assert torch.equal(up @ down, expected)


def test_pack_gain_threshold_and_unavailable_measurements():
    assert needs_pack_moves(0.24, 1.0)
    assert not needs_pack_moves(0.25, 1.0)
    assert not needs_pack_moves(0.5, 1.0)
    assert not needs_pack_moves(0.0, 0.0)
    assert not needs_pack_moves(float("nan"), 1.0)


def test_invalid_terms_fail_closed():
    vector = torch.ones(2)
    with pytest.raises(KeyError):
        term_matrices([PackTerm("missing", "i", 1.0)], {}, {"i": vector})
    with pytest.raises(ValueError):
        term_matrices([PackTerm("o", "i", float("inf"))], {"o": vector}, {"i": vector})
