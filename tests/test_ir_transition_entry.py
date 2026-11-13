# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_share_transition import planted

from kilora import compute
from kilora.ir import check_invariants
from kilora.ir.default import default_program
from kilora.program import interpret


def test_transition_application_matches_native(monkeypatch):
    monkeypatch.delenv("KILORA_SHARE_TRANSITION_KIND", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    torch.manual_seed(115)
    s = dataclasses.replace(
        planted(), down=torch.randn(12, 16), up=torch.randn(12, 16), per_expert=4
    )
    matrix = compute.share_transition(s)
    last = torch.softmax(torch.randn(2, 7, 3), -1)
    program = default_program("transition", sizes={"batch": 2, "tokens": 7, "experts": 3})
    output = interpret(program, {"last_weights": last, "matrix": matrix}, {})
    assert torch.equal(output["weights"], last @ matrix)
    check_invariants("transition", output)


@pytest.mark.parametrize("empty", [False, True])
def test_entry_logits_match_native(monkeypatch, empty):
    for name in ("KILORA_DERIVED_DIFFUSE", "KILORA_DERIVED_MODE", "KILORA_DERIVED_ROWS"):
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(116)
    s = dataclasses.replace(
        planted(), down=torch.randn(12, 16), up=torch.randn(12, 16), per_expert=4, tau=1.25
    )
    x = torch.randn(2, 7, 16)
    if empty:
        s.down[:4].zero_()
        x[0, 0].zero_()
    kin, rows, _, _, _ = compute.derived_keys(s)
    reads = (x @ kin.flatten(0, 1).T).float().unflatten(-1, kin.shape[:2])
    energy = torch.linalg.vector_norm(x, dim=-1, dtype=torch.float32).square()
    inputs = {
        "reads": reads,
        "energy": energy,
        "rows": rows,
        "key_width": torch.tensor(float(kin.shape[-1])),
        "tau": torch.tensor(s.tau),
        "epsilon": torch.tensor(compute.SCORE_EPS),
        "temperature_in": torch.tensor(1.0),
    }
    program = default_program("entry", sizes={"batch": 2, "tokens": 7, "experts": 3, "per": 4})
    output = interpret(program, inputs, {})
    assert torch.equal(output["logits"], compute.derived_logits(s, x))
    check_invariants("entry", output)


def test_prepared_cosine_transition_matches_native_with_zero_row(monkeypatch):
    from kilora.ir.default import prepare_cosine_transition

    monkeypatch.delenv("KILORA_SHARE_TRANSITION_KIND", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    torch.manual_seed(1151)
    s = dataclasses.replace(
        planted(), down=torch.randn(12, 16), up=torch.randn(12, 16), per_expert=4
    )
    s.up[:4].zero_()
    expected = compute.share_transition(s)
    got = prepare_cosine_transition(s.down, s.up, s.per_expert)
    assert torch.equal(got, expected)
    assert torch.equal(got[0], torch.zeros_like(got[0]))
    assert torch.all(got >= 0)
    assert torch.allclose(got[1:].sum(-1), torch.ones(2), rtol=0, atol=1e-6)


@pytest.mark.parametrize("empty", [False, True])
def test_entry_from_input_matches_native(monkeypatch, empty):
    for name in ("KILORA_DERIVED_DIFFUSE", "KILORA_DERIVED_MODE", "KILORA_DERIVED_ROWS"):
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(117)
    s = dataclasses.replace(
        planted(), down=torch.randn(12, 16), up=torch.randn(12, 16), per_expert=4, tau=1.25
    )
    x = torch.randn(2, 7, 16)
    if empty:
        s.down[:4].zero_()
        x[0, 0].zero_()
    keys, rows, _, _, _ = compute.derived_keys(s)
    inputs = {
        "x": x,
        "keys": keys,
        "rows": rows,
        "key_width": torch.tensor(float(keys.shape[-1])),
        "tau": torch.tensor(s.tau),
        "epsilon": torch.tensor(compute.SCORE_EPS),
        "temperature_in": torch.tensor(1.0),
    }
    program = default_program(
        "entry", "from_input", sizes={"batch": 2, "tokens": 7, "experts": 3, "per": 4, "key": 16}
    )
    output = interpret(program, inputs, {})
    assert torch.equal(output["logits"], compute.derived_logits(s, x))
    check_invariants("entry", output)


@pytest.mark.parametrize("per,key", [(4, 16), (12, 16), (5, 3)])
def test_derived_basis_preparation_matches_native(monkeypatch, per, key):
    from kilora.ir.default import prepare_derived_bases

    monkeypatch.delenv("KILORA_DERIVED_ROWS", raising=False)
    torch.manual_seed(118 + per)
    s = dataclasses.replace(
        planted(), down=torch.randn(3 * per, key), up=torch.randn(3 * per, key), per_expert=per
    )
    s.down[:per].zero_()
    s.up[per : 2 * per].zero_()
    kin, rin, kout, rout, _ = compute.derived_keys(s)
    actual_in, count_in = prepare_derived_bases(s.down, per)
    actual_out, count_out = prepare_derived_bases(s.up, per)
    assert torch.equal(actual_in, kin)
    assert torch.equal(count_in, rin)
    assert torch.equal(actual_out, kout)
    assert torch.equal(count_out, rout)


def test_prepared_basis_through_entry_matches_native_entries(monkeypatch):
    from kilora.ir.default import prepare_derived_bases

    for name in (
        "KILORA_DERIVED_DIFFUSE",
        "KILORA_DERIVED_MODE",
        "KILORA_DERIVED_ROWS",
        "KILORA_DERIVED_EXCLUSION",
        "KILORA_DERIVED_ENTRY_TEMP",
    ):
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(119)
    s = dataclasses.replace(
        planted(), down=torch.randn(12, 16), up=torch.randn(12, 16), per_expert=4, tau=1.1
    )
    x = torch.randn(2, 7, 16)
    keys, rows = prepare_derived_bases(s.down, s.per_expert)
    program = default_program(
        "entry", "from_input", sizes={"batch": 2, "tokens": 7, "experts": 3, "per": 4, "key": 16}
    )
    inputs = {
        "x": x,
        "keys": keys,
        "rows": rows,
        "key_width": torch.tensor(16.0),
        "tau": torch.tensor(s.tau),
        "epsilon": torch.tensor(compute.SCORE_EPS),
        "temperature_in": torch.tensor(1.0),
    }
    logits = interpret(program, inputs, {})["logits"]
    assert torch.equal(logits, compute.derived_logits(s, x))
    with compute.dropout_key(22, 3):
        expected = compute.derived_entries(s, x, "tiny", 2)
    monkeypatch.setattr(compute, "derived_logits", lambda _s, _x: logits)
    with compute.dropout_key(22, 3):
        actual = compute.derived_entries(s, x, "tiny", 2)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("experts,per,key", [(2, 1, 2), (5, 3, 8), (4, 8, 13), (3, 9, 33)])
def test_prepared_cosine_varied_shapes(monkeypatch, experts, per, key):
    from kilora.ir.default import prepare_cosine_transition

    monkeypatch.delenv("KILORA_SHARE_TRANSITION_KIND", raising=False)
    monkeypatch.delenv("KILORA_DERIVED_DIFFUSE", raising=False)
    torch.manual_seed(120 + per)
    s = dataclasses.replace(
        planted(),
        down=torch.randn(experts * per, key),
        up=torch.randn(experts * per, key),
        per_expert=per,
    )
    s.down[:per].zero_()
    s.up[-per:].zero_()
    assert torch.equal(prepare_cosine_transition(s.down, s.up, per), compute.share_transition(s))


def test_derived_basis_rank_deficient_matches_native(monkeypatch):
    from kilora.ir.default import prepare_derived_bases

    monkeypatch.delenv("KILORA_DERIVED_ROWS", raising=False)
    torch.manual_seed(121)
    down = torch.randn(12, 16)
    down[1:4] = down[0]
    down[4:8] = 0
    s = dataclasses.replace(planted(), down=down, up=torch.randn(12, 16), per_expert=4)
    kin, rows, _, _, _ = compute.derived_keys(s)
    result, counts = prepare_derived_bases(s.down, s.per_expert)
    assert torch.equal(result, kin)
    assert torch.equal(counts, rows)
    assert counts.tolist() == [1, 0, 4]
