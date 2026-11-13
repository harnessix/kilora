# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.adaptive_gate import adaptive_keep, change_stats
from kilora.ir.defaults import default_bindings, default_program
from kilora.ir.parts import check_invariants
from kilora.program import interpret


@pytest.mark.parametrize("batch,tokens,cols,out", [(2, 7, 4, 11), (1, 13, 7, 19), (3, 5, 9, 4)])
@pytest.mark.parametrize("base_reference", [False, True])
@pytest.mark.parametrize("has_history", [False, True])
@pytest.mark.parametrize("coherence", [False, True])
@pytest.mark.parametrize("has_before", [False, True])
def test_adaptive_gate_matches_native(
    monkeypatch, has_history, coherence, has_before, batch, tokens, cols, out, base_reference
):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1" if base_reference else "0")
    generator = torch.Generator().manual_seed(641)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    z = rand(batch, tokens, cols)
    before = rand(batch, tokens, cols) if has_before else None
    history = rand(batch, tokens, cols) if has_history else None
    up = rand(cols, out)
    norm = rand(batch, tokens).abs() + 0.1
    rms = rand(batch, out).abs() + 0.1
    first_norm, first_rms = change_stats(z if before is None else before, up, up @ up.t())
    kappa = 1.7
    inputs = dict(
        coefficients=z,
        before=before if before is not None else torch.zeros_like(z),
        prior_history=history if history is not None else torch.zeros_like(z),
        has_history=torch.tensor(has_history),
        up=up,
        base_norm=norm,
        base_rms=rms,
        first_norm=torch.zeros_like(norm) if base_reference else first_norm,
        first_rms=torch.zeros_like(rms) if base_reference else first_rms,
        first_step=torch.tensor(not has_before and not base_reference),
    )
    sizes = dict(batch=batch, tokens=tokens, cols=cols, out=out)
    result = interpret(
        default_program("gate", "adaptive", sizes),
        inputs,
        default_bindings("gate", "adaptive", tokens=tokens, kappa=kappa, coherence=coherence),
    )
    expected = adaptive_keep(
        z,
        before,
        history,
        up,
        norm,
        rms,
        kappa,
        coherence,
        first_norm=first_norm,
        first_rms=first_rms,
    )
    for name, value in zip(("token", "channel", "history"), expected, strict=True):
        assert torch.equal(result[name], value), name
    check_invariants("gate", result, sizes=sizes)


def test_gate_zero_direction_and_parameter_checks(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    sizes = dict(batch=1, tokens=3, cols=2, out=4)
    z = torch.zeros(1, 3, 2)
    up = torch.ones(2, 4)
    inputs = dict(
        coefficients=z,
        before=z,
        prior_history=z,
        has_history=torch.tensor(True),
        up=up,
        base_norm=torch.ones(1, 3),
        base_rms=torch.ones(1, 4),
        first_norm=torch.zeros(1, 3),
        first_rms=torch.zeros(1, 4),
        first_step=torch.tensor(False),
    )
    actual = interpret(
        default_program("gate", "adaptive", sizes),
        inputs,
        default_bindings("gate", "adaptive", tokens=3),
    )
    expected = adaptive_keep(z, z, z, up, inputs["base_norm"], inputs["base_rms"])
    for name, value in zip(("token", "channel", "history"), expected, strict=True):
        assert torch.equal(actual[name], value)
    check_invariants("gate", actual, sizes=sizes)
    with pytest.raises(ValueError, match="kappa"):
        default_bindings("gate", "adaptive", tokens=3, kappa=0)
