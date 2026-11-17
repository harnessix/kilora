# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from kilora import compute


@pytest.mark.parametrize("width", [16, 64, 2048])
def test_chance_score_by_rank(monkeypatch, width):
    ranks = torch.tensor([0, 1, 4, 16])
    keys = torch.zeros(4, 16, width)
    for expert, rank in enumerate(ranks.tolist()):
        keys[expert, :rank] = torch.eye(width)[:rank]
    monkeypatch.setattr(compute, "derived_keys", lambda s: (keys, ranks, None, None, None))
    monkeypatch.setattr(compute, "derived_diffusion", lambda: False)
    setting = SimpleNamespace(tau=1.0, form="input", ir_parts={})
    count = 24000 if width < 2048 else 6000
    x = torch.randn(1, count, width, generator=torch.Generator().manual_seed(27))
    scores = compute._derived_logits(setting, x)
    assert torch.isfinite(scores).all()
    assert torch.equal(scores[..., 0], torch.zeros_like(scores[..., 0]))
    assert scores.mean((0, 1)).abs().max() < 0.07
    assert abs(scores[..., 1].std().item() - 1) < 0.09
    if width == 16:
        assert torch.equal(scores[..., 3], torch.zeros_like(scores[..., 3]))


def test_state_score_keeps_previous_rule(monkeypatch):
    gen = torch.Generator().manual_seed(81)
    keys = torch.linalg.qr(torch.randn(16, 4, generator=gen)).Q.T.reshape(2, 2, 16)
    ranks = torch.full((2,), 2)
    monkeypatch.setattr(compute, "derived_keys", lambda s: (keys, ranks, None, None, None))
    monkeypatch.setattr(compute, "derived_diffusion", lambda: False)
    setting = SimpleNamespace(tau=1.0, form="state", ir_parts={}, expert_mask=None)
    x = torch.randn(1, 5, 16, generator=gen)
    proj = (x @ keys.flatten(0, 1).T).unflatten(-1, (2, 2))
    energy = torch.linalg.vector_norm(x, dim=-1, keepdim=True).square()
    expected = proj.square().sum(-1) / (energy + compute.SCORE_EPS)
    expected = expected / (ranks.float() / 16)
    assert torch.equal(compute.derived_logits(setting, x), expected)


def test_ir_chance_score_matches_eager(monkeypatch):
    from kilora.ir.driver import _run

    gen = torch.Generator().manual_seed(92)
    keys = torch.zeros(4, 16, 16)
    ranks = torch.tensor([0, 1, 4, 16])
    for e, rank in enumerate(ranks.tolist()):
        keys[e, :rank] = torch.eye(16)[:rank]
    x = torch.randn(1, 5, 16, generator=gen)
    monkeypatch.setattr(compute, "derived_keys", lambda s: (keys, ranks, None, None, None))
    monkeypatch.setattr(compute, "derived_diffusion", lambda: False)
    setting = SimpleNamespace(tau=2.0, form="input", ir_parts={}, expert_mask=None)
    inputs = {"x": x, "keys": keys, "rows": ranks}
    for name, value in (
        ("key_width", 16),
        ("tau", 2),
        ("epsilon", compute.SCORE_EPS),
        ("temperature_in", 1),
    ):
        inputs[name] = torch.tensor(float(value))
    actual = _run(
        "entry",
        "chance_from_input",
        inputs,
        {"batch": 1, "tokens": 5, "experts": 4, "per": 16, "key": 16},
    )["logits"]
    assert torch.equal(actual, compute.derived_logits(setting, x))


def _shared_input_layer(out_width):
    up = torch.zeros(2, out_width)
    up[0, 1] = 8
    up[1, 0] = 8
    return compute.LayerSet(
        form="input",
        import_score="derived",
        router=torch.eye(out_width),
        keys_in=torch.zeros(2, 1, out_width),
        down=torch.eye(4)[:2],
        up=up,
        per_expert=1,
        carry="drop",
        gate_mode="none",
    )


@pytest.mark.parametrize("out_width", [4, 6])
@pytest.mark.parametrize("batched", [False, True])
def test_full_walks_score_shared_input_after_different_entries(monkeypatch, out_width, batched):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "0")
    layer = _shared_input_layer(out_width)
    x = torch.tensor([[[2.0, 1.0, 0.0, 0.0]]])
    y0 = torch.zeros(1, 1, out_width)
    entries = [torch.eye(2)[i].reshape(1, 1, 2) for i in range(2)]
    with torch.no_grad(), compute.eager():
        expected = compute._gate(
            compute.pooled_derived_logits(layer, x, "layer") * compute.derived_lift_after(),
            layer,
            0.0,
        )
        if batched:
            _, states, _ = compute._derived_batched_paths(layer, x, y0, None, "layer", entries, 2)
        else:
            states = []
            for entry in entries:
                stats = {}
                compute._run_input(
                    x, y0, [layer], [0], 2, [None], "layer", entry=entry, stats=stats
                )
                states.append(stats)
    for state in states:
        torch.testing.assert_close(state["gates"], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("out_width", [4, 6])
@pytest.mark.parametrize("engine", ["eager", "ir"])
def test_keys_chain_scores_shared_input_at_every_step(monkeypatch, out_width, engine):
    import contextlib

    from kilora import ir

    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_FUSED", "0")
    layer = _shared_input_layer(out_width)
    x = torch.tensor([[[2.0, 1.0, 0.0, 0.0]]])
    y0 = torch.zeros(1, 1, out_width)
    gates = []
    with torch.no_grad(), compute.eager():
        expected = compute._gate(compute.pooled_derived_logits(layer, x, "layer"), layer, 0.0)
        with (
            ir.interpret_all() if engine == "ir" else contextlib.nullcontext(),
            compute.recording_shares(lambda _, rows: gates.append(rows[0].clone())),
        ):
            compute.run_layer(x, y0, [layer], 2, target="layer")
    assert len(gates) == 2
    assert all(torch.equal(gate, expected) for gate in gates)
