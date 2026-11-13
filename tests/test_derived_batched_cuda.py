# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import threading

import pytest
import torch
from test_share_transition import planted

from kilora import compute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_derived_cuda_matches_loop(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setenv("KILORA_ADAPTIVE_KERNEL", "0")
    monkeypatch.setenv("KILORA_SHARE_TRANSITION", "1")
    gen = torch.Generator().manual_seed(710)
    width, rank, experts = 2048, 32, 3
    down = torch.randn(experts * rank, width, generator=gen) / width**0.5
    up = torch.randn(experts * rank, width, generator=gen) / (experts * rank) ** 0.5
    down[16:32] = 0
    up[16:32] = 0
    down[56:64] = 0
    up[56:64] = 0
    s = dataclasses.replace(
        planted(),
        router=torch.eye(experts, width, device="cuda"),
        keys_in=torch.eye(experts, device="cuda").unsqueeze(1),
        down=down.to("cuda", torch.bfloat16),
        up=up.to("cuda", torch.bfloat16),
        per_expert=rank,
        gate_mode="adaptive",
        path_width=3,
        route_freeze=0.4,
        scale=0.65,
    )
    x = torch.randn(1, 256, width, generator=gen).to("cuda", torch.bfloat16)
    y = torch.randn(1, 256, width, generator=gen).to("cuda", torch.bfloat16)
    original = compute._derived_path_note
    records = []
    current = []
    gate_records = []
    gate_current = []
    thread_id = threading.get_ident()
    split = compute.split_gate

    def gate(*args, **kwargs):
        result = split(*args, **kwargs)
        if threading.get_ident() == thread_id:
            gate_current.append((args[1].detach().clone(), *(v.detach().clone() for v in result)))
        return result

    monkeypatch.setattr(compute, "split_gate", gate)

    def note(stats, gates, logits, active=None):
        result = original(stats, gates, logits, active)
        if threading.get_ident() == thread_id:
            current.append(
                (
                    gates.detach().clone(),
                    stats["score"].clone(),
                    stats.get(
                        "converged", torch.zeros_like(stats["score"], dtype=torch.bool)
                    ).clone(),
                )
            )
        return result

    monkeypatch.setattr(compute, "_derived_path_note", note)
    outputs = []
    for enabled in ("0", "1", "fast"):
        monkeypatch.setenv("KILORA_DERIVED_BATCHED", enabled)
        current = []
        gate_current = []
        with (
            torch.no_grad(),
            compute.derived_call(),
            compute.sampling_progress(0.2),
            compute.dropout_key(4, 1),
        ):
            outputs.append(compute.run_layer(x, y, [s], 3, target="layer"))
        records.append(current)
        gate_records.append(gate_current)
    assert len(records[0]) == len(records[1]) == 9
    for path in range(3):
        for step in range(3):
            for expected, actual in zip(
                records[0][path * 3 + step], records[1][step * 3 + path], strict=True
            ):
                torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    assert len(gate_records[0]) == len(gate_records[1]) == 9
    for path in range(3):
        for step in range(3):
            expected = gate_records[0][path * 3 + step]
            actual = gate_records[1][step * 3 + path]
            for before, after in zip(expected, actual, strict=True):
                torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-4)
    difference = (outputs[0].float() - outputs[1].float()).abs()
    for step in range(3):
        before = gate_records[0][step][0]
        after = gate_records[2][step * 3][0]
        print(f"fast path 0 step {step + 1} coefficient max", (before - after).abs().max().item())
    fast_difference = (outputs[0].float() - outputs[2].float()).abs()
    print("corrected output mean/max", difference.mean().item(), difference.max().item())
    print("fast output mean/max", fast_difference.mean().item(), fast_difference.max().item())
    assert difference.mean().item() < 0.5 / 255, (difference.mean().item(), difference.max().item())
