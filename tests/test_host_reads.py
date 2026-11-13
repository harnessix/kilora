# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora import modfile
from kilora.compute import LayerSet
from kilora.host_copy import all_finite, host_copies, host_copy_map, nonzero_flags

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def sync_counts(monkeypatch):
    counts = {"bool": 0, "cpu": 0, "item": 0, "sync": 0}
    for name, attr in (("bool", "__bool__"), ("cpu", "cpu"), ("item", "item")):
        original = getattr(torch.Tensor, attr)

        def counted(self, *a, _original=original, _name=name, **k):
            counts[_name] += 1
            return _original(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, attr, counted)
    sync = torch.cuda.synchronize

    def counted_sync(*a, **k):
        counts["sync"] += 1
        return sync(*a, **k)

    monkeypatch.setattr(torch.cuda, "synchronize", counted_sync)
    return counts


def _tensors(n, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = [torch.randn(3, 4, generator=gen).to(device) for _ in range(n)]
    out[0] = out[0].to(torch.bfloat16)
    out.append(torch.randn(4, 3, generator=gen).to(device).t())
    out.append(torch.zeros(0, device=device))
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_host_copies_match_the_old_copies_bit_for_bit(device):
    values = _tensors(5, device)
    for old in (lambda v: v.cpu().clone(), lambda v: v.cpu().contiguous()):
        new = host_copies(values, old)
        for a, b in zip(new, values, strict=True):
            ref = old(b)
            assert a.device.type == "cpu" and a.dtype == ref.dtype and a.shape == ref.shape
            assert torch.equal(a.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8))
    mapped = host_copy_map({f"k{i}": v for i, v in enumerate(values)}, lambda v: v.cpu())
    assert list(mapped) == [f"k{i}" for i in range(len(values))]


@pytest.mark.parametrize("device", DEVICES)
def test_the_finite_verdict_matches_the_old_check(device):
    values = _tensors(4, device)
    old = all(bool(torch.isfinite(v).all()) for v in values)
    assert all_finite(values) == old is True
    for bad in (float("nan"), float("inf")):
        broken = [*values[:2], torch.tensor([1.0, bad], device=device), *values[2:]]
        assert all_finite(broken) is False
    assert all_finite([]) is True


@pytest.mark.parametrize("device", DEVICES)
def test_the_nonzero_flags_match_the_old_check(device):
    values = [
        torch.zeros(3, device=device),
        torch.tensor([0.0, 2.0], device=device),
        torch.zeros(0, device=device),
        torch.tensor([-0.0], device=device),
    ]
    assert nonzero_flags(values) == [bool(v.count_nonzero()) for v in values]


def _old_digest(manifest, tensors):
    import hashlib
    import json

    h = hashlib.sha256()
    body = {k: v for k, v in manifest.items() if k != "digest"}
    h.update(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(tensors):
        t = tensors[name].contiguous().cpu()
        h.update(f"\n{name}|{str(t.dtype).removeprefix('torch.')}|{list(t.shape)}\n".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return "sha256-" + h.hexdigest()


def _old_layout(tensors):
    chunks = []
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        chunks.append(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return chunks


@pytest.mark.parametrize("device", DEVICES)
def test_the_digest_and_the_saved_bytes_stay_the_same(device):
    values = _tensors(4, device)
    tensors = {f"t{i}": v for i, v in enumerate(values)}
    manifest = {"id": "x", "digest": "old"}
    assert modfile.compute_digest(manifest, tensors) == _old_digest(manifest, tensors)
    _, chunks = modfile._layout(tensors)
    assert chunks == _old_layout(tensors)


@pytest.mark.parametrize("device", DEVICES)
def test_the_host_reads_do_not_grow_with_the_tensor_count(device, sync_counts):
    def reads(n):
        values = [v.float() for v in _tensors(n, device, seed=n)]
        for key in sync_counts:
            sync_counts[key] = 0
        all_finite(values)
        nonzero_flags(values)
        return sync_counts["bool"] + sync_counts["item"] + sync_counts["sync"]

    assert reads(3) == reads(12)


def _old_gates(ins, parts, active):
    return [
        torch.ones_like(part)
        if active.sum() == 1
        else (part * active.sum()).clamp(max=cap)
        if route == "multiply"
        else 1 + part
        if route == "add"
        else torch.ones_like(part)
        for (route, cap), part in zip(ins, parts, strict=True)
    ]


def _new_gates(ins, parts, active):
    total = active.sum()
    single = total == 1
    return [
        torch.where(single, torch.ones_like(part), (part * total).clamp(max=cap))
        if route == "multiply"
        else torch.where(single, torch.ones_like(part), 1 + part)
        if route == "add"
        else torch.ones_like(part)
        for (route, cap), part in zip(ins, parts, strict=True)
    ]


@pytest.mark.parametrize("active", [[True, False, False], [True, True, False], [True] * 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_route_gates_chosen_on_the_device_are_the_same_bits(active, dtype):
    gen = torch.Generator().manual_seed(7)
    parts = [torch.rand(2, 5, 1, generator=gen).to(dtype) for _ in range(3)]
    ins = [("multiply", 1.5), ("add", 2.0), ("none", 1.0)]
    mask = torch.tensor(active)
    for old, new in zip(_old_gates(ins, parts, mask), _new_gates(ins, parts, mask), strict=True):
        assert old.dtype == new.dtype and torch.equal(old, new)


def test_the_gate_site_in_compute_has_no_host_read_per_call():
    import inspect

    from kilora import compute

    assert "if active.sum() == 1" not in inspect.getsource(compute)


def test_the_column_keys_built_on_the_device_are_the_same_bits():
    from kilora import compute

    s = LayerSet(
        router=torch.randn(8, 32),
        keys_in=torch.randn(3, 2, 8),
        per_expert=2,
        down=torch.randn(6, 32),
        up=torch.randn(6, 32),
        key="k",
        expert_ids=torch.tensor([5, 2**33 + 7, 2**31 - 1]),
    )
    ck = compute.column_keys(s)
    cp, cols = 8, s.up.shape[0]
    old = torch.zeros(cp, dtype=torch.int64)
    old[:cols] = ck.cpu()
    old = torch.where(old >= 2**31, old - 2**32, old).to(torch.int32)
    new = torch.zeros(cp, dtype=torch.int64, device=s.up.device)
    new[:cols] = ck.to(s.up.device)
    new = torch.where(new >= 2**31, new - 2**32, new).to(torch.int32)
    assert torch.equal(old, new)
