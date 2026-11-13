# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from test_lowrank_depth import _sets
from test_route_freeze import _call, _freeze

from kilora import compute
from kilora import route_freeze as R


def _fold(sets, gates, lead):
    parts = []
    for layer, qb in zip(sets, gates, strict=True):
        up = layer.up.float()
        up = up * qb.float() if qb is not None else up.expand(*lead, *up.shape)
        parts.append(up)
    return torch.cat(parts, -2)


def _stored_tensors(obj, found, seen):
    mark = id(obj)
    if mark in seen:
        return
    seen.add(mark)
    if isinstance(obj, torch.Tensor):
        found.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _stored_tensors(value, found, seen)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _stored_tensors(value, found, seen)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _stored_tensors(getattr(obj, field.name), found, seen)


def _folded_shape(entry, sets):
    cols = sum(layer.up.shape[0] for layer in sets)
    return (*entry["scale"].shape[:-1], cols, sets[0].up.shape[-1])


def _assert_no_folded_storage(entry, shape):
    assert not dict.__contains__(entry, "up")
    found = []
    _stored_tensors(entry, found, set())
    assert not any(t.dtype == torch.float32 and tuple(t.shape) == shape for t in found)


def _product(sets, x, y0, scale, folded):
    dt = y0.dtype
    reads = []
    for layer in sets:
        src = y0 if layer.form == "state" else x
        reads.append(F.linear(src.to(dt), layer.down.to(dt)).float())
    coef = torch.cat(reads, -1) * scale.unsqueeze(-2)
    return y0 + torch.matmul(coef, folded).to(dt)


def _layers(dtype, n=2):
    layers = []
    for layer in _sets(n, 8, experts=2, per_expert=2, router_dim=4, seed=11):
        layers.append(
            dataclasses.replace(
                layer,
                up=layer.up.to(dtype),
                down=layer.down.to(dtype),
                form="state",
                gate_mode="none",
            )
        )
    return layers


def _record(layers, mode):
    gen = torch.Generator().manual_seed(19)
    batch, tokens = 2, 5
    steps = []
    for _ in range(2):
        step = []
        for layer in layers:
            experts = layer.keys_in.shape[0]
            logits = torch.randn(batch, tokens, experts, generator=gen)
            step.append(torch.softmax(logits, -1))
        steps.append(step)
    saved = []
    gates = {}
    for i, layer in enumerate(layers):
        use_gate = mode == "gate" or (mode == "mixed" and i == 0)
        if not use_gate:
            saved.append(None)
            continue
        qa = torch.rand(batch, tokens, 1, generator=gen)
        qb = torch.rand(batch, 1, layer.up.shape[-1], generator=gen)
        gates[("layer", i)] = (qa, qb)
        saved.append(qb.detach().clone())
    rec = {"routes": {"layer": steps}, "gates": gates}
    return rec, saved, (batch, tokens)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["none", "gate", "mixed"])
def test_frozen_product_matches_the_old_fold_and_stores_no_copy(dtype, mode):
    layers = _layers(dtype)
    rec, saved, lead_in = _record(layers, mode)
    entry = R.build(rec, layers, [0, 1], "layer", [None, None], 2, lead_in)
    if mode != "none":
        rec["gates"][("layer", 0)][1].add_(0.25)
    lead = tuple(rec["routes"]["layer"][0][0].shape[:-2])
    assert entry["scale"].shape[:-1] == lead
    shape = _folded_shape(entry, layers)
    _assert_no_folded_storage(entry, shape)
    for layer, weight, qb, original in zip(
        layers, entry["ups"], entry["up_gates"], saved, strict=True
    ):
        assert weight is layer.up
        if original is None:
            assert qb is None
        else:
            assert torch.equal(qb, original)
            assert tuple(qb.shape) == (*lead, 1, layer.up.shape[-1])
    if mode != "none":
        assert entry["up_gates"][0] is not rec["gates"][("layer", 0)][1]
        assert not torch.equal(entry["up_gates"][0], rec["gates"][("layer", 0)][1])
    folded = _fold(layers, saved, lead)
    assert tuple(folded.shape) == shape
    first, second = entry["up"], entry["up"]
    assert torch.equal(first, folded)
    assert torch.equal(second, folded)
    assert first.data_ptr() != second.data_ptr()
    _assert_no_folded_storage(entry, shape)
    assert first.data_ptr() not in {t.data_ptr() for t in _held(entry)}
    gen = torch.Generator().manual_seed(23)
    y0 = torch.randn(2, 5, layers[0].up.shape[-1], generator=gen).to(dtype)
    x = torch.randn(2, 5, layers[0].down.shape[-1], generator=gen).to(dtype)
    assert torch.equal(R.output(entry, x, y0), _product(layers, x, y0, entry["scale"], folded))


def _held(entry):
    found = []
    _stored_tensors(entry, found, set())
    return found


@pytest.mark.parametrize("gate", ["none", "adaptive"])
def test_live_state_freeze_replays_the_rebuilt_product(gate):
    layers = _freeze(
        [dataclasses.replace(s, gate_mode=gate) for s in _sets(2, 16, 4, 2, 8, seed=3)]
    )
    cache: dict = {}
    _call(layers, 3, 0.4, cache)
    entry = cache["layer"]
    lead = entry["scale"].shape[:-1]
    shape = _folded_shape(entry, layers)
    _assert_no_folded_storage(entry, shape)
    for layer, weight, qb in zip(layers, entry["ups"], entry["up_gates"], strict=True):
        assert weight is layer.up
        if gate == "none":
            assert qb is None
        else:
            assert qb is not None and qb.shape[-2:] == (1, layer.up.shape[-1])
    folded = _fold(layers, entry["up_gates"], lead)
    assert torch.equal(entry["up"], folded)
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(5))
    x = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(6))
    assert torch.equal(R.output(entry, x, y0), _product(layers, x, y0, entry["scale"], folded))
    assert torch.equal(_call(layers, 3, 0.7, cache, x=x, y0=y0, seed=9), R.output(entry, x, y0))


def test_import_chain_freeze_stores_no_folded_up(monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_FIXED_READ", raising=False)
    monkeypatch.setenv("KILORA_FUSED", "0")
    gen = torch.Generator().manual_seed(71)
    layer = dataclasses.replace(
        _sets(1, 16, 4, 2, 8, seed=71)[0],
        form="input",
        down=torch.randn(8, 16, generator=gen),
        route_freeze=0.3,
        gate_mode="none",
    )
    x = torch.randn(2, 9, 16, generator=gen)
    y0 = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(72))
    cache: dict = {}
    _call([layer], 3, 0.3, cache, x=x, y0=y0)
    entry = cache["layer"]
    assert entry.get("chain_sets") is not None
    shape = _folded_shape(entry, [layer])
    _assert_no_folded_storage(entry, shape)
    assert entry["ups"][0] is layer.up
    assert entry["up_gates"] == [None]
    folded = _fold([layer], entry["up_gates"], entry["scale"].shape[:-1])
    assert torch.equal(entry["up"], folded)
    assert not dict.__contains__(entry, "up")


def _weight_cast_keys(layer):
    weights = (layer.up, layer.down)
    hits = []
    for key in compute._DERIVED:
        if key[0] != "as-torch.float32":
            continue
        for src in key[1:]:
            if src is None:
                continue
            ptr, _version, shape, _stride, dtype, device = src
            for weight in weights:
                if (
                    ptr == weight.data_ptr()
                    and tuple(shape) == tuple(weight.shape)
                    and dtype == weight.dtype
                    and device == str(weight.device)
                ):
                    hits.append(key)
    return hits


@pytest.mark.parametrize("gate", ["none", "adaptive"])
def test_import_sampling_keeps_no_float32_weight_cast(monkeypatch, gate):
    monkeypatch.delenv("KILORA_IMPORT_FIXED_READ", raising=False)
    monkeypatch.setenv("KILORA_FUSED", "0")
    gen = torch.Generator().manual_seed(3)
    dtype = torch.bfloat16
    layer = compute.LayerSet(
        router=torch.randn(4, 8, generator=gen).to(dtype),
        keys_in=torch.randn(2, 1, 4, generator=gen).to(dtype),
        down=torch.randn(4, 8, generator=gen).to(dtype),
        up=torch.randn(4, 8, generator=gen).to(dtype),
        per_expert=2,
        form="input",
        carry="drop",
        import_score="keys",
        iteration="accumulate",
        gate_mode=gate,
    )
    x = torch.randn(2, 3, 8, generator=gen).to(dtype)
    y = torch.randn(2, 3, 8, generator=gen).to(dtype)
    saved = dict(compute._DERIVED)
    compute._DERIVED.clear()
    try:
        with torch.no_grad():
            compute.run_layer(x, y, [layer], 3, target="census")
        assert _weight_cast_keys(layer) == []
        kinds = [key[0] for key in compute._DERIVED]
        assert "import-read-cross" in kinds
        assert any(kind.startswith("depth-metric-") for kind in kinds)
    finally:
        compute._DERIVED.clear()
        compute._DERIVED.update(saved)
