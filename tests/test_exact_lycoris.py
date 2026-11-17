# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.exact_lycoris import apply_exact, make_exact
from kilora.multi_import import lycoris_delta


@pytest.mark.parametrize("kind", ["lokr", "loha"])
def test_exact_components_match_dense_reference(kind):
    gen = torch.Generator().manual_seed(34)
    if kind == "lokr":
        tensors = {
            "lokr_w1": torch.randn(3, 2, generator=gen),
            "lokr_w2_a": torch.randn(4, 2, generator=gen),
            "lokr_w2_b": torch.randn(2, 5, generator=gen),
            "alpha": torch.tensor(3.0),
        }
    else:
        tensors = {
            "hada_w1_a": torch.randn(12, 2, generator=gen),
            "hada_w1_b": torch.randn(2, 10, generator=gen),
            "hada_w2_a": torch.randn(12, 2, generator=gen),
            "hada_w2_b": torch.randn(2, 10, generator=gen),
            "alpha": torch.tensor(3.0),
        }
    weight = -0.7
    parts = make_exact(tensors, kind, weight, 10000)
    x = torch.randn(2, 4, 10, generator=gen)
    expected = x @ (lycoris_delta(tensors)[0] * weight).t()
    torch.testing.assert_close(apply_exact(parts, x, block=3), expected, rtol=1e-5, atol=1e-5)
    size = sum(t.numel() * t.element_size() for t in parts)
    assert make_exact(tensors, kind, weight, size - 1) is None
    assert make_exact(tensors, kind, weight, size) is not None


@pytest.mark.parametrize("kind", ["lokr", "loha"])
def test_routed_import_applies_exact_change_and_keeps_shares(tmp_path, monkeypatch, kind):
    import dataclasses

    from test_multi_import import Q, ToyAnima, _loha, _lokr, lora_weights, to_keys, write

    from kilora import compose, compute
    from kilora.multi_import import import_loras

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = ToyAnima().eval()
    lyco, delta = (_lokr if kind == "lokr" else _loha)()
    plain = lora_weights(model, layers=[Q], rank=4, seed=5)
    paths = [write(tmp_path, "lyco", lyco), write(tmp_path, "plain", to_keys(plain, "kohya"))]
    imported = import_loras(paths, model, rank=4, lycoris_apply="exact")
    expert = imported.adapter_set.experts[Q]
    assert 0 in expert.exact
    size = sum(t.numel() * t.element_size() for t in expert.exact[0])
    assert size <= (expert.down[0].numel() + expert.up[0].numel()) * 4
    adapter = compose(model, [imported.adapter_set], mix="stack")
    layer = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))[0]
    layer = dataclasses.replace(layer, import_score="read_share")
    x = torch.randn(2, 5, 16)
    y0 = torch.randn(2, 5, 16)
    shares = []
    with compute.recording_shares(lambda target, gates: shares.append(gates[0].clone())):
        out = compute.run_layer(x, y0, [layer], 1, target=Q)
    old_shares = []
    with compute.recording_shares(lambda target, gates: old_shares.append(gates[0].clone())):
        compute.run_layer(x, y0, [dataclasses.replace(layer, exact={})], 1, target=Q)
    assert torch.equal(shares[0], old_shares[0])
    down, up = plain[Q]
    wanted = (
        y0 + (x @ delta.t()) * shares[0][..., :1] + (x @ down.t() @ up.t()) * shares[0][..., 1:2]
    )
    torch.testing.assert_close(out, wanted, rtol=2e-5, atol=2e-5)
    adapter.remove()


@pytest.mark.parametrize("depth", [1, 2, 2.5])
@pytest.mark.parametrize("mode", ["none", "adaptive"])
@pytest.mark.parametrize("fixed_read", [False, True])
def test_fused_exact_matches_eager_and_keeps_kernel_routes(
    tmp_path, monkeypatch, depth, mode, fixed_read
):
    import dataclasses
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("requires the Triton interpreter")
    pytest.importorskip("triton")
    from test_multi_import import Q, ToyAnima, _lokr, lora_weights, to_keys, write

    from kilora import compose, compute, fused
    from kilora.multi_import import import_loras

    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1" if fixed_read else "0")
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = ToyAnima().eval()
    lyco, _ = _lokr()
    plain = lora_weights(model, layers=[Q], rank=4, seed=5)
    imported = import_loras(
        [write(tmp_path, "lyco", lyco), write(tmp_path, "plain", to_keys(plain, "kohya"))],
        model,
        rank=4,
        lycoris_apply="exact",
    )
    adapter = compose(model, [imported.adapter_set], mix="stack")
    layer = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))[0]
    layer = dataclasses.replace(layer, import_score="read_share", gate_mode=mode)
    x = torch.randn(2, 5, 16)
    y0 = torch.randn(2, 5, 16)
    new_trace, old_trace = [], []
    with torch.no_grad():
        fused._TRACE.append(new_trace)
        try:
            actual = fused.run_fused(y0, [layer], depth, target=Q, x=x, name=f"{Q}#input0")
        finally:
            fused._TRACE.pop()
        fused._TRACE.append(old_trace)
        try:
            fused.run_fused(
                y0, [dataclasses.replace(layer, exact={})], depth, target=Q, x=x, name=f"{Q}#input0"
            )
        finally:
            fused._TRACE.pop()
        without_trace = fused.run_fused(y0, [layer], depth, target=Q, x=x, name=f"{Q}#input0")
        expected = compute._run_input(x, y0, [layer], [0], depth, [None], Q)
    assert len(new_trace) == len(old_trace)
    if fixed_read or depth == 1:
        assert all(torch.equal(a["p"], b["p"]) for a, b in zip(new_trace, old_trace, strict=True))
    else:
        assert all(torch.isfinite(step["p"]).all() for step in new_trace)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(without_trace, expected, rtol=2e-4, atol=2e-4)
    adapter.remove()


def test_report_lists_exact_scoring_and_memory_fallback(tmp_path, monkeypatch):
    from test_multi_import import Q, ToyAnima, _loha, _lokr, lora_weights, to_keys, write

    from kilora.multi_import import import_loras

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = ToyAnima().eval()
    plain = lora_weights(model, layers=[Q], rank=4, seed=5)
    lora_path = write(tmp_path, "plain", to_keys(plain, "kohya"))
    lokr_path = write(tmp_path, "lokr", _lokr()[0])
    exact = import_loras([lokr_path, lora_path], model, rank=4, lycoris_apply="exact")
    assert "energy kept 100% (exact change)" in exact.report()
    assert "energy kept by" not in exact.report()
    loha_path = write(tmp_path, "loha", _loha()[0])
    small = lora_weights(model, layers=[Q], rank=1, seed=5)
    small_path = write(tmp_path, "small", to_keys(small, "kohya"))
    fallback = import_loras([loha_path, small_path], model, rank=1, lycoris_apply="exact")
    assert 0 not in fallback.adapter_set.experts[Q].exact
    assert Q in fallback.report() and "exceeded" in fallback.report()


def test_exact_components_survive_adapter_and_mod_roundtrip(tmp_path, monkeypatch):
    from test_multi_import import Q, ToyAnima, _lokr, lora_weights, to_keys, write

    from kilora import compose, load_mod
    from kilora.multi_import import import_loras

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = ToyAnima().eval()
    plain = lora_weights(model, layers=[Q], rank=4, seed=5)
    imported = import_loras(
        [write(tmp_path, "lyco", _lokr()[0]), write(tmp_path, "plain", to_keys(plain, "kohya"))],
        model,
        rank=4,
        lycoris_apply="exact",
    )
    adapter = compose(model, [imported.adapter_set], mix="stack")
    original = imported.adapter_set.experts[Q].exact[0]
    exported = adapter.to_sets()[0].experts[Q].exact[0]
    assert all(torch.equal(a, b) for a, b in zip(original, exported, strict=True))
    file = tmp_path / "test.kmod"
    adapter.save(file, id="test", name="Test", license="CC0-1.0", base_model={"family": "tiny"})
    loaded = load_mod(file).sets[0].experts[Q].exact[0]
    assert all(torch.equal(a, b) for a, b in zip(original, loaded, strict=True))
    adapter.remove()


def test_plain_routing_keeps_existing_product_bits():
    from kilora import compute

    gen = torch.Generator().manual_seed(909)
    layer = compute.LayerSet(
        router=torch.randn(4, 8, generator=gen),
        keys_in=torch.randn(2, 1, 4, generator=gen),
        down=torch.randn(6, 8, generator=gen),
        up=torch.randn(6, 8, generator=gen),
        per_expert=3,
        form="input",
        import_score="read_share",
        gate_mode="none",
    )
    x = torch.randn(2, 5, 8, generator=gen)
    y0 = torch.randn(2, 5, 8, generator=gen)
    shares = []
    with compute.recording_shares(lambda target, gates: shares.append(gates[0])):
        actual = compute.run_layer(x, y0, [layer], 1)
    read = torch.nn.functional.linear(x, layer.down).float()
    expected = y0 + (read * shares[0].float().repeat_interleave(3, dim=-1)) @ layer.up
    assert torch.equal(actual, expected)


def test_frozen_routes_apply_exact_change():
    from kilora import compute, route_freeze

    gen = torch.Generator().manual_seed(19)
    w1 = torch.randn(2, 2, generator=gen)
    w2 = torch.randn(4, 4, generator=gen)
    layer = compute.LayerSet(
        router=torch.randn(4, 8, generator=gen),
        keys_in=torch.randn(2, 1, 4, generator=gen),
        down=torch.randn(6, 8, generator=gen),
        up=torch.randn(6, 8, generator=gen),
        per_expert=3,
        form="input",
        exact={0: (w1, w2)},
        scale=0.7,
    )
    x = torch.randn(2, 5, 8, generator=gen)
    y0 = torch.randn(2, 5, 8, generator=gen)
    gate = torch.rand(2, 5, 2, generator=gen)
    record = {"routes": {"layer": [[gate]]}, "gates": {}}
    entry = route_freeze.build(record, [layer], [0], "layer", [None], 1, y0.shape[:-1])
    actual = route_freeze.output(entry, x, y0)
    baseline = route_freeze.output({**entry, "exact": []}, x, y0)
    approx = (x @ layer.down[:3].t()) @ layer.up[:3]
    extra = (x @ torch.kron(w1, w2).t() - approx) * gate[..., 0].mean(-1)[:, None, None] * 0.7
    torch.testing.assert_close(actual, baseline + extra, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fast", [False, True])
def test_mixed_state_and_input_preserve_routes_and_apply_exact(fast, monkeypatch):
    import dataclasses

    from kilora import compute

    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    gen = torch.Generator().manual_seed(90)
    w1 = torch.randn(2, 2, generator=gen)
    w2 = torch.randn(4, 4, generator=gen)

    def layer(form, exact=None):
        return compute.LayerSet(
            router=torch.randn(4, 8, generator=gen),
            keys_in=torch.randn(2, 1, 4, generator=gen),
            down=torch.randn(6, 8, generator=gen),
            up=torch.randn(6, 8, generator=gen),
            per_expert=3,
            form=form,
            exact=exact or {},
            gate_mode="none",
            scale=0.7,
        )

    imported = layer("input", {0: (w1, w2)})
    state = layer("state")
    x = torch.randn(2, 5, 8, generator=gen)
    y0 = torch.randn(2, 5, 8, generator=gen)

    def run(setting):
        seen = []
        with (
            torch.no_grad(),
            compute.recording_shares(lambda target, gates: seen.append([g.clone() for g in gates])),
        ):
            if fast:
                out = compute._run_lowrank(y0, [state, setting], 2, False, x=x)
            else:
                out = compute._run_loop(y0, [state, setting], 2, False, 0, None, None, "t", x=x)
        return out, seen

    actual, shares = run(imported)
    baseline, old_shares = run(dataclasses.replace(imported, exact={}))
    for a, b in zip(shares, old_shares, strict=True):
        assert all(torch.equal(c, d) for c, d in zip(a, b, strict=True))
    weights = sum(step[1][..., :1] for step in shares) * 0.7
    approximate = (x @ imported.down[:3].t()) @ imported.up[:3]
    extra = (x @ torch.kron(w1, w2).t() - approximate) * weights
    torch.testing.assert_close(actual, baseline + extra, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("depth", [2, 4])
def test_batched_derived_walks_apply_exact_change(monkeypatch, depth):
    from kilora import compute
    from kilora.path_walks import select_weights

    monkeypatch.setenv("KILORA_DERIVED_MODE", "full")
    monkeypatch.setenv("KILORA_FUSED", "0")
    gen = torch.Generator().manual_seed(31)
    layer = compute.LayerSet(
        router=torch.randn(4, 8, generator=gen),
        keys_in=torch.randn(2, 1, 4, generator=gen),
        down=torch.randn(6, 8, generator=gen),
        up=torch.randn(6, 8, generator=gen),
        per_expert=3,
        form="input",
        exact={0: (torch.randn(2, 2, generator=gen), torch.randn(4, 4, generator=gen))},
        import_score="derived",
        gate_mode="none",
    )
    x = torch.randn(2, 5, 8, generator=gen)
    y0 = torch.randn(2, 5, 8, generator=gen)
    entries = [torch.tensor([1.0, 0.0]).expand(2, 5, 2), torch.tensor([0.0, 1.0]).expand(2, 5, 2)]
    with torch.no_grad():
        actual, states, _ = compute._derived_batched_paths(
            layer, x, y0, None, "test", entries, depth
        )
        paths = []
        separate_states = []
        for entry in entries:
            stats = {}
            paths.append(
                compute._run_input(
                    x, y0, [layer], [0], depth, [None], "test", entry=entry, stats=stats
                )
            )
            stats.setdefault("converged", torch.zeros_like(stats["score"], dtype=torch.bool))
            separate_states.append(stats)
        weights = select_weights(separate_states, layer.path_select)
        expected = sum((path - y0) * weights[..., i : i + 1] for i, path in enumerate(paths))
    for a, b in zip(states, separate_states, strict=True):
        assert torch.equal(a["gates"] > 0, b["gates"] > 0)
        assert torch.equal(a["gates"].argmax(-1), b["gates"].argmax(-1))
        torch.testing.assert_close(a["gates"], b["gates"], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("width", [1, 2])
def test_path_walks_use_exact_change(width):
    import dataclasses

    from kilora import compute
    from kilora.path_walks import _combine, select_weights, walks_batched

    gen = torch.Generator().manual_seed(71)
    w1 = torch.randn(2, 2, generator=gen)
    w2 = torch.randn(4, 4, generator=gen)
    layer = compute.LayerSet(
        router=torch.randn(4, 8, generator=gen),
        keys_in=torch.randn(2, 1, 4, generator=gen),
        down=torch.randn(6, 8, generator=gen),
        up=torch.randn(6, 8, generator=gen),
        per_expert=3,
        form="input",
        iteration="path",
        path_width=width,
        exact={0: (w1, w2)},
        gate_mode="none",
    )
    x = torch.randn(2, 5, 8, generator=gen)
    y0 = torch.randn(2, 5, 8, generator=gen)
    with torch.no_grad():
        seen = []
        with compute.recording_shares(lambda target, gates: seen.append(gates[0].clone())):
            actual = compute._run_path(y0, [layer], 2, False, 0, None, "test", x=x)
        baseline = compute._run_path(
            y0, [dataclasses.replace(layer, exact={})], 2, False, 0, None, "test", x=x
        )
        if width == 1:
            gate = seen[-1]
        else:
            records = walks_batched(y0, [layer], 2, False, 0, "test", [None], x, width)
            weights = select_weights(records, layer.path_select)
            gate = _combine([record["gates"][0] for record in records], weights)
        approx = (x @ layer.down[:3].t()) @ layer.up[:3]
        extra = (x @ torch.kron(w1, w2).t() - approx) * gate[..., :1]
    torch.testing.assert_close(actual, baseline + extra, rtol=2e-5, atol=2e-5)
