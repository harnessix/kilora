# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open

from kilora import key_cache, key_graph
from kilora.bake import ModulePlan, TargetPlan
from kilora.comfyui.state_keys import KeyPass
from kilora.compute import LayerSet
from kilora.path_walks import transition_prior


def layer():
    eye = torch.eye(2)
    return LayerSet(
        router=eye,
        keys_in=eye[:, None, :],
        keys_out=eye[:, None, :],
        down=eye,
        up=eye,
        key="toy",
        routing_select="top1",
        gate_mode="none",
    )


def test_transition_and_powers():
    s = layer()
    matrix, powers = key_graph.build(s, 4)
    prior = transition_prior(s)
    torch.testing.assert_close(matrix, prior / prior.sum(-1, keepdim=True))
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2))
    torch.testing.assert_close(powers[3], matrix @ matrix @ matrix)
    torch.testing.assert_close(powers[4], powers[3] @ matrix)


def test_full_depth_probe_on_planted_layer():
    s = layer()
    target = TargetPlan(sets=[s], depth=4, name="toy")
    plan = ModulePlan(parts=[(0, -1, target)])
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    args = {"input": x, "timestep": torch.ones(1), "c": {}}
    probe = KeyPass([plan], [], "toy")
    probe._measure_graph(lambda inp, t: plan.run(inp, inp), args)
    assert s.graph_distances.shape == (4,)
    assert s.graph_distances.max() < 1e-6
    report = next(iter(key_graph.graph_report(plan).values()))
    assert report["qualifies"]
    assert abs(report["rho"] - 1.0) < 1e-6
    assert 0 <= report["damped_rho"] < 1
    assert report["entropy_rate"] == 0
    assert report["loop_share"] == 1
    assert all(row["two_hop"] < 1e-6 for row in report["hops"][1:])
    s.keys_out = torch.randn(s.keys_out.shape, generator=torch.Generator().manual_seed(17))
    _, unchanged = key_graph.build(s, 4)
    torch.testing.assert_close(key_graph.distances([torch.eye(2)] * 4, unchanged), torch.zeros(4))
    s.up = torch.randn(s.up.shape, generator=torch.Generator().manual_seed(17))
    matrix, powers = key_graph.build(s, 4)
    weights = [torch.eye(2)] * 4
    assert key_graph.distances(weights, powers)[1:].min() > 0.01
    assert matrix.shape == (2, 2)


def test_graph_probe_includes_input_and_state_sets_on_attention_mounts(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    args = {"input": x, "timestep": torch.ones(1), "c": {}}
    results = []
    for name in ("layers.0.attention.to_q", "layers.0.feed_forward.w1"):
        sets = [layer(), replace(layer(), key="import", form="input")]
        plan = ModulePlan(parts=[(0, -1, TargetPlan(sets=sets, depth=4, name=name))])
        KeyPass([plan], [], "toy")._measure_graph(
            lambda inp, t, current=plan: current.run(inp, inp), args
        )
        assert all(s.graph_distances is not None for s in sets)
        results.append([s.graph_distances for s in sets])
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_cache_formats_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("KILORA_KEY_GRAPH", raising=False)
    keys = {
        "layer": {
            "keys_in": torch.eye(2)[:, None],
            "keys_out": torch.eye(2)[:, None],
            "rows_in": torch.ones(2, dtype=torch.long),
            "rows_out": torch.ones(2, dtype=torch.long),
        }
    }
    path = key_cache.store("same", keys, 0.5)
    old = Path(path).read_bytes()
    with safe_open(path, framework="pt") as f:
        assert '"format": 2' in f.metadata()["kilora_key_cache"]
    monkeypatch.setenv("KILORA_KEY_GRAPH", "0")
    key_cache.store("same", keys, 0.5)
    assert Path(path).read_bytes() == old
    monkeypatch.setenv("KILORA_KEY_GRAPH", "1")
    assert key_cache.load("same") is None
    matrix, powers = key_graph.build(layer(), 4)
    keys["layer"].update(
        graph_transition=matrix,
        graph_powers=powers,
        graph_distances=torch.zeros(4),
        graph_stats=torch.zeros(5),
        graph_hops=torch.zeros(3, 2),
        graph_merges=torch.zeros(0, 4, dtype=torch.long),
    )
    key_cache.store("same", keys, 0.5)
    with safe_open(path, framework="pt") as f:
        assert '"format": 4' in f.metadata()["kilora_key_cache"]
    loaded = key_cache.load("same")
    for name, value in keys["layer"].items():
        torch.testing.assert_close(loaded["layer"][name], value)


def test_imported_graph_is_cached_and_restored(denoiser, chain, folders, monkeypatch):
    from safetensors.torch import load_file, save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _call, _sets

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    chain = None
    for i in range(3):
        path = folders["loras"] / f"s{i}.safetensors"
        data = load_file(str(path))
        gen = torch.Generator().manual_seed(100 + i)
        stem = "lora_unet_layers_0_feed_forward_w1"
        data[f"{stem}.lora_down.weight"] = torch.randn(2, 16, generator=gen)
        data[f"{stem}.lora_up.weight"] = torch.randn(32, 2, generator=gen)
        save_file(data, str(path))
        (chain,) = N.KiLoRAImportLoRA().add(path.name, 1.0, sets=chain)

    monkeypatch.setenv("KILORA_KEY_GRAPH", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    base = FakePatcher(FakeBaseModel(denoiser))
    patched, _ = patching.apply(base, chain, 3, patching.LiveEnv(), joint=True)
    x, cond = torch.randn(1, 5, 4), torch.ones(1, 2)
    first = _call(patched, x, cond)
    sets = _sets(patched)
    iterated = sets
    assert iterated
    assert all(s.graph_distances is not None and len(s.graph_distances) == 3 for s in iterated)
    saved = [
        (
            s.graph_distances.clone(),
            s.graph_stats.clone(),
            s.graph_hops.clone(),
            s.graph_merges.clone(),
        )
        for s in iterated
    ]
    again, _ = patching.apply(base, chain, 3, patching.LiveEnv(), joint=True)
    torch.testing.assert_close(_call(again, x, cond), first)
    restored = _sets(again)
    for a, b in zip(restored, saved, strict=True):
        for actual, expected in zip(
            (a.graph_distances, a.graph_stats, a.graph_hops, a.graph_merges), b, strict=True
        ):
            torch.testing.assert_close(actual, expected, equal_nan=True)
    assert "cache" in patching.applied(again)["key_pass"].report


from test_comfyui import denoiser, folders
from test_comfyui_state_keys import chain


def test_graph_probe_does_not_change_output(denoiser, chain, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_comfyui_state_keys import _call

    from kilora.comfyui import patching

    base = FakePatcher(FakeBaseModel(denoiser))
    x, cond = torch.randn(1, 5, 4), torch.ones(1, 2)
    monkeypatch.delenv("KILORA_KEY_GRAPH", raising=False)
    plain, _ = patching.apply(base, chain, 3, patching.LiveEnv(), joint=True)
    torch.manual_seed(8)
    expected = _call(plain, x, cond)
    after_plain = torch.rand(4)
    monkeypatch.setenv("KILORA_KEY_GRAPH", "1")
    graph, _ = patching.apply(base, chain, 3, patching.LiveEnv(), joint=True)
    torch.manual_seed(8)
    actual = _call(graph, x, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(4), after_plain, rtol=0, atol=0)


def test_loop_gain_and_import():
    s = layer()
    s.up = torch.tensor([[0.5, 1.0], [0.0, -0.25]])
    rho, damped = key_graph.loop_gain(s, torch.tensor([0.5, 0.8]))
    assert abs(rho - 0.5) < 1e-6
    assert abs(damped - 0.25) < 1e-6
    s.form = "input"
    assert key_graph.loop_gain(s, torch.ones(2)) == (None, None)


def test_sequence_entropy_and_merge_contexts():
    repeated = [torch.eye(4)] * 8
    entropy, loop, _ = key_graph.sequence_metrics(repeated)
    assert entropy == 0 and loop == 1
    generator = torch.Generator().manual_seed(42)
    choices = torch.randint(4, (12000, 8), generator=generator)
    weights = [torch.nn.functional.one_hot(choices[:, i], 4) for i in range(8)]
    entropy, loop, _ = key_graph.sequence_metrics(weights)
    assert abs(entropy - 2.0) < 0.02
    assert loop < 0.001
    choices = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 3]])
    weights = [torch.nn.functional.one_hot(choices[:, i], 5) for i in range(4)]
    _, _, merges = key_graph.sequence_metrics(weights)
    assert [1, 2, 1, 4] in merges.tolist()


def test_hops_match_planted_graph():
    t = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    powers = torch.stack([torch.eye(2), t, t @ t])
    weights = [torch.eye(2)]
    for _ in range(4):
        weights.append(weights[-1] @ t)
    hops = key_graph.hop_distances(weights, powers)
    assert hops[:, 0].max() < 1e-6
    assert hops[1:, 1].max() < 1e-6
