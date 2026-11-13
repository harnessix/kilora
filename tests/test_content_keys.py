# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_modes import DIM, LAYERS, Toy, _lora

from kilora import adapter as adapter_module
from kilora import compose, subspace_keys
from kilora.compute import recording_routes
from kilora.multi_import import import_loras

LABELS = torch.tensor([0] * 32 + [1] * 32)


def _tokens(seed=3):
    g = torch.Generator().manual_seed(seed)
    u, v = torch.randn(DIM, generator=g), torch.randn(DIM, generator=g)
    x = torch.where(LABELS[:, None] == 0, u, v) * 2 + 0.5 * torch.randn(64, DIM, generator=g)
    return x[None]


def _paths(tmp_path, seeds):
    model = Toy().eval()
    paths = []
    for i, seed in enumerate(seeds):
        p = tmp_path / f"l{i}.safetensors"
        _lora(model, p, rank=4, seed=seed, alpha=4.0, layers=LAYERS)
        paths.append(str(p))
    return model, paths


def _routing(adapter, call):
    with torch.no_grad(), recording_routes() as routes:
        call()
    out = {}
    for name, steps in routes.items():
        p0 = steps[0][0][..., 0].reshape(-1).float()
        out[name.split("#")[0]] = (p0, p0)
    return out


def _cluster_gap(gap):
    return float((gap[LABELS == 0].mean() - gap[LABELS == 1].mean()).abs())


def _run(tmp_path, monkeypatch, seeds, contrast=False):
    model, paths = _paths(tmp_path, seeds)
    routed = import_loras(paths, model).adapter_set
    x = _tokens()
    monkeypatch.setattr(adapter_module, "AUTO_KEYS", False)
    m0 = Toy().eval()
    a0 = compose(m0, [routed], depth=2)
    random = _routing(a0, lambda: m0(x))
    a0.remove()
    monkeypatch.setattr(adapter_module, "AUTO_KEYS", True)
    monkeypatch.setattr(subspace_keys, "CONTRAST", contrast)
    m = Toy().eval()
    adapter = compose(m, [routed], depth=2)
    with torch.no_grad():
        m(x)
    assert adapter._set_info[0]["key_source"] == "states"
    return random, _routing(adapter, lambda: m(x))


@pytest.mark.parametrize("contrast", [False, True])
@pytest.mark.parametrize("top", ["", "0.8"])
@pytest.mark.parametrize("score", ["keys", "read_share"])
def test_state_keys_split_two_loras_by_cluster(tmp_path, monkeypatch, contrast, top, score):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    monkeypatch.setenv("KILORA_KEY_TARGET", top)
    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    random, state = _run(tmp_path, monkeypatch, (11, 29), contrast)
    for t in state:
        gap_r, gap_s = _cluster_gap(random[t][0]), _cluster_gap(state[t][0])
        p0 = state[t][1]
        a, b = p0[LABELS == 0], p0[LABELS == 1]
        print(
            f"MODE {t} score={score} top={top or 0.6} contrast={contrast}: "
            f"p0 gap random {gap_r:.3f} state {gap_s:.3f} | "
            f"p0 A mean {a.mean():.3f} [{a.quantile(0.1):.2f},{a.quantile(0.9):.2f}] "
            f"B mean {b.mean():.3f} [{b.quantile(0.1):.2f},{b.quantile(0.9):.2f}]"
        )
        assert torch.isfinite(p0).all()
    p0 = state["blocks.0.mlp.layer2"][1]
    split = abs(float(p0[LABELS == 0].mean() - p0[LABELS == 1].mean()))
    if score == "read_share" or top or contrast:
        assert split >= 0.9, split


@pytest.mark.parametrize("score", ["keys", "read_share"])
def test_experts_reading_the_same_subspace_route_evenly(tmp_path, monkeypatch, score):
    from safetensors.torch import load_file, save_file

    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    model, paths = _paths(tmp_path, (7, 7))
    sd = load_file(paths[1])
    g = torch.Generator().manual_seed(99)
    for k in [k for k in sd if k.endswith("lora_up.weight")]:
        sd[k] = torch.randn(sd[k].shape, generator=g) * 0.2
    save_file(sd, paths[1])
    routed = import_loras(paths, model).adapter_set
    m = Toy().eval()
    adapter = compose(m, [routed], depth=2)
    x = _tokens()
    with torch.no_grad():
        y = m(x)
    assert torch.isfinite(y).all()
    from kilora.compute import orthonormal_keys

    for site in adapter.sites:
        k = orthonormal_keys(site.tensors["s0_keys_in"].detach().float())
        assert torch.isfinite(k).all()
        overlap = torch.einsum("mr,nr->mn", k[0], k[1]).pow(2).sum()
        assert abs(float(overlap) - k.shape[1]) < 1e-3


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_state_keys_split_separated_clusters_with_first_change_reference(monkeypatch, scope):
    from kilora import compute

    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "0")
    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    states = torch.eye(2)[:, None, :].expand(2, 32, 2).clone()
    layer = compute.LayerSet(
        router=torch.eye(2),
        keys_in=torch.eye(2).unsqueeze(1),
        down=torch.eye(2),
        up=torch.eye(2) * 0.2,
        tau=10.0,
        routing_scope=scope,
        gate_mode="adaptive",
    )
    routes = []
    with torch.no_grad(), compute.recording_shares(lambda _, gs: routes.append(gs[0])):
        output = compute.run_layer(states, states, [layer], 3)
    for weights in routes:
        split = (weights[0, :, 0].mean() - weights[1, :, 0].mean()).abs()
        assert split >= 0.9
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :, 0].eq(0), states[:, :, 0].eq(0))
    assert torch.equal(output[:, :, 1].eq(0), states[:, :, 1].eq(0))
