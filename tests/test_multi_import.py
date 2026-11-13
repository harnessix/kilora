# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kilora import LoRAImportError, compose, compute
from kilora.multi_import import LoRASource, MultiImport, import_loras

DIM, CTX, HIDDEN = 16, 8, 24


class _Attn(nn.Module):
    def __init__(self, kv_dim):
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM)
        self.k_proj = nn.Linear(kv_dim, DIM)
        self.v_proj = nn.Linear(kv_dim, DIM)
        self.output_proj = nn.Linear(DIM, DIM)

    def forward(self, x, kv):
        q, k, v = self.q_proj(x), self.k_proj(kv), self.v_proj(kv)
        w = torch.softmax(q @ k.transpose(-1, -2) / DIM**0.5, dim=-1)
        return self.output_proj(w @ v)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIM, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, DIM)

    def forward(self, x):
        return self.layer2(torch.nn.functional.gelu(self.layer1(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn(DIM)
        self.cross_attn = _Attn(CTX)
        self.mlp = _MLP()

    def forward(self, x, ctx):
        x = x + self.self_attn(x, x)
        x = x + self.cross_attn(x, ctx)
        return x + self.mlp(x)


class ToyAnima(nn.Module):
    def __init__(self, n_blocks=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList(_Block() for _ in range(n_blocks))

    def forward(self, x, ctx):
        for block in self.blocks:
            x = block(x, ctx)
        return x


LAYERS = [
    "blocks.0.self_attn.q_proj",
    "blocks.0.cross_attn.k_proj",
    "blocks.1.self_attn.output_proj",
    "blocks.1.mlp.layer1",
    "blocks.1.mlp.layer2",
]


@pytest.fixture
def model():
    return ToyAnima().eval()


def lora_weights(model, layers=LAYERS, rank=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name in layers:
        m = model.get_submodule(name)
        out[name] = (
            torch.randn(rank, m.in_features, generator=gen) * 0.3,
            torch.randn(m.out_features, rank, generator=gen) * 0.3,
        )
    return out


def to_keys(weights, layout, alpha=None):
    sd = {}
    for name, (a, b) in weights.items():
        if layout == "kohya":
            stem = "lora_unet_" + name.replace(".", "_")
            sd[f"{stem}.lora_down.weight"] = a
            sd[f"{stem}.lora_up.weight"] = b
            if alpha is not None:
                sd[f"{stem}.alpha"] = torch.tensor(float(alpha))
        else:
            prefix = "diffusion_model." if layout == "comfyui" else "transformer."
            sd[f"{prefix}{name}.lora_A.weight"] = a
            sd[f"{prefix}{name}.lora_B.weight"] = b
            if alpha is not None:
                sd[f"{prefix}{name}.alpha"] = torch.tensor(float(alpha))
    return sd


def write(tmp_path, name, sd, metadata=None):
    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata=metadata)
    return str(path)


def expert_delta(ex, e, x):
    return x @ ex.down[e, 0].t() @ ex.up[e, 0]


def only_expert(adapter_set, e):
    experts = {}
    for t, ex in adapter_set.experts.items():
        experts[t] = dataclasses.replace(
            ex,
            down=ex.down[e : e + 1],
            up=ex.up[e : e + 1],
            keys_in=ex.keys_in[e : e + 1],
            keys_out=ex.keys_out[e : e + 1],
            rows_in=ex.rows_in[e : e + 1],
            rows_out=ex.rows_out[e : e + 1],
        )
    return dataclasses.replace(adapter_set, experts=experts)


def check_plain_lora(model, adapter_set, weights, scale, tol=1e-5):
    ref = ToyAnima().eval()
    compose(model, [adapter_set], mix="stack")
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, (a, b) in weights.items():
            m = model.get_submodule(name)
            x = torch.randn(2, 5, m.in_features, generator=gen)
            want = ref.get_submodule(name)(x) + scale * (x @ a.t() @ b.t())
            got = m(x)
            err = ((got - want).norm() / want.norm()).item()
            assert err < tol, (name, err)


@pytest.mark.parametrize("layout", ["kohya", "peft", "comfyui"])
def test_each_format_imports_as_the_plain_lora(model, tmp_path, layout):
    w = lora_weights(model, rank=4)
    path = write(tmp_path, f"lora-{layout}", to_keys(w, layout, alpha=2.0))
    result = import_loras([path], model)
    assert isinstance(result, MultiImport)
    assert result.sources[0]["format"] == layout
    assert set(result.adapter_set.experts) == set(LAYERS)
    assert result.unmatched == {f"lora-{layout}": []}
    check_plain_lora(model, result.adapter_set, w, 0.5)


def test_kohya_names_resolve_against_the_model(model, tmp_path):
    w = lora_weights(model, layers=["blocks.0.self_attn.output_proj", "blocks.1.mlp.layer2"])
    result = import_loras([write(tmp_path, "k", to_keys(w, "kohya"))], model)
    assert list(result.adapter_set.experts) == list(w)


def test_set_fields(model, tmp_path):
    path = write(tmp_path, "a", to_keys(lora_weights(model), "peft"))
    s = import_loras([path], model, router_dim=16).adapter_set
    assert (s.form, s.aggregation, s.strength, s.key_source) == (
        "input",
        "accumulate",
        1.0,
        "random",
    )
    assert s.router_dim == 16


@pytest.mark.parametrize("alpha,rank,weight", [(8.0, 4, 1.0), (1.0, 4, 0.7), (None, 2, -1.5)])
def test_alpha_rank_and_weight_scale_exactly(model, tmp_path, alpha, rank, weight):
    w = lora_weights(model, rank=rank, seed=3)
    path = write(tmp_path, "s", to_keys(w, "kohya", alpha=alpha))
    result = import_loras([LoRASource(path=path, weight=weight)], model)
    base = alpha / rank if alpha is not None else 1.0
    assert result.sources[0]["scale"] == base
    assert result.sources[0]["alpha"] == alpha
    gen = torch.Generator().manual_seed(1)
    for name, (a, b) in w.items():
        x = torch.randn(3, a.shape[1], generator=gen)
        want = weight * base * (x @ a.t() @ b.t())
        got = expert_delta(result.adapter_set.experts[name], 0, x)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    check_plain_lora(model, result.adapter_set, w, weight * base)


def test_peft_adapter_config_gives_the_scale(model, tmp_path):
    w = lora_weights(model, rank=4)
    path = write(tmp_path, "p", to_keys(w, "peft"))
    (tmp_path / "adapter_config.json").write_text('{"r": 4, "lora_alpha": 2}')
    result = import_loras([path], model)
    assert result.sources[0]["scale"] == 0.5
    check_plain_lora(model, result.adapter_set, w, 0.5)


def test_one_expert_per_lora_in_order_and_padding(model, tmp_path):
    ranks = [2, 5, 3]
    weights = [lora_weights(model, rank=r, seed=10 + i) for i, r in enumerate(ranks)]
    paths = [
        write(tmp_path, f"l{i}", to_keys(w, "kohya", alpha=1.0)) for i, w in enumerate(weights)
    ]
    result = import_loras(paths, model)
    assert [s["name"] for s in result.sources] == ["l0", "l1", "l2"]
    assert [s["rank"] for s in result.sources] == ranks
    gen = torch.Generator().manual_seed(2)
    for name in LAYERS:
        ex = result.adapter_set.experts[name]
        assert ex.down.shape[:3] == (3, 1, 5)
        x = torch.randn(4, ex.down.shape[-1], generator=gen)
        for e, (w, r) in enumerate(zip(weights, ranks, strict=True)):
            a, b = w[name]
            assert torch.count_nonzero(ex.down[e, 0, r:]) == 0
            assert torch.count_nonzero(ex.up[e, 0, r:]) == 0
            torch.testing.assert_close(
                expert_delta(ex, e, x), (x @ a.t() @ b.t()) / r, rtol=1e-6, atol=1e-6
            )
    for e, (w, r) in enumerate(zip(weights, ranks, strict=True)):
        check_plain_lora(ToyAnima().eval(), only_expert(result.adapter_set, e), w, 1.0 / r)


def test_padding_does_not_change_the_output(model, tmp_path):
    small = lora_weights(model, rank=2, seed=1)
    big = lora_weights(model, rank=6, seed=2)
    p_small = write(tmp_path, "small", to_keys(small, "peft"))
    p_big = write(tmp_path, "big", to_keys(big, "peft"))
    alone = import_loras([p_small], model).adapter_set
    padded = only_expert(import_loras([p_small, p_big], model).adapter_set, 0)
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(1, 3, DIM)
    ctx = torch.randn(1, 4, CTX, generator=gen)
    outs = []
    for s in (alone, padded):
        m = ToyAnima().eval()
        compose(m, [s], mix="stack")
        with torch.no_grad():
            outs.append(m(x, ctx))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-6, atol=1e-6)


def test_layer_changed_by_some_loras_only(model, tmp_path):
    w0 = lora_weights(model, layers=LAYERS[:2], seed=1)
    w1 = lora_weights(model, layers=LAYERS[1:], seed=2)
    result = import_loras(
        [write(tmp_path, "a", to_keys(w0, "peft")), write(tmp_path, "b", to_keys(w1, "peft"))],
        model,
    )
    experts = result.adapter_set.experts
    assert list(experts) == LAYERS
    for name in LAYERS:
        assert experts[name].down.shape[0] == 2
    assert torch.count_nonzero(experts[LAYERS[0]].up[1]) == 0
    assert torch.count_nonzero(experts[LAYERS[4]].up[0]) == 0
    assert torch.count_nonzero(experts[LAYERS[1]].up[0]) > 0
    assert torch.count_nonzero(experts[LAYERS[1]].up[1]) > 0
    assert [s["targets"] for s in result.sources] == [2, 4]


def test_unmatched_keys_are_reported_and_skipped(model, tmp_path):
    w = lora_weights(model, layers=LAYERS[:2])
    sd = to_keys(w, "kohya", alpha=3.0)
    extra = {
        "lora_unet_blocks_7_self_attn_q_proj.lora_down.weight": torch.zeros(3, DIM),
        "lora_unet_blocks_7_self_attn_q_proj.lora_up.weight": torch.zeros(DIM, 3),
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.zeros(3, 4),
        "lora_unet_blocks_0_self_attn_q_proj.hada_w1_a": torch.zeros(3, 3),
    }
    path = write(tmp_path, "mixed", {**sd, **extra})
    result = import_loras([path], model)
    assert sorted(result.unmatched["mixed"]) == sorted(extra)
    assert list(result.adapter_set.experts) == LAYERS[:2]
    report = result.report()
    assert "mixed" in report and "4 skipped" in report and "6 keys matched" in report
    assert "lora_unet_blocks (3)" in report and "lora_te1_text_model_encoder_layers (1)" in report
    check_plain_lora(model, result.adapter_set, w, 1.0)


def test_shape_mismatch_names_the_file_and_key(model, tmp_path):
    w = lora_weights(model, layers=LAYERS[:1])
    sd = to_keys(w, "peft")
    sd["transformer.blocks.1.mlp.layer1.lora_A.weight"] = torch.zeros(3, DIM + 1)
    sd["transformer.blocks.1.mlp.layer1.lora_B.weight"] = torch.zeros(HIDDEN, 3)
    good = write(tmp_path, "good", to_keys(w, "peft"))
    bad = write(tmp_path, "bad", sd)
    with pytest.raises(LoRAImportError) as info:
        import_loras([good, bad], model)
    msg = str(info.value)
    assert "bad.safetensors" in msg and "transformer.blocks.1.mlp.layer1.lora_A.weight" in msg

    sd = to_keys(w, "kohya")
    sd["lora_unet_blocks_1_mlp_layer2.lora_down.weight"] = torch.zeros(3, HIDDEN)
    sd["lora_unet_blocks_1_mlp_layer2.lora_up.weight"] = torch.zeros(DIM + 2, 3)
    with pytest.raises(LoRAImportError, match=r"bad2\.safetensors.*lora_unet_blocks_1_mlp_layer2"):
        import_loras([write(tmp_path, "bad2", sd)], model)


def test_sources_records(model, tmp_path):
    w = lora_weights(model, rank=4)
    p0 = write(tmp_path, "first", to_keys(w, "kohya", alpha=2.0), {"modelspec.license": "MIT"})
    p1 = write(tmp_path, "second", to_keys(lora_weights(model, LAYERS[:3], rank=2), "comfyui"))
    result = import_loras(
        [
            LoRASource(path=p0, weight=0.5, repo="someone/first", revision="abc123"),
            LoRASource(path=p1, name="style", license="Apache-2.0"),
        ],
        model,
    )
    with open(p0, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert {
        k: v
        for k, v in result.sources[0].items()
        if k not in ("coverage", "coverage_counts", "skipped_reasons", "global_mount_counts")
    } == {
        "name": "first",
        "file": "first.safetensors",
        "sha256": digest,
        "repo": "someone/first",
        "revision": "abc123",
        "license": "MIT",
        "format": "kohya",
        "rank": 4,
        "alpha": 2.0,
        "scale": 0.5,
        "weight": 0.5,
        "targets": 5,
    }
    second = result.sources[1]
    assert (second["name"], second["file"], second["license"]) == (
        "style",
        "second.safetensors",
        "Apache-2.0",
    )
    assert (second["format"], second["rank"], second["alpha"], second["scale"]) == (
        "comfyui",
        2,
        None,
        1.0,
    )
    assert (second["repo"], second["revision"], second["targets"]) == (None, None, 3)
    assert result.adapter_set.imported_from == {"sources": result.sources}
    p2 = write(tmp_path, "third", to_keys(w, "peft"))
    assert import_loras([p2], model).sources[0]["license"] == "NOASSERTION"
    lines = result.report().splitlines()
    assert any("first" in ln and "kohya" in ln and "rank 4" in ln for ln in lines)
    assert any("style" in ln and "comfyui" in ln and "rank 2" in ln for ln in lines)


def test_plain_string_is_a_source(model, tmp_path):
    path = write(tmp_path, "x", to_keys(lora_weights(model), "peft"))
    a = import_loras([path], model)
    b = import_loras([LoRASource(path=path)], model)
    assert a.sources == b.sources


def test_duplicate_names_and_no_match_stop(model, tmp_path):
    w = lora_weights(model)
    p = write(tmp_path, "x", to_keys(w, "peft"))
    with pytest.raises(LoRAImportError, match="unique"):
        import_loras([p, p], model)
    none = write(tmp_path, "none", {"lora_te_x.lora_down.weight": torch.zeros(2, 2)})
    with pytest.raises(LoRAImportError, match=r"none\.safetensors"):
        import_loras([none], model)


@pytest.mark.parametrize("depth", [1, 3])
def test_fast_path_equals_the_loop(tmp_path, monkeypatch, depth):
    base = ToyAnima().eval()
    paths = [
        write(tmp_path, f"f{i}", to_keys(lora_weights(base, rank=2 + i, seed=i), "kohya", 1.0))
        for i in range(3)
    ]
    adapter_set = import_loras(paths, base).adapter_set
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(2, 6, DIM, generator=gen)
    ctx = torch.randn(2, 4, CTX, generator=gen)
    outs = {}
    for fast in (False, True):
        monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
        m = ToyAnima().eval()
        compose(m, [adapter_set], depth=depth, mix="stack")
        with torch.no_grad():
            outs[fast] = m(x, ctx)
    torch.testing.assert_close(outs[False], outs[True], rtol=1e-4, atol=1e-5)
    m = ToyAnima().eval()
    compose(m, [adapter_set], depth=depth, mix="stack")
    with torch.no_grad():
        assert torch.equal(m(x, ctx), outs[True])


Q = "blocks.0.self_attn.q_proj"
QK = "lora_unet_blocks_0_self_attn_q_proj"


def _lokr(seed=1):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(4, 4, generator=g)
    w2_a, w2_b = torch.randn(4, 2, generator=g), torch.randn(2, 4, generator=g)
    keys = {
        f"{QK}.lokr_w1": w1,
        f"{QK}.lokr_w2_a": w2_a,
        f"{QK}.lokr_w2_b": w2_b,
        f"{QK}.alpha": torch.tensor(1.0),
    }
    return keys, torch.kron(w1, w2_a @ w2_b) * (1.0 / 2)


def _loha(seed=2):
    g = torch.Generator().manual_seed(seed)
    shapes = {"w1_a": (16, 2), "w1_b": (2, 16), "w2_a": (16, 2), "w2_b": (2, 16)}
    w = {n: torch.randn(*shape, generator=g) for n, shape in shapes.items()}
    keys = {f"{QK}.hada_{n}": t for n, t in w.items()} | {f"{QK}.alpha": torch.tensor(2.0)}
    return keys, (w["w1_a"] @ w["w1_b"]) * (w["w2_a"] @ w["w2_b"]) * (2.0 / 2)


def _delta(result, e=0):
    ex = result.adapter_set.experts[Q]
    return ex.up[e, 0].t() @ ex.down[e, 0]


@pytest.mark.parametrize("kind", ["lokr", "loha"])
def test_lycoris_layers_are_exact_at_full_rank_and_cut_by_svd(model, tmp_path, kind, monkeypatch):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    keys, delta = (_lokr if kind == "lokr" else _loha)()
    path = write(tmp_path, kind, keys)
    full = import_loras([path], model, rank=16)
    torch.testing.assert_close(_delta(full), delta, atol=1e-4, rtol=1e-4)
    assert full.sources[0]["svd_error"] < 1e-5 and kind in full.sources[0]["format"]
    cut = import_loras([path], model, rank=2)
    sv = torch.linalg.svdvals(delta.double())
    want = float(sv[2:].norm() / sv.norm())
    assert abs(cut.sources[0]["svd_error"] - want) < 1e-5 and want > 0.01
    assert "cut to rank 2 by SVD" in cut.report()


def test_routed_lokr_cut_reports_the_energy_kept_per_layer(model, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    keys, delta = _lokr()
    path = write(tmp_path, "lokr", keys)
    sv = torch.linalg.svdvals(delta.double())
    full = import_loras([path], model, rank=16)
    assert abs(full.sources[0]["svd_energy"][Q] - 1.0) < 1e-5
    assert "energy kept" in full.report() and "Warning" not in full.report()
    cut = import_loras([path], model, rank=1)
    want = float(sv[:1].square().sum() / sv.square().sum())
    assert abs(cut.sources[0]["svd_energy"][Q] - want) < 1e-4 and want < 0.9
    notes = cut.cut_notes()
    assert any(n.startswith("Warning") and Q in n and "mix=stack" in n for n in notes)


def test_mixed_import_of_a_lora_and_a_lokr_skips_other_modules(model, tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    w = lora_weights(model, rank=3)
    plain = write(tmp_path, "plain", to_keys(w, "kohya"))
    keys, delta = _lokr()
    keys["lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.lokr_w1"] = torch.zeros(2, 2)
    keys["lora_unet_llm_adapter_blocks_0_cross_attn_o_proj.lokr_w2"] = torch.zeros(4, 4)
    lokr = write(tmp_path, "lokr", keys)
    result = import_loras([plain, lokr], model)
    assert result.sources[1]["svd_rank"] == 3
    sv = torch.linalg.svdvals(delta.double())
    assert abs(result.sources[1]["svd_error"] - float(sv[3:].norm() / sv.norm())) < 1e-5
    assert "lora_unet_llm_adapter_blocks (2)" in result.report()
    torch.testing.assert_close(_delta(result, 0), w[Q][1] @ w[Q][0], atol=1e-5, rtol=1e-5)


def test_unknown_format_names_the_weights_it_found(model, tmp_path):
    path = write(tmp_path, "dora", {f"{QK}.dora_scale": torch.ones(16)})
    with pytest.raises(LoRAImportError, match="unknown format.*dora_scale"):
        import_loras([path], model)


def test_input_form_routes_are_recorded(model, tmp_path):
    paths = [
        write(tmp_path, f"l{i}", to_keys(lora_weights(model, rank=3, seed=i), "kohya"))
        for i in range(2)
    ]
    result = import_loras(paths, model)
    compose(model, [result.adapter_set], mix="stack")
    x = torch.randn(1, 4, DIM, generator=torch.Generator().manual_seed(0))
    ctx = torch.randn(1, 3, CTX, generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), compute.recording_routes() as routes:
        model(x, ctx)
    assert routes and all(len(v) >= 1 for v in routes.values())


def test_adaln_keys_become_a_fixed_section(tmp_path):
    from safetensors.torch import save_file

    model = ToyAnima()
    tree = ToyAnima()
    tree.blocks[0].adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 3 * DIM))
    gen = torch.Generator().manual_seed(4)
    a, b = torch.randn(2, DIM, generator=gen), torch.randn(DIM, 2, generator=gen)
    m, n = torch.randn(2, DIM, generator=gen), torch.randn(3 * DIM, 2, generator=gen)
    sd = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": a,
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": b,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_down.weight": m,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_up.weight": n,
        "lora_unet_blocks_0_adaln_modulation_cross_attn_1.alpha": torch.tensor(1.0),
        "lora_unet_blocks_7_adaln_modulation_mlp_1.lora_down.weight": m.clone(),
        "lora_unet_blocks_7_adaln_modulation_mlp_1.lora_up.weight": n.clone(),
    }
    path = tmp_path / "style.safetensors"
    save_file(sd, str(path))
    result = import_loras([LoRASource(path=str(path), weight=0.5)], model, full_model=tree)
    assert sorted(result.adapter_set.experts) == ["blocks.0.self_attn.q_proj"]
    (section,) = result.sections
    assert section.kind == "fixed-lora"
    down, up = section.data["blocks.0.adaln_modulation_cross_attn.1"]
    torch.testing.assert_close(up @ down, 0.5 * 0.5 * n @ m)
    assert any("blocks_7" in k for k in result.unmatched["style"])


def test_lycoris_cut_is_a_fast_seeded_randomized_svd():
    import time

    from kilora.multi_import import low_rank

    gen = torch.Generator().manual_seed(5)
    delta = torch.randn(512, 6, generator=gen) @ torch.randn(6, 768, generator=gen)
    t0 = time.time()
    down, up, error = low_rank(delta, 6, device="cpu")
    assert time.time() - t0 < 5.0
    assert error < 1e-4
    again = low_rank(delta, 6, device="cpu")
    assert torch.equal(again[0], down) and torch.equal(again[1], up)


def test_routed_lokr_expert_at_weight_one_is_the_exact_change(model, tmp_path, monkeypatch):
    import kilora.multi_import as mi

    monkeypatch.setattr(mi, "LOKR_ENERGY", 1.0)
    keys, delta = _lokr()
    path = write(tmp_path, "lokr", keys)
    result = import_loras([path], model, rank=1)
    torch.testing.assert_close(_delta(result), delta, atol=1e-4, rtol=1e-4)
    assert result.sources[0]["svd_ranks"][Q] == int(torch.linalg.matrix_rank(delta))


def test_routed_lokr_rank_grows_per_layer_to_the_energy_target(model, tmp_path):
    keys, delta = _lokr()
    path = write(tmp_path, "lokr", keys)
    result = import_loras([path], model, rank=1)
    sv = torch.linalg.svdvals(delta.double())
    share = sv.square().cumsum(0) / sv.square().sum()
    want = int((share < 0.95).sum()) + 1
    src = result.sources[0]
    assert src["svd_ranks"][Q] == want and want > 1
    assert abs(src["svd_energy"][Q] - float(share[want - 1])) < 1e-4
    assert "LoKr ranks" in result.report() and "Warning" not in result.report()


def test_kron_cut_equals_the_truncated_svd_of_the_full_product():
    from kilora.multi_import import _kron_spectrum

    gen = torch.Generator().manual_seed(9)
    w1, w2 = torch.randn(3, 4, generator=gen), torch.randn(5, 2, generator=gen)
    spec = _kron_spectrum({"lokr_w1": w1, "lokr_w2": w2}, 0.5)
    full = 0.5 * torch.kron(w1, w2)
    u, sv, vh = torch.linalg.svd(full.double(), full_matrices=False)
    down, up, error = spec.cut(4)
    want = (u[:, :4] * sv[:4]) @ vh[:4]
    torch.testing.assert_close((up @ down).double(), want, atol=1e-5, rtol=1e-5)
    assert abs(error - float(sv[4:].norm() / sv.norm())) < 1e-5


def test_routed_lokr_ranks_stay_under_the_fused_column_cap(model, tmp_path, monkeypatch):
    import kilora.fused as fused

    keys, delta = _lokr()
    path = write(tmp_path, "lokr", keys)
    monkeypatch.setattr(
        fused, "max_columns", lambda router_dim, device=None, tiled=False: None if tiled else 3
    )
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "read_share")
    tiled = import_loras([path], model, rank=1)
    assert tiled.sources[0]["svd_ranks"][Q] > 3 and tiled.sources[0]["rank_cap"] is None
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "keys")
    capped = import_loras([path], model, rank=1)
    assert capped.sources[0]["svd_ranks"][Q] == 3 and capped.sources[0]["rank_cap"] == 3
    assert "capped at 3 per expert" in capped.report()
    monkeypatch.setenv("KILORA_LOKR_RANKS", "energy")
    free = import_loras([path], model, rank=1)
    assert free.sources[0]["svd_ranks"][Q] > 3 and free.sources[0]["rank_cap"] is None


def test_max_columns_without_cuda_is_no_cap():
    from kilora.fused import max_columns

    assert max_columns(32, "cpu") is None


def test_columns_for_limit_is_the_largest_size_that_fits():
    from kilora.fused import _shared_need, columns_for_limit

    for limit in (101376, 232448, 49152):
        cols = columns_for_limit(limit, 32)
        assert _shared_need(cols, 32) <= limit < _shared_need(cols * 2, 32)
    assert columns_for_limit(101376, 32) == 256
    assert columns_for_limit(1000, 32) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_max_columns_is_the_largest_size_that_fits():
    from kilora.fused import fits, max_columns

    cols = max_columns(32)
    probe = torch.empty(0, device="cuda")
    assert cols and fits(probe, cols, 32) and not fits(probe, cols * 2, 32)


def test_routed_lokr_part_svds_are_cached_on_disk(model, tmp_path, monkeypatch):
    import kilora.key_cache as key_cache

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    keys, _ = _lokr()
    path = write(tmp_path, "lokr", keys)
    first = import_loras([path], model, rank=1)
    calls = []
    real = torch.linalg.svd
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: calls.append(1) or real(*a, **k))
    second = import_loras([path], model, rank=1)
    assert not calls
    a, b = first.adapter_set.experts[Q], second.adapter_set.experts[Q]
    assert torch.equal(a.down, b.down) and torch.equal(a.up, b.up)


def test_kron_spectrum_with_a_negative_strength():
    from kilora.multi_import import _kron_spectrum

    gen = torch.Generator().manual_seed(4)
    w1, w2 = torch.randn(2, 3, generator=gen), torch.randn(4, 2, generator=gen)
    down, up, error = _kron_spectrum({"lokr_w1": w1, "lokr_w2": w2}, -0.7).cut(6)
    torch.testing.assert_close(up @ down, -0.7 * torch.kron(w1, w2), atol=1e-5, rtol=1e-5)
    assert error < 1e-5


def test_content_keys_are_the_top_right_singular_direction():
    from kilora.multi_import import _content_keys

    gen = torch.Generator().manual_seed(6)
    up = torch.randn(1, 1, 5, 12, generator=gen)
    router = torch.randn(4, 12, generator=gen)
    key = _content_keys(up, router, [{"sha256": "x"}], "t")[0, 0]
    _, _, vh = torch.linalg.svd((up[0, 0] @ router.t()).double(), full_matrices=False)
    want = vh[0].float()
    want = want * (1.0 if float(want[want.abs().argmax()]) >= 0 else -1.0)
    torch.testing.assert_close(key, want / want.norm(), atol=1e-5, rtol=1e-5)


def test_joint_global_shares_across_calls_and_reset(tmp_path):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui.patching import JOINT_SET, _key_mask_by_step
    from kilora.fixed_change import fixed_copy
    from kilora.sections import layer_changes

    tree = ToyAnima()
    tree.blocks[0].adaln_modulation_cross_attn = nn.Linear(DIM, DIM, bias=False)
    tree.llm_adapter = nn.Module()
    tree.llm_adapter.proj = nn.Linear(DIM, DIM, bias=False)
    target = "blocks.0.self_attn.q_proj"
    mounts = ["blocks.0.adaln_modulation_cross_attn", "llm_adapter.proj"]
    paths = []
    for i in range(2):
        weights = lora_weights(tree, [target, *mounts], rank=2, seed=i)
        paths.append(write(tmp_path, str(i), to_keys(weights, "kohya")))
    result = import_loras(paths, ToyAnima(), full_model=tree)
    assert {s.meta["mount_kind"] for s in result.sections} == {"adaln", "text-adapter"}
    changes = layer_changes(result.sections, 1.0)
    patches = {}
    full = {}
    x = torch.randn(2, DIM)
    for path in mounts:
        for c in changes[path]:
            c.tag, c.axis = JOINT_SET, "creative"
        base = tree.get_submodule(path)
        patches[path] = fixed_copy(base, changes[path])
        full[path] = [c.add(x, torch.zeros_like(base(x))) for c in changes[path]]

    layer_set = SimpleNamespace(
        key=JOINT_SET, keys_in=torch.zeros(2, 1, 1), down=torch.ones(1), route_freeze=None
    )
    plans = {target: SimpleNamespace(sets=[layer_set])}
    patcher = FakePatcher(FakeBaseModel(tree))
    _key_mask_by_step(patcher, plans=plans, mount_share_tags={JOINT_SET})
    wrapper = patcher.model_options["model_function_wrapper"]
    gates = torch.tensor([[[0.2, 0.8], [0.4, 0.6]], [[0.7, 0.3], [0.9, 0.1]]])

    def apply_model(inp, timestep, **kwargs):
        outputs = {path: patch(x) for path, patch in patches.items()}
        with compute.layer_set_share_observation(target, [layer_set]):
            compute._note_routes(target, [gates], [layer_set])
        return outputs

    def call(step):
        return wrapper(apply_model, {"input": x, "timestep": step, "c": {}})

    first = call(3.0)
    second = call(2.0)
    restarted = call(3.0)
    for path in mounts:
        base = tree.get_submodule(path)(x)
        torch.testing.assert_close(first[path], base + 0.5 * full[path][0] + 0.5 * full[path][1])
        torch.testing.assert_close(restarted[path], first[path], rtol=0, atol=0)
        weights = gates.mean(dim=1)
        want = base + weights[:, :1] * full[path][0] + weights[:, 1:] * full[path][1]
        torch.testing.assert_close(second[path], want)

    one = import_loras(paths[:1], ToyAnima(), full_model=tree)
    for path, cs in layer_changes(one.sections, 1.0).items():
        for c in cs:
            c.tag, c.axis = JOINT_SET, "creative"
        base = tree.get_submodule(path)
        torch.testing.assert_close(fixed_copy(base, cs)(x), cs[0].add(x, base(x)), rtol=0, atol=0)
        for c in changes[path]:
            c.tag = "stack-entry"
        expected = base(x)
        for c in changes[path]:
            expected = c.add(x, expected)
        torch.testing.assert_close(fixed_copy(base, changes[path])(x), expected, rtol=0, atol=0)


def test_joint_report_accounts_for_plain_and_lokr_keys(tmp_path):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui.patching import ModEntry, joint_choice

    tree = ToyAnima()
    tree.blocks[0].adaln_modulation_cross_attn = nn.Linear(DIM, DIM, bias=False)
    tree.llm_adapter = nn.Module()
    tree.llm_adapter.proj = nn.Linear(DIM, DIM, bias=False)
    target = "blocks.0.self_attn.q_proj"
    mounts = ["blocks.0.adaln_modulation_cross_attn", "llm_adapter.proj"]
    plain = to_keys(lora_weights(tree, [target, *mounts], rank=2), "kohya")
    plain["lora_unet_blocks_9_adaln_modulation_cross_attn.lora_down.weight"] = torch.ones(2, DIM)
    plain["lora_unet_blocks_9_adaln_modulation_cross_attn.lora_up.weight"] = torch.ones(DIM, 2)
    lyco = to_keys(lora_weights(tree, [target], rank=2, seed=2), "kohya")
    for path in mounts:
        stem = "lora_unet_" + path.replace(".", "_")
        lyco[stem + ".lokr_w1"] = torch.eye(4)
        lyco[stem + ".lokr_w2"] = torch.eye(4)
    lyco["lora_unet_llm_adapter_proj.dora_scale"] = torch.ones(DIM)
    paths = [write(tmp_path, "plain", plain), write(tmp_path, "lyco", lyco)]
    result = import_loras(paths, ToyAnima(), full_model=tree)
    for record, state in zip(result.sources, [plain, lyco], strict=True):
        coverage = record["coverage"]
        assert set().union(*map(set, coverage.values())) == set(state)
        assert sum(record["coverage_counts"].values()) == len(state)
    global_lokr = set(lyco) - set(to_keys(lora_weights(tree, [target], rank=2, seed=2), "kohya"))
    global_lokr.remove("lora_unet_llm_adapter_proj.dora_scale")
    assert global_lokr == set(result.sources[1]["coverage"]["scaled-global"])
    assert "lora_unet_llm_adapter_proj.dora_scale" in result.sources[1]["coverage"]["skipped"]
    entries = [ModEntry("lora", path, axis="creative") for path in paths]
    choice, _ = joint_choice(
        FakePatcher(FakeBaseModel(tree)), entries, SimpleNamespace(family=None), ToyAnima()
    )
    report = "\n".join(choice.notes)
    assert "previous-step mean share, 1/2 before the first step" in report
    for record in result.sources:
        for key in record["coverage"]["routed"] + record["coverage"]["scaled-global"]:
            assert key not in report
    assert "adaln 2" in report and "text-adapter 2" in report
    assert "skipped (no matching model layer or shape)" in report


def test_share_observer_excludes_text_and_padding(monkeypatch):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.call_values import current_mount_share
    from kilora.comfyui import patching
    from kilora.family_spec import TokenLayout

    layout = TokenLayout("image2d", (1, 1), text="before")
    monkeypatch.setattr(patching, "layout_resolver", lambda *_: lambda _: layout)
    layer_set = SimpleNamespace(
        key=patching.JOINT_SET, keys_in=torch.zeros(2, 1, 1), down=torch.ones(1), route_freeze=None
    )
    patcher = FakePatcher(FakeBaseModel(ToyAnima()))
    patching._key_mask_by_step(
        patcher,
        plans={"layer": SimpleNamespace(sets=[layer_set])},
        mount_share_tags={patching.JOINT_SET},
    )
    wrapper = patcher.model_options["model_function_wrapper"]
    gates = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.25, 0.75],
                [0.25, 0.75],
                [0.25, 0.75],
                [0.25, 0.75],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ]
    )

    def apply_model(inp, timestep, **kwargs):
        share = current_mount_share(patching.JOINT_SET, 0, 2)
        with compute.layer_set_share_observation("layer", [layer_set]):
            compute._note_routes("layer", [gates], [layer_set])
        return share

    args = {
        "input": torch.zeros(1, 1, 2, 2),
        "timestep": 2.0,
        "c": {"c_crossattn": torch.zeros(1, 2, 1)},
    }
    assert wrapper(apply_model, args) == 0.5
    args["timestep"] = 1.0
    torch.testing.assert_close(wrapper(apply_model, args), torch.tensor([0.25]))


def test_joint_report_limits_skips_per_file():
    from kilora.comfyui.patching import _joint_coverage_notes

    sources = []
    for i in range(2):
        skipped = [f"file{i}_key{j:02}" for j in range(25)]
        sources.append(
            {
                "name": f"file{i}",
                "coverage_counts": {"routed": 100, "skipped": len(skipped)},
                "global_mount_counts": dict.fromkeys(
                    ("adaln", "text-adapter", "embedder", "final", "other"), 2
                ),
                "coverage": {
                    "routed": ["routed_key"],
                    "scaled-global": ["global_key"],
                    "skipped": skipped,
                },
                "skipped_reasons": dict.fromkeys(skipped, "unmatched"),
            }
        )
    notes = _joint_coverage_notes(sources)
    for src in sources:
        own = [line for line in notes if line.startswith(src["name"] + ":")]
        assert len(own) == 22
        assert "and 5 more" in own[-1]
        assert len(src["coverage"]["skipped"]) == 25
    assert "routed_key" not in "\n".join(notes)
    assert "global_key" not in "\n".join(notes)


def test_share_observer_stops_after_representative_first_layer(monkeypatch):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher
    from test_fused import _set

    from kilora import fused
    from kilora.call_values import current_mount_share
    from kilora.comfyui import patching

    layer_set = SimpleNamespace(
        key=patching.JOINT_SET, keys_in=torch.zeros(2, 1, 1), down=torch.ones(1), route_freeze=None
    )
    plans = {f"layer{i}": SimpleNamespace(sets=[layer_set]) for i in range(8)}
    patcher = FakePatcher(FakeBaseModel(ToyAnima()))
    patching._key_mask_by_step(patcher, plans=plans, mount_share_tags={patching.JOINT_SET})
    wrapper = patcher.model_options["model_function_wrapper"]
    gates = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]]])
    observed = []
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "device_ok", lambda _: True)
    monkeypatch.setattr(fused, "fits", lambda *a, **kw: True)
    monkeypatch.setattr(fused, "enabled", lambda: True)
    fused_set = _set(10)
    y0 = torch.zeros(1, 4, 10)

    def apply_model(inp, timestep, **kwargs):
        share = current_mount_share(patching.JOINT_SET, 0, 2)
        for target in plans:
            with compute.layer_set_share_observation(target, [layer_set]):
                observed.append(bool(compute._SET_SHARES))
                if not compute._SET_SHARES:
                    assert fused._fused_ok(y0, [fused_set])
                compute._note_routes(target, [gates], [layer_set])
        return share

    args = {"input": torch.zeros(1, 1, 2, 2), "timestep": 2.0, "c": {}}
    assert wrapper(apply_model, args) == 0.5
    args["timestep"] = 1.0
    torch.testing.assert_close(
        wrapper(apply_model, args), gates.mean(dim=1)[:, 0], atol=1e-6, rtol=0
    )
    assert sum(observed[:8]) <= 4 and sum(observed[8:]) <= 4
    assert not compute._SET_SHARES


def test_share_sampling_handles_split_rows_and_separate_sets():
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.call_values import current_mount_share, route_rows
    from kilora.comfyui import patching

    def layer_set(key):
        return SimpleNamespace(
            key=key, keys_in=torch.zeros(2, 1, 1), down=torch.ones(1), route_freeze=None
        )

    a, b = layer_set("a"), layer_set("b")
    plans = {"a": SimpleNamespace(sets=[a]), "b": SimpleNamespace(sets=[b])}
    patcher = FakePatcher(FakeBaseModel(ToyAnima()))
    patching._key_mask_by_step(patcher, plans=plans, mount_share_tags={"a", "b"})
    wrapper = patcher.model_options["model_function_wrapper"]
    enabled = []

    def sample(target, layer, gates):
        with compute.layer_set_share_observation(target, [layer]):
            enabled.append(bool(compute._SET_SHARES))
            for row, gate in enumerate(gates):
                with route_rows(torch.tensor([row])):
                    compute._note_routes(target, [gate], [layer])
                    compute._note_routes(target, [gate], [layer])

    def apply_model(inp, timestep, **kwargs):
        shares = (current_mount_share("a", 0, 2), current_mount_share("b", 0, 2))
        sample("layer-a", a, [torch.tensor([[[0.2, 0.8]]]), torch.tensor([[[0.7, 0.3]]])])
        sample("later-a", a, [torch.tensor([[[0.9, 0.1]]])] * 2)
        sample("layer-b", b, [torch.tensor([[[0.4, 0.6]]]), torch.tensor([[[0.6, 0.4]]])])
        return shares

    args = {"input": torch.zeros(2, 1, 2, 2), "timestep": 2.0, "c": {}}
    assert wrapper(apply_model, args) == (0.5, 0.5)
    args["timestep"] = 1.0
    a_share, b_share = wrapper(apply_model, args)
    torch.testing.assert_close(a_share, torch.tensor([0.2, 0.7]))
    torch.testing.assert_close(b_share, torch.tensor([0.4, 0.6]))
    assert enabled == [True, False, True] * 2


def test_share_sampling_caps_unsuccessful_layers():
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    layer_set = SimpleNamespace(
        key="a", keys_in=torch.zeros(2, 1, 1), down=torch.ones(1), route_freeze=None
    )
    patcher = FakePatcher(FakeBaseModel(ToyAnima()))
    patching._key_mask_by_step(
        patcher, plans={"layer": SimpleNamespace(sets=[layer_set])}, mount_share_tags={"a"}
    )
    wrapper = patcher.model_options["model_function_wrapper"]
    observed = []

    def apply_model(inp, timestep, **kwargs):
        for i in range(8):
            with compute.layer_set_share_observation(str(i), [layer_set]):
                observed.append(bool(compute._SET_SHARES))
        return inp

    wrapper(apply_model, {"input": torch.zeros(1, 1, 2, 2), "timestep": 1.0, "c": {}})
    assert observed == [True] * 4 + [False] * 4


def test_single_axis_coverage_alongside_routed_set(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    tree = ToyAnima()
    tree.blocks[0].adaln_modulation_cross_attn = nn.Linear(DIM, DIM, bias=False)
    tree.llm_adapter = nn.Module()
    tree.llm_adapter.proj = nn.Linear(DIM, DIM, bias=False)
    target = "blocks.0.self_attn.q_proj"
    mounts = ["blocks.0.adaln_modulation_cross_attn", "llm_adapter.proj"]
    states = [
        to_keys(lora_weights(tree, [target, *mounts], rank=2, seed=i), "kohya") for i in range(3)
    ]
    states[2]["lora_unet_blocks_9_adaln_modulation_cross_attn.lora_down.weight"] = torch.ones(
        2, DIM
    )
    states[2]["lora_unet_blocks_9_adaln_modulation_cross_attn.lora_up.weight"] = torch.ones(DIM, 2)
    paths = [write(tmp_path, f"axis{i}", state) for i, state in enumerate(states)]
    patcher = FakePatcher(FakeBaseModel(tree))
    binding = SimpleNamespace(family=None)
    monkeypatch.setattr(patching, "diffusers_shapes", lambda *args: None)
    monkeypatch.setattr(patching, "shape_model", lambda *args: ToyAnima())
    entries = [patching.ModEntry("lora", path, axis="creative") for path in paths[:2]]
    entries.append(patching.ModEntry("lora", paths[2], axis="ip"))
    choices = patching.choices_for(patcher, entries, binding, joint=True)
    routed, _ = patching.joint_choice(patcher, entries[:2], binding, ToyAnima())
    assert choices[0].notes == routed.notes
    notes = choices[1].notes
    assert notes[0] == (
        "axis2: 2 routed / scaled-global (adaln 2, text-adapter 2, embedder 0, final 0, other 0)"
        " / 2 skipped; single."
    )
    assert len(states[2]) == 2 + 2 + 2 + 2
    assert len(notes) == 3
    assert all("skipped (no matching model layer or shape)" in line for line in notes[1:])


def test_single_lycoris_coverage_counts_exact_sections(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import patching

    tree = ToyAnima()
    target = "blocks.0.self_attn.q_proj"
    state = to_keys(lora_weights(tree, ["blocks.1.mlp.layer1"], rank=2), "kohya")
    stem = "lora_unet_" + target.replace(".", "_")
    state[stem + ".lokr_w1"] = torch.eye(4)
    state[stem + ".lokr_w2"] = torch.eye(4)
    state[stem + ".dora_scale"] = torch.ones(DIM)
    path = write(tmp_path, "single_lyco", state)
    monkeypatch.setattr(patching, "diffusers_shapes", lambda *args: None)
    monkeypatch.setattr(patching, "shape_model", lambda *args: ToyAnima())
    (choice,) = patching.choices_for(
        FakePatcher(FakeBaseModel(tree)),
        [patching.ModEntry("lora", path, axis="ip")],
        SimpleNamespace(family=None),
        joint=True,
    )
    assert choice.notes[0] == (
        "single_lyco: 2 routed / scaled-global (adaln 0, text-adapter 0, embedder 0, "
        "final 0, other 2) / 1 skipped; single."
    )
    assert len(state) == 5
    assert stem + ".dora_scale: skipped (unsupported or excluded weight)" in choice.notes[1]


def test_single_report_limits_skips(tmp_path):
    from kilora.comfyui.patching import _joint_coverage_notes
    from kilora.lora_import import import_lora

    tree = ToyAnima()
    state = to_keys(lora_weights(tree, ["blocks.0.self_attn.q_proj"], rank=2), "kohya")
    state.update({f"lora_te_invalid_{i:02}": torch.ones(1) for i in range(25)})
    result = import_lora(state, tree)
    notes = _joint_coverage_notes([result.coverage], single=True)
    assert "2 routed" in notes[0] and "25 skipped; single." in notes[0]
    assert len(notes) == 22
    assert notes[-1].endswith("and 5 more skipped keys")
