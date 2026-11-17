# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import os

import pytest
import torch
from test_comfyui import denoiser, folders
from test_lowrank_depth import _sets
from test_output_gate import _gated

from kilora import adaptive_gate
from kilora import compute as C
from kilora.adaptive_gate import _token_gram, adaptive_keep, resolve_mode, token_cross
from kilora.compute import LayerSet, dropout_key, run_layer, split_gate, step_carry


@pytest.fixture(autouse=True)
def _keys_arm(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")


def _y0(seed=1, width=16, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 9, width, generator=g) + offset


def _run(y0, sets, depth, fast, monkeypatch, regions=None):
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, target="layer", regions=regions)


def _mode(sets, mode, **kw):
    return [dataclasses.replace(s, gate_mode=mode, **kw) for s in sets]


@pytest.mark.parametrize(
    "shape",
    [(2, 4096, 32), (2, 3, 512, 32), (2, 77, 32), (2, 4096, 8)]
    + [(2, tokens, 8) for tokens in (96, 48, 24, 12, 6)],
)
@pytest.mark.parametrize("cross", [False, True])
def test_token_gram_matches_direct_product(shape, cross):
    g = torch.Generator().manual_seed(17)
    a = torch.rand(shape, generator=g)
    b = torch.rand((*shape[:-1], shape[-1] + 3), generator=g) if cross else a
    actual = token_cross(a, b) if cross else _token_gram(a)
    torch.testing.assert_close(actual, a.transpose(-1, -2) @ b, rtol=1e-5, atol=0)


@pytest.mark.parametrize("tokens", [512, 77])
@pytest.mark.parametrize("cross", [False, True])
def test_token_products_compile(tokens, cross):
    g = torch.Generator().manual_seed(19)
    a = torch.rand(2, tokens, 8, generator=g)
    b = torch.rand(2, tokens, 8, generator=g)
    product = token_cross if cross else _token_gram
    compiled = torch.compile(product, backend="eager", dynamic=False, fullgraph=True)
    args = (a, b) if cross else (a,)
    torch.testing.assert_close(compiled(*args), product(*args), rtol=1e-5, atol=0)


def test_token_gram_noncontiguous_signed_input():
    v = torch.randn(2, 32, 512, generator=torch.Generator().manual_seed(18)).transpose(-1, -2)
    expected = v.transpose(-1, -2) @ v
    torch.testing.assert_close(_token_gram(v), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_infinite_kappa_without_direction_is_no_gate_bit_for_bit(carry, monkeypatch):
    plain = _mode(_sets(2, 16, 4, 2, 8, seed=2, scales=[1.0, -0.7]), "none", carry=carry)
    open_gate = _mode(plain, "adaptive", gate_kappa=math.inf, gate_coherence=False)
    for depth in (1, 3, 2.5):
        with dropout_key(1, 0.5):
            loop = _run(_y0(), open_gate, depth, False, monkeypatch)
            assert torch.equal(loop, _run(_y0(), plain, depth, False, monkeypatch))
            fast = _run(_y0(), open_gate[:1], depth, True, monkeypatch)
            want = _run(_y0(), plain[:1], depth, True, monkeypatch)
        if depth == 1:
            assert torch.equal(fast, want)
        else:
            torch.testing.assert_close(fast, want, rtol=1e-6, atol=1e-6)


def _toy(gains, width=8, kappa=1.0, mode="adaptive", carry="drop"):
    cols = len(gains)
    down = torch.zeros(cols, width)
    up = torch.zeros(cols, width)
    for c, gain in enumerate(gains):
        down[c, c] = gain
        up[c, c] = 1.0
    return LayerSet(
        router=torch.eye(4, width),
        keys_in=torch.eye(4)[None, :2],
        per_expert=cols,
        down=down,
        up=up,
        carry=carry,
        gate_mode=mode,
        gate_kappa=kappa,
    )


@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
def test_growing_channel_is_damped_and_stays_bounded(kappa, carry, monkeypatch):

    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    y0 = _y0(width=8, offset=1.0)
    rms = y0.pow(2).mean(-2).sqrt()[..., 0]
    free = _run(y0, [_toy([0.6], kappa=kappa, mode="none", carry=carry)], 7, False, monkeypatch)
    free = (free - y0).mean(-2)[..., 0].abs()
    first = (
        (_run(y0, [_toy([0.6], mode="none", carry=carry)], 1, False, monkeypatch) - y0)
        .mean(-2)[..., 0]
        .abs()
    )
    assert bool((free <= C.IMPORT_DEPTH_BUDGET * first + 1e-6).all())
    assert bool((free > first).all())
    gate = _toy([0.6], kappa=kappa, carry=carry)
    for depth in range(1, 8):
        accs = []
        for seed in range(40):
            with dropout_key(seed, 0.5):
                y = _run(y0, [gate], depth, False, monkeypatch)
            accs.append((y - y0).mean(-2)[..., 0].abs())
        mean = torch.stack(accs).mean(0)
        assert bool((mean <= 2 * kappa * rms).all()), (depth, mean, rms)
    assert bool((mean < free).all())


def test_expected_channel_change_is_bounded(monkeypatch):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    g = torch.Generator().manual_seed(3)
    up = torch.eye(4)
    y0 = torch.randn(1, 10, 4, generator=g)
    y0_norm, y0_rms = y0.norm(dim=-1), y0.pow(2).mean(-2).sqrt()
    for kappa in (0.5, 1.0, 2.0):
        for size in (0.1, 1.0, 3.0, 30.0):
            z = torch.randn(1, 10, 4, generator=g) + size
            _, qb, _ = adaptive_keep(z, None, None, up, y0_norm, y0_rms, kappa, False)
            acc = z.pow(2).mean(-2).sqrt()
            peak = kappa * y0_rms / (math.e * math.log(2))
            assert bool((qb[:, 0] * acc <= peak * (1 + 1e-5)).all())


def test_oscillating_channel_is_damped_more_than_a_consistent_one():
    g = torch.Generator().manual_seed(0)
    up = torch.eye(2)
    y0 = torch.randn(1, 12, 2, generator=g) + 3.0
    y0_norm = y0.norm(dim=-1)
    y0_rms = y0.pow(2).mean(-2).sqrt()
    step = torch.rand(1, 12, 1, generator=g) * 0.2 + 0.1
    z = torch.zeros(1, 12, 2)
    hist = None
    keeps = []
    for k in range(6):
        dz = torch.cat([step, step * (-1) ** k], -1)
        qa, qb, hist = adaptive_keep(z + dz, z, hist, up, y0_norm, y0_rms)
        z = z + dz
        keeps.append(qb[0, 0])
    later = torch.stack(keeps[1:])
    assert bool((later[:, 1] < 0.5 * later[:, 0]).all()), later


def test_oscillating_channel_in_the_loop(monkeypatch):
    y0 = _y0(width=8, offset=1.0)
    with dropout_key(0, 0.5), C.recording_gates() as stats:
        _run(y0, [_toy([0.3, -1.6], kappa=4.0)], 6, False, monkeypatch)
    qbs = [q.reshape(2, 8) for q in stats[1::2]][1:]
    mean = torch.stack(qbs).mean(0)
    assert bool((mean[:, 1] < mean[:, 0]).all()), mean


@pytest.mark.parametrize("mode", ["adaptive", "both"])
@pytest.mark.parametrize("carry", ["dropped", "drop"])
@pytest.mark.parametrize("region", [False, True])
def test_fast_path_equals_the_loop(mode, carry, region, monkeypatch):
    base = _sets(2, 16, 4, 2, 8, seed=8, scales=[1.0, -0.7])
    sets = _mode(_gated("separable", base) if mode == "both" else base, mode, carry=carry)
    assert C._fast_path_ok(_y0(), sets, 0.0, None, None)
    regions = None
    if region:
        r = (torch.arange(9) < 5).float()[None].expand(2, 9)
        regions = [r, 1 - r * 0.5]
    for depth in (1, 3, 5, 2.5):
        with dropout_key(1, 0.4):
            fast = _run(_y0(), sets, depth, True, monkeypatch, regions)
            loop = _run(_y0(), sets, depth, False, monkeypatch, regions)
        torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("carry", ["dropped", "drop"])
def test_fast_path_equals_the_loop_across_sampler_steps(carry, monkeypatch):
    sets = _mode(_sets(2, 16, 4, 2, 8, seed=5, scales=[1.0, 0.8]), "adaptive", carry=carry)
    outs = {}
    for fast in (False, True):
        monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
        cache: dict = {}
        outs[fast] = []
        for i in range(5):
            y0 = _y0() * (1 + 0.05 * i)
            with dropout_key(0, 5 - i), torch.no_grad(), step_carry(cache):
                outs[fast].append(run_layer(y0, y0, sets, 3, target="layer"))
            assert cache["layer"]["gates"] is not None
    for a, b in zip(outs[False], outs[True], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fast", [False, True])
def test_replay_is_bit_identical_and_keyed(fast, monkeypatch):
    sets = _mode(_sets(2, 16, 4, 2, 8, seed=3), "adaptive")
    with dropout_key(2, 0.3):
        a = _run(_y0(), sets, 4, fast, monkeypatch)
        b = _run(_y0(), sets, 4, fast, monkeypatch)
    with dropout_key(2, 0.7):
        c = _run(_y0(), sets, 4, fast, monkeypatch)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_both_is_the_product_of_the_two_gates():
    s = _gated("separable", _sets(1, 16, 4, 2, 8, seed=6))[0]
    g = torch.Generator().manual_seed(1)
    y0 = _y0()
    y0_norm, y0_rms = C._y0_gate_stats(y0)
    weights = torch.softmax(torch.randn(2, 9, 4, generator=g), -1)
    zs = [torch.randn(2, 9, 8, generator=g) for _ in range(3)]
    hists = {m: {} for m in ("learned", "adaptive", "both")}
    with dropout_key(0, 0.2):
        for k, z in enumerate(zs, 1):
            gates = {}
            for m, hist in hists.items():
                sm = dataclasses.replace(s, gate_mode=m)
                args = (y0_norm, y0_rms, weights, "layer", 0, k, None)
                gates[m] = split_gate(sm, z, hist, *args)
            for part in (0, 1):
                product = gates["learned"][part] * gates["adaptive"][part]
                assert torch.equal(gates["both"][part], product)


def test_sampled_gate_masks_match_separate_comparisons(monkeypatch):
    s = _mode(_sets(1, 16, 4, 2, 8, seed=9), "adaptive")[0]
    y0 = _y0()
    z = torch.randn(2, 9, 8, generator=torch.Generator().manual_seed(10))
    stats = C._y0_gate_stats(y0)
    weights = torch.full((2, 9, 4), 0.25)
    qa = torch.rand(2, 9, 1, generator=torch.Generator().manual_seed(11))
    qb = torch.rand(2, 1, 16, generator=torch.Generator().manual_seed(12))
    u = torch.rand(qa.numel() + qb.numel(), generator=torch.Generator().manual_seed(13))
    monkeypatch.setattr(C, "adaptive_keep", lambda *args, **kwargs: (qa, qb, z))
    got = split_gate(s, z, {}, *stats, weights, "layer", 0, 1, None, uniforms=lambda *args: u)
    assert torch.equal(got[0], (u[: qa.numel()].reshape(qa.shape) < qa).float())
    assert torch.equal(got[1], (u[qa.numel() :].reshape(qb.shape) < qb).float())


def test_auto_mode_follows_the_trained_weights():
    assert resolve_mode("auto", True) == "both"
    assert resolve_mode("auto", False) == "adaptive"
    assert resolve_mode("learned", False) == "learned"
    with pytest.raises(ValueError):
        resolve_mode("ctm", False)
    s = _sets(1, 16, 4, 2, 8, seed=7)[0]
    assert C.uses_adaptive(dataclasses.replace(s, gate_mode="auto"))
    assert C.learned_kind(dataclasses.replace(s, gate_mode="auto")) == "none"
    t = _gated("separable", [s])[0]
    assert C.learned_kind(dataclasses.replace(t, gate_mode="auto")) == "separable"
    assert C.learned_kind(dataclasses.replace(t, gate_mode="adaptive")) == "none"


def test_mod_file_keeps_the_gate_mode_and_kappa(model, tmp_path):
    from kilora import KiLoRAConfig, attach, load_mod
    from kilora.modfile import save_mod

    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=2,
            routes=1,
            router_dim=8,
            depth=2,
            train_depth=None,
            output_gate="none",
        ),
    )
    assert adapter.settings.gate_mode == "learned"
    adapter.settings.gate_mode, adapter.settings.gate_kappa = "adaptive", 0.5
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    path = tmp_path / "g.kmod.safetensors"
    save_mod(path, sets, settings, id="g")
    mod = load_mod(path)
    assert (mod.settings.gate_mode, mod.settings.gate_kappa) == ("adaptive", 0.5)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
def test_imported_set_at_depth_7_stays_bounded(kappa, fast, monkeypatch):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    y0 = _y0(width=8, offset=1.0)
    rms = y0.pow(2).mean(-2).sqrt()[..., 0]
    s = dataclasses.replace(_toy([1.0], carry="dropped"), form="input", gate_kappa=kappa)
    monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)

    def change(set_, depth, seed):
        with dropout_key(seed, 0.5), torch.no_grad():
            y = run_layer(y0.clone(), y0.clone(), [set_], depth, target="layer")
        return (y - y0).mean(-2)[..., 0].abs()

    plain = dataclasses.replace(s, gate_mode="none")
    free = change(plain, 7, 0)
    first = change(plain, 1, 0)
    assert bool((free <= 2 * first + 1e-6).all())
    for depth in range(1, 8):
        mean = torch.stack([change(s, depth, seed) for seed in range(40)]).mean(0)
        assert bool((mean <= 2 * kappa * rms).all()), (depth, mean, rms)


def test_apply_node_gate_choice_reaches_the_imported_set(denoiser, folders, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    seen = []
    real = patching.plan_targets

    def spy(scoped, *a, **k):
        seen.append({(s.gate_mode, s.gate_kappa) for s in scoped})
        return real(scoped, *a, **k)

    monkeypatch.setattr(patching, "plan_targets", spy)
    gen = torch.Generator().manual_seed(3)
    sd = {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": torch.randn(2, 16, generator=gen),
        "lora_unet_layers_0_attention_to_q.lora_up.weight": torch.randn(16, 2, generator=gen),
        "lora_unet_layers_0_attention_to_q.alpha": torch.tensor(1.0),
    }
    save_file(sd, str(folders["loras"] / "style.safetensors"))
    (chain,) = N.KiLoRAImportLoRA().add("style.safetensors", 1.0)
    for gate, kappa in (("mod", 0.0), ("none", 0.0), ("adaptive", 0.5)):
        N.KiLoRAApply().apply(
            FakePatcher(FakeBaseModel(denoiser)), chain, -1, output_gate=gate, gate_kappa=kappa
        )
    assert seen == [{("auto", 1.0)}, {("none", 1.0)}, {("adaptive", 0.5)}]


def test_zero_mean_texture_change_is_limited(monkeypatch):
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", "1")
    g = torch.Generator().manual_seed(4)
    y0 = torch.randn(1, 16, 2, generator=g)
    y0_norm, y0_rms = y0.norm(dim=-1), y0.pow(2).mean(-2).sqrt()
    signs = torch.tensor([1.0, -1.0]).repeat(8)[None, :, None]
    z = torch.cat([signs * 5.0, torch.full((1, 16, 1), 0.01)], -1)
    _, qb, _ = adaptive_keep(z, None, None, torch.eye(2), y0_norm, y0_rms)
    assert float(z[..., 0].mean()) == 0.0
    assert float(qb[0, 0, 0]) < 0.1 < 0.9 < float(qb[0, 0, 1])


@pytest.mark.parametrize("form", ["state", "input"])
def test_last_step_uses_the_expectation(form, monkeypatch):
    y0 = _y0(width=8, offset=1.0)
    s = dataclasses.replace(_toy([0.6, -0.4]), form=form)
    outs = []
    for seed in (0, 1, 2):
        with dropout_key(seed, 0.5):
            outs.append(_run(y0, [s], 1, False, monkeypatch))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    assert not torch.equal(outs[0], y0)


@pytest.mark.parametrize("fast", [False, True])
def test_head_history_freezes_rates_and_ignores_later_changes(fast, monkeypatch):
    s = dataclasses.replace(_toy([0.3, -0.2]), gate_head_limit=2)
    y0 = _y0(width=8)
    original = C.adaptive_keep
    calls = []

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(tuple(t.clone() for t in result[:2]))
        return result

    monkeypatch.setattr(C, "adaptive_keep", track)
    with dropout_key(5, 0.5), C.recording_gates() as stats:
        first = _run(y0, [s], 5, fast, monkeypatch)
    assert len(calls) == 2 and len(stats) == 10
    for n in range(2, 5):
        assert torch.equal(stats[2 * n], stats[2])
        assert torch.equal(stats[2 * n + 1], stats[3])
    calls.clear()

    def perturb(*args, **kwargs):
        z = args[1]
        if args[8] > 2:
            args = (args[0], z * 11, *args[2:])
        return real_split(*args, **kwargs)

    real_split = C.split_gate
    monkeypatch.setattr(C, "split_gate", perturb)
    with dropout_key(5, 0.5), C.recording_gates() as changed:
        _run(y0, [s], 5, fast, monkeypatch)
    assert len(calls) == 2
    for a, b in zip(stats, changed, strict=True):
        assert torch.equal(a, b)
    assert torch.isfinite(first).all()


def test_head_history_carries_count_and_rates(monkeypatch):
    s = dataclasses.replace(_toy([0.3, -0.2]), gate_head_limit=2)
    y0 = _y0(width=8)
    cache = {}
    with step_carry(cache), dropout_key(5, 0.5), C.recording_gates() as stats:
        _run(y0, [s], 1, False, monkeypatch)
        _run(y0, [s], 4, False, monkeypatch)
    for n in range(2, 5):
        assert torch.equal(stats[2 * n], stats[2])
        assert torch.equal(stats[2 * n + 1], stats[3])
    assert cache["layer"]["gates"][0]["head_count"] == 2


def test_head_history_fast_matches_loop(monkeypatch):
    s = dataclasses.replace(_toy([0.3, -0.2]), gate_head_limit=2)
    with dropout_key(5, 0.5):
        loop = _run(_y0(width=8), [s], 5, False, monkeypatch)
        fast = _run(_y0(width=8), [s], 5, True, monkeypatch)
    torch.testing.assert_close(fast, loop, rtol=1e-5, atol=1e-6)


def test_gate_history_setting_and_fixed_after():
    from kilora.adaptive_gate import parse_gate_history

    assert parse_gate_history("full") is None
    assert parse_gate_history("head:2") == 2
    for bad in ("head:0", "head:x", "tail:2"):
        with pytest.raises(ValueError, match="full or head:k"):
            parse_gate_history(bad)
    s = _toy([0.2])
    assert C.gate_fixed_after(dataclasses.replace(s, gate_head_limit=2)) == 2
    assert C.gate_fixed_after(s) is None
    assert C.gate_fixed_after(dataclasses.replace(s, gate_head_limit=2, gate_mode="none")) is None
    learned = dataclasses.replace(s, gate_head_limit=2, gate_mode="learned")
    assert C.gate_fixed_after(learned) is None


def test_explicit_full_matches_default(monkeypatch):
    from kilora.adaptive_gate import parse_gate_history

    s = _toy([0.3, -0.2])
    with dropout_key(7, 0.5), C.recording_gates() as baseline:
        a = _run(_y0(width=8), [s], 5, False, monkeypatch)
    with dropout_key(7, 0.5), C.recording_gates() as explicit:
        explicit_set = dataclasses.replace(s, gate_head_limit=parse_gate_history("full"))
        b = _run(_y0(width=8), [explicit_set], 5, False, monkeypatch)
    assert torch.equal(a, b)
    assert all(torch.equal(x, y) for x, y in zip(baseline, explicit, strict=True))


def test_head_history_path_single_walk(monkeypatch):
    s = dataclasses.replace(_toy([0.3, -0.2]), gate_head_limit=2, iteration="path")
    calls = []
    original = C.adaptive_keep

    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(C, "adaptive_keep", track)
    with dropout_key(5, 0.5):
        y = _run(_y0(width=8), [s], 5, False, monkeypatch)
    assert len(calls) == 2
    assert torch.isfinite(y).all()
    with pytest.raises(ValueError, match="multiple path walks"):
        _run(_y0(width=8), [dataclasses.replace(s, path_width=2)], 5, False, monkeypatch)


def test_head_history_input_validation_on_apply(denoiser, folders):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    node = N.KiLoRAApply()
    model = FakePatcher(FakeBaseModel(denoiser))
    for bad in ("head:0", "head:x", "tail:2"):
        with pytest.raises(ValueError, match="full or head:k"):
            node.apply(model, patching.ModChain(), 1, gate_history=bad)
    patched, _, _ = node.apply(model, patching.ModChain(), 1, gate_history="head:2")
    assert patching.applied(patched)["gate_history"] == "head:2"


def test_apply_head_history_checks_resolved_gate_and_width(denoiser, folders, monkeypatch):
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora import KiLoRAConfig, KiLoRAError, attach
    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    adapter = attach(
        denoiser,
        KiLoRAConfig(
            target_modules=["layers.0.attention.to_out.0"],
            num_experts=4,
            routes=2,
            router_dim=8,
        ),
    )
    adapter.settings.gate_mode = "adaptive"
    adapter.settings.path_width = 2
    adapter.settings.iteration = "path"
    adapter.save(
        folders["kilora"] / "gate-1.0.0.kmod",
        id="gate",
        version="1.0.0",
        base_model={"family": "kilora-tiny-1"},
    )
    adapter.remove()
    (chain,) = N.KiLoRASelect().select("gate-1.0.0.kmod", 1.0, 1.0)
    base = FakePatcher(FakeBaseModel(denoiser))
    node = N.KiLoRAApply()
    message = (
        "gate_history head:k works with path width 1 only. "
        "Set path_width to 1, or gate_history to full."
    )
    with monkeypatch.context() as m:
        m.setattr(patching, "clear", lambda *_: pytest.fail("model cloned before validation"))
        for width in (0, 2):
            with pytest.raises(KiLoRAError) as exc:
                node.apply(base, chain, 3, gate_history="head:2", path_width=width)
            assert str(exc.value) == message

    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(9))
    for history, width, gate in (("head:2", 1, "mod"), ("full", 2, "mod"), ("head:2", 2, "none")):
        patched, _, _ = node.apply(
            base, chain, 3, gate_history=history, path_width=width, output_gate=gate
        )
        assert torch.isfinite(patched.run(x)).all()


_KERNELS = pytest.mark.skipif(
    adaptive_gate.triton is None
    or (not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"),
    reason="needs CUDA or the Triton interpreter",
)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_KERNEL_CHOICE = C.fused_gate_enabled


def _torch_gate(monkeypatch, on):
    choice = _KERNEL_CHOICE if on else (lambda z: False)
    monkeypatch.setattr(adaptive_gate, "fused_gate_enabled", choice)
    monkeypatch.setattr(C, "fused_gate_enabled", choice)


def _gate_inputs(seed, batch=2, tokens=150, cols=40, out=72, history=True, active=False):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, tokens, cols, generator=g)
    before = torch.randn(batch, tokens, cols, generator=g)
    hist = torch.randn(batch, tokens, cols, generator=g) if history else None
    up = torch.randn(cols, out, generator=g) / cols**0.5
    norm = torch.rand(batch, tokens, generator=g) * 4 + 0.1
    rms = torch.rand(batch, out, generator=g) + 0.05
    mask = (torch.rand(batch, tokens, 1, generator=g) > 0.3) if active else None
    if mask is not None:
        z, before = z * mask, before * mask
    return [None if t is None else t.to(_DEVICE) for t in (z, before, hist, up, norm, rms, mask)]


@_KERNELS
@pytest.mark.parametrize("shape", [(2, 150, 40, 72), (1, 9, 8, 24), (3, 70, 96, 130)])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("first", ["none", "given", "made"])
@pytest.mark.parametrize("coherence", [True, False])
def test_gate_kernels_match_the_torch_path(shape, history, active, first, coherence, monkeypatch):
    monkeypatch.delenv("KILORA_GATE_BASE_REFERENCE", raising=False)
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    batch, tokens, cols, out = shape
    z, before, hist, up, norm, rms, mask = _gate_inputs(
        7, batch, tokens, cols, out, history=history, active=active
    )
    gram = up @ up.t()
    first_norm, first_rms = adaptive_gate.change_stats(z - before, up, gram, mask)
    ref = {} if first == "none" else dict(first_norm=first_norm, first_rms=first_rms)
    _torch_gate(monkeypatch, False)
    expected = adaptive_keep(
        z,
        before,
        hist,
        up,
        norm,
        rms,
        coherence=coherence,
        active=mask,
        gram=gram,
        first_step=False,
        **ref,
    )
    _torch_gate(monkeypatch, True)
    with dropout_key(5, 2):
        base = C.dropout_base("blk#0#adaptive", 3)
        got = adaptive_gate.gate_call(
            z,
            before,
            hist,
            up,
            gram,
            norm,
            rms,
            1.0,
            coherence,
            mask,
            **({} if first != "given" else ref),
            first_change=first == "made",
            base=base,
        )
        u = C.dropout_uniforms((got[0].numel() + got[1].numel(),), "blk#0#adaptive", 3, z.device)
    qa, qb, new, made_norm, made_rms, mask_a, mask_b = got
    for e, r in zip(expected, (qa, qb, new), strict=True):
        torch.testing.assert_close(r, e, rtol=1e-5, atol=1e-5)
    if first == "made":
        torch.testing.assert_close(made_norm, first_norm, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(made_rms, first_rms, rtol=1e-5, atol=1e-6)
    else:
        assert made_norm is None and made_rms is None
    n = qa.numel()
    draws = (u < torch.cat((qa.reshape(-1), qb.reshape(-1)))).float()
    assert torch.equal(mask_a.reshape(-1), draws[:n])
    assert torch.equal(mask_b.reshape(-1), draws[n:])


@_KERNELS
@pytest.mark.parametrize("cols", [24, 96])
@pytest.mark.parametrize("reference", ["0", "1"])
@pytest.mark.parametrize("masks", ["sample", "expect"])
@pytest.mark.parametrize("active", [False, True])
def test_split_gate_kernels_match_the_torch_path(cols, reference, masks, active, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("KILORA_GATE_BASE_REFERENCE", reference)
    monkeypatch.setenv("KILORA_MASKS", masks)
    g = torch.Generator().manual_seed(11)
    out, tokens = 40, 70
    s = LayerSet(
        router=torch.randn(8, out, generator=g),
        keys_in=torch.randn(3, 3, 8, generator=g),
        keys_out=torch.randn(3, 2, 8, generator=g),
        per_expert=cols // 3,
        down=torch.randn(cols, 20, generator=g),
        up=(torch.randn(cols, out, generator=g) / cols**0.5),
        form="input",
        gate_mode="adaptive",
    )
    s.up = s.up.to(_DEVICE)
    steps = [torch.randn(2, tokens, cols, generator=g).to(_DEVICE) for _ in range(4)]
    norm = (torch.rand(2, tokens, generator=g) * 3 + 0.1).to(_DEVICE)
    rms = (torch.rand(2, out, generator=g) + 0.05).to(_DEVICE)
    act = (torch.rand(2, tokens, 1, generator=g) > 0.3).to(_DEVICE) if active else None
    last = None
    if active:
        last = (torch.rand(2, tokens, 1, generator=g) > 0.5).to(_DEVICE)

    def run(on):
        _torch_gate(monkeypatch, on)
        hist, parts = {}, []
        with torch.no_grad(), dropout_key(7, 3):
            for k, z in enumerate(steps, start=1):
                a, b = split_gate(
                    s,
                    z,
                    hist,
                    norm,
                    rms,
                    None,
                    "blk",
                    0,
                    k,
                    None,
                    last=k == 4 if last is None else last & (k == 4),
                    active=act,
                )
                parts.append((a, b, hist["m"], *hist["q"]))
        return parts, hist

    expected, old = run(False)
    got, new = run(True)
    for step, (e_step, g_step) in enumerate(zip(expected, got, strict=True)):
        a0, b0, m0, qa0, qb0 = e_step
        a1, b1, m1, qa1, qb1 = g_step
        torch.testing.assert_close(qa1, qa0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(qb1, qb0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(m1, m0, rtol=1e-5, atol=1e-5)
        if masks == "sample" and step < 3:
            assert torch.equal(a1, a0) and torch.equal(b1, b0)
        torch.testing.assert_close(a1, a0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b1, b0, rtol=1e-5, atol=1e-5)
    if reference == "0":
        torch.testing.assert_close(new["first_norm"], old["first_norm"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new["first_rms"], old["first_rms"], rtol=1e-5, atol=1e-6)
