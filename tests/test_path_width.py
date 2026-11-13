# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
import torch
from test_comfyui import denoiser, folders
from test_lowrank_depth import _sets

from kilora import compute as C
from kilora import path_walks as P
from kilora.compute import LayerSet, dropout_key, run_layer


def _y0(seed=1, width=16):
    return torch.randn(2, 9, width, generator=torch.Generator().manual_seed(seed))


def _run(y0, sets, depth, **kw):
    with torch.no_grad():
        return run_layer(y0.clone(), y0.clone(), sets, depth, target="layer", **kw)


def _path(sets, **kw):
    return [dataclasses.replace(s, iteration="path", **kw) for s in sets]


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("n_sets", [1, 2])
def test_width_one_is_the_path_mode_bit_for_bit(n_sets, stop, monkeypatch):
    plain = _path(_sets(n_sets, 16, 4, 2, 8, seed=3), fixed_point_stop=stop)
    explicit = _path(plain, path_width=1, path_select="mix")
    for depth in (1, 3, 5):
        with dropout_key(0, 0.5):
            assert torch.equal(_run(_y0(), plain, depth), _run(_y0(), explicit, depth))


def _cycle_set(tau=40.0, gate_mode="none"):

    e = torch.eye(6)
    keys_in = torch.stack([e[0], e[1], e[2]])[:, None, :]
    keys_out = torch.stack([0.6 * e[0] + 0.8 * e[2], e[1], 0.8 * e[0] + 0.6 * e[2]])[:, None, :]
    width = 6
    down = torch.zeros(3, width)
    down[0, 0], down[1, 1], down[2, 0] = 1.0, 1.0, 1.0
    up = torch.zeros(3, width)
    up[0, 3], up[1, 4], up[2, 5] = 1.0, 1.0, 1.0
    return LayerSet(
        router=torch.eye(6),
        keys_in=keys_in,
        keys_out=keys_out,
        per_expert=1,
        down=down,
        up=up,
        tau=tau,
        iteration="path",
        gate_mode=gate_mode,
    )


def _cycle_y0():
    y0 = torch.zeros(1, 2, 6)
    y0[0, 0, :2] = torch.tensor([0.8, 0.6])
    y0[0, 1, :2] = torch.tensor([0.6, 0.8])
    return y0


def _expert_of(y, y0):
    return (y - y0)[..., 3:6].abs().argmax(-1)


def _derived_cycle_set():
    s = _cycle_set()
    s.down = torch.eye(6)[:3]
    s.up = torch.zeros(3, 6)
    s.up[0, 0], s.up[0, 2], s.up[0, 3] = -2.0, 4.0, 1.0
    s.up[1, 1], s.up[1, 4] = 4.0, 1.0
    s.up[2, 0], s.up[2, 5] = 4.0, 1.0
    return s


def test_two_walks_recover_a_wrong_first_step(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    y0 = _cycle_y0()
    s = dataclasses.replace(_derived_cycle_set(), carry="drop")
    with dropout_key(0, 0.5):
        one = _run(y0, [dataclasses.replace(s, path_width=1)], 8)
        two = _run(y0, [dataclasses.replace(s, path_width=2)], 4)
    assert _expert_of(one, y0).tolist() == [[_expert_of(one, y0)[0, 0].item(), 1]]
    assert _expert_of(one, y0)[0, 0].item() in (0, 2)
    assert _expert_of(two, y0).tolist() == [[1, 1]]
    assert (two - y0)[0, 0, 4] >= y0[0, 0, 1]


def test_walk_zero_is_the_single_walk_and_walks_are_independent(monkeypatch):
    sets = _path(_sets(2, 16, 4, 2, 8, seed=5))
    y0 = _y0()

    def run(width):
        with torch.no_grad(), dropout_key(0, 0.5):
            state, _ = P.walks(
                y0, sets, 4, False, 0.0, None, "layer", [None] * 2, None, None, width
            )
        return state

    w2, w3 = run(2), run(3)
    for a, b in zip(w2, w3[:2], strict=True):
        for za, zb in zip(a["zs"], b["zs"], strict=True):
            assert torch.equal(za, zb)
        assert torch.equal(a["score"], b["score"])
    one = _run(y0, sets, 4)
    pick0 = [dataclasses.replace(s, path_width=3, path_select="score") for s in sets]
    real = P.select_weights

    def first(state, select):
        w = real(state, select)
        return torch.zeros_like(w).index_fill(-1, torch.tensor([0]), 1.0)

    monkeypatch.setattr(P, "select_weights", first)
    assert torch.equal(_run(y0, pick0, 4), one)


def test_mix_with_one_dominant_walk_equals_score():
    state = [
        {"score": torch.tensor([[3.0, 0.0]]), "converged": torch.tensor([[False, False]])},
        {"score": torch.tensor([[1.0, 2.0]]), "converged": torch.tensor([[False, False]])},
    ]
    assert torch.equal(P.select_weights(state, "mix"), P.select_weights(state, "score"))


def test_converged_walk_wins_over_a_higher_score():
    state = [
        {"score": torch.tensor([[5.0]]), "converged": torch.tensor([[False]])},
        {"score": torch.tensor([[1.0]]), "converged": torch.tensor([[True]])},
    ]
    assert P.select_weights(state, "converged").tolist() == [[[0.0, 1.0]]]
    assert P.select_weights(state, "score").tolist() == [[[1.0, 0.0]]]


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        P.check_path_width(0)
    with pytest.raises(ValueError):
        P.check_path_select("best")


@pytest.mark.parametrize("gate_mode", ["none", "adaptive"])
def test_wide_walks_keep_one_up_product_and_the_gate(gate_mode):
    y0 = _cycle_y0()
    s = _cycle_set(gate_mode=gate_mode)
    with dropout_key(0, 0.5):
        y = _run(y0, [dataclasses.replace(s, path_width=3)], 4)
    assert torch.isfinite(y).all()
    assert torch.equal(y[..., :3], y0[..., :3])
    assert C.PATH_SELECTS == ("converged", "score", "mix")


def test_mod_file_keeps_the_path_width(model, tmp_path):
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
        ),
    )
    adapter.settings.path_width, adapter.settings.path_select = 3, "score"
    sets, settings = adapter.to_sets(), adapter.settings
    adapter.remove()
    path = tmp_path / "w.kmod.safetensors"
    save_mod(path, sets, settings, id="w")
    mod = load_mod(path)
    assert (mod.settings.path_width, mod.settings.path_select) == (3, "score")


def test_apply_node_path_choices_reach_the_sets(denoiser, folders, monkeypatch):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import nodes as N
    from kilora.comfyui import patching

    seen = []
    real = patching.plan_targets

    def spy(scoped, *a, **k):
        seen.append({(s.path_width, s.path_select) for s in scoped})
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
    for width, select in ((0, "mod"), (3, "score")):
        N.KiLoRAApply().apply(
            FakePatcher(FakeBaseModel(denoiser)), chain, -1, path_width=width, path_select=select
        )
    assert seen == [{(1, "converged")}, {(3, "score")}]


def test_transition_prior_is_symmetric_only_for_equal_keys():
    s = _derived_cycle_set()
    prior = P.transition_prior(s)
    assert prior.shape == (3, 3)
    kin, kout = C._set_keys(s)
    expected = torch.einsum("jar,ibr->ijab", kin, kout).square().sum((-1, -2))
    expected /= (kout.norm(dim=-1) > 0).sum(-1).clamp_min(1)[:, None]
    torch.testing.assert_close(prior, expected)
    assert not torch.allclose(prior, prior.t())
    same = dataclasses.replace(s, up=s.down.clone(), keys_out=torch.randn_like(s.keys_out))
    p2 = P.transition_prior(same)
    torch.testing.assert_close(p2, p2.t())


def test_search_options_off_are_bit_identical():
    sets = _path(_sets(2, 16, 4, 2, 8, seed=7), path_width=3)
    off = [dataclasses.replace(s, path_prior_min=0.0, path_starts="top") for s in sets]
    with dropout_key(0, 0.5):
        assert torch.equal(_run(_y0(), sets, 5), _run(_y0(), off, 5))


def test_pruning_takes_no_transition_below_the_prior():
    y0 = _cycle_y0()
    s = dataclasses.replace(_derived_cycle_set(), path_width=2, path_prior_min=0.5)
    with torch.no_grad(), dropout_key(0, 0.5):
        state, _ = P.walks(y0, [s], 3, False, 0.0, None, "layer", [None], None, None, 2)
    assert state[0]["nodes"][0][0, 0].item() in (0, 2)
    prior = P.transition_prior(s)
    assert prior[0, 0] < 0.5 and prior[2, 2] < 0.5


def test_diverse_starts_avoid_overlapping_outputs():
    e = torch.eye(4)
    keys_in = torch.stack([e[0], e[1], e[2]])[:, None, :]
    keys_out = torch.stack([e[3], e[3], e[2]])[:, None, :]
    s = LayerSet(
        router=torch.eye(4),
        keys_in=keys_in,
        keys_out=keys_out,
        per_expert=1,
        down=keys_in[:, 0],
        up=keys_out[:, 0],
        tau=40.0,
        iteration="path",
    )
    q = torch.tensor([[[0.7, 0.6, 0.39, 0.0]]])
    keys = [C._set_keys(s)]
    scores = P._scores([q], [s], keys)
    top = P._start_orders(scores, [s], keys, "top", 2)[0]
    diverse = P._start_orders(scores, [s], keys, "diverse", 2)[0]
    assert top[0, 0, :2].tolist() == [0, 1]
    assert diverse[0, 0, :2].tolist() == [0, 2]


def _best(state):
    return torch.stack([st["score"] for st in state], -1).amax(-1)


@pytest.mark.parametrize("seed", range(6))
def test_merging_never_lowers_the_best_path_score(seed):
    sets = _path(_sets(1, 16, 6, 1, 6, seed=seed), path_width=3)
    y0 = _y0(seed)
    with torch.no_grad(), dropout_key(0, 0.5):
        plain, _ = P.walks(y0, sets, 6, False, 0.0, None, "l", [None], None, None, 3)
        merged_sets = [dataclasses.replace(s, path_merge=True) for s in sets]
        merged, _ = P.walks(y0, merged_sets, 6, False, 0.0, None, "l", [None], None, None, 3)
    assert bool((_best(merged) >= _best(plain) - 1e-4).all())


@pytest.mark.parametrize("seed", range(6))
def test_beam_never_ends_below_the_greedy_walk(seed):
    sets = _path(_sets(1, 16, 6, 1, 6, seed=seed), path_width=3)
    y0 = _y0(seed)
    with torch.no_grad(), dropout_key(0, 0.5):
        greedy, _ = P.walks(y0, sets, 6, False, 0.0, None, "l", [None], None, None, 3)
        beam_sets = [dataclasses.replace(s, path_beam=True) for s in sets]
        beam, _ = P.walks(y0, beam_sets, 6, False, 0.0, None, "l", [None], None, None, 3)
    assert torch.equal(beam[0]["score"], greedy[0]["score"])
    assert bool((_best(beam) >= greedy[0]["score"]).all())
    assert bool((_best(beam) >= _best(greedy) - 1e-4).all())


def _shared_sets(seed=8, n=2):
    sets = _sets(n, 16, 4, 1, 8, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    out = []
    for s in sets:
        out.append(
            dataclasses.replace(
                s,
                router=sets[0].router,
                keys_out=torch.randn(4, 4, 8, generator=g),
                iteration="path",
                path_width=3,
            )
        )
    return out


def test_joint_prior_blocks_are_the_sets_own_priors():
    sets = _shared_sets()
    joint = P.joint_transition_prior(sets)
    assert joint.shape == (8, 8)
    torch.testing.assert_close(joint[:4, :4], P.transition_prior(sets[0]))
    torch.testing.assert_close(joint[4:, 4:], P.transition_prior(sets[1]))
    assert float(joint[:4, 4:].abs().sum()) > 0


@pytest.mark.parametrize("seed", range(4))
def test_joint_beam_keeps_the_greedy_walk_and_never_ends_below_it(seed):
    sets = _shared_sets(seed)
    y0 = _y0(seed)
    with torch.no_grad(), dropout_key(0, 0.5):
        greedy, _ = P.walks(y0, sets, 5, True, 0.0, None, "l", [None] * 2, None, None, 3)
        beam_sets = [dataclasses.replace(s, path_beam=True) for s in sets]
        beam, _ = P.walks(y0, beam_sets, 5, True, 0.0, None, "l", [None] * 2, None, None, 3)
    assert torch.equal(beam[0]["score"], greedy[0]["score"])
    for a, b in zip(beam[0]["zs"], greedy[0]["zs"], strict=True):
        assert torch.equal(a, b)
    assert bool((_best(beam) >= greedy[0]["score"]).all())
    with torch.no_grad(), dropout_key(0, 0.5):
        y = _run(y0, beam_sets, 5, shared=True)
    assert torch.isfinite(y).all()


def test_per_set_beam_is_each_sets_own_beam():
    sets = _path(_sets(2, 16, 4, 1, 8, seed=9), path_width=3, path_beam=True)
    y0 = _y0(9)
    with dropout_key(0, 0.5):
        both = _run(y0, sets, 4)
        alone = [_run(y0, [s], 4) - y0 for s in sets]
    torch.testing.assert_close(both, y0 + alone[0] + alone[1], rtol=1e-5, atol=1e-6)


def test_chosen_score_counts_only_sets_with_a_weight():
    scores = [torch.tensor([[[0.9, 0.1]]]), torch.tensor([[[0.7, 0.2]]])]
    gates = [torch.tensor([[[0.0, 1.0]]]), torch.zeros(1, 1, 2)]
    torch.testing.assert_close(P._chosen_score(scores, gates), torch.tensor([[0.1]]))


def _run_x(x, y0, sets, depth):
    with torch.no_grad(), dropout_key(0, 0.5):
        return run_layer(x.clone(), y0.clone(), sets, depth, target="layer")


def test_imported_set_takes_path_width():
    x = _cycle_y0()
    y0 = x.clone()
    imported = dataclasses.replace(_cycle_set(), form="input")
    old = _run_x(x, y0, [imported], 4)
    assert torch.equal(_run_x(x, y0, [dataclasses.replace(imported, path_width=1)], 4), old)
    two = _run_x(x, y0, [dataclasses.replace(imported, path_width=2)], 4)
    assert not torch.equal(two, old)
    assert _expert_of(two, y0).tolist() == [[1, 1]]
    frac = _run_x(x, y0, [dataclasses.replace(imported, path_width=2)], 3.5)
    assert torch.isfinite(frac).all()


def test_imported_and_state_sets_walk_together_with_path_width():
    x = _cycle_y0()
    y0 = x.clone()
    imported = dataclasses.replace(_cycle_set(), form="input")
    state = _path(_sets(1, 6, 2, 1, 6, seed=4))[0]
    old = _run_x(x, y0, [state, imported], 4)
    same = [dataclasses.replace(s, path_width=1) for s in (state, imported)]
    assert torch.equal(_run_x(x, y0, same, 4), old)
    wide = [dataclasses.replace(s, path_width=2) for s in (state, imported)]
    assert not torch.equal(_run_x(x, y0, wide, 4), old)


@pytest.mark.parametrize("form", ["state", "input"])
@pytest.mark.parametrize(
    "routing", [("token", "sparsemax"), ("tensor", "sparsemax"), ("token", "top1")]
)
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("select", ["converged", "score", "mix"])
def test_batched_walks_equal_the_walk_loop(form, routing, width, select, monkeypatch):
    opts = dict(routing_scope=routing[0], routing_select=routing[1], gate_mode="adaptive")
    s = _path(_sets(1, 16, 4, 2, 8, seed=31), path_width=width, path_select=select, **opts)[0]
    if form == "input":
        s = dataclasses.replace(s, form="input")
    y0 = _y0(32)
    region = (torch.arange(9) < 6).float()[None].expand(2, 9)
    outs = {}
    for mode in ("loop", "batched"):
        monkeypatch.setenv("KILORA_WALKS", mode)
        outs[mode] = [_run(y0, [s], d) for d in (1, 3, 4)]
        with torch.no_grad(), dropout_key(0, 0.5):
            outs[mode].append(
                run_layer(y0.clone(), y0.clone(), [s], 3, target="layer", regions=[region])
            )
    for a, b in zip(outs["loop"], outs["batched"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
