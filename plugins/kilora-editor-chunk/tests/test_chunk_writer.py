# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from kilora_editor_chunk import Chunk, ChunkEditor, ChunkState, ChunkWriter

NAMES = ["layers.0.attention.to_q", "layers.0.attention.to_out.0"]
SHAPES = {h: [6, 4] for h in NAMES}


def _editor():
    q, _ = torch.linalg.qr(torch.randn(48, 20, generator=torch.Generator().manual_seed(0)))
    rows = q.t()
    return ChunkEditor(
        NAMES,
        SHAPES,
        {"to_q": 2.0, "to_k": 1.0, "to_v": 3.0},
        [
            Chunk("common", rows[:12], torch.ones(12)),
            Chunk("retro", rows[12:], torch.ones(8)),
        ],
    )


def _loss_fn(target):
    def loss(pert, t, seed):
        return float(sum(((v - target[h]) ** 2).sum() for h, v in pert.items()))

    return loss


def _run(ed, n=5):
    st = ChunkState.new(ed)
    tgt = {h: torch.randn(*SHAPES[h], generator=torch.Generator().manual_seed(3)) for h in NAMES}
    w = ChunkWriter(
        ed,
        times=[0.7, 0.95],
        weights=[0.5, 0.5],
        eps=[0.1, 0.1],
        write_norm=0.2,
        per_event=4,
        measure="full",
    )
    for e in range(n):
        st.push()
        w.step(st, _loss_fn(tgt), {h: [0.4, 0.6] for h in NAMES}, seed=e)
    return st


def test_file_round_trip_and_orthogonality(tmp_path):
    ed = _editor()
    files = ed.save_packs(tmp_path, "1.0.0", "z-image", "Apache-2.0")
    ed2 = ChunkEditor.load_packs([files["common"], files["retro"]])
    assert ed2.family_names() == ["common", "retro"]
    assert torch.allclose(ed2.chunks[0].q, ed.chunks[0].q, atol=0.02)
    bad = ChunkEditor(
        NAMES, SHAPES, ed.alpha, [ed.chunks[0], Chunk("dup", ed.chunks[0].q[:2], torch.ones(2))]
    )
    try:
        bad.check_orthogonal()
        raise AssertionError("not orthogonal chunks were accepted")
    except ValueError:
        pass


def test_writer_is_deterministic():
    ed = _editor()
    a, b = _run(ed), _run(ed)
    assert all(torch.equal(a.coef[k], b.coef[k]) for k in a.coef)
    assert all(
        torch.equal(torch.nan_to_num(a.hist[k]), torch.nan_to_num(b.hist[k])) for k in a.hist
    )


def test_undo_reset_save_reload_bit_identical(tmp_path):
    ed = _editor()
    st = _run(ed, 4)
    before = st.snapshot()
    st.push()
    ChunkWriter(ed, [0.7], [1.0], [0.1], 0.2, per_event=4).step(st, lambda p, t, s: 0.0, {}, 9)
    assert st.undo()
    assert all(torch.equal(st.coef[k], before["coef"][k]) for k in st.coef)
    st.save(tmp_path / "s.kstate", ed)
    st2 = ChunkState.load(tmp_path / "s.kstate")
    assert all(torch.equal(st2.coef[k], st.coef[k]) for k in st.coef) and st2.enabled == st.enabled
    st2.reset()
    assert all(float(v.abs().sum()) == 0 for v in st2.coef.values())
    assert st2.undo() and all(torch.equal(st2.coef[k], st.coef[k]) for k in st.coef)


def test_disabling_a_chunk_is_exact_and_undoable():
    ed = _editor()
    st = _run(ed, 6)
    full = st.expanded_u(ed)
    st.set_enabled("retro", False)
    off = st.expanded_u(ed)
    alone = ChunkState(
        {"common": st.coef["common"].clone(), "retro": torch.zeros_like(st.coef["retro"])},
        st.hist,
        ["common", "retro"],
    )
    ref = alone.expanded_u(ed)
    assert all(torch.equal(off[h], ref[h]) for h in NAMES)
    assert st.undo()
    back = st.expanded_u(ed)
    assert all(torch.equal(back[h], full[h]) for h in NAMES)


def test_chunk_and_total_caps():
    ed = _editor()
    ed.chunks[1].mean_keys = {h: torch.eye(4) * 4 for h in NAMES}
    st = ChunkState.new(ed)
    tgt = {h: torch.randn(*SHAPES[h], generator=torch.Generator().manual_seed(3)) for h in NAMES}
    w = ChunkWriter(ed, [0.7], [1.0], [0.1], write_norm=1.0, per_event=20, chunk_cap=0.3, decay=0.0)
    st.push()
    w.step(st, _loss_fn(tgt), {}, 0)
    for c in ed.chunks:
        assert w._norm({c.family: st.coef[c.family]}) <= 0.3 + 1e-5
    w2 = ChunkWriter(
        ed, [0.7], [1.0], [0.1], write_norm=1.0, per_event=20, total_cap=0.5, decay=0.0
    )
    st2 = ChunkState.new(ed)
    for e in range(4):
        w2.step(st2, _loss_fn(tgt), {}, e)
    assert w2._norm({f: st2.coef[f] for f in st2.enabled}) <= 0.5 + 1e-5


def test_packs_split_and_join(tmp_path):
    ed = _editor()
    files = ed.save_packs(tmp_path / "packs", "1.0.0", "z-image", "Apache-2.0")
    assert set(files) == {"common", "retro"}
    assert files["retro"].name == "chunk-retro-1.0.0.kedit-pack.safetensors"
    again = ed.save_packs(tmp_path / "again", "1.0.0", "z-image", "Apache-2.0")
    assert again["retro"].read_bytes() == files["retro"].read_bytes()
    joined = ChunkEditor.load_packs([files["common"], files["retro"]])
    assert joined.family_names() == ["common", "retro"]
    only = ChunkEditor.load_packs([files["common"]])
    assert only.family_names() == ["common"]


def test_one_sided_shares_the_base_forward():
    ed = _editor()
    tgt = {h: torch.randn(*SHAPES[h], generator=torch.Generator().manual_seed(3)) for h in NAMES}
    calls, hists = {}, {}
    for diff in ("central", "one-sided"):
        n = [0]
        inner = _loss_fn(tgt)

        def loss(pert, t, seed, inner=inner, n=n):
            n[0] += 1
            return inner(pert, t, seed)

        st = ChunkState.new(ed)
        w = ChunkWriter(
            ed, [0.7, 0.95], [0.5, 0.5], [1e-3, 1e-3], 0.2, per_event=4, diff=diff, measure="full"
        )
        w.step(st, loss, {}, 0)
        calls[diff] = n[0]
        hists[diff] = torch.cat([torch.nan_to_num(v[-1]) for v in st.hist.values()])
    assert ChunkWriter(ed, [0.7], [1.0], [0.1], 0.2).diff == "central"
    assert calls == {"central": 16, "one-sided": 10}
    cos = torch.nn.functional.cosine_similarity(hists["central"], hists["one-sided"], dim=0)
    assert float(cos) > 0.999


def _linear_setup(sigma):
    names = ["layers.0.attention.to_q", "layers.0.attention.to_v"]
    shapes = {h: [8, 4] for h in names}
    q, _ = torch.linalg.qr(torch.randn(64, 48, generator=torch.Generator().manual_seed(0)))
    rows = q.t()
    ed = ChunkEditor(
        names,
        shapes,
        {"to_q": 2.0, "to_k": 1.0, "to_v": 3.0},
        [Chunk("common", rows, torch.ones(48))],
    )
    g = {h: torch.randn(*shapes[h], generator=torch.Generator().manual_seed(1)) for h in names}
    noise = torch.Generator().manual_seed(2)

    def loss(pert, t, seed):
        clean = float(sum((g[h] * v).sum() for h, v in pert.items()))
        return clean + sigma * float(torch.randn((), generator=noise))

    alpha = {h: ed.alpha[h.rsplit(".", 1)[-1]] for h in names}
    true = torch.tensor(
        [
            sum(
                alpha[h] * float((g[h] * d[0]).sum())
                for h, d in ed.per_hook(rows[j : j + 1]).items()
            )
            for j in range(48)
        ]
    )
    return ed, loss, true


def test_compressed_recovers_the_slopes():
    for sigma in (0.0, 0.02):
        ed, loss, true = _linear_setup(sigma)
        st = ChunkState.new(ed)
        w = ChunkWriter(ed, [0.7], [1.0], [0.1], 0.0, decay=0.0)
        assert w.measure == "compressed"
        for e in range(8):
            w.step(st, loss, {}, seed=e)
        err = float((st.prior["slopes"] - true).norm() / true.norm())
        assert err < 0.15, (sigma, err)


def test_compressed_counts_and_round_trip(tmp_path):
    ed, loss, _ = _linear_setup(0.0)
    n = [0]

    def counted(pert, t, seed):
        n[0] += 1
        return loss(pert, t, seed)

    st = ChunkState.new(ed)
    w = ChunkWriter(ed, [0.7], [1.0], [0.1], 0.2)
    w.step(st, counted, {}, seed=0)
    assert n[0] == 2 * 12
    n[0] = 0
    ChunkWriter(ed, [0.7], [1.0], [0.1], 0.2, combos=6, diff="one-sided").step(
        st, counted, {}, seed=1
    )
    assert n[0] == 6 + 1
    st.save(tmp_path / "s.safetensors", ed)
    back = ChunkState.load(tmp_path / "s.safetensors")
    assert all(torch.equal(back.prior[k], st.prior[k]) for k in st.prior)
    st.push()
    w.step(st, loss, {}, seed=2)
    st.undo()
    assert all(torch.equal(back.prior[k], st.prior[k]) for k in st.prior)


@pytest.mark.parametrize("seed", [0, 1, 3, 17, 822])
@pytest.mark.parametrize("use_modulator", [False, True])
@pytest.mark.parametrize("custom_norm", [False, True])
def test_interpreted_write_matches_uncapped_native_step(seed, use_modulator, custom_norm):
    from kilora.ir.defaults import interpret_live_write

    ed = _editor()
    state = ChunkState.new(ed, experts=2)
    generator = torch.Generator().manual_seed(seed)
    for family in state.coef:
        state.coef[family] = torch.randn(state.coef[family].shape, generator=generator)
        state.hist[family] = torch.randn(state.hist[family].shape, generator=generator)
        state.hist[family][1:3, ::2] = float("nan")
    previous = {name: value.clone() for name, value in state.coef.items()}
    multipliers = {
        name: torch.linspace(0.3, 1.2, value.shape[1]) for name, value in state.hist.items()
    }
    joined = torch.cat(list(multipliers.values()))
    norm = (
        (lambda values: sum(float(v.abs().sum()) for v in values.values())) if custom_norm else None
    )
    writer = ChunkWriter(
        ed,
        [0.7],
        [1.0],
        [0.1],
        0.2,
        per_event=4,
        measure="full",
        modulate=(lambda extra, slopes: joined) if use_modulator else None,
        norm=norm,
    )
    shares = torch.tensor([0.4, 0.6])
    writer.step(state, _loss_fn({h: torch.ones(*SHAPES[h]) for h in NAMES}), {"a": [0.4, 0.6]}, 4)
    result = interpret_live_write(
        state.hist,
        previous,
        shares,
        writer.write_norm,
        writer.decay,
        multipliers=multipliers if use_modulator else None,
        norm=norm,
    )
    assert all(torch.equal(result[name], state.coef[name]) for name in previous)


def test_interpreted_write_with_no_measured_slopes():
    from kilora.ir.defaults import interpret_live_write

    state = ChunkState.new(_editor(), experts=2)
    result = interpret_live_write(state.hist, state.coef, torch.tensor([0.5, 0.5]), 0.2, 0.02)
    assert all(torch.equal(result[name], value) for name, value in state.coef.items())


def test_expanded_u_without_node_coefficients_matches_original_expression():
    ed = _editor()
    st = _run(ed, 3)
    expected = None
    for chunk in ed.chunks:
        if chunk.family not in st.enabled:
            continue
        part = {
            h: torch.einsum("ej,jor->eor", st.coef[chunk.family], direction)
            for h, direction in ed.per_hook(chunk.q).items()
        }
        expected = part if expected is None else {h: expected[h] + part[h] for h in expected}
    assert all(torch.equal(st.expanded_u(ed)[h], expected[h]) for h in ed.names)
