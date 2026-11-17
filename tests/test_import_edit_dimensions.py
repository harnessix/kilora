# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest
import torch
from test_import_modes import LAYERS, Toy, _lora

from kilora.live import SESSIONS
from kilora.live_imports import imported_session_for, moves
from kilora.multi_import import import_loras


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _session(tmp_path):
    model = Toy().eval()
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    sources = [
        dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            path=str(p),
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    return imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")


def test_identity_strength_and_space(tmp_path):
    session = _session(tmp_path)
    target = next(iter(session.state))
    clean = session._source.experts[target]
    assert torch.equal(session.sets()[0].experts[target].up, clean.up)
    core_moves = [
        m for m in moves(session) if m.target == target and m.expert == 0 and m.part == "core"
    ]
    assert len(core_moves) == 4
    aligned = session._source.experts[target]
    session.set_options(core_dims=1)
    assert (
        len(
            [
                m
                for m in session.moves()
                if m.target == target and m.expert == 0 and m.part == "core"
            ]
        )
        == 1
    )
    core_moves[0].apply(session.state, 0.3)
    changed = session.sets()[0].experts[target]
    a, b = aligned.down[0, 0], aligned.up[0, 0].T
    delta = changed.up[0, 0].T @ a - b @ a
    qb = torch.linalg.qr(b).Q
    qa = torch.linalg.qr(a.T).Q
    assert (
        torch.linalg.norm(delta - qb @ (qb.T @ delta @ qa) @ qa.T) / torch.linalg.norm(delta) < 1e-6
    )
    assert torch.equal(changed.routing_up, clean.up)
    session.state[target]["strength_0"].zero_()
    changed = session.sets()[0].experts[target]
    assert torch.count_nonzero(changed.up[0]) == 0
    assert torch.equal(changed.up[1], aligned.up[1])
    session.edits = "off"
    assert session.sets()[0] is session._plain_set


def test_save_load_edits(tmp_path):
    session = _session(tmp_path)
    target = next(iter(session.state))
    session.state[target]["core_0"][0, 1] = 0.25
    session.state[target]["strength_0"][0] = 0.5
    edit_path = tmp_path / "dimensions.kmod"
    session.save_edits(edit_path)
    saved = session.save_point("dimensions")
    session.reset()
    session.load(saved)
    assert session.state[target]["core_0"][0, 1] == 0.25
    assert session.state[target]["strength_0"][0] == 0.5
    session.reset()
    session.load_edits(edit_path)
    assert session.state[target]["core_0"][0, 1] == 0.25
    assert session.state[target]["strength_0"][0] == 0.5


def test_lokr_modes(tmp_path, monkeypatch):
    from test_lycoris_apply import paths_for
    from test_multi_import import Q

    from kilora.exact_lycoris import apply_exact

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model, paths = paths_for(tmp_path)
    sources = [
        dict(
            sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            path=p,
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    for mode in ("truncated", "exact"):
        imported = import_loras(paths, model, rank=4, lycoris_apply=mode)
        shapes = {t: list(e.down.shape) for t, e in imported.adapter_set.experts.items()}
        session = imported_session_for(imported.adapter_set, sources, "toy", shapes)
        original = session._source.experts[Q]
        own = [m for m in moves(session) if m.target == Q and m.expert == 0]
        assert sum(m.part == "strength" for m in own) == 1
        assert sum(m.part == "core" for m in own) == (4 if mode == "truncated" else 0)
        assert torch.equal(session.sets()[0].experts[Q].up, original.up)
        if mode == "exact":
            x = torch.randn(2, 16)
            baseline = apply_exact(original.exact[0], x)
            session.state[Q]["strength_0"][0] = 0.5
            edited = session.sets()[0].experts[Q]
            torch.testing.assert_close(apply_exact(edited.exact[0], x), baseline * 0.5)


def test_derived_routing_uses_clean_weights(tmp_path):
    import dataclasses

    from kilora import compose, compute

    session = _session(tmp_path)
    target = next(iter(session.state))
    model = Toy().eval()
    before = compose(model, session.sets())
    site_index = before.targets.index(target)
    clean_layer = before._layer_sets(site_index, before.sites[site_index], torch.device("cpu"))[0]
    clean_layer = dataclasses.replace(clean_layer, import_score="derived")
    x = torch.randn(1, 3, clean_layer.down.shape[-1])
    clean_logits = compute.derived_logits(clean_layer, x)
    before.remove()
    session.state[target]["strength_0"].zero_()
    after = compose(model, session.sets())
    site_index = after.targets.index(target)
    edited_layer = after._layer_sets(site_index, after.sites[site_index], torch.device("cpu"))[0]
    edited_layer = dataclasses.replace(edited_layer, import_score="derived")
    edited_logits = compute.derived_logits(edited_layer, x)
    assert torch.equal(clean_logits, edited_logits)
    clean_routes, edited_routes = [], []
    y = torch.randn(1, 3, clean_layer.up.shape[-1])
    with (
        torch.no_grad(),
        compute.recording_shares(lambda _, gates: clean_routes.append(gates[0].clone())),
    ):
        compute.run_layer(x, y, [clean_layer], 1, target=target)
    with (
        torch.no_grad(),
        compute.recording_shares(lambda _, gates: edited_routes.append(gates[0].clone())),
    ):
        compute.run_layer(x, y, [edited_layer], 1, target=target)
    assert len(clean_routes) == len(edited_routes) > 0
    assert all(torch.equal(a, b) for a, b in zip(clean_routes, edited_routes, strict=True))
    after.remove()


def test_exact_fixed_section_strength(tmp_path):
    from test_lycoris_apply import paths_for
    from test_multi_import import Q

    from kilora.multi_import import LoRASource
    from kilora.sections import layer_changes

    model, paths = paths_for(tmp_path)
    sources = [
        dict(
            sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            path=p,
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    imported = import_loras(
        [LoRASource(path=p, name=f"{i}-{Path(p).stem}") for i, p in enumerate(paths)],
        model,
        rank=4,
        lycoris_mode="exact",
    )
    shapes = {t: list(e.down.shape) for t, e in imported.adapter_set.experts.items()}
    session = imported_session_for(
        imported.adapter_set, sources, "toy", shapes, sections=imported.sections
    )
    own = [m for m in moves(session) if m.target == Q and m.expert == 0]
    assert [m.part for m in own] == ["strength"]
    x, zero = torch.randn(2, 16), torch.zeros(2, 16)
    original = layer_changes(session.sections(), 1.0)[Q][0].add(x, zero)
    session.state[Q]["strength_0"][0] = 0.5
    edited = layer_changes(session.sections(), 1.0)[Q][0].add(x, zero)
    torch.testing.assert_close(edited, original * 0.5)


def test_unpadded_rank_and_absent_expert(tmp_path):
    model = Toy().eval()
    paths = [tmp_path / f"rank{i}.safetensors" for i in range(2)]
    _lora(model, paths[0], rank=2, seed=0, layers=LAYERS[:1])
    _lora(model, paths[1], rank=4, seed=1, layers=LAYERS)
    sources = [
        dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            path=str(p),
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    imported = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in imported.experts.items()}
    session = imported_session_for(imported, sources, "toy", shapes)
    target = LAYERS[0]
    assert session.state[target]["core_0"].shape == (2, 2)
    session.state[target]["core_0"][0, 1] = 0.5
    ex = session.sets()[0].experts[target]
    assert torch.count_nonzero(ex.up[0, 0, 2:]) == 0
    assert "core_0" not in session.state[LAYERS[1]]
    assert "strength_0" not in session.state[LAYERS[1]]


def test_exact_import_fallback_stays_unedited(tmp_path, monkeypatch):
    from test_multi_import import Q, ToyAnima, _loha, lora_weights, to_keys, write

    monkeypatch.setenv("KILORA_LOKR_BUDGET_MIB", "0")
    model = ToyAnima().eval()
    paths = [
        write(tmp_path, "loha", _loha()[0]),
        write(tmp_path, "small", to_keys(lora_weights(model, layers=[Q], rank=1), "kohya")),
    ]
    sources = [
        dict(
            sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            path=p,
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    imported = import_loras(paths, model, rank=1, lycoris_apply="exact").adapter_set
    assert 0 in imported.experts[Q].exact_unavailable
    shapes = {t: list(e.down.shape) for t, e in imported.experts.items()}
    session = imported_session_for(imported, sources, "toy", shapes)
    assert not [m for m in moves(session) if m.target == Q and m.expert == 0]
    assert torch.equal(session.sets()[0].experts[Q].up, imported.experts[Q].up)


def test_core_moves_on_rank_deficient_layer(tmp_path):
    model = Toy().eval()
    path = tmp_path / "flat.safetensors"
    _lora(model, path, rank=4, seed=0, alpha=4.0, layers=LAYERS)
    routed = import_loras([str(path)], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    target = next(iter(routed.experts))
    ex = routed.experts[target]
    ex.down[0, 0, 1:] = 0
    ex.up[0, 0, 2:] = 0
    source = dict(sha256="flat", path=str(path), strength=1.0, axis="both")
    session = imported_session_for(routed, [source], "toy", shapes)
    own = [m for m in moves(session) if m.target == target and m.expert == 0]
    assert sum(m.part == "strength" for m in own) == 1
    assert sum(m.part == "core" for m in own) == 1
    assert all(torch.isfinite(m.direction).all() for m in own)
