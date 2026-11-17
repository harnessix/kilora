# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import types

import pytest
import torch
from safetensors.torch import save_file
from test_import_modes import DIM, LAYERS, Toy

from kilora import compose
from kilora.editors import commit_write
from kilora.live import SESSIONS
from kilora.live_experts import RouteRecord
from kilora.live_imports import ImportEditor, imported_session_for
from kilora.multi_import import import_loras
from kilora.path_merge import (
    KeepMeans,
    MergeProfile,
    _fold_reference,
    fold_import_paths,
    mean_keeps,
)
from kilora.recording import GenerationRecord

_PASS = {"gap": 1.0, "spread": 1.0}


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _pin(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    monkeypatch.setenv("KILORA_READ_DROPOUT", "off")
    monkeypatch.setenv("KILORA_MASKS", "expect")


def _small_lora(model, path, *, rank=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    stored = {}
    for name in LAYERS:
        layer = model.get_submodule(name)
        down = torch.randn(rank, layer.in_features, generator=gen) * 0.02
        up = torch.randn(layer.out_features, rank, generator=gen) * 0.02
        stem = "lora_unet_" + name.replace(".", "_")
        stored[f"{stem}.lora_down.weight"] = down
        stored[f"{stem}.lora_up.weight"] = up
        stored[f"{stem}.alpha"] = torch.tensor(float(rank))
    save_file(stored, str(path))


def _world(tmp_path):
    model = Toy().eval()
    path = tmp_path / "chain.safetensors"
    _small_lora(model, path)
    routed = import_loras([str(path)], model).adapter_set
    shapes = {name: list(expert.down.shape) for name, expert in routed.experts.items()}
    base = {key: value.detach().clone() for key, value in model.state_dict().items()}
    original = path.read_bytes()

    def open_session(tag):
        digest = hashlib.sha256(tag.encode()).hexdigest()
        sources = [dict(sha256=digest, path=str(path), strength=1.0, axis="both")]
        return imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves" / tag)

    return model, path, base, original, open_session


def _inputs():
    return torch.randn(2, 3, DIM, generator=torch.Generator().manual_seed(7))


def _render(model, base, session, depth, x):
    model.load_state_dict(base)
    handle = compose(model, session.sets(use_pending=False), depth=depth)
    try:
        with torch.no_grad():
            return model(x).detach().clone()
    finally:
        handle.remove()


def _measure(model, base, session, x):
    def measure(trial):
        _render(model, base, session, 7, x)
        total = 0.0
        for target, parts in session.state.items():
            up = session._source.experts[target].up
            delta = None if trial is None else trial.du.get(target)
            flat = None if delta is None else delta.reshape_as(up)
            for key, value in parts.items():
                if not key.startswith("strength_"):
                    continue
                expert = int(key.removeprefix("strength_"))
                shift = 0.0
                if flat is not None:
                    source = up[expert, 0].float()
                    shift = float((flat[expert, 0].float() * source).sum() / source.square().sum())
                total += (float(value) + shift - 1.5) ** 2
        return total

    return measure


def _commit(session, model, base, x, *, path_merge, guard_score, routes=None):
    session.learning = True
    session.set_options(
        core_dims=0,
        controller="rprop",
        measure="full",
        editor_options={
            "path_merge": path_merge,
            "path_angle": 0.0,
            "merge_residual": "drop",
        },
    )
    editor = ImportEditor(session)
    editor.configure(session.editor_options)
    record = GenerationRecord(measure=_measure(model, base, session, x), guard_score=guard_score)
    if routes is not None:
        record.route_record = routes
    report = commit_write(session, editor, record)
    return editor, report


def _pass(_trial):
    return dict(_PASS)


def _equal_state(left, right):
    assert set(left.state) == set(right.state)
    for target, parts in left.state.items():
        other = right.state[target]
        assert set(parts) == set(other)
        for key, value in parts.items():
            assert torch.equal(value, other[key])


def test_path_merge_off_matches(tmp_path, monkeypatch):
    _pin(monkeypatch)
    model, path, base, original, open_session = _world(tmp_path)
    x = _inputs()
    first = open_session("off-a")
    second = open_session("off-b")
    _, first_report = _commit(first, model, base, x, path_merge=False, guard_score=_pass)
    _, second_report = _commit(second, model, base, x, path_merge=False, guard_score=_pass)
    assert len(first.commits) == 1 and len(second.commits) == 1
    assert not first.commits[0].source.endswith("+merge")
    assert not second.commits[0].source.endswith("+merge")
    assert "source=merge" not in first_report
    assert "source=merge" not in second_report
    _equal_state(first, second)
    assert torch.equal(_render(model, base, first, 1, x), _render(model, base, second, 1, x))
    assert torch.equal(_render(model, base, first, 7, x), _render(model, base, second, 7, x))
    assert path.read_bytes() == original


def test_merge_moves_depth_undo_reset_and_export(tmp_path, monkeypatch):
    _pin(monkeypatch)
    model, path, base, original, open_session = _world(tmp_path)
    x = _inputs()
    clean = open_session("clean")
    off = open_session("off")
    on = open_session("on")
    _, off_report = _commit(off, model, base, x, path_merge=False, guard_score=_pass)
    editor, report = _commit(on, model, base, x, path_merge=True, guard_score=_pass)
    assert "source=merge" not in off_report
    assert len(on.commits) == 1
    assert on.commits[-1].source.endswith("+merge")
    assert "source=merge" in report
    assert "source=merge" in editor.note
    before = _render(model, base, off, 1, x)
    target = _render(model, base, off, 7, x)
    after = _render(model, base, on, 1, x)
    before_distance = torch.linalg.norm(before - target)
    after_distance = torch.linalg.norm(after - target)
    assert after_distance < before_distance
    assert on.editor_state.path_node_versions
    exported = tmp_path / "out.kmod"
    on.export(exported)
    assert exported.is_file() and exported.stat().st_size > 0
    assert exported.resolve() != path.resolve()
    assert path.read_bytes() == original
    on.undo()
    assert len(on.commits) == 0
    assert on.editor_state.event == 0 and not on.editor_state.rules
    _equal_state(clean, on)
    assert torch.equal(_render(model, base, clean, 1, x), _render(model, base, on, 1, x))
    assert torch.equal(_render(model, base, clean, 7, x), _render(model, base, on, 7, x))
    assert on.editor_state.path_node_versions
    on.reset()
    history = on.editor_state
    assert history.path_node_versions == {}
    assert history.path_node_values == {}
    assert history.path_graph.stats == (0, 0, 0, 0)


def test_refused_merge_keeps_state_and_graph(tmp_path, monkeypatch):
    _pin(monkeypatch)
    model, _path, base, _original, open_session = _world(tmp_path)
    x = _inputs()
    off = open_session("off")
    guarded = open_session("guard")
    _commit(off, model, base, x, path_merge=False, guard_score=_pass)
    calls = {"n": 0}

    def guard_score(_trial):
        calls["n"] += 1
        if calls["n"] >= 4:
            return {"gap": 0.0, "spread": 1.0}
        return dict(_PASS)

    _editor, report = _commit(guarded, model, base, x, path_merge=True, guard_score=guard_score)
    assert calls["n"] >= 4
    assert "Merge refused" in report
    assert len(guarded.commits) == 1
    assert not guarded.commits[0].source.endswith("+merge")
    _equal_state(off, guarded)
    history = guarded.editor_state
    assert history.path_node_versions == {}
    assert history.path_node_values == {}
    assert history.path_graph.stats == (0, 0, 0, 0)


def _three_experts(tmp_path):
    model = Toy().eval()
    paths = [tmp_path / f"expert{i}.safetensors" for i in range(3)]
    for i, path in enumerate(paths):
        _small_lora(model, path, rank=2, seed=i)
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {name: list(expert.down.shape) for name, expert in routed.experts.items()}
    sources = [
        dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), path=str(p), strength=1.0)
        for p in paths
    ]
    for source in sources:
        source["axis"] = "both"
    return imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves3")


def _fold_inputs(session, *, depth=4, paths=3, seed=3):
    gen = torch.Generator().manual_seed(seed)
    before = {t: {k: v.detach().clone() for k, v in p.items()} for t, p in session.state.items()}
    after = {t: {k: v.detach().clone() for k, v in p.items()} for t, p in session.state.items()}
    for parts in after.values():
        for key, value in parts.items():
            if key.startswith("core_"):
                noise = torch.randn(value.shape, generator=gen).to(value)
                parts[key] = value + 0.05 * noise
    experts = session._source.num_experts
    keeps = KeepMeans(experts)
    for target in session.state:
        for step in range(depth - 1):
            for _path in range(paths):
                keeps.add((target, step), 0.5 + 0.5 * torch.rand(4, experts, 2, generator=gen))
    return before, after, mean_keeps(keeps, experts)


def _close(left, right):
    assert left is not None and right is not None
    assert set(left) == set(right)
    for target, parts in left.items():
        assert set(parts) == set(right[target])
        for key, value in parts.items():
            other = right[target][key]
            assert value.shape == other.shape, (target, key)
            if key.startswith(("core_", "strength_")):
                assert torch.allclose(value.float(), other.float(), rtol=1e-4, atol=1e-6), (
                    target,
                    key,
                )


def _expert_of(key):
    prefix, sep, number = key.rpartition("_")
    return int(number) if sep and prefix in ("core", "strength") and number.isdigit() else None


def test_a_target_of_every_expert_folds_as_before(tmp_path):
    session = _three_experts(tmp_path)
    before, after, keeps = _fold_inputs(session)
    options = {"path_angle": 0.05}
    reference = _fold_reference(session, options, types.SimpleNamespace(), before, after, keeps)
    every = {t: list(range(session._source.num_experts)) for t in session.state}
    for target in (None, every):
        history = types.SimpleNamespace()
        merged = fold_import_paths(session, options, history, before, after, keeps, target=target)
        _close(merged, reference)
        assert history.path_node_versions
        assert set(history.path_node_values) == set(history.path_node_versions)


def test_the_fold_visits_only_the_target_and_keeps_the_rest(tmp_path):
    session = _three_experts(tmp_path)
    before, after, keeps = _fold_inputs(session)
    experts = session._source.num_experts
    assert experts == 3
    hook = sorted(session.state)[0]
    target = {hook: [0, 2]}
    profile = MergeProfile()
    merged = fold_import_paths(
        session, {}, types.SimpleNamespace(), before, after, keeps, target=target, profile=profile
    )
    assert profile.hooks == 1
    assert profile.experts == 2
    assert 0 < profile.steps <= 3 * 2 * 2
    assert profile.rotations <= profile.steps
    full = MergeProfile()
    fold_import_paths(
        session, {}, types.SimpleNamespace(), before, after, keeps, target=None, profile=full
    )
    assert full.hooks == full.graph_hooks == profile.graph_hooks
    assert full.experts == full.graph_hooks * experts
    assert profile.steps < full.steps <= full.graph_hooks * 3 * experts * experts
    assert merged is not None
    for name, parts in merged.items():
        for key, value in parts.items():
            expert = _expert_of(key)
            if name == hook and (expert is None or expert in target[hook]):
                continue
            assert torch.equal(value, after[name][key]), (name, key)
    assert any(not torch.equal(merged[hook][f"core_{e}"], after[hook][f"core_{e}"]) for e in (0, 2))


def test_the_fold_visits_do_not_grow_with_the_path_count(tmp_path):
    session = _three_experts(tmp_path)
    counts = []
    for paths in (1, 3, 6):
        before, after, keeps = _fold_inputs(session, paths=paths)
        profile = MergeProfile()
        profile.paths = paths
        fold_import_paths(
            session, {}, types.SimpleNamespace(), before, after, keeps, profile=profile
        )
        counts.append((profile.hooks, profile.experts, profile.steps, profile.rotations))
        assert f"paths={paths} " in profile.line()
    assert counts[0] == counts[1] == counts[2]


@pytest.mark.parametrize("mode", ["keys", "full"])
def test_the_merge_after_a_write_folds_only_the_target(tmp_path, monkeypatch, mode):
    _pin(monkeypatch)
    monkeypatch.setenv("KILORA_DERIVED_MODE", mode)
    monkeypatch.setenv("KILORA_WRITE_PROFILE", "1")
    model, _path, base, _original, open_session = _world(tmp_path)
    x = _inputs()
    session = open_session(f"target-{mode}")
    start = {t: {k: v.detach().clone() for k, v in p.items()} for t, p in session.state.items()}
    chosen = sorted(session.state)[0]
    experts = session._source.num_experts
    routes = RouteRecord()
    routes.mass[chosen] = torch.ones(experts)
    routes.ran.add(chosen)
    editor, _report = _commit(
        session, model, base, x, path_merge=True, guard_score=_pass, routes=routes
    )
    assert "merge_profile:" in editor.note
    assert " hooks=1 " in editor.note
    assert f" experts={experts} " in editor.note
    for name, parts in session.state.items():
        if name == chosen:
            continue
        for key, value in parts.items():
            assert torch.equal(value, start[name][key]), (name, key)


def _held_commit(session, model, base, x, held, guard_score, path_merge=True):
    measure = _measure(model, base, session, x)
    measure.held = held
    session.learning = True
    session.set_options(
        speed=1.0,
        core_dims=0,
        controller="rprop",
        measure="full",
        editor_options={"path_merge": path_merge, "path_angle": 0.0, "merge_residual": "drop"},
    )
    editor = ImportEditor(session)
    editor.configure(session.editor_options)
    record = GenerationRecord(measure=measure, guard_score=guard_score)
    return editor, commit_write(session, editor, record)


def _merge_raises(trial):
    return 1.0 if trial.du or trial.terms else 0.0


def _merge_lowers(trial):
    return 0.5 if trial.du or trial.terms else 1.0


def _write_stays(off, session, report):
    assert len(session.commits) == 1
    assert not session.commits[0].source.endswith("+merge")
    assert "the write stays" in report
    _equal_state(off, session)
    history = session.editor_state
    assert history.path_node_versions == {}
    assert history.path_graph.stats == (0, 0, 0, 0)


def test_a_merge_that_raises_the_held_out_loss_is_dropped(tmp_path, monkeypatch):
    _pin(monkeypatch)
    model, _path, base, _original, open_session = _world(tmp_path)
    x = _inputs()
    off = open_session("off")
    held = open_session("held")
    _held_commit(off, model, base, x, _merge_raises, _pass, path_merge=False)
    _editor, report = _held_commit(held, model, base, x, _merge_raises, _pass)
    assert "Merge dropped (the held-out loss 0 before, 1 after the merge" in report
    assert "source=merge" not in report
    _write_stays(off, held, report)


def test_a_merge_that_fails_the_guard_is_dropped(tmp_path, monkeypatch):
    _pin(monkeypatch)
    model, _path, base, _original, open_session = _world(tmp_path)
    x = _inputs()
    off = open_session("off")
    guarded = open_session("guard")
    _held_commit(off, model, base, x, _merge_lowers, _pass, path_merge=False)

    def guard_score(trial):
        merged = trial is not None and bool(trial.du or trial.terms)
        if merged and guarded.commits:
            return {"gap": 1.0, "spread": 0.5}
        return dict(_PASS)

    _editor, report = _held_commit(guarded, model, base, x, _merge_lowers, guard_score)
    assert "Merge refused and dropped (the guard:" in report
    _write_stays(off, guarded, report)


def test_a_merge_that_lowers_the_held_out_loss_and_passes_the_guard_is_accepted(
    tmp_path, monkeypatch
):
    _pin(monkeypatch)
    model, _path, base, _original, open_session = _world(tmp_path)
    x = _inputs()
    on = open_session("on")
    editor, report = _held_commit(on, model, base, x, _merge_lowers, _pass)
    assert "Merge accepted (held-out loss 1 before, 0.5 after the merge" in report
    assert on.commits[-1].source.endswith("+merge")
    assert "source=merge" in report


def test_a_write_with_no_target_change_runs_no_fold(tmp_path, monkeypatch):
    from kilora import path_merge

    session = _three_experts(tmp_path)
    before, after, keeps = _fold_inputs(session)
    hook = sorted(session.state)[0]
    for key in ("core_0", "core_2"):
        after[hook][key] = before[hook][key].clone()
    for name in session.state:
        if name != hook:
            after[name] = {k: v.clone() for k, v in before[name].items()}

    def no_fold(*args, **kwargs):
        raise AssertionError("the fold ran")

    monkeypatch.setattr(path_merge, "_fold_hook", no_fold)
    monkeypatch.setattr(path_merge, "_project_mixes", no_fold)
    monkeypatch.setattr(path_merge, "_import_graph", no_fold)
    history = types.SimpleNamespace()
    profile = MergeProfile()
    merged = fold_import_paths(
        session, {}, history, before, after, keeps, target={hook: [0, 2]}, profile=profile
    )
    assert merged is None
    assert profile.skipped == path_merge.NO_TARGET_CHANGE
    assert profile.pass_seconds == profile.project_seconds == 0.0
    assert "skipped=no_target_change" in profile.line()
    assert history.path_node_versions == {}


def _old_store_leftover(session, history, parts, target, expert, leftover):
    from kilora.import_terms import add_direction

    if float(leftover.norm()) <= 1e-5:
        return
    ex = session._source.experts[target]
    rank = leftover.shape[0]
    down = ex.down[expert, 0, :rank].detach().float()
    up = ex.up[expert, 0, :rank].detach().float().T
    extra = up @ leftover.to(up) @ down
    if float(extra.norm()) <= 1e-6:
        return
    left, values, right = torch.linalg.svd(extra, full_matrices=False)
    if not values.numel() or float(values[0]) <= 1e-8:
        return
    scale = float(values[0]) ** 0.5
    serial = int(getattr(history, "path_term_serial", 0))
    history.path_term_serial = serial + 1
    row, col = add_direction(
        parts,
        f"path-{serial}-o",
        (left[:, 0] * scale).contiguous(),
        f"path-{serial}-i",
        (right[0] * scale).contiguous(),
    )
    coeff = parts["terms_c"].clone()
    coeff[row, col] = coeff[row, col] + 1
    parts["terms_c"] = coeff


def _old_project(session, history, before, merged, old_u, new_u, experts, core_moves):
    import math

    from kilora.path_merge import _blend_state, _cap_scale

    for hook, updated in new_u.items():
        if hook not in merged:
            continue
        parts = merged[hook]
        previous = old_u[hook]
        for expert in sorted(experts.get(hook, ())):
            core_key, strength_key = f"core_{expert}", f"strength_{expert}"
            core = parts.get(core_key)
            strength = parts.get(strength_key)
            if core is None or strength is None or tuple(core.shape) != tuple(previous.shape[1:]):
                continue
            delta = (updated[expert] - previous[expert]).float()
            if not bool(torch.count_nonzero(delta)):
                continue
            current = core.float()
            denom = float(current.square().sum())
            if denom:
                step = float((delta * current).sum()) / denom
                if math.isfinite(step) and step:
                    parts[strength_key] = strength + step * torch.ones_like(strength)
                    delta = delta - step * current
            strength0 = float(strength.detach().reshape(()))
            if float(delta.norm()) > 1e-5 and strength0:
                for move in core_moves(hook, expert):
                    tangent = strength0 * move.direction.float()
                    width = float(tangent.square().sum())
                    if not width:
                        continue
                    step = float((delta * tangent).sum()) / width
                    if not math.isfinite(step) or not step:
                        continue
                    move.apply(merged, step)
                    delta = delta - step * tangent
            base = float((updated[expert] - previous[expert]).float().norm())
            if float(delta.norm()) > max(1e-5, 1e-4 * base):
                _old_store_leftover(session, history, parts, hook, expert, delta)
    return _blend_state(before, merged, _cap_scale(before, merged))


def _term_weight(parts):
    if "terms_c" not in parts:
        return None
    return parts["terms_out"].float() @ parts["terms_c"].float() @ parts["terms_in"].float().T


def _project_inputs(session, seed=11):
    from kilora.path_merge import _hook_ranks, _mixes

    _before, after, _keeps = _fold_inputs(session)
    experts, hooks = _hook_ranks(session, after)
    mixes = _mixes(after, after, hooks, experts)
    gen = torch.Generator().manual_seed(seed)
    old_u = {h: mixes[h][1] for h, _rank in hooks}
    new_u = {h: u + 0.02 * torch.randn(u.shape, generator=gen) for h, u in old_u.items()}
    return after, old_u, new_u


@pytest.mark.parametrize("dims", [1, 2])
def test_the_projection_touches_only_the_target_and_matches_the_loop(tmp_path, dims):
    from kilora.path_merge import _clone_state, _pair_core_moves, _project_mixes

    session = _three_experts(tmp_path)
    session.core_dims = dims
    after, old_u, new_u = _project_inputs(session)
    hooks = sorted(new_u)
    target = {hooks[0]: [0, 2], hooks[1]: [1]}
    moves = _pair_core_moves(session)
    asked = []

    def counted(hook, expert):
        asked.append((hook, expert))
        return moves(hook, expert)

    counts = {}
    new = _project_mixes(
        session,
        types.SimpleNamespace(),
        after,
        _clone_state(after),
        old_u,
        new_u,
        experts=target,
        core_moves=counted,
        counts=counts,
    )
    old = _old_project(
        session,
        types.SimpleNamespace(),
        after,
        _clone_state(after),
        old_u,
        new_u,
        target,
        _pair_core_moves(session),
    )
    assert sorted(asked) == sorted((h, e) for h, chosen in target.items() for e in chosen)
    assert counts["hooks"] == 2 and counts["experts"] == 3
    assert counts["moves"] > 0
    if dims == 1:
        assert counts["terms"] > 0
    for name, parts in new.items():
        chosen = target.get(name, ())
        for key, value in parts.items():
            expert = _expert_of(key)
            if expert is not None and expert not in chosen:
                assert torch.equal(value, after[name][key]), (name, key)
            elif expert is not None:
                assert torch.allclose(
                    value.float(), old[name][key].float(), rtol=1e-4, atol=1e-6
                ), (name, key)
        mine, theirs = _term_weight(parts), _term_weight(old[name])
        assert (mine is None) == (theirs is None), name
        if mine is not None:
            assert torch.allclose(mine, theirs, rtol=1e-3, atol=1e-7), name
    for name in set(old) - set(target):
        assert _term_weight(old[name]) is None


def test_the_projection_keeps_the_core_change_on_the_active_rows(tmp_path):
    from kilora.path_merge import _clone_state, _pair_core_moves, _project_mixes

    session = _three_experts(tmp_path)
    after, old_u, new_u = _project_inputs(session)
    hook = sorted(new_u)[0]
    target = {hook: [0, 1]}
    rank = int(after[hook]["core_0"].shape[0])
    mask = torch.zeros(rank, rank)
    mask[0, :] = 1.0

    def active(name, expert, size):
        assert name == hook and expert in (0, 1) and size == rank
        return mask if expert == 0 else None

    def project(rows):
        return _project_mixes(
            session,
            types.SimpleNamespace(),
            after,
            _clone_state(after),
            old_u,
            new_u,
            experts=target,
            core_moves=_pair_core_moves(session),
            active=rows,
        )

    free, kept = project(None), project(active)
    old0 = after[hook]["core_0"]
    assert torch.equal((kept[hook]["core_0"] - old0)[1:], torch.zeros_like(old0[1:]))
    assert torch.allclose(kept[hook]["core_0"][0], free[hook]["core_0"][0])
    assert torch.equal(kept[hook]["core_1"], free[hook]["core_1"])
    assert torch.equal(kept[hook]["strength_0"], free[hook]["strength_0"])


def test_the_merge_profile_counts_the_projection(tmp_path):
    session = _three_experts(tmp_path)
    before, after, keeps = _fold_inputs(session)
    hook = sorted(session.state)[0]
    profile = MergeProfile()
    fold_import_paths(
        session,
        {},
        types.SimpleNamespace(),
        before,
        after,
        keeps,
        target={hook: [0, 2]},
        profile=profile,
    )
    assert profile.project["hooks"] == 1
    assert 0 < profile.project["experts"] <= 2
    line = profile.line()
    assert " project_hooks=1 " in line and " project_experts=" in line
    assert " project_moves=" in line and " project_terms=" in line
    assert profile.project_seconds <= profile.fold_seconds


def test_the_merge_check_reuses_the_loss_of_the_write(tmp_path):
    from kilora.path_merge import merge_loss_check

    session = _three_experts(tmp_path)
    state = session.state
    merged = {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}
    calls = []

    def held(trial):
        calls.append(trial)
        return 0.25

    record = types.SimpleNamespace(measure=types.SimpleNamespace(held=held))
    editor = types.SimpleNamespace(_accepted={"after": 0.5, "state": state})
    dropped, note = merge_loss_check(editor, session, record, merged)
    assert not dropped and "held-out loss 0.5 before, 0.25 after" in note
    assert len(calls) == 1
    other = {t: {k: v + 1 for k, v in p.items()} for t, p in state.items()}
    editor = types.SimpleNamespace(_accepted={"after": 0.5, "state": other})
    calls.clear()
    dropped, note = merge_loss_check(editor, session, record, merged)
    assert len(calls) == 2 and dropped and "0.25 before, 0.25 after" in note
