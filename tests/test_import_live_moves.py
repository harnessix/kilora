# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
import types

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_import_modes import LAYERS, Toy

from kilora import compose
from kilora.editors import commit_write, write_outcome
from kilora.live import SESSIONS
from kilora.live_imports import ImportEditor, imported_session_for, moves
from kilora.modfile import MANIFEST_KEY, compute_digest
from kilora.multi_import import import_loras
from kilora.recording import GenerationRecord

RANK = 16
LIVE = LAYERS[0]
ZERO = LAYERS[1]


@pytest.fixture(autouse=True)
def _fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _kohya(path, layers):
    sd = {}
    for name, (down, up) in layers.items():
        stem = "lora_unet_" + name.replace(".", "_")
        sd[f"{stem}.lora_down.weight"] = down.contiguous()
        sd[f"{stem}.lora_up.weight"] = up.contiguous()
    save_file(sd, str(path))


def _source(path):
    data = path.read_bytes()
    return dict(
        sha256=hashlib.sha256(data).hexdigest(),
        path=str(path),
        strength=1.0,
        axis="both",
    )


def _session(tmp_path, paths):
    model = Toy().eval()
    routed = import_loras([str(path) for path in paths], model).adapter_set
    shapes = {target: list(ex.down.shape) for target, ex in routed.experts.items()}
    sources = [_source(path) for path in paths]
    session = imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")
    return routed, session


def _render(sets, name, x):
    model = Toy().eval()
    handle = compose(model, sets, depth=1, mix="stack")
    try:
        with torch.no_grad():
            return model.get_submodule(name)(x).clone()
    finally:
        handle.remove()


def _plain(name, x, layers):
    base = Toy().eval()
    with torch.no_grad():
        out = base.get_submodule(name)(x)
        for down, up in layers:
            out = out + x @ down.T @ up.T
    return out


def _padded_rank1(model, name, *, live_row, seed):
    module = model.get_submodule(name)
    gen = torch.Generator().manual_seed(seed)
    down = torch.zeros(RANK, module.in_features)
    down[live_row] = torch.randn(module.in_features, generator=gen)
    up = torch.randn(module.out_features, RANK, generator=gen)
    return down, up


def _zeros(model, name):
    module = model.get_submodule(name)
    return torch.zeros(RANK, module.in_features), torch.zeros(module.out_features, RANK)


def _rewrite_core(path, key, core):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata()
        tensors = {name: handle.get_tensor(name).contiguous() for name in handle.keys()}
    tensors[key] = core.contiguous()
    manifest = json.loads(meta[MANIFEST_KEY])
    manifest["digest"] = compute_digest(manifest, tensors)
    meta[MANIFEST_KEY] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    save_file(tensors, str(path), metadata=meta)


def test_rank1_padded_to_16_moves_and_changes_the_layer(tmp_path):
    model = Toy().eval()
    down, up = _padded_rank1(model, LIVE, live_row=0, seed=1)
    path = tmp_path / "rank1.safetensors"
    _kohya(path, {LIVE: (down, up)})
    routed, session = _session(tmp_path, [path])
    assert routed.experts[LIVE].down.shape[2] == RANK
    assert routed.experts[LIVE].routing_down is None

    core_moves = [
        m for m in moves(session) if m.target == LIVE and m.expert == 0 and m.part == "core"
    ]
    expert = session._source.experts[LIVE]
    assert expert.ranks[0] == 1
    assert torch.count_nonzero(expert.down[0, 0, 1:]) == 0
    assert session.state[LIVE]["core_0"].shape == (1, 1)
    assert len(core_moves) == 1
    move = core_moves[0]
    assert move.direction.shape == (1, 1)
    assert float(move.direction.abs().sum()) > 0

    module = model.get_submodule(LIVE)
    x = torch.randn(2, 3, module.in_features, generator=torch.Generator().manual_seed(2))
    before = _render(session.sets(), LIVE, x)
    step = 0.3 / float(move.direction.abs().max().clamp_min(1e-8))
    move.apply(session.state, step)
    after = _render(session.sets(), LIVE, x)
    assert torch.allclose(before, _plain(LIVE, x, [(down, up)]), rtol=1e-5, atol=1e-5)
    assert float((after - before).norm()) > 1e-3


def test_all_zero_mount_has_no_moves(tmp_path):
    model = Toy().eval()
    live = _padded_rank1(model, LIVE, live_row=0, seed=3)
    path = tmp_path / "with_zero.safetensors"
    _kohya(path, {LIVE: live, ZERO: _zeros(model, ZERO)})
    _routed, session = _session(tmp_path, [path])
    moves(session)
    assert session._source.experts[ZERO].ranks[0] == 0
    assert "core_0" not in session.state[ZERO]
    assert "strength_0" not in session.state[ZERO]
    assert not [m for m in moves(session) if m.target == ZERO]

    other = tmp_path / "other.safetensors"
    second = _padded_rank1(model, LIVE, live_row=1, seed=4)
    absent = _padded_rank1(model, ZERO, live_row=0, seed=5)
    _kohya(tmp_path / "absent.safetensors", {ZERO: absent})
    _kohya(other, {LIVE: second})
    SESSIONS.clear()
    _routed, session = _session(tmp_path, [tmp_path / "absent.safetensors", other])
    moves(session)
    assert session._source.experts[LIVE].ranks[0] == 0
    assert not [m for m in moves(session) if m.target == LIVE and m.expert == 0]
    assert any(m.target == LIVE and m.expert == 1 and m.part == "core" for m in moves(session))


def test_zero_writes_match_the_source_product(tmp_path):
    model = Toy().eval()
    gen = torch.Generator().manual_seed(6)
    full_down = torch.randn(4, model.get_submodule(LIVE).in_features, generator=gen)
    full_up = torch.randn(model.get_submodule(LIVE).out_features, 4, generator=gen)
    thin = _padded_rank1(model, ZERO, live_row=2, seed=8)
    path = tmp_path / "both.safetensors"
    _kohya(path, {LIVE: (full_down, full_up), ZERO: thin})
    routed, session = _session(tmp_path, [path])
    files = {LIVE: (full_down, full_up), ZERO: thin}
    for name, (down, up) in files.items():
        expert = routed.experts[name]
        stored = expert.up[0, 0].T @ expert.down[0, 0]
        assert torch.allclose(stored, up @ down, rtol=1e-5, atol=1e-5)
        module = model.get_submodule(name)
        x = torch.randn(2, 4, module.in_features, generator=torch.Generator().manual_seed(9))
        got = _render(session.sets(), name, x)
        assert torch.allclose(got, _plain(name, x, [(down, up)]), rtol=1e-5, atol=1e-5)
    moves(session)
    for name in files:
        rendered = session.sets()[0].experts[name]
        assert torch.equal(rendered.down, routed.experts[name].down)
        assert torch.equal(rendered.up, routed.experts[name].up)
        assert rendered.routing_down is None


def test_old_edit_file_matches_the_source_core(tmp_path):
    model = Toy().eval()
    live_row = 7
    down, up = _padded_rank1(model, LIVE, live_row=live_row, seed=10)
    path = tmp_path / "old.safetensors"
    _kohya(path, {LIVE: (down, up)})
    _routed, session = _session(tmp_path, [path])
    moves(session)
    assert session.state[LIVE]["core_0"].shape == (1, 1)
    module = model.get_submodule(LIVE)
    x = torch.randn(2, 3, module.in_features, generator=torch.Generator().manual_seed(11))
    zero = _render(session.sets(), LIVE, x)
    assert torch.allclose(zero, _plain(LIVE, x, [(down, up)]), rtol=1e-5, atol=1e-5)

    session.state[LIVE]["core_0"][0, 0] = 0.7
    fresh = tmp_path / "fresh.kmod"
    session.save_edits(fresh)
    session.reset()
    session.load_edits(fresh)
    assert session.state[LIVE]["core_0"].shape == (1, 1)
    assert float(session.state[LIVE]["core_0"][0, 0]) == pytest.approx(0.7)

    edit = tmp_path / "legacy.kmod"
    session.reset()
    session.save_edits(edit)
    key = f"kilora/edit/{LIVE}/core_0"
    source_up = session._source.experts[LIVE]._source_up[0, 0]
    source_down = session._source.experts[LIVE]._source_down[0, 0]

    def expect(core):
        product = source_up.double().T @ core.double() @ source_down.double()
        base = Toy().eval()
        with torch.no_grad():
            plain = base.get_submodule(LIVE)(x)
        return plain + x @ product.float().T

    identity = torch.eye(RANK)
    _rewrite_core(edit, key, identity)
    session.load_edits(edit)
    assert session.state[LIVE]["core_0"].shape == (1, 1)
    assert torch.allclose(_render(session.sets(), LIVE, x), zero, rtol=1e-5, atol=1e-5)

    null = torch.eye(RANK)
    null[0, 0] = 1.4
    _rewrite_core(edit, key, null)
    session.load_edits(edit)
    assert torch.allclose(_render(session.sets(), LIVE, x), zero, rtol=1e-5, atol=1e-5)

    live = torch.eye(RANK)
    live[live_row, live_row] = 1.4
    _rewrite_core(edit, key, live)
    session.load_edits(edit)
    got = _render(session.sets(), LIVE, x)
    want = expect(live)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    assert float((got - zero).norm()) > 1e-3
    assert float(session.state[LIVE]["core_0"][0, 0]) == pytest.approx(1.4, abs=1e-4)

    SESSIONS.clear()
    _routed, again = _session(tmp_path, [path])
    assert LIVE not in again._aligned_keys
    again.load_edits(fresh)
    assert again.state[LIVE]["core_0"].shape == (1, 1)
    assert float(again.state[LIVE]["core_0"][0, 0]) == pytest.approx(0.7)


def test_skip_line_names_the_count():
    editor = object.__new__(ImportEditor)
    editor._note_moves = {}
    editor._full_moves = ""
    editor._response_skips = 2
    assert editor._moves_note() == " Skipped 2 moves below the response floor."
    editor._response_skips = 0
    assert editor._moves_note() == ""


def _routed_render(sets, name, x):
    model = Toy().eval()
    handle = compose(model, sets, depth=1)
    try:
        with torch.no_grad():
            return model.get_submodule(name)(x).clone()
    finally:
        handle.remove()


def test_routed_zero_writes_match_main_and_one_write_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    gen = torch.Generator().manual_seed(12)
    module = model.get_submodule(LIVE)
    dense = (
        torch.randn(4, module.in_features, generator=gen),
        torch.randn(module.out_features, 4, generator=gen),
    )
    first = tmp_path / "dense.safetensors"
    second = tmp_path / "thin.safetensors"
    _kohya(first, {LIVE: dense})
    _kohya(second, {LIVE: _padded_rank1(model, LIVE, live_row=5, seed=13)})
    routed, session = _session(tmp_path, [first, second])
    imported = routed.experts[LIVE]
    assert imported.routing_down is None

    x = torch.randn(2, 5, module.in_features, generator=torch.Generator().manual_seed(14))
    main = _routed_render([routed], LIVE, x)
    assert torch.allclose(_routed_render(session.sets(), LIVE, x), main, rtol=1e-5, atol=1e-5)

    written = [m for m in moves(session) if m.target == LIVE and m.expert == 1 and m.part == "core"]
    aligned = session._source.experts[LIVE]
    plain = session._plain[LIVE]
    assert aligned.ranks[1] == 1
    assert torch.equal(aligned.routing_down, imported.down)
    assert torch.equal(aligned.routing_up, imported.up)
    assert aligned.routing_source == plain.routing_source
    assert session.sets()[0].experts[LIVE] is plain
    zero = _routed_render(session.sets(), LIVE, x)
    assert torch.allclose(zero, main, rtol=1e-5, atol=1e-5)

    assert len(written) == 1
    move = written[0]
    move.apply(session.state, 0.5 / float(move.direction.abs().max().clamp_min(1e-8)))
    after = _routed_render(session.sets(), LIVE, x)
    assert float((after - zero).norm()) > 1e-4


def test_alignment_runs_once_per_expert_in_the_session(tmp_path, monkeypatch):
    from kilora import live_imports

    calls = []
    real = live_imports.align_rows

    def counted(up, down, *args, **kwargs):
        calls.append(1)
        return real(up, down, *args, **kwargs)

    monkeypatch.setattr(live_imports, "align_rows", counted)
    model = Toy().eval()
    gen = torch.Generator().manual_seed(15)
    module = model.get_submodule(LIVE)
    dense = (
        torch.randn(3, module.in_features, generator=gen),
        torch.randn(module.out_features, 3, generator=gen),
    )
    first = tmp_path / "a.safetensors"
    second = tmp_path / "b.safetensors"
    _kohya(first, {LIVE: dense})
    _kohya(second, {LIVE: _padded_rank1(model, LIVE, live_row=2, seed=16)})
    routed, session = _session(tmp_path, [first, second])
    x = torch.randn(2, 3, module.in_features, generator=torch.Generator().manual_seed(17))
    _render(session.sets(), LIVE, x)
    assert not calls
    pairs = sum(int(ex.down.shape[0]) for ex in routed.experts.values())
    moves(session)
    once = len(calls)
    assert 0 < once <= pairs
    moves(session)
    session.moves()
    editor = ImportEditor(session)
    editor.begin_write()
    assert len(calls) == once
    assert session.sets()[0].experts[LIVE] is session._plain[LIVE]


def test_write_outcome_is_logged(caplog):
    rule = types.SimpleNamespace(
        needs_path=False,
        _response_skips=3,
        _refused_note="",
        next_state=lambda state, record: None,
        write_note=lambda: "Skipped 3 moves below the response floor.",
    )
    session = types.SimpleNamespace(learning=True, state={})
    with caplog.at_level(logging.INFO, logger="kilora.editors"):
        commit_write(session, rule, GenerationRecord(images=1))
    lines = [r.getMessage() for r in caplog.records if r.name == "kilora.editors"]
    assert any("live write: no moves (3 skipped below the response floor)." in m for m in lines)

    rule._refused_note = " Not written (loss 1 before, 2 after the halved step)."
    assert write_outcome(rule, {"kind": "no_write"}).startswith(
        "live write: refused (Not written (loss 1 before"
    )
    assert write_outcome(rule, {"kind": "refused", "why": "gap 0.1"}).startswith(
        "live write: refused (gap 0.1)."
    )
    assert write_outcome(rule, {"kind": "committed"}).startswith("live write: committed.")
    assert write_outcome(rule, {}) == ""


def test_a_queued_imported_write_ends_with_an_outcome_line(tmp_path, monkeypatch, caplog):
    from test_import_edit_dimensions import _session as dimensions_session

    from kilora.comfyui.live_write import QueuedWrite
    from kilora.live_queue import WriteQueue

    session = dimensions_session(tmp_path)
    session.learning = True
    session.set_options(core_dims=0)
    with monkeypatch.context() as patch:
        patch.setattr(session, "align_for_moves", lambda: False)
        editor = ImportEditor(session)
        move = next(m for m in session.moves() if m.part == "strength")
    assert not session._aligned_keys
    session.moves = lambda: [move]
    target, expert = move.target, move.expert

    def loss(trial):
        delta = trial.du.get(target)
        source = session._source.experts[target].up[expert, 0].float()
        shift = (
            0.0
            if delta is None
            else float(
                (delta.reshape_as(session._source.experts[target].up)[expert, 0] * source).sum()
                / source.square().sum()
            )
        )
        return (float(session.state[target][f"strength_{expert}"]) + shift - 1.5) ** 2

    record = GenerationRecord(measure=loss, guard_score=lambda _: {"gap": 1.0, "spread": 1.0})
    before = float(session.state[target][f"strength_{expert}"])
    queue = WriteQueue()
    try:
        with caplog.at_level(logging.INFO):
            queue.submit(QueuedWrite(session, editor, record, None))
            assert queue.wait_idle(60)
    finally:
        queue.close()
    assert session._aligned_keys
    lines = [r.getMessage() for r in caplog.records if r.name == "kilora.editors"]
    assert any(m.startswith(("live write: committed.", "live write: refused (")) for m in lines)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    [(number, report)] = queue.reports
    assert number == 1 and "failed" not in report
    assert before < float(session.state[target][f"strength_{expert}"]) < 1.5


def test_a_failing_queued_write_is_logged_at_error_with_its_traceback(caplog):
    from kilora.live_queue import Preempted, WriteQueue

    def stopped():
        raise Preempted(0.0)

    def broken():
        commit_write(
            types.SimpleNamespace(learning=True, state={}),
            types.SimpleNamespace(
                needs_path=False,
                next_state=lambda state, record: 1 / 0,
            ),
            GenerationRecord(images=1),
        )

    queue = WriteQueue(max_pending=10)
    try:
        with caplog.at_level(logging.INFO):
            for run in (stopped, broken, lambda: "written"):
                queue.submit(run)
            assert queue.wait_idle(30)
    finally:
        queue.close()
    assert [n for n, _ in queue.reports] == [1, 2, 3]
    assert queue.reports[2][1] == "written"
    assert "failed" in queue.reports[0][1] and "failed" in queue.reports[1][1]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert all(r.exc_info for r in errors)
    messages = [r.getMessage() for r in errors]
    assert "live write 1 failed" in messages and "live write 2 failed" in messages
    assert any(m.startswith("live write: failed (ZeroDivisionError") for m in messages)
