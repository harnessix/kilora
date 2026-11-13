# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import kilora_editor_chunk.plugin as cl
import torch
from kilora_editor_chunk import Chunk, ChunkEditor, ChunkState
from test_chunk_plugin import _record
from test_chunk_plugin import setup as chunk_setup

from kilora.cross_layer import RLSLink, pass_back
from kilora.editors import commit_write
from kilora.path_graph import PathGraphCache
from kilora.recording import PathRecorder
from kilora.sets import KeyedExperts


def _standin():
    names = ["first.to_q", "second.to_q"]
    ed = ChunkEditor(
        names,
        {h: [4, 4] for h in names},
        {"to_q": 1.0},
        [Chunk("common", torch.eye(32), torch.ones(32))],
    )
    keys = torch.eye(2)[:, None, :]
    editor = object.__new__(cl.ChunkLive)
    editor.editor, editor.experts, editor.targets = ed, 2, names
    editor.live_set = SimpleNamespace(
        experts={h: KeyedExperts(keys_in=keys) for h in names},
        router_for=lambda _: torch.eye(4)[:2],
    )
    editor.v0 = {h: torch.eye(4) for h in names}
    editor.col_expert = {h: torch.tensor([0, 0, 1, 1]) for h in names}
    editor.path_graph = PathGraphCache()
    editor.options = {"path_angle": 0.0}
    editor.session = SimpleNamespace(
        editor_state=SimpleNamespace(
            links={},
            forest=SimpleNamespace(current="main", branches={"main": SimpleNamespace(paths=[])}),
        )
    )
    chunks = ChunkState.new(ed)
    chunks.coef["common"][0, :16] = 0.1 * torch.eye(4).reshape(-1)
    chunks.coef["common"][0, 16:] = 0.2 * torch.eye(4).reshape(-1)
    state = {h: {"down": torch.eye(4)} for h in names}
    return editor, chunks, state


def test_pair_recording_learns_map_and_off_is_identical():
    a = {name: torch.eye(2) for name in ("a", "b")}
    off = PathRecorder(a, a)
    on = PathRecorder(a, a, rows=[1], pair_targets=["a", "b"])
    control = PathRecorder(a, a, rows=[1])
    matrix = torch.tensor([[1.5, -0.2], [0.3, 0.8]])
    gen = torch.Generator().manual_seed(3)
    for _ in range(4):
        x = torch.randn(2, 400, 2, generator=gen)
        y = x @ matrix
        for recorder in (on, control):
            with recorder.recording():
                recorder.observe("a", x)
                recorder.observe("b", y)
    with off.recording():
        off.observe("a", x)
    assert not off.finish().pairs
    observed, baseline = on.finish(), control.finish()
    for name in a:
        assert torch.equal(observed.c_kk[name], baseline.c_kk[name])
        assert torch.equal(observed.c_vk[name], baseline.c_vk[name])
    pairs = observed.pairs[("a", "b")]
    assert len(pairs) == 4 and all(x.shape[0] == 256 for x, _ in pairs)
    link = RLSLink(2, 2)
    for x, y in pairs:
        link.update(x, y)
    torch.testing.assert_close(link.weights.float(), matrix, atol=1e-4, rtol=1e-4)


def test_cross_graph_and_passed_target(monkeypatch):
    editor, chunks, state = _standin()
    a, b = editor.targets
    link = RLSLink(4, 4)
    link.weights = torch.diag(torch.tensor([2.0, 2.0, 1.0, 1.0], dtype=torch.float64))
    link._error_weight = 1.0
    editor.session.editor_state.links[(a, b)] = link
    editor.options["cross_layer"] = "rls"
    seen = {}
    original = editor.path_graph.get

    def capture(version, layers, cross, versions):
        seen.update(cross)
        return original(version, layers, cross, versions)

    monkeypatch.setattr(editor.path_graph, "get", capture)
    original_u = chunks.expanded_u(editor.editor)
    editor._write_target = {h: torch.zeros_like(v) for h, v in original_u.items()}
    editor._write_target[b][0] = torch.eye(4)
    editor._path_merge(chunks, state, {a: [1.0, 0.0], b: [1.0, 0.0]}, {})
    torch.testing.assert_close(seen[(a, b)], torch.tensor([1.0, 0.5]))
    composite = original_u[b][0] @ torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])) @ original_u[a][0]
    expected = original_u[a][0] + composite + pass_back(link.weights.T, torch.eye(4))
    torch.testing.assert_close(
        chunks.expanded_u(editor.editor)[a][0], expected, atol=1e-5, rtol=1e-5
    )


def test_zero_precision_is_off():
    editor, chunks, state = _standin()
    base = chunks.snapshot()
    shares = {h: [1.0, 0.0] for h in editor.targets}
    editor._path_merge(chunks, state, shares, {})
    expected = chunks.expanded_u(editor.editor)
    chunks.restore(base)
    editor.options["cross_layer"] = "rls"
    link = RLSLink(4, 4)
    editor.session.editor_state.links[tuple(editor.targets)] = link
    editor._path_merge(chunks, state, shares, {})
    actual = chunks.expanded_u(editor.editor)
    assert all(torch.equal(expected[h], actual[h]) for h in expected)


def test_link_state_undo_and_load(chunk_setup):
    session, path, target = chunk_setup
    editor = cl.ChunkLive(str(path))
    editor.check(session)
    editor.configure({"cross_layer": "rls", "path_merge": False})
    names = editor.targets[:2]
    if len(names) < 2:
        names = ["a", "b"]
    from kilora.recording import PathStatistics

    record = _record(session, target)
    x = torch.eye(4)
    record.stats = PathStatistics({}, {}, {}, {tuple(names): [(x, x)]})
    result = commit_write(session, editor, record)
    assert "commits" in result
    saved = session.editor_state.state_tensors()
    name = session.save_point("links")
    assert session.editor_state.links
    session.undo()
    assert not session.editor_state.links
    session.load(name)
    assert session.editor_state.links
    session.editor_state.load_tensors(saved)
    torch.testing.assert_close(
        session.editor_state.links[tuple(names)].weights,
        RLSLink.from_tensors(
            {k.split("/")[-1]: v for k, v in saved.items() if k.startswith("links/0/")}
        ).weights,
    )


def test_accounting_for_single_and_combined_writes(chunk_setup, monkeypatch):
    session, path, target = chunk_setup
    editor = cl.ChunkLive(str(path))
    editor.check(session)
    editor.configure({"cross_layer": "rls"})

    def merge(chunks, state, shares, keeps, frozen_keys):
        family = editor.editor.chunks[0].family
        hook = editor.targets[0]
        amount = 0.002 if editor.options["cross_layer"] == "rls" else 0.001
        key = (family, hook)
        chunks.node_coef[key] = (
            chunks.node_coef.get(key, torch.zeros_like(chunks.coef[family])) + amount
        )
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    record = _record(session, target)
    report = commit_write(session, editor, record)
    assert "merge_equiv=" in report and "merge_equiv_off=" in report
    records = [_record(session, target), _record(session, target)]
    report = commit_write(session, editor, records[0], records=records)
    assert "merge_equiv=" in report and "merge_equiv_off=" in report


def test_identity_link_preserves_matching_target():
    editor, chunks, state = _standin()
    a, b = editor.targets
    chunks.coef["common"][0, 16:] = torch.eye(4).reshape(-1)
    initial = chunks.snapshot()
    editor._write_target = {
        h: torch.zeros_like(v) for h, v in chunks.expanded_u(editor.editor).items()
    }
    editor._write_target[b][0] = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    shares = {h: [1.0, 0.0] for h in editor.targets}
    editor._path_merge(chunks, state, shares, {})
    expected = chunks.expanded_u(editor.editor)
    chunks.restore(initial)
    link = RLSLink(4, 4)
    link.weights = torch.eye(4, dtype=torch.float64)
    link._error_weight = 1.0
    editor.session.editor_state.links[(a, b)] = link
    editor.options["cross_layer"] = "rls"
    editor._path_merge(chunks, state, shares, {})
    actual = chunks.expanded_u(editor.editor)
    for h in editor.targets:
        torch.testing.assert_close(actual[h], expected[h], atol=1e-6, rtol=1e-6)


def test_disabled_option_matches_default_commit(chunk_setup):
    session, path, target = chunk_setup
    editor = cl.ChunkLive(str(path))
    editor.check(session)
    assert not editor.recorder(session).pair_targets
    record = _record(session, target)
    commit_write(session, editor, record)
    expected = session.editor_state.state_tensors()
    expected_up = {h: parts["up"].clone() for h, parts in session.state.items()}
    while session.commits:
        session.undo()
    editor.configure({"cross_layer": "off"})
    assert not editor.recorder(session).pair_targets
    commit_write(session, editor, record)
    actual = session.editor_state.state_tensors()
    assert actual.keys() == expected.keys()
    assert all(
        torch.equal(
            actual[k].contiguous().view(torch.uint8), expected[k].contiguous().view(torch.uint8)
        )
        for k in actual
    )
    assert all(torch.equal(session.state[h]["up"], v) for h, v in expected_up.items())


def test_jvp_recording_and_cached_linear_response(monkeypatch):
    from kilora.cross_layer import sequential_links

    editor, _, state = _standin()
    a, b = editor.targets
    matrix = torch.tensor(
        [[1.5, -0.2, 0.1, 0.0], [0.3, 0.8, 0.0, 0.0], [0.0, 0.0, 0.5, 0.2], [0.0, 0.0, 0.1, 1.2]]
    )
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(4, 4, bias=False))
    model[1].weight.data.copy_(matrix.T)
    block = sequential_links(model, ["0", "1"])[("0", "1")]
    calls = []

    def counted(x):
        calls.append(1)
        return block(x)

    editor.session.state = state
    editor.session.cross_layer_blocks = {(a, b): counted}
    editor.options["cross_layer"] = "jvp"
    editor.weight_version = 0
    recorder = PathRecorder(
        {h: torch.eye(4) for h in (a, b)},
        editor.v0,
        rows=[1],
        pair_targets=[a, b],
        capture_pair_inputs=True,
    )
    x = torch.randn(2, 400, 4)
    with recorder.recording():
        recorder.observe(a, x)
        recorder.observe(b, block(x))
    stats = recorder.finish()
    assert stats.pair_inputs[(a, b)][0][0].shape == (256, 4)
    editor._update_links(SimpleNamespace(stats=stats))
    link = editor._cross_link((a, b))
    torch.testing.assert_close(link.weights, matrix)
    torch.testing.assert_close(
        link.weights,
        editor.session.editor_state.links[(a, b)].weights.float(),
        atol=1e-4,
        rtol=1e-4,
    )
    assert link.residual_variance < 1e-10
    count = len(calls)
    assert editor._cross_link((a, b)) is link and len(calls) == count
    editor.weight_version = getattr(editor, "weight_version", 0) + 1
    editor._cross_link((a, b))
    assert len(calls) > count


def test_jvp_falls_back_for_unisolated_pair(caplog):
    caplog.set_level("INFO")
    editor, _, state = _standin()
    a, b = editor.targets
    editor.session.state = state
    link = RLSLink(4, 4)
    editor.session.editor_state.links[(a, b)] = link
    editor.options["cross_layer"] = "jvp"
    assert editor._cross_link((a, b)) is link
    assert editor._cross_link((a, b)) is link
    assert caplog.text.count("uses RLS") == 1


def test_jvp_commit_accounting(chunk_setup, monkeypatch):
    session, path, target = chunk_setup
    editor = cl.ChunkLive(str(path))
    editor.check(session)
    editor.configure({"cross_layer": "jvp"})

    def merge(chunks, state, shares, keeps, frozen_keys):
        family, hook = editor.editor.chunks[0].family, editor.targets[0]
        amount = 0.002 if editor.options["cross_layer"] == "jvp" else 0.001
        key = (family, hook)
        chunks.node_coef[key] = (
            chunks.node_coef.get(key, torch.zeros_like(chunks.coef[family])) + amount
        )
        editor.path_report = (1, 0.0, 0.0)

    monkeypatch.setattr(editor, "_path_merge", merge)
    report = commit_write(session, editor, _record(session, target))
    assert "merge_equiv=" in report and "merge_equiv_off=" in report
    for phase in ("response", "link_update", "merge_accounting", "other"):
        assert f"cross_layer_ms:{phase}=" in report
    editor.configure({"cross_layer": "off"})
    report = commit_write(session, editor, _record(session, target))
    assert "cross_layer_ms:" not in report


def test_jvp_projection_cache_two_pairs():
    from kilora.cross_layer import measured_link

    torch.manual_seed(42)
    editor, _, state = _standin()
    a, b = editor.targets
    c = "third"
    editor.v0[c] = editor.v0[b].clone()
    state[c] = {key: value.clone() for key, value in state[b].items()}
    editor.session.state = state
    editor.options["cross_layer"] = "jvp"
    blocks = {(a, b): torch.nn.Tanh(), (b, c): torch.nn.Sigmoid()}
    editor.session.cross_layer_blocks = blocks
    x = torch.randn(5, 4)
    editor._jvp_samples = {pair: [(x, block(x))] for pair, block in blocks.items()}
    for pair, block in blocks.items():
        bases = [editor.v0[h].float() @ cl._down(state[h]).float() for h in pair]
        expected, variance = measured_link(block, editor._jvp_samples[pair], *bases)
        actual = editor._cross_link(pair)
        assert torch.equal(actual.weights, expected.T)
        assert actual.residual_variance == variance
    cached = editor._jvp_bases[b][1]
    editor.weight_version = 1
    for pair in blocks:
        editor._cross_link(pair)
    assert editor._jvp_bases[b][1] is cached


def test_jvp_variance_and_unsafe_replay():
    from kilora.cross_layer import measured_link, sequential_links

    x = torch.tensor([[-2.0, 1.0], [1.0, -2.0]])
    basis = torch.eye(2)
    response, variance = measured_link(torch.relu, [(x, torch.relu(x))], basis, basis)
    torch.testing.assert_close(response, basis * 0.5)
    assert variance > 0
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Softmax(dim=0))
    assert sequential_links(model, ["0", "1"]) == {}


def test_jvp_response_once_per_pair_and_timed(monkeypatch):
    from kilora import cross_layer

    torch.manual_seed(7)
    editor, _, state = _standin()
    a, b = editor.targets
    c = "third"
    editor.v0[c] = editor.v0[b].clone()
    state[c] = {key: value.clone() for key, value in state[b].items()}
    editor.session.state = state
    editor.options["cross_layer"] = "jvp"
    blocks = {(a, b): torch.nn.Tanh(), (b, c): torch.nn.Sigmoid()}
    editor.session.cross_layer_blocks = blocks
    x = torch.randn(5, 4)
    editor._jvp_samples = {pair: [(x, block(x))] for pair, block in blocks.items()}
    calls = []
    real = cross_layer.measured_link

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(cross_layer, "measured_link", counted)
    editor._cross_layer_times = {
        "response": 0.0,
        "link_update": 0.0,
        "merge_accounting": 0.0,
        "other": 0.0,
    }
    for _ in range(2):
        for pair in blocks:
            assert editor._cross_link(pair) is not None
    assert len(calls) == len(blocks)
    assert editor._cross_layer_times["response"] > 0.0
    assert all(
        value == 0.0 for phase, value in editor._cross_layer_times.items() if phase != "response"
    )
    editor.weight_version = getattr(editor, "weight_version", 0) + 1
    for pair in blocks:
        editor._cross_link(pair)
    assert len(calls) == 2 * len(blocks)


def test_cross_layer_off_does_not_capture_pairs(chunk_setup, tmp_path):
    from test_chunk_plugin import _frac_file

    session, path, _ = chunk_setup
    editor = _frac_file(session, path, tmp_path, write_frac=0.05, cap_frac=0.12)
    editor.configure({"cross_layer": "off"})
    recorder = editor.recorder(session)
    with recorder.recording():
        for target in editor.targets:
            recorder.observe(target, torch.randn(2, recorder.a[target].shape[-1]))
    assert not recorder.pairs and not recorder.pair_inputs
    assert not recorder._call and not recorder._raw_call
    stats = recorder.finish()
    assert not stats.pairs and not stats.pair_inputs
