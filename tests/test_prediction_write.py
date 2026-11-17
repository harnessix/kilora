# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_prediction_cross_step import _weights, imported

from kilora import editors
from kilora import prediction_write as pw
from kilora.editors import Editor, commit_write
from kilora.live import open_session
from kilora.recording import (
    CrossStepError,
    GenerationRecord,
    LevelError,
    PathStatistics,
    PredictionError,
    orthonormal_rows,
)
from kilora.testing import make_live_mod

R, RANK, OUT, K = 3, 4, 5, 2
TARGET = "blocks.0.mount"


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _toy(seed=0):
    g = _gen(seed)
    m = torch.randn(R, OUT, RANK, generator=g, dtype=torch.float64)

    def response(z):
        return torch.einsum("jor,...r->...oj", m, z.double())

    p_out = orthonormal_rows(torch.randn(OUT, RANK, generator=g))
    block = orthonormal_rows(torch.randn(OUT, K, generator=g))
    proj = torch.randn(RANK, OUT, generator=g, dtype=torch.float64)
    sigmas = (2.0, 1.0, 0.5)
    zs = [torch.randn(RANK, generator=g, dtype=torch.float64) for _ in sigmas]
    cross = {TARGET: []}
    for step in (1, 2):
        cross[TARGET].append(
            CrossStepError(
                step + 1,
                sigmas[step],
                0.5,
                zs[step],
                torch.zeros(RANK),
                torch.zeros(RANK),
                torch.randn(RANK, generator=g, dtype=torch.float64),
                torch.tensor(0.3 + step),
                0,
                zs[step - 1],
                sigmas[step - 1],
            )
        )
    depth = {
        TARGET: [
            PredictionError(
                0.8,
                torch.zeros(2, RANK),
                torch.zeros(2, RANK),
                torch.zeros(2, RANK),
                torch.randn(2, RANK, generator=g, dtype=torch.float64),
                torch.tensor(0.7),
                0,
                zs[1],
            )
        ]
    }

    def level(step, sigma):
        return LevelError(
            step,
            sigma,
            0.5,
            0,
            torch.zeros(K),
            torch.zeros(K),
            torch.randn(K, generator=g, dtype=torch.float64),
            torch.tensor(0.4),
        )

    hier = {"blocks.0": [level(2, sigmas[1]), level(3, sigmas[2])]}
    clamp = {"blocks.0": [level(3, sigmas[2])]}
    return SimpleToy(m, response, p_out, block, proj, cross, depth, hier, clamp, zs)


class SimpleToy:
    def __init__(self, m, response, p_out, block, proj, cross, depth, hier, clamp, zs):
        self.m, self.response, self.p_out, self.block, self.proj = m, response, p_out, block, proj
        self.cross, self.depth, self.hier, self.clamp, self.zs = cross, depth, hier, clamp, zs

    def terms(self):
        return pw.build_terms(
            self.cross,
            {(TARGET, 0): ("node", self.response)},
            {(TARGET, 0): self.p_out},
            depth=self.depth,
            depth_proj={TARGET: self.proj},
            hier=self.hier,
            clamp=self.clamp,
            block_frame=lambda target, width: self.block,
            block_of=lambda target: 0,
        )

    def hand(self):
        out = []
        z_at = {round(e.sigma, 9): e.z for e in self.cross[TARGET]}
        for e in self.cross[TARGET]:
            span = e.sigma_prev - e.sigma
            a = self.p_out.double() @ self.response((e.z - e.z_prev) / span)
            out.append(("cross", e.error, a, float(e.precision)))
        for e in self.depth[TARGET]:
            out.append(
                ("depth", e.error.mean(0), -(self.proj @ self.response(e.z)), float(e.precision))
            )
        for name, errors, sign in (("hier", self.hier, 1.0), ("clamp", self.clamp, -1.0)):
            for e in errors["blocks.0"]:
                z = z_at.get(round(e.sigma, 9))
                if z is None:
                    continue
                a = sign * (self.block.double() @ self.response(z))
                out.append((name, e.error, a, float(e.precision)))
        return out


def _prior(seed=9):
    g = _gen(seed)
    c = torch.randn(R, generator=g, dtype=torch.float64)
    mu = torch.randn(R, generator=g, dtype=torch.float64)
    root = torch.randn(R, R, generator=g, dtype=torch.float64)
    return pw.NodePrior(c, mu, root @ root.t() + 0.5 * torch.eye(R, dtype=torch.float64))


def _reference(hand, prior, weights):
    rows, rhs = [], []
    for source, error, a, pi in hand:
        w = (weights[source] * pi) ** 0.5
        if w == 0:
            continue
        rows.append(w * a)
        rhs.append(-w * error)
    chol = torch.linalg.cholesky(prior.inv_cov)
    rows.append(chol.t())
    rhs.append(-chol.t() @ (prior.c - prior.mu))
    a, b = torch.cat(rows), torch.cat(rhs)
    return torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)


@pytest.mark.parametrize("only", [*pw.SOURCES, None])
def test_the_write_is_the_closed_form_map(only):
    toy, prior = _toy(), _prior()
    weights = (
        dict(pw.DEFAULT_WEIGHTS)
        if only is None
        else {name: (1.3 if name == only else 0.0) for name in pw.SOURCES}
    )
    terms = toy.terms()
    hand = toy.hand()
    assert sorted(t.source for t in terms) == sorted(h[0] for h in hand)
    for term, (source, error, a, pi) in zip(terms, hand, strict=True):
        assert term.source == source
        assert torch.allclose(term.response.double(), a, atol=1e-12)
        assert torch.allclose(term.error.double(), error.double(), atol=1e-12)
        precision = torch.as_tensor(term.precision, dtype=torch.float64)
        assert torch.allclose(precision, torch.as_tensor(pi, dtype=torch.float64), atol=1e-12)
    result = pw.solve_map(terms, {"node": prior}, weights)
    expected = _reference(hand, prior, weights)
    assert torch.allclose(result.delta["node"], expected, atol=1e-9)
    dc = result.delta["node"]
    grad = prior.inv_cov @ (prior.c + dc - prior.mu)
    for source, error, a, pi in hand:
        grad = grad + weights[source] * pi * a.t() @ (error + a @ dc)
    assert grad.abs().max() < 1e-9
    assert result.counts.get("clamp", 0) == 1
    assert result.steps["cross"] == 2 and result.nodes["cross"] == 1
    note = result.note(0.5)
    assert "source=prediction" in note and "write_norm=0.5" in note and "solve_s=" in note
    assert "eps_hier_steps=2" in note and "pi_eps2_cross=" in note
    assert pw.NO_TARGET not in note


def test_without_a_target_the_report_says_so():
    toy = _toy()
    toy.clamp = {}
    result = pw.solve_map(toy.terms(), {"node": _prior()})
    assert not result.has_target and pw.NO_TARGET in result.note()
    assert result.delta["node"].abs().sum() > 0


def test_one_batched_solve_per_coefficient_count():
    priors = {i: _prior(i) for i in range(4)}
    result = pw.solve_map([], priors)
    for i, prior in priors.items():
        assert torch.allclose(result.delta[i], prior.mu - prior.c, atol=1e-10)


def test_the_settings_are_checked():
    with pytest.raises(ValueError):
        pw.check_learn("guess")
    with pytest.raises(ValueError):
        pw.check_weights({"other": 1.0})
    with pytest.raises(ValueError):
        pw.check_weights({"cross": -1.0})
    with pytest.raises(ValueError):
        pw.check_prior_scale(0.0)
    assert pw.check_weights({"cross": 2})["cross"] == 2.0


def _never(*args, **kwargs):
    raise AssertionError("record.measure must not run with learn = prediction")


@pytest.fixture
def chunk(tmp_path, monkeypatch):
    from test_chunk_live import _packs

    monkeypatch.delenv(editors.GUARD_ENV, raising=False)
    make_live_mod(tmp_path / "live.kmod")
    session = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    session.learning = True
    return session, _packs(session, tmp_path)


def _chunk_stats(editor, session, seed=3):
    g = _gen(seed)
    frames = editor._cross_frames(session)
    cross = {}
    for target, items in frames.items():
        rank = items[0].p_in.shape[0]
        z = [torch.randn(rank, generator=g, dtype=torch.float64) for _ in range(3)]
        cross[target] = [
            CrossStepError(
                step + 1,
                2.0 - step * 0.5,
                0.5,
                z[step],
                torch.zeros(rank),
                torch.zeros(rank),
                torch.randn(items[0].p_out.shape[0], generator=g, dtype=torch.float64),
                torch.tensor(1.0),
                0,
                z[step - 1],
                2.0 - (step - 1) * 0.5,
            )
            for step in (1, 2)
        ]
    return PathStatistics({}, {}, {}, cross_errors=cross)


def _chunk_write(session, path, record):
    from kilora.editors import editor_for

    editor = editor_for(path)
    editor.check(session)
    record.stats = _chunk_stats(editor, session)
    return editor, commit_write(session, editor, record)


def _ups(session):
    return {t: p["up"].clone() for t, p in session.state.items()}


def test_chunk_prediction_write_runs_no_measurement(chunk):
    session, path = chunk
    session.set_options(learn="prediction")
    before = _ups(session)
    calls = []

    def score(trial):
        calls.append(trial)
        return {"gap": 1.0, "spread": 1.0}

    editor, report = _chunk_write(
        session, path, GenerationRecord(measure=_never, guard_score=score)
    )
    note = editor.write_note()
    assert "source=prediction" in note and pw.NO_TARGET in note and "write_norm=" in note
    assert "Guard deferred to the next generation." in report
    assert not calls
    assert any(not torch.equal(session.state[t]["up"], before[t]) for t in before)
    assert session.deferred_guard is not None and len(session.commits) == 1
    assert session.editor_state.chunks.node_coef
    session.undo()
    assert all(torch.equal(session.state[t]["up"], before[t]) for t in before)
    assert session.deferred_guard is None


def test_chunk_measure_is_unchanged_by_the_prediction_settings(chunk, tmp_path):
    from test_chunk_live import _packs, _record

    session, path = chunk
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    make_live_mod(other_dir / "live.kmod")
    other = open_session(other_dir / "live.kmod", saves_dir=other_dir / "saves")
    other.learning = True
    other_path = _packs(other, other_dir)
    other.set_options(
        learn="measure", prediction_weights={"cross": 3.0, "clamp": 0.0}, prediction_prior=7.0
    )
    g = _gen(5)
    target = {
        t: torch.randn(int(p["up"].shape[-1]), 4, generator=g) for t, p in session.state.items()
    }
    from kilora.editors import editor_for

    for s, p in ((session, path), (other, other_path)):
        editor = editor_for(p)
        editor.check(s)
        report = commit_write(s, editor, _record(s, target))
        assert s.deferred_guard is None and not editor.provisional
        assert "source=prediction" not in report + editor.write_note()
    assert len(session.commits) == len(other.commits) == 1
    for t in session.state:
        for part in session.state[t]:
            assert torch.equal(session.state[t][part], other.state[t][part])


class Provisional(Editor):
    def next_state(self, state, record):
        self.provisional = True
        return {t: {**p, "up": p["up"] + 0.1} for t, p in state.items()}

    def guard_trial(self, state, new):
        return "trial"


class Quiet(Editor):
    def next_state(self, state, record):
        return None

    def guard_trial(self, state, new):
        return "trial"


@pytest.fixture
def plain(tmp_path, monkeypatch):
    monkeypatch.delenv(editors.GUARD_ENV, raising=False)
    make_live_mod(tmp_path / "live.kmod")
    s = open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")
    s.learning = True
    return s


def _scores(with_write, without_write):
    def score(trial):
        return {"gap": with_write if trial is None else without_write, "spread": 1.0}

    return score


def test_the_deferred_guard_rolls_back_a_failing_write(plain):
    before = _ups(plain)
    first = commit_write(plain, Provisional(), GenerationRecord(guard_score=_never))
    assert "Guard deferred" in first and plain.deferred_guard is not None
    assert not all(torch.equal(plain.state[t]["up"], before[t]) for t in before)
    report = commit_write(plain, Quiet(), GenerationRecord(guard_score=_scores(0.5, 1.0)))
    assert "The previous prediction write was rolled back" in report
    assert "prompt following" in report
    assert all(torch.equal(plain.state[t]["up"], before[t]) for t in before)
    assert plain.deferred_guard is None and len(plain.commits) == 0


def test_the_deferred_guard_keeps_a_passing_write(plain):
    commit_write(plain, Provisional(), GenerationRecord(guard_score=_never))
    after = _ups(plain)
    report = commit_write(plain, Quiet(), GenerationRecord(guard_score=_scores(1.0, 1.0)))
    assert "Guard of the previous prediction write: passed." in report
    assert all(torch.equal(plain.state[t]["up"], after[t]) for t in after)
    assert plain.deferred_guard is None and len(plain.commits) == 1


def test_the_deferred_guard_waits_for_a_generation_with_a_score(plain):
    commit_write(plain, Provisional(), GenerationRecord(guard_score=_never))
    commit_write(plain, Quiet(), GenerationRecord())
    assert plain.deferred_guard is not None


def _imported_stats(session, seed=4):
    from kilora.live_imports import import_source_frames

    g = _gen(seed)
    cross = {}
    for target, items in import_source_frames(session).items():
        for node, (_, p_in, p_out, _) in enumerate(items):
            rank = p_in.shape[0]
            z = [torch.randn(rank, generator=g, dtype=torch.float64) for _ in range(3)]
            cross.setdefault(target, []).extend(
                CrossStepError(
                    step + 1,
                    2.0 - step * 0.5,
                    0.5,
                    z[step],
                    torch.zeros(rank),
                    torch.zeros(rank),
                    torch.randn(p_out.shape[0], generator=g, dtype=torch.float64),
                    torch.tensor(1.0),
                    node,
                    z[step - 1],
                    2.0 - (step - 1) * 0.5,
                )
                for step in (1, 2)
            )
    return PathStatistics({}, {}, {}, cross_errors=cross)


def test_imported_prediction_write_moves_core_strength_and_terms(imported):
    from kilora.live_imports import ImportEditor, TermMove, import_source_frames

    session, model = imported
    session.align_for_moves()
    sources = import_source_frames(session, _weights(session, model))
    target = next(iter(sources))
    ex = session._source.experts[target]
    g = _gen(8)
    term = TermMove(
        f"{target}:term:o:i",
        target,
        "o",
        "i",
        torch.randn(ex.up.shape[-1], generator=g),
        torch.randn(ex.down.shape[-1], generator=g),
    )
    term.apply(session.state, 0.01)
    session.set_options(learn="prediction")
    before = {t: {k: v.clone() for k, v in parts.items()} for t, parts in session.state.items()}
    editor = ImportEditor(session)
    record = GenerationRecord(measure=_never, stats=_imported_stats(session))
    new = editor.next_state(session.state, record)
    assert new is not None and editor.provisional
    changed = {
        key.split("_")[0]
        for t, parts in new.items()
        for key, value in parts.items()
        if key in before[t] and not torch.equal(value, before[t][key])
    }
    assert {"core", "strength", "terms"} <= changed
    assert "source=prediction" in editor.note and pw.NO_TARGET in editor.note
    for t, parts in before.items():
        for key, value in parts.items():
            assert torch.equal(session.state[t][key], value)


def test_imported_measure_stays_the_measured_write(imported):
    from kilora.live_imports import ImportEditor

    session, _ = imported
    editor = ImportEditor(session)
    assert pw.session_learn(session) == "measure"
    assert editor.next_state(session.state, GenerationRecord()) is None
    assert not editor.provisional


def test_the_setting_survives_a_save_and_load(plain):
    plain.set_options(
        learn="prediction", prediction_weights={"clamp": 2.0, "hier": 0.0}, prediction_prior=3.0
    )
    name = plain.save_point("prediction")
    plain.set_options(learn="measure", prediction_weights={}, prediction_prior=1.0)
    assert plain.learn == "measure"
    plain.load(name)
    assert plain.learn == "prediction" and plain.prediction_prior == 3.0
    assert plain.prediction_weights["clamp"] == 2.0 and plain.prediction_weights["hier"] == 0.0
    assert plain.prediction_weights["cross"] == pw.DEFAULT_WEIGHTS["cross"]


def test_a_default_save_records_no_prediction_setting(plain):
    assert plain._saved_options() is None
