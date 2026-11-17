# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from test_image_terms import MOUNTS, RANK, Model, _latent, _setup

from kilora import image_terms as it


class Move:
    def __init__(self, target, index):
        self.target, self.index = target, index
        self.id = f"{target}:{index}"

    def apply(self, new, step):
        c = new[self.target]["c"].clone()
        c[self.index] += step
        new[self.target] = {**new[self.target], "c": c}


class Editor:
    def __init__(self):
        self.frames, self.responses, self.p_out, self.priors = _setup()
        self.note = ""

    def prediction_setup(self, state):
        by_target = {name: [Move(name, k) for k in range(RANK)] for name in MOUNTS}
        return by_target, self.responses, self.p_out, self.priors

    def recorder(self, session):
        return SimpleNamespace(frames=self.frames)

    def apply_prediction(self, state, by_target, result):
        new = {t: dict(parts) for t, parts in state.items()}
        written, total = {}, 0.0
        for target, delta in result.delta.items():
            for move, step in zip(by_target[target], delta.tolist(), strict=True):
                if step:
                    move.apply(new, step)
                    written[move.id] = (move, step)
                    total += step * step
        return (new, written, total**0.5) if written else None

    def guard_trial(self, state, new):
        return ("trial", new)


class Forest:
    def __init__(self):
        self.branches, self.current, self.serial = ["main"], "main", 0

    def fork(self):
        self.serial += 1
        self.branches.append(f"branch-{self.serial}")
        return self.branches[-1]

    def checkout(self, branch):
        self.current = branch


class Session:
    def __init__(self):
        self.state = {name: {"c": torch.zeros(RANK, dtype=torch.float64)} for name in MOUNTS}
        self.editor_state = Forest()
        self.model = Model()
        self.revision, self.commits = 0, []
        self.pending = self._pending_info = None

    def editor_snapshot(self):
        return None

    def commit(self, editor_before=None):
        self.state = self.pending
        self.commits.append(self._pending_info.message)
        self.pending = self._pending_info = None
        return "Saved."


COUNT = 6
STUDENT = _latent(10)
SHARED = _latent(11)


def _items():
    return [
        SimpleNamespace(index=i, latent=_latent(20 + i), caption=f"image {i}") for i in range(COUNT)
    ]


class Recorder:
    def __init__(self, gain=0.5, drift=0.0):
        self.captured, self.measured = [], []
        self.gain, self.drift_value = gain, drift

    def samples(self, items, cue, settings):
        self.captured += [item.index for item in items]
        return [
            SimpleNamespace(
                target={"hidden_states": STUDENT + SHARED + 0.01 * item.latent},
                student={"hidden_states": STUDENT.clone()},
            )
            for item in items
        ]

    def losses(self, items, cue, settings, trial):
        self.measured.append(([item.index for item in items], trial is not None))
        return [1.0 if trial is None else self.gain for _ in items]

    def drift(self, trial):
        return self.drift_value


def _learn(session, recorder, **kwargs):
    return it.learn_from_images(
        session,
        _items(),
        "style",
        None,
        samples=recorder.samples,
        losses=recorder.losses,
        drift=recorder.drift,
        editor=Editor(),
        model=session.model,
        **kwargs,
    )


def test_holdout_takes_every_fifth_image():
    assert it.holdout_indices(3) == [2]
    assert it.holdout_indices(6) == [4]
    assert it.holdout_indices(8) == [4, 7]
    assert it.holdout_indices(10) == [4, 9]
    with pytest.raises(ValueError, match="at least 3 images"):
        it.holdout_indices(2)


def test_improving_set_passes_and_adds_one_branch_with_source_images():
    session, recorder = Session(), Recorder(gain=0.5)
    report = _learn(session, recorder)
    assert report.passed and report.committed and not report.warning
    assert report.held_out == [4] and report.terms > 0
    assert len(session.commits) == 1 and "source=images" in session.commits[0]
    assert session.editor_state.branches == ["main", "branch-1"]
    assert session.editor_state.current == report.branch == "branch-1"
    assert any(float(parts["c"].abs().sum()) > 0 for parts in session.state.values())
    assert report.relative_gain == pytest.approx(0.5)
    assert report.equivalent_writes == pytest.approx(1.0)
    text = report.text()
    assert "relative gain 50.00%" in text and it.RIGHTS_NOTE in text
    assert session.write_norms == [pytest.approx(report.norm)]


def test_worse_held_out_loss_commits_nothing():
    session, recorder = Session(), Recorder(gain=1.5)
    before = {t: parts["c"].clone() for t, parts in session.state.items()}
    report = _learn(session, recorder)
    assert not report.passed and not report.committed
    assert session.commits == [] and session.editor_state.branches == ["main"]
    for target, parts in session.state.items():
        assert torch.equal(parts["c"], before[target])
    assert report.with_change == 1.5 and report.without == 1.0
    assert "Guard failed" in report.text() and "Not committed." in report.text()


def test_ignore_guard_commits_and_warns():
    session, recorder = Session(), Recorder(gain=1.5)
    report = _learn(session, recorder, ignore_guard=True)
    assert not report.passed and report.committed
    assert "Warning" in report.warning and report.warning in report.text()
    assert len(session.commits) == 1
    assert "source=images" in session.commits[0] and "guard=ignored" in session.commits[0]


def test_drift_over_the_bound_fails_the_guard():
    session, recorder = Session(), Recorder(gain=0.5, drift=0.5)
    report = _learn(session, recorder, drift_bound=0.15)
    assert not report.passed and not report.committed and session.commits == []
    assert "drift" in report.message


def test_held_out_images_contribute_no_terms(monkeypatch):
    seen = []
    original = it.image_terms

    def spy(session, samples, **kwargs):
        terms = original(session, samples, **kwargs)
        seen.extend(terms)
        return terms

    monkeypatch.setattr(it, "image_terms", spy)
    session, recorder = Session(), Recorder()
    report = _learn(session, recorder)
    assert recorder.captured == [0, 1, 2, 3, 5]
    assert seen and {term.step[0] for term in seen} == {0, 1, 2, 3, 5}
    assert all(term.source == "images" for term in seen)
    assert report.fitted == [0, 1, 2, 3, 5]
    assert recorder.measured == [([4], False), ([4], True)]


def test_node_is_registered_with_a_live_display_name(monkeypatch):
    from kilora.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from kilora.comfyui import help as help_texts
    from kilora.comfyui.nodes import FOLDERS

    key = "KiLoRALiveLearnFromImages"
    assert key in NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS[key].startswith("KiLoRA Live: ")
    cls = NODE_CLASS_MAPPINGS[key]
    assert cls.CATEGORY == "KiLoRA/live mode"
    assert it.RIGHTS_NOTE in cls.DESCRIPTION
    inputs = cls.INPUT_TYPES()
    assert list(inputs["required"]) == ["path", "vae", "cue", "ignore_guard"]
    assert list(inputs["optional"])[:4] == ["images", "captions", "folder", "clip"]
    monkeypatch.setattr(FOLDERS, "list", lambda *a, **k: [])
    node = help_texts.node_defs("en", NODE_CLASS_MAPPINGS)[key]
    assert it.RIGHTS_NOTE in node["description"]
    assert "responsible for the rights" in node["inputs"]["images"]["tooltip"]


def test_node_needs_an_image_source():
    from kilora.comfyui.nodes import KiLoRALiveLearnFromImages

    with pytest.raises(ValueError, match="either images or folder"):
        KiLoRALiveLearnFromImages().learn(SimpleNamespace(), None, "style")


def test_image_learning_solves_terms_without_a_per_write_direction_estimate(monkeypatch):
    from kilora import prediction_write, write_estimates

    def unexpected(*_args, **_kwargs):
        raise AssertionError("image learning has no per-write gradient estimate")

    monkeypatch.setattr(write_estimates, "accumulate_write_estimate", unexpected)
    solves = []
    solve = prediction_write.solve_map

    def capture(terms, priors, weights):
        solves.append(len(terms))
        return solve(terms, priors, weights)

    monkeypatch.setattr(prediction_write, "solve_map", capture)
    session, recorder = Session(), Recorder(gain=2.0)
    for _ in range(2):
        report = _learn(session, recorder)
        assert not report.committed
    assert len(solves) == 2 and min(solves) > 0
    assert not hasattr(session, "_write_estimates")
    assert session.commits == []
