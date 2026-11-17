# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_edit_dimensions import _session as _imported

from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.comfyui.live_write import (
    PathHandle,
    batch_conditions,
    flow_pairs,
    flow_seed,
    image_loss,
    timing_note,
)
from kilora.errors import KiLoRAError
from kilora.live import SESSIONS, open_session, read_save
from kilora.recording import Perturbation
from kilora.regions import masked_mean
from kilora.testing import make_live_mod

TIMESTEPS = [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]
EVENT, RETRY = 2, 1


@pytest.fixture(autouse=True)
def _clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _in(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return _imported(folder)


def _plain(tmp_path):
    make_live_mod(tmp_path / "live.kmod")
    return open_session(tmp_path / "live.kmod", saves_dir=tmp_path / "saves")


def _model(seen):
    def apply_model(x_t, t, c_crossattn):
        seen.append((x_t.clone(), t.clone()))
        rows = c_crossattn.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))
        return 0.5 * x_t + 0.1 * t.reshape(-1, *([1] * (x_t.dim() - 1))) + rows

    return apply_model


def _handle(session, seen):
    return PathHandle(
        session=session,
        rule=None,
        recorder=None,
        timesteps=list(TIMESTEPS),
        conditioning={"c_crossattn": torch.full((1, 2), 0.3)},
        empty_conditioning={"c_crossattn": torch.full((1, 2), -0.2)},
        apply_model=_model(seen),
    )


def _x0():
    return torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(11))


def _pairs(session, count):
    return flow_pairs(None, (1, 5, 4), count, count, flow_seed(session, RETRY))


def _noised(x0, sigma, noise):
    return sigma * noise + (1 - sigma) * x0


def _loss(x0, x, sigma):
    joined = batch_conditions(
        {"c_crossattn": torch.full((1, 2), 0.3)}, {"c_crossattn": torch.full((1, 2), -0.2)}, 1
    )
    t = torch.full((1,), sigma, dtype=x0.dtype)
    outs = _model([])(torch.cat([x, x]), torch.cat([t, t]), **joined).chunk(2)
    losses = [float(masked_mean((o.float() - x0.float()).pow(2), None)) for o in outs]
    return sum(losses) / len(losses) / sigma**2, outs


def _gap(x0, outs):
    loss_c = float(masked_mean((outs[0].float() - x0.float()).pow(2), None))
    loss_e = float(masked_mean((outs[1].float() - x0.float()).pow(2), None))
    return loss_e - loss_c


def _run(session):
    seen = []
    measure = image_loss(_handle(session, seen), _x0(), EVENT, retry=RETRY)
    base = measure(None)
    trials = [measure(Perturbation({}, {})) for _ in range(2)]
    guard = (measure.guard_score(None), measure.guard_score(Perturbation({}, {})))
    return measure, base, trials, guard, seen


@pytest.mark.parametrize("kind", ["imported", "plain"])
def test_one_pair_measures_the_velocity_loss(tmp_path, kind):
    session = _imported(tmp_path) if kind == "imported" else _plain(tmp_path)
    if kind == "imported":
        assert session.measure_samples == 4
        session.set_options(measure_samples=1)
    assert session.measure_samples == 1
    measure, base, trials, guard, seen = _run(session)
    x0 = _x0()
    (pair,), (held,) = _pairs(session, 1)
    sigma, noise = pair
    x_t = _noised(x0, sigma, noise)
    loss, outs = _loss(x0, x_t, sigma)
    assert base == pytest.approx(loss, rel=1e-6)
    assert trials == pytest.approx([loss, loss], rel=1e-6)
    for x, t in seen[:3]:
        assert torch.allclose(x, torch.cat([x_t, x_t]))
        assert torch.equal(t, torch.full((2,), sigma))
    _, alt_outs = _loss(x0, _noised(x0, *held), held[0])
    expected = {"gap": _gap(x0, outs), "spread": float(outs[0].float().std())}
    expected["gap_alt"] = _gap(x0, alt_outs)
    for score in guard:
        assert score == pytest.approx(expected, rel=1e-6)
    assert measure.calls[0] == 3 + 1 + 2
    timing = measure.timing
    assert len(timing["base_s"]) == 1 and len(timing["trial_s"]) == 2
    assert measure.samples == [(sigma, flow_seed(session, RETRY))]
    assert "measure_samples" not in timing_note(type("R", (), {"measure": measure})())


def test_four_pairs_are_the_same_for_every_trial(tmp_path):
    session = _imported(tmp_path)
    assert session.measure_samples == 4
    measure, base, trials, guard, seen = _run(session)
    x0 = _x0()
    pairs, held = _pairs(session, 4)
    assert [s for s, _seed in measure.samples] == [s for s, _n in pairs]
    assert [s for s, _seed in measure.held_samples] == [s for s, _n in held]
    first, second = seen[1:5], seen[5:9]
    for (sigma, noise), (x, t), (x2, t2) in zip(pairs, first, second, strict=True):
        expected = _noised(x0, sigma, noise)
        assert torch.allclose(x, torch.cat([expected, expected]))
        assert torch.equal(t, torch.full((2,), sigma))
        assert torch.equal(x, x2) and torch.equal(t, t2)
    per_pair = [_loss(x0, _noised(x0, s, n), s)[0] for s, n in pairs]
    assert measure.sample_losses == pytest.approx(per_pair, rel=1e-6)
    assert trials[0] == pytest.approx(sum(per_pair) / 4, rel=1e-6)
    assert trials[1] == trials[0]
    assert base == pytest.approx(per_pair[0], rel=1e-6)
    timing = measure.timing
    assert len(timing["base_s"]) == 1 and len(timing["trial_s"]) == 2
    assert measure.calls[0] == 1 + 2 * 4 + 1 + 2
    note = timing_note(type("R", (), {"measure": measure})())
    assert "measure_samples=4" in note


def test_the_guard_reads_the_first_pair_and_the_first_held_out_pair(tmp_path):
    session = _imported(tmp_path)
    measure, _base, _trials, guard, _seen = _run(session)
    x0 = _x0()
    pairs, held = _pairs(session, 4)
    _, outs = _loss(x0, _noised(x0, *pairs[0]), pairs[0][0])
    _, alt = _loss(x0, _noised(x0, *held[0]), held[0][0])
    assert guard[0]["gap"] == pytest.approx(_gap(x0, outs), rel=1e-6)
    assert guard[0]["gap_alt"] == pytest.approx(_gap(x0, alt), rel=1e-6)


def test_the_setting_is_checked_and_survives_a_save(tmp_path):
    imported = _in(tmp_path / "a")
    for bad in (0, -1, True, 2.0):
        with pytest.raises(KiLoRAError, match="measure_samples"):
            imported.set_options(measure_samples=bad)
    assert imported.measure_samples == 4
    imported.set_options(measure_samples=2)
    name = imported.save_point("two")
    imported.set_options(measure_samples=6)
    imported.load(name)
    assert imported.measure_samples == 2
    imported.set_options(measure_samples=4)
    old = imported.save_point("default")
    manifest = read_save(tmp_path / "a" / "saves" / imported.mod_id / f"{old}.ksave").manifest
    assert "options" not in manifest
    imported.set_options(measure_samples=7)
    imported.load(old)
    assert imported.measure_samples == 4

    SESSIONS.clear()
    plain = _plain(tmp_path)
    assert plain.measure_samples == 1
    with pytest.raises(KiLoRAError, match="measure_samples"):
        plain.set_options(measure_samples=0)
    plain.set_options(measure_samples=3)
    name = plain.save_point("three")
    plain.set_options(measure_samples=1)
    plain.load(name)
    assert plain.measure_samples == 3
    plain.set_options(measure_samples=1)
    old = plain.save_point("default")
    plain.set_options(measure_samples=5)
    plain.load(old)
    assert plain.measure_samples == 1


def test_the_options_node_sets_the_samples_of_imported_sessions(tmp_path, monkeypatch):
    session = _in(tmp_path / "a")
    monkeypatch.setattr(patching, "sessions_of", lambda model: [session])
    monkeypatch.setattr(patching, "reapply", lambda model: (model, ""))
    types = N.KiLoRALiveOptions.INPUT_TYPES()
    order = list(types["optional"])
    assert order[-2:] == ["measure_samples", "learn"]
    assert types["optional"]["measure_samples"][0] == "INT"
    assert types["optional"]["measure_samples"][1]["default"] == 0
    N.KiLoRALiveOptions().set(object(), session.mod_id, False, 0.0, measure_samples=0)
    assert session.measure_samples == 4
    N.KiLoRALiveOptions().set(object(), session.mod_id, False, 0.0, measure_samples=8)
    assert session.measure_samples == 8
    N.KiLoRALiveOptions().set(object(), session.mod_id, False, 0.0)
    assert session.measure_samples == 8

    plain = _plain(tmp_path)
    monkeypatch.setattr(patching, "sessions_of", lambda model: [plain])
    with pytest.raises(KiLoRAError) as measure_error:
        N.KiLoRALiveOptions().set(object(), plain.mod_id, False, 0.0, measure="full")
    with pytest.raises(KiLoRAError) as samples_error:
        N.KiLoRALiveOptions().set(object(), plain.mod_id, False, 0.0, measure_samples=2)
    assert str(samples_error.value) == str(measure_error.value)
    assert plain.measure_samples == 1
