# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import logging

import pytest
import torch

from kilora import Settings, compose, compute, fused, load_mod, save_mod
from kilora.bake import ScopedSet, plan_targets
from kilora.learned_cross import cross
from kilora.routing_features import RoutingFeatures
from kilora.sections import Section
from kilora.sets import new_keyed_set


@pytest.fixture(autouse=True)
def _keys_arm(monkeypatch):
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")


def _set():
    s = new_keyed_set(
        "s",
        {"layer": (4, 4)},
        num_experts=2,
        routes=1,
        rank=1,
        router_dim=4,
        router_seed=0,
        seed=0,
    )
    ex = s.experts["layer"]
    ex.keys_in[:] = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]])
    ex.keys_out[:] = ex.keys_in
    ex.down.zero_()
    ex.down[0, 0, 0, 0] = 1
    ex.down[1, 0, 0, 1] = 1
    ex.up[0] = 1
    ex.up[1] = -1
    s.routers[4] = torch.eye(4)
    s.log_tau = 0.0
    return s


def _section(active=True, width=4):
    down = torch.ones(1, width)
    up = torch.zeros(width, 1)
    if active:
        up[0, 0] = 1
    return Section("learned-cross", "cross", {"down": down, "up": up, "bias": torch.zeros(width)})


def _model():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.layer.weight.copy_(torch.eye(4))
    return model


def test_tensor_persistence_and_layer_output(tmp_path):
    section = _section()
    path = tmp_path / "cross.kmod"
    manifest = save_mod(path, [_set()], Settings(depth=1), sections=[section])
    assert "section.learned-cross" in manifest["requires_features"]
    mod = load_mod(path)
    (back,) = mod.sections
    for key in section.data:
        assert torch.equal(back.data[key], section.data[key])
    x = torch.ones(1, 2, 4)
    d = back.data
    torch.testing.assert_close(cross(x, x, d), x * (x @ d["down"].T @ d["up"].T + d["bias"]) + x)
    model = _model()
    adapter = compose(model, [path])
    with torch.no_grad():
        first = model.layer(x)
    adapter.remove()
    path2 = tmp_path / "roundtrip.kmod"
    save_mod(path2, mod.sets, mod.settings, sections=mod.sections)
    compose(model, [path2])
    with torch.no_grad():
        assert torch.equal(first, model.layer(x))


def test_imported_and_live_sets_are_refused(tmp_path):
    s = _set()
    for invalid, fields in [
        (dataclasses.replace(s, imported_from={}), {}),
        (dataclasses.replace(s, extra={"live": True}), {}),
        (s, {"live": {"enabled": True}}),
    ]:
        with pytest.raises(ValueError, match="static, non-imported"):
            save_mod(tmp_path / "bad.kmod", [invalid], Settings(), sections=[_section()], **fields)


def test_invalid_shapes_and_width(tmp_path):
    section = _section()
    section.data["up"] = torch.ones(4, 2)
    with pytest.raises(ValueError, match="shapes"):
        save_mod(tmp_path / "bad.kmod", [_set()], Settings(), sections=[section])
    save_mod(tmp_path / "fixed-width.kmod", [_set()], Settings(), sections=[_section(width=3)])
    from unittest.mock import patch

    import kilora.modfile as modfile

    path = tmp_path / "valid.kmod"
    save_mod(path, [_set()], Settings(), sections=[_section()])
    real = modfile._unpack_sets

    def wrong_width(*args):
        sets, settings = real(*args)
        sets[0].routers[4] = torch.eye(3, 4)
        return sets, settings

    with patch.object(modfile, "_unpack_sets", wrong_width):
        assert load_mod(path).sections[0].kind == "learned-cross"


def test_zero_and_planted_cross_route_real_layer(tmp_path, monkeypatch):
    x = torch.ones(1, 2, 4)
    model = _model()
    base_path, zero_path, active_path = [tmp_path / f"{n}.kmod" for n in ("base", "zero", "active")]
    save_mod(base_path, [_set()], Settings(depth=1))
    save_mod(zero_path, [_set()], Settings(depth=1), sections=[_section(False)])
    save_mod(active_path, [_set()], Settings(depth=1), sections=[_section()])
    weights = []
    real = compute._gates_from_logits

    def record(*args, **kwargs):
        result = real(*args, **kwargs)
        weights.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(compute, "_gates_from_logits", record)
    outputs = []
    for path in (base_path, zero_path, active_path):
        adapter = compose(model, [path])
        with torch.no_grad():
            outputs.append(model.layer(x).clone())
        adapter.remove()
    assert torch.equal(outputs[0], outputs[1])
    assert weights[-1][..., 0].mean() > weights[-2][..., 0].mean()
    assert not torch.equal(outputs[0], outputs[2])


def test_plan_and_fused_fallback(tmp_path, caplog, monkeypatch):
    path = tmp_path / "cross.kmod"
    save_mod(path, [_set()], Settings(depth=1), sections=[_section()])
    (s,) = load_mod(path).sets
    plan = plan_targets([ScopedSet(s)], 1)["layer"]
    plan = plan.to(torch.device("cpu"), torch.float32)
    (layer,) = plan.sets
    assert layer.learned_cross is not None
    monkeypatch.setattr(fused, "_NOTED", set())
    x = torch.ones(1, 2, 4)
    with caplog.at_level(logging.INFO, logger="kilora"):
        assert not fused.fused_ok(x, [layer])
        assert not fused.fused_ok(x, [layer])
    assert caplog.text.count("learned cross") == 1
    with torch.no_grad():
        result = compute.run_layer(x, x, [layer], 1)
        with compute.eager():
            eager = compute.run_layer(x, x, [layer], 1)
    assert torch.equal(result, eager)


def test_zero_cross_allows_fused_like_base(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_FUSED", "1")
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "device_ok", lambda _: True)
    monkeypatch.setattr(compute, "_EAGER", False)
    layers = []
    for name, sections in (("base", []), ("zero", [_section(False)])):
        path = tmp_path / f"{name}.kmod"
        save_mod(path, [_set()], Settings(depth=1), sections=sections)
        (s,) = load_mod(path).sets
        layers.append(plan_targets([ScopedSet(s)], 1)["layer"].sets[0])
    assert layers[1].learned_cross_zero
    x = torch.ones(1, 2, 4)
    assert fused.fused_ok(x, [layers[0]])
    assert fused.fused_ok(x, [layers[1]])


@pytest.mark.parametrize("loop", [False, True])
def test_zero_section_matches_base_at_depth(tmp_path, monkeypatch, loop):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", not loop)
    torch.manual_seed(7)
    x = torch.randn(2, 5, 4)
    model = _model()
    model.layer.weight.data.copy_(torch.randn(4, 4))
    paths = [tmp_path / "base.kmod", tmp_path / "zero.kmod"]
    save_mod(paths[0], [_set()], Settings(depth=3))
    save_mod(paths[1], [_set()], Settings(depth=3), sections=[_section(False)])
    outputs = []
    for path in paths:
        adapter = compose(model, [path])
        with torch.no_grad(), compute.eager():
            outputs.append(model.layer(x).clone())
        adapter.remove()
    assert torch.equal(*outputs)


def test_score_uses_projected_layer_output(tmp_path, monkeypatch):
    path = tmp_path / "cross.kmod"
    save_mod(path, [_set()], Settings(depth=1), sections=[_section()])
    (s,) = load_mod(path).sets
    layer = plan_targets([ScopedSet(s)], 1)["layer"].sets[0]
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    y = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    captured = []
    real = compute._gates_from_logits

    def record(logits, *args, **kwargs):
        captured.append(logits[0].clone())
        return real(logits, *args, **kwargs)

    monkeypatch.setattr(compute, "_gates_from_logits", record)
    with torch.no_grad():
        compute.run_layer(x, y, [layer], 1)
    kin, _ = compute._set_keys(layer)
    delta = cross(y, y, s.learned_cross) - y
    expected = layer.tau * (
        compute.derived_logits(layer, y) / layer.tau
        + compute.cross_energy(delta, y, kin, compute.derived_keys(layer)[1])
    )
    torch.testing.assert_close(captured[0], expected)


def test_cross_term_continuity_and_gradient(tmp_path, monkeypatch):
    import kilora.routing_features as routing_features

    path = tmp_path / "cross.kmod"
    save_mod(path, [_set()], Settings(depth=1), sections=[_section()])
    (s,) = load_mod(path).sets
    layer = plan_targets([ScopedSet(s)], 1)["layer"].sets[0]
    token = torch.ones(1, 2, 4)
    monkeypatch.setattr(
        routing_features, "features_for_set", lambda _: RoutingFeatures(token, token, None, None)
    )
    keys, _ = compute._set_keys(layer)
    base = layer.tau * compute.norm_energy(token, keys, layer.rows_in)
    planted = layer.learned_cross["up"].clone()
    terms = []
    for scale in (0.0, 1e-3, 0.1, 1.0):
        layer.learned_cross["up"] = planted * scale
        terms.append((compute._learned_cross_logits(base, layer, token, keys) - base)[0, 0, 0])
    assert terms[0] == 0
    assert all(a < b for a, b in zip(terms, terms[1:], strict=False))
    torch.testing.assert_close(terms[1] / terms[-1], torch.tensor(1e-6), rtol=0.03, atol=0)

    up = torch.zeros_like(planted, requires_grad=True)
    layer.learned_cross["up"] = up
    layer.learned_cross["bias"] = torch.tensor([0.1, 0.0, 0.0, 0.0])
    logits = compute._learned_cross_logits(base, layer, token, keys)
    weight = compute.sparsemax(logits)[..., 0].sum()
    (gradient,) = torch.autograd.grad(weight, up)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("beam", [False, True])
def test_cross_runs_in_wide_paths(tmp_path, beam):
    path = tmp_path / "cross.kmod"
    settings = Settings(depth=2, iteration="path", path_width=2, path_beam=beam)
    save_mod(path, [_set()], settings, sections=[_section()])
    model = _model()
    compose(model, [path])
    x = torch.ones(1, 2, 4)
    with torch.no_grad():
        first = model.layer(x)
        assert torch.equal(first, model.layer(x))
