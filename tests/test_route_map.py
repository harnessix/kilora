# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_comfyui import FakeBaseModel, denoiser, folders
from test_modlist import write_mod
from test_path_write import WrapPatcher

from kilora import KiLoRAError
from kilora import compute as C
from kilora.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from kilora.comfyui import nodes as N
from kilora.comfyui import route_map as R


def _patched(denoiser, folders, depth=2):
    write_mod(
        denoiser, folders["kilora"], "ink", seed=1, target_modules=["layers.*.feed_forward.w2"]
    )
    base = WrapPatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink-1.0.0.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(base, chain, depth)
    return WrapPatcher(patched.model, patched.object_patches, patched.model_options)


def _x(seed=0):
    return torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(seed))


def test_nodes_are_registered():
    assert NODE_CLASS_MAPPINGS["KiLoRARouteMap"] is R.KiLoRARouteMap
    assert NODE_DISPLAY_NAME_MAPPINGS["KiLoRARouteMap"] == "KiLoRA Route Map"
    assert R.KiLoRARouteRecord.RETURN_TYPES == ("MODEL", R.ROUTES)
    assert R.KiLoRARouteMap.RETURN_TYPES == ("IMAGE", "STRING")
    assert list(R.KiLoRARouteMap.INPUT_TYPES()["required"]) == ["routes", "image"]


def test_shares_sum_to_one_per_token(denoiser, folders):
    patched = _patched(denoiser, folders)
    model, routes = R.KiLoRARouteRecord().record(patched)
    model.sample(_x(), steps=3)
    shares = routes.shares()
    assert shares and routes.loras == 4
    for s in shares.values():
        assert s.shape == (6, 4)
        torch.testing.assert_close(s.sum(-1), torch.ones(6))
        assert torch.all(s >= 0)
    first = {t: s.clone() for t, s in shares.items()}
    model.sample(_x(5), steps=3)
    assert routes.counts == {t: 3 * 2 for t in first}
    assert any(not torch.equal(first[t], s) for t, s in routes.shares().items())


def test_recording_does_not_change_the_output(denoiser, folders):
    patched = _patched(denoiser, folders)
    model, _ = R.KiLoRARouteRecord().record(patched)
    want = patched.sample(_x(), steps=2)
    got = model.sample(_x(), steps=2)
    for a, b in zip(want, got, strict=True):
        torch.testing.assert_close(a, b)


def test_map_size_and_report(denoiser, folders):
    patched = _patched(denoiser, folders)
    model, routes = R.KiLoRARouteRecord().record(patched)
    model.sample(_x(), steps=2)
    image = torch.rand(2, 24, 16, 3)
    out, report = R.KiLoRARouteMap().draw(routes, image, labels="ink, parchment")
    assert out.shape == (2, 24 + R.LEGEND, 16 * 4, 3)
    assert out.dtype == torch.float32 and out.min() >= 0 and out.max() <= 1
    assert "1. ink (red)" in report and "2. parchment (blue)" in report and "4. LoRA 4" in report
    pad = 4
    torch.testing.assert_close(
        out[0, 24 + pad + 1, 16 + pad + 1], torch.tensor(R.COLORS[1]), atol=1 / 255, rtol=0
    )


def test_token_grid_follows_the_image_aspect():
    assert R.token_grid(6, 20, 30) == (2, 3)
    assert R.token_grid(64, 64, 64) == (8, 8)
    assert R.token_grid(32, 64, 128) == (4, 8)
    assert R.token_grid(5, 10, 10) in {(1, 5), (5, 1)}


def test_no_record_is_a_clear_error():
    with pytest.raises(KiLoRAError, match="KiLoRA Route Record"):
        R.render(R.RouteShares(), torch.zeros(1, 8, 8, 3))


def test_no_recording_cost_without_the_node(denoiser, folders, monkeypatch):
    patched = _patched(denoiser, folders)
    calls = []
    monkeypatch.setattr(R.RouteShares, "observe", lambda self, *a: calls.append(a))
    seen = []
    real = C._note_routes

    def spy(target, gates, sets=None):
        seen.append((list(C._SHARES), list(C._ROUTES)))
        real(target, gates, sets)

    monkeypatch.setattr(C, "_note_routes", spy)
    patched.sample(_x(), steps=2)
    assert seen
    assert all(shares == [] and routes == [] for shares, routes in seen)
    assert calls == []
    before = len(seen)
    model, _ = R.KiLoRARouteRecord().record(patched)
    model.sample(_x(), steps=2)
    assert len(calls) == len(seen) - before == before
    assert not C._SHARES


def test_fused_path_steps_aside_while_shares_are_recorded(monkeypatch):
    from kilora import compute, fused

    notes = []
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(fused, "enabled", lambda: True)
    monkeypatch.setattr(fused, "device_ok", lambda t: True)
    monkeypatch.setattr(fused, "_note", notes.append)
    y0 = torch.randn(1, 4, 16)
    with compute.recording_shares(lambda target, gates: None):
        assert not fused.fused_ok(y0, [])
    assert notes == ["routes or gates are being recorded"]
