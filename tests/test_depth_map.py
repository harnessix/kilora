# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_comfyui_dropin import FakeClip, WrappedPatcher, comfy, sample

from kilora.comfyui import dropin, patching
from kilora.comfyui import nodes as N
from kilora.compute import sampling_progress
from kilora.depth_map import (
    budget_depths,
    detail_energy,
    feather,
    luminance,
    make_auto_depth_map,
    make_depth_map,
    normalized,
    smooth_grid,
)
from kilora.region import latent_size


def _build(**kw):
    args = dict(
        min_depth=1.0, max_depth=3.0, invert=False, gamma=1.0, feather=0, start=0.3, base=-1.0
    )
    args.update(kw)
    return N.KiLoRADepthMap().build(**args)["result"]


def test_a_mask_maps_to_the_depth_range():
    mask = torch.zeros(1, 8, 8)
    mask[:, :, 4:] = 1.0
    m, preview = _build(mask=mask, min_depth=2.0, max_depth=5.0)
    depths = m.depths()
    assert depths[..., :4].eq(2).all() and depths[..., 4:].eq(5).all()
    assert m.base == 4.0
    assert m.start == 0.3
    assert preview.shape == (1, 8, 8, 3)
    inverted, _ = _build(mask=mask, min_depth=2.0, max_depth=5.0, invert=True)
    assert inverted.depths()[..., :4].eq(5).all() and inverted.depths()[..., 4:].eq(2).all()
    assert _build(mask=mask, base=1.5)[0].base == 1.5


def test_an_image_maps_by_its_luminance():
    image = torch.zeros(1, 4, 4, 3)
    image[:, :2] = 1.0
    image[:, 2:, :, 1] = 1.0
    m, _ = _build(image=image, min_depth=1.0, max_depth=3.0)
    torch.testing.assert_close(m.depths()[0, :2], torch.full((2, 4), 3.0))
    torch.testing.assert_close(m.depths()[0, 2:], torch.full((2, 4), 1.0 + 2 * 0.587))
    assert luminance(torch.ones(1, 2, 2, 1)).shape == (1, 2, 2)
    squared, _ = _build(image=image, gamma=2.0)
    torch.testing.assert_close(squared.values[0, 2:], torch.full((2, 4), 0.587**2))
    both, _ = _build(mask=torch.zeros(1, 4, 4), image=image)
    assert both.depths().eq(1).all()


def test_feather_smooths_the_edge():
    mask = torch.zeros(1, 16, 16)
    mask[:, :, 8:] = 1.0
    hard = make_depth_map(mask, feather_px=0).values[0, 0]
    soft = make_depth_map(mask, feather_px=4).values[0, 0]
    assert set(hard.tolist()) == {0.0, 1.0}
    assert 0 < soft[7] < 0.5 < soft[8] < 1
    assert torch.all(soft[1:] >= soft[:-1] - 1e-6)
    assert soft[0] < 1e-3 and soft[-1] > 1 - 1e-3
    assert torch.equal(feather(mask, 0), mask)


def test_bad_settings_are_refused():
    with pytest.raises(patching.KiLoRAError, match="weights"):
        _build(detail_weight=-1.0)
    with pytest.raises(ValueError, match="start"):
        make_depth_map(torch.ones(1, 2, 2), start=1.5)
    with pytest.raises(ValueError, match="gamma"):
        make_depth_map(torch.ones(1, 2, 2), gamma=0)


def _chain(folders, targets):
    gen = torch.Generator().manual_seed(3)
    for i, (target, width) in enumerate(targets):
        stem = f"lora_unet_layers_0_{target}"
        sd = {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(width, 2, generator=gen),
            f"{stem}.alpha": torch.tensor(1.0),
        }
        save_file(sd, str(folders["loras"] / f"s{i}.safetensors"))
    (c,) = N.KiLoRAImportLoRA().add("s0.safetensors", 1.0)
    (c,) = N.KiLoRAImportLoRA().add("s1.safetensors", 0.8, sets=c)
    return c


@pytest.fixture
def chain(folders):
    return _chain(folders, [("attention_to_q", 16), ("attention_to_v", 16)])


@pytest.fixture
def mlp_chain(folders):
    return _chain(folders, [("feed_forward_w1", 32), ("feed_forward_w1", 32)])


def _run(patcher, progress, x):
    with latent_size((4, 4)), sampling_progress(progress):
        return patcher.run(x)


def _apply(denoiser, chain, depth, depth_map=None):
    kw = {} if depth_map is None else {"depth_map": depth_map}
    patched, report, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, depth, **kw
    )
    return patched


def test_start_and_base_hold_through_apply(denoiser, mlp_chain):
    chain = mlp_chain
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    uniform = {d: _run(_apply(denoiser, chain, d), 0.0, x) for d in (1, 2, 3)}
    assert not torch.equal(uniform[1], uniform[3])
    full, _ = _build(mask=torch.ones(1, 32, 32), min_depth=1.0, max_depth=3.0, base=2.0)
    patched = _apply(denoiser, chain, 1, full)
    assert torch.equal(_run(patched, 0.1, x), uniform[2])
    assert torch.equal(_run(patched, 0.5, x), uniform[3])
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    split, _ = _build(mask=mask, min_depth=1.0, max_depth=3.0)
    patched = _apply(denoiser, chain, 1, split)
    assert torch.equal(_run(patched, 0.0, x), uniform[2])
    late = _run(patched, 0.5, x)
    assert not torch.equal(late, uniform[1]) and not torch.equal(late, uniform[3])
    again, report = patching.reapply(patched)
    assert patching.applied(again)["token_depth"] is split
    assert patching.applied(again)["joint"] is True and "routed-loras" in report
    assert torch.equal(_run(again, 0.5, x), late)


def test_attention_follows_depth_map(denoiser, chain):
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    uniform = {d: _run(_apply(denoiser, chain, d), 0.0, x) for d in (1, 2, 3)}
    assert not torch.equal(uniform[1], uniform[3])
    full, _ = _build(mask=torch.ones(1, 32, 32), min_depth=1.0, max_depth=3.0, base=2.0)
    patched = _apply(denoiser, chain, 1, full)
    assert torch.equal(_run(patched, 0.1, x), uniform[2])
    assert torch.equal(_run(patched, 0.5, x), uniform[3])
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    split, _ = _build(mask=mask, min_depth=1.0, max_depth=3.0)
    patched = _apply(denoiser, chain, 1, split)
    late = _run(patched, 0.5, x)
    assert not torch.equal(late, uniform[1]) and not torch.equal(late, uniform[3])


def test_no_map_gives_todays_result(denoiser, chain):
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    plain = _apply(denoiser, chain, 2)
    none = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 2, depth_map=None)[0]
    assert torch.equal(_run(none, 0.5, x), _run(plain, 0.5, x))
    assert patching.applied(none)["token_depth"] is None


def test_the_drop_in_loader_passes_the_map(denoiser, folders, chain, comfy):
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    m, _ = _build(mask=mask, min_depth=1.0, max_depth=3.0)
    node = dropin.KiLoRALoRALoader()
    model, clip = WrappedPatcher(FakeBaseModel(denoiser)), FakeClip()
    model, clip = node.load_lora(model, clip, "s0.safetensors", 1.0, 0.0)
    model, clip = node.load_lora(model, clip, "s1.safetensors", 0.8, 0.0, depth_map=m)
    assert dropin.state_of(model).settings["token_depth"] is m
    x = torch.randn(1, 16, 4, generator=torch.Generator().manual_seed(4))
    with latent_size((4, 4)), sampling_progress(0.5):
        got = sample(model, x)
    want = patching.apply(
        FakePatcher(FakeBaseModel(denoiser)),
        chain,
        -1,
        patching.LiveEnv(),
        joint=True,
        token_depth=m,
        depth_steps=N.COMFY_DEPTH_STEPS,
    )[0]
    assert torch.equal(got, _run(want, 0.5, x))


def test_no_mask_or_image_gives_an_auto_map():
    m, preview = _build()
    assert m.auto is not None and not m.ready and m.measured() is None
    assert preview.shape == (1, 64, 64, 3) and preview.eq(0.5).all()
    assert m.base == 2.0
    mask = torch.ones(1, 4, 4)
    assert _build(mask=mask, source="auto")[0].auto is not None
    assert _build(mask=mask, source="mask")[0].auto is None
    assert _build(image=torch.ones(1, 4, 4, 3), source="image")[0].auto is None
    assert _build(source="image")[0].auto is not None


def test_the_quantile_budget_keeps_the_mean_depth():
    gen = torch.Generator().manual_seed(5)
    score = torch.rand(2, 8, 8, generator=gen)
    score[1, :4] = 0.25
    for mean in (1.0, 1.7, 2.5, 3.0):
        depths = budget_depths(score, 1.0, 3.0, mean)
        torch.testing.assert_close(depths.mean((1, 2)), torch.full((2,), mean), atol=1e-4, rtol=0)
        assert depths.min() >= 1.0 - 1e-6 and depths.max() <= 3.0 + 1e-6
        order = score[0].flatten().argsort()
        assert torch.all(depths[0].flatten()[order].diff() >= -1e-6)
        tied = depths[1, :4]
        assert torch.allclose(tied, tied.flatten()[0].expand_as(tied))
    assert torch.allclose(budget_depths(score, 1.0, 3.0, 9.0), torch.full_like(score, 3.0))
    assert budget_depths(score, 2.0, 2.0, 1.0).eq(2.0).all()


def _latent():
    latent = torch.zeros(1, 4, 4, 4)
    rows, cols = torch.meshgrid(torch.arange(4), torch.arange(2), indexing="ij")
    latent[..., :2] = ((rows + cols) % 2 * 2 - 1).float()
    return latent


def _tokens(latent):
    return latent.flatten(2).transpose(1, 2).contiguous()


def _auto_patched(denoiser, mlp_chain, **kw):
    args = dict(
        source="auto",
        min_depth=1.0,
        max_depth=3.0,
        feather=0,
        start=0.3,
        change_weight=0.0,
        smooth=0.0,
    )
    args.update(kw)
    m, _ = _build(**args)
    return m, _apply(denoiser, mlp_chain, 2, m)


def _invoke(patched, progress, calls=None):
    def invoke(args):
        if calls is not None:
            calls.append(progress)
        _run(patched, progress, _tokens(args["input"]))
        return args["input"]

    return invoke


def test_auto_gives_detail_more_depth(denoiser, mlp_chain):
    m, patched = _auto_patched(denoiser, mlp_chain)
    assert m.mean_target() == 2.0
    args = {"input": _latent()}
    m.call(_invoke(patched, 0.4), args, 0.4)
    depths = m.measured()
    assert depths is not None and m.ready
    assert depths.shape[-2:] == (2, 2)
    d = m.depths()
    assert float(d[..., 0].mean()) > float(d[..., 1].mean())
    torch.testing.assert_close(d.mean(), torch.tensor(2.0), atol=1e-4, rtol=0)
    m2, patched2 = _auto_patched(denoiser, mlp_chain, change_weight=1.0)
    m2.call(_invoke(patched2, 0.4), args, 0.4)
    assert m2.ready and m2.depths().shape[-2:] == (4, 4)
    torch.testing.assert_close(m2.depths().mean(), torch.tensor(2.0), atol=1e-4, rtol=0)
    x = _tokens(_latent())
    late = _run(patched, 0.5, x)
    uniform = _run(_apply(denoiser, mlp_chain, 2), 0.5, x)
    assert not torch.equal(late, uniform)


def test_auto_measures_once_per_generation(denoiser, mlp_chain, monkeypatch):
    m, patched = _auto_patched(denoiser, mlp_chain)
    measured = []
    finish = m.finish
    monkeypatch.setattr(m, "finish", lambda *a, **k: (measured.append(a[-2]), finish(*a, **k)))
    args = {"input": _latent()}
    for p in (0.0, 0.1, 0.2):
        m.call(_invoke(patched, p), args, p)
    assert measured == [] and not m.ready
    for p in (0.3, 0.4, 0.6, 1.0):
        m.call(_invoke(patched, p), args, p)
    assert measured == [0.3]
    first = m.values
    m.call(_invoke(patched, 1.0), args, 1.0)
    assert m.values is first
    m.call(_invoke(patched, 0.0), args, 0.0)
    assert not m.ready
    for p in (0.1, 0.3, 0.6):
        m.call(_invoke(patched, p), args, p)
    assert measured == [0.3, 0.3]


def test_auto_calls_after_the_measurement_read_no_host_values(
    denoiser,
    mlp_chain,
    monkeypatch,
):
    m, patched = _auto_patched(denoiser, mlp_chain, change_weight=1.0)
    m.call(_invoke(patched, 0.4), {"input": _latent()}, 0.4)
    assert m.ready
    same = make_depth_map(m.values.clone(), m.min_depth, m.max_depth, start=m.start, base=m.base)
    baseline = _apply(denoiser, mlp_chain, 2, same)
    _run(baseline, 0.1, _tokens(_latent()))
    reads = []
    for name in ("item", "tolist", "cpu"):
        original = getattr(torch.Tensor, name)

        def counted(self, *a, _original=original, _name=name, **k):
            reads.append(_name)
            return _original(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, name, counted)
    m.call(lambda args: None, {"input": _latent()}, 0.6)
    assert reads == []
    x = _tokens(_latent())
    _run(patched, 0.6, x)
    auto_reads = list(reads)
    reads.clear()
    _run(baseline, 0.6, x)
    assert auto_reads == reads


def _total_variation(t):
    return float((t[:, 1:] - t[:, :-1]).abs().sum() + (t[:, :, 1:] - t[:, :, :-1]).abs().sum())


def test_smoothing_lowers_the_variation_of_a_noise_score():
    gen = torch.Generator().manual_seed(11)
    noise = torch.rand(2, 32, 32, generator=gen)
    assert smooth_grid(noise, 0.0) is noise
    smooth = smooth_grid(noise, 2.0)
    assert smooth.shape == noise.shape
    assert _total_variation(smooth) < 0.5 * _total_variation(noise)
    assert _total_variation(smooth_grid(noise, 4.0)) < _total_variation(smooth)
    flat = torch.full((1, 6, 5), 0.7)
    torch.testing.assert_close(smooth_grid(flat, 2.0), flat)


def _measure(smooth, mean_depth=2.5, latent=None):
    m = make_auto_depth_map(1.0, 3.0, 0.3, 2.0, mean_depth, 1.0, 0.0, 0.0, 0, smooth)
    x = (
        latent
        if latent is not None
        else torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
    )
    measurement = m.begin_call(0.3)
    assert measurement is not None
    m.finish(measurement, x, x, 0.3)
    return m, x


def test_smooth_zero_keeps_the_depths_of_the_unsmoothed_map():
    m, x = _measure(0.0)
    score = normalized(detail_energy(x, (8, 8)))
    expected = budget_depths(score, 1.0, 3.0, 2.5)
    torch.testing.assert_close(m.depths(), expected, atol=1e-6, rtol=0)
    smoothed, _ = _measure(2.0, latent=x)
    assert not torch.equal(smoothed.values, m.values)
    assert make_auto_depth_map().auto.smooth == 2.0
    with pytest.raises(ValueError):
        make_auto_depth_map(smooth=-1.0)


def test_smoothing_keeps_the_mean_depth():
    for smooth in (0.0, 1.0, 2.0, 5.0):
        m, _ = _measure(smooth, mean_depth=2.5)
        torch.testing.assert_close(m.depths().mean(), torch.tensor(2.5), atol=1e-4, rtol=0)


def test_smoothing_runs_on_the_device_without_a_host_read(monkeypatch):
    reads = []
    for name in ("item", "tolist", "cpu"):
        original = getattr(torch.Tensor, name)

        def counted(self, *a, _original=original, _name=name, **k):
            reads.append(_name)
            return _original(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, name, counted)
    smooth_grid(torch.rand(1, 8, 8), 2.0)
    assert reads == []


def test_the_measured_map_is_shown_once_per_generation(denoiser, mlp_chain, monkeypatch):
    from kilora.comfyui import depth_preview

    shown = []
    monkeypatch.setattr(depth_preview, "show_measured", lambda v, node: shown.append((v, node)))
    m, _ = _build(source="auto", change_weight=0.0, unique_id="7")
    patched = _apply(denoiser, mlp_chain, 2, m)
    args = {"input": _latent()}
    for p in (0.0, 0.1, 0.3, 0.4, 0.6, 1.0):
        m.call(_invoke(patched, p), args, p)
    assert len(shown) == 1 and shown[0][1] == "7"
    assert shown[0][0] is m.values
    m.call(_invoke(patched, 0.0), args, 0.0)
    for p in (0.3, 0.6):
        m.call(_invoke(patched, p), args, p)
    assert len(shown) == 2
    mask, _ = _build(mask=torch.ones(1, 4, 4), unique_id="7")
    assert mask.on_measured is None and len(shown) == 2


def test_show_measured_writes_a_temp_png_and_sends_it(tmp_path, monkeypatch):
    import sys
    import types

    from PIL import Image

    from kilora.comfyui import depth_preview

    sent = []
    server = types.SimpleNamespace(
        send_sync=lambda *a: sent.append(a), last_prompt_id="p1", client_id="c1"
    )
    module = types.ModuleType("server")
    module.PromptServer = types.SimpleNamespace(instance=server)
    monkeypatch.setitem(sys.modules, "server", module)
    monkeypatch.setattr(depth_preview, "_temp_folder", lambda: tmp_path)
    values = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
    assert depth_preview.show_measured(values, None) is None and not sent
    image = depth_preview.show_measured(values, "12")
    assert image is not None and image["type"] == "temp" and image["subfolder"] == ""
    pixels = Image.open(tmp_path / image["filename"]).convert("L")
    assert pixels.size == (2, 2) and pixels.getpixel((1, 0)) == 255
    event, data, client = sent[0]
    assert event == "executed" and client == "c1"
    assert data["node"] == "12" and data["prompt_id"] == "p1"
    assert data["output"] == {"images": [image]}
