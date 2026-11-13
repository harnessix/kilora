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
from kilora.depth_map import feather, luminance, make_depth_map
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
    with pytest.raises(patching.KiLoRAError, match="mask or an image"):
        _build()
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
