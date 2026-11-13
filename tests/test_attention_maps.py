# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_comfyui import denoiser, folders

from kilora.attention_maps import AttentionMaps, alpha_from_maps, trigger_positions


def test_positions_and_unknown_are_disabled():
    ids = {"t5xxl_ids": torch.tensor([7, 2, 3, 9, 2, 3])}
    positions, report = trigger_positions("anima", ids, {"character": [2, 3]})
    assert positions == {"character": (1, 2, 4, 5)}
    assert report is None
    for family, cond, triggers in [
        ("flux", ids, {"a": [2]}),
        ("anima", {}, {"a": [2]}),
        ("anima", ids, {"a": [100]}),
    ]:
        positions, report = trigger_positions(family, cond, triggers)
        assert positions == {}
        assert "off" in report


def test_cross_attention_full_softmax_middle_blocks_and_chaining():
    collector = AttentionMaps({"subject": (1,)}, blocks=6)
    patches = collector.kind_patches()
    calls = []
    override = collector.comfy_override(lambda func, *a, **kw: calls.append(1) or func(*a, **kw))
    q = torch.tensor([[[0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    k = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]])

    def attention(q, k, v, heads):
        return q

    for block in (0, 2, 3, 5):
        patches["attn2_patch"](q, k, k, extra_options={"block_index": block})
        assert override(attention, q, k, k, 1) is q
    result = collector.finish_step()["subject"]
    assert result.shape == (1, 4)
    assert result.argmax().item() == 1
    assert torch.allclose(result[0, 0], torch.tensor(1 / 3))
    assert len(calls) == 4
    assert collector.finish_step() == {}


def test_normalization_blur_and_per_entry_moving_average():
    maps = {"a": torch.zeros(1, 49), "b": torch.zeros(1, 49)}
    maps["a"][0, 24] = 2
    first = alpha_from_maps(maps, share=0.5)
    blurred = torch.zeros(1, 49)
    blurred.view(7, 7)[1:6, 1:6] = 1 / 25
    expected = torch.maximum(0.5 + blurred - blurred.mean(), torch.full_like(blurred, 0.5))
    torch.testing.assert_close(first["ip"], expected)
    assert first["ip"][0, 24] == first["ip"][0, 16]
    torch.testing.assert_close(first["creative"], torch.ones_like(first["ip"]))
    second = alpha_from_maps({k: torch.zeros_like(v) for k, v in maps.items()}, first)
    torch.testing.assert_close(second["ip"], torch.ones_like(blurred))
    torch.testing.assert_close(second.history["a"], first.history["a"] * 0.5)
    assert alpha_from_maps({}, None) == {}


def test_flat_or_weak_maps_give_the_budget_share_exactly():
    flat = alpha_from_maps({"a": torch.full((2, 24), 0.004)}, grid=(4, 6))
    assert torch.equal(flat["ip"], torch.ones(2, 24))
    weak = torch.full((1, 64), 0.0003)
    weak.view(8, 8)[2:6, 2:6] = 0.004
    alpha = alpha_from_maps({"a": weak}, grid=(8, 8), share=1 / 3)
    assert torch.equal(alpha["ip"], torch.full((1, 64), 1 / 3))


def _blob(level):
    ys, xs = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    blob = torch.exp(-((ys - 7.5) ** 2 + (xs - 7.5) ** 2) / 18)
    return (0.002 + level * blob).reshape(1, 256)


def test_clear_map_keeps_the_mean_and_moves_the_share_to_the_subject():
    alpha = alpha_from_maps({"a": _blob(0.028)}, grid=(16, 16), share=0.5)["ip"]
    assert float(alpha.min()) > 0 and float(alpha.max()) < 1
    torch.testing.assert_close(alpha.mean(), torch.tensor(0.5), rtol=0, atol=1e-5)
    image = alpha.view(16, 16)
    assert float(image[7:9, 7:9].min()) > 0.6 and float(image[0, 0]) < 0.45


def test_strong_map_saturates_on_the_subject():
    value = torch.full((1, 256), 1e-4)
    value.view(16, 16)[4:12, 4:12] = 0.05
    alpha = alpha_from_maps({"a": value}, grid=(16, 16))["ip"].view(16, 16)
    assert float(alpha[6:10, 6:10].min()) == 1.0 and float(alpha[0, 0]) < 0.8


def test_hard_alpha_feathers_and_combines_entries(monkeypatch):
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    value = torch.full((2, 32, 32), 1e-4)
    value[:, 8:24, 8:24] = 0.05
    value[1] = value[1].roll(5, dims=1)
    maps = {"a": value.flatten(1), "b": value.flip(-1).flatten(1)}
    combined = alpha_from_maps(maps, grid=(32, 32))
    a = alpha_from_maps({"a": maps["a"]}, grid=(32, 32))["ip"]
    b = alpha_from_maps({"b": maps["b"]}, grid=(32, 32))["ip"]
    image = a.view(2, 32, 32)
    assert image[0, 16, 16] == 1 and image[0, 0, 0] == 0
    assert ((image[0] > 0) & (image[0] < 1)).any()
    assert torch.equal(combined["ip"], torch.maximum(a, b))
    assert torch.equal(combined["creative"], torch.ones_like(combined["ip"]))


def test_hard_alpha_confidence_and_soft_history(monkeypatch):
    flat = torch.full((2, 256), 0.05)
    weak = torch.full((2, 256), 0.00001)
    weak.view(2, 16, 16)[:, 4:12, 4:12] = 0.004
    strong = weak * 20
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    for value in (flat, weak):
        alpha = alpha_from_maps({"a": value}, grid=(16, 16), share=1 / 3)
        assert torch.equal(alpha["ip"], torch.full_like(value, 1 / 3))
    hard_first = alpha_from_maps({"a": strong}, grid=(16, 16))
    monkeypatch.setenv("KILORA_ALPHA_HARD", "0")
    soft_first = alpha_from_maps({"a": strong}, grid=(16, 16))
    assert torch.equal(hard_first.history["a"], soft_first.history["a"])
    soft_second = alpha_from_maps({"a": strong.flip(-1)}, soft_first, grid=(16, 16))
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    hard_second = alpha_from_maps({"a": strong.flip(-1)}, hard_first, grid=(16, 16))
    assert torch.equal(hard_second.history["a"], soft_second.history["a"])


def test_disabled_hard_alpha_preserves_original_calculation(monkeypatch):
    from kilora.attention_maps import map_confidence

    value = torch.rand(2, 64, generator=torch.Generator().manual_seed(13)) ** 4
    scale = torch.quantile(value, 0.99, dim=-1, keepdim=True)
    normalized = (value / scale.clamp_min(1e-12)).clamp(0, 1).reshape(2, 1, 8, 8)
    blurred = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(normalized, (2, 2, 2, 2), mode="replicate"), 5, stride=1
    ).flatten(1)
    expected = 1.0 + map_confidence(value)[:, None] * (blurred - blurred.mean(-1, keepdim=True))
    expected = expected.clamp(0, 1)
    for setting in (None, "0"):
        if setting is None:
            monkeypatch.delenv("KILORA_ALPHA_HARD", raising=False)
        else:
            monkeypatch.setenv("KILORA_ALPHA_HARD", setting)
        result = alpha_from_maps({"a": value}, grid=(8, 8))
        assert torch.equal(result["ip"], expected)
        assert torch.equal(result["creative"], torch.ones_like(expected))


@pytest.mark.parametrize("hard", ["0", "1"])
def test_alpha_level_override(monkeypatch, hard):
    from kilora.attention_maps import map_confidence

    monkeypatch.setenv("KILORA_ALPHA_HARD", hard)
    value = torch.full((1, 256), 0.00001)
    value.view(16, 16)[4:12, 4:12] = 0.006
    monkeypatch.delenv("KILORA_ALPHA_LEVEL", raising=False)
    assert torch.equal(map_confidence(value), torch.zeros(1))
    default = alpha_from_maps({"a": value}, grid=(16, 16))
    assert torch.equal(default["ip"], torch.ones_like(value))
    monkeypatch.setenv("KILORA_ALPHA_LEVEL", "0.002,0.01")
    torch.testing.assert_close(map_confidence(value), torch.tensor([0.5]))
    changed = alpha_from_maps({"a": value}, grid=(16, 16))
    assert not torch.equal(changed["ip"], default["ip"])


def test_default_confidence_matches_baseline(monkeypatch):
    import math

    from kilora.attention_maps import FULL_CONTRAST, LEVEL_FLOOR, LEVEL_FULL, map_confidence

    monkeypatch.delenv("KILORA_ALPHA_LEVEL", raising=False)
    value = torch.rand(4, 256, generator=torch.Generator().manual_seed(4)) ** 4 * 0.08
    p99 = torch.quantile(value, 0.99, dim=-1)
    p50 = torch.quantile(value, 0.5, dim=-1)
    ratio = torch.where(p99 > 0, p99 / p50.clamp_min(p99 * 1e-6 + 1e-30), torch.ones_like(p99))
    contrast = (ratio.clamp_min(1.0).log() / math.log(FULL_CONTRAST)).clamp(0, 1)
    level = ((p99 - LEVEL_FLOOR) / (LEVEL_FULL - LEVEL_FLOOR)).clamp(0, 1)
    assert torch.equal(map_confidence(value), contrast * level)


@pytest.mark.parametrize("setting", ["", "0.1", "0.1,0.1", "-1,1", "0,nan", "a,b", "0,1,2"])
def test_invalid_alpha_level_override(monkeypatch, setting):
    from kilora.attention_maps import map_confidence

    monkeypatch.setenv("KILORA_ALPHA_LEVEL", setting)
    with pytest.raises(ValueError, match="KILORA_ALPHA_LEVEL"):
        map_confidence(torch.ones(1, 4))


def test_alphas_keep_independent_strength_per_token():
    from kilora.compute import axis_budget

    value = torch.rand(2, 36, generator=torch.Generator().manual_seed(3)) ** 4
    alphas = alpha_from_maps({"a": value, "b": value.flip(-1)}, grid=(6, 6))
    w = axis_budget({"ip": alphas["ip"], "creative": alphas["creative"]})
    torch.testing.assert_close(w["creative"], torch.ones(2, 36))
    assert torch.equal(w["ip"], alphas["ip"])
    assert (w["ip"] + w["creative"] > 1).any()
    assert float(alphas["ip"].min()) >= 0 and float(alphas["ip"].max()) <= 1


def test_conditional_rows_and_chunked_softmax():
    generator = torch.Generator().manual_seed(22)
    q = torch.randn(4, 2051, 8, generator=generator)
    k = torch.randn(4, 7, 8, generator=generator)
    mask = torch.randn(4, 1, 2051, 7, generator=generator)
    collector = AttentionMaps({"a": (1, 3), "b": (5,)}, blocks=3, block=1)
    collector.collect(q, k, 2, mask=mask, cond_or_uncond=[1, 0])
    result = collector.finish_step()
    q_cond = q[2:].reshape(2, 2051, 2, 4).transpose(1, 2)
    k_cond = k[2:].reshape(2, 7, 2, 4).transpose(1, 2)
    scores = q_cond @ k_cond.transpose(-1, -2) / 2 + mask[2:]
    probabilities = scores.softmax(-1)
    for name, positions in collector.positions.items():
        expected = probabilities[..., list(positions)].sum(-1).mean(1)
        assert torch.allclose(result[name], expected, atol=1e-6)
    q[:2] *= 100
    k[:2] *= 100
    collector.collect(q, k, 2, mask=mask, cond_or_uncond=[1, 0])
    for name, value in collector.finish_step().items():
        assert torch.equal(value, result[name])
    collector.collect(q, k, 2, cond_or_uncond=[1, 1])
    assert collector.finish_step() == {}


def test_patch_passes_conditional_rows_to_collector():
    collector = AttentionMaps({"a": (0,)}, blocks=3)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 2, 4)
    collector.kind_patches()["attn2_patch"](
        q, k, k, extra_options={"block_index": 1, "cond_or_uncond": [1, 0]}
    )
    collector.comfy_override()(lambda *args: q, q, k, k, 1)
    assert collector.finish_step()["a"].shape == (1, 3)


def test_one_axis_keeps_its_full_strength_on_a_flat_map():
    from kilora.compute import budget_share

    assert budget_share({"ip"}) == 1.0
    assert budget_share({"ip", "creative", "functional"}) == 1.0
    assert budget_share({"ip", "creative", "other"}) == 1.0
    flat = alpha_from_maps({"a": torch.full((1, 24), 0.004)}, grid=(4, 6), share=1.0)
    torch.testing.assert_close(flat["ip"], torch.ones(1, 24))


def test_apply_gives_auto_alpha_the_share_of_its_axes(denoiser, folders, monkeypatch):
    from test_axes import _save
    from test_comfyui import FakeBaseModel, FakePatcher

    from kilora.comfyui import auto_alpha
    from kilora.comfyui import nodes as N

    shares = []
    real = auto_alpha.AutoAlpha.__init__

    def spy(self, *args, **kwargs):
        real(self, *args, **kwargs)
        shares.append(self.share)

    monkeypatch.setattr(auto_alpha.AutoAlpha, "__init__", spy)
    _save(folders, "person", 1)
    _save(folders, "ink", 2)
    (alone,) = N.KiLoRAImportLoRA().add("person.safetensors", 1.0, axis="ip")
    (mixed,) = N.KiLoRAImportLoRA().add("ink.safetensors", 1.0, sets=alone, axis="creative")
    for chain in (alone, mixed):
        N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 1, axis_alpha="auto")
    assert shares == [1.0, 1.0]
