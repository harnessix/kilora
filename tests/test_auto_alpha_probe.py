# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kilora.attention_maps import AttentionMaps
from kilora.call_values import current_axis_alphas
from kilora.comfyui import auto_alpha
from kilora.comfyui.auto_alpha import AutoAlpha, ProbeAttentionMaps


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(3, 4, bias=True)
        self.k_norm = nn.LayerNorm(2)

    def keys(self, context):
        key = self.k_proj(context).reshape(context.shape[0], -1, 2, 2)
        return self.k_norm(key).transpose(1, 2)


def _model():
    return SimpleNamespace(blocks=[SimpleNamespace(cross_attn=ToyAttention()) for _ in range(3)])


def _collector(model, embedding, positions=None):
    probes = {"cat": ([[embedding, {}]], embedding.shape[1])}
    collector = ProbeAttentionMaps(positions or {}, 3, probes, model)
    collector.block = 1
    return collector


def _attention(q, k, v, heads, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("skip_reshape", [False, True])
@pytest.mark.parametrize("mask_kind", [None, "bool", "additive"])
def test_probe_map_matches_appended_context(tokens, skip_reshape, mask_kind):
    torch.manual_seed(713)
    model = _model()
    layer = model.blocks[1].cross_attn
    context, embedding = torch.randn(1, 5, 3), torch.randn(1, tokens, 3)
    q, k = torch.randn(1, 2, 7, 2), layer.keys(context)
    probe_keys = layer.keys(embedding)
    mask = None
    if mask_kind is not None:
        mask = torch.ones(1, 1, 7, 5, dtype=torch.bool)
        mask[..., 0] = False
        if mask_kind == "additive":
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -float("inf"))
    collector = _collector(model, embedding)
    query, keys = q, k
    if not skip_reshape:
        query = q.transpose(1, 2).reshape(1, 7, 4)
        keys = k.transpose(1, 2).reshape(1, 5, 4)
    collector.collect(query, keys, 2, skip_reshape, mask)
    expected = []
    for index in range(tokens):
        appended = torch.cat((k, probe_keys[..., index : index + 1, :]), dim=-2)
        scores = (q @ appended.transpose(-1, -2)) / math.sqrt(2)
        if mask is not None:
            if mask.dtype == torch.bool:
                extended = torch.cat((mask, torch.ones_like(mask[..., :1])), dim=-1)
                scores = scores.masked_fill(~extended, -float("inf"))
            else:
                scores = scores + torch.cat((mask, torch.zeros_like(mask[..., :1])), dim=-1)
        expected.append(scores.softmax(-1)[..., -1].mean(1))
    torch.testing.assert_close(collector.finish_step()["cat"], torch.stack(expected).mean(0))


def test_probe_attention_output_is_bit_identical_and_chains_override():
    torch.manual_seed(714)
    model = _model()
    q, k, v = (torch.randn(1, 2, count, 2) for count in (4, 5, 5))
    collector = _collector(model, torch.randn(1, 2, 3))
    calls = []

    def previous(func, *args, **kwargs):
        calls.append(1)
        return func(*args, **kwargs)

    collector.kind_patches()["attn2_patch"](q, k, v, extra_options={"block_index": 1})
    before = tuple(t.clone() for t in (q, k, v))
    actual = collector.comfy_override(previous)(_attention, q, k, v, 2, skip_reshape=True)
    assert torch.equal(actual, _attention(q, k, v, 2))
    assert all(torch.equal(a, b) for a, b in zip(before, (q, k, v), strict=True))
    assert calls == [1]
    assert "cat" in collector.finish_step()


def test_probes_only_sample_middle_blocks_and_positive_cfg_rows():
    torch.manual_seed(715)
    model = _model()
    embedding = torch.randn(1, 1, 3)
    q, k = torch.randn(4, 2, 4, 2), torch.randn(4, 2, 5, 2)
    collector = _collector(model, embedding)
    collector.block = 0
    collector.collect(q, k, 2, True, cond_or_uncond=[1, 0])
    assert collector.finish_step() == {}
    collector.block = 1
    collector.collect(q, k, 2, True, cond_or_uncond=[1, 0])
    reference = _collector(model, embedding)
    reference.collect(q[2:], k[2:], 2, True)
    assert torch.equal(collector.finish_step()["cat"], reference.finish_step()["cat"])
    collector.collect(q, k, 2, True, cond_or_uncond=[1])
    assert collector.finish_step() == {}


class ToyClip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, value):
        ids = [7, 8, 0] if value else [0]
        return {"t5xxl": [[(i, 1.0) for i in ids]]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return [[torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]), {}]]


def _auto(clip, model, path="cat.safetensors", triggers=None, reports=None):
    entry = SimpleNamespace(enabled=True, kind="lora", axis="ip", path=path)
    return AutoAlpha(
        "anima",
        {path: [7, 8]} if triggers is None else triggers,
        3,
        1,
        chain=[entry],
        clip=clip,
        denoiser=model,
        reports=reports,
    )


def _conditioning(path="cat.safetensors", baked_path=None, ids=(1, 2, 0)):
    return {
        "t5xxl_ids": torch.tensor(ids),
        "kilora_mentions": [
            {
                "name": "cat",
                "path": path,
                "baked_path": baked_path,
                "trigger_words": ["cat"],
            }
        ],
    }


def test_mentions_cache_per_file_and_encoder_and_report_source_names():
    model, clip, reports = _model(), ToyClip(), []
    auto = _auto(clip, model, reports=reports)
    first, message = auto._collector(_conditioning())
    assert message is None
    assert isinstance(first, ProbeAttentionMaps)
    assert first.probes["cat.safetensors"][1] == 2
    second, _ = auto._collector(_conditioning())
    assert second.probes["cat.safetensors"] is first.probes["cat.safetensors"]
    assert len(clip.encoded) == 1
    assert "cat.safetensors" in auto.report and "probe keys" in auto.report
    assert reports == [auto.report]
    other_clip = ToyClip()
    _auto(other_clip, model)._collector(_conditioning())
    assert len(other_clip.encoded) == 1


def test_baked_path_resolves_trigger_ids_from_source(monkeypatch):
    monkeypatch.setattr(auto_alpha, "_metadata", lambda path: {})
    monkeypatch.setattr(auto_alpha, "trigger_words_of", lambda path: ["cat"])
    auto = _auto(ToyClip(), _model(), "baked.safetensors", triggers={})
    collector, message = auto._collector(_conditioning(baked_path="baked.safetensors"))
    assert message is None
    assert isinstance(collector, ProbeAttentionMaps)
    assert set(collector.probes) == {"baked.safetensors"}
    assert "cat.safetensors" in auto.report


def test_trigger_in_prompt_keeps_existing_positions_without_encoding():
    clip = ToyClip()
    auto = _auto(clip, _model())
    collector, message = auto._collector(_conditioning(ids=(1, 7, 8, 0)))
    assert message is None
    assert type(collector) is AttentionMaps
    assert collector.positions == {"cat.safetensors": (1, 2)}
    assert clip.encoded == []
    assert auto.report == ""


def test_without_mentions_matches_existing_collector_and_alpha_bit_for_bit():
    torch.manual_seed(716)
    clip, model = ToyClip(), _model()
    auto = _auto(clip, model)
    reference = AutoAlpha("anima", {"cat.safetensors": [7, 8]}, 3, 1)
    q, k, v = (torch.randn(1, 2, count, 2) for count in (4, 5, 5))
    seen = []

    def invoke(args):
        seen.append({name: value.clone() for name, value in current_axis_alphas().items()})
        options = args["c"]["transformer_options"]
        options["patches"]["attn2_patch"][-1](q, k, v, extra_options={"block_index": 1})
        return options["optimized_attention_override"](_attention, q, k, v, 2, skip_reshape=True)

    for step in (3, 2):
        args = {
            "input": torch.ones(1, 1, 2, 2),
            "timestep": torch.tensor([step]),
            "c": {"t5xxl_ids": torch.tensor([1, 7, 8, 0, 0])},
        }
        expected, actual = reference.call(args, invoke), auto.call(args, invoke)
        assert torch.equal(actual, expected)
        assert seen[-1].keys() == seen[-2].keys()
        assert all(torch.equal(seen[-1][axis], seen[-2][axis]) for axis in seen[-1])
        maps_a, maps_b = reference.collector.finish_step(), auto.collector.finish_step()
        assert all(torch.equal(maps_a[name], maps_b[name]) for name in maps_a)
        for item, maps in ((reference, maps_a), (auto, maps_b)):
            item.collector._sums = maps
            item.collector._counts = {name: 1 for name in maps}
    assert clip.encoded == []


def test_unmentioned_missing_trigger_and_invalid_prompt_ids_disable_collection():
    auto = _auto(ToyClip(), _model())
    conditioning = _conditioning(path="another.safetensors")
    collector, message = auto._collector(conditioning)
    assert collector is None and "not in the prompt" in message
    conditioning = _conditioning()
    conditioning.pop("t5xxl_ids")
    collector, message = auto._collector(conditioning)
    assert collector is None and "off" in message
    assert auto.clip.encoded == []


def test_mixed_prompt_and_probe_entries_preserve_existing_map():
    torch.manual_seed(717)
    model = _model()
    collector = _collector(model, torch.randn(1, 2, 3), {"dog": (1, 2)})
    reference = AttentionMaps({"dog": (1, 2)}, 3, block=1)
    q, k = torch.randn(1, 2, 4, 2), torch.randn(1, 2, 5, 2)
    collector.collect(q, k, 2, True)
    reference.collect(q, k, 2, True)
    maps = collector.finish_step()
    assert set(maps) == {"cat", "dog"}
    assert torch.equal(maps["dog"], reference.finish_step()["dog"])


def test_sampler_metadata_drives_probe_maps_and_previous_step_alpha():
    torch.manual_seed(718)
    clip, model = ToyClip(), _model()
    auto = _auto(clip, model)
    q, k, v = (torch.randn(1, 2, count, 2) for count in (4, 5, 5))
    seen = []

    def sample(args):
        def invoke(call):
            seen.append(dict(current_axis_alphas()))
            options = call["c"]["transformer_options"]
            options["patches"]["attn2_patch"][-1](q, k, v, extra_options={"block_index": 1})
            return options["optimized_attention_override"](
                _attention, q, k, v, 2, skip_reshape=True
            )

        return auto.call({"input": args["input"], "timestep": args["sigma"], "c": {}}, invoke)

    options = {"sampler_calc_cond_batch_function": sample}
    auto_alpha.install_sampler_hook(options)
    hook = options["sampler_calc_cond_batch_function"]
    for step in (3, 2):
        actual = hook(
            {
                "conds": [[_conditioning()], None],
                "input": torch.ones(1, 1, 2, 2),
                "sigma": torch.tensor([step]),
            }
        )
        assert torch.equal(actual, _attention(q, k, v, 2))
    assert seen[0] == {}
    assert set(seen[1]) == {"ip", "creative"}
    assert len(auto.stats) == 1
    assert len(clip.encoded) == 1
    assert "cat.safetensors" in auto.report
    assert auto_alpha._conditioning.get() is None


def test_adapter_runs_once_for_cached_context_and_end_token_is_not_a_probe():
    torch.manual_seed(719)
    model = _model()
    calls = []

    def adapt(tensor, ids, t5xxl_weights=None):
        calls.append(ids.clone())
        return tensor + 0.25

    model.preprocess_text_embeds = adapt
    embedding = torch.randn(1, 3, 3)
    caption = [[embedding, {"t5xxl_ids": torch.tensor([7, 8, 0])}]]
    collector = ProbeAttentionMaps({}, 3, {"cat": (caption, 2)}, model)
    collector.block = 1
    q, k = torch.randn(1, 2, 4, 2), torch.randn(1, 2, 5, 2)
    collector.collect(q, k, 2, True)
    collector.collect(q, k, 2, True)
    assert len(calls) == 1
    reference = _collector(model, (embedding + 0.25)[:, :2])
    reference.collect(q, k, 2, True)
    torch.testing.assert_close(collector.finish_step()["cat"], reference.finish_step()["cat"])


def test_failed_projection_leaves_attention_output_unchanged_and_no_partial_maps():
    model = _model()
    collector = _collector(model, torch.randn(1, 2, 3), {"dog": (1,)})
    model.blocks[1].cross_attn.k_proj = nn.Linear(9, 4)
    q, k, v = (torch.randn(1, 2, count, 2) for count in (4, 5, 5))
    collector.kind_patches()["attn2_patch"](q, k, v, extra_options={"block_index": 1})
    actual = collector.comfy_override()(_attention, q, k, v, 2, skip_reshape=True)
    assert torch.equal(actual, _attention(q, k, v, 2))
    assert collector.failed
    assert collector.finish_step() == {}


def test_encoder_failure_is_conservative_and_does_not_change_output():
    class BrokenClip(ToyClip):
        def encode_from_tokens_scheduled(self, tokens):
            raise ValueError("unavailable encoder")

    auto = _auto(BrokenClip(), _model())
    x = torch.randn(1, 1, 2, 2)
    args = {"input": x, "timestep": torch.tensor([3]), "c": _conditioning()}
    assert torch.equal(auto.call(args, lambda call: call["input"] * 0.7), x * 0.7)
    assert auto.collector is None
    assert auto.alphas == {}


def test_baked_mention_does_not_drop_another_entries_prompt_trigger(monkeypatch):
    monkeypatch.setattr(auto_alpha, "_metadata", lambda path: {})
    monkeypatch.setattr(auto_alpha, "trigger_words_of", lambda path: ["cat"])
    clip, model = ToyClip(), _model()
    auto = _auto(clip, model, "baked.safetensors", triggers={})
    auto.chain.append(SimpleNamespace(enabled=True, kind="lora", axis="ip", path="dog.safetensors"))
    collector, message = auto._collector(
        _conditioning(baked_path="baked.safetensors", ids=(1, 7, 8, 0))
    )
    assert message is None
    assert type(collector) is AttentionMaps
    assert set(collector.positions) == {"baked.safetensors", "dog.safetensors"}
    assert clip.encoded == []
