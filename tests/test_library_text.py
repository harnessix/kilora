# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json

import torch
from safetensors.torch import save_file

from kilora.library.text import (
    TextFeatureStore,
    embed_text,
    encoder_identity,
    read_text_features,
)


def test_metadata_and_sidecars(tmp_path):
    path = tmp_path / "red_fox.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        str(path),
        metadata={
            "modelspec.trigger_phrase": "foxperson, red fox",
            "ss_tag_frequency": json.dumps({"a": {"fur": 10, "forest": 4}, "b": {"forest": 9}}),
            "ss_output_name": "portrait",
        },
    )
    path.with_suffix(".civitai.info").write_text(
        json.dumps({"trainedWords": ["foxperson", "special word"], "tags": ["animal"]}),
        encoding="utf-8",
    )
    features = read_text_features(path)
    assert features.trigger_words == ("foxperson", "red fox", "special word")
    assert features.tags == ("forest", "fur", "portrait", "animal")
    assert features.filename_words == ("red", "fox")


def test_sidecar_formats_and_top_twenty(tmp_path):
    path = tmp_path / "subject_name.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        str(path),
        metadata={"ss_tag_frequency": json.dumps({"set": {f"tag{i}": i for i in range(1, 26)}})},
    )
    path.with_suffix(".txt").write_text("extra, words", encoding="utf-8")
    path.with_suffix(".json").write_text('{"trainedWords": ["trigger"]}', encoding="utf-8")
    features = read_text_features(path)
    assert len(features.tags) == 20
    assert features.tags[0] == "tag25"
    assert features.tags[-1] == "tag6"
    assert features.trigger_words == ("trigger",)
    assert "subject" in features.text() and "name" in features.text()


def test_fallback_deterministic_and_similar():
    first = embed_text("watercolor forest fox")
    assert torch.equal(first, embed_text("watercolor forest fox"))
    assert torch.allclose(first.norm(), torch.tensor(1.0))
    assert torch.dot(first, embed_text("watercolor forest wolf")) > torch.dot(
        first, embed_text("chrome spaceship engine")
    )
    assert torch.equal(embed_text(""), torch.zeros(512))


class FakeClip:
    def __init__(self, count=3):
        self.cond_stage_model = torch.nn.Linear(count, 1, bias=False)
        self.text = None

    def tokenize(self, text):
        self.text = text
        return "tokens"

    def encode_from_tokens(self, tokens, return_pooled=False):
        assert tokens == "tokens" and return_pooled
        return torch.zeros(1, 2, 3), torch.tensor([[1.0, 2.0, 3.0]])


def test_clip_and_per_encoder_store(tmp_path):
    clip = FakeClip()
    other = FakeClip(4)
    vector = embed_text("forest", clip)
    assert clip.text == "forest"
    assert torch.equal(vector, torch.tensor([1.0, 2.0, 3.0]))
    assert encoder_identity(clip) != encoder_identity(other)
    store = TextFeatureStore(tmp_path / "features.safetensors")
    store.put("abc", vector, clip)
    store.put("abc", embed_text("forest"))
    store.save()
    loaded = TextFeatureStore(store.path)
    assert torch.equal(loaded.get("abc", clip).float(), vector)
    assert loaded.get("abc").shape == (512,)
    assert loaded.get("abc", other) is None


class MeanClip(FakeClip):
    def tokenize(self, text):
        return {"attention_mask": torch.tensor([[1, 1, 0]])}

    def encode_from_tokens(self, tokens, return_pooled=False):
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]]), None


def test_mean_encoder_and_cache():
    clip = MeanClip()
    assert torch.equal(embed_text("test", clip), torch.tensor([2.0, 3.0]))
    assert encoder_identity(clip).endswith("-mean")
    assert not encoder_identity(FakeClip()).endswith("-mean")
    clip.cond_stage_model.parameters = lambda: (_ for _ in ()).throw(AssertionError("recounted"))
    assert encoder_identity(clip).endswith("-mean")


def test_trigger_txt_and_index_vectors(tmp_path):
    from safetensors.torch import load_file

    from kilora.library import LibraryIndex

    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "red_fox.safetensors"
    save_file(
        {
            "layers.0.attention.to_q.lora_down.weight": torch.ones(2, 3),
            "layers.0.attention.to_q.lora_up.weight": torch.ones(4, 2),
        },
        str(path),
    )
    path.with_suffix(".txt").write_text("trigger: foxperson, red fox\nfur, forest")
    index = LibraryIndex.open(tmp_path / "index")
    clip = MeanClip()
    index.sync([folder], clip=clip)
    digest = next(iter(index.records))
    assert index.records[digest]["trigger_words"] == ["foxperson", "red fox"]
    assert index.records[digest]["tags"] == ["fur", "forest"]
    store = TextFeatureStore(index.path / "features.safetensors")
    assert torch.equal(store.get(digest, clip).float(), torch.tensor([2.0, 3.0]))
    assert store.get(digest) is not None
    path.unlink()
    index.sync([folder])
    assert not any(key.startswith(digest + "/") for key in load_file(str(store.path)))


def test_mean_encoder_without_mask_uses_prompt_tokens():
    class Unpooled(FakeClip):
        def tokenize(self, text):
            ids = [101, 11, 12, 102, 0, 0] if text else [101, 102, 0, 0, 0, 0]
            return {"llm": [[(token, 1.0) for token in ids]]}

        def encode_from_tokens(self, tokens, return_pooled=False):
            return torch.tensor([[[99.0], [2.0], [4.0], [80.0], [90.0], [90.0]]]), None

    assert torch.equal(embed_text("two words", Unpooled()), torch.tensor([3.0]))
