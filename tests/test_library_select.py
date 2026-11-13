# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file

from kilora.library import LibraryIndex, retrieve, select, text


def _write(path, row):
    save_file(
        {
            "layers.0.attention.to_q.lora_down.weight": torch.tensor([row]),
            "layers.0.attention.to_q.lora_up.weight": torch.ones(2, 1),
        },
        str(path),
    )


class _Clip:
    def tokenize(self, text):
        return text

    def encode_from_tokens(self, tokens, return_pooled=False):
        return torch.zeros(1, 1, 3), torch.tensor([[1.0, 2.0, 3.0]])


def test_select_sync_and_fallback(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    _write(folder / "red_fox.safetensors", [1.0, 0.0, 0.0])
    _write(folder / "blue_ocean.safetensors", [0.0, 1.0, 0.0])
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    groups = index.retrieval_inputs("z-image")["groups"]
    probe = {group: torch.tensor([[1.0, 0.0, 0.0]]) for group in groups}
    chosen = select(index, "red fox", probe, "z-image", 1)
    assert chosen[0].path == str(folder / "red_fox.safetensors")
    assert chosen[0].sha256 in index.records
    assert not chosen[0].text_fallback
    assert len(select(index, "red fox", probe, "z-image", 10, min_relevance=0)) == 2
    assert all(
        x.text_fallback
        for x in select(index, "red fox", probe, "z-image", 10, clip=_Clip(), min_relevance=0)
    )
    (folder / "red_fox.safetensors").unlink()
    _write(folder / "green_forest.safetensors", [1.0, 0.0, 0.0])
    index.sync([folder])
    chosen = select(index, "green forest", probe, "z-image", 10, min_relevance=0)
    assert {x.path for x in chosen} == {
        str(folder / "blue_ocean.safetensors"),
        str(folder / "green_forest.safetensors"),
    }
    assert chosen[0].path.endswith("green_forest.safetensors")


def test_feature_file_cached_until_sync(tmp_path, monkeypatch):
    import kilora.library.index as module

    folder = tmp_path / "loras"
    folder.mkdir()
    _write(folder / "one.safetensors", [1.0, 0.0, 0.0])
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    original = module.load_file
    reads = []

    def counted(path, *args, **kwargs):
        if str(path).endswith("features.safetensors"):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module, "load_file", counted)
    index.retrieval_inputs("z-image")
    index.retrieval_inputs("z-image", _Clip())
    assert len(reads) <= 1
    index.sync([folder])
    reads.clear()
    index.retrieval_inputs("z-image")
    index.retrieval_inputs("z-image")
    assert len(reads) <= 1


def test_mixed_encoders_and_reopened_index(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    _write(folder / "red_fox.safetensors", [1.0, 0.0, 0.0])
    clip = _Clip()
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder], clip=clip)
    _write(folder / "blue_ocean.safetensors", [0.0, 1.0, 0.0])
    index.sync([folder])
    index = LibraryIndex.open(index.path)
    chosen = select(index, "blue ocean", {}, "z-image", 10, clip=clip, min_relevance=0)
    by_name = {x.path.rsplit("/", 1)[-1]: x for x in chosen}
    assert not by_name["red_fox.safetensors"].text_fallback
    assert by_name["blue_ocean.safetensors"].text_fallback
    assert by_name["blue_ocean.safetensors"].score > 0


def test_region_selection_and_global_compatibility(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    _write(folder / "one.safetensors", [1.0, 0.0, 0.0])
    _write(folder / "two.safetensors", [0.0, 1.0, 0.0])
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    inputs = dict(index.retrieval_inputs("z-image"))
    fallback = inputs.pop("fallback")
    probe = {
        group: torch.tensor([[[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, 1.0, 0]]])
        for group in inputs["groups"]
    }
    masks = {"a": torch.tensor([[1.0, 0], [1, 0]]), "b": torch.tensor([[0.0, 1], [0, 1]])}
    chosen = select(index, "", probe, "z-image", 1, regions=masks)
    assert {item.path.rsplit("/", 1)[-1]: item.regions for item in chosen} == {
        "one.safetensors": ("a",),
        "two.safetensors": ("b",),
    }
    both = select(index, "", probe, "z-image", 1, regions={"a": masks["a"], "c": masks["a"]})
    assert len(both) == 1 and both[0].regions == ("a", "c")
    assert select(index, "", probe, "z-image", 1, regions={"empty": torch.zeros(2, 2)}) == []

    query = text.embed_text("")
    original = retrieve(
        **inputs,
        text_query=query,
        cluster_query=query,
        fallback_query=query,
        fallback_digests=fallback,
        probe=probe,
        family="z-image",
        k=2,
    )
    global_results = select(index, "", probe, "z-image", 2)
    assert [(r.sha256, r.score) for r in global_results] == [(r.digest, r.score) for r in original]
    assert all(not result.regions for result in global_results)


def test_region_probe_weights_partial_patches_and_skips_non_image_tokens():
    from kilora.library import _region_probe
    from kilora.library.retrieve import _content_share

    probe = {
        "image": torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]]),
        "text": torch.ones(3, 2),
    }
    masked = _region_probe(probe, torch.tensor([[1.0, 0.25], [1.0, 0.25]]))
    assert "text" not in masked
    assert _content_share(torch.tensor([1.0, 0]), masked, {"image": 2}) == pytest.approx(0.8)
