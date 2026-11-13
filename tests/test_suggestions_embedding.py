# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import torch
from test_comfyui import folders

from kilora.comfyui import nodes, patching
from kilora.suggestions import embedding_suggest


def test_embedding_scores_and_missing_candidates():
    vectors = {
        "a": torch.tensor([1.0, 0.0]),
        "near": torch.tensor([0.9, 0.1]),
        "direction": torch.tensor([1.0, 1.0]),
        "covered": torch.tensor([1.0, 0.0]),
    }
    installed = {name: name for name in ("a", "near", "direction", "covered", "missing")}
    singles = Counter({"a": 1, "missing": 1})
    pairs = Counter({("a", "missing"): 1})
    similar, missing = embedding_suggest(
        ["a"], installed, singles, pairs, vectors, ["a"], "similar"
    )
    assert missing and ("missing", 1.0) in similar
    assert dict(similar)["covered"] > dict(similar)["near"] > dict(similar)["direction"]
    complementary, _ = embedding_suggest(
        ["a"], installed, singles, Counter(), vectors, ["a"], "complementary"
    )
    assert complementary[0][0] == "direction"
    liked, _ = embedding_suggest(
        ["a"], installed, singles, Counter(), vectors, ["direction"], "near_liked"
    )
    assert liked[0][0] == "direction"


def test_node_liked_fallback_and_index_reuse(tmp_path, folders, monkeypatch):
    import kilora.library.index as index_module
    import kilora.references as references
    from kilora import palette, usage_history
    from kilora.key_cache import file_sha256
    from kilora.library import LibraryIndex

    paths = [folders["loras"] / (name + ".safetensors") for name in ("current", "liked", "near")]
    for path in paths:
        path.write_bytes(path.stem.encode())
    hashes = [file_sha256(path) for path in paths]
    vectors = dict(
        zip(
            hashes,
            (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.9])),
            strict=True,
        )
    )
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    monkeypatch.setattr(usage_history, "counts", lambda **kwargs: (Counter(), Counter()))

    class Index:
        records = {h: {"family": "test"} for h in hashes}

        def sync(self, *args, **kwargs):
            pass

        def embedding_inputs(self, family, model):
            return vectors

    Index.paths = {str(p.resolve()): h for p, h in zip(paths, hashes, strict=True)}
    monkeypatch.setattr(LibraryIndex, "open", lambda *args: Index())
    monkeypatch.setattr(nodes.FOLDERS, "dirs", lambda kind: [str(folders["loras"])])
    node = nodes.KiLoRASuggestions()
    chain = patching.ModChain((patching.ModEntry("lora", str(paths[0])),))
    assert node.show(chain)["result"][0] == node.show(chain, "unused", "co_usage")["result"][0]
    assert "using co-usage" in node.show(chain, "", "similar")["result"][0]
    assert "No kept images" in node.show(chain, "model", "near_liked")["result"][0]
    entry = palette.PaletteEntry("lora", paths[1].name, hashes[1])
    palette_path = tmp_path / "palettes" / "liked.kpalette.json"
    palette_path.parent.mkdir()
    palette.save_palette(palette_path, palette.Palette("liked", [entry]))
    state = palette.ShuffleState()
    palette.reward_shuffle(
        state,
        [palette.Resolved(entry, str(paths[1]))],
        (0, 1),
        "keep",
        {str(paths[1]): ("a" * 64, [0.0, 1.0])},
    )
    assert state.embedding

    palette.save_shuffle_state(palette_path, state)

    def fail(*args, **kwargs):
        raise AssertionError("LoRA read again")

    monkeypatch.setattr("kilora.key_cache.file_sha256", fail)
    monkeypatch.setattr(index_module, "_content", fail)
    monkeypatch.setattr(references, "_sha256", fail)
    report = node.show(chain, "model", "near_liked")["result"][0]
    assert report.splitlines()[0].startswith("liked.safetensors")
    assert "No kept images" not in report
    paths[2].unlink()
    assert "near.safetensors" not in node.show(chain, "model", "near_liked")["result"][0]


def test_real_index_reuses_cached_lora_reads(tmp_path, folders, monkeypatch):
    from safetensors.torch import save_file

    import kilora.library.index as index_module
    import kilora.references as references
    from kilora import palette
    from kilora.library import LibraryIndex, basis, coeffs, embed

    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(palette, "PALETTE_DIR", str(tmp_path / "palettes"))
    monkeypatch.setattr(nodes.FOLDERS, "dirs", lambda kind: [str(folders["loras"])])
    paths = [folders["loras"] / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        save_file(
            {
                "layers.0.attention.to_q.lora_down.weight": torch.tensor([[1.0, float(i)]]),
                "layers.0.attention.to_q.lora_up.weight": torch.ones(2, 1),
            },
            str(path),
        )
    index = LibraryIndex.open()
    index.sync([folders["loras"]])
    basis.build_basis(index, rank=1, top=1)
    model = embed.LayerChain(["layers.0.attention.to_q"], 1, d=4)
    model_path = tmp_path / "model"
    model.save(model_path)
    chain = patching.ModChain((patching.ModEntry("lora", str(paths[0])),))
    node = nodes.KiLoRASuggestions()
    assert paths[1].name in node.show(chain, str(model_path), "similar")["result"][0]

    def fail(*args, **kwargs):
        raise AssertionError("LoRA read again")

    monkeypatch.setattr(coeffs, "lora_coefficients", fail)
    monkeypatch.setattr(index_module, "_content", fail)
    monkeypatch.setattr(index_module, "read_text_features", fail)
    monkeypatch.setattr(references, "_sha256", fail)
    monkeypatch.setattr("kilora.key_cache.file_sha256", fail)
    assert paths[1].name in node.show(chain, str(model_path), "similar")["result"][0]
