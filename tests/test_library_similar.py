# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import shutil

import pytest
import torch
from safetensors.torch import save_file

from kilora.library import LibraryIndex


def _save(path, rows, stem="layers.0.attention.to_q"):
    down = torch.tensor(rows, dtype=torch.float32)
    save_file(
        {f"{stem}.lora_down.weight": down, f"{stem}.lora_up.weight": torch.ones(2, len(rows))},
        str(path),
    )


def test_similar_reuses_index_and_finds_other_rank(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    first, second = folder / "rank1.safetensors", folder / "rank2.safetensors"
    _save(first, [[2, 0, 0]])
    _save(second, [[2, 0, 0], [0, 0.1, 0]])
    for i in range(4):
        _save(folder / f"other{i}.safetensors", [[0, 1, i + 1]])
    _save(folder / "different.safetensors", [[2, 0, 0]], "lora_unet_layers_0_proj_ffn_up_gate")
    shutil.copyfile(first, folder / "duplicate.safetensors")
    index = LibraryIndex.open(tmp_path / "index")
    assert index.sync([folder]).skipped == 0
    import kilora.library.index as module
    import kilora.references as references

    def fail(*args, **kwargs):
        raise AssertionError("indexed weights were read again")

    monkeypatch.setattr(module, "_content", fail)
    monkeypatch.setattr(references, "_sha256", fail)
    index = LibraryIndex.open(index.path)
    for query, expected in ((first, second), (second, first)):
        results = index.similar(query, 3)
        assert str(expected) in [r.path for r in results] or (
            query == second and str(folder / "duplicate.safetensors") in [r.path for r in results]
        )
        assert results[0].score > 0.99
        assert all("different" not in r.path for r in results)
        assert all(r.sha256 != index.paths[str(query)] for r in results)
    assert index.similar(first, 0) == []
    with pytest.raises(ValueError):
        index.similar(first, -1)
    second.unlink()
    assert str(second) not in [r.path for r in index.similar(first, 10)]


def test_unindexed_and_empty_features(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "unknown.safetensors"
    _save(path, [[1, 0]], "unrecognized.0")
    index = LibraryIndex.open(tmp_path / "index")
    with pytest.raises(ValueError, match="sync"):
        index.similar(path, 1)
    index.sync([folder])
    assert index.similar(path, 1) == []


def test_node_syncs_folders_and_lists_scores(tmp_path, monkeypatch):
    from kilora.comfyui import nodes

    folder = tmp_path / "loras"
    folder.mkdir()
    _save(folder / "first.safetensors", [[1, 0]])
    _save(folder / "second.safetensors", [[1, 0], [0, 0.1]])
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(nodes.FOLDERS, "path", lambda _, name: str(folder / name))
    monkeypatch.setattr(
        nodes.FOLDERS,
        "_fp",
        lambda: type("Folders", (), {"get_folder_paths": lambda self, kind: [str(folder)]})(),
    )
    node = nodes.KiLoRASimilarLoRAs()
    result = node.find("first.safetensors", 3)
    assert "second.safetensors" in result["result"][0]
    assert "1.0000" in result["result"][0]
    assert result["ui"]["text"] == list(result["result"])


def test_index_is_opt_in_and_leaves_model_output_unchanged(tmp_path):
    from kilora.testing import TinyDenoiser

    folder = tmp_path / "loras"
    folder.mkdir()
    _save(folder / "one.safetensors", [[1, 0]])
    model = TinyDenoiser(seed=0).eval()
    x = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        before = model(x)
        LibraryIndex.open(tmp_path / "index").sync([folder])
        after = model(x)
    assert torch.equal(before, after)


def test_embedding_node_reuses_coefficients_and_falls_back(tmp_path, monkeypatch):
    import kilora.library.index as index_module
    import kilora.references as references
    from kilora.comfyui import nodes
    from kilora.library import basis, coeffs, embed

    folder = tmp_path / "loras"
    folder.mkdir()
    first, second = folder / "first.safetensors", folder / "second.safetensors"
    _save(first, [[1, 0]])
    _save(second, [[1, 0], [0, 0]])
    index_path = tmp_path / "index"
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(index_path))
    monkeypatch.setattr(nodes.FOLDERS, "path", lambda _, name: str(folder / name))
    monkeypatch.setattr(
        nodes.FOLDERS,
        "_fp",
        lambda: type("Folders", (), {"get_folder_paths": lambda self, kind: [str(folder)]})(),
    )
    index = LibraryIndex.open(index_path)
    index.sync([folder])
    built = basis.build_basis(index, rank=1, top=1)
    assert "layers.0.attention.to_q" in built.mounts
    torch.manual_seed(4)
    model = embed.LayerChain(["layers.0.attention.to_q"], 1, d=4)
    model_path = tmp_path / "model"
    model.save(model_path)
    node = nodes.KiLoRASimilarLoRAs()
    old = node.find(first.name, 3)["result"][0]
    assert old == node.find(first.name, 3, "keys", "unused")["result"][0]
    for query, expected in ((first, second), (second, first)):
        result = node.find(query.name, 3, "embedding", str(model_path))["result"][0]
        assert expected.name in result
        assert "1.0000" in result

    def fail(*args, **kwargs):
        raise AssertionError("LoRA read again")

    monkeypatch.setattr(coeffs, "lora_coefficients", fail)
    monkeypatch.setattr(index_module, "_content", fail)
    monkeypatch.setattr(index_module, "read_text_features", fail)
    monkeypatch.setattr(references, "_sha256", fail)
    assert second.name in node.find(first.name, 3, "embedding", str(model_path))["result"][0]
    with torch.no_grad():
        model.initial.add_(1)
    model.save(model_path)
    assert second.name in node.find(first.name, 3, "embedding", str(model_path))["result"][0]
    bad = tmp_path / "bad"
    embed.LayerChain(["wrong.mount"], 1, d=4).save(bad)
    corrupt = tmp_path / "corrupt"
    model.save(corrupt)
    (corrupt / "model.safetensors").write_bytes(b"invalid")
    for path in ("", str(bad), str(corrupt)):
        report = node.find(first.name, 3, "embedding", path)["result"][0]
        assert "using content keys" in report
        assert report.endswith(old)
