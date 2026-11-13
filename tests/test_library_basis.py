# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from safetensors.torch import save_file

from kilora.library import basis as builder
from kilora.library.index import LibraryIndex
from kilora.multi_import import lycoris_delta

STEM = "layers.0.attention.to_q"


def write_lora(path, up, down, metadata=None, extra=None):
    tensors = {STEM + ".lora_up.weight": up, STEM + ".lora_down.weight": down}
    tensors.update(extra or {})
    save_file(tensors, str(path), metadata=metadata)


def indexed(tmp_path, folder):
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    return index


def test_span_sources_and_persistence(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "one.safetensors"
    up = torch.tensor([[1.0], [2.0], [3.0]])
    down = torch.tensor([[2.0, -1.0]])
    write_lora(path, up, down, {"license": "Apache-2.0"})
    index = indexed(tmp_path, folder)
    result = builder.build_basis(index)
    mount = result.mounts[STEM]
    target = up @ down
    projected = mount.u @ (mount.u.T @ target @ mount.v) @ mount.v.T
    assert torch.allclose(projected.square().sum() / target.square().sum(), torch.tensor(1.0))
    assert mount.output_sources[0][0]["file"] == str(path)
    assert mount.output_sources[0][0]["weight"] == 1.0
    loaded = builder.load_basis(index)
    assert torch.equal(loaded.mounts[STEM].u, mount.u)


def test_lokr_delta(tmp_path):
    path = tmp_path / "k.safetensors"
    parts = {"lokr_w1": torch.tensor([[1.0, 2.0]]), "lokr_w2": torch.tensor([[2.0], [3.0]])}
    save_file({STEM + "." + k: v for k, v in parts.items()}, str(path))
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        actual = builder._delta(file, {k: STEM + "." + k for k in parts}, "cpu")
    assert torch.equal(actual, lycoris_delta(parts)[0])


def test_license_and_text_keys(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    up, down = torch.ones(2, 1), torch.ones(1, 2)
    write_lora(
        folder / "plain.safetensors",
        up,
        down,
        extra={
            "lora_te_x.lora_up.weight": up.clone(),
            "text_encoder.x.lora_down.weight": down.clone(),
        },
    )
    write_lora(folder / "restricted.safetensors", up, down, {"usage": "No derivatives"})
    index = indexed(tmp_path, folder)
    result = builder.build_basis(index)
    assert result.files[str(folder / "plain.safetensors")]["text_encoder_keys"] == 2
    assert result.skipped[0]["file"] == str(folder / "restricted.safetensors")
    assert result.flagged[0]["file"] == str(folder / "plain.safetensors")


def test_update_reads_only_new_and_matches_rebuild(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    write_lora(folder / "a.safetensors", torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]]))
    index = indexed(tmp_path, folder)
    original = builder.build_basis(index)
    write_lora(folder / "b.safetensors", torch.tensor([[0.0], [2.0]]), torch.tensor([[0.0, 1.0]]))
    index.sync([folder])
    read = builder._read
    seen = []

    def spy(path, *args):
        seen.append(path.name)
        return read(path, *args)

    monkeypatch.setattr(builder, "_read", spy)
    updated = builder.update_basis(builder.load_basis(index), index)
    assert seen == ["b.safetensors"]
    rebuilt = builder.build_basis(index)
    for attr in ("u", "v"):
        a, b = getattr(updated.mounts[STEM], attr), getattr(rebuilt.mounts[STEM], attr)
        assert torch.allclose(a @ a.T, b @ b.T, atol=1e-6)
    assert original.files.keys() < updated.files.keys()


def test_bad_files_are_skipped_and_progress_is_logged(tmp_path, caplog):
    folder = tmp_path / "loras"
    folder.mkdir()
    up, down = torch.ones(8, 1), torch.ones(1, 8)
    good = folder / "good.safetensors"
    bad = folder / "unsupported.safetensors"
    corrupt = folder / "corrupt.safetensors"
    write_lora(good, up, down)
    write_lora(bad, up, down, extra={"unexpected.weight": torch.ones(1)})
    write_lora(corrupt, up, down)
    index = indexed(tmp_path, folder)
    corrupt.write_bytes(b"not safetensors")
    with caplog.at_level("INFO", logger=builder.__name__):
        result = builder.build_basis(index)
    skipped = {entry["file"]: entry["reason"] for entry in result.skipped}
    assert set(skipped) == {str(bad), str(corrupt)}
    assert "unexpected.weight" in skipped[str(bad)]
    assert skipped[str(corrupt)]
    assert STEM in result.mounts
    assert all(path.name in caplog.text for path in (good, bad, corrupt))


def test_factorized_and_truncated_directions_match_full_svd(tmp_path):
    torch.manual_seed(42)
    top = 3
    up, down = torch.randn(64, 5), torch.randn(5, 72)
    plain = tmp_path / "plain.safetensors"
    write_lora(plain, up, down, extra={STEM + ".alpha": torch.tensor(2.5)})
    a = torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]))
    b = torch.diag(
        torch.tensor([8.0, 0.02, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001])
    )
    other = tmp_path / "other.safetensors"
    save_file({STEM + ".lokr_w1": a, STEM + ".lokr_w2": b}, str(other))
    for path, target in ((plain, up @ down * 0.5), (other, torch.kron(a, b))):
        parts, _, _, _ = builder._read(path, top, "cpu")
        left, right = parts[STEM]
        u, _, vh = torch.linalg.svd(target, full_matrices=False)
        for weighted, expected in ((left, u[:, :top]), (right, vh[:top].T)):
            actual, _ = torch.linalg.qr(weighted)
            assert torch.linalg.matrix_norm(actual @ actual.T - expected @ expected.T) < 1e-4


def test_svd_paths_are_saved_per_file_and_mount(tmp_path, caplog):
    folder = tmp_path / "loras"
    folder.mkdir()
    plain = folder / "plain.safetensors"
    other = folder / "other.safetensors"
    write_lora(plain, torch.ones(64, 1), torch.ones(1, 64))
    save_file(
        {
            STEM + ".lokr_w1": torch.eye(8),
            STEM + ".lokr_w2": torch.eye(8),
            "layers.1.attention.to_q.lokr_w1": torch.eye(2),
            "layers.1.attention.to_q.lokr_w2": torch.eye(2),
        },
        str(other),
    )
    index = indexed(tmp_path, folder)
    with caplog.at_level("INFO", logger=builder.__name__):
        builder.build_basis(index, top=2)
    loaded = builder.load_basis(index)
    assert loaded.files[str(plain)]["svd_paths"] == {STEM: "low_rank_qr"}
    assert loaded.files[str(other)]["svd_paths"] == {
        STEM: "svd_lowrank",
        "layers.1.attention.to_q": "full_svd",
    }
    assert "low_rank_qr=1" in caplog.text
    assert "svd_lowrank=1, full_svd=1" in caplog.text
