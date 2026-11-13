# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time

import pytest
import torch
from safetensors.torch import load_file, save_file

from kilora.library import LibraryIndex
from kilora.library.content import _layer_rows, _svd_rows, extract_content


def test_truncated_directions_match_full_svd_and_preserve_rng():
    generator = torch.Generator().manual_seed(123)
    left, _ = torch.linalg.qr(torch.randn(100, 24, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(128, 24, generator=generator))
    spectrum = torch.tensor([12.0, 10.0, 8.0, 6.0] + [0.1] * 20)
    matrix = (left * spectrum) @ right.T
    _, expected_values, expected = torch.linalg.svd(matrix, full_matrices=False)
    state = torch.random.get_rng_state().clone()
    directions, values = _svd_rows(matrix, 4)
    assert torch.equal(torch.random.get_rng_state(), state)
    again, _ = _svd_rows(matrix, 4)
    assert torch.equal(directions, again)
    residual = directions - (directions @ expected[:4].T) @ expected[:4]
    assert torch.linalg.matrix_norm(residual, ord=2) < 1e-3
    torch.testing.assert_close(values, expected_values[:4], rtol=1e-4, atol=0)


def test_batched_rows_match_individual_layers():
    torch.manual_seed(22)
    layers = {
        str(i): {
            "down": torch.randn(3, 16, dtype=torch.float16),
            "up": torch.randn(8, 3, dtype=torch.float16),
            "alpha": torch.tensor(float(i + 1)),
        }
        for i in range(5)
    }
    expected = torch.cat([_layer_rows(parts, 4) for parts in layers.values()])
    _, values, right = torch.linalg.svd(expected, full_matrices=False)
    result = extract_content(layers, dict.fromkeys(layers, "g"), {"g": 16})
    torch.testing.assert_close(result.singular_values["g"], values[:4])
    torch.testing.assert_close(result.directions["g"].abs(), right[:4].abs(), atol=1e-3, rtol=1e-3)


def test_old_index_recomputes_content_once(tmp_path, monkeypatch):
    import kilora.library.index as index_module

    folder = tmp_path / "loras"
    folder.mkdir()
    save_file(
        {
            "layers.0.attention.to_q.lora_down.weight": torch.randn(2, 3),
            "layers.0.attention.to_q.lora_up.weight": torch.randn(4, 2),
        },
        str(folder / "one.safetensors"),
    )
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    digest = next(iter(index.records))
    file = index.path / "library.json"
    data = json.loads(file.read_text())
    del data["records"][digest]["content_version"]
    file.write_text(json.dumps(data))
    original = index_module._content
    calls = []

    def content(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(index_module, "_content", content)
    index = LibraryIndex.open(index.path)
    index.sync([folder])
    assert len(calls) == 1
    assert index.records[digest]["content_version"] == 2
    assert load_file(str(index.path / "features.safetensors"))[f"{digest}/summary"].numel()
    LibraryIndex.open(index.path).sync([folder])
    assert len(calls) == 1


@pytest.mark.perf
def test_anima_sized_sync_speed(tmp_path, monkeypatch):
    if not os.environ.get("KILORA_PERF"):
        pytest.skip("timing check; set KILORA_PERF=1 to run it")
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("requires two CPU threads")
    import kilora.library.index as index_module

    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        generator = torch.Generator().manual_seed(42)
        shared = [
            {
                "down": torch.randn(32, 2048, generator=generator, dtype=torch.float16),
                "up": torch.randn(2048, 32, generator=generator, dtype=torch.float16),
            }
            for _ in range(30)
        ]
        layers = {f"{g:02}-{i:02}": shared[i] for g in range(76) for i in range(30)}
        groups = {name: name[:2] for name in layers}
        widths = {f"{g:02}": 2048 for g in range(76)}
        folder = tmp_path / "loras"
        folder.mkdir()
        for i in range(10):
            save_file(
                {
                    "layers.0.attention.to_q.lora_down.weight": torch.full((1, 2), float(i)),
                    "layers.0.attention.to_q.lora_up.weight": torch.ones(2, 1),
                },
                str(folder / f"{i}.safetensors"),
            )
        monkeypatch.setattr(
            index_module, "_content", lambda *args: extract_content(layers, groups, widths)
        )
        start = time.monotonic()
        report = LibraryIndex.open(tmp_path / "index").sync([folder])
        per_file = (time.monotonic() - start) / 10
        print(f"synthetic sync seconds per file: {per_file:.3f}")
        assert report.added == 10 and not report.skipped and not report.reasons
        assert per_file < 3
    finally:
        torch.set_num_threads(original_threads)


def test_failed_upgrade_keeps_content_and_retries(tmp_path, monkeypatch):
    import kilora.library.index as index_module

    folder = tmp_path / "loras"
    folder.mkdir()
    save_file(
        {
            "layers.0.attention.to_q.lora_down.weight": torch.randn(2, 3),
            "layers.0.attention.to_q.lora_up.weight": torch.randn(4, 2),
        },
        str(folder / "one.safetensors"),
    )
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    digest = next(iter(index.records))
    before = load_file(str(index.path / "features.safetensors"))[f"{digest}/summary"].clone()
    index.records[digest].pop("content_version")
    original = index_module._content

    def fail(*args):
        raise RuntimeError("temporary extraction failure")

    monkeypatch.setattr(index_module, "_content", fail)
    report = index.sync([folder])
    assert any("temporary extraction failure" in reason for reason in report.reasons)
    assert "content_version" not in index.records[digest]
    torch.testing.assert_close(
        load_file(str(index.path / "features.safetensors"))[f"{digest}/summary"], before
    )
    monkeypatch.setattr(index_module, "_content", original)
    index.sync([folder])
    assert index.records[digest]["content_version"] == 2
