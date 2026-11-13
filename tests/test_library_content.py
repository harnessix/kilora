# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import gzip
import json

import torch
from torch.profiler import ProfilerActivity, profile

from kilora.library.content import extract_content, lokr_directions


def test_planted_read_subspace_and_summary():
    torch.manual_seed(11)
    q, _ = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64))
    down = q.T.float()
    layers = {"a": {"down": down, "up": torch.diag(torch.tensor([4.0, 2.0, 1.0]))}}
    result = extract_content(layers, {"a": "attention"}, {"attention": 12, "mlp": 12}, rank=3)
    actual = result.directions["attention"].double()
    assert torch.linalg.matrix_norm(actual - (actual @ q) @ q.T, ord=2) < 1e-4
    torch.testing.assert_close(result.singular_values["attention"], torch.tensor([4.0, 2.0, 1.0]))
    torch.testing.assert_close(result.summary[:12], actual[0].float() * 4)
    assert not result.summary[12:].any()


def test_lokr_matches_full_svd():
    torch.manual_seed(5)
    a, b = torch.randn(5, 4), torch.randn(6, 3)
    d, s = lokr_directions(a, b, 4)
    _, expected_s, expected_d = torch.linalg.svd(torch.kron(a, b), full_matrices=False)
    torch.testing.assert_close(s, expected_s[:4], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(d.abs(), expected_d[:4].abs(), rtol=1e-4, atol=1e-5)


def test_lokr_does_not_allocate_full_product(tmp_path):
    torch.manual_seed(8)
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True
    ) as prof:
        lokr_directions(a, b, 4)
    timeline = tmp_path / "memory.raw.json.gz"
    prof.export_memory_timeline(str(timeline))
    with gzip.open(timeline, "rt") as stream:
        events = json.load(stream)
    current = peak = 0
    for _, _, size, _ in events:
        current += size
        peak = max(peak, current)
    assert peak < 4 * (a.numel() + b.numel()) * a.element_size()


def test_group_stacks_layers_and_handles_loha_and_lokr():
    layers = {
        "a": {"down": torch.tensor([[1.0, 0.0]]), "up": torch.tensor([[2.0]])},
        "b": {"down": torch.tensor([[0.0, 1.0]]), "up": torch.tensor([[3.0]])},
        "c": {"lokr_w1": torch.eye(1), "lokr_w2": torch.diag(torch.tensor([4.0, 1.0]))},
        "d": {
            "hada_w1_a": torch.eye(2),
            "hada_w1_b": torch.eye(2),
            "hada_w2_a": torch.eye(2),
            "hada_w2_b": torch.eye(2),
        },
    }
    result = extract_content(layers, dict.fromkeys(layers, "g"), {"g": 2}, rank=2)
    torch.testing.assert_close(result.singular_values["g"], torch.tensor([21.0, 11.0]).sqrt())


def test_presets_group_every_target_and_split_widths():
    from kilora.families import FAMILIES
    from kilora.presets import PRESETS

    for family, preset in PRESETS.items():
        if FAMILIES[family].root == "text_encoder":
            continue
        for target in preset.full_target_modules:
            layer = target.replace("*", "0")
            assert preset.content_group(layer, 7) is not None, (family, layer)
            assert preset.content_group(layer, 7) != preset.content_group(layer, 8)


def test_index_adds_and_removes_content_and_reports_unmatched(tmp_path):
    from safetensors.torch import load_file, save_file

    from kilora.library import LibraryIndex

    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "one.safetensors"
    save_file(
        {
            "layers.0.attention.to_q.lora_down.weight": torch.ones(2, 3),
            "layers.0.attention.to_q.lora_up.weight": torch.ones(4, 2),
        },
        str(path),
    )
    index = LibraryIndex.open(tmp_path / "index")
    report = index.sync([folder])
    assert report.added == 1
    digest = next(iter(index.records))
    features = load_file(str(index.path / "features.safetensors"))
    assert features[f"{digest}/summary"].numel() == 3
    assert any(k.endswith("/directions") for k in features)
    path.unlink()
    index.sync([folder])
    assert not load_file(str(index.path / "features.safetensors"))
    save_file(
        {"other.lora_down.weight": torch.ones(2, 3), "other.lora_up.weight": torch.ones(4, 2)},
        str(path),
    )
    report = index.sync([folder])
    digest = next(iter(index.records))
    assert report.added == 1 and report.skipped == 0
    assert any("no layers matched" in reason for reason in report.reasons)
    assert load_file(str(index.path / "features.safetensors"))[f"{digest}/summary"].numel() == 0


def test_flux_depth_bands_cover_both_block_lists():
    from kilora.families import FAMILIES
    from kilora.presets import PRESETS

    preset = PRESETS["flux"]
    double = {preset.content_group(f"double_blocks.{i}.img_attn.qkv", 4)[0] for i in range(19)}
    single = {preset.content_group(f"single_blocks.{i}.linear1", 4)[0] for i in range(38)}
    assert double == {"0", "1"}
    assert single == {"1", "2", "3"}
