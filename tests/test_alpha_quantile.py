# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest
import torch
import torch.nn.functional as F

import kilora.attention_maps as attention_maps
from kilora.attention_maps import alpha_from_maps, map_confidence


def _peak():
    y, x = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    return (0.00001 + 0.00599 * torch.exp(-((x - 7.1) ** 2 + (y - 7.4) ** 2) / 15)).reshape(1, -1)


def _feather(owner):
    mask = owner.float().reshape(-1, 1, 16, 16)
    return F.avg_pool2d(F.pad(mask, (1, 1, 1, 1), mode="replicate"), 3, stride=1).flatten(1)


@pytest.mark.parametrize("hard", ["0", "1"])
def test_unset_matches_existing_rule(monkeypatch, hard):
    monkeypatch.setenv("KILORA_ALPHA_HARD", hard)
    monkeypatch.delenv("KILORA_ALPHA_QUANTILE", raising=False)
    value = _peak() * 20
    result = alpha_from_maps({"a": value}, grid=(16, 16))
    blurred = result.history["a"]
    confidence = map_confidence(value)[:, None]
    placement = (
        _feather(blurred >= 0.5) if hard == "1" else blurred - blurred.mean(-1, keepdim=True)
    )
    expected = (1.0 + confidence * (placement - 1.0 if hard == "1" else placement)).clamp(0, 1)
    assert torch.equal(result["ip"], expected)
    assert torch.equal(result["creative"], torch.ones_like(expected))


def test_quantile_selects_top_tokens_per_entry_and_row(monkeypatch):
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", "0.25")
    value = _peak() * 20
    rows = torch.cat((value, value.flip(-1)), dim=0)
    result = alpha_from_maps({"a": rows, "b": rows.flip(-1)}, grid=(16, 16))
    expected_entries = []
    for name, raw in (("a", rows), ("b", rows.flip(-1))):
        m = result.history[name]
        owner = m >= torch.quantile(m, 0.75, dim=-1, keepdim=True)
        assert (owner.sum(-1) == 64).all()
        assert all(m[i, owner[i]].min() >= m[i, ~owner[i]].max() for i in range(2))
        expected_entries.append(
            (1.0 + map_confidence(raw)[:, None] * (_feather(owner) - 1.0)).clamp(0, 1)
        )
    assert torch.equal(result["ip"], torch.stack(expected_entries).amax(0))
    assert torch.equal(result["creative"], torch.ones_like(result["ip"]))


def test_weak_map_and_flat_map(monkeypatch):
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", "0.25")
    monkeypatch.setenv("KILORA_ALPHA_LEVEL", "0.002,0.01")
    value = _peak()
    value = value * (0.006 / torch.quantile(value, 0.99))
    result = alpha_from_maps({"a": value}, grid=(16, 16))
    image = result["ip"].view(16, 16)
    assert image[7, 7] > image[0, 0]
    assert image[0, 0] < 1.0
    flat = alpha_from_maps({"a": torch.full((1, 256), 0.006)}, grid=(16, 16))
    assert torch.equal(flat["ip"], torch.ones(1, 256))


@pytest.mark.parametrize("setting", ["", "bad", "0", "1", "-0.1", "1.1", "nan", "inf"])
def test_invalid_quantile(monkeypatch, setting):
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", setting)
    with pytest.raises(ValueError, match="KILORA_ALPHA_QUANTILE"):
        alpha_from_maps({"a": _peak()}, grid=(16, 16))


def test_soft_mode_ignores_quantile_and_logs_once(monkeypatch, caplog):
    monkeypatch.delenv("KILORA_ALPHA_HARD", raising=False)
    monkeypatch.delenv("KILORA_ALPHA_QUANTILE", raising=False)
    monkeypatch.setattr(attention_maps, "_quantile_ignored_logged", False)
    value = _peak()
    baseline = alpha_from_maps({"a": value}, grid=(16, 16))
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", "invalid")
    with caplog.at_level(logging.WARNING, logger=attention_maps.__name__):
        for _ in range(2):
            result = alpha_from_maps({"a": value}, grid=(16, 16))
            assert torch.equal(result["ip"], baseline["ip"])
    assert sum("KILORA_ALPHA_QUANTILE" in r.message for r in caplog.records) == 1


def test_ties_flat_rows_and_setting_read_each_call(monkeypatch):
    monkeypatch.setenv("KILORA_ALPHA_HARD", "1")
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", "0.25")
    value = torch.full((2, 256), 0.00001)
    value[0].view(16, 16)[4:12, 4:12] = 0.08
    value[1] = 0.08
    masks = []
    pool = F.avg_pool2d

    def capture(input, kernel_size, *args, **kwargs):
        if kernel_size == 3:
            masks.append(input[:, :, 1:-1, 1:-1].clone())
        return pool(input, kernel_size, *args, **kwargs)

    monkeypatch.setattr(attention_maps.F, "avg_pool2d", capture)
    first = alpha_from_maps({"a": value}, grid=(16, 16))
    m = first.history["a"]
    threshold = torch.quantile(m[0], 0.75)
    assert torch.equal(masks[0][0].flatten().bool(), m[0] >= threshold)
    assert masks[0][0].sum() >= 64
    assert masks[0][1].sum() == 0
    monkeypatch.setenv("KILORA_ALPHA_QUANTILE", "0.5")
    second = alpha_from_maps({"a": value}, first, grid=(16, 16))
    assert torch.equal(second.history["a"], first.history["a"])
    assert masks[1][0].sum() > masks[0][0].sum()
