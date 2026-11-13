# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch
from torch import nn

from kilora.comfyui.crop_positions import global_positions, supported


class Rope3D(nn.Module):
    def __init__(self, dim=12, max_size=64):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        self.calls = []

    def generate_embeddings(self, B_T_H_W_C, fps=None, device=None, dtype=None):
        _, t, h, w = B_T_H_W_C[:4]
        part = self.dim // 6
        freqs = 1.0 / (100.0 ** (torch.arange(part, dtype=torch.float64) / part))
        seq = torch.arange(max(h, w, t), dtype=torch.float64)
        th, tw, tt = (torch.outer(seq[:n], freqs) for n in (h, w, t))
        em = torch.cat(
            [
                tt[:, None, None, :].expand(t, h, w, part),
                th[None, :, None, :].expand(t, h, w, part),
                tw[None, None, :, :].expand(t, h, w, part),
            ]
            * 2,
            dim=-1,
        )
        return em.reshape(t * h * w, 1, 1, -1).float()

    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        out = self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps, device=device, dtype=dtype)
        self.calls.append(out)
        return out


class LearnableAxes(nn.Module):
    def __init__(self, dim=4, size=32):
        super().__init__()
        self.h = nn.Parameter(torch.randn(size, dim))
        self.w = nn.Parameter(torch.randn(size, dim))
        self.calls = []

    def generate_embeddings(self, B_T_H_W_C, fps=None, device=None, dtype=None):
        b, t, h, w = B_T_H_W_C[:4]
        emb = self.h[:h][None, None, :, None] + self.w[:w][None, None, None, :]
        return emb.expand(b, t, h, w, -1)

    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        out = self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps, device=device)
        self.calls.append(out)
        return out


class MiniTrainDIT(nn.Module):
    def __init__(self, extra=False):
        super().__init__()
        self.patch_spatial = 2
        self.pos_embedder = Rope3D()
        self.extra_pos_embedder = LearnableAxes() if extra else None

    def forward(self, x, timestep, **kwargs):
        if kwargs.get("fail"):
            raise RuntimeError("model failed")
        b, c, t, h, w = x.shape
        p = self.patch_spatial
        tokens = x.reshape(b, c, t, h // p, p, w // p, p).mean((4, 6)).permute(0, 2, 3, 4, 1)
        rope = self.last_rope = self.pos_embedder(tokens, fps=None, device=x.device)
        out = tokens.mean(-1).reshape(b, -1) + rope[:, 0, 0, :].sum(-1)
        if self.extra_pos_embedder is not None:
            extra = self.last_extra = self.extra_pos_embedder(tokens, fps=None, device=x.device)
            out = out + extra.sum(-1).reshape(b, -1)
        return out


def _call(dit, boxes, x, **extra):
    from kilora.comfyui.patching import _key_mask_by_step

    class Patcher:
        pass

    patcher = Patcher()
    patcher.model_options = {}
    patcher.model = types.SimpleNamespace(diffusion_model=dit)
    _key_mask_by_step(patcher)
    c = dict(extra)
    if boxes is not None:
        c["kilora_crop_window"] = torch.tensor(boxes)
    return patcher.model_options["model_function_wrapper"](
        dit, {"input": x, "timestep": torch.tensor([1.0]), "c": c}
    )


def test_supported_detects_the_family():
    assert supported(MiniTrainDIT())
    assert not supported(nn.Linear(2, 2))


def test_crop_call_gets_full_grid_rows_at_the_window():
    dit = MiniTrainDIT(extra=True)
    top, left, full_h, full_w = 8, 12, 32, 40
    crop = torch.randn(1, 3, 1, 16, 8)
    _call(dit, [[top, left, full_h, full_w]], crop)
    got = dit.last_rope
    full = dit.pos_embedder.generate_embeddings(torch.Size((1, 1, 16, 20, 3)))
    expected = full.reshape(1, 16, 20, 1, 1, -1)[:, 4:12, 6:10].reshape(8 * 4, 1, 1, -1)
    assert torch.equal(got, expected)
    local = dit.pos_embedder.generate_embeddings(torch.Size((1, 1, 8, 4, 3)))
    assert not torch.equal(got, local)
    extra = dit.last_extra
    full_extra = dit.extra_pos_embedder.generate_embeddings(torch.Size((1, 1, 16, 20, 3)))
    assert torch.equal(extra, full_extra[:, :, 4:12, 6:10])


def test_call_without_window_is_unchanged():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 1, 8, 8)
    dit = MiniTrainDIT(extra=True)
    plain = dit(x, torch.tensor([1.0]))
    plain_rows = dit.last_rope
    out = _call(dit, None, x)
    assert torch.equal(out, plain)
    assert torch.equal(dit.last_rope, plain_rows)
    assert "forward" not in dit.pos_embedder.__dict__


def test_full_window_is_unchanged():
    x = torch.randn(1, 3, 1, 8, 8)
    dit = MiniTrainDIT()
    plain = dit(x, torch.tensor([1.0]))
    assert torch.equal(_call(dit, [[0, 0, 8, 8]], x), plain)


def test_embedders_are_restored_after_call_and_exception():
    dit = MiniTrainDIT(extra=True)
    originals = (dit.pos_embedder.forward, dit.extra_pos_embedder.forward)
    _call(dit, [[4, 4, 16, 16]], torch.randn(1, 3, 1, 8, 8))
    assert "forward" not in dit.pos_embedder.__dict__
    assert "forward" not in dit.extra_pos_embedder.__dict__
    with pytest.raises(RuntimeError, match="model failed"):
        _call(dit, [[4, 4, 16, 16]], torch.randn(1, 3, 1, 8, 8), fail=True)
    assert "forward" not in dit.pos_embedder.__dict__
    assert "forward" not in dit.extra_pos_embedder.__dict__
    assert (dit.pos_embedder.forward, dit.extra_pos_embedder.forward) == originals
    with pytest.raises(RuntimeError), global_positions(dit, [[4, 4, 16, 16]], 1):
        assert "forward" in dit.pos_embedder.__dict__
        raise RuntimeError
    assert "forward" not in dit.pos_embedder.__dict__


def test_rows_with_different_windows_keep_local_positions(caplog):
    dit = MiniTrainDIT()
    x = torch.randn(2, 3, 1, 8, 8)
    plain = dit(x, torch.tensor([1.0]))
    local = dit.last_rope
    with caplog.at_level("WARNING", logger="kilora"):
        out = _call(dit, [[0, 0, 16, 16], [8, 8, 16, 16]], x)
    assert torch.equal(out, plain)
    assert torch.equal(dit.last_rope, local)
    assert "different windows" in caplog.text


def test_rows_with_one_window_share_the_slice():
    dit = MiniTrainDIT()
    x = torch.randn(2, 3, 1, 8, 8)
    _call(dit, [[8, 8, 16, 16], [8, 8, 16, 16]], x)
    full = dit.pos_embedder.generate_embeddings(torch.Size((1, 1, 8, 8, 3)))
    expected = full.reshape(1, 8, 8, 1, 1, -1)[:, 4:8, 4:8].reshape(16, 1, 1, -1)
    assert torch.equal(dit.last_rope, expected)


def test_other_families_are_unchanged():
    model = types.SimpleNamespace(pos_embedder=Rope3D(), patch_spatial=2)
    with global_positions(model, [[4, 4, 16, 16]], 1) as active:
        assert not active
        assert "forward" not in model.pos_embedder.__dict__
