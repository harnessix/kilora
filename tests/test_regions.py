# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.comfyui.live_write import PathHandle, image_loss
from kilora.regions import Region, RegionBook, region_keys
from kilora.subspace_keys import subspace_distance


def _handle(apply_model, caption=2.0, empty=-1.0):
    return PathHandle(
        session=None,
        rule=None,
        recorder=None,
        timesteps=[0.2, 0.5, 0.8],
        conditioning={"c_crossattn": torch.full((1, 3), caption)},
        empty_conditioning={"c_crossattn": torch.full((1, 3), empty)},
        apply_model=apply_model,
    )


def _model(x_t, t, c_crossattn):
    return torch.tanh(x_t) * 0.3 + c_crossattn.mean()


def test_full_mask_is_bit_identical_to_no_mask():
    x0 = torch.randn(1, 4, 6, 8, generator=torch.Generator().manual_seed(0))
    plain = image_loss(_handle(_model), x0, event=2)(None)
    full = image_loss(_handle(_model), x0, event=2, mask=torch.ones(1, 1, 6, 8))(None)
    assert plain == full
    big = image_loss(_handle(_model), x0, event=2, mask=torch.ones(1, 48, 64))(None)
    assert plain == big


def test_part_mask_measures_only_the_region():
    x0 = torch.randn(1, 4, 6, 8, generator=torch.Generator().manual_seed(1))
    x0[..., 4:] += 5.0
    zero = lambda x_t, t, c_crossattn: torch.zeros_like(x_t)
    mask = torch.zeros(1, 1, 6, 8)
    mask[..., :4] = 1.0
    value = image_loss(_handle(zero), x0, event=0, mask=mask)(None)
    assert abs(value - float(x0[..., :4].pow(2).mean())) < 1e-5
    whole = image_loss(_handle(zero), x0, event=0)(None)
    assert value < whole / 4
    soft = mask * 0.5
    soft[..., 4:] = 0.25
    value = image_loss(_handle(zero), x0, event=0, mask=soft)(None)
    w = soft.expand_as(x0)
    assert abs(value - float((x0.pow(2) * w).sum() / w.sum())) < 1e-4


def _per_row(x_t, c):
    return torch.zeros_like(x_t) + c.flatten(1).mean(1).reshape(-1, *([1] * (x_t.dim() - 1)))


def test_region_caption_replaces_the_prompt():
    x0 = torch.zeros(1, 4, 6, 8)
    mask = torch.zeros(1, 1, 6, 8)
    mask[..., :2, :] = 1.0
    const = lambda x_t, t, c_crossattn: _per_row(x_t, c_crossattn)
    region = {"c_crossattn": torch.full((1, 3), 3.0)}
    value = image_loss(_handle(const), x0, event=0, mask=mask, region_conditioning=region)(None)
    assert abs(value - (9.0 + 1.0) / 2) < 1e-6


def _planted_probes(signal, other, grid=(6, 8), d=16, steps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    gh, gw = grid
    entries = []
    for k in range(steps):
        a = torch.randn(gh, gw, signal.shape[0], generator=g) * 3.0
        b = torch.randn(gh, gw, other.shape[0], generator=g) * 6.0
        q = 0.05 * torch.randn(gh, gw, d, generator=g)
        q[:, : gw // 2] += a[:, : gw // 2] @ signal
        q[:, gw // 2 :] += b[:, gw // 2 :] @ other
        prev = None if k == 0 else [torch.ones(1, gh * gw, 2)]
        entries.append(
            {
                "k": k,
                "queries": [q.reshape(1, gh * gw, d)],
                "prev": prev,
                "gates": [torch.ones(1, gh * gw, 2)],
            }
        )
    entries.append({"k": "output", "grads": None})
    text = {"k": 0, "queries": [torch.randn(1, 7, d, generator=g)], "prev": None, "gates": None}
    return {"layer.attn": entries, "layer.text": [text]}


def test_new_expert_keys_match_the_planted_region_signal():
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=torch.Generator().manual_seed(3)))
    signal, other = q[:, :4].t(), q[:, 4:8].t()
    mask = torch.zeros(1, 1, 6, 8)
    mask[..., :4] = 1.0
    keys = region_keys(_planted_probes(signal, other), mask)
    assert set(keys) == {"layer.attn"}
    assert subspace_distance(keys["layer.attn"]["keys_in"], signal) < 0.2
    assert subspace_distance(keys["layer.attn"]["keys_out"], signal) < 0.2
    unmasked = region_keys(_planted_probes(signal, other), torch.ones(1, 1, 6, 8))
    assert subspace_distance(unmasked["layer.attn"]["keys_in"], other) < 0.2

    experts = {"layer.attn": torch.stack([other, q[:, 8:12].t()])}
    book = RegionBook()
    entry = book.target_for(Region(mask, name="red scarf", caption="red scarf"), keys, experts)
    assert entry.new and entry.name == "red scarf"
    assert subspace_distance(entry.keys["layer.attn"]["keys_in"], signal) < 0.2
    near = {"layer.attn": torch.stack([other, signal])}
    routed = RegionBook().target_for(Region(mask, name="scarf"), keys, near)
    assert not routed.new and routed.index == 1
    assert book.find("  Red   Scarf ") is entry
    assert book.target_for(Region(mask, name="red scarf"), {}, experts) is entry
    assert entry.writes == 2


def test_book_round_trip():
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=torch.Generator().manual_seed(4)))
    book = RegionBook()
    keys = {"t": {"keys_in": q[:4].clone(), "keys_out": q[4:8].clone()}}
    book.target_for(Region(torch.ones(1, 1, 2, 2), name="cat", caption="a cat"), keys, {})
    again = RegionBook.from_tensors(book.state_tensors())
    (e,) = again.experts
    assert (e.name, e.caption, e.new, e.index) == ("cat", "a cat", True, 0)
    assert torch.equal(e.keys["t"]["keys_in"], q[:4])
    assert RegionBook.from_tensors({}).experts == []
