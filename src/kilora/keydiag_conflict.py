# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch


def overlaps(s):
    a = s.down.detach().float().reshape(-1, s.per_expert, s.down.shape[-1])
    b = s.up.detach().float().reshape(-1, s.per_expert, s.up.shape[-1])
    aa = torch.einsum("erd,fsd->efrs", a, a)
    bb = torch.einsum("erd,fsd->efrs", b, b)
    norms = torch.einsum("eers,eers->e", aa, bb).clamp_min(0).sqrt()
    denom = (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    ww = (aa * bb).sum((-1, -2)) / denom
    wr = None
    if a.shape[-1] == b.shape[-1]:
        cross = torch.einsum("erd,fsd->efrs", a, b)
        wr = torch.einsum("efrs,fesr->ef", cross, cross) / denom
    return ww, wr


def conflict(s):
    if s.down is None or s.up is None:
        return {"skipped": "no read or write matrix"}
    with torch.no_grad():
        ww, wr = overlaps(s)
        pairs = [(float(ww[e, f]), e, f) for e in range(len(ww)) for f in range(e + 1, len(ww))]
        value, e, f = min(pairs) if pairs else (None, None, None)
        wr_pairs = (
            [(float(wr[e, f]), e, f) for e in range(len(wr)) for f in range(e + 1, len(wr))]
            if wr is not None
            else []
        )
        wr_value, wr_e, wr_f = min(wr_pairs) if wr_pairs else (None, None, None)
        return {
            "write_write": ww.cpu().tolist(),
            "write_read": None if wr is None else wr.cpu().tolist(),
            "write_read_negative_share": None if wr is None else float((wr < 0).float().mean()),
            "most_negative_pair": None if value is None else [e, f],
            "most_negative_value": value,
            "write_read_most_negative_pair": None if wr_value is None else [wr_e, wr_f],
            "write_read_most_negative_value": wr_value,
        }
