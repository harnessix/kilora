# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from kilora.testing import TinyDenoiser

try:
    from kilora_editor_chunk import PLUGIN as _CHUNK_PLUGIN
except ImportError:
    _CHUNK_PLUGIN = None
else:
    from kilora.editors import register_editor

    register_editor(_CHUNK_PLUGIN, "tests/conftest.py")

TARGET = "layers.*.attention.to_out.0"


@pytest.fixture
def model():
    return TinyDenoiser(channels=4, width=16, num_layers=2, seed=0).eval()


@pytest.fixture
def x():
    return torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(1))


def randomize_up(adapter, set_index=0, seed=0, scale=0.3):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for site in adapter.sites:
            key = f"s{set_index}_up"
            if key in site.tensors:
                p = site.tensors[key]
                p.copy_(torch.randn(p.shape, generator=gen) * scale)


def reference_layer(
    x,
    y0,
    sets,
    depth,
    shared=False,
    grad_last_only=False,
    target="",
    routing_scope="token",
    routing_select="sparsemax",
    carry="dropped",
):

    from kilora.compute import read_dropout

    flat = []
    for s in sets:
        e_n, per = s["down"].shape[0], s["down"].shape[1] * s["down"].shape[2]
        flat.append((e_n, per, s["down"].reshape(e_n * per, -1), s["up"].reshape(e_n * per, -1)))
    derived = all(s.get("score") == "derived" for s in sets)
    alive = [None] * len(sets)
    refs = [None] * len(sets)
    bases = []
    if derived:
        for _, per, down, _ in flat:
            blocks = down.detach().reshape(-1, per, down.shape[-1]).float()
            _, values, vh = torch.linalg.svd(blocks, full_matrices=False)
            count = min(8, per, down.shape[-1])
            tol = (
                values.amax(-1, keepdim=True)
                * max(blocks.shape[-2:])
                * torch.finfo(values.dtype).eps
            )
            rows = (values > tol).sum(-1).clamp(max=count)
            keys = vh[:, :count] * (torch.arange(count)[None, :] < rows[:, None])[..., None]
            bases.append((keys, rows))
    zs = [None] * len(sets)
    held = [None] * len(sets)
    y = carried = y0
    prev = None
    for k in range(1, depth + 1):
        ctx = torch.enable_grad() if (not grad_last_only or k == depth) else torch.no_grad()
        with ctx:
            logits = []
            for i, s in enumerate(sets):
                if derived:
                    keys, rows = bases[i]
                    proj = torch.einsum("...d,erd->...er", carried, keys)
                    energy = carried.float().norm(dim=-1, keepdim=True).square()
                    score = s.get("tau", 1.0) * proj.square().sum(-1) / (energy + 1e-6)
                    score = score / (rows.clamp_min(1).float() / carried.shape[-1])
                    score = score.masked_fill(rows == 0, -1e9)
                    if shared and len(sets) > 1:
                        import math
                        import os

                        score = score + s.get("log_prior", 0.0)
                        if os.environ.get("KILORA_FILE_BALANCE", "1") != "0":
                            score = score - math.log(score.shape[-1])
                    logits.append(score)
                    continue
                q = carried @ s["router"].t()
                if prev is not None:
                    kout = _gram_schmidt(s.get("keys_out", s["keys"]))
                    q = sum(
                        prev[i][..., e : e + 1] * ((q @ kout[e].t()) @ kout[e])
                        for e in range(kout.shape[0])
                    )
                logits.append(_reference_logits(q, s))
            gates = _reference_gates(logits, shared, routing_scope, routing_select)
            prev = gates
            for i, (s, gate, (_, per, down, _up)) in enumerate(zip(sets, gates, flat, strict=True)):
                part = s["strength"] * (carried @ down.t()) * gate.repeat_interleave(per, -1)
                if derived and alive[i] is not None:
                    part = part * alive[i]
                zs[i] = part if s["mode"] == "replace" or zs[i] is None else zs[i] + part
                if derived:
                    size = (zs[i] @ _up).detach().float().norm(dim=-1)
                    if refs[i] is None:
                        refs[i] = size
                    else:
                        ratio = (2 * refs[i] / size.clamp_min(1e-30)).clamp(max=1)
                        zs[i] = zs[i] * ratio[..., None]
            y = y0 + sum(z @ f[3] for z, f in zip(zs, flat, strict=True))
            if k < depth:
                carried = y0
                for i, (z, (_, per, down, up)) in enumerate(zip(zs, flat, strict=True)):
                    m = read_dropout(z, y0 @ down.t(), per, f"{target}#{i}", k)
                    if derived:
                        alive[i] = m if alive[i] is None else alive[i] * m
                    if carry == "drop":
                        carried = carried + (m * z) @ up
                    else:
                        old = held[i]
                        if old is None:
                            old = sets[i]["strength"] * (y0 @ down.t())
                        held[i] = old + (1 - m) * (z - old)
                        carried = carried + held[i] @ up
        if grad_last_only and k < depth:
            y, carried = y.detach(), carried.detach()
            zs = [z.detach() for z in zs]
            held = [None if h is None else h.detach() for h in held]
    return y


def _gram_schmidt(keys):
    rows = []
    for key in keys:
        basis = []
        for v in key:
            for b in basis:
                v = v - (v @ b) * b
            basis.append(v / v.norm())
        rows.append(torch.stack(basis))
    return torch.stack(rows)


def _reference_logits(q, s):
    k = _gram_schmidt(s["keys"])
    proj = torch.einsum("...r,emr->...em", q, k)
    share = proj.pow(2).sum(-1) / (q.pow(2).sum(-1, keepdim=True) + 1e-6)
    return s.get("tau", 1.0) * share / (k.shape[1] / k.shape[2])


def _reference_gates(logits, shared, scope, select):
    sizes = [lg.shape[-1] for lg in logits]

    def pick(z):
        if select == "top1":
            return torch.nn.functional.one_hot(z.argmax(-1), z.shape[-1]).to(z.dtype)
        return _sparsemax(z)

    chosen = [lg.mean(-2, keepdim=True) if scope == "tensor" else lg for lg in logits]
    if shared and len(logits) > 1:
        ps = list(torch.split(pick(torch.cat(chosen, -1)), sizes, -1))
    else:
        ps = [pick(c) for c in chosen]
    return [p.expand_as(lg) for p, lg in zip(ps, logits, strict=True)]


def _sparsemax(z):
    out = torch.empty_like(z)
    flat, res = z.reshape(-1, z.shape[-1]), out.view(-1, z.shape[-1])
    for row in range(flat.shape[0]):
        v = flat[row]
        srt = sorted(v.tolist(), reverse=True)
        j = max(i for i in range(1, len(srt) + 1) if 1 + i * srt[i - 1] > sum(srt[:i]))
        tau = (sum(srt[:j]) - 1) / j
        res[row] = (v - tau).clamp(min=0)
    return out


@pytest.fixture(autouse=True)
def _synchronous_live_writes(request, monkeypatch):
    if "async_live" not in request.keywords:
        monkeypatch.setenv("KILORA_LIVE_ASYNC", "0")


@pytest.fixture(autouse=True)
def _key_cache_in_tmp(tmp_path_factory, monkeypatch):
    from kilora import key_cache

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path_factory.mktemp("key-cache")))
