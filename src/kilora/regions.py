# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .subspace_keys import subspace_distance

NEW_EXPERT_DISTANCE = 0.6


@dataclass
class Region:
    mask: torch.Tensor
    name: str = ""
    caption: str = ""
    probes: dict | None = None
    ensure_probes: Any = None

    def recorded_probes(self) -> dict:
        if self.probes is None and self.ensure_probes is not None:
            self.ensure_probes()
        return self.probes or {}


def latent_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    m = mask.detach().float()
    while m.dim() < 4:
        m = m.unsqueeze(0)
    if m.shape[1] != 1:
        m = m.mean(dim=1, keepdim=True)
    size = tuple(like.shape[-2:])
    if tuple(m.shape[-2:]) != size:
        m = F.interpolate(m, size=size, mode="area" if m.shape[-1] >= size[-1] else "bilinear")
    if m.shape[0] != like.shape[0]:
        m = m[:1].expand(like.shape[0], -1, -1, -1)
    return m.clamp(0.0, 1.0)


def is_full(mask: torch.Tensor | None) -> bool:
    return mask is None or bool((mask == 1).all())


def masked_mean(err: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if is_full(mask):
        return err.mean()
    m = mask.to(err.device, err.dtype)
    while m.dim() < err.dim():
        m = m.unsqueeze(2)
    m = m.expand_as(err)
    total = m.sum()
    if float(total) <= 0:
        raise ValueError("the region mask is empty")
    return (err * m).sum() / total


def token_weights(mask: torch.Tensor, tokens: int) -> torch.Tensor | None:
    h, w = mask.shape[-2:]
    gh = max(1, round(math.sqrt(tokens * h / w)))
    if tokens % gh:
        return None
    gw = tokens // gh
    if gh > h or gw > w:
        return None
    return F.adaptive_avg_pool2d(mask.float(), (gh, gw)).flatten(1)


def region_keys(
    probes: Mapping[str, list], mask: torch.Tensor, m_in: int = 4, m_out: int = 4, pos: int = 0
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for target, entries in probes.items():
        sums: dict[str, torch.Tensor] = {}
        for entry in entries:
            if entry.get("k") == "output" or entry.get("queries") is None:
                continue
            q = entry["queries"][pos]
            if q is None:
                continue
            q = q.detach().float().cpu()
            if q.dim() == 2:
                q = q.unsqueeze(0)
            q = q.reshape(q.shape[0], -1, q.shape[-1])
            w = token_weights(mask[:1].expand(q.shape[0], -1, -1, -1), q.shape[1])
            if w is None:
                continue
            moment = torch.einsum("bn,bnr,bns->rs", w, q, q)
            parts = ["keys_in"] + (["keys_out"] if entry.get("prev") is not None else [])
            for part in parts:
                sums[part] = sums[part] + moment if part in sums else moment
        if "keys_in" not in sums:
            continue
        keys = {}
        for part, m in (("keys_in", m_in), ("keys_out", m_out)):
            moment = sums.get(part, sums["keys_in"])
            _, vectors = torch.linalg.eigh(moment.double())
            keys[part] = vectors[:, -m:].flip(-1).t().float().contiguous()
        out[target] = keys
    return out


def expert_distances(
    keys: Mapping[str, Mapping[str, torch.Tensor]], experts: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    rows = []
    for target, k in keys.items():
        if target not in experts:
            continue
        ex = experts[target].float()
        a = k["keys_in"]
        rows.append(torch.tensor([_distance(a, _orth(ex[e])) for e in range(ex.shape[0])]))
    if not rows:
        return torch.empty(0)
    return torch.stack(rows).mean(0)


def _orth(k: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(k.t().double())
    return q.t().float()


def _distance(a: torch.Tensor, b: torch.Tensor) -> float:
    m = min(a.shape[0], b.shape[0])
    return subspace_distance(a[:m], b[:m])


@dataclass
class RegionExpert:
    name: str
    caption: str
    index: int
    new: bool
    keys: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    writes: int = 0


@dataclass
class RegionBook:
    experts: list[RegionExpert] = field(default_factory=list)

    def find(self, name: str) -> RegionExpert | None:
        key = _name_key(name)
        return next((e for e in self.experts if _name_key(e.name) == key), None)

    def target_for(
        self,
        region: Region,
        keys: Mapping[str, Mapping[str, torch.Tensor]],
        experts: Mapping[str, torch.Tensor],
        threshold: float | None = None,
    ) -> RegionExpert:
        threshold = NEW_EXPERT_DISTANCE if threshold is None else threshold
        found = self.find(region.name) if region.name else None
        if found is not None:
            found.writes += 1
            return found
        d = expert_distances(keys, experts)
        index = int(d.argmin()) if d.numel() else 0
        new = not d.numel() or float(d[index]) >= threshold
        entry = RegionExpert(
            name=region.name,
            caption=region.caption,
            index=index,
            new=new,
            keys={t: {p: v.clone() for p, v in k.items()} for t, k in keys.items()},
            writes=1,
        )
        self.experts.append(entry)
        return entry

    def state_tensors(self, prefix: str = "region/") -> dict[str, torch.Tensor]:
        if not self.experts:
            return {}
        meta = [
            {
                "name": e.name,
                "caption": e.caption,
                "index": e.index,
                "new": e.new,
                "writes": e.writes,
                "targets": sorted(e.keys),
            }
            for e in self.experts
        ]
        data = json.dumps(meta).encode("utf-8")
        out = {f"{prefix}meta": torch.tensor(list(data), dtype=torch.uint8)}
        for n, e in enumerate(self.experts):
            for t, k in e.keys.items():
                for p, v in k.items():
                    out[f"{prefix}{n}/{t}/{p}"] = v.clone()
        return out

    @staticmethod
    def from_tensors(t: Mapping[str, torch.Tensor], prefix: str = "region/") -> RegionBook:
        raw = t.get(f"{prefix}meta")
        if raw is None:
            return RegionBook()
        meta = json.loads(bytes(raw.to(torch.uint8).tolist()).decode("utf-8"))
        experts = []
        for n, m in enumerate(meta):
            keys = {
                target: {
                    p: t[f"{prefix}{n}/{target}/{p}"].clone()
                    for p in ("keys_in", "keys_out")
                    if f"{prefix}{n}/{target}/{p}" in t
                }
                for target in m["targets"]
            }
            experts.append(
                RegionExpert(
                    m["name"],
                    m["caption"],
                    int(m["index"]),
                    bool(m["new"]),
                    keys,
                    int(m.get("writes", 0)),
                )
            )
        return RegionBook(experts)


def _name_key(name: str) -> str:
    return " ".join(name.lower().split())
