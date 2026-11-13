# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from kilora.editor_packs import PACK_SUFFIX, read_pack, write_pack

FORMAT = "kilora-chunk-editor"
EDITOR_VERSION = "1.0.0"
COMMON = "common"
STATE_FORMAT = "kilora-chunk-state"
V0_LAYOUT = "nested-v1"
LIVE_RANK = (4, 64)
HISTORY = 8
KINDS = ("to_q", "to_k", "to_v", "to_out.0")


def _load_residual(tensors):
    result = {}
    for key, up in tensors.items():
        if key.startswith("residual/") and key.endswith("/up"):
            expert, hook = key[len("residual/") : -len("/up")].split("/", 1)
            result[hook, int(expert)] = (up.clone(), tensors[key[:-2] + "down"].clone())
    return result


def pack_id_of(family: str) -> str:
    return "chunk-" + family.replace("_", "-")


def hook_kind(name: str) -> str:
    return next(k for k in KINDS if name.endswith(k))


@dataclass
class Chunk:
    family: str
    q: torch.Tensor
    eigenvalues: torch.Tensor
    mean_keys: dict[str, torch.Tensor] = field(default_factory=dict)
    input_basis: dict[str, torch.Tensor] = field(default_factory=dict)
    input_eigenvalues: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ChunkEditor:
    names: list[str]
    shapes: dict[str, list[int]]
    alpha: dict[str, float]
    chunks: list[Chunk]

    EXACT_TENSORS = (
        "eigenvalues",
        "keys.*",
        "input_basis.*",
        "input_eigenvalues.*",
    )

    def save_packs(
        self,
        folder,
        version: str,
        base_family: str,
        license: str,
        storage: str = "int8-block",
        license_notice: str | None = None,
        collection: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = {}
        for c in self.chunks:
            common = c.family == COMMON
            tensors = {"q": c.q.float(), "eigenvalues": c.eigenvalues.float()}
            tensors.update({f"keys.{h}": k.float() for h, k in c.mean_keys.items()})
            tensors.update({f"input_basis.{h}": v.float() for h, v in c.input_basis.items()})
            tensors.update(
                {f"input_eigenvalues.{h}": v.float() for h, v in c.input_eigenvalues.items()}
            )
            pack_id = pack_id_of(c.family)
            manifest = {
                "id": pack_id,
                "version": version,
                "role": "common" if common else "family",
                "editor": {"kind": FORMAT, "version": f">={EDITOR_VERSION}"},
                "base_models": [{"family": base_family}],
                "mod_families": [] if common else [c.family],
                "requires": {} if common else {pack_id_of(COMMON): f">={version}"},
                "storage": storage,
                "license": license,
                **({"license_notice": license_notice} if license_notice else {}),
                "training": {
                    "state": "trained",
                    **({"collection": dict(collection)} if collection else {}),
                },
                "chunk": {
                    "family": c.family,
                    "hooks": self.names,
                    "shapes": self.shapes,
                    "alpha": self.alpha,
                    "v0_layout": V0_LAYOUT,
                    "live_rank": list(LIVE_RANK),
                },
            }
            f = folder / f"{pack_id}-{version}{PACK_SUFFIX}"
            write_pack(f, manifest, tensors, exact=self.EXACT_TENSORS)
            out[c.family] = f
        return out

    @staticmethod
    def load_packs(files: Sequence, tol: float = 0.05) -> ChunkEditor:
        base, chunks = None, []
        for f in files:
            m, t = read_pack(f)
            if m.editor_kind != FORMAT:
                raise ValueError(f"{f}: a pack for the editor {m.editor_kind!r}")
            c = m.raw["chunk"]
            if c.get("v0_layout") != V0_LAYOUT:
                raise ValueError(f"{f}: the V0 layout {c.get('v0_layout')!r} is not supported")
            if base is None:
                base = c
            elif c["hooks"] != base["hooks"] or c["shapes"] != base["shapes"]:
                raise ValueError("these packs are for different hook points")
            keys = {k[5:]: v for k, v in t.items() if k.startswith("keys.")}
            basis = {
                k[len("input_basis.") :]: v for k, v in t.items() if k.startswith("input_basis.")
            }
            values = {
                k[len("input_eigenvalues.") :]: v
                for k, v in t.items()
                if k.startswith("input_eigenvalues.")
            }
            chunks.append(Chunk(c["family"], t["q"], t["eigenvalues"], keys, basis, values))
        if base is None:
            raise ValueError("no packs")
        ed = ChunkEditor(base["hooks"], base["shapes"], base["alpha"], chunks)
        return ed

    def coherence(self) -> float:
        parts = [c.q.float() for c in self.chunks if len(c.q)]
        if not parts:
            return 0.0
        rows = torch.cat(parts, dim=0)
        rows = torch.nn.functional.normalize(rows, dim=1)
        gram = (rows @ rows.T).abs()
        gram.fill_diagonal_(0)
        return float(gram.max()) if gram.numel() else 0.0

    def check_orthogonal(self, tol: float = 0.05) -> None:
        for i, a in enumerate(self.chunks):
            for b in self.chunks[i + 1 :]:
                worst = float((a.q @ b.q.t()).abs().max()) if len(a.q) and len(b.q) else 0.0
                if worst > tol:
                    raise ValueError(
                        f"chunks {a.family} and {b.family} are not orthogonal "
                        f"(largest inner product {worst:.3f}, limit {tol})"
                    )

    @staticmethod
    def from_basis_files(
        files: Sequence[str], alpha: Mapping[str, float] | None = None
    ) -> ChunkEditor:
        cs = [torch.load(f, map_location="cpu") for f in files]
        chunks = [
            Chunk(
                c.get("family", Path(f).stem),
                c["q"].float(),
                c["eigenvalues"].float(),
                dict(c.get("mean_keys") or {}),
            )
            for c, f in zip(cs, files, strict=True)
        ]
        return ChunkEditor(
            list(cs[0]["names"]), dict(cs[0]["shapes"]), dict(alpha or cs[0]["alpha"]), chunks
        )

    def family_names(self) -> list[str]:
        return [c.family for c in self.chunks]

    def per_hook(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        out, off = {}, 0
        for h in self.names:
            n = int(torch.Size(self.shapes[h]).numel())
            out[h] = q[:, off : off + n].reshape(q.shape[0], *self.shapes[h])
            off += n
        return out


@dataclass
class ChunkState:
    coef: dict[str, torch.Tensor]
    hist: dict[str, torch.Tensor]
    enabled: list[str]
    events: int = 0
    stack: list[dict] = field(default_factory=list)
    extra: dict[str, torch.Tensor] = field(default_factory=dict)
    prior: dict[str, torch.Tensor] = field(default_factory=dict)

    node_coef: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)
    residual: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @staticmethod
    def new(
        editor: ChunkEditor, experts: int = 2, enabled: Sequence[str] | None = None
    ) -> ChunkState:
        coef = {c.family: torch.zeros(experts, c.q.shape[0]) for c in editor.chunks}
        hist = {c.family: torch.full((HISTORY, c.q.shape[0]), float("nan")) for c in editor.chunks}
        return ChunkState(coef, hist, list(enabled or editor.family_names()))

    def snapshot(self) -> dict:
        return {
            "coef": {k: v.clone() for k, v in self.coef.items()},
            "hist": {k: v.clone() for k, v in self.hist.items()},
            "enabled": list(self.enabled),
            "events": self.events,
            "extra": {k: v.clone() for k, v in self.extra.items()},
            "prior": {k: v.clone() for k, v in self.prior.items()},
            "node_coef": {k: v.clone() for k, v in self.node_coef.items()},
            "residual": {k: (u.clone(), v.clone()) for k, (u, v) in self.residual.items()},
        }

    def restore(self, snap: Mapping) -> None:
        self.coef = {k: v.clone() for k, v in snap["coef"].items()}
        self.hist = {k: v.clone() for k, v in snap["hist"].items()}
        self.enabled = list(snap["enabled"])
        self.events = int(snap["events"])
        self.extra = {k: v.clone() for k, v in snap.get("extra", {}).items()}
        self.prior = {k: v.clone() for k, v in snap.get("prior", {}).items()}
        self.node_coef = {k: v.clone() for k, v in snap.get("node_coef", {}).items()}
        self.residual = {
            k: (u.clone(), v.clone()) for k, (u, v) in snap.get("residual", {}).items()
        }

    def push(self) -> None:
        self.stack.append(self.snapshot())

    def undo(self) -> bool:
        if not self.stack:
            return False
        self.restore(self.stack.pop())
        return True

    def reset(self) -> None:
        self.push()
        for k in self.coef:
            self.coef[k] = torch.zeros_like(self.coef[k])
            self.hist[k] = torch.full_like(self.hist[k], float("nan"))
        self.extra = {k: torch.zeros_like(v) for k, v in self.extra.items()}
        self.prior = {}
        self.node_coef = {}
        self.residual = {}
        self.events = 0

    def set_enabled(self, family: str, on: bool) -> None:
        self.push()
        if on and family not in self.enabled:
            self.enabled.append(family)
        if not on and family in self.enabled:
            self.enabled.remove(family)

    def expanded_u(
        self, editor: ChunkEditor, hooks: Sequence[str] | None = None
    ) -> dict[str, torch.Tensor]:
        out = None
        for c in editor.chunks:
            if c.family not in self.enabled:
                continue
            ph = editor.per_hook(c.q)
            part = {
                h: torch.einsum(
                    "ej,jor->eor",
                    self.coef[c.family] + self.node_coef[(c.family, h)]
                    if (c.family, h) in self.node_coef
                    else self.coef[c.family],
                    d,
                )
                for h, d in ph.items()
                if hooks is None or h in hooks
            }
            out = part if out is None else {h: out[h] + part[h] for h in out}
        if out is None:
            e = next(iter(self.coef.values())).shape[0]
            out = {
                h: torch.zeros(e, *editor.shapes[h])
                for h in editor.names
                if hooks is None or h in hooks
            }
        for (hook, expert), (up, down) in self.residual.items():
            if hook in out:
                out[hook][expert] = out[hook][expert] + (up @ down).to(out[hook])
        return out

    def add_residual(self, hook: str, expert: int, value: torch.Tensor) -> None:
        if not bool(torch.count_nonzero(value)):
            return
        up = value.detach().clone()
        down = torch.eye(value.shape[1], dtype=value.dtype, device=value.device)
        old = self.residual.get((hook, expert))
        if old is not None:
            up, down = torch.cat((old[0], up), 1), torch.cat((old[1], down), 0)
        self.residual[hook, expert] = (up, down)

    def compact_residual(self, budget: int) -> int:
        for key, (up, down) in list(self.residual.items()):
            if up.shape[1] <= budget:
                continue
            qu, ru = torch.linalg.qr(up, mode="reduced")
            qv, rv = torch.linalg.qr(down.T, mode="reduced")
            left, values, right = torch.linalg.svd(ru @ rv.T, full_matrices=False)
            self.residual[key] = ((qu @ left) * values, right @ qv.T)
        return sum(up.shape[1] for up, _ in self.residual.values())

    def save(self, path, editor: ChunkEditor) -> None:
        t = {}
        for k in self.coef:
            t[f"coef/{k}"] = self.coef[k].contiguous()
            t[f"hist/{k}"] = self.hist[k].contiguous()
        for k, v in self.extra.items():
            t[f"extra/{k}"] = v.contiguous()
        for k, v in self.prior.items():
            t[f"prior/{k}"] = v.contiguous()
        for (family, hook), v in self.node_coef.items():
            t[f"node_coef/{family}/{hook}"] = v.contiguous()
        for (hook, expert), (up, down) in self.residual.items():
            t[f"residual/{expert}/{hook}/up"] = up.contiguous()
            t[f"residual/{expert}/{hook}/down"] = down.contiguous()
        for h, u in self.expanded_u(editor).items():
            t[f"u_expanded/{h}"] = u.contiguous()
        meta = {"format": STATE_FORMAT, "enabled": self.enabled, "events": self.events}
        save_file(t, str(path), metadata={"kilora.manifest": json.dumps(meta)})

    @staticmethod
    def load(path) -> ChunkState:
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            m = json.loads(f.metadata()["kilora.manifest"])
        t = load_file(str(path))
        coef = {k[5:]: v for k, v in t.items() if k.startswith("coef/")}
        hist = {k[5:]: v for k, v in t.items() if k.startswith("hist/")}
        extra = {k[6:]: v for k, v in t.items() if k.startswith("extra/")}
        prior = {k[6:]: v for k, v in t.items() if k.startswith("prior/")}
        node_coef = {
            tuple(k[len("node_coef/") :].split("/", 1)): v
            for k, v in t.items()
            if k.startswith("node_coef/")
        }
        return ChunkState(
            coef,
            hist,
            list(m["enabled"]),
            int(m["events"]),
            extra=extra,
            prior=prior,
            node_coef=node_coef,
            residual=_load_residual(t),
        )


@dataclass
class EditBranch:
    id: str
    parent: str | None
    state: ChunkState
    seed: int | None = None
    temperature: float | None = None
    depth: int | None = None
    paths: list = field(default_factory=list)
    action: str | None = None
    writes: list[int] = field(default_factory=list)
    merge_pairs: list = field(default_factory=list)


class BranchStore:
    def __init__(self, state: ChunkState):
        self.branches = {"main": EditBranch("main", None, state)}
        self.head = self.current = "main"
        self.serial = 0

    @property
    def state(self) -> ChunkState:
        return self.branches[self.current].state

    def fork(self, parent: str | None = None, *, seed=None, temperature=None, depth=None) -> str:
        import copy

        parent = self.head if parent is None else parent
        source = self.branches[parent].state
        self.serial += 1
        branch_id = f"branch-{self.serial}"
        state = ChunkState(
            dict(source.coef),
            {k: v.clone() for k, v in source.hist.items()},
            list(source.enabled),
            source.events,
            copy.deepcopy(source.stack),
            {k: v.clone() for k, v in source.extra.items()},
            {k: v.clone() for k, v in source.prior.items()},
            dict(source.node_coef),
            {k: (u.clone(), v.clone()) for k, (u, v) in source.residual.items()},
        )
        self.branches[branch_id] = EditBranch(branch_id, parent, state, seed, temperature, depth)
        return branch_id

    def checkout(self, branch_id: str) -> None:
        if branch_id not in self.branches:
            raise KeyError(branch_id)
        self.current = branch_id

    def promote(self, branch_id: str) -> None:
        self.checkout(branch_id)
        self.head = branch_id

    def delete(self, branch_id: str) -> None:
        if branch_id == self.head or branch_id == self.current:
            raise ValueError("cannot delete the head or current branch")
        del self.branches[branch_id]

    def list(self) -> list[dict]:
        import copy

        return [
            dict(
                id=b.id,
                parent=b.parent,
                seed=b.seed,
                temperature=b.temperature,
                depth=b.depth,
                paths=copy.deepcopy(b.paths),
                action=b.action,
                writes=list(b.writes),
                head=b.id == self.head,
                current=b.id == self.current,
            )
            for b in self.branches.values()
        ]


@dataclass
class ChunkWriter:
    editor: ChunkEditor
    times: Sequence[float]
    weights: Sequence[float]
    eps: Sequence[float]
    write_norm: float
    per_event: int = 16
    decay: float = 0.02
    measure_px: int = 512
    chunk_cap: float | None = None
    total_cap: float | None = None
    norm: Any = None
    effect_cap: float | None = None
    diff: str = "central"
    modulate: Any = None
    measure: str = "compressed"
    combos: int | None = None
    ridge_range: tuple[float, float] = (1e-6, 1e6)
    noise_rate: float = 0.3
    drift: float = 0.01
    ir_parts: Any = None
    explore: float = 0.0
    draw: bool = False

    @staticmethod
    def exploration_scale(chunk, counts, explore):

        eig = chunk.eigenvalues.to(counts).clamp_min(0)
        prior = eig / eig.max().clamp_min(1e-30)
        score = prior.sqrt() * (torch.log1p(counts.sum()) / (1 + counts)).sqrt()
        return (1 + explore * score).clamp(max=3.0)

    def _rotation(self, state: ChunkState) -> list[tuple[str, int]]:
        pool = self._pool(state)
        if not pool:
            return []
        start = (state.events * self.per_event) % len(pool)
        return [pool[(start + i) % len(pool)] for i in range(min(self.per_event, len(pool)))]

    def step(
        self,
        state: ChunkState,
        loss: Callable[[Mapping[str, torch.Tensor], float, int], float],
        route_share: Mapping[str, Sequence[float]] | Callable[[], Mapping],
        seed: int,
    ) -> dict[str, float]:
        fam_rows = {c.family: c for c in self.editor.chunks}
        al = {h: self.alpha_of(h) for h in self.editor.names}
        slopes, bins = {}, {}
        if self.diff not in ("central", "one-sided"):
            raise ValueError(f"unknown difference {self.diff!r}")
        if self.measure not in ("compressed", "full"):
            raise ValueError(f"unknown measurement {self.measure!r}")
        pool = self._pool(state)
        chosen = self._rotation(state) if self.measure == "full" else pool
        base = {}
        if self.diff == "one-sided" and chosen:
            zero = {h: torch.zeros(self.editor.shapes[h]) for h in self.editor.names}
            base = {t_: loss(zero, t_, seed) for t_ in self.times}

        def measure(d):
            s, per_bin = 0.0, []
            for t_, w, e in zip(self.times, self.weights, self.eps, strict=True):
                plus = {h: e * al[h] * v for h, v in d.items()}
                if self.diff == "one-sided":
                    y = (loss(plus, t_, seed) - base[t_]) / e
                else:
                    minus = {h: -v for h, v in plus.items()}
                    y = (loss(plus, t_, seed) - loss(minus, t_, seed)) / (2 * e)
                s += w * y
                per_bin.append(y)
            return s, per_bin

        if self.measure == "full":
            for fam, j in chosen:
                d = self.editor.per_hook(fam_rows[fam].q[j : j + 1])
                slopes[(fam, j)], bins[(fam, j)] = measure({h: v[0] for h, v in d.items()})
        elif pool:
            est, _ = self._compressed(state, pool, measure, seed)
            for i, key in enumerate(pool):
                slopes[key] = float(est[i])
                bins[key] = [float(est[i])] * len(self.times)
        for fam in state.coef:
            row = torch.full((state.hist[fam].shape[1],), float("nan"))
            for (f_, j), s in slopes.items():
                if f_ == fam:
                    row[j] = s
            state.hist[fam] = torch.cat([state.hist[fam][1:], row[None]])
        step = {}
        for c in self.editor.chunks:
            if c.family not in state.enabled:
                continue
            h_ = state.hist[c.family]
            m = torch.nanmean(h_, 0)
            step[c.family] = -torch.nan_to_num(m, nan=0.0)
        info = {}
        if self.modulate is not None:
            n_d = sum(c.q.shape[0] for c in self.editor.chunks)
            sl = torch.full((n_d, len(self.times)), float("nan"))
            off, offs = 0, {}
            for c in self.editor.chunks:
                offs[c.family] = off
                off += c.q.shape[0]
            for (f_, j), ys in bins.items():
                sl[offs[f_] + j] = torch.tensor(ys)
            with torch.no_grad():
                mult = self.modulate(state.extra, sl)
            for f_ in step:
                k = step[f_].shape[0]
                step[f_] = step[f_] * mult[offs[f_] : offs[f_] + k]
        if callable(route_share):
            route_share = route_share()
        share = (
            torch.tensor([sum(v) / len(v) for v in zip(*route_share.values(), strict=True)])
            if route_share
            else None
        )
        e_n = next(iter(state.coef.values())).shape[0]
        sh = share if share is not None and len(share) == e_n else torch.full((e_n,), 1.0 / e_n)
        from kilora.compute import _IR_INTERPRET_ALL

        override = (self.ir_parts or {}).get("live_write")
        if _IR_INTERPRET_ALL.get() or override is not None:
            from types import SimpleNamespace

            from kilora.ir.defaults import interpret_live_write
            from kilora.ir.dispatch import active_overrides, interpreted_part

            histories = {f: -v.unsqueeze(0) for f, v in step.items()}
            zeros = {f: torch.zeros_like(state.coef[f]) for f in step}
            if override is not None:
                interpreted_part([SimpleNamespace(ir_parts={"live_write": override})])
            with active_overrides({"live_write": override} if override is not None else {}):
                tmp = interpret_live_write(
                    histories, zeros, sh, self.write_norm, 0.0, norm=self._norm
                )
        else:
            tmp = {f: sh[:, None] * v[None] for f, v in step.items()}
            norm = self._norm(tmp)
            fac = self.write_norm / norm if norm > 0 else 0.0
            tmp = {f: fac * v for f, v in tmp.items()}
        if self.explore > 0 and self.draw:
            for chunk in self.editor.chunks:
                if chunk.family in tmp:
                    count = state.extra.get("explore_count/" + chunk.family)
                    if count is None:
                        count = torch.zeros(chunk.q.shape[0], dtype=tmp[chunk.family].dtype)
                    scale = self.exploration_scale(chunk, count, self.explore)
                    tmp[chunk.family] = tmp[chunk.family] * scale.to(tmp[chunk.family])
            norm = self._norm(tmp)
            if norm > self.write_norm:
                tmp = {f: v * (self.write_norm / norm) for f, v in tmp.items()}
        old_coef = {f: v.clone() for f, v in state.coef.items()} if self.explore > 0 else None
        if self.chunk_cap is not None:
            for f, v in tmp.items():
                en = self._norm({f: v})
                if en > self.chunk_cap:
                    tmp[f] = v * (self.chunk_cap / en)
        for fam in state.coef:
            if fam in state.enabled:
                state.coef[fam] = (1 - self.decay) * state.coef[fam] + tmp.get(fam, 0)
        if self.total_cap is not None:
            en = self._norm({f: state.coef[f] for f in state.enabled})
            if en > self.total_cap:
                for f in state.enabled:
                    state.coef[f] = state.coef[f] * (self.total_cap / en)
        if self.effect_cap is not None:
            en = self.effect_norm({f: state.coef[f] for f in state.enabled})
            if en > self.effect_cap:
                for f in state.enabled:
                    state.coef[f] = state.coef[f] * (self.effect_cap / en)
        if old_coef is not None:
            for chunk in self.editor.chunks:
                key = "explore_count/" + chunk.family
                previous = state.extra.get(key, torch.zeros(chunk.q.shape[0]))
                delta = (state.coef[chunk.family] - old_coef[chunk.family]).abs().sum(dim=0)
                state.extra[key] = previous.to(delta) * 0.9 + delta
        state.events += 1
        return {**{f"{f}:{j}": s for (f, j), s in slopes.items()}, **info}

    def _pool(self, state: ChunkState) -> list[tuple[str, int]]:
        return [
            (c.family, j)
            for c in self.editor.chunks
            if c.family in state.enabled
            for j in range(c.q.shape[0])
        ]

    def _compressed(self, state, pool, measure, seed):
        n = len(pool)
        m = self.combos if self.combos is not None else max(1, round(n / 4))
        m = max(1, min(int(m), n))
        g = torch.Generator().manual_seed((int(seed) * 1_000_003 + state.events * 7919) % (2**63))
        psi = torch.randn(m, n, generator=g, dtype=torch.float64)
        psi = psi / psi.norm(dim=1, keepdim=True)
        offsets, off = {}, 0
        for c in self.editor.chunks:
            if c.family in state.enabled:
                offsets[c.family] = off
                off += c.q.shape[0]
        meas = torch.zeros(m, dtype=torch.float64)
        per_bin = []
        for r in range(m):
            d = None
            for c in self.editor.chunks:
                if c.family not in offsets:
                    continue
                k = c.q.shape[0]
                w = psi[r, offsets[c.family] : offsets[c.family] + k].to(c.q.dtype)
                part = {h: v[0] for h, v in self.editor.per_hook(w[None] @ c.q).items()}
                d = part if d is None else {h: d[h] + part[h] for h in d}
            s, pb = measure(d)
            meas[r] = s
            per_bin.append(pb)
        prev = state.prior.get("slopes")
        cov = state.prior.get("cov")
        fresh = prev is None or prev.numel() != n or cov is None or cov.shape != (n, n)
        if fresh:
            prev = torch.zeros(n, dtype=torch.float64)
            p0 = max(float(meas.pow(2).mean()), 1e-12)
            cov = p0 * torch.eye(n, dtype=torch.float64)
            noise = torch.tensor([self.ridge_range[0] * p0, p0], dtype=torch.float64)
        else:
            prev, cov = prev.double(), cov.double()
            noise = state.prior["noise"].double()
            cov = cov + self.drift * float(noise[1]) * torch.eye(n, dtype=torch.float64)
        resid0 = meas - psi @ prev
        pred = psi @ cov @ psi.t()
        if not fresh:
            sig = max(float(resid0.pow(2).mean()) - float(pred.diagonal().mean()), 0.0)
            sig = (1 - self.noise_rate) * float(noise[0]) + self.noise_rate * sig
            noise = torch.tensor(
                [max(sig, self.ridge_range[0] * float(noise[1])), float(noise[1])],
                dtype=torch.float64,
            )
        gain = (
            cov
            @ psi.t()
            @ torch.linalg.inv(pred + float(noise[0]) * torch.eye(m, dtype=torch.float64))
        )
        est = prev + gain @ resid0
        cov = cov - gain @ psi @ cov
        cov = (cov + cov.t()) / 2
        lam = float(noise[0]) / max(float(cov.diagonal().mean()), 1e-30)
        state.prior = {
            "slopes": est.float(),
            "cov": cov.float(),
            "noise": noise.float(),
            "lambda": torch.tensor([lam]),
        }
        return est.float(), per_bin

    def alpha_of(self, hook: str) -> float:
        a = self.editor.alpha
        return float(a.get(hook_kind(hook), sum(a.values()) / len(a)))

    def effect_norm(self, coefs: Mapping[str, torch.Tensor]) -> float:
        rows = {c.family: c for c in self.editor.chunks}
        if all(not rows[f].mean_keys for f in coefs):
            chunks = [c for c in self.editor.chunks if c.family in coefs]
            if not chunks:
                return 0.0
            directions = torch.cat([c.q.float() for c in chunks])
            gram = directions @ directions.T
            identity = torch.eye(len(directions), dtype=gram.dtype, device=gram.device)
            if torch.allclose(gram, identity, atol=1e-6, rtol=1e-6):
                return sum(float(v.pow(2).sum()) for v in coefs.values()) ** 0.5
            values = torch.cat([coefs[c.family].float() for c in chunks], dim=1)
            return float((values @ directions).square().sum()) ** 0.5
        tot = 0.0
        for f, v in coefs.items():
            c = rows[f]
            if not c.mean_keys:
                tot += float(v.pow(2).sum())
                continue
            ph = self.editor.per_hook(c.q)
            for h, d in ph.items():
                du = torch.einsum("ej,jor->eor", v.float(), d)
                k = c.mean_keys.get(h)
                tot += float(
                    du.pow(2).sum()
                    if k is None
                    else torch.einsum("eor,rs,eos->", du, k.float(), du)
                )
        return tot**0.5

    def _norm(self, coefs: Mapping[str, torch.Tensor]) -> float:
        if self.norm is not None:
            return float(self.norm(coefs))
        chunks = [c for c in self.editor.chunks if c.family in coefs]
        if not chunks:
            return 0.0
        rows = torch.cat([c.q.float() for c in chunks])
        gram = rows @ rows.T
        identity = torch.eye(len(rows), device=gram.device, dtype=gram.dtype)
        if torch.allclose(gram, identity, atol=1e-6, rtol=1e-6):
            return float(sum(float(v.pow(2).sum()) for v in coefs.values())) ** 0.5
        values = torch.cat([coefs[c.family].float() for c in chunks], dim=1)
        return float((values @ rows).square().sum()) ** 0.5


def editor_entry():
    return {"editor": ChunkEditor, "state": ChunkState, "writer": ChunkWriter}
