# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import derived_cache
from .adapter import imported_settings
from .errors import KiLoRAError
from .host_copy import nonzero_flags
from .import_controller import (
    DECAY,
    DECAY_AFTER,
    MEASURE_COMBOS,
    SHRINK,
    SLOPE_EPS,
    TRUST_RATIO,
    TRUST_STEPS,
    WRITE_FRAC,
    combinations,
    compressed_update,
    sign_step,
)
from .import_terms import (
    PACK_GAIN_EVENTS,
    PACK_TERM_MOVES,
    TERM_FIXED,
    TERM_PARTS,
    add_direction,
    change_norm,
    decode_ids,
    difference,
    has_terms,
    pack_directions,
    pack_moves_start,
    term_pair,
)
from .layout import Layout
from .live import SESSIONS, LiveSession, LiveSpec, Pairing
from .modfile import Mod, load_mod, save_mod


class ImportedLiveSession(LiveSession):
    MEASURE_SAMPLES = 4

    def __init__(self, adapter_set, sources, family, shapes, saves_dir, key, sections=()):
        experts = {
            target: dataclasses.replace(
                ex,
                down=ex.down.detach().clone(),
                up=ex.up.detach().clone(),
                routing_source=derived_cache.source(ex),
            )
            for target, ex in adapter_set.experts.items()
        }
        for ex in experts.values():
            derived_cache.stamp(ex, ex.routing_source)
        self._source = dataclasses.replace(adapter_set, experts=experts)
        targets = tuple(experts)
        first = next(iter(experts.values()))
        layout = Layout(
            id=key,
            targets=targets,
            widths=tuple(experts[t].up.shape[-1] for t in targets),
            in_widths=tuple(experts[t].down.shape[-1] for t in targets),
            experts=first.down.shape[0],
            per_expert=first.down.shape[1] * first.down.shape[2],
            key_dim=first.keys_in.shape[-1],
        )
        spec = LiveSpec(layout, "plugin", adapter_set.name)
        manifest = {"id": key, "version": "1.0.0", "digest": key, "live": spec.to_manifest()}
        super().__init__(
            Mod(manifest, [self._source], imported_settings()), Pairing(True), saves_dir
        )
        self.sources = [dict(source) for source in sources]
        self.family = family
        self.shapes = json.loads(json.dumps(shapes))
        self.clean = {t: {"down": ex.down, "up": ex.up} for t, ex in experts.items()}
        self._sections = list(sections)
        self._section_pairs = set()
        for section in self._sections:
            if section.kind not in ("fixed-lokr", "fixed-delta") or not section.meta.get("lycoris"):
                continue
            expert = self._section_expert(section)
            if expert is not None:
                self._section_pairs.update((target, expert) for target in section.data)
        extra_targets = sorted({target for target, _ in self._section_pairs} - set(targets))
        if extra_targets:
            self.layout = dataclasses.replace(
                self.layout,
                targets=self.layout.targets + tuple(extra_targets),
                widths=self.layout.widths + (0,) * len(extra_targets),
                in_widths=self.layout.in_widths + (0,) * len(extra_targets),
            )
        self.state = self._reset_state()
        self.edits = "on"
        self.core_dims = 2
        self.controller = "r1"
        self.measure = "compressed"
        self.measure_combos = MEASURE_COMBOS

    def _section_expert(self, section):
        matches = [
            i
            for i, source in enumerate(self.sources)
            if Path(source.get("path", "")).stem == section.name
        ]
        if len(matches) == 1:
            return matches[0]
        index, separator, name = section.name.partition("-")
        if separator and index.isdigit():
            expert = int(index)
            if (
                expert < len(self.sources)
                and Path(self.sources[expert].get("path", "")).stem == name
            ):
                return expert
        return None

    def _reset_state(self):
        state = {}
        for target, ex in self._source.experts.items():
            ranks = ex.ranks or tuple(ex.down.shape[2] for _ in range(ex.down.shape[0]))
            state[target] = {}
            for expert, rank in enumerate(ranks):
                if rank and expert not in ex.exact_unavailable:
                    if expert not in ex.exact:
                        state[target][f"core_{expert}"] = torch.eye(rank)
                    state[target][f"strength_{expert}"] = torch.ones(1)
        for target, expert in self._section_pairs:
            state.setdefault(target, {}).setdefault(f"strength_{expert}", torch.ones(1))
        return state

    def _neutral(self, state=None):
        neutral = self._reset_state()
        for target, parts in (self.state if state is None else state).items():
            if has_terms(parts):
                row = neutral.setdefault(target, {})
                for key in TERM_FIXED:
                    row[key] = parts[key].clone()
                row["terms_c"] = torch.zeros_like(parts["terms_c"])
        return neutral

    @torch.no_grad()
    def _terms_set(self, state):
        from .compute import orthonormal_keys
        from .sets import AdapterSet, KeyedExperts

        experts = {}
        gen = torch.Generator().manual_seed(0)
        for target, parts in state.items():
            pair = term_pair(parts)
            if pair is None:
                continue
            up, down = pair
            experts[target] = KeyedExperts(
                down=down[None, None].clone(),
                up=up.T[None, None].clone(),
                keys_in=orthonormal_keys(torch.randn(1, 1, self._source.router_dim, generator=gen)),
            )
        if not experts:
            return None
        return AdapterSet(
            name="edit-terms",
            part_type=("lora",),
            experts=experts,
            routers={w: r.clone() for w, r in self._source.routers.items()},
            aggregation="accumulate",
            form="input",
            strength=1.0,
            extra={
                "trained_with_kilora": False,
                "live_id": f"{self.mod_id}:edit-terms",
                "apply_once": True,
            },
        )

    @torch.no_grad()
    def sections(self, use_pending=True):
        if self.edits == "off":
            return self._sections
        state = (
            self._pinned_state
            if self.pinned is not None
            else self.pending
            if use_pending and self.pending is not None
            else self.state
        )
        result = []
        for section in self._sections:
            expert = self._section_expert(section)
            if (
                section.kind not in ("fixed-lokr", "fixed-delta")
                or not section.meta.get("lycoris")
                or expert is None
            ):
                result.append(section)
                continue
            data = dict(section.data)
            for target, value in data.items():
                strength = float(state[target][f"strength_{expert}"].item())
                if strength != 1.0:
                    if section.kind == "fixed-lokr":
                        data[target] = (value[0] * strength, value[1])
                    else:
                        data[target] = value * strength
            result.append(dataclasses.replace(section, data=data))
        return result

    @torch.no_grad()
    def sets(self, use_pending=True):
        if self.edits == "off":
            return [self._source]
        state = (
            self._pinned_state
            if self.pinned is not None
            else self.pending
            if use_pending and self.pending is not None
            else self.state
        )
        experts = {}
        for target, ex in self._source.experts.items():
            parts = state[target]
            up = ex.up.clone()
            exact = dict(ex.exact)
            changed = False
            for expert in range(up.shape[0]):
                core = parts.get(f"core_{expert}")
                strength = parts.get(f"strength_{expert}")
                if strength is None:
                    continue
                if expert in exact:
                    if float(strength.item()) != 1.0:
                        components = exact[expert]
                        exact[expert] = (
                            components[0] * strength.to(components[0]),
                            *components[1:],
                        )
                        changed = True
                    continue
                if core is None:
                    continue
                rank = core.shape[0]
                if (
                    torch.equal(core, torch.eye(rank, dtype=core.dtype, device=core.device))
                    and float(strength.item()) == 1.0
                ):
                    continue
                changed = True
                with torch.no_grad():
                    up[expert, 0, :rank] = strength.to(up) * (
                        core.to(up).T @ ex.up[expert, 0, :rank]
                    )
            experts[target] = (
                dataclasses.replace(ex, up=up, exact=exact, routing_down=ex.down, routing_up=ex.up)
                if changed
                else ex
            )
        result = [dataclasses.replace(self._source, experts=experts)]
        terms = self._terms_set(state)
        return result if terms is None else [*result, terms]

    def set_options(
        self,
        *args,
        edits=None,
        core_dims=None,
        controller=None,
        measure=None,
        measure_combos=None,
        **kwargs,
    ):
        if controller is not None and controller not in ("r1", "rprop"):
            raise KiLoRAError("controller must be r1 or rprop")
        if measure is not None and measure not in ("compressed", "full"):
            raise KiLoRAError("measure must be compressed or full")
        if measure_combos is not None and (
            isinstance(measure_combos, bool)
            or not isinstance(measure_combos, int)
            or measure_combos < 1
        ):
            raise KiLoRAError("measure_combos must be a positive integer")
        if core_dims is not None and (
            isinstance(core_dims, bool) or not isinstance(core_dims, int) or core_dims < 0
        ):
            raise KiLoRAError("core_dims must be a non-negative integer")
        if edits is not None and edits not in ("on", "off"):
            raise KiLoRAError("edits must be on or off")
        result = super().set_options(*args, **kwargs)
        if edits is not None:
            self.edits = edits
        if core_dims is not None:
            self.core_dims = core_dims
        if controller is not None:
            self.controller = controller
        if measure is not None:
            self.measure = measure
        if measure_combos is not None:
            self.measure_combos = measure_combos
        return result

    def _load_editor_state(self, tensors, kind, state=None):
        if tensors is not None and kind == "import-r1":
            tensors = {
                key if key.startswith("prior/") else f"prior/history/{key}": value
                for key, value in tensors.items()
            }
        return super()._load_editor_state(tensors, kind, state)

    def moves(self):
        return moves(self, core_dims=self.core_dims)

    def save_edits(self, path):
        tensors = {
            f"kilora/edit/{target}/{part}": value
            for target, parts in self.state.items()
            for part, value in parts.items()
        }
        return save_mod(
            path,
            [],
            self.mod.settings,
            extra_tensors=tensors,
            id=self.mod_id,
            name=self.mod_id,
            version="1.0.0",
            base_model={"family": self.family},
            edits={
                "sources": self.sources,
                "shapes": self.shapes,
                "dimensions": {
                    target: {part: list(value.shape) for part, value in parts.items()}
                    for target, parts in self.state.items()
                },
            },
        )

    @torch.no_grad()
    def export(self, path, format="kmod", include_sources=False, export_rank=None):
        import os

        destination = Path(path)
        for source in self.sources:
            original = Path(source["path"])
            if destination.resolve() == original.resolve() or (
                destination.exists()
                and original.exists()
                and os.path.samefile(destination, original)
            ):
                raise KiLoRAError("export cannot overwrite a source file")
        if format not in ("kmod", "lora"):
            raise KiLoRAError("export format must be kmod or lora")
        if export_rank is not None and (
            isinstance(export_rank, bool) or not isinstance(export_rank, int) or export_rank < 1
        ):
            raise KiLoRAError("export_rank must be a positive integer")
        if format == "kmod":
            tensors = {
                f"kilora/edit/{target}/{part}": value
                for target, parts in self.state.items()
                for part, value in parts.items()
            }
            sets = [dataclasses.replace(s, frozen=True) for s in self.sets(use_pending=False)]
            return save_mod(
                path,
                sets if include_sources else [],
                self.mod.settings,
                sections=self.sections(use_pending=False) if include_sources else None,
                extra_tensors=tensors,
                id=self.mod_id,
                name=self.mod_id,
                version="1.0.0",
                base_model={"family": self.family},
                edits={
                    "sources": self.sources,
                    "shapes": self.shapes,
                    "options": {
                        "edits": self.edits,
                        "core_dims": self.core_dims,
                        "measure_samples": self.measure_samples,
                    },
                    "dimensions": {
                        t: {p: list(v.shape) for p, v in parts.items()}
                        for t, parts in self.state.items()
                    },
                },
            )

        from .comfyui.takeover import _write

        changes = {}
        ranks = {}
        for setting in self.sets(use_pending=False):
            for target, ex in setting.experts.items():
                value = torch.zeros(ex.up.shape[-1], ex.down.shape[-1], dtype=torch.float64)
                for expert in range(ex.down.shape[0]):
                    if expert in ex.exact:
                        parts = [p.detach().cpu().double() for p in ex.exact[expert]]
                        change = (
                            torch.kron(*parts)
                            if len(parts) == 2
                            else (parts[0] @ parts[1]) * (parts[2] @ parts[3])
                        )
                    else:
                        change = (
                            ex.up[expert, 0].detach().cpu().double().T
                            @ ex.down[expert, 0].detach().cpu().double()
                        )
                    value += change * setting.strength
                changes[target] = changes.get(target, 0) + value
                ranks[target] = ranks.get(target, 0) + sum(
                    ex.ranks or (ex.down.shape[2],) * ex.down.shape[0]
                )
        for section in self.sections(use_pending=False):
            if section.kind not in ("fixed-lora", "fixed-delta", "fixed-lokr"):
                raise KiLoRAError(f"cannot export section {section.kind} as a plain LoRA")
            for target, value in section.data.items():
                if section.kind == "fixed-lora":
                    down, up = value
                    delta = up.double().cpu() @ down.double().cpu()
                    ranks[target] = ranks.get(target, 0) + down.shape[0]
                elif section.kind == "fixed-lokr":
                    delta = torch.kron(*(v.double().cpu() for v in value))
                else:
                    delta = value.double().cpu()
                changes[target] = changes.get(target, 0) + delta
                ranks.setdefault(target, min(delta.shape))
        state, residuals = {}, {}
        for target, delta in changes.items():
            rank = min(export_rank or max(1, ranks[target]), *delta.shape)
            down, up, error = export_low_rank(delta, rank)
            state[f"transformer.{target}.lora_A.weight"] = down
            state[f"transformer.{target}.lora_B.weight"] = up
            residuals[target] = error
        note = {"note": "Routing is not kept in a plain LoRA.", "relative_svd_residual": residuals}
        _write(state, 0, path=path, metadata={"kilora.export": json.dumps(note, sort_keys=True)})
        return note

    def load_edits(self, path):
        from safetensors import safe_open

        mod = load_mod(path)
        if mod.manifest.get("id") != self.mod_id:
            raise KiLoRAError("edit set belongs to a different import")
        expected = self._reset_state()
        with safe_open(str(path), framework="pt", device="cpu") as file:
            names = {
                f"kilora/edit/{target}/{part}"
                for target, parts in expected.items()
                for part in parts
            }
            saved_names = file.keys()
            edit_names = {name for name in saved_names if name.startswith("kilora/edit/")}
            for name in sorted(edit_names - names):
                target, _, part = name.removeprefix("kilora/edit/").rpartition("/")
                if part not in TERM_PARTS or target not in expected:
                    continue
                group = {f"kilora/edit/{target}/{key}" for key in TERM_PARTS}
                if not group <= edit_names:
                    raise KiLoRAError("edit set has incomplete terms")
                names |= group
                expected[target].update(
                    {key: file.get_tensor(f"kilora/edit/{target}/{key}") for key in TERM_PARTS}
                )
            if edit_names != names:
                raise KiLoRAError("edit set does not match the imported layers")
            for target, parts in expected.items():
                if has_terms(parts):
                    self._check_terms(target, parts)
                for part, value in parts.items():
                    if part in TERM_PARTS:
                        continue
                    saved = file.get_tensor(f"kilora/edit/{target}/{part}")
                    if saved.shape != value.shape or not torch.isfinite(saved).all():
                        raise KiLoRAError("invalid edit tensor")
                    parts[part] = saved.float()
        self.undo_log.push(
            self.state, expected, self.commits, self.editor_snapshot(), self.previous_snapshot()
        )
        self.state = expected
        self.pending = None
        self.revision += 1

    def _check_terms(self, target, parts):
        ex = self._source.experts.get(target)
        try:
            out_ids, in_ids = decode_ids(parts["terms_ids"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise KiLoRAError("invalid edit terms") from exc
        p_out, p_in, c = parts["terms_out"], parts["terms_in"], parts["terms_c"]
        if (
            ex is None
            or p_out.ndim != 2
            or p_in.ndim != 2
            or p_out.shape != (ex.up.shape[-1], len(out_ids))
            or p_in.shape != (ex.down.shape[-1], len(in_ids))
            or c.shape != (len(out_ids), len(in_ids))
            or not all(torch.isfinite(t).all() for t in (p_out, p_in, c))
        ):
            raise KiLoRAError("invalid edit terms")
        parts["terms_out"], parts["terms_in"] = p_out.float(), p_in.float()
        parts["terms_c"] = c.float()

    def _write(self, kind, name):
        result = super()._write(kind, name)
        if self.saves_dir is not None:
            self.save_edits(Path(self.saves_dir) / self.mod_id / "edits.kmod")
        return result


def imported_session_for(
    adapter_set, sources, family, shapes, saves_dir=None, *, sections=(), create=True
):
    sources = [dict(source) for source in sources]
    identity = sorted((str(s["sha256"]), float(s["strength"]), str(s["axis"])) for s in sources)
    payload = json.dumps(
        [
            identity,
            family,
            shapes,
            [
                (target, sorted(ex.exact), ex.exact_unavailable)
                for target, ex in sorted(adapter_set.experts.items())
            ],
            [
                (s.kind, s.name, sorted(s.data))
                for s in sections
                if s.kind in ("fixed-lokr", "fixed-delta")
            ],
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    key = "imp-" + hashlib.sha256(payload.encode()).hexdigest()[:60]
    if key not in SESSIONS and create:
        SESSIONS[key] = ImportedLiveSession(
            adapter_set, sources, family, shapes, saves_dir, key, sections
        )
    return SESSIONS.get(key)


@dataclasses.dataclass(frozen=True)
class Move:
    id: str
    target: str
    expert: int
    part: str
    direction: torch.Tensor

    @torch.no_grad()
    def apply(self, state, delta):
        key = f"{self.part}_{self.expert}"
        state[self.target][key] = state[self.target][key] + delta * self.direction.to(
            state[self.target][key]
        )


@dataclasses.dataclass(frozen=True)
class TermMove:
    id: str
    target: str
    output_id: str
    input_id: str
    output: torch.Tensor
    input: torch.Tensor
    expert: int = -1
    part: str = "term"

    @torch.no_grad()
    def apply(self, state, delta):
        parts = state[self.target]
        i, j = add_direction(parts, self.output_id, self.output, self.input_id, self.input)
        c = parts["terms_c"].clone()
        c[i, j] += delta
        parts["terms_c"] = c

    @property
    def effect(self):
        return float(self.output.double().norm() * self.input.double().norm())


def term_moves(state):
    result = []
    for target, parts in state.items():
        if not has_terms(parts):
            continue
        out_ids, in_ids = decode_ids(parts["terms_ids"])
        for i, j in parts["terms_c"].nonzero().tolist():
            result.append(
                TermMove(
                    f"{target}:term:{out_ids[i]}:{in_ids[j]}",
                    target,
                    out_ids[i],
                    in_ids[j],
                    parts["terms_out"][:, i],
                    parts["terms_in"][:, j],
                )
            )
    return result


def _gram_root(gram):

    w, v = torch.linalg.eigh(gram)
    return torch.sqrt(w.clamp_min(0))[:, None] * v.T


def _build_grams(ex, expert, rank):
    a = ex.down[expert, 0, :rank].detach().double()
    b = ex.up[expert, 0, :rank].detach().double().T
    return a @ a.T, b.T @ b


def _grams(session, target, ex, expert, rank):
    cache = session.__dict__.setdefault("_grams", {})
    key = (target, expert, rank)
    entry = cache.get(key)
    if entry is None or entry[0] is not ex:
        ga, gb = _build_grams(ex, expert, rank)
        source = max(float(torch.trace(gb @ ga)), 0.0) ** 0.5
        entry = cache[key] = (ex, ga, gb, source)
    return entry[1:]


def _gram_norm(x, gb, ga):
    return max(float(torch.trace(x.T @ gb @ x @ ga)), 0.0) ** 0.5


TRUST_ROOT_TOL = 1e-12


def _trust_coefficients(jobs):
    result = [None] * len(jobs)
    groups = {}
    for i, job in enumerate(jobs):
        groups.setdefault((job[2].shape[0], job[4].device), []).append(i)
    for (rank, device), index in groups.items():
        s0 = torch.stack([jobs[i][0] for i in index]).reshape(-1, 1, 1)
        s1 = torch.stack([jobs[i][1] for i in index]).reshape(-1, 1, 1)
        m0 = torch.stack([jobs[i][2] for i in index])
        m1 = torch.stack([jobs[i][3] for i in index])
        ga = torch.stack([jobs[i][4] for i in index])
        gb = torch.stack([jobs[i][5] for i in index])
        ds, dm = s1 - s0, m1 - m0
        eye = torch.eye(rank, dtype=torch.float64, device=device)
        x = torch.stack([s0 * m0 - eye, s0 * dm + ds * m0, ds * dm], dim=1)
        z = gb[:, None] @ x @ ga[:, None]
        inner = torch.einsum("nirc,njrc->nij", x, z).cpu()
        for k, i in enumerate(index):
            g = inner[k]
            limit = jobs[i][6]
            result[i] = [
                float(g[0, 0]) - limit * limit,
                2 * float(g[0, 1]),
                2 * float(g[0, 2]) + float(g[1, 1]),
                2 * float(g[1, 2]),
                float(g[2, 2]),
            ]
    return result


def _trust_root(coefficients):
    p = np.asarray(coefficients, dtype=np.float64)
    scale = float(np.abs(p).max()) if p.size else 0.0
    if not np.isfinite(p).all() or scale <= 0:
        return None
    desc = p[::-1].copy()
    while desc.size and abs(desc[0]) <= TRUST_ROOT_TOL * scale:
        desc = desc[1:]
    if desc.size < 2:
        return None
    roots = np.roots(desc)
    real = [
        float(r.real)
        for r in roots
        if abs(r.imag) <= 1e-9 * (1 + abs(r)) and -TRUST_ROOT_TOL <= r.real <= 1 + TRUST_ROOT_TOL
    ]
    if not real:
        return None
    t = max(real)
    slope = np.polyder(desc)
    for _ in range(2):
        d = float(np.polyval(slope, t))
        if d == 0 or not np.isfinite(d):
            break
        step = t - float(np.polyval(desc, t)) / d
        if not 0.0 <= step <= 1.0 or abs(step - t) > 1e-6:
            break
        t = step
    return min(max(t, 0.0), 1.0)


def _core_frame(session, target, ex, expert, rank):
    cache = session.__dict__.setdefault("_core_frames", {})
    key = (target, expert, rank)
    if key not in cache:
        ga, gb, _ = _grams(session, target, ex, expert, rank)

        def frame(device):
            rb, ra = _gram_root(gb.to(device)), _gram_root(ga.to(device))
            u, sv, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
            left = (torch.linalg.pinv(rb, rtol=1e-10) @ u).float().cpu()
            right = (torch.linalg.pinv(ra, rtol=1e-10) @ vh.T).float().cpu()
            usable = int((sv > sv.max() * 1e-10).sum()) if sv.numel() else 0
            return left, right, usable

        result = (torch.zeros(rank, 0), torch.zeros(rank, 0), 0)
        for device in dict.fromkeys((ga.device, torch.device("cpu"))):
            try:
                result = frame(device)
                break
            except torch.linalg.LinAlgError:
                continue
        cache[key] = result
    return cache[key]


def import_source_frames(session, weights=None):
    from .recording import orthonormal_rows

    cache = session.__dict__.get("_cross_frames")
    if cache is not None:
        return cache
    cache = {}
    weights = weights or {}
    with torch.no_grad():
        for target, ex in session._source.experts.items():
            weight = weights.get(target)
            if weight is None or target not in session.state:
                continue
            ranks = ex.ranks or tuple(ex.down.shape[2] for _ in range(ex.down.shape[0]))
            for expert, rank in enumerate(ranks):
                if not rank or f"strength_{expert}" not in session.state[target]:
                    continue
                a = ex.down[expert, 0, :rank].detach().double()
                b = ex.up[expert, 0, :rank].detach().double().T
                norms = a.norm(dim=1, keepdim=True)
                p_out = orthonormal_rows(b)
                if p_out is None or not bool((norms > 0).all()):
                    continue
                p_in = a / norms
                if weight.shape != (b.shape[0], a.shape[1]):
                    continue
                w = weight.detach()
                base = p_out.to(w.device) @ (w @ p_in.to(w.device, w.dtype).t()).double()
                cache.setdefault(target, []).append((expert, p_in, p_out, base.cpu()))
    session.__dict__["_cross_frames"] = cache
    return cache


def import_frames(session):
    from .recording import CrossFrame

    out = {}
    with torch.no_grad():
        for target, items in import_source_frames(session).items():
            ex = session._source.experts[target]
            parts = session.state.get(target, {})
            ups, downs = [], []
            ranks = ex.ranks or tuple(ex.down.shape[2] for _ in range(ex.down.shape[0]))
            for expert, rank in enumerate(ranks):
                strength = parts.get(f"strength_{expert}")
                if not rank or strength is None:
                    continue
                core = parts.get(f"core_{expert}")
                b = ex.up[expert, 0, :rank].detach().double().T
                mix = strength.double().reshape(()) * (
                    core.double() if core is not None else torch.eye(rank, dtype=torch.float64)
                )
                ups.append(b @ mix.to(b.device))
                downs.append(ex.down[expert, 0, :rank].detach().double())
            up = torch.cat(ups, 1) if ups else None
            down = torch.cat(downs, 0) if downs else None
            out[target] = [
                CrossFrame(p_in, p_out, "input", None, up, down, base)
                for _, p_in, p_out, base in items
            ]
    return out


def import_block_recorder(session):
    from .recording import BlockFrames, BlockStreamRecorder

    frames = session.__dict__.get("_block_frames")
    if frames is None:
        outputs = {}
        for target, items in import_source_frames(session).items():
            rows = [p_out for _, _, p_out, _ in items]
            if rows:
                outputs[target] = torch.cat(rows, 0)
        frames = BlockFrames(outputs)
        session.__dict__["_block_frames"] = frames
    return BlockStreamRecorder(frames) if frames.outputs else None


def moves(session, core_dims=2):

    if isinstance(core_dims, bool) or not isinstance(core_dims, int) or core_dims < 0:
        raise ValueError("core_dims must be a non-negative integer")
    result = []
    with torch.no_grad():
        for target, ex in session._source.experts.items():
            for expert in range(ex.down.shape[0]):
                core = session.state[target].get(f"core_{expert}")
                if core is None:
                    if f"strength_{expert}" in session.state[target]:
                        result.append(
                            Move(
                                f"{target}:{expert}:strength",
                                target,
                                expert,
                                "strength",
                                torch.ones(1),
                            )
                        )
                    continue
                rank = core.shape[0]
                left, right, usable = _core_frame(session, target, ex, expert, rank)
                count = min(core_dims, usable)
                for i in range(count):
                    for j in range(count):
                        direction = torch.outer(left[:, i], right[:, j])
                        result.append(
                            Move(
                                f"{target}:{expert}:core:{i}:{j}", target, expert, "core", direction
                            )
                        )
                result.append(
                    Move(f"{target}:{expert}:strength", target, expert, "strength", torch.ones(1))
                )
    for target, expert in sorted(session._section_pairs):
        if not any(
            m.target == target and m.expert == expert and m.part == "strength" for m in result
        ):
            result.append(
                Move(f"{target}:{expert}:strength", target, expert, "strength", torch.ones(1))
            )
    return result


IMPORT_WRITE_FRAC = WRITE_FRAC
IMPORT_CAP_FRAC = 2e-2
IMPORT_DECAY = 0.02
IMPORT_EPS = SLOPE_EPS
IMPORT_HISTORY = 8


EXPORT_EXACT_SVD_MAX = 4096 * 4096


def export_low_rank(delta, rank, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    devices = [torch.device(device)]
    if devices[0].type != "cpu":
        devices.append(torch.device("cpu"))
    if delta.numel() <= EXPORT_EXACT_SVD_MAX:
        for place in devices:
            d = delta.detach().to(place, torch.float64)
            try:
                u, sv, vh = torch.linalg.svd(d, full_matrices=False)
            except torch.linalg.LinAlgError:
                continue
            r = min(rank, sv.shape[0])
            down = vh[:r]
            up = u[:, :r] * sv[:r]
            norm = float(d.norm())
            error = float((d - up @ down).norm()) / norm if norm else 0.0
            return down.float().cpu().contiguous(), up.float().cpu().contiguous(), error
    from .multi_import import low_rank

    return low_rank(delta.detach().to("cpu", torch.float64), rank, device="cpu")


def subspace_gradient(psi, meas):
    psi = psi.double()
    return torch.linalg.pinv(psi) @ meas.double()


def cap_step(step):
    return max(-IMPORT_EPS, min(IMPORT_EPS, float(step)))


@torch.no_grad()
def between(state, new, t):
    out = {}
    for target, parts in new.items():
        old = state.get(target, {})
        out[target] = {}
        for key, value in parts.items():
            if key in TERM_FIXED:
                out[target][key] = value.clone()
                continue
            base = old.get(key)
            if base is None:
                base = torch.zeros_like(value)
            elif base.shape != value.shape:
                grown = torch.zeros_like(value)
                grown[tuple(slice(0, size) for size in base.shape)] = base.to(value)
                base = grown
            else:
                base = base.to(value)
            out[target][key] = base + t * (value - base)
    return out


NOT_WRITTEN = "not written: the step did not lower the measured loss"


class ImportHistory:
    def __init__(self, directions):
        self.history = {move.id: torch.full((IMPORT_HISTORY,), float("nan")) for move in directions}
        self.rules = {}
        self.event = 0
        self.max_dims = 0
        self.compressed = None
        self.measure_events = 0
        self.gains = torch.zeros(0, dtype=torch.float64)
        from .recording import BlockMaps

        self.block_maps = BlockMaps()

    def reset(self):
        for row in self.history.values():
            row.fill_(float("nan"))
        self.rules = {}
        self.event = 0
        self.max_dims = 0
        self.compressed = None
        self.measure_events = 0
        self.gains = torch.zeros(0, dtype=torch.float64)

    def add_gain(self, gain):
        gains = torch.cat((self.gains, torch.tensor([float(gain)], dtype=torch.float64)))
        if gains.shape[0] > 2 * PACK_GAIN_EVENTS:
            gains = torch.cat((gains[:PACK_GAIN_EVENTS], gains[-PACK_GAIN_EVENTS:]))
        self.gains = gains

    def state_tensors(self):
        result = {f"prior/history/{key}": value.clone() for key, value in self.history.items()}
        result["prior/event"] = torch.tensor(self.event, dtype=torch.int64)
        result["prior/max_dims"] = torch.tensor(self.max_dims, dtype=torch.int64)
        if self.gains.numel():
            result["prior/gains"] = self.gains.clone()
        result.update({f"prior/rule/{key}": value.clone() for key, value in self.rules.items()})
        if self.measure_events:
            result["prior/compressed/events"] = torch.tensor(self.measure_events, dtype=torch.int64)
        if self.compressed is not None:
            for name in ("slopes", "cov", "noise"):
                result[f"prior/compressed/{name}"] = self.compressed[name].clone()
            for i, key in enumerate(self.compressed["ids"]):
                result[f"prior/compressed/order/{key}"] = torch.tensor(i, dtype=torch.int64)
        maps = getattr(self, "block_maps", None)
        if maps is not None:
            result.update(maps.state_tensors())
        return result

    def load_tensors(self, tensors):
        from .recording import BlockMaps

        self.block_maps = BlockMaps.from_tensors(tensors)
        self.history = {
            key.removeprefix("prior/history/"): value.clone()
            for key, value in tensors.items()
            if key.startswith("prior/history/")
        }
        self.event = int(tensors.get("prior/event", torch.tensor(0)))
        self.max_dims = int(tensors.get("prior/max_dims", torch.tensor(0)))
        gains = tensors.get("prior/gains")
        self.gains = (
            torch.zeros(0, dtype=torch.float64)
            if gains is None
            else gains.detach().to("cpu", torch.float64).reshape(-1).clone()
        )
        self.rules = {
            key.removeprefix("prior/rule/"): value.clone()
            for key, value in tensors.items()
            if key.startswith("prior/rule/")
        }
        self.measure_events = int(tensors.get("prior/compressed/events", torch.tensor(0)))
        order = {
            key.removeprefix("prior/compressed/order/"): int(value)
            for key, value in tensors.items()
            if key.startswith("prior/compressed/order/")
        }
        self.compressed = None
        if order and all(f"prior/compressed/{n}" in tensors for n in ("slopes", "cov", "noise")):
            self.compressed = {
                "ids": sorted(order, key=order.get),
                **{
                    name: tensors[f"prior/compressed/{name}"].clone().double()
                    for name in ("slopes", "cov", "noise")
                },
            }


class ImportTrial:
    replay_all = True

    def __init__(self, session, old, new):
        self.set_key = session._source.extra.get("live_id", session._source.name)
        self.du = {}
        self.exact = {}
        self.exact_norm = 0.0
        self.fixed_changes = {
            (session.mod_id, target, expert): float(
                new[target][f"strength_{expert}"] - old[target][f"strength_{expert}"]
            )
            for target, expert in session._section_pairs
        }
        candidates: list = []
        for target, ex in session._source.experts.items():
            delta = torch.zeros_like(ex.up)
            exact = {}
            for expert, parts in ex.exact.items():
                name = f"strength_{expert}"
                if name not in old[target]:
                    continue
                change = float(new[target][name] - old[target][name])
                if change:
                    exact[expert] = (parts[0] * float(new[target][name]), *parts[1:])
                    size = 1.0
                    for part in parts:
                        size *= float(part.float().norm())
                    self.exact_norm += (change * size) ** 2
            if exact:
                self.exact[target] = exact
            for expert in range(ex.up.shape[0]):
                key, strength = f"core_{expert}", f"strength_{expert}"
                if key not in old[target] or key not in new[target]:
                    continue
                rank = old[target][key].shape[0]
                base = ex.up[expert, 0, :rank]
                before = old[target][strength].to(base) * (old[target][key].to(base).T @ base)
                after = new[target][strength].to(base) * (new[target][key].to(base).T @ base)
                delta[expert, 0, :rank] = after - before
            if exact:
                candidates.append((target, delta, True))
            elif delta.numel():
                candidates.append((target, delta, False))
        unknown = [delta for _, delta, exact in candidates if not exact]
        flags = iter(nonzero_flags(unknown))
        for target, delta, exact in candidates:
            if exact or next(flags):
                self.du[target] = delta.reshape(-1, delta.shape[-1])
        candidates.clear()
        self.terms = {}
        for target in set(old) | set(new):
            change = difference(old.get(target, {}), new.get(target, {}))
            if change is not None:
                self.terms[target] = change
        self.terms_norm = sum(change_norm(*v) ** 2 for v in self.terms.values()) ** 0.5

    def input_delta(self, target, x):
        change = self.terms.get(target)
        if change is None:
            return None
        up, down = change
        return (x.float() @ down.T.to(x.device)) @ up.T.to(x.device)

    def delta(self, target, h):
        return None

    def attention_sets(self, target, sets):
        if target not in self.du:
            return sets
        return [
            dataclasses.replace(
                s,
                up=s.up + self.du[target].to(s.up),
                routing_down=s.routing_down if s.routing_down is not None else s.down,
                routing_up=s.routing_up if s.routing_up is not None else s.up,
                exact={
                    **s.exact,
                    **{
                        expert: tuple(part.to(s.up) for part in parts)
                        for expert, parts in self.exact.get(target, {}).items()
                    },
                },
            )
            if s.key == self.set_key and s.up is not None
            else s
            for s in sets
        ]


NOTE_LARGEST = 5
NOTE_PARTS = {"core": "core", "strength": "strength", "term": "pack term"}


TIMING_ENV = "KILORA_IMPORT_TIMING"


def note_full():
    return os.environ.get("KILORA_IMPORT_NOTE_FULL") == "1"


class ImportEditor:
    needs_path = False
    api_version = "1"
    supports = None

    def __init__(self, session):
        self.session = session
        self.options = {}
        self.note = ""
        self._estimated, self._dim_steps = None, {}
        self._subspace, self._subspace_samples, self._decayed = None, None, None
        self._accepted, self._refused_note = None, ""
        self._note_moves, self._full_moves = {}, ""
        if session.editor_state is None:
            session.attach_editor_state(ImportHistory(session.moves()), "import-r1")

    def configure(self, options):
        self.options = dict(options)

    def check(self, session):
        if session is not self.session:
            raise KiLoRAError("import editor belongs to a different session")

    def recorder(self, session):
        if not getattr(session, "record_prediction", True):
            return None
        from .recording import CrossStepRecorder

        frames = import_frames(session)
        if not frames:
            return None
        return CrossStepRecorder(frames, blocks=import_block_recorder(session))

    def guard_trial(self, state, new):
        return ImportTrial(self.session, state, new)

    def write_note(self):
        if self.session.state is not getattr(self, "_before", None):
            return self.note
        return getattr(self, "_refused_note", "")

    def _trial_norm(self, trial):
        total = trial.exact_norm + sum(float(v.float().norm()) ** 2 for v in trial.du.values())
        for section in self.session._sections:
            if section.kind not in ("fixed-lokr", "fixed-delta") or not section.meta.get("lycoris"):
                continue
            expert = self.session._section_expert(section)
            for target, value in section.data.items():
                amount = trial.fixed_changes.get((self.session.mod_id, target, expert), 0.0)
                if not amount:
                    continue
                if isinstance(value, tuple):
                    size = 1.0
                    for tensor in value:
                        size *= float(tensor.float().norm())
                else:
                    size = float(value.float().norm())
                total += (amount * size) ** 2
        return total**0.5

    def _moves_note(self):
        if note_full():
            return self._full_moves
        if not self._note_moves:
            return ""
        counts = {}
        for move, step in self._note_moves.values():
            label = NOTE_PARTS.get(move.part, move.part)
            measured, written = counts.get(label, (0, 0))
            counts[label] = (measured + 1, written + bool(step))
        measured = sum(c[0] for c in counts.values())
        written = sum(c[1] for c in counts.values())
        parts = ", ".join(f"{label} {m}/{w}" for label, (m, w) in counts.items())
        text = f" Moves: {measured} measured, {written} written ({parts})."
        largest = sorted(
            ((move, step) for move, step in self._note_moves.values() if step),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:NOTE_LARGEST]
        if largest:
            text += " Largest: " + ", ".join(_note_entry(m, s) for m, s in largest) + "."
        return text

    def guard_refused(self):
        if self.session.controller != "rprop":
            return
        store = self.session.editor_state
        for key, initial in getattr(self, "_refused_moves", {}).items():
            row = store.rules.get(
                key, torch.tensor([0.0, initial, initial, 0.0, 0.0], dtype=torch.float64)
            ).clone()
            row[1] *= SHRINK
            store.rules[key] = row

    def _pair_norm(self, state, target, expert):
        ex = self.session._source.experts.get(target)
        core = state[target].get(f"core_{expert}")
        if core is None:
            parts = ex.exact.get(expert) if ex is not None else None
            present = bool(parts) and all(bool(t.count_nonzero()) for t in parts)
            for section in self.session._sections:
                if (
                    section.kind in ("fixed-lokr", "fixed-delta")
                    and self.session._section_expert(section) == expert
                    and target in section.data
                ):
                    value = section.data[target]
                    tensors = value if isinstance(value, tuple) else (value,)
                    present |= all(bool(t.count_nonzero()) for t in tensors)
            if not present:
                return None
            return abs(float(state[target][f"strength_{expert}"]) - 1.0), 1.0
        if ex is None:
            return None
        rank = core.shape[0]
        ga, gb, source = _grams(self.session, target, ex, expert, rank)
        eye = torch.eye(rank, dtype=torch.float64, device=ga.device)
        c = (
            state[target][f"strength_{expert}"].double().to(ga.device) * core.double().to(ga.device)
            - eye
        )
        return _gram_norm(c, gb, ga), source

    @torch.no_grad()
    def _measure_compressed(self, state, record, directions, samples=False):
        if not directions:
            return ({}, None, None) if samples else {}
        store = self.session.editor_state
        ids = [move.id for move in directions]
        count = int(getattr(self.session, "measure_combos", MEASURE_COMBOS))
        digest = hashlib.sha256(self.session.mod_id.encode()).digest()
        seed = int.from_bytes(digest[:7], "big") * 1_000_003 + store.measure_events * 7919
        psi = combinations(count, len(ids), seed)
        meas = torch.zeros(count, dtype=torch.float64)

        def clone():
            return {t: {k: v.clone() for k, v in p.items()} for t, p in state.items()}

        for r in range(count):
            plus, minus = clone(), clone()
            for move, weight in zip(directions, psi[r].tolist(), strict=True):
                move.apply(plus, IMPORT_EPS * weight)
                move.apply(minus, -IMPORT_EPS * weight)
            meas[r] = (
                record.measure(ImportTrial(self.session, state, plus))
                - record.measure(ImportTrial(self.session, state, minus))
            ) / (2 * IMPORT_EPS)
        if not bool(torch.isfinite(meas).all()):
            return None
        est, store.compressed = compressed_update(ids, psi, meas, store.compressed)
        store.measure_events += 1
        estimates = dict(zip(ids, est.tolist(), strict=True))
        return (estimates, psi, meas) if samples else estimates

    @torch.no_grad()
    def _measure_subspace(self, state, record, directions):
        result = self._measure_compressed(state, record, directions, samples=True)
        if result is None:
            return False
        estimated, psi, meas = result
        self._estimated = estimated
        ids = [move.id for move in directions]
        if psi is None:
            self._subspace, self._subspace_samples = {}, (ids, None, None)
            return True
        grad = subspace_gradient(psi, meas)
        if not bool(torch.isfinite(grad).all()):
            return False
        self._subspace = dict(zip(ids, grad.tolist(), strict=True))
        self._subspace_samples = (ids, psi, meas)
        return True

    @torch.no_grad()
    def _trust_radius(self, state, new, rows, all_moves, pair_of):
        store = self.session.editor_state
        neutral = None
        jobs, solves = [], []
        for target, expert in {(m.target, m.expert) for m in all_moves.values()}:
            pair = self._pair_norm(new, target, expert)
            if pair is None or pair[1] <= 0 or pair[0] <= TRUST_RATIO * pair[1]:
                continue
            limit = TRUST_RATIO * pair[1]
            before = pair_of(target, expert)
            if before and before[0] <= limit:
                base = state
            else:
                if neutral is None:
                    neutral = self.session._reset_state()
                base = neutral
            proposed = {
                k: v.clone()
                for k, v in new[target].items()
                if k in (f"core_{expert}", f"strength_{expert}")
            }
            job = self._trust_job(base, proposed, target, expert, limit)
            if job is not None:
                jobs.append(job)
            solves.append((target, expert, base, proposed, limit, job is not None))
        roots = iter(_trust_coefficients(jobs))
        closed = bisected = 0
        for target, expert, base, proposed, limit, has_job in solves:
            lo = _trust_root(next(roots)) if has_job else None
            if lo is None:
                lo = self._bisect_trust(new, base, proposed, target, expert, limit)
                bisected += 1
            else:
                closed += 1
            for k, value in proposed.items():
                new[target][k] = base[target][k] + lo * (value - base[target][k])
            for key, row in rows.items():
                move = all_moves.get(key)
                if move and (move.target, move.expert) == (target, expert):
                    old = (
                        float(store.rules[key][4]) if base is state and key in store.rules else 0.0
                    )
                    row[4] = old + lo * (float(row[4]) - old)
        return closed, bisected

    def _trust_job(self, base, proposed, target, expert, limit):
        skey, ckey = f"strength_{expert}", f"core_{expert}"
        if skey not in proposed or skey not in base.get(target, {}):
            return None
        if ckey in proposed:
            if ckey not in base[target]:
                return None
            ex = self.session._source.experts.get(target)
            if ex is None:
                return None
            rank = proposed[ckey].shape[0]
            ga, gb, _ = _grams(self.session, target, ex, expert, rank)
            m0 = base[target][ckey].double().to(ga.device)
            m1 = proposed[ckey].double().to(ga.device)
        else:
            ga = gb = m0 = m1 = torch.ones(1, 1, dtype=torch.float64)

        def cast(x):
            return x.double().to(ga.device).reshape(())

        s0, s1 = cast(base[target][skey]), cast(proposed[skey])
        return s0, s1, m0, m1, ga, gb, float(limit)

    def _bisect_trust(self, new, base, proposed, target, expert, limit):
        lo, hi = 0.0, 1.0
        for _ in range(TRUST_STEPS):
            mid = (lo + hi) / 2
            for k, value in proposed.items():
                new[target][k] = base[target][k] + mid * (value - base[target][k])
            if self._pair_norm(new, target, expert)[0] <= limit:
                lo = mid
            else:
                hi = mid
        return lo

    def _timing_line(self):
        t = getattr(self, "_rprop_time", None)
        if not t or os.environ.get(TIMING_ENV, "").strip() not in ("1", "on", "true"):
            return ""
        return (
            f" Controller: measurement {t['measure_s']:.2f} s, step {t['step_s']:.2f} s, "
            f"trust radius {t['trust_s']:.2f} s ({t['closed']} closed form, "
            f"{t['bisected']} bisected)."
        )

    @torch.no_grad()
    def _rprop(self, state, record, directions, extra=()):
        if not directions or self.session.speed <= 0:
            return None
        store = self.session.editor_state

        def clone(src):
            return {t: {k: v.clone() for k, v in p.items()} for t, p in src.items()}

        started = time.perf_counter()
        measure_s = 0.0
        new = clone(state)
        rows = {key: value.clone() for key, value in store.rules.items()}
        event = store.event + 1
        norms = {}

        def pair_of(target, expert):
            key = (target, expert)
            if key not in norms:
                norms[key] = self._pair_norm(state, target, expert)
            return norms[key]

        self._refused_moves = {}
        max_dims = max(store.max_dims, self.session.core_dims, 2)
        all_moves = {m.id: m for m in moves(self.session, max_dims)}
        all_moves.update({m.id: m for m in term_moves(state)})
        all_moves.update({m.id: m for m in directions})
        touched = {m.id for m in directions}
        for key, row in rows.items():
            move = all_moves.get(key)
            if move is not None and key not in touched and event - int(row[3]) >= DECAY_AFTER:
                change = float(row[4]) * (DECAY - 1)
                move.apply(new, change)
                row[4] += change
        estimated = None
        subspace = getattr(self.session, "measure", "full") == "compressed"
        if subspace:
            self._decayed = clone(new)
            measured = [
                m
                for m in directions
                if (pair := pair_of(m.target, m.expert)) is not None and pair[1] > 0
            ]
            clock = time.perf_counter()
            ok = self._measure_subspace(state, record, [*measured, *extra])
            measure_s += time.perf_counter() - clock
            if not ok:
                return None
            estimated = self._subspace
        for move in directions:
            pair = pair_of(move.target, move.expert)
            if pair is None or pair[1] <= 0:
                continue
            if estimated is not None:
                slope = estimated[move.id]
            else:
                clock = time.perf_counter()
                plus, minus = clone(state), clone(state)
                move.apply(plus, IMPORT_EPS)
                move.apply(minus, -IMPORT_EPS)
                slope = (
                    record.measure(ImportTrial(self.session, state, plus))
                    - record.measure(ImportTrial(self.session, state, minus))
                ) / (2 * IMPORT_EPS)
                measure_s += time.perf_counter() - clock
            if not torch.isfinite(torch.tensor(slope)):
                return None
            core = state[move.target].get(f"core_{move.expert}")
            if core is None:
                effect = pair[1]
            else:
                effect_state = {move.target: {k: v.clone() for k, v in state[move.target].items()}}
                move.apply(effect_state, IMPORT_EPS)
                ex = self.session._source.experts[move.target]
                ga, gb, _ = _grams(self.session, move.target, ex, move.expert, core.shape[0])

                def effective(st, move=move):
                    return (
                        st[move.target][f"strength_{move.expert}"].double()
                        * st[move.target][f"core_{move.expert}"].double()
                    )

                d = (effective(effect_state) - effective(state)).to(ga.device) / IMPORT_EPS
                effect = _gram_norm(d, gb, ga)
            if effect <= 0:
                continue
            initial = (
                IMPORT_WRITE_FRAC
                * pair[1]
                / effect
                * (self.session.speed / self.session.spec.speed)
            )
            row = rows.get(
                move.id, torch.tensor([0.0, initial, initial, 0.0, 0.0], dtype=torch.float64)
            )
            self._refused_moves[move.id] = float(row[2])
            step, sign, delta = sign_step(float(slope), int(row[0]), float(row[1]), float(row[2]))
            if subspace:
                step = cap_step(step)
            move.apply(new, step)
            rows[move.id] = torch.tensor(
                [sign, delta, float(row[2]), event, float(row[4]) + step], dtype=torch.float64
            )
        if not self._refused_moves:
            return None
        trust_start = time.perf_counter()
        closed, bisected = self._trust_radius(state, new, rows, all_moves, pair_of)
        done = time.perf_counter()
        self._rprop_time = {
            "measure_s": measure_s,
            "step_s": max(trust_start - started - measure_s, 0.0),
            "trust_s": done - trust_start,
            "closed": closed,
            "bisected": bisected,
        }
        self._dim_steps = {
            key: float(rows[key][4]) - (float(store.rules[key][4]) if key in store.rules else 0.0)
            for key in self._refused_moves
        }
        store.rules, store.event, store.max_dims = rows, event, max_dims
        self._note_moves = {
            key: (all_moves[key], self._dim_steps[key]) for key in self._refused_moves
        }
        self._full_moves = " Moves: " + ", ".join(self._refused_moves) + "."
        return new

    @torch.no_grad()
    def _record_hierarchy(self, record):
        from .recording import MAP_DECAY, BlockMaps, record_hierarchy

        store = self.session.editor_state
        stats = getattr(record, "stats", None)
        if store is None or stats is None:
            return
        if getattr(store, "block_maps", None) is None:
            store.block_maps = BlockMaps()
        store.block_maps.decay = float(getattr(self.session, "hier_decay", MAP_DECAY))
        record_hierarchy(store.block_maps, stats, getattr(record, "extra", None))

    def next_state(self, state, record):
        self.note = ""
        self._note_moves, self._full_moves = {}, ""
        self._before = state
        self._refused_note, self._accepted = "", None
        self._record_hierarchy(record)
        self.provisional = False
        from .prediction_write import session_learn

        if session_learn(self.session) == "prediction":
            if self.session.edits == "off":
                return None
            return self._prediction_write(state, record)
        if record.measure is None or self.session.edits == "off":
            return None
        directions = self.session.moves()
        self._estimated, self._dim_steps = None, {}
        self._subspace, self._subspace_samples, self._decayed = None, None, None
        self._rprop_time = None
        found, active = None, []
        compressed = getattr(self.session, "measure", "full") == "compressed"
        rules_before = {}
        if compressed:
            found = self.pack_moves()
            active = self._pack_gate(state, found[0], directions)
            rules_before = {k: v.clone() for k, v in self.session.editor_state.rules.items()}
        if self.session.controller == "rprop":
            new = self._rprop(state, record, directions, active)
        else:
            new = self._r1(state, record, directions, active)
        if new is None:
            self._decayed = None
        new = self._with_pack_moves(state, new, record, directions, found, active)
        if compressed and new is not None:
            new = self._accept(state, record, new, rules_before)
            if new is None:
                return None
        self.note = self._moves_note() + self.note + self._accept_line() + self._timing_line()
        self.note += self._prediction_line(record)
        return new

    def _prediction_moves(self, state):
        found = list(self.session.moves()) + term_moves(state)
        if getattr(self.session, "editor_packs", None):
            found += self.pack_moves()[0]
        return list({move.id: move for move in found}.values())

    def frame_axes(self, state=None):
        sources = getattr(self.session, "sources", None) or []
        axes = {}
        for target, frames in import_source_frames(self.session).items():
            for node, (expert, *_rest) in enumerate(frames):
                source = sources[expert] if 0 <= expert < len(sources) else {}
                axes[(target, node)] = str(source.get("axis") or "creative")
        return axes

    @torch.no_grad()
    def prediction_setup(self, state):
        from . import prediction_write as pw

        sources = import_source_frames(self.session)
        if not sources:
            return None
        experts = self.session._source.experts
        directions = [
            m
            for m in self._prediction_moves(state)
            if m.target in sources
            and (isinstance(m, TermMove) or m.expert not in experts[m.target].exact)
        ]
        by_target = {}
        for move in directions:
            by_target.setdefault(move.target, []).append(move)
        scale = pw.session_prior_scale(self.session)
        responses, p_out, priors = {}, {}, {}
        for target, moves_ in by_target.items():
            ex = self.session._source.experts[target]
            parts = state[target]
            columns = []
            for move in moves_:
                if isinstance(move, TermMove):
                    columns.append(("term", move.output.double(), move.input.double(), None))
                    continue
                rank = parts.get(f"core_{move.expert}")
                rank = rank.shape[0] if rank is not None else ex.down.shape[2]
                a = ex.down[move.expert, 0, :rank].detach().double().cpu()
                b = ex.up[move.expert, 0, :rank].detach().double().T.cpu()
                core = parts.get(f"core_{move.expert}")
                core = torch.eye(rank, dtype=torch.float64) if core is None else core.double()
                s = parts[f"strength_{move.expert}"].double().reshape(())
                mix = s * move.direction.double() if move.part == "core" else core
                columns.append(("lora", b, a, mix))

            def response(x, columns=columns):
                out = []
                for kind, left, right, mix in columns:
                    if kind == "term":
                        out.append((x @ right).unsqueeze(-1) * left)
                    else:
                        out.append(((x @ right.t()) @ mix.t()) @ left.t())
                return torch.stack(out, -1)

            for node, (_, p_in, p_o, _) in enumerate(sources[target]):
                pinv = torch.linalg.pinv(p_in.double())

                def frame_response(z, pinv=pinv, response=response):
                    return response(z.double() @ pinv.t())

                responses[(target, node)] = (target, frame_response)
                p_out[(target, node)] = p_o
            r = len(columns)
            zero = torch.zeros(r, dtype=torch.float64)
            priors[target] = pw.NodePrior(
                zero, zero.clone(), torch.eye(r, dtype=torch.float64) / scale
            )
        return by_target, responses, p_out, priors

    @torch.no_grad()
    def _prediction_write(self, state, record):
        from . import prediction_write as pw
        from .recording import block_index

        stats = getattr(record, "stats", None)
        if stats is None or self.session.speed <= 0:
            return None
        setup = self.prediction_setup(state)
        if setup is None:
            return None
        by_target, responses, p_out, priors = setup
        hier = getattr(stats, "hier_errors", None) or {}
        clamp = getattr(stats, "clamp_errors", None) or {}
        names = list(hier) + list(clamp)
        attr = names[0].rpartition(".")[0] if names else "blocks"
        frames = self.session.__dict__.get("_block_frames")

        def block_frame(target, width):
            index = block_index(target, attr)
            if frames is None or index is None:
                return None
            return frames.frame(attr, index, width)

        terms = pw.build_terms(
            getattr(stats, "cross_errors", None) or {},
            responses,
            p_out,
            hier=hier,
            clamp=clamp,
            block_frame=block_frame,
            block_of=lambda target: block_index(target, attr),
        )
        result = pw.solve_map(terms, priors, pw.session_weights(self.session))
        applied = self.apply_prediction(state, by_target, result)
        if applied is None:
            return None
        new, written, norm = applied
        self._note_moves = written
        self.note = self._moves_note() + result.note(norm) + self._prediction_line(record)
        self.provisional = True
        return new

    @torch.no_grad()
    def apply_prediction(self, state, by_target, result):
        new = {t: dict(parts) for t, parts in state.items()}
        written = {}
        for target, delta in result.delta.items():
            for move, step in zip(by_target[target], delta.tolist(), strict=True):
                if step and math.isfinite(step):
                    move.apply(new, float(step))
                    written[move.id] = (move, float(step))
        if not written:
            return None
        lora_moves = {k: m for k, (m, _) in written.items() if not isinstance(m, TermMove)}
        if lora_moves:
            self._trust_radius(
                state, new, {}, lora_moves, lambda t, e: self._pair_norm(state, t, e)
            )
        trial = ImportTrial(self.session, state, new)
        norm = (self._trial_norm(trial) ** 2 + trial.terms_norm**2) ** 0.5
        return new, written, norm

    @staticmethod
    def _prediction_line(record):
        from .recording import prediction_debug, prediction_note, prediction_report

        stats = getattr(record, "stats", None)
        cross = getattr(stats, "cross_errors", None) or {}
        hier = getattr(stats, "hier_errors", None) or {}
        clamp = getattr(stats, "clamp_errors", None) or {}
        seconds = (getattr(record, "extra", None) or {}).get("target_forward_s")
        line = prediction_note({}, cross, hier, clamp, seconds)
        if prediction_debug():
            line += prediction_report({}, cross, hier, clamp)
        return line

    @torch.no_grad()
    def _change_norm(self, old, new):
        trial = ImportTrial(self.session, old, new)
        return (self._trial_norm(trial) ** 2 + trial.terms_norm**2) ** 0.5

    @torch.no_grad()
    def _accept(self, state, record, new, rules_before):
        before = float(record.measure(ImportTrial(self.session, state, state)))
        after = float(record.measure(ImportTrial(self.session, state, new)))
        halved = False
        if not after < before:
            new = between(state, new, 0.5)
            halved = True
            after = float(record.measure(ImportTrial(self.session, state, new)))
            if not after < before:
                self._refused_note = (
                    f" {NOT_WRITTEN} (loss {before:.6g} before, {after:.6g} after the halved step)."
                )
                return None
            self._halve_steps(rules_before)
        decayed = state if self._decayed is None else self._decayed
        if halved:
            decayed = between(state, decayed, 0.5)
        self._accepted = {
            "before": before,
            "after": after,
            "halved": halved,
            "decay": self._change_norm(state, decayed),
            "moves": self._change_norm(decayed, new),
        }
        self.session.editor_state.add_gain(before - after)
        return new

    def _halve_steps(self, rules_before):
        store = self.session.editor_state
        for key, row in list(store.rules.items()):
            old = float(rules_before[key][4]) if key in rules_before else 0.0
            if float(row[4]) == old:
                continue
            row = row.clone()
            row[4] = old + 0.5 * (float(row[4]) - old)
            store.rules[key] = row
        self._dim_steps = {key: 0.5 * step for key, step in self._dim_steps.items()}
        self._note_moves = {
            key: (move, 0.5 * step) for key, (move, step) in self._note_moves.items()
        }

    def _accept_line(self):
        a = self._accepted
        if not a:
            return ""
        halved = ", after halving the step" if a["halved"] else ", step not halved"
        return (
            f" Accept check: measured loss {a['before']:.6g} before, {a['after']:.6g}"
            f" after{halved}. Change: decay {a['decay']:.4g}, moves {a['moves']:.4g}."
        )

    def _r1(self, state, record, directions, extra=()):
        if not directions:
            return None

        def clone():
            return {t: {k: v.clone() for k, v in parts.items()} for t, parts in state.items()}

        from .compute import observing_set_share_layers

        shares = []
        set_key = self.session._source.extra.get("live_id", self.session._source.name)

        def observe(_target, gates, sets):
            for gate, layer in zip(gates, sets or (), strict=False):
                if layer.key == set_key:
                    shares.append(gate.float().reshape(-1, gate.shape[-1]).mean(0).cpu())

        def wants(_target, sets):
            return any(layer.key == set_key for layer in sets)

        def recording_set_shares(fn):
            return observing_set_share_layers(wants, fn)

        slopes = []
        history = self.session.editor_state.history
        subspace = getattr(self.session, "measure", "full") == "compressed"
        if subspace:
            if not self._measure_subspace(state, record, [*directions, *extra]):
                return None
            slopes = [-float(self._subspace[move.id]) for move in directions]
        for move in () if subspace else directions:
            plus, minus = clone(), clone()
            move.apply(plus, IMPORT_EPS)
            move.apply(minus, -IMPORT_EPS)
            with recording_set_shares(observe):
                slope = (
                    record.measure(ImportTrial(self.session, state, plus))
                    - record.measure(ImportTrial(self.session, state, minus))
                ) / (2 * IMPORT_EPS)
            if not torch.isfinite(torch.tensor(slope)):
                return None
            previous = history.get(move.id, torch.full((IMPORT_HISTORY,), float("nan")))
            row = torch.cat((previous[1:], torch.tensor([slope])))
            history[move.id] = row
            slopes.append(-float(torch.nanmean(row)))
        if shares:
            average_share = torch.stack(shares).mean(0)
            slopes = [
                slope * (float(average_share[move.expert]) if move.expert >= 0 else 1.0)
                for move, slope in zip(directions, slopes, strict=True)
            ]
        weights = torch.tensor(slopes)
        norm = float(weights.norm())
        if not norm:
            return None
        host_norms = getattr(self.session, "host_norms", {}) or {}
        scale = (
            sum(float(host_norms[t]) ** 2 for t in state) ** 0.5
            if all(t in host_norms for t in state)
            else 0.0
        )
        if not scale:
            neutral = self.session._reset_state()
            blank = {t: {k: v.clone() for k, v in parts.items()} for t, parts in neutral.items()}
            for parts in blank.values():
                for key, value in parts.items():
                    if key.startswith("strength_"):
                        value.zero_()
            scale = self._trial_norm(ImportTrial(self.session, blank, neutral))
        if not scale:
            return None
        raw = clone()
        for move, weight in zip(directions, weights, strict=True):
            move.apply(raw, float(weight) * IMPORT_EPS)
        trial = ImportTrial(self.session, state, raw)
        effect = self._trial_norm(trial) / IMPORT_EPS
        if not effect:
            return None
        speed = max(float(self.session.speed), 0.0) / self.session.spec.speed
        if not speed:
            return None
        step_scale = IMPORT_WRITE_FRAC * scale * speed / effect
        new = clone()
        neutral = self.session._neutral(state)
        for target in new:
            for part in new[target]:
                if part in TERM_FIXED:
                    continue
                new[target][part] = neutral[target][part] + (
                    state[target][part] - neutral[target][part]
                ) * (1 - IMPORT_DECAY)
        if subspace:
            self._decayed = {t: {k: v.clone() for k, v in p.items()} for t, p in new.items()}
        steps = [float(weight) * step_scale for weight in weights]
        if subspace:
            steps = [cap_step(step) for step in steps]
        for move, step in zip(directions, steps, strict=True):
            move.apply(new, step)
        total_trial = ImportTrial(self.session, neutral, new)
        total = self._trial_norm(total_trial)
        ratio = 1.0
        if total > IMPORT_CAP_FRAC * scale:
            ratio = IMPORT_CAP_FRAC * scale / total
            for target in new:
                for part in new[target]:
                    if part in TERM_PARTS:
                        continue
                    new[target][part] = neutral[target][part] + ratio * (
                        new[target][part] - neutral[target][part]
                    )
        if subspace and ratio != 1.0:
            for target, parts in self._decayed.items():
                for part in parts:
                    if part not in TERM_PARTS:
                        parts[part] = neutral[target][part] + ratio * (
                            parts[part] - neutral[target][part]
                        )
        self._dim_steps = {m.id: step * ratio for m, step in zip(directions, steps, strict=True)}
        self._note_moves = {
            m.id: (m, step * ratio) for m, step in zip(directions, steps, strict=True)
        }
        self._full_moves = (
            " Moves: "
            + ", ".join(
                f"{m.id}={step * ratio:.4g}" for m, step in zip(directions, steps, strict=True)
            )
            + "."
        )
        return new

    def _read_pack(self, pack):
        if isinstance(pack, tuple) and len(pack) == 2:
            return pack
        path = pack if isinstance(pack, (str, Path)) else getattr(pack, "path", None)
        if path is None:
            return None
        cache = self.__dict__.setdefault("_packs", {})
        if str(path) not in cache:
            from .editor_packs import read_pack

            try:
                cache[str(path)] = read_pack(path)
            except (OSError, ValueError) as exc:
                cache[str(path)] = None
                self.__dict__.setdefault("_pack_problems", []).append(str(exc))
        return cache[str(path)]

    def pack_moves(self):
        result, skipped = [], []
        self.uncovered = 0
        session = self.session
        for pack in getattr(session, "editor_packs", None) or []:
            loaded = self._read_pack(pack)
            if loaded is None:
                continue
            manifest, tensors = loaded
            chunk = (getattr(manifest, "raw", None) or {}).get("chunk") or {}
            names, shapes = chunk.get("hooks"), chunk.get("shapes")
            if not names or not shapes or "q" not in tensors:
                continue
            prefix = "input_basis."
            basis = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
            for target in names:
                ex = session._source.experts.get(target)
                if ex is None:
                    self.uncovered += 1
                    continue
                if target not in basis:
                    skipped.append(f"{manifest.id}:{target}")
                    continue
                try:
                    outputs, inputs = pack_directions(
                        names, shapes, tensors["q"], basis, target, manifest.id
                    )
                except ValueError:
                    skipped.append(f"{manifest.id}:{target}")
                    continue
                count = 0
                for output_id, output in outputs.items():
                    if count >= PACK_TERM_MOVES:
                        break
                    input_id = f"{manifest.id}:in:{output_id.rsplit(':', 1)[1]}"
                    input_ = inputs.get(input_id)
                    if (
                        input_ is None
                        or output.numel() != ex.up.shape[-1]
                        or input_.numel() != ex.down.shape[-1]
                        or not bool(output.count_nonzero())
                        or not bool(input_.count_nonzero())
                    ):
                        continue
                    result.append(
                        TermMove(
                            f"{target}:term:{output_id}:{input_id}",
                            target,
                            output_id,
                            input_id,
                            output.float(),
                            input_.float(),
                        )
                    )
                    count += 1
        return result, skipped

    def _at_trust_radius(self, state, directions):
        pairs = {
            (m.target, m.expert)
            for m in directions
            if not isinstance(m, TermMove) and getattr(m, "expert", -1) >= 0
        }
        if not pairs:
            return False
        for target, expert in pairs:
            pair = self._pair_norm(state, target, expert)
            if pair is None or pair[1] <= 0:
                return False
            if pair[0] < TRUST_RATIO * pair[1] * (1 - 1e-6):
                return False
        return True

    def _pack_gate(self, state, candidates, directions):

        host = getattr(self.session, "host_norms", None) or {}
        candidates = [m for m in candidates if float(host.get(m.target, 0.0)) > 0]
        if not candidates or self.session.speed <= 0:
            return []
        store = self.session.editor_state
        if pack_moves_start(store.gains.tolist()) or (
            store.gains.shape[0] >= PACK_GAIN_EVENTS and self._at_trust_radius(state, directions)
        ):
            return candidates
        return []

    def _estimated_gain(self, directions):
        if self._estimated is None:
            return None
        return -sum(
            float(self._estimated[m.id]) * float(self._dim_steps.get(m.id, 0.0))
            for m in directions
            if m.id in self._estimated
        )

    @torch.no_grad()
    def _with_pack_moves(self, state, new, record, directions=(), found=None, active=()):

        compressed = found is not None
        candidates, skipped = found if compressed else self.pack_moves()
        uncovered = getattr(self, "uncovered", 0)
        if not candidates and not skipped and not uncovered:
            return new
        measured = written = 0
        if candidates and self.session.speed > 0:
            base = state if new is None else new
            store = self.session.editor_state
            if compressed:
                if active and self._subspace is not None:
                    slopes = [self._subspace[m.id] for m in active]
                    stepped = self._pack_step(state, base, record, active, slopes, subspace=True)
                    if stepped is not None:
                        new, measured, written = stepped
            else:
                before = record.measure(ImportTrial(self.session, state, state))
                gain = before - record.measure(ImportTrial(self.session, state, base))
                store.add_gain(float(gain))
                if pack_moves_start(store.gains.tolist()) or (
                    store.gains.shape[0] >= PACK_GAIN_EVENTS
                    and self._at_trust_radius(base, directions)
                ):
                    stepped = self._pack_step(state, base, record, candidates)
                    if stepped is not None:
                        new, measured, written = stepped
        self.note += f" Pack moves: {measured} measured, {written} written."
        if skipped:
            self.note += " Pack hooks without input_basis: " + ", ".join(skipped) + "."
        if uncovered:
            self.note += f" Pack hooks skipped without an import: {uncovered}."
        return new

    @torch.no_grad()
    def _pack_step(self, state, base, record, candidates, slopes=None, subspace=False):

        host = getattr(self.session, "host_norms", None) or {}
        if slopes is not None:
            slopes = [
                float(slope)
                for m, slope in zip(candidates, slopes, strict=True)
                if float(host.get(m.target, 0.0)) > 0
            ]
        candidates = [m for m in candidates if float(host.get(m.target, 0.0)) > 0]
        if not candidates:
            return None

        def clone(src):
            return {t: {k: v.clone() for k, v in p.items()} for t, p in src.items()}

        if slopes is None:
            slopes = []
            for move in candidates:
                plus, minus = clone(base), clone(base)
                move.apply(plus, IMPORT_EPS)
                move.apply(minus, -IMPORT_EPS)
                slope = (
                    record.measure(ImportTrial(self.session, state, plus))
                    - record.measure(ImportTrial(self.session, state, minus))
                ) / (2 * IMPORT_EPS)
                if not torch.isfinite(torch.tensor(slope)):
                    return None
                slopes.append(float(slope))
        elif not all(torch.isfinite(torch.tensor(slopes, dtype=torch.float64)).tolist()):
            return None
        speed = max(float(self.session.speed), 0.0) / self.session.spec.speed
        if not speed:
            return None
        store = self.session.editor_state
        new = clone(base)
        steps = []
        if self.session.controller == "rprop":
            refused = self.__dict__.setdefault("_refused_moves", {})
            for move, slope in zip(candidates, slopes, strict=True):
                effect = move.effect
                if effect <= 0:
                    steps.append(0.0)
                    continue
                initial = IMPORT_WRITE_FRAC * float(host[move.target]) / effect * speed
                row = store.rules.get(
                    move.id, torch.tensor([0.0, initial, initial, 0.0, 0.0], dtype=torch.float64)
                )
                refused[move.id] = float(row[2])
                step, sign, delta = sign_step(slope, int(row[0]), float(row[1]), float(row[2]))
                if subspace:
                    step = cap_step(step)
                move.apply(new, step)
                store.rules[move.id] = torch.tensor(
                    [sign, delta, float(row[2]), store.event, float(row[4]) + step],
                    dtype=torch.float64,
                )
                steps.append(step)
        else:
            history = store.history
            weights = []
            for move, slope in zip(candidates, slopes, strict=True):
                if subspace:
                    weights.append(-float(slope))
                    continue
                previous = history.get(move.id, torch.full((IMPORT_HISTORY,), float("nan")))
                row = torch.cat((previous[1:], torch.tensor([slope])))
                history[move.id] = row
                weights.append(-float(torch.nanmean(row)))
            raw = clone(base)
            for move, weight in zip(candidates, weights, strict=True):
                move.apply(raw, weight * IMPORT_EPS)
            effect = ImportTrial(self.session, base, raw).terms_norm / IMPORT_EPS
            if not effect:
                return None
            targets = {m.target for m in candidates}
            reference = sum(float(host[t]) ** 2 for t in targets) ** 0.5
            scale = IMPORT_WRITE_FRAC * reference * speed / effect
            for move, weight in zip(candidates, weights, strict=True):
                step = cap_step(weight * scale) if subspace else weight * scale
                move.apply(new, step)
                steps.append(step)
        for target in {m.target for m in candidates}:
            pair = term_pair(new[target])
            if pair is None:
                continue
            limit = TRUST_RATIO * IMPORT_WRITE_FRAC * float(host[target])
            size = change_norm(*pair)
            if size > limit:
                new[target]["terms_c"] = new[target]["terms_c"] * (limit / size)
        written = sum(1 for step in steps if step)
        for move, step in zip(candidates, steps, strict=True):
            self._note_moves[move.id] = (move, float(step))
        return new, len(candidates), written


def _note_entry(move, step):
    if move.part == "core":
        kind = "core " + ":".join(move.id.split(":")[-2:])
    else:
        kind = NOTE_PARTS.get(move.part, move.part)
    lora = f" LoRA {move.expert}" if move.expert >= 0 else ""
    return f"{move.target}{lora} {kind} = {step:.4g}"
