# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kilora.compute import (
    DerivedKeyVersions,
    LayerSet,
    derived_keys,
    hard_gates,
    recording_read_keep_raw,
    route_shares,
    router_logits,
    tensor_gate,
    token_gate,
)
from kilora.editors import Editor, EditorFileInfo, EditorPlugin, EditorSupport
from kilora.errors import KiLoRAError, ModFormatError
from kilora.i18n import text
from kilora.path_graph import PathGraphCache
from kilora.path_rotation import rotate_pair

from .editor import BranchStore, ChunkEditor, ChunkState, ChunkWriter, EditBranch, _load_residual

log = logging.getLogger("kilora.editor.chunk")

KIND = "chunk"
FORMAT = "kilora-editor"
FILE_VERSION = "1.0"
MANIFEST_KEY = "kilora.manifest"
DEFAULT_V0_SEED = 0
DEFAULT_TOTAL_CAP_WRITES = 10.0
EDITABLE_PART = "up"
DEFAULT_WRITE_FRAC = 1e-3
DEFAULT_CAP_FRAC = 2e-2


def save_chunk_editor(
    path,
    packs,
    base_family: str,
    write_norm: float,
    eps: float = 0.1,
    per_event: int = 16,
    decay: float = 0.02,
    chunk_cap: float | None = None,
    total_cap: float | None = None,
    total_cap_writes: float | None = DEFAULT_TOTAL_CAP_WRITES,
    write_frac: float | None = None,
    cap_frac: float | None = None,
    per_hook_cap: bool = False,
    diff: str = "central",
    measure: str = "compressed",
    combos: int | None = None,
    version: str = "1.0.0",
    license: str = "Apache-2.0",
    license_notice: str | None = None,
) -> dict:

    if total_cap is None and total_cap_writes is not None:
        total_cap = float(write_norm) * float(total_cap_writes)
    folder = Path(path).resolve().parent
    names = []
    for p in packs:
        p = Path(p).resolve()
        if p.parent != folder:
            raise ValueError(f"{p.name}: the packs lie in the folder of the editor file")
        names.append(p.name)
    ed = ChunkEditor.load_packs([folder / n for n in names])
    manifest = {
        "format": FORMAT,
        "format_version": FILE_VERSION,
        "kind": KIND,
        "id": "chunk",
        "version": version,
        "license": license,
        **({"license_notice": license_notice} if license_notice else {}),
        "base_models": [{"family": base_family}],
        "training": {"state": "trained"},
        "supports": EditorSupport(hooks=list(ed.names)).to_manifest(),
        "packs": names,
        "writer": {
            "write_norm": float(write_norm),
            "eps": float(eps),
            "per_event": int(per_event),
            "decay": float(decay),
            "chunk_cap": chunk_cap,
            "total_cap": total_cap,
            "diff": diff,
            "measure": measure,
            "combos": combos,
            **({"write_frac": float(write_frac)} if write_frac is not None else {}),
            **({"cap_frac": float(cap_frac)} if cap_frac is not None else {}),
            **({"per_hook_cap": True} if per_hook_cap else {}),
        },
    }
    tensors = {"kilora/chunk/write_norm": torch.tensor([float(write_norm)])}
    save_file(tensors, os.fspath(path), metadata={MANIFEST_KEY: json.dumps(manifest)})
    return manifest


def read_manifest(path) -> dict[str, Any]:
    label = os.path.basename(os.fspath(path))
    try:
        with safe_open(os.fspath(path), framework="pt") as f:
            m = json.loads((f.metadata() or {})[MANIFEST_KEY])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ModFormatError(f"{label}: cannot read the editor ({exc})") from None
    if m.get("format") != FORMAT or m.get("kind") != KIND:
        raise ModFormatError(f"{label}: not a chunk editor")
    if m.get("format_version") != FILE_VERSION:
        raise ModFormatError(f"{label}: unknown format version {m.get('format_version')}")
    if (m.get("training") or {}).get("state") != "trained":
        raise KiLoRAError(f"{label}: the editor is not trained")
    return m


class RoutedPerturbation:
    def __init__(
        self,
        base,
        routes,
        routing="tensor",
        tau=1.0,
        observed_routing=None,
        observed_select="sparsemax",
        trial_up=None,
        set_key=None,
    ):
        self.trial_up, self.set_key = trial_up, set_key
        self.base, self.routes, self.routing, self.tau = base, routes, routing, tau
        self.observed_routing, self.observed_select = observed_routing, observed_select
        self.proj, self.du = base.proj, base.du
        self.counts: dict[str, torch.Tensor] = {}
        self.observed_counts: dict[str, torch.Tensor] = {}

    def attention_sets(self, target, sets):
        if self.trial_up is None or target not in self.trial_up:
            return sets
        delta = self.trial_up[target]
        return [
            dataclasses.replace(s, up=s.up + delta.to(s.up))
            if s.key == self.set_key and s.up is not None
            else s
            for s in sets
        ]

    def delta(self, target, h):
        r = self.routes.get(target)
        if r is not None:
            router, keys = r
            q = torch.nn.functional.linear(h.float(), router.to(h.device))
            lg = router_logits(q, keys.to(h.device), self.tau)
            n = route_shares(lg, self.routing).cpu()
            self.counts[target] = self.counts.get(target, 0) + n.double()
            if self.observed_routing is not None:
                measured = route_shares(lg, self.observed_routing, self.observed_select).cpu()
                self.observed_counts[target] = (
                    self.observed_counts.get(target, 0) + measured.double()
                )
        return self.base.delta(target, h)


class RoutedWrite:
    def __init__(self, down, dup, col_expert, routes, routing="tensor", tau=1.0, set_key=None):
        self.set_key = set_key
        self.down, self.du, self.col_expert = down, dup, col_expert
        self.routes, self.routing, self.tau = routes, routing, tau

    def attention_sets(self, target, sets):
        if target not in self.du:
            return sets
        return [
            dataclasses.replace(s, up=s.up + self.du[target].t().to(s.up))
            if s.key == self.set_key and s.up is not None
            else s
            for s in sets
        ]

    def delta(self, target, h):
        if target not in self.du:
            return None
        c = h @ self.down[target].to(h.device, h.dtype).t()
        r = self.routes.get(target)
        if r is not None:
            router, keys = r
            lg = router_logits(
                torch.nn.functional.linear(h.float(), router.to(h.device)),
                keys.to(h.device),
                self.tau,
            )
            with hard_gates():
                w = tensor_gate(lg) if self.routing == "tensor" else token_gate(lg)
            c = c * w[..., self.col_expert[target].to(h.device)].to(c.dtype)
        return c @ self.du[target].to(h.device, h.dtype).t()


class ChunkEditorState:
    def __init__(self, editor: ChunkEditor, base: dict[str, torch.Tensor], experts: int):
        self.editor, self.base0 = editor, {t: v.clone() for t, v in base.items()}
        self.experts = experts
        from kilora.recording import BlockMaps

        self.block_maps = BlockMaps()
        self._initialize()

    def _initialize(self) -> None:
        from kilora.regions import RegionBook

        self.forest = BranchStore(ChunkState.new(self.editor, experts=self.experts))
        self.merge_v0 = {}
        self._branch_regions = {}
        self.base = {t: v.clone() for t, v in self.base0.items()}
        self.regions = RegionBook()
        self.links = {}
        self.prediction_errors = {}
        self.cross_errors = {}
        self.hier_errors = {}
        self.clamp_errors = {}
        self.target_forward_s = None

    def reset(self) -> None:
        from kilora.regions import RegionBook

        self.chunks.reset()
        self.links = {}
        self.regions = RegionBook()
        self.forest.branches[self.forest.current].paths = []

    def restore_branch(self, tensors) -> None:
        old = object.__new__(ChunkEditorState)
        old.editor, old.base0, old.experts = self.editor, self.base0, self.experts
        old.load_tensors(tensors)
        branch = self.forest.branches[self.forest.current]
        source = old.forest.branches.get(branch.id)
        if source is None:
            source = old.forest.branches.get(branch.parent, old.forest.branches[old.forest.current])
        if old.merge_v0:
            self.merge_v0 = old.merge_v0
        elif not any(b.merge_pairs for b in self.forest.branches.values() if b.id != branch.id):
            self.merge_v0 = {}
        branch.state = source.state
        branch.paths = list(source.paths) if source.id == branch.id else []
        branch.merge_pairs = list(source.merge_pairs) if source.id == branch.id else []
        self.regions = old._branch_regions.get(source.id, old.regions)
        self.links = old.links

    @property
    def chunks(self):
        return self.forest.state

    def fork(self, parent=None, **draw):
        import copy

        parent = self.forest.current if parent is None else parent
        branch = self.forest.fork(parent, **draw)
        source = self.regions if parent == self.forest.current else self._branch_regions[parent]
        self._branch_regions[branch] = copy.deepcopy(source)
        return branch

    def checkout(self, branch_id):
        self._branch_regions[self.forest.current] = self.regions
        self.forest.checkout(branch_id)
        self.regions = self._branch_regions[branch_id]

    def promote(self, branch_id):
        self.checkout(branch_id)
        self.forest.promote(branch_id)

    def delete(self, branch_id):
        self.forest.delete(branch_id)
        self._branch_regions.pop(branch_id, None)

    def start_from(self, state) -> None:
        self.base0 = {t: p[EDITABLE_PART].float().clone() for t, p in state.items()}
        self._initialize()

    def state_tensors(self) -> dict[str, torch.Tensor]:
        out = {f"coef/{k}": v.clone() for k, v in self.chunks.coef.items()}
        out.update({f"branch/v0/{k}": v.clone() for k, v in self.merge_v0.items()})
        out.update({f"hist/{k}": v.clone() for k, v in self.chunks.hist.items()})
        out.update({f"base/{t}": v.clone() for t, v in self.base.items()})
        out.update({f"extra/{k}": v.clone() for k, v in self.chunks.extra.items()})
        out.update({f"prior/{k}": v.clone() for k, v in self.chunks.prior.items()})
        maps = getattr(self, "block_maps", None)
        if maps is not None:
            out.update(maps.state_tensors())
        out.update({f"node_coef/{f}/{h}": v.clone() for (f, h), v in self.chunks.node_coef.items()})
        for (h, e), (up, down) in self.chunks.residual.items():
            out[f"residual/{e}/{h}/up"] = up.clone()
            out[f"residual/{e}/{h}/down"] = down.clone()
        names = self.editor.family_names()
        out["enabled"] = torch.tensor([float(n in self.chunks.enabled) for n in names])
        out["events"] = torch.tensor([float(self.chunks.events)])
        out.update(self.regions.state_tensors())
        if self.links:
            pairs = list(self.links)
            out["links/metadata"] = torch.tensor(
                list(json.dumps(pairs).encode()), dtype=torch.uint8
            )
            for index, pair in enumerate(pairs):
                out.update(
                    {f"links/{index}/{k}": v for k, v in self.links[pair].state_tensors().items()}
                )
        if (
            len(self.forest.branches) > 1
            or self.forest.serial
            or any(b.merge_pairs for b in self.forest.branches.values())
        ):
            metadata = {
                "head": self.forest.head,
                "current": self.forest.current,
                "serial": self.forest.serial,
                "branches": self.forest.list(),
            }
            out["branch/metadata"] = torch.tensor(
                list(json.dumps(metadata).encode()), dtype=torch.uint8
            )
            for index, branch in enumerate(self.forest.branches.values()):
                prefix = f"branch/{index}/"
                st = branch.state
                for pair_index, pair in enumerate(branch.merge_pairs):
                    pair_prefix = prefix + f"merge/{pair_index}/"
                    out[pair_prefix + "write"] = torch.tensor([pair[0]], dtype=torch.int64)
                    for side, snap in zip(("before", "after"), pair[1:], strict=True):
                        out[pair_prefix + side + "/enabled"] = torch.tensor(
                            [float(n in snap["enabled"]) for n in names]
                        )
                        for group in ("coef", "node_coef"):
                            for key, value in snap[group].items():
                                key = "/".join(key) if isinstance(key, tuple) else key
                                out[pair_prefix + side + "/" + group + "/" + key] = value.clone()
                for group in ("coef", "hist", "extra", "prior"):
                    out.update(
                        {prefix + group + "/" + k: v.clone() for k, v in getattr(st, group).items()}
                    )
                out.update(
                    {prefix + f"node_coef/{f}/{h}": v.clone() for (f, h), v in st.node_coef.items()}
                )
                for (h, e), (up, down) in st.residual.items():
                    out[prefix + f"residual/{e}/{h}/up"] = up.clone()
                    out[prefix + f"residual/{e}/{h}/down"] = down.clone()
                out[prefix + "enabled"] = torch.tensor([float(n in st.enabled) for n in names])
                out[prefix + "events"] = torch.tensor([st.events], dtype=torch.int64)
                book = (
                    self.regions
                    if branch.id == self.forest.current
                    else self._branch_regions[branch.id]
                )
                out.update({prefix + k: v.clone() for k, v in book.state_tensors().items()})
        return out

    def load_forest(self, tensors, *, allow_replace=True) -> bool:
        from kilora.live import _fixed_part

        current, incoming = _fixed_part(self.state_tensors()), _fixed_part(tensors)
        if set(current) != set(incoming) or any(
            current[k].shape != incoming[k].shape for k in current
        ):
            raise KiLoRAError("the saved forest does not fit this editor")
        saved = object.__new__(ChunkEditorState)
        saved.editor, saved.base0, saved.experts = self.editor, self.base0, self.experts
        saved.load_tensors(tensors)
        if (
            allow_replace
            and len(self.forest.branches) == 1
            and self.chunks.events == 0
            and not any(torch.count_nonzero(v) for v in self.chunks.coef.values())
            and not any(torch.count_nonzero(v) for v in self.chunks.node_coef.values())
        ):
            self.load_tensors(tensors)
            return False
        if set(saved.base) != set(self.base) or any(
            not torch.equal(saved.base[k], self.base[k]) for k in self.base
        ):
            raise KiLoRAError("the saved forest has a different editor base")
        self._branch_regions[self.forest.current] = self.regions
        if saved.merge_v0:
            self.merge_v0.update(saved.merge_v0)
        ids = {}
        for old in saved.forest.branches:
            self.forest.serial += 1
            ids[old] = f"branch-{self.forest.serial}"
        for old, branch in saved.forest.branches.items():
            new = ids[old]
            self.forest.branches[new] = EditBranch(
                new,
                ids.get(branch.parent),
                branch.state,
                branch.seed,
                branch.temperature,
                branch.depth,
                list(branch.paths),
                branch.action,
                list(branch.writes),
                list(branch.merge_pairs),
            )
            self._branch_regions[new] = saved._branch_regions.get(old, saved.regions)
        return True

    def load_tensors(self, t) -> None:
        from kilora.cross_layer import RLSLink

        self.merge_v0 = {
            k[len("branch/v0/") :]: v.clone() for k, v in t.items() if k.startswith("branch/v0/")
        }
        self.links = {}
        if "links/metadata" in t:
            pairs = json.loads(bytes(t["links/metadata"].tolist()).decode())
            for index, pair in enumerate(pairs):
                prefix = f"links/{index}/"
                self.links[tuple(pair)] = RLSLink.from_tensors(
                    {k[len(prefix) :]: v for k, v in t.items() if k.startswith(prefix)}
                )
        names = self.editor.family_names()
        coef = {k[5:]: v.clone() for k, v in t.items() if k.startswith("coef/")}
        hist = {k[5:]: v.clone() for k, v in t.items() if k.startswith("hist/")}
        enabled = [n for n, on in zip(names, t["enabled"].tolist(), strict=True) if on]
        extra = {k[6:]: v.clone() for k, v in t.items() if k.startswith("extra/")}
        from kilora.recording import BlockMaps

        self.block_maps = BlockMaps.from_tensors(t)
        prior = {
            k[6:]: v.clone()
            for k, v in t.items()
            if k.startswith("prior/") and not k.startswith(BlockMaps.PREFIX)
        }
        node_coef = {
            tuple(k[len("node_coef/") :].split("/", 1)): v.clone()
            for k, v in t.items()
            if k.startswith("node_coef/")
        }
        self.forest = BranchStore(
            ChunkState(
                coef,
                hist,
                enabled,
                int(t["events"][0]),
                extra=extra,
                prior=prior,
                node_coef=node_coef,
                residual=_load_residual(t),
            )
        )
        self._branch_regions = {}
        self.base = {k[5:]: v.clone() for k, v in t.items() if k.startswith("base/")}
        from kilora.regions import RegionBook

        self.regions = RegionBook.from_tensors(t)
        if "branch/metadata" in t:
            metadata = json.loads(bytes(t["branch/metadata"].tolist()).decode())
            self.forest.branches = {}
            self.forest.head, self.forest.current = metadata["head"], metadata["current"]
            self.forest.serial = metadata["serial"]
            for index, item in enumerate(metadata["branches"]):
                prefix = f"branch/{index}/"
                part = {k[len(prefix) :]: v for k, v in t.items() if k.startswith(prefix)}

                def group(name, part=part):
                    return {
                        k[len(name) + 1 :]: v.clone()
                        for k, v in part.items()
                        if k.startswith(name + "/")
                    }

                enabled = [n for n, on in zip(names, part["enabled"].tolist(), strict=True) if on]
                state = ChunkState(
                    group("coef"),
                    group("hist"),
                    enabled,
                    int(part["events"][0]),
                    extra=group("extra"),
                    prior=group("prior"),
                    node_coef={tuple(k.split("/", 1)): v for k, v in group("node_coef").items()},
                    residual=_load_residual(part),
                )
                branch = EditBranch(
                    item["id"],
                    item["parent"],
                    state,
                    item["seed"],
                    item["temperature"],
                    item["depth"],
                    item["paths"],
                    item.get("action"),
                    item.get("writes", []),
                )
                pair_indices = sorted(
                    {int(k.split("/")[1]) for k in part if k.startswith("merge/")}
                )
                for pair_index in pair_indices:
                    pp = f"merge/{pair_index}/"
                    snapshots = []
                    for side in ("before", "after"):
                        sp = pp + side + "/"
                        coef = {
                            k[len(sp + "coef/") :]: v.clone()
                            for k, v in part.items()
                            if k.startswith(sp + "coef/")
                        }
                        node = {
                            tuple(k[len(sp + "node_coef/") :].split("/", 1)): v.clone()
                            for k, v in part.items()
                            if k.startswith(sp + "node_coef/")
                        }
                        enabled = [
                            n
                            for n, on in zip(names, part[sp + "enabled"].tolist(), strict=True)
                            if on
                        ]
                        snapshots.append({"coef": coef, "node_coef": node, "enabled": enabled})
                    branch.merge_pairs.append((int(part[pp + "write"][0]), *snapshots))
                self.forest.branches[branch.id] = branch
                self._branch_regions[branch.id] = RegionBook.from_tensors(part)
            self.regions = self._branch_regions[self.forest.current]


def missing_packs_message(editor: str, folder: Path, missing: list[str]) -> str:
    return (
        f"The editor {editor} needs these direction packs, which are not in {folder}: "
        f"{', '.join(missing)}. The packs ship as separate files with the editor: download "
        "them from the same place as the editor file and put them next to it, in the same "
        "folder. To make packs for your own live mod, see 'Build packs' in the "
        "kilora-editor-chunk README."
    )


class ChunkLive(Editor):
    def __init__(self, path: str):
        self.manifest = read_manifest(path)
        folder = Path(path).resolve().parent
        missing = [n for n in self.manifest["packs"] if not (folder / n).is_file()]
        if missing:
            raise FileNotFoundError(missing_packs_message(Path(path).name, folder, missing))
        self.editor = ChunkEditor.load_packs([folder / n for n in self.manifest["packs"]])
        self.supports = EditorSupport.from_manifest(self.manifest["supports"])
        self.session: Any = None
        self.writer: ChunkWriter | None = None
        self._gain: float | None = None
        self.scale: float = 0.0
        self.scale_reference = ""
        self.hook_scale: dict[str, float] = {}
        self.options: dict[str, Any] = {}
        self.last_note = f"Direction set coherence: {self.editor.coherence():.3f}"
        self.path_report = None
        self.path_graph = PathGraphCache()
        self.weight_version = 0
        self._jvp_cache = {}
        self._jvp_samples = {}
        self._jvp_fallbacks = set()
        self._jvp_write_pairs = {}
        self._jvp_first_fallback = None
        self.derived_key_versions = DerivedKeyVersions()
        self._stage_writes = False
        self._staged_merge = None
        self._staged_merge_off = None
        self.merge_accounting = None
        self.merge_recording_seconds = 0.0
        self.provisional = False

    def check(self, session: Any) -> None:
        from kilora.live_basis import default_v0

        family = (session.mod.manifest.get("base_model") or {}).get("family")
        families = [b.get("family") for b in self.manifest["base_models"]]
        if family not in families:
            raise KiLoRAError(f"This editor is for {', '.join(families)}, not for {family}.")
        missing = [t for t in session.state if t not in self.editor.shapes]
        if missing:
            raise KiLoRAError(f"The editor has no directions for the hook point {missing[0]}.")
        live = session.mod.manifest.get("live") or {}
        cols = {t: _columns(p["down"]) for t, p in session.state.items()}
        ranks = {self.editor.shapes[t][1] for t in session.state}
        if len(ranks) != 1:
            raise KiLoRAError("The editor's hook points have different live ranks.")
        for t, p in session.state.items():
            if p["up"].shape[-1] != self.editor.shapes[t][0]:
                raise KiLoRAError(f"{t}: the editor is for width {self.editor.shapes[t][0]}.")
        seed = int(live.get("v0_seed", DEFAULT_V0_SEED))
        self.targets = list(session.state)
        self.v0 = default_v0(self.targets, cols, seed=seed, rank=ranks.pop())
        live_name = (session.mod.manifest.get("live") or {}).get("live_set")
        live_set = next(s for s in session.mod.sets if s.name == live_name)
        self.live_set = live_set
        state_routers = (
            {
                width: torch.eye(width)
                for width in {ex.keys_in.shape[-1] for ex in live_set.experts.values()}
            }
            if live_set.form == "state"
            else {}
        )
        self.routes = {
            t: (
                state_routers[live_set.experts[t].keys_in.shape[-1]]
                if live_set.form == "state"
                else live_set.router_for(t).float(),
                live_set.experts[t].keys_in.float(),
            )
            for t in self.targets
        }
        self.experts = int(next(iter(self.routes.values()))[1].shape[0])
        self.tau = math.exp(live_set.log_tau)
        self.col_expert = {t: torch.arange(cols[t]) // (cols[t] // self.experts) for t in cols}
        if getattr(session, "editor_state_kind", None) != KIND:
            base = {t: p["up"].float() for t, p in session.state.items()}
            state = ChunkEditorState(self.editor, base, self.experts)
            note = session.attach_editor_state(state, KIND)
            if note:
                log.info(note)
        host = getattr(session, "host_norms", None) or {}
        if host and all(t in host for t in self.targets):
            self.hook_scale = {t: float(host[t]) for t in self.targets}
            self.scale_reference = "host"
        else:
            clean = getattr(session, "clean", None) or session.state
            self.hook_scale = {
                t: float(clean[t][EDITABLE_PART].float().norm()) for t in self.targets if t in clean
            }
            self.scale_reference = "live state"
        self.scale = math.sqrt(sum(v * v for v in self.hook_scale.values()))
        self._gain = None
        self.session = session
        session.editor_state.expand_live = self._expanded_state

    def _cross_frames(self, session):
        from kilora.recording import CrossFrame, orthonormal_rows

        cached = getattr(self, "_cross_cache", None)
        if cached is not None and cached[0] is session:
            return cached[1]
        frames = {}
        with torch.no_grad():
            for t in self.targets:
                if t not in session.state or t not in self.v0:
                    continue
                down = _down(session.state[t])
                up = session.state[t]["up"].float().reshape(down.shape[0], -1).t()
                v0 = self.v0[t].float()
                p_out = orthonormal_rows(up @ v0.t())
                if p_out is None:
                    continue
                frames[t] = [
                    CrossFrame((v0 @ down).double(), p_out, "output", None, up.clone(), down)
                ]
        self._cross_cache = (session, frames)
        return frames

    def _block_recorder(self, session, frames):
        from kilora.recording import BlockFrames, BlockStreamRecorder

        if not frames:
            return None
        cached = getattr(self, "_block_cache", None)
        if cached is None or cached[0] is not session:
            outputs = {t: items[0].p_out for t, items in frames.items() if items}
            cached = (session, BlockFrames(outputs))
            self._block_cache = cached
        return BlockStreamRecorder(cached[1])

    def recorder(self, session):
        session.editor_state.prediction_errors = {}
        session.editor_state.cross_errors = {}
        from kilora.recording import CrossStepRecorder, PathRecorder

        cross, blocks = None, None
        if getattr(session, "record_prediction", True):
            frames = self._cross_frames(session)
            cross = CrossStepRecorder(frames) if frames else None
            blocks = self._block_recorder(session, frames)
        return PathRecorder(
            {t: _down(session.state[t]) for t in self.targets},
            self.v0,
            pair_targets=(
                self.targets if self.options.get("cross_layer") in ("rls", "jvp") else ()
            ),
            prediction_set_key=session.mod_id,
            capture_pair_inputs=(
                set(getattr(session, "cross_layer_blocks", {}))
                - set(getattr(session, "cross_layer_inner", {}))
                if self.options.get("cross_layer") == "jvp"
                else set()
            ),
            cross=cross,
            blocks=blocks,
        )

    def _update_links(self, record):
        from kilora.cross_layer import RLSLink

        stats = getattr(record, "stats", None)
        if self.options.get("cross_layer", "off") not in ("rls", "jvp") or stats is None:
            return
        if self.options.get("cross_layer") == "jvp":
            self._jvp_samples = getattr(stats, "pair_inputs", {})
        links = self.session.editor_state.links
        for pair, observations in stats.pairs.items():
            for x, y in observations:
                if pair not in links:
                    links[pair] = RLSLink(x.shape[-1], y.shape[-1])
                links[pair].update(x, y)

    def _cross_layer_stamp(self, phase, started):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._cross_layer_times[phase] += (time.perf_counter() - started) * 1000

    def _cross_link(self, pair):
        link = self.session.editor_state.links.get(pair)
        if self.options.get("cross_layer") != "jvp":
            return link
        from kilora.cross_layer import measured_link

        if not hasattr(self, "_jvp_write_pairs"):
            self._jvp_write_pairs = {}
        cache = getattr(self, "_jvp_cache", {})
        key = (getattr(self, "weight_version", 0), pair)
        if key in cache:
            self._jvp_write_pairs[pair] = None
            return cache[key]
        blocks = getattr(self.session, "cross_layer_blocks", {})
        inner = getattr(self.session, "cross_layer_inner", {})
        samples = getattr(self, "_jvp_samples", {}).get(pair)
        if pair in inner or (pair in blocks and samples):
            timing = hasattr(self, "_cross_layer_times")
            started = time.perf_counter() if timing else 0.0
            try:
                cached = getattr(self, "_jvp_bases", {})
                bases = []
                for h in pair:
                    down = _down(self.session.state[h]).detach().float()
                    old = cached.get(h)
                    if old is None or not torch.equal(old[0], down):
                        old = (down.clone(), self.v0[h].detach().float() @ down)
                        cached[h] = old
                    bases.append(old[1])
                self._jvp_bases = cached
                if pair in inner:
                    with torch.no_grad():
                        weight = inner[pair].to(bases[0])
                        response = bases[1] @ weight @ bases[0].T
                        del weight
                    if not bool(torch.isfinite(response).all()):
                        raise ValueError("link response is not finite")
                    variance = 0.0
                else:
                    response, variance = measured_link(blocks[pair], samples, *bases)
                from types import SimpleNamespace

                link = SimpleNamespace(weights=response.T, residual_variance=variance)
                cache = {k: v for k, v in cache.items() if k[0] == key[0]}
                cache[key] = link
                self._jvp_cache = cache
                self._jvp_write_pairs[pair] = None
                if timing:
                    self._cross_layer_stamp("response", started)
                return link
            except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
                reason = str(exc)
                if timing:
                    self._cross_layer_stamp("response", started)
        else:
            reason = "no isolated replay or paired activations"
        self._jvp_write_pairs.setdefault(pair, reason)
        if getattr(self, "_jvp_first_fallback", None) is None:
            self._jvp_first_fallback = reason
        fallbacks = getattr(self, "_jvp_fallbacks", set())
        if pair not in fallbacks:
            log.info("Cross-layer pair %s uses RLS: %s", pair, reason)
            fallbacks.add(pair)
            self._jvp_fallbacks = fallbacks
        return link

    def _writer(self, times) -> ChunkWriter:
        w = self.manifest["writer"]
        diff = self.options.get("diff") or w.get("diff", "central")
        measure = self.options.get("measure") or w.get("measure", "compressed")
        combos = self.options.get("combos") or w.get("combos")
        return ChunkWriter(
            self.editor,
            times=list(times),
            weights=[1.0 / len(times)] * len(times),
            eps=[w["eps"]] * len(times),
            write_norm=self._write_norm(w) * self._write_scale(),
            per_event=w["per_event"],
            decay=w["decay"],
            chunk_cap=w["chunk_cap"],
            total_cap=self._total_cap(w),
            diff=diff,
            measure=measure,
            combos=None if combos is None else int(combos),
            norm=self.change_norm,
            effect_cap=self._effect_cap(w),
            ir_parts=getattr(getattr(self, "live_set", None), "ir_parts", {}),
        )

    def _effect_cap(self, w) -> float | None:
        cap = self._total_cap(w)
        on = self.options.get("cap_effect", w.get("cap_effect", False))
        if cap is None or not on:
            return None
        if self._gain is None:
            gen = torch.Generator().manual_seed(0)
            fams = self.editor.chunks
            writer = ChunkWriter(self.editor, [None], [1.0], [0.1], write_norm=1.0)
            eff = chg = 0.0
            for _ in range(8):
                coefs = {
                    c.family: torch.randn(self.experts, c.q.shape[0], generator=gen) for c in fams
                }
                eff += writer.effect_norm(coefs) ** 2
                chg += self.change_norm(coefs) ** 2
            self._gain = math.sqrt(eff / chg) if chg > 0 else 1.0
        return cap * self._gain

    def change_norm(self, coefs) -> float:
        rows = {c.family: c for c in self.editor.chunks}
        tot = 0.0
        du = None
        for f, v in coefs.items():
            ph = self.editor.per_hook(rows[f].q)
            part = {h: torch.einsum("ej,jor->eor", v.float(), d) for h, d in ph.items()}
            du = part if du is None else {h: du[h] + part[h] for h in du}
        for t in self.targets if du else ():
            v0 = self.v0[t].float()
            for e in range(self.experts):
                m = (self.col_expert[t] == e).float()
                tot += float(((du[t][e] @ v0) * m).pow(2).sum())
        return tot**0.5

    def write_note(self) -> str:
        w, n, e = self.writer, self.state_norm(), self.state_effect_norm()
        from kilora.recording import prediction_debug, prediction_note, prediction_report

        st = getattr(self.session, "editor_state", None)
        errors = getattr(st, "prediction_errors", {}) or {}
        cross = getattr(st, "cross_errors", {}) or {}
        hier = getattr(st, "hier_errors", {}) or {}
        clamp = getattr(st, "clamp_errors", {}) or {}
        observations = [item for items in errors.values() for item in items]
        if w is None or n is None:
            return ""
        eps = prediction_note(errors, cross, hier, clamp, getattr(st, "target_forward_s", None))
        cap = f" of cap {w.total_cap:.4g}" if w.total_cap is not None else " (no cap)"
        eff = f", effect norm {e:.4g}" if e is not None else ""
        if w.effect_cap is not None:
            eff += f" of effect cap {w.effect_cap:.4g}"
        report = self.path_report
        path = (
            f" Path merge: {report[0]} steps, merged norm {report[1]:.4g}, angle {report[2]:.4g}."
            if report is not None
            else ""
        )
        if report is not None:
            path += f" Merge recording: {self.merge_recording_seconds:.4g} s."
        if self.merge_accounting is not None:
            a, b, c = self.merge_accounting
            path += f" source=merge merge_equiv={a:.4g} merge_repeat={b:.4g} merge_new={c:.4g}."
        if observations:
            for target, items in errors.items():
                if not items:
                    continue
                mean = sum(float(item.error.abs().mean()) for item in items) / len(items)
                eps += f" {target}: mean_abs_error={mean:.4g}"
        if prediction_debug():
            eps += prediction_report(errors, cross, hier, clamp)
        pin = getattr(self.session, "pinned", None)
        pinned = f"{pin['branch']}@{pin['revision']}" if pin else "none"
        return (
            f"State: change norm {n:.4g}{cap}{eff}."
            + path
            + eps
            + f" pinned={pinned}. {self.last_note}"
        )

    def state_effect_norm(self) -> float | None:
        w, st = self.writer, getattr(self.session, "editor_state", None)
        if w is None or st is None:
            return None
        return w.effect_norm({f: st.chunks.coef[f] for f in st.chunks.enabled})

    def state_norm(self) -> float | None:
        w, st = self.writer, getattr(self.session, "editor_state", None)
        if w is None or st is None:
            return None
        if st.chunks.node_coef or st.chunks.residual:
            return self._expanded_norm(st.chunks.expanded_u(self.editor))
        return w._norm({f: st.chunks.coef[f] for f in st.chunks.enabled})

    def _expanded_norm(self, values):
        return math.sqrt(
            sum(
                float(
                    (
                        (values[h][e].float() @ self.v0[h].float())
                        * (self.col_expert[h] == e).float()
                    )
                    .square()
                    .sum()
                )
                for h in self.targets
                for e in range(self.experts)
            )
        )

    def _write_scale(self) -> float:
        session = self.session
        default = float(getattr(getattr(session, "spec", None), "speed", 0.0) or 0.0)
        speed = getattr(session, "speed", None)
        if speed is None or default <= 0:
            return 1.0
        return max(float(speed), 0.0) / default

    def _write_norm(self, w) -> float:
        if w.get("write_frac") is not None and self.scale:
            return float(w["write_frac"]) * self.scale
        return float(w["write_norm"])

    def _total_cap(self, w) -> float | None:
        if w.get("cap_frac") is not None and self.scale:
            return float(w["cap_frac"]) * self.scale
        cap = w.get("total_cap")
        return float(cap) if cap is not None else self._write_norm(w) * DEFAULT_TOTAL_CAP_WRITES

    def _times(self, record) -> list:
        import inspect as _inspect

        bins = int(self.options.get("time_bins") or 1)
        try:
            takes_time = "time" in _inspect.signature(record.measure).parameters
        except (TypeError, ValueError):
            takes_time = False
        if bins == 2 and takes_time:
            return list(self.manifest["writer"].get("times", [0.7, 0.95]))
        self.last_note = "" if bins == 1 else "time_bins 2: this host measures at one time"
        return [None]

    def _apply_packs(self, chunks) -> None:
        packs = self.options.get("packs")
        names = self.editor.family_names()
        want = names if packs is None else [f for f in names if f in set(packs)]
        if want != chunks.enabled:
            chunks.enabled = want

    def next_combined_state(self, state, records):
        st = self.session.editor_state
        chunks = st.chunks
        initial = chunks.snapshot()
        sums = {group: {} for group in ("coef", "node_coef")}
        last = None
        merge_sums = {group: {} for group in ("coef", "node_coef")}
        off_sums = {group: {} for group in ("coef", "node_coef")}
        merge_paths = []
        self._staged_merge = None
        for record in records:
            chunks.restore(initial)
            self._staged_merge = None
            self._staged_merge_off = None
            result = self.next_state(state, record)
            if result is None:
                continue
            last = chunks.snapshot()
            staged = self._staged_merge
            staged_off = self._staged_merge_off
            if staged_off is not None:
                off_before, off_after = staged_off
                for group in off_sums:
                    for key in off_before[group].keys() | off_after[group].keys():
                        exemplar = off_after[group].get(key, off_before[group].get(key))
                        start = off_before[group].get(key, torch.zeros_like(exemplar))
                        end = off_after[group].get(key, torch.zeros_like(exemplar))
                        delta = end - start
                        off_sums[group][key] = (
                            off_sums[group].get(key, torch.zeros_like(delta)) + delta
                        )
            if staged is not None:
                base_merge, merged, paths = staged
                merge_paths.extend(paths)
                for group in merge_sums:
                    for key in base_merge[group].keys() | merged[group].keys():
                        exemplar = merged[group].get(key, base_merge[group].get(key))
                        start = base_merge[group].get(key, torch.zeros_like(exemplar))
                        end = merged[group].get(key, torch.zeros_like(exemplar))
                        delta = end - start
                        merge_sums[group][key] = (
                            merge_sums[group].get(key, torch.zeros_like(delta)) + delta
                        )
            for group in sums:
                base = initial[group]
                for key, value in last[group].items():
                    delta = value - base.get(key, torch.zeros_like(value))
                    sums[group][key] = sums[group].get(key, torch.zeros_like(delta)) + delta
        if last is None:
            chunks.restore(initial)
            return None
        chunks.restore(last)
        for group, updates in sums.items():
            target = getattr(chunks, group)
            base = initial[group]
            for key, delta in updates.items():
                target[key] = base.get(key, torch.zeros_like(delta)) + delta
        chunks.events = initial["events"] + len(records)
        if (
            self._stage_writes
            and self.options.get("path_merge", True)
            and not getattr(self, "provisional", False)
        ):
            if "merge_write_directions" in initial["prior"]:
                chunks.prior["merge_write_directions"] = initial["prior"][
                    "merge_write_directions"
                ].clone()
            else:
                chunks.prior.pop("merge_write_directions", None)
            combined = chunks.snapshot()
            self._remember_write(initial, combined)
            combined = chunks.snapshot()
            if self.options.get("cross_layer", "off") in ("rls", "jvp"):
                for group, updates in off_sums.items():
                    target = getattr(chunks, group)
                    for key, delta in updates.items():
                        target[key] = target.get(key, torch.zeros_like(delta)) + delta
                self._bound_path_write(chunks, combined, self.writer)
                self._staged_merge_off = (combined, chunks.snapshot())
                chunks.restore(combined)
            for group, updates in merge_sums.items():
                target = getattr(chunks, group)
                for key, delta in updates.items():
                    target[key] = target.get(key, torch.zeros_like(delta)) + delta
            self._bound_path_write(chunks, combined, self.writer)
            merged = chunks.snapshot()
            chunks.restore(combined)
            self._staged_merge = (combined, merged, merge_paths)
        return self._expanded_state(state)

    def _record_hierarchy(self, record, stats):
        from kilora.recording import MAP_DECAY, BlockMaps, record_hierarchy

        st = self.session.editor_state
        extra = getattr(record, "extra", None) or {}
        if getattr(st, "block_maps", None) is None:
            st.block_maps = BlockMaps()
        st.block_maps.decay = float(getattr(self.session, "hier_decay", MAP_DECAY))
        record_hierarchy(st.block_maps, stats, extra)
        st.hier_errors = getattr(stats, "hier_errors", {}) if stats is not None else {}
        st.clamp_errors = getattr(stats, "clamp_errors", {}) if stats is not None else {}
        st.target_forward_s = extra.get("target_forward_s")

    def next_state(self, state, record):
        from kilora.recording import Perturbation

        if self.session is None:
            raise KiLoRAError("check() was not called")
        stats = getattr(record, "stats", None)
        self.session.editor_state.prediction_errors = (
            getattr(stats, "prediction_errors", {}) if stats is not None else {}
        )
        self.session.editor_state.cross_errors = (
            getattr(stats, "cross_errors", {}) if stats is not None else {}
        )
        self._record_hierarchy(record, stats)
        self.provisional = False
        from kilora.prediction_write import session_learn

        if session_learn(self.session) == "prediction":
            return self._prediction_write(state, record, stats)
        if record.measure is None or self._write_scale() <= 0:
            return None
        st = self.session.editor_state
        timing = self.options.get("cross_layer") == "jvp"
        if timing:
            if not hasattr(self, "_cross_layer_times"):
                self._cross_layer_times = {
                    "response": 0.0,
                    "link_update": 0.0,
                    "merge_accounting": 0.0,
                    "other": 0.0,
                }
            write_started = started = time.perf_counter()
            prior_times = sum(self._cross_layer_times.values())
        self._update_links(record)
        if timing:
            self._cross_layer_stamp("link_update", started)
        proj = {t: self.v0[t].float() @ _down(state[t]) for t in self.targets}

        counts: dict[str, torch.Tensor] = {}
        raw_keeps: dict[tuple[str, int], list[torch.Tensor]] = {}
        keeps: dict[tuple[str, int], list[torch.Tensor]] = {}
        merge_on = self.options.get("path_merge", True)
        self.path_report = None

        recording_time = 0.0

        def observe_keep(name, step, set_key, values):
            nonlocal recording_time
            started = time.perf_counter()
            hook, sep, index = name.rpartition("#")
            if not sep or hook not in self.targets or set_key != self.session.mod_id:
                return
            if values.shape[-2] == self.experts:
                raw_keeps.setdefault((hook, step - 1), []).append(values)
            recording_time += time.perf_counter() - started

        measurement_index = 0
        pass_keeps: dict[tuple[str, int], list[torch.Tensor]] = {}
        pending_keeps = None

        def loss(pert, time_, _seed):
            nonlocal measurement_index, pass_keeps, pending_keeps
            p = RoutedPerturbation(
                Perturbation(proj, {t: pert[t].float() for t in self.targets}),
                self.routes,
                getattr(self.session.mod.settings, "routing", "tensor"),
                self.tau,
                observed_select=getattr(self.session.mod.settings, "routing_select", "sparsemax"),
                trial_up={t: (pert[t].float() @ self.v0[t].float()).t() for t in self.targets},
                set_key=self.session.mod_id,
            )
            is_base = w.diff == "one-sided" and measurement_index == 0
            measurement_index += 1
            if merge_on and (is_base or w.diff == "central"):
                pass_keeps = {}

                def observe_pass(name, step, set_key, values):
                    hook, sep, _ = name.rpartition("#")
                    if (
                        sep
                        and hook in self.targets
                        and set_key == self.session.mod_id
                        and values.shape[-2] == self.experts
                    ):
                        pass_keeps.setdefault((hook, step - 1), []).append(values)

                with recording_read_keep_raw(observe_keep if is_base else observe_pass):
                    value = record.measure(p) if time_ is None else record.measure(p, time=time_)
                if w.diff == "central":
                    if pending_keeps is None:
                        pending_keeps = pass_keeps
                    else:
                        for key in pending_keeps.keys() & pass_keeps.keys():
                            plus, minus = pending_keeps[key], pass_keeps[key]
                            if len(plus) == len(minus):
                                for a, b in zip(plus, minus, strict=True):
                                    if a.shape == b.shape:
                                        raw_keeps.setdefault(key, []).append((a + b) * 0.5)
                        pending_keeps = None
            else:
                value = record.measure(p) if time_ is None else record.measure(p, time=time_)
            for t, n in p.counts.items():
                counts[t] = counts.get(t, 0) + n
            return value

        draw = (getattr(record, "extra", None) or {}).get("draw")
        if draw is None:
            extra = getattr(record, "extra", None) or {}
            if float(extra.get("entry_temperature", 0) or 0) > 0:
                draw = {
                    "seed": extra.get("seed"),
                    "temperature": extra["entry_temperature"],
                    "depth": extra.get("depth"),
                }
        if draw and float(draw.get("temperature", 0) or 0) > 0:
            branch = st.fork(
                seed=draw.get("seed"), temperature=draw["temperature"], depth=draw.get("depth")
            )
            st.checkout(branch)
            state = self._expanded_state(state)
        self._apply_packs(st.chunks)
        w = self._writer(self._times(record))
        w.explore = float(self.options.get("explore", 0))
        w.draw = bool(draw and float(draw.get("temperature", 0) or 0) > 0)
        self._explore_draw = w.draw and w.explore > 0
        self.writer = w
        if not w._rotation(st.chunks):
            return None

        key_layers = {}
        if merge_on:
            for h in self.targets:
                ex = self.live_set.experts[h]
                down = _down(state[h]).clone()
                up = state[h]["up"].float().reshape(down.shape[0], -1).clone()
                key_layers[h] = LayerSet(
                    router=self.live_set.router_for(h),
                    keys_in=ex.keys_in,
                    down=down,
                    up=up,
                    per_expert=down.shape[0] // self.experts,
                )
            with self.derived_key_versions.write_window(self.weight_version, key_layers.values()):
                frozen_keys = {h: derived_keys(layer) for h, layer in key_layers.items()}

        def shares() -> dict[str, list[float]]:
            return {
                t: (n / n.sum()).tolist()
                for t, n in counts.items()
                if t in self.targets and float(n.sum()) > 0
            }

        region = (getattr(record, "extra", None) or {}).get("region")
        if region is not None:
            target = self._region_target(st, region)
            self.last_note = (
                f"region {target.name or '(unnamed)'}: "
                f"{'new expert' if target.new else 'expert'} {target.index}"
            )

            def shares() -> dict[str, list[float]]:
                one = [float(e == target.index) for e in range(self.experts)]
                return {t: list(one) for t in self.targets}

        snapshot_start = time.perf_counter()
        before = st.chunks.snapshot() if merge_on or st.chunks.residual else None
        before_u = st.chunks.expanded_u(self.editor) if merge_on else None
        history = st.chunks.prior.get("merge_write_directions") if merge_on else None
        snapshot_seconds = time.perf_counter() - snapshot_start
        w.step(st.chunks, loss, shares, seed=0)
        if history is not None:
            st.chunks.prior["merge_write_directions"] = history
        if merge_on:
            snapshot_start = time.perf_counter()
            after_write = st.chunks.snapshot()
            snapshot_seconds += time.perf_counter() - snapshot_start
            self.path_snapshot_seconds = snapshot_seconds
            after_refs = (dict(st.chunks.coef), dict(st.chunks.node_coef))
            paths_before = len(st.forest.branches[st.forest.current].paths)
            after_u = st.chunks.expanded_u(self.editor)
            self._write_target = {h: after_u[h] - before_u[h] for h in self.targets}
            recording_start = time.perf_counter()
            for key, values in raw_keeps.items():
                keeps[key] = [
                    v.float().reshape(-1, self.experts, v.shape[-1]).mean((0, 2)).cpu()
                    for v in values
                ]
            measured_shares = shares()
            self.merge_recording_seconds = recording_time + time.perf_counter() - recording_start
            self._graph_state = before
            try:
                if self._stage_writes and self.options.get("cross_layer", "off") in ("rls", "jvp"):
                    cross_mode = self.options["cross_layer"]
                    self.options["cross_layer"] = "off"
                    try:
                        self._path_merge(st.chunks, state, measured_shares, keeps, frozen_keys)
                        if self.path_report and self.path_report[0]:
                            self._bound_path_write(st.chunks, after_write, w)
                        self._staged_merge_off = (after_write, st.chunks.snapshot())
                    finally:
                        self.options["cross_layer"] = cross_mode
                        st.chunks.restore(after_write)
                        del st.forest.branches[st.forest.current].paths[paths_before:]
                if timing:
                    started = time.perf_counter()
                    response_before = self._cross_layer_times["response"]
                self._path_merge(st.chunks, state, measured_shares, keeps, frozen_keys)
                if timing:
                    self._cross_layer_stamp("other", started)
                    self._cross_layer_times["other"] -= (
                        self._cross_layer_times["response"] - response_before
                    )
            finally:
                self._graph_state = None
            if self.path_report:
                merged = st.chunks.expanded_u(self.editor)
                steps, _, angle = self.path_report
                self.path_report = (
                    steps,
                    self._expanded_norm({h: merged[h] - after_u[h] for h in self.targets}),
                    angle,
                )
            if self.path_report and self.path_report[0]:
                self._bound_path_write(st.chunks, after_write if self._stage_writes else before, w)
            if self._stage_writes:
                compact_start = time.perf_counter()
                term_count = st.chunks.compact_residual(int(self.options.get("term_budget", 64)))
                self.merge_compaction = (term_count, time.perf_counter() - compact_start)
                branch = st.forest.branches[st.forest.current]
                paths = branch.paths[paths_before:]
                del branch.paths[paths_before:]
                merged = st.chunks.snapshot()
                st.chunks.restore(after_write)
                st.chunks.coef, st.chunks.node_coef = after_refs
                self._remember_write(before, after_write)
                after_write = st.chunks.snapshot()
                self._staged_merge = (after_write, merged, paths)
            self._write_target = None
            log.debug("path graph recomputed nodes=%d/%d transitions=%d/%d", *self.path_graph.stats)
        else:
            if before is not None:
                self._bound_path_write(st.chunks, before, w)
            self.rotation_residual = self.rotation_residual_mean = 0.0
            log.debug("path graph recomputed nodes=0/0 transitions=0/0 (path merge off)")
            log.debug("path rotation projection relative residual: max=0 mean=0 (steps=0)")
        compact_start = time.perf_counter()
        terms = st.chunks.compact_residual(int(self.options.get("term_budget", 64)))
        compact_seconds = time.perf_counter() - compact_start
        self.last_note = (
            f"{self.last_note} terms={terms} compaction_s={compact_seconds:.4g}".strip()
        )
        result = self._expanded_state(state)
        if timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - write_started) * 1000
            measured = sum(self._cross_layer_times.values()) - prior_times
            self._cross_layer_times["other"] += max(0.0, elapsed - measured)
        return result

    def _prediction_write(self, state, record, stats):
        from kilora import prediction_write as pw
        from kilora.recording import block_index

        if stats is None or self._write_scale() <= 0:
            return None
        st = self.session.editor_state
        self._apply_packs(st.chunks)
        families = [c for c in self.editor.chunks if c.family in st.chunks.enabled]
        frames = self._cross_frames(self.session)
        if not families or not frames:
            return None
        w = self._writer([None])
        self.writer = w
        self.path_report = None
        self.merge_accounting = None
        scale = pw.session_prior_scale(self.session)
        per_hook = {c.family: self.editor.per_hook(c.q) for c in families}
        variance = []
        for c in families:
            k = int(c.q.shape[0])
            eig = c.eigenvalues.detach().double().reshape(-1)
            eig = eig[:k] if eig.numel() >= k else torch.ones(k, dtype=torch.float64)
            variance.append(eig.clamp_min(1e-12))
        variance = torch.cat(variance)
        responses, p_out, priors, depth_proj = {}, {}, {}, {}
        for t in self.targets:
            if t not in frames or not frames[t]:
                continue
            m = torch.cat([per_hook[c.family][t].double() for c in families], 0)

            def response(z, m=m):
                return torch.einsum("jor,...r->...oj", m, z.to(m))

            responses[(t, 0)] = (t, response)
            p_out[(t, 0)] = frames[t][0].p_out
            depth_proj[t] = frames[t][0].p_in
            current = torch.cat(
                [
                    (
                        st.chunks.coef[c.family]
                        + st.chunks.node_coef.get((c.family, t), torch.zeros(()))
                    )
                    .double()
                    .mean(0)
                    for c in families
                ]
            )
            priors[t] = pw.NodePrior(
                current, torch.zeros_like(current), torch.diag(1.0 / (scale * variance))
            )
        blocks = getattr(self, "_block_cache", None)
        block_frames = blocks[1] if blocks is not None and blocks[0] is self.session else None

        def block_frame(target, width):
            index = block_index(target, "blocks")
            if block_frames is None or index is None:
                return None
            return block_frames.frame("blocks", index, width)

        terms = pw.build_terms(
            getattr(stats, "cross_errors", {}) or {},
            responses,
            p_out,
            depth=getattr(stats, "prediction_errors", {}) or {},
            depth_proj=depth_proj,
            hier=getattr(st, "hier_errors", {}) or {},
            clamp=getattr(st, "clamp_errors", {}) or {},
            block_frame=block_frame,
            block_of=lambda target: block_index(target, "blocks"),
        )
        result = pw.solve_map(terms, priors, pw.session_weights(self.session))
        before = st.chunks.snapshot()
        before_u = st.chunks.expanded_u(self.editor)
        for t, delta in result.delta.items():
            offset = 0
            for c in families:
                k = int(c.q.shape[0])
                part = delta[offset : offset + k].float()
                offset += k
                key = (c.family, t)
                base = st.chunks.node_coef.get(key)
                if base is None:
                    base = torch.zeros_like(st.chunks.coef[c.family])
                st.chunks.node_coef[key] = base + part.unsqueeze(0).to(base)
        st.chunks.events += 1
        self._bound_path_write(st.chunks, before, w)
        after_u = st.chunks.expanded_u(self.editor)
        norm = self._expanded_norm({h: after_u[h] - before_u[h] for h in self.targets})
        self.last_note = result.note(norm).strip()
        self.provisional = True
        return self._expanded_state(state)

    def _coefficient_vector(self, snapshot):
        parts = []
        for chunk in self.editor.chunks:
            for hook in self.targets:
                coef = snapshot["coef"][chunk.family]
                node = snapshot["node_coef"].get((chunk.family, hook))
                parts.append((coef + node if node is not None else coef).reshape(-1).double())
        return torch.cat(parts)

    def _remember_write(self, before, after):
        delta = self._coefficient_vector(after) - self._coefficient_vector(before)
        if not bool(delta.norm()):
            return
        chunks = self.session.editor_state.chunks
        history = chunks.prior.get("merge_write_directions")
        row = delta.float().unsqueeze(0)
        chunks.prior["merge_write_directions"] = (
            row if history is None else torch.cat((history, row), dim=0)[-8:]
        )

    @staticmethod
    def _merge_units(delta, history):
        delta = delta.double().reshape(-1)
        norm = float(delta.norm())
        if not norm or history is None or not history.numel():
            return 0.0, 0.0, 0.0 if not norm else 1.0
        rows = history.double().reshape(-1, delta.numel())
        norms = torch.linalg.vector_norm(rows, dim=1)
        rows, norms = rows[norms > 0], norms[norms > 0]
        if not len(norms):
            return 0.0, 0.0, 1.0
        median = float(torch.quantile(norms, 0.5))
        _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
        rank = int(
            (singular > singular.max() * max(rows.shape) * torch.finfo(rows.dtype).eps).sum()
        )
        basis = vh[:rank]
        projected = (delta @ basis.T) @ basis
        return (
            norm / median,
            float(projected.norm()) / norm,
            float((delta - projected).norm()) / norm,
        )

    def begin_write(self):
        self._stage_writes = True
        self._staged_merge = None
        self._staged_merge_off = None
        self.merge_accounting = None
        self._jvp_write_pairs = {}
        self._jvp_first_fallback = None
        if self.options.get("cross_layer") == "jvp":
            self._cross_layer_times = {
                "response": 0.0,
                "link_update": 0.0,
                "merge_accounting": 0.0,
                "other": 0.0,
            }

    def finish_write(self, session, records):
        self._stage_writes = False
        staged, self._staged_merge = self._staged_merge, None
        result = (
            ""
            if staged is None
            else self.commit_path_merge(session, records, staged, source="merge")
        )
        if self.options.get("cross_layer") == "jvp":
            if "cross_layer_jvp=" not in result:
                result += self._jvp_write_note()
            pairs = self._jvp_write_pairs
            if pairs and all(reason is not None for reason in pairs.values()):
                result += " " + text("msg.cross_layer_jvp_unavailable")
        return result

    def _jvp_write_note(self):
        pairs = self._jvp_write_pairs
        used = sum(reason is None for reason in pairs.values())
        note = f" cross_layer_jvp={used}/{len(pairs)}"
        if used < len(pairs):
            note += (
                f" cross_layer_fallback={json.dumps(self._jvp_first_fallback, ensure_ascii=False)}"
            )
        return note

    def commit_path_merge(self, session, records, staged, *, source):

        from kilora.editors import guard_check
        from kilora.live import LearnResult

        timing = self.options.get("cross_layer") == "jvp" and hasattr(self, "_cross_layer_times")
        if timing:
            started = time.perf_counter()
        before, merged, paths = staged
        chunks = session.editor_state.chunks
        delta = self._coefficient_vector(merged) - self._coefficient_vector(before)
        before_u, merged_u = self._snapshot_residual(before), self._snapshot_residual(merged)
        residual_changed = any(not torch.equal(before_u[h], merged_u[h]) for h in self.targets)
        if not bool(delta.norm()) and not residual_changed:
            return ""
        self.merge_accounting = self._merge_units(delta, chunks.prior.get("merge_write_directions"))
        off_equiv = ""
        if self.options.get("cross_layer", "off") in ("rls", "jvp") and self._staged_merge_off:
            off_before, off_after = self._staged_merge_off
            off_delta = self._coefficient_vector(off_after) - self._coefficient_vector(off_before)
            off_units = self._merge_units(off_delta, chunks.prior.get("merge_write_directions"))[0]
            off_equiv = f" merge_equiv_off={off_units:.4g}"
        if self.options.get("cross_layer") == "jvp":
            off_equiv += self._jvp_write_note()
        if timing:
            self._cross_layer_stamp("merge_accounting", started)
            started = time.perf_counter()
        editor_before = session.editor_snapshot()
        chunks.restore(merged)
        chunks.prior = {k: v.clone() for k, v in before["prior"].items()}
        session.editor_state.forest.branches[session.editor_state.forest.current].paths.extend(
            paths
        )
        new = self._expanded_state(session.state)
        if not all(bool(torch.isfinite(v).all()) for parts in new.values() for v in parts.values()):
            session.editor_state.load_tensors(editor_before)
            return " Merge refused: non-finite state."
        for record in records:
            refused, note = guard_check(self, session, record, new)
            if refused:
                session.editor_state.load_tensors(editor_before)
                return f" Merge refused: {note}"
        session.pending = {
            h: {k: v.detach().cpu().clone() for k, v in parts.items()} for h, parts in new.items()
        }
        session._pending_info = LearnResult(
            sum(r.images for r in records), 1.0, 0.0, False, False, source
        )
        if self.options.get("keep_merge_pairs", False):
            session.editor_state.merge_v0 = {
                k: v.detach().cpu().clone() for k, v in self.v0.items()
            }

            def pair_snapshot(snap):
                return {
                    "coef": {k: v.detach().cpu().clone() for k, v in snap["coef"].items()},
                    "node_coef": {
                        k: v.detach().cpu().clone() for k, v in snap["node_coef"].items()
                    },
                    "enabled": list(snap["enabled"]),
                }

            branch = session.editor_state.forest.branches[session.editor_state.forest.current]
            branch.merge_pairs.append(
                (len(session.commits) + 1, pair_snapshot(before), pair_snapshot(merged))
            )
            del branch.merge_pairs[:-32]
        result = session.commit(editor_before=editor_before)
        if timing:
            self._cross_layer_stamp("other", started)
        timing_note = (
            "".join(
                f" cross_layer_ms:{phase}={value:.4g}"
                for phase, value in self._cross_layer_times.items()
            )
            if timing
            else ""
        )
        a, b, c = self.merge_accounting
        residual = getattr(self, "merge_residual_share", 0.0)
        term_count, seconds = getattr(self, "merge_compaction", (0, 0.0))
        residual_note = (
            " (explore draw: no residuals kept)" if getattr(self, "_explore_draw", False) else ""
        )
        return (
            f" {result} source={source} merge_equiv={a:.4g}{off_equiv} "
            f"merge_repeat={b:.4g} merge_new={c:.4g} residual_kept={residual:.4g}{residual_note} "
            f"terms={term_count} compaction_s={seconds:.4g}{timing_note}."
        )

    def _snapshot_residual(self, snapshot):
        result = {h: torch.zeros(self.experts, *self.editor.shapes[h]) for h in self.targets}
        for (hook, expert), (up, down) in snapshot.get("residual", {}).items():
            result[hook][expert] += (up @ down).to(result[hook])
        return result

    def _expanded_state(self, state):
        st = self.session.editor_state

        u = st.chunks.expanded_u(self.editor)
        new = {t: {p: v.clone() for p, v in parts.items()} for t, parts in state.items()}
        for t in self.targets:
            up = state[t]["up"]
            b0 = st.base[t].reshape(-1, up.shape[-1]).t()
            b = b0.clone()
            for e in range(self.experts):
                m = (self.col_expert[t] == e).float()
                b = b + (u[t][e].float() @ self.v0[t].float()) * m
            if self._per_hook_cap() is not None:
                lim = self._per_hook_cap() * self.hook_scale.get(t, 0.0)
                d = b - b0
                n = float(d.norm())
                if lim > 0 and n > lim:
                    b = b0 + d * (lim / n)
            new[t]["up"] = b.t().reshape(up.shape).to(up.dtype)
        self.weight_version += 1
        self.derived_key_versions.advance(self.weight_version)
        return new

    def _path_merge(self, chunks, state, shares, keeps, frozen_keys=None):
        depths = {
            h: max((k + 2 for name, k in keeps if name == h), default=1) for h in self.targets
        }
        layers = []
        for name in self.targets:
            ex = self.live_set.experts[name]
            kin, rin, kout, rout = (
                frozen_keys[name][:4]
                if frozen_keys is not None
                else (ex.keys_in, ex.rows_in, ex.keys_out, ex.rows_out)
            )
            layer = LayerSet(
                router=self.live_set.router_for(name),
                keys_in=kin,
                keys_out=kout,
                rows_in=rin,
                rows_out=rout,
            )
            layers.append((name, layer, depths[name]))
        cross = {
            (a, b): torch.ones(self.experts)
            for a, b in zip(self.targets, self.targets[1:], strict=False)
        }
        if self.options.get("cross_layer", "off") in ("rls", "jvp"):
            from kilora.cross_layer import variance_weight

            for (a, b), flat in cross.items():
                link = self._cross_link((a, b))
                if link is None:
                    continue
                precision = variance_weight(link.residual_variance)
                energy = link.weights.float().square().sum(dim=1)
                basis = self.v0[a].float().cpu()
                if energy.numel() != basis.shape[0]:
                    continue
                transmission = torch.stack(
                    [
                        (basis[:, self.col_expert[a].cpu() == e].square().sum(dim=1) * energy).sum()
                        for e in range(self.experts)
                    ]
                ).sqrt()
                peak = transmission.max()
                if peak > 0:
                    transmission = transmission / peak
                    cross[(a, b)] = flat * (1 - precision) + transmission * precision

        def tensor_version(tensor):
            if tensor is None:
                return None
            if tensor.is_inference():
                return object()
            return (id(tensor), tensor._version)

        versions = tuple(
            (
                h,
                depths[h],
                *(
                    tensor_version(getattr(self.live_set.experts[h], field))
                    for field in ("keys_in", "keys_out", "rows_in", "rows_out")
                ),
            )
            for h in self.targets
        )
        snapshot = getattr(self, "_graph_state", None) or chunks.snapshot()
        old = getattr(self, "_graph_node_values", {})
        node_versions = getattr(self, "_graph_node_versions", {})
        values = {}
        for h in self.targets:
            for e in range(self.experts):
                key = (h, e)
                parts = []
                for family in sorted(snapshot["coef"]):
                    parts.append(
                        (
                            family,
                            family in snapshot["enabled"],
                            snapshot["coef"][family][e].detach().clone(),
                            snapshot["node_coef"].get((family, h)),
                        )
                    )
                parts = [
                    (
                        f,
                        enabled,
                        coef,
                        torch.zeros_like(coef) if delta is None else delta[e].detach().clone(),
                    )
                    for f, enabled, coef, delta in parts
                ]
                residual = snapshot.get("residual", {}).get(key)
                if residual is not None:
                    value = (residual[0] @ residual[1]).detach().clone().reshape(-1)
                else:
                    value = torch.zeros(math.prod(self.editor.shapes[h]))
                parts.append(("__residual__", True, value, torch.zeros_like(value)))
                previous = old.get(key)
                same = (
                    previous is not None
                    and len(previous) == len(parts)
                    and all(
                        a[0:2] == b[0:2] and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
                        for a, b in zip(previous, parts, strict=True)
                    )
                )
                if not same:
                    node_versions[key] = node_versions.get(key, 0) + 1
                values[key] = parts
        self._graph_node_values, self._graph_node_versions = values, node_versions
        graph = self.path_graph.get((id(self.live_set), versions), layers, cross, node_versions)

        def stamp():
            if torch.cuda.is_available() and any(v.is_cuda for v in self.v0.values()):
                torch.cuda.synchronize()
            return time.perf_counter()

        section_start = stamp()
        observed_keep = {h: torch.stack(v).mean(0) for h, v in keeps.items() if v}
        u = chunks.expanded_u(self.editor)
        updates = {h: torch.zeros_like(v) for h, v in u.items()}
        rotation_edges = []
        spread_edges = []
        reads = {}

        def read_for(hook, expert):
            key = (hook, expert)
            if key not in reads:
                mask = (self.col_expert[hook] == expert).float()
                reads[key] = (self.v0[hook].float() * mask) @ _down(state[hook])
            return reads[key]

        owned = set()

        def add_coefficients(hook, expert, coeff, active):
            offset = 0
            for chunk in active:
                n = chunk.q.shape[0]
                key = (chunk.family, hook)
                if key not in owned:
                    old = chunks.node_coef.get(key)
                    chunks.node_coef[key] = (
                        old.clone()
                        if old is not None
                        else torch.zeros_like(chunks.coef[chunk.family])
                    )
                    owned.add(key)
                chunks.node_coef[key][expert] += coeff[offset : offset + n].to(
                    chunks.node_coef[key]
                )
                offset += n

        hook_order = {name: i for i, name in enumerate(self.targets)}
        ordered_graph = sorted(
            graph,
            key=lambda edge: (
                hook_order[edge.predecessor.hook],
                edge.predecessor.depth,
                edge.predecessor.expert,
                hook_order[edge.successor.hook],
                edge.successor.depth,
                edge.successor.expert,
            ),
        )
        edge_start = stamp()
        setup_seconds = edge_start - section_start
        for edge in ordered_graph:
            pred, succ = edge.predecessor, edge.successor
            if pred.hook == succ.hook:
                q = observed_keep.get((pred.hook, pred.depth))
                if q is None:
                    continue
                weight = edge.weight * float(q[pred.expert])
            else:
                if pred.hook not in shares or succ.hook not in shares:
                    continue
                weight = shares[pred.hook][pred.expert] * shares[succ.hook][succ.expert]
                if self.options.get("cross_layer", "off") in ("rls", "jvp"):
                    weight *= edge.weight
            if weight == 0:
                continue
            down = _down(state[succ.hook])
            if down.shape[1] != u[pred.hook].shape[1]:
                continue
            read = read_for(succ.hook, succ.expert)
            core = read @ u[pred.hook][pred.expert].float()
            updates[pred.hook][pred.expert] += weight * (u[succ.hook][succ.expert].float() @ core)
            rotation_edges.append((pred, succ, weight))
            spread_edges.append((pred, succ, weight, read))
            editor_state = getattr(getattr(self, "session", None), "editor_state", None)
            if editor_state is not None:
                forest = editor_state.forest
                forest.branches[forest.current].paths.append(
                    {
                        "predecessor": [pred.hook, pred.depth, pred.expert],
                        "successor": [succ.hook, succ.depth, succ.expert],
                        "p": weight,
                    }
                )

        spread_start = stamp()
        edge_seconds = spread_start - edge_start
        target = getattr(self, "_write_target", None)
        if target is not None:
            predecessors = {pred for pred, _, _, _ in spread_edges}
            successors = {succ for _, succ, _, _ in spread_edges}
            signals = {
                node: target[node.hook][node.expert].float().clone()
                for node in successors - predecessors
            }
            for pred, succ, weight, read in reversed(spread_edges):
                signal = signals.get(succ)
                if signal is None:
                    continue
                propagated = read.T @ (u[succ.hook][succ.expert].float().T @ signal)
                if pred.hook != succ.hook and self.options.get("cross_layer", "off") in (
                    "rls",
                    "jvp",
                ):
                    from kilora.cross_layer import pass_back, variance_weight

                    link = self._cross_link((pred.hook, succ.hook))
                    if link is not None and signal.shape[0] == propagated.shape[0]:
                        precision = variance_weight(link.residual_variance)
                        if precision:
                            passed = pass_back(link.weights.T, signal)
                            if passed.shape == propagated.shape:
                                propagated = propagated * (1 - precision) + passed * precision
                propagated = weight * propagated
                signals[pred] = signals.get(pred, torch.zeros_like(propagated)) + propagated
                updates[pred.hook][pred.expert] += propagated

        truncation_start = stamp()
        spread_seconds = truncation_start - spread_start
        residual_size = projected_size = 0.0
        keep_residual = self.options.get("merge_residual", "keep") == "keep" and not getattr(
            self, "_explore_draw", False
        )
        merged_norm = math.sqrt(sum(float(v.square().sum()) for v in updates.values()))
        self.path_report = (
            len(spread_edges),
            merged_norm,
            float(self.options.get("path_angle", 0.05)),
        )
        for h, update in updates.items():
            if not torch.count_nonzero(update):
                continue
            active = [c for c in self.editor.chunks if c.family in chunks.enabled]
            if not active:
                continue
            directions = torch.cat([self.editor.per_hook(c.q)[h] for c in active], dim=0)
            basis = directions.reshape(directions.shape[0], -1).t().float()
            for e in range(self.experts):
                if not torch.count_nonzero(update[e]):
                    continue
                candidate = u[h][e].float() + update[e]
                if torch.equal(candidate, u[h][e]):
                    continue
                q, r = torch.linalg.qr(candidate, mode="reduced")
                left, singular, right = torch.linalg.svd(r, full_matrices=False)
                truncated = (q @ left * singular) @ right
                delta = (truncated - u[h][e].float()).reshape(-1)
                coeff = torch.linalg.lstsq(basis, delta[:, None]).solution[:, 0]
                add_coefficients(h, e, coeff, active)
                projected = basis @ coeff
                residual = delta - projected
                if keep_residual:
                    residual_size += float(residual.square().sum())
                projected_size += float(projected.square().sum())
                if keep_residual and float(residual.norm()) > 1e-6 * float(delta.norm()):
                    chunks.add_residual(h, e, residual.reshape_as(update[e]))

        rotation_start = stamp()
        truncation_seconds = rotation_start - truncation_start
        residuals = []
        angle = float(getattr(self, "options", {}).get("path_angle", 0.05))
        active = [c for c in self.editor.chunks if c.family in chunks.enabled]
        bases = {}
        for index, (pred, succ, weight) in enumerate(rotation_edges):
            if (pred.hook, pred.expert) == (succ.hook, succ.expert) or not active or angle == 0:
                continue
            current = chunks.expanded_u(self.editor, {pred.hook, succ.hook})
            bp = current[pred.hook][pred.expert].float()
            bs = current[succ.hook][succ.expert].float()
            read = read_for(succ.hook, succ.expert)
            rp, ra = rotate_pair(
                bp,
                read,
                f"{pred.hook}:{pred.expert}:{succ.hook}:{succ.expert}",
                index,
                angle * min(abs(weight), 1.0),
            )
            desired = bs @ ra
            projected_s = torch.linalg.lstsq(read.T, desired.T).solution.T
            for node, delta in ((pred, rp - bp), (succ, projected_s - bs)):
                if node.hook not in bases:
                    directions = torch.cat(
                        [self.editor.per_hook(c.q)[node.hook] for c in active], dim=0
                    )
                    bases[node.hook] = directions.reshape(directions.shape[0], -1).T.float()
                coeff = torch.linalg.lstsq(bases[node.hook], delta.reshape(-1, 1)).solution[:, 0]
                add_coefficients(node.hook, node.expert, coeff, active)
                remainder = delta - (bases[node.hook] @ coeff).reshape_as(delta)
                if keep_residual:
                    residual_size += float(remainder.square().sum())
                projected_size += float((delta - remainder).square().sum())
                if keep_residual and float(remainder.norm()) > 1e-6 * float(delta.norm()):
                    chunks.add_residual(node.hook, node.expert, remainder)
            after = chunks.expanded_u(self.editor, {pred.hook, succ.hook})
            actual = (
                after[succ.hook][succ.expert].float() @ read @ after[pred.hook][pred.expert].float()
            )
            original = bs @ read @ bp
            residuals.append(float((actual - original).norm() / original.norm().clamp_min(1e-30)))
        rotation_seconds = stamp() - rotation_start
        self.merge_residual_share = (
            math.sqrt(residual_size / (residual_size + projected_size))
            if keep_residual and residual_size + projected_size
            else 0.0
        )
        self.path_merge_timings = {
            "snapshot": getattr(self, "path_snapshot_seconds", 0.0),
            "setup": setup_seconds,
            "edge": edge_seconds,
            "spread": spread_seconds,
            "truncation": truncation_seconds,
            "rotation": rotation_seconds,
        }
        log.info(
            "path merge edges=%d spread_edges=%d nodes=%d seconds=%s",
            len(graph),
            len(spread_edges),
            len({node for edge in graph for node in (edge.predecessor, edge.successor)}),
            self.path_merge_timings,
        )
        self.rotation_residual = max(residuals, default=0.0)
        self.rotation_residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
        log.debug(
            "path rotation projection relative residual: max=%.8g mean=%.8g (steps=%d)",
            self.rotation_residual,
            self.rotation_residual_mean,
            len(residuals),
        )

    def _bound_path_write(self, chunks, before, writer):
        def copy_coefficients():
            return (
                {k: v.clone() for k, v in chunks.coef.items()},
                {k: v.clone() for k, v in chunks.node_coef.items()},
                {k: (u.clone(), v.clone()) for k, (u, v) in chunks.residual.items()},
            )

        def assign(start, end, scale):
            for group, old, new in zip(
                (chunks.coef, chunks.node_coef), start[:2], end[:2], strict=True
            ):
                group.clear()
                for key in old.keys() | new.keys():
                    exemplar = new.get(key, old.get(key))
                    a = old.get(key, torch.zeros_like(exemplar))
                    b = new.get(key, torch.zeros_like(exemplar))
                    group[key] = a + (b - a) * scale
            chunks.residual = {}
            for key in start[2].keys() | end[2].keys():
                pieces = []
                if key in start[2] and scale != 1:
                    up, down = start[2][key]
                    pieces.append((up * (1 - scale), down))
                if key in end[2] and scale != 0:
                    up, down = end[2][key]
                    pieces.append((up * scale, down))
                if pieces:
                    chunks.residual[key] = (
                        torch.cat([p[0] for p in pieces], 1),
                        torch.cat([p[1] for p in pieces], 0),
                    )

        def hook_norms():
            expanded = chunks.expanded_u(self.editor)
            result = {}
            for h, values in expanded.items():
                v0 = self.v0[h].float()
                result[h] = math.sqrt(
                    sum(
                        float(
                            ((values[e].float() @ v0) * (self.col_expert[h] == e).float())
                            .square()
                            .sum()
                        )
                        for e in range(self.experts)
                    )
                )
            return result

        old = (before["coef"], before["node_coef"], before.get("residual", {}))
        new = copy_coefficients()
        assign(old, new, 0.0)
        old_u = chunks.expanded_u(self.editor)
        assign(old, new, 1.0)
        new_u = chunks.expanded_u(self.editor)
        delta_norm = math.sqrt(
            sum(
                float(
                    (
                        ((new_u[h][e] - old_u[h][e]).float() @ self.v0[h].float())
                        * (self.col_expert[h] == e).float()
                    )
                    .square()
                    .sum()
                )
                for h in self.targets
                for e in range(self.experts)
            )
        )
        if delta_norm > writer.write_norm:
            assign(old, new, max(writer.write_norm, 0.0) / delta_norm)
        norms = hook_norms()
        total = math.sqrt(sum(n * n for n in norms.values()))
        scale = (
            min(1.0, writer.total_cap / total) if writer.total_cap is not None and total else 1.0
        )
        cap = self._per_hook_cap()
        if cap is not None:
            for h, n in norms.items():
                limit = cap * self.hook_scale.get(h, 0.0)
                if limit > 0 and n > limit:
                    scale = min(scale, limit / n)
        if scale < 1.0:
            current = copy_coefficients()
            assign(({}, {}, {}), current, scale)
        if self.path_report:
            steps, norm, angle = self.path_report
            self.path_report = (
                steps,
                norm * min(1.0, writer.write_norm / delta_norm) * scale if delta_norm else 0.0,
                angle,
            )

    def checkout_branch(self, branch_id, *, promote=False):
        if self.session.pending is not None:
            raise KiLoRAError("finish the pending update before switching branches")
        st = self.session.editor_state
        if promote:
            self.session.unpin()
            st.promote(branch_id)
        else:
            st.checkout(branch_id)
        self.session.state = self._expanded_state(self.session.state)
        self.session.revision += 1

    def _per_hook_cap(self) -> float | None:
        w = self.manifest["writer"]
        on = self.options.get("per_hook_cap", w.get("per_hook_cap", False))
        frac = w.get("cap_frac")
        return float(frac) if on and frac is not None else None

    def guard_trial(self, state, new):
        if self.session is None:
            return None
        down, dup = {}, {}
        for t in self.targets:
            up = state[t]["up"]
            down[t] = _down(state[t])
            dup[t] = (new[t]["up"].float() - up.float()).reshape(-1, up.shape[-1]).t()
        return RoutedWrite(
            down,
            dup,
            self.col_expert,
            self.routes,
            getattr(self.session.mod.settings, "routing", "tensor"),
            self.tau,
            set_key=self.session.mod_id,
        )

    def _region_target(self, st, region):
        from kilora.regions import region_keys

        mask = region.mask
        probes = region.recorded_probes()
        keys = region_keys(probes, mask) if probes else {}
        experts = {t: k for t, (_, k) in self.routes.items()}
        return st.regions.target_for(region, keys, experts)


def _down(parts) -> torch.Tensor:
    down = parts["down"].float()
    return down.reshape(_columns(down), down.shape[-1])


def inspect(path: str) -> EditorFileInfo:
    m = read_manifest(path)
    ed = ChunkLive(path).editor
    rows = ", ".join(f"{c.family} {c.q.shape[0]}" for c in ed.chunks)
    return EditorFileInfo(
        label="editor (chunk)",
        targets=list(ed.names),
        facts=[f"packs: {rows} directions", f"write norm {m['writer']['write_norm']}"],
    )


PLUGIN = EditorPlugin(KIND, ChunkLive, inspect=inspect, title="Chunked editor")


def _columns(down: torch.Tensor) -> int:
    return int(down.numel() // down.shape[-1])
