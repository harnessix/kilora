# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import logging
import math
import time

import torch

from .compute import LayerSet
from .i18n import text
from .import_terms import TERM_FIXED, add_direction
from .path_rotation import rotate_pair

log = logging.getLogger("kilora.editor.chunk")


def mean_keeps(raw_keeps, experts):
    if isinstance(raw_keeps, KeepMeans):
        return raw_keeps.means()
    return {
        key: [v.float().reshape(-1, experts, v.shape[-1]).mean((0, 2)).cpu() for v in values]
        for key, values in raw_keeps.items()
    }


class KeepMeans:
    def __init__(self, experts):
        self.experts = int(experts)
        self._sums = {}
        self._counts = {}

    def add(self, key, values):
        mean = values.detach().float().reshape(-1, self.experts, values.shape[-1]).mean((0, 2))
        mean = mean.double().cpu()
        old = self._sums.get(key)
        self._sums[key] = mean if old is None else old + mean
        self._counts[key] = self._counts.get(key, 0) + 1

    def __bool__(self):
        return bool(self._sums)

    def __len__(self):
        return len(self._sums)

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self._sums.values())

    def means(self):
        return {k: [(v / self._counts[k]).float()] for k, v in self._sums.items()}


def share_until_write(mapping):
    return dict(mapping)


def _down(parts) -> torch.Tensor:
    down = parts["down"].float()
    return down.reshape(_columns(down), down.shape[-1])


def _columns(down: torch.Tensor) -> int:
    return int(down.numel() // down.shape[-1])


def finish_write(self, session, records):
    self._stage_writes = False
    staged, self._staged_merge = self._staged_merge, None
    result = (
        "" if staged is None else self.commit_path_merge(session, records, staged, source="merge")
    )
    if self.options.get("cross_layer") == "jvp":
        if "cross_layer_jvp=" not in result:
            result += self._jvp_write_note()
        pairs = self._jvp_write_pairs
        if pairs and all(reason is not None for reason in pairs.values()):
            result += " " + text("msg.cross_layer_jvp_unavailable")
    return result


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
    session.editor_state.forest.branches[session.editor_state.forest.current].paths.extend(paths)
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
        session.editor_state.merge_v0 = {k: v.detach().cpu().clone() for k, v in self.v0.items()}

        def pair_snapshot(snap):
            return {
                "coef": {k: v.detach().cpu().clone() for k, v in snap["coef"].items()},
                "node_coef": {k: v.detach().cpu().clone() for k, v in snap["node_coef"].items()},
                "enabled": list(snap["enabled"]),
            }

        branch = session.editor_state.forest.branches[session.editor_state.forest.current]
        branch.merge_pairs.append(
            (len(session.commits), pair_snapshot(before), pair_snapshot(merged))
        )
        del branch.merge_pairs[:-32]
    result = session.commit_part(source, editor_before=editor_before)
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


def _tensor_version(tensor):
    if tensor is None:
        return None
    if tensor.is_inference():
        return object()
    return (id(tensor), tensor._version)


def _path_merge(self, chunks, state, shares, keeps, frozen_keys=None):
    depths = {h: max((k + 2 for name, k in keeps if name == h), default=1) for h in self.targets}
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

    versions = tuple(
        (
            h,
            depths[h],
            *(
                _tensor_version(getattr(self.live_set.experts[h], field))
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
                    old.clone() if old is not None else torch.zeros_like(chunks.coef[chunk.family])
                )
                owned.add(key)
            chunks.node_coef[key][expert] += coeff[offset : offset + n].to(chunks.node_coef[key])
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
    scale = min(1.0, writer.total_cap / total) if writer.total_cap is not None and total else 1.0
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


def graph_snapshot(history):
    return (
        copy.deepcopy(history.path_graph),
        copy.deepcopy(history.path_node_versions),
        copy.deepcopy(history.path_node_values),
    )


def restore_graph(history, snapshot):
    graph, versions, values = snapshot
    history.path_graph = graph
    history.path_node_versions = versions
    history.path_node_values = values


def _clone_state(state):
    return {t: {k: v.detach().clone() for k, v in parts.items()} for t, parts in state.items()}


def _same_state(left, right):
    if set(left) != set(right):
        return False
    for target, parts in left.items():
        other = right[target]
        if set(parts) != set(other):
            return False
        for key, value in parts.items():
            if value.shape != other[key].shape or not torch.equal(value, other[key]):
                return False
    return True


def _all_finite(state):
    for parts in state.values():
        for value in parts.values():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                return False
    return True


class _Direction:
    def __init__(self, family, q):
        self.family = family
        self.q = q


class _LinearEditor:
    def __init__(self, names, shapes):
        self.names = list(names)
        self.shapes = dict(shapes)
        total = sum(int(torch.Size(self.shapes[h]).numel()) for h in self.names)
        self.chunks = [_Direction("mix", torch.eye(total))]

    def per_hook(self, q):
        out, off = {}, 0
        for hook in self.names:
            n = int(torch.Size(self.shapes[hook]).numel())
            out[hook] = q[:, off : off + n].reshape(q.shape[0], *self.shapes[hook])
            off += n
        return out


class _LinearState:
    def __init__(self, experts, editor):
        width = editor.chunks[0].q.shape[0]
        self.coef = {"mix": torch.zeros(experts, width)}
        self.node_coef = {}
        self.enabled = ["mix"]
        self.residual = {}

    def snapshot(self):
        return {
            "coef": {k: v.clone() for k, v in self.coef.items()},
            "enabled": list(self.enabled),
            "node_coef": {k: v.clone() for k, v in self.node_coef.items()},
            "residual": {k: (u.clone(), v.clone()) for k, (u, v) in self.residual.items()},
        }

    def expanded_u(self, editor, hooks=None):
        out = None
        for chunk in editor.chunks:
            if chunk.family not in self.enabled:
                continue
            per_hook = editor.per_hook(chunk.q)
            part = {
                hook: torch.einsum(
                    "ej,jor->eor",
                    self.coef[chunk.family] + self.node_coef[(chunk.family, hook)]
                    if (chunk.family, hook) in self.node_coef
                    else self.coef[chunk.family],
                    direction,
                )
                for hook, direction in per_hook.items()
                if hooks is None or hook in hooks
            }
            out = part if out is None else {hook: out[hook] + part[hook] for hook in out}
        if out is None:
            count = next(iter(self.coef.values())).shape[0]
            out = {
                hook: torch.zeros(count, *editor.shapes[hook])
                for hook in editor.names
                if hooks is None or hook in hooks
            }
        for (hook, expert), (up, down) in self.residual.items():
            if hook in out:
                out[hook][expert] = out[hook][expert] + (up @ down).to(out[hook])
        return out

    def add_residual(self, hook, expert, value):
        if not bool(torch.count_nonzero(value)):
            return
        up = value.detach().clone()
        down = torch.eye(value.shape[1], dtype=value.dtype, device=value.device)
        old = self.residual.get((hook, expert))
        if old is not None:
            up, down = torch.cat((old[0], up), 1), torch.cat((old[1], down), 0)
        self.residual[hook, expert] = (up, down)


class _ImportHost:
    def __init__(self, session, options, history, targets, experts, editor, mixes):
        self.session = None
        self.options = dict(options)
        self.options["cross_layer"] = "off"
        self.targets = list(targets)
        self.experts = experts
        self.live_set = session._source
        self.path_graph = history.path_graph
        self._graph_node_values = history.path_node_values
        self._graph_node_versions = history.path_node_versions
        self._write_target = {hook: mixes[hook][1] - mixes[hook][0] for hook in self.targets}
        self._explore_draw = False
        self.path_snapshot_seconds = 0.0
        self.editor = editor
        self.v0 = {}
        self.col_expert = {}
        self.reads = {}
        for hook in self.targets:
            rank = editor.shapes[hook][0]
            eye = torch.eye(rank)
            self.v0[hook] = eye.repeat(1, experts)
            self.col_expert[hook] = torch.arange(experts * rank) // rank
            self.reads[hook] = {"down": eye.repeat(experts, 1)}


def _ensure_graph(history):
    from .path_graph import PathGraphCache

    if not hasattr(history, "path_graph"):
        history.path_graph = PathGraphCache()
        history.path_node_versions = {}
        history.path_node_values = {}


def _hook_ranks(session, state):
    source = session._source
    experts = source.num_experts
    hooks = []
    for name, ex in source.experts.items():
        parts = state.get(name)
        if parts is None or ex.keys_in.shape[0] != experts or ex.down is None:
            continue
        if ex.down.shape[0] != experts:
            continue
        ranks = []
        usable = True
        for expert in range(experts):
            core = parts.get(f"core_{expert}")
            if core is None:
                continue
            if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] < 1:
                usable = False
                break
            ranks.append(core.shape[0])
        if usable and ranks and all(rank == ranks[0] for rank in ranks):
            hooks.append((name, ranks[0]))
    return experts, hooks


def _mix_row(parts, expert, rank):
    core = None if parts is None else parts.get(f"core_{expert}")
    strength = None if parts is None else parts.get(f"strength_{expert}")
    if core is None or strength is None or tuple(core.shape) != (rank, rank):
        return torch.zeros(rank, rank)
    return (strength.detach().reshape(()).to(core) * core.detach()).float()


def _mixes(before, after, hooks, experts):
    out = {}
    for hook, rank in hooks:
        old = torch.stack([_mix_row(before.get(hook), e, rank) for e in range(experts)])
        new = torch.stack([_mix_row(after.get(hook), e, rank) for e in range(experts)])
        out[hook] = (old, new)
    return out


def _fill_linear(linear, editor, mixes, experts):
    total = editor.chunks[0].q.shape[0]
    offset = 0
    for hook in editor.names:
        rank = editor.shapes[hook][0]
        width = rank * rank
        block = torch.zeros(experts, total)
        new = mixes[hook][1]
        for expert in range(experts):
            block[expert, offset : offset + width] = new[expert].reshape(-1)
        linear.node_coef[("mix", hook)] = block
        offset += width


def _cap_scale(before, merged):
    scale = 1.0
    for target, parts in merged.items():
        old = before.get(target, {})
        for key, value in parts.items():
            if not (key.startswith("core_") or key.startswith("strength_")):
                continue
            base = old.get(key)
            if base is None or base.shape != value.shape:
                continue
            base_norm = float(base.float().norm())
            step_norm = float((value - base).float().norm())
            if step_norm > base_norm:
                scale = min(scale, base_norm / step_norm if base_norm else 0.0)
    return scale


def _blend_state(before, merged, scale):
    if scale == 1.0:
        return merged
    out = _clone_state(merged)
    for target, parts in out.items():
        old = before.get(target, {})
        for key, value in list(parts.items()):
            if key in TERM_FIXED:
                continue
            base = old.get(key)
            if base is None:
                if key == "terms_c":
                    parts[key] = value * scale
                continue
            if base.shape == value.shape:
                parts[key] = base + scale * (value - base)
                continue
            if key != "terms_c":
                continue
            grown = torch.zeros_like(value)
            grown[tuple(slice(0, size) for size in base.shape)] = base.to(value)
            parts[key] = grown + scale * (value - grown)
    return out


def _top_pair(up, leftover, down):

    rank = leftover.shape[0]
    small = leftover.to(up)
    if up.shape[0] < rank or down.shape[1] < rank:
        extra = up @ small @ down
        if float(extra.norm()) <= 1e-6:
            return None
        left, values, right = torch.linalg.svd(extra, full_matrices=False)
        if not values.numel():
            return None
        return left[:, 0], right[0], float(values[0])
    q_up, r_up = torch.linalg.qr(up, mode="reduced")
    q_down, r_down = torch.linalg.qr(down.T, mode="reduced")
    core = r_up @ small @ r_down.T
    if float(core.norm()) <= 1e-6:
        return None
    left, values, right = torch.linalg.svd(core, full_matrices=False)
    if not values.numel():
        return None
    return q_up @ left[:, 0], q_down @ right[0], float(values[0])


def _store_leftover(session, history, parts, target, expert, leftover):
    if float(leftover.norm()) <= 1e-5:
        return False
    ex = session._source.experts[target]
    rank = leftover.shape[0]
    down = ex.down[expert, 0, :rank].detach().float()
    up = ex.up[expert, 0, :rank].detach().float().T
    pair = _top_pair(up, leftover, down)
    if pair is None or pair[2] <= 1e-8:
        return False
    left, right, value = pair
    scale = value**0.5
    serial = int(getattr(history, "path_term_serial", 0))
    history.path_term_serial = serial + 1
    row, col = add_direction(
        parts,
        f"path-{serial}-o",
        (left * scale).contiguous(),
        f"path-{serial}-i",
        (right * scale).contiguous(),
    )
    coeff = parts["terms_c"].clone()
    coeff[row, col] = coeff[row, col] + 1
    parts["terms_c"] = coeff
    return True


def _session_core_moves(session):
    grouped = {}
    for move in session.moves():
        if move.part == "core":
            grouped.setdefault((move.target, move.expert), []).append(move)
    return lambda hook, expert: grouped.get((hook, expert), ())


def _pair_core_moves(session):

    from .live_imports import Move, _core_frame

    align = getattr(session, "align_for_moves", None)
    if align is not None:
        align()
    dims = int(getattr(session, "core_dims", 2))
    cache = {}

    def build(target, expert):
        ex = session._source.experts.get(target)
        if ex is None or ex.down is None or expert >= ex.down.shape[0]:
            return ()
        width = int(ex.down.shape[2])
        ranks = tuple(width for _ in range(ex.down.shape[0])) if ex.ranks is None else ex.ranks
        kept = int(ranks[expert]) if expert < len(ranks) else width
        core = session.state.get(target, {}).get(f"core_{expert}")
        if kept <= 0 or core is None:
            return ()
        rank = min(int(core.shape[0]), kept)
        if rank <= 0:
            return ()
        with torch.no_grad():
            left, right, usable = _core_frame(session, target, ex, expert, rank)
            count = min(dims, usable)
            return [
                Move(
                    f"{target}:{expert}:core:{i}:{j}",
                    target,
                    expert,
                    "core",
                    torch.outer(left[:, i], right[:, j]),
                )
                for i in range(count)
                for j in range(count)
            ]

    def lookup(target, expert):
        key = (target, expert)
        if key not in cache:
            cache[key] = build(target, expert)
        return cache[key]

    return lookup


def _project_mixes(
    session,
    history,
    before,
    merged,
    old_u,
    new_u,
    experts=None,
    core_moves=None,
    active=None,
    counts=None,
):

    core_moves = _session_core_moves(session) if core_moves is None else core_moves
    visited = {"hooks": 0, "experts": 0, "moves": 0, "terms": 0}
    for hook, updated in new_u.items():
        if hook not in merged:
            continue
        parts = merged[hook]
        previous = old_u[hook]
        shape = tuple(previous.shape[1:])
        chosen = range(updated.shape[0]) if experts is None else sorted(experts.get(hook, ()))
        rows = [
            e
            for e in chosen
            if 0 <= e < updated.shape[0]
            and parts.get(f"strength_{e}") is not None
            and parts.get(f"core_{e}") is not None
            and tuple(parts[f"core_{e}"].shape) == shape
        ]
        if not rows:
            continue
        index = torch.tensor(rows, dtype=torch.long)
        change = (updated[index] - previous[index]).float()
        live = change.flatten(1).ne(0).any(1)
        if not bool(live.any()):
            continue
        rows = [e for e, keep in zip(rows, live.tolist(), strict=True) if keep]
        change = change[live]
        visited["hooks"] += 1
        visited["experts"] += len(rows)
        cores = torch.stack([parts[f"core_{e}"].detach().float().to(change) for e in rows])
        strength0 = torch.tensor(
            [float(parts[f"strength_{e}"].detach().reshape(())) for e in rows],
            dtype=change.dtype,
            device=change.device,
        )
        denom = cores.square().sum((1, 2)).double()
        along = (change * cores).sum((1, 2)).double()
        step = torch.where(denom != 0, along / torch.where(denom != 0, denom, 1.0), 0.0)
        step = torch.where(torch.isfinite(step), step, 0.0)
        delta = change - step.to(change)[:, None, None] * cores
        lists = [list(core_moves(hook, e)) for e in rows]
        width = max((len(m) for m in lists), default=0)
        moved = torch.zeros_like(change)
        if width:
            directions = torch.zeros(len(rows), width, *shape, dtype=change.dtype)
            present = torch.zeros(len(rows), width, dtype=torch.bool)
            for n, moves in enumerate(lists):
                for m, move in enumerate(moves):
                    directions[n, m] = move.direction.detach().float()
                    present[n, m] = True
            directions, present = directions.to(change.device), present.to(change.device)
            usable = (delta.flatten(1).norm(dim=1) > 1e-5) & (strength0 != 0)
            for m in range(width):
                tangent = strength0[:, None, None] * directions[:, m]
                size = tangent.square().sum((1, 2)).double()
                ratio = (delta * tangent).sum((1, 2)).double() / torch.where(size != 0, size, 1.0)
                ok = usable & present[:, m] & (size != 0) & torch.isfinite(ratio) & (ratio != 0)
                ratio = torch.where(ok, ratio, 0.0).to(change)
                moved = moved + ratio[:, None, None] * directions[:, m]
                delta = delta - ratio[:, None, None] * tangent
                visited["moves"] += int(ok.sum())
        base = change.flatten(1).norm(dim=1)
        rest = delta.flatten(1).norm(dim=1)
        steps = step.tolist()
        for n, expert in enumerate(rows):
            core_key, strength_key = f"core_{expert}", f"strength_{expert}"
            if steps[n]:
                strength = parts[strength_key]
                parts[strength_key] = strength + steps[n] * torch.ones_like(strength)
            if bool(moved[n].ne(0).any()):
                core = parts[core_key]
                parts[core_key] = core + moved[n].to(core)
            leftover = float(rest[n]) > max(1e-5, 1e-4 * float(base[n]))
            if leftover and _store_leftover(session, history, parts, hook, expert, delta[n]):
                visited["terms"] += 1
    out = _blend_state(before, merged, _cap_scale(before, merged))
    if active is not None and experts is not None:
        for hook, chosen in experts.items():
            parts, old = out.get(hook), before.get(hook, {})
            if parts is None:
                continue
            for expert in chosen:
                key = f"core_{expert}"
                if key not in parts or key not in old:
                    continue
                mask = active(hook, expert, int(old[key].shape[0]))
                if mask is not None:
                    base_core = old[key]
                    parts[key] = base_core + (parts[key] - base_core) * mask.to(base_core)
    if counts is not None:
        for key, value in visited.items():
            counts[key] = counts.get(key, 0) + value
    return out


PROFILE_ENV = "KILORA_WRITE_PROFILE"


def wants_profile() -> bool:
    import os

    return os.environ.get(PROFILE_ENV, "").strip() in ("1", "on", "true")


class MergeProfile:
    def __init__(self) -> None:
        self.paths = 0
        self.hooks = 0
        self.experts = 0
        self.steps = 0
        self.rotations = 0
        self.graph_hooks = 0
        self.fold_seconds = 0.0
        self.graph_seconds = 0.0
        self.pass_seconds = 0.0
        self.project_seconds = 0.0
        self.hook_seconds: dict[str, float] = {}
        self.project: dict[str, int] = {}
        self.skipped = ""

    def line(self) -> str:
        slow = max(self.hook_seconds.items(), key=lambda item: item[1], default=("", 0.0))
        return (
            f" merge_profile: fold_s={self.fold_seconds:.4g} graph_s={self.graph_seconds:.4g}"
            f" pass_s={self.pass_seconds:.4g} project_s={self.project_seconds:.4g}"
            f" paths={self.paths} hooks={self.hooks} experts={self.experts}"
            f" steps={self.steps} rotations={self.rotations} graph_hooks={self.graph_hooks}"
            f" hook_s_max={slow[1]:.4g}{f' ({slow[0]})' if slow[0] else ''}"
            f" project_hooks={self.project.get('hooks', 0)}"
            f" project_experts={self.project.get('experts', 0)}"
            f" project_moves={self.project.get('moves', 0)}"
            f" project_terms={self.project.get('terms', 0)}"
            f"{f' skipped={self.skipped}' if self.skipped else ''}."
        )


def _visit(order, experts, target):
    if target is None:
        return {h: list(range(experts)) for h in order}
    out = {}
    for h in order:
        chosen = sorted({int(e) for e in target.get(h, ()) if 0 <= int(e) < experts})
        if chosen:
            out[h] = chosen
    return out


def _node_values(history, order, experts, mixes):
    old = history.path_node_values
    versions = history.path_node_versions
    values = {}
    for h in order:
        new = mixes[h][1]
        for e in range(experts):
            value = new[e].detach().reshape(-1).clone()
            previous = old.get((h, e))
            same = (
                isinstance(previous, torch.Tensor)
                and previous.shape == value.shape
                and torch.equal(previous, value)
            )
            if not same:
                versions[(h, e)] = versions.get((h, e), 0) + 1
            values[(h, e)] = value
    history.path_node_values = values
    return versions


def _import_graph(session, history, order, depths, experts, versions):
    source = session._source
    layers = []
    for name in order:
        ex = source.experts[name]
        layer = LayerSet(
            router=source.router_for(name),
            keys_in=ex.keys_in,
            keys_out=ex.keys_out,
            rows_in=ex.rows_in,
            rows_out=ex.rows_out,
        )
        layers.append((name, layer, depths[name]))
    cross = {(a, b): torch.ones(experts) for a, b in zip(order, order[1:], strict=False)}
    key = tuple(
        (
            h,
            depths[h],
            *(
                _tensor_version(getattr(source.experts[h], field))
                for field in ("keys_in", "keys_out", "rows_in", "rows_out")
            ),
        )
        for h in order
    )
    return history.path_graph.get((id(source), key), layers, cross, versions)


def _path_steps(graph, order, depths, experts, keeps, visit):

    observed = {key: torch.stack(v).mean(0).tolist() for key, v in keeps.items() if v}
    hook_order = {name: i for i, name in enumerate(order)}
    weights = {
        h: torch.zeros(max(depths[h] - 1, 0), experts, experts, dtype=torch.float64) for h in visit
    }
    steps = {h: [] for h in visit}
    chosen = {h: set(experts_) for h, experts_ in visit.items()}
    ordered = sorted(
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
    index = 0
    for edge in ordered:
        pred, succ = edge.predecessor, edge.successor
        if pred.hook != succ.hook:
            continue
        q = observed.get((pred.hook, pred.depth))
        if q is None:
            continue
        weight = edge.weight * float(q[pred.expert])
        if weight == 0:
            continue
        position, index = index, index + 1
        inside = chosen.get(pred.hook)
        if inside is None or pred.expert not in inside or succ.expert not in inside:
            continue
        weights[pred.hook][pred.depth, pred.expert, succ.expert] = weight
        steps[pred.hook].append((pred.depth, pred.expert, succ.expert, weight, position))
    return weights, steps


def _fold_hook(hook, old, new, weights, steps, angle):

    u = new.float()
    w = weights.to(u)
    updates = torch.matmul(torch.einsum("ef,fij->eij", w.sum(0), u), u)
    depth = w.shape[0] + 1
    if depth > 1:
        signal = (new - old).float()
        live = w.ne(0)
        preds = live.any(2)
        succs = live.any(1)
        carried = torch.where(succs[-1][:, None, None], signal, torch.zeros_like(signal))
        for k in range(depth - 2, -1, -1):
            passed = torch.einsum("ef,fij->eij", w[k], torch.matmul(u.transpose(-1, -2), carried))
            updates = updates + passed
            if k:
                ends = succs[k - 1] & ~preds[k]
                carried = passed + torch.where(
                    ends[:, None, None], signal, torch.zeros_like(signal)
                )
    cur = u.clone()
    changed = [e for e in range(u.shape[0]) if bool(torch.count_nonzero(updates[e]))]
    if changed:
        index = torch.tensor(changed)
        candidate = u[index] + updates[index]
        q, r = torch.linalg.qr(candidate, mode="reduced")
        left, singular, right = torch.linalg.svd(r, full_matrices=False)
        truncated = (torch.matmul(q, left) * singular[..., None, :]) @ right
        for row, e in enumerate(changed):
            if not torch.equal(candidate[row], u[e]):
                cur[e] = u[e] + (truncated[row] - u[e])
    rotations = 0
    if angle != 0:
        eye = torch.eye(u.shape[-1], dtype=u.dtype, device=u.device)
        for _k, e, f, weight, position in steps:
            if e == f:
                continue
            rp, ra = rotate_pair(
                cur[e], eye, f"{hook}:{e}:{hook}:{f}", position, angle * min(abs(weight), 1.0)
            )
            cur[f] = cur[f] @ ra
            cur[e] = rp
            rotations += 1
    return cur, rotations


NO_TARGET_CHANGE = "no_target_change"


def _target_changed(mixes, visit):
    for h, chosen in visit.items():
        old, new = mixes[h]
        index = torch.tensor(chosen, dtype=torch.long)
        if bool((new[index] != old[index]).any()):
            return True
    return False


def fold_import_paths(
    session,
    options,
    history,
    before_state,
    after_state,
    keeps,
    target=None,
    profile=None,
    active=None,
):

    started = time.perf_counter()
    _ensure_graph(history)
    experts, hooks = _hook_ranks(session, after_state)
    if not experts or not hooks:
        return None
    order = [name for name, _rank in hooks]
    mixes = _mixes(before_state, after_state, hooks, experts)
    visit = _visit(order, experts, target)
    if target is not None and not _target_changed(mixes, visit):
        if profile is not None:
            profile.skipped = NO_TARGET_CHANGE
            profile.graph_hooks = len(order)
            profile.fold_seconds = time.perf_counter() - started
        return None
    depths = {h: max((k + 2 for name, k in keeps if name == h), default=1) for h in order}
    versions = _node_values(history, order, experts, mixes)
    graph = _import_graph(session, history, order, depths, experts, versions)
    weights, steps = _path_steps(graph, order, depths, experts, keeps, visit)
    graph_done = time.perf_counter()
    angle = float(options.get("path_angle", 0.05))
    folded = {}
    rotations = 0
    hook_seconds = {}
    for h in visit:
        hook_start = time.perf_counter()
        folded[h], count = _fold_hook(h, *mixes[h], weights[h], steps[h], angle)
        rotations += count
        hook_seconds[h] = time.perf_counter() - hook_start
    pass_done = time.perf_counter()
    merged = _project_mixes(
        session,
        history,
        after_state,
        _clone_state(after_state),
        {h: mixes[h][1] for h in folded},
        folded,
        experts=visit,
        core_moves=_pair_core_moves(session),
        active=active,
        counts=None if profile is None else profile.project,
    )
    done = time.perf_counter()
    if profile is not None:
        profile.hooks = len(visit)
        profile.experts = sum(len(v) for v in visit.values())
        profile.steps = sum(len(v) for v in steps.values())
        profile.rotations = rotations
        profile.graph_hooks = len(order)
        profile.graph_seconds = graph_done - started
        profile.pass_seconds = pass_done - graph_done
        profile.project_seconds = done - pass_done
        profile.fold_seconds = done - started
        profile.hook_seconds = hook_seconds
    if not _all_finite(merged) or _same_state(after_state, merged):
        return None
    return merged


def _fold_reference(session, options, history, before_state, after_state, keeps):
    _ensure_graph(history)
    experts, hooks = _hook_ranks(session, after_state)
    if not experts or not hooks:
        return None
    names = [name for name, _rank in hooks]
    shapes = {name: (rank, rank) for name, rank in hooks}
    mixes = _mixes(before_state, after_state, hooks, experts)
    editor = _LinearEditor(names, shapes)
    linear = _LinearState(experts, editor)
    _fill_linear(linear, editor, mixes, experts)
    host = _ImportHost(session, options, history, names, experts, editor, mixes)
    _path_merge(host, linear, host.reads, {}, keeps)
    history.path_node_values = host._graph_node_values
    history.path_node_versions = host._graph_node_versions
    folded = linear.expanded_u(editor)
    merged = _project_mixes(
        session,
        history,
        after_state,
        _clone_state(after_state),
        {hook: mixes[hook][1] for hook in names},
        folded,
    )
    if not _all_finite(merged) or _same_state(after_state, merged):
        return None
    return merged


def commit_import_merge(editor, session, records, staged):

    from .live import LearnResult

    before, merged, snapshot = staged
    history = session.editor_state
    if not _all_finite(merged) or _same_state(before, merged):
        restore_graph(history, snapshot)
        if not _all_finite(merged):
            return " Merge refused: non-finite state."
        return ""
    editor_before = session.editor_snapshot()
    notes = []
    for record in records:
        what = "Merge dropped"
        dropped, note = merge_loss_check(editor, session, record, merged)
        if not dropped:
            dropped, why = merge_guard(editor, session, record, merged)
            if dropped:
                what, note = "Merge refused and dropped", f"the guard: {why}"
            else:
                notes.append(note + (f"; guard: {why}" if why else ""))
        if dropped:
            if editor_before is not None:
                history.load_tensors(editor_before)
            restore_graph(history, snapshot)
            return f" {what} ({note}); the write stays."
    session.pending = {
        target: {key: value.detach().cpu().clone() for key, value in parts.items()}
        for target, parts in merged.items()
    }
    images = sum(record.images for record in records)
    session._pending_info = LearnResult(images, 1.0, 0.0, False, False, "merge")
    saved = session.commit_part("merge", editor_before=editor_before)
    return (
        f" Merge accepted ({'; '.join(n for n in notes if n) or 'checked'}). {saved} source=merge"
    )


def merge_loss_check(editor, session, record, merged) -> tuple[bool, str]:
    from .live_imports import ImportTrial, accept_measure

    measure = getattr(record, "measure", None)
    if measure is None:
        return False, "no measurement"
    run = accept_measure(measure)
    if not getattr(run, "held_out", False):
        return False, "no held-out pairs"
    state = session.state
    before = _written_loss(editor, state)
    if before is None:
        before = float(run(ImportTrial(session, state, state)))
    after = float(run(ImportTrial(session, state, merged)))
    note = f"held-out loss {before:.6g} before, {after:.6g} after the merge"
    if not (math.isfinite(after) and after < before):
        return True, f"the {note}: no decrease"
    return False, note


def _equal_states(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        return False
    for target, parts in left.items():
        other = right[target]
        if set(parts) != set(other):
            return False
        for key, value in parts.items():
            if value.shape != other[key].shape:
                return False
            if not torch.equal(value.detach().cpu(), other[key].detach().cpu()):
                return False
    return True


def _written_loss(editor, state):
    accepted = getattr(editor, "_accepted", None)
    if not isinstance(accepted, dict):
        return None
    after = accepted.get("after")
    if after is None or not math.isfinite(float(after)):
        return None
    if not _equal_states(accepted.get("state"), state):
        return None
    return float(after)


def merge_guard(editor, session, record, merged) -> tuple[bool, str]:
    from .editors import guard_enabled, guard_verdict

    if not guard_enabled(session) or getattr(record, "guard_score", None) is None:
        return False, "guard off"
    trial = getattr(editor, "guard_trial", None)
    if trial is None:
        return False, "guard off"
    state = session.state
    before = record.guard_score(trial(state, state))
    after = record.guard_score(trial(state, merged))
    return guard_verdict(session, before, after)
