# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import time

import torch
from kilora_editor_chunk.plugin import (
    KiLoRAError,
    LayerSet,
    RoutedPerturbation,
    _down,
    derived_keys,
    log,
    recording_read_keep_raw,
    rotate_pair,
)


def reference_path_merge(self, chunks, state, shares, keeps, frozen_keys=None):
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
    graph = self.path_graph.get(
        (id(self.live_set), versions, self.weight_version if frozen_keys is not None else None),
        layers,
        cross,
    )

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
        if weight == 0:
            continue
        down = _down(state[succ.hook])
        if down.shape[1] != u[pred.hook].shape[1]:
            continue
        mask = (self.col_expert[succ.hook] == succ.expert).float()
        read = (self.v0[succ.hook].float() * mask) @ down
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
            propagated = weight * (read.T @ (u[succ.hook][succ.expert].float().T @ signal))
            signals[pred] = signals.get(pred, torch.zeros_like(propagated)) + propagated
            updates[pred.hook][pred.expert] += propagated

    truncation_start = stamp()
    spread_seconds = truncation_start - spread_start
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
            offset = 0
            for c in active:
                n = c.q.shape[0]
                key = (c.family, h)
                if key not in chunks.node_coef:
                    chunks.node_coef[key] = torch.zeros_like(chunks.coef[c.family])
                chunks.node_coef[key] = chunks.node_coef[key].clone()
                chunks.node_coef[key][e] += coeff[offset : offset + n].to(chunks.node_coef[key])
                offset += n

    rotation_start = stamp()
    truncation_seconds = rotation_start - truncation_start
    residuals = []
    angle = float(getattr(self, "options", {}).get("path_angle", 0.05))
    active = [c for c in self.editor.chunks if c.family in chunks.enabled]
    for index, (pred, succ, weight) in enumerate(rotation_edges):
        if (pred.hook, pred.expert) == (succ.hook, succ.expert) or not active or angle == 0:
            continue
        current = chunks.expanded_u(self.editor)
        bp = current[pred.hook][pred.expert].float()
        bs = current[succ.hook][succ.expert].float()
        mask = (self.col_expert[succ.hook] == succ.expert).float()
        read = (self.v0[succ.hook].float() * mask) @ _down(state[succ.hook]).float()
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
            directions = torch.cat([self.editor.per_hook(c.q)[node.hook] for c in active], dim=0)
            basis = directions.reshape(directions.shape[0], -1).T.float()
            coeff = torch.linalg.lstsq(basis, delta.reshape(-1, 1)).solution[:, 0]
            offset = 0
            for c in active:
                n = c.q.shape[0]
                key = (c.family, node.hook)
                if key not in chunks.node_coef:
                    chunks.node_coef[key] = torch.zeros_like(chunks.coef[c.family])
                chunks.node_coef[key] = chunks.node_coef[key].clone()
                chunks.node_coef[key][node.expert] += coeff[offset : offset + n].to(
                    chunks.node_coef[key]
                )
                offset += n
        after = chunks.expanded_u(self.editor)
        actual = (
            after[succ.hook][succ.expert].float() @ read @ after[pred.hook][pred.expert].float()
        )
        original = bs @ read @ bp
        residuals.append(float((actual - original).norm() / original.norm().clamp_min(1e-30)))
    rotation_seconds = stamp() - rotation_start
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


def reference_next_state(self, state, record):
    from kilora.recording import Perturbation

    if self.session is None:
        raise KiLoRAError("check() was not called")
    if record.measure is None or self._write_scale() <= 0:
        return None
    st = self.session.editor_state
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

    def loss(pert, time_, _seed):
        nonlocal measurement_index
        p = RoutedPerturbation(
            Perturbation(proj, {t: pert[t].float() for t in self.targets}),
            self.routes,
            getattr(self.session.mod.settings, "routing", "tensor"),
            self.tau,
            observed_select=getattr(self.session.mod.settings, "routing_select", "sparsemax"),
        )
        is_base = w.diff == "one-sided" and measurement_index == 0
        measurement_index += 1
        if merge_on and is_base:
            with recording_read_keep_raw(observe_keep):
                value = record.measure(p) if time_ is None else record.measure(p, time=time_)
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

    before = st.chunks.snapshot() if merge_on else None
    before_u = st.chunks.expanded_u(self.editor) if merge_on else None
    history = st.chunks.prior.get("merge_write_directions") if merge_on else None
    w.step(st.chunks, loss, shares, seed=0)
    if history is not None:
        st.chunks.prior["merge_write_directions"] = history
    if merge_on:
        after_write = st.chunks.snapshot()
        after_refs = (dict(st.chunks.coef), dict(st.chunks.node_coef))
        paths_before = len(st.forest.branches[st.forest.current].paths)
        after_u = st.chunks.expanded_u(self.editor)
        self._write_target = {h: after_u[h] - before_u[h] for h in self.targets}
        recording_start = time.perf_counter()
        for key, values in raw_keeps.items():
            keeps[key] = [
                v.float().reshape(-1, self.experts, v.shape[-1]).mean((0, 2)).cpu() for v in values
            ]
        measured_shares = shares()
        self.merge_recording_seconds = recording_time + time.perf_counter() - recording_start
        self._path_merge(st.chunks, state, measured_shares, keeps, frozen_keys)
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
    else:
        self.rotation_residual = self.rotation_residual_mean = 0.0
        log.debug("path rotation projection relative residual: max=0 mean=0 (steps=0)")
    return self._expanded_state(state)
