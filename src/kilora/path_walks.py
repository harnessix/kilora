# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F

from . import ablation_settings
from .adaptive_gate import base_reference, change_stats
from .compute import (
    PATH_SELECTS,
    _cache_get,
    _cache_put,
    _candidate_mask,
    _gate,
    _input_read,
    _learned_cross_logits,
    _mixed_route,
    _note_routes,
    _path_output,
    _set_keys,
    _tensor_key,
    norm_energy,
    out_project,
    region_mean,
    sparsemax,
    uses_adaptive,
)

_CONVERGED = 1.0e6


def check_path_select(select: str) -> str:
    if select not in PATH_SELECTS:
        raise ValueError(f"unknown path_select {select!r}, expected one of {PATH_SELECTS}")
    return select


PATH_STARTS = ("top", "diverse")


def check_path_starts(starts: str) -> str:
    if starts not in PATH_STARTS:
        raise ValueError(f"unknown path_starts {starts!r}, expected one of {PATH_STARTS}")
    return starts


def check_path_width(width: int) -> int:
    if isinstance(width, bool) or int(width) != width or int(width) < 1:
        raise ValueError(f"path_width must be a whole number >= 1, got {width!r}")
    return int(width)


def _scores(qs, lora, keys) -> list[torch.Tensor]:
    return [
        _learned_cross_logits(norm_energy(q.float(), kin, s.rows_in), s, q, kin, scale=1.0)
        for q, s, (kin, _) in zip(qs, lora, keys, strict=True)
    ]


def _candidate_scores(scores, qs, lora, keys, regions):
    ordered = []
    for score, query, s, (kin, _), region in zip(scores, qs, lora, keys, regions, strict=True):
        mask = _candidate_mask(s, query, kin, region)
        if mask is None:
            ordered.append(score)
            continue
        scoped = region_mean(score, region) if s.routing_scope == "tensor" else score
        ordered.append(scoped.masked_fill(~mask, -torch.inf).expand_as(score))
    return ordered


def _chosen_score(scores, gates) -> torch.Tensor:
    total = None
    for sc, g in zip(scores, gates, strict=True):
        g = g.detach()
        part = sc.gather(-1, g.argmax(-1, keepdim=True))[..., 0]
        part = part * (g.amax(-1) > 0)
        total = part if total is None else total + part
    return total


def transition_prior(s, keys=None) -> torch.Tensor:
    tensors = (
        [s.down, s.up] if s.form == "state" else [s.keys_in, s.rows_in, s.keys_out, s.rows_out]
    )
    key = _tensor_key("transition-prior", tensors)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    kin, kout = keys if keys is not None else _set_keys(s)
    with torch.no_grad():
        cross = torch.einsum("jar,ibr->ijab", kin.float(), kout.float())
        rows = (kout.float().norm(dim=-1) > 0).sum(-1).clamp(min=1).float()
        prior = cross.pow(2).sum((-1, -2)) / rows[:, None]
    _cache_put(key, prior)
    return prior


def _padded_keys(keys) -> tuple[torch.Tensor, torch.Tensor]:
    m = max(max(kin.shape[1], kout.shape[1]) for kin, kout in keys)

    def pad(t):
        return F.pad(t.float(), (0, 0, 0, m - t.shape[1]))

    return (
        torch.cat([pad(kin) for kin, _ in keys], 0),
        torch.cat([pad(kout) for _, kout in keys], 0),
    )


def joint_transition_prior(lora, keys=None) -> torch.Tensor:
    keys = keys if keys is not None else [_set_keys(s) for s in lora]
    kin, kout = _padded_keys(keys)
    with torch.no_grad():
        cross = torch.einsum("jar,ibr->ijab", kin, kout)
        rows = (kout.norm(dim=-1) > 0).sum(-1).clamp(min=1).float()
        return cross.pow(2).sum((-1, -2)) / rows[:, None]


def _output_overlap(s, keys) -> torch.Tensor:
    _, kout = keys
    cross = torch.einsum("jar,ibr->ijab", kout.float(), kout.float())
    rows = (kout.float().norm(dim=-1) > 0).sum(-1).clamp(min=1).float()
    return cross.pow(2).sum((-1, -2)) / rows[:, None]


def _one_hot(index: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(like).scatter(-1, index, 1.0)


def _start_orders(scores, lora, keys, starts: str, width: int) -> list[torch.Tensor]:
    out = []
    for sc, s, k in zip(scores, lora, keys, strict=True):
        sc = sc.detach()
        n = sc.shape[-1]
        if starts == "top" or n == 1:
            out.append(sc.argsort(-1, descending=True))
            continue
        overlap = _output_overlap(s, k).to(sc.device)
        picked = torch.zeros_like(sc, dtype=torch.bool)
        order = [sc.argmax(-1, keepdim=True)]
        picked.scatter_(-1, order[0], True)
        for _ in range(1, n):
            pf = picked.float()
            near = (pf[..., :, None] * overlap).amax(-2)
            near = near.masked_fill(picked | ~torch.isfinite(sc), float("inf"))
            least = near.amin(-1, keepdim=True)
            ok = (near <= least + 1e-6) & ~picked
            nxt = sc.masked_fill(~ok, float("-inf")).argmax(-1, keepdim=True)
            order.append(nxt)
            picked.scatter_(-1, nxt, True)
        out.append(torch.cat(order, -1))
    return out


def _forced(orders, rank, scores) -> list[torch.Tensor]:
    gates = []
    for order, sc in zip(orders, scores, strict=True):
        if isinstance(rank, int):
            r = torch.full(order.shape[:-1] + (1,), min(rank, order.shape[-1] - 1))
            r = r.to(order.device)
        else:
            r = rank.clamp(max=order.shape[-1] - 1)[..., None]
        gates.append(_one_hot(order.gather(-1, r), sc))
    return gates


def _pruned_gates(qs, lora, keys, scores, nodes, threshold, noise_scale, regions):
    gates = []
    for s, sc, k, node, r, q in zip(lora, scores, keys, nodes, regions, qs, strict=True):
        prior = transition_prior(s, k).to(sc.device)
        allowed = prior[node] >= threshold
        allowed = allowed | ~allowed.any(-1, keepdim=True)
        logits = (s.tau * sc).masked_fill(~allowed, -1e9)
        mask = _candidate_mask(s, q, k[0], r)
        gates.append(_gate(logits, s, noise_scale, r, mask))
    return gates


def _set_nodes(gates) -> list[torch.Tensor]:
    return [g.detach().argmax(-1) for g in gates]


def _walk_step(st, gates, scores, c0, lora, keys, qs, keep, fixed, dtype, k):
    new = []
    for s, gate, c in zip(lora, gates, c0, strict=True):
        weight = gate.to(dtype).repeat_interleave(s.per_expert, dim=-1)
        scale = s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale
        z = c * weight * scale
        new.append(z if keep else z.detach())
    node = torch.cat(gates, -1).detach().argmax(-1)
    gained = _chosen_score(scores, gates).detach()
    st["score"] = gained if st["score"] is None else st["score"] + gained
    exact_gates = gates
    if st["node"] is not None:
        same = node == st["node"]
        st["converged"] = same if st["converged"] is None else st["converged"] | same
        if fixed:
            done = same.all(-1)
            st["done"] = done if st["done"] is None else st["done"] | done
            hold = st["done"][..., None, None]
            new = [torch.where(hold, o, z) for o, z in zip(st["zs"], new, strict=True)]
            if any(s.exact for s in lora):
                exact_gates = [
                    torch.where(hold, old, gate)
                    for old, gate in zip(st.get("exact_gates", gates), gates, strict=True)
                ]
    if k == 1:
        st["first_zs"] = [z.detach() for z in new]
    st["prev_zs"], st["zs"], st["node"] = st["zs"], new, node
    st["nodes"] = _set_nodes(gates)
    st["gates"], st["k"] = gates, k
    if any(s.exact for s in lora):
        st["exact_gates"] = exact_gates
    st["qs"] = [out_project(q, kout, g) for q, (_, kout), g in zip(qs, keys, gates, strict=True)]


def _merge(state, q0, orders, next_rank, total):
    width = len(state)
    qcat = [torch.cat([q.float() for q in st["qs"]], -1) for st in state]
    restart = [torch.zeros_like(state[0]["node"], dtype=torch.bool) for _ in state]
    for w in range(1, width):
        for v in range(w):
            a, b = state[w], state[v]
            same = torch.stack(
                [x == y for x, y in zip(a["nodes"], b["nodes"], strict=True)], -1
            ).all(-1)
            dup = same & (F.cosine_similarity(qcat[w], qcat[v], dim=-1) > 0.999)
            lower = a["score"] <= b["score"]
            restart[w] = restart[w] | (dup & lower)
            restart[v] = restart[v] | (dup & ~lower)
    for st, r in zip(state, restart, strict=True):
        r = r & (next_rank < total)
        st["restart"] = r
        if not bool(r.any()):
            continue
        st["restart_rank"] = next_rank.clone()
        next_rank = torch.where(r, next_rank + 1, next_rank)
        st["qs"] = [torch.where(r[..., None], q0i, q) for q0i, q in zip(q0, st["qs"], strict=True)]
        st["score"] = torch.where(r, torch.zeros_like(st["score"]), st["score"])
        if st["converged"] is not None:
            st["converged"] = st["converged"] & ~r
        st["node"] = torch.where(r, torch.full_like(st["node"], -1), st["node"])
    return next_rank


def _beam_step(state, lora, keys, c0, keep, fixed, dtype, k, shared, noise_scale, regions):
    s = lora[0]
    width = len(state)
    qs = [st["qs"][0] for st in state]
    scores = [_scores([q], lora, keys)[0] for q in qs]
    gates0 = _mixed_route([qs[0]], lora, keys, None, shared, noise_scale, regions)
    ranked = [
        _candidate_scores([sc], [q], lora, keys, regions)[0]
        for sc, q in zip(scores, qs, strict=True)
    ]
    cand = torch.stack(
        [st["score"][..., None] + sc for st, sc in zip(state, ranked, strict=True)], -2
    )
    if s.routing_scope == "tensor" and 0 < s.candidates < scores[0].shape[-1]:
        cand = torch.stack(
            [
                region_mean(st["score"][..., None], regions[0]) + sc
                for st, sc in zip(state, ranked, strict=True)
            ],
            -2,
        )
    experts = cand.shape[-1]
    flat = cand.flatten(-2)
    first = (
        gates0[0].detach().argmax(-1, keepdim=True)
        if 0 < s.candidates < scores[0].shape[-1]
        else scores[0].detach().argmax(-1, keepdim=True)
    )
    flat = flat.scatter(-1, first, float("-inf"))
    best = flat.topk(width - 1, dim=-1).indices if width > 1 else None
    old = [dict(st) for st in state]
    for w in range(width):
        st = state[w]
        if w == 0:
            _walk_step(st, gates0, [scores[0]], c0, lora, keys, [qs[0]], keep, fixed, dtype, k)
            continue
        pick = best[..., w - 1 : w]
        parent, expert = pick // experts, pick % experts

        def take(values, parent=parent):
            stacked = torch.stack(values, -2)
            idx = parent[..., None].expand(*parent.shape[:-1], 1, stacked.shape[-1])
            return stacked.gather(-2, idx)[..., 0, :]

        q_par = take([o["qs"][0] for o in old])
        sc_par = take(scores)
        st["score"] = take([o["score"][..., None] for o in old])[..., 0]
        st["node"] = take([o["node"][..., None].float() for o in old])[..., 0].long()
        if old[0]["converged"] is not None:
            conv = [o["converged"][..., None].float() for o in old]
            st["converged"] = take(conv)[..., 0].bool()
        st["zs"] = [take([o["zs"][0] for o in old])]
        gate = [_one_hot(expert, sc_par)]
        _walk_step(st, gate, [sc_par], c0, lora, keys, [q_par], keep, fixed, dtype, k)


def _joint_scores(q, keys, lora) -> torch.Tensor:
    return torch.cat(
        _scores([q] * len(lora), lora, keys),
        -1,
    )


def _node_gates(node, sizes, like) -> list[torch.Tensor]:
    gates, off = [], 0
    for n, ref in zip(sizes, like, strict=True):
        inside = (node >= off) & (node < off + n)
        local = (node - off).clamp(0, n - 1)
        gates.append(_one_hot(local, ref) * inside.to(ref.dtype))
        off += n
    return gates


def _project_node(q, node, keys, sizes) -> torch.Tensor:
    kout = _padded_keys(keys)[1]
    k = kout[node[..., 0]]
    return torch.einsum("...mr,...m->...r", k, torch.einsum("...mr,...r->...m", k, q.float()))


def _joint_beam_step(state, lora, keys, c0, keep, fixed, dtype, k, shared, regions, sizes):
    width = len(state)
    q0_shared = state[0]["shared_q"] if k == 1 else sum(q.float() for q in state[0]["qs"])
    q_of = [q0_shared] + [st["shared_q"] for st in state[1:]]
    split = _scores(state[0]["qs"], lora, keys)
    scores = [torch.cat(split, -1)] + [_joint_scores(q, keys, lora) for q in q_of[1:]]
    ranked = []
    for j, q in enumerate(q_of):
        qs_for_score = state[0]["qs"] if j == 0 else [q] * len(lora)
        pieces = list(torch.split(scores[j], sizes, -1))
        ranked.append(torch.cat(_candidate_scores(pieces, qs_for_score, lora, keys, regions), -1))
    old = [dict(st) for st in state]
    gates0 = _mixed_route(state[0]["qs"], lora, keys, None, shared, 0.0, regions)
    _walk_step(state[0], gates0, split, c0, lora, keys, state[0]["qs"], keep, fixed, dtype, k)
    if width == 1:
        return
    first = torch.cat(gates0, -1).detach().argmax(-1, keepdim=True)
    n = scores[0].shape[-1]
    if k == 1:
        flat = ranked[0].scatter(-1, first, float("-inf"))
        picks = flat.topk(min(width - 1, n - 1), dim=-1).indices
        parents = torch.zeros_like(picks)
        nodes = picks
        base = [torch.zeros_like(scores[0][..., 0]) for _ in range(width)]
    else:
        cand = torch.stack(
            [o["score"][..., None] + sc for o, sc in zip(old, ranked, strict=True)], -2
        )
        flat = cand.flatten(-2).scatter(-1, first, float("-inf"))
        best = flat.topk(width - 1, dim=-1).indices
        parents, nodes = best // n, best % n
        base = None
    for w in range(1, width):
        st = state[w]
        r = min(w - 1, nodes.shape[-1] - 1)
        node = nodes[..., r : r + 1]
        parent = parents[..., r : r + 1]

        def take(values, parent=parent):
            stacked = torch.stack(values, -2)
            idx = parent[..., None].expand(*parent.shape[:-1], 1, stacked.shape[-1])
            return stacked.gather(-2, idx)[..., 0, :]

        q_par = take([q.float() for q in q_of])
        sc_par = take(scores)
        if base is None:
            st["score"] = take([o["score"][..., None] for o in old])[..., 0]
            st["node"] = take([o["node"][..., None].float() for o in old])[..., 0].long()
            if old[0]["converged"] is not None:
                st["converged"] = take([o["converged"][..., None].float() for o in old])[
                    ..., 0
                ].bool()
            st["zs"] = [take([o["zs"][i] for o in old]) for i in range(len(lora))]
        gates = _node_gates(node, sizes, [torch.zeros_like(x) for x in split])
        qs = [q_par for _ in lora]
        if any(0 < s.candidates < n for s, n in zip(lora, sizes, strict=True)):
            masks = [
                _candidate_mask(s, q_par, kin, r)
                for s, (kin, _), r in zip(lora, keys, regions, strict=True)
            ]
            invalid = torch.zeros_like(node, dtype=torch.bool)
            for gate, mask in zip(gates, masks, strict=True):
                if mask is not None:
                    invalid |= (gate.bool() & ~mask).any(-1, keepdim=True)
            if invalid.any():
                pieces = _scores(qs, lora, keys)
                ranked_q = _candidate_scores(pieces, qs, lora, keys, regions)
                replacement = torch.cat(ranked_q, -1).argmax(-1, keepdim=True)
                node = torch.where(invalid, replacement, node)
                gates = _node_gates(node, sizes, [torch.zeros_like(v) for v in split])
        _walk_step(
            st,
            gates,
            list(torch.split(sc_par, sizes, -1)),
            c0,
            lora,
            keys,
            qs,
            keep,
            fixed,
            dtype,
            k,
        )
        st["shared_q"] = _project_node(q_par, node, keys, sizes)


def _derived_walks(
    y0, lora, depth, noise_scale, grad_steps, target, regions, x, width, shared=False
):
    from . import compute as c

    width = min(
        width,
        min((s.candidates for s in lora if 0 < s.candidates < s.keys_in.shape[0]), default=width),
    )
    keys = [
        (c.derived_keys(s)[0], c.derived_keys(s)[2])
        if c.import_score(s) == "derived"
        else c._set_keys(s)
        for s in lora
    ]
    joint_choice = shared or any(s.form == "input" for s in lora)
    records = []
    for w in range(width):
        record = {
            "zs": [],
            "prev_zs": [],
            "gates": None,
            "node": None,
            "nodes": None,
            "converged": None,
            "score": None,
            "k": 0,
        }

        def queries(state, _record=record):
            result = []
            for index, setting in enumerate(lora):
                if setting.form == "state":
                    query = state
                elif c.import_score(setting) == "derived":
                    query = x
                else:
                    query = F.linear(state.float(), setting.router.float())
                    if c.import_score(setting) == "keys" and _record["gates"] is not None:
                        query = c.out_project(query, keys[index][1], _record["gates"][index])
                result.append(query)
            return result

        def scores(state, _queries=queries):
            result = []
            for setting, query, (basis, _) in zip(lora, _queries(state), keys, strict=True):
                mode = c.import_score(setting)
                if mode == "derived":
                    score = c.derived_logits(setting, query)
                elif mode == "read_share":
                    read = c._input_read(setting, x, x.dtype)
                    energy = c.read_energy(setting, read)
                    score = setting.tau * energy / energy.sum(-1, keepdim=True).clamp_min(1e-12)
                else:
                    score = setting.tau * c.norm_energy(query, basis, setting.rows_in)
                if setting.form == "state" or mode == "keys":
                    score = c._cross_logits(score, setting, basis)
                    score = c._learned_cross_logits(score, setting, query, basis)
                result.append(score / setting.tau)
            return result

        def region_orders(orders, values, joint=False):
            if not any(s.expert_mask is not None for s in lora):
                return orders
            allowed = [
                c._mask_derived_logits(s, torch.zeros_like(value)).eq(0)
                if c.import_score(s) == "derived"
                else torch.ones_like(value, dtype=torch.bool)
                for s, value in zip(lora, values, strict=True)
            ]
            if joint:
                allowed = [torch.cat(allowed, -1)]
                values = [torch.cat(values, -1)]
            result = []
            for order, valid, value in zip(orders, allowed, values, strict=True):
                best = value.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
                result.append(torch.where(valid.gather(-1, order), order, best))
            return result

        def choose(state, gates, k, _walk=w, _record=record, _scores=scores, _queries=queries):
            if k == 1 and _walk > 0:
                values = _scores(state)
                ranked = _candidate_scores(values, _queries(state), lora, keys, regions)
                if not joint_choice:
                    orders = _start_orders(ranked, lora, keys, lora[0].path_starts, width)
                    return _forced(region_orders(orders, ranked), _walk, values)
                parts = []
                for score, s in zip(ranked, lora, strict=True):
                    prior = s.log_prior if len(lora) > 1 else 0.0
                    if len(lora) > 1 and os.environ.get("KILORA_FILE_BALANCE", "1") != "0":
                        prior = prior + 1.0 / (len(lora) * score.shape[-1])
                    parts.append(s.tau * score + prior)
                joint = torch.cat(parts, -1)
                output_keys = [c.derived_keys(s)[2] for s in lora]
                outputs = _padded_keys([(kout, kout) for kout in output_keys])
                orders = _start_orders([joint], [lora[0]], [outputs], lora[0].path_starts, width)
                orders = region_orders(orders, parts, joint=True)
                forced = _forced(orders, _walk, [joint])[0]
                return list(forced.split([v.shape[-1] for v in values], -1))
            if k > 1 and lora[0].path_prior_min > 0:
                logits = []
                for score, s, key, node in zip(
                    _scores(state), lora, keys, _record["nodes"], strict=True
                ):
                    allowed = transition_prior(s, key).to(score.device)[node] >= s.path_prior_min
                    allowed = allowed | ~allowed.any(-1, keepdim=True)
                    value = s.tau * score
                    if (
                        joint_choice
                        and len(lora) > 1
                        and os.environ.get("KILORA_FILE_BALANCE", "1") != "0"
                    ):
                        value = value + 1.0 / (len(lora) * value.shape[-1])
                    logits.append(value.masked_fill(~allowed, -1e9))
                masks = [
                    c._candidate_mask(s, query, key[0], region)
                    for s, query, key, region in zip(
                        lora, _queries(state), keys, regions, strict=True
                    )
                ]
                return c._gates_from_logits(logits, lora, joint_choice, noise_scale, regions, masks)
            return gates

        def capture(state, gates, zs, output, k, _record=record, _scores=scores):
            node = (
                torch.cat(gates, -1).detach().argmax(-1)
                if joint_choice
                else torch.stack(_set_nodes(gates), -1)
            )
            if _record["node"] is not None:
                same = node == _record["node"]
                if not joint_choice:
                    same = same.all(-1)
                _record["converged"] = same
            gained = _chosen_score(_scores(state), gates).detach()
            _record["score"] = gained if _record["score"] is None else _record["score"] + gained
            _record["prev_zs"], _record["zs"] = _record["zs"], zs
            if k == 1:
                _record["first_zs"] = [z.detach() for z in zs]
            _record.update(gates=gates, node=node, nodes=_set_nodes(gates), k=k, output=output)

        name = target if w == 0 else f"{target}#walk{w}"
        c._run_input(
            x,
            y0,
            lora,
            list(range(len(lora))),
            depth,
            regions,
            name,
            stats={"filter_routes": choose, "capture_step": capture},
            noise_scale=noise_scale,
            grad_steps=grad_steps,
            shared=shared,
        )
        if record["converged"] is None:
            record["converged"] = torch.zeros_like(record["score"], dtype=torch.bool)
        records.append(record)
    frozen = grad_steps is not None and not grad_steps(depth)
    return records, frozen


def walks(y0, lora, depth, shared, noise_scale, grad_steps, target, regions, stop, x, width):
    from . import compute as c

    if c._derived_state_sets(lora):
        return _derived_walks(
            y0, lora, depth, noise_scale, grad_steps, target, regions, x, width, shared
        )
    dtype = y0.dtype
    keys = [_set_keys(s) for s in lora]
    c0 = [
        F.linear(y0.to(dtype), s.down.to(dtype)) if s.form == "state" else _input_read(s, x, dtype)
        for s in lora
    ]
    q0 = [F.linear(y0.float(), s.router.float()) for s in lora]
    fixed = lora[0].fixed_point_stop if stop is None else stop
    opts = lora[0]
    starts = check_path_starts(opts.path_starts)
    threshold = float(opts.path_prior_min)
    beam = bool(opts.path_beam) and len(lora) == 1
    joint = (
        bool(opts.path_beam)
        and len(lora) > 1
        and shared
        and all(s.form == "state" for s in lora)
        and all(s.router.shape == lora[0].router.shape for s in lora)
        and all(torch.equal(s.router, lora[0].router) for s in lora[1:])
    )
    sizes = [kin.shape[0] for kin, _ in keys]
    merge = bool(opts.path_merge)
    state = [
        {
            "qs": list(q0),
            "zs": [],
            "prev_zs": [],
            "node": None,
            "done": None,
            "gates": None,
            "k": 0,
            "converged": None,
            "score": None,
            "nodes": None,
            "restart": None,
        }
        for _ in range(width)
    ]
    if joint:
        for st in state:
            st["shared_q"] = q0[0]
    orders = None
    next_rank = None
    frozen = False
    for k in range(1, depth + 1):
        keep = grad_steps is None or grad_steps(k)
        with contextlib.nullcontext() if keep else torch.no_grad():
            if joint:
                _joint_beam_step(
                    state, lora, keys, c0, keep, fixed, dtype, k, shared, regions, sizes
                )
                frozen = not keep
                continue
            if beam and k > 1:
                if frozen:
                    for st in state:
                        st["qs"] = [q.detach() for q in st["qs"]]
                _beam_step(state, lora, keys, c0, keep, fixed, dtype, k, shared, 0.0, regions)
                frozen = not keep
                continue
            for w, st in enumerate(state):
                if st["done"] is not None and bool(st["done"].all()):
                    continue
                qs = [q.detach() for q in st["qs"]] if frozen else st["qs"]
                scores = _scores(qs, lora, keys)
                if k == 1 and w > 0:
                    ranked = _candidate_scores(scores, qs, lora, keys, regions)
                    if starts == "top" and not merge:
                        gates = []
                        for sc in ranked:
                            rank = min(w, sc.shape[-1] - 1)
                            top = sc.detach().topk(rank + 1, dim=-1).indices[..., rank : rank + 1]
                            gates.append(torch.zeros_like(sc).scatter(-1, top, 1.0))
                    else:
                        if orders is None:
                            orders = _start_orders(ranked, lora, keys, starts, width)
                        gates = _forced(orders, w, scores)
                elif threshold > 0 and st["nodes"] is not None:
                    gates = _pruned_gates(
                        qs, lora, keys, scores, st["nodes"], threshold, noise_scale, regions
                    )
                else:
                    gates = _mixed_route(qs, lora, keys, None, shared, noise_scale, regions)
                restart = st.get("restart")
                if restart is not None and bool(restart.any()):
                    forced = _forced(orders, st["restart_rank"], scores)
                    gates = [
                        torch.where(restart[..., None], f, g)
                        for f, g in zip(forced, gates, strict=True)
                    ]
                    st["restart"] = None
                if w == 0:
                    _note_routes(target, gates, lora)
                _walk_step(st, gates, scores, c0, lora, keys, qs, keep, fixed, dtype, k)
            if merge and width > 1:
                if orders is None:
                    scores0 = _candidate_scores(_scores(q0, lora, keys), q0, lora, keys, regions)
                    orders = _start_orders(scores0, lora, keys, starts, width)
                if next_rank is None:
                    next_rank = torch.full_like(state[0]["node"], width)
                total = min(
                    min(o.shape[-1] for o in orders),
                    min(
                        (s.candidates for s in lora if 0 < s.candidates < s.keys_in.shape[0]),
                        default=10**9,
                    ),
                )
                next_rank = _merge(state, q0, orders, next_rank, total)
        frozen = not keep
        if all(st["done"] is not None and bool(st["done"].all()) for st in state):
            break
    for st in state:
        if st["converged"] is None:
            st["converged"] = torch.zeros_like(st["score"], dtype=torch.bool)
    return state, frozen


def select_weights(state: list[dict], select: str) -> torch.Tensor:
    check_path_select(select)
    score = torch.stack([st["score"].float() for st in state], -1)
    if select == "mix":
        return sparsemax(score)
    if select == "converged":
        conv = torch.stack([st["converged"] for st in state], -1)
        score = score + _CONVERGED * conv.float()
    best = score.argmax(-1, keepdim=True)
    return torch.zeros_like(score).scatter(-1, best, 1.0)


def _combine(parts: list, weights: torch.Tensor) -> torch.Tensor:
    out = None
    for w, p in enumerate(parts):
        term = weights[..., w : w + 1].to(p.dtype) * p
        out = term if out is None else out + term
    return out


def _batched_ok(lora, stop, grad_steps) -> bool:
    if ablation_settings.force_loop() or len(lora) != 1 or grad_steps is not None:
        return False
    if any(0 < s.candidates < s.keys_in.shape[0] for s in lora):
        return False
    if any(s.learned_cross is not None and not s.learned_cross_zero for s in lora):
        return False
    s = lora[0]
    fixed = s.fixed_point_stop if stop is None else stop
    options = (s.path_starts, s.path_prior_min, s.path_merge, s.path_beam)
    if fixed or options != ("top", 0.0, False, False):
        return False
    tensors = [s.down, s.up, s.router, s.keys_in, s.keys_out]
    return not (torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors))


def walks_batched(y0, lora, depth, shared, noise_scale, target, regions, x, width):
    s = lora[0]
    dtype = y0.dtype
    keys = [_set_keys(s)]
    kin, kout = keys[0]
    c0 = F.linear(y0.to(dtype), s.down.to(dtype)) if s.form == "state" else _input_read(s, x, dtype)
    scale = s.scale.to(dtype) if isinstance(s.scale, torch.Tensor) else s.scale
    q = F.linear(y0.float(), s.router.float())
    first_zs = None
    zs = prev = gates = node = score = conv = None
    k_last = 0
    for k in range(1, depth + 1):
        if k == 1:
            sc = norm_energy(q, kin, s.rows_in)
            first = _mixed_route([q], lora, keys, None, shared, noise_scale, regions)[0]
            order = sc.topk(min(width, sc.shape[-1]), dim=-1).indices
            others = []
            for w in range(1, width):
                r = min(w, sc.shape[-1] - 1)
                others.append(torch.zeros_like(sc).scatter(-1, order[..., r : r + 1], 1.0))
            g = torch.stack([first, *others])
            q = q.expand(width, *q.shape)
            sc = sc.expand(width, *sc.shape)
        else:
            sc = norm_energy(q, kin, s.rows_in)
            g = _mixed_route([q], lora, keys, None, shared, noise_scale, regions)[0]
        _note_routes(target, [g[0]], lora)
        weight = g.to(dtype).repeat_interleave(s.per_expert, dim=-1)
        z = c0 * weight * scale
        if k == 1:
            first_zs = z.detach()
        n = g.detach().argmax(-1)
        gained = sc.gather(-1, n[..., None])[..., 0] * (g.detach().amax(-1) > 0)
        score = gained if score is None else score + gained
        if node is not None:
            same = n == node
            conv = same if conv is None else conv | same
        prev, zs, node, gates, k_last = zs, z, n, g, k
        q = out_project(q, kout, g)
    conv = conv if conv is not None else torch.zeros_like(score, dtype=torch.bool)
    return [
        {
            "zs": [zs[w]],
            "first_zs": [first_zs[w]],
            "prev_zs": [prev[w]] if prev is not None else [],
            "gates": [gates[w]],
            "k": k_last,
            "converged": conv[w],
            "score": score[w],
        }
        for w in range(width)
    ]


def _first_gate_histories(lora, state, weights):
    if base_reference():
        return None
    histories = []
    for i, s in enumerate(lora):
        hist = {}
        if uses_adaptive(s):
            first = _combine([st["first_zs"][i] for st in state], weights)
            up = s.up.detach().float()
            hist["first_norm"], hist["first_rms"] = change_stats(first, up, up @ up.t())
        histories.append(hist)
    return histories


def run_wide(y0, lora, depth, shared, noise_scale, grad_steps, target, regions, stop, x):
    regions = regions if regions is not None else [None] * len(lora)
    width = check_path_width(lora[0].path_width)
    from . import compute as c

    if c._derived_state_sets(lora):
        if (
            lora[0].path_beam
            and len(lora) > 1
            and not shared
            and all(s.form == "state" for s in lora)
        ):
            result = y0
            for s, region in zip(lora, regions, strict=True):
                single = run_wide(
                    y0, [s], depth, False, noise_scale, grad_steps, target, [region], stop, x
                )
                result = result + (single - y0)
            return result
        state, frozen = _derived_walks(
            y0, lora, depth, noise_scale, grad_steps, target, regions, x, width, shared
        )
        weights = select_weights(state, lora[0].path_select)
        result = _combine([st["output"] for st in state], weights)
        return result.detach() if frozen else result
    width = min(
        width,
        min((s.candidates for s in lora if 0 < s.candidates < s.keys_in.shape[0]), default=width),
    )
    if lora[0].path_beam and len(lora) > 1 and not shared:
        return _run_per_set(y0, lora, depth, noise_scale, grad_steps, target, regions, stop, x)
    if _batched_ok(lora, stop, grad_steps) and os.environ.get("KILORA_WALKS", "") != "loop":
        state = walks_batched(y0, lora, depth, shared, noise_scale, target, regions, x, width)
        frozen = False
    else:
        state, frozen = walks(
            y0, lora, depth, shared, noise_scale, grad_steps, target, regions, stop, x, width
        )
    weights = select_weights(state, lora[0].path_select)
    n = len(lora)
    zs = [_combine([st["zs"][i] for st in state], weights) for i in range(n)]
    gates = [_combine([st["gates"][i] for st in state], weights) for i in range(n)]
    exact_gates = (
        [
            _combine([st.get("exact_gates", st["gates"])[i] for st in state], weights)
            for i in range(n)
        ]
        if any(s.exact for s in lora)
        else None
    )
    prev_zs: list = []
    if all(st["prev_zs"] for st in state):
        prev_zs = [_combine([st["prev_zs"][i] for st in state], weights) for i in range(n)]
    last_k = max(st["k"] for st in state)
    y = _path_output(
        y0,
        lora,
        zs,
        prev_zs,
        gates,
        last_k,
        regions,
        target,
        head_hists=_first_gate_histories(lora, state, weights),
        x=x,
        exact_gates=exact_gates,
    )
    return y.detach() if frozen else y


def _run_per_set(y0, lora, depth, noise_scale, grad_steps, target, regions, stop, x):
    zs, gates, exact_gates, prev_zs, ks, frozen = [], [], [], [], [], False
    histories = []
    width = check_path_width(lora[0].path_width)
    for i, s in enumerate(lora):
        set_width = min(width, s.candidates) if 0 < s.candidates < s.keys_in.shape[0] else width
        state, fz = walks(
            y0,
            [s],
            depth,
            False,
            noise_scale,
            grad_steps,
            f"{target}",
            [regions[i]],
            stop,
            x,
            set_width,
        )
        frozen = frozen or fz
        weights = select_weights(state, s.path_select)
        histories.extend(_first_gate_histories([s], state, weights) or [{}])
        zs.append(_combine([st["zs"][0] for st in state], weights))
        gates.append(_combine([st["gates"][0] for st in state], weights))
        if s.exact:
            exact_gates.append(
                _combine([st.get("exact_gates", st["gates"])[0] for st in state], weights)
            )
        else:
            exact_gates.append(gates[-1])
        if all(st["prev_zs"] for st in state):
            prev_zs.append(_combine([st["prev_zs"][0] for st in state], weights))
        ks.append(max(st["k"] for st in state))
    prev = prev_zs if len(prev_zs) == len(lora) else []
    y = _path_output(
        y0,
        lora,
        zs,
        prev,
        gates,
        max(ks),
        regions,
        target,
        head_hists=histories,
        x=x,
        exact_gates=exact_gates,
    )
    return y.detach() if frozen else y
