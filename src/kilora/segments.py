# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import os

import torch

from . import compute


def discover(
    s,
    *,
    limit,
    min_similarity=0.5,
    max_columns=None,
    host_norm=None,
    min_triple=0.05,
    epsilon=0.1,
):

    if s.form != "input" or s.down.shape[1] != s.up.shape[1] or limit <= 0:
        return []
    key = compute._tensor_key(
        f"segments-{s.per_expert}-{limit}-{min_similarity}-{max_columns}-"
        f"{host_norm}-{min_triple}-{epsilon}-"
        f"{os.environ.get('KILORA_DERIVED_ROWS', '8')}",
        [s.down, s.up],
    )
    cached = compute._cache_get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        kin, rin, kout, rout = compute.derived_keys(s)[:4]
        count = len(kin)
        down = s.down.detach().float().reshape(count, s.per_expert, -1)
        up = s.up.detach().float().reshape(count, s.per_expert, -1)
        aa = down @ down.transpose(-1, -2)
        bb = up @ up.transpose(-1, -2)
        norms = (aa * bb).sum((-1, -2)).clamp_min(0).sqrt()
        strength = down.new_zeros((count, count))
        for a in range(count):
            for b in range(count):
                m = down[b] @ up[a].T
                strength[a, b] = torch.trace(m.T @ bb[b] @ m @ aa[a]).clamp_min(0).sqrt() / (
                    norms[a] * norms[b]
                ).clamp_min(1e-30)
        strength /= strength.sum(-1, keepdim=True).clamp_min(1e-30)

        def similarities(bases, rows):
            out = down.new_zeros((count, count))
            for a in range(count):
                for b in range(count):
                    na, nb = int(rows[a]), int(rows[b])
                    if na and nb:
                        out[a, b] = (bases[a, :na] @ bases[b, :nb].T).square().sum() / min(na, nb)
            return out

        si, so = similarities(kin, rin), similarities(kout, rout)
        allowed = None
        if host_norm is not None:
            allowed, _ = operator_decisions(s, host_norm, min_triple=min_triple, epsilon=epsilon)
        candidates = []
        for a in range(count):
            for b in range(count):
                if a == b or strength[a, b] <= 0:
                    continue
                similar = ((si[a] + so[b]) * 0.5) >= min_similarity
                if allowed is not None:
                    decision = allowed.get((a, b)) if isinstance(allowed, dict) else allowed[a, b]
                    if decision is None:
                        continue
                    similar = similar & decision
                if similar.any():
                    candidates.append((float(strength[a, b]), a, b))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        used = s.down.shape[0]
        result = []
        for _, a, b in candidates:
            cost = 2 * s.per_expert
            if max_columns is not None and used + cost > max_columns:
                continue
            result.append((a, b))
            used += cost
            if len(result) == limit:
                break
        compute._cache_put(key, result)
        return result


def fold(s, a, b):
    r = s.per_expert
    return (
        torch.cat((s.down[a * r : (a + 1) * r], s.down[b * r : (b + 1) * r])),
        torch.cat((s.up[a * r : (a + 1) * r], s.up[b * r : (b + 1) * r])),
    )


def entry_logits(s, x, pairs):
    base = compute.derived_logits(s, x)
    if not pairs:
        return base
    m = int(os.environ.get("KILORA_DERIVED_ROWS", "8"))
    key = compute._tensor_key(f"segment-bases-{s.per_expert}-{m}-{pairs}", [s.down])
    cached = compute._cache_get(key)
    if cached is not None:
        bases, rows = cached
        extra = s.tau * compute.norm_energy(x.float(), bases, rows)
        return torch.cat((base, extra), dim=-1)
    bases = []
    counts = []
    width = min(m, 2 * s.per_expert, s.down.shape[-1])
    for a, b in pairs:
        down, _ = fold(s, a, b)
        _, values, vh = torch.linalg.svd(down.detach().float(), full_matrices=False)
        tol = values.max() * max(down.shape) * torch.finfo(values.dtype).eps
        count = min(width, int((values > tol).sum().item()))
        basis = down.new_zeros((width, down.shape[-1]), dtype=torch.float32)
        basis[:count] = vh[:count]
        bases.append(basis)
        counts.append(count)
    rows = torch.tensor(counts, device=x.device).clamp_min(1)
    bases = torch.stack(bases)
    compute._cache_put(key, (bases, rows))
    extra = s.tau * compute.norm_energy(x.float(), bases, rows)
    return torch.cat((base, extra), dim=-1)


def prepare(s):
    count = s.down.shape[0] // s.per_expert
    limit = int(os.environ.get("KILORA_SEGMENTS", str(count)))
    minimum = float(os.environ.get("KILORA_SEGMENT_MIN_SIM", "0.5"))
    fused = compute._fused_module()
    budget = (
        None if fused is None else fused.max_columns(s.router.shape[-1], s.down.device, tiled=True)
    )
    if (
        s.host_weight_norm is None
        or not math.isfinite(s.host_weight_norm)
        or s.host_weight_norm <= 0
    ):
        return []
    pairs = discover(
        s,
        limit=limit,
        min_similarity=minimum,
        max_columns=budget,
        host_norm=s.host_weight_norm,
        min_triple=float(os.environ.get("KILORA_SEGMENT_MIN_TRIPLE", "0.05")),
        epsilon=float(os.environ.get("KILORA_SEGMENT_EPS", "0.1")),
    )
    if pairs:
        entry_logits(s, s.down.new_zeros((1, s.down.shape[-1])), pairs)
    return pairs


def operator_metrics(s, host_norm):

    if not isinstance(host_norm, (float, int)) or not 0 < host_norm < float("inf"):
        raise ValueError("host_norm must be finite and positive")
    with torch.no_grad():
        count = s.down.shape[0] // s.per_expert
        down = s.down.detach().float().reshape(count, s.per_expert, -1)
        up = s.up.detach().float().reshape(count, s.per_expert, -1)
        if down.shape[-1] != up.shape[-1]:
            raise ValueError("operator metrics require square layers")
        aa = down @ down.transpose(-1, -2)
        bb = up @ up.transpose(-1, -2)
        norms = (aa * bb).sum((-1, -2)).clamp_min(0).sqrt()
        match = down.new_zeros((count, count))
        triple = down.new_zeros((count, count, count)) if count <= 10 else {}
        for a in range(count):
            for b in range(count):
                m = down[b] @ up[a].T
                denominator = norms[a] * host_norm
                value = torch.trace(m.T @ bb[b] @ m @ aa[a]).clamp_min(0).sqrt()
                match[a, b] = value / denominator.clamp_min(1e-30)
        if count <= 10:
            pairs = [(a, b) for a in range(count) for b in range(count)]
        else:
            pair_strength = match / (norms / host_norm).clamp_min(1e-30)[None, :]
            indices = pair_strength.flatten().topk(min(10, count * count)).indices.tolist()
            pairs = [(i // count, i % count) for i in indices]
        for a, b in pairs:
            values = []
            for c in range(count):
                value = torch.trace((down[c] @ up[b].T) @ (down[b] @ up[a].T) @ (down[a] @ up[c].T))
                values.append(value / (norms[a] * norms[b] * norms[c]).clamp_min(1e-30))
            triple[a, b] = torch.stack(values)
        return norms, norms / host_norm, match, triple


def shortcut_residual(match, gain, a, b, c):
    if gain[b] <= 0:
        return match.new_tensor(float("inf"))
    return (match[a, c] - match[a, b] * match[b, c] / gain[b]).abs()


def operator_decisions(s, host_norm, *, skip_gain=0.01, min_triple=0.05, epsilon=0.1):

    if not 0 <= min_triple < float("inf") or not 0 <= epsilon < float("inf"):
        raise ValueError("segment triple and epsilon must be finite and nonnegative")
    if not 0 <= skip_gain < float("inf"):
        raise ValueError("segment skip gain must be finite and nonnegative")
    _, gains, match, triple = operator_metrics(s, host_norm)
    count = len(gains)
    if isinstance(triple, dict):
        pairs = triple.keys()
        segments, skips = {}, {}
    else:
        pairs = ((a, b) for a in range(count) for b in range(count))
        segments = torch.zeros_like(triple, dtype=torch.bool)
        skips = torch.zeros_like(segments)
    for a, b in pairs:
        coherent = triple[a, b] > min_triple
        residual = (match[a] - match[a, b] * match[b] / gains[b].clamp_min(1e-30)).abs()
        residual = torch.where(gains[b] > 0, residual, float("inf"))
        valid = residual < epsilon
        segments[a, b] = valid & coherent
        skips[a, b] = valid & (gains[b] < skip_gain)
    return segments, skips


def skip_decisions(s):
    if (
        os.environ.get("KILORA_SEGMENTS") == "0"
        or s.host_weight_norm is None
        or not math.isfinite(s.host_weight_norm)
        or s.host_weight_norm <= 0
        or s.form != "input"
        or s.iteration != "accumulate"
        or s.down.shape[1] != s.up.shape[1]
    ):
        return None
    threshold = float(os.environ.get("KILORA_SEGMENT_SKIP_GAIN", "0.01"))
    epsilon = float(os.environ.get("KILORA_SEGMENT_EPS", "0.1"))
    if not 0 <= threshold < float("inf") or not 0 <= epsilon < float("inf"):
        raise ValueError("segment skip gain and epsilon must be finite and nonnegative")
    if threshold == 0 or epsilon == 0:
        return None
    key = compute._tensor_key(
        f"segment-skip-{s.per_expert}-{s.host_weight_norm}-{threshold}-{epsilon}",
        [s.down, s.up],
    )
    cached = compute._cache_get(key)
    if cached is not None:
        return cached
    _, skips = operator_decisions(s, s.host_weight_norm, skip_gain=threshold, epsilon=epsilon)
    compute._cache_put(key, skips)
    return skips


def has_skips(skips):
    if skips is None:
        return False
    if isinstance(skips, dict):
        return any(bool(row.any()) for row in skips.values())
    return bool(skips.any())


def skip_mask(skips, previous, current, following):
    a = previous.argmax(-1)
    b = current.argmax(-1)
    c = following.argmax(-1)
    if isinstance(skips, dict):
        mask = torch.zeros_like(a, dtype=torch.bool)
        for (source, relay), destinations in skips.items():
            mask |= (a == source) & (b == relay) & destinations[c]
    else:
        mask = skips[a, b, c]
    for route in (previous, current, following):
        mask &= (route == 1).sum(-1) == 1
        mask &= (route != 0).sum(-1) == 1
    return mask
