# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import torch

WRITE_FRAC = 1e-3
GROWTH = 1.2
SHRINK = 0.5
MAX_INITIAL = 8.0
TRUST_RATIO = 0.5
TRUST_STEPS = 32
SLOPE_EPS = 0.01
DECAY_AFTER = 5
DECAY = 0.9
MEASURE_COMBOS = 8
NOISE_FLOOR = 1e-6
NOISE_RATE = 0.3
DRIFT = 0.01


def sign_step(slope, last_sign, delta, initial_delta):
    if not all(math.isfinite(x) for x in (slope, last_sign, delta, initial_delta)):
        raise ValueError("controller inputs must be finite")
    if delta < 0 or initial_delta < 0 or last_sign not in (-1, 0, 1):
        raise ValueError("invalid controller state")
    sign = (slope > 0) - (slope < 0)
    if sign and sign == last_sign:
        delta = min(delta * GROWTH, initial_delta * MAX_INITIAL)
    elif sign and last_sign and sign != last_sign:
        return 0.0, sign, delta * SHRINK
    return -sign * delta, sign, delta


def combinations(count, size, seed):
    g = torch.Generator().manual_seed(int(seed) % (2**63))
    signs = torch.randint(0, 2, (count, size), generator=g, dtype=torch.int64)
    return (signs * 2 - 1).double() / math.sqrt(size)


def compressed_update(ids, psi, meas, prior):
    n, m = len(ids), psi.shape[0]
    meas = meas.double()
    eye = torch.eye(n, dtype=torch.float64)
    fresh = prior is None or not set(ids) & set(prior["ids"])
    if fresh:
        prev = torch.zeros(n, dtype=torch.float64)
        p0 = max(float(meas.pow(2).mean()), 1e-12)
        cov = p0 * eye
        noise = torch.tensor([NOISE_FLOOR * p0, p0], dtype=torch.float64)
    else:
        noise = prior["noise"].double()
        where = {key: i for i, key in enumerate(prior["ids"])}
        old = torch.tensor([where.get(key, -1) for key in ids], dtype=torch.int64)
        known = old >= 0
        prev = torch.zeros(n, dtype=torch.float64)
        prev[known] = prior["slopes"].double()[old[known]]
        cov = float(noise[1]) * eye
        rows = known.nonzero().flatten()
        if rows.numel():
            sub = prior["cov"].double()[old[rows]][:, old[rows]]
            cov[rows.unsqueeze(1), rows.unsqueeze(0)] = sub
        cov = cov + DRIFT * float(noise[1]) * eye
    resid = meas - psi @ prev
    pred = psi @ cov @ psi.t()
    if not fresh:
        sig = max(float(resid.pow(2).mean()) - float(pred.diagonal().mean()), 0.0)
        sig = (1 - NOISE_RATE) * float(noise[0]) + NOISE_RATE * sig
        noise = torch.tensor(
            [max(sig, NOISE_FLOOR * float(noise[1])), float(noise[1])], dtype=torch.float64
        )
    gain = (
        cov @ psi.t() @ torch.linalg.inv(pred + float(noise[0]) * torch.eye(m, dtype=torch.float64))
    )
    est = prev + gain @ resid
    cov = cov - gain @ psi @ cov
    cov = (cov + cov.t()) / 2
    return est, {"ids": list(ids), "slopes": est, "cov": cov, "noise": noise}
