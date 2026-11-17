# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging

import torch


def same_write_state(left, right):
    return left.keys() == right.keys() and all(
        left[t].keys() == right[t].keys()
        and all(torch.equal(v.detach().cpu(), right[t][k]) for k, v in p.items())
        for t, p in left.items()
    )


def copy_write_state(state):
    return {t: {k: v.detach().cpu().clone() for k, v in p.items()} for t, p in state.items()}


def reset_write_estimates(session):
    session._write_estimates = {}
    session._write_estimate_state = None
    session._write_estimate_context = None


def accumulate_write_estimate(session, estimate):

    from .live_imports import RESPONSE_FLOOR

    if hasattr(estimate, "mean_gradients"):
        return estimate.mean_response_note
    forest = getattr(getattr(session, "editor_state", None), "forest", None)
    context = (getattr(session, "revision", None), getattr(forest, "current", None))
    state = getattr(session, "_write_estimate_state", None)
    if (
        state is None
        or getattr(session, "_write_estimate_context", None) != context
        or not same_write_state(state, estimate.state)
    ):
        session._write_estimate_context = context
        session._write_estimate_state = copy_write_state(estimate.state)
        session._write_estimates = {}
    totals = session._write_estimates
    gradients = estimate.gradients()
    averaged, notes = {}, []
    for target in sorted(estimate.target):
        keys = sorted(key for key in gradients if key[0] == target)
        signature = tuple(keys)
        previous = totals.get(target)
        if previous is None or previous[0] != signature:
            previous = (signature, 0, {})
        _, count, sums = previous
        count += 1
        current = {}
        for key in keys:
            core, strength = gradients[key]
            old_core, old_strength = sums.get(key, (torch.zeros_like(core), 0.0))
            current[key] = (old_core + core.detach().cpu(), old_strength + strength)
            averaged[key] = (current[key][0] / count, current[key][1] / count)
        totals[target] = (signature, count, current)
        response = (
            sum(
                float(core.square().sum()) + strength**2
                for key, (core, strength) in averaged.items()
                if key[0] == target
            )
            ** 0.5
        )
        raw = (
            sum(float(gradients[key][0].square().sum()) + gradients[key][1] ** 2 for key in keys)
            ** 0.5
        )
        notes.append(
            f"Write target {target}: estimated response {raw:.9g}, "
            f"mean response {response:.9g} over {count} estimates, floor {RESPONSE_FLOOR:g}."
        )
    estimate.mean_gradients = averaged
    estimate.mean_response_note = " ".join(notes) or "Write estimate has no traversed target."
    logging.getLogger(__name__).info(estimate.mean_response_note)
    return estimate.mean_response_note


def mean_write_values(session, state, values, *, kind, valid=None):

    from types import SimpleNamespace

    gradients = {
        ((kind, key), 0): (torch.as_tensor(value).detach().cpu().double(), 0.0)
        for key, value in values.items()
    }
    estimate = SimpleNamespace(
        state=state,
        target={target: [0] for target, _ in gradients},
        gradients=lambda: gradients,
    )
    note = accumulate_write_estimate(session, estimate)
    means = {}
    for target, (_signature, count, sums) in session._write_estimates.items():
        if not isinstance(target, tuple) or target[0] != kind:
            continue
        key = target[1]
        if valid is None or key in valid:
            means[key] = sums[(target, 0)][0] / count
    return means, note
