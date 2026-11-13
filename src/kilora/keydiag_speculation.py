# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os

import torch

from . import compute


def speculation(record, index, tolerance=1e-5):

    s = record["sets"][index]
    if s.form != "input" or s.down is None or s.up is None:
        return {"skipped": "not an imported input set"}
    if compute.import_score(s) != "derived" or compute.derived_mode() != "full":
        return {"skipped": "requires derived full mode"}
    if "KILORA_SHARE_TRANSITION" in os.environ:
        return {"skipped": "requires default transition gamma"}
    if compute.derived_exclusion():
        return {"skipped": "exclusion is enabled"}
    if s.routing_scope != "token":
        return {"skipped": "tensor routing cannot be replayed on a token subset"}
    if record.get("has_regions"):
        return {"skipped": "regional masks were not captured"}
    if record.get("has_schedule"):
        return {"skipped": "scheduled depth was not captured"}
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    depth = record["depth"]
    if not isinstance(depth, (int, float)) or depth <= 1:
        return {"skipped": "depth has no later route"}
    steps = compute._depth_parts(depth)[0]
    with torch.no_grad():
        transition = compute.share_transition(s)
        if transition is None:
            return {"skipped": "transition unavailable"}
        x = record["x"].to(s.down.device).unsqueeze(0)
        gamma = compute.share_transition_gamma(s)
        seed, step_key = record.get("dropout_key", (0, 0))
        with compute._DROPOUT_KEY.pushed((seed, step_key)):
            entries = compute.derived_entries(
                s, x, f"{record['target']}#input{index}", s.path_width
            )
        logits = compute.derived_logits(s, x) * compute.derived_lift_after()
        all_argmax = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        all_close = all_argmax.clone()
        all_exact = all_argmax.clone()
        step_argmax = [torch.ones_like(all_argmax) for _ in range(steps - 1)]
        step_close = [v.clone() for v in step_argmax]
        step_exact = [v.clone() for v in step_argmax]
        for entry in entries:
            true = predicted = entry
            for k in range(steps - 1):
                true = compute._gate(
                    logits + gamma * (true.float() @ transition + 1e-6).log(), s, 0.0
                )
                predicted = compute._gate(
                    gamma * (predicted.float() @ transition + 1e-6).log(), s, 0.0
                )
                same_choice = true.argmax(-1) == predicted.argmax(-1)
                close = (true - predicted).abs().amax(-1) <= tolerance
                exact = (true == predicted).all(-1)
                step_argmax[k] &= same_choice
                step_close[k] &= close
                step_exact[k] &= exact
                all_argmax &= same_choice
                all_close &= close
                all_exact &= exact

        def rate(mask):
            return mask.float().mean().item()

        return {
            "tokens": x.shape[-2],
            "walks": len(entries),
            "gamma": gamma,
            "tolerance": tolerance,
            "argmax_all": rate(all_argmax),
            "weights_within_tolerance_all": rate(all_close),
            "argmax_and_tolerance_all": rate(all_argmax & all_close),
            "weights_exact_all": rate(all_exact),
            "depth_steps": [
                {
                    "depth": k + 2,
                    "argmax": rate(a),
                    "weights_within_tolerance": rate(b),
                    "weights_exact": rate(c),
                    "argmax_and_tolerance": rate(a & b),
                }
                for k, (a, b, c) in enumerate(zip(step_argmax, step_close, step_exact, strict=True))
            ],
        }
