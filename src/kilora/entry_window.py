# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math

import torch


def check_window(start, end):
    for name, value in (("start", start), ("end", end)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be a number from 0 to 1")


def windowed_sets(sets, windows):
    return [dataclasses.replace(s, extra={**s.extra, "entry_windows": windows}) for s in sets]


def windowed_sections(sections, windows):
    result = []
    for section in sections:
        index = section.meta.get("expert_index", 0)
        window = windows[index if len(windows) > 1 else 0]
        result.append(dataclasses.replace(section, meta={**section.meta, "entry_window": window}))
    return result


def active_layer_sets(sets, progress):

    result, positions = [], []
    for position, s in enumerate(sets):
        windows = s.entry_windows
        if not windows or all(w == (0.0, 1.0) for w in windows):
            result.append(s)
            positions.append(position)
            continue
        count = s.keys_in.shape[0]
        if len(windows) == 1:
            windows = windows * count
        if len(windows) != count:
            raise ValueError("entry window count does not match expert count")
        indices = tuple(i for i, (start, end) in enumerate(windows) if start <= progress <= end)
        if not indices:
            continue
        positions.append(position)
        if len(indices) == count:
            result.append(dataclasses.replace(s, entry_windows=(), route_freeze=None))
            continue

        def experts(value, indices=indices):
            return None if value is None else value[list(indices)]

        columns = [i * s.per_expert + j for i in indices for j in range(s.per_expert)]

        def cols(value, columns=columns):
            return None if value is None else value[columns]

        def sequence(value, indices=indices):
            return None if value is None else tuple(value[i] for i in indices)

        result.append(
            dataclasses.replace(
                s,
                entry_windows=(),
                window_indices=indices,
                window_count=count,
                down=cols(s.down),
                up=cols(s.up),
                routing_down=cols(s.routing_down),
                routing_up=cols(s.routing_up),
                routing_source=None,
                derived_cache_source=None,
                keys_in=experts(s.keys_in),
                keys_out=experts(s.keys_out),
                rows_in=experts(s.rows_in),
                rows_out=experts(s.rows_out),
                expert_ids=experts(s.expert_ids),
                expert_mask=None if s.expert_mask is None else s.expert_mask[..., columns],
                expert_inputs=sequence(s.expert_inputs),
                expert_offsets=sequence(s.expert_offsets),
                exact={i: s.exact[old] for i, old in enumerate(indices) if old in s.exact},
                gate=None if s.gate is None else {k: experts(v) for k, v in s.gate.items()},
                route_freeze=None,
                share_transition_weighted=None,
                share_transition_base=None,
                step_metric=None,
            )
        )
    return result, positions


def full_window_gates(gates, sets):
    result = []
    for gate, s in zip(gates, sets, strict=True):
        indices = getattr(s, "window_indices", None)
        if indices:
            indices = torch.tensor(indices, device=gate.device)
            full = gate.new_zeros((*gate.shape[:-1], s.window_count))
            gate = full.index_copy(-1, indices, gate)
        result.append(gate)
    return result
