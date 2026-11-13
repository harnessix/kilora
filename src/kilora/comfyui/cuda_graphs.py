# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

log = logging.getLogger("kilora.cuda_graphs")

ENABLED = False
_WARNED: list[bool] = []


def requested() -> bool:
    if os.environ.get("KILORA_CUDA_GRAPHS", "0").strip() != "1":
        return False
    if not ENABLED:
        if not _WARNED:
            _WARNED.append(True)
            log.warning("KILORA_CUDA_GRAPHS is ignored: CUDA graphs are disabled in this release")
        return False
    return True


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
    if isinstance(value, dict):
        return ("dict", tuple((k, _signature(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_signature(v) for v in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    raise ValueError("unsupported conditioning value")


def _copy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_tree(v) for v in value)
    if isinstance(value, list):
        return [_copy_tree(v) for v in value]
    return value


def _update(dst, src):
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, dict):
        for k in src:
            _update(dst[k], src[k])
    elif isinstance(src, (tuple, list)):
        for a, b in zip(dst, src, strict=True):
            _update(a, b)


def safe(out, args, frozen, targets, carry, progress, *, mutable=False):
    from .. import compute, linear_jump, route_freeze
    from ..recording import active_perturbation, active_recorder

    if (
        carry
        or mutable
        or torch.is_grad_enabled()
        or not targets
        or args["input"].device.type != "cuda"
        or compute._PROBES
        or compute._ROUTES
        or compute._SHARES
        or route_freeze.RECORDS
        or route_freeze._OFF
        or linear_jump._RECORDS
        or active_recorder() is not None
        or active_perturbation() is not None
    ):
        return False
    if any(
        name not in frozen or threshold is None or progress < threshold
        for name, threshold in targets.items()
    ):
        return False
    try:
        import comfy.model_management as mm

        if mm.vram_state in (mm.VRAMState.LOW_VRAM, mm.VRAMState.NO_VRAM):
            return False
    except (ImportError, AttributeError):
        return False
    return not getattr(out.model, "model_lowvram", False)


class Graphs:
    def __init__(self):
        self.buckets = OrderedDict()
        self.failed = set()
        self.caches = {}
        self.pool = None

    def generation(self, kind, cache):
        old = self.caches.get(kind)
        if old is not None and old is not cache:
            self.buckets.clear()
            self.failed.clear()
            self.pool = None
            self.caches.clear()
        self.caches[kind] = cache

    def call(self, args, cache, invoke):
        try:
            key = (
                id(cache),
                tuple((name, id(entry)) for name, entry in cache.items()),
                _signature(args),
            )
            hash(key)
        except (TypeError, ValueError):
            return invoke(args)
        if key in self.failed:
            return invoke(args)
        try:
            bucket = self.buckets.get(key)
            if bucket is None:
                static = _copy_tree(args)
                device = args["input"].device
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                torch.cuda.reset_peak_memory_stats(device)
                before = torch.cuda.memory_allocated(device)
                with torch.cuda.stream(stream):
                    invoke(static)
                torch.cuda.current_stream(device).wait_stream(stream)
                if key[1] != tuple((name, id(entry)) for name, entry in cache.items()):
                    return invoke(args)
                peak = max(0, torch.cuda.max_memory_allocated(device) - before)
                free, _ = torch.cuda.mem_get_info(device)
                if free < 2 * peak:
                    return invoke(args)
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    output = invoke(static)
                if not isinstance(output, torch.Tensor):
                    raise TypeError("graph output must be a tensor")
                bucket = (graph, static, output)
                self.buckets[key] = bucket
                if len(self.buckets) > 4:
                    self.buckets.popitem(last=False)
            else:
                self.buckets.move_to_end(key)
                graph, static, output = bucket
                _update(static, args)
                graph.replay()
            return bucket[2].clone()
        except Exception:
            self.buckets.pop(key, None)
            self.failed.add(key)
            log.warning("CUDA graph disabled for this bucket", exc_info=True)
            return invoke(args)
