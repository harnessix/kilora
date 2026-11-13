# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import logging
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

from . import key_cache

log = logging.getLogger("kilora.keys")
_FORMAT = 2


def backend(device):
    device = torch.device(device)
    result = {
        "device": str(device),
        "cuda": torch.version.cuda,
        "threads": torch.get_num_threads(),
        "precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["capability"] = torch.cuda.get_device_capability(device)
        result["tf32"] = torch.backends.cuda.matmul.allow_tf32
    return result


def identity(records, target, settings):
    if not records or any(not r.get("sha256") for r in records):
        return None
    sources = [
        {
            "sha256": record["sha256"],
            "weight": record.get("weight"),
            "rank": record.get("svd_ranks", {}).get(target),
        }
        for record in records
    ]
    payload = {
        "format": _FORMAT,
        "sources": sources,
        "target": target,
        "settings": settings,
        "torch": torch.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def expert_identities(records, target, down, up):
    if not records or any(not r.get("sha256") for r in records):
        return None
    result = []
    for record, a, b in zip(records, down, up, strict=True):
        digest = hashlib.sha256()
        digest.update(json.dumps([_FORMAT, record["sha256"], target], sort_keys=True).encode())
        for tensor in (a, b):
            tensor = tensor.detach().contiguous().cpu()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        result.append(digest.hexdigest())
    return tuple(result)


_EXPERT_FORMAT = "strength-free-2"


class Source(tuple):
    unit_up: torch.Tensor | None

    def __new__(cls, layer_id, expert_ids, unit_up=None):
        value = super().__new__(cls, (layer_id, expert_ids))
        value.unit_up = unit_up
        return value

    def __reduce__(self):
        return (Source, (self[0], self[1], self.unit_up))


def unit_up(value):
    return getattr(value, "unit_up", None)


def strength_free_identities(records, target, by_expert):

    if not records or any(not r.get("sha256") for r in records):
        return None
    result = []
    for e, record in enumerate(records):
        weight = float(record.get("weight", 1.0))
        sign = (weight > 0) - (weight < 0)
        digest = hashlib.sha256()
        payload = [_FORMAT, _EXPERT_FORMAT, record["sha256"], target, sign]
        digest.update(json.dumps(payload).encode())
        part = by_expert.get(e)
        if part is None:
            digest.update(b"no change")
        else:
            down, up = part
            down = down.detach().contiguous().cpu()
            digest.update(str((tuple(down.shape), down.dtype)).encode())
            digest.update(down.view(torch.uint8).numpy().tobytes())
            digest.update(str((tuple(up.shape), up.dtype)).encode())
        result.append(digest.hexdigest())
    return tuple(result)


def stamp(ex, value):
    with contextlib.suppress(RuntimeError):
        ex._derived_source = (value, ex.down, ex.down._version, ex.up, ex.up._version)


def source(ex):
    v = getattr(ex, "_derived_source", None)
    try:
        if (
            v
            and v[1] is ex.down
            and v[3] is ex.up
            and v[2] == ex.down._version
            and v[4] == ex.up._version
        ):
            return v[0]
    except RuntimeError:
        pass
    return None


def routing_source(ex):

    if getattr(ex, "routing_down", None) is None:
        return None
    return getattr(ex, "routing_source", None) or None


def bind(layer, value):
    with contextlib.suppress(AttributeError, RuntimeError):
        layer.derived_cache_source = (
            value,
            layer.down,
            layer.down._version,
            layer.up,
            layer.up._version,
        )
    return layer


def layer_source(layer):
    v = layer.derived_cache_source
    try:
        if (
            v
            and v[0]
            and v[1] is layer.down
            and v[3] is layer.up
            and v[2] == layer.down._version
            and v[4] == layer.up._version
        ):
            return v[0]
    except RuntimeError:
        pass
    return None


def _path(identity, kind, settings):
    payload = [identity, kind, settings, _FORMAT]
    name = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return os.path.join(key_cache.cache_dir(), "derived", name + ".safetensors")


STATS: collections.Counter = collections.Counter()

_MEMORY: collections.OrderedDict = collections.OrderedDict()
_MEMORY_BUDGET = int(os.environ.get("KILORA_DERIVED_MEMORY_MB", "4096")) << 20
_memory_bytes = 0


def clear_memory():
    global _memory_bytes
    _MEMORY.clear()
    _memory_bytes = 0


def _signature(path):
    s = os.stat(path)
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _remember(path, signature, tensors):
    global _memory_bytes
    old = _MEMORY.pop(path, None)
    if old is not None:
        _memory_bytes -= sum(t.numel() * t.element_size() for t in old[1].values())
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    if size > _MEMORY_BUDGET:
        return
    _MEMORY[path] = (signature, tensors)
    _memory_bytes += size
    while _memory_bytes > _MEMORY_BUDGET and _MEMORY:
        _, (_, dropped) = _MEMORY.popitem(last=False)
        _memory_bytes -= sum(t.numel() * t.element_size() for t in dropped.values())


def load(identity, kind, settings, names, device="cpu"):
    if not identity:
        return None
    path = _path(identity, kind, settings)
    try:
        signature = _signature(path)
        kept = _MEMORY.get(path)
        if kept is not None and kept[0] == signature:
            _MEMORY.move_to_end(path)
            STATS["memory_hits"] += 1
            tensors = kept[1]
        else:
            STATS["reads"] += 1
            tensors = load_file(path, device="cpu")
            _remember(path, signature, tensors)
        if set(tensors) != set(names):
            return None
        return tuple(tensors[name].to(device, copy=True) for name in names)
    except FileNotFoundError:
        STATS["misses"] += 1
        return None
    except Exception as exc:
        log.warning("cannot read derived cache %s: %s", path, exc)
        return None


def store(identity, kind, settings, names, values):
    if not identity:
        return
    path = _path(identity, kind, settings)
    tmp = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        os.close(fd)
        tensors = {
            name: value.detach().to("cpu", copy=True).contiguous()
            for name, value in zip(names, values, strict=True)
        }
        save_file(tensors, tmp)
        os.replace(tmp, path)
        _remember(path, _signature(path), tensors)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("cannot write derived cache %s: %s", path, exc)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
