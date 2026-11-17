# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import sys
import tempfile
import threading
import time

log = logging.getLogger("kilora.prepare")
_lock = threading.Lock()
_active = 0
_worker = None
_stop = threading.Event()
_shutting_down = False
_JOIN_TIMEOUT = 30.0


@contextlib.contextmanager
def model_call():
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1


def busy():
    from ..live_queue import writes_busy

    with _lock:
        if _active:
            return True
    if writes_busy():
        return True
    try:
        from server import PromptServer

        queue = PromptServer.instance.prompt_queue
        return bool(queue.get_tasks_remaining())
    except (ImportError, AttributeError):
        return False


def _settings_path():
    from ..key_cache import cache_dir

    return os.path.join(os.path.dirname(cache_dir()), "prepare-startup.json")


def enabled():
    if os.environ.get("KILORA_PREPARE_AT_STARTUP", "1") == "0":
        return False
    try:
        with open(_settings_path(), encoding="utf-8") as stream:
            return json.load(stream).get("enabled", True) is True
    except (OSError, ValueError, AttributeError):
        return True


def set_enabled(value):
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"enabled": bool(value)}, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if value:
        start()
    else:
        _stop.set()


class _Stopped(Exception):
    pass


def _checkpoint(device, label="LoRA library"):
    if _stop.is_set():
        raise _Stopped
    if busy():
        log.info("Startup preparation paused: %s", label)
        if _stop.is_set():
            raise _Stopped
        import torch

        if torch.device(device).type == "cuda":
            if _stop.is_set():
                raise _Stopped
            torch.cuda.synchronize(device)
            if _stop.is_set():
                raise _Stopped
            torch.cuda.empty_cache()
        while busy():
            if _stop.wait(0.2):
                raise _Stopped
        log.info("Startup preparation resumed: %s", label)


_SAFETENSORS_DTYPES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}
_HEADER_LIMIT = 100 << 20
_CHECKPOINT_FAMILIES = frozenset({"sd15", "sdxl"})


def _header(path):
    with open(path, "rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size <= _HEADER_LIMIT:
            raise ValueError(f"no safetensors header of a valid size: {size}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("the safetensors header is not an object")
    metadata = header.pop("__metadata__", None) or {}
    entries = {}
    for key, info in header.items():
        entries[key] = (info["dtype"], tuple(int(n) for n in info["shape"]))
    return entries, metadata


def _stand_ins(entries):
    import torch

    state = {}
    for key, (name, shape) in entries.items():
        dtype = getattr(torch, _SAFETENSORS_DTYPES.get(name, ""), None)
        if isinstance(dtype, torch.dtype):
            state[key] = torch.empty(shape, dtype=dtype, device="meta")
    return state


def _family(names):
    from ..bindings import BINDINGS
    from ..families import detect_names

    names = set(names)
    spec = detect_names(sorted(names))
    if spec is not None:
        return spec.family
    for binding in BINDINGS.values():
        if binding.matches_comfy(names):
            return binding.family
    return None


def _lora_families(path):
    from ..prepare import lora_families

    entries, _ = _header(path)
    return lora_families(entries, path)


def _comfy_module(name):
    import importlib

    from ..host import services

    comfy = services().comfy
    found = getattr(comfy, name.rsplit(".", 1)[-1], None) if comfy is not None else None
    if found is not None and name != "folder_paths":
        return found
    if name == "folder_paths":
        from .nodes import FOLDERS

        return FOLDERS._fp()
    return importlib.import_module(name)


def _unet_prefix(sd, detection):
    with contextlib.suppress(Exception):
        prefix = detection.unet_prefix_from_state_dict(sd)
        if isinstance(prefix, str) and any(k.startswith(prefix) for k in sd):
            return prefix
    from ..family_spec import DIFFUSION_PREFIXES

    for prefix in DIFFUSION_PREFIXES:
        if any(k.startswith(prefix) for k in sd):
            return prefix
    return ""


def _weight_dtype(sd):
    counts = {}
    for tensor in sd.values():
        counts[tensor.dtype] = counts.get(tensor.dtype, 0) + tensor.numel()
    return max(counts, key=counts.get) if counts else None


def _predict_file(path, device):
    import torch

    from ..prepare import PLAN_DTYPES

    entries, metadata = _header(path)
    sd = _stand_ins(entries)
    detection = _comfy_module("comfy.model_detection")
    mm = _comfy_module("comfy.model_management")
    utils = _comfy_module("comfy.utils")
    prefix = _unet_prefix(sd, detection)
    if prefix:
        sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sd:
        raise ValueError("no diffusion model keys")
    family = _family(k.rsplit(".", 1)[0] for k in sd)
    try:
        weight_dtype = utils.weight_dtype(sd)
    except Exception:
        weight_dtype = _weight_dtype(sd)
    try:
        parameters = utils.calculate_parameters(sd)
    except Exception:
        parameters = sum(t.numel() for t in sd.values())
    try:
        config = detection.model_config_from_unet(sd, "", metadata=metadata)
    except TypeError:
        config = detection.model_config_from_unet(sd, "")
    if config is None:
        raise ValueError("ComfyUI does not detect the model")
    supported = getattr(config, "supported_inference_dtypes", None)
    try:
        load_device = mm.get_torch_device()
    except Exception:
        load_device = torch.device(device)
    unet_dtype = mm.unet_dtype(
        model_params=parameters, supported_dtypes=supported, weight_dtype=weight_dtype
    )
    cast = mm.unet_manual_cast(unet_dtype, load_device, supported)
    dtype = unet_dtype if cast is None else cast
    if dtype not in PLAN_DTYPES:
        raise ValueError(f"the plan dtype {dtype} is not float32, float16 or bfloat16")
    return family, dtype


def _predict(device, library):
    families = set()
    for path in library:
        with contextlib.suppress(Exception):
            families.update(_lora_families(path))
    if not families:
        return set(), [], "the family of the library files is unknown"
    try:
        folder_paths = _comfy_module("folder_paths")
    except Exception as exc:
        return set(), [], f"no ComfyUI model folders ({exc})"
    folders = ["diffusion_models"]
    if families & _CHECKPOINT_FAMILIES:
        folders.append("checkpoints")
    dtypes, files, failures = set(), [], []
    for folder in folders:
        try:
            names = list(folder_paths.get_filename_list(folder))
        except Exception:
            names = []
        for name in names:
            if not str(name).lower().endswith(".safetensors"):
                continue
            try:
                path = folder_paths.get_full_path(folder, name)
                family, dtype = _predict_file(path, device)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                continue
            if family in families:
                dtypes.add(dtype)
                files.append(name)
    if dtypes:
        return dtypes, files, None
    if failures:
        return set(), [], "the detection failed: " + "; ".join(failures)
    return set(), [], "no diffusion model file of the families " + ", ".join(sorted(families))


def _found_plan_dtype(device, library):
    import torch

    from ..prepare import PLAN_DTYPES, plan_dtype

    named = os.environ.get("KILORA_PLAN_DTYPE", "").strip()
    if named:
        dtype = getattr(torch, named, None)
        if dtype in PLAN_DTYPES:
            return dtype, "environment", [], None
        log.warning("KILORA_PLAN_DTYPE=%s is not float32, float16 or bfloat16", named)
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None
    if mm is not None:
        found = set()
        for loaded in getattr(mm, "current_loaded_models", ()) or ():
            with contextlib.suppress(Exception):
                model = getattr(loaded, "model", None)
                inner = getattr(model, "model", None)
                if inner is not None and hasattr(inner, "diffusion_model"):
                    found.add(plan_dtype(inner))
        if len(found) == 1 and None not in found:
            return found.pop(), "loaded model", [], None
    reason = None
    try:
        dtypes, files, reason = _predict(device, library)
    except Exception as exc:
        dtypes, files, reason = set(), [], f"the prediction failed: {exc}"
    if dtypes:
        ordered = tuple(d for d in PLAN_DTYPES if d in dtypes)
        return (ordered[0] if len(ordered) == 1 else ordered), "header prediction", files, None
    if mm is None:
        return None, None, [], reason
    try:
        from comfy.cli_args import args

        forced = (
            torch.float32
            if getattr(args, "fp32_unet", False)
            else torch.float16
            if getattr(args, "fp16_unet", False)
            else torch.bfloat16
            if getattr(args, "bf16_unet", False)
            else None
        )
        if forced is None:
            return None, None, [], reason
        cast = mm.unet_manual_cast(forced, torch.device(device))
        dtype = forced if cast is None else cast
        if dtype in PLAN_DTYPES:
            return dtype, "forced flag", [], None
        return None, None, [], reason
    except Exception:
        return None, None, [], reason


def _plan_dtype(device, library=()):

    dtype, source, files, reason = _found_plan_dtype(device, library)
    if dtype is None:
        log.info("No plan dtype prediction: %s", reason or "no model, header or flag")
    else:
        log.info(
            "Plan dtype %s from the %s%s",
            dtype,
            source,
            f" of {', '.join(map(str, files))}" if files else "",
        )
    return dtype


def _run(roots):
    from ..prepare import (
        layer_ranks,
        library_paddings,
        prepare_file,
        prepare_lokr,
        prepare_paddings,
    )
    from .prepare_node import apply_device, display_path, selected_files

    started = time.monotonic()
    counts = {"prepared": 0, "current": 0, "other": 0, "unmapped": 0, "failed": 0}
    try:
        if hasattr(os, "nice") and sys.platform.startswith("linux"):
            with contextlib.suppress(OSError):
                os.nice(19)
        if _stop.is_set():
            raise _Stopped
        device = apply_device()
        label = "LoRA library"
        checkpoint = lambda: _checkpoint(device, label)
        paths, _ = selected_files(roots, "")
        ranks = {}
        labels = {}
        for selected in paths:
            path = os.path.realpath(selected)
            label = display_path(selected, roots)
            checkpoint()
            log.info("Preparing LoRA: %s", label)
            try:
                new, current, skipped = prepare_lokr(
                    path, checkpoint=checkpoint, detailed=True, device=device
                )
                checkpoint()
                derived = False
                try:
                    result = prepare_file(path, checkpoint=checkpoint, device=device)
                    ranks[path] = layer_ranks(result)
                    labels[path] = label
                    derived = True
                except ValueError:
                    counts["unmapped"] += 1
                counts["current" if skipped else "prepared" if new or derived else "other"] += 1
            except _Stopped:
                raise
            except Exception:
                counts["failed"] += 1
                log.exception("Failed to prepare LoRA: %s", label)
        added = {"files": 0, "entries": 0, "bytes": 0}
        dtype = _plan_dtype(device, list(ranks))
        if dtype is None:
            log.info(
                "The plan dtype of the next render is unknown: LoRA paddings are stored "
                "in float32, float16 and bfloat16 (set KILORA_PLAN_DTYPE to store one)"
            )
        else:
            log.info("LoRA paddings are stored in the plan dtype %s", dtype)
        for path, wanted in library_paddings(ranks).items():
            if not wanted:
                continue
            label = labels[path]
            checkpoint()
            try:
                report = prepare_paddings(
                    path, wanted, checkpoint=checkpoint, device=device, dtype=dtype
                )
            except _Stopped:
                raise
            except Exception:
                counts["failed"] += 1
                log.exception("Failed to prepare LoRA paddings: %s", labels[path])
                continue
            if report["entries"]:
                added["files"] += 1
                added["entries"] += report["entries"]
                added["bytes"] += report["bytes"]
                log.info(
                    "Prepared LoRA paddings: %s: paddings %s, %d entries, %d bytes",
                    labels[path],
                    report["paddings"],
                    report["entries"],
                    report["bytes"],
                )
        counts["paddings"] = added
        label = ", ".join(labels.values()) or "LoRA library"
        counts["sizes"] = _prepare_sizes(list(labels), checkpoint)
        log.info("Startup LoRA preparation: %s in %.2f s", counts, time.monotonic() - started)
    except _Stopped:
        log.info("Startup preparation stopped")
    except Exception as exc:
        log.warning("Startup LoRA preparation failed: %s", exc)


def base_model_files():
    named = os.environ.get("KILORA_SIZE_BASE", "").strip()
    if named:
        return [p for p in named.split(os.pathsep) if p and os.path.isfile(p)]
    try:
        from .nodes import FOLDERS

        return [
            FOLDERS.path("diffusion_models", n)
            for n in FOLDERS.list("diffusion_models", ".safetensors")
        ]
    except Exception:
        return []


def _prepare_sizes(paths, checkpoint):
    from ..prepare import prepare_sizes

    done = {"bases": 0, "files": 0}
    if not paths:
        return done
    for base in base_model_files():
        checkpoint()
        try:
            sizes = prepare_sizes(paths, base, checkpoint=checkpoint)
        except _Stopped:
            raise
        except Exception:
            log.exception("Failed to prepare LoRA sizes on base: %s", os.path.basename(base))
            continue
        stored = sum(1 for v in sizes.values() if v is not None)
        if stored:
            done["bases"] += 1
            done["files"] += stored
            log.info("Prepared LoRA sizes on %s: %d files", os.path.basename(base), stored)
    return done


def shutdown():
    global _shutting_down
    with _lock:
        _shutting_down = True
        _stop.set()
        worker = _worker
    if worker is not None and worker is not threading.current_thread():
        worker.join(_JOIN_TIMEOUT)
        if worker.is_alive():
            log.warning("Startup preparation did not stop within %.1f s", _JOIN_TIMEOUT)


atexit.register(shutdown)


def start():
    global _worker
    if not enabled():
        return None
    try:
        from .nodes import FOLDERS

        roots = FOLDERS.dirs("loras")
    except (ImportError, AttributeError, OSError):
        return None
    if not roots:
        return None
    with _lock:
        if _shutting_down:
            return None
        if _worker is not None and _worker.is_alive():
            return _worker
        _stop.clear()
        _worker = threading.Thread(target=_run, args=(roots,), name="kilora-prepare", daemon=True)
        _worker.start()
        return _worker
