# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from kilora.call_values import host_list, host_scalar, host_values
from kilora.comfyui import patching
from kilora.comfyui.layer_crop import sampling_progress

_NAMES = ("item", "__bool__", "__float__", "__int__", "tolist")


def _old_sampling_progress(sigma, sample_sigmas=None, first=None):
    values = []
    if sample_sigmas is not None:
        try:
            values = [float(v) for v in sample_sigmas.flatten() if float(v) > 0]
        except (AttributeError, TypeError, ValueError):
            values = []
    if values:
        high, low = max(values), min(values)
        if high <= low:
            return 0.0
        return min(1.0, max(0.0, (high - sigma) / (high - low)))
    if first is None or first <= 0:
        return 0.0
    return min(1.0, max(0.0, 1 - sigma / first))


def _old_progress(args):
    sigmas = (args.get("c") or {}).get("transformer_options", {}).get("sample_sigmas")
    t = args.get("timestep")
    if sigmas is None or t is None or len(sigmas) < 2:
        return 0.0
    sigma = float(t.detach().flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
    values = torch.as_tensor(sigmas, dtype=torch.float64).flatten()[:-1]
    index = int(torch.argmin((values - sigma).abs()))
    return index / max(len(values) - 1, 1)


class _OldStepCarry:
    def __init__(self):
        self.caches = {}

    def cache_for(self, args):
        kind = tuple(
            (args.get("c") or {}).get("transformer_options", {}).get("cond_or_uncond") or ()
        )
        x, t = args["input"], args["timestep"]
        step = float(t.detach().flatten()[0]) if isinstance(t, torch.Tensor) else float(t)
        shape = tuple(x.shape)
        entry = self.caches.get(kind)
        sigmas = (args.get("c") or {}).get("transformer_options", {}).get("sample_sigmas")
        first = sigmas is not None and len(sigmas) and float(sigmas[0]) == step
        if entry is None or first or step >= entry["step"] or entry["shape"] != shape:
            entry = {"step": step, "shape": shape, "state": {}}
            self.caches[kind] = entry
        entry["step"] = step
        return entry["state"]


def _schedule(n, top, dtype):
    values = [top * (1 - i / n) ** 1.7 + 0.0137 * i for i in range(n)] + [0.0]
    return torch.tensor(values, dtype=dtype)


def _args(sigmas, i, dtype, kind=(0, 1)):
    return {
        "input": torch.zeros(2, 4, 8, 8),
        "timestep": sigmas[i : i + 1].clone().to(dtype),
        "c": {"transformer_options": {"sample_sigmas": sigmas, "cond_or_uncond": list(kind)}},
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_progress_values_match_the_per_call_reads(dtype):
    runs = [_schedule(9, 14.6, dtype), _schedule(9, 3.3, dtype), _schedule(6, 7.9, dtype)]
    changed = _schedule(9, 14.6, dtype)
    runs += [changed, changed]
    for run, sigmas in enumerate(runs):
        if run == len(runs) - 1:
            sigmas.mul_(0.37)
        for i in range(len(sigmas) - 1):
            args = _args(sigmas, i, dtype)
            sigma = float(args["timestep"][0])
            with host_values():
                assert patching._progress(args) == _old_progress(args)
                assert patching._progress(args) == _old_progress(args)
            assert patching._progress(args) == _old_progress(args)
            assert sampling_progress(sigma, sigmas) == _old_sampling_progress(sigma, sigmas)
            off = sigma * 0.913
            assert sampling_progress(off, sigmas) == _old_sampling_progress(off, sigmas)
            assert host_list(sigmas) == tuple(float(v) for v in sigmas)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_step_cache_matches_the_per_call_reads(dtype):
    new, old = patching.StepCarry(), _OldStepCarry()
    calls = []
    for top, n in ((14.6, 7), (3.3, 7), (14.6, 5)):
        sigmas = _schedule(n, top, dtype)
        for i in range(n):
            for kind in ((0,), (1,), (0, 1)):
                calls.append(_args(sigmas, i, dtype, kind))
    calls += calls[:6]
    for marks, args in enumerate(calls):
        with host_values():
            a = new.cache_for(args)
        b = old.cache_for(args)
        assert ("mark" in a) == ("mark" in b)
        a.setdefault("mark", marks)
        b.setdefault("mark", marks)
        assert a["mark"] == b["mark"]


def test_host_scalar_reads_a_timestep_once_per_model_call():
    t = torch.tensor([3.25, 1.0])
    counts = []
    original = torch.Tensor.__float__

    def counted(tensor):
        counts.append(1)
        return original(tensor)

    with patch.object(torch.Tensor, "__float__", counted):
        with host_values():
            assert [host_scalar(t) for _ in range(4)] == [3.25] * 4
        assert len(counts) == 1
        with host_values():
            assert host_scalar(t) == 3.25
            t.fill_(0.5)
            assert host_scalar(t) == 0.5
        assert len(counts) == 3
        assert host_scalar(t) == 0.5
        assert len(counts) == 4


def test_host_list_reads_a_new_tensor_again():
    with torch.inference_mode():
        first = torch.tensor([4.0, 2.0, 0.0])
        assert host_list(first) == (4.0, 2.0, 0.0)
        second = torch.tensor([9.0, 1.5, 0.0])
        assert host_list(second) == (9.0, 1.5, 0.0)
    assert host_list(torch.tensor([1, 2])) == (1, 2)


def test_sigma_list_is_read_once_per_run():
    sigmas = _schedule(9, 14.6, torch.float32)
    counts = []
    originals = {n: getattr(torch.Tensor, n) for n in _NAMES}

    def wrap(name):
        def counted(tensor, *a, **k):
            counts.append(name)
            return originals[name](tensor, *a, **k)

        return counted

    calls = [_args(sigmas, i, torch.float32) for i in range(len(sigmas) - 1)]
    with patch.multiple(torch.Tensor, **{n: wrap(n) for n in originals}):
        carry = patching.StepCarry()
        for args in calls:
            before = len(counts)
            with host_values():
                patching._progress(args)
                patching._progress(args)
                carry.cache_for(args)
                carry.cache_for(args)
                sampling_progress(host_scalar(args["timestep"]), sigmas)
            scalars = [n for n in counts[before:] if n != "tolist"]
            assert len(scalars) <= 1, counts[before:]
    assert counts.count("tolist") == 1, counts
