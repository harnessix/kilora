# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import shutil

import torch
from test_comfyui import denoiser, folders
from test_path_write import WrapPatcher, _setup, fresh_sessions

from kilora.comfyui import nodes as N


def _record(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "path.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)

    def start():
        return N.KiLoRALiveRecord().record(wp, "path.kedit.safetensors")

    return session, start


def _sample(model, x, steps=3, mode="together"):

    fn = model.model_options["model_function_wrapper"]
    n = x.shape[0]

    def apply_model(inp, t, **c):
        return model.run(inp)

    def call(batch, cou, cond):
        c = {"c_crossattn": cond}
        fn(apply_model, {"input": batch, "timestep": t, "c": c, "cond_or_uncond": cou})

    for i in range(steps):
        t = torch.full((n,), float(steps - i))
        pos, neg = torch.ones(n, 2), -torch.ones(n, 2)
        if mode == "together":
            call(torch.cat([-x, x]), [1, 0], torch.cat([neg, pos]))
        elif mode == "split":
            call(-x, [1], neg)
            call(x, [0], pos)
        else:
            call(x, [0], pos)


def _stats(handle):
    return {t: (s.vk / s.n, s.cc / s.n, s.c2 / s.n) for t, s in handle.recorder.sums.items()}


RTOL = 1e-6


def _close(a, b):
    assert set(a) == set(b) and a
    for t in a:
        for p, q in zip(a[t], b[t], strict=True):
            assert float((p - q).norm()) <= RTOL * float(q.norm()) + 1e-300, (t, p, q)


def test_batch_statistics_are_the_mean_over_images(denoiser, folders, tmp_path):
    session, start = _record(denoiser, folders, tmp_path)
    x0, x1 = torch.randn(2, 1, 5, 4, generator=torch.Generator().manual_seed(3))
    singles = []
    for x in (x0, x1):
        model, handle = start()
        _sample(model, x)
        singles.append(_stats(handle))
    model, handle = start()
    _sample(model, torch.cat([x0, x1]))
    mean = {
        t: tuple((a + b) / 2 for a, b in zip(singles[0][t], singles[1][t], strict=True))
        for t in singles[0]
    }
    _close(_stats(handle), mean)
    model, twice = start()
    _sample(model, torch.cat([x0, x0]))
    _close(_stats(twice), singles[0])


def test_batch_makes_one_write_and_one_commit(denoiser, folders, tmp_path):
    session, start = _record(denoiser, folders, tmp_path)
    x = torch.randn(3, 5, 4, generator=torch.Generator().manual_seed(4))
    model, handle = start()
    _sample(model, x)
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": x})["result"]
    assert "1 commits" in report
    assert len(session.commits) == 1 and session.commits[0].images == 3
    assert len(session.undo_log) == 1
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": x})["result"]
    assert "Nothing was recorded" in report and len(session.commits) == 1


def test_guidance_records_the_conditional_rows_only(denoiser, folders, tmp_path):
    session, start = _record(denoiser, folders, tmp_path)
    x = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(5))
    model, plain = start()
    _sample(model, x, mode="none")
    model, guided = start()
    _sample(model, x, mode="together")
    _close(_stats(guided), _stats(plain))
    assert {s.n for s in guided.recorder.sums.values()} == {3 * 2 * 5}
    assert torch.equal(guided.conditioning["c_crossattn"], torch.ones(2, 2))
    assert guided.timesteps == [3.0, 2.0, 1.0]


def test_split_guidance_passes_skip_the_unconditional_call(denoiser, folders, tmp_path):
    session, start = _record(denoiser, folders, tmp_path)
    x = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(6))
    model, together = start()
    _sample(model, x, mode="together")
    model, split = start()
    _sample(model, x, mode="split")
    _close(_stats(split), _stats(together))
    assert torch.equal(split.conditioning["c_crossattn"], torch.ones(2, 2))
    assert split.timesteps == [3.0, 2.0, 1.0] and split.calls == 3
    _, report = N.KiLoRALiveWrite().write(split, {"samples": x})["result"]
    assert "1 commits" in report and session.commits[0].images == 2
