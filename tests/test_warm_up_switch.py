# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import torch

from kilora import fused
from kilora.comfyui import patching


def test_apply_reuses_shapes_and_reports_only_new_work(monkeypatch, caplog):
    monkeypatch.setattr(fused, "_WARMED", set())
    monkeypatch.setattr(fused, "_WARM_STATS", [0, 0.0])
    monkeypatch.setattr(fused, "enabled", lambda: True)
    monkeypatch.setattr(fused, "HAVE_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused, "_comfy_device", lambda: torch.device("cpu"))
    work = []

    def warm_up(device, templates, dtypes):
        jobs = []
        for dtype in dtypes:
            for template, depth in templates:
                key = (str(device), dtype, fused.shape_key(template, depth))
                if key not in fused._WARMED:
                    fused._WARMED.add(key)
                    jobs.append(key)
        work.append(jobs)
        fused._WARM_STATS[0] += len(jobs)

    def spawn(call):
        call()
        return object()

    monkeypatch.setattr(fused, "warm_up", warm_up)
    monkeypatch.setattr(fused, "_spawn", spawn)
    first = fused._default_template(per=16)
    same = fused._default_template(per=16)
    new = fused._default_template(per=32)
    model = SimpleNamespace(manual_cast_dtype=torch.float32)
    patcher = SimpleNamespace(model=model)

    def apply(*sets):
        plans = {str(i): SimpleNamespace(sets=[item], depth=1) for i, item in enumerate(sets)}
        patching._warm_up_shapes(patcher, plans)

    def report():
        with fused.sample_compile_report():
            pass

    with caplog.at_level(logging.INFO, logger="kilora"):
        apply(first)
        report()
        apply(same)
        report()
        assert len(work) == 1
        assert "1 shapes warmed" in caplog.text
        assert "0 shapes warmed" in caplog.text
        apply(same, new)
        report()
    assert [len(jobs) for jobs in work] == [1, 1]
    assert work[-1][0][2] == fused.shape_key(new)

    model.manual_cast_dtype = torch.float16
    apply(same)
    assert len(work) == 3
    assert work[-1][0][:2] == ("cpu", torch.float16)

    fused.warm_up_sets([([same], 1)], device="cuda:0", dtypes=(torch.float32,))
    assert len(work) == 4
    assert work[-1][0][:2] == ("cuda:0", torch.float32)


def test_wait_ignores_finished_threads(monkeypatch):
    class Finished:
        def is_alive(self):
            return False

        def join(self):
            raise AssertionError("finished warm-up must not be waited on")

    monkeypatch.setattr(fused, "_WARM_THREADS", [Finished()])
    fused.wait_for_warm_up()
