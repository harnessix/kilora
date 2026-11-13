# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import threading

import torch

from kilora import fused


def test_warm_up_is_a_no_op_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    before = threading.active_count()
    assert fused.start_warm_up() is None
    assert fused.warm_up() is False
    assert fused.warm_up("cpu") is False
    assert fused.warm_up_sets([([], 1)]) is None
    assert threading.active_count() == before


def test_default_shapes_cover_imported_sets_and_both_routings():
    shapes = fused._default_templates()
    forms = {(t.form, t.routing_scope, d) for t, d in shapes}
    assert {("input", "token", 1), ("input", "tensor", 1), ("input", "token", 3)} <= forms
    assert {t.up.shape[0] for t, _ in shapes if t.form == "input"} == {64, 128, 256}
    keys = {fused.shape_key(t, d) for t, d in shapes}
    assert len(keys) == len(shapes)


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    assert fused.cache_dir() == str(tmp_path)
