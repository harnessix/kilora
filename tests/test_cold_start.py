# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from types import SimpleNamespace

import pytest
import torch
from test_fused_input import _imported

from kilora import fused


def _read_share(rank, width=256):
    return _imported(1, experts=4, per_expert=rank, width=width, import_score="read_share")


def test_read_share_ranks_share_a_key():
    assert fused.shape_key(_read_share(263)) == fused.shape_key(_read_share(333))


def test_synthetic_key_matches_real_sizes():
    original = _read_share(263, width=256)
    synthetic = fused._synthetic(original, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert synthetic.import_score == "read_share"
    assert fused.shape_key(original, 3, tokens=4096) == fused.shape_key(synthetic, 3, tokens=32)


def test_wide_read_share_is_warmed(monkeypatch, caplog):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setattr(fused, "_WARMED", set())
    monkeypatch.setattr(fused, "_TOO_BIG", set())
    monkeypatch.setattr(fused, "_shared_limit", lambda _: 99 * 1024)
    calls = []
    monkeypatch.setattr(fused, "run_fused", lambda *args, **kwargs: calls.append(args[1][0]))
    with caplog.at_level(logging.INFO, logger="kilora"):
        assert fused.warm_up("cpu", [(_read_share(263), 1)], (torch.float32,))
    assert len(calls) == 1
    assert calls[0].import_score == "read_share"
    assert "do not fit the shared memory" not in caplog.text


def test_cache_under_comfy_user_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setitem(
        sys.modules, "folder_paths", SimpleNamespace(get_user_directory=lambda: str(tmp_path))
    )
    assert fused.cache_dir() == str(tmp_path / "kilora" / "triton")
    assert fused.os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "kilora" / "triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "custom"))
    assert fused.cache_dir() == str(tmp_path / "custom")


def test_sample_report_excludes_disk_hits_and_restores_listener(caplog):
    import triton

    knobs = getattr(triton, "knobs", None)
    if knobs is None or not hasattr(knobs.compilation, "listener"):
        pytest.skip("Triton does not expose a compilation listener")
    previous = knobs.compilation.listener
    with caplog.at_level(logging.INFO, logger="kilora"), fused.sample_compile_report():
        knobs.compilation.listener(cache_hit=True)
        knobs.compilation.listener(cache_hit=False)
    assert knobs.compilation.listener is previous
    assert "1 kernels compiled" in caplog.text


def test_narrow_ranks_keep_their_own_key():
    assert fused.shape_key(_read_share(16)) != fused.shape_key(_read_share(32))
    assert fused.shape_key(_read_share(65)) == fused.shape_key(_read_share(333))


@pytest.mark.parametrize(
    ("rank", "kernel"), [(32, "_read_share_kernel"), (96, "_read_share_wide_kernel")]
)
def test_read_share_kernel_by_rank(monkeypatch, rank, kernel):
    monkeypatch.setenv("KILORA_FUSED_CPU", "1")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    used = []
    for name in ("_read_share_kernel", "_read_share_wide_kernel"):
        launch = getattr(fused, name)

        class Launcher:
            def __getitem__(self, grid, _name=name, _launch=launch):
                used.append(_name)
                return _launch[grid]

        monkeypatch.setattr(fused, name, Launcher())
    s = _read_share(rank, width=64)
    x = torch.randn(1, 8, s.down.shape[1])
    y0 = torch.randn(1, 8, 64)
    with torch.no_grad():
        fused.run_fused(y0, [s], 2, target="t", x=x, name="t#input0")
    assert used and set(used) == {kernel}


def test_compile_workers_default_one():
    assert fused._COMPILE_WORKERS == 1
