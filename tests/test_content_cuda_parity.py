# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.multi_import import _content_keys
from kilora.sets import make_router


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rank,n_out,router_dim", [(16, 2048, 32), (32, 2048, 32), (64, 4096, 32)])
def test_individual_content_keys_compare_with_batched_cuda(
    rank, n_out, router_dim, record_property
):
    generator = torch.Generator().manual_seed(123)
    up = torch.randn(3, 1, rank, n_out, generator=generator)
    router = make_router(router_dim, n_out, 42)
    records = [{"sha256": str(i)} for i in range(3)]
    batched = _content_keys(up, router, records, "layer", device="cuda")
    separate = torch.cat(
        [
            _content_keys(up[i : i + 1], router, records[i : i + 1], "layer", device="cuda")
            for i in range(3)
        ]
    )
    record_property("individual_bitwise_equal", torch.equal(batched, separate))
    torch.testing.assert_close(batched, separate, rtol=1e-4, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_prepared_mix_matches_fresh_without_eigh(monkeypatch, tmp_path):
    from test_import_modes import Toy, _lora

    from kilora import derived_cache, key_cache
    from kilora.multi_import import import_loras
    from kilora.prepare import prepare_mix

    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "cache"))
    model = Toy().eval()
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.safetensors"
        _lora(model, path, rank=3, seed=i)
        paths.append(str(path))
    with monkeypatch.context() as patch:
        patch.setattr(derived_cache, "load", lambda *args, **kwargs: None)
        patch.setattr(derived_cache, "store", lambda *args, **kwargs: None)
        fresh = import_loras(paths, model, device="cuda").adapter_set
    prepare_mix(paths, model, device="cuda")
    monkeypatch.setattr(torch.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigensolve"))
    cached = import_loras(paths, model, device="cuda").adapter_set
    for target in fresh.experts:
        assert torch.equal(fresh.experts[target].keys_in, cached.experts[target].keys_in)
